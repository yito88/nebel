use std::{
    os::unix::fs::FileExt,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::Result;

use crate::{
    handle::{CollectionInner, PendingNotify},
    segment::{SealedSegment, WritableSegment},
    snapshot::SegmentSnapshot,
    storage::CompactionEntry,
    types::{Level, Manifest, SegId, SegmentMeta, SegmentState},
};

// How long the compaction coordinator sleeps between checks when idle.
const COMPACTION_INTERVAL: Duration = Duration::from_millis(500);

// ---------------------------------------------------------------------------
// SegmentInfo
// ---------------------------------------------------------------------------

/// Per-segment summary used by trigger evaluation and candidate selection.
#[derive(Debug, Clone)]
pub(crate) struct SegmentInfo {
    pub seg_id: SegId,
    pub num_vectors: usize,
    pub tombstone_count: usize,
    pub level: Level,
}

impl SegmentInfo {
    pub fn live_vectors(&self) -> usize {
        self.num_vectors.saturating_sub(self.tombstone_count)
    }

    pub fn tombstone_ratio(&self) -> f64 {
        if self.num_vectors == 0 {
            0.0
        } else {
            self.tombstone_count as f64 / self.num_vectors as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Trigger evaluation
// ---------------------------------------------------------------------------

/// Returns the set of levels that are eligible for compaction.
///
/// For levels 0..top-1: triggers when `segment_count >= threshold` OR
///   any segment at that level has `tombstone_ratio > tombstone_threshold`.
/// For the top level: triggers only on tombstone ratio (same-level consolidation).
pub(crate) fn evaluate_triggers(
    infos: &[SegmentInfo],
    num_levels: usize,
    level_count_thresholds: &[usize],
    tombstone_threshold: f64,
) -> Vec<Level> {
    let mut eligible = Vec::new();

    for i in 0..num_levels {
        let level = Level::new(i as u8);
        let at_level: Vec<&SegmentInfo> = infos.iter().filter(|s| s.level == level).collect();
        if at_level.is_empty() {
            continue;
        }
        let threshold = level_count_thresholds.get(i).copied().unwrap_or(usize::MAX);
        let count_trigger = !level.is_top(num_levels) && at_level.len() >= threshold;
        let tombstone_trigger = at_level.iter().any(|s| s.tombstone_ratio() > tombstone_threshold);

        if count_trigger || tombstone_trigger {
            eligible.push(level);
        }
    }
    eligible
}

// ---------------------------------------------------------------------------
// Candidate selection
// ---------------------------------------------------------------------------

/// Select input segments for a compaction at `level`.
///
/// Returns an empty Vec if not enough segments can be gathered to justify a merge.
pub(crate) fn select_candidates(
    level: Level,
    infos: &[SegmentInfo],
    base_capacity: usize,
    num_levels: usize,
    packing_fill_factor: f64,
) -> Vec<SegId> {
    let target_live = (level.output(num_levels).capacity(base_capacity) as f64 * packing_fill_factor) as usize;

    let mut at_level: Vec<&SegmentInfo> =
        infos.iter().filter(|s| s.level == level).collect();

    // Sort: tombstone_ratio desc, seg_id asc (older first as tie-breaker).
    at_level.sort_by(|a, b| {
        b.tombstone_ratio()
            .partial_cmp(&a.tombstone_ratio())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.seg_id.as_u32().cmp(&b.seg_id.as_u32()))
    });

    // Accumulate until we have enough live vectors for the target.
    let mut selected: Vec<SegId> = Vec::new();
    let mut live_sum = 0usize;
    for seg in &at_level {
        selected.push(seg.seg_id);
        live_sum += seg.live_vectors();
        if live_sum >= target_live {
            break;
        }
    }

    // Require at least 2 segments to avoid trivial single-segment "merges" unless
    // the single segment has a high tombstone ratio (worth cleaning up anyway).
    if selected.len() < 2 {
        let single_high_tombstone = selected.first().and_then(|id| {
            infos.iter().find(|s| s.seg_id == *id)
        }).map(|s| s.tombstone_ratio() > 0.0).unwrap_or(false);

        if !single_high_tombstone {
            return Vec::new();
        }
    }

    selected
}

// ---------------------------------------------------------------------------
// Merge execution
// ---------------------------------------------------------------------------

/// Build a new sealed segment from the live vectors of `input_segs`.
///
/// Returns the new `SealedSegment`, its metadata, and the list of
/// `CompactionEntry` items for storage commit.
///
/// This runs without holding `apply_state` — the correctness check is in
/// `Storage::commit_compaction`.
pub(crate) fn run_merge(
    inner: &CollectionInner,
    input_segs: Vec<(SegId, usize)>,
    new_seg_id: SegId,
    output_level: Level,
) -> Result<(SealedSegment, SegmentMeta, Vec<CompactionEntry>)> {
    let id = &inner.id;
    let storage = &*inner.storage;
    let schema = &*inner.schema;
    let dimension = schema.dimension;
    let new_dir = inner.seg_dir(new_seg_id);

    // For each input segment, snapshot live entries from storage.
    // This gives us (doc_id, metadata) per internal_id (None = tombstoned).
    let mut all_entries: Vec<(SegId, Vec<Option<(String, Option<serde_json::Value>)>>)> =
        Vec::new();
    for (seg_id, num_vectors) in &input_segs {
        let entries = storage.load_segment_live_entries(id, *seg_id, *num_vectors)?;
        all_entries.push((*seg_id, entries));
    }

    // Create a new writable segment to accumulate live vectors.
    let mut ws = WritableSegment::create(
        new_seg_id,
        new_dir,
        &schema.metric,
        &schema.segment_params,
    )?;

    let mut compaction_entries: Vec<CompactionEntry> = Vec::new();

    let record_size = dimension * 4;
    for (seg_id, live) in &all_entries {
        let vec_path = inner.seg_dir(*seg_id).join("vectors.seg");
        let vec_file = std::fs::File::open(&vec_path)?;
        for (i, entry_opt) in live.iter().enumerate() {
            let Some((doc_id, metadata)) = entry_opt else {
                continue;
            };
            let mut buf = vec![0u8; record_size];
            vec_file.read_exact_at(&mut buf, i as u64 * record_size as u64)?;
            let vector: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let new_internal_id = ws.insert(&vector, dimension)?;
            compaction_entries.push(CompactionEntry {
                doc_id: doc_id.clone(),
                new_internal_id,
                metadata: metadata.clone(),
            });
        }
    }

    let num_vectors = ws.num_vectors();
    let new_sealed = ws.persist_as_sealed(output_level)?;

    let new_seg_meta = SegmentMeta {
        seg_id: new_seg_id,
        num_vectors,
        state: SegmentState::Sealed,
        tombstone_count: 0,
        level: output_level,
    };

    Ok((new_sealed, new_seg_meta, compaction_entries))
}

// ---------------------------------------------------------------------------
// Post-merge state update
// ---------------------------------------------------------------------------

/// After `commit_compaction` succeeds, update `apply_state` under lock and
/// publish the new snapshot.
fn apply_compaction_result(
    inner: &Arc<CollectionInner>,
    new_sealed: SealedSegment,
    new_seg_meta: &SegmentMeta,
    removed_seg_ids: &[SegId],
    new_manifest: Manifest,
) {
    let new_arc = Arc::new(new_sealed);
    let mut state = inner.apply_state.lock().unwrap();
    state
        .sealed_segs
        .retain(|s| !removed_seg_ids.contains(&s.seg_id()));
    state.sealed_segs.push(Arc::clone(&new_arc));
    state.manifest = new_manifest;
    inner.publish_snapshot(&state);
    drop(state);
    let _ = new_seg_meta; // already committed to storage
}

// ---------------------------------------------------------------------------
// Per-level merge task
// ---------------------------------------------------------------------------

fn run_level_compaction(
    inner: Arc<CollectionInner>,
    level: Level,
    level_busy: Arc<Vec<AtomicBool>>,
) {
    let schema = Arc::clone(&inner.schema);
    let storage = Arc::clone(&inner.storage);
    let id = inner.id.clone();
    let num_levels = schema.compaction_params.num_levels;

    // Build SegmentInfo from the published snapshot (read lock only).
    let infos: Vec<SegmentInfo> = {
        let snap = inner.snapshot.read().unwrap();
        snap.segs
            .iter()
            .filter_map(|s| match s {
                SegmentSnapshot::Sealed(ss) => Some(ss.meta()),
                _ => None,
            })
            .map(|m| SegmentInfo {
                seg_id: m.seg_id,
                num_vectors: m.num_vectors,
                tombstone_count: m.tombstone_count,
                level: m.level,
            })
            .collect()
    };

    let candidates = select_candidates(
        level,
        &infos,
        schema.segment_params.segment_capacity,
        num_levels,
        schema.compaction_params.packing_fill_factor,
    );
    if candidates.is_empty() {
        level_busy[level.as_usize()].store(false, Ordering::Release);
        return;
    }

    // Build input (seg_id, num_vectors) pairs from infos — no locking needed.
    let candidate_set: std::collections::HashSet<SegId> =
        candidates.iter().copied().collect();
    let input_segs: Vec<(SegId, usize)> = infos
        .iter()
        .filter(|info| candidate_set.contains(&info.seg_id))
        .map(|info| (info.seg_id, info.num_vectors))
        .collect();

    // Allocate a new seg_id under the apply_state lock.
    let new_seg_id = {
        let mut state = inner.apply_state.lock().unwrap();
        let new_id = state.manifest.next_seg_id;
        state.manifest.next_seg_id = new_id.next();
        new_id
    };

    let output_level = level.output(num_levels);

    // Run the merge (no lock held).
    let removed_seg_ids: Vec<SegId> = input_segs.iter().map(|(seg_id, _)| *seg_id).collect();
    let merge_result = run_merge(&inner, input_segs, new_seg_id, output_level);

    let (new_sealed, new_seg_meta, compaction_entries) = match merge_result {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[compaction] merge failed at {}: {}", level, e);
            level_busy[level.as_usize()].store(false, Ordering::Release);
            return;
        }
    };

    // Build new manifest (remove old segs, add new).
    let new_manifest = {
        let state = inner.apply_state.lock().unwrap();
        let removed_set: std::collections::HashSet<SegId> =
            removed_seg_ids.iter().copied().collect();
        let mut active = state
            .manifest
            .active_segments
            .iter()
            .copied()
            .filter(|id| !removed_set.contains(id))
            .collect::<Vec<_>>();
        active.push(new_seg_id);
        Manifest {
            active_segments: active,
            writable_segment: state.manifest.writable_segment,
            next_seg_id: state.manifest.next_seg_id,
        }
    };

    // Commit atomically to storage.
    if let Err(e) = storage.commit_compaction(
        &id,
        &new_seg_meta,
        &new_manifest,
        &compaction_entries,
        &removed_seg_ids,
    ) {
        eprintln!("[compaction] commit_compaction failed at {}: {}", level, e);
        level_busy[level.as_usize()].store(false, Ordering::Release);
        return;
    }

    // Update in-memory state.
    apply_compaction_result(&inner, new_sealed, &new_seg_meta, &removed_seg_ids, new_manifest);

    level_busy[level.as_usize()].store(false, Ordering::Release);
}

// ---------------------------------------------------------------------------
// Compaction coordinator loop
// ---------------------------------------------------------------------------

pub(crate) fn compaction_worker_loop(
    weak: Weak<CollectionInner>,
    notify: PendingNotify,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        // Wait for a notification or timeout.
        {
            let guard = notify.0.lock().unwrap();
            let _ = notify.1.wait_timeout_while(guard, COMPACTION_INTERVAL, |hw| {
                !*hw && !shutdown.load(Ordering::Acquire)
            });
        }

        if shutdown.load(Ordering::Acquire) {
            return;
        }

        let inner = match weak.upgrade() {
            Some(i) => i,
            None => return,
        };

        // Reset notification flag.
        *notify.0.lock().unwrap() = false;

        let schema = Arc::clone(&inner.schema);
        let num_levels = schema.compaction_params.num_levels;
        let level_busy = Arc::clone(&inner.level_busy);

        // Snapshot segment infos from the published snapshot (read lock only).
        let infos: Vec<SegmentInfo> = {
            let snap = Arc::clone(&*inner.snapshot.read().unwrap());
            snap.segs
                .iter()
                .filter_map(|s| match s {
                    SegmentSnapshot::Sealed(ss) => Some(ss.meta()),
                    _ => None,
                })
                .map(|m| SegmentInfo {
                    seg_id: m.seg_id,
                    num_vectors: m.num_vectors,
                    tombstone_count: m.tombstone_count,
                    level: m.level,
                })
                .collect()
        };

        // Evaluate triggers.
        let eligible = evaluate_triggers(
            &infos,
            num_levels,
            &schema.compaction_params.level_count_thresholds,
            schema.compaction_params.tombstone_threshold,
        );

        // For each eligible level not already compacting, spawn a thread.
        for level in eligible {
            let idx = level.as_usize();
            if idx >= level_busy.len() {
                continue;
            }
            // Try to claim the level.
            if level_busy[idx]
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue; // Already compacting at this level.
            }

            let inner_clone = Arc::clone(&inner);
            let busy_clone = Arc::clone(&level_busy);
            thread::spawn(move || run_level_compaction(inner_clone, level, busy_clone));
        }

        // inner drops here.
    }
}
