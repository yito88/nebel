use std::{
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
    storage::CompactionEntry,
    types::{InternalId, Manifest, SegId, SegmentMeta, SegmentState},
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
    pub level: u8,
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

/// Level capacity multipliers: level `l` holds up to `capacity * 2^l` vectors.
fn level_capacity(base_capacity: usize, level: usize) -> usize {
    base_capacity * (1 << level)
}

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
) -> Vec<usize> {
    let top = num_levels.saturating_sub(1);
    let mut eligible = Vec::new();

    for level in 0..num_levels {
        let at_level: Vec<&SegmentInfo> = infos.iter().filter(|s| s.level as usize == level).collect();
        if at_level.is_empty() {
            continue;
        }
        let threshold = level_count_thresholds.get(level).copied().unwrap_or(usize::MAX);
        let count_trigger = level < top && at_level.len() >= threshold;
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
    level: usize,
    infos: &[SegmentInfo],
    base_capacity: usize,
    num_levels: usize,
    packing_fill_factor: f64,
) -> Vec<SegId> {
    let top = num_levels.saturating_sub(1);
    // For the top level, "next level" = same level (tombstone cleanup).
    let target_capacity = if level < top {
        level_capacity(base_capacity, level + 1)
    } else {
        level_capacity(base_capacity, level)
    };
    let target_live = (target_capacity as f64 * packing_fill_factor) as usize;

    let mut at_level: Vec<&SegmentInfo> =
        infos.iter().filter(|s| s.level as usize == level).collect();

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
    input_segs: Vec<Arc<SealedSegment>>,
    new_seg_id: SegId,
    output_level: u8,
) -> Result<(SealedSegment, SegmentMeta, Vec<CompactionEntry>)> {
    let id = &inner.id;
    let storage = &*inner.storage;
    let schema = &*inner.schema;
    let dimension = schema.dimension;
    let new_dir = inner.seg_dir(new_seg_id);

    // For each input segment, snapshot live entries from storage.
    // This gives us (doc_id, metadata) per internal_id (None = tombstoned).
    let mut all_entries: Vec<(Arc<SealedSegment>, Vec<Option<(String, Option<serde_json::Value>)>>)> =
        Vec::new();
    for seg in &input_segs {
        let entries =
            storage.load_segment_live_entries(id, seg.seg_id(), seg.num_vectors())?;
        all_entries.push((Arc::clone(seg), entries));
    }

    // Create a new writable segment to accumulate live vectors.
    let mut ws = WritableSegment::create(
        new_seg_id,
        new_dir,
        &schema.metric,
        &schema.segment_params,
    )?;

    let mut compaction_entries: Vec<CompactionEntry> = Vec::new();

    for (seg, live) in &all_entries {
        for (i, entry_opt) in live.iter().enumerate() {
            let Some((doc_id, metadata)) = entry_opt else {
                continue;
            };
            let internal_id = InternalId::from_u32(i as u32);
            let vector = seg.read_vector(internal_id, dimension)?;
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
    level: usize,
    level_busy: Arc<Vec<AtomicBool>>,
) {
    let schema = Arc::clone(&inner.schema);
    let storage = Arc::clone(&inner.storage);
    let id = inner.id.clone();
    let num_levels = schema.compaction_params.num_levels;
    let top = num_levels.saturating_sub(1);

    // Snapshot current sealed segment info under lock.
    let (seg_infos, input_seg_arcs, new_seg_id) = {
        let mut state = inner.apply_state.lock().unwrap();

        // Build SegmentInfo list by reading tombstone counts from storage.
        let mut infos: Vec<SegmentInfo> = Vec::new();
        for ss in &state.sealed_segs {
            let tc = storage
                .get_segment(&id, ss.seg_id())
                .ok()
                .flatten()
                .map(|m| m.tombstone_count)
                .unwrap_or(0);
            infos.push(SegmentInfo {
                seg_id: ss.seg_id(),
                num_vectors: ss.num_vectors(),
                tombstone_count: tc,
                level: ss.level(),
            });
        }

        let candidates = select_candidates(
            level,
            &infos,
            schema.segment_params.segment_capacity,
            num_levels,
            schema.compaction_params.packing_fill_factor,
        );
        if candidates.is_empty() {
            drop(state);
            level_busy[level].store(false, Ordering::Release);
            return;
        }

        let candidate_set: std::collections::HashSet<SegId> =
            candidates.iter().copied().collect();
        let input_arcs: Vec<Arc<SealedSegment>> = state
            .sealed_segs
            .iter()
            .filter(|s| candidate_set.contains(&s.seg_id()))
            .map(Arc::clone)
            .collect();

        let new_id = state.manifest.next_seg_id;
        state.manifest.next_seg_id = new_id.next();

        (infos, input_arcs, new_id)
    };

    // Determine output level.
    let output_level = if level < top {
        (level + 1).min(top) as u8
    } else {
        level as u8
    };

    // Run the merge (no lock held).
    let removed_seg_ids: Vec<SegId> = input_seg_arcs.iter().map(|s| s.seg_id()).collect();
    let merge_result = run_merge(&inner, input_seg_arcs, new_seg_id, output_level);

    let (new_sealed, new_seg_meta, compaction_entries) = match merge_result {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[compaction] merge failed at level {}: {}", level, e);
            level_busy[level].store(false, Ordering::Release);
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
        eprintln!("[compaction] commit_compaction failed at level {}: {}", level, e);
        level_busy[level].store(false, Ordering::Release);
        return;
    }

    // Update in-memory state.
    apply_compaction_result(&inner, new_sealed, &new_seg_meta, &removed_seg_ids, new_manifest);

    let _ = seg_infos; // consumed above
    level_busy[level].store(false, Ordering::Release);
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
        let storage = Arc::clone(&inner.storage);
        let id = inner.id.clone();
        let num_levels = schema.compaction_params.num_levels;
        let level_busy = Arc::clone(&inner.level_busy);

        // Snapshot segment infos under lock.
        let infos: Vec<SegmentInfo> = {
            let state = inner.apply_state.lock().unwrap();
            state
                .sealed_segs
                .iter()
                .map(|ss| {
                    let tc = storage
                        .get_segment(&id, ss.seg_id())
                        .ok()
                        .flatten()
                        .map(|m| m.tombstone_count)
                        .unwrap_or(0);
                    SegmentInfo {
                        seg_id: ss.seg_id(),
                        num_vectors: ss.num_vectors(),
                        tombstone_count: tc,
                        level: ss.level(),
                    }
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
            if level >= level_busy.len() {
                continue;
            }
            // Try to claim the level.
            if level_busy[level]
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
