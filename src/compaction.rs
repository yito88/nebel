use std::{
    os::unix::fs::FileExt,
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

use anyhow::Result;

use crate::{
    apply::PendingNotify,
    handle::CollectionInner,
    segment::{SealedSegment, WritableSegment},
    snapshot::SegmentSnapshot,
    storage::CompactionEntry,
    types::{Level, Manifest, SegId, SegmentMeta, SegmentState},
};

// ---------------------------------------------------------------------------
// SegmentInfo
// ---------------------------------------------------------------------------

/// Per-segment summary used by trigger evaluation and candidate selection.
#[derive(Debug, Clone)]
struct SegmentInfo {
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
fn evaluate_triggers(
    infos: &[SegmentInfo],
    num_levels: usize,
    level_count_multiplier: usize,
    tombstone_threshold: f64,
) -> Vec<Level> {
    let mut eligible = Vec::new();

    for i in 0..num_levels {
        let level = Level::new(i as u8);
        let at_level: Vec<&SegmentInfo> = infos.iter().filter(|s| s.level == level).collect();
        if at_level.is_empty() {
            continue;
        }
        let threshold = level_count_multiplier.pow((i + 2) as u32);
        let count_trigger = !level.is_top(num_levels) && at_level.len() >= threshold;
        let tombstone_trigger = at_level
            .iter()
            .any(|s| s.tombstone_ratio() > tombstone_threshold);

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
fn select_candidates(
    level: Level,
    infos: &[SegmentInfo],
    base_capacity: usize,
    num_levels: usize,
    tombstone_threshold: f64,
) -> std::collections::HashSet<SegId> {
    let packing_fill_factor = 1.0 - tombstone_threshold;
    let target_live =
        (level.output(num_levels).capacity(base_capacity) as f64 * packing_fill_factor) as usize;

    let mut at_level: Vec<&SegmentInfo> = infos.iter().filter(|s| s.level == level).collect();

    // Sort: tombstone_ratio desc, seg_id asc (older first as tie-breaker).
    at_level.sort_by(|a, b| {
        b.tombstone_ratio()
            .partial_cmp(&a.tombstone_ratio())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.seg_id.as_u32().cmp(&b.seg_id.as_u32()))
    });

    // Accumulate until we have enough live vectors for the target.
    let mut selected: std::collections::HashSet<SegId> = std::collections::HashSet::new();
    let mut live_sum = 0usize;
    for seg in &at_level {
        selected.insert(seg.seg_id);
        live_sum += seg.live_vectors();
        if live_sum >= target_live {
            break;
        }
    }

    // Require at least 2 segments to avoid trivial single-segment "merges" unless
    // the single segment has a high tombstone ratio (worth cleaning up anyway).
    if selected.len() < 2 {
        let single_high_tombstone = selected
            .iter()
            .next()
            .and_then(|id| infos.iter().find(|s| s.seg_id == *id))
            .map(|s| s.tombstone_ratio() > 0.0)
            .unwrap_or(false);

        if !single_high_tombstone {
            return std::collections::HashSet::new();
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
fn run_merge(
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

    // Create a new writable segment to accumulate live vectors.
    // Size the HNSW index to the output level's capacity so it is never over-allocated.
    let output_capacity = output_level.capacity(schema.segment_params.segment_capacity);
    let mut ws = WritableSegment::create(
        new_seg_id,
        new_dir,
        &schema.metric,
        &schema.segment_params,
        output_capacity,
    )?;

    let mut compaction_entries: Vec<CompactionEntry> = Vec::new();

    let record_size = dimension * 4;
    let chunk_vec_num = schema.segment_params.insert_batch_size;
    let mut buf = vec![0u8; chunk_vec_num * record_size];
    for (seg_id, num_vectors) in &input_segs {
        let mut live = storage.load_segment_live_entries(id, *seg_id, *num_vectors)?;
        let vec_path = inner.seg_dir(*seg_id).join("vectors.seg");
        let vec_file = std::fs::File::open(&vec_path)?;

        for (chunk_idx, chunk) in live.chunks_mut(chunk_vec_num).enumerate() {
            let byte_offset = (chunk_idx * chunk_vec_num * record_size) as u64;
            vec_file.read_exact_at(&mut buf[..chunk.len() * record_size], byte_offset)?;

            let mut live_meta: Vec<(
                String,
                Option<serde_json::Value>,
                Option<crate::types::DocLocation>,
            )> = Vec::new();
            let mut vectors: Vec<Vec<f32>> = Vec::new();
            for (i, entry_opt) in chunk.iter_mut().enumerate() {
                let Some((doc_id, metadata, prev_location)) = entry_opt.take() else {
                    continue;
                };
                let start = i * record_size;
                vectors.push(
                    buf[start..start + record_size]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
                live_meta.push((doc_id, metadata, prev_location));
            }

            let new_ids = ws.insert_batch(&vectors, dimension)?;
            for ((doc_id, metadata, prev_location), new_internal_id) in
                live_meta.into_iter().zip(new_ids)
            {
                compaction_entries.push(CompactionEntry {
                    doc_id,
                    new_internal_id,
                    metadata,
                    prev_location,
                });
            }
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
        schema.compaction_params.tombstone_threshold,
    );
    if candidates.is_empty() {
        level_busy[level.as_usize()].store(false, Ordering::Release);
        return;
    }

    // Build input (seg_id, num_vectors) pairs from infos — no locking needed.
    let input_segs: Vec<(SegId, usize)> = infos
        .iter()
        .filter(|info| candidates.contains(&info.seg_id))
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

    // Build manifest, commit to storage, and update in-memory state under one lock
    // so that concurrent WAL applies cannot modify the manifest between these steps.
    let commit_result = {
        let mut state = inner.apply_state.lock().unwrap();
        let mut active = state
            .manifest
            .active_segments
            .iter()
            .copied()
            .filter(|id| !removed_seg_ids.contains(id))
            .collect::<Vec<_>>();
        active.push(new_seg_id);
        let new_manifest = Manifest {
            active_segments: active,
            writable_segment: state.manifest.writable_segment,
            next_seg_id: state.manifest.next_seg_id,
        };
        let result = storage.commit_compaction(
            &id,
            &new_seg_meta,
            &new_manifest,
            &compaction_entries,
            &removed_seg_ids,
        );
        if result.is_ok() {
            let new_arc = Arc::new(new_sealed);
            state
                .sealed_segs
                .retain(|s| !removed_seg_ids.contains(&s.seg_id()));
            state.sealed_segs.push(Arc::clone(&new_arc));
            state.manifest = new_manifest;
            inner.publish_snapshot(&state);
        }
        result
    };
    match commit_result {
        Ok(()) => {
            for seg_id in &removed_seg_ids {
                let seg_dir = inner.seg_dir(*seg_id);
                if let Err(e) = std::fs::remove_dir_all(&seg_dir) {
                    eprintln!(
                        "[compaction] failed to delete segment dir {:?}: {}",
                        seg_dir, e
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("[compaction] commit_compaction failed at {}: {}", level, e);
        }
    }

    level_busy[level.as_usize()].store(false, Ordering::Release);
}

// ---------------------------------------------------------------------------
// CompactionWorkerGuard
// ---------------------------------------------------------------------------

/// Joins the background compaction coordinator when the last `CollectionHandle` drops.
pub(crate) struct CompactionWorkerGuard {
    pub(crate) shutdown: Arc<AtomicBool>,
    pub(crate) notify: PendingNotify,
    pub(crate) handle: Mutex<Option<thread::JoinHandle<()>>>,
}

impl Drop for CompactionWorkerGuard {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        {
            let mut pending = self.notify.0.lock().unwrap();
            *pending = true;
        }
        self.notify.1.notify_one();
        if let Some(h) = self.handle.lock().unwrap().take() {
            let _ = h.join();
        }
    }
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
        // Block until notified (or shutdown).
        {
            let guard = notify.0.lock().unwrap();
            drop(
                notify
                    .1
                    .wait_while(guard, |hw| !*hw && !shutdown.load(Ordering::Acquire)),
            );
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
            schema.compaction_params.level_count_multiplier,
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
