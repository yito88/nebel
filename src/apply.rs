use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex, RwLock, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::Result;

use crate::{
    handle::CollectionInner,
    segment::{SealedSegment, WritableSegment},
    types::{Level, Manifest, SegmentMeta, SegmentState, VectorEntry},
    wal::{ApplyCursor, WalId, WalOp, WalRecord},
};

pub(crate) const APPLY_BATCH_MAX: usize = 2048;
const APPLY_INTERVAL: Duration = Duration::from_millis(50);

// ---------------------------------------------------------------------------
// PendingNotify
// ---------------------------------------------------------------------------

/// Notification channel held by both `CollectionInner` and the apply worker.
/// Lightweight (just a bool + condvar), no `Storage` reference.
/// This allows the worker to wait WITHOUT holding an `Arc<CollectionInner>`,
/// so `Storage`/`Database` can be freed as soon as the last handle drops.
pub(crate) type PendingNotify = Arc<(Mutex<bool>, std::sync::Condvar)>;

// ---------------------------------------------------------------------------
// ApplyWorkerGuard
// ---------------------------------------------------------------------------

/// Joins the background apply worker when the last `CollectionHandle` drops.
/// Declared before `inner` in `CollectionHandle` so Rust's field-drop order
/// guarantees the thread exits before `CollectionInner` (and `Storage`) is freed.
pub(crate) struct ApplyWorkerGuard {
    pub(crate) shutdown: Arc<AtomicBool>,
    pub(crate) notify: PendingNotify,
    pub(crate) handle: Mutex<Option<thread::JoinHandle<()>>>,
}

impl Drop for ApplyWorkerGuard {
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
// ApplyState
// ---------------------------------------------------------------------------

/// Mutable state that the apply worker and sealing helpers share under a single `Mutex`.
pub(crate) struct ApplyState {
    pub(crate) sealed_segs: Vec<Arc<SealedSegment>>,
    pub(crate) writable_seg: Arc<RwLock<WritableSegment>>,
    pub(crate) manifest: Manifest,
    pub(crate) applied_seq: u64,
}

// ---------------------------------------------------------------------------
// notify_worker
// ---------------------------------------------------------------------------

pub(crate) fn notify_worker(notify: &PendingNotify) {
    *notify.0.lock().unwrap() = true;
    notify.1.notify_one();
}

// ---------------------------------------------------------------------------
// Background apply worker
// ---------------------------------------------------------------------------

pub(crate) fn apply_worker_loop(
    weak: Weak<CollectionInner>,
    notify: PendingNotify,
    shutdown: Arc<AtomicBool>,
) {
    // The cursor is local to this thread — no need to persist it.
    // After recovery, WAL is always wiped and starts fresh from wal_id=1.
    let mut cursor_opt: Option<ApplyCursor> = None;

    loop {
        // Wait for work WITHOUT holding Arc<CollectionInner>.
        {
            let guard = notify.0.lock().unwrap();
            let _ = notify.1.wait_timeout_while(guard, APPLY_INTERVAL, |hw| {
                !*hw && !shutdown.load(Ordering::Acquire)
            });
        }

        // Check shutdown before upgrading so we never resurrect the Arc
        // after the last CollectionHandle has been dropped.
        if shutdown.load(Ordering::Acquire) {
            return;
        }

        let inner = match weak.upgrade() {
            Some(i) => i,
            None => return,
        };

        // Initialize cursor on the first iteration.
        let cursor = cursor_opt.get_or_insert_with(|| {
            let applied = inner.apply_state.lock().unwrap().applied_seq;
            ApplyCursor {
                current_wal_id: WalId::first(),
                current_offset: 0,
                last_applied_seq: applied,
            }
        });

        // Only work if there are records durably written beyond what we've applied.
        let durable = inner.durable_seq.load(Ordering::Acquire);
        if durable <= cursor.last_applied_seq {
            *notify.0.lock().unwrap() = false;
            continue;
        }

        // Snapshot segment paths briefly under lock; read files without holding it.
        let segments = inner
            .wal
            .lock()
            .unwrap()
            .segment_paths_from(cursor.current_wal_id);

        let (pending, new_cursor) =
            match read_records_from_cursor(&segments, cursor, durable, APPLY_BATCH_MAX) {
                Ok(r) => r,
                Err(_) => {
                    *notify.0.lock().unwrap() = false;
                    continue;
                }
            };

        if pending.is_empty() {
            *notify.0.lock().unwrap() = false;
            continue;
        }

        // Apply records.
        let mut state = inner.apply_state.lock().unwrap();
        let seg_id_before = state.manifest.next_seg_id;
        let _ = apply_records(&inner, &mut state, &pending);

        let applied = state.applied_seq;
        let sealed = state.manifest.next_seg_id != seg_id_before;
        inner.publish_snapshot(&state);
        drop(state);

        inner.visible_seq.fetch_max(applied + 1, Ordering::Release);
        inner.visible.1.notify_all();
        if sealed {
            notify_worker(&inner.compaction_notify);
        }

        // Advance cursor; keep physical position from new_cursor, update logical seq.
        *cursor = ApplyCursor {
            last_applied_seq: applied,
            ..new_cursor
        };

        // Drop fully-applied closed WAL segments.
        inner.wal.lock().unwrap().truncate_applied(applied);

        // Clear pending flag; re-set if more records may exist.
        *notify.0.lock().unwrap() = false;
        // inner drops here.
    }
}

// ---------------------------------------------------------------------------
// read_records_from_cursor
// ---------------------------------------------------------------------------

/// Read up to `batch_size` WAL records starting from `cursor`, stopping at `up_to_seq`.
/// Returns the records and an updated cursor reflecting the new position.
/// Does NOT hold the WAL mutex — callers must snapshot segment paths first.
fn read_records_from_cursor(
    segments: &[(WalId, PathBuf)],
    cursor: &ApplyCursor,
    up_to_seq: u64,
    batch_size: usize,
) -> Result<(Vec<WalRecord>, ApplyCursor)> {
    use std::io::Read;

    let mut records = Vec::new();
    let mut new_cursor = ApplyCursor {
        current_wal_id: cursor.current_wal_id,
        current_offset: cursor.current_offset,
        last_applied_seq: cursor.last_applied_seq,
    };

    'outer: for (wal_id, path) in segments
        .iter()
        .filter(|(id, _)| *id >= cursor.current_wal_id)
    {
        if !path.exists() {
            continue;
        }

        let start_offset = if *wal_id == cursor.current_wal_id {
            cursor.current_offset as usize
        } else {
            0
        };

        let mut file = fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut pos = start_offset;
        while pos + 4 <= data.len() {
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if pos + len > data.len() {
                break;
            }
            let body = &data[pos..pos + len];
            let rec = match serde_json::from_slice::<WalRecord>(body) {
                Ok(r) => r,
                Err(_) => break,
            };
            pos += len;
            if pos < data.len() && data[pos] == b'\n' {
                pos += 1;
            }

            if rec.seq > up_to_seq {
                break 'outer;
            }

            new_cursor.current_wal_id = *wal_id;
            new_cursor.current_offset = pos as u64;
            new_cursor.last_applied_seq = rec.seq;
            records.push(rec);

            if records.len() >= batch_size {
                break 'outer;
            }
        }
    }

    Ok((records, new_cursor))
}

// ---------------------------------------------------------------------------
// apply_records / flush_upsert_batch
// ---------------------------------------------------------------------------

/// Apply a slice of WAL records to `state`, batching consecutive `Upsert` operations.
/// Non-upsert records are handled one at a time via `apply_entry`.
pub(crate) fn apply_records(
    inner: &CollectionInner,
    state: &mut ApplyState,
    records: &[WalRecord],
) -> Result<()> {
    let mut i = 0;
    while i < records.len() {
        if records[i].seq <= state.applied_seq {
            i += 1;
            continue;
        }
        if matches!(records[i].op, WalOp::Upsert { .. }) {
            let start = i;
            while i < records.len()
                && records[i].seq > state.applied_seq
                && matches!(records[i].op, WalOp::Upsert { .. })
            {
                i += 1;
            }
            flush_upsert_batch(inner, state, &records[start..i])?;
        } else {
            apply_entry(inner, state, &records[i])?;
            i += 1;
        }
    }
    Ok(())
}

/// Insert a slice of `Upsert` WAL records into the writable segment, splitting across
/// segment-capacity boundaries as needed. Each chunk is committed as one redb transaction.
fn flush_upsert_batch(
    inner: &CollectionInner,
    state: &mut ApplyState,
    records: &[WalRecord],
) -> Result<()> {
    let schema = &*inner.schema;
    let storage = &*inner.storage;
    let id = &inner.id;
    let capacity = schema.segment_params.segment_capacity;

    let mut offset = 0;
    while offset < records.len() {
        let current_count = state.writable_seg.read().unwrap().num_vectors();
        if current_count >= capacity {
            seal_and_new_segment(inner, state)?;
        }
        let remaining_capacity = capacity - state.writable_seg.read().unwrap().num_vectors();
        let chunk_size = (records.len() - offset).min(remaining_capacity);
        let chunk = &records[offset..offset + chunk_size];

        let vectors: Vec<Vec<f32>> = chunk
            .iter()
            .map(|r| {
                let WalOp::Upsert { vector, .. } = &r.op else {
                    unreachable!()
                };
                vector.clone()
            })
            .collect();

        let internal_ids = {
            let mut ws = state.writable_seg.write().unwrap();
            ws.insert_batch(&vectors, schema.dimension)?
        };

        let seg_id = state.manifest.writable_segment;
        let num_vectors = state.writable_seg.read().unwrap().num_vectors();

        let entries: Vec<VectorEntry<'_>> = chunk
            .iter()
            .zip(&internal_ids)
            .map(|(r, &iid)| {
                let WalOp::Upsert {
                    doc_id, metadata, ..
                } = &r.op
                else {
                    unreachable!()
                };
                VectorEntry {
                    doc_id: doc_id.as_str(),
                    internal_id: iid,
                    metadata: metadata.as_ref(),
                }
            })
            .collect();

        let last_seq = chunk.last().unwrap().seq;
        storage.apply_upsert_batch(
            id,
            &SegmentMeta {
                seg_id,
                num_vectors,
                state: SegmentState::Writable,
                tombstone_count: 0,
                level: Level::ZERO,
            },
            &entries,
            last_seq,
        )?;

        state.applied_seq = last_seq;
        offset += chunk_size;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// apply_entry
// ---------------------------------------------------------------------------

/// Apply a single WAL record to `state`. Shared between the apply worker and recovery.
fn apply_entry(inner: &CollectionInner, state: &mut ApplyState, record: &WalRecord) -> Result<()> {
    let storage = &*inner.storage;
    let schema = &*inner.schema;
    let id = &inner.id;

    match &record.op {
        WalOp::Upsert {
            doc_id,
            vector,
            metadata,
        } => {
            if state.writable_seg.read().unwrap().num_vectors()
                >= schema.segment_params.segment_capacity
            {
                seal_and_new_segment(inner, state)?;
            }
            let internal_id = {
                let mut ws = state.writable_seg.write().unwrap();
                ws.insert(vector, schema.dimension)?
            };
            let seg_id = state.manifest.writable_segment;
            let num_vectors = state.writable_seg.read().unwrap().num_vectors();
            storage.apply_upsert(
                id,
                doc_id.as_str(),
                &SegmentMeta {
                    seg_id,
                    num_vectors,
                    state: SegmentState::Writable,
                    tombstone_count: 0,
                    level: Level::ZERO,
                },
                &[VectorEntry {
                    doc_id: doc_id.as_str(),
                    internal_id,
                    metadata: metadata.as_ref(),
                }],
                record.seq,
            )?;
        }
        WalOp::Delete { doc_id } => {
            storage.apply_delete(id, doc_id, record.seq)?;
        }
        WalOp::UpdateMetadata { doc_id, metadata } => {
            storage.apply_update_metadata(id, doc_id, metadata, record.seq)?;
        }
    }

    state.applied_seq = record.seq;
    Ok(())
}

// ---------------------------------------------------------------------------
// Sealing helper
// ---------------------------------------------------------------------------

/// Seal the current writable segment and create a fresh one, updating `state` and storage.
pub(crate) fn seal_and_new_segment(inner: &CollectionInner, state: &mut ApplyState) -> Result<()> {
    let storage = &*inner.storage;
    let old_id = state.manifest.writable_segment;
    let new_id = state.manifest.next_seg_id;
    let new_dir = inner.seg_dir(new_id);

    let num_vectors = state.writable_seg.read().unwrap().num_vectors();
    // Newly sealed segments are always L0.
    let sealed = state
        .writable_seg
        .read()
        .unwrap()
        .persist_as_sealed(Level::ZERO)?;
    let sealed_arc = Arc::new(sealed);

    let new_ws = WritableSegment::create(
        new_id,
        new_dir,
        &inner.schema.metric,
        &inner.schema.segment_params,
    )?;

    state.sealed_segs.push(sealed_arc);
    state.manifest.active_segments.push(new_id);
    state.manifest.writable_segment = new_id;
    state.manifest.next_seg_id = new_id.next();
    state.writable_seg = Arc::new(RwLock::new(new_ws));

    storage.seal_segment(
        &inner.id,
        &SegmentMeta {
            seg_id: old_id,
            num_vectors,
            state: SegmentState::Sealed,
            tombstone_count: 0,
            level: Level::ZERO,
        },
        &SegmentMeta::new(new_id),
        &state.manifest,
    )?;

    Ok(())
}
