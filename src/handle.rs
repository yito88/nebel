use std::{
    cmp::Ordering as CmpOrdering,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex, RwLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use anyhow::{Result, bail};
use rayon::prelude::*;
use serde_json::Value;

#[cfg(feature = "testing")]
use crate::segment::compute_distance;
use crate::{
    segment::{SealedSegment, WritableSegment},
    snapshot::{CollectionSnapshot, SegmentSnapshot},
    storage::Storage,
    types::{
        CollectionId, CollectionSchema, InternalId, Level, Manifest, SearchHit, SegId, SegmentMeta,
        SegmentState, VectorEntry, WriteToken,
    },
    wal::{ApplyCursor, Wal, WalId, WalOp, WalRecord},
};

const INGEST_BATCH_SIZE: usize = 2048;
const APPLY_BATCH_MAX: usize = 2048;
const APPLY_INTERVAL: Duration = Duration::from_millis(50);

// ---------------------------------------------------------------------------
// PendingNotify
// ---------------------------------------------------------------------------

/// Notification channel held by both `CollectionInner` and the apply worker.
/// Lightweight (just a bool + condvar), no `Storage` reference.
/// This allows the worker to wait WITHOUT holding an `Arc<CollectionInner>`,
/// so `Storage`/`Database` can be freed as soon as the last handle drops.
pub(crate) type PendingNotify = Arc<(Mutex<bool>, Condvar)>;

// ---------------------------------------------------------------------------
// ApplyWorkerGuard
// ---------------------------------------------------------------------------

/// Joins the background apply worker when the last `CollectionHandle` drops.
/// Declared before `inner` in `CollectionHandle` so Rust's field-drop order
/// guarantees the thread exits before `CollectionInner` (and `Storage`) is freed.
struct ApplyWorkerGuard {
    shutdown: Arc<AtomicBool>,
    notify: PendingNotify,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
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
// CollectionInner
// ---------------------------------------------------------------------------

/// Shared core state for a single collection, owned behind an `Arc`.
pub(crate) struct CollectionInner {
    pub(crate) id: CollectionId,
    pub(crate) schema: Arc<CollectionSchema>,
    pub(crate) storage: Arc<Storage>,
    pub(crate) base_dir: PathBuf,
    pub(crate) apply_state: Mutex<ApplyState>,
    pub(crate) snapshot: RwLock<Arc<CollectionSnapshot>>,
    pub(crate) wal: Mutex<Wal>,
    /// Shared with the apply worker thread — worker waits on this without holding
    /// an Arc<CollectionInner>, so the database can be freed promptly on drop.
    pub(crate) notify: PendingNotify,
    /// Shared with the compaction worker — notified after each seal.
    pub(crate) compaction_notify: PendingNotify,
    /// Per-level in-progress flags. Prevents concurrent merges at the same level.
    /// Length == `schema.compaction_params.num_levels`.
    pub(crate) level_busy: Arc<Vec<AtomicBool>>,
    /// Highest seq that has been fsync'd to WAL. Updated by writers after sync_all.
    /// The apply worker only processes records up to this value.
    pub(crate) durable_seq: AtomicU64,
    pub(crate) visible_seq: AtomicU64,
    pub(crate) visible: (Mutex<()>, Condvar),
}

impl CollectionInner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: CollectionId,
        storage: Arc<Storage>,
        base_dir: PathBuf,
        manifest: Manifest,
        sealed_segs: Vec<Arc<SealedSegment>>,
        writable_seg: Arc<RwLock<WritableSegment>>,
        applied_seq: u64,
        schema: Arc<CollectionSchema>,
    ) -> Result<Arc<Self>> {
        let mut wal = Wal::open_dir(&base_dir.join(id.as_str()).join("wal"))?;
        wal.rotation_bytes = schema.wal_segment_bytes;
        // Seq allocation lives inside Wal (under its mutex). Set the starting
        // value so new records continue from where recovery left off.
        wal.reset_next_seq(applied_seq + 1);

        let initial_segs = sealed_segs
            .iter()
            .map(|s| SegmentSnapshot::Sealed(Arc::clone(s)))
            .chain(std::iter::once(SegmentSnapshot::Writable(Arc::clone(
                &writable_seg,
            ))))
            .collect();
        let initial_snap = Arc::new(CollectionSnapshot {
            schema: Arc::clone(&schema),
            segs: initial_segs,
        });
        let num_levels = schema.compaction_params.num_levels;
        let level_busy = Arc::new(
            (0..num_levels)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>(),
        );
        Ok(Arc::new(Self {
            id,
            schema,
            storage,
            base_dir,
            apply_state: Mutex::new(ApplyState {
                sealed_segs,
                writable_seg,
                manifest,
                applied_seq,
            }),
            snapshot: RwLock::new(initial_snap),
            wal: Mutex::new(wal),
            notify: Arc::new((Mutex::new(false), Condvar::new())),
            compaction_notify: Arc::new((Mutex::new(false), Condvar::new())),
            level_busy,
            // durable_seq starts at applied_seq: after recovery WAL is empty, so the
            // worker's cursor (last_applied_seq = applied_seq) won't read anything
            // until a real write bumps durable_seq above applied_seq.
            durable_seq: AtomicU64::new(applied_seq),
            visible_seq: AtomicU64::new(applied_seq),
            visible: (Mutex::new(()), Condvar::new()),
        }))
    }

    pub(crate) fn publish_snapshot(&self, state: &ApplyState) {
        let segs = state
            .sealed_segs
            .iter()
            .map(|s| SegmentSnapshot::Sealed(Arc::clone(s)))
            .chain(std::iter::once(SegmentSnapshot::Writable(Arc::clone(
                &state.writable_seg,
            ))))
            .collect();
        let snap = Arc::new(CollectionSnapshot {
            schema: Arc::clone(&self.schema),
            segs,
        });
        *self.snapshot.write().unwrap() = snap;
    }

    pub(crate) fn seg_dir(&self, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(self.id.as_str())
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }
}

// ---------------------------------------------------------------------------
// CompactionWorkerGuard
// ---------------------------------------------------------------------------

/// Joins the background compaction coordinator when the last `CollectionHandle` drops.
struct CompactionWorkerGuard {
    shutdown: Arc<AtomicBool>,
    notify: PendingNotify,
    handle: Mutex<Option<thread::JoinHandle<()>>>,
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
// CollectionHandle
// ---------------------------------------------------------------------------

/// Cheaply cloneable handle to a collection. Spawns a background apply worker on creation.
#[derive(Clone)]
pub struct CollectionHandle {
    /// Dropped before `inner` — joins the apply worker thread so the redb lock is
    /// released deterministically before `Storage` is freed.
    _guard: Arc<ApplyWorkerGuard>,
    /// Dropped before `inner` — joins the compaction coordinator thread.
    _compaction_guard: Arc<CompactionWorkerGuard>,
    pub(crate) inner: Arc<CollectionInner>,
}

impl CollectionHandle {
    pub(crate) fn from_arc(inner: Arc<CollectionInner>) -> Self {
        // Spawn apply worker.
        let notify = Arc::clone(&inner.notify);
        let shutdown = Arc::new(AtomicBool::new(false));
        let weak = Arc::downgrade(&inner);
        let shutdown2 = Arc::clone(&shutdown);
        let notify2 = Arc::clone(&notify);
        let handle = thread::spawn(move || apply_worker_loop(weak, notify2, shutdown2));
        let guard = Arc::new(ApplyWorkerGuard {
            shutdown,
            notify,
            handle: Mutex::new(Some(handle)),
        });

        // Spawn compaction coordinator.
        let compact_notify = Arc::clone(&inner.compaction_notify);
        let compact_shutdown = Arc::new(AtomicBool::new(false));
        let weak2 = Arc::downgrade(&inner);
        let compact_shutdown2 = Arc::clone(&compact_shutdown);
        let compact_notify2 = Arc::clone(&compact_notify);
        let compact_handle = thread::spawn(move || {
            crate::compaction::compaction_worker_loop(weak2, compact_notify2, compact_shutdown2)
        });
        let compaction_guard = Arc::new(CompactionWorkerGuard {
            shutdown: compact_shutdown,
            notify: compact_notify,
            handle: Mutex::new(Some(compact_handle)),
        });

        Self {
            _guard: guard,
            _compaction_guard: compaction_guard,
            inner,
        }
    }

    // -----------------------------------------------------------------------
    // Write operations — WAL path
    // -----------------------------------------------------------------------

    pub fn upsert(
        &self,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<WriteToken> {
        let inner = &self.inner;
        if vector.len() != inner.schema.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                inner.schema.dimension,
                vector.len()
            );
        }
        let seq = inner.wal.lock().unwrap().append(WalOp::Upsert {
            doc_id: doc_id.to_string(),
            vector: vector.to_vec(),
            metadata,
        })?;
        inner.durable_seq.fetch_max(seq, Ordering::Release);
        notify_worker(&inner.notify);
        Ok(WriteToken(seq + 1))
    }

    /// Upsert a batch of vectors atomically: all entries in the batch are written to the WAL with
    /// a single `fsync`, so either the entire batch is durable or none of it is.
    ///
    /// # Duplicate `doc_id` handling
    ///
    /// If the same `doc_id` appears more than once in `entries`, only the **last** occurrence is
    /// kept. Earlier entries for the same `doc_id` are silently discarded before writing to the
    /// WAL.
    ///
    /// # Batch size limit
    ///
    /// `entries` must contain at most [`APPLY_BATCH_MAX`] entries. Larger slices are rejected with
    /// an error.
    ///
    /// # Visibility
    ///
    /// The returned [`WriteToken`] can be passed to [`Collection::wait_visible`] to block until
    /// all vectors in the batch have been applied and are visible to search.
    pub fn upsert_batch(&self, entries: &[(&str, &[f32], Option<Value>)]) -> Result<WriteToken> {
        if entries.is_empty() {
            return Ok(WriteToken(self.inner.durable_seq.load(Ordering::Acquire)));
        }
        if entries.len() > APPLY_BATCH_MAX {
            bail!(
                "batch size {} exceeds maximum of {}",
                entries.len(),
                APPLY_BATCH_MAX
            );
        }
        let inner = &self.inner;
        let expected_dim = inner.schema.dimension;
        for (i, (_, vector, _)) in entries.iter().enumerate() {
            if vector.len() != expected_dim {
                bail!(
                    "dimension mismatch at index {}: expected {}, got {}",
                    i,
                    expected_dim,
                    vector.len()
                );
            }
        }
        // Deduplicate by doc_id: last entry wins.
        let mut seen: std::collections::HashMap<&str, usize> =
            std::collections::HashMap::with_capacity(entries.len());
        for (i, (doc_id, _, _)) in entries.iter().enumerate() {
            seen.insert(doc_id, i);
        }
        let mut deduped: Vec<usize> = seen.into_values().collect();
        deduped.sort_unstable();
        let ops: Vec<WalOp> = deduped
            .iter()
            .map(|&i| {
                let (doc_id, vector, metadata) = &entries[i];
                WalOp::Upsert {
                    doc_id: doc_id.to_string(),
                    vector: vector.to_vec(),
                    metadata: metadata.clone(),
                }
            })
            .collect();
        let last_seq = inner.wal.lock().unwrap().append_batch(ops)?;
        inner.durable_seq.fetch_max(last_seq, Ordering::Release);
        notify_worker(&inner.notify);
        Ok(WriteToken(last_seq + 1))
    }

    pub fn delete(&self, doc_id: &str) -> Result<WriteToken> {
        let inner = &self.inner;
        let seq = inner.wal.lock().unwrap().append(WalOp::Delete {
            doc_id: doc_id.to_string(),
        })?;
        inner.durable_seq.fetch_max(seq, Ordering::Release);
        notify_worker(&inner.notify);
        Ok(WriteToken(seq + 1))
    }

    pub fn update_metadata(&self, doc_id: &str, metadata: Value) -> Result<WriteToken> {
        let inner = &self.inner;
        let seq = inner.wal.lock().unwrap().append(WalOp::UpdateMetadata {
            doc_id: doc_id.to_string(),
            metadata,
        })?;
        inner.durable_seq.fetch_max(seq, Ordering::Release);
        notify_worker(&inner.notify);
        Ok(WriteToken(seq + 1))
    }

    // -----------------------------------------------------------------------
    // Search
    // -----------------------------------------------------------------------

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        include_metadata: bool,
        include_vector: bool,
    ) -> Result<Vec<SearchHit>> {
        let snap = { Arc::clone(&*self.inner.snapshot.read().unwrap()) };
        search_snapshot(
            &self.inner,
            &snap,
            query,
            k,
            include_metadata,
            include_vector,
        )
    }

    #[cfg(feature = "testing")]
    pub fn search_exact(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        let snap = { Arc::clone(&*self.inner.snapshot.read().unwrap()) };
        search_exact_snapshot(&self.inner, &snap, query, k)
    }

    // -----------------------------------------------------------------------
    // Ingest (fast-path: bypasses WAL, directly applies to segments)
    // -----------------------------------------------------------------------

    #[cfg(feature = "testing")]
    pub fn ingest_file(&self, file_path: impl AsRef<Path>) -> Result<usize> {
        let inner = &self.inner;
        let storage = &*inner.storage;
        let dimension = inner.schema.dimension;
        let record_size = dimension * 4;

        let mut file = fs::File::open(file_path)?;
        let mut buf = vec![0u8; record_size * INGEST_BATCH_SIZE];
        let mut count = 0usize;

        loop {
            let n_bytes = read_up_to(&mut file, &mut buf)?;
            if n_bytes == 0 {
                break;
            }
            if n_bytes % record_size != 0 {
                bail!("partial record: {} trailing bytes", n_bytes % record_size);
            }
            let vectors: Vec<Vec<f32>> = buf[..n_bytes]
                .chunks_exact(record_size)
                .map(|c| {
                    c.chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect()
                })
                .collect();
            let doc_ids: Vec<String> = (count..count + vectors.len())
                .map(|i| format!("doc_{}", i))
                .collect();

            let mut state = inner.apply_state.lock().unwrap();
            if state.writable_seg.read().unwrap().num_vectors()
                >= inner.schema.segment_params.segment_capacity
            {
                seal_and_new_segment(inner, &mut state)?;
            }
            let internal_ids = {
                let mut ws = state.writable_seg.write().unwrap();
                ws.insert_batch(&vectors, dimension)?
            };
            let seg_id = state.manifest.writable_segment;
            let num_vectors = state.writable_seg.read().unwrap().num_vectors();
            let entries: Vec<VectorEntry<'_>> = doc_ids
                .iter()
                .zip(internal_ids.iter())
                .map(|(doc_id, &internal_id)| VectorEntry {
                    doc_id: doc_id.as_str(),
                    internal_id,
                    metadata: None,
                })
                .collect();
            storage.write_vector_entries(
                &inner.id,
                &SegmentMeta {
                    seg_id,
                    num_vectors,
                    state: SegmentState::Writable,
                    tombstone_count: 0,
                    level: Level::ZERO,
                },
                &entries,
            )?;
            count += vectors.len();
        }

        // Publish snapshot so searches immediately see the ingested data, then
        // notify compaction so it sees the up-to-date snapshot.
        {
            let state = inner.apply_state.lock().unwrap();
            inner.publish_snapshot(&state);
        }
        notify_worker(&inner.compaction_notify);

        Ok(count)
    }

    // -----------------------------------------------------------------------
    // Visibility
    // -----------------------------------------------------------------------

    pub fn wait_visible(&self, token: WriteToken) -> Result<()> {
        let inner = &self.inner;
        if inner.visible_seq.load(Ordering::Acquire) >= token.0 {
            return Ok(());
        }
        let guard = inner.visible.0.lock().unwrap();
        drop(inner.visible.1.wait_while(guard, |_| {
            inner.visible_seq.load(Ordering::Acquire) < token.0
        }));
        Ok(())
    }

    // -----------------------------------------------------------------------
    pub fn set_wal_rotation_bytes(&self, n: u64) {
        self.inner.wal.lock().unwrap().rotation_bytes = n;
    }

    pub fn wal_segment_count(&self) -> usize {
        self.inner.wal.lock().unwrap().segments.len()
    }
}

// ---------------------------------------------------------------------------
// Background apply worker
// ---------------------------------------------------------------------------

fn notify_worker(notify: &PendingNotify) {
    *notify.0.lock().unwrap() = true;
    notify.1.notify_one();
}

fn apply_worker_loop(
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
pub(crate) fn apply_entry(
    inner: &CollectionInner,
    state: &mut ApplyState,
    record: &WalRecord,
) -> Result<()> {
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

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn search_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
    include_metadata: bool,
    include_vector: bool,
) -> Result<Vec<SearchHit>> {
    let dimension = snap.schema.dimension;
    if query.len() != dimension {
        bail!(
            "dimension mismatch: expected {}, got {}",
            dimension,
            query.len()
        );
    }
    let storage = &*inner.storage;
    let id = &inner.id;

    let sealed: Vec<&Arc<SealedSegment>> = snap
        .segs
        .iter()
        .filter_map(|s| match s {
            SegmentSnapshot::Sealed(ss) => Some(ss),
            _ => None,
        })
        .collect();
    let writable = snap.segs.iter().find_map(|s| match s {
        SegmentSnapshot::Writable(w) => Some(w),
        _ => None,
    });

    let (sealed_cands, ws_cands) = rayon::join(
        || {
            sealed
                .par_iter()
                .filter(|s| s.num_vectors() > 0)
                .flat_map(|seg| {
                    let sid = seg.seg_id();
                    seg.search(query, k)
                        .unwrap_or_default()
                        .into_iter()
                        .map(move |(id, dist)| (sid, id, dist))
                        .collect::<Vec<_>>()
                })
                .collect()
        },
        || {
            writable
                .map(|w| {
                    let ws = w.read().unwrap();
                    let sid = ws.seg_id();
                    let hits = if ws.num_vectors() > 0 {
                        ws.search(query, k).unwrap_or_default()
                    } else {
                        vec![]
                    };
                    drop(ws);
                    hits.into_iter()
                        .map(|(id, dist)| (sid, id, dist))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        },
    );
    let mut candidates: Vec<(SegId, InternalId, f32)> = sealed_cands;
    candidates.extend(ws_cands);
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(CmpOrdering::Equal));

    // Resolve tombstones, doc_ids, and metadata in a single read transaction.
    let keys: Vec<(SegId, InternalId)> = candidates.iter().map(|&(s, i, _)| (s, i)).collect();
    let resolved = storage.resolve_candidates(id, &keys, include_metadata)?;

    let mut hits = Vec::new();
    for ((seg_id, internal_id, distance), resolved) in candidates.iter().zip(resolved.into_iter()) {
        if hits.len() >= k {
            break;
        }
        let Some(r) = resolved else { continue };
        let vector = if include_vector {
            let from_sealed = sealed
                .iter()
                .find(|s| s.seg_id() == *seg_id)
                .map(|s| s.read_vector(*internal_id, dimension))
                .transpose()?;
            if from_sealed.is_some() {
                from_sealed
            } else if let Some(w) = writable {
                let ws = w.read().unwrap();
                if ws.seg_id() == *seg_id {
                    Some(ws.read_vector(*internal_id, dimension)?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: *distance,
            metadata: r.metadata,
            vector,
        });
    }
    Ok(hits)
}

#[cfg(feature = "testing")]
fn search_exact_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
) -> Result<Vec<SearchHit>> {
    let dimension = snap.schema.dimension;
    if query.len() != dimension {
        bail!(
            "dimension mismatch: expected {}, got {}",
            dimension,
            query.len()
        );
    }
    let storage = &*inner.storage;
    let id = &inner.id;
    let metric = &snap.schema.metric;

    // Load all tombstones in one read transaction instead of per-vector lookups.
    let tombstones = storage.load_tombstone_set(id)?;

    struct HeapEntry {
        distance: f32,
        seg_id: SegId,
        internal_id: InternalId,
    }

    let sealed: Vec<&Arc<SealedSegment>> = snap
        .segs
        .iter()
        .filter_map(|s| match s {
            SegmentSnapshot::Sealed(ss) => Some(ss),
            _ => None,
        })
        .collect();
    let writable = snap.segs.iter().find_map(|s| match s {
        SegmentSnapshot::Writable(w) => Some(w),
        _ => None,
    });

    let (sealed_cands, ws_cands): (Vec<HeapEntry>, Vec<HeapEntry>) = rayon::join(
        || {
            sealed
                .par_iter()
                .flat_map(|seg| {
                    let sid = seg.seg_id();
                    (0..seg.num_vectors() as u32)
                        .map(InternalId::from_u32)
                        .filter_map(|internal_id| {
                            if tombstones.contains(&(sid, internal_id)) {
                                return None;
                            }
                            let vector = seg.read_vector(internal_id, dimension).ok()?;
                            let distance = compute_distance(metric, query, &vector);
                            Some(HeapEntry {
                                distance,
                                seg_id: sid,
                                internal_id,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        },
        || {
            writable
                .map(|w| {
                    let ws = w.read().unwrap();
                    let sid = ws.seg_id();
                    let n = ws.num_vectors() as u32;
                    let entries: Vec<HeapEntry> = (0..n)
                        .map(InternalId::from_u32)
                        .filter_map(|internal_id| {
                            if tombstones.contains(&(sid, internal_id)) {
                                return None;
                            }
                            let vector = ws.read_vector(internal_id, dimension).ok()?;
                            let distance = compute_distance(metric, query, &vector);
                            Some(HeapEntry {
                                distance,
                                seg_id: sid,
                                internal_id,
                            })
                        })
                        .collect();
                    drop(ws);
                    entries
                })
                .unwrap_or_default()
        },
    );
    let mut candidates = sealed_cands;
    candidates.extend(ws_cands);

    candidates.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(CmpOrdering::Equal)
    });
    let top_k: Vec<HeapEntry> = candidates.into_iter().take(k).collect();

    // Resolve doc_ids in a single read transaction.
    let keys: Vec<(SegId, InternalId)> = top_k.iter().map(|e| (e.seg_id, e.internal_id)).collect();
    let resolved = storage.resolve_candidates(id, &keys, false)?;

    let mut hits = Vec::with_capacity(top_k.len());
    for (entry, r) in top_k.iter().zip(resolved.into_iter()) {
        let Some(r) = r else { continue };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: entry.distance,
            metadata: None,
            vector: None,
        });
    }
    Ok(hits)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read_up_to(r: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut total = 0;
    while total < buf.len() {
        match r.read(&mut buf[total..])? {
            0 => break,
            n => total += n,
        }
    }
    Ok(total)
}
