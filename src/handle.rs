use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};

use std::collections::HashMap;

use anyhow::{Result, bail};

#[cfg(feature = "testing")]
use crate::{
    apply::seal_and_new_segment,
    search::search_exact_snapshot,
    types::{Level, SegmentMeta, SegmentState, VectorEntry},
};
use crate::{
    apply::{ApplyState, ApplyWorkerGuard, PendingNotify, apply_worker_loop, notify_worker},
    compaction::CompactionWorkerGuard,
    metadata::MetadataValue,
    search::search_snapshot,
    segment::{SealedSegment, WritableSegment},
    snapshot::{CollectionSnapshot, SegmentSnapshot},
    storage::Storage,
    types::{CollectionId, CollectionSchema, Manifest, SearchHit, SegId, WriteToken},
    wal::{Wal, WalOp},
};
#[cfg(feature = "testing")]
use std::{fs, io::Read, path::Path};

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

/// One entry for [`CollectionHandle::upsert_batch`]: `(doc_id, vector, metadata)`.
pub type UpsertEntry<'a> = (&'a str, &'a [f32], Option<HashMap<String, MetadataValue>>);

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
        metadata: Option<HashMap<String, MetadataValue>>,
    ) -> Result<WriteToken> {
        let inner = &self.inner;
        if vector.len() != inner.schema.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                inner.schema.dimension,
                vector.len()
            );
        }
        if let Some(ref m) = metadata {
            match inner.schema.metadata_schema.as_ref() {
                Some(ms) => {
                    ms.validate(m)?;
                }
                None => bail!("collection has no metadata schema; metadata must be None"),
            }
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
    /// `entries` must contain at most `SegmentParams::insert_batch_size` entries (default 2048).
    /// Larger slices are rejected with an error.
    ///
    /// # Visibility
    ///
    /// The returned [`WriteToken`] can be passed to [`CollectionHandle::wait_visible`] to block
    /// until all vectors in the batch have been applied and are visible to search.
    pub fn upsert_batch(&self, entries: &[UpsertEntry<'_>]) -> Result<WriteToken> {
        if entries.is_empty() {
            return Ok(WriteToken(self.inner.durable_seq.load(Ordering::Acquire)));
        }
        let inner = &self.inner;
        let batch_max = inner.schema.segment_params.insert_batch_size;
        if entries.len() > batch_max {
            bail!(
                "batch size {} exceeds maximum of {}",
                entries.len(),
                batch_max
            );
        }
        let expected_dim = inner.schema.dimension;
        for (i, (_, vector, metadata)) in entries.iter().enumerate() {
            if vector.len() != expected_dim {
                bail!(
                    "dimension mismatch at index {}: expected {}, got {}",
                    i,
                    expected_dim,
                    vector.len()
                );
            }
            if let Some(m) = metadata {
                match inner.schema.metadata_schema.as_ref() {
                    Some(ms) => {
                        ms.validate(m)?;
                    }
                    None => bail!("collection has no metadata schema; metadata must be None"),
                }
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

    pub fn update_metadata(
        &self,
        doc_id: &str,
        metadata: HashMap<String, MetadataValue>,
    ) -> Result<WriteToken> {
        let inner = &self.inner;
        match inner.schema.metadata_schema.as_ref() {
            Some(ms) => {
                ms.validate(&metadata)?;
            }
            None => bail!("collection has no metadata schema; cannot update metadata"),
        }
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
    pub fn search_exact(
        &self,
        query: &[f32],
        k: usize,
        include_metadata: bool,
    ) -> Result<Vec<SearchHit>> {
        let snap = { Arc::clone(&*self.inner.snapshot.read().unwrap()) };
        search_exact_snapshot(&self.inner, &snap, query, k, include_metadata)
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
        let ingest_batch_size = inner.schema.segment_params.insert_batch_size;
        let mut buf = vec![0u8; record_size * ingest_batch_size];
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
    #[cfg(feature = "testing")]
    pub fn wal_segment_count(&self) -> usize {
        self.inner.wal.lock().unwrap().segments.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "testing")]
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
