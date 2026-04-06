use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

use anyhow::{Result, anyhow, bail};

use crate::{
    handle::{CollectionHandle, CollectionInner, apply_entry},
    segment::{SealedSegment, WritableSegment},
    storage::Storage,
    types::{CollectionId, CollectionSchema, Manifest, SegId, SegmentMeta, SegmentState},
    wal::{Wal, WalRecord},
};

const DB_NAME: &str = "nebel.redb";

pub struct Db {
    pub(crate) base_dir: PathBuf,
    pub(crate) storage: Arc<Storage>,
    collections: Mutex<HashMap<CollectionId, CollectionHandle>>,
}

impl Db {
    /// Open (or create) a Nebel database rooted at `base_dir`.
    pub fn open(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir)?;
        let db_path = base_dir.join(DB_NAME);
        let storage = Arc::new(Storage::open(&db_path)?);
        Ok(Self {
            base_dir,
            storage,
            collections: Mutex::new(HashMap::new()),
        })
    }

    /// Create a new collection from a [`CollectionSchema`]. Returns a cloneable handle.
    pub fn create_collection(&self, schema: CollectionSchema) -> Result<CollectionHandle> {
        let id = schema.name.clone();
        if self.storage.get_collection(&id)?.is_some() {
            bail!("collection '{}' already exists", id);
        }
        let manifest = Manifest {
            active_segments: vec![SegId::FIRST],
            writable_segment: SegId::FIRST,
            next_seg_id: SegId::FIRST.next(),
        };
        self.storage
            .put_new_collection(&schema, &manifest, &SegmentMeta::new(SegId::FIRST))?;

        // Ensure collection directory exists for WAL etc.
        fs::create_dir_all(self.base_dir.join(id.as_str()))?;

        // Create initial writable segment.
        let seg_dir = self.seg_dir(&id, SegId::FIRST);
        let ws = WritableSegment::create(
            SegId::FIRST,
            seg_dir,
            &schema.metric,
            &schema.segment_params,
        )?;
        let ws_arc = Arc::new(RwLock::new(ws));
        let schema_arc = Arc::new(schema);
        let inner = CollectionInner::new(
            id.clone(),
            Arc::clone(&self.storage),
            self.base_dir.clone(),
            manifest,
            vec![],
            ws_arc,
            0,
            schema_arc,
        )?;
        let handle = CollectionHandle::from_arc(inner);
        self.collections.lock().unwrap().insert(id, handle.clone());
        Ok(handle)
    }

    /// Return a handle to an existing collection, loading it from disk if needed.
    pub fn collection(&self, name: &str) -> Result<CollectionHandle> {
        let id = CollectionId::new(name);
        {
            let guard = self.collections.lock().unwrap();
            if let Some(h) = guard.get(&id) {
                return Ok(h.clone());
            }
        }
        let handle = self.load_collection_inner(&id)?;
        self.collections.lock().unwrap().insert(id, handle.clone());
        Ok(handle)
    }

    pub(crate) fn load_collection_inner(&self, id: &CollectionId) -> Result<CollectionHandle> {
        let schema = self
            .storage
            .get_collection(id)?
            .ok_or_else(|| anyhow!("collection '{}' not found", id))?;
        let manifest = self
            .storage
            .get_manifest(id)?
            .ok_or_else(|| anyhow!("manifest not found for '{}'", id))?;

        let mut sealed_segs: Vec<Arc<SealedSegment>> = Vec::new();
        let mut writable_seg_opt: Option<Arc<RwLock<WritableSegment>>> = None;

        for &seg_id in &manifest.active_segments {
            let meta = self
                .storage
                .get_segment(id, seg_id)?
                .ok_or_else(|| anyhow!("segment {} not found for '{}'", seg_id, id))?;
            let dir = self.seg_dir(id, seg_id);
            match meta.state {
                SegmentState::Writable => {
                    let ws = WritableSegment::open(
                        seg_id,
                        dir,
                        schema.dimension,
                        meta.num_vectors,
                        &schema.metric,
                        &schema.segment_params,
                    )?;
                    writable_seg_opt = Some(Arc::new(RwLock::new(ws)));
                }
                SegmentState::Sealed => {
                    let ss = SealedSegment::open(
                        seg_id,
                        dir,
                        meta.num_vectors,
                        &schema.metric,
                        schema.segment_params.ef_search,
                        meta.level,
                        meta.tombstone_count,
                    )?;
                    sealed_segs.push(Arc::new(ss));
                }
            }
        }

        let writable_seg =
            writable_seg_opt.ok_or_else(|| anyhow!("no writable segment found for '{}'", id))?;

        let schema_arc = Arc::new(schema);
        let applied_seq_stored = self.storage.get_applied_seq(id)?.unwrap_or(0);

        // --- Recovery ---
        //
        // Each WAL segment is applied and deleted individually.  The segment file is
        // only removed after its records are applied and applied_seq is persisted, so a
        // crash at any point leaves surviving files for the next open to re-read and
        // re-apply (records already applied are skipped via the per-record seq check).

        // Step 1: Create CollectionInner.
        // Wal::open_dir marks every pre-existing segment Closed and opens a fresh
        // Active segment at max_id.next(), so new writes never touch recovery files.
        let inner = CollectionInner::new(
            id.clone(),
            Arc::clone(&self.storage),
            self.base_dir.clone(),
            manifest,
            sealed_segs,
            writable_seg,
            applied_seq_stored,
            schema_arc,
        )?;

        // Step 2: Process each pre-existing (Closed) WAL segment in ascending ID order.
        // Snapshot the list first; new writes only go to the Active segment.
        let recovery_segs = inner.wal.lock().unwrap().recovery_segments();
        for (wal_id, path) in recovery_segs {
            let records = Wal::read_segment(&path)?;
            Self::apply_and_persist(&records, &inner)?;
            inner.wal.lock().unwrap().remove_segment(wal_id)?;
        }

        // Step 4: Sync sequence counters with the final applied_seq.
        {
            let state = inner.apply_state.lock().unwrap();
            let applied = state.applied_seq;
            if applied > applied_seq_stored {
                inner.wal.lock().unwrap().reset_next_seq(applied + 1);
                inner
                    .durable_seq
                    .store(applied, std::sync::atomic::Ordering::Release);
                inner.publish_snapshot(&state);
                drop(state);
                inner
                    .visible_seq
                    .store(applied + 1, std::sync::atomic::Ordering::Release);
            }
        }

        Ok(CollectionHandle::from_arc(inner))
    }

    /// Apply records (skipping already-applied ones by seq).
    fn apply_and_persist(records: &[WalRecord], inner: &CollectionInner) -> Result<()> {
        let mut state = inner.apply_state.lock().unwrap();
        for record in records {
            if record.seq <= state.applied_seq {
                continue;
            }
            apply_entry(inner, &mut state, record)?;
        }
        Ok(())
    }

    fn seg_dir(&self, id: &CollectionId, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(id.as_str())
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }
}
