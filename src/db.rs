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
    wal::Wal,
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
                    )?;
                    sealed_segs.push(Arc::new(ss));
                }
            }
        }

        let writable_seg =
            writable_seg_opt.ok_or_else(|| anyhow!("no writable segment found for '{}'", id))?;

        let schema_arc = Arc::new(schema);
        let applied_seq_stored = self.storage.get_applied_seq(id)?.unwrap_or(0);

        // --- Recovery: read all WAL records, wipe WAL, then replay ---

        let col_dir = self.base_dir.join(id.as_str());
        let old_wal_path = col_dir.join("wal.log"); // legacy single-file format
        let wal_dir = col_dir.join("wal");           // current segmented format

        // Step 1: Read all pending WAL records before touching the directory.
        let all_records = if old_wal_path.exists() && !wal_dir.exists() {
            Wal::read_legacy(&old_wal_path)?
        } else {
            Wal::read_all_from_dir(&wal_dir)?
        };
        let pending: Vec<_> = all_records
            .into_iter()
            .filter(|r| r.seq > applied_seq_stored)
            .collect();

        // Step 2: Wipe all WAL files so CollectionInner::new() opens a clean directory.
        if old_wal_path.exists() {
            fs::remove_file(&old_wal_path)?;
        }
        if wal_dir.exists() {
            fs::remove_dir_all(&wal_dir)?;
        }

        // Step 3: Create CollectionInner — opens fresh wal/ dir, creates 000001.log.
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

        // Step 4: Replay pending WAL records against the freshly-opened inner.
        if !pending.is_empty() {
            let mut state = inner.apply_state.lock().unwrap();
            for record in &pending {
                apply_entry(&inner, &mut state, record)?;
            }
            self.storage.put_applied_seq(id, state.applied_seq)?;
            let new_next = state.applied_seq + 1;
            inner
                .next_seq
                .store(new_next, std::sync::atomic::Ordering::Relaxed);
            // durable_seq must also advance so the worker doesn't scan empty WAL
            // looking for records that were already replayed here.
            inner
                .durable_seq
                .store(state.applied_seq, std::sync::atomic::Ordering::Release);
            let visible = state.applied_seq + 1;
            inner.publish_snapshot(&state);
            drop(state);
            inner
                .visible_seq
                .store(visible, std::sync::atomic::Ordering::Release);
        }

        Ok(CollectionHandle::from_arc(inner))
    }

    fn seg_dir(&self, id: &CollectionId, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(id.as_str())
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }
}
