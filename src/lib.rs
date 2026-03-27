mod context;
pub mod dataset;
pub mod eval;
mod segment;
mod storage;
pub mod types;

use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

const DB_NAME: &str = "nebel.redb";
const INGEST_BATCH_SIZE: usize = 2048;

use context::CollectionContext;
use segment::{Segment, WritableSegment};
use storage::Storage;
use types::{
    CollectionId, CollectionSchema, Manifest, SearchHit, SegId, SegmentMeta, SegmentState,
};

pub struct Nebel {
    base_dir: PathBuf,
    storage: Storage,
    collections: HashMap<CollectionId, CollectionContext>,
}

impl Nebel {
    /// Open (or create) a Nebel database rooted at `base_dir`.
    pub fn open(base_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir)?;
        let db_path = base_dir.join(DB_NAME);
        let storage = Storage::open(&db_path)?;
        Ok(Self {
            base_dir,
            storage,
            collections: HashMap::new(),
        })
    }

    /// Load an existing collection into memory, loading indices for all active
    /// segments.
    ///
    /// This must be called after [`Nebel::open`] to make a previously-created
    /// collection available for search and mutation.
    pub fn load_collection(&mut self, id: &CollectionId) -> Result<()> {
        let schema = self
            .storage
            .get_collection(id)?
            .ok_or_else(|| anyhow!("collection '{}' not found", id))?;
        let manifest = self
            .storage
            .get_manifest(id)?
            .ok_or_else(|| anyhow!("manifest not found for '{}'", id))?;

        let mut segments = HashMap::new();
        for &seg_id in &manifest.active_segments {
            let meta = self
                .storage
                .get_segment(id, seg_id)?
                .ok_or_else(|| anyhow!("segment {} not found for '{}'", seg_id, id))?;
            let dir = self.seg_dir(id, seg_id);
            let seg = match meta.state {
                SegmentState::Writable => Segment::Writable(WritableSegment::open(
                    seg_id,
                    dir,
                    schema.dimension,
                    meta.num_vectors,
                    &schema.metric,
                    &schema.segment_params,
                )?),
                SegmentState::Sealed => Segment::Sealed(segment::SealedSegment::open(
                    seg_id,
                    dir,
                    meta.num_vectors,
                    &schema.metric,
                    schema.segment_params.ef_search,
                )?),
            };
            segments.insert(seg_id, seg);
        }

        self.collections.insert(
            id.clone(),
            CollectionContext {
                schema,
                manifest,
                segments,
            },
        );
        Ok(())
    }

    /// Create a new collection from a [`CollectionSchema`].
    ///
    /// The collection id is taken from `schema.name`. Use [`CollectionSchema::new`]
    /// to build a schema with default [`SegmentParams`].
    pub fn create_collection(&mut self, schema: CollectionSchema) -> Result<()> {
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
        let seg = WritableSegment::create(
            SegId::FIRST,
            self.seg_dir(&id, SegId::FIRST),
            &schema.metric,
            &schema.segment_params,
        )?;
        let mut segments = HashMap::new();
        segments.insert(SegId::FIRST, Segment::Writable(seg));
        self.collections.insert(
            id,
            CollectionContext {
                schema,
                manifest,
                segments,
            },
        );
        Ok(())
    }

    /// Create a new writable segment for `collection`, sealing the previous
    /// writable segment and updating the manifest.
    pub fn add_writable_segment(&mut self, id: &CollectionId) -> Result<SegId> {
        let ctx = self
            .collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))?;
        let old_id = ctx.manifest.writable_segment;
        let new_id = ctx.manifest.next_seg_id;
        let new_dir = self.seg_dir(id, new_id);

        let ctx = self.collections.get_mut(id).expect("collection present");

        // Seal the old writable segment.
        let old_seg = ctx
            .segments
            .remove(&old_id)
            .ok_or_else(|| anyhow!("old writable segment not found"))?;
        let (sealed, num_vectors) = match old_seg {
            Segment::Writable(w) => {
                let nv = w.num_vectors();
                let sealed = w.seal()?;
                (sealed, nv)
            }
            Segment::Sealed(_) => bail!("writable segment is already sealed"),
        };
        ctx.segments.insert(old_id, Segment::Sealed(sealed));

        let old_meta = SegmentMeta {
            seg_id: old_id,
            num_vectors,
            state: SegmentState::Sealed,
        };
        self.storage.put_segment(id, &old_meta)?;

        // Create the new writable segment.
        self.storage.put_segment(id, &SegmentMeta::new(new_id))?;
        let metric = self.collections[id].schema.metric.clone();
        let params = self.collections[id].schema.segment_params.clone();
        let seg = WritableSegment::create(new_id, new_dir, &metric, &params)?;

        let ctx = self.collections.get_mut(id).expect("collection present");
        ctx.segments.insert(new_id, Segment::Writable(seg));
        ctx.manifest.active_segments.push(new_id);
        ctx.manifest.writable_segment = new_id;
        ctx.manifest.next_seg_id = new_id.next();
        self.storage.put_manifest(id, &ctx.manifest)?;

        Ok(new_id)
    }

    /// Ingest vectors from a binary file into `collection`.
    ///
    /// File format: contiguous f32 little-endian values with no header.
    /// Each record is exactly `dimension * 4` bytes.
    /// Doc IDs are auto-assigned as `"doc_0"`, `"doc_1"`, …
    ///
    /// Vectors are read in batches of [`INGEST_BATCH_SIZE`] and inserted into
    /// the HNSW index with `parallel_insert`.
    ///
    /// Returns the number of vectors ingested.
    pub fn ingest_file(&mut self, id: &CollectionId, file_path: impl AsRef<Path>) -> Result<usize> {
        let dimension = self.get_ctx(id)?.schema.dimension;
        let record_size = dimension * 4;

        let mut file = fs::File::open(file_path)?;
        let mut buf = vec![0u8; record_size * INGEST_BATCH_SIZE];
        let mut count = 0;

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
            if self.get_ctx(id)?.ensure_writable_capacity() {
                self.add_writable_segment(id)?;
            }
            let ctx = self.collections.get_mut(id).expect("collection present");
            ctx.insert_vector_batch(&mut self.storage, &doc_ids, &vectors)?;
            count += vectors.len();
        }
        Ok(count)
    }

    /// Insert or replace a document.
    ///
    /// If `doc_id` already exists its old vector is tombstoned before the new
    /// one is appended.
    pub fn upsert(
        &mut self,
        id: &CollectionId,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        if self.get_ctx(id)?.ensure_writable_capacity() {
            self.add_writable_segment(id)?;
        }
        let ctx = self.collections.get_mut(id).expect("collection present");
        ctx.upsert(&mut self.storage, doc_id, vector, metadata)
    }

    /// Mark a document as deleted (tombstone). Returns an error if not found.
    pub fn delete(&mut self, id: &CollectionId, doc_id: &str) -> Result<()> {
        let ctx = self
            .collections
            .get_mut(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))?;
        ctx.delete(&mut self.storage, doc_id)
    }

    /// Overwrite the stored metadata for `doc_id` without touching its vector.
    pub fn update_metadata(
        &mut self,
        id: &CollectionId,
        doc_id: &str,
        metadata: Value,
    ) -> Result<()> {
        let ctx = self
            .collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))?;
        ctx.update_metadata(&mut self.storage, doc_id, metadata)
    }

    /// Return the `k` nearest neighbours to `query` across all active segments,
    /// filtering tombstoned entries.
    ///
    /// Scores are raw L2 distances (lower = closer).
    pub fn search(
        &self,
        id: &CollectionId,
        query: &[f32],
        k: usize,
        include_metadata: bool,
        include_vector: bool,
    ) -> Result<Vec<SearchHit>> {
        let ctx = self
            .collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))?;
        ctx.search(&self.storage, query, k, include_metadata, include_vector)
    }

    /// Brute-force exact search (for testing recall of the HNSW index).
    #[doc(hidden)]
    pub fn search_exact(
        &self,
        id: &CollectionId,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchHit>> {
        let ctx = self
            .collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))?;
        ctx.search_exact(&self.storage, query, k)
    }

    // --- internal helpers ---

    fn get_ctx(&self, id: &CollectionId) -> Result<&CollectionContext> {
        self.collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))
    }

    fn seg_dir(&self, id: &CollectionId, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(id.as_str())
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }
}

/// Read up to `buf.len()` bytes, stopping at EOF. Returns bytes read.
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
