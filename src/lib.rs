mod segment;
mod storage;
pub mod types;

use std::{
    cmp::Ordering,
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

const DB_NAME: &str = "nebel.redb";
const INGEST_BATCH_SIZE: usize = 512;
const SEGMENT_CAPACITY: usize = 100_000;

use segment::Segment;
use storage::Storage;
use types::{
    CollectionId, CollectionSchema, Manifest, Metric, SearchHit, SegId, SegmentMeta, VectorEntry,
};

pub struct Nebel {
    base_dir: PathBuf,
    storage: Storage,
    /// In-memory segments per collection, keyed by (collection, seg_id).
    segments: HashMap<CollectionId, HashMap<SegId, Segment>>,
    /// Cached manifests per collection.
    manifests: HashMap<CollectionId, Manifest>,
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
            segments: HashMap::new(),
            manifests: HashMap::new(),
        })
    }

    /// Load an existing collection into memory, rebuilding the HNSW indices
    /// from the persisted vectors for all active segments.
    ///
    /// This must be called after [`Nebel::open`] to make a previously-created
    /// collection available for search and mutation.
    pub fn load_collection(&mut self, id: &CollectionId) -> Result<()> {
        let schema = self.get_schema(id)?;
        let manifest = self
            .storage
            .get_manifest(id)?
            .ok_or_else(|| anyhow!("manifest not found for '{}'", id))?;
        for &seg_id in &manifest.active_segments {
            self.load_segment(id, seg_id, schema.dimension)?;
        }
        self.manifests.insert(id.clone(), manifest);
        Ok(())
    }

    /// Create a new collection with the given `name`, vector `dimension`, and distance `metric`.
    pub fn create_collection(
        &mut self,
        id: &CollectionId,
        dimension: usize,
        metric: Metric,
    ) -> Result<()> {
        if self.storage.get_collection(id)?.is_some() {
            bail!("collection '{}' already exists", id);
        }
        let schema = CollectionSchema {
            name: id.clone(),
            dimension,
            metric,
        };
        let manifest = Manifest {
            active_segments: vec![SegId::FIRST],
            writable_segment: SegId::FIRST,
            next_seg_id: SegId::FIRST.next(),
        };
        self.storage
            .put_new_collection(&schema, &manifest, &SegmentMeta::new(SegId::FIRST))?;
        let seg = self.create_segment(id, SegId::FIRST, dimension)?;
        self.segments
            .entry(id.clone())
            .or_default()
            .insert(SegId::FIRST, seg);
        self.manifests.insert(id.clone(), manifest);
        Ok(())
    }

    /// Create a new writable segment for `collection`, updating the manifest.
    ///
    /// The previous writable segment remains in `active_segments` (for search)
    /// but new writes will go to the returned segment.
    pub fn add_writable_segment(&mut self, id: &CollectionId) -> Result<SegId> {
        let schema = self.get_schema(id)?;
        let manifest = self.get_manifest(id)?;
        let new_id = manifest.next_seg_id;

        self.storage
            .put_segment(id, &SegmentMeta::new(new_id))?;

        let seg = self.create_segment(id, new_id, schema.dimension)?;
        self.segments
            .entry(id.clone())
            .or_default()
            .insert(new_id, seg);

        let mut manifest = manifest;
        manifest.active_segments.push(new_id);
        manifest.writable_segment = new_id;
        manifest.next_seg_id = new_id.next();
        self.storage.put_manifest(id, &manifest)?;
        self.manifests.insert(id.clone(), manifest);

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
    pub fn ingest_file(
        &mut self,
        id: &CollectionId,
        file_path: impl AsRef<Path>,
    ) -> Result<usize> {
        let schema = self.get_schema(id)?;
        let record_size = schema.dimension * 4;

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
            let manifest = self.writable_manifest(id, schema.dimension)?;
            self.insert_vector_batch(id, &schema, &manifest, &doc_ids, &vectors)?;
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
        let schema = self.get_schema(id)?;
        if vector.len() != schema.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                schema.dimension,
                vector.len()
            );
        }
        if let Some(loc) = self.storage.get_doc_location(id, doc_id)? {
            self.storage
                .set_tombstone(id, loc.seg_id, loc.internal_id)?;
        }
        let manifest = self.writable_manifest(id, schema.dimension)?;
        self.insert_vector(id, &schema, &manifest, doc_id, vector, metadata)?;
        Ok(())
    }

    /// Mark a document as deleted (tombstone). Returns an error if not found.
    pub fn delete(&mut self, id: &CollectionId, doc_id: &str) -> Result<()> {
        let loc = self
            .storage
            .get_doc_location(id, doc_id)?
            .ok_or_else(|| anyhow!("document '{}' not found", doc_id))?;
        self.storage
            .set_tombstone(id, loc.seg_id, loc.internal_id)?;
        self.storage.remove_doc_location(id, doc_id)?;
        Ok(())
    }

    /// Overwrite the stored metadata for `doc_id` without touching its vector.
    pub fn update_metadata(
        &mut self,
        id: &CollectionId,
        doc_id: &str,
        metadata: Value,
    ) -> Result<()> {
        let loc = self
            .storage
            .get_doc_location(id, doc_id)?
            .ok_or_else(|| anyhow!("document '{}' not found", doc_id))?;
        self.storage
            .put_metadata(id, loc.seg_id, loc.internal_id, &metadata)?;
        Ok(())
    }

    /// Return the `k` nearest neighbours to `query` across all active segments,
    /// filtering tombstoned entries.
    ///
    /// Scores are raw L2 distances (lower = closer).
    pub fn search(
        &mut self,
        id: &CollectionId,
        query: &[f32],
        k: usize,
        include_metadata: bool,
        include_vector: bool,
    ) -> Result<Vec<SearchHit>> {
        let schema = self.get_schema(id)?;
        let manifest = self.get_manifest(id)?;
        if query.len() != schema.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                schema.dimension,
                query.len()
            );
        }

        // Collect candidates from all active segments.
        let mut candidates: Vec<(SegId, u32, f32)> = Vec::new();
        for &seg_id in &manifest.active_segments {
            let seg = self.load_segment(id, seg_id, schema.dimension)?;
            if seg.num_vectors == 0 {
                continue;
            }
            let raw = seg.search(query, k)?;
            for (internal_id, distance) in raw {
                candidates.push((seg_id, internal_id, distance));
            }
        }

        // Sort by distance ascending, take top k after filtering tombstones.
        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        let mut hits = Vec::new();
        for (seg_id, internal_id, distance) in candidates {
            if hits.len() >= k {
                break;
            }
            if self.storage.is_tombstoned(id, seg_id, internal_id)? {
                continue;
            }
            let doc_id = self.reverse_lookup(id, seg_id, internal_id)?;
            let metadata = if include_metadata {
                self.storage.get_metadata(id, seg_id, internal_id)?
            } else {
                None
            };
            let vector = if include_vector {
                let seg = self.load_segment(id, seg_id, schema.dimension)?;
                Some(seg.read_vector(internal_id)?)
            } else {
                None
            };
            hits.push(SearchHit {
                doc_id,
                score: distance,
                metadata,
                vector,
            });
        }

        Ok(hits)
    }

    // --- internal helpers ---

    fn get_schema(&self, id: &CollectionId) -> Result<CollectionSchema> {
        self.storage
            .get_collection(id)?
            .ok_or_else(|| anyhow!("collection '{}' not found", id))
    }

    fn get_manifest(&self, id: &CollectionId) -> Result<Manifest> {
        if let Some(m) = self.manifests.get(id) {
            return Ok(m.clone());
        }
        self.storage
            .get_manifest(id)?
            .ok_or_else(|| anyhow!("manifest not found for '{}'", id))
    }

    /// Return the current manifest, rotating to a new writable segment if
    /// the current one has reached [`SEGMENT_CAPACITY`].
    fn writable_manifest(&mut self, id: &CollectionId, dimension: usize) -> Result<Manifest> {
        let manifest = self.get_manifest(id)?;
        let seg = self.load_segment(id, manifest.writable_segment, dimension)?;
        if seg.num_vectors >= SEGMENT_CAPACITY {
            self.add_writable_segment(id)?;
            return self.get_manifest(id);
        }
        Ok(manifest)
    }

    fn create_segment(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        dimension: usize,
    ) -> Result<Segment> {
        let dir = self.seg_dir(id, seg_id);
        Segment::create(seg_id, dir, dimension)
    }

    fn seg_dir(&self, id: &CollectionId, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(id.as_str())
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }

    fn load_segment(
        &mut self,
        id: &CollectionId,
        seg_id: SegId,
        dimension: usize,
    ) -> Result<&mut Segment> {
        // Compute dir and load from storage before borrowing self.segments.
        let needs_load = !self
            .segments
            .get(id)
            .is_some_and(|m| m.contains_key(&seg_id));
        if needs_load {
            let meta = self
                .storage
                .get_segment(id, seg_id)?
                .ok_or_else(|| anyhow!("segment {} not found for '{}'", seg_id, id))?;
            let dir = self.seg_dir(id, seg_id);
            let seg = Segment::open(seg_id, dir, dimension, meta.num_vectors)?;
            self.segments
                .entry(id.clone())
                .or_default()
                .insert(seg_id, seg);
        }
        Ok(self
            .segments
            .get_mut(id)
            .expect("collection present")
            .get_mut(&seg_id)
            .expect("segment present"))
    }

    fn insert_vector(
        &mut self,
        id: &CollectionId,
        schema: &CollectionSchema,
        manifest: &Manifest,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let seg_id = manifest.writable_segment;
        let seg = self.load_segment(id, seg_id, schema.dimension)?;
        let internal_id = seg.insert(vector)?;
        let num_vectors = seg.num_vectors;

        self.write_vector_entries(
            id,
            seg_id,
            num_vectors,
            &[VectorEntry {
                doc_id,
                internal_id,
                metadata: metadata.as_ref(),
            }],
        )
    }

    fn insert_vector_batch(
        &mut self,
        id: &CollectionId,
        schema: &CollectionSchema,
        manifest: &Manifest,
        doc_ids: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        let seg_id = manifest.writable_segment;
        let seg = self.load_segment(id, seg_id, schema.dimension)?;
        let internal_ids = seg.insert_batch(vectors)?;
        let num_vectors = seg.num_vectors;

        let entries: Vec<VectorEntry<'_>> = doc_ids
            .iter()
            .zip(internal_ids.iter())
            .map(|(doc_id, &internal_id)| VectorEntry {
                doc_id: doc_id.as_str(),
                internal_id,
                metadata: None,
            })
            .collect();

        self.write_vector_entries(id, seg_id, num_vectors, &entries)
    }

    /// Write segment metadata, doc locations, reverse-doc mappings, and
    /// optional per-vector metadata in a single redb transaction.
    fn write_vector_entries(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        num_vectors: usize,
        entries: &[VectorEntry<'_>],
    ) -> Result<()> {
        let seg_meta = SegmentMeta {
            seg_id,
            num_vectors,
        };
        self.storage
            .write_vector_entries(id, &seg_meta, entries)
    }

    fn reverse_lookup(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        internal_id: u32,
    ) -> Result<String> {
        self.storage
            .get_reverse_doc(id, seg_id, internal_id)?
            .ok_or_else(|| {
                anyhow!(
                    "reverse mapping not found for ({}, {})",
                    seg_id,
                    internal_id
                )
            })
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
