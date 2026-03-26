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
const INGEST_BATCH_SIZE: usize = 512;

use segment::Segment;
use storage::Storage;
use types::{CollectionSchema, Metric, SearchHit, SegId, SegmentMeta, VectorEntry};

pub struct Nebel {
    base_dir: PathBuf,
    storage: Storage,
    /// In-memory active segment per collection.
    segments: HashMap<String, Segment>,
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
        })
    }

    /// Create a new collection with the given `name`, vector `dimension`, and distance `metric`.
    pub fn create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        metric: Metric,
    ) -> Result<()> {
        if self.storage.get_collection(name)?.is_some() {
            bail!("collection '{}' already exists", name);
        }
        let schema = CollectionSchema {
            name: name.to_string(),
            dimension,
            metric,
            active_seg_id: SegId::FIRST,
        };
        self.storage
            .put_new_collection(&schema, &SegmentMeta::new(SegId::FIRST))?;
        let seg = self.create_segment(name, SegId::FIRST, dimension)?;
        self.segments.insert(name.to_string(), seg);
        Ok(())
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
    pub fn ingest_file(&mut self, collection: &str, file_path: impl AsRef<Path>) -> Result<usize> {
        let schema = self.get_schema(collection)?;
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
            self.insert_vector_batch(collection, &schema, &doc_ids, &vectors)?;
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
        collection: &str,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let schema = self.get_schema(collection)?;
        if vector.len() != schema.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                schema.dimension,
                vector.len()
            );
        }
        if let Some(loc) = self.storage.get_doc_location(collection, doc_id)? {
            self.storage
                .set_tombstone(collection, loc.seg_id, loc.internal_id)?;
        }
        self.insert_vector(collection, &schema, doc_id, vector, metadata)?;
        Ok(())
    }

    /// Mark a document as deleted (tombstone). Returns an error if not found.
    pub fn delete(&mut self, collection: &str, doc_id: &str) -> Result<()> {
        let loc = self
            .storage
            .get_doc_location(collection, doc_id)?
            .ok_or_else(|| anyhow!("document '{}' not found", doc_id))?;
        self.storage
            .set_tombstone(collection, loc.seg_id, loc.internal_id)?;
        self.storage.remove_doc_location(collection, doc_id)?;
        Ok(())
    }

    /// Overwrite the stored metadata for `doc_id` without touching its vector.
    pub fn update_metadata(
        &mut self,
        collection: &str,
        doc_id: &str,
        metadata: Value,
    ) -> Result<()> {
        let loc = self
            .storage
            .get_doc_location(collection, doc_id)?
            .ok_or_else(|| anyhow!("document '{}' not found", doc_id))?;
        self.storage
            .put_metadata(collection, loc.seg_id, loc.internal_id, &metadata)?;
        Ok(())
    }

    /// Return the `k` nearest neighbours to `query` across all segments,
    /// filtering tombstoned entries.
    ///
    /// Scores are raw L2 distances (lower = closer).
    pub fn search(
        &mut self,
        collection: &str,
        query: &[f32],
        k: usize,
        include_metadata: bool,
        include_vector: bool,
    ) -> Result<Vec<SearchHit>> {
        let schema = self.get_schema(collection)?;
        if query.len() != schema.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                schema.dimension,
                query.len()
            );
        }

        let seg = self.load_segment(collection, schema.active_seg_id, schema.dimension)?;
        if seg.num_vectors == 0 {
            return Ok(vec![]);
        }

        let raw = seg.search(query, k)?;

        let mut hits = Vec::new();
        for (internal_id, distance) in raw {
            if self
                .storage
                .is_tombstoned(collection, schema.active_seg_id, internal_id)?
            {
                continue;
            }
            let doc_id = self.reverse_lookup(collection, schema.active_seg_id, internal_id)?;
            let metadata = if include_metadata {
                self.storage
                    .get_metadata(collection, schema.active_seg_id, internal_id)?
            } else {
                None
            };
            let vector = if include_vector {
                let seg = self.load_segment(collection, schema.active_seg_id, schema.dimension)?;
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

    fn get_schema(&self, collection: &str) -> Result<CollectionSchema> {
        self.storage
            .get_collection(collection)?
            .ok_or_else(|| anyhow!("collection '{}' not found", collection))
    }

    fn create_segment(&self, collection: &str, seg_id: SegId, dimension: usize) -> Result<Segment> {
        let dir = self.seg_dir(collection, seg_id);
        Segment::create(seg_id, dir, dimension)
    }

    fn seg_dir(&self, collection: &str, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(collection)
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }

    fn load_segment(
        &mut self,
        collection: &str,
        seg_id: SegId,
        dimension: usize,
    ) -> Result<&mut Segment> {
        if !self.segments.contains_key(collection) {
            let meta = self
                .storage
                .get_segment(collection, seg_id)?
                .ok_or_else(|| anyhow!("segment {} not found for '{}'", seg_id, collection))?;
            let dir = self.seg_dir(collection, seg_id);
            let seg = Segment::open(seg_id, dir, dimension, meta.num_vectors)?;
            self.segments.insert(collection.to_string(), seg);
        }
        Ok(self
            .segments
            .get_mut(collection)
            .expect("should be present"))
    }

    fn insert_vector(
        &mut self,
        collection: &str,
        schema: &CollectionSchema,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let seg_id = schema.active_seg_id;
        let seg = self.load_segment(collection, seg_id, schema.dimension)?;
        let internal_id = seg.insert(vector)?;
        let num_vectors = seg.num_vectors;

        self.write_vector_entries(
            collection,
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
        collection: &str,
        schema: &CollectionSchema,
        doc_ids: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        let seg_id = schema.active_seg_id;
        let seg = self.load_segment(collection, seg_id, schema.dimension)?;
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

        self.write_vector_entries(collection, seg_id, num_vectors, &entries)
    }

    /// Write segment metadata, doc locations, reverse-doc mappings, and
    /// optional per-vector metadata in a single redb transaction.
    fn write_vector_entries(
        &self,
        collection: &str,
        seg_id: SegId,
        num_vectors: usize,
        entries: &[VectorEntry<'_>],
    ) -> Result<()> {
        let seg_meta = SegmentMeta {
            seg_id,
            num_vectors,
        };
        self.storage
            .write_vector_entries(collection, &seg_meta, entries)
    }

    fn reverse_lookup(&self, collection: &str, seg_id: SegId, internal_id: u32) -> Result<String> {
        self.storage
            .get_reverse_doc(collection, seg_id, internal_id)?
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
