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

use segment::{Segment, WritableSegment};
use storage::Storage;
use types::{
    CollectionId, CollectionSchema, Manifest, Metric, SearchHit, SegId, SegmentMeta, SegmentState,
    VectorEntry,
};

struct CollectionContext {
    schema: CollectionSchema,
    manifest: Manifest,
    segments: HashMap<SegId, Segment>,
}

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
                )?),
                SegmentState::Sealed => Segment::Sealed(segment::SealedSegment::open(
                    seg_id,
                    dir,
                    meta.num_vectors,
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
        let seg = WritableSegment::create(SegId::FIRST, self.seg_dir(id, SegId::FIRST))?;
        let mut segments = HashMap::new();
        segments.insert(SegId::FIRST, Segment::Writable(seg));
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
        self.storage
            .put_segment(id, &SegmentMeta::new(new_id))?;
        let seg = WritableSegment::create(new_id, new_dir)?;

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
    pub fn ingest_file(
        &mut self,
        id: &CollectionId,
        file_path: impl AsRef<Path>,
    ) -> Result<usize> {
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
            self.ensure_writable_capacity(id)?;
            self.insert_vector_batch(id, &doc_ids, &vectors)?;
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
        let dimension = self.get_ctx(id)?.schema.dimension;
        if vector.len() != dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                dimension,
                vector.len()
            );
        }
        if let Some(loc) = self.storage.get_doc_location(id, doc_id)? {
            self.storage
                .set_tombstone(id, loc.seg_id, loc.internal_id)?;
        }
        self.ensure_writable_capacity(id)?;
        self.insert_vector(id, doc_id, vector, metadata)?;
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
        let ctx = self
            .collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))?;
        let dimension = ctx.schema.dimension;
        if query.len() != dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                dimension,
                query.len()
            );
        }

        // Collect candidates from all active segments.
        let mut candidates: Vec<(SegId, u32, f32)> = Vec::new();
        let active_segments = ctx.manifest.active_segments.clone();
        for &seg_id in &active_segments {
            let seg = self
                .collections
                .get(id)
                .and_then(|c| c.segments.get(&seg_id))
                .ok_or_else(|| anyhow!("segment {} not loaded", seg_id))?;
            if seg.num_vectors() == 0 {
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
                let ctx = self.collections.get_mut(id).expect("collection present");
                match ctx.segments.get_mut(&seg_id) {
                    Some(Segment::Writable(w)) => {
                        Some(w.read_vector(internal_id, dimension)?)
                    }
                    _ => None,
                }
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

    fn get_ctx(&self, id: &CollectionId) -> Result<&CollectionContext> {
        self.collections
            .get(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))
    }

    fn get_ctx_mut(&mut self, id: &CollectionId) -> Result<&mut CollectionContext> {
        self.collections
            .get_mut(id)
            .ok_or_else(|| anyhow!("collection '{}' not loaded", id))
    }

    /// Rotate to a new writable segment if the current one has reached
    /// [`SEGMENT_CAPACITY`].
    fn ensure_writable_capacity(&mut self, id: &CollectionId) -> Result<()> {
        let ctx = self.get_ctx(id)?;
        let writable_id = ctx.manifest.writable_segment;
        let at_capacity = ctx
            .segments
            .get(&writable_id)
            .is_some_and(|s| s.num_vectors() >= SEGMENT_CAPACITY);
        if at_capacity {
            self.add_writable_segment(id)?;
        }
        Ok(())
    }

    fn seg_dir(&self, id: &CollectionId, seg_id: SegId) -> PathBuf {
        self.base_dir
            .join(id.as_str())
            .join(format!("seg_{:03}", seg_id.as_u32()))
    }

    fn insert_vector(
        &mut self,
        id: &CollectionId,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let ctx = self.get_ctx_mut(id)?;
        let seg_id = ctx.manifest.writable_segment;
        let dimension = ctx.schema.dimension;
        let seg = match ctx.segments.get_mut(&seg_id) {
            Some(Segment::Writable(w)) => w,
            _ => bail!("expected writable segment"),
        };
        let internal_id = seg.insert(vector, dimension)?;
        let num_vectors = seg.num_vectors();

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
        doc_ids: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        let ctx = self.get_ctx_mut(id)?;
        let seg_id = ctx.manifest.writable_segment;
        let dimension = ctx.schema.dimension;
        let seg = match ctx.segments.get_mut(&seg_id) {
            Some(Segment::Writable(w)) => w,
            _ => bail!("expected writable segment"),
        };
        let internal_ids = seg.insert_batch(vectors, dimension)?;
        let num_vectors = seg.num_vectors();

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
            state: SegmentState::Writable,
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
