use std::{cmp::Ordering, collections::HashMap};

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

use crate::segment::Segment;
use crate::storage::Storage;
use crate::types::{
    CollectionId, CollectionSchema, Manifest, SearchHit, SegId, SegmentMeta, SegmentState,
    VectorEntry,
};

const SEGMENT_CAPACITY: usize = 100_000;

pub(crate) struct CollectionContext {
    pub(crate) schema: CollectionSchema,
    pub(crate) manifest: Manifest,
    pub(crate) segments: HashMap<SegId, Segment>,
}

impl CollectionContext {
    pub(crate) fn search(
        &self,
        storage: &Storage,
        query: &[f32],
        k: usize,
        include_metadata: bool,
        include_vector: bool,
    ) -> Result<Vec<SearchHit>> {
        let dimension = self.schema.dimension;
        if query.len() != dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                dimension,
                query.len()
            );
        }

        let mut candidates: Vec<(SegId, u32, f32)> = Vec::new();
        for &seg_id in &self.manifest.active_segments {
            let seg = self
                .segments
                .get(&seg_id)
                .ok_or_else(|| anyhow!("segment {} not loaded", seg_id))?;
            if seg.num_vectors() == 0 {
                continue;
            }
            for (internal_id, distance) in seg.search(query, k)? {
                candidates.push((seg_id, internal_id, distance));
            }
        }

        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        let id = &self.schema.name;
        let mut hits = Vec::new();
        for (seg_id, internal_id, distance) in candidates {
            if hits.len() >= k {
                break;
            }
            if storage.is_tombstoned(id, seg_id, internal_id)? {
                continue;
            }
            let doc_id = storage
                .get_reverse_doc(id, seg_id, internal_id)?
                .ok_or_else(|| {
                    anyhow!(
                        "reverse mapping not found for ({}, {})",
                        seg_id,
                        internal_id
                    )
                })?;
            let metadata = if include_metadata {
                storage.get_metadata(id, seg_id, internal_id)?
            } else {
                None
            };
            let vector = if include_vector {
                match self.segments.get(&seg_id) {
                    Some(Segment::Writable(w)) => Some(w.read_vector(internal_id, dimension)?),
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

    pub(crate) fn upsert(
        &mut self,
        storage: &mut Storage,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let dimension = self.schema.dimension;
        if vector.len() != dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                dimension,
                vector.len()
            );
        }
        let id = &self.schema.name;
        if let Some(loc) = storage.get_doc_location(id, doc_id)? {
            storage.set_tombstone(id, loc.seg_id, loc.internal_id)?;
        }
        self.insert_vector(storage, doc_id, vector, metadata)
    }

    pub(crate) fn delete(&mut self, storage: &mut Storage, doc_id: &str) -> Result<()> {
        let id = &self.schema.name;
        let loc = storage
            .get_doc_location(id, doc_id)?
            .ok_or_else(|| anyhow!("document '{}' not found", doc_id))?;
        storage.set_tombstone(id, loc.seg_id, loc.internal_id)?;
        storage.remove_doc_location(id, doc_id)
    }

    pub(crate) fn update_metadata(
        &self,
        storage: &mut Storage,
        doc_id: &str,
        metadata: Value,
    ) -> Result<()> {
        let id = &self.schema.name;
        let loc = storage
            .get_doc_location(id, doc_id)?
            .ok_or_else(|| anyhow!("document '{}' not found", doc_id))?;
        storage.put_metadata(id, loc.seg_id, loc.internal_id, &metadata)
    }

    pub(crate) fn ensure_writable_capacity(&self) -> bool {
        let writable_id = self.manifest.writable_segment;
        self.segments
            .get(&writable_id)
            .is_some_and(|s| s.num_vectors() >= SEGMENT_CAPACITY)
    }

    pub(crate) fn insert_vector(
        &mut self,
        storage: &mut Storage,
        doc_id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = self.schema.name.clone();
        let seg_id = self.manifest.writable_segment;
        let dimension = self.schema.dimension;
        let seg = match self.segments.get_mut(&seg_id) {
            Some(Segment::Writable(w)) => w,
            _ => bail!("expected writable segment"),
        };
        let internal_id = seg.insert(vector, dimension)?;
        let num_vectors = seg.num_vectors();
        self.write_vector_entries(
            storage,
            &id,
            seg_id,
            num_vectors,
            &[VectorEntry {
                doc_id,
                internal_id,
                metadata: metadata.as_ref(),
            }],
        )
    }

    pub(crate) fn insert_vector_batch(
        &mut self,
        storage: &mut Storage,
        doc_ids: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        let id = self.schema.name.clone();
        let seg_id = self.manifest.writable_segment;
        let dimension = self.schema.dimension;
        let seg = match self.segments.get_mut(&seg_id) {
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
        self.write_vector_entries(storage, &id, seg_id, num_vectors, &entries)
    }

    fn write_vector_entries(
        &self,
        storage: &Storage,
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
        storage.write_vector_entries(id, &seg_meta, entries)
    }
}
