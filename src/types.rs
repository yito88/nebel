use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegId(u32);

impl SegId {
    pub const FIRST: Self = Self(0);

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for SegId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionId(String);

impl CollectionId {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CollectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Metric {
    L2,
    Cosine,
    Dot,
}

impl FromStr for Metric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" => Ok(Metric::L2),
            "cosine" => Ok(Metric::Cosine),
            "dot" => Ok(Metric::Dot),
            _ => Err(format!(
                "unknown metric: '{}' (expected l2, cosine, or dot)",
                s
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentParams {
    /// Number of bi-directional links per node in the HNSW graph.
    /// Higher = better recall, more memory. Typical range: 8–48.
    pub m: usize,
    /// Size of the candidate list during index construction.
    /// Higher = better index quality, slower inserts.
    pub ef_construction: usize,
    /// Size of the candidate list during search.
    /// Higher = better recall, slower queries.
    pub ef_search: usize,
    /// Number of vectors a writable segment holds before it is sealed.
    pub segment_capacity: usize,
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            segment_capacity: 100_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: CollectionId,
    pub dimension: usize,
    pub metric: Metric,
    pub segment_params: SegmentParams,
}

impl CollectionSchema {
    /// Create a schema with the given identity fields and default [`SegmentParams`].
    pub fn new(name: CollectionId, dimension: usize, metric: Metric) -> Self {
        Self {
            name,
            dimension,
            metric,
            segment_params: SegmentParams::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub active_segments: Vec<SegId>,
    pub writable_segment: SegId,
    pub next_seg_id: SegId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentState {
    Writable,
    Sealed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub seg_id: SegId,
    pub num_vectors: usize,
    pub state: SegmentState,
}

impl SegmentMeta {
    pub fn new(seg_id: SegId) -> Self {
        Self {
            seg_id,
            num_vectors: 0,
            state: SegmentState::Writable,
        }
    }
}

/// Location of a document within storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocLocation {
    pub seg_id: SegId,
    pub internal_id: u32,
}

/// A single vector entry to be written in a batch transaction.
#[derive(Debug)]
pub struct VectorEntry<'a> {
    pub doc_id: &'a str,
    pub internal_id: u32,
    pub metadata: Option<&'a serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_id: String,
    /// Raw distance (lower = closer for L2).
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
    pub vector: Option<Vec<f32>>,
}
