use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegId(u32);

impl SegId {
    pub const FIRST: Self = Self(0);

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SegId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Metric {
    L2,
    Cosine,
    Dot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub active_seg_id: SegId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub seg_id: SegId,
    pub num_vectors: usize,
}

impl SegmentMeta {
    pub fn new(seg_id: SegId) -> Self {
        Self {
            seg_id,
            num_vectors: 0,
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
