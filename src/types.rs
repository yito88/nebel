use serde::{Deserialize, Serialize};

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
    pub active_seg_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub seg_id: u32,
    pub num_vectors: usize,
}

/// Location of a document within storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocLocation {
    pub seg_id: u32,
    pub internal_id: u32,
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_id: String,
    /// Raw distance (lower = closer for L2).
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
    pub vector: Option<Vec<f32>>,
}
