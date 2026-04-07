use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Level
// ---------------------------------------------------------------------------

/// Compaction level for a sealed segment.
/// L0 is the ingest level (freshly sealed); higher levels hold larger, older segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Level(u8);

impl Level {
    pub const ZERO: Self = Self(0);

    pub fn new(v: u8) -> Self {
        Self(v)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    pub fn is_top(self, num_levels: usize) -> bool {
        self.as_usize() + 1 >= num_levels
    }

    /// The output level for a compaction that starts at `self`.
    /// Promotes to `self + 1`, capped at the top level.
    /// At the top level the output stays the same (tombstone cleanup in place).
    pub fn output(self, num_levels: usize) -> Self {
        if self.is_top(num_levels) {
            self
        } else {
            Self(self.0 + 1)
        }
    }

    /// Maximum number of vectors a segment at this level can hold:
    /// `base_capacity * 2^level`.
    pub fn capacity(self, base_capacity: usize) -> usize {
        base_capacity * (1 << self.as_usize())
    }
}

impl Default for Level {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CompactionParams
// ---------------------------------------------------------------------------

const DEFAULT_NUM_LEVELS: usize = 4;
const DEFAULT_COUNT_THRESHOLDS: [usize; 4] = [4, 8, 16, 32];
const DEFAULT_TOMBSTONE_THRESHOLD: f64 = 0.2;
const DEFAULT_PACKING_FILL_FACTOR: f64 = 0.8;

/// Policy parameters for the tiered compaction worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionParams {
    /// Number of compaction levels (L0 = ingest level, top level = highest).
    pub num_levels: usize,
    /// Minimum sealed segment count per level before the count-trigger fires.
    /// `level_count_thresholds[i]` is the threshold for level `i`.
    /// Length must equal `num_levels`.
    pub level_count_thresholds: Vec<usize>,
    /// Tombstone ratio above which a level becomes eligible for compaction.
    pub tombstone_threshold: f64,
    /// Target fill factor: accumulate input segments until
    /// `live_vectors >= next_level_capacity * packing_fill_factor`.
    pub packing_fill_factor: f64,
}

impl Default for CompactionParams {
    fn default() -> Self {
        Self {
            num_levels: DEFAULT_NUM_LEVELS,
            level_count_thresholds: DEFAULT_COUNT_THRESHOLDS.to_vec(),
            tombstone_threshold: DEFAULT_TOMBSTONE_THRESHOLD,
            packing_fill_factor: DEFAULT_PACKING_FILL_FACTOR,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SegId(u32);

impl SegId {
    pub const FIRST: Self = Self(0);

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn from_u32(v: u32) -> Self {
        Self(v)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InternalId(u32);

impl InternalId {
    pub fn from_u32(v: u32) -> Self {
        Self(v)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for InternalId {
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

const DEFAULT_M: usize = 16;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const DEFAULT_EF_SEARCH: usize = 50;
const DEFAULT_SEGMENT_CAPACITY: usize = 500_000;

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
            m: DEFAULT_M,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            segment_capacity: DEFAULT_SEGMENT_CAPACITY,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    pub name: CollectionId,
    pub dimension: usize,
    pub metric: Metric,
    pub segment_params: SegmentParams,
    #[serde(default)]
    pub compaction_params: CompactionParams,
}

impl CollectionSchema {
    /// Create a schema with the given identity fields and default [`SegmentParams`].
    pub fn new(name: CollectionId, dimension: usize, metric: Metric) -> Self {
        Self {
            name,
            dimension,
            metric,
            segment_params: SegmentParams::default(),
            compaction_params: CompactionParams::default(),
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
    /// Number of tombstoned entries in this segment.
    /// Incremented atomically with each tombstone write; avoids table scans.
    #[serde(default)]
    pub tombstone_count: usize,
    /// Compaction level: L0 = freshly sealed, incremented on each promotion.
    /// The top level compacts within itself.
    #[serde(default)]
    pub level: Level,
}

impl SegmentMeta {
    pub fn new(seg_id: SegId) -> Self {
        Self {
            seg_id,
            num_vectors: 0,
            state: SegmentState::Writable,
            tombstone_count: 0,
            level: Level::ZERO,
        }
    }
}

/// Location of a document within storage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocLocation {
    pub seg_id: SegId,
    pub internal_id: InternalId,
}

/// A single vector entry to be written in a batch transaction.
#[derive(Debug)]
pub struct VectorEntry<'a> {
    pub doc_id: &'a str,
    pub internal_id: InternalId,
    pub metadata: Option<&'a serde_json::Value>,
}

/// Monotonic sequence token returned by every write operation.
/// Pass to `wait_visible` to block until the write is search-visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WriteToken(pub u64);

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub doc_id: String,
    /// Raw distance (lower = closer for L2).
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
    pub vector: Option<Vec<f32>>,
}
