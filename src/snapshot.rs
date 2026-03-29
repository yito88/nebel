use std::sync::{Arc, RwLock};

use crate::{
    segment::{SealedSegment, WritableSegment},
    types::CollectionSchema,
};

/// A point-in-time searchable view of a collection.
///
/// Search loads one snapshot at request start and uses it throughout,
/// so a single request sees a consistent set of segments.
pub(crate) struct CollectionSnapshot {
    pub schema: Arc<CollectionSchema>,
    /// Immutable sealed segments, shared by Arc.
    pub sealed_segs: Vec<Arc<SealedSegment>>,
    /// Live writable segment — searched under a read lock.
    pub writable_seg: Arc<RwLock<WritableSegment>>,
    /// visible_seq at the time this snapshot was published.
    pub visible_seq: u64,
}
