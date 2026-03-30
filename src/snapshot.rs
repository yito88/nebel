use std::sync::{Arc, RwLock};

use crate::{
    segment::{SealedSegment, WritableSegment},
    types::CollectionSchema,
};

/// A segment as stored in a [`CollectionSnapshot`] — either a sealed (immutable) segment
/// or the live writable segment (behind a lock).
pub(crate) enum SegmentSnapshot {
    Sealed(Arc<SealedSegment>),
    Writable(Arc<RwLock<WritableSegment>>),
}

/// A point-in-time searchable view of a collection.
///
/// Search loads one snapshot at request start and uses it throughout,
/// so a single request sees a consistent set of segments.
pub(crate) struct CollectionSnapshot {
    pub schema: Arc<CollectionSchema>,
    /// Ordered segment list: sealed segments first, writable segment last.
    pub segs: Vec<SegmentSnapshot>,
}
