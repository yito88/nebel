use std::cmp::Ordering as CmpOrdering;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::{
    filter::{FilterExpr, FilterResult},
    handle::CollectionInner,
    segment::{SealedSegment, compute_distance},
    snapshot::{CollectionSnapshot, SegmentSnapshot},
    types::{InternalId, SearchHit, SegId},
};

const EXACT_THRESHOLD: usize = 5_000;

pub(crate) fn search_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
    filter: Option<&FilterExpr>,
    include_metadata: bool,
    include_vector: bool,
) -> Result<Vec<SearchHit>> {
    let dimension = snap.schema.dimension;
    if query.len() != dimension {
        bail!(
            "dimension mismatch: expected {}, got {}",
            dimension,
            query.len()
        );
    }
    // Planner: evaluate filter first, then decide strategy.
    if let Some(f) = filter {
        let schema =
            snap.schema.metadata_schema.as_ref().ok_or_else(|| {
                anyhow::anyhow!("collection has no metadata schema; cannot filter")
            })?;
        let filter_result = inner.storage.evaluate_filter(&inner.id, f, schema)?;
        return if filter_result.matched_count <= EXACT_THRESHOLD {
            filtered_exact_snapshot(inner, snap, query, k, &filter_result, include_metadata)
        } else {
            ann_post_filter_snapshot(
                inner,
                snap,
                query,
                k,
                &filter_result,
                include_metadata,
                include_vector,
            )
        };
    }

    let storage = &*inner.storage;
    let id = &inner.id;

    let sealed: Vec<&Arc<SealedSegment>> = snap
        .segs
        .iter()
        .filter_map(|s| match s {
            SegmentSnapshot::Sealed(ss) => Some(ss),
            _ => None,
        })
        .collect();
    let writable = snap.segs.iter().find_map(|s| match s {
        SegmentSnapshot::Writable(w) => Some(w),
        _ => None,
    });

    let (sealed_cands, ws_cands) = std::thread::scope(|scope| {
        let sealed_handle = scope.spawn(|| {
            sealed
                .par_iter()
                .filter(|s| s.num_vectors() > 0)
                .flat_map(|seg| {
                    let sid = seg.seg_id();
                    seg.search(query, k)
                        .unwrap_or_default()
                        .into_iter()
                        .map(move |(id, dist)| (sid, id, dist))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        let ws_cands: Vec<(SegId, InternalId, f32)> = writable
            .map(|w| {
                let ws = w.read().unwrap();
                let sid = ws.seg_id();
                let hits = if ws.num_vectors() > 0 {
                    ws.search(query, k).unwrap_or_default()
                } else {
                    vec![]
                };
                drop(ws);
                hits.into_iter()
                    .map(|(id, dist)| (sid, id, dist))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        (sealed_handle.join().unwrap(), ws_cands)
    });
    let mut candidates: Vec<(SegId, InternalId, f32)> = sealed_cands;
    candidates.extend(ws_cands);
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(CmpOrdering::Equal));

    // Resolve tombstones, doc_ids, and metadata in a single read transaction.
    let keys: Vec<(SegId, InternalId)> = candidates.iter().map(|&(s, i, _)| (s, i)).collect();
    let resolved = storage.resolve_candidates(
        id,
        &keys,
        include_metadata,
        false,
        snap.schema.metadata_schema.as_ref(),
    )?;

    let mut hits = Vec::new();
    for ((seg_id, internal_id, distance), resolved) in candidates.iter().zip(resolved.into_iter()) {
        if hits.len() >= k {
            break;
        }
        let Some(r) = resolved else { continue };
        let vector = if include_vector {
            let from_sealed = sealed
                .iter()
                .find(|s| s.seg_id() == *seg_id)
                .map(|s| s.read_vector(*internal_id, dimension))
                .transpose()?;
            if from_sealed.is_some() {
                from_sealed
            } else if let Some(w) = writable {
                let ws = w.read().unwrap();
                if ws.seg_id() == *seg_id {
                    Some(ws.read_vector(*internal_id, dimension)?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: *distance,
            metadata: r.metadata,
            vector,
        });
    }
    Ok(hits)
}

#[cfg(feature = "testing")]
pub(crate) fn search_exact_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
    include_metadata: bool,
) -> Result<Vec<SearchHit>> {
    let dimension = snap.schema.dimension;
    if query.len() != dimension {
        bail!(
            "dimension mismatch: expected {}, got {}",
            dimension,
            query.len()
        );
    }
    let storage = &*inner.storage;
    let id = &inner.id;
    let metric = &snap.schema.metric;

    // Load all tombstones in one read transaction instead of per-vector lookups.
    let tombstones = storage.load_tombstone_set(id)?;

    struct HeapEntry {
        distance: f32,
        seg_id: SegId,
        internal_id: InternalId,
    }

    let sealed: Vec<&Arc<SealedSegment>> = snap
        .segs
        .iter()
        .filter_map(|s| match s {
            SegmentSnapshot::Sealed(ss) => Some(ss),
            _ => None,
        })
        .collect();
    let writable = snap.segs.iter().find_map(|s| match s {
        SegmentSnapshot::Writable(w) => Some(w),
        _ => None,
    });

    let (sealed_cands, ws_cands): (Vec<HeapEntry>, Vec<HeapEntry>) = std::thread::scope(|scope| {
        let sealed_handle = scope.spawn(|| {
            sealed
                .par_iter()
                .flat_map(|seg| {
                    let sid = seg.seg_id();
                    (0..seg.num_vectors() as u32)
                        .map(InternalId::from_u32)
                        .filter_map(|internal_id| {
                            if tombstones.contains(&(sid, internal_id)) {
                                return None;
                            }
                            let vector = seg.read_vector(internal_id, dimension).ok()?;
                            let distance = compute_distance(metric, query, &vector);
                            Some(HeapEntry {
                                distance,
                                seg_id: sid,
                                internal_id,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<HeapEntry>>()
        });
        let ws_cands: Vec<HeapEntry> = writable
            .map(|w| {
                let ws = w.read().unwrap();
                let sid = ws.seg_id();
                let n = ws.num_vectors() as u32;
                let entries: Vec<HeapEntry> = (0..n)
                    .map(InternalId::from_u32)
                    .filter_map(|internal_id| {
                        if tombstones.contains(&(sid, internal_id)) {
                            return None;
                        }
                        let vector = ws.read_vector(internal_id, dimension).ok()?;
                        let distance = compute_distance(metric, query, &vector);
                        Some(HeapEntry {
                            distance,
                            seg_id: sid,
                            internal_id,
                        })
                    })
                    .collect();
                drop(ws);
                entries
            })
            .unwrap_or_default();
        (sealed_handle.join().unwrap(), ws_cands)
    });
    let mut candidates = sealed_cands;
    candidates.extend(ws_cands);

    candidates.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(CmpOrdering::Equal)
    });
    let top_k: Vec<HeapEntry> = candidates.into_iter().take(k).collect();

    // Resolve doc_ids in a single read transaction.
    let keys: Vec<(SegId, InternalId)> = top_k.iter().map(|e| (e.seg_id, e.internal_id)).collect();
    let resolved = storage.resolve_candidates(
        id,
        &keys,
        include_metadata,
        false,
        snap.schema.metadata_schema.as_ref(),
    )?;

    let mut hits = Vec::with_capacity(top_k.len());
    for (entry, r) in top_k.iter().zip(resolved.into_iter()) {
        let Some(r) = r else { continue };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: entry.distance,
            metadata: r.metadata,
            vector: None,
        });
    }
    Ok(hits)
}

// ---------------------------------------------------------------------------
// Filtered exact search
// ---------------------------------------------------------------------------

/// Brute-force distance scan limited to `filter_result.doc_ords`.
///
/// Used when the filter selects few enough documents that loading every matched
/// vector is cheaper than a full ANN + post-filter pass.
pub(crate) fn filtered_exact_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
    filter_result: &FilterResult,
    include_metadata: bool,
) -> Result<Vec<SearchHit>> {
    let dimension = snap.schema.dimension;
    if query.len() != dimension {
        bail!(
            "dimension mismatch: expected {}, got {}",
            dimension,
            query.len()
        );
    }
    let metric = &snap.schema.metric;

    // Resolve doc_ords → (doc_id, DocLocation).
    let locations = inner
        .storage
        .get_doc_locations(&inner.id, &filter_result.doc_ords)?;

    // Build a lookup from SegId → segment snapshot for fast access.
    let sealed: Vec<&Arc<SealedSegment>> = snap
        .segs
        .iter()
        .filter_map(|s| match s {
            SegmentSnapshot::Sealed(ss) => Some(ss),
            _ => None,
        })
        .collect();
    let writable = snap.segs.iter().find_map(|s| match s {
        SegmentSnapshot::Writable(w) => Some(w),
        _ => None,
    });

    // Compute distances for each matched doc.
    struct Entry {
        seg_id: SegId,
        internal_id: InternalId,
        distance: f32,
    }

    let mut entries: Vec<Entry> = Vec::with_capacity(locations.len());
    for item in locations.into_iter().flatten() {
        let (_doc_id, loc) = item;
        let vector = if let Some(seg) = sealed.iter().find(|s| s.seg_id() == loc.seg_id) {
            seg.read_vector(loc.internal_id, dimension).ok()
        } else if let Some(w) = writable {
            let ws = w.read().unwrap();
            if ws.seg_id() == loc.seg_id {
                ws.read_vector(loc.internal_id, dimension).ok()
            } else {
                None
            }
        } else {
            None
        };
        if let Some(v) = vector {
            entries.push(Entry {
                seg_id: loc.seg_id,
                internal_id: loc.internal_id,
                distance: compute_distance(metric, query, &v),
            });
        }
    }

    entries.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(CmpOrdering::Equal)
    });
    entries.truncate(k);

    // Resolve metadata (and tombstone-check) for the top-k.
    let keys: Vec<(SegId, InternalId)> =
        entries.iter().map(|e| (e.seg_id, e.internal_id)).collect();
    let resolved = inner.storage.resolve_candidates(
        &inner.id,
        &keys,
        include_metadata,
        false,
        snap.schema.metadata_schema.as_ref(),
    )?;

    let mut hits = Vec::with_capacity(entries.len());
    for (entry, r) in entries.iter().zip(resolved.into_iter()) {
        let Some(r) = r else { continue };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: entry.distance,
            metadata: r.metadata,
            vector: None,
        });
    }
    Ok(hits)
}

// ---------------------------------------------------------------------------
// ANN + post-filter
// ---------------------------------------------------------------------------

/// ANN search with a larger candidate pool, then retain only documents whose
/// `doc_ord` is in `filter_result`.
pub(crate) fn ann_post_filter_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
    filter_result: &FilterResult,
    include_metadata: bool,
    include_vector: bool,
) -> Result<Vec<SearchHit>> {
    let dimension = snap.schema.dimension;
    if query.len() != dimension {
        bail!(
            "dimension mismatch: expected {}, got {}",
            dimension,
            query.len()
        );
    }

    let filter_set: HashSet<u64> = filter_result.doc_ords.iter().copied().collect();
    let candidate_k = (k * 10).max(100);

    let storage = &*inner.storage;
    let id = &inner.id;

    let sealed: Vec<&Arc<SealedSegment>> = snap
        .segs
        .iter()
        .filter_map(|s| match s {
            SegmentSnapshot::Sealed(ss) => Some(ss),
            _ => None,
        })
        .collect();
    let writable = snap.segs.iter().find_map(|s| match s {
        SegmentSnapshot::Writable(w) => Some(w),
        _ => None,
    });

    let (sealed_cands, ws_cands) = std::thread::scope(|scope| {
        let sealed_handle = scope.spawn(|| {
            sealed
                .par_iter()
                .filter(|s| s.num_vectors() > 0)
                .flat_map(|seg| {
                    let sid = seg.seg_id();
                    seg.search(query, candidate_k)
                        .unwrap_or_default()
                        .into_iter()
                        .map(move |(id, dist)| (sid, id, dist))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });
        let ws_cands: Vec<(SegId, InternalId, f32)> = writable
            .map(|w| {
                let ws = w.read().unwrap();
                let sid = ws.seg_id();
                let hits = if ws.num_vectors() > 0 {
                    ws.search(query, candidate_k).unwrap_or_default()
                } else {
                    vec![]
                };
                drop(ws);
                hits.into_iter()
                    .map(|(id, dist)| (sid, id, dist))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        (sealed_handle.join().unwrap(), ws_cands)
    });
    let mut candidates: Vec<(SegId, InternalId, f32)> = sealed_cands;
    candidates.extend(ws_cands);
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(CmpOrdering::Equal));

    // Resolve candidates with doc_ord so we can apply the filter set.
    let keys: Vec<(SegId, InternalId)> = candidates.iter().map(|&(s, i, _)| (s, i)).collect();
    let resolved = storage.resolve_candidates(
        id,
        &keys,
        include_metadata,
        true, // need_doc_ord for post-filter check
        snap.schema.metadata_schema.as_ref(),
    )?;

    let mut hits = Vec::new();
    for ((seg_id, internal_id, distance), r) in candidates.iter().zip(resolved.into_iter()) {
        if hits.len() >= k {
            break;
        }
        let Some(r) = r else { continue };
        // Keep only docs that matched the filter.
        if !r.doc_ord.is_some_and(|ord| filter_set.contains(&ord)) {
            continue;
        }
        let vector = if include_vector {
            let from_sealed = sealed
                .iter()
                .find(|s| s.seg_id() == *seg_id)
                .map(|s| s.read_vector(*internal_id, dimension))
                .transpose()?;
            if from_sealed.is_some() {
                from_sealed
            } else if let Some(w) = writable {
                let ws = w.read().unwrap();
                if ws.seg_id() == *seg_id {
                    Some(ws.read_vector(*internal_id, dimension)?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: *distance,
            metadata: r.metadata,
            vector,
        });
    }
    Ok(hits)
}
