use std::cmp::Ordering as CmpOrdering;
use std::sync::Arc;

use anyhow::{Result, bail};
use rayon::prelude::*;

#[cfg(feature = "testing")]
use crate::segment::compute_distance;
use crate::{
    handle::CollectionInner,
    segment::SealedSegment,
    snapshot::{CollectionSnapshot, SegmentSnapshot},
    types::{InternalId, SearchHit, SegId},
};

pub(crate) fn search_snapshot(
    inner: &CollectionInner,
    snap: &CollectionSnapshot,
    query: &[f32],
    k: usize,
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
    let resolved = storage.resolve_candidates(id, &keys, false, None)?;

    let mut hits = Vec::with_capacity(top_k.len());
    for (entry, r) in top_k.iter().zip(resolved.into_iter()) {
        let Some(r) = r else { continue };
        hits.push(SearchHit {
            doc_id: r.doc_id,
            score: entry.distance,
            metadata: None,
            vector: None,
        });
    }
    Ok(hits)
}
