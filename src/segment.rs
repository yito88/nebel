use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};
use hnsw_rs::prelude::*;

use crate::types::{InternalId, Level, Metric, SegId, SegmentMeta, SegmentParams, SegmentState};

const MAX_ELEMENTS: usize = 1_000_000;
const MAX_LAYER: usize = 16;
/// 256 MB buffer for rebuilding the HNSW index in chunks.
const REBUILD_BUF_BYTES: usize = 256 * 1024 * 1024;
const INDEX_BASENAME: &str = "index";

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

/// Metric-polymorphic wrapper around the `hnsw_rs` index types.
enum HnswIndex {
    L2(Hnsw<'static, f32, DistL2>),
    Cosine(Hnsw<'static, f32, DistCosine>),
    Dot(Hnsw<'static, f32, DistDot>),
}

impl HnswIndex {
    fn new(metric: &Metric, m: usize, ef_construction: usize) -> Self {
        match metric {
            Metric::L2 => HnswIndex::L2(Hnsw::new(
                m,
                MAX_ELEMENTS,
                MAX_LAYER,
                ef_construction,
                DistL2 {},
            )),
            Metric::Cosine => HnswIndex::Cosine(Hnsw::new(
                m,
                MAX_ELEMENTS,
                MAX_LAYER,
                ef_construction,
                DistCosine {},
            )),
            Metric::Dot => HnswIndex::Dot(Hnsw::new(
                m,
                MAX_ELEMENTS,
                MAX_LAYER,
                ef_construction,
                DistDot {},
            )),
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbour> {
        match self {
            HnswIndex::L2(idx) => idx.search(query, k, ef),
            HnswIndex::Cosine(idx) => idx.search(query, k, ef),
            HnswIndex::Dot(idx) => idx.search(query, k, ef),
        }
    }

    fn insert(&self, data: (&[f32], usize)) {
        match self {
            HnswIndex::L2(idx) => idx.insert(data),
            HnswIndex::Cosine(idx) => idx.insert(data),
            HnswIndex::Dot(idx) => idx.insert(data),
        }
    }

    fn parallel_insert(&self, data: &[(&Vec<f32>, usize)]) {
        match self {
            HnswIndex::L2(idx) => idx.parallel_insert(data),
            HnswIndex::Cosine(idx) => idx.parallel_insert(data),
            HnswIndex::Dot(idx) => idx.parallel_insert(data),
        }
    }

    fn file_dump(&self, dir: &Path, basename: &str) -> Result<()> {
        match self {
            HnswIndex::L2(idx) => idx
                .file_dump(dir, basename)
                .map(|_| ())
                .map_err(|e| anyhow!(e)),
            HnswIndex::Cosine(idx) => idx
                .file_dump(dir, basename)
                .map(|_| ())
                .map_err(|e| anyhow!(e)),
            HnswIndex::Dot(idx) => idx
                .file_dump(dir, basename)
                .map(|_| ())
                .map_err(|e| anyhow!(e)),
        }
    }

    fn metric(&self) -> Metric {
        match self {
            HnswIndex::L2(_) => Metric::L2,
            HnswIndex::Cosine(_) => Metric::Cosine,
            HnswIndex::Dot(_) => Metric::Dot,
        }
    }
}

// ---------------------------------------------------------------------------
// SegMeta
// ---------------------------------------------------------------------------

/// Shared in-memory metadata carried by both writable and sealed segments.
#[allow(dead_code)]
struct SegMeta {
    seg_id: SegId,
    num_vectors: usize,
    dir: PathBuf,
    /// Compaction level. Only meaningful for sealed segments.
    level: Level,
}

// ---------------------------------------------------------------------------
// WritableSegment
// ---------------------------------------------------------------------------

/// An append-only segment that accepts new vectors and maintains a live HNSW index.
pub struct WritableSegment {
    meta: SegMeta,
    index: HnswIndex,
    ef_search: usize,
    vector_file: fs::File,
}

impl WritableSegment {
    /// Create a new empty writable segment on disk.
    pub fn create(
        seg_id: SegId,
        dir: PathBuf,
        metric: &Metric,
        params: &SegmentParams,
    ) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        let vector_file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(dir.join("vectors.seg"))?;
        let index = HnswIndex::new(metric, params.m, params.ef_construction);
        Ok(Self {
            meta: SegMeta {
                seg_id,
                num_vectors: 0,
                dir,
                level: Level::ZERO,
            },
            index,
            ef_search: params.ef_search,
            vector_file,
        })
    }

    /// Open an existing writable segment and rebuild the index from vectors.
    pub fn open(
        seg_id: SegId,
        dir: PathBuf,
        dimension: usize,
        num_vectors: usize,
        metric: &Metric,
        params: &SegmentParams,
    ) -> Result<Self> {
        let vector_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join("vectors.seg"))?;
        let index = HnswIndex::new(metric, params.m, params.ef_construction);
        let mut seg = Self {
            meta: SegMeta {
                seg_id,
                num_vectors,
                dir,
                level: Level::ZERO,
            },
            index,
            ef_search: params.ef_search,
            vector_file,
        };
        seg.rebuild_index(dimension)?;
        Ok(seg)
    }

    pub fn seg_id(&self) -> SegId {
        self.meta.seg_id
    }

    pub fn num_vectors(&self) -> usize {
        self.meta.num_vectors
    }

    /// Non-consuming seal: persist the index to disk and return a new `SealedSegment`
    /// while keeping this `WritableSegment` intact for ongoing searches.
    /// `level` is the compaction level to assign to the resulting sealed segment.
    pub(crate) fn persist_as_sealed(&self, level: Level) -> Result<SealedSegment> {
        let metric = self.index.metric();
        self.index.file_dump(&self.meta.dir, INDEX_BASENAME)?;
        let (index, index_io) = load_index(&self.meta.dir, &metric)?;
        let vector_file = fs::File::open(self.meta.dir.join("vectors.seg"))?;
        Ok(SealedSegment {
            meta: SegmentMeta {
                seg_id: self.meta.seg_id,
                num_vectors: self.meta.num_vectors,
                state: SegmentState::Sealed,
                tombstone_count: 0,
                level,
            },
            index,
            ef_search: self.ef_search,
            vector_file,
            index_io,
        })
    }

    /// Append a vector to the segment and return its internal_id.
    pub fn insert(&mut self, vector: &[f32], dimension: usize) -> Result<InternalId> {
        let internal_id = self.meta.num_vectors;
        let offset = (internal_id * dimension * 4) as u64;
        self.vector_file.seek(SeekFrom::Start(offset))?;
        self.vector_file.write_all(&f32_to_bytes(vector))?;
        self.vector_file.flush()?;
        self.index.insert((vector, internal_id));
        self.meta.num_vectors += 1;
        Ok(InternalId::from_u32(internal_id as u32))
    }

    /// Append a batch of vectors, write them to the file in one pass, and
    /// insert them into the index with `parallel_insert`.
    ///
    /// Returns the assigned internal_ids in the same order as `vectors`.
    pub fn insert_batch(
        &mut self,
        vectors: &[Vec<f32>],
        dimension: usize,
    ) -> Result<Vec<InternalId>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        let first_id = self.meta.num_vectors;
        let offset = (first_id * dimension * 4) as u64;
        self.vector_file.seek(SeekFrom::Start(offset))?;
        for v in vectors {
            self.vector_file.write_all(&f32_to_bytes(v))?;
        }
        self.vector_file.flush()?;

        let data: Vec<(&Vec<f32>, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (v, first_id + i))
            .collect();
        self.index.parallel_insert(&data);

        let ids = (first_id..first_id + vectors.len())
            .map(|i| InternalId::from_u32(i as u32))
            .collect();
        self.meta.num_vectors += vectors.len();
        Ok(ids)
    }

    /// Nearest-neighbour search. Returns (internal_id, distance) pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(InternalId, f32)>> {
        let ef = self.ef_search.max(k);
        let neighbours = self.index.search(query, k, ef);
        Ok(neighbours
            .into_iter()
            .map(|n| (InternalId::from_u32(n.d_id as u32), n.distance))
            .collect())
    }

    /// Read a raw vector by internal_id.
    pub fn read_vector(&self, internal_id: InternalId, dimension: usize) -> Result<Vec<f32>> {
        let record_size = dimension * 4;
        let offset = internal_id.as_usize() as u64 * record_size as u64;
        let mut buf = vec![0u8; record_size];
        self.vector_file.read_exact_at(&mut buf, offset)?;
        Ok(bytes_to_f32(&buf))
    }

    fn rebuild_index(&mut self, dimension: usize) -> Result<()> {
        if self.meta.num_vectors == 0 {
            return Ok(());
        }
        let record_size = dimension * 4;
        let vectors_per_chunk = REBUILD_BUF_BYTES / record_size;
        let chunk_bytes = vectors_per_chunk * record_size;
        let mut buf = vec![0u8; chunk_bytes];
        let mut offset = 0usize;

        self.vector_file.seek(SeekFrom::Start(0))?;
        while offset < self.meta.num_vectors {
            let remaining = self.meta.num_vectors - offset;
            let n = remaining.min(vectors_per_chunk);
            let read_bytes = n * record_size;
            self.vector_file.read_exact(&mut buf[..read_bytes])?;

            let vectors: Vec<Vec<f32>> = buf[..read_bytes]
                .chunks_exact(record_size)
                .map(bytes_to_f32)
                .collect();

            let data: Vec<(&Vec<f32>, usize)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (v, offset + i))
                .collect();
            self.index.parallel_insert(&data);
            offset += n;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SealedSegment
// ---------------------------------------------------------------------------

/// A read-only segment whose HNSW index has been persisted to disk and memory-mapped.
pub struct SealedSegment {
    meta: SegmentMeta,
    index: HnswIndex,
    ef_search: usize,
    /// Open file handle to vectors.seg. Kept open so that reads remain valid even
    /// after the segment directory is removed by compaction (Unix unlink semantics).
    vector_file: fs::File,
    #[allow(dead_code)]
    index_io: Box<HnswIo>,
}

impl SealedSegment {
    pub fn seg_id(&self) -> SegId {
        self.meta.seg_id
    }

    pub fn num_vectors(&self) -> usize {
        self.meta.num_vectors
    }

    pub fn meta(&self) -> &SegmentMeta {
        &self.meta
    }

    /// Open an existing sealed segment by loading the persisted index.
    pub fn open(
        seg_id: SegId,
        dir: PathBuf,
        num_vectors: usize,
        metric: &Metric,
        ef_search: usize,
        level: Level,
        tombstone_count: usize,
    ) -> Result<Self> {
        let (index, index_io) = load_index(&dir, metric)?;
        let vector_file = fs::File::open(dir.join("vectors.seg"))?;
        Ok(Self {
            meta: SegmentMeta {
                seg_id,
                num_vectors,
                state: SegmentState::Sealed,
                tombstone_count,
                level,
            },
            index,
            ef_search,
            vector_file,
            index_io,
        })
    }

    /// Nearest-neighbour search. Returns (internal_id, distance) pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(InternalId, f32)>> {
        let ef = self.ef_search.max(k);
        let neighbours = self.index.search(query, k, ef);
        Ok(neighbours
            .into_iter()
            .map(|n| (InternalId::from_u32(n.d_id as u32), n.distance))
            .collect())
    }

    /// Read a raw vector by internal_id from the on-disk vectors.seg file.
    pub fn read_vector(&self, internal_id: InternalId, dimension: usize) -> Result<Vec<f32>> {
        let record_size = dimension * 4;
        let offset = internal_id.as_usize() as u64 * record_size as u64;
        let mut buf = vec![0u8; record_size];
        self.vector_file.read_exact_at(&mut buf, offset)?;
        Ok(bytes_to_f32(&buf))
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Load a persisted HNSW index from disk.
///
/// Returns the index and the `HnswIo` that owns the memory-mapped backing data.
/// The caller must keep the `Box<HnswIo>` alive for as long as the `Hnsw` is used.
fn load_index(dir: &Path, metric: &Metric) -> Result<(HnswIndex, Box<HnswIo>)> {
    let mut reloader = Box::new(HnswIo::new(dir, INDEX_BASENAME));
    // SAFETY: The Box<HnswIo> is stored alongside the Hnsw in SealedSegment,
    // ensuring the mmap backing lives as long as the index.
    let reloader_ref: &'static mut HnswIo = unsafe { &mut *(&mut *reloader as *mut HnswIo) };
    let index = match metric {
        Metric::L2 => {
            let idx: Hnsw<f32, DistL2> = reloader_ref.load_hnsw()?;
            HnswIndex::L2(idx)
        }
        Metric::Cosine => {
            let idx: Hnsw<f32, DistCosine> = reloader_ref.load_hnsw()?;
            HnswIndex::Cosine(idx)
        }
        Metric::Dot => {
            let idx: Hnsw<f32, DistDot> = reloader_ref.load_hnsw()?;
            HnswIndex::Dot(idx)
        }
    };
    Ok((index, reloader))
}

/// Compute the raw distance between two vectors for the given metric.
/// The result is consistent with hnsw_rs conventions (lower = closer).
#[cfg(feature = "testing")]
pub fn compute_distance(metric: &Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => {
            let sq: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
            sq.sqrt()
        }
        Metric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = norm_a * norm_b;
            if denom == 0.0 { 1.0 } else { 1.0 - dot / denom }
        }
        Metric::Dot => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            -dot
        }
    }
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
