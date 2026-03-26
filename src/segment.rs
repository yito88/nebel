use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use hnsw_rs::prelude::*;

use crate::types::SegId;

const MAX_ELEMENTS: usize = 1_000_000;
const MAX_LAYER: usize = 16;
const HNSW_M: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EF_SEARCH: usize = 50;
/// 256 MB buffer for rebuilding the HNSW index in chunks.
const REBUILD_BUF_BYTES: usize = 256 * 1024 * 1024;
const INDEX_BASENAME: &str = "index";

// ---------------------------------------------------------------------------
// Shared in-memory metadata
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct SegMeta {
    seg_id: SegId,
    num_vectors: usize,
    dir: PathBuf,
}

// ---------------------------------------------------------------------------
// WritableSegment
// ---------------------------------------------------------------------------

pub struct WritableSegment {
    meta: SegMeta,
    index: Hnsw<'static, f32, DistL2>,
    vector_file: fs::File,
}

impl WritableSegment {
    /// Create a new empty writable segment on disk.
    pub fn create(seg_id: SegId, dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        let vector_file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(dir.join("vectors.seg"))?;
        let index = Hnsw::new(HNSW_M, MAX_ELEMENTS, MAX_LAYER, EF_CONSTRUCTION, DistL2 {});
        Ok(Self {
            meta: SegMeta {
                seg_id,
                num_vectors: 0,
                dir,
            },
            index,
            vector_file,
        })
    }

    /// Open an existing writable segment and rebuild the index from vectors.
    pub fn open(seg_id: SegId, dir: PathBuf, dimension: usize, num_vectors: usize) -> Result<Self> {
        let vector_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join("vectors.seg"))?;
        let index = Hnsw::new(HNSW_M, MAX_ELEMENTS, MAX_LAYER, EF_CONSTRUCTION, DistL2 {});
        let mut seg = Self {
            meta: SegMeta {
                seg_id,
                num_vectors,
                dir,
            },
            index,
            vector_file,
        };
        seg.rebuild_index(dimension)?;
        Ok(seg)
    }

    pub fn num_vectors(&self) -> usize {
        self.meta.num_vectors
    }

    /// Seal this segment: persist the index to disk and return a `SealedSegment`.
    pub fn seal(self) -> Result<SealedSegment> {
        self.index.file_dump(&self.meta.dir, INDEX_BASENAME)?;
        let (index, index_io) = load_index(&self.meta.dir)?;
        Ok(SealedSegment {
            meta: self.meta,
            index,
            index_io,
        })
    }

    /// Append a vector to the segment and return its internal_id.
    pub fn insert(&mut self, vector: &[f32], dimension: usize) -> Result<u32> {
        let internal_id = self.meta.num_vectors;
        let offset = (internal_id * dimension * 4) as u64;
        self.vector_file.seek(SeekFrom::Start(offset))?;
        self.vector_file.write_all(&f32_to_bytes(vector))?;
        self.vector_file.flush()?;
        self.index.insert((vector, internal_id));
        self.meta.num_vectors += 1;
        Ok(internal_id as u32)
    }

    /// Append a batch of vectors, write them to the file in one pass, and
    /// insert them into the index with `parallel_insert`.
    ///
    /// Returns the assigned internal_ids in the same order as `vectors`.
    pub fn insert_batch(&mut self, vectors: &[Vec<f32>], dimension: usize) -> Result<Vec<u32>> {
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
            .map(|i| i as u32)
            .collect();
        self.meta.num_vectors += vectors.len();
        Ok(ids)
    }

    /// Nearest-neighbour search. Returns (internal_id, distance) pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let ef = EF_SEARCH.max(k);
        let neighbours = self.index.search(query, k, ef);
        Ok(neighbours
            .into_iter()
            .map(|n| (n.d_id as u32, n.distance))
            .collect())
    }

    /// Read a raw vector by internal_id.
    pub fn read_vector(&mut self, internal_id: u32, dimension: usize) -> Result<Vec<f32>> {
        let record_size = dimension * 4;
        let offset = internal_id as u64 * record_size as u64;
        let mut buf = vec![0u8; record_size];
        self.vector_file.seek(SeekFrom::Start(offset))?;
        self.vector_file.read_exact(&mut buf)?;
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

pub struct SealedSegment {
    meta: SegMeta,
    index: Hnsw<'static, f32, DistL2>,
    #[allow(dead_code)]
    index_io: Box<HnswIo>,
}

impl SealedSegment {
    /// Open an existing sealed segment by loading the persisted index.
    pub fn open(seg_id: SegId, dir: PathBuf, num_vectors: usize) -> Result<Self> {
        let (index, index_io) = load_index(&dir)?;
        Ok(Self {
            meta: SegMeta {
                seg_id,
                num_vectors,
                dir,
            },
            index,
            index_io,
        })
    }

    /// Nearest-neighbour search. Returns (internal_id, distance) pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        let ef = EF_SEARCH.max(k);
        let neighbours = self.index.search(query, k, ef);
        Ok(neighbours
            .into_iter()
            .map(|n| (n.d_id as u32, n.distance))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Segment enum
// ---------------------------------------------------------------------------

pub enum Segment {
    Writable(WritableSegment),
    Sealed(SealedSegment),
}

#[allow(dead_code)]
impl Segment {
    pub fn seg_id(&self) -> SegId {
        match self {
            Segment::Writable(s) => s.meta.seg_id,
            Segment::Sealed(s) => s.meta.seg_id,
        }
    }

    pub fn num_vectors(&self) -> usize {
        match self {
            Segment::Writable(s) => s.meta.num_vectors,
            Segment::Sealed(s) => s.meta.num_vectors,
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        match self {
            Segment::Writable(s) => s.search(query, k),
            Segment::Sealed(s) => s.search(query, k),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Load a persisted HNSW index from disk.
///
/// Returns the index and the `HnswIo` that owns the memory-mapped backing data.
/// The caller must keep the `Box<HnswIo>` alive for as long as the `Hnsw` is used.
fn load_index(dir: &Path) -> Result<(Hnsw<'static, f32, DistL2>, Box<HnswIo>)> {
    let mut reloader = Box::new(HnswIo::new(dir, INDEX_BASENAME));
    // SAFETY: The Box<HnswIo> is stored alongside the Hnsw in SealedSegment,
    // ensuring the mmap backing lives as long as the index.
    let reloader_ref: &'static mut HnswIo = unsafe { &mut *(&mut *reloader as *mut HnswIo) };
    let index: Hnsw<f32, DistL2> = reloader_ref.load_hnsw()?;
    Ok((index, reloader))
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
