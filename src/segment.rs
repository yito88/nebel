use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

use anyhow::{Result, bail};
use hnsw_rs::prelude::*;

use crate::types::SegId;

const MAX_ELEMENTS: usize = 1_000_000;
const MAX_LAYER: usize = 16;
const HNSW_M: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EF_SEARCH: usize = 50;
/// 256 MB buffer for rebuilding the HNSW index in chunks.
const REBUILD_BUF_BYTES: usize = 256 * 1024 * 1024;

#[allow(dead_code)]
pub struct Segment {
    pub seg_id: SegId,
    pub dimension: usize,
    pub num_vectors: usize,
    dir: PathBuf,
    hnsw: Hnsw<'static, f32, DistL2>,
    vector_file: fs::File,
}

impl Segment {
    /// Create a new empty segment on disk.
    pub fn create(seg_id: SegId, dir: PathBuf, dimension: usize) -> Result<Self> {
        fs::create_dir_all(&dir)?;
        let vector_file = fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(dir.join("vectors.seg"))?;
        let hnsw = Hnsw::new(HNSW_M, MAX_ELEMENTS, MAX_LAYER, EF_CONSTRUCTION, DistL2 {});
        Ok(Self {
            seg_id,
            dimension,
            num_vectors: 0,
            dir,
            hnsw,
            vector_file,
        })
    }

    /// Open an existing segment and rebuild the HNSW index from vectors.
    pub fn open(seg_id: SegId, dir: PathBuf, dimension: usize, num_vectors: usize) -> Result<Self> {
        let vector_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join("vectors.seg"))?;
        let hnsw = Hnsw::new(HNSW_M, MAX_ELEMENTS, 16, EF_CONSTRUCTION, DistL2 {});
        let mut seg = Self {
            seg_id,
            dimension,
            num_vectors,
            dir,
            hnsw,
            vector_file,
        };
        seg.rebuild_hnsw()?;
        Ok(seg)
    }

    fn rebuild_hnsw(&mut self) -> Result<()> {
        if self.num_vectors == 0 {
            return Ok(());
        }
        let record_size = self.dimension * 4;
        let vectors_per_chunk = REBUILD_BUF_BYTES / record_size;
        let chunk_bytes = vectors_per_chunk * record_size;
        let mut buf = vec![0u8; chunk_bytes];
        let mut offset = 0usize;

        self.vector_file.seek(SeekFrom::Start(0))?;
        while offset < self.num_vectors {
            let remaining = self.num_vectors - offset;
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
            self.hnsw.parallel_insert(&data);
            offset += n;
        }
        Ok(())
    }

    /// Append a batch of vectors, write them to the file in one pass, and
    /// insert them into the HNSW index with `parallel_insert`.
    ///
    /// Returns the assigned internal_ids in the same order as `vectors`.
    pub fn insert_batch(&mut self, vectors: &[Vec<f32>]) -> Result<Vec<u32>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }
        for v in vectors {
            if v.len() != self.dimension {
                bail!(
                    "dimension mismatch: expected {}, got {}",
                    self.dimension,
                    v.len()
                );
            }
        }
        let first_id = self.num_vectors;
        let offset = (first_id * self.dimension * 4) as u64;
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
        self.hnsw.parallel_insert(&data);

        let ids = (first_id..first_id + vectors.len())
            .map(|i| i as u32)
            .collect();
        self.num_vectors += vectors.len();
        Ok(ids)
    }

    /// Append a vector to the segment and return its internal_id.
    pub fn insert(&mut self, vector: &[f32]) -> Result<u32> {
        if vector.len() != self.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            );
        }
        let internal_id = self.num_vectors;
        let offset = (internal_id * self.dimension * 4) as u64;
        self.vector_file.seek(SeekFrom::Start(offset))?;
        self.vector_file.write_all(&f32_to_bytes(vector))?;
        self.vector_file.flush()?;
        self.hnsw.insert((vector, internal_id));
        self.num_vectors += 1;
        Ok(internal_id as u32)
    }

    /// HNSW nearest-neighbour search. Returns (internal_id, distance) pairs.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>> {
        if query.len() != self.dimension {
            bail!(
                "dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            );
        }
        let ef = EF_SEARCH.max(k);
        let neighbours = self.hnsw.search(query, k, ef);
        Ok(neighbours
            .into_iter()
            .map(|n| (n.d_id as u32, n.distance))
            .collect())
    }

    /// Read a raw vector by internal_id.
    pub fn read_vector(&mut self, internal_id: u32) -> Result<Vec<f32>> {
        let record_size = self.dimension * 4;
        let offset = internal_id as u64 * record_size as u64;
        let mut buf = vec![0u8; record_size];
        self.vector_file.seek(SeekFrom::Start(offset))?;
        self.vector_file.read_exact(&mut buf)?;
        Ok(bytes_to_f32(&buf))
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
