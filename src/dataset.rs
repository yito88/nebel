use std::{fs, io::Write, path::Path};

use anyhow::{Result, bail};

/// Read vectors from .fvecs format (SIFT convention):
/// each record = 4-byte i32 dimension + dimension * 4-byte f32 values.
pub fn read_fvecs(path: &Path) -> Result<(usize, Vec<Vec<f32>>)> {
    let data = fs::read(path)?;
    if data.len() < 4 {
        bail!("file too small");
    }
    let dim = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let record_size = 4 + dim * 4;
    if data.len() % record_size != 0 {
        bail!(
            "file size {} not divisible by record size {} (dim={})",
            data.len(),
            record_size,
            dim
        );
    }
    let n = data.len() / record_size;
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * record_size + 4;
        let floats: Vec<f32> = data[offset..offset + dim * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        vectors.push(floats);
    }
    Ok((dim, vectors))
}

/// Read vectors from .bvecs format (SIFT convention):
/// each record = 4-byte i32 dimension + dimension * 1-byte u8 values (cast to f32).
pub fn read_bvecs(path: &Path) -> Result<(usize, Vec<Vec<f32>>)> {
    let data = fs::read(path)?;
    if data.len() < 4 {
        bail!("file too small");
    }
    let dim = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let record_size = 4 + dim;
    if data.len() % record_size != 0 {
        bail!(
            "file size {} not divisible by record size {} (dim={})",
            data.len(),
            record_size,
            dim
        );
    }
    let n = data.len() / record_size;
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * record_size + 4;
        let floats: Vec<f32> = data[offset..offset + dim]
            .iter()
            .map(|&b| b as f32)
            .collect();
        vectors.push(floats);
    }
    Ok((dim, vectors))
}

/// Read neighbor indices from .ivecs format (SIFT groundtruth convention):
/// each record = 4-byte i32 count + count * 4-byte i32 neighbor indices.
/// Returns (k per record, Vec of neighbor-index lists).
pub fn read_ivecs(path: &Path) -> Result<(usize, Vec<Vec<i32>>)> {
    let data = fs::read(path)?;
    if data.len() < 4 {
        bail!("file too small");
    }
    let k = i32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let record_size = 4 + k * 4;
    if data.len() % record_size != 0 {
        bail!(
            "file size {} not divisible by record size {} (k={})",
            data.len(),
            record_size,
            k
        );
    }
    let n = data.len() / record_size;
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * record_size + 4;
        let ids: Vec<i32> = data[offset..offset + k * 4]
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        results.push(ids);
    }
    Ok((k, results))
}

/// Auto-detect format by extension and read vectors.
pub fn load_vectors(path: &str) -> Result<(usize, Vec<Vec<f32>>)> {
    let p = Path::new(path);
    match p.extension().and_then(|e| e.to_str()) {
        Some("fvecs") => read_fvecs(p),
        Some("bvecs") => read_bvecs(p),
        _ => read_fvecs(p),
    }
}

/// Load groundtruth from .ivecs file and convert indices to doc IDs ("doc_0", "doc_1", ...).
pub fn load_groundtruth(path: &str, k: usize) -> Result<Vec<Vec<String>>> {
    let p = Path::new(path);
    let (gt_k, raw) = read_ivecs(p)?;
    let mut results = Vec::with_capacity(raw.len());
    for ids in &raw {
        let doc_ids: Vec<String> = ids
            .iter()
            .take(k.min(gt_k))
            .map(|&idx| format!("doc_{}", idx))
            .collect();
        results.push(doc_ids);
    }
    Ok(results)
}

/// Write vectors to a raw f32 LE file (no header) for `Nebel::ingest_file`.
pub fn write_raw_f32(path: &Path, vectors: &[Vec<f32>]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    for v in vectors {
        for &val in v {
            f.write_all(&val.to_le_bytes())?;
        }
    }
    Ok(())
}
