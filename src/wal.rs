use std::{
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const WAL_ROTATION_BYTES: u64 = 64 * 1024 * 1024; // 64 MB

// ---------------------------------------------------------------------------
// Record types (unchanged)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WalRecord {
    pub seq: u64,
    pub op: WalOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum WalOp {
    Upsert {
        doc_id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    Delete {
        doc_id: String,
    },
    UpdateMetadata {
        doc_id: String,
        metadata: Value,
    },
}

// ---------------------------------------------------------------------------
// Segment metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WalSegmentState {
    Active,
    Closed,
}

pub(crate) struct WalSegmentMeta {
    pub wal_id: u64,
    pub path: PathBuf,
    pub start_seq: u64,       // seq of first record; 0 if empty
    pub end_seq: Option<u64>, // None while Active
    pub byte_size: u64,       // sum of (4 + body.len() + 1) for all records
    pub record_count: u64,
    pub state: WalSegmentState,
}

// ---------------------------------------------------------------------------
// Apply cursor
// ---------------------------------------------------------------------------

pub(crate) struct ApplyCursor {
    pub current_wal_id: u64,
    pub current_offset: u64,
    pub last_applied_seq: u64,
}

// ---------------------------------------------------------------------------
// Filename helpers
// ---------------------------------------------------------------------------

pub(crate) fn wal_filename(wal_id: u64) -> String {
    format!("{:06}.log", wal_id)
}

pub(crate) fn parse_wal_id(filename: &str) -> Option<u64> {
    filename.strip_suffix(".log")?.parse().ok()
}

// ---------------------------------------------------------------------------
// Wal
// ---------------------------------------------------------------------------

pub(crate) struct Wal {
    wal_dir: PathBuf,
    /// All segments sorted by wal_id ascending. Last entry is always Active.
    pub(crate) segments: Vec<WalSegmentMeta>,
    writer: BufWriter<fs::File>,
}

impl Wal {
    /// Open (or create) a segmented WAL directory.
    ///
    /// - If the directory is empty, creates `000001.log` as the first active segment.
    /// - If segments already exist, reconstructs metadata by scanning them, marks all
    ///   but the last as Closed, and opens the last in append mode.
    pub(crate) fn open_dir(wal_dir: &Path) -> Result<Self> {
        fs::create_dir_all(wal_dir)?;

        let mut id_paths: Vec<(u64, PathBuf)> = fs::read_dir(wal_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                parse_wal_id(&name).map(|id| (id, e.path()))
            })
            .collect();
        id_paths.sort_by_key(|(id, _)| *id);

        if id_paths.is_empty() {
            let path = wal_dir.join(wal_filename(1));
            let file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            let seg = WalSegmentMeta {
                wal_id: 1,
                path,
                start_seq: 0,
                end_seq: None,
                byte_size: 0,
                record_count: 0,
                state: WalSegmentState::Active,
            };
            return Ok(Self {
                wal_dir: wal_dir.to_path_buf(),
                segments: vec![seg],
                writer: BufWriter::new(file),
            });
        }

        // Reconstruct metadata for existing segments.
        let mut segments = Vec::with_capacity(id_paths.len());
        for (i, (wal_id, path)) in id_paths.iter().enumerate() {
            let is_last = i == id_paths.len() - 1;
            let (start_seq, end_seq, byte_size, record_count) = scan_segment_meta(path)?;
            segments.push(WalSegmentMeta {
                wal_id: *wal_id,
                path: path.clone(),
                start_seq,
                end_seq: if is_last { None } else { end_seq },
                byte_size,
                record_count,
                state: if is_last {
                    WalSegmentState::Active
                } else {
                    WalSegmentState::Closed
                },
            });
        }

        let active_path = &segments.last().unwrap().path;
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(active_path)?;

        Ok(Self {
            wal_dir: wal_dir.to_path_buf(),
            segments,
            writer: BufWriter::new(file),
        })
    }

    /// Append a record durably. Format: 4-byte LE length + JSON body + '\n'.
    /// Rotates to a new segment file if the active file exceeds WAL_ROTATION_BYTES.
    pub(crate) fn append(&mut self, record: &WalRecord) -> Result<()> {
        let body = serde_json::to_vec(record)?;
        let len = body.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&body)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;

        let record_byte_size = (4 + body.len() + 1) as u64;
        let active = self.segments.last_mut().unwrap();
        if active.record_count == 0 {
            active.start_seq = record.seq;
        }
        active.end_seq = Some(record.seq);
        active.record_count += 1;
        active.byte_size += record_byte_size;

        if active.byte_size >= WAL_ROTATION_BYTES {
            self.rotate()?;
        }
        Ok(())
    }

    /// Close the current active segment and open a new one with the next wal_id.
    fn rotate(&mut self) -> Result<()> {
        let new_wal_id = self.segments.last().unwrap().wal_id + 1;
        self.segments.last_mut().unwrap().state = WalSegmentState::Closed;

        let new_path = self.wal_dir.join(wal_filename(new_wal_id));
        let new_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_path)?;
        self.writer = BufWriter::new(new_file);
        self.segments.push(WalSegmentMeta {
            wal_id: new_wal_id,
            path: new_path,
            start_seq: 0,
            end_seq: None,
            byte_size: 0,
            record_count: 0,
            state: WalSegmentState::Active,
        });
        Ok(())
    }

    /// Return (wal_id, path) for all segments with wal_id >= the given id, sorted.
    /// Used by the apply worker to snapshot the segment list under a brief lock.
    pub(crate) fn segment_paths_from(&self, wal_id: u64) -> Vec<(u64, PathBuf)> {
        self.segments
            .iter()
            .filter(|s| s.wal_id >= wal_id)
            .map(|s| (s.wal_id, s.path.clone()))
            .collect()
    }

    /// Read all valid records from all segments in a WAL directory, in wal_id order.
    /// Used for recovery and drain_wal.
    pub(crate) fn read_all_from_dir(wal_dir: &Path) -> Result<Vec<WalRecord>> {
        if !wal_dir.exists() {
            return Ok(vec![]);
        }
        let mut id_paths: Vec<(u64, PathBuf)> = fs::read_dir(wal_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                parse_wal_id(&name).map(|id| (id, e.path()))
            })
            .collect();
        id_paths.sort_by_key(|(id, _)| *id);

        let mut records = Vec::new();
        for (_, path) in &id_paths {
            records.extend(read_segment_records(path)?);
        }
        Ok(records)
    }

    /// Read all valid records from a legacy single-file `wal.log` (migration path).
    pub(crate) fn read_legacy(path: &Path) -> Result<Vec<WalRecord>> {
        read_segment_records(path)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Read all valid records from a single segment file.
/// Truncated last records are silently ignored (crash-resilient).
fn read_segment_records(path: &Path) -> Result<Vec<WalRecord>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut records = Vec::new();
    let mut pos = 0usize;

    while pos + 4 <= data.len() {
        let len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + len > data.len() {
            break;
        }
        let body = &data[pos..pos + len];
        match serde_json::from_slice::<WalRecord>(body) {
            Ok(rec) => records.push(rec),
            Err(_) => break,
        }
        pos += len;
        if pos < data.len() && data[pos] == b'\n' {
            pos += 1;
        }
    }

    Ok(records)
}

/// Scan a segment file to reconstruct its metadata.
/// Returns (start_seq, end_seq, byte_size, record_count).
fn scan_segment_meta(path: &Path) -> Result<(u64, Option<u64>, u64, u64)> {
    if !path.exists() {
        return Ok((0, None, 0, 0));
    }
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut pos = 0usize;
    let mut start_seq = 0u64;
    let mut end_seq = None;
    let mut record_count = 0u64;
    let mut byte_size = 0u64;

    while pos + 4 <= data.len() {
        let len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        if pos + 4 + len > data.len() {
            break;
        }
        let body = &data[pos + 4..pos + 4 + len];
        match serde_json::from_slice::<WalRecord>(body) {
            Ok(rec) => {
                byte_size += (4 + len + 1) as u64;
                if record_count == 0 {
                    start_seq = rec.seq;
                }
                end_seq = Some(rec.seq);
                record_count += 1;
                pos += 4 + len;
                if pos < data.len() && data[pos] == b'\n' {
                    pos += 1;
                }
            }
            Err(_) => break,
        }
    }

    Ok((start_seq, end_seq, byte_size, record_count))
}
