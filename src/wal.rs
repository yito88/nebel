use std::{
    fs,
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::DEFAULT_WAL_SEGMENT_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct WalId(pub u64);

impl WalId {
    pub(crate) fn first() -> Self {
        WalId(1)
    }

    pub(crate) fn next(self) -> Self {
        WalId(self.0 + 1)
    }

    pub(crate) fn filename(self) -> String {
        format!("{:06}.log", self.0)
    }

    pub(crate) fn parse(filename: &str) -> Option<Self> {
        filename.strip_suffix(".log")?.parse().ok().map(WalId)
    }
}

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
    pub wal_id: WalId,
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
    pub current_wal_id: WalId,
    pub current_offset: u64,
    pub last_applied_seq: u64,
}

// ---------------------------------------------------------------------------
// Wal
// ---------------------------------------------------------------------------

pub(crate) struct Wal {
    wal_dir: PathBuf,
    /// All segments sorted by wal_id ascending. Last entry is always Active.
    pub(crate) segments: Vec<WalSegmentMeta>,
    writer: BufWriter<fs::File>,
    pub(crate) rotation_bytes: u64,
    /// Monotonically increasing sequence counter. Allocated and incremented
    /// inside `append`, which is always called under `Mutex<Wal>`, so no
    /// atomics are needed.
    next_seq: u64,
}

impl Wal {
    /// Open (or create) a segmented WAL directory.
    ///
    /// - **Empty / absent directory**: creates `WalId::first()` as the sole Active segment.
    /// - **Existing segments**: marks every existing segment Closed (they are recovery
    ///   candidates), then creates a new Active segment at `max_id.next()`.  The caller
    ///   is responsible for draining the Closed segments via `remove_segment` once their
    ///   records have been applied.
    pub(crate) fn open_dir(wal_dir: &Path) -> Result<Self> {
        fs::create_dir_all(wal_dir)?;

        let mut id_paths: Vec<(WalId, PathBuf)> = fs::read_dir(wal_dir)?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                WalId::parse(&name).map(|id| (id, e.path()))
            })
            .collect();
        id_paths.sort_by_key(|(id, _)| *id);

        // Fresh directory — create the first segment.
        if id_paths.is_empty() {
            let path = wal_dir.join(WalId::first().filename());
            let file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            return Ok(Self {
                wal_dir: wal_dir.to_path_buf(),
                segments: vec![WalSegmentMeta {
                    wal_id: WalId::first(),
                    path,
                    start_seq: 0,
                    end_seq: None,
                    byte_size: 0,
                    record_count: 0,
                    state: WalSegmentState::Active,
                }],
                writer: BufWriter::new(file),
                rotation_bytes: DEFAULT_WAL_SEGMENT_BYTES,
                next_seq: 0,
            });
        }

        // Existing segments: mark all Closed, open a new Active at max+1.
        let mut segments = Vec::with_capacity(id_paths.len() + 1);
        for (wal_id, path) in &id_paths {
            let (start_seq, end_seq, byte_size, record_count) = scan_segment_meta(path)?;
            segments.push(WalSegmentMeta {
                wal_id: *wal_id,
                path: path.clone(),
                start_seq,
                end_seq,
                byte_size,
                record_count,
                state: WalSegmentState::Closed,
            });
        }

        let new_wal_id = segments.last().unwrap().wal_id.next();
        let new_path = wal_dir.join(new_wal_id.filename());
        let new_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&new_path)?;
        segments.push(WalSegmentMeta {
            wal_id: new_wal_id,
            path: new_path,
            start_seq: 0,
            end_seq: None,
            byte_size: 0,
            record_count: 0,
            state: WalSegmentState::Active,
        });

        Ok(Self {
            wal_dir: wal_dir.to_path_buf(),
            segments,
            writer: BufWriter::new(new_file),
            rotation_bytes: DEFAULT_WAL_SEGMENT_BYTES,
            next_seq: 0,
        })
    }

    /// Snapshot the (wal_id, path) of every Closed segment, in ascending ID order.
    /// These are the segments that need to be replayed and removed during recovery.
    pub(crate) fn recovery_segments(&self) -> Vec<(WalId, PathBuf)> {
        self.segments
            .iter()
            .filter(|s| s.state == WalSegmentState::Closed)
            .map(|s| (s.wal_id, s.path.clone()))
            .collect()
    }

    /// Remove all `Closed` segments whose last record has been applied (`end_seq <= applied_seq`).
    pub(crate) fn truncate_applied(&mut self, applied_seq: u64) {
        let to_remove: Vec<WalId> = self
            .segments
            .iter()
            .filter(|s| s.state == WalSegmentState::Closed)
            .filter(|s| s.end_seq.is_none_or(|end| end <= applied_seq))
            .map(|s| s.wal_id)
            .collect();
        for wal_id in to_remove {
            let _ = self.remove_segment(wal_id);
        }
    }

    /// Delete a segment file and remove it from the in-memory list.
    pub(crate) fn remove_segment(&mut self, wal_id: WalId) -> Result<()> {
        if let Some(pos) = self.segments.iter().position(|s| s.wal_id == wal_id) {
            let seg = self.segments.remove(pos);
            fs::remove_file(&seg.path)?;
        }
        Ok(())
    }

    /// Allocate the next sequence number, append the operation durably, and
    /// return the assigned seq. Format: 4-byte LE length + JSON body + '\n'.
    /// Rotates to a new segment file if the active file exceeds DEFAULT_WAL_SEGMENT_BYTES.
    ///
    /// Because seq allocation and WAL write happen together under `Mutex<Wal>`,
    /// seq order is guaranteed to match physical write order, which is required
    /// for the apply worker's durable-seq tracking to be correct.
    pub(crate) fn append(&mut self, op: WalOp) -> Result<u64> {
        let seq = self.write_record(op)?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        if self.segments.last().unwrap().byte_size >= self.rotation_bytes {
            self.rotate()?;
        }
        Ok(seq)
    }

    /// Like [`append`], but writes all ops under a single fsync.
    /// Returns the seq of the last record. Caller must ensure `ops` is non-empty.
    pub(crate) fn append_batch(&mut self, ops: Vec<WalOp>) -> Result<u64> {
        debug_assert!(!ops.is_empty());
        let mut last_seq = 0u64;
        for op in ops {
            last_seq = self.write_record(op)?;
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        if self.segments.last().unwrap().byte_size >= self.rotation_bytes {
            self.rotate()?;
        }
        Ok(last_seq)
    }

    /// Write one record to the `BufWriter` and update segment metadata.
    /// Does NOT flush or sync — caller is responsible.
    fn write_record(&mut self, op: WalOp) -> Result<u64> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let record = WalRecord { seq, op };
        let body = serde_json::to_vec(&record)?;
        let len = body.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&body)?;
        self.writer.write_all(b"\n")?;
        let record_byte_size = (4 + body.len() + 1) as u64;
        let active = self.segments.last_mut().unwrap();
        if active.record_count == 0 {
            active.start_seq = seq;
        }
        active.end_seq = Some(seq);
        active.record_count += 1;
        active.byte_size += record_byte_size;
        Ok(seq)
    }

    /// Reset the next sequence number. Called after recovery to continue from
    /// the highest applied seq rather than from 0.
    pub(crate) fn reset_next_seq(&mut self, next: u64) {
        self.next_seq = next;
    }

    /// Close the current active segment and open a new one with the next wal_id.
    fn rotate(&mut self) -> Result<()> {
        let new_wal_id = self.segments.last().unwrap().wal_id.next();
        self.segments.last_mut().unwrap().state = WalSegmentState::Closed;

        let new_path = self.wal_dir.join(new_wal_id.filename());
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
    pub(crate) fn segment_paths_from(&self, wal_id: WalId) -> Vec<(WalId, PathBuf)> {
        self.segments
            .iter()
            .filter(|s| s.wal_id >= wal_id)
            .map(|s| (s.wal_id, s.path.clone()))
            .collect()
    }

    /// Read all valid records from a single segment file.
    pub(crate) fn read_segment(path: &Path) -> Result<Vec<WalRecord>> {
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
