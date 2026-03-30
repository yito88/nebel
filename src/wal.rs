use std::{
    fs,
    io::{BufWriter, Read, Write},
    path::Path,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

pub(crate) struct Wal {
    writer: BufWriter<fs::File>,
}

impl Wal {
    /// Open or create a WAL file. Appends to existing content.
    pub(crate) fn open(path: &Path) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Append a record durably. Format: 4-byte LE length + JSON body + '\n'.
    pub(crate) fn append(&mut self, record: &WalRecord) -> Result<()> {
        let body = serde_json::to_vec(record)?;
        let len = body.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&body)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }

    /// Read all valid records from a WAL file.
    /// Truncated last records are silently ignored (crash-resilient).
    pub(crate) fn read_from(path: &Path) -> Result<Vec<WalRecord>> {
        if !path.exists() {
            return Ok(vec![]);
        }
        let mut file = fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut records = Vec::new();
        let mut pos = 0usize;

        while pos + 4 <= data.len() {
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if pos + len > data.len() {
                // Truncated record — stop here.
                break;
            }
            let body = &data[pos..pos + len];
            match serde_json::from_slice::<WalRecord>(body) {
                Ok(rec) => records.push(rec),
                Err(_) => break, // Corrupted record — stop
            }
            pos += len;
            // Skip optional newline.
            if pos < data.len() && data[pos] == b'\n' {
                pos += 1;
            }
        }

        Ok(records)
    }
}
