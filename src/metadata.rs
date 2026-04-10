use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

pub type FieldId = u32;
pub type DocOrd = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int64,
    Float64,
    Bool,
    Bytes,
    Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub id: FieldId,
    pub name: String,
    pub ty: FieldType,
    pub filterable: bool,
}

/// The collection-level metadata schema (immutable after creation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSchema {
    pub fields: Vec<FieldSchema>,
}

impl MetadataSchema {
    pub fn field_by_name(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn field_by_id(&self, id: FieldId) -> Option<&FieldSchema> {
        self.fields.iter().find(|f| f.id == id)
    }

    /// Validate a name-keyed metadata map against the schema and return a field_id-keyed map.
    /// Returns an error on unknown fields, type mismatches, or invalid Float64 values.
    pub fn validate(
        &self,
        raw: &HashMap<String, MetadataValue>,
    ) -> Result<HashMap<FieldId, MetadataValue>> {
        let mut out = HashMap::with_capacity(raw.len());
        for (name, value) in raw {
            let field = self
                .field_by_name(name)
                .ok_or_else(|| anyhow!("unknown metadata field '{}'", name))?;
            let matches = match (&field.ty, value) {
                (FieldType::String, MetadataValue::String(_)) => true,
                (FieldType::Int64, MetadataValue::Int64(_)) => true,
                (FieldType::Float64, MetadataValue::Float64(f)) => {
                    if f.is_nan() || f.is_infinite() {
                        bail!("field '{}': Float64 must be finite (got {})", name, f);
                    }
                    true
                }
                (FieldType::Bool, MetadataValue::Bool(_)) => true,
                (FieldType::Bytes, MetadataValue::Bytes(_)) => true,
                (FieldType::Timestamp, MetadataValue::Timestamp(_)) => true,
                _ => false,
            };
            if !matches {
                bail!(
                    "field '{}': expected {:?}, got {}",
                    name,
                    field.ty,
                    value.type_name()
                );
            }
            out.insert(field.id, value.clone());
        }
        Ok(out)
    }

    /// Convert a name-keyed map to a field_id-keyed map, silently skipping unknown fields.
    /// Used at apply time when the metadata was already validated at write time.
    pub fn names_to_ids(
        &self,
        raw: &HashMap<String, MetadataValue>,
    ) -> HashMap<FieldId, MetadataValue> {
        raw.iter()
            .filter_map(|(name, value)| self.field_by_name(name).map(|f| (f.id, value.clone())))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    String(String),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Bytes(Vec<u8>),
    Timestamp(i64), // epoch micros
}

impl MetadataValue {
    fn type_name(&self) -> &'static str {
        match self {
            MetadataValue::String(_) => "String",
            MetadataValue::Int64(_) => "Int64",
            MetadataValue::Float64(_) => "Float64",
            MetadataValue::Bool(_) => "Bool",
            MetadataValue::Bytes(_) => "Bytes",
            MetadataValue::Timestamp(_) => "Timestamp",
        }
    }
}

// ---------------------------------------------------------------------------
// Binary encoding / decoding
// ---------------------------------------------------------------------------

#[repr(u8)]
enum ValueTag {
    String = 1,
    Int64 = 2,
    Float64 = 3,
    Bool = 4,
    Bytes = 5,
    Timestamp = 6,
}

impl ValueTag {
    fn try_from(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::String),
            2 => Some(Self::Int64),
            3 => Some(Self::Float64),
            4 => Some(Self::Bool),
            5 => Some(Self::Bytes),
            6 => Some(Self::Timestamp),
            _ => None,
        }
    }
}

/// Encode a `MetadataValue` to bytes: `[tag:u8][payload...]`.
pub fn encode_value(v: &MetadataValue) -> Vec<u8> {
    match v {
        MetadataValue::String(s) => {
            let mut out = vec![ValueTag::String as u8];
            out.extend_from_slice(s.as_bytes());
            out
        }
        MetadataValue::Int64(i) => {
            let mut out = vec![ValueTag::Int64 as u8];
            out.extend_from_slice(&i.to_be_bytes());
            out
        }
        MetadataValue::Float64(f) => {
            let mut out = vec![ValueTag::Float64 as u8];
            out.extend_from_slice(&f.to_be_bytes());
            out
        }
        MetadataValue::Bool(b) => vec![ValueTag::Bool as u8, u8::from(*b)],
        MetadataValue::Bytes(b) => {
            let mut out = vec![ValueTag::Bytes as u8];
            out.extend_from_slice(b);
            out
        }
        MetadataValue::Timestamp(t) => {
            let mut out = vec![ValueTag::Timestamp as u8];
            out.extend_from_slice(&t.to_be_bytes());
            out
        }
    }
}

/// Decode bytes produced by [`encode_value`] back into a `MetadataValue`.
pub fn decode_value(b: &[u8]) -> Result<MetadataValue> {
    if b.is_empty() {
        bail!("empty metadata value bytes");
    }
    match ValueTag::try_from(b[0]) {
        Some(ValueTag::String) => {
            let s = std::str::from_utf8(&b[1..])
                .map_err(|_| anyhow!("invalid UTF-8 in String metadata value"))?;
            Ok(MetadataValue::String(s.to_string()))
        }
        Some(ValueTag::Int64) => {
            if b.len() < 9 {
                bail!("truncated Int64 metadata value");
            }
            Ok(MetadataValue::Int64(i64::from_be_bytes(
                b[1..9].try_into().unwrap(),
            )))
        }
        Some(ValueTag::Float64) => {
            if b.len() < 9 {
                bail!("truncated Float64 metadata value");
            }
            Ok(MetadataValue::Float64(f64::from_be_bytes(
                b[1..9].try_into().unwrap(),
            )))
        }
        Some(ValueTag::Bool) => {
            if b.len() < 2 {
                bail!("truncated Bool metadata value");
            }
            Ok(MetadataValue::Bool(b[1] != 0))
        }
        Some(ValueTag::Bytes) => Ok(MetadataValue::Bytes(b[1..].to_vec())),
        Some(ValueTag::Timestamp) => {
            if b.len() < 9 {
                bail!("truncated Timestamp metadata value");
            }
            Ok(MetadataValue::Timestamp(i64::from_be_bytes(
                b[1..9].try_into().unwrap(),
            )))
        }
        None => bail!("unknown metadata value tag: {}", b[0]),
    }
}

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

/// Build the METADATA_VALUES redb key: `"{collection}\0" + doc_ord_be8 + field_id_be4`.
pub fn meta_value_key(collection: &str, doc_ord: DocOrd, field_id: FieldId) -> Vec<u8> {
    let mut key = format!("{}\0", collection).into_bytes();
    key.extend_from_slice(&doc_ord.to_be_bytes());
    key.extend_from_slice(&field_id.to_be_bytes());
    key
}

/// Build the prefix for all METADATA_VALUES entries of one document:
/// `"{collection}\0" + doc_ord_be8`.
pub fn meta_value_prefix(collection: &str, doc_ord: DocOrd) -> Vec<u8> {
    let mut key = format!("{}\0", collection).into_bytes();
    key.extend_from_slice(&doc_ord.to_be_bytes());
    key
}

/// Build the DOC_ORD_TO_ID key: `"{collection}\0" + doc_ord_be8`.
pub fn doc_ord_key(collection: &str, doc_ord: DocOrd) -> Vec<u8> {
    let mut key = format!("{}\0", collection).into_bytes();
    key.extend_from_slice(&doc_ord.to_be_bytes());
    key
}
