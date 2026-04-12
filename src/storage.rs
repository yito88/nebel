use std::collections::HashMap;

use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};

use crate::{
    filter::{FilterExpr, FilterResult, eval_expr},
    metadata::{
        DocOrd, FieldId, MetadataSchema, MetadataValue, decode_value, doc_ord_key, encode_value,
        meta_value_key, meta_value_prefix,
    },
    types::{
        CollectionId, CollectionSchema, DocLocation, InternalId, Manifest, SegId, SegmentMeta,
        VectorEntry,
    },
};

/// `(doc_id, prev_location)` — `prev_location` is the DOC_MAP value at the time of the read,
/// used by compaction to detect concurrent updates.
type LiveEntries = Vec<Option<(String, Option<DocLocation>)>>;

// ---------------------------------------------------------------------------
// Table definitions
// ---------------------------------------------------------------------------

/// collection_name -> CollectionSchema (JSON)
const COLLECTIONS: TableDefinition<&str, &str> = TableDefinition::new("collections");

/// "{collection}\0{seg_id}" -> SegmentMeta (JSON)
const SEGMENTS: TableDefinition<&str, &str> = TableDefinition::new("segments");

/// "{collection}\0{doc_id}" -> DocLocation (JSON)
const DOC_MAP: TableDefinition<&str, &str> = TableDefinition::new("doc_map");

/// "{collection}\0{seg_id}\0{internal_id}" -> bool (true = deleted)
const TOMBSTONES: TableDefinition<&str, bool> = TableDefinition::new("tombstones");

/// "{collection}\0{seg_id}\0{internal_id}" -> doc_id string
const REVERSE_DOC: TableDefinition<&str, &str> = TableDefinition::new("reverse_doc");

/// collection_name -> Manifest (JSON)
const MANIFESTS: TableDefinition<&str, &str> = TableDefinition::new("manifests");

/// collection_name -> last applied WAL seq (u64)
const APPLIED_SEQ: TableDefinition<&str, u64> = TableDefinition::new("applied_seq");

// --- New metadata tables ---

/// "{collection}\0{field_name}" -> field_id (u32)
const FIELD_NAME_TO_ID: TableDefinition<&str, u32> = TableDefinition::new("field_name_to_id");

/// "{collection}\0{field_id:010}" -> FieldSchema (JSON)
const FIELD_ID_TO_SCHEMA: TableDefinition<&str, &str> = TableDefinition::new("field_id_to_schema");

/// "{collection}\0{doc_id}" -> doc_ord (u64)
const DOC_ID_TO_ORD: TableDefinition<&str, u64> = TableDefinition::new("doc_id_to_ord");

/// "{collection}\0" + doc_ord_be8 -> doc_id string
const DOC_ORD_TO_ID: TableDefinition<&[u8], &str> = TableDefinition::new("doc_ord_to_id");

/// "{collection}\0" + doc_ord_be8 + field_id_be4 -> [tag][payload]
const METADATA_VALUES: TableDefinition<&[u8], &[u8]> = TableDefinition::new("metadata_values");

/// collection_name -> next_doc_ord (u64)
const COLLECTION_META: TableDefinition<&str, u64> = TableDefinition::new("collection_meta");

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

fn seg_key(collection: &str, seg_id: SegId) -> String {
    format!("{}\0{}", collection, seg_id)
}

fn doc_key(collection: &str, doc_id: &str) -> String {
    format!("{}\0{}", collection, doc_id)
}

fn meta_key(collection: &str, seg_id: SegId, internal_id: InternalId) -> String {
    format!("{}\0{}\0{}", collection, seg_id, internal_id)
}

fn field_name_key(collection: &str, field_name: &str) -> String {
    format!("{}\0{}", collection, field_name)
}

fn field_id_key(collection: &str, field_id: FieldId) -> String {
    format!("{}\0{:010}", collection, field_id)
}

fn doc_id_to_ord_key(collection: &str, doc_id: &str) -> String {
    format!("{}\0{}", collection, doc_id)
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

pub struct Storage {
    db: Database,
}

impl Storage {
    /// Open (or create) the redb database and ensure all tables exist.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let db = Database::create(path)?;
        let wtxn = db.begin_write()?;
        wtxn.open_table(COLLECTIONS)?;
        wtxn.open_table(SEGMENTS)?;
        wtxn.open_table(DOC_MAP)?;
        wtxn.open_table(TOMBSTONES)?;
        wtxn.open_table(REVERSE_DOC)?;
        wtxn.open_table(MANIFESTS)?;
        wtxn.open_table(APPLIED_SEQ)?;
        wtxn.open_table(FIELD_NAME_TO_ID)?;
        wtxn.open_table(FIELD_ID_TO_SCHEMA)?;
        wtxn.open_table(DOC_ID_TO_ORD)?;
        wtxn.open_table(DOC_ORD_TO_ID)?;
        wtxn.open_table(METADATA_VALUES)?;
        wtxn.open_table(COLLECTION_META)?;
        wtxn.commit()?;
        Ok(Self { db })
    }

    /// Atomically persist a new collection schema, manifest, and initial segment metadata.
    pub fn put_new_collection(
        &self,
        schema: &CollectionSchema,
        manifest: &Manifest,
        seg_meta: &SegmentMeta,
    ) -> Result<()> {
        let name = schema.name.as_str();
        let schema_json = serde_json::to_string(schema)?;
        let manifest_json = serde_json::to_string(manifest)?;
        let sk = seg_key(name, seg_meta.seg_id);
        let seg_json = serde_json::to_string(seg_meta)?;
        let wtxn = self.db.begin_write()?;
        {
            let mut col_table = wtxn.open_table(COLLECTIONS)?;
            col_table.insert(name, schema_json.as_str())?;
            let mut manifest_table = wtxn.open_table(MANIFESTS)?;
            manifest_table.insert(name, manifest_json.as_str())?;
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            seg_table.insert(sk.as_str(), seg_json.as_str())?;

            // Initialize DocOrd counter and write field schemas if metadata schema is present.
            let mut col_meta_table = wtxn.open_table(COLLECTION_META)?;
            col_meta_table.insert(name, 0u64)?;

            if let Some(ms) = &schema.metadata_schema {
                let mut fn_table = wtxn.open_table(FIELD_NAME_TO_ID)?;
                let mut fi_table = wtxn.open_table(FIELD_ID_TO_SCHEMA)?;
                for field in &ms.fields {
                    let fk = field_name_key(name, &field.name);
                    fn_table.insert(fk.as_str(), field.id)?;
                    let fik = field_id_key(name, field.id);
                    let field_json = serde_json::to_string(field)?;
                    fi_table.insert(fik.as_str(), field_json.as_str())?;
                }
            }
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Load a collection schema by name.
    pub fn get_collection(&self, id: &CollectionId) -> Result<Option<CollectionSchema>> {
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(COLLECTIONS)?;
        match table.get(id.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Load the manifest for a collection.
    pub fn get_manifest(&self, id: &CollectionId) -> Result<Option<Manifest>> {
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(MANIFESTS)?;
        match table.get(id.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Atomically seal an old segment, register a new writable segment, and update the manifest.
    pub fn seal_segment(
        &self,
        id: &CollectionId,
        sealed_meta: &SegmentMeta,
        new_meta: &SegmentMeta,
        manifest: &Manifest,
    ) -> Result<()> {
        let col = id.as_str();
        let sealed_key = seg_key(col, sealed_meta.seg_id);
        let sealed_json = serde_json::to_string(sealed_meta)?;
        let new_key = seg_key(col, new_meta.seg_id);
        let new_json = serde_json::to_string(new_meta)?;
        let manifest_json = serde_json::to_string(manifest)?;

        let wtxn = self.db.begin_write()?;
        {
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            seg_table.insert(sealed_key.as_str(), sealed_json.as_str())?;
            seg_table.insert(new_key.as_str(), new_json.as_str())?;
            let mut manifest_table = wtxn.open_table(MANIFESTS)?;
            manifest_table.insert(col, manifest_json.as_str())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Load segment metadata.
    pub fn get_segment(&self, id: &CollectionId, seg_id: SegId) -> Result<Option<SegmentMeta>> {
        let key = seg_key(id.as_str(), seg_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(SEGMENTS)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Resolve a batch of candidates in a single read transaction.
    ///
    /// For each `(seg_id, internal_id)`, checks the tombstone, looks up the doc_id,
    /// and optionally fetches typed metadata from `METADATA_VALUES`.
    /// Returns one `Option<ResolvedCandidate>` per input — `None` if tombstoned or missing.
    pub fn resolve_candidates(
        &self,
        id: &CollectionId,
        candidates: &[(SegId, InternalId)],
        include_metadata: bool,
        need_doc_ord: bool,
        metadata_schema: Option<&MetadataSchema>,
    ) -> Result<Vec<Option<ResolvedCandidate>>> {
        let col = id.as_str();
        let rtxn = self.db.begin_read()?;
        let tomb_table = rtxn.open_table(TOMBSTONES)?;
        let rev_table = rtxn.open_table(REVERSE_DOC)?;
        let open_ord = (include_metadata && metadata_schema.is_some()) || need_doc_ord;
        let doc_ord_table = if open_ord {
            Some(rtxn.open_table(DOC_ID_TO_ORD)?)
        } else {
            None
        };
        let meta_table = if include_metadata && metadata_schema.is_some() {
            Some(rtxn.open_table(METADATA_VALUES)?)
        } else {
            None
        };

        let mut results = Vec::with_capacity(candidates.len());
        for &(seg_id, internal_id) in candidates {
            let key = meta_key(col, seg_id, internal_id);
            if tomb_table
                .get(key.as_str())?
                .map(|v| v.value())
                .unwrap_or(false)
            {
                results.push(None);
                continue;
            }
            let doc_id = match rev_table.get(key.as_str())? {
                Some(v) => v.value().to_string(),
                None => {
                    results.push(None);
                    continue;
                }
            };

            let mut resolved_ord: Option<DocOrd> = None;
            let metadata = if let Some(ord_table) = doc_ord_table.as_ref() {
                let dk = doc_id_to_ord_key(col, &doc_id);
                if let Some(ord_v) = ord_table.get(dk.as_str())? {
                    let doc_ord = ord_v.value();
                    resolved_ord = Some(doc_ord);

                    if let (Some(mv_table), Some(schema)) = (meta_table.as_ref(), metadata_schema) {
                        let prefix = meta_value_prefix(col, doc_ord);
                        let end = meta_value_key(col, doc_ord, u32::MAX);
                        let mut map = HashMap::new();
                        for entry in mv_table.range(prefix.as_slice()..=end.as_slice())? {
                            let (k, v) = entry?;
                            let key_bytes = k.value();
                            if key_bytes.len() < 4 {
                                continue;
                            }
                            let field_id = u32::from_be_bytes(
                                key_bytes[key_bytes.len() - 4..].try_into().unwrap(),
                            );
                            if let (Some(field), Ok(val)) =
                                (schema.field_by_id(field_id), decode_value(v.value()))
                            {
                                map.insert(field.name.clone(), val);
                            }
                        }
                        if map.is_empty() { None } else { Some(map) }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            results.push(Some(ResolvedCandidate {
                doc_id,
                doc_ord: resolved_ord,
                metadata,
            }));
        }
        Ok(results)
    }

    /// Atomically apply an upsert WAL record: tombstone the old location (if any),
    /// write new segment metadata, doc/reverse entries, DocOrd mappings, typed metadata
    /// values, and advance applied_seq.
    pub fn apply_upsert(
        &self,
        id: &CollectionId,
        doc_id: &str,
        seg_meta: &SegmentMeta,
        entries: &[VectorEntry<'_>],
        seq: u64,
    ) -> Result<()> {
        let col = id.as_str();
        let sk = seg_key(col, seg_meta.seg_id);
        let seg_json = serde_json::to_string(seg_meta)?;
        let old_dk = doc_key(col, doc_id);

        let wtxn = self.db.begin_write()?;
        {
            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let old_loc: Option<DocLocation> = match doc_table.get(old_dk.as_str())? {
                Some(v) => Some(serde_json::from_str(v.value())?),
                None => None,
            };
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            if let Some(loc) = &old_loc {
                let tk = meta_key(col, loc.seg_id, loc.internal_id);
                let mut tomb_table = wtxn.open_table(TOMBSTONES)?;
                tomb_table.insert(tk.as_str(), true)?;
                let old_sk = seg_key(col, loc.seg_id);
                let updated_meta = if let Some(v) = seg_table.get(old_sk.as_str())? {
                    let mut meta: SegmentMeta = serde_json::from_str(v.value())?;
                    meta.tombstone_count += 1;
                    Some(serde_json::to_string(&meta)?)
                } else {
                    None
                };
                if let Some(updated) = updated_meta {
                    seg_table.insert(old_sk.as_str(), updated.as_str())?;
                }
            }
            seg_table.insert(sk.as_str(), seg_json.as_str())?;

            let mut rev_table = wtxn.open_table(REVERSE_DOC)?;
            let mut ord_table = wtxn.open_table(DOC_ID_TO_ORD)?;
            let mut ord_rev_table = wtxn.open_table(DOC_ORD_TO_ID)?;
            let mut meta_vals_table = wtxn.open_table(METADATA_VALUES)?;
            let mut col_meta_table = wtxn.open_table(COLLECTION_META)?;

            let mut next_doc_ord = col_meta_table.get(col)?.map(|v| v.value()).unwrap_or(0);

            for e in entries {
                let dk = doc_key(col, e.doc_id);
                let loc = DocLocation {
                    seg_id: seg_meta.seg_id,
                    internal_id: e.internal_id,
                };
                let loc_json = serde_json::to_string(&loc)?;
                doc_table.insert(dk.as_str(), loc_json.as_str())?;

                let mk = meta_key(col, seg_meta.seg_id, e.internal_id);
                rev_table.insert(mk.as_str(), e.doc_id)?;

                // Assign or look up DocOrd.
                let ord_key = doc_id_to_ord_key(col, e.doc_id);
                let doc_ord = if let Some(v) = ord_table.get(ord_key.as_str())? {
                    v.value()
                } else {
                    let new_ord = next_doc_ord;
                    next_doc_ord += 1;
                    ord_table.insert(ord_key.as_str(), new_ord)?;
                    let ok = doc_ord_key(col, new_ord);
                    ord_rev_table.insert(ok.as_slice(), e.doc_id)?;
                    new_ord
                };

                // Write typed metadata values.
                if let Some(meta_map) = e.metadata {
                    for (&field_id, value) in meta_map {
                        let vk = meta_value_key(col, doc_ord, field_id);
                        let encoded = encode_value(value);
                        meta_vals_table.insert(vk.as_slice(), encoded.as_slice())?;
                    }
                }
            }

            col_meta_table.insert(col, next_doc_ord)?;

            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Atomically apply a batch of upsert WAL records in a single redb transaction.
    pub fn apply_upsert_batch(
        &self,
        id: &CollectionId,
        seg_meta: &SegmentMeta,
        entries: &[VectorEntry<'_>],
        seq: u64,
    ) -> Result<()> {
        let col = id.as_str();
        let sk = seg_key(col, seg_meta.seg_id);
        let seg_json = serde_json::to_string(seg_meta)?;

        let wtxn = self.db.begin_write()?;
        {
            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            let mut tomb_table = wtxn.open_table(TOMBSTONES)?;

            // Tombstone old locations, accumulate per-segment tombstone counts.
            let mut tombstone_increments: HashMap<SegId, usize> = HashMap::new();
            for e in entries {
                let dk = doc_key(col, e.doc_id);
                if let Some(v) = doc_table.get(dk.as_str())? {
                    let old_loc: DocLocation = serde_json::from_str(v.value())?;
                    let tk = meta_key(col, old_loc.seg_id, old_loc.internal_id);
                    tomb_table.insert(tk.as_str(), true)?;
                    *tombstone_increments.entry(old_loc.seg_id).or_default() += 1;
                }
            }

            // Flush tombstone_count increments per segment.
            for (old_seg_id, inc) in &tombstone_increments {
                let old_sk = seg_key(col, *old_seg_id);
                let updated = if let Some(v) = seg_table.get(old_sk.as_str())? {
                    let mut meta: SegmentMeta = serde_json::from_str(v.value())?;
                    meta.tombstone_count += inc;
                    Some(serde_json::to_string(&meta)?)
                } else {
                    None
                };
                if let Some(updated) = updated {
                    seg_table.insert(old_sk.as_str(), updated.as_str())?;
                }
            }

            seg_table.insert(sk.as_str(), seg_json.as_str())?;

            let mut rev_table = wtxn.open_table(REVERSE_DOC)?;
            let mut ord_table = wtxn.open_table(DOC_ID_TO_ORD)?;
            let mut ord_rev_table = wtxn.open_table(DOC_ORD_TO_ID)?;
            let mut meta_vals_table = wtxn.open_table(METADATA_VALUES)?;
            let mut col_meta_table = wtxn.open_table(COLLECTION_META)?;

            let mut next_doc_ord = col_meta_table.get(col)?.map(|v| v.value()).unwrap_or(0);

            for e in entries {
                let dk = doc_key(col, e.doc_id);
                let loc = DocLocation {
                    seg_id: seg_meta.seg_id,
                    internal_id: e.internal_id,
                };
                let loc_json = serde_json::to_string(&loc)?;
                doc_table.insert(dk.as_str(), loc_json.as_str())?;

                let mk = meta_key(col, seg_meta.seg_id, e.internal_id);
                rev_table.insert(mk.as_str(), e.doc_id)?;

                // Assign or look up DocOrd.
                let ord_key = doc_id_to_ord_key(col, e.doc_id);
                let doc_ord = if let Some(v) = ord_table.get(ord_key.as_str())? {
                    v.value()
                } else {
                    let new_ord = next_doc_ord;
                    next_doc_ord += 1;
                    ord_table.insert(ord_key.as_str(), new_ord)?;
                    let ok = doc_ord_key(col, new_ord);
                    ord_rev_table.insert(ok.as_slice(), e.doc_id)?;
                    new_ord
                };

                // Write typed metadata values.
                if let Some(meta_map) = e.metadata {
                    for (&field_id, value) in meta_map {
                        let vk = meta_value_key(col, doc_ord, field_id);
                        let encoded = encode_value(value);
                        meta_vals_table.insert(vk.as_slice(), encoded.as_slice())?;
                    }
                }
            }

            col_meta_table.insert(col, next_doc_ord)?;

            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Atomically apply a delete WAL record: tombstone and remove doc location (if it exists),
    /// clean up DocOrd and metadata values, and advance applied_seq.
    pub fn apply_delete(&self, id: &CollectionId, doc_id: &str, seq: u64) -> Result<()> {
        let col = id.as_str();
        let dk = doc_key(col, doc_id);
        let wtxn = self.db.begin_write()?;
        {
            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let old_loc: Option<DocLocation> = match doc_table.get(dk.as_str())? {
                Some(v) => Some(serde_json::from_str(v.value())?),
                None => None,
            };
            if let Some(loc) = &old_loc {
                let tk = meta_key(col, loc.seg_id, loc.internal_id);
                let mut tomb_table = wtxn.open_table(TOMBSTONES)?;
                tomb_table.insert(tk.as_str(), true)?;
                doc_table.remove(dk.as_str())?;
                let old_sk = seg_key(col, loc.seg_id);
                let mut seg_table = wtxn.open_table(SEGMENTS)?;
                let updated_meta = if let Some(v) = seg_table.get(old_sk.as_str())? {
                    let mut meta: SegmentMeta = serde_json::from_str(v.value())?;
                    meta.tombstone_count += 1;
                    Some(serde_json::to_string(&meta)?)
                } else {
                    None
                };
                if let Some(updated) = updated_meta {
                    seg_table.insert(old_sk.as_str(), updated.as_str())?;
                }
            }

            // Remove DocOrd mappings and METADATA_VALUES.
            let ord_key = doc_id_to_ord_key(col, doc_id);
            let mut ord_table = wtxn.open_table(DOC_ID_TO_ORD)?;
            if let Some(ord_v) = ord_table.remove(ord_key.as_str())? {
                let doc_ord = ord_v.value();
                let ok = doc_ord_key(col, doc_ord);
                let mut ord_rev_table = wtxn.open_table(DOC_ORD_TO_ID)?;
                ord_rev_table.remove(ok.as_slice())?;

                // Range-delete all METADATA_VALUES entries for this doc_ord.
                let prefix = meta_value_prefix(col, doc_ord);
                let end = meta_value_key(col, doc_ord, u32::MAX);
                let mut meta_vals_table = wtxn.open_table(METADATA_VALUES)?;
                let to_delete: Vec<Vec<u8>> = meta_vals_table
                    .range(prefix.as_slice()..=end.as_slice())?
                    .filter_map(|e| e.ok())
                    .map(|(k, _)| k.value().to_vec())
                    .collect();
                for k in &to_delete {
                    meta_vals_table.remove(k.as_slice())?;
                }
            }

            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Atomically apply an update-metadata WAL record: look up DocOrd and write new typed
    /// metadata values for fields present in `meta`, then advance applied_seq.
    pub fn apply_update_metadata(
        &self,
        id: &CollectionId,
        doc_id: &str,
        meta: &HashMap<FieldId, MetadataValue>,
        seq: u64,
    ) -> Result<()> {
        let col = id.as_str();
        let ord_key = doc_id_to_ord_key(col, doc_id);
        let wtxn = self.db.begin_write()?;
        {
            let ord_table = wtxn.open_table(DOC_ID_TO_ORD)?;
            if let Some(ord_v) = ord_table.get(ord_key.as_str())? {
                let doc_ord = ord_v.value();
                let mut meta_vals_table = wtxn.open_table(METADATA_VALUES)?;
                for (&field_id, value) in meta {
                    let vk = meta_value_key(col, doc_ord, field_id);
                    let encoded = encode_value(value);
                    meta_vals_table.insert(vk.as_slice(), encoded.as_slice())?;
                }
            }
            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Load all tombstoned `(seg_id, internal_id)` pairs for a collection in
    /// a single read transaction.
    #[cfg(feature = "testing")]
    pub fn load_tombstone_set(&self, id: &CollectionId) -> Result<TombstoneSet> {
        let col = id.as_str();
        let prefix = format!("{}\0", col);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(TOMBSTONES)?;
        let mut set = TombstoneSet::new();
        for entry in table.iter()? {
            let (k, v) = entry?;
            let key = k.value();
            if !key.starts_with(&prefix) {
                continue;
            }
            if !v.value() {
                continue;
            }
            let rest = &key[prefix.len()..];
            if let Some((seg_str, id_str)) = rest.split_once('\0')
                && let (Ok(seg_id), Ok(internal_id)) =
                    (seg_str.parse::<u32>(), id_str.parse::<u32>())
            {
                set.insert((SegId::from_u32(seg_id), InternalId::from_u32(internal_id)));
            }
        }
        Ok(set)
    }

    /// Write segment metadata, doc locations, reverse-doc mappings, DocOrd mappings,
    /// and typed metadata values in a single redb transaction.
    pub fn write_vector_entries(
        &self,
        id: &CollectionId,
        seg_meta: &SegmentMeta,
        entries: &[VectorEntry<'_>],
    ) -> Result<()> {
        let col = id.as_str();
        let sk = seg_key(col, seg_meta.seg_id);
        let seg_json = serde_json::to_string(seg_meta)?;

        let wtxn = self.db.begin_write()?;
        {
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            seg_table.insert(sk.as_str(), seg_json.as_str())?;

            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let mut rev_table = wtxn.open_table(REVERSE_DOC)?;
            let mut ord_table = wtxn.open_table(DOC_ID_TO_ORD)?;
            let mut ord_rev_table = wtxn.open_table(DOC_ORD_TO_ID)?;
            let mut meta_vals_table = wtxn.open_table(METADATA_VALUES)?;
            let mut col_meta_table = wtxn.open_table(COLLECTION_META)?;

            let mut next_doc_ord = col_meta_table.get(col)?.map(|v| v.value()).unwrap_or(0);

            for e in entries {
                let dk = doc_key(col, e.doc_id);
                let loc = DocLocation {
                    seg_id: seg_meta.seg_id,
                    internal_id: e.internal_id,
                };
                let loc_json = serde_json::to_string(&loc)?;
                doc_table.insert(dk.as_str(), loc_json.as_str())?;

                let mk = meta_key(col, seg_meta.seg_id, e.internal_id);
                rev_table.insert(mk.as_str(), e.doc_id)?;

                let ord_key = doc_id_to_ord_key(col, e.doc_id);
                let doc_ord = if let Some(v) = ord_table.get(ord_key.as_str())? {
                    v.value()
                } else {
                    let new_ord = next_doc_ord;
                    next_doc_ord += 1;
                    ord_table.insert(ord_key.as_str(), new_ord)?;
                    let ok = doc_ord_key(col, new_ord);
                    ord_rev_table.insert(ok.as_slice(), e.doc_id)?;
                    new_ord
                };

                if let Some(meta_map) = e.metadata {
                    for (&field_id, value) in meta_map {
                        let vk = meta_value_key(col, doc_ord, field_id);
                        let encoded = encode_value(value);
                        meta_vals_table.insert(vk.as_slice(), encoded.as_slice())?;
                    }
                }
            }

            col_meta_table.insert(col, next_doc_ord)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Scan all live documents for a collection, evaluate `filter` against their
    /// typed metadata, and return the matching [`FilterResult`].
    ///
    /// "Live" means the document has an entry in `DOC_ID_TO_ORD` (i.e. it has
    /// been upserted at least once and not yet deleted).
    pub fn evaluate_filter(
        &self,
        id: &CollectionId,
        filter: &FilterExpr,
        schema: &MetadataSchema,
    ) -> Result<FilterResult> {
        let col = id.as_str();
        let prefix = format!("{}\0", col);

        let rtxn = self.db.begin_read()?;
        let ord_table = rtxn.open_table(DOC_ID_TO_ORD)?;
        let mv_table = rtxn.open_table(METADATA_VALUES)?;

        let mut matching = Vec::new();

        for entry in ord_table.iter()? {
            let (k, v) = entry?;
            let key = k.value();
            if !key.starts_with(&prefix) {
                continue;
            }
            let doc_ord: DocOrd = v.value();

            // Load all metadata fields for this doc_ord.
            let meta_prefix = meta_value_prefix(col, doc_ord);
            let meta_end = meta_value_key(col, doc_ord, u32::MAX);
            let mut fields: HashMap<FieldId, MetadataValue> = HashMap::new();
            for meta_entry in mv_table.range(meta_prefix.as_slice()..=meta_end.as_slice())? {
                let (mk, mv) = meta_entry?;
                let key_bytes = mk.value();
                if key_bytes.len() < 4 {
                    continue;
                }
                let field_id =
                    u32::from_be_bytes(key_bytes[key_bytes.len() - 4..].try_into().unwrap());
                if schema.field_by_id(field_id).is_some()
                    && let Ok(val) = decode_value(mv.value())
                {
                    fields.insert(field_id, val);
                }
            }

            if eval_expr(filter, &fields) {
                matching.push(doc_ord);
            }
        }

        let matched_count = matching.len();
        Ok(FilterResult {
            doc_ords: matching,
            matched_count,
        })
    }

    /// Resolve doc_id and current [`DocLocation`] for each `doc_ord`.
    ///
    /// Returns `None` for a given doc_ord if it was deleted between filter
    /// evaluation and this call (defensive).
    pub fn get_doc_locations(
        &self,
        id: &CollectionId,
        doc_ords: &[DocOrd],
    ) -> Result<Vec<Option<(String, DocLocation)>>> {
        let col = id.as_str();
        let rtxn = self.db.begin_read()?;
        let ord_rev_table = rtxn.open_table(DOC_ORD_TO_ID)?;
        let doc_table = rtxn.open_table(DOC_MAP)?;

        let mut out = Vec::with_capacity(doc_ords.len());
        for &doc_ord in doc_ords {
            let ok = doc_ord_key(col, doc_ord);
            let doc_id = match ord_rev_table.get(ok.as_slice())? {
                Some(v) => v.value().to_string(),
                None => {
                    out.push(None);
                    continue;
                }
            };
            let dk = doc_key(col, &doc_id);
            match doc_table.get(dk.as_str())? {
                Some(v) => {
                    let loc: DocLocation = serde_json::from_str(v.value())?;
                    out.push(Some((doc_id, loc)));
                }
                None => out.push(None),
            }
        }
        Ok(out)
    }

    /// Get the last applied WAL sequence number for a collection.
    pub fn get_applied_seq(&self, id: &CollectionId) -> Result<Option<u64>> {
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(APPLIED_SEQ)?;
        Ok(table.get(id.as_str())?.map(|v| v.value()))
    }

    /// Read doc_id and prev_location for every internal_id in a segment.
    ///
    /// Returns one entry per internal_id (index = internal_id).
    /// The entry is `None` when the slot is tombstoned or has no reverse-doc mapping.
    pub fn load_segment_live_entries(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        num_vectors: usize,
    ) -> Result<LiveEntries> {
        let col = id.as_str();
        let rtxn = self.db.begin_read()?;
        let tomb_table = rtxn.open_table(TOMBSTONES)?;
        let rev_table = rtxn.open_table(REVERSE_DOC)?;
        let doc_table = rtxn.open_table(DOC_MAP)?;

        let mut out = Vec::with_capacity(num_vectors);
        for i in 0..num_vectors {
            let internal_id = InternalId::from_u32(i as u32);
            let key = meta_key(col, seg_id, internal_id);
            if tomb_table
                .get(key.as_str())?
                .map(|v| v.value())
                .unwrap_or(false)
            {
                out.push(None);
                continue;
            }
            let doc_id = match rev_table.get(key.as_str())? {
                Some(v) => v.value().to_string(),
                None => {
                    out.push(None);
                    continue;
                }
            };
            let dk = doc_key(col, &doc_id);
            let prev_location: Option<DocLocation> = match doc_table.get(dk.as_str())? {
                Some(v) => Some(serde_json::from_str(v.value())?),
                None => None,
            };
            out.push(Some((doc_id, prev_location)));
        }
        Ok(out)
    }

    /// Atomically commit a compaction result:
    /// - Insert new segment metadata and update the manifest
    /// - For each merged doc: update DOC_MAP/REVERSE_DOC to the new location,
    ///   OR tombstone the new entry if the doc was deleted/moved during the merge
    /// - Remove old segment metadata and tombstone entries for removed segments
    ///
    /// Note: METADATA_VALUES / DOC_ID_TO_ORD / DOC_ORD_TO_ID are keyed by stable
    /// DocOrd and do not need to be updated during compaction.
    pub fn commit_compaction(
        &self,
        id: &CollectionId,
        new_seg_meta: &SegmentMeta,
        new_manifest: &Manifest,
        entries: &[CompactionEntry],
        removed_seg_ids: &[SegId],
    ) -> Result<()> {
        let col = id.as_str();
        let new_sk = seg_key(col, new_seg_meta.seg_id);
        let new_seg_json = serde_json::to_string(new_seg_meta)?;
        let manifest_json = serde_json::to_string(new_manifest)?;

        let removed_set: std::collections::HashSet<SegId> =
            removed_seg_ids.iter().copied().collect();

        let wtxn = self.db.begin_write()?;
        {
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            seg_table.insert(new_sk.as_str(), new_seg_json.as_str())?;

            let mut manifest_table = wtxn.open_table(MANIFESTS)?;
            manifest_table.insert(col, manifest_json.as_str())?;

            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let mut rev_table = wtxn.open_table(REVERSE_DOC)?;
            let mut tomb_table = wtxn.open_table(TOMBSTONES)?;

            for entry in entries {
                let new_mk = meta_key(col, new_seg_meta.seg_id, entry.new_internal_id);
                let dk = doc_key(col, &entry.doc_id);

                let current_loc: Option<DocLocation> = match doc_table.get(dk.as_str())? {
                    Some(v) => Some(serde_json::from_str(v.value())?),
                    None => None,
                };

                let should_tombstone = match &current_loc {
                    None => true,
                    Some(loc) if !removed_set.contains(&loc.seg_id) => true,
                    Some(loc) => Some(loc) != entry.prev_location.as_ref(),
                };

                if should_tombstone {
                    tomb_table.insert(new_mk.as_str(), true)?;
                } else {
                    let new_loc = DocLocation {
                        seg_id: new_seg_meta.seg_id,
                        internal_id: entry.new_internal_id,
                    };
                    let loc_json = serde_json::to_string(&new_loc)?;
                    doc_table.insert(dk.as_str(), loc_json.as_str())?;
                    rev_table.insert(new_mk.as_str(), entry.doc_id.as_str())?;
                }
            }

            // Remove old segment metadata.
            for &old_seg_id in removed_seg_ids {
                let old_sk = seg_key(col, old_seg_id);
                seg_table.remove(old_sk.as_str())?;
            }

            // Remove tombstone entries for removed segments.
            let prefix_base = format!("{}\0", col);
            let mut to_delete: Vec<String> = Vec::new();
            for entry in tomb_table.iter()? {
                let (k, _) = entry?;
                let key = k.value();
                if !key.starts_with(&prefix_base) {
                    continue;
                }
                let rest = &key[prefix_base.len()..];
                if let Some((seg_str, _)) = rest.split_once('\0')
                    && let Ok(n) = seg_str.parse::<u32>()
                    && removed_set.contains(&SegId::from_u32(n))
                {
                    to_delete.push(key.to_string());
                }
            }
            for k in &to_delete {
                tomb_table.remove(k.as_str())?;
            }
        }
        wtxn.commit()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Result of resolving a single candidate in [`Storage::resolve_candidates`].
pub struct ResolvedCandidate {
    pub doc_id: String,
    /// Present when `need_doc_ord` or `include_metadata` was true.
    pub doc_ord: Option<DocOrd>,
    pub metadata: Option<HashMap<String, MetadataValue>>,
}

/// A single document entry produced by a compaction merge.
pub struct CompactionEntry {
    pub doc_id: String,
    pub new_internal_id: InternalId,
    /// DOC_MAP value at the time `load_segment_live_entries` read this doc.
    pub prev_location: Option<DocLocation>,
}

/// Set of tombstoned `(seg_id, internal_id)` pairs for a collection.
#[cfg(feature = "testing")]
pub type TombstoneSet = std::collections::HashSet<(SegId, InternalId)>;
