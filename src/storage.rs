use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};

use crate::types::{
    CollectionId, CollectionSchema, DocLocation, Manifest, SegId, SegmentMeta, VectorEntry,
};

// table definitions

/// collection_name -> CollectionSchema (JSON)
const COLLECTIONS: TableDefinition<&str, &str> = TableDefinition::new("collections");

/// "{collection}\0{seg_id}" -> SegmentMeta (JSON)
const SEGMENTS: TableDefinition<&str, &str> = TableDefinition::new("segments");

/// "{collection}\0{doc_id}" -> DocLocation (JSON)
const DOC_MAP: TableDefinition<&str, &str> = TableDefinition::new("doc_map");

/// "{collection}\0{seg_id}\0{internal_id}" -> bool (true = deleted)
const TOMBSTONES: TableDefinition<&str, bool> = TableDefinition::new("tombstones");

/// "{collection}\0{seg_id}\0{internal_id}" -> metadata JSON string
const METADATA: TableDefinition<&str, &str> = TableDefinition::new("metadata");

/// "{collection}\0{seg_id}\0{internal_id}" -> doc_id string
const REVERSE_DOC: TableDefinition<&str, &str> = TableDefinition::new("reverse_doc");

/// collection_name -> Manifest (JSON)
const MANIFESTS: TableDefinition<&str, &str> = TableDefinition::new("manifests");

/// collection_name -> last applied WAL seq (u64)
const APPLIED_SEQ: TableDefinition<&str, u64> = TableDefinition::new("applied_seq");

// key helpers

fn seg_key(collection: &str, seg_id: SegId) -> String {
    format!("{}\0{}", collection, seg_id)
}

fn doc_key(collection: &str, doc_id: &str) -> String {
    format!("{}\0{}", collection, doc_id)
}

fn meta_key(collection: &str, seg_id: SegId, internal_id: u32) -> String {
    format!("{}\0{}\0{}", collection, seg_id, internal_id)
}

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
        wtxn.open_table(METADATA)?;
        wtxn.open_table(REVERSE_DOC)?;
        wtxn.open_table(MANIFESTS)?;
        wtxn.open_table(APPLIED_SEQ)?;
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
    /// All three writes happen in a single redb transaction.
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

    /// Return `true` if the slot is tombstoned.
    pub fn is_tombstoned(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        internal_id: u32,
    ) -> Result<bool> {
        let key = meta_key(id.as_str(), seg_id, internal_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(TOMBSTONES)?;
        Ok(table.get(key.as_str())?.map(|v| v.value()).unwrap_or(false))
    }

    /// Atomically apply an upsert WAL record: tombstone the old location (if any),
    /// write new segment metadata, doc/reverse/vector-metadata entries, and advance applied_seq.
    /// The old doc location is read inside the transaction, making the full operation atomic.
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
        let prepared: Vec<_> = entries
            .iter()
            .map(|e| {
                let dk = doc_key(col, e.doc_id);
                let loc = DocLocation {
                    seg_id: seg_meta.seg_id,
                    internal_id: e.internal_id,
                };
                let loc_json = serde_json::to_string(&loc)?;
                let mk = meta_key(col, seg_meta.seg_id, e.internal_id);
                let meta_json = match e.metadata {
                    Some(m) => Some(serde_json::to_string(m)?),
                    None => None,
                };
                Ok((dk, loc_json, mk, e.doc_id, meta_json))
            })
            .collect::<Result<Vec<_>>>()?;

        let wtxn = self.db.begin_write()?;
        {
            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let old_loc: Option<DocLocation> = match doc_table.get(old_dk.as_str())? {
                Some(v) => Some(serde_json::from_str(v.value())?),
                None => None,
            };
            if let Some(loc) = &old_loc {
                let tk = meta_key(col, loc.seg_id, loc.internal_id);
                let mut tomb_table = wtxn.open_table(TOMBSTONES)?;
                tomb_table.insert(tk.as_str(), true)?;
            }
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            seg_table.insert(sk.as_str(), seg_json.as_str())?;
            let mut rev_table = wtxn.open_table(REVERSE_DOC)?;
            let mut meta_table = wtxn.open_table(METADATA)?;
            for (dk, loc_json, mk, entry_doc_id, meta_json) in &prepared {
                doc_table.insert(dk.as_str(), loc_json.as_str())?;
                rev_table.insert(mk.as_str(), *entry_doc_id)?;
                if let Some(mj) = meta_json {
                    meta_table.insert(mk.as_str(), mj.as_str())?;
                }
            }
            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Atomically apply a delete WAL record: tombstone and remove doc location (if it exists),
    /// and advance applied_seq.
    /// The doc location is read inside the transaction, making the full operation atomic.
    pub fn apply_delete(
        &self,
        id: &CollectionId,
        doc_id: &str,
        seq: u64,
    ) -> Result<()> {
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
            }
            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Atomically apply an update-metadata WAL record: write the new metadata (if the doc exists)
    /// and advance applied_seq.
    /// The doc location is read inside the transaction, making the full operation atomic.
    pub fn apply_update_metadata(
        &self,
        id: &CollectionId,
        doc_id: &str,
        meta: &serde_json::Value,
        seq: u64,
    ) -> Result<()> {
        let col = id.as_str();
        let dk = doc_key(col, doc_id);
        let wtxn = self.db.begin_write()?;
        {
            let doc_table = wtxn.open_table(DOC_MAP)?;
            let old_loc: Option<DocLocation> = match doc_table.get(dk.as_str())? {
                Some(v) => Some(serde_json::from_str(v.value())?),
                None => None,
            };
            if let Some(loc) = &old_loc {
                let mk = meta_key(col, loc.seg_id, loc.internal_id);
                let json = serde_json::to_string(meta)?;
                let mut meta_table = wtxn.open_table(METADATA)?;
                meta_table.insert(mk.as_str(), json.as_str())?;
            }
            let mut seq_table = wtxn.open_table(APPLIED_SEQ)?;
            seq_table.insert(col, seq)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Retrieve user-defined metadata for a vector slot.
    pub fn get_metadata(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        internal_id: u32,
    ) -> Result<Option<serde_json::Value>> {
        let key = meta_key(id.as_str(), seg_id, internal_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(METADATA)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Write segment metadata, doc locations, reverse-doc mappings, and
    /// optional per-vector metadata in a single redb transaction.
    pub fn write_vector_entries(
        &self,
        id: &CollectionId,
        seg_meta: &SegmentMeta,
        entries: &[VectorEntry<'_>],
    ) -> Result<()> {
        let col = id.as_str();
        let seg_key = seg_key(col, seg_meta.seg_id);
        let seg_json = serde_json::to_string(seg_meta)?;

        // Pre-serialise doc locations so we don't do it inside the txn.
        let prepared: Vec<_> = entries
            .iter()
            .map(|e| {
                let dk = doc_key(col, e.doc_id);
                let loc = DocLocation {
                    seg_id: seg_meta.seg_id,
                    internal_id: e.internal_id,
                };
                let loc_json = serde_json::to_string(&loc)?;
                let mk = meta_key(col, seg_meta.seg_id, e.internal_id);
                let meta_json = match e.metadata {
                    Some(m) => Some(serde_json::to_string(m)?),
                    None => None,
                };
                Ok((dk, loc_json, mk, e.doc_id, meta_json))
            })
            .collect::<Result<Vec<_>>>()?;

        let wtxn = self.db.begin_write()?;
        {
            let mut seg_table = wtxn.open_table(SEGMENTS)?;
            seg_table.insert(seg_key.as_str(), seg_json.as_str())?;

            let mut doc_table = wtxn.open_table(DOC_MAP)?;
            let mut rev_table = wtxn.open_table(REVERSE_DOC)?;
            let mut meta_table = wtxn.open_table(METADATA)?;

            for (dk, loc_json, mk, doc_id, meta_json) in &prepared {
                doc_table.insert(dk.as_str(), loc_json.as_str())?;
                rev_table.insert(mk.as_str(), *doc_id)?;
                if let Some(mj) = meta_json {
                    meta_table.insert(mk.as_str(), mj.as_str())?;
                }
            }
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Get the last applied WAL sequence number for a collection.
    pub fn get_applied_seq(&self, id: &CollectionId) -> Result<Option<u64>> {
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(APPLIED_SEQ)?;
        Ok(table.get(id.as_str())?.map(|v| v.value()))
    }

    /// Look up the `doc_id` for a given `(seg_id, internal_id)`.
    pub fn get_reverse_doc(
        &self,
        id: &CollectionId,
        seg_id: SegId,
        internal_id: u32,
    ) -> Result<Option<String>> {
        let key = meta_key(id.as_str(), seg_id, internal_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(REVERSE_DOC)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(v.value().to_string())),
            None => Ok(None),
        }
    }
}
