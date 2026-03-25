use anyhow::Result;
use redb::{Database, TableDefinition};

use crate::types::{CollectionSchema, DocLocation, SegmentMeta};

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

// key helpers

pub fn seg_key(collection: &str, seg_id: u32) -> String {
    format!("{}\0{}", collection, seg_id)
}

pub fn doc_key(collection: &str, doc_id: &str) -> String {
    format!("{}\0{}", collection, doc_id)
}

pub fn meta_key(collection: &str, seg_id: u32, internal_id: u32) -> String {
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
        wtxn.commit()?;
        Ok(Self { db })
    }

    /// Persist a collection schema.
    pub fn put_collection(&self, schema: &CollectionSchema) -> Result<()> {
        let json = serde_json::to_string(schema)?;
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(COLLECTIONS)?;
            table.insert(schema.name.as_str(), json.as_str())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Load a collection schema by name.
    pub fn get_collection(&self, name: &str) -> Result<Option<CollectionSchema>> {
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(COLLECTIONS)?;
        match table.get(name)? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Persist segment metadata.
    pub fn put_segment(&self, collection: &str, meta: &SegmentMeta) -> Result<()> {
        let key = seg_key(collection, meta.seg_id);
        let json = serde_json::to_string(meta)?;
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(SEGMENTS)?;
            table.insert(key.as_str(), json.as_str())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Load segment metadata.
    pub fn get_segment(&self, collection: &str, seg_id: u32) -> Result<Option<SegmentMeta>> {
        let key = seg_key(collection, seg_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(SEGMENTS)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Write the `doc_id -> (seg_id, internal_id)` mapping.
    pub fn put_doc_location(
        &self,
        collection: &str,
        doc_id: &str,
        loc: &DocLocation,
    ) -> Result<()> {
        let key = doc_key(collection, doc_id);
        let json = serde_json::to_string(loc)?;
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(DOC_MAP)?;
            table.insert(key.as_str(), json.as_str())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Look up the storage location for a document.
    pub fn get_doc_location(&self, collection: &str, doc_id: &str) -> Result<Option<DocLocation>> {
        let key = doc_key(collection, doc_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(DOC_MAP)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Remove the doc_map entry for a document (called on delete).
    pub fn remove_doc_location(&self, collection: &str, doc_id: &str) -> Result<()> {
        let key = doc_key(collection, doc_id);
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(DOC_MAP)?;
            table.remove(key.as_str())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Mark a vector slot as deleted.
    pub fn set_tombstone(&self, collection: &str, seg_id: u32, internal_id: u32) -> Result<()> {
        let key = meta_key(collection, seg_id, internal_id);
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(TOMBSTONES)?;
            table.insert(key.as_str(), true)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Return `true` if the slot is tombstoned.
    pub fn is_tombstoned(&self, collection: &str, seg_id: u32, internal_id: u32) -> Result<bool> {
        let key = meta_key(collection, seg_id, internal_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(TOMBSTONES)?;
        Ok(table.get(key.as_str())?.map(|v| v.value()).unwrap_or(false))
    }

    /// Store (or overwrite) user-defined metadata for a vector slot.
    pub fn put_metadata(
        &self,
        collection: &str,
        seg_id: u32,
        internal_id: u32,
        meta: &serde_json::Value,
    ) -> Result<()> {
        let key = meta_key(collection, seg_id, internal_id);
        let json = serde_json::to_string(meta)?;
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(METADATA)?;
            table.insert(key.as_str(), json.as_str())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Retrieve user-defined metadata for a vector slot.
    pub fn get_metadata(
        &self,
        collection: &str,
        seg_id: u32,
        internal_id: u32,
    ) -> Result<Option<serde_json::Value>> {
        let key = meta_key(collection, seg_id, internal_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(METADATA)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(serde_json::from_str(v.value())?)),
            None => Ok(None),
        }
    }

    /// Store the reverse mapping `(seg_id, internal_id) -> doc_id`.
    pub fn put_reverse_doc(
        &self,
        collection: &str,
        seg_id: u32,
        internal_id: u32,
        doc_id: &str,
    ) -> Result<()> {
        let key = meta_key(collection, seg_id, internal_id);
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(REVERSE_DOC)?;
            table.insert(key.as_str(), doc_id)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Look up the `doc_id` for a given `(seg_id, internal_id)`.
    pub fn get_reverse_doc(
        &self,
        collection: &str,
        seg_id: u32,
        internal_id: u32,
    ) -> Result<Option<String>> {
        let key = meta_key(collection, seg_id, internal_id);
        let rtxn = self.db.begin_read()?;
        let table = rtxn.open_table(REVERSE_DOC)?;
        match table.get(key.as_str())? {
            Some(v) => Ok(Some(v.value().to_string())),
            None => Ok(None),
        }
    }
}
