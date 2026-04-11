use std::{collections::HashMap, io::Write, thread, time::Duration};

use nebel::{
    Db,
    metadata::{FieldSchema, FieldType, MetadataSchema, MetadataValue},
    types::{CollectionId, CollectionSchema, Metric},
};
use tempfile::tempdir;

fn make_db() -> (Db, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let db = Db::open(dir.path()).unwrap();
    (db, dir)
}

fn col(name: &str) -> CollectionId {
    CollectionId::new(name)
}

#[test]
fn create_and_search() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    let t = col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
    col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.01, 0.0], 1, false, false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "a");
}

#[test]
fn tombstone_after_delete() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    col.upsert("x", &[1.0, 0.0], None).unwrap();
    col.upsert("y", &[0.0, 1.0], None).unwrap();
    let t = col.delete("x").unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 2, false, false).unwrap();
    assert!(
        !hits.iter().any(|h| h.doc_id == "x"),
        "deleted doc should not appear"
    );
}

#[test]
fn upsert_replaces_vector() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    col.upsert("a", &[1.0, 0.0], None).unwrap();
    let t = col.upsert("a", &[0.0, 1.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[0.0, 1.0], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "a");
    assert!(hits[0].score < 0.01, "updated vector should be very close");
}

#[test]
fn metadata_roundtrip() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![
            FieldSchema {
                id: 0,
                name: "label".into(),
                ty: FieldType::String,
                filterable: false,
            },
            FieldSchema {
                id: 1,
                name: "value".into(),
                ty: FieldType::Int64,
                filterable: false,
            },
        ],
    });
    let col = db.create_collection(schema).unwrap();

    let meta = HashMap::from([
        ("label".to_string(), MetadataValue::String("test".into())),
        ("value".to_string(), MetadataValue::Int64(42)),
    ]);
    let t = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    let meta = hits[0].metadata.as_ref().unwrap();
    assert!(matches!(meta["label"], MetadataValue::String(ref s) if s == "test"));
}

#[test]
fn update_metadata_only() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "v".into(),
            ty: FieldType::Int64,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    col.upsert("doc", &[1.0, 0.0], None).unwrap();
    let t = col
        .update_metadata(
            "doc",
            HashMap::from([("v".to_string(), MetadataValue::Int64(99))]),
        )
        .unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    let meta = hits[0].metadata.as_ref().unwrap();
    assert!(matches!(meta["v"], MetadataValue::Int64(99)));
}

#[test]
fn include_vector_in_search() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    let t = col.upsert("v", &[1.0, 2.0, 3.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 2.0, 3.0], 1, false, true).unwrap();
    let vec = hits[0].vector.as_ref().unwrap();
    assert!((vec[0] - 1.0).abs() < 1e-6);
    assert!((vec[1] - 2.0).abs() < 1e-6);
    assert!((vec[2] - 3.0).abs() < 1e-6);
}

#[test]
fn ingest_binary_file() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 4, Metric::L2))
        .unwrap();

    let tmp = tempdir().unwrap();
    let path = tmp.path().join("vecs.bin");
    let mut f = std::fs::File::create(&path).unwrap();
    for vec in [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ] {
        for v in vec {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    drop(f);

    let count = col.ingest_file(&path).unwrap();
    assert_eq!(count, 3);

    let hits = col.search(&[1.0, 0.0, 0.0, 0.0], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "doc_0");
}

#[test]
fn load_collection_restores_data() {
    let dir = tempdir().unwrap();
    let id = col("col");

    {
        let db = Db::open(dir.path()).unwrap();
        let mut schema = CollectionSchema::new(id.clone(), 3, Metric::L2);
        schema.metadata_schema = Some(MetadataSchema {
            fields: vec![FieldSchema {
                id: 0,
                name: "tag".into(),
                ty: FieldType::String,
                filterable: false,
            }],
        });
        let col = db.create_collection(schema).unwrap();

        let meta = HashMap::from([("tag".to_string(), MetadataValue::String("hello".into()))]);
        col.upsert("a", &[1.0, 0.0, 0.0], Some(meta)).unwrap();
        col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
        let t = col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
        col.wait_visible(t).unwrap();
        col.delete("c").unwrap();
    }

    let db = Db::open(dir.path()).unwrap();
    let col = db.collection(id.as_str()).unwrap();

    let hits = col.search(&[1.0, 0.01, 0.0], 3, true, true).unwrap();

    assert_eq!(hits[0].doc_id, "a");
    assert!(
        !hits.iter().any(|h| h.doc_id == "c"),
        "deleted doc must not reappear"
    );
    assert!(
        matches!(hits[0].metadata.as_ref().unwrap()["tag"], MetadataValue::String(ref s) if s == "hello")
    );

    let vec = hits[0].vector.as_ref().unwrap();
    assert!((vec[0] - 1.0).abs() < 1e-6);
}

#[test]
fn multi_segment_search() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 3, Metric::L2);
    schema.segment_params.segment_capacity = 2;
    let col = db.create_collection(schema).unwrap();

    col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
    // "c" triggers auto-seal of seg0 (capacity=2) before inserting.
    col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
    let t = col.upsert("d", &[0.9, 0.1, 0.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0, 0.0], 2, false, false).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "a");
    assert_eq!(hits[1].doc_id, "d");
}

#[test]
fn multi_segment_tombstone() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.segment_params.segment_capacity = 1;
    let col = db.create_collection(schema).unwrap();

    col.upsert("x", &[1.0, 0.0], None).unwrap();
    // "y" triggers auto-seal of seg0 (capacity=1) before inserting.
    col.upsert("y", &[0.9, 0.1], None).unwrap();
    let t = col.delete("x").unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 2, false, false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "y");
}

#[test]
fn search_exact_matches_brute_force() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
    col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
    let t = col.upsert("d", &[0.9, 0.1, 0.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let query = [1.0, 0.05, 0.0];
    let exact = col.search_exact(&query, 3, false).unwrap();
    let approx = col.search(&query, 3, false, false).unwrap();

    assert_eq!(exact.len(), 3);
    assert_eq!(approx.len(), 3);
    for (e, a) in exact.iter().zip(approx.iter()) {
        assert_eq!(e.doc_id, a.doc_id);
        assert!(
            (e.score - a.score).abs() < 1e-3,
            "score mismatch: exact={} approx={}",
            e.score,
            a.score
        );
    }
}

#[test]
fn search_exact_respects_tombstones() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    col.upsert("x", &[1.0, 0.0], None).unwrap();
    col.upsert("y", &[0.0, 1.0], None).unwrap();
    let t = col.delete("x").unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search_exact(&[1.0, 0.0], 2, false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "y");
}

#[test]
fn search_exact_multi_segment() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 3, Metric::L2);
    schema.segment_params.segment_capacity = 2;
    let col = db.create_collection(schema).unwrap();

    col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
    // "c" triggers auto-seal of seg0 (capacity=2) before inserting.
    col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
    let t = col.upsert("d", &[0.9, 0.1, 0.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search_exact(&[1.0, 0.0, 0.0], 2, false).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "a");
    assert_eq!(hits[1].doc_id, "d");
}

#[test]
fn load_collection_multi_segment() {
    let dir = tempdir().unwrap();
    let id = col("col");

    {
        let db = Db::open(dir.path()).unwrap();
        let mut schema = CollectionSchema::new(id.clone(), 3, Metric::L2);
        schema.segment_params.segment_capacity = 1;
        let col = db.create_collection(schema).unwrap();
        col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
        // "b" triggers auto-seal of seg0 (capacity=1) before inserting.
        let t = col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
        col.wait_visible(t).unwrap();
    }

    let db = Db::open(dir.path()).unwrap();
    let col = db.collection(id.as_str()).unwrap();

    let hits = col.search(&[1.0, 0.0, 0.0], 2, false, false).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "a");
    assert_eq!(hits[1].doc_id, "b");
}

#[test]
fn recovery_from_wal_after_crash() {
    let dir = tempdir().unwrap();
    let schema = CollectionSchema::new(col("test"), 3, Metric::Cosine);

    // Write records then drop without wait_visible — simulates crash mid-apply.
    {
        let db = Db::open(dir.path()).unwrap();
        let col = db.create_collection(schema).unwrap();
        col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
        col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();
        col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
        // drop without wait_visible
    }

    // Reopen — recovery applies WAL records synchronously in load_collection_inner.
    let db2 = Db::open(dir.path()).unwrap();
    let col2 = db2.collection("test").unwrap();
    let hits = col2.search_exact(&[1.0, 0.0, 0.0], 3, false).unwrap();
    assert_eq!(hits.len(), 3);
    let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(ids.contains(&"a") && ids.contains(&"b") && ids.contains(&"c"));
}

#[test]
fn wal_segment_rotation() {
    let (db, _dir) = make_db();
    // Set a tiny rotation threshold so a handful of records triggers it.
    let mut schema = CollectionSchema::new(col("test"), 3, Metric::Cosine);
    schema.wal_segment_bytes = 512;
    let col = db.create_collection(schema).unwrap();
    assert_eq!(col.wal_segment_count(), 1);

    // Write enough records to exceed the threshold several times.
    let mut last = col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    for i in 0..20u32 {
        last = col
            .upsert(&format!("doc{i}"), &[i as f32, 0.0, 0.0], None)
            .unwrap();
    }
    col.wait_visible(last).unwrap();

    // After all records are applied, closed segments are pruned; only the active segment remains.
    assert_eq!(col.wal_segment_count(), 1);
}

// ---------------------------------------------------------------------------
// Compaction tests
// ---------------------------------------------------------------------------

/// Build a schema with a small segment capacity and aggressive L0 threshold
/// so compaction fires after inserting a handful of vectors.
fn make_compaction_schema(name: &str) -> CollectionSchema {
    let mut s = CollectionSchema::new(col(name), 3, Metric::L2);
    s.segment_params.segment_capacity = 5;
    s.compaction_params.num_levels = 2;
    s.compaction_params.level_count_multiplier = 1;
    s
}

/// Wait long enough for the background compaction thread to finish.
fn wait_for_compaction() {
    thread::sleep(Duration::from_millis(2000));
}

/// After two L0 segments are sealed, the compaction worker should merge them
/// into one L1 segment. All vectors must remain searchable.
#[test]
fn compaction_merges_l0_segments() {
    let (db, _dir) = make_db();
    let col = db.create_collection(make_compaction_schema("c")).unwrap();

    // Insert 11 vectors. The 6th insert auto-seals seg0 (capacity=5); the 11th
    // auto-seals seg1, leaving two sealed L0 segments and triggering compaction.
    let mut last = col.upsert("d0", &[0.0, 0.0, 0.0], None).unwrap();
    for i in 1..11u32 {
        last = col
            .upsert(&format!("d{i}"), &[i as f32, 0.0, 0.0], None)
            .unwrap();
    }
    col.wait_visible(last).unwrap();
    wait_for_compaction();

    // All 11 vectors must still be findable after the L0→L1 merge.
    for i in 0..11u32 {
        let hits = col.search_exact(&[i as f32, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(hits.len(), 1, "d{i} not found after compaction");
        assert_eq!(hits[0].doc_id, format!("d{i}"));
    }
}

/// Tombstones committed before the compaction snapshot is taken must be
/// honoured: the compacted segment should not contain the deleted vectors.
#[test]
fn compaction_removes_tombstones() {
    let (db, _dir) = make_db();
    let col = db.create_collection(make_compaction_schema("c")).unwrap();

    // Insert 10 vectors (seg0 auto-seals after d4), then delete d0–d4.
    let mut last = col.upsert("d0", &[0.0, 0.0, 0.0], None).unwrap();
    for i in 1..10u32 {
        last = col
            .upsert(&format!("d{i}"), &[i as f32, 0.0, 0.0], None)
            .unwrap();
    }
    col.wait_visible(last).unwrap();
    for i in 0..5u32 {
        last = col.delete(&format!("d{i}")).unwrap();
    }
    col.wait_visible(last).unwrap();

    // Insert d10 — this auto-seals seg1 (now at capacity), leaving two sealed
    // L0 segments and triggering the compaction worker. Tombstones for d0–d4
    // are already committed, so the merge will skip them.
    last = col.upsert("d10", &[10.0, 0.0, 0.0], None).unwrap();
    col.wait_visible(last).unwrap();
    wait_for_compaction();

    for i in 0..5u32 {
        let hits = col.search_exact(&[i as f32, 0.0, 0.0], 5, false).unwrap();
        assert!(
            !hits.iter().any(|h| h.doc_id == format!("d{i}")),
            "d{i} should have been removed by compaction"
        );
    }
    for i in 5..10u32 {
        let hits = col.search_exact(&[i as f32, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(
            hits[0].doc_id,
            format!("d{i}"),
            "d{i} should survive compaction"
        );
    }
}

/// Compacted-away segment directories must be deleted from disk after the commit.
#[test]
fn compaction_deletes_old_segment_dirs() {
    let (db, dir) = make_db();
    let col = db.create_collection(make_compaction_schema("c")).unwrap();

    // Insert 11 vectors: seg_000 auto-seals after d4, seg_001 auto-seals after
    // d9 when d10 arrives, leaving two sealed L0 segments that trigger compaction.
    // The compaction worker allocates seg_003 as the merged output
    // (seg_002 is the writable segment holding d10).
    let mut last = col.upsert("d0", &[0.0, 0.0, 0.0], None).unwrap();
    for i in 1..11u32 {
        last = col
            .upsert(&format!("d{i}"), &[i as f32, 0.0, 0.0], None)
            .unwrap();
    }
    col.wait_visible(last).unwrap();
    wait_for_compaction();

    let base = dir.path().join("c");
    assert!(
        !base.join("seg_000").exists(),
        "seg_000 should be deleted after compaction"
    );
    assert!(
        !base.join("seg_001").exists(),
        "seg_001 should be deleted after compaction"
    );
    assert!(
        base.join("seg_003").exists(),
        "merged seg_003 should exist after compaction"
    );
}

/// Metadata stored alongside vectors must be intact after compaction.
#[test]
fn compaction_preserves_metadata() {
    let (db, _dir) = make_db();
    let mut schema = make_compaction_schema("c");
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "n".into(),
            ty: FieldType::Int64,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    // Insert 11 vectors with metadata. The 11th auto-seals seg1, triggering compaction.
    let mut last = col
        .upsert(
            "d0",
            &[0.0, 0.0, 0.0],
            Some(HashMap::from([("n".to_string(), MetadataValue::Int64(0))])),
        )
        .unwrap();
    for i in 1..11u32 {
        last = col
            .upsert(
                &format!("d{i}"),
                &[i as f32, 0.0, 0.0],
                Some(HashMap::from([(
                    "n".to_string(),
                    MetadataValue::Int64(i as i64),
                )])),
            )
            .unwrap();
    }
    col.wait_visible(last).unwrap();
    wait_for_compaction();

    for i in 0..10u32 {
        let hits = col.search(&[i as f32, 0.0, 0.0], 1, true, false).unwrap();
        assert_eq!(hits[0].doc_id, format!("d{i}"));
        assert!(
            matches!(hits[0].metadata.as_ref().unwrap()["n"], MetadataValue::Int64(v) if v == i as i64),
            "metadata for d{i} corrupted after compaction"
        );
    }
}

// ---------------------------------------------------------------------------
// upsert_batch tests
// ---------------------------------------------------------------------------

#[test]
fn upsert_batch_basic() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    let token = col
        .upsert_batch(&[
            ("a", [1.0f32, 0.0, 0.0].as_slice(), None),
            ("b", [0.0, 1.0, 0.0].as_slice(), None),
            ("c", [0.0, 0.0, 1.0].as_slice(), None),
        ])
        .unwrap();
    col.wait_visible(token).unwrap();

    let hits = col.search(&[1.0, 0.01, 0.0], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "a");
    let hits = col.search(&[0.0, 1.0, 0.01], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "b");
    let hits = col.search(&[0.0, 0.01, 1.0], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "c");
}

#[test]
fn upsert_batch_empty_is_noop() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    let token = col.upsert_batch(&[]).unwrap();
    col.wait_visible(token).unwrap();
}

#[test]
fn upsert_batch_dimension_mismatch() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    let result = col.upsert_batch(&[
        ("a", [1.0f32, 0.0, 0.0].as_slice(), None),
        ("b", [0.0, 1.0].as_slice(), None), // wrong dim
    ]);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("index 1"), "error should mention index: {msg}");
}

#[test]
fn upsert_batch_with_metadata() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "label".into(),
            ty: FieldType::String,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    let x_meta = HashMap::from([("label".to_string(), MetadataValue::String("x".into()))]);
    let token = col
        .upsert_batch(&[
            ("x", [1.0f32, 0.0].as_slice(), Some(x_meta)),
            ("y", [0.0, 1.0].as_slice(), None),
        ])
        .unwrap();
    col.wait_visible(token).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].doc_id, "x");
    assert!(
        matches!(hits[0].metadata.as_ref().unwrap()["label"], MetadataValue::String(ref s) if s == "x")
    );
}

// ---------------------------------------------------------------------------
// Metadata: schema validation
// ---------------------------------------------------------------------------

/// Helper: build a schema with one field of each type.
fn all_types_schema(name: &str) -> CollectionSchema {
    let mut schema = CollectionSchema::new(CollectionId::new(name), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![
            FieldSchema {
                id: 0,
                name: "s".into(),
                ty: FieldType::String,
                filterable: false,
            },
            FieldSchema {
                id: 1,
                name: "i".into(),
                ty: FieldType::Int64,
                filterable: false,
            },
            FieldSchema {
                id: 2,
                name: "f".into(),
                ty: FieldType::Float64,
                filterable: false,
            },
            FieldSchema {
                id: 3,
                name: "b".into(),
                ty: FieldType::Bool,
                filterable: false,
            },
            FieldSchema {
                id: 4,
                name: "t".into(),
                ty: FieldType::Timestamp,
                filterable: false,
            },
        ],
    });
    schema
}

#[test]
fn metadata_correct_types_accepted() {
    let (db, _dir) = make_db();
    let col = db.create_collection(all_types_schema("col")).unwrap();

    let meta = HashMap::from([
        ("s".to_string(), MetadataValue::String("hello".into())),
        ("i".to_string(), MetadataValue::Int64(-42)),
        ("f".to_string(), MetadataValue::Float64(2.5)),
        ("b".to_string(), MetadataValue::Bool(true)),
        (
            "t".to_string(),
            MetadataValue::Timestamp(1_700_000_000_000_000),
        ),
    ]);
    let t = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    let m = hits[0].metadata.as_ref().unwrap();
    assert!(matches!(m["s"], MetadataValue::String(ref v) if v == "hello"));
    assert!(matches!(m["i"], MetadataValue::Int64(-42)));
    assert!(matches!(m["f"], MetadataValue::Float64(v) if (v - 2.5).abs() < 1e-10));
    assert!(matches!(m["b"], MetadataValue::Bool(true)));
    assert!(matches!(
        m["t"],
        MetadataValue::Timestamp(1_700_000_000_000_000)
    ));
}

#[test]
fn metadata_wrong_type_rejected() {
    let (db, _dir) = make_db();
    let col = db.create_collection(all_types_schema("col")).unwrap();

    // Pass an Int64 where String is expected.
    let meta = HashMap::from([("s".to_string(), MetadataValue::Int64(99))]);
    let err = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("field 's'"),
        "error should name the field: {msg}"
    );
}

#[test]
fn metadata_unknown_field_rejected() {
    let (db, _dir) = make_db();
    let col = db.create_collection(all_types_schema("col")).unwrap();

    let meta = HashMap::from([("nonexistent".to_string(), MetadataValue::String("x".into()))]);
    let err = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("nonexistent"),
        "error should name the field: {msg}"
    );
}

#[test]
fn metadata_no_schema_rejects_any_metadata() {
    let (db, _dir) = make_db();
    // Collection created with no metadata_schema.
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    let meta = HashMap::from([("x".to_string(), MetadataValue::Bool(false))]);
    assert!(col.upsert("doc", &[1.0, 0.0], Some(meta)).is_err());
}

// ---------------------------------------------------------------------------
// Metadata: persistence across reopen
// ---------------------------------------------------------------------------

#[test]
fn metadata_persists_across_reopen() {
    let dir = tempdir().unwrap();
    let id = CollectionId::new("col");

    {
        let db = Db::open(dir.path()).unwrap();
        let mut schema = CollectionSchema::new(id.clone(), 2, Metric::L2);
        schema.metadata_schema = Some(MetadataSchema {
            fields: vec![
                FieldSchema {
                    id: 0,
                    name: "tag".into(),
                    ty: FieldType::String,
                    filterable: false,
                },
                FieldSchema {
                    id: 1,
                    name: "count".into(),
                    ty: FieldType::Int64,
                    filterable: false,
                },
            ],
        });
        let col = db.create_collection(schema).unwrap();
        let meta = HashMap::from([
            ("tag".to_string(), MetadataValue::String("persist".into())),
            ("count".to_string(), MetadataValue::Int64(7)),
        ]);
        let t = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap();
        col.wait_visible(t).unwrap();
    }

    // Reopen and verify both fields survive.
    let db = Db::open(dir.path()).unwrap();
    let col = db.collection(id.as_str()).unwrap();
    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    let m = hits[0].metadata.as_ref().unwrap();
    assert!(matches!(m["tag"],   MetadataValue::String(ref s) if s == "persist"));
    assert!(matches!(m["count"], MetadataValue::Int64(7)));
}

// ---------------------------------------------------------------------------
// Metadata: overwrite — stale values not readable after re-upsert
// ---------------------------------------------------------------------------

#[test]
fn metadata_overwrite_replaces_value() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "v".into(),
            ty: FieldType::Int64,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    let t = col
        .upsert(
            "doc",
            &[1.0, 0.0],
            Some(HashMap::from([("v".to_string(), MetadataValue::Int64(1))])),
        )
        .unwrap();
    col.wait_visible(t).unwrap();

    // Re-upsert with a new value.
    let t = col
        .upsert(
            "doc",
            &[1.0, 0.0],
            Some(HashMap::from([("v".to_string(), MetadataValue::Int64(2))])),
        )
        .unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    // Only the latest value is visible.
    assert!(matches!(
        hits[0].metadata.as_ref().unwrap()["v"],
        MetadataValue::Int64(2)
    ));
}

// ---------------------------------------------------------------------------
// Metadata: delete removes metadata
// ---------------------------------------------------------------------------

#[test]
fn metadata_deleted_doc_not_returned() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "x".into(),
            ty: FieldType::String,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    let meta = HashMap::from([("x".to_string(), MetadataValue::String("alive".into()))]);
    col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap();
    let t = col.delete("doc").unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    assert!(hits.is_empty(), "deleted doc must not appear in results");
}

// ---------------------------------------------------------------------------
// Metadata: edge cases
// ---------------------------------------------------------------------------

#[test]
fn metadata_partial_fields_allowed() {
    // A document may supply only a subset of the schema's fields.
    let (db, _dir) = make_db();
    let col = db.create_collection(all_types_schema("col")).unwrap();

    // Only supply the "s" field, leave the rest absent.
    let meta = HashMap::from([("s".to_string(), MetadataValue::String("only-s".into()))]);
    let t = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    let m = hits[0].metadata.as_ref().unwrap();
    assert!(matches!(m["s"], MetadataValue::String(ref v) if v == "only-s"));
    // Absent fields must not appear.
    assert!(!m.contains_key("i"));
    assert!(!m.contains_key("f"));
}

#[test]
fn metadata_bool_boundaries() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "flag".into(),
            ty: FieldType::Bool,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    for (doc, val) in [("t", true), ("f", false)] {
        let meta = HashMap::from([("flag".to_string(), MetadataValue::Bool(val))]);
        col.upsert(doc, &[1.0, 0.0], Some(meta)).unwrap();
    }
    let t = col.upsert("z", &[0.0, 1.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 2, true, false).unwrap();
    for hit in &hits {
        if hit.doc_id == "t" {
            assert!(matches!(
                hit.metadata.as_ref().unwrap()["flag"],
                MetadataValue::Bool(true)
            ));
        } else if hit.doc_id == "f" {
            assert!(matches!(
                hit.metadata.as_ref().unwrap()["flag"],
                MetadataValue::Bool(false)
            ));
        }
    }
}

#[test]
fn metadata_int64_boundaries() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "n".into(),
            ty: FieldType::Int64,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    for (doc, val) in [("min", i64::MIN), ("max", i64::MAX), ("zero", 0i64)] {
        let meta = HashMap::from([("n".to_string(), MetadataValue::Int64(val))]);
        col.upsert(doc, &[1.0, 0.0], Some(meta)).unwrap();
    }
    let t = col.upsert("z", &[0.0, 1.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 3, true, false).unwrap();
    for hit in &hits {
        let expected = match hit.doc_id.as_str() {
            "min" => Some(i64::MIN),
            "max" => Some(i64::MAX),
            "zero" => Some(0),
            _ => None,
        };
        if let Some(v) = expected {
            assert!(
                matches!(hit.metadata.as_ref().unwrap()["n"], MetadataValue::Int64(got) if got == v)
            );
        }
    }
}

#[test]
fn metadata_string_boundaries() {
    let (db, _dir) = make_db();
    let mut schema = CollectionSchema::new(col("col"), 2, Metric::L2);
    schema.metadata_schema = Some(MetadataSchema {
        fields: vec![FieldSchema {
            id: 0,
            name: "s".into(),
            ty: FieldType::String,
            filterable: false,
        }],
    });
    let col = db.create_collection(schema).unwrap();

    let cases = [
        ("empty", ""),
        ("ascii", "hello world"),
        ("unicode", "日本語テスト"),
        ("nullish", "null\x00byte"),
    ];
    for (doc, val) in &cases {
        let meta = HashMap::from([("s".to_string(), MetadataValue::String((*val).into()))]);
        col.upsert(doc, &[1.0, 0.0], Some(meta)).unwrap();
    }
    let t = col.upsert("z", &[0.0, 1.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 5, true, false).unwrap();
    for hit in &hits {
        if let Some((_, expected)) = cases.iter().find(|(id, _)| *id == hit.doc_id) {
            assert!(
                matches!(hit.metadata.as_ref().unwrap()["s"], MetadataValue::String(ref s) if s == expected),
                "mismatch for doc '{}'",
                hit.doc_id
            );
        }
    }
}

#[test]
fn metadata_float_nan_rejected() {
    let (db, _dir) = make_db();
    let col = db.create_collection(all_types_schema("col")).unwrap();

    let nan_meta = HashMap::from([("f".to_string(), MetadataValue::Float64(f64::NAN))]);
    assert!(col.upsert("doc", &[1.0, 0.0], Some(nan_meta)).is_err());
}

#[test]
fn metadata_float_inf_rejected() {
    let (db, _dir) = make_db();
    let col = db.create_collection(all_types_schema("col")).unwrap();

    for val in [f64::INFINITY, f64::NEG_INFINITY] {
        let meta = HashMap::from([("f".to_string(), MetadataValue::Float64(val))]);
        let err = col.upsert("doc", &[1.0, 0.0], Some(meta)).unwrap_err();
        assert!(
            err.to_string().contains("finite"),
            "expected 'finite' in error: {err}"
        );
    }
}
