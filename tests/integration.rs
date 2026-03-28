use std::io::Write;

use nebel::{
    Nebel,
    types::{CollectionId, CollectionSchema, Metric, SegmentParams},
};
use tempfile::tempdir;

// tempfile is only in dev-deps; pull it in below via Cargo.toml.

fn make_db() -> (Nebel, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let db = Nebel::open(dir.path()).unwrap();
    (db, dir)
}

fn col(name: &str) -> CollectionId {
    CollectionId::new(name)
}

#[test]
fn create_and_search() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
        .unwrap();

    db.upsert(&id, "a", &[1.0, 0.0, 0.0], None).unwrap();
    db.upsert(&id, "b", &[0.0, 1.0, 0.0], None).unwrap();
    db.upsert(&id, "c", &[0.0, 0.0, 1.0], None).unwrap();

    let hits = db.search(&id, &[1.0, 0.01, 0.0], 1, false, false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "a");
}

#[test]
fn tombstone_after_delete() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 2, Metric::L2))
        .unwrap();

    db.upsert(&id, "x", &[1.0, 0.0], None).unwrap();
    db.upsert(&id, "y", &[0.0, 1.0], None).unwrap();

    db.delete(&id, "x").unwrap();

    // Search near "x" — should return "y", not "x".
    let hits = db.search(&id, &[1.0, 0.0], 2, false, false).unwrap();
    assert!(
        !hits.iter().any(|h| h.doc_id == "x"),
        "deleted doc should not appear"
    );
}

#[test]
fn upsert_replaces_vector() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 2, Metric::L2))
        .unwrap();

    db.upsert(&id, "a", &[1.0, 0.0], None).unwrap();
    // Replace "a" with a new vector close to (0,1).
    db.upsert(&id, "a", &[0.0, 1.0], None).unwrap();

    let hits = db.search(&id, &[0.0, 1.0], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "a");
    assert!(hits[0].score < 0.01, "updated vector should be very close");
}

#[test]
fn metadata_roundtrip() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 2, Metric::L2))
        .unwrap();

    let meta = serde_json::json!({"label": "test", "value": 42});
    db.upsert(&id, "doc", &[1.0, 0.0], Some(meta.clone()))
        .unwrap();

    let hits = db.search(&id, &[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].metadata.as_ref().unwrap()["label"], "test");
}

#[test]
fn update_metadata_only() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 2, Metric::L2))
        .unwrap();

    db.upsert(&id, "doc", &[1.0, 0.0], None).unwrap();
    db.update_metadata(&id, "doc", serde_json::json!({"v": 99}))
        .unwrap();

    let hits = db.search(&id, &[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].metadata.as_ref().unwrap()["v"], 99);
}

#[test]
fn include_vector_in_search() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
        .unwrap();

    db.upsert(&id, "v", &[1.0, 2.0, 3.0], None).unwrap();

    let hits = db.search(&id, &[1.0, 2.0, 3.0], 1, false, true).unwrap();
    let vec = hits[0].vector.as_ref().unwrap();
    assert!((vec[0] - 1.0).abs() < 1e-6);
    assert!((vec[1] - 2.0).abs() < 1e-6);
    assert!((vec[2] - 3.0).abs() < 1e-6);
}

#[test]
fn ingest_binary_file() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 4, Metric::L2))
        .unwrap();

    // Write 3 vectors as raw f32 LE.
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

    let count = db.ingest_file(&id, &path).unwrap();
    assert_eq!(count, 3);

    let hits = db
        .search(&id, &[1.0, 0.0, 0.0, 0.0], 1, false, false)
        .unwrap();
    assert_eq!(hits[0].doc_id, "doc_0");
}

#[test]
fn load_collection_restores_data() {
    let dir = tempdir().unwrap();
    let id = col("col");

    // Create a collection and insert vectors in one Nebel instance.
    {
        let mut db = Nebel::open(dir.path()).unwrap();
        db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
            .unwrap();

        let meta = serde_json::json!({"tag": "hello"});
        db.upsert(&id, "a", &[1.0, 0.0, 0.0], Some(meta)).unwrap();
        db.upsert(&id, "b", &[0.0, 1.0, 0.0], None).unwrap();
        db.upsert(&id, "c", &[0.0, 0.0, 1.0], None).unwrap();
        // Delete one so we can verify tombstones survive reload.
        db.delete(&id, "c").unwrap();
    }

    // Re-open from disk and load the collection.
    let mut db = Nebel::open(dir.path()).unwrap();
    db.load_collection(&id).unwrap();

    // Search should return "a" nearest to the query, and "c" must stay deleted.
    let hits = db.search(&id, &[1.0, 0.01, 0.0], 3, true, true).unwrap();

    assert_eq!(hits[0].doc_id, "a");
    assert!(
        !hits.iter().any(|h| h.doc_id == "c"),
        "deleted doc must not reappear"
    );

    // Metadata should survive the reload.
    assert_eq!(hits[0].metadata.as_ref().unwrap()["tag"], "hello");

    // Vector should survive the reload.
    let vec = hits[0].vector.as_ref().unwrap();
    assert!((vec[0] - 1.0).abs() < 1e-6);
}

#[test]
fn multi_segment_search() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
        .unwrap();

    // Insert into segment 0.
    db.upsert(&id, "a", &[1.0, 0.0, 0.0], None).unwrap();
    db.upsert(&id, "b", &[0.0, 1.0, 0.0], None).unwrap();

    // Add a second writable segment.
    db.add_writable_segment(&id).unwrap();

    // Insert into segment 1.
    db.upsert(&id, "c", &[0.0, 0.0, 1.0], None).unwrap();
    db.upsert(&id, "d", &[0.9, 0.1, 0.0], None).unwrap();

    // Search should merge results across both segments.
    // Query near [1,0,0]: closest should be "a" then "d".
    let hits = db.search(&id, &[1.0, 0.0, 0.0], 2, false, false).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "a");
    assert_eq!(hits[1].doc_id, "d");
}

#[test]
fn multi_segment_tombstone() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 2, Metric::L2))
        .unwrap();

    db.upsert(&id, "x", &[1.0, 0.0], None).unwrap();

    db.add_writable_segment(&id).unwrap();
    db.upsert(&id, "y", &[0.9, 0.1], None).unwrap();

    // Delete "x" from segment 0 — should not appear in cross-segment search.
    db.delete(&id, "x").unwrap();

    let hits = db.search(&id, &[1.0, 0.0], 2, false, false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "y");
}

#[test]
fn search_exact_matches_brute_force() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
        .unwrap();

    db.upsert(&id, "a", &[1.0, 0.0, 0.0], None).unwrap();
    db.upsert(&id, "b", &[0.0, 1.0, 0.0], None).unwrap();
    db.upsert(&id, "c", &[0.0, 0.0, 1.0], None).unwrap();
    db.upsert(&id, "d", &[0.9, 0.1, 0.0], None).unwrap();

    let query = [1.0, 0.05, 0.0];
    let exact = db.search_exact(&id, &query, 3).unwrap();
    let approx = db.search(&id, &query, 3, false, false).unwrap();

    // Exact search should return the same top-k ordering as HNSW on this small dataset.
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
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 2, Metric::L2))
        .unwrap();

    db.upsert(&id, "x", &[1.0, 0.0], None).unwrap();
    db.upsert(&id, "y", &[0.0, 1.0], None).unwrap();
    db.delete(&id, "x").unwrap();

    let hits = db.search_exact(&id, &[1.0, 0.0], 2).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "y");
}

#[test]
fn search_exact_multi_segment() {
    let (mut db, _dir) = make_db();
    let id = col("col");
    db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
        .unwrap();

    db.upsert(&id, "a", &[1.0, 0.0, 0.0], None).unwrap();
    db.upsert(&id, "b", &[0.0, 1.0, 0.0], None).unwrap();

    db.add_writable_segment(&id).unwrap();

    db.upsert(&id, "c", &[0.0, 0.0, 1.0], None).unwrap();
    db.upsert(&id, "d", &[0.9, 0.1, 0.0], None).unwrap();

    let hits = db.search_exact(&id, &[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "a");
    assert_eq!(hits[1].doc_id, "d");
}

#[test]
fn load_collection_multi_segment() {
    let dir = tempdir().unwrap();
    let id = col("col");

    {
        let mut db = Nebel::open(dir.path()).unwrap();
        db.create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
            .unwrap();
        db.upsert(&id, "a", &[1.0, 0.0, 0.0], None).unwrap();

        db.add_writable_segment(&id).unwrap();
        db.upsert(&id, "b", &[0.0, 1.0, 0.0], None).unwrap();
    }

    // Re-open and load — should recover both segments.
    let mut db = Nebel::open(dir.path()).unwrap();
    db.load_collection(&id).unwrap();

    let hits = db.search(&id, &[1.0, 0.0, 0.0], 2, false, false).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "a");
    assert_eq!(hits[1].doc_id, "b");
}

#[test]
fn ingest_file_parallel_basic() {
    let dir = tempdir().unwrap();
    let mut db = Nebel::open(dir.path()).unwrap();
    let id = col("col");

    // Use a small segment capacity to force multiple segments.
    let schema = CollectionSchema {
        name: id.clone(),
        dimension: 4,
        metric: Metric::L2,
        segment_params: SegmentParams {
            segment_capacity: 3,
            ..SegmentParams::default()
        },
    };
    db.create_collection(schema).unwrap();

    // Write 7 vectors → should create 3 segments (3 + 3 + 1).
    let tmp = tempdir().unwrap();
    let path = tmp.path().join("vecs.bin");
    let mut f = std::fs::File::create(&path).unwrap();
    let vectors: Vec<[f32; 4]> = vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.9, 0.1, 0.0, 0.0],
    ];
    for vec in &vectors {
        for &v in vec {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    drop(f);

    let count = db.ingest_file_parallel(&id, &path).unwrap();
    assert_eq!(count, 7);

    // Verify search works across segments.
    let hits = db
        .search(&id, &[1.0, 0.0, 0.0, 0.0], 2, false, false)
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].doc_id, "doc_0");
    assert_eq!(hits[1].doc_id, "doc_6"); // [0.9, 0.1, 0.0, 0.0]

    // Verify exact search also works.
    let exact = db.search_exact(&id, &[1.0, 0.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(exact.len(), 2);
    assert_eq!(exact[0].doc_id, "doc_0");
    assert_eq!(exact[1].doc_id, "doc_6");
}
