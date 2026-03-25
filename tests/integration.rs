use std::io::Write;

use nebel::{Nebel, types::Metric};
use tempfile::tempdir;

// tempfile is only in dev-deps; pull it in below via Cargo.toml.

fn make_db() -> (Nebel, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let db = Nebel::open(dir.path()).unwrap();
    (db, dir)
}

#[test]
fn create_and_search() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 3, Metric::L2).unwrap();

    db.upsert("col", "a", &[1.0, 0.0, 0.0], None).unwrap();
    db.upsert("col", "b", &[0.0, 1.0, 0.0], None).unwrap();
    db.upsert("col", "c", &[0.0, 0.0, 1.0], None).unwrap();

    let hits = db
        .search("col", &[1.0, 0.01, 0.0], 1, false, false)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "a");
}

#[test]
fn tombstone_after_delete() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 2, Metric::L2).unwrap();

    db.upsert("col", "x", &[1.0, 0.0], None).unwrap();
    db.upsert("col", "y", &[0.0, 1.0], None).unwrap();

    db.delete("col", "x").unwrap();

    // Search near "x" — should return "y", not "x".
    let hits = db.search("col", &[1.0, 0.0], 2, false, false).unwrap();
    assert!(
        !hits.iter().any(|h| h.doc_id == "x"),
        "deleted doc should not appear"
    );
}

#[test]
fn upsert_replaces_vector() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 2, Metric::L2).unwrap();

    db.upsert("col", "a", &[1.0, 0.0], None).unwrap();
    // Replace "a" with a new vector close to (0,1).
    db.upsert("col", "a", &[0.0, 1.0], None).unwrap();

    let hits = db.search("col", &[0.0, 1.0], 1, false, false).unwrap();
    assert_eq!(hits[0].doc_id, "a");
    assert!(hits[0].score < 0.01, "updated vector should be very close");
}

#[test]
fn metadata_roundtrip() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 2, Metric::L2).unwrap();

    let meta = serde_json::json!({"label": "test", "value": 42});
    db.upsert("col", "doc", &[1.0, 0.0], Some(meta.clone()))
        .unwrap();

    let hits = db.search("col", &[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].metadata.as_ref().unwrap()["label"], "test");
}

#[test]
fn update_metadata_only() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 2, Metric::L2).unwrap();

    db.upsert("col", "doc", &[1.0, 0.0], None).unwrap();
    db.update_metadata("col", "doc", serde_json::json!({"v": 99}))
        .unwrap();

    let hits = db.search("col", &[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].metadata.as_ref().unwrap()["v"], 99);
}

#[test]
fn include_vector_in_search() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 3, Metric::L2).unwrap();

    db.upsert("col", "v", &[1.0, 2.0, 3.0], None).unwrap();

    let hits = db.search("col", &[1.0, 2.0, 3.0], 1, false, true).unwrap();
    let vec = hits[0].vector.as_ref().unwrap();
    assert!((vec[0] - 1.0).abs() < 1e-6);
    assert!((vec[1] - 2.0).abs() < 1e-6);
    assert!((vec[2] - 3.0).abs() < 1e-6);
}

#[test]
fn ingest_binary_file() {
    let (mut db, _dir) = make_db();
    db.create_collection("col", 4, Metric::L2).unwrap();

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

    let count = db.ingest_file("col", &path).unwrap();
    assert_eq!(count, 3);

    let hits = db
        .search("col", &[1.0, 0.0, 0.0, 0.0], 1, false, false)
        .unwrap();
    assert_eq!(hits[0].doc_id, "doc_0");
}
