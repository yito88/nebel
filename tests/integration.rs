use std::io::Write;

use nebel::{
    Db,
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
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    let meta = serde_json::json!({"label": "test", "value": 42});
    let t = col.upsert("doc", &[1.0, 0.0], Some(meta.clone())).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].metadata.as_ref().unwrap()["label"], "test");
}

#[test]
fn update_metadata_only() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    col.upsert("doc", &[1.0, 0.0], None).unwrap();
    let t = col
        .update_metadata("doc", serde_json::json!({"v": 99}))
        .unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search(&[1.0, 0.0], 1, true, false).unwrap();
    assert_eq!(hits[0].metadata.as_ref().unwrap()["v"], 99);
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
        let col = db
            .create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
            .unwrap();

        let meta = serde_json::json!({"tag": "hello"});
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
    assert_eq!(hits[0].metadata.as_ref().unwrap()["tag"], "hello");

    let vec = hits[0].vector.as_ref().unwrap();
    assert!((vec[0] - 1.0).abs() < 1e-6);
}

#[test]
fn multi_segment_search() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();

    col.add_writable_segment().unwrap();

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
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 2, Metric::L2))
        .unwrap();

    col.upsert("x", &[1.0, 0.0], None).unwrap();

    col.add_writable_segment().unwrap();
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
    let exact = col.search_exact(&query, 3).unwrap();
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

    let hits = col.search_exact(&[1.0, 0.0], 2).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].doc_id, "y");
}

#[test]
fn search_exact_multi_segment() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("col"), 3, Metric::L2))
        .unwrap();

    col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    col.upsert("b", &[0.0, 1.0, 0.0], None).unwrap();

    col.add_writable_segment().unwrap();

    col.upsert("c", &[0.0, 0.0, 1.0], None).unwrap();
    let t = col.upsert("d", &[0.9, 0.1, 0.0], None).unwrap();
    col.wait_visible(t).unwrap();

    let hits = col.search_exact(&[1.0, 0.0, 0.0], 2).unwrap();
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
        let col = db
            .create_collection(CollectionSchema::new(id.clone(), 3, Metric::L2))
            .unwrap();
        col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();

        col.add_writable_segment().unwrap();
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
    let hits = col2.search_exact(&[1.0, 0.0, 0.0], 3).unwrap();
    assert_eq!(hits.len(), 3);
    let ids: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
    assert!(ids.contains(&"a") && ids.contains(&"b") && ids.contains(&"c"));
}

#[test]
fn wal_segment_rotation() {
    let (db, _dir) = make_db();
    let col = db
        .create_collection(CollectionSchema::new(col("test"), 3, Metric::Cosine))
        .unwrap();

    // Set a tiny rotation threshold so a handful of records triggers it.
    col.set_wal_rotation_bytes(512);
    assert_eq!(col.wal_segment_count(), 1);

    // Write enough records to exceed the threshold several times.
    let mut last = col.upsert("a", &[1.0, 0.0, 0.0], None).unwrap();
    for i in 0..20u32 {
        last = col
            .upsert(&format!("doc{i}"), &[i as f32, 0.0, 0.0], None)
            .unwrap();
    }
    col.wait_visible(last).unwrap();

    // Multiple segments should now exist.
    assert!(col.wal_segment_count() > 1);
}
