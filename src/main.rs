use nebel::{
    Db,
    types::{CollectionId, CollectionSchema, Metric},
};

fn main() -> anyhow::Result<()> {
    let db = Db::open("data")?;

    let col_id = CollectionId::new("demo");
    let col = db.create_collection(CollectionSchema::new(col_id, 4, Metric::L2))?;

    col.upsert("a", &[1.0, 0.0, 0.0, 0.0], None)?;
    col.upsert("b", &[0.0, 1.0, 0.0, 0.0], None)?;
    col.upsert("c", &[0.0, 0.0, 1.0, 0.0], None)?;

    let hits = col.search(&[1.0, 0.1, 0.0, 0.0], 2, false, false)?;
    for h in &hits {
        println!("doc_id={} score={:.4}", h.doc_id, h.score);
    }

    col.delete("a")?;
    println!("\nafter delete 'a':");
    let hits = col.search(&[1.0, 0.1, 0.0, 0.0], 2, false, false)?;
    for h in &hits {
        println!("doc_id={} score={:.4}", h.doc_id, h.score);
    }

    Ok(())
}
