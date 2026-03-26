use nebel::{
    Nebel,
    types::{CollectionId, Metric},
};

fn main() -> anyhow::Result<()> {
    let mut db = Nebel::open("data")?;

    let col = CollectionId::new("demo");
    db.create_collection(&col, 4, Metric::L2)?;

    db.upsert(&col, "a", &[1.0, 0.0, 0.0, 0.0], None)?;
    db.upsert(&col, "b", &[0.0, 1.0, 0.0, 0.0], None)?;
    db.upsert(&col, "c", &[0.0, 0.0, 1.0, 0.0], None)?;

    let hits = db.search(&col, &[1.0, 0.1, 0.0, 0.0], 2, false, false)?;
    for h in &hits {
        println!("doc_id={} score={:.4}", h.doc_id, h.score);
    }

    db.delete(&col, "a")?;
    println!("\nafter delete 'a':");
    let hits = db.search(&col, &[1.0, 0.1, 0.0, 0.0], 2, false, false)?;
    for h in &hits {
        println!("doc_id={} score={:.4}", h.doc_id, h.score);
    }

    Ok(())
}
