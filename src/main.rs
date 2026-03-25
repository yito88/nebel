use nebel::{Nebel, types::Metric};

fn main() -> anyhow::Result<()> {
    let mut db = Nebel::open("data")?;

    db.create_collection("demo", 4, Metric::L2)?;

    db.upsert("demo", "a", &[1.0, 0.0, 0.0, 0.0], None)?;
    db.upsert("demo", "b", &[0.0, 1.0, 0.0, 0.0], None)?;
    db.upsert("demo", "c", &[0.0, 0.0, 1.0, 0.0], None)?;

    let hits = db.search("demo", &[1.0, 0.1, 0.0, 0.0], 2, false, false)?;
    for h in &hits {
        println!("doc_id={} score={:.4}", h.doc_id, h.score);
    }

    db.delete("demo", "a")?;
    println!("\nafter delete 'a':");
    let hits = db.search("demo", &[1.0, 0.1, 0.0, 0.0], 2, false, false)?;
    for h in &hits {
        println!("doc_id={} score={:.4}", h.doc_id, h.score);
    }

    Ok(())
}
