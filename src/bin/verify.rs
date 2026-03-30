use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Result, bail};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use rayon::prelude::*;

use nebel::{
    Db,
    dataset::{load_groundtruth, load_vectors, write_raw_f32},
    eval::{hits_to_ids, recall_at_k},
    types::{CollectionId, CollectionSchema, Metric, SegmentParams},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "verify",
    about = "Verify search correctness against groundtruth"
)]
struct Cli {
    /// Path to base vectors (.fvecs / .bvecs)
    #[arg(long)]
    base_file: String,

    /// Path to query vectors (.fvecs / .bvecs)
    #[arg(long)]
    query_file: String,

    /// Path to groundtruth file (.ivecs)
    #[arg(long)]
    ground_truth_file: String,

    /// Top-k
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Number of queries to verify
    #[arg(long, default_value_t = 100)]
    num_queries: usize,

    /// Distance metric
    #[arg(long, default_value = "l2")]
    metric: Metric,

    /// Random seed for deterministic query subset selection
    #[arg(long, default_value_t = 42)]
    seed: u64,

    // --- HNSW / segment params ---
    #[arg(long, default_value_t = 16)]
    m: usize,

    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    #[arg(long, default_value_t = 50)]
    ef_search: usize,

    #[arg(long, default_value_t = 100_000)]
    segment_capacity: usize,

    /// Persist ingested DB to this directory and reuse it on subsequent runs.
    /// If omitted, a temporary directory is used and discarded after the run.
    #[arg(long)]
    data_dir: Option<String>,

    /// Number of concurrent query threads.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    let params = SegmentParams {
        m: cli.m,
        ef_construction: cli.ef_construction,
        ef_search: cli.ef_search,
        segment_capacity: cli.segment_capacity,
    };

    // 1. Load base vectors
    println!("Loading base vectors from {}...", cli.base_file);
    let (dim, base_vectors) = load_vectors(&cli.base_file)?;
    println!("  {} base vectors, dimension={}", base_vectors.len(), dim);

    // 2. Load query vectors
    println!("Loading query vectors from {}...", cli.query_file);
    let (qdim, all_queries) = load_vectors(&cli.query_file)?;
    if qdim != dim {
        bail!("dimension mismatch: base={} query={}", dim, qdim);
    }
    println!("  {} query vectors available", all_queries.len());

    // 3. Load groundtruth
    println!("Loading groundtruth from {}...", cli.ground_truth_file);
    let all_groundtruth = load_groundtruth(&cli.ground_truth_file, cli.k)?;
    if all_groundtruth.len() != all_queries.len() {
        bail!(
            "groundtruth count ({}) != query count ({})",
            all_groundtruth.len(),
            all_queries.len()
        );
    }

    // 4. Select query subset
    let indices: Vec<usize> = if cli.num_queries >= all_queries.len() {
        (0..all_queries.len()).collect()
    } else {
        let mut idx: Vec<usize> = (0..all_queries.len()).collect();
        let mut rng = StdRng::seed_from_u64(cli.seed);
        idx.shuffle(&mut rng);
        idx.truncate(cli.num_queries);
        idx.sort();
        idx
    };
    let num_queries = indices.len();
    println!("  Using {} queries for verification", num_queries);

    // 5. Open or create DB, ingest if needed
    let col_id = CollectionId::new("verify");

    enum DbDir {
        Temp(tempfile::TempDir),
        Persistent(PathBuf),
    }
    impl DbDir {
        fn path(&self) -> &Path {
            match self {
                DbDir::Temp(t) => t.path(),
                DbDir::Persistent(p) => p.as_path(),
            }
        }
    }

    let (db_dir, col, reused) = if let Some(ref dir) = cli.data_dir {
        let path = PathBuf::from(dir);
        let db = Db::open(&path)?;
        let (col, reused) = match db.collection(col_id.as_str()) {
            Ok(h) => (h, true),
            Err(_) => {
                let schema = CollectionSchema {
                    name: col_id.clone(),
                    dimension: dim,
                    metric: cli.metric.clone(),
                    segment_params: params.clone(),
                };
                (db.create_collection(schema)?, false)
            }
        };
        (DbDir::Persistent(path), col, reused)
    } else {
        let tmp = tempfile::tempdir()?;
        println!("DB temp dir: {}", tmp.path().display());
        let db = Db::open(tmp.path())?;
        let schema = CollectionSchema {
            name: col_id.clone(),
            dimension: dim,
            metric: cli.metric.clone(),
            segment_params: params.clone(),
        };
        let col = db.create_collection(schema)?;
        (DbDir::Temp(tmp), col, false)
    };

    if reused {
        println!("Reusing existing DB at {}", db_dir.path().display());
    } else {
        println!("Ingesting base vectors...");
        let ingest_start = Instant::now();
        let tmp_base = db_dir.path().join("base_vectors.raw");
        write_raw_f32(&tmp_base, &base_vectors)?;
        let num_ingested = col.ingest_file(&tmp_base)?;
        println!(
            "  Ingested {} vectors in {:.2}s",
            num_ingested,
            ingest_start.elapsed().as_secs_f64()
        );
    }

    // 6. Run search_exact and search(ANN) for each query
    println!(
        "Running verification ({} queries, k={}, concurrency={})...",
        num_queries, cli.k, cli.concurrency,
    );

    struct QueryResult {
        query_idx: usize,
        exact_recall: f64,
        ann_recall: f64,
    }

    let results: Vec<QueryResult> = if cli.concurrency <= 1 {
        let mut out = Vec::with_capacity(num_queries);
        for (qi, &query_idx) in indices.iter().enumerate() {
            let query = &all_queries[query_idx];
            let gt = &all_groundtruth[query_idx];
            let exact_r = recall_at_k(gt, &hits_to_ids(&col.search_exact(query, cli.k)?), cli.k);
            let ann_r = recall_at_k(
                gt,
                &hits_to_ids(&col.search(query, cli.k, false, false)?),
                cli.k,
            );
            out.push(QueryResult {
                query_idx,
                exact_recall: exact_r,
                ann_recall: ann_r,
            });
            if (qi + 1) % 10 == 0 || qi + 1 == num_queries {
                print!("\r  {}/{}", qi + 1, num_queries);
            }
        }
        out
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cli.concurrency)
            .build()?;
        let mut out: Vec<QueryResult> = pool.install(|| {
            indices
                .par_iter()
                .map(|&query_idx| {
                    let query = &all_queries[query_idx];
                    let gt = &all_groundtruth[query_idx];
                    let exact_r = recall_at_k(
                        gt,
                        &hits_to_ids(&col.search_exact(query, cli.k).unwrap()),
                        cli.k,
                    );
                    let ann_r = recall_at_k(
                        gt,
                        &hits_to_ids(&col.search(query, cli.k, false, false).unwrap()),
                        cli.k,
                    );
                    QueryResult {
                        query_idx,
                        exact_recall: exact_r,
                        ann_recall: ann_r,
                    }
                })
                .collect()
        });
        out.sort_by_key(|r| r.query_idx);
        out
    };
    println!();

    let exact_recalls: Vec<f64> = results.iter().map(|r| r.exact_recall).collect();
    let ann_recalls: Vec<f64> = results.iter().map(|r| r.ann_recall).collect();
    let exact_low_recall: Vec<(usize, f64)> = results
        .iter()
        .filter(|r| r.exact_recall < 1.0)
        .map(|r| (r.query_idx, r.exact_recall))
        .collect();
    drop(results);

    // 7. Compute and print results
    let avg_exact_recall = exact_recalls.iter().sum::<f64>() / num_queries as f64;
    let avg_ann_recall = ann_recalls.iter().sum::<f64>() / num_queries as f64;

    println!("\n===== Verification Results =====");
    println!("Queries:                    {}", num_queries);
    println!("k:                          {}", cli.k);
    println!("Metric:                     {:?}", cli.metric);
    println!(
        "HNSW params:                m={} ef_c={} ef_s={} cap={}",
        params.m, params.ef_construction, params.ef_search, params.segment_capacity
    );
    println!("---");
    println!(
        "search_exact avg recall@{}:  {:.4}",
        cli.k, avg_exact_recall
    );
    println!("search (ANN) avg recall@{}:  {:.4}", cli.k, avg_ann_recall);
    println!(
        "exact queries with recall < 1.0: {}",
        exact_low_recall.len()
    );

    if !exact_low_recall.is_empty() {
        println!("\nWARNING: search_exact missed groundtruth on these queries:");
        for (idx, r) in &exact_low_recall {
            println!("  query_idx={} recall={:.4}", idx, r);
        }
    }

    Ok(())
}
