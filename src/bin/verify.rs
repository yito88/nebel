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
    types::{CollectionId, CollectionSchema, Metric, SegmentParams, WriteToken},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    Recall,
    Mutation,
}

#[derive(Parser)]
#[command(
    name = "verify",
    about = "Verify search correctness against groundtruth"
)]
struct Cli {
    /// Path to base vectors (.fvecs / .bvecs)
    #[arg(long)]
    base_file: String,

    /// Path to query vectors (.fvecs / .bvecs) — required for --mode recall
    #[arg(long)]
    query_file: Option<String>,

    /// Path to groundtruth file (.ivecs) — required for --mode recall
    #[arg(long)]
    ground_truth_file: Option<String>,

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

    /// Number of concurrent query/mutation threads.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Verification mode
    #[arg(long, default_value = "recall")]
    mode: Mode,

    /// Number of mutation checks (mutation mode only)
    #[arg(long, default_value_t = 100)]
    num_mutations: usize,
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

    // 2. Open or create DB, ingest if needed
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
                    compaction_params: Default::default(),
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
            compaction_params: Default::default(),
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

    match cli.mode {
        Mode::Recall => run_recall_mode(&cli, &col, dim, &base_vectors)?,
        Mode::Mutation => {
            if cli.data_dir.is_some() {
                println!(
                    "WARNING: mutation mode modifies the collection; \
                     do not reuse this data_dir for recall verification."
                );
            }
            run_mutation_mode(&cli, &col, &base_vectors)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Recall mode
// ---------------------------------------------------------------------------

fn run_recall_mode(
    cli: &Cli,
    col: &nebel::CollectionHandle,
    dim: usize,
    _base_vectors: &[Vec<f32>],
) -> Result<()> {
    let query_file = cli
        .query_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--query-file is required for --mode recall"))?;
    let gt_file = cli
        .ground_truth_file
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--ground-truth-file is required for --mode recall"))?;

    println!("Loading query vectors from {}...", query_file);
    let (qdim, all_queries) = load_vectors(query_file)?;
    if qdim != dim {
        bail!("dimension mismatch: base={} query={}", dim, qdim);
    }
    println!("  {} query vectors available", all_queries.len());

    println!("Loading groundtruth from {}...", gt_file);
    let all_groundtruth = load_groundtruth(gt_file, cli.k)?;
    if all_groundtruth.len() != all_queries.len() {
        bail!(
            "groundtruth count ({}) != query count ({})",
            all_groundtruth.len(),
            all_queries.len()
        );
    }

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

    let avg_exact_recall = exact_recalls.iter().sum::<f64>() / num_queries as f64;
    let avg_ann_recall = ann_recalls.iter().sum::<f64>() / num_queries as f64;

    println!("\n===== Verification Results =====");
    println!("Queries:                    {}", num_queries);
    println!("k:                          {}", cli.k);
    println!("Metric:                     {:?}", cli.metric);
    println!(
        "HNSW params:                m={} ef_c={} ef_s={} cap={}",
        cli.m, cli.ef_construction, cli.ef_search, cli.segment_capacity
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

// ---------------------------------------------------------------------------
// Mutation mode
// ---------------------------------------------------------------------------

enum MutationOp {
    Insert { doc_id: String, vector: Vec<f32> },
    Update { doc_id: String, new_vector: Vec<f32> },
    Delete { doc_id: String, original_vector: Vec<f32> },
}

fn run_mutation_mode(
    cli: &Cli,
    col: &nebel::CollectionHandle,
    base_vectors: &[Vec<f32>],
) -> Result<()> {
    let n = base_vectors.len();
    let num_mutations = cli.num_mutations.min(n);
    let n_inserts = num_mutations / 3;
    let n_updates = num_mutations / 3;
    let n_deletes = num_mutations - n_inserts - n_updates;

    // Build a shuffled pool of base indices, partitioned into non-overlapping slices.
    let mut pool: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(cli.seed);
    pool.shuffle(&mut rng);

    let update_indices = pool[..n_updates].to_vec();
    let delete_indices = pool[n_updates..n_updates + n_deletes].to_vec();
    let insert_indices = pool[n_updates + n_deletes..n_updates + n_deletes + n_inserts].to_vec();

    // Phase A: build mutation list
    let mut ops: Vec<MutationOp> = Vec::with_capacity(num_mutations);

    for (i, &base_idx) in insert_indices.iter().enumerate() {
        ops.push(MutationOp::Insert {
            doc_id: format!("new_doc_{}", i),
            vector: base_vectors[base_idx].clone(),
        });
    }
    for &base_idx in &update_indices {
        let replacement_idx = (base_idx + 1) % n;
        ops.push(MutationOp::Update {
            doc_id: format!("doc_{}", base_idx),
            new_vector: base_vectors[replacement_idx].clone(),
        });
    }
    for &base_idx in &delete_indices {
        ops.push(MutationOp::Delete {
            doc_id: format!("doc_{}", base_idx),
            original_vector: base_vectors[base_idx].clone(),
        });
    }

    println!(
        "Submitting {} mutations (inserts={}, updates={}, deletes={}, concurrency={})...",
        num_mutations, n_inserts, n_updates, n_deletes, cli.concurrency,
    );

    // Phase B: submit all mutations concurrently, collect tokens
    let tokens: Vec<WriteToken> = if cli.concurrency <= 1 {
        ops.iter()
            .map(|op| match op {
                MutationOp::Insert { doc_id, vector } => col.upsert(doc_id, vector, None),
                MutationOp::Update { doc_id, new_vector } => col.upsert(doc_id, new_vector, None),
                MutationOp::Delete { doc_id, .. } => col.delete(doc_id),
            })
            .collect::<Result<_>>()?
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cli.concurrency)
            .build()?;
        pool.install(|| {
            ops.par_iter()
                .map(|op| match op {
                    MutationOp::Insert { doc_id, vector } => {
                        col.upsert(doc_id, vector, None).unwrap()
                    }
                    MutationOp::Update { doc_id, new_vector } => {
                        col.upsert(doc_id, new_vector, None).unwrap()
                    }
                    MutationOp::Delete { doc_id, .. } => col.delete(doc_id).unwrap(),
                })
                .collect()
        })
    };

    // Phase C: wait once on the highest token — covers all prior writes
    let max_token = tokens.into_iter().max().expect("at least one mutation");
    col.wait_visible(max_token)?;

    // Phase D: check all mutations
    struct MutationResult {
        kind: &'static str,
        doc_id: String,
        passed: bool,
        detail: String,
    }

    let k_delete = 10.min(n);
    let mut results: Vec<MutationResult> = Vec::with_capacity(num_mutations);

    for op in &ops {
        match op {
            MutationOp::Insert { doc_id, vector } => {
                // k=2: handles tie with the existing doc that shares the same base vector
                let hits = col.search_exact(vector, 2)?;
                let found = hits.iter().any(|h| h.doc_id == *doc_id);
                let detail = if found {
                    String::new()
                } else {
                    format!(
                        "expected '{}' in top-2, got {:?}",
                        doc_id,
                        hits.iter().map(|h| h.doc_id.as_str()).collect::<Vec<_>>()
                    )
                };
                results.push(MutationResult { kind: "insert", doc_id: doc_id.clone(), passed: found, detail });
            }
            MutationOp::Update { doc_id, new_vector } => {
                let hits = col.search_exact(new_vector, 2)?;
                let found = hits.iter().any(|h| h.doc_id == *doc_id);
                let detail = if found {
                    String::new()
                } else {
                    format!(
                        "expected '{}' in top-2 after update, got {:?}",
                        doc_id,
                        hits.iter().map(|h| h.doc_id.as_str()).collect::<Vec<_>>()
                    )
                };
                results.push(MutationResult { kind: "update", doc_id: doc_id.clone(), passed: found, detail });
            }
            MutationOp::Delete { doc_id, original_vector } => {
                let hits = col.search_exact(original_vector, k_delete)?;
                let absent = hits.iter().all(|h| h.doc_id != *doc_id);
                let detail = if absent {
                    String::new()
                } else {
                    format!("'{}' still appears in top-{} results after deletion", doc_id, k_delete)
                };
                results.push(MutationResult { kind: "delete", doc_id: doc_id.clone(), passed: absent, detail });
            }
        }
    }

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = total - passed;

    println!("\n===== Mutation Verification Results =====");
    println!("Total mutations: {}  (inserts={}, updates={}, deletes={})", total, n_inserts, n_updates, n_deletes);
    println!("Passed: {}", passed);
    println!("Failed: {}", failed);

    if failed > 0 {
        println!("\nFailed checks:");
        for r in results.iter().filter(|r| !r.passed) {
            println!("  [{}] doc_id='{}' -- {}", r.kind, r.doc_id, r.detail);
        }
        bail!("mutation verification failed: {}/{} checks failed", failed, total);
    }

    println!("All mutation checks passed.");
    Ok(())
}
