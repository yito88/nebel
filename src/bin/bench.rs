use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Result, bail};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use nebel::{
    Db,
    dataset::{load_vectors, write_raw_f32},
    eval::{hits_to_ids, percentile, recall_at_k},
    types::{CollectionId, CollectionSchema, Metric, SegmentParams},
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "bench", about = "Benchmark for nebel vector DB")]
struct Cli {
    /// Path to base vectors (SIFT .bvecs / .fvecs)
    #[arg(long)]
    base_file: String,

    /// Path to query vectors
    #[arg(long)]
    query_file: String,

    /// Logical dataset name for reporting
    #[arg(long, default_value = "sift")]
    dataset_name: String,

    /// Top-k
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Number of queries to benchmark (subset of query file)
    #[arg(long, default_value_t = 100)]
    num_queries: usize,

    /// Warmup queries before measurement
    #[arg(long, default_value_t = 10)]
    warmup_queries: usize,

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

    // --- optional file paths ---
    /// Save exact results to this path (JSON)
    #[arg(long)]
    save_exact_results: Option<String>,

    /// Load exact results from this path instead of recomputing
    #[arg(long)]
    load_exact_results: Option<String>,

    /// Save benchmark summary to this path (JSON)
    #[arg(long)]
    save_summary: Option<String>,

    /// Persist ingested DB to this directory and reuse it on subsequent runs.
    /// If omitted, a temporary directory is used and discarded after the run.
    #[arg(long)]
    data_dir: Option<String>,

    /// Number of concurrent query threads during the ANN benchmark phase.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Skip recall computation — exact results are neither loaded nor computed.
    /// Only throughput and latency are reported.
    #[arg(long, default_value_t = false)]
    no_recall: bool,
}

// ---------------------------------------------------------------------------
// Output structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExactResults {
    dataset_name: String,
    metric: Metric,
    k: usize,
    query_results: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkSummary {
    dataset_name: String,
    metric: Metric,
    k: usize,
    num_base_vectors: usize,
    num_queries: usize,
    warmup_queries: usize,

    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,

    #[serde(skip_serializing_if = "Option::is_none")]
    avg_recall_at_k: Option<f64>,

    concurrency: usize,
    throughput_qps: f64,

    params: SegmentParams,
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

    // 3. Select query subset
    let total_needed = cli.warmup_queries + cli.num_queries;
    let queries: Vec<&Vec<f32>> = if total_needed >= all_queries.len() {
        all_queries.iter().collect()
    } else {
        let mut indices: Vec<usize> = (0..all_queries.len()).collect();
        let mut rng = StdRng::seed_from_u64(cli.seed);
        indices.shuffle(&mut rng);
        indices.truncate(total_needed);
        indices.sort();
        indices.iter().map(|&i| &all_queries[i]).collect()
    };
    let warmup_queries = &queries[..cli.warmup_queries.min(queries.len())];
    let bench_queries = &queries[cli.warmup_queries.min(queries.len())..];
    let num_bench = bench_queries.len().min(cli.num_queries);
    let bench_queries = &bench_queries[..num_bench];

    println!(
        "  Using {} warmup + {} benchmark queries",
        warmup_queries.len(),
        bench_queries.len()
    );

    // 4. Open or create DB, ingest if needed
    let col_id = CollectionId::new(&cli.dataset_name);

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
        let n = col.ingest_file(&tmp_base)?;
        println!(
            "  Ingested {} vectors in {:.2}s",
            n,
            ingest_start.elapsed().as_secs_f64()
        );
    }

    // 5. Warmup
    println!("Running {} warmup queries...", warmup_queries.len());
    for q in warmup_queries {
        let _ = col.search(q, cli.k, false, false)?;
    }

    // 6. Exact results: load or compute (skipped when --no-recall)
    let exact_results: Option<ExactResults> = if cli.no_recall {
        None
    } else {
        let auto_exact = cli
            .data_dir
            .as_ref()
            .map(|d| PathBuf::from(d).join("exact_results.json"));
        let load_exact = cli
            .load_exact_results
            .as_deref()
            .map(PathBuf::from)
            .or(auto_exact.clone());
        let save_exact = cli
            .save_exact_results
            .as_deref()
            .map(PathBuf::from)
            .or(auto_exact);

        let results = if let Some(ref path) = load_exact.filter(|p| p.exists()) {
            println!("Loading exact results from {}...", path.display());
            let data = fs::read_to_string(path)?;
            serde_json::from_str(&data)?
        } else {
            println!(
                "Computing exact results for {} queries (brute-force)...",
                bench_queries.len()
            );
            let exact_start = Instant::now();
            let mut query_results = Vec::with_capacity(bench_queries.len());
            for (i, q) in bench_queries.iter().enumerate() {
                let hits = col.search_exact(q, cli.k)?;
                query_results.push(hits_to_ids(&hits));
                if (i + 1) % 10 == 0 || i + 1 == bench_queries.len() {
                    print!("\r  {}/{}", i + 1, bench_queries.len());
                }
            }
            println!(" done in {:.2}s", exact_start.elapsed().as_secs_f64());
            let r = ExactResults {
                dataset_name: cli.dataset_name.clone(),
                metric: cli.metric.clone(),
                k: cli.k,
                query_results,
            };
            if let Some(ref path) = save_exact {
                println!("Saving exact results to {}...", path.display());
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, serde_json::to_string_pretty(&r)?)?;
            }
            r
        };
        Some(results)
    };

    // 7. ANN benchmark
    println!(
        "Running ANN benchmark ({} queries, k={}, concurrency={})...",
        bench_queries.len(),
        cli.k,
        cli.concurrency,
    );

    let bench_start = Instant::now();
    let mut latencies_ms: Vec<f64>;
    let recalls: Vec<f64>;
    if cli.concurrency <= 1 {
        let mut lats = Vec::with_capacity(bench_queries.len());
        let mut recs = Vec::with_capacity(bench_queries.len());
        for (i, q) in bench_queries.iter().enumerate() {
            let start = Instant::now();
            let hits = col.search(q, cli.k, false, false)?;
            lats.push(start.elapsed().as_secs_f64() * 1000.0);
            if let Some(ref er) = exact_results {
                recs.push(recall_at_k(
                    &er.query_results[i],
                    &hits_to_ids(&hits),
                    cli.k,
                ));
            }
        }
        latencies_ms = lats;
        recalls = recs;
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cli.concurrency)
            .build()?;
        let (lats, opt_recs): (Vec<f64>, Vec<Option<f64>>) = pool.install(|| {
            (0..bench_queries.len())
                .into_par_iter()
                .map(|i| {
                    let start = Instant::now();
                    let hits = col.search(bench_queries[i], cli.k, false, false).unwrap();
                    let latency = start.elapsed().as_secs_f64() * 1000.0;
                    let recall = exact_results
                        .as_ref()
                        .map(|er| recall_at_k(&er.query_results[i], &hits_to_ids(&hits), cli.k));
                    (latency, recall)
                })
                .unzip()
        });
        latencies_ms = lats;
        recalls = opt_recs.into_iter().flatten().collect();
    };
    let wall_secs = bench_start.elapsed().as_secs_f64();
    let throughput_qps = bench_queries.len() as f64 / wall_secs;

    // 8. Compute stats
    latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
    let p50 = percentile(&latencies_ms, 50.0);
    let p95 = percentile(&latencies_ms, 95.0);
    let p99 = percentile(&latencies_ms, 99.0);
    let avg_recall: Option<f64> = if recalls.is_empty() {
        None
    } else {
        Some(recalls.iter().sum::<f64>() / recalls.len() as f64)
    };

    let summary = BenchmarkSummary {
        dataset_name: cli.dataset_name.clone(),
        metric: cli.metric.clone(),
        k: cli.k,
        num_base_vectors: base_vectors.len(),
        num_queries: bench_queries.len(),
        warmup_queries: warmup_queries.len(),
        avg_latency_ms: avg_latency,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        p99_latency_ms: p99,
        avg_recall_at_k: avg_recall,
        concurrency: cli.concurrency,
        throughput_qps,
        params,
    };

    // 9. Print summary
    println!("\n===== Benchmark Summary =====");
    println!("Dataset:          {}", summary.dataset_name);
    println!("Metric:           {:?}", summary.metric);
    println!("k:                {}", summary.k);
    println!("Base vectors:     {}", summary.num_base_vectors);
    println!("Queries:          {}", summary.num_queries);
    println!("Warmup queries:   {}", summary.warmup_queries);
    println!("Concurrency:      {}", summary.concurrency);
    println!("---");
    println!("Throughput:       {:.1} QPS", summary.throughput_qps);
    println!("Avg latency:      {:.3} ms", summary.avg_latency_ms);
    println!("P50 latency:      {:.3} ms", summary.p50_latency_ms);
    println!("P95 latency:      {:.3} ms", summary.p95_latency_ms);
    println!("P99 latency:      {:.3} ms", summary.p99_latency_ms);
    println!("---");
    if let Some(avg_recall) = summary.avg_recall_at_k {
        println!("Avg recall@{}:    {:.4}", summary.k, avg_recall);
        println!("---");
    }
    println!(
        "HNSW params:      m={} ef_c={} ef_s={} cap={}",
        summary.params.m,
        summary.params.ef_construction,
        summary.params.ef_search,
        summary.params.segment_capacity
    );

    // 10. Save summary if requested
    if let Some(ref path) = cli.save_summary {
        println!("\nSaving summary to {}...", path);
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&summary)?;
        fs::write(path, json)?;
    }

    Ok(())
}
