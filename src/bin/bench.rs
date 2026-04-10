use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Result, bail};
use clap::Parser;
use rand::Rng;
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use nebel::{
    Db,
    dataset::{load_vectors, write_raw_f32},
    eval::{hits_to_ids, percentile, recall_at_k},
    types::{CollectionId, CollectionSchema, DEFAULT_WAL_SEGMENT_BYTES, Metric, SegmentParams},
};

// ---------------------------------------------------------------------------
// Mode enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
enum BenchMode {
    Search,
    Write,
    Mix,
}

#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum, Serialize, Deserialize)]
enum WritePattern {
    Append,
    Update,
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "bench", about = "Benchmark for nebel vector DB")]
struct Cli {
    /// Benchmark workload mode
    #[arg(long, default_value = "search")]
    mode: BenchMode,

    /// Path to base vectors (SIFT .bvecs / .fvecs)
    #[arg(long)]
    base_file: String,

    /// Path to query vectors (required for search mode)
    #[arg(long)]
    query_file: Option<String>,

    /// Logical dataset name for reporting
    #[arg(long, default_value = "sift")]
    dataset_name: String,

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

    #[arg(long, default_value_t = 500_000)]
    segment_capacity: usize,

    // --- shared execution ---
    /// Number of concurrent threads during the benchmark phase.
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// Persist ingested DB to this directory and reuse it on subsequent runs.
    /// If omitted, a temporary directory is used and discarded after the run.
    #[arg(long)]
    data_dir: Option<String>,

    // --- search-mode options ---
    /// Top-k
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Number of queries to benchmark (subset of query file)
    #[arg(long, default_value_t = 100)]
    num_queries: usize,

    /// Warmup queries before measurement
    #[arg(long, default_value_t = 10)]
    warmup_queries: usize,

    /// Skip recall computation — exact results are neither loaded nor computed.
    /// Only throughput and latency are reported.
    #[arg(long, default_value_t = false)]
    no_recall: bool,

    /// Save exact results to this path (JSON)
    #[arg(long)]
    save_exact_results: Option<String>,

    /// Load exact results from this path instead of recomputing
    #[arg(long)]
    load_exact_results: Option<String>,

    /// Save benchmark summary to this path (JSON)
    #[arg(long)]
    save_summary: Option<String>,

    // --- write-mode options ---
    /// Number of write operations to benchmark
    #[arg(long, default_value_t = 1000)]
    num_writes: usize,

    /// Warmup writes before measurement
    #[arg(long, default_value_t = 0)]
    warmup_writes: usize,

    /// Write pattern: append new doc IDs or update existing preloaded ones
    #[arg(long, default_value = "append")]
    write_pattern: WritePattern,

    /// Insert N vectors before the write benchmark starts (write mode only)
    #[arg(long, default_value_t = 0)]
    preload: usize,

    /// Wait for each write to become search-visible before proceeding (write mode only)
    #[arg(long, default_value_t = false)]
    wait_visible: bool,

    // --- mix-mode options ---
    /// Fraction of operations that are reads, e.g. 0.8 = 80% reads / 20% writes (mix mode)
    #[arg(long, default_value_t = 0.8)]
    read_write_ratio: f64,

    /// Total number of operations in mix benchmark
    #[arg(long, default_value_t = 1000)]
    num_ops: usize,

    /// Warmup operations before mix measurement
    #[arg(long, default_value_t = 0)]
    warmup_ops: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WriteSummary {
    dataset_name: String,
    metric: Metric,
    num_preloaded: usize,
    num_writes: usize,
    warmup_writes: usize,
    write_pattern: WritePattern,

    avg_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,

    concurrency: usize,
    wait_visible: bool,
    throughput_ops: f64,

    params: SegmentParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MixSummary {
    dataset_name: String,
    metric: Metric,
    num_preloaded: usize,
    num_ops: usize,
    warmup_ops: usize,
    read_write_ratio: f64,
    write_pattern: WritePattern,
    k: usize,
    concurrency: usize,
    wait_visible: bool,

    total_read_ops: usize,
    total_write_ops: usize,
    throughput_ops: f64,

    read_avg_latency_ms: f64,
    read_p50_latency_ms: f64,
    read_p95_latency_ms: f64,
    read_p99_latency_ms: f64,

    write_avg_latency_ms: f64,
    write_p50_latency_ms: f64,
    write_p95_latency_ms: f64,
    write_p99_latency_ms: f64,

    params: SegmentParams,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let cli = Cli::parse();
    let params = SegmentParams {
        m: cli.m,
        ef_construction: cli.ef_construction,
        ef_search: cli.ef_search,
        segment_capacity: cli.segment_capacity,
        ..SegmentParams::default()
    };

    // Validate mode-specific required args
    if cli.mode == BenchMode::Search && cli.query_file.is_none() {
        bail!("--query-file is required for search mode");
    }
    if cli.mode == BenchMode::Write && cli.write_pattern == WritePattern::Update && cli.preload == 0
    {
        bail!("--write-pattern update requires --preload N > 0");
    }
    if cli.mode == BenchMode::Mix && cli.query_file.is_none() {
        bail!("--query-file is required for mix mode");
    }
    if cli.mode == BenchMode::Mix && cli.write_pattern == WritePattern::Update && cli.preload == 0 {
        bail!("--write-pattern update requires --preload N > 0 in mix mode");
    }
    if cli.mode == BenchMode::Mix && !(cli.read_write_ratio > 0.0 && cli.read_write_ratio < 1.0) {
        bail!("--read-write-ratio must be in (0.0, 1.0)");
    }

    // 1. Load base vectors
    println!("Loading base vectors from {}...", cli.base_file);
    let (dim, base_vectors) = load_vectors(&cli.base_file)?;
    println!("  {} base vectors, dimension={}", base_vectors.len(), dim);

    // 2. Open or create DB
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
                    compaction_params: Default::default(),
                    wal_segment_bytes: DEFAULT_WAL_SEGMENT_BYTES,
                    metadata_schema: None,
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
            wal_segment_bytes: DEFAULT_WAL_SEGMENT_BYTES,
            metadata_schema: None,
        };
        let col = db.create_collection(schema)?;
        (DbDir::Temp(tmp), col, false)
    };

    if reused {
        println!("Reusing existing DB at {}", db_dir.path().display());
    }

    match cli.mode {
        // -----------------------------------------------------------------------
        // Search mode
        // -----------------------------------------------------------------------
        BenchMode::Search => {
            if !reused {
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

            // Load query vectors
            let query_file = cli.query_file.as_deref().unwrap();
            println!("Loading query vectors from {}...", query_file);
            let (qdim, all_queries) = load_vectors(query_file)?;
            if qdim != dim {
                bail!("dimension mismatch: base={} query={}", dim, qdim);
            }
            println!("  {} query vectors available", all_queries.len());

            // Select query subset
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

            // Warmup
            println!("Running {} warmup queries...", warmup_queries.len());
            for q in warmup_queries {
                let _ = col.search(q, cli.k, false, false)?;
            }

            // Exact results: load or compute (skipped when --no-recall)
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

            // ANN benchmark
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
                            let recall = exact_results.as_ref().map(|er| {
                                recall_at_k(&er.query_results[i], &hits_to_ids(&hits), cli.k)
                            });
                            (latency, recall)
                        })
                        .unzip()
                });
                latencies_ms = lats;
                recalls = opt_recs.into_iter().flatten().collect();
            };
            let wall_secs = bench_start.elapsed().as_secs_f64();
            let throughput_qps = bench_queries.len() as f64 / wall_secs;

            // Compute stats
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

            // Print summary
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

            // Save summary if requested
            if let Some(ref path) = cli.save_summary {
                println!("\nSaving summary to {}...", path);
                if let Some(parent) = Path::new(path).parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, serde_json::to_string_pretty(&summary)?)?;
            }
        }

        // -----------------------------------------------------------------------
        // Write mode
        // -----------------------------------------------------------------------
        BenchMode::Write => {
            // Preload N vectors into the DB before benchmarking
            let num_preloaded = if !reused && cli.preload > 0 {
                println!("Preloading {} vectors...", cli.preload);
                let preload_start = Instant::now();
                let preload_count = cli.preload.min(base_vectors.len());
                let preload_vecs = &base_vectors[..preload_count];
                let tmp_preload = db_dir.path().join("preload_vectors.raw");
                write_raw_f32(&tmp_preload, preload_vecs)?;
                let n = col.ingest_file(&tmp_preload)?;
                println!(
                    "  Preloaded {} vectors in {:.2}s",
                    n,
                    preload_start.elapsed().as_secs_f64()
                );
                n
            } else if reused {
                cli.preload
            } else {
                0
            };

            let base_len = base_vectors.len();

            // Warmup writes
            println!("Running {} warmup writes...", cli.warmup_writes);
            for i in 0..cli.warmup_writes {
                let vec_idx = i % base_len;
                let doc_id = match cli.write_pattern {
                    WritePattern::Append => format!("warmup_{}", i),
                    WritePattern::Update => format!("doc_{}", i % num_preloaded),
                };
                col.upsert(&doc_id, &base_vectors[vec_idx], None)?;
            }

            // Write benchmark
            println!(
                "Running write benchmark ({} writes, pattern={:?}, concurrency={})...",
                cli.num_writes, cli.write_pattern, cli.concurrency,
            );

            let bench_start = Instant::now();
            let mut latencies_ms: Vec<f64>;

            if cli.concurrency <= 1 {
                let mut lats = Vec::with_capacity(cli.num_writes);
                for i in 0..cli.num_writes {
                    let vec_idx = i % base_len;
                    let doc_id = match cli.write_pattern {
                        WritePattern::Append => format!("bench_{}", i),
                        WritePattern::Update => format!("doc_{}", i % num_preloaded),
                    };
                    let start = Instant::now();
                    let token = col.upsert(&doc_id, &base_vectors[vec_idx], None)?;
                    if cli.wait_visible {
                        col.wait_visible(token)?;
                    }
                    lats.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                latencies_ms = lats;
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(cli.concurrency)
                    .build()?;
                let write_pattern = cli.write_pattern.clone();
                let num_writes = cli.num_writes;
                let wait_visible = cli.wait_visible;
                latencies_ms = pool.install(|| {
                    (0..num_writes)
                        .into_par_iter()
                        .map(|i| {
                            let vec_idx = i % base_len;
                            let doc_id = match write_pattern {
                                WritePattern::Append => format!("bench_{}", i),
                                WritePattern::Update => format!("doc_{}", i % num_preloaded),
                            };
                            let start = Instant::now();
                            let token = col.upsert(&doc_id, &base_vectors[vec_idx], None).unwrap();
                            if wait_visible {
                                col.wait_visible(token).unwrap();
                            }
                            start.elapsed().as_secs_f64() * 1000.0
                        })
                        .collect()
                });
            }

            let wall_secs = bench_start.elapsed().as_secs_f64();
            let throughput_ops = cli.num_writes as f64 / wall_secs;

            // Compute stats
            latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let avg_latency = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
            let p50 = percentile(&latencies_ms, 50.0);
            let p95 = percentile(&latencies_ms, 95.0);
            let p99 = percentile(&latencies_ms, 99.0);

            let summary = WriteSummary {
                dataset_name: cli.dataset_name.clone(),
                metric: cli.metric.clone(),
                num_preloaded,
                num_writes: cli.num_writes,
                warmup_writes: cli.warmup_writes,
                write_pattern: cli.write_pattern.clone(),
                avg_latency_ms: avg_latency,
                p50_latency_ms: p50,
                p95_latency_ms: p95,
                p99_latency_ms: p99,
                concurrency: cli.concurrency,
                wait_visible: cli.wait_visible,
                throughput_ops,
                params,
            };

            // Print summary
            println!("\n===== Write Benchmark Summary =====");
            println!("Dataset:          {}", summary.dataset_name);
            println!("Metric:           {:?}", summary.metric);
            println!("Preloaded:        {}", summary.num_preloaded);
            println!("Writes:           {}", summary.num_writes);
            println!("Warmup writes:    {}", summary.warmup_writes);
            println!("Pattern:          {:?}", summary.write_pattern);
            println!("Concurrency:      {}", summary.concurrency);
            println!("Wait visible:     {}", summary.wait_visible);
            println!("---");
            println!("Throughput:       {:.1} ops/sec", summary.throughput_ops);
            println!("Avg latency:      {:.3} ms", summary.avg_latency_ms);
            println!("P50 latency:      {:.3} ms", summary.p50_latency_ms);
            println!("P95 latency:      {:.3} ms", summary.p95_latency_ms);
            println!("P99 latency:      {:.3} ms", summary.p99_latency_ms);
            println!("---");
            println!(
                "HNSW params:      m={} ef_c={} ef_s={} cap={}",
                summary.params.m,
                summary.params.ef_construction,
                summary.params.ef_search,
                summary.params.segment_capacity
            );

            // Save summary if requested
            if let Some(ref path) = cli.save_summary {
                println!("\nSaving summary to {}...", path);
                if let Some(parent) = Path::new(path).parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, serde_json::to_string_pretty(&summary)?)?;
            }
        }

        // -----------------------------------------------------------------------
        // Mix mode
        // -----------------------------------------------------------------------
        BenchMode::Mix => {
            // Preload N vectors before benchmarking
            let num_preloaded = if !reused && cli.preload > 0 {
                println!("Preloading {} vectors...", cli.preload);
                let preload_start = Instant::now();
                let preload_count = cli.preload.min(base_vectors.len());
                let preload_vecs = &base_vectors[..preload_count];
                let tmp_preload = db_dir.path().join("preload_vectors.raw");
                write_raw_f32(&tmp_preload, preload_vecs)?;
                let n = col.ingest_file(&tmp_preload)?;
                println!(
                    "  Preloaded {} vectors in {:.2}s",
                    n,
                    preload_start.elapsed().as_secs_f64()
                );
                n
            } else if reused {
                cli.preload
            } else {
                0
            };

            // Load query vectors
            let query_file = cli.query_file.as_deref().unwrap();
            println!("Loading query vectors from {}...", query_file);
            let (qdim, all_queries) = load_vectors(query_file)?;
            if qdim != dim {
                bail!("dimension mismatch: base={} query={}", dim, qdim);
            }
            println!("  {} query vectors available", all_queries.len());

            let base_len = base_vectors.len();
            let query_len = all_queries.len();
            let read_write_ratio = cli.read_write_ratio;

            // Pre-compute operation assignments deterministically (true = read, false = write)
            let total_ops = cli.warmup_ops + cli.num_ops;
            let op_is_read: Vec<bool> = {
                let mut rng = StdRng::seed_from_u64(cli.seed);
                (0..total_ops)
                    .map(|_| rng.random_bool(read_write_ratio))
                    .collect()
            };

            // Shared write counter to assign unique append doc IDs across threads
            let write_counter = Arc::new(AtomicUsize::new(num_preloaded));

            // Warmup
            println!("Running {} warmup operations...", cli.warmup_ops);
            for (i, &is_read) in op_is_read[..cli.warmup_ops].iter().enumerate() {
                if is_read {
                    let q = &all_queries[i % query_len];
                    let _ = col.search(q, cli.k, false, false)?;
                } else {
                    let vec_idx = i % base_len;
                    let doc_id = match cli.write_pattern {
                        WritePattern::Append => {
                            format!("warmup_{}", write_counter.fetch_add(1, Ordering::Relaxed))
                        }
                        WritePattern::Update => format!("doc_{}", i % num_preloaded.max(1)),
                    };
                    col.upsert(&doc_id, &base_vectors[vec_idx], None)?;
                }
            }

            // Benchmark
            println!(
                "Running mix benchmark ({} ops, read_write_ratio={:.2}, write_pattern={:?}, concurrency={})...",
                cli.num_ops, read_write_ratio, cli.write_pattern, cli.concurrency,
            );

            let bench_ops = &op_is_read[cli.warmup_ops..];
            let bench_start = Instant::now();

            let results: Vec<(bool, f64)> = if cli.concurrency <= 1 {
                let mut res = Vec::with_capacity(cli.num_ops);
                for (i, &is_read) in bench_ops.iter().enumerate() {
                    if is_read {
                        let q = &all_queries[i % query_len];
                        let start = Instant::now();
                        let _ = col.search(q, cli.k, false, false)?;
                        res.push((true, start.elapsed().as_secs_f64() * 1000.0));
                    } else {
                        let vec_idx = i % base_len;
                        let doc_id = match cli.write_pattern {
                            WritePattern::Append => {
                                format!("mix_{}", write_counter.fetch_add(1, Ordering::Relaxed))
                            }
                            WritePattern::Update => {
                                format!("doc_{}", i % num_preloaded.max(1))
                            }
                        };
                        let start = Instant::now();
                        let token = col.upsert(&doc_id, &base_vectors[vec_idx], None)?;
                        if cli.wait_visible {
                            col.wait_visible(token)?;
                        }
                        res.push((false, start.elapsed().as_secs_f64() * 1000.0));
                    }
                }
                res
            } else {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(cli.concurrency)
                    .build()?;
                let write_pattern = cli.write_pattern.clone();
                let wait_visible = cli.wait_visible;
                let wc = Arc::clone(&write_counter);
                pool.install(|| {
                    bench_ops
                        .par_iter()
                        .enumerate()
                        .map(|(i, &is_read)| {
                            if is_read {
                                let q = &all_queries[i % query_len];
                                let start = Instant::now();
                                let _ = col.search(q, cli.k, false, false).unwrap();
                                (true, start.elapsed().as_secs_f64() * 1000.0)
                            } else {
                                let vec_idx = i % base_len;
                                let doc_id = match write_pattern {
                                    WritePattern::Append => {
                                        format!("mix_{}", wc.fetch_add(1, Ordering::Relaxed))
                                    }
                                    WritePattern::Update => {
                                        format!("doc_{}", i % num_preloaded.max(1))
                                    }
                                };
                                let start = Instant::now();
                                let token =
                                    col.upsert(&doc_id, &base_vectors[vec_idx], None).unwrap();
                                if wait_visible {
                                    col.wait_visible(token).unwrap();
                                }
                                (false, start.elapsed().as_secs_f64() * 1000.0)
                            }
                        })
                        .collect()
                })
            };

            let wall_secs = bench_start.elapsed().as_secs_f64();
            let throughput_ops = cli.num_ops as f64 / wall_secs;

            // Split latencies by op type
            let mut read_lats: Vec<f64> = results
                .iter()
                .filter(|(is_read, _)| *is_read)
                .map(|(_, lat)| *lat)
                .collect();
            let mut write_lats: Vec<f64> = results
                .iter()
                .filter(|(is_read, _)| !is_read)
                .map(|(_, lat)| *lat)
                .collect();

            read_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            write_lats.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let read_avg = if read_lats.is_empty() {
                0.0
            } else {
                read_lats.iter().sum::<f64>() / read_lats.len() as f64
            };
            let write_avg = if write_lats.is_empty() {
                0.0
            } else {
                write_lats.iter().sum::<f64>() / write_lats.len() as f64
            };

            let summary = MixSummary {
                dataset_name: cli.dataset_name.clone(),
                metric: cli.metric.clone(),
                num_preloaded,
                num_ops: cli.num_ops,
                warmup_ops: cli.warmup_ops,
                read_write_ratio,
                write_pattern: cli.write_pattern.clone(),
                k: cli.k,
                concurrency: cli.concurrency,
                wait_visible: cli.wait_visible,
                total_read_ops: read_lats.len(),
                total_write_ops: write_lats.len(),
                throughput_ops,
                read_avg_latency_ms: read_avg,
                read_p50_latency_ms: percentile(&read_lats, 50.0),
                read_p95_latency_ms: percentile(&read_lats, 95.0),
                read_p99_latency_ms: percentile(&read_lats, 99.0),
                write_avg_latency_ms: write_avg,
                write_p50_latency_ms: percentile(&write_lats, 50.0),
                write_p95_latency_ms: percentile(&write_lats, 95.0),
                write_p99_latency_ms: percentile(&write_lats, 99.0),
                params,
            };

            // Print summary
            println!("\n===== Mix Benchmark Summary =====");
            println!("Dataset:          {}", summary.dataset_name);
            println!("Metric:           {:?}", summary.metric);
            println!("Preloaded:        {}", summary.num_preloaded);
            println!("Ops:              {}", summary.num_ops);
            println!("Warmup ops:       {}", summary.warmup_ops);
            println!("Read/write ratio: {:.2}", summary.read_write_ratio);
            println!("Write pattern:    {:?}", summary.write_pattern);
            println!("Concurrency:      {}", summary.concurrency);
            println!("Wait visible:     {}", summary.wait_visible);
            println!("---");
            println!(
                "Total ops:        {} reads, {} writes",
                summary.total_read_ops, summary.total_write_ops
            );
            println!("Throughput:       {:.1} ops/sec", summary.throughput_ops);
            println!("---");
            println!("Read  avg:        {:.3} ms", summary.read_avg_latency_ms);
            println!("Read  P50:        {:.3} ms", summary.read_p50_latency_ms);
            println!("Read  P95:        {:.3} ms", summary.read_p95_latency_ms);
            println!("Read  P99:        {:.3} ms", summary.read_p99_latency_ms);
            println!("---");
            println!("Write avg:        {:.3} ms", summary.write_avg_latency_ms);
            println!("Write P50:        {:.3} ms", summary.write_p50_latency_ms);
            println!("Write P95:        {:.3} ms", summary.write_p95_latency_ms);
            println!("Write P99:        {:.3} ms", summary.write_p99_latency_ms);
            println!("---");
            println!(
                "HNSW params:      m={} ef_c={} ef_s={} cap={}",
                summary.params.m,
                summary.params.ef_construction,
                summary.params.ef_search,
                summary.params.segment_capacity
            );

            // Save summary if requested
            if let Some(ref path) = cli.save_summary {
                println!("\nSaving summary to {}...", path);
                if let Some(parent) = Path::new(path).parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, serde_json::to_string_pretty(&summary)?)?;
            }
        }
    }

    Ok(())
}
