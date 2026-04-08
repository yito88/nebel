# Nebel

Nebel is a lightweight embedded vector database written in Rust. It stores vectors on local disk, indexes them with HNSW, persists collection metadata in `redb`, and uses a write-ahead log (WAL) plus a background apply worker to make writes durable before they become search-visible.

The current codebase is a storage engine and evaluation harness, not a networked service. You embed it in a Rust process and work with collections through the library API.

## Current capabilities

- Embedded Rust API via [`Db`](src/db.rs) and [`CollectionHandle`](src/handle.rs)
- Approximate nearest-neighbor search with HNSW
- Exact brute-force search for correctness and recall checks
- Per-document metadata stored as JSON
- Delete via tombstones
- Near-real-time visibility with `WriteToken` + `wait_visible`
- WAL-backed recovery for normal write operations
- Multi-segment collections with automatic sealing at capacity
- Benchmark and verification CLIs for ANN quality and latency testing

## What Nebel is not yet

- No server, REST API, or gRPC interface
- No filtering, hybrid search, or metadata predicates
- No collection deletion or schema migration support

## Storage model

Each collection lives under a directory inside the database root:

- `nebel.redb`: shared metadata store
- `<collection>/seg_###/vectors.seg`: raw vector data for each segment
- `<collection>/seg_###/index*`: persisted HNSW files for sealed segments
- `<collection>/wal/*.log`: segmented WAL files for normal writes

The `redb` database stores:

- collection schemas
- segment metadata and manifest state
- document-to-segment mappings
- tombstones
- metadata JSON
- reverse lookup from `(segment, internal_id)` to `doc_id`
- last applied WAL sequence

## API overview

### Open a database and create a collection

```rust
use nebel::{
    Db,
    types::{CollectionId, CollectionSchema, Metric},
};

let db = Db::open("data")?;

let schema = CollectionSchema::new(CollectionId::new("docs"), 4, Metric::L2);
let col = db.create_collection(schema)?;
```

To reopen an existing collection:

```rust
let db = Db::open("data")?;
let col = db.collection("docs")?;
```

### Write and wait for visibility

Writes are durable once the WAL append returns, but they are eventually(not immediately) visible to search. Every write returns a `WriteToken`; call `wait_visible` if the next read must observe it.

```rust
let token = col.upsert("doc-1", &[1.0, 0.0, 0.0, 0.0], None)?;
col.wait_visible(token)?;
```

Supported write operations:

- `upsert(doc_id, vector, metadata)`
- `upsert_batch(&[doc_id, vector, metadata])`
- `delete(doc_id)`
- `update_metadata(doc_id, metadata)`

### Search

Approximate search:

```rust
let hits = col.search(&[1.0, 0.1, 0.0, 0.0], 5, true, false)?;
```

Parameters:

- query vector
- `k`
- `include_metadata`
- `include_vector`

Exact search:

```rust
let hits = col.search_exact(&[1.0, 0.1, 0.0, 0.0], 5)?;
```

`SearchHit.score` is the raw distance produced by the configured metric implementation. Lower scores rank first.

## Consistency and durability

The write path is:

1. Append a WAL record and `fsync` it.
2. Return a `WriteToken` to the caller.
3. A background worker applies durable WAL records into the active segment and metadata store.
4. A fresh snapshot is published.
5. `wait_visible(token)` unblocks once the write is visible to search.

This gives Nebel near-real-time reads with explicit visibility control instead of fully synchronous write application.

On restart, Nebel:

1. Reopens collection metadata and known segments.
2. Marks pre-existing WAL segments as recovery inputs.
3. Replays closed WAL segments in order.
4. Persists the recovered applied sequence.
5. Opens a fresh active WAL segment for new writes.

## Segments

Each collection always has one writable segment and zero or more sealed segments.

- Writable segment: accepts inserts and maintains an in-memory HNSW index
- Sealed segment: immutable, persisted to disk, and reopened as read-only

When the writable segment reaches `segment_capacity`, Nebel seals it and creates a new writable segment. Search runs across all active segments and merges candidates by distance.

Default segment parameters are defined in [`SegmentParams`](src/types.rs):

- `m = 16`
- `ef_construction = 200`
- `ef_search = 50`
- `segment_capacity = 100_000`

## Command-line tools

`cargo run --bin verify -- ...`

- Loads a dataset plus ground truth
- Builds or reuses a Nebel collection
- Compares ANN and exact results against expected neighbors

`cargo run --bin bench -- ...`

- Loads base and query vectors
- Builds or reuses a Nebel collection
- Measures latency, throughput, and optional recall

The helper script [`scripts/download_sift.sh`](scripts/download_sift.sh) downloads a SIFT dataset variant used by these tools.

Latest published daily benchmark and verification results:

- [GitHub Pages report](https://yito88.github.io/nebel/)

## Status

The implementation already supports end-to-end persistence, recovery, ANN search, exact search, and evaluation workflows. It is still an early-stage engine, and some operational features that a production vector DB would usually need are intentionally not built yet.
