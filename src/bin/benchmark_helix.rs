//! Benchmark comparison: PardusDB vs HelixDB for vector similarity search
//!
//! Prerequisites for HelixDB comparison:
//! 1. Install Helix CLI:
//!    curl -sSL "https://install.helix-db.com" | bash
//!
//! 2. Initialize and deploy HelixDB:
//!    mkdir helix_bench && cd helix_bench
//!    helix init
//!    helix push dev
//!
//! 3. Build with HelixDB support:
//!    cargo run --release --features helix --bin benchmark_helix
//!
//! Or run without HelixDB:
//!    cargo run --release --bin benchmark_helix

use std::time::{Duration, Instant};

const DIM: usize = 128;
const NUM_VECTORS: usize = 10_000;
const NUM_QUERIES: usize = 100;
const K: usize = 10;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f32 / (n * dim) as f32).collect())
        .collect()
}

fn format_duration(d: Duration) -> String {
    if d.as_micros() < 1000 {
        format!("{}µs", d.as_micros())
    } else if d.as_millis() < 1000 {
        format!("{}ms", d.as_millis())
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}

fn format_ops(count: usize, duration: Duration) -> String {
    let ops = count as f64 / duration.as_secs_f64();
    if ops >= 1_000.0 {
        format!("{:.1}K/s", ops / 1_000.0)
    } else {
        format!("{:.1}/s", ops)
    }
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    insert_time: Duration,
    search_time: Duration,
    single_search: Duration,
}

fn pardusdb_benchmark() -> BenchmarkResult {
    use pardusdb::{ConcurrentDatabase, Value};

    println!("  [PardusDB] Starting benchmark...");

    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute(&format!(
        "CREATE TABLE vectors (embedding VECTOR({}), id INTEGER);",
        DIM
    )).unwrap();

    let vectors = generate_vectors(NUM_VECTORS, DIM);
    let query_vectors = generate_vectors(NUM_QUERIES, DIM);

    // Insert using batch for better performance
    println!("  [PardusDB] Inserting {} vectors...", NUM_VECTORS);
    let insert_start = Instant::now();

    // Use batch insert
    const BATCH_SIZE: usize = 1000;
    for chunk in vectors.chunks(BATCH_SIZE) {
        let batch_vectors: Vec<Vec<f32>> = chunk.to_vec();
        let batch_metadata: Vec<Vec<(&str, Value)>> = chunk.iter()
            .enumerate()
            .map(|(i, _)| vec![("id", Value::Integer(i as i64))])
            .collect();
        conn.insert_batch_direct("vectors", batch_vectors, batch_metadata).unwrap();
    }
    let insert_time = insert_start.elapsed();

    // Search
    println!("  [PardusDB] Running {} similarity searches...", NUM_QUERIES);
    let search_start = Instant::now();
    for query in &query_vectors {
        let _ = conn.search_similar("vectors", query, K, 100).unwrap();
    }
    let search_time = search_start.elapsed();
    let single_search = search_time / NUM_QUERIES as u32;

    BenchmarkResult {
        name: "PardusDB".to_string(),
        insert_time,
        search_time,
        single_search,
    }
}

#[cfg(feature = "helix")]
fn helix_benchmark() -> Option<BenchmarkResult> {
    use reqwest::blocking::Client;
    use serde_json::json;
    use std::thread;
    use std::time::Duration;

    println!("  [HelixDB] Connecting to HelixDB...");

    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let base_url = "http://localhost:6969";

    // Test connection
    let response = client.get(format!("{}/health", base_url)).send();
    if response.is_err() {
        println!("  [HelixDB] Connection failed: {}", response.unwrap_err());
        return None;
    }

    println!("  [HelixDB] Connected successfully!");

    let vectors = generate_vectors(NUM_VECTORS, DIM);
    let query_vectors = generate_vectors(NUM_QUERIES, DIM);

    // Insert benchmark with small delay to avoid overwhelming the server
    println!("  [HelixDB] Inserting {} vectors...", NUM_VECTORS);
    let insert_start = Instant::now();

    for (i, vec) in vectors.iter().enumerate() {
        let vector: Vec<f64> = vec.iter().map(|&x| x as f64).collect();

        // Retry logic for resilience
        let mut retries = 0;
        loop {
            let response = client.post(format!("{}/query/insertVector", base_url))
                .json(&json!({
                    "doc_id": i,
                    "vector": vector
                }))
                .send();

            match response {
                Ok(_) => break,
                Err(e) if retries < 3 => {
                    retries += 1;
                    println!("  [HelixDB] Retry {} for vector {}: {}", retries, i, e);
                    thread::sleep(Duration::from_millis(100));
                    // Try to reconnect
                    let _ = client.get(format!("{}/health", base_url)).send();
                }
                Err(e) => {
                    println!("  [HelixDB] Insert error at {}: {}", i, e);
                    return None;
                }
            }
        }

        // Small delay every 100 inserts to avoid overwhelming the server
        if i > 0 && i % 100 == 0 {
            thread::sleep(Duration::from_millis(1));
        }
    }
    let insert_time = insert_start.elapsed();

    // Search benchmark
    println!("  [HelixDB] Running {} similarity searches...", NUM_QUERIES);
    let search_start = Instant::now();

    for query_vec in &query_vectors {
        let vector: Vec<f64> = query_vec.iter().map(|&x| x as f64).collect();
        let _ = client.post(format!("{}/query/searchVector", base_url))
            .json(&json!({
                "vector": vector,
                "k": K
            }))
            .send();
    }
    let search_time = search_start.elapsed();
    let single_search = search_time / NUM_QUERIES as u32;

    Some(BenchmarkResult {
        name: "HelixDB".to_string(),
        insert_time,
        search_time,
        single_search,
    })
}

#[cfg(not(feature = "helix"))]
fn helix_benchmark() -> Option<BenchmarkResult> {
    None
}

fn print_comparison_table() {
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PardusDB vs HelixDB - Feature Comparison                        ║");
    println!("╠═════════════════════╦═════════════════════════╦════════════════════════════════════╣");
    println!("║ Feature             ║ PardusDB                ║ HelixDB                            ║");
    println!("╠═════════════════════╬═════════════════════════╬════════════════════════════════════╣");
    println!("║ Architecture        ║ Embedded (SQLite-like)  ║ Server (Docker)                    ║");
    println!("║ Implementation      ║ Rust (native)           ║ Rust (native)                      ║");
    println!("║ Vector Index        ║ HNSW (optimized)        ║ HNSW                               ║");
    println!("║ Graph Support       ║ No                      ║ Yes                                ║");
    println!("║ Deployment          ║ Single binary/file      ║ Docker + CLI                       ║");
    println!("║ Setup Time          ║ 0 seconds               ║ 5-10 minutes                       ║");
    println!("║ Memory Overhead     ║ Minimal (~50MB)         ║ Docker container overhead          ║");
    println!("║ Query Language      ║ SQL-like                ║ HelixQL                            ║");
    println!("║ Network Latency     ║ None (in-process)       ║ HTTP API overhead                  ║");
    println!("║ Persistence         ║ Single file (.pardus)   ║ LMDB                               ║");
    println!("║ License             ║ MIT                     ║ AGPL-3.0                           ║");
    println!("║ Best For            ║ Vector similarity       ║ Graph + Vector                     ║");
    println!("╚═════════════════════╩═════════════════════════╩════════════════════════════════════╝");
    println!();
}

fn main() {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("              PardusDB vs HelixDB - Vector Similarity Benchmark                        ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Configuration:");
    println!("  • Vector dimension: {}", DIM);
    println!("  • Number of vectors: {}", NUM_VECTORS);
    println!("  • Number of queries: {}", NUM_QUERIES);
    println!("  • Top-K: {}", K);
    println!();

    let mut results = Vec::new();

    // Run PardusDB benchmark
    println!("[1/2] Running PardusDB benchmark...");
    results.push(pardusdb_benchmark());
    println!();

    // Try HelixDB benchmark
    println!("[2/2] Running HelixDB benchmark...");
    if let Some(helix_result) = helix_benchmark() {
        results.push(helix_result);
    } else {
        println!();
        println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
        println!("│ HelixDB Not Available                                                               │");
        println!("├─────────────────────────────────────────────────────────────────────────────────────┤");
        println!("│ To run HelixDB comparison:                                                         │");
        println!("│                                                                                     │");
        println!("│   # Install Helix CLI                                                               │");
        println!("│   curl -sSL \"https://install.helix-db.com\" | bash                                  │");
        println!("│                                                                                     │");
        println!("│   # Initialize and deploy                                                           │");
        println!("│   mkdir helix_bench && cd helix_bench                                               │");
        println!("│   helix init                                                                        │");
        println!("│   helix push dev                                                                    │");
        println!("│                                                                                     │");
        println!("│   # Build and run with HelixDB support                                              │");
        println!("│   cargo run --release --features helix --bin benchmark_helix                        │");
        println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
    }
    println!();

    // Print results table
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Benchmark Results                                        ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("┌──────────────┬─────────────────────┬───────────────────────┬─────────────────────┐");
    println!("│ Database     │ Insert ({} vectors) │ Search ({} queries)   │ Single Search       │", NUM_VECTORS, NUM_QUERIES);
    println!("├──────────────┼─────────────────────┼───────────────────────┼─────────────────────┤");

    for r in &results {
        println!("│ {:<12} │ {:>10} ({:>6}) │ {:>10} ({:>6}) │ {:>10}          │",
            r.name,
            format_duration(r.insert_time),
            format_ops(NUM_VECTORS, r.insert_time),
            format_duration(r.search_time),
            format_ops(NUM_QUERIES, r.search_time),
            format_duration(r.single_search)
        );
    }
    println!("└──────────────┴─────────────────────┴───────────────────────┴─────────────────────┘");
    println!();

    // Speedup comparison
    if results.len() > 1 {
        let pardus = &results[0];
        let helix = &results[1];

        let insert_speedup = helix.insert_time.as_secs_f64() / pardus.insert_time.as_secs_f64();
        let search_speedup = helix.search_time.as_secs_f64() / pardus.search_time.as_secs_f64();

        println!("═══════════════════════════════════════════════════════════════════════════════════════");
        println!("                              Performance Comparison                                  ");
        println!("═══════════════════════════════════════════════════════════════════════════════════════");
        println!();
        println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
        println!("│ PardusDB vs HelixDB Speedup                                                         │");
        println!("├─────────────────────────────────────────────────────────────────────────────────────┤");
        println!("│   Insert:    {:>8.1}x faster                                                        │", insert_speedup);
        println!("│   Search:    {:>8.1}x faster                                                        │", search_speedup);
        println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // Print comparison table
    print_comparison_table();

    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Benchmark Complete!                                      ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
}
