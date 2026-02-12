//! Benchmark comparing batch insert vs individual inserts
//!
//! Run: cargo run --release --bin benchmark_batch

use std::time::{Duration, Instant};

const DIM: usize = 128;
const NUM_VECTORS: usize = 10_000;
const BATCH_SIZE: usize = 100;

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

fn benchmark_individual_insert(vectors: &[Vec<f32>]) -> Duration {
    use pardusdb::{ConcurrentDatabase, Value};

    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute(&format!(
        "CREATE TABLE test (embedding VECTOR({}), id INTEGER)",
        DIM
    )).unwrap();

    println!("  Testing individual inserts...");
    let start = Instant::now();

    for (i, vec) in vectors.iter().enumerate() {
        conn.insert_direct(
            "test",
            vec.clone(),
            vec![("id", Value::Integer(i as i64))],
        ).unwrap();
    }

    start.elapsed()
}

fn benchmark_batch_insert(vectors: &[Vec<f32>], batch_size: usize) -> Duration {
    use pardusdb::{ConcurrentDatabase, Value};

    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute(&format!(
        "CREATE TABLE test (embedding VECTOR({}), id INTEGER)",
        DIM
    )).unwrap();

    println!("  Testing batch inserts (batch_size={})...", batch_size);
    let start = Instant::now();

    for chunk in vectors.chunks(batch_size) {
        let vectors_batch: Vec<Vec<f32>> = chunk.to_vec();
        let metadata: Vec<Vec<(&str, Value)>> = chunk.iter()
            .enumerate()
            .map(|(i, _)| vec![("id", Value::Integer(i as i64))])
            .collect();

        conn.insert_batch_direct("test", vectors_batch, metadata).unwrap();
    }

    start.elapsed()
}

fn benchmark_search_after_insert(vectors: &[Vec<f32>], num_queries: usize) {
    use pardusdb::{ConcurrentDatabase, Value};

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Search Performance After Batch Insert                                               │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────┤");

    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute(&format!(
        "CREATE TABLE test (embedding VECTOR({}), id INTEGER)",
        DIM
    )).unwrap();

    // Batch insert
    let vectors_batch: Vec<Vec<f32>> = vectors.to_vec();
    let metadata: Vec<Vec<(&str, Value)>> = vectors.iter()
        .enumerate()
        .map(|(i, _)| vec![("id", Value::Integer(i as i64))])
        .collect();

    conn.insert_batch_direct("test", vectors_batch, metadata).unwrap();

    // Search benchmark
    let query_vectors: Vec<Vec<f32>> = vectors.iter().take(num_queries).cloned().collect();

    let start = Instant::now();
    for query in &query_vectors {
        let _ = conn.search_similar("test", query, 10, 100).unwrap();
    }
    let search_time = start.elapsed();

    println!("│ Search ({} queries):   {:>10} ({})", num_queries, format_duration(search_time), format_ops(num_queries, search_time));
    println!("│ Single search:        {:>10}", format_duration(search_time / num_queries as u32));
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
}

fn main() {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("              PardusDB Batch Insert Benchmark                                          ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Configuration:");
    println!("  • Vector dimension: {}", DIM);
    println!("  • Number of vectors: {}", NUM_VECTORS);
    println!("  • Batch size: {}", BATCH_SIZE);
    println!();

    let vectors = generate_vectors(NUM_VECTORS, DIM);

    // Individual inserts
    println!("[1/3] Benchmarking individual inserts...");
    let individual_time = benchmark_individual_insert(&vectors);
    println!();

    // Batch inserts with different sizes
    println!("[2/3] Benchmarking batch inserts...");
    let mut batch_times: Vec<(usize, Duration)> = Vec::new();

    for batch_size in [50, 100, 250, 500, 1000].iter() {
        let time = benchmark_batch_insert(&vectors, *batch_size);
        batch_times.push((*batch_size, time));
    }
    println!();

    // Search after batch insert
    println!("[3/3] Testing search after batch insert...");
    benchmark_search_after_insert(&vectors, 100);
    println!();

    // Print results
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Results                                                  ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();

    println!("┌────────────────────────┬─────────────────────┬─────────────────┬──────────────────┐");
    println!("│ Method                 │ Insert ({} vecs)  │ Throughput      │ Speedup          │", NUM_VECTORS);
    println!("├────────────────────────┼─────────────────────┼─────────────────┼──────────────────┤");

    println!("│ {:<22} │ {:>10} ({:>6}) │ {:>15} │ {:>16} │",
        "Individual",
        format_duration(individual_time),
        format_ops(NUM_VECTORS, individual_time),
        format_ops(NUM_VECTORS, individual_time),
        "1.0x (baseline)"
    );

    for (batch_size, time) in &batch_times {
        let speedup = individual_time.as_secs_f64() / time.as_secs_f64();
        println!("│ {:<22} │ {:>10} ({:>6}) │ {:>15} │ {:>16.1}x │",
            format!("Batch (size={})", batch_size),
            format_duration(*time),
            format_ops(NUM_VECTORS, *time),
            format_ops(NUM_VECTORS, *time),
            speedup
        );
    }

    println!("└────────────────────────┴─────────────────────┴─────────────────┴──────────────────┘");
    println!();

    // Best speedup
    if let Some((batch_size, time)) = batch_times.iter().max_by(|a, b| {
        let speedup_a = individual_time.as_secs_f64() / a.1.as_secs_f64();
        let speedup_b = individual_time.as_secs_f64() / b.1.as_secs_f64();
        speedup_a.partial_cmp(&speedup_b).unwrap()
    }) {
        let best_speedup = individual_time.as_secs_f64() / time.as_secs_f64();
        println!("┌──────────────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Best Configuration: Batch size {}", batch_size);
        println!("│ Speedup: {:.1}x faster than individual inserts", best_speedup);
        println!("│ Insert throughput: {}", format_ops(NUM_VECTORS, *time));
        println!("└──────────────────────────────────────────────────────────────────────────────────────┘");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Benchmark Complete!                                      ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
}
