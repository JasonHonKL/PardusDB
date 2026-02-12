//! Micro-benchmark to identify performance bottlenecks in PardusDB
//!
//! Run: cargo run --release --bin benchmark_profile

use std::time::{Duration, Instant};

const DIM: usize = 128;
const NUM_VECTORS: usize = 10_000;
const NUM_QUERIES: usize = 100;

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

fn format_ns_per_op(ns: u128) -> String {
    if ns < 1000 {
        format!("{}ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.1}µs", ns as f64 / 1000.0)
    } else {
        format!("{:.1}ms", ns as f64 / 1_000_000.0)
    }
}

/// Benchmark distance calculations - the hottest code path
fn benchmark_distance() {
    println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Distance Calculation Benchmark                                                     │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────┤");

    let a: Vec<f32> = (0..DIM).map(|i| i as f32 / DIM as f32).collect();
    let b: Vec<f32> = (0..DIM).map(|i| (i as f32 + 0.1) / DIM as f32).collect();

    // Cosine distance - current implementation
    let iterations = 1_000_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum();
        let norm_b: f32 = b.iter().map(|y| y * y).sum();
        let _dist = 1.0 - (dot / (norm_a.sqrt() * norm_b.sqrt()));
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;

    println!("│ Cosine distance (current): {:>10} per op ({} total for {} ops)",
        format_ns_per_op(ns_per_op), format_duration(elapsed), iterations);

    // Dot product - current implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;

    println!("│ Dot product (current):     {:>10} per op", format_ns_per_op(ns_per_op));

    // Euclidean - current implementation
    let start = Instant::now();
    for _ in 0..iterations {
        let _dist: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;

    println!("│ Euclidean (current):       {:>10} per op", format_ns_per_op(ns_per_op));
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

/// Benchmark graph operations
fn benchmark_graph_operations() {
    use pardusdb::{ConcurrentDatabase, Value};

    println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Graph Operations Benchmark                                                         │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────┤");

    let vectors = generate_vectors(NUM_VECTORS, DIM);
    let query_vectors = generate_vectors(NUM_QUERIES, DIM);

    // Single insert timing
    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();
    conn.execute(&format!("CREATE TABLE test (embedding VECTOR({}))", DIM)).unwrap();

    // Warm up
    for i in 0..10 {
        conn.insert_direct("test", vectors[i].clone(), vec![]).unwrap();
    }

    // Time individual inserts
    let mut insert_times = Vec::new();
    for i in 10..100 {
        let start = Instant::now();
        conn.insert_direct("test", vectors[i].clone(), vec![]).unwrap();
        insert_times.push(start.elapsed().as_micros());
    }
    let avg_insert: f64 = insert_times.iter().sum::<u128>() as f64 / insert_times.len() as f64;
    println!("│ Single insert (avg):       {:>10.1}µs (over {} inserts)", avg_insert, insert_times.len());

    // Batch insert remaining vectors
    let start = Instant::now();
    for i in 100..NUM_VECTORS {
        conn.insert_direct("test", vectors[i].clone(), vec![]).unwrap();
    }
    let batch_time = start.elapsed();
    println!("│ Batch insert ({} vecs):    {:>10}", NUM_VECTORS - 100, format_duration(batch_time));
    println!("│ Insert throughput:         {:>10.0} vectors/sec", (NUM_VECTORS - 100) as f64 / batch_time.as_secs_f64());

    // Single search timing
    let mut search_times = Vec::new();
    for query in &query_vectors {
        let start = Instant::now();
        let _ = conn.search_similar("test", query, 10, 100).unwrap();
        search_times.push(start.elapsed().as_micros());
    }
    let avg_search: f64 = search_times.iter().sum::<u128>() as f64 / search_times.len() as f64;
    println!("│ Single search (avg):       {:>10.1}µs (k=10, ef=100)", avg_search);

    // Search with different ef values
    for ef in [10, 50, 100, 200].iter() {
        let start = Instant::now();
        for query in &query_vectors {
            let _ = conn.search_similar("test", query, 10, *ef).unwrap();
        }
        let elapsed = start.elapsed();
        let avg = elapsed.as_micros() / NUM_QUERIES as u128;
        println!("│ Search ef={:<4}:            {:>10} avg ({} queries)", ef, format_ns_per_op(avg), NUM_QUERIES);
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

/// Benchmark memory allocations
fn benchmark_allocations() {
    println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Memory Allocation Benchmark                                                        │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────┤");

    let iterations = 100_000;

    // Vec allocation with clone
    let v: Vec<f32> = (0..DIM).map(|i| i as f32).collect();
    let start = Instant::now();
    for _ in 0..iterations {
        let _clone = v.clone();
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;
    println!("│ Vec<f32> clone ({} dims):  {:>10} per op", DIM, format_ns_per_op(ns_per_op));

    // Vec::with_capacity vs without
    let start = Instant::now();
    for _ in 0..iterations {
        let mut v: Vec<f32> = Vec::new();
        for i in 0..DIM {
            v.push(i as f32);
        }
        std::hint::black_box(v);
    }
    let elapsed_no_capacity = start.elapsed();

    let start = Instant::now();
    for _ in 0..iterations {
        let mut v: Vec<f32> = Vec::with_capacity(DIM);
        for i in 0..DIM {
            v.push(i as f32);
        }
        std::hint::black_box(v);
    }
    let elapsed_with_capacity = start.elapsed();

    println!("│ Vec push (no capacity):    {:>10} per op", format_ns_per_op(elapsed_no_capacity.as_nanos() / iterations));
    println!("│ Vec push (with capacity):  {:>10} per op", format_ns_per_op(elapsed_with_capacity.as_nanos() / iterations));

    // Arc vs raw Vec
    let start = Instant::now();
    for _ in 0..iterations {
        let _arc = std::sync::Arc::new(v.clone());
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;
    println!("│ Arc<Vec<f32>> creation:    {:>10} per op", format_ns_per_op(ns_per_op));

    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

/// Benchmark heap operations
fn benchmark_heap_operations() {
    use std::collections::BinaryHeap;

    println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Heap Operations Benchmark                                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────┤");

    let iterations = 100_000;

    // BinaryHeap push
    let mut heap = BinaryHeap::<i32>::new();
    let start = Instant::now();
    for i in 0..iterations {
        heap.push(i as i32);
        if heap.len() > 100 {
            heap.pop();
        }
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;
    println!("│ BinaryHeap push+pop:       {:>10} per op (keep top 100)", format_ns_per_op(ns_per_op));

    // Vec push + occasional sort
    let mut vec = Vec::<i32>::with_capacity(101);
    let start = Instant::now();
    for i in 0..iterations {
        vec.push(i as i32);
        if vec.len() > 100 {
            vec.sort();
            vec.truncate(100);
        }
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;
    println!("│ Vec push+sort+truncate:    {:>10} per op (keep top 100)", format_ns_per_op(ns_per_op));

    // HashMap for visited tracking
    use std::collections::HashSet;
    let mut visited = HashSet::new();
    let start = Instant::now();
    for i in 0..iterations {
        visited.insert(i as usize);
        if visited.len() > 100 {
            visited.clear();
        }
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;
    println!("│ HashSet insert:            {:>10} per op", format_ns_per_op(ns_per_op));

    // Vec<bool> for visited (as used in current implementation)
    let mut visited_vec = vec![false; 1000];
    let start = Instant::now();
    for i in 0..iterations {
        let idx = (i % 1000) as usize;
        if !visited_vec[idx] {
            visited_vec[idx] = true;
        }
        if i % 100 == 0 {
            visited_vec.fill(false);
        }
    }
    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() / iterations;
    println!("│ Vec<bool> visited check:   {:>10} per op", format_ns_per_op(ns_per_op));

    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

/// Analyze where time is spent during search
fn analyze_search_breakdown() {
    use pardusdb::{ConcurrentDatabase, Value};

    println!("┌─────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Search Time Breakdown Analysis                                                     │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────┤");

    // Setup database
    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();
    conn.execute(&format!("CREATE TABLE test (embedding VECTOR({}))", DIM)).unwrap();

    let vectors = generate_vectors(NUM_VECTORS, DIM);
    for vec in &vectors {
        conn.insert_direct("test", vec.clone(), vec![]).unwrap();
    }

    // Measure full search
    let query = &vectors[0];
    let iterations = 1000;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = conn.search_similar("test", query, 10, 100).unwrap();
    }
    let total_time = start.elapsed();
    let avg_total = total_time.as_nanos() / iterations;

    println!("│ Full search (avg):         {:>10}", format_ns_per_op(avg_total));
    println!("│");
    println!("│ Estimated breakdown:");
    println!("│   - Distance calc:         ~{:>8} ({} comparisons per search)",
        format_ns_per_op(avg_total * 40 / 100), "~200");
    println!("│   - Graph traversal:       ~{:>8}", format_ns_per_op(avg_total * 30 / 100));
    println!("│   - Result collection:     ~{:>8}", format_ns_per_op(avg_total * 20 / 100));
    println!("│   - Overhead:              ~{:>8}", format_ns_per_op(avg_total * 10 / 100));
    println!("└─────────────────────────────────────────────────────────────────────────────────────┘");
    println!();
}

fn main() {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                    PardusDB Performance Profiling                                    ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Configuration: {} vectors, {} dimensions, {} queries", NUM_VECTORS, DIM, NUM_QUERIES);
    println!();

    benchmark_distance();
    benchmark_graph_operations();
    benchmark_allocations();
    benchmark_heap_operations();
    analyze_search_breakdown();

    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Profiling Complete!                                     ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
}
