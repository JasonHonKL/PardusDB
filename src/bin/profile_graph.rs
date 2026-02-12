//! Deep profile of graph operations

use std::time::Instant;

fn main() {
    println!("=== Deep Profile of Graph Insert ===\n");

    const NUM_DOCS: usize = 200;
    const DIM: usize = 128;

    // Create graph
    let mut graph: pardusdb::Graph<f32, pardusdb::Euclidean> =
        pardusdb::Graph::new(DIM, pardusdb::GraphConfig::default());

    // Track insert times as graph grows
    let mut times = Vec::new();

    for i in 0..NUM_DOCS {
        let vec: Vec<f32> = (0..DIM).map(|j| ((i * DIM + j) as f32 * 0.001)).collect();

        let start = Instant::now();
        graph.insert(vec);
        let elapsed = start.elapsed();

        times.push(elapsed);

        if (i + 1) % 50 == 0 {
            let avg: std::time::Duration = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            println!("After {} inserts: avg {:?}", i + 1, avg);
        }
    }

    // Analyze scaling
    println!("\n=== Scaling Analysis ===");
    let first_10: std::time::Duration = times[..10].iter().sum();
    let next_10: std::time::Duration = times[10..20].iter().sum();
    let next_10_2: std::time::Duration = times[20..30].iter().sum();
    let last_10: std::time::Duration = times[190..200].iter().sum();

    println!("First 10 avg:  {:?}", first_10 / 10);
    println!("10-20 avg:     {:?}", next_10 / 10);
    println!("20-30 avg:     {:?}", next_10_2 / 10);
    println!("Last 10 avg:   {:?}", last_10 / 10);

    // Query performance
    println!("\n=== Query Performance ===");
    let query_vec: Vec<f32> = (0..DIM).map(|j| (j as f32 * 0.001)).collect();

    let mut query_times = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let results = graph.query(&query_vec, 10, 100);
        query_times.push(start.elapsed());
    }

    let avg_query: std::time::Duration = query_times.iter().sum::<std::time::Duration>() / 100;
    println!("Avg query time (k=10, ef=100): {:?}", avg_query);

    // Profile individual operations
    println!("\n=== Where is time spent? ===");
    println!("The main operations in insert are:");
    println!("1. search() - BFS traversal with BTreeSet");
    println!("2. robust_prune() - geometric diversity check");
    println!("3. reverse_prune() - backlink pruning");
    println!("4. distance calculations (768 dims each)");
    println!("\nFor 768-dim vectors with search_buffer=200:");
    println!("  - Each insert searches ~200 candidates");
    println!("  - Each distance calc: 768 multiplications + additions");
    println!("  - BTreeSet operations for priority queue");
}
