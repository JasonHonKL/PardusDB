//! Accuracy Benchmark: PardusDB vs Neo4j
//!
//! Compares search accuracy (recall, precision) against brute-force ground truth.
//!
//! Run: cargo run --release --bin benchmark_accuracy
//! With Neo4j: cargo run --release --features neo4j --bin benchmark_accuracy

use std::time::Instant;

const DIM: usize = 128;
const NUM_VECTORS: usize = 5_000;
const NUM_QUERIES: usize = 50;
const K: usize = 10;

fn generate_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| (0..dim).map(|j| (i * dim + j) as f32 / (n * dim) as f32).collect())
        .collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let mag_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 1.0;
    }
    (1.0 - dot / (mag_a * mag_b)) as f32
}

/// Brute-force exact search for ground truth
fn brute_force_search(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_distance(query, v)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).collect()
}

#[derive(Debug, Clone)]
struct AccuracyMetrics {
    recall_at_k: f64,      // What fraction of true neighbors were found
    precision_at_k: f64,   // What fraction of results are correct
    mrr: f64,              // Mean Reciprocal Rank
    avg_distance_error: f64, // Average distance difference from ground truth
}

fn calculate_metrics(results: &[(usize, f32)], ground_truth: &[(usize, f32)], k: usize) -> AccuracyMetrics {
    let result_ids: std::collections::HashSet<usize> = results.iter().take(k).map(|(id, _)| *id).collect();
    let gt_ids: std::collections::HashSet<usize> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();

    // Recall: how many ground truth items were found
    let hits = result_ids.intersection(&gt_ids).count();
    let recall = hits as f64 / k as f64;

    // Precision: same as recall for top-k (we return exactly k items)
    let precision = recall;

    // MRR: mean reciprocal rank
    let mut mrr = 0.0;
    for (rank, (id, _)) in results.iter().take(k).enumerate() {
        if gt_ids.contains(id) {
            mrr += 1.0 / (rank + 1) as f64;
        }
    }
    mrr /= k as f64;

    // Average distance error
    let mut total_error = 0.0;
    let mut count = 0;
    for (id, dist) in results.iter().take(k) {
        if let Some((_, gt_dist)) = ground_truth.iter().take(k).find(|(gt_id, _)| gt_id == id) {
            total_error += (dist - gt_dist).abs() as f64;
            count += 1;
        }
    }
    let avg_error = if count > 0 { total_error / count as f64 } else { 0.0 };

    AccuracyMetrics {
        recall_at_k: recall,
        precision_at_k: precision,
        mrr,
        avg_distance_error: avg_error,
    }
}

fn pardusdb_accuracy_test(vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> (AccuracyMetrics, AccuracyMetrics, AccuracyMetrics) {
    use pardusdb::{ConcurrentDatabase, Value};

    println!("  [PardusDB] Setting up database...");
    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute(&format!(
        "CREATE TABLE vectors (embedding VECTOR({}), vec_id INTEGER);",
        DIM
    )).unwrap();

    // Insert vectors with their index as vec_id
    for (i, vec) in vectors.iter().enumerate() {
        conn.insert_direct(
            "vectors",
            vec.clone(),
            vec![("vec_id", Value::Integer(i as i64))],
        ).unwrap();
    }

    println!("  [PardusDB] Running {} queries...", queries.len());

    let mut k10_metrics = Vec::new();
    let mut k5_metrics = Vec::new();
    let mut k1_metrics = Vec::new();

    for query in queries {
        // Get ground truth
        let gt_10 = brute_force_search(vectors, query, 10);
        let gt_5: Vec<_> = gt_10.iter().take(5).cloned().collect();
        let gt_1: Vec<_> = gt_10.iter().take(1).cloned().collect();

        // PardusDB search
        let results = conn.search_similar("vectors", query, 10, 100).unwrap();

        let pardus_results: Vec<(usize, f32)> = results.iter()
            .map(|(row_id, values, _dist)| {
                // Get vec_id from values
                let vec_id = values.iter()
                    .find_map(|v| if let Value::Integer(i) = v { Some(*i as usize) } else { None })
                    .unwrap_or_else(|| {
                        // row_id is 1-indexed, our vector index is 0-indexed
                        if *row_id > 0 { *row_id as usize - 1 } else { 0 }
                    });
                // Recalculate distance for fair comparison
                let dist = cosine_distance(&vectors[vec_id], query);
                (vec_id, dist)
            })
            .collect();

        k10_metrics.push(calculate_metrics(&pardus_results, &gt_10, 10));
        k5_metrics.push(calculate_metrics(&pardus_results, &gt_5, 5));
        k1_metrics.push(calculate_metrics(&pardus_results, &gt_1, 1));
    }

    let avg_k10 = average_metrics(&k10_metrics);
    let avg_k5 = average_metrics(&k5_metrics);
    let avg_k1 = average_metrics(&k1_metrics);

    (avg_k10, avg_k5, avg_k1)
}

#[cfg(feature = "neo4j")]
fn neo4j_accuracy_test(vectors: &[Vec<f32>], queries: &[Vec<f32>]) -> Option<(AccuracyMetrics, AccuracyMetrics, AccuracyMetrics)> {
    use neo4rs::{Graph, query};
    use std::sync::OnceLock;

    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let rt = RT.get_or_init(|| tokio::runtime::Runtime::new().unwrap());

    rt.block_on(async {
        println!("  [Neo4j] Connecting...");
        let graph = match Graph::new("bolt://localhost:7687", "neo4j", "password123").await {
            Ok(g) => g,
            Err(e) => {
                println!("  [Neo4j] Connection failed: {}", e);
                return None;
            }
        };

        println!("  [Neo4j] Connected, setting up database...");
        let _ = graph.run(query("MATCH (n:Vector) DETACH DELETE n")).await;

        // Try to create vector index, ignore if already exists
        let _ = graph.run(query(
            "CALL db.index.vector.createNodeIndex('vector_index', 'Vector', 'embedding', 128, 'cosine')"
        )).await;

        // Insert vectors
        println!("  [Neo4j] Inserting {} vectors...", vectors.len());
        for (i, vec) in vectors.iter().enumerate() {
            let embedding: Vec<f64> = vec.iter().map(|&x| x as f64).collect();
            if let Err(e) = graph.run(query("CREATE (v:Vector {id: $id, embedding: $embedding})")
                .param("id", i as i64)
                .param("embedding", embedding)
            ).await {
                println!("  [Neo4j] Insert error at {}: {}", i, e);
                return None;
            }
        }

        println!("  [Neo4j] Running {} queries...", queries.len());

        let mut k10_metrics = Vec::new();
        let mut k5_metrics = Vec::new();
        let mut k1_metrics = Vec::new();

        for (q_idx, query_vec) in queries.iter().enumerate() {
            let gt_10 = brute_force_search(vectors, query_vec, 10);
            let gt_5: Vec<_> = gt_10.iter().take(5).cloned().collect();
            let gt_1: Vec<_> = gt_10.iter().take(1).cloned().collect();

            let embedding: Vec<f64> = query_vec.iter().map(|&x| x as f64).collect();
            let result = graph.execute(query(
                "CALL db.index.vector.queryNodes('vector_index', 10, $embedding) YIELD node, score RETURN node.id as id, score"
            ).param("embedding", embedding)).await;

            let mut neo4j_results = Vec::new();
            match result {
                Ok(mut rows) => {
                    while let Ok(Some(row)) = rows.next().await {
                        if let (Ok(id), Ok(score)) = (row.get::<i64>("id"), row.get::<f64>("score")) {
                            // Convert similarity score to distance (cosine distance = 1 - similarity)
                            let dist = 1.0 - score as f32;
                            neo4j_results.push((id as usize, dist));
                        }
                    }
                }
                Err(e) => {
                    if q_idx == 0 {
                        println!("  [Neo4j] Query error: {}", e);
                    }
                }
            }

            // Debug first query
            if q_idx == 0 {
                println!("  [Neo4j] First query returned {} results", neo4j_results.len());
                if !neo4j_results.is_empty() {
                    println!("  [Neo4j] First 3 results: {:?}", &neo4j_results[..3.min(neo4j_results.len())]);
                }
            }

            k10_metrics.push(calculate_metrics(&neo4j_results, &gt_10, 10));
            k5_metrics.push(calculate_metrics(&neo4j_results, &gt_5, 5));
            k1_metrics.push(calculate_metrics(&neo4j_results, &gt_1, 1));
        }

        let _ = graph.run(query("MATCH (n:Vector) DETACH DELETE n")).await;

        Some((average_metrics(&k10_metrics), average_metrics(&k5_metrics), average_metrics(&k1_metrics)))
    })
}

#[cfg(not(feature = "neo4j"))]
fn neo4j_accuracy_test(_vectors: &[Vec<f32>], _queries: &[Vec<f32>]) -> Option<(AccuracyMetrics, AccuracyMetrics, AccuracyMetrics)> {
    None
}

fn average_metrics(metrics: &[AccuracyMetrics]) -> AccuracyMetrics {
    let n = metrics.len() as f64;
    AccuracyMetrics {
        recall_at_k: metrics.iter().map(|m| m.recall_at_k).sum::<f64>() / n,
        precision_at_k: metrics.iter().map(|m| m.precision_at_k).sum::<f64>() / n,
        mrr: metrics.iter().map(|m| m.mrr).sum::<f64>() / n,
        avg_distance_error: metrics.iter().map(|m| m.avg_distance_error).sum::<f64>() / n,
    }
}

fn print_metrics_table(name: &str, k10: &AccuracyMetrics, k5: &AccuracyMetrics, k1: &AccuracyMetrics) {
    println!("┌──────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ {:84} │", format!("{} Accuracy Results", name));
    println!("├───────────────┬──────────────┬──────────────┬──────────────┬───────────────────────┤");
    println!("│ Metric        │    K=10      │    K=5       │    K=1       │ Description           │");
    println!("├───────────────┼──────────────┼──────────────┼──────────────┼───────────────────────┤");
    println!("│ Recall@K      │    {:>6.1}%   │    {:>6.1}%   │    {:>6.1}%   │ True neighbors found  │", k10.recall_at_k * 100.0, k5.recall_at_k * 100.0, k1.recall_at_k * 100.0);
    println!("│ Precision@K   │    {:>6.1}%   │    {:>6.1}%   │    {:>6.1}%   │ Correct results ratio │", k10.precision_at_k * 100.0, k5.precision_at_k * 100.0, k1.precision_at_k * 100.0);
    println!("│ MRR           │    {:>6.3}    │    {:>6.3}    │    {:>6.3}    │ Mean Reciprocal Rank  │", k10.mrr, k5.mrr, k1.mrr);
    println!("│ Distance Err  │    {:>6.4}    │    {:>6.4}    │    {:>6.4}    │ Avg distance diff     │", k10.avg_distance_error, k5.avg_distance_error, k1.avg_distance_error);
    println!("└───────────────┴──────────────┴──────────────┴──────────────┴───────────────────────┘");
}

fn main() {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("              PardusDB vs Neo4j - Accuracy Benchmark                                   ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("Test Configuration:");
    println!("  • Vector dimension: {}", DIM);
    println!("  • Number of vectors: {}", NUM_VECTORS);
    println!("  • Number of queries: {}", NUM_QUERIES);
    println!("  • Ground truth: Brute-force exact search");
    println!();

    // Generate vectors
    let vectors = generate_vectors(NUM_VECTORS, DIM);

    // Use some of the stored vectors as queries (with slight perturbation) plus some random ones
    // This ensures queries are similar to stored vectors
    let mut queries = Vec::new();
    for i in 0..NUM_QUERIES {
        // Use every Nth vector as a query, with small perturbation
        let base_idx = (i * NUM_VECTORS / NUM_QUERIES) % NUM_VECTORS;
        let mut query = vectors[base_idx].clone();
        // Add small perturbation
        for val in &mut query {
            *val += (i as f32 * 0.0001) - 0.00005;
        }
        queries.push(query);
    }

    // Compute ground truth
    println!("[0/2] Computing brute-force ground truth...");
    let start = Instant::now();
    let _ground_truth: Vec<Vec<(usize, f32)>> = queries.iter()
        .map(|q| brute_force_search(&vectors, q, K))
        .collect();
    println!("  Ground truth computed in {:?}", start.elapsed());
    println!();

    // PardusDB accuracy
    println!("[1/2] Testing PardusDB accuracy...");
    let (pardus_k10, pardus_k5, pardus_k1) = pardusdb_accuracy_test(&vectors, &queries);
    println!();

    // Neo4j accuracy
    println!("[2/2] Testing Neo4j accuracy...");
    let neo4j_result = neo4j_accuracy_test(&vectors, &queries);
    println!();

    // Print results
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Accuracy Results                                         ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();

    print_metrics_table("PardusDB", &pardus_k10, &pardus_k5, &pardus_k1);

    if let Some((neo4j_k10, neo4j_k5, neo4j_k1)) = neo4j_result {
        println!();
        print_metrics_table("Neo4j", &neo4j_k10, &neo4j_k5, &neo4j_k1);

        // Comparison
        println!();
        println!("┌──────────────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Accuracy Comparison (PardusDB vs Neo4j)                                              │");
        println!("├───────────────┬────────────────────────┬────────────────────────┬───────────────────┤");
        println!("│ Metric        │ PardusDB               │ Neo4j                  │ Winner            │");
        println!("├───────────────┼────────────────────────┼────────────────────────┼───────────────────┤");

        let recall_winner = if pardus_k10.recall_at_k >= neo4j_k10.recall_at_k { "PardusDB" } else { "Neo4j" };
        let mrr_winner = if pardus_k10.mrr >= neo4j_k10.mrr { "PardusDB" } else { "Neo4j" };
        let err_winner = if pardus_k10.avg_distance_error <= neo4j_k10.avg_distance_error { "PardusDB" } else { "Neo4j" };

        println!("│ Recall@10     │ {:>20.1}%  │ {:>20.1}%  │ {:17} │", pardus_k10.recall_at_k * 100.0, neo4j_k10.recall_at_k * 100.0, recall_winner);
        println!("│ MRR           │ {:>20.3}  │ {:>20.3}  │ {:17} │", pardus_k10.mrr, neo4j_k10.mrr, mrr_winner);
        println!("│ Distance Err  │ {:>20.4}  │ {:>20.4}  │ {:17} │", pardus_k10.avg_distance_error, neo4j_k10.avg_distance_error, err_winner);
        println!("└───────────────┴────────────────────────┴────────────────────────┴───────────────────┘");
    } else {
        println!();
        println!("┌──────────────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Neo4j Not Available - Run with --features neo4j for comparison                      │");
        println!("└──────────────────────────────────────────────────────────────────────────────────────┘");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!("                              Benchmark Complete!                                      ");
    println!("═══════════════════════════════════════════════════════════════════════════════════════");
    println!();
}
