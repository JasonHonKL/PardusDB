//! Profile where the time is spent

use std::time::Instant;

fn main() {
    println!("=== Profiling Insert Performance ===\n");

    const NUM_DOCS: usize = 100;
    const DIM: usize = 768;

    // Generate test data
    let embeddings: Vec<Vec<f32>> = (0..NUM_DOCS)
        .map(|i| (0..DIM).map(|j| (i * DIM + j) as f32).collect())
        .collect();

    let documents: Vec<String> = (0..NUM_DOCS)
        .map(|i| format!("Document {}", i))
        .collect();

    // Test 1: SQL parsing overhead
    println!("Test 1: SQL String Formatting + Parsing");
    let start = Instant::now();
    for (i, (content, embedding)) in documents.iter().zip(embeddings.iter()).enumerate() {
        // This is what the benchmark does
        let vec_str: String = embedding.iter()
            .map(|f| format!("{:.6}", f))
            .collect::<Vec<_>>()
            .join(", ");

        let _sql = format!(
            "INSERT INTO docs (embedding, content) VALUES ([{}], '{}');",
            vec_str, content
        );
        // Note: Not even executing, just formatting!
    }
    let format_time = start.elapsed();
    println!("  String formatting only: {:?}", format_time);
    println!("  Per doc: {:?}", format_time / NUM_DOCS as u32);

    // Test 2: Direct API (bypass SQL)
    println!("\nTest 2: Direct API (no SQL parsing)");
    let mut db = pardusdb::Database::in_memory();
    db.execute(&format!("CREATE TABLE docs (id INTEGER, embedding VECTOR({}), content TEXT);", DIM)).unwrap();

    let start = Instant::now();
    for (i, (content, embedding)) in documents.iter().zip(embeddings.iter()).enumerate() {
        // Direct insert bypassing SQL
        let values = vec![
            pardusdb::Value::Integer(i as i64),
            pardusdb::Value::Vector(embedding.clone()),
            pardusdb::Value::Text(content.clone()),
        ];
        // We need to add a direct API...
    }
    println!("  (Need to add direct API first)");

    // Test 3: Graph insert only
    println!("\nTest 3: Graph Insert Only (no SQL, no table overhead)");
    let mut graph: pardusdb::Graph<f32, pardusdb::Euclidean> = pardusdb::Graph::new(DIM, pardusdb::GraphConfig::default());

    let start = Instant::now();
    for embedding in &embeddings {
        graph.insert(embedding.clone());
    }
    let graph_time = start.elapsed();
    println!("  Graph insert: {:?}", graph_time);
    println!("  Per doc: {:?}", graph_time / NUM_DOCS as u32);

    // Test 4: Vector cloning overhead
    println!("\nTest 4: Vector Cloning Overhead");
    let start = Instant::now();
    for embedding in &embeddings {
        let _cloned = embedding.clone();
    }
    let clone_time = start.elapsed();
    println!("  Just cloning vectors: {:?}", clone_time);
    println!("  Per doc: {:?}", clone_time / NUM_DOCS as u32);

    // Test 5: Full SQL path
    println!("\nTest 5: Full SQL Path (what benchmark does)");
    let mut db = pardusdb::Database::in_memory();
    db.execute(&format!("CREATE TABLE docs (embedding VECTOR({}), content TEXT);", DIM)).unwrap();

    let start = Instant::now();
    for (content, embedding) in documents.iter().zip(embeddings.iter()) {
        let vec_str: String = embedding.iter()
            .map(|f| format!("{:.6}", f))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "INSERT INTO docs (embedding, content) VALUES ([{}], '{}');",
            vec_str, content
        );
        db.execute(&sql).unwrap();
    }
    let sql_time = start.elapsed();
    println!("  Full SQL path: {:?}", sql_time);
    println!("  Per doc: {:?}", sql_time / NUM_DOCS as u32);

    // Summary
    println!("\n=== Summary ===");
    println!("String formatting: {:?}", format_time / NUM_DOCS as u32);
    println!("Graph only:        {:?}", graph_time / NUM_DOCS as u32);
    println!("Full SQL:          {:?}", sql_time / NUM_DOCS as u32);
    println!("\nOverhead breakdown:");
    println!("  SQL overhead: ~{:?}", (sql_time - graph_time) / NUM_DOCS as u32);
}
