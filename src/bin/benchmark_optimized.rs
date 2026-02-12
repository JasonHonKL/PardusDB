//! Benchmark comparing SQL parsing vs direct API vs batch operations

use std::time::Instant;

fn main() {
    println!("=== PardusDB Optimized API Benchmark ===\n");

    const NUM_DOCS: usize = 500;
    const DIM: usize = 768;

    // Generate random embeddings
    println!("Generating {} embeddings (dim={})...", NUM_DOCS, DIM);
    let start = Instant::now();
    let embeddings: Vec<Vec<f32>> = (0..NUM_DOCS)
        .map(|i| (0..DIM).map(|j| ((i * DIM + j) as f32 / (NUM_DOCS * DIM) as f32)).collect())
        .collect();
    println!("Generated in {:?}\n", start.elapsed());

    // Test 1: SQL parsing every time (slowest)
    println!("--- Test 1: SQL Parsing (baseline) ---");
    let mut db = pardusdb::Database::in_memory();
    db.execute(&format!("CREATE TABLE docs (embedding VECTOR({}), content TEXT);", DIM)).unwrap();

    let start = Instant::now();
    for (i, embedding) in embeddings.iter().enumerate() {
        let vec_str: String = embedding.iter().map(|f| format!("{:.6}", f)).collect::<Vec<_>>().join(", ");
        let sql = format!("INSERT INTO docs (embedding, content) VALUES ([{}], 'doc{}');", vec_str, i);
        db.execute(&sql).unwrap();
    }
    let sql_time = start.elapsed();
    println!("SQL insert: {:?} ({:.2?}/doc)\n", sql_time, sql_time / NUM_DOCS as u32);

    // Test 2: Direct API (faster)
    println!("--- Test 2: Direct API ---");
    let mut db = pardusdb::Database::in_memory();
    db.execute(&format!("CREATE TABLE docs2 (embedding VECTOR({}), content TEXT);", DIM)).unwrap();

    let start = Instant::now();
    for (i, embedding) in embeddings.iter().enumerate() {
        db.insert_direct(
            "docs2",
            embedding.clone(),
            vec![("content", pardusdb::Value::Text(format!("doc{}", i)))]
        ).unwrap();
    }
    let direct_time = start.elapsed();
    println!("Direct insert: {:?} ({:.2?}/doc)", direct_time, direct_time / NUM_DOCS as u32);
    println!("Speedup vs SQL: {:.1}x\n", sql_time.as_secs_f64() / direct_time.as_secs_f64());

    // Test 3: Batch insert (fastest for bulk operations)
    println!("--- Test 3: Batch Insert ---");
    let mut db = pardusdb::Database::in_memory();
    db.execute(&format!("CREATE TABLE docs3 (embedding VECTOR({}), content TEXT);", DIM)).unwrap();

    let start = Instant::now();
    {
        let mut batch = pardusdb::BatchInserter::new(&mut db, "docs3", &["embedding", "content"]);
        for (i, embedding) in embeddings.iter().enumerate() {
            batch.insert(vec![
                pardusdb::Value::Vector(embedding.clone()),
                pardusdb::Value::Text(format!("doc{}", i)),
            ]).unwrap();
        }
    }
    let batch_time = start.elapsed();
    println!("Batch insert: {:?} ({:.2?}/doc)", batch_time, batch_time / NUM_DOCS as u32);
    println!("Speedup vs SQL: {:.1}x", sql_time.as_secs_f64() / batch_time.as_secs_f64());
    println!("Speedup vs Direct: {:.1}x\n", direct_time.as_secs_f64() / batch_time.as_secs_f64());

    // Query benchmarks
    println!("--- Query Performance ---");

    // Warm up
    let _ = db.search_similar("docs3", &embeddings[0], 10, 100).unwrap();

    let mut total_query = std::time::Duration::ZERO;
    for i in 0..50 {
        let start = Instant::now();
        let results = db.search_similar("docs3", &embeddings[i], 10, 100).unwrap();
        total_query += start.elapsed();
    }

    println!("50 queries in {:?}", total_query);
    println!("Avg query: {:?}\n", total_query / 50);

    // Summary
    println!("=== Summary ===");
    println!("Documents: {}", NUM_DOCS);
    println!("Dimension: {}", DIM);
    println!("SQL insert:    {:?}/doc", sql_time / NUM_DOCS as u32);
    println!("Direct insert: {:?}/doc ({:.1}x faster than SQL)", direct_time / NUM_DOCS as u32, sql_time.as_secs_f64() / direct_time.as_secs_f64());
    println!("Batch insert:  {:?}/doc ({:.1}x faster than SQL)", batch_time / NUM_DOCS as u32, sql_time.as_secs_f64() / batch_time.as_secs_f64());
    println!("Query:         {:?}", total_query / 50);
}
