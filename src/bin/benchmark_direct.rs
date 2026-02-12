//! Benchmark with direct API (no SQL overhead)

use std::time::Instant;

fn main() {
    println!("=== PardusDB Direct API Benchmark ===\n");

    const NUM_DOCS: usize = 1000;
    const DIM: usize = 768;

    // Generate random embeddings
    println!("Generating {} embeddings (dim={})...", NUM_DOCS, DIM);
    let start = Instant::now();
    let embeddings: Vec<Vec<f32>> = (0..NUM_DOCS)
        .map(|i| (0..DIM).map(|j| ((i * DIM + j) as f32 / (NUM_DOCS * DIM) as f32)).collect())
        .collect();
    println!("Generated in {:?}\n", start.elapsed());

    // Create database
    let mut db = pardusdb::Database::in_memory();
    db.execute(&format!("CREATE TABLE docs (embedding VECTOR({}), content TEXT, category TEXT);", DIM)).unwrap();

    // Insert using direct API
    println!("Inserting {} documents (direct API)...", NUM_DOCS);
    let start = Instant::now();
    for (i, embedding) in embeddings.iter().enumerate() {
        let category = if i < NUM_DOCS / 3 { "A" }
                       else if i < 2 * NUM_DOCS / 3 { "B" }
                       else { "C" };

        db.insert_direct(
            "docs",
            embedding.clone(),
            vec![
                ("content", pardusdb::Value::Text(format!("Document {}", i))),
                ("category", pardusdb::Value::Text(category.to_string())),
            ]
        ).unwrap();

        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{}...", i + 1, NUM_DOCS);
        }
    }
    let insert_time = start.elapsed();
    println!("Inserted in {:?} ({:.2?}/doc)\n", insert_time, insert_time / NUM_DOCS as u32);

    // Query using direct API
    println!("Querying (direct API)...");
    let mut total_query = std::time::Duration::ZERO;

    for i in (0..100).map(|i| i * 10) {
        let start = Instant::now();
        let results = db.search_similar("docs", &embeddings[i], 10, 100).unwrap();
        total_query += start.elapsed();
    }

    println!("100 queries in {:?}", total_query);
    println!("Avg query: {:?}\n", total_query / 100);

    // Compare with SQL path
    println!("--- Comparison (SQL vs Direct) ---");

    // Single insert comparison
    let vec = embeddings[0].clone();

    // SQL path
    let vec_str: String = vec.iter().map(|f| format!("{:.6}", f)).collect::<Vec<_>>().join(", ");
    let sql = format!("INSERT INTO docs (embedding, content) VALUES ([{}], 'test');", vec_str);
    let start = Instant::now();
    db.execute(&sql).unwrap();
    let sql_time = start.elapsed();

    // Direct path
    let start = Instant::now();
    db.insert_direct("docs", vec.clone(), vec![("content", pardusdb::Value::Text("test".into()))]).unwrap();
    let direct_time = start.elapsed();

    println!("SQL insert:    {:?}", sql_time);
    println!("Direct insert: {:?}", direct_time);
    println!("Speedup:       {:.1}x", sql_time.as_secs_f64() / direct_time.as_secs_f64());

    // Summary
    println!("\n=== Summary ===");
    println!("Documents: {}", NUM_DOCS);
    println!("Dimension: {}", DIM);
    println!("Insert (direct): {:?}", insert_time / NUM_DOCS as u32);
    println!("Query (direct):  {:?}", total_query / 100);
}
