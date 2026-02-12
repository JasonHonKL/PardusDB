//! Benchmark test with random embeddings

use std::time::Instant;

fn main() {
    println!("=== PardusDB Benchmark (Random Embeddings) ===\n");

    const NUM_DOCS: usize = 1000;
    const DIM: usize = 768;

    // Generate random embeddings
    println!("Generating {} random embeddings (dim={})...", NUM_DOCS, DIM);
    let start = Instant::now();
    let embeddings: Vec<Vec<f32>> = (0..NUM_DOCS)
        .map(|i| {
            // Create semi-random vectors with some structure
            (0..DIM)
                .map(|j| ((i * DIM + j) as f32 / (NUM_DOCS * DIM) as f32 * 2.0 - 1.0))
                .collect()
        })
        .collect();
    println!("Generated in {:?}\n", start.elapsed());

    // Generate documents
    let documents: Vec<String> = (0..NUM_DOCS)
        .map(|i| format!("Document {}: This is test content for document number {}.", i, i))
        .collect();

    // Create in-memory database
    println!("Creating database and inserting documents...");
    let mut db = pardusdb::Database::in_memory();

    // Create table
    db.execute(
        &format!("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR({}), content TEXT, category TEXT);", DIM)
    ).unwrap();

    // Insert documents
    let start = Instant::now();
    for (i, (content, embedding)) in documents.iter().zip(embeddings.iter()).enumerate() {
        let vec_str: String = embedding.iter()
            .map(|f| format!("{:.6}", f))
            .collect::<Vec<_>>()
            .join(", ");

        let category = if i < NUM_DOCS / 3 { "A" }
                       else if i < 2 * NUM_DOCS / 3 { "B" }
                       else { "C" };

        let sql = format!(
            "INSERT INTO docs (embedding, content, category) VALUES ([{}], '{}', '{}');",
            vec_str, content.replace("'", "''"), category
        );
        db.execute(&sql).unwrap();

        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{} documents...", i + 1, NUM_DOCS);
        }
    }
    let insert_time = start.elapsed();
    println!("Inserted {} documents in {:?} ({:.2?}/doc)\n",
        NUM_DOCS, insert_time, insert_time / NUM_DOCS as u32);

    // Test queries
    println!("--- Testing Queries ---\n");

    // Query 1: Similarity search
    println!("Query 1: Find top 10 similar to document 0");
    let query_vec: String = embeddings[0].iter()
        .map(|f| format!("{:.6}", f))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT * FROM docs WHERE embedding SIMILARITY [{}] LIMIT 10;",
        query_vec
    );

    let start = Instant::now();
    let result = db.execute(&sql).unwrap();
    let query_time = start.elapsed();
    println!("Query time: {:?}", query_time);
    println!("Result preview (first 2):\n{}\n", format_result(&result, 2));

    // Query 2: Filtered search
    println!("Query 2: Find top 5 in category 'B'");
    let sql = "SELECT * FROM docs WHERE category = 'B' LIMIT 5;";
    let start = Instant::now();
    let result = db.execute(sql).unwrap();
    let query_time = start.elapsed();
    println!("Query time: {:?}", query_time);
    println!("Result: {}\n", result);

    // Query 3: Multiple random queries for average
    println!("Query 3: 100 random similarity queries");
    let mut total_time = std::time::Duration::ZERO;
    let query_indices: Vec<usize> = (0..100).map(|i| (i * 10) % NUM_DOCS).collect();

    for &i in &query_indices {
        let query_vec: String = embeddings[i].iter()
            .map(|f| format!("{:.6}", f))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT * FROM docs WHERE embedding SIMILARITY [{}] LIMIT 10;",
            query_vec
        );
        let start = Instant::now();
        db.execute(&sql).unwrap();
        total_time += start.elapsed();
    }
    println!("Total time for 100 queries: {:?}", total_time);
    println!("Average query time: {:?}\n", total_time / 100);

    // Query 4: Update test
    println!("Query 4: Update 100 documents");
    let start = Instant::now();
    for i in (0..100).map(|i| i * 10 + 1) {
        let sql = format!("UPDATE docs SET category = 'X' WHERE id = {};", i);
        db.execute(&sql).unwrap();
    }
    let update_time = start.elapsed();
    println!("Updated 100 documents in {:?} ({:.2?}/update)\n", update_time, update_time / 100);

    // Query 5: Delete test
    println!("Query 5: Delete 50 documents");
    let start = Instant::now();
    for i in (0..50).map(|i| i * 20 + 1) {
        let sql = format!("DELETE FROM docs WHERE id = {};", i);
        db.execute(&sql).unwrap();
    }
    let delete_time = start.elapsed();
    println!("Deleted 50 documents in {:?} ({:.2?}/delete)\n", delete_time, delete_time / 50);

    // Show final state
    println!("Final state:");
    println!("{}", db.execute("SHOW TABLES;").unwrap());

    // Summary
    println!("\n=== Summary ===");
    println!("Documents: {}", NUM_DOCS);
    println!("Embedding dimension: {}", DIM);
    println!("Insert time: {:?} ({:.2?}/doc)", insert_time, insert_time / NUM_DOCS as u32);
    println!("Avg similarity query: {:?}", total_time / 100);
    println!("Update: {:.2?}/doc", update_time / 100);
    println!("Delete: {:.2?}/doc", delete_time / 50);
}

fn format_result(result: &pardusdb::ExecuteResult, max_lines: usize) -> String {
    let s = format!("{}", result);
    let lines: Vec<&str> = s.lines().take(max_lines + 1).collect();
    lines.join("\n")
}
