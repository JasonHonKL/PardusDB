//! Benchmark test with Ollama embeddings

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() {
    println!("=== PardusDB + Ollama Embedding Benchmark ===\n");

    // Check if ollama is available
    let ollama_check = Command::new("ollama").arg("list").output();
    if ollama_check.is_err() {
        println!("Error: Ollama not found. Please install Ollama first.");
        return;
    }

    // Generate test documents
    println!("Generating 1000 test documents...");
    let documents = generate_documents(1000);
    println!("Generated {} documents.\n", documents.len());

    // Get embeddings from Ollama
    println!("Getting embeddings from Ollama (embeddinggemma)...");
    let start = Instant::now();
    let embeddings = get_embeddings_batch(&documents);
    let embedding_time = start.elapsed();
    println!("Got {} embeddings in {:?}\n", embeddings.len(), embedding_time);

    // Create in-memory database
    println!("Creating database and inserting documents...");
    let mut db = pardusdb::Database::in_memory();

    // Create table
    db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(768), content TEXT, category TEXT);"
    ).unwrap();

    // Insert documents
    let start = Instant::now();
    for (i, (content, embedding)) in documents.iter().zip(embeddings.iter()).enumerate() {
        let vec_str: String = embedding.iter()
            .map(|f| format!("{:.6}", f))
            .collect::<Vec<_>>()
            .join(", ");

        let category = if i < 333 { "A" } else if i < 666 { "B" } else { "C" };

        let sql = format!(
            "INSERT INTO docs (embedding, content, category) VALUES ([{}], '{}', '{}');",
            vec_str, content.replace("'", "''"), category
        );
        db.execute(&sql).unwrap();

        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/1000 documents...", i + 1);
        }
    }
    let insert_time = start.elapsed();
    println!("Inserted 1000 documents in {:?}\n", insert_time);

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
    println!("Result: {}", result);
    println!("Query time: {:?}\n", query_time);

    // Query 2: Filtered search
    println!("Query 2: Find top 5 in category 'B'");
    let sql = "SELECT * FROM docs WHERE category = 'B' LIMIT 5;";
    let start = Instant::now();
    let result = db.execute(sql).unwrap();
    let query_time = start.elapsed();
    println!("Result: {}", result);
    println!("Query time: {:?}\n", query_time);

    // Query 3: Multiple random queries
    println!("Query 3: 10 random similarity queries");
    let mut total_time = std::time::Duration::ZERO;
    for i in [100, 250, 500, 750, 900, 50, 150, 350, 600, 850] {
        let query_vec: String = embeddings[i].iter()
            .map(|f| format!("{:.6}", f))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT * FROM docs WHERE embedding SIMILARITY [{}] LIMIT 5;",
            query_vec
        );
        let start = Instant::now();
        db.execute(&sql).unwrap();
        total_time += start.elapsed();
    }
    println!("Average query time: {:?}", total_time / 10);

    // Summary
    println!("\n=== Summary ===");
    println!("Documents: 1000");
    println!("Embedding dimension: 768");
    println!("Embedding time: {:?}", embedding_time);
    println!("Insert time: {:?}", insert_time);
    println!("Avg query time: {:?}", total_time / 10);
}

/// Generate random test documents
fn generate_documents(count: usize) -> Vec<String> {
    let topics = [
        "machine learning", "artificial intelligence", "deep learning",
        "neural networks", "natural language processing", "computer vision",
        "data science", "statistics", "algorithms", "programming",
        "software engineering", "databases", "web development", "cloud computing",
        "cybersecurity", "blockchain", "robotics", "autonomous vehicles",
        "quantum computing", "bioinformatics"
    ];

    let templates = [
        "An introduction to {}: fundamentals and applications",
        "Advanced techniques in {} for modern systems",
        "The future of {}: trends and predictions",
        "Practical guide to implementing {} solutions",
        "Research advances in {} methodology",
        "Case studies in {} for enterprise applications",
        "Best practices for {} development",
        "{} optimization and performance tuning",
        "Scaling {} systems for production",
        "Security considerations in {} architecture"
    ];

    let mut docs = Vec::with_capacity(count);
    for i in 0..count {
        let topic = topics[i % topics.len()];
        let template = templates[(i / topics.len()) % templates.len()];
        let doc = format!("Document {}: {}", i + 1, template.replace("{}", topic));
        docs.push(doc);
    }
    docs
}

/// Get embeddings from Ollama
fn get_embeddings_batch(documents: &[String]) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::with_capacity(documents.len());

    for (i, doc) in documents.iter().enumerate() {
        if i % 100 == 0 {
            print!("\r  Embedding document {}/{}...", i + 1, documents.len());
            std::io::stdout().flush().unwrap();
        }

        let embedding = get_single_embedding(doc);
        embeddings.push(embedding);
    }
    println!();

    embeddings
}

/// Get a single embedding from Ollama
fn get_single_embedding(text: &str) -> Vec<f32> {
    let output = Command::new("ollama")
        .args(["run", "embeddinggemma"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .expect("Failed to run ollama");

    // The output contains the embedding as JSON array
    // Parse it - the embeddinggemma model outputs just the vector
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Try to parse the output as a JSON array
    // The format might vary, so we handle different cases
    parse_embedding_output(&stdout).unwrap_or_else(|| {
        // Fallback: generate a dummy embedding for testing
        println!("Warning: Could not parse embedding, using dummy");
        vec![0.0; 768]
    })
}

fn parse_embedding_output(output: &str) -> Option<Vec<f32>> {
    // Try to find a JSON array in the output
    let output = output.trim();

    // If it starts with '[', try to parse as JSON array
    if output.starts_with('[') {
        // Find the end of the array
        if let Some(end) = output.find(']') {
            let array_str = &output[..=end];
            // Parse the array
            let values: Vec<f32> = array_str
                .trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if !values.is_empty() {
                return Some(values);
            }
        }
    }

    None
}
