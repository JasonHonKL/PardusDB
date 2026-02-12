//! Simple RAG Example with PardusDB
//!
//! This example demonstrates a basic RAG workflow using PardusDB:
//! 1. Create an in-memory database
//! 2. Store documents with embeddings
//! 3. Query similar documents
//!
//! Run: cargo run --example simple_rag --release

use pardusdb::{ConcurrentDatabase, Connection, Value};

/// Sample documents for the RAG example
const DOCUMENTS: &[&str] = &[
    "Rust is a systems programming language focused on safety, speed, and concurrency.",
    "PardusDB is a SQLite-like vector database written in Rust for similarity search.",
    "Vector databases store embeddings which are numerical representations of data.",
    "RAG combines retrieval of relevant documents with language model generation.",
    "Embeddings capture semantic meaning, allowing similarity-based search.",
    "Graph-based search uses connections between vectors to find nearest neighbors.",
    "SQLite is an embedded database that stores everything in a single file.",
    "HNSW is a graph-based algorithm for approximate nearest neighbor search.",
];

/// Simple embedding function that creates deterministic embeddings from text.
/// In a real application, you would use a proper embedding model like:
/// - OpenAI text-embedding-ada-002
/// - Ollama with nomic-embed-text
/// - sentence-transformers
fn create_embedding(text: &str, dim: usize) -> Vec<f32> {
    // This is a simple hash-based embedding for demonstration.
    // NOT suitable for production - use a real embedding model!
    let bytes = text.as_bytes();
    let mut embedding = vec![0.0f32; dim];

    for (i, byte) in bytes.iter().cycle().take(dim * 4).enumerate() {
        let idx = i % dim;
        let shift = (i / dim) as u32 % 8;
        let contribution = (*byte as f32 / 255.0) / (1 << shift) as f32;
        embedding[idx] += contribution;
    }

    // Normalize the embedding
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    embedding
}

fn main() {
    println!("============================================================");
    println!("  PardusDB Simple RAG Example");
    println!("============================================================");
    println!();

    // Configuration
    const EMBEDDING_DIM: usize = 128;

    // Create in-memory database
    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    println!("Creating PardusDB in-memory database...");

    // Create table
    conn.execute(&format!(
        "CREATE TABLE documents (embedding VECTOR({}), content TEXT);",
        EMBEDDING_DIM
    )).expect("Failed to create table");

    println!("Created table 'documents' with {}-dimensional vectors\n", EMBEDDING_DIM);

    // Insert documents
    println!("=== Ingesting Documents ===");

    let mut doc_embeddings: Vec<(&str, Vec<f32>)> = Vec::new();

    for (i, doc) in DOCUMENTS.iter().enumerate() {
        let embedding = create_embedding(doc, EMBEDDING_DIM);
        doc_embeddings.push((*doc, embedding.clone()));

        conn.insert_direct(
            "documents",
            embedding,
            vec![("content", Value::Text(doc.to_string()))],
        ).expect("Failed to insert document");

        println!("  [{}/{}] Inserted: {}...", i + 1, DOCUMENTS.len(), &doc[..50.min(doc.len())]);
    }

    println!("\nDocuments stored: {}\n", DOCUMENTS.len());

    // Query examples
    println!("=== RAG Query Examples ===\n");

    let queries = [
        "What is a vector database?",
        "How does RAG work?",
        "Tell me about Rust programming language",
    ];

    for query in queries.iter() {
        println!("Query: \"{}\"", query);

        let query_embedding = create_embedding(query, EMBEDDING_DIM);

        // Search using PardusDB
        let results = conn.search_similar("documents", &query_embedding, 3, 100)
            .expect("Failed to search");

        println!("\nTop 3 similar documents (from PardusDB):");
        for (i, (row_id, values, distance)) in results.iter().enumerate() {
            // Extract content from values
            let content = values.iter()
                .find_map(|v| if let Value::Text(s) = v { Some(s.as_str()) } else { None })
                .unwrap_or("Unknown");

            let similarity = 1.0 - distance;
            println!("  [{}] Row ID: {}, Distance: {:.4} (similarity: {:.4})", i + 1, row_id, distance, similarity);
            println!("      \"{}\"", if content.len() > 70 { &content[..70] } else { content });
        }
        println!();
    }

    // Demonstrate batch insert
    println!("=== Batch Insert Demo ===\n");

    // Create a new table for batch demo
    conn.execute(&format!(
        "CREATE TABLE batch_docs (embedding VECTOR({}), id INTEGER);",
        EMBEDDING_DIM
    )).expect("Failed to create batch table");

    // Prepare batch data
    let batch_vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| create_embedding(&format!("Document {}", i), EMBEDDING_DIM))
        .collect();

    let batch_metadata: Vec<Vec<(&str, Value)>> = (0..100)
        .map(|i| vec![("id", Value::Integer(i as i64))])
        .collect();

    // Batch insert
    let start = std::time::Instant::now();
    conn.insert_batch_direct("batch_docs", batch_vectors, batch_metadata)
        .expect("Failed to batch insert");
    let elapsed = start.elapsed();

    println!("Inserted 100 documents in {:?}", elapsed);
    println!("Throughput: {:.0} docs/sec", 100.0 / elapsed.as_secs_f64());

    // Summary
    println!("\n============================================================");
    println!("  Summary");
    println!("============================================================");
    println!("Database: In-memory");
    println!("Documents in 'documents' table: {}", DOCUMENTS.len());
    println!("Documents in 'batch_docs' table: 100");
    println!("Embedding dimension: {}", EMBEDDING_DIM);
    println!();
    println!("Key features demonstrated:");
    println!("  - CREATE TABLE with VECTOR column");
    println!("  - Individual inserts with insert_direct()");
    println!("  - Batch inserts with insert_batch_direct()");
    println!("  - Similarity search with search_similar()");
    println!("  - High-performance batch operations");
}
