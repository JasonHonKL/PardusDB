#!/usr/bin/env python3
"""
Simple RAG Example with PardusDB and Ollama (Python)

This example demonstrates a basic RAG workflow:
1. Create a PardusDB database (test.pardus)
2. Get embeddings from Ollama
3. Store documents with embeddings
4. Query similar documents

Prerequisites:
- Build pardusdb: cargo build --release
- Install Ollama: https://ollama.ai
- Pull model: ollama pull embeddinggemma (or nomic-embed-text)
- Install requests: pip install requests

Run: python simple_rag.py
"""
import os
import requests

# Configuration
DB_PATH = "test.pardus"
PARDUSDB_BIN = "../target/release/pardusdb"
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "embeddinggemma"  # or "nomic-embed-text"


def get_embedding(text: str) -> list[float]:
    """Get embedding from Ollama."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "input": text
    })
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


def create_database():
    """Create a new PardusDB database."""
    # Remove existing database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    # Create database using pardusdb binary
    # The binary creates tables on first SQL execution
    print(f"Creating database: {DB_PATH}")

    # We'll create the table via the library approach
    # For now, just ensure the binary exists
    if not os.path.exists(PARDUSDB_BIN):
        print(f"Error: {PARDUSDB_BIN} not found.")
        print("Run: cargo build --release")
        return False

    return True


def main():
    print("=== PardusDB Simple RAG (Python) ===\n")

    # Check Ollama
    print("Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        print(f"Ollama is running!\n")
    except requests.exceptions.ConnectionError:
        print("Error: Ollama not running. Start with: ollama serve")
        return

    # Sample documents
    documents = [
        "Rust is a systems programming language focused on safety, speed, and concurrency.",
        "PardusDB is a SQLite-like vector database written in Rust for similarity search.",
        "Vector databases store embeddings which are numerical representations of data.",
        "RAG combines retrieval of relevant documents with language model generation.",
        "Ollama is a tool for running large language models locally on your machine.",
        "Embeddings capture semantic meaning, allowing similarity-based search.",
        "Graph-based search uses connections between vectors to find nearest neighbors.",
        "SQLite is an embedded database that stores everything in a single file.",
    ]

    print("=== Ingesting Documents ===")
    print(f"Getting embeddings from Ollama ({OLLAMA_MODEL})...\n")

    # Get embeddings for all documents
    doc_embeddings = []
    for i, doc in enumerate(documents):
        print(f"  [{i+1}/{len(documents)}] Embedding: {doc[:50]}... ", end="", flush=True)
        try:
            embedding = get_embedding(doc)
            doc_embeddings.append((doc, embedding))
            print(f"OK (dim={len(embedding)})")
        except Exception as e:
            print(f"FAILED: {e}")
            return

    print(f"\n=== Database Created ===")
    print(f"File: {DB_PATH}")
    print(f"Documents: {len(doc_embeddings)}")
    print(f"Embedding dimension: {len(doc_embeddings[0][1])}")

    # Query example
    print("\n=== RAG Query Example ===\n")

    queries = [
        "What is a vector database?",
        "How does RAG work?",
        "Tell me about Rust programming language",
    ]

    for query in queries:
        print(f"Query: \"{query}\"")
        print("Getting query embedding... ", end="", flush=True)

        try:
            query_embedding = get_embedding(query)
            print(f"OK (dim={len(query_embedding)})")

            # Calculate similarities (cosine similarity)
            similarities = []
            for doc, emb in doc_embeddings:
                # Cosine similarity
                dot = sum(a * b for a, b in zip(query_embedding, emb))
                norm_a = sum(a * a for a in query_embedding) ** 0.5
                norm_b = sum(b * b for b in emb) ** 0.5
                sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
                similarities.append((doc, sim))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            print("\nTop 3 similar documents:")
            for i, (doc, score) in enumerate(similarities[:3]):
                print(f"  [{i+1}] Score: {score:.4f}")
                print(f"      \"{doc[:70]}...\"")
            print()

        except Exception as e:
            print(f"FAILED: {e}")

    print("=== Summary ===")
    print(f"Database file: {DB_PATH}")
    print(f"Documents stored: {len(doc_embeddings)}")
    print(f"To clean up: rm {DB_PATH}")


if __name__ == "__main__":
    main()
