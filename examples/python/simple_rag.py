#!/usr/bin/env python3
"""
Simple RAG Example with PardusDB and Ollama (Python)

This example demonstrates a basic RAG workflow:
1. Create a PardusDB database
2. Get embeddings from Ollama
3. Store documents with embeddings in PardusDB
4. Query similar documents using PardusDB's vector search

Prerequisites:
- Build pardusdb: cargo build --release
- Install Ollama: https://ollama.ai
- Pull model: ollama pull nomic-embed-text
- Install requests: pip install requests

Run: python simple_rag.py
"""
import os
import json
import subprocess
import tempfile
import requests
from typing import List, Tuple

# Configuration
DB_PATH = "rag_db.pardus"
PARDUSDB_BIN = os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "pardusdb")
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "nomic-embed-text"  # or "embeddinggemma"


def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "input": text
    })
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


class PardusDB:
    """Simple wrapper for PardusDB using subprocess."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # Remove existing database
        if os.path.exists(db_path):
            os.remove(db_path)

    def execute(self, sql: str) -> str:
        """Execute SQL command and return output."""
        # Create a temporary file with the SQL command
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
            f.write(sql)
            f.write("\n")
            temp_path = f.name

        try:
            # Run pardusdb with the SQL file
            result = subprocess.run(
                [PARDUSDB_BIN, self.db_path],
                input=sql + "\n",
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def create_table(self, table_name: str, embedding_dim: int):
        """Create a table with vector column."""
        sql = f"CREATE TABLE {table_name} (embedding VECTOR({embedding_dim}), content TEXT);"
        return self.execute(sql)

    def insert(self, table_name: str, embedding: List[float], content: str):
        """Insert a document with embedding."""
        # Format embedding as JSON array
        emb_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
        # Escape single quotes in content
        content_escaped = content.replace("'", "''")
        sql = f"INSERT INTO {table_name} (embedding, content) VALUES ({emb_str}, '{content_escaped}');"
        return self.execute(sql)

    def search_similar(self, table_name: str, query_embedding: List[float], k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar documents."""
        # Format embedding as JSON array
        emb_str = "[" + ",".join(f"{x:.6f}" for x in query_embedding) + "]"
        sql = f"SELECT * FROM {table_name} WHERE embedding SIMILARITY {emb_str} LIMIT {k};"
        output = self.execute(sql)

        # Parse output to extract results
        results = []
        lines = output.strip().split('\n')
        for line in lines:
            if 'distance=' in line and 'content=' in line:
                # Parse the line to extract content and distance
                # Format: id=X, distance=Y.YYYY, values=[Vector([...]), Text("...")]
                try:
                    # Extract distance
                    dist_start = line.find('distance=') + 9
                    dist_end = line.find(',', dist_start)
                    distance = float(line[dist_start:dist_end])

                    # Extract content from Text("...")
                    text_start = line.find('Text("') + 6
                    text_end = line.rfind('")')
                    content = line[text_start:text_end]

                    results.append((content, distance))
                except:
                    pass

        return results


def main():
    print("=" * 60)
    print("  PardusDB Simple RAG Example with Ollama")
    print("=" * 60)
    print()

    # Check pardusdb binary
    if not os.path.exists(PARDUSDB_BIN):
        print(f"Error: PardusDB binary not found at {PARDUSDB_BIN}")
        print("Build it with: cargo build --release")
        return

    print(f"PardusDB binary: {PARDUSDB_BIN}")

    # Check Ollama
    print("Checking Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"Ollama is running!\n")
    except requests.exceptions.ConnectionError:
        print("Error: Ollama not running. Start with: ollama serve")
        return
    except requests.exceptions.Timeout:
        print("Error: Ollama connection timed out")
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

    # Get embedding dimension from first document
    print(f"Getting embeddings from Ollama (model: {OLLAMA_MODEL})...\n")
    sample_embedding = get_embedding("test")
    embedding_dim = len(sample_embedding)
    print(f"Embedding dimension: {embedding_dim}\n")

    # Create database and table
    print("Creating PardusDB database...")
    db = PardusDB(DB_PATH)
    result = db.create_table("documents", embedding_dim)
    print(f"Created table 'documents'\n")

    # Insert documents
    print("=== Ingesting Documents ===")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}/{len(documents)}] Embedding: {doc[:40]}... ", end="", flush=True)
        try:
            embedding = get_embedding(doc)
            db.insert("documents", embedding, doc)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            return

    print(f"\nDatabase: {DB_PATH}")
    print(f"Documents stored: {len(documents)}\n")

    # Query examples
    print("=== RAG Query Examples ===\n")

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

            # Search using PardusDB
            results = db.search_similar("documents", query_embedding, k=3)

            print("\nTop 3 similar documents (from PardusDB):")
            for i, (doc, distance) in enumerate(results):
                similarity = 1 - distance  # Convert distance to similarity
                print(f"  [{i+1}] Distance: {distance:.4f} (similarity: {similarity:.4f})")
                print(f"      \"{doc[:70]}...\"" if len(doc) > 70 else f"      \"{doc}\"")
            print()

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"Database file: {DB_PATH}")
    print(f"Documents stored: {len(documents)}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"To clean up: rm {DB_PATH}")
    print()
    print("This example used PardusDB's built-in vector similarity search")
    print("instead of computing similarities manually in Python!")


if __name__ == "__main__":
    main()
