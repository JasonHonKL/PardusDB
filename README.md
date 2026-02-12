# PardusDB

**A fast, SQLite-like embedded vector database with graph-based approximate nearest neighbor search**  
**Open-source project from the team behind [Pardus AI](https://pardusai.org/)**

PardusDB is designed for developers building local AI applications — RAG pipelines, semantic search, recommendation systems, or any project that needs lightweight, persistent vector storage without external dependencies.

While [Pardus AI](https://pardusai.org/) gives non-technical users a powerful no-code platform to ask questions of their CSV, JSON, and PDF data in plain English, PardusDB gives developers the same speed and privacy in an embeddable, fully open-source vector database.

## Features

- **Single-file storage** — Everything lives in one `.pardus` file, just like SQLite
- **Multiple tables** — Store different vector dimensions and metadata in the same database
- **Familiar SQL-like syntax** — CREATE, INSERT, SELECT, UPDATE, DELETE feel natural
- **Fast vector similarity search** — Graph-based approximate nearest neighbor search
- **Thread-safe** — Safe concurrent reads in multi-threaded applications
- **Full transactions** — BEGIN/COMMIT/ROLLBACK for atomic operations
- **Optional GPU acceleration** — For large batch inserts and queries
- **Zero external dependencies** — Pure Rust, MIT licensed

## Installation

```bash
git clone https://github.com/pardus-ai/pardusdb
cd pardusdb
cargo build --release
```

The binary will be at `target/release/pardusdb`.

## Quick Start

### Interactive REPL

```bash
./target/release/pardusdb
```

```
╔═══════════════════════════════════════════════════════════════╗
║                    PardusDB REPL                      ║
║          Vector Database with SQL Interface           ║
╚═══════════════════════════════════════════════════════════════╝

pardusdb [memory]> .create mydb.pardus
Created and opened: mydb.pardus

pardusdb [mydb.pardus]> CREATE TABLE docs (embedding VECTOR(768), content TEXT);
Table 'docs' created

pardusdb [mydb.pardus]> INSERT INTO docs (embedding, content) 
VALUES ([0.1, 0.2, 0.3, ...], 'Hello World');
Inserted row with id=1

pardusdb [mydb.pardus]> SELECT * FROM docs 
WHERE embedding SIMILARITY [0.1, 0.2, 0.3, ...] LIMIT 5;

Found 1 similar rows:
  id=1, distance=0.0000, values=[Vector([...]), Text("Hello World")]

pardusdb [mydb.pardus]> quit
Saved to: mydb.pardus
Goodbye!
```

### Command Line

```bash
# Persistent file
./target/release/pardusdb mydata.pardus

# In-memory only
./target/release/pardusdb
```

## SQL Syntax

### Supported Data Types

| Type      | Description                  | Example                |
|-----------|------------------------------|------------------------|
| `VECTOR(n)` | n-dimensional float vector   | `VECTOR(768)`          |
| `TEXT`    | UTF-8 string                 | `'hello world'`        |
| `INTEGER` | 64-bit integer               | `42`                   |
| `FLOAT`   | 64-bit float                 | `3.14`                 |
| `BOOLEAN` | true/false                   | `true`                 |

### Basic Operations

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    embedding VECTOR(768),
    title TEXT,
    category TEXT,
    score FLOAT
);

INSERT INTO documents (embedding, title, category, score)
VALUES ([0.1, 0.2, ...], 'Introduction to Rust', 'tutorial', 0.95);

SELECT * FROM documents WHERE category = 'tutorial' LIMIT 10;

UPDATE documents SET score = 0.99 WHERE id = 1;

DELETE FROM documents WHERE id = 1;
```

### Vector Similarity Search

```sql
SELECT * FROM documents
WHERE embedding SIMILARITY [0.12, 0.24, ...]
LIMIT 10;
```

Results are automatically ordered by distance (closest first).

### Utility Commands

```sql
SHOW TABLES;
DROP TABLE documents;
```

## REPL Commands

| Command         | Description                       |
|-----------------|-----------------------------------|
| `.create <file>`| Create and open a new database    |
| `.open <file>`  | Open an existing database         |
| `.save`         | Force save current database       |
| `.tables`       | List tables                       |
| `.clear`        | Clear screen                      |
| `help`          | Show help                         |
| `quit`          | Exit (auto-saves if file open)    |

## Performance (Apple Silicon M-series)

| Operation                  | Time          |
|----------------------------|---------------|
| Single insert              | ~140 µs/doc   |
| Query (k=10)               | ~42 µs        |
| Batch insert (1,000 docs)  | ~140 ms       |

## Python Integration & RAG Example

See `examples/python/simple_rag.py` — a complete retrieval-augmented generation demo using Ollama for embeddings and PardusDB as the vector store.

```bash
cd examples/python
pip install requests ollama
python simple_rag.py
```

## Why We Built PardusDB

The Pardus AI team built PardusDB because we believe private, local-first AI tools should be accessible to everyone — from individual developers to large teams.

PardusDB gives you the low-level building block for fast, private vector search, while [Pardus AI](https://pardusai.org/) delivers the high-level no-code experience for analysts, marketers, and business users who just want answers from their data.

If you enjoy working with PardusDB, we’d love for you to try [Pardus AI](https://pardusai.org/) — upload your spreadsheets or documents and ask questions in plain English. Free tier available, no credit card required.

## License

MIT License — use it freely in personal and commercial projects.

---

⭐ Star us on GitHub if you find this useful!  
🚀 Building something cool with PardusDB? Share it with us on X or Discord — we’d love to hear from you.

**Pardus AI** — https://pardusai.org/