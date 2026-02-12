# PardusDB

A SQLite-like embedded vector database with graph-based approximate nearest neighbor search.

## Features

- **Single-file storage** - Like SQLite, everything in one `.pardus` file
- **Multiple tables** - Store different vector types in the same database
- **SQL-like syntax** - Familiar CREATE, INSERT, SELECT, UPDATE, DELETE
- **Vector similarity search** - Find nearest neighbors using graph traversal
- **Thread-safe** - Concurrent reads, safe for multi-threaded applications
- **Transactions** - Begin/commit/rollback for atomic operations
- **Optional GPU acceleration** - For large-scale batch operations

## Installation

```bash
# Build from source
git clone https://github.com/your-repo/pardusdb
cd pardusdb
cargo build --release
```

The binary will be at `target/release/pardusdb`.

## Quick Start

### Interactive Mode (REPL)

```bash
./target/release/pardusdb
```

```
╔═══════════════════════════════════════════════════════════════╗
║                        PardusDB REPL                          ║
║              Vector Database with SQL Interface               ║
╚═══════════════════════════════════════════════════════════════╝

pardusdb [memory]> .create mydb.pardus
Created and opened: mydb.pardus

pardusdb [mydb.pardus]> CREATE TABLE docs (embedding VECTOR(768), content TEXT);
Table 'docs' created

pardusdb [mydb.pardus]> INSERT INTO docs (embedding, content) VALUES ([0.1, 0.2, 0.3, ...], 'Hello World');
Inserted row with id=1

pardusdb [mydb.pardus]> SELECT * FROM docs WHERE embedding SIMILARITY [0.1, 0.2, 0.3, ...] LIMIT 5;
Found 1 similar rows:
  id=1, distance=0.0000, values=[Vector([...]), Text("Hello World")]

pardusdb [mydb.pardus]> quit
Saved to: mydb.pardus
Goodbye!
```

### Command Line

```bash
# Create/open a database file
./target/release/pardusdb mydata.pardus

# In-memory mode (no file)
./target/release/pardusdb
```

## SQL Syntax

### Data Types

| Type | Description | Example |
|------|-------------|---------|
| `VECTOR(n)` | n-dimensional float vector | `VECTOR(768)` |
| `TEXT` | String value | `'hello world'` |
| `INTEGER` | 64-bit integer | `42` |
| `FLOAT` | 64-bit float | `3.14` |
| `BOOLEAN` | true/false | `true` |

### CREATE TABLE

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    embedding VECTOR(768),
    title TEXT,
    category TEXT,
    score FLOAT
);
```

### INSERT

```sql
INSERT INTO documents (embedding, title, category, score)
VALUES ([0.1, 0.2, 0.3, ...], 'Introduction to Rust', 'tutorial', 0.95);
```

### SELECT

```sql
-- Select all
SELECT * FROM documents;

-- With WHERE clause
SELECT * FROM documents WHERE category = 'tutorial';
SELECT * FROM documents WHERE score > 0.9;

-- With LIMIT
SELECT * FROM documents LIMIT 10;
```

### Vector Similarity Search

```sql
-- Find 10 most similar documents
SELECT * FROM documents
WHERE embedding SIMILARITY [0.1, 0.2, 0.3, ...]
LIMIT 10;
```

Results are automatically sorted by distance (closest first).

### UPDATE

```sql
UPDATE documents SET score = 0.99 WHERE id = 1;
```

### DELETE

```sql
DELETE FROM documents WHERE id = 1;
```

### Other Commands

```sql
SHOW TABLES;           -- List all tables
DROP TABLE documents;  -- Delete a table
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `.create <file>` | Create a new database file |
| `.open <file>` | Open an existing database |
| `.save` | Save current database |
| `.tables` | List all tables |
| `.clear` | Clear screen |
| `help` | Show help |
| `quit` | Exit (auto-saves if file open) |

## Example Workflow

```sql
-- Create a document store
CREATE TABLE docs (
    id INTEGER PRIMARY KEY,
    embedding VECTOR(768),
    title TEXT,
    content TEXT
);

-- Insert documents with embeddings
INSERT INTO docs (embedding, title, content)
VALUES ([0.15, 0.23, ...], 'Rust Guide', 'Introduction to Rust programming');

INSERT INTO docs (embedding, title, content)
VALUES ([0.42, 0.11, ...], 'Python Tips', 'Advanced Python techniques');

-- Find similar documents
SELECT * FROM docs
WHERE embedding SIMILARITY [0.14, 0.22, ...]
LIMIT 5;

-- Update a document
UPDATE docs SET title = 'Rust Programming Guide' WHERE id = 1;

-- Clean up
DELETE FROM docs WHERE id = 2;
```

## Graph Configuration

The similarity search uses a graph-based algorithm. Tune for your needs:

```sql
-- Higher max_neighbors = better recall, more memory
-- Higher search_buffer = better recall, slower search
```

Default settings work well for most cases. Adjust if you need:
- **Higher recall**: Increase `search_buffer` (default: 64)
- **Lower memory**: Decrease `max_neighbors` (default: 16)

## Performance

On Apple Silicon (M-series):

| Operation | Time |
|-----------|------|
| Insert | ~140µs/doc |
| Query (k=10) | ~42µs |
| Batch insert (1000 docs) | ~140ms |

## Python Integration

See `examples/python/simple_rag.py` for a complete RAG example using Ollama for embeddings.

```bash
cd examples/python
pip install requests
python simple_rag.py
```

## License

MIT
