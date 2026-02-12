# PardusDB Python SDK

A simple, Pythonic interface for PardusDB vector database.

## Installation

```bash
pip install pardusdb
```

Or install from source:

```bash
cd sdk/python
pip install -e .
```

## Prerequisites

The PardusDB binary must be installed and available in your PATH:

```bash
git clone https://github.com/pardus-ai/pardusdb
cd pardusdb
cargo build --release
./setup.sh  # Installs 'pardusdb' to /usr/local/bin
```

## Quick Start

```python
from pardusdb import PardusDB

# Create or open a database
db = PardusDB("my_vectors.pardus")

# Create a table for vector storage
db.create_table(
    "documents",
    vector_dim=768,
    metadata_schema={
        "title": "str",
        "content": "str",
        "category": "str",
    }
)

# Insert vectors with metadata
db.insert(
    vector=[0.1, 0.2, 0.3, ...],  # Your 768-dim embedding
    metadata={
        "title": "Introduction to Rust",
        "content": "Rust is a systems programming language...",
        "category": "tutorial"
    }
)

# Batch insert
db.insert_batch(
    vectors=[
        [0.1, 0.2, ...],
        [0.3, 0.4, ...],
        [0.5, 0.6, ...],
    ],
    metadata_list=[
        {"title": "Doc 1", "content": "...", "category": "news"},
        {"title": "Doc 2", "content": "...", "category": "blog"},
        {"title": "Doc 3", "content": "...", "category": "docs"},
    ]
)

# Search for similar vectors
results = db.search(
    query_vector=[0.15, 0.25, ...],  # Query embedding
    k=5
)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance:.4f}")
    print(f"Metadata: {result.metadata}")

# Close the database
db.close()
```

## Usage with Context Manager

```python
from pardusDB import PardusDB

with PardusDB("my_vectors.pardus") as db:
    db.create_table("docs", vector_dim=384, metadata_schema={"text": "str"})

    # Your operations here...
    # Database automatically saves and closes
```

## RAG Example

```python
from pardusdb import PardusDB
import numpy as np

# Mock embedding function - replace with your actual embedding model
def embed(text: str) -> list[float]:
    # Use OpenAI, SentenceTransformers, etc.
    return np.random.randn(384).tolist()

# Initialize database
db = PardusDB("knowledge_base.pardus")
db.create_table("documents", vector_dim=384, metadata_schema={"text": "str", "source": "str"})

# Index documents
documents = [
    {"text": "PardusDB is a fast vector database", "source": "docs/intro.txt"},
    {"text": "Vector similarity search uses HNSW algorithm", "source": "docs/tech.txt"},
    {"text": "The database supports batch inserts for performance", "source": "docs/perf.txt"},
]

for doc in documents:
    embedding = embed(doc["text"])
    db.insert(vector=embedding, metadata={"text": doc["text"], "source": doc["source"]})

# Query
query = "How does the search algorithm work?"
query_embedding = embed(query)
results = db.search(query_embedding, k=3)

print(f"Query: {query}\n")
print("Relevant documents:")
for r in results:
    print(f"  - {r.metadata.get('text', 'N/A')}")
```

## API Reference

### PardusDB

```python
class PardusDB:
    def __init__(self, path: str | None = None, binary_path: str | None = None):
        """
        Initialize PardusDB client.

        Args:
            path: Path to .pardus file. None for in-memory database.
            binary_path: Path to pardusdb binary. None to auto-detect.
        """
```

#### Table Operations

| Method | Description |
|--------|-------------|
| `create_table(name, vector_dim, metadata_schema=None)` | Create a new vector table |
| `use(name)` | Set current table for operations |
| `drop_table(name)` | Delete a table |
| `list_tables()` | List all tables |

#### Insert Operations

| Method | Description |
|--------|-------------|
| `insert(vector, metadata=None, table=None)` | Insert single vector |
| `insert_batch(vectors, metadata_list=None, table=None)` | Insert multiple vectors |

#### Search Operations

| Method | Description |
|--------|-------------|
| `search(query_vector, k=10, table=None)` | Find similar vectors |

#### CRUD Operations

| Method | Description |
|--------|-------------|
| `get(row_id, table=None)` | Get row by ID |
| `update(row_id, metadata, table=None)` | Update row metadata |
| `delete(row_id, table=None)` | Delete row |
| `delete_all(table=None)` | Delete all rows |

#### Utility

| Method | Description |
|--------|-------------|
| `raw_sql(sql)` | Execute raw SQL |
| `close()` | Close connection |

### VectorResult

```python
@dataclass
class VectorResult:
    id: int           # Row ID
    distance: float   # Similarity distance (lower = more similar)
    metadata: dict    # Associated metadata
```

## Error Handling

```python
from pardusdb import PardusDB, PardusDBError, QueryError, TableNotFoundError

try:
    db = PardusDB("mydb.pardus")
    results = db.search([0.1, 0.2, ...], k=5)
except TableNotFoundError as e:
    print(f"Table {e.table_name} doesn't exist")
except QueryError as e:
    print(f"Query failed: {e.message}")
except PardusDBError as e:
    print(f"Error: {e.message}")
```

## License

MIT License
