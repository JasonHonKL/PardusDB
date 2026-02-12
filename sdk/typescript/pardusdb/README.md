# PardusDB TypeScript SDK

A simple, type-safe interface for PardusDB vector database.

## Installation

```bash
npm install pardusdb
```

Or with yarn:

```bash
yarn add pardusdb
```

Or install from source:

```bash
cd sdk/typescript/pardusdb
npm install
npm run build
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

```typescript
import { PardusDB, VectorResult } from "pardusdb";

// Create or open a database
const db = new PardusDB("my_vectors.pardus");

// Create a table for vector storage
await db.createTable(
  "documents",
  768, // vector dimension
  {
    title: "str",
    content: "str",
    category: "str",
  }
);

// Insert a vector with metadata
const id = await db.insert(
  [0.1, 0.2, 0.3, /* ... */], // Your 768-dim embedding
  {
    title: "Introduction to Rust",
    content: "Rust is a systems programming language...",
    category: "tutorial",
  }
);
console.log(`Inserted row with ID: ${id}`);

// Batch insert
await db.insertBatch(
  [
    [0.1, 0.2, /* ... */],
    [0.3, 0.4, /* ... */],
    [0.5, 0.6, /* ... */],
  ],
  [
    { title: "Doc 1", content: "...", category: "news" },
    { title: "Doc 2", content: "...", category: "blog" },
    { title: "Doc 3", content: "...", category: "docs" },
  ]
);

// Search for similar vectors
const results: VectorResult[] = await db.search(
  [0.15, 0.25, /* ... */], // Query embedding
  5 // k
);

for (const result of results) {
  console.log(`ID: ${result.id}, Distance: ${result.distance.toFixed(4)}`);
}

// Close the database
await db.close();
```

## RAG Example

```typescript
import { PardusDB, VectorResult } from "pardusdb";

// Mock embedding function - replace with your actual embedding model
async function embed(text: string): Promise<number[]> {
  // Use OpenAI, SentenceTransformers, etc.
  return Array.from({ length: 384 }, () => Math.random());
}

async function main() {
  // Initialize database
  const db = new PardusDB("knowledge_base.pardus");
  await db.createTable("documents", 384, { text: "str", source: "str" });

  // Index documents
  const documents = [
    { text: "PardusDB is a fast vector database", source: "docs/intro.txt" },
    { text: "Vector similarity search uses HNSW algorithm", source: "docs/tech.txt" },
    { text: "The database supports batch inserts for performance", source: "docs/perf.txt" },
  ];

  for (const doc of documents) {
    const embedding = await embed(doc.text);
    await db.insert(embedding, { text: doc.text, source: doc.source });
  }

  // Query
  const query = "How does the search algorithm work?";
  const queryEmbedding = await embed(query);
  const results = await db.search(queryEmbedding, 3);

  console.log(`Query: ${query}\n`);
  console.log("Relevant documents:");
  for (const r of results) {
    console.log(`  - ID ${r.id} (distance: ${r.distance.toFixed(4)})`);
  }

  await db.close();
}

main().catch(console.error);
```

## API Reference

### PardusDB

```typescript
class PardusDB {
  constructor(dbPath?: string, binaryPath?: string);
}
```

#### Table Operations

| Method | Description |
|--------|-------------|
| `createTable(name, vectorDim, metadataSchema?, ifNotExists?)` | Create a new vector table |
| `use(name)` | Set current table for operations |
| `dropTable(name, ifExists?)` | Delete a table |
| `listTables()` | List all tables |

#### Insert Operations

| Method | Description |
|--------|-------------|
| `insert(vector, metadata?, options?)` | Insert single vector |
| `insertBatch(vectors, metadataList?, options?)` | Insert multiple vectors |

#### Search Operations

| Method | Description |
|--------|-------------|
| `search(queryVector, kOrOptions?)` | Find similar vectors |

#### CRUD Operations

| Method | Description |
|--------|-------------|
| `get(rowId, options?)` | Get row by ID |
| `update(rowId, metadata, options?)` | Update row metadata |
| `delete(rowId, options?)` | Delete row |
| `deleteAll(options?)` | Delete all rows |

#### Utility

| Method | Description |
|--------|-------------|
| `rawSql(sql)` | Execute raw SQL |
| `close()` | Close connection |

### VectorResult

```typescript
interface VectorResult {
  id: number;        // Row ID
  distance: number;  // Similarity distance (lower = more similar)
  metadata: Record<string, unknown>;  // Associated metadata
}
```

### Error Classes

```typescript
class PardusDBError extends Error {}
class ConnectionError extends PardusDBError {}
class QueryError extends PardusDBError {
  query?: string;
}
class TableNotFoundError extends PardusDBError {
  tableName: string;
}
```

## Error Handling

```typescript
import {
  PardusDB,
  PardusDBError,
  QueryError,
  TableNotFoundError
} from "pardusdb";

try {
  const db = new PardusDB("mydb.pardus");
  const results = await db.search([0.1, 0.2, ...], 5);
} catch (e) {
  if (e instanceof TableNotFoundError) {
    console.log(`Table ${e.tableName} doesn't exist`);
  } else if (e instanceof QueryError) {
    console.log(`Query failed: ${e.message}`);
  } else if (e instanceof PardusDBError) {
    console.log(`Error: ${e.message}`);
  }
}
```

## Type Definitions

```typescript
// Column types for metadata schema
type ColumnType = "TEXT" | "INTEGER" | "FLOAT" | "BOOLEAN" | "VECTOR";

interface MetadataSchema {
  [columnName: string]: ColumnType | "str" | "int" | "float" | "bool" | "string";
}

interface SearchOptions {
  k?: number;
  table?: string;
}

interface InsertOptions {
  table?: string;
}
```

## License

MIT License
