# PardusDB MCP Server

Model Context Protocol (MCP) server for [PardusDB](https://github.com/pardus-ai/pardusdb) - A fast, SQLite-like embedded vector database.

This server enables AI agents like Claude to interact with PardusDB for vector similarity search, making it easy to build RAG applications, semantic search, and recommendation systems.

## Features

- **Vector Storage**: Store and manage embedding vectors with metadata
- **Similarity Search**: Fast HNSW-based vector similarity search
- **Batch Operations**: Efficient batch insert for bulk data loading
- **Raw SQL**: Execute any PardusDB SQL command
- **Simple Integration**: Works with Claude Desktop and other MCP clients

## Prerequisites

1. **Install PardusDB**:
   ```bash
   git clone https://github.com/pardus-ai/pardusdb
   cd pardusdb
   ./setup.sh
   ```

2. **Verify installation**:
   ```bash
   pardusdb --version
   # or just run: pardusdb
   ```

## Installation

### Option 1: npm (Recommended)

```bash
npm install -g @pardusai/pardusdb-mcp
```

### Option 2: From Source

```bash
cd mcp
npm install
npm run build
npm link
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pardusdb": {
      "command": "pardusdb-mcp"
    }
  }
}
```

### With custom database path:

```json
{
  "mcpServers": {
    "pardusdb": {
      "command": "pardusdb-mcp",
      "env": {
        "PARDUSDB_PATH": "/path/to/your/database.pardus"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `pardusdb_create_database` | Create a new database file |
| `pardusdb_open_database` | Open an existing database |
| `pardusdb_create_table` | Create a table for vectors |
| `pardusdb_insert_vector` | Insert a single vector |
| `pardusdb_batch_insert` | Insert multiple vectors |
| `pardusdb_search_similar` | Search for similar vectors |
| `pardusdb_execute_sql` | Execute raw SQL |
| `pardusdb_list_tables` | List all tables |
| `pardusdb_use_table` | Set current table |
| `pardusdb_status` | Get connection status |

## Usage Examples

### Creating a Database and Table

```
User: Create a new PardusDB database at data/knowledge.pardus with a table called "documents" for 768-dimensional vectors with title and content columns.

Claude: I'll create the database and table for you.
[Uses pardusdb_create_database, then pardusdb_create_table]
```

### Inserting Vectors

```
User: Insert this embedding [0.1, 0.2, 0.3, ...] into the documents table with title "Introduction" and content "Hello world".

Claude: [Uses pardusdb_insert_vector]
Vector inserted with ID: 1
```

### Searching for Similar Vectors

```
User: Search for documents similar to this embedding [0.15, 0.25, 0.35, ...], return top 5 results.

Claude: [Uses pardusdb_search_similar]
Found 5 similar vectors:
1. ID: 1, Distance: 0.0012, Title: "Introduction"
2. ID: 3, Distance: 0.0034, Title: "Getting Started"
...
```

### RAG Workflow Example

```
User: I want to build a RAG system. Create a knowledge base, index these documents, and answer my question.

Claude: I'll help you build a RAG system:
1. [pardusdb_create_database] - Create knowledge base
2. [pardusdb_create_table] - Create documents table with vector_dim=768
3. [pardusdb_batch_insert] - Index all documents with embeddings
4. [pardusdb_search_similar] - Find relevant context for your question
5. Generate answer using retrieved context
```

## Tool Parameters

### pardusdb_create_database

```json
{
  "path": "data/mydb.pardus"
}
```

### pardusdb_create_table

```json
{
  "name": "documents",
  "vector_dim": 768,
  "metadata_schema": {
    "title": "str",
    "content": "str",
    "category": "str",
    "score": "float"
  }
}
```

### pardusdb_insert_vector

```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "title": "Document Title",
    "content": "Document content..."
  },
  "table": "documents"
}
```

### pardusdb_batch_insert

```json
{
  "vectors": [
    [0.1, 0.2, ...],
    [0.3, 0.4, ...]
  ],
  "metadata_list": [
    {"title": "Doc 1", "content": "..."},
    {"title": "Doc 2", "content": "..."}
  ],
  "table": "documents"
}
```

### pardusdb_search_similar

```json
{
  "query_vector": [0.15, 0.25, ...],
  "k": 10,
  "table": "documents"
}
```

### pardusdb_execute_sql

```json
{
  "sql": "SELECT * FROM documents WHERE category = 'tutorial' LIMIT 10"
}
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run locally for testing
npm run dev
```

## Architecture

```
┌─────────────────┐     MCP      ┌─────────────────┐
│   Claude/AI     │ ◄────────► │  PardusDB MCP   │
│     Agent       │             │     Server      │
└─────────────────┘             └────────┬────────┘
                                         │
                                         │ spawn
                                         ▼
                                ┌─────────────────┐
                                │    pardusdb     │
                                │    binary       │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │    .pardus      │
                                │   database      │
                                └─────────────────┘
```

## Troubleshooting

### "pardusdb binary not found"

Make sure PardusDB is installed and in your PATH:
```bash
which pardusdb
# Should return: /usr/local/bin/pardusdb
```

### "Database file not found"

The database file must exist before opening. Use `pardusdb_create_database` first.

### "No table specified"

Either use `pardusdb_use_table` to set a current table, or provide the `table` parameter in each operation.

## License

MIT License

## Links

- **PardusDB**: https://github.com/pardus-ai/pardusdb
- **Pardus AI**: https://pardusai.org/
- **MCP Documentation**: https://modelcontextprotocol.io/
