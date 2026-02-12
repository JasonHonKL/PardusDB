//! PardusDB - A single-file embedded vector database.
//!
//! Inspired by SQLite's embedded design, PardusDB provides vector similarity search
//! using a graph-based approach with geometric diversity pruning.
//!
//! # Features
//!
//! - Single-file persistence (like SQLite)
//! - Multiple tables per file
//! - SQL-like syntax for CRUD operations
//! - Custom metadata columns (TEXT, INTEGER, FLOAT, BOOLEAN)
//! - Multiple distance metrics (Cosine, Dot Product, Euclidean)
//!
//! # SQL-like Syntax
//!
//! ```sql
//! -- Create a table with vector and metadata columns
//! CREATE TABLE documents (
//!     id INTEGER PRIMARY KEY,
//!     embedding VECTOR(768),
//!     title TEXT,
//!     content TEXT,
//!     score FLOAT
//! );
//!
//! -- Insert with vector and metadata
//! INSERT INTO documents (embedding, title, content, score)
//! VALUES ([0.1, 0.2, ...], 'Title', 'Content', 0.95);
//!
//! -- Query by vector similarity
//! SELECT * FROM documents
//! WHERE embedding SIMILARITY [0.1, 0.2, ...]
//! LIMIT 10;
//!
//! -- Update and delete
//! UPDATE documents SET score = 0.99 WHERE id = 5;
//! DELETE FROM documents WHERE id = 5;
//! ```
//!
//! # Example
//!
//! ```rust
//! use pardusdb::Database;
//!
//! // Create an in-memory database
//! let mut db = Database::in_memory();
//!
//! // Execute SQL commands
//! db.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);")?;
//! db.execute("INSERT INTO docs (embedding, title) VALUES ([0.1, 0.2, 0.3], 'Hello');")?;
//!
//! // Query
//! let result = db.execute("SELECT * FROM docs LIMIT 10;")?;
//! println!("{}", result);
//! # Ok::<(), pardusdb::MarsError>(())
//! ```

pub mod concurrent;
pub mod database;
pub mod db;
pub mod distance;
pub mod error;
pub mod graph;
pub mod node;
pub mod parser;
pub mod prepared;
pub mod schema;
pub mod storage;
pub mod table;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-exports for convenience
pub use database::{Database, ExecuteResult, TableInfo};
pub use db::{Config, SearchResult, VectorDB, CosineDB, DotProductDB, EuclideanDB};
pub use distance::{Distance, Numeric, Cosine, DotProduct, Euclidean};
pub use error::{MarsError, Result};
pub use graph::{Graph, GraphConfig};
pub use node::{Candidate, Node, NodeId};
pub use parser::{Command, ComparisonOp, parse};
pub use prepared::{BatchInserter, PreparedStatement, StatementCache};
pub use schema::{Column, ColumnType, Row, Schema, Value};
pub use table::Table;

#[cfg(feature = "gpu")]
pub use gpu::{GpuDistance, GpuError};

// Concurrent module re-exports
pub use concurrent::{ConcurrentDatabase, Connection, DatabaseInner, DatabasePool, ScopedTransaction};
