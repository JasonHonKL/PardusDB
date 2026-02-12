//! Concurrent database support for multi-threaded applications.
//!
//! This module provides thread-safe access to PardusDB using `Arc<RwLock>` pattern,
//! allowing multiple concurrent readers or exclusive writer access.
//!
//! # Architecture
//!
//! - `ConcurrentDatabase`: Thread-safe database wrapper using `Arc<RwLock<DatabaseInner>>`
//! - `Connection`: A handle to the database for executing operations
//! - `Transaction`: Batched operations with commit/rollback support
//!
//! # Concurrency Model
//!
//! - **Read operations**: Multiple threads can read simultaneously (shared lock)
//! - **Write operations**: Exclusive access required (exclusive lock)
//! - **Transactions**: All operations in a transaction are atomic
//!
//! # Example
//!
//! ```rust
//! use pardusdb::concurrent::{ConcurrentDatabase, Connection};
//! use std::sync::Arc;
//! use std::thread;
//!
//! // Create a concurrent database
//! let db = ConcurrentDatabase::in_memory();
//!
//! // Create table first
//! let mut conn = db.connect();
//! conn.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();
//!
//! // Share across threads
//! let db = Arc::new(db);
//! let mut handles = vec![];
//!
//! for i in 0..4 {
//!     let db_clone = Arc::clone(&db);
//!     let handle = thread::spawn(move || {
//!         let mut conn = db_clone.connect();
//!         conn.execute(&format!(
//!             "INSERT INTO docs (embedding, title) VALUES ([0.1, 0.2, 0.3], 'Doc {}');",
//!             i
//!         )).unwrap();
//!     });
//!     handles.push(handle);
//! }
//!
//! for handle in handles {
//!     handle.join().unwrap();
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use serde::{Deserialize, Serialize};

use crate::database::{ExecuteResult, TableInfo};
use crate::error::{MarsError, Result};
use crate::graph::GraphConfig;
use crate::parser::{parse, Command, ComparisonOp};
use crate::schema::{Column, ColumnType, Row, Schema, Value};
use crate::table::Table;

/// File header with database metadata
#[derive(Serialize, Deserialize)]
struct DbHeader {
    pub version: u32,
    pub table_count: u32,
}

/// Serialized table data
#[derive(Serialize, Deserialize)]
struct TableData {
    pub schema: Schema,
    pub rows: Vec<Row>,
    pub centroid: Vec<f32>,
    pub next_id: u64,
}

/// Internal database state
pub struct DatabaseInner {
    pub tables: HashMap<String, Table>,
    pub config: GraphConfig,
    pub path: Option<PathBuf>,
}

/// A thread-safe database that can be shared across threads.
///
/// Uses `RwLock` internally to allow multiple concurrent readers
/// or one exclusive writer.
pub struct ConcurrentDatabase {
    inner: RwLock<DatabaseInner>,
}

impl ConcurrentDatabase {
    /// Create an in-memory concurrent database.
    pub fn in_memory() -> Self {
        ConcurrentDatabase {
            inner: RwLock::new(DatabaseInner {
                tables: HashMap::new(),
                config: GraphConfig::default(),
                path: None,
            }),
        }
    }

    /// Create with custom graph configuration.
    pub fn with_config(config: GraphConfig) -> Self {
        ConcurrentDatabase {
            inner: RwLock::new(DatabaseInner {
                tables: HashMap::new(),
                config,
                path: None,
            }),
        }
    }

    /// Open or create a concurrent database file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            Self::load(&path)
        } else {
            Self::create_new(&path)
        }
    }

    fn create_new(path: &Path) -> Result<Self> {
        let db = ConcurrentDatabase {
            inner: RwLock::new(DatabaseInner {
                tables: HashMap::new(),
                config: GraphConfig::default(),
                path: Some(path.to_path_buf()),
            }),
        };

        // Write empty database
        db.save()?;

        Ok(db)
    }

    fn load(path: &Path) -> Result<Self> {
        use std::fs::File;
        use std::io::{BufReader, Read};

        let mut file = File::open(path)?;
        let mut reader = BufReader::new(&mut file);

        // Read header
        let mut header_buf = [0u8; 8];
        reader.read_exact(&mut header_buf)?;

        let _version =
            u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]);
        let table_count =
            u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);

        // Read tables
        let mut tables = HashMap::new();

        for _ in 0..table_count {
            let mut size_buf = [0u8; 8];
            reader.read_exact(&mut size_buf)?;
            let size = u64::from_le_bytes(size_buf) as usize;

            let mut table_buf = vec![0u8; size];
            reader.read_exact(&mut table_buf)?;

            let table_data: TableData = bincode::deserialize(&table_buf)
                .map_err(|e| MarsError::InvalidFormat(format!("Failed to deserialize table: {}", e)))?;

            let mut table = Table::new(table_data.schema, GraphConfig::default())?;

            for row in table_data.rows {
                if let Some(vec_idx) = table.schema.columns.iter().position(|c| {
                    matches!(c.data_type, ColumnType::Vector(_))
                }) {
                    if let Some(vec) = row.values.get(vec_idx).and_then(|v| v.as_vector()) {
                        table.graph.insert(vec.to_vec());
                    }
                }
                let id = row.id;
                table.rows.insert(id, row);
            }

            table.next_id = table_data.next_id;
            tables.insert(table.name().to_string(), table);
        }

        Ok(ConcurrentDatabase {
            inner: RwLock::new(DatabaseInner {
                tables,
                config: GraphConfig::default(),
                path: Some(path.to_path_buf()),
            }),
        })
    }

    /// Save database to file.
    ///
    /// This acquires a read lock and saves the current state to disk.
    pub fn save(&self) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::{BufWriter, Write};

        let inner = self.inner.read().unwrap();

        let path = match &inner.path {
            Some(p) => p,
            None => return Ok(()),
        };

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);

        let header = DbHeader {
            version: 1,
            table_count: inner.tables.len() as u32,
        };
        writer.write_all(&header.version.to_le_bytes())?;
        writer.write_all(&header.table_count.to_le_bytes())?;

        for table in inner.tables.values() {
            let table_data = TableData {
                schema: table.schema.clone(),
                rows: table.rows.values().cloned().collect(),
                centroid: table.graph.centroid().to_vec(),
                next_id: table.next_id,
            };

            let serialized = bincode::serialize(&table_data)
                .map_err(|e| MarsError::InvalidFormat(format!("Failed to serialize table: {}", e)))?;

            writer.write_all(&(serialized.len() as u64).to_le_bytes())?;
            writer.write_all(&serialized)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Create a new connection to this database.
    ///
    /// The connection can be used to execute operations. Each connection
    /// maintains its own transaction state.
    pub fn connect(&self) -> Connection<'_> {
        Connection {
            db: self,
            transaction: None,
        }
    }

    /// Get a read guard for direct access.
    pub fn read(&self) -> RwLockReadGuard<'_, DatabaseInner> {
        self.inner.read().unwrap()
    }

    /// Get a write guard for direct access.
    pub fn write(&self) -> RwLockWriteGuard<'_, DatabaseInner> {
        self.inner.write().unwrap()
    }

    /// Execute a read operation with a read lock.
    pub fn with_read<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&DatabaseInner) -> T,
    {
        let guard = self.inner.read().unwrap();
        f(&guard)
    }

    /// Execute a write operation with a write lock.
    pub fn with_write<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut DatabaseInner) -> T,
    {
        let mut guard = self.inner.write().unwrap();
        f(&mut guard)
    }
}

/// A connection to a concurrent database.
///
/// Connections can execute operations and manage transactions.
/// Each connection maintains its own transaction state.
pub struct Connection<'a> {
    db: &'a ConcurrentDatabase,
    transaction: Option<TransactionState>,
}

/// Transaction state for a connection.
struct TransactionState {
    operations: Vec<PendingOperation>,
}

/// A pending operation in a transaction.
enum PendingOperation {
    CreateTable { name: String, columns: Vec<crate::parser::ColumnDef> },
    DropTable { name: String, if_exists: bool },
    Insert { table: String, columns: Vec<String>, values: Vec<Vec<Value>> },
    Update { table: String, assignments: Vec<(String, Value)>, where_clause: Option<crate::parser::WhereClause> },
    Delete { table: String, where_clause: Option<crate::parser::WhereClause> },
}

impl<'a> Connection<'a> {
    /// Execute a SQL command.
    ///
    /// If a transaction is active, the operation is queued for later execution.
    /// Otherwise, it's executed immediately.
    pub fn execute(&mut self, sql: &str) -> Result<ExecuteResult> {
        let command = parse(sql)?;

        if let Some(ref mut tx) = self.transaction {
            // Queue operation for transaction
            let pending = match command {
                Command::CreateTable { name, columns } => {
                    PendingOperation::CreateTable { name, columns }
                }
                Command::DropTable { name, if_exists } => {
                    PendingOperation::DropTable { name, if_exists }
                }
                Command::Insert { table, columns, values } => {
                    PendingOperation::Insert { table, columns, values }
                }
                Command::Update { table, assignments, where_clause } => {
                    PendingOperation::Update { table, assignments, where_clause }
                }
                Command::Delete { table, where_clause } => {
                    PendingOperation::Delete { table, where_clause }
                }
                Command::Select { .. } => {
                    // SELECT is immediate even in transaction
                    return self.execute_command(command);
                }
                Command::ShowTables => {
                    return self.execute_command(command);
                }
            };
            tx.operations.push(pending);
            Ok(ExecuteResult::Insert { id: 0 }) // Placeholder
        } else {
            self.execute_command(command)
        }
    }

    fn execute_command(&mut self, command: Command) -> Result<ExecuteResult> {
        match command {
            Command::CreateTable { name, columns } => self.create_table(name, columns),
            Command::DropTable { name, if_exists } => self.drop_table(name, if_exists),
            Command::Insert { table, columns, values } => self.insert_multi(table, columns, values),
            Command::Select { table, columns, where_clause, order_by, limit, offset, distinct } => {
                self.select(table, columns, where_clause.as_ref(), order_by.as_ref(), limit, offset, distinct)
            }
            Command::Update { table, assignments, where_clause } => {
                self.update(table, assignments, where_clause.as_ref())
            }
            Command::Delete { table, where_clause } => self.delete(table, where_clause.as_ref()),
            Command::ShowTables => self.show_tables(),
        }
    }

    /// Begin a new transaction.
    ///
    /// All subsequent operations will be queued until `commit()` is called.
    pub fn begin(&mut self) -> Result<()> {
        if self.transaction.is_some() {
            return Err(MarsError::InvalidFormat("Transaction already in progress".into()));
        }
        self.transaction = Some(TransactionState {
            operations: Vec::new(),
        });
        Ok(())
    }

    /// Commit the current transaction.
    ///
    /// All queued operations are executed atomically with an exclusive lock.
    pub fn commit(&mut self) -> Result<Vec<ExecuteResult>> {
        let tx = self.transaction.take()
            .ok_or_else(|| MarsError::InvalidFormat("No transaction in progress".into()))?;

        let mut results = Vec::new();
        let mut guard = self.db.inner.write().unwrap();

        for op in tx.operations {
            let result = self.execute_pending(&mut guard, op)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Rollback the current transaction.
    ///
    /// All queued operations are discarded.
    pub fn rollback(&mut self) -> Result<()> {
        self.transaction.take();
        Ok(())
    }

    /// Check if a transaction is active.
    pub fn in_transaction(&self) -> bool {
        self.transaction.is_some()
    }

    fn execute_pending(
        &self,
        inner: &mut DatabaseInner,
        op: PendingOperation,
    ) -> Result<ExecuteResult> {
        match op {
            PendingOperation::CreateTable { name, columns } => {
                Self::create_table_inner(inner, name, columns)
            }
            PendingOperation::DropTable { name, if_exists } => {
                Self::drop_table_inner(inner, name, if_exists)
            }
            PendingOperation::Insert { table, columns, values } => {
                Self::insert_inner(inner, table, columns, values)
            }
            PendingOperation::Update { table, assignments, where_clause } => {
                Self::update_inner(inner, table, assignments, where_clause.as_ref())
            }
            PendingOperation::Delete { table, where_clause } => {
                Self::delete_inner(inner, table, where_clause.as_ref())
            }
        }
    }

    fn create_table_inner(
        inner: &mut DatabaseInner,
        name: String,
        columns: Vec<crate::parser::ColumnDef>,
    ) -> Result<ExecuteResult> {
        if inner.tables.contains_key(&name) {
            return Err(MarsError::InvalidConfig(format!("Table '{}' already exists", name)));
        }

        let mut schema = Schema::new(&name);
        for col_def in columns {
            let is_vector = matches!(col_def.data_type, ColumnType::Vector(_));
            let col_name = col_def.name.clone();

            let mut col = Column::new(&col_def.name, col_def.data_type);
            col.primary_key = col_def.primary_key;
            col.nullable = !col_def.not_null;
            schema.columns.push(col);

            if is_vector {
                schema.vector_column = Some(col_name);
            }
        }

        let table = Table::new(schema, inner.config.clone())?;
        inner.tables.insert(name.clone(), table);

        Ok(ExecuteResult::CreateTable { name })
    }

    fn drop_table_inner(inner: &mut DatabaseInner, name: String, if_exists: bool) -> Result<ExecuteResult> {
        if inner.tables.remove(&name).is_none() && !if_exists {
            return Err(MarsError::InvalidFormat(format!("Table '{}' does not exist", name)));
        }
        Ok(ExecuteResult::DropTable { name })
    }

    fn insert_inner(
        inner: &mut DatabaseInner,
        table_name: String,
        columns: Vec<String>,
        values: Vec<Vec<Value>>,
    ) -> Result<ExecuteResult> {
        let table = inner.tables.get_mut(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let mut last_id = 0u64;
        for row_values in values {
            last_id = table.insert(&columns, row_values)?;
        }
        Ok(ExecuteResult::Insert { id: last_id })
    }

    fn update_inner(
        inner: &mut DatabaseInner,
        table_name: String,
        assignments: Vec<(String, Value)>,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<ExecuteResult> {
        let table = inner.tables.get_mut(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let count = table.update(&assignments, where_clause)?;
        Ok(ExecuteResult::Update { count })
    }

    fn delete_inner(
        inner: &mut DatabaseInner,
        table_name: String,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<ExecuteResult> {
        let table = inner.tables.get_mut(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let count = table.delete(where_clause)?;
        Ok(ExecuteResult::Delete { count })
    }

    fn create_table(&mut self, name: String, columns: Vec<crate::parser::ColumnDef>) -> Result<ExecuteResult> {
        let mut guard = self.db.inner.write().unwrap();
        Self::create_table_inner(&mut guard, name, columns)
    }

    fn drop_table(&mut self, name: String, if_exists: bool) -> Result<ExecuteResult> {
        let mut guard = self.db.inner.write().unwrap();
        Self::drop_table_inner(&mut guard, name, if_exists)
    }

    fn insert_multi(&mut self, table: String, columns: Vec<String>, values: Vec<Vec<Value>>) -> Result<ExecuteResult> {
        let mut guard = self.db.inner.write().unwrap();
        Self::insert_inner(&mut guard, table, columns, values)
    }

    fn select(
        &self,
        table_name: String,
        columns: Vec<crate::parser::SelectColumn>,
        where_clause: Option<&crate::parser::WhereClause>,
        order_by: Option<&crate::parser::OrderBy>,
        limit: Option<usize>,
        offset: Option<usize>,
        distinct: bool,
    ) -> Result<ExecuteResult> {
        let guard = self.db.inner.read().unwrap();

        let table = guard.tables.get(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        // Check for vector similarity query
        if let Some(wc) = where_clause {
            for cond in &wc.conditions {
                if cond.operator == ComparisonOp::Similar {
                    if let crate::parser::ConditionValue::Single(Value::Vector(query_vec)) = &cond.value {
                        let k = limit.unwrap_or(10);
                        let results = table.select_by_similarity(query_vec, k, 100);
                        return Ok(ExecuteResult::SelectSimilar { results });
                    }
                }
            }
        }

        // Convert SelectColumn to column names
        let col_names: Vec<String> = columns.iter()
            .filter_map(|c| match c {
                crate::parser::SelectColumn::Column(name) => Some(name.clone()),
                _ => None,
            })
            .collect();

        let is_star = columns.iter().any(|c| matches!(c, crate::parser::SelectColumn::All));

        let rows = table.select(
            if is_star { &[] } else { &col_names },
            where_clause,
            limit,
            offset,
            order_by,
            distinct,
        );
        Ok(ExecuteResult::Select { rows })
    }

    fn update(
        &mut self,
        table_name: String,
        assignments: Vec<(String, Value)>,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<ExecuteResult> {
        let mut guard = self.db.inner.write().unwrap();
        Self::update_inner(&mut guard, table_name, assignments, where_clause)
    }

    fn delete(
        &mut self,
        table_name: String,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<ExecuteResult> {
        let mut guard = self.db.inner.write().unwrap();
        Self::delete_inner(&mut guard, table_name, where_clause)
    }

    fn show_tables(&self) -> Result<ExecuteResult> {
        let guard = self.db.inner.read().unwrap();

        let tables: Vec<TableInfo> = guard.tables.values()
            .map(|t| TableInfo {
                name: t.name().to_string(),
                rows: t.len(),
                dimension: t.schema.get_vector_dimension().unwrap_or(0),
            })
            .collect();

        Ok(ExecuteResult::ShowTables { tables })
    }

    /// Direct insert without SQL parsing.
    pub fn insert_direct(
        &mut self,
        table_name: &str,
        vector: Vec<f32>,
        metadata: Vec<(&str, Value)>,
    ) -> Result<u64> {
        let mut guard = self.db.inner.write().unwrap();

        let table = guard.tables.get_mut(table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let mut row_values: Vec<Value> = table.schema.columns.iter()
            .map(|_| Value::Null)
            .collect();

        for (i, col) in table.schema.columns.iter().enumerate() {
            if matches!(col.data_type, ColumnType::Vector(_)) {
                row_values[i] = Value::Vector(vector.clone());
            }
        }

        for (col_name, value) in metadata {
            if let Some(idx) = table.schema.columns.iter().position(|c| &c.name == col_name) {
                row_values[idx] = value;
            }
        }

        table.insert_row(row_values)
    }

    /// Batch insert without SQL parsing - significantly faster than individual inserts.
    pub fn insert_batch_direct(
        &mut self,
        table_name: &str,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<Vec<(&str, Value)>>,
    ) -> Result<Vec<u64>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let mut guard = self.db.inner.write().unwrap();

        let table = guard.tables.get_mut(table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let batch_size = vectors.len();
        let mut rows: Vec<Vec<Value>> = Vec::with_capacity(batch_size);

        for (i, vector) in vectors.into_iter().enumerate() {
            let mut row_values: Vec<Value> = table.schema.columns.iter()
                .map(|_| Value::Null)
                .collect();

            // Set vector column
            for (j, col) in table.schema.columns.iter().enumerate() {
                if matches!(col.data_type, ColumnType::Vector(_)) {
                    row_values[j] = Value::Vector(vector.clone());
                }
            }

            // Set metadata if provided
            if let Some(meta) = metadata.get(i) {
                for (col_name, value) in meta {
                    if let Some(idx) = table.schema.columns.iter().position(|c| &c.name == *col_name) {
                        row_values[idx] = value.clone();
                    }
                }
            }

            rows.push(row_values);
        }

        table.insert_batch(rows)
    }

    /// Direct similarity search without SQL parsing.
    pub fn search_similar(
        &self,
        table_name: &str,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u64, Vec<Value>, f32)>> {
        let guard = self.db.inner.read().unwrap();

        let table = guard.tables.get(table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let results = table.select_by_similarity(query, k, ef_search);

        Ok(results.into_iter()
            .map(|(row, dist)| (row.id, row.values, dist))
            .collect())
    }

    /// Get table names.
    pub fn table_names(&self) -> Vec<String> {
        let guard = self.db.inner.read().unwrap();
        guard.tables.keys().cloned().collect()
    }

    /// Get the underlying database reference.
    pub fn database(&self) -> &'a ConcurrentDatabase {
        self.db
    }
}

/// A scoped transaction that automatically rolls back if not committed.
///
/// This provides RAII-style transaction management.
pub struct ScopedTransaction<'a> {
    conn: &'a mut Connection<'a>,
    committed: bool,
}

impl<'a> ScopedTransaction<'a> {
    /// Create a new scoped transaction.
    pub fn new(conn: &'a mut Connection<'a>) -> Result<Self> {
        conn.begin()?;
        Ok(ScopedTransaction {
            conn,
            committed: false,
        })
    }

    /// Commit the transaction.
    pub fn commit(mut self) -> Result<Vec<ExecuteResult>> {
        self.committed = true;
        self.conn.commit()
    }
}

impl<'a> Drop for ScopedTransaction<'a> {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.conn.rollback();
        }
    }
}

/// A thread-safe database pool for managing multiple connections.
///
/// This is useful when you need to share database access across
/// many threads without passing references.
#[derive(Clone)]
pub struct DatabasePool {
    db: Arc<ConcurrentDatabase>,
}

impl DatabasePool {
    /// Create a new pool from a concurrent database.
    pub fn new(db: ConcurrentDatabase) -> Self {
        DatabasePool {
            db: Arc::new(db),
        }
    }

    /// Create an in-memory database pool.
    pub fn in_memory() -> Self {
        DatabasePool {
            db: Arc::new(ConcurrentDatabase::in_memory()),
        }
    }

    /// Open a database file and create a pool.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = ConcurrentDatabase::open(path)?;
        Ok(DatabasePool {
            db: Arc::new(db),
        })
    }

    /// Get a connection from the pool.
    pub fn connect(&self) -> Connection<'_> {
        self.db.connect()
    }

    /// Get a reference to the underlying database.
    pub fn database(&self) -> &ConcurrentDatabase {
        &self.db
    }

    /// Save the database.
    pub fn save(&self) -> Result<()> {
        self.db.save()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_concurrent_insert() {
        // Create Arc first, then get connections from it
        let db = Arc::new(ConcurrentDatabase::in_memory());
        let mut conn = db.connect();

        conn.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();

        // Insert from multiple threads
        let mut handles = vec![];

        for i in 0..4 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let mut conn = db_clone.connect();
                conn.execute(&format!(
                    "INSERT INTO docs (embedding, title) VALUES ([{:.1}, 0.0, 0.0], 'Doc {}');",
                    i as f32 * 0.1, i
                )).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all inserts
        let result = conn.execute("SELECT * FROM docs;").unwrap();
        if let ExecuteResult::Select { rows } = result {
            assert_eq!(rows.len(), 4);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_concurrent_read_write() {
        // Create Arc first
        let db = Arc::new(ConcurrentDatabase::in_memory());
        let mut conn = db.connect();

        conn.execute("CREATE TABLE docs (embedding VECTOR(3), value INTEGER);").unwrap();

        for i in 0..10 {
            conn.execute(&format!(
                "INSERT INTO docs (embedding, value) VALUES ([0.0, 0.0, 0.0], {});",
                i
            )).unwrap();
        }

        // Reader thread
        let db_reader = Arc::clone(&db);
        let reader = thread::spawn(move || {
            let mut conn = db_reader.connect();
            let result = conn.execute("SELECT * FROM docs;").unwrap();
            if let ExecuteResult::Select { rows } = result {
                rows.len()
            } else {
                0
            }
        });

        // Writer thread
        let db_writer = Arc::clone(&db);
        let writer = thread::spawn(move || {
            let mut conn = db_writer.connect();
            conn.execute("INSERT INTO docs (embedding, value) VALUES ([1.0, 0.0, 0.0], 999);").unwrap();
        });

        writer.join().unwrap();
        let read_count = reader.join().unwrap();

        assert!(read_count >= 10);
    }

    #[test]
    fn test_transaction_commit() {
        let db = ConcurrentDatabase::in_memory();
        let mut conn = db.connect();

        conn.execute("CREATE TABLE docs (embedding VECTOR(3), value INTEGER);").unwrap();

        conn.begin().unwrap();
        conn.execute("INSERT INTO docs (embedding, value) VALUES ([0.1, 0.2, 0.3], 1);").unwrap();
        conn.execute("INSERT INTO docs (embedding, value) VALUES ([0.4, 0.5, 0.6], 2);").unwrap();
        let results = conn.commit().unwrap();

        assert_eq!(results.len(), 2);

        let result = conn.execute("SELECT * FROM docs;").unwrap();
        if let ExecuteResult::Select { rows } = result {
            assert_eq!(rows.len(), 2);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_transaction_rollback() {
        let db = ConcurrentDatabase::in_memory();
        let mut conn = db.connect();

        conn.execute("CREATE TABLE docs (embedding VECTOR(3), value INTEGER);").unwrap();
        conn.execute("INSERT INTO docs (embedding, value) VALUES ([0.1, 0.2, 0.3], 1);").unwrap();

        conn.begin().unwrap();
        conn.execute("INSERT INTO docs (embedding, value) VALUES ([0.4, 0.5, 0.6], 2);").unwrap();
        conn.rollback().unwrap();

        let result = conn.execute("SELECT * FROM docs;").unwrap();
        if let ExecuteResult::Select { rows } = result {
            assert_eq!(rows.len(), 1); // Only the original row
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_database_pool() {
        let pool = DatabasePool::in_memory();
        let mut conn = pool.connect();

        conn.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();
        drop(conn); // Release the connection

        let pool_clone = pool.clone();
        let handle = thread::spawn(move || {
            let mut conn = pool_clone.connect();
            conn.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'Test');").unwrap();
        });

        handle.join().unwrap();

        let mut conn = pool.connect();
        let result = conn.execute("SELECT * FROM docs;").unwrap();
        if let ExecuteResult::Select { rows } = result {
            assert_eq!(rows.len(), 1);
        } else {
            panic!("Expected Select result");
        }
    }

    #[test]
    fn test_direct_api_concurrent() {
        let db = ConcurrentDatabase::in_memory();
        let mut conn = db.connect();

        conn.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();

        // Direct insert
        let id = conn.insert_direct("docs", vec![1.0, 0.0, 0.0], vec![
            ("title", Value::Text("Direct".to_string()))
        ]).unwrap();
        assert!(id > 0);

        // Direct search
        let results = conn.search_similar("docs", &[1.0, 0.0, 0.0], 10, 100).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_persistence() {
        let temp_path = std::env::temp_dir().join("pardusdb_concurrent_test.pardus");
        let _ = std::fs::remove_file(&temp_path);

        // Create and populate
        {
            let db = ConcurrentDatabase::open(&temp_path).unwrap();
            let mut conn = db.connect();
            conn.execute("CREATE TABLE docs (embedding VECTOR(2), title TEXT);").unwrap();
            conn.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0], 'Test');").unwrap();
            db.save().unwrap();
        }

        // Reopen and verify
        {
            let db = ConcurrentDatabase::open(&temp_path).unwrap();
            let mut conn = db.connect();
            let result = conn.execute("SELECT * FROM docs;").unwrap();
            if let ExecuteResult::Select { rows } = result {
                assert_eq!(rows.len(), 1);
            } else {
                panic!("Expected Select result");
            }
        }

        let _ = std::fs::remove_file(&temp_path);
    }
}
