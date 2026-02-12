use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{MarsError, Result};
use crate::graph::GraphConfig;
use crate::parser::{Command, ComparisonOp, ConditionValue, OrderBy, SelectColumn, WhereClause, parse};
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

/// The main database - manages multiple tables in a single file
pub struct Database {
    tables: HashMap<String, Table>,
    config: GraphConfig,
    path: Option<PathBuf>,
}

impl Database {
    /// Create an in-memory database
    pub fn in_memory() -> Self {
        Database {
            tables: HashMap::new(),
            config: GraphConfig::default(),
            path: None,
        }
    }

    /// Set graph configuration
    pub fn with_config(mut self, config: GraphConfig) -> Self {
        self.config = config;
        self
    }

    /// Open or create a database file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            Self::load(&path)
        } else {
            Self::create_new(&path)
        }
    }

    /// Create a new database file
    fn create_new(path: &Path) -> Result<Self> {
        let db = Database {
            tables: HashMap::new(),
            config: GraphConfig::default(),
            path: Some(path.to_path_buf()),
        };

        // Write empty database
        db.save()?;

        Ok(db)
    }

    /// Load database from file
    fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut reader = BufReader::new(&mut file);

        // Read header
        let mut header_buf = [0u8; 8];
        reader.read_exact(&mut header_buf)?;

        let version = u32::from_le_bytes([header_buf[0], header_buf[1], header_buf[2], header_buf[3]]);
        let table_count = u32::from_le_bytes([header_buf[4], header_buf[5], header_buf[6], header_buf[7]]);

        // Read tables
        let mut tables = HashMap::new();

        for _ in 0..table_count {
            // Read table size
            let mut size_buf = [0u8; 8];
            reader.read_exact(&mut size_buf)?;
            let size = u64::from_le_bytes(size_buf) as usize;

            // Read table data
            let mut table_buf = vec![0u8; size];
            reader.read_exact(&mut table_buf)?;

            let table_data: TableData = bincode::deserialize(&table_buf)
                .map_err(|e| MarsError::InvalidFormat(format!("Failed to deserialize table: {}", e)))?;

            // Reconstruct table
            let mut table = Table::new(table_data.schema, GraphConfig::default())?;

            // Restore rows and graph
            for row in table_data.rows {
                // Extract vector and insert into graph
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

        Ok(Database {
            tables,
            config: GraphConfig::default(),
            path: Some(path.to_path_buf()),
        })
    }

    /// Save database to file
    pub fn save(&self) -> Result<()> {
        let path = match &self.path {
            Some(p) => p,
            None => return Ok(()), // In-memory, no save needed
        };

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        let mut writer = BufWriter::new(file);

        // Write header
        let header = DbHeader {
            version: 1,
            table_count: self.tables.len() as u32,
        };
        writer.write_all(&header.version.to_le_bytes())?;
        writer.write_all(&header.table_count.to_le_bytes())?;

        // Write tables
        for table in self.tables.values() {
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

    /// Execute a SQL-like command
    pub fn execute(&mut self, sql: &str) -> Result<ExecuteResult> {
        let command = parse(sql)?;
        self.execute_command(command)
    }

    /// Direct insert without SQL parsing - much faster
    pub fn insert_direct(
        &mut self,
        table_name: &str,
        vector: Vec<f32>,
        metadata: Vec<(&str, Value)>,
    ) -> Result<u64> {
        let table = self.tables.get_mut(table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        // Build row values
        let mut row_values: Vec<Value> = table.schema.columns.iter()
            .map(|_| Value::Null)
            .collect();

        // Find vector column and set it
        for (i, col) in table.schema.columns.iter().enumerate() {
            if matches!(col.data_type, ColumnType::Vector(_)) {
                row_values[i] = Value::Vector(vector.clone());
            }
        }

        // Set metadata
        for (col_name, value) in metadata {
            if let Some(idx) = table.schema.columns.iter().position(|c| &c.name == col_name) {
                row_values[idx] = value;
            }
        }

        // Insert
        table.insert_row(row_values)
    }

    /// Direct similarity search without SQL parsing
    pub fn search_similar(
        &self,
        table_name: &str,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u64, Vec<Value>, f32)>> {
        let table = self.tables.get(table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let results = table.select_by_similarity(query, k, ef_search);

        Ok(results.into_iter()
            .map(|(row, dist)| (row.id, row.values, dist))
            .collect())
    }

    fn execute_command(&mut self, command: Command) -> Result<ExecuteResult> {
        match command {
            Command::CreateTable { name, columns } => {
                self.create_table(name, columns)
            }
            Command::DropTable { name, if_exists } => {
                self.drop_table(name, if_exists)
            }
            Command::Insert { table, columns, values } => {
                self.insert_multi(table, columns, values)
            }
            Command::Select { table, columns, where_clause, order_by, limit, offset, distinct } => {
                self.select(table, columns, where_clause.as_ref(), order_by.as_ref(), limit, offset, distinct)
            }
            Command::Update { table, assignments, where_clause } => {
                self.update(table, assignments, where_clause.as_ref())
            }
            Command::Delete { table, where_clause } => {
                self.delete(table, where_clause.as_ref())
            }
            Command::ShowTables => {
                self.show_tables()
            }
        }
    }

    fn create_table(&mut self, name: String, columns: Vec<crate::parser::ColumnDef>) -> Result<ExecuteResult> {
        if self.tables.contains_key(&name) {
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

        let table = Table::new(schema, self.config.clone())?;
        self.tables.insert(name.clone(), table);

        Ok(ExecuteResult::CreateTable { name })
    }

    fn drop_table(&mut self, name: String, if_exists: bool) -> Result<ExecuteResult> {
        if self.tables.remove(&name).is_none() {
            if if_exists {
                return Ok(ExecuteResult::DropTable { name });
            }
            return Err(MarsError::InvalidFormat(format!("Table '{}' does not exist", name)));
        }
        Ok(ExecuteResult::DropTable { name })
    }

    fn insert_multi(&mut self, table_name: String, columns: Vec<String>, values: Vec<Vec<Value>>) -> Result<ExecuteResult> {
        let table = self.tables.get_mut(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let mut last_id = 0u64;
        for row_values in values {
            last_id = table.insert(&columns, row_values)?;
        }
        Ok(ExecuteResult::Insert { id: last_id })
    }

    fn select(
        &self,
        table_name: String,
        columns: Vec<SelectColumn>,
        where_clause: Option<&WhereClause>,
        order_by: Option<&OrderBy>,
        limit: Option<usize>,
        offset: Option<usize>,
        distinct: bool,
    ) -> Result<ExecuteResult> {
        let table = self.tables.get(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        // Check for vector similarity query
        if let Some(wc) = where_clause {
            for cond in &wc.conditions {
                if cond.operator == ComparisonOp::Similar {
                    if let ConditionValue::Single(Value::Vector(query_vec)) = &cond.value {
                        let k = limit.unwrap_or(10);
                        let results = table.select_by_similarity(query_vec, k, 100);
                        return Ok(ExecuteResult::SelectSimilar { results });
                    }
                }
            }
        }

        // Check for aggregate functions
        let has_aggregates = columns.iter().any(|c| matches!(c, SelectColumn::Aggregate { .. }));
        if has_aggregates {
            return self.execute_aggregates(table, &columns, where_clause);
        }

        // Convert SelectColumn to column names
        let col_names: Vec<String> = columns.iter()
            .filter_map(|c| match c {
                SelectColumn::Column(name) => Some(name.clone()),
                _ => None,
            })
            .collect();

        let is_star = columns.iter().any(|c| matches!(c, SelectColumn::All));

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

    fn execute_aggregates(&self, table: &Table, columns: &[SelectColumn], where_clause: Option<&WhereClause>) -> Result<ExecuteResult> {
        use crate::parser::AggregateFunc;

        // Get matching rows
        let matching_rows: Vec<&Row> = table.rows.values()
            .filter(|row| table.matches_where(row, where_clause))
            .collect();

        let mut results = Vec::new();

        for col in columns {
            match col {
                SelectColumn::Aggregate { func, column, alias } => {
                    let value = match func {
                        AggregateFunc::Count => {
                            if column == "*" {
                                Value::Integer(matching_rows.len() as i64)
                            } else {
                                let idx = table.column_index(column).unwrap_or(0);
                                let count = matching_rows.iter()
                                    .filter(|r| !matches!(r.values.get(idx), Some(Value::Null) | None))
                                    .count();
                                Value::Integer(count as i64)
                            }
                        }
                        AggregateFunc::Sum => {
                            let idx = table.column_index(column).unwrap_or(0);
                            let sum: f64 = matching_rows.iter()
                                .filter_map(|r| match r.values.get(idx) {
                                    Some(Value::Integer(i)) => Some(*i as f64),
                                    Some(Value::Float(f)) => Some(*f),
                                    _ => None,
                                })
                                .sum();
                            Value::Float(sum)
                        }
                        AggregateFunc::Avg => {
                            let idx = table.column_index(column).unwrap_or(0);
                            let values: Vec<f64> = matching_rows.iter()
                                .filter_map(|r| match r.values.get(idx) {
                                    Some(Value::Integer(i)) => Some(*i as f64),
                                    Some(Value::Float(f)) => Some(*f),
                                    _ => None,
                                })
                                .collect();
                            if values.is_empty() {
                                Value::Null
                            } else {
                                Value::Float(values.iter().sum::<f64>() / values.len() as f64)
                            }
                        }
                        AggregateFunc::Min => {
                            let idx = table.column_index(column).unwrap_or(0);
                            matching_rows.iter()
                                .filter_map(|r| r.values.get(idx))
                                .filter(|v| !matches!(v, Value::Null))
                                .min_by(|a, b| table.values_compare(a, b).unwrap_or(std::cmp::Ordering::Equal))
                                .cloned()
                                .unwrap_or(Value::Null)
                        }
                        AggregateFunc::Max => {
                            let idx = table.column_index(column).unwrap_or(0);
                            matching_rows.iter()
                                .filter_map(|r| r.values.get(idx))
                                .filter(|v| !matches!(v, Value::Null))
                                .max_by(|a, b| table.values_compare(a, b).unwrap_or(std::cmp::Ordering::Equal))
                                .cloned()
                                .unwrap_or(Value::Null)
                        }
                    };

                    let name = alias.clone().unwrap_or_else(|| format!("{:?}({})", func, column));
                    results.push((name, value));
                }
                SelectColumn::Column(name) => {
                    // For non-aggregate columns in aggregate query, take first value
                    if let Some(row) = matching_rows.first() {
                        if let Some(idx) = table.column_index(name) {
                            results.push((name.clone(), row.values.get(idx).cloned().unwrap_or(Value::Null)));
                        }
                    }
                }
                SelectColumn::All => {}
            }
        }

        Ok(ExecuteResult::Aggregate { results })
    }

    fn update(
        &mut self,
        table_name: String,
        assignments: Vec<(String, Value)>,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<ExecuteResult> {
        let table = self.tables.get_mut(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let count = table.update(&assignments, where_clause)?;
        Ok(ExecuteResult::Update { count })
    }

    fn delete(
        &mut self,
        table_name: String,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<ExecuteResult> {
        let table = self.tables.get_mut(&table_name)
            .ok_or_else(|| MarsError::InvalidFormat(format!("Table '{}' does not exist", table_name)))?;

        let count = table.delete(where_clause)?;
        Ok(ExecuteResult::Delete { count })
    }

    fn show_tables(&self) -> Result<ExecuteResult> {
        let tables: Vec<TableInfo> = self.tables.values()
            .map(|t| TableInfo {
                name: t.name().to_string(),
                rows: t.len(),
                dimension: t.schema.get_vector_dimension().unwrap_or(0),
            })
            .collect();

        Ok(ExecuteResult::ShowTables { tables })
    }

    /// Get table names
    pub fn table_names(&self) -> Vec<&str> {
        self.tables.keys().map(|s| s.as_str()).collect()
    }

    /// Get table by name
    pub fn get_table(&self, name: &str) -> Option<&Table> {
        self.tables.get(name)
    }
}

/// Result of executing a command
#[derive(Debug)]
pub enum ExecuteResult {
    CreateTable { name: String },
    DropTable { name: String },
    Insert { id: u64 },
    Select { rows: Vec<Row> },
    SelectSimilar { results: Vec<(Row, f32)> },
    Aggregate { results: Vec<(String, Value)> },
    Update { count: usize },
    Delete { count: usize },
    ShowTables { tables: Vec<TableInfo> },
}

/// Table information
#[derive(Debug, Clone)]
pub struct TableInfo {
    pub name: String,
    pub rows: usize,
    pub dimension: usize,
}

impl std::fmt::Display for ExecuteResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecuteResult::CreateTable { name } => write!(f, "Table '{}' created", name),
            ExecuteResult::DropTable { name } => write!(f, "Table '{}' dropped", name),
            ExecuteResult::Insert { id } => write!(f, "Inserted row with id={}", id),
            ExecuteResult::Select { rows } => {
                writeln!(f, "Found {} rows:", rows.len())?;
                for row in rows {
                    writeln!(f, "  id={}, values={:?}", row.id, row.values)?;
                }
                Ok(())
            }
            ExecuteResult::SelectSimilar { results } => {
                writeln!(f, "Found {} similar rows:", results.len())?;
                for (row, dist) in results {
                    writeln!(f, "  id={}, distance={:.4}, values={:?}", row.id, dist, row.values)?;
                }
                Ok(())
            }
            ExecuteResult::Aggregate { results } => {
                writeln!(f, "Aggregate results:")?;
                for (name, value) in results {
                    writeln!(f, "  {} = {:?}", name, value)?;
                }
                Ok(())
            }
            ExecuteResult::Update { count } => write!(f, "Updated {} rows", count),
            ExecuteResult::Delete { count } => write!(f, "Deleted {} rows", count),
            ExecuteResult::ShowTables { tables } => {
                writeln!(f, "Tables ({}):", tables.len())?;
                for t in tables {
                    writeln!(f, "  {} ({} rows, dim={})", t.name, t.rows, t.dimension)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_table() {
        let mut db = Database::in_memory();

        let result = db.execute(
            "CREATE TABLE docs (id INTEGER, embedding VECTOR(128), title TEXT);"
        ).unwrap();

        assert!(matches!(result, ExecuteResult::CreateTable { .. }));
        assert!(db.get_table("docs").is_some());
    }

    #[test]
    fn test_insert_and_select() {
        let mut db = Database::in_memory();

        db.execute("CREATE TABLE docs (id INTEGER, embedding VECTOR(3), title TEXT);").unwrap();

        db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'First');").unwrap();
        db.execute("INSERT INTO docs (embedding, title) VALUES ([0.0, 1.0, 0.0], 'Second');").unwrap();

        let result = db.execute("SELECT * FROM docs;").unwrap();
        assert!(matches!(result, ExecuteResult::Select { rows } if rows.len() == 2));
    }

    #[test]
    fn test_delete() {
        let mut db = Database::in_memory();

        db.execute("CREATE TABLE docs (id INTEGER, embedding VECTOR(3), title TEXT);").unwrap();
        db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'Test');").unwrap();

        let result = db.execute("DELETE FROM docs WHERE id = 1;").unwrap();
        assert!(matches!(result, ExecuteResult::Delete { count: 1 }));
    }
}
