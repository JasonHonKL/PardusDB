use std::collections::HashMap;
use std::sync::Arc;

use crate::distance::{Distance, Euclidean, Numeric};
use crate::error::{MarsError, Result};
use crate::graph::{Graph, GraphConfig};
use crate::node::NodeId;
use crate::schema::{Column, ColumnType, Row, Schema, Value};

/// A table in the database containing vectors and metadata
pub struct Table {
    pub schema: Schema,
    pub graph: Graph<f32, Euclidean>,
    pub(crate) rows: HashMap<u64, Row>,
    pub(crate) next_id: u64,
}

impl Table {
    pub fn new(schema: Schema, config: GraphConfig) -> Result<Self> {
        let dimension = schema.get_vector_dimension()
            .ok_or_else(|| MarsError::InvalidConfig("Table must have a VECTOR column".into()))?;

        Ok(Table {
            schema,
            graph: Graph::new(dimension, config),
            rows: HashMap::new(),
            next_id: 1,
        })
    }

    /// Get the table name
    pub fn name(&self) -> &str {
        &self.schema.name
    }

    /// Get the number of rows
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Insert a row with values
    pub fn insert(&mut self, columns: &[String], values: Vec<Value>) -> Result<u64> {
        // Validate and build complete row
        let row_values = self.build_row_values(columns, values)?;
        self.insert_row(row_values)
    }

    /// Insert a row with pre-built values (faster, no validation)
    pub fn insert_row(&mut self, mut row_values: Vec<Value>) -> Result<u64> {

        // Auto-generate ID
        let id = self.next_id;
        self.next_id += 1;

        // If there's an 'id' column, set it to the auto-generated ID
        if let Some(idx) = self.column_index("id") {
            row_values[idx] = Value::Integer(id as i64);
        }

        // Extract vector
        let vector = self.extract_vector(&row_values)?;

        // Insert into graph
        let _graph_id = self.graph.insert(vector);

        // Create row
        let row = Row::new(id, row_values);
        self.rows.insert(id, row);

        Ok(id)
    }

    /// Select rows matching conditions
    pub fn select(
        &self,
        columns: &[String],
        where_clause: Option<&crate::parser::WhereClause>,
        limit: Option<usize>,
    ) -> Vec<Row> {
        let mut results: Vec<&Row> = self.rows.values()
            .filter(|row| self.matches_where(row, where_clause))
            .collect();

        // Apply limit
        if let Some(n) = limit {
            results.truncate(n);
        }

        // Project columns
        results.into_iter()
            .map(|row| self.project_row(row, columns))
            .collect()
    }

    /// Select by vector similarity
    pub fn select_by_similarity(
        &self,
        query_vector: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Vec<(Row, f32)> {
        let candidates = self.graph.query(query_vector, k, ef_search);

        candidates.into_iter()
            .filter_map(|c| {
                // Graph node ID corresponds to row ID - 1 (first insert gets graph_id=0, row_id=1)
                let row_id = (c.id as u64) + 1;
                self.rows.get(&row_id).map(|row| {
                    (self.project_row(row, &[]), c.distance)
                })
            })
            .collect()
    }

    /// Update rows matching conditions
    pub fn update(
        &mut self,
        assignments: &[(String, Value)],
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<usize> {
        let matching_ids: Vec<u64> = self.rows.values()
            .filter(|row| self.matches_where(row, where_clause))
            .map(|row| row.id)
            .collect();

        // Precompute column indices
        let assignment_indices: Vec<(Option<usize>, Value)> = assignments.iter()
            .map(|(col_name, value)| (self.column_index(col_name), value.clone()))
            .collect();

        let count = matching_ids.len();

        for id in matching_ids {
            if let Some(row) = self.rows.get_mut(&id) {
                for (idx_opt, value) in &assignment_indices {
                    if let Some(idx) = idx_opt {
                        row.values[*idx] = value.clone();
                    }
                }
            }
        }

        Ok(count)
    }

    /// Delete rows matching conditions
    pub fn delete(
        &mut self,
        where_clause: Option<&crate::parser::WhereClause>,
    ) -> Result<usize> {
        let matching_ids: Vec<u64> = self.rows.values()
            .filter(|row| self.matches_where(row, where_clause))
            .map(|row| row.id)
            .collect();

        let count = matching_ids.len();

        for id in &matching_ids {
            self.rows.remove(id);
            // Note: We should also delete from graph, but need to map row ID to graph ID
            let graph_id = (*id - 1) as NodeId;
            self.graph.delete(graph_id);
        }

        Ok(count)
    }

    /// Get a row by ID
    pub fn get(&self, id: u64) -> Option<&Row> {
        self.rows.get(&id)
    }

    /// Build row values from column names and provided values
    fn build_row_values(&self, columns: &[String], values: Vec<Value>) -> Result<Vec<Value>> {
        let mut row_values: Vec<Value> = self.schema.columns.iter()
            .map(|_| Value::Null)
            .collect();

        for (i, col_name) in columns.iter().enumerate() {
            let idx = self.column_index(col_name)
                .ok_or_else(|| MarsError::InvalidFormat(format!("Unknown column: {}", col_name)))?;

            row_values[idx] = values.get(i)
                .ok_or_else(|| MarsError::InvalidFormat(format!("Missing value for column: {}", col_name)))?
                .clone();
        }

        Ok(row_values)
    }

    /// Extract vector from row values
    fn extract_vector(&self, values: &[Value]) -> Result<Vec<f32>> {
        let vec_col = self.schema.vector_column.as_ref()
            .ok_or_else(|| MarsError::InvalidConfig("No vector column defined".into()))?;

        let idx = self.column_index(vec_col)
            .ok_or_else(|| MarsError::InvalidConfig("Vector column not found".into()))?;

        match &values[idx] {
            Value::Vector(v) => Ok(v.clone()),
            _ => Err(MarsError::InvalidFormat("Vector column must contain a vector".into())),
        }
    }

    /// Get column index by name
    fn column_index(&self, name: &str) -> Option<usize> {
        self.schema.columns.iter().position(|c| c.name == name)
    }

    /// Check if a row matches where clause
    fn matches_where(&self, row: &Row, where_clause: Option<&crate::parser::WhereClause>) -> bool {
        match where_clause {
            None => true,
            Some(wc) => wc.conditions.iter().all(|cond| {
                let idx = match self.column_index(&cond.column) {
                    Some(i) => i,
                    None => return false,
                };

                let row_val = &row.values[idx];
                self.compare_values(row_val, &cond.operator, &cond.value)
            }),
        }
    }

    /// Compare two values
    fn compare_values(&self, a: &Value, op: &crate::parser::ComparisonOp, b: &Value) -> bool {
        use crate::parser::ComparisonOp::*;

        match op {
            Eq => self.values_equal(a, b),
            Ne => !self.values_equal(a, b),
            Lt => self.values_compare(a, b) == Some(std::cmp::Ordering::Less),
            Le => self.values_compare(a, b).map(|o| o != std::cmp::Ordering::Greater).unwrap_or(false),
            Gt => self.values_compare(a, b) == Some(std::cmp::Ordering::Greater),
            Ge => self.values_compare(a, b).map(|o| o != std::cmp::Ordering::Less).unwrap_or(false),
            Similar => false, // Similarity is handled separately
        }
    }

    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::Integer(i1), Value::Integer(i2)) => i1 == i2,
            (Value::Float(f1), Value::Float(f2)) => f1 == f2,
            (Value::Text(s1), Value::Text(s2)) => s1 == s2,
            (Value::Boolean(b1), Value::Boolean(b2)) => b1 == b2,
            (Value::Integer(i), Value::Float(f)) => (*i as f64) == *f,
            (Value::Float(f), Value::Integer(i)) => *f == (*i as f64),
            _ => false,
        }
    }

    fn values_compare(&self, a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
        match (a, b) {
            (Value::Integer(i1), Value::Integer(i2)) => i1.partial_cmp(i2),
            (Value::Float(f1), Value::Float(f2)) => f1.partial_cmp(f2),
            (Value::Text(s1), Value::Text(s2)) => s1.partial_cmp(s2),
            (Value::Integer(i), Value::Float(f)) => (*i as f64).partial_cmp(f),
            (Value::Float(f), Value::Integer(i)) => f.partial_cmp(&(*i as f64)),
            _ => None,
        }
    }

    /// Project row to specified columns
    fn project_row(&self, row: &Row, columns: &[String]) -> Row {
        if columns.is_empty() {
            return row.clone(); // SELECT *
        }

        let values: Vec<Value> = columns.iter()
            .filter_map(|name| {
                self.column_index(name).map(|idx| row.values[idx].clone())
            })
            .collect();

        Row::new(row.id, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ColumnType;

    fn create_test_schema() -> Schema {
        Schema::new("test")
            .column("id", ColumnType::Integer)
            .column("embedding", ColumnType::Vector(3))
            .column("title", ColumnType::Text)
    }

    #[test]
    fn test_table_creation() {
        let schema = create_test_schema();
        let table = Table::new(schema, GraphConfig::default()).unwrap();

        assert_eq!(table.name(), "test");
        assert!(table.is_empty());
    }

    #[test]
    fn test_insert() {
        let schema = create_test_schema();
        let mut table = Table::new(schema, GraphConfig::default()).unwrap();

        let id = table.insert(
            &["embedding".to_string(), "title".to_string()],
            vec![Value::Vector(vec![1.0, 2.0, 3.0]), Value::Text("Test".to_string())],
        ).unwrap();

        assert_eq!(id, 1);
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_select() {
        let schema = create_test_schema();
        let mut table = Table::new(schema, GraphConfig::default()).unwrap();

        table.insert(
            &["embedding".to_string(), "title".to_string()],
            vec![Value::Vector(vec![1.0, 0.0, 0.0]), Value::Text("First".to_string())],
        ).unwrap();

        table.insert(
            &["embedding".to_string(), "title".to_string()],
            vec![Value::Vector(vec![0.0, 1.0, 0.0]), Value::Text("Second".to_string())],
        ).unwrap();

        let rows = table.select(&[], None, None);
        assert_eq!(rows.len(), 2);
    }
}
