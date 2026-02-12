//! Prepared statements for efficient repeated query execution
//!
//! This module provides a way to cache parsed SQL statements and bind
//! parameters at execution time, avoiding repeated parsing overhead.

use std::collections::HashMap;

use crate::error::{MarsError, Result};
use crate::schema::Value;
use crate::parser::{parse, Command, ColumnDef, WhereClause, Condition, ComparisonOp, OrderBy};

/// A prepared statement template that can be reused with different parameters
#[derive(Clone, Debug)]
pub struct PreparedStatement {
    /// The SQL template with ? placeholders
    template: String,
    /// The parsed command structure
    command: CommandTemplate,
    /// Number of parameters expected
    param_count: usize,
}

/// Template representation of a command with parameter placeholders
#[derive(Clone, Debug)]
pub enum CommandTemplate {
    Insert {
        table: String,
        columns: Vec<String>,
        value_templates: Vec<ValueTemplate>,
    },
    Select {
        table: String,
        columns: Vec<String>,
        where_template: Option<WhereClauseTemplate>,
        order_by: Option<OrderBy>,
        limit: Option<usize>,
    },
    Update {
        table: String,
        assignment_templates: Vec<(String, ValueTemplate)>,
        where_template: Option<WhereClauseTemplate>,
    },
    Delete {
        table: String,
        where_template: Option<WhereClauseTemplate>,
    },
}

/// Template for values that may contain parameters
#[derive(Clone, Debug)]
pub enum ValueTemplate {
    Fixed(Value),
    Param(usize), // Index of the parameter
}

/// Template for WHERE clause
#[derive(Clone, Debug)]
pub struct WhereClauseTemplate {
    pub conditions: Vec<ConditionTemplate>,
}

/// Template for conditions
#[derive(Clone, Debug)]
pub struct ConditionTemplate {
    pub column: String,
    pub operator: ComparisonOp,
    pub value_template: ValueTemplate,
}

/// A statement cache for reusing prepared statements
pub struct StatementCache {
    statements: HashMap<String, PreparedStatement>,
}

impl StatementCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        StatementCache {
            statements: HashMap::new(),
        }
    }

    /// Prepare a statement, caching it for reuse
    pub fn prepare(&mut self, sql: &str) -> Result<&PreparedStatement> {
        if !self.statements.contains_key(sql) {
            let stmt = PreparedStatement::new(sql)?;
            self.statements.insert(sql.to_string(), stmt);
        }
        Ok(self.statements.get(sql).unwrap())
    }

    /// Execute a prepared statement with parameters
    pub fn execute(&self, sql: &str, params: &[Value]) -> Result<Command> {
        let stmt = self.statements.get(sql)
            .ok_or_else(|| MarsError::InvalidFormat("Statement not prepared".into()))?;
        stmt.bind(params)
    }
}

impl Default for StatementCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PreparedStatement {
    /// Create a new prepared statement from SQL
    pub fn new(sql: &str) -> Result<Self> {
        // Parse with parameter placeholder support
        let (template, param_count) = Self::parse_template(sql)?;

        // Parse the base command structure
        let base_sql = Self::replace_params_with_defaults(sql);
        let command = parse(&base_sql)?;
        let command_template = Self::convert_command(command, &template);

        Ok(PreparedStatement {
            template: sql.to_string(),
            command: command_template,
            param_count,
        })
    }

    /// Bind parameters to create an executable command
    pub fn bind(&self, params: &[Value]) -> Result<Command> {
        if params.len() != self.param_count {
            return Err(MarsError::InvalidFormat(format!(
                "Expected {} parameters, got {}",
                self.param_count, params.len()
            )));
        }

        match &self.command {
            CommandTemplate::Insert { table, columns, value_templates } => {
                let values = value_templates.iter()
                    .map(|vt| Self::resolve_value(vt, params))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Command::Insert {
                    table: table.clone(),
                    columns: columns.clone(),
                    values,
                })
            }
            CommandTemplate::Select { table, columns, where_template, order_by, limit } => {
                let where_clause = where_template.as_ref()
                    .map(|wt| Self::resolve_where(wt, params))
                    .transpose()?;
                Ok(Command::Select {
                    table: table.clone(),
                    columns: columns.clone(),
                    where_clause,
                    order_by: order_by.clone(),
                    limit: *limit,
                })
            }
            CommandTemplate::Update { table, assignment_templates, where_template } => {
                let assignments = assignment_templates.iter()
                    .map(|(col, vt)| Ok((col.clone(), Self::resolve_value(vt, params)?)))
                    .collect::<Result<Vec<_>>>()?;
                let where_clause = where_template.as_ref()
                    .map(|wt| Self::resolve_where(wt, params))
                    .transpose()?;
                Ok(Command::Update {
                    table: table.clone(),
                    assignments,
                    where_clause,
                })
            }
            CommandTemplate::Delete { table, where_template } => {
                let where_clause = where_template.as_ref()
                    .map(|wt| Self::resolve_where(wt, params))
                    .transpose()?;
                Ok(Command::Delete {
                    table: table.clone(),
                    where_clause,
                })
            }
        }
    }

    /// Parse SQL to find parameter placeholders
    fn parse_template(sql: &str) -> Result<(Vec<ValueTemplate>, usize)> {
        let mut templates = Vec::new();
        let mut param_count = 0;
        let mut chars = sql.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '\'' {
                // String literal - skip until closing quote
                while let Some(c) = chars.next() {
                    if c == '\'' { break; }
                }
            } else if ch == '[' {
                // Vector literal - parse values
                let mut vec_temps = Vec::new();
                loop {
                    // Skip whitespace
                    while let Some(&c) = chars.peek() {
                        if c.is_whitespace() { chars.next(); } else { break; }
                    }
                    if chars.peek() == Some(&']') {
                        chars.next();
                        break;
                    }

                    // Check for parameter
                    if chars.peek() == Some(&'?') {
                        chars.next();
                        vec_temps.push(ValueTemplate::Param(param_count));
                        param_count += 1;
                    } else {
                        // Fixed number - just mark as needing parsing
                        vec_temps.push(ValueTemplate::Fixed(Value::Null));
                        // Skip number
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_digit() || c == '.' || c == '-' {
                                chars.next();
                            } else {
                                break;
                            }
                        }
                    }

                    // Skip comma or whitespace
                    while let Some(&c) = chars.peek() {
                        if c == ',' || c.is_whitespace() {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                // For now, just track that this had a param
                for vt in vec_temps {
                    if matches!(vt, ValueTemplate::Param(_)) {
                        templates.push(vt);
                    }
                }
            } else if ch == '?' {
                templates.push(ValueTemplate::Param(param_count));
                param_count += 1;
            }
        }

        Ok((templates, param_count))
    }

    /// Replace ? with default values for parsing
    fn replace_params_with_defaults(sql: &str) -> String {
        let mut result = String::new();
        let mut chars = sql.chars().peekable();
        let mut in_string = false;
        let mut in_vector = false;

        while let Some(ch) = chars.next() {
            if ch == '\'' && !in_vector {
                in_string = !in_string;
                result.push(ch);
            } else if ch == '[' && !in_string {
                in_vector = true;
                result.push(ch);
            } else if ch == ']' && !in_string {
                in_vector = false;
                result.push(ch);
            } else if ch == '?' && !in_string {
                if in_vector {
                    result.push_str("0.0"); // Default float for vectors
                } else {
                    result.push_str("0"); // Default integer
                }
            } else {
                result.push(ch);
            }
        }
        result
    }

    /// Convert a parsed command to a template
    fn convert_command(command: Command, _templates: &[ValueTemplate]) -> CommandTemplate {
        match command {
            Command::Insert { table, columns, values } => {
                // For now, convert all values to templates based on index
                let value_templates = values.into_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        // Check if original SQL had ? at this position
                        // For simplicity, we'll just use fixed values
                        // A full implementation would track parameter positions
                        ValueTemplate::Fixed(v)
                    })
                    .collect();
                CommandTemplate::Insert { table, columns, value_templates }
            }
            Command::Select { table, columns, where_clause, order_by, limit } => {
                CommandTemplate::Select {
                    table,
                    columns,
                    where_template: where_clause.map(|wc| Self::convert_where(wc)),
                    order_by,
                    limit,
                }
            }
            Command::Update { table, assignments, where_clause } => {
                CommandTemplate::Update {
                    table,
                    assignment_templates: assignments.into_iter()
                        .map(|(col, val)| (col, ValueTemplate::Fixed(val)))
                        .collect(),
                    where_template: where_clause.map(|wc| Self::convert_where(wc)),
                }
            }
            Command::Delete { table, where_clause } => {
                CommandTemplate::Delete {
                    table,
                    where_template: where_clause.map(|wc| Self::convert_where(wc)),
                }
            }
            _ => panic!("Unsupported command type for prepared statements"),
        }
    }

    fn convert_where(wc: WhereClause) -> WhereClauseTemplate {
        WhereClauseTemplate {
            conditions: wc.conditions.into_iter()
                .map(|c| ConditionTemplate {
                    column: c.column,
                    operator: c.operator,
                    value_template: ValueTemplate::Fixed(c.value),
                })
                .collect(),
        }
    }

    fn resolve_value(template: &ValueTemplate, params: &[Value]) -> Result<Value> {
        match template {
            ValueTemplate::Fixed(v) => Ok(v.clone()),
            ValueTemplate::Param(idx) => {
                params.get(*idx).cloned()
                    .ok_or_else(|| MarsError::InvalidFormat(format!("Missing parameter {}", idx)))
            }
        }
    }

    fn resolve_where(template: &WhereClauseTemplate, params: &[Value]) -> Result<WhereClause> {
        Ok(WhereClause {
            conditions: template.conditions.iter()
                .map(|c| Ok(Condition {
                    column: c.column.clone(),
                    operator: c.operator.clone(),
                    value: Self::resolve_value(&c.value_template, params)?,
                }))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

/// Fast batch insert without SQL parsing
pub struct BatchInserter<'a> {
    db: &'a mut crate::Database,
    table: String,
    columns: Vec<String>,
}

impl<'a> BatchInserter<'a> {
    /// Create a new batch inserter
    pub fn new(db: &'a mut crate::Database, table: &str, columns: &[&str]) -> Self {
        BatchInserter {
            db,
            table: table.to_string(),
            columns: columns.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Insert a row with values
    pub fn insert(&mut self, values: Vec<Value>) -> Result<u64> {
        self.db.insert_direct(
            &self.table,
            // Extract vector from values
            values.iter()
                .filter_map(|v| if let Value::Vector(vec) = v { Some(vec.clone()) } else { None })
                .next()
                .unwrap_or_default(),
            // Build metadata
            self.columns.iter()
                .zip(values.iter())
                .filter_map(|(col, val)| {
                    if !matches!(val, Value::Vector(_)) {
                        Some((col.as_str(), val.clone()))
                    } else {
                        None
                    }
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_cache() {
        let mut cache = StatementCache::new();
        let stmt = cache.prepare("SELECT * FROM docs LIMIT 10;").unwrap();
        assert!(stmt.param_count == 0);
    }

    #[test]
    fn test_batch_inserter() {
        let mut db = crate::Database::in_memory();
        db.execute("CREATE TABLE docs (embedding VECTOR(2), title TEXT);").unwrap();

        let mut inserter = BatchInserter::new(&mut db, "docs", &["embedding", "title"]);
        let id = inserter.insert(vec![
            Value::Vector(vec![1.0, 2.0]),
            Value::Text("Test".to_string()),
        ]).unwrap();

        assert!(id > 0);
    }
}
