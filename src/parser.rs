//! Enhanced SQL parser for PardusDB
//!
//! Supports a comprehensive subset of SQL including:
//! - CREATE TABLE, DROP TABLE
//! - INSERT (single and multi-row)
//! - SELECT with WHERE, ORDER BY, LIMIT, OFFSET, DISTINCT
//! - UPDATE, DELETE
//! - Aggregate functions: COUNT, SUM, AVG, MIN, MAX
//! - LIKE, IN, BETWEEN, IS NULL, IS NOT NULL
//! - AND, OR in WHERE clauses

use crate::error::{MarsError, Result};
use crate::schema::{ColumnType, Value};

/// SQL command types
#[derive(Clone, Debug)]
pub enum Command {
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
    },
    DropTable {
        name: String,
        if_exists: bool,
    },
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Vec<Value>>,  // Support multiple rows
    },
    Select {
        table: String,
        columns: Vec<SelectColumn>,
        where_clause: Option<WhereClause>,
        group_by: Option<Vec<String>>,  // NEW: GROUP BY columns
        having: Option<WhereClause>,    // NEW: HAVING clause
        order_by: Option<OrderBy>,
        limit: Option<usize>,
        offset: Option<usize>,
        distinct: bool,
    },
    Join {
        left_table: String,
        right_table: String,
        join_type: JoinType,
        left_column: String,   // Column from left table
        right_column: String,  // Column from right table
        columns: Vec<JoinColumn>,  // Columns to select
        where_clause: Option<WhereClause>,
        order_by: Option<OrderBy>,
        limit: Option<usize>,
        offset: Option<usize>,
    },
    Update {
        table: String,
        assignments: Vec<(String, Value)>,
        where_clause: Option<WhereClause>,
    },
    Delete {
        table: String,
        where_clause: Option<WhereClause>,
    },
    ShowTables,
}

/// JOIN types
#[derive(Clone, Debug, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
}

/// Column selection for JOIN queries
#[derive(Clone, Debug)]
pub enum JoinColumn {
    All,                              // *
    TableColumn { table: String, column: String },  // table.column
}

/// Column selection - either a regular column or an aggregate function
#[derive(Clone, Debug)]
pub enum SelectColumn {
    All,                           // *
    Column(String),                // column_name
    Aggregate { func: AggregateFunc, column: String, alias: Option<String> },
}

/// Aggregate function types
#[derive(Clone, Debug, PartialEq)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Clone, Debug)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: ColumnType,
    pub primary_key: bool,
    pub not_null: bool,
    pub unique: bool,  // NEW: UNIQUE constraint
    pub default: Option<Value>,
}

#[derive(Clone, Debug, Default)]
pub struct WhereClause {
    pub conditions: Vec<Condition>,
    pub connectors: Vec<BoolConnector>,  // AND/OR between conditions
}

#[derive(Clone, Debug)]
pub enum BoolConnector {
    And,
    Or,
}

#[derive(Clone, Debug)]
pub struct Condition {
    pub column: String,
    pub operator: ComparisonOp,
    pub value: ConditionValue,
}

#[derive(Clone, Debug)]
pub enum ConditionValue {
    Single(Value),
    List(Vec<Value>),       // For IN clause
    Range(Value, Value),    // For BETWEEN
    NullCheck,              // For IS NULL / IS NOT NULL
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Similar,    // Vector similarity
    Like,       // Pattern matching
    NotLike,
    In,         // IN clause
    NotIn,
    Between,    // BETWEEN
    NotBetween,
    IsNull,     // IS NULL
    IsNotNull,  // IS NOT NULL
}

#[derive(Clone, Debug)]
pub struct OrderBy {
    pub column: String,
    pub ascending: bool,
}

/// High-performance SQL parser
pub struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(input: &'a str) -> Self {
        Parser { input, pos: 0 }
    }

    pub fn parse(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let keyword = self.read_keyword_upper()?;

        match keyword.as_str() {
            "CREATE" => self.parse_create(),
            "DROP" => self.parse_drop(),
            "INSERT" => self.parse_insert(),
            "SELECT" => self.parse_select(),
            "UPDATE" => self.parse_update(),
            "DELETE" => self.parse_delete(),
            "SHOW" => self.parse_show(),
            _ => Err(MarsError::InvalidFormat(format!("Unknown command: {}", keyword))),
        }
    }

    // ==================== CREATE TABLE ====================
    fn parse_create(&mut self) -> Result<Command> {
        self.expect_keyword("TABLE")?;
        self.skip_whitespace();
        let name = self.read_identifier()?;

        self.skip_whitespace();
        self.expect_char('(')?;

        let mut columns = Vec::new();
        loop {
            self.skip_whitespace();
            let col_name = self.read_identifier()?;
            self.skip_whitespace();
            let col_type = self.parse_column_type()?;
            self.skip_whitespace();

            let mut primary_key = false;
            let mut not_null = false;
            let mut unique = false;
            let mut default = None;

            loop {
                let keyword = self.peek_keyword_upper();
                match keyword.as_str() {
                    "PRIMARY" => {
                        self.read_keyword()?;
                        self.expect_keyword("KEY")?;
                        primary_key = true;
                    }
                    "NOT" => {
                        self.read_keyword()?;
                        self.expect_keyword("NULL")?;
                        not_null = true;
                    }
                    "UNIQUE" => {
                        self.read_keyword()?;
                        unique = true;
                    }
                    "DEFAULT" => {
                        self.read_keyword()?;
                        self.skip_whitespace();
                        default = Some(self.parse_value()?);
                    }
                    _ => break,
                }
                self.skip_whitespace();
            }

            columns.push(ColumnDef {
                name: col_name,
                data_type: col_type,
                primary_key,
                not_null,
                unique,
                default,
            });

            self.skip_whitespace();
            if self.peek_char() == Some(')') {
                self.advance();
                break;
            }
            self.expect_char(',')?;
        }

        self.skip_trailing_semicolon();
        Ok(Command::CreateTable { name, columns })
    }

    // ==================== DROP TABLE ====================
    fn parse_drop(&mut self) -> Result<Command> {
        self.expect_keyword("TABLE")?;
        self.skip_whitespace();

        let if_exists = if self.peek_keyword_upper() == "IF" {
            self.read_keyword()?;
            self.expect_keyword("EXISTS")?;
            self.skip_whitespace();
            true
        } else {
            false
        };

        let name = self.read_identifier()?;
        self.skip_trailing_semicolon();

        Ok(Command::DropTable { name, if_exists })
    }

    // ==================== INSERT ====================
    fn parse_insert(&mut self) -> Result<Command> {
        self.expect_keyword("INTO")?;
        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();
        let mut columns = Vec::new();

        if self.peek_char() == Some('(') {
            self.advance();
            loop {
                self.skip_whitespace();
                columns.push(self.read_identifier()?);
                self.skip_whitespace();
                if self.peek_char() == Some(')') {
                    self.advance();
                    break;
                }
                self.expect_char(',')?;
            }
        }

        self.skip_whitespace();
        self.expect_keyword("VALUES")?;

        let mut all_values = Vec::new();
        loop {
            self.skip_whitespace();
            self.expect_char('(')?;

            let mut values = Vec::new();
            loop {
                self.skip_whitespace();
                values.push(self.parse_value()?);
                self.skip_whitespace();
                if self.peek_char() == Some(')') {
                    self.advance();
                    break;
                }
                self.expect_char(',')?;
            }
            all_values.push(values);

            self.skip_whitespace();
            if self.peek_char() == Some(',') {
                self.advance();
                continue;
            }
            break;
        }

        self.skip_trailing_semicolon();
        Ok(Command::Insert { table, columns, values: all_values })
    }

    // ==================== SELECT ====================
    fn parse_select(&mut self) -> Result<Command> {
        self.skip_whitespace();

        // DISTINCT
        let mut distinct = false;
        if self.peek_keyword_upper() == "DISTINCT" {
            self.read_keyword()?;
            distinct = true;
            self.skip_whitespace();
        }

        // Columns - could be SelectColumn or JoinColumn depending on if JOIN is present
        let mut select_columns = Vec::new();
        let mut join_columns = Vec::new();
        let is_star = self.peek_char() == Some('*');

        if is_star {
            self.advance();
            select_columns.push(SelectColumn::All);
            join_columns.push(JoinColumn::All);
        } else {
            loop {
                self.skip_whitespace();
                // Check if it's a table.column format (for JOIN)
                let col = self.read_identifier()?;
                self.skip_whitespace();

                if self.peek_char() == Some('.') {
                    // It's table.column format
                    self.advance(); // consume '.'
                    self.skip_whitespace();
                    let column_name = self.read_identifier()?;
                    join_columns.push(JoinColumn::TableColumn {
                        table: col.clone(),
                        column: column_name.clone(),
                    });
                    // Also add as SelectColumn for non-JOIN case
                    select_columns.push(SelectColumn::Column(column_name));
                } else {
                    // Regular column
                    // Check if it's an aggregate function
                    let col_upper = col.to_uppercase();
                    if ["COUNT", "SUM", "AVG", "MIN", "MAX"].contains(&col_upper.as_str()) {
                        // Parse aggregate function
                        self.expect_char('(')?;
                        self.skip_whitespace();
                        let agg_col = if self.peek_char() == Some('*') {
                            self.advance();
                            "*".to_string()
                        } else {
                            self.read_identifier()?
                        };
                        self.skip_whitespace();
                        self.expect_char(')')?;

                        select_columns.push(SelectColumn::Aggregate {
                            func: match col_upper.as_str() {
                                "COUNT" => AggregateFunc::Count,
                                "SUM" => AggregateFunc::Sum,
                                "AVG" => AggregateFunc::Avg,
                                "MIN" => AggregateFunc::Min,
                                "MAX" => AggregateFunc::Max,
                                _ => return Err(MarsError::InvalidFormat(format!("Unknown aggregate: {}", col))),
                            },
                            column: agg_col,
                            alias: None,
                        });
                    } else {
                        select_columns.push(SelectColumn::Column(col));
                    }
                }

                self.skip_whitespace();
                if self.peek_char() == Some(',') {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.skip_whitespace();
        self.expect_keyword("FROM")?;
        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();

        // Check for JOIN
        let join_keyword = self.peek_keyword_upper();
        if ["INNER", "LEFT", "RIGHT", "JOIN"].contains(&join_keyword.as_str()) {
            return self.parse_join(table, join_columns);
        }

        // Regular SELECT without JOIN
        let where_clause = self.parse_where()?;

        // GROUP BY
        self.skip_whitespace();
        let group_by = self.parse_group_by()?;

        // HAVING
        self.skip_whitespace();
        let having = if group_by.is_some() {
            self.parse_having()?
        } else {
            None
        };

        self.skip_whitespace();
        let order_by = self.parse_order_by()?;

        self.skip_whitespace();
        let limit = self.parse_limit()?;

        self.skip_whitespace();
        let offset = self.parse_offset()?;

        self.skip_trailing_semicolon();

        Ok(Command::Select {
            table,
            columns: select_columns,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
            distinct,
        })
    }

    /// Parse JOIN clause (called from parse_select when JOIN is detected)
    fn parse_join(&mut self, left_table: String, columns: Vec<JoinColumn>) -> Result<Command> {
        self.skip_whitespace();

        // Parse join type
        let join_type = match self.peek_keyword_upper().as_str() {
            "INNER" => {
                self.read_keyword()?;
                self.skip_whitespace();
                self.expect_keyword("JOIN")?;
                JoinType::Inner
            }
            "LEFT" => {
                self.read_keyword()?;
                self.skip_whitespace();
                self.expect_keyword("JOIN")?;
                JoinType::Left
            }
            "RIGHT" => {
                self.read_keyword()?;
                self.skip_whitespace();
                self.expect_keyword("JOIN")?;
                JoinType::Right
            }
            "JOIN" => {
                self.read_keyword()?;
                JoinType::Inner
            }
            _ => return Err(MarsError::InvalidFormat("Expected JOIN type".into())),
        };

        self.skip_whitespace();
        let right_table = self.read_identifier()?;

        self.skip_whitespace();
        self.expect_keyword("ON")?;

        self.skip_whitespace();
        // Parse ON condition: table1.column = table2.column
        let left_col_table = self.read_identifier()?;
        self.skip_whitespace();
        self.expect_char('.')?;
        self.skip_whitespace();
        let left_column = self.read_identifier()?;

        self.skip_whitespace();
        self.expect_char('=')?;

        self.skip_whitespace();
        let right_col_table = self.read_identifier()?;
        self.skip_whitespace();
        self.expect_char('.')?;
        self.skip_whitespace();
        let right_column = self.read_identifier()?;

        // Validate that the tables in ON clause match our tables
        let (left_col, right_col) = if left_col_table.to_lowercase() == left_table.to_lowercase() {
            (left_column, right_column)
        } else {
            (right_column, left_column)
        };

        self.skip_whitespace();
        let where_clause = self.parse_where()?;

        self.skip_whitespace();
        let order_by = self.parse_order_by()?;

        self.skip_whitespace();
        let limit = self.parse_limit()?;

        self.skip_whitespace();
        let offset = self.parse_offset()?;

        self.skip_trailing_semicolon();

        Ok(Command::Join {
            left_table,
            right_table,
            join_type,
            left_column: left_col,
            right_column: right_col,
            columns,
            where_clause,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_select_column(&mut self) -> Result<SelectColumn> {
        let keyword = self.peek_keyword_upper();

        // Check for aggregate functions
        match keyword.as_str() {
            "COUNT" | "SUM" | "AVG" | "MIN" | "MAX" => {
                let func = match keyword.as_str() {
                    "COUNT" => AggregateFunc::Count,
                    "SUM" => AggregateFunc::Sum,
                    "AVG" => AggregateFunc::Avg,
                    "MIN" => AggregateFunc::Min,
                    "MAX" => AggregateFunc::Max,
                    _ => unreachable!(),
                };
                self.read_keyword()?;
                self.skip_whitespace();
                self.expect_char('(')?;
                self.skip_whitespace();

                let column = if self.peek_char() == Some('*') {
                    self.advance();
                    "*".to_string()
                } else {
                    self.read_identifier()?
                };

                self.skip_whitespace();
                self.expect_char(')')?;

                // Check for alias
                self.skip_whitespace();
                let alias = if self.peek_keyword_upper() == "AS" {
                    self.read_keyword()?;
                    self.skip_whitespace();
                    Some(self.read_identifier()?)
                } else {
                    None
                };

                Ok(SelectColumn::Aggregate { func, column, alias })
            }
            _ => Ok(SelectColumn::Column(self.read_identifier()?))
        }
    }

    // ==================== UPDATE ====================
    fn parse_update(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();
        self.expect_keyword("SET")?;

        let mut assignments = Vec::new();
        loop {
            self.skip_whitespace();
            let col = self.read_identifier()?;
            self.skip_whitespace();
            self.expect_char('=')?;
            self.skip_whitespace();
            let val = self.parse_value()?;
            assignments.push((col, val));

            self.skip_whitespace();
            if self.peek_char() == Some(',') {
                self.advance();
            } else {
                break;
            }
        }

        self.skip_whitespace();
        let where_clause = self.parse_where()?;

        self.skip_trailing_semicolon();
        Ok(Command::Update { table, assignments, where_clause })
    }

    // ==================== DELETE ====================
    fn parse_delete(&mut self) -> Result<Command> {
        self.expect_keyword("FROM")?;
        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();
        let where_clause = self.parse_where()?;

        self.skip_trailing_semicolon();
        Ok(Command::Delete { table, where_clause })
    }

    // ==================== SHOW ====================
    fn parse_show(&mut self) -> Result<Command> {
        self.expect_keyword("TABLES")?;
        self.skip_trailing_semicolon();
        Ok(Command::ShowTables)
    }

    // ==================== WHERE CLAUSE ====================
    fn parse_where(&mut self) -> Result<Option<WhereClause>> {
        self.skip_whitespace();
        if self.peek_keyword_upper() != "WHERE" {
            return Ok(None);
        }
        self.read_keyword()?;

        let mut conditions = Vec::new();
        let mut connectors = Vec::new();

        loop {
            self.skip_whitespace();
            let condition = self.parse_condition()?;
            conditions.push(condition);

            self.skip_whitespace();
            let connector = self.peek_keyword_upper();
            match connector.as_str() {
                "AND" => {
                    self.read_keyword()?;
                    connectors.push(BoolConnector::And);
                }
                "OR" => {
                    self.read_keyword()?;
                    connectors.push(BoolConnector::Or);
                }
                _ => break,
            }
        }

        Ok(Some(WhereClause { conditions, connectors }))
    }

    fn parse_condition(&mut self) -> Result<Condition> {
        self.skip_whitespace();
        let column = self.read_identifier()?;
        self.skip_whitespace();

        // Check for IS NULL / IS NOT NULL
        let keyword = self.peek_keyword_upper();
        if keyword == "IS" {
            self.read_keyword()?;
            self.skip_whitespace();

            let is_not = if self.peek_keyword_upper() == "NOT" {
                self.read_keyword()?;
                self.skip_whitespace();
                true
            } else {
                false
            };

            self.expect_keyword("NULL")?;

            return Ok(Condition {
                column,
                operator: if is_not { ComparisonOp::IsNotNull } else { ComparisonOp::IsNull },
                value: ConditionValue::NullCheck,
            });
        }

        // Check for NOT prefix (NOT IN, NOT BETWEEN, NOT LIKE)
        let negated = if keyword == "NOT" {
            self.read_keyword()?;
            self.skip_whitespace();
            true
        } else {
            false
        };

        let next_keyword = self.peek_keyword_upper();

        // IN clause
        if next_keyword == "IN" {
            self.read_keyword()?;
            self.skip_whitespace();
            self.expect_char('(')?;

            let mut values = Vec::new();
            loop {
                self.skip_whitespace();
                values.push(self.parse_value()?);
                self.skip_whitespace();
                if self.peek_char() == Some(')') {
                    self.advance();
                    break;
                }
                self.expect_char(',')?;
            }

            return Ok(Condition {
                column,
                operator: if negated { ComparisonOp::NotIn } else { ComparisonOp::In },
                value: ConditionValue::List(values),
            });
        }

        // BETWEEN
        if next_keyword == "BETWEEN" {
            self.read_keyword()?;
            self.skip_whitespace();
            let low = self.parse_value()?;

            self.skip_whitespace();
            self.expect_keyword("AND")?;
            self.skip_whitespace();
            let high = self.parse_value()?;

            return Ok(Condition {
                column,
                operator: if negated { ComparisonOp::NotBetween } else { ComparisonOp::Between },
                value: ConditionValue::Range(low, high),
            });
        }

        // LIKE
        if next_keyword == "LIKE" {
            self.read_keyword()?;
            self.skip_whitespace();
            let pattern = self.parse_value()?;

            return Ok(Condition {
                column,
                operator: if negated { ComparisonOp::NotLike } else { ComparisonOp::Like },
                value: ConditionValue::Single(pattern),
            });
        }

        // SIMILARITY (for vectors)
        if next_keyword == "SIMILARITY" {
            self.read_keyword()?;
            self.skip_whitespace();
            let vec = self.parse_value()?;

            return Ok(Condition {
                column,
                operator: ComparisonOp::Similar,
                value: ConditionValue::Single(vec),
            });
        }

        // Standard comparison operators
        let operator = self.parse_comparison_op()?;
        self.skip_whitespace();
        let value = self.parse_value()?;

        Ok(Condition {
            column,
            operator,
            value: ConditionValue::Single(value),
        })
    }

    fn parse_comparison_op(&mut self) -> Result<ComparisonOp> {
        let ch = self.peek_char().ok_or_else(|| {
            MarsError::InvalidFormat("Unexpected end of input".into())
        })?;

        match ch {
            '=' => {
                self.advance();
                Ok(ComparisonOp::Eq)
            }
            '!' | '<' => {
                if ch == '!' {
                    self.advance();
                    self.expect_char('=')?;
                    Ok(ComparisonOp::Ne)
                } else {
                    self.advance();
                    if self.peek_char() == Some('=') {
                        self.advance();
                        Ok(ComparisonOp::Le)
                    } else if self.peek_char() == Some('>') {
                        self.advance();
                        Ok(ComparisonOp::Ne)
                    } else {
                        Ok(ComparisonOp::Lt)
                    }
                }
            }
            '>' => {
                self.advance();
                if self.peek_char() == Some('=') {
                    self.advance();
                    Ok(ComparisonOp::Ge)
                } else {
                    Ok(ComparisonOp::Gt)
                }
            }
            _ => Err(MarsError::InvalidFormat(format!("Expected comparison operator, got '{}'", ch)))
        }
    }

    // ==================== ORDER BY, LIMIT, OFFSET ====================
    fn parse_order_by(&mut self) -> Result<Option<OrderBy>> {
        self.skip_whitespace();
        if self.peek_keyword_upper() != "ORDER" {
            return Ok(None);
        }
        self.read_keyword()?;
        self.expect_keyword("BY")?;

        self.skip_whitespace();
        let column = self.read_identifier()?;

        self.skip_whitespace();
        let mut ascending = true;
        match self.peek_keyword_upper().as_str() {
            "ASC" => {
                self.read_keyword()?;
                ascending = true;
            }
            "DESC" => {
                self.read_keyword()?;
                ascending = false;
            }
            _ => {}
        }

        Ok(Some(OrderBy { column, ascending }))
    }

    // ==================== GROUP BY ====================
    fn parse_group_by(&mut self) -> Result<Option<Vec<String>>> {
        self.skip_whitespace();
        if self.peek_keyword_upper() != "GROUP" {
            return Ok(None);
        }
        self.read_keyword()?;
        self.expect_keyword("BY")?;

        let mut columns = Vec::new();
        loop {
            self.skip_whitespace();
            columns.push(self.read_identifier()?);
            self.skip_whitespace();
            if self.peek_char() == Some(',') {
                self.advance();
            } else {
                break;
            }
        }

        Ok(Some(columns))
    }

    // ==================== HAVING ====================
    fn parse_having(&mut self) -> Result<Option<WhereClause>> {
        self.skip_whitespace();
        if self.peek_keyword_upper() != "HAVING" {
            return Ok(None);
        }
        self.read_keyword()?;

        // Reuse parse_where logic but for HAVING
        let mut conditions = Vec::new();
        let mut connectors = Vec::new();

        loop {
            self.skip_whitespace();
            let condition = self.parse_condition()?;
            conditions.push(condition);

            self.skip_whitespace();
            let connector = self.peek_keyword_upper();
            match connector.as_str() {
                "AND" => {
                    self.read_keyword()?;
                    connectors.push(BoolConnector::And);
                }
                "OR" => {
                    self.read_keyword()?;
                    connectors.push(BoolConnector::Or);
                }
                _ => break,
            }
        }

        Ok(Some(WhereClause { conditions, connectors }))
    }

    fn parse_limit(&mut self) -> Result<Option<usize>> {
        self.skip_whitespace();
        if self.peek_keyword_upper() != "LIMIT" {
            return Ok(None);
        }
        self.read_keyword()?;
        self.skip_whitespace();
        let n = self.read_integer()? as usize;
        Ok(Some(n))
    }

    fn parse_offset(&mut self) -> Result<Option<usize>> {
        self.skip_whitespace();
        if self.peek_keyword_upper() != "OFFSET" {
            return Ok(None);
        }
        self.read_keyword()?;
        self.skip_whitespace();
        let n = self.read_integer()? as usize;
        Ok(Some(n))
    }

    // ==================== VALUE PARSING ====================
    fn parse_value(&mut self) -> Result<Value> {
        self.skip_whitespace();

        let ch = self.peek_char().ok_or_else(|| {
            MarsError::InvalidFormat("Unexpected end of input".into())
        })?;

        match ch {
            '\'' | '"' => {
                self.advance();
                let s = self.read_string_content(ch)?;
                Ok(Value::Text(s))
            }
            '[' => {
                self.advance();
                let nums = self.read_vector_content()?;
                Ok(Value::Vector(nums))
            }
            't' | 'T' | 'f' | 'F' => {
                let kw = self.read_keyword_upper()?;
                match kw.as_str() {
                    "TRUE" => Ok(Value::Boolean(true)),
                    "FALSE" => Ok(Value::Boolean(false)),
                    _ => Err(MarsError::InvalidFormat(format!("Unknown keyword: {}", kw))),
                }
            }
            'n' | 'N' => {
                let kw = self.read_keyword_upper()?;
                if kw == "NULL" {
                    Ok(Value::Null)
                } else {
                    Err(MarsError::InvalidFormat(format!("Unknown keyword: {}", kw)))
                }
            }
            '-' | '0'..='9' => {
                let (n, has_decimal) = self.read_number()?;
                if has_decimal {
                    Ok(Value::Float(n))
                } else {
                    Ok(Value::Integer(n as i64))
                }
            }
            _ => Err(MarsError::InvalidFormat(format!("Unexpected character: {}", ch))),
        }
    }

    fn read_vector_content(&mut self) -> Result<Vec<f32>> {
        let mut nums = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek_char() == Some(']') {
                self.advance();
                break;
            }
            let (n, _) = self.read_number()?;
            nums.push(n as f32);
            self.skip_whitespace();
            if self.peek_char() == Some(',') {
                self.advance();
            }
        }
        Ok(nums)
    }

    fn parse_column_type(&mut self) -> Result<ColumnType> {
        let type_name = self.read_keyword_upper()?;

        match type_name.as_str() {
            "VECTOR" => {
                self.skip_whitespace();
                self.expect_char('(')?;
                self.skip_whitespace();
                let dim = self.read_integer()? as usize;
                self.skip_whitespace();
                self.expect_char(')')?;
                Ok(ColumnType::Vector(dim))
            }
            "TEXT" | "VARCHAR" | "STRING" | "CHAR" => Ok(ColumnType::Text),
            "INTEGER" | "INT" | "BIGINT" | "SMALLINT" | "TINYINT" => Ok(ColumnType::Integer),
            "FLOAT" | "REAL" | "DOUBLE" | "DECIMAL" | "NUMERIC" => Ok(ColumnType::Float),
            "BOOLEAN" | "BOOL" => Ok(ColumnType::Boolean),
            "BLOB" | "BINARY" => Ok(ColumnType::Blob),
            _ => Err(MarsError::InvalidFormat(format!("Unknown type: {}", type_name))),
        }
    }

    // ==================== LOW-LEVEL HELPERS ====================

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_trailing_semicolon(&mut self) {
        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn advance(&mut self) {
        if self.pos < self.input.len() {
            self.pos += self.input[self.pos..].chars().next().unwrap().len_utf8();
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<()> {
        let ch = self.peek_char().ok_or_else(|| {
            MarsError::InvalidFormat(format!("Expected '{}', got end of input", expected))
        })?;
        if ch != expected {
            return Err(MarsError::InvalidFormat(format!("Expected '{}', got '{}'", expected, ch)));
        }
        self.advance();
        Ok(())
    }

    fn peek_keyword_upper(&self) -> String {
        let start = self.pos;
        let mut end = start;
        for ch in self.input[start..].chars() {
            if ch.is_alphanumeric() || ch == '_' || ch == '*' {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        self.input[start..end].to_uppercase()
    }

    fn read_keyword(&mut self) -> Result<String> {
        self.skip_whitespace();  // Skip leading whitespace
        let start = self.pos;
        let mut end = start;
        for ch in self.input[start..].chars() {
            if ch.is_alphanumeric() || ch == '_' || ch == '*' {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        if end == start {
            return Err(MarsError::InvalidFormat("Expected keyword".into()));
        }
        let keyword = self.input[start..end].to_string();
        self.pos = end;
        Ok(keyword)
    }

    fn read_keyword_upper(&mut self) -> Result<String> {
        let kw = self.read_keyword()?;
        Ok(kw.to_uppercase())
    }

    fn expect_keyword(&mut self, expected: &str) -> Result<()> {
        let kw = self.read_keyword_upper()?;
        if kw != expected.to_uppercase() {
            return Err(MarsError::InvalidFormat(format!("Expected '{}', got '{}'", expected, kw)));
        }
        Ok(())
    }

    fn read_identifier(&mut self) -> Result<String> {
        self.read_keyword()
    }

    fn read_integer(&mut self) -> Result<i64> {
        let (n, _) = self.read_number()?;
        Ok(n as i64)
    }

    fn read_number(&mut self) -> Result<(f64, bool)> {
        let start = self.pos;
        let mut end = start;
        let mut has_dot = false;

        for ch in self.input[start..].chars() {
            if ch == '-' && end == start {
                end += 1;
            } else if ch == '.' && !has_dot {
                has_dot = true;
                end += 1;
            } else if ch.is_ascii_digit() {
                end += 1;
            } else {
                break;
            }
        }

        if end == start {
            return Err(MarsError::InvalidFormat("Expected number".into()));
        }

        let num_str = &self.input[start..end];
        self.pos = end;
        let n: f64 = num_str.parse().map_err(|_| MarsError::InvalidFormat("Invalid number".into()))?;
        Ok((n, has_dot))
    }

    fn read_string_content(&mut self, quote: char) -> Result<String> {
        let mut result = String::new();
        loop {
            let ch = self.peek_char().ok_or_else(|| {
                MarsError::InvalidFormat("Unterminated string".into())
            })?;
            if ch == quote {
                // Check for escaped quote ('')
                let next_pos = self.pos + quote.len_utf8();
                if next_pos < self.input.len() {
                    let next_char = self.input[next_pos..].chars().next();
                    if next_char == Some(quote) {
                        result.push(quote);
                        self.pos = next_pos + quote.len_utf8();
                        continue;
                    }
                }
                self.advance();
                break;
            }
            if ch == '\\' {
                self.advance();
                let escaped = self.peek_char().ok_or_else(|| {
                    MarsError::InvalidFormat("Unterminated escape sequence".into())
                })?;
                match escaped {
                    'n' => result.push('\n'),
                    't' => result.push('\t'),
                    'r' => result.push('\r'),
                    '\\' => result.push('\\'),
                    '\'' => result.push('\''),
                    '"' => result.push('"'),
                    _ => result.push(escaped),
                }
                self.advance();
            } else {
                result.push(ch);
                self.advance();
            }
        }
        Ok(result)
    }
}

/// Parse a SQL command string
pub fn parse(input: &str) -> Result<Command> {
    Parser::new(input).parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_create_table() {
        let sql = "CREATE TABLE documents (id INTEGER PRIMARY KEY, embedding VECTOR(768), title TEXT);";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::CreateTable { name, columns } => {
                assert_eq!(name, "documents");
                assert_eq!(columns.len(), 3);
                assert!(columns[0].primary_key);
                assert_eq!(columns[1].data_type, ColumnType::Vector(768));
            }
            _ => panic!("Expected CreateTable"),
        }
    }

    #[test]
    fn test_parse_insert_multirow() {
        let sql = "INSERT INTO docs (id, name) VALUES (1, 'a'), (2, 'b'), (3, 'c');";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Insert { table, columns, values } => {
                assert_eq!(table, "docs");
                assert_eq!(values.len(), 3);
            }
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_parse_select_aggregate() {
        let sql = "SELECT COUNT(*), AVG(score) FROM users;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { columns, .. } => {
                assert_eq!(columns.len(), 2);
                assert!(matches!(columns[0], SelectColumn::Aggregate { func: AggregateFunc::Count, .. }));
                assert!(matches!(columns[1], SelectColumn::Aggregate { func: AggregateFunc::Avg, .. }));
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_select_distinct() {
        let sql = "SELECT DISTINCT category FROM products;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { distinct, .. } => {
                assert!(distinct);
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_where_like() {
        let sql = "SELECT * FROM users WHERE name LIKE 'John%';";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { where_clause: Some(wc), .. } => {
                assert_eq!(wc.conditions.len(), 1);
                assert_eq!(wc.conditions[0].operator, ComparisonOp::Like);
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_where_in() {
        let sql = "SELECT * FROM users WHERE id IN (1, 2, 3);";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { where_clause: Some(wc), .. } => {
                assert_eq!(wc.conditions[0].operator, ComparisonOp::In);
                if let ConditionValue::List(values) = &wc.conditions[0].value {
                    assert_eq!(values.len(), 3);
                } else {
                    panic!("Expected List");
                }
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_where_between() {
        let sql = "SELECT * FROM products WHERE price BETWEEN 10 AND 100;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { where_clause: Some(wc), .. } => {
                assert_eq!(wc.conditions[0].operator, ComparisonOp::Between);
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_where_is_null() {
        let sql = "SELECT * FROM users WHERE deleted_at IS NULL;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { where_clause: Some(wc), .. } => {
                assert_eq!(wc.conditions[0].operator, ComparisonOp::IsNull);
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_where_or() {
        let sql = "SELECT * FROM users WHERE id = 1 OR id = 2;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { where_clause: Some(wc), .. } => {
                assert_eq!(wc.conditions.len(), 2);
                assert_eq!(wc.connectors.len(), 1);
                assert!(matches!(wc.connectors[0], BoolConnector::Or));
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_limit_offset() {
        let sql = "SELECT * FROM users LIMIT 10 OFFSET 20;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { limit, offset, .. } => {
                assert_eq!(limit, Some(10));
                assert_eq!(offset, Some(20));
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_drop_if_exists() {
        let sql = "DROP TABLE IF EXISTS temp;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::DropTable { name, if_exists } => {
                assert_eq!(name, "temp");
                assert!(if_exists);
            }
            _ => panic!("Expected DropTable"),
        }
    }

    #[test]
    fn test_parse_order_by_desc() {
        let sql = "SELECT * FROM products ORDER BY price DESC;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { order_by: Some(ob), .. } => {
                assert_eq!(ob.column, "price");
                assert!(!ob.ascending);
            }
            _ => panic!("Expected Select"),
        }
    }
}
