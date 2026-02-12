use std::fmt;

use crate::error::{MarsError, Result};
use crate::schema::{ColumnType, Schema, Value};

/// SQL-like command types
#[derive(Clone, Debug)]
pub enum Command {
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
    },
    DropTable {
        name: String,
    },
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Value>,
    },
    Select {
        table: String,
        columns: Vec<String>,  // empty means *
        where_clause: Option<WhereClause>,
        order_by: Option<OrderBy>,
        limit: Option<usize>,
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

#[derive(Clone, Debug)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: ColumnType,
    pub primary_key: bool,
    pub not_null: bool,
}

#[derive(Clone, Debug)]
pub struct WhereClause {
    pub conditions: Vec<Condition>,
}

#[derive(Clone, Debug)]
pub struct Condition {
    pub column: String,
    pub operator: ComparisonOp,
    pub value: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Similar,  // For vector similarity
}

#[derive(Clone, Debug)]
pub struct OrderBy {
    pub column: String,
    pub ascending: bool,
}

/// Simple SQL-like parser
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

        let keyword = self.read_keyword()?;

        match keyword.to_uppercase().as_str() {
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

    fn parse_create(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let next = self.read_keyword()?;
        if next.to_uppercase() != "TABLE" {
            return Err(MarsError::InvalidFormat("Expected TABLE after CREATE".into()));
        }

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

            loop {
                let keyword = self.peek_keyword();
                match keyword.to_uppercase().as_str() {
                    "PRIMARY" => {
                        self.read_keyword()?;
                        self.skip_whitespace();
                        let key_kw = self.read_keyword()?;
                        if key_kw.to_uppercase() != "KEY" {
                            return Err(MarsError::InvalidFormat("Expected KEY after PRIMARY".into()));
                        }
                        primary_key = true;
                    }
                    "NOT" => {
                        self.read_keyword()?;
                        self.skip_whitespace();
                        let null_kw = self.read_keyword()?;
                        if null_kw.to_uppercase() != "NULL" {
                            return Err(MarsError::InvalidFormat("Expected NULL after NOT".into()));
                        }
                        not_null = true;
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
            });

            self.skip_whitespace();
            if self.peek_char() == Some(')') {
                self.advance();
                break;
            }
            self.expect_char(',')?;
        }

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::CreateTable { name, columns })
    }

    fn parse_drop(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let next = self.read_keyword()?;
        if next.to_uppercase() != "TABLE" {
            return Err(MarsError::InvalidFormat("Expected TABLE after DROP".into()));
        }

        self.skip_whitespace();
        let name = self.read_identifier()?;

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::DropTable { name })
    }

    fn parse_insert(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let into = self.read_keyword()?;
        if into.to_uppercase() != "INTO" {
            return Err(MarsError::InvalidFormat("Expected INTO after INSERT".into()));
        }

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
        let values_kw = self.read_keyword()?;
        if values_kw.to_uppercase() != "VALUES" {
            return Err(MarsError::InvalidFormat("Expected VALUES".into()));
        }

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

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::Insert { table, columns, values })
    }

    fn parse_select(&mut self) -> Result<Command> {
        self.skip_whitespace();

        let mut columns = Vec::new();
        if self.peek_char() == Some('*') {
            self.advance();
        } else {
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
        }

        self.skip_whitespace();
        let from = self.read_keyword()?;
        if from.to_uppercase() != "FROM" {
            return Err(MarsError::InvalidFormat("Expected FROM".into()));
        }

        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();
        let where_clause = self.parse_where()?;

        self.skip_whitespace();
        let order_by = self.parse_order_by()?;

        self.skip_whitespace();
        let limit = self.parse_limit()?;

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::Select {
            table,
            columns,
            where_clause,
            order_by,
            limit,
        })
    }

    fn parse_update(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();
        let set = self.read_keyword()?;
        if set.to_uppercase() != "SET" {
            return Err(MarsError::InvalidFormat("Expected SET".into()));
        }

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

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::Update {
            table,
            assignments,
            where_clause,
        })
    }

    fn parse_delete(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let from = self.read_keyword()?;
        if from.to_uppercase() != "FROM" {
            return Err(MarsError::InvalidFormat("Expected FROM".into()));
        }

        self.skip_whitespace();
        let table = self.read_identifier()?;

        self.skip_whitespace();
        let where_clause = self.parse_where()?;

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::Delete { table, where_clause })
    }

    fn parse_show(&mut self) -> Result<Command> {
        self.skip_whitespace();
        let tables = self.read_keyword()?;
        if tables.to_uppercase() != "TABLES" {
            return Err(MarsError::InvalidFormat("Expected TABLES after SHOW".into()));
        }

        self.skip_whitespace();
        if self.peek_char() == Some(';') {
            self.advance();
        }

        Ok(Command::ShowTables)
    }

    fn parse_where(&mut self) -> Result<Option<WhereClause>> {
        self.skip_whitespace();
        let keyword = self.peek_keyword();
        if keyword.to_uppercase() != "WHERE" {
            return Ok(None);
        }
        self.read_keyword()?;

        let mut conditions = Vec::new();
        loop {
            self.skip_whitespace();
            let column = self.read_identifier()?;
            self.skip_whitespace();

            let operator = self.parse_comparison_op()?;
            self.skip_whitespace();

            let value = self.parse_value()?;
            conditions.push(Condition {
                column,
                operator,
                value,
            });

            self.skip_whitespace();
            let and_or = self.peek_keyword();
            if and_or.to_uppercase() == "AND" {
                self.read_keyword()?;
                continue;
            }
            break;
        }

        Ok(Some(WhereClause { conditions }))
    }

    fn parse_order_by(&mut self) -> Result<Option<OrderBy>> {
        self.skip_whitespace();
        let keyword = self.peek_keyword();
        if keyword.to_uppercase() != "ORDER" {
            return Ok(None);
        }
        self.read_keyword()?;

        self.skip_whitespace();
        let by = self.read_keyword()?;
        if by.to_uppercase() != "BY" {
            return Err(MarsError::InvalidFormat("Expected BY after ORDER".into()));
        }

        self.skip_whitespace();
        let column = self.read_identifier()?;

        self.skip_whitespace();
        let mut ascending = true;
        let dir = self.peek_keyword();
        if dir.to_uppercase() == "ASC" {
            self.read_keyword()?;
            ascending = true;
        } else if dir.to_uppercase() == "DESC" {
            self.read_keyword()?;
            ascending = false;
        }

        Ok(Some(OrderBy { column, ascending }))
    }

    fn parse_limit(&mut self) -> Result<Option<usize>> {
        self.skip_whitespace();
        let keyword = self.peek_keyword();
        if keyword.to_uppercase() != "LIMIT" {
            return Ok(None);
        }
        self.read_keyword()?;

        self.skip_whitespace();
        let n = self.read_number()? as usize;
        Ok(Some(n))
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
            '!' => {
                self.advance();
                self.expect_char('=')?;
                Ok(ComparisonOp::Ne)
            }
            '<' => {
                self.advance();
                if self.peek_char() == Some('=') {
                    self.advance();
                    Ok(ComparisonOp::Le)
                } else {
                    Ok(ComparisonOp::Lt)
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
            _ => {
                // Check for SIMILARITY keyword
                let kw = self.peek_keyword();
                if kw.to_uppercase() == "SIMILARITY" {
                    self.read_keyword()?;
                    return Ok(ComparisonOp::Similar);
                }
                Err(MarsError::InvalidFormat(format!("Expected comparison operator, got {}", ch)))
            }
        }
    }

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
                let mut nums = Vec::new();
                loop {
                    self.skip_whitespace();
                    if self.peek_char() == Some(']') {
                        self.advance();
                        break;
                    }
                    let n = self.read_number()?;
                    nums.push(n as f32);
                    self.skip_whitespace();
                    if self.peek_char() == Some(',') {
                        self.advance();
                    }
                }
                Ok(Value::Vector(nums))
            }
            't' | 'f' => {
                let kw = self.read_keyword()?;
                match kw.to_lowercase().as_str() {
                    "true" => Ok(Value::Boolean(true)),
                    "false" => Ok(Value::Boolean(false)),
                    "null" => Ok(Value::Null),
                    _ => Err(MarsError::InvalidFormat(format!("Unknown keyword: {}", kw))),
                }
            }
            'n' => {
                let kw = self.read_keyword()?;
                if kw.to_lowercase() == "null" {
                    Ok(Value::Null)
                } else {
                    Err(MarsError::InvalidFormat(format!("Unknown keyword: {}", kw)))
                }
            }
            '-' | '0'..='9' => {
                let (n, has_decimal) = self.read_number_with_decimal()?;
                if has_decimal {
                    Ok(Value::Float(n))
                } else {
                    Ok(Value::Integer(n as i64))
                }
            }
            _ => Err(MarsError::InvalidFormat(format!("Unexpected character: {}", ch))),
        }
    }

    fn parse_column_type(&mut self) -> Result<ColumnType> {
        let type_name = self.read_keyword()?;

        match type_name.to_uppercase().as_str() {
            "VECTOR" => {
                self.skip_whitespace();
                self.expect_char('(')?;
                self.skip_whitespace();
                let dim = self.read_number()? as usize;
                self.skip_whitespace();
                self.expect_char(')')?;
                Ok(ColumnType::Vector(dim))
            }
            "TEXT" | "VARCHAR" | "STRING" => Ok(ColumnType::Text),
            "INTEGER" | "INT" => Ok(ColumnType::Integer),
            "FLOAT" | "REAL" | "DOUBLE" => Ok(ColumnType::Float),
            "BOOLEAN" | "BOOL" => Ok(ColumnType::Boolean),
            "BLOB" => Ok(ColumnType::Blob),
            _ => Err(MarsError::InvalidFormat(format!("Unknown type: {}", type_name))),
        }
    }

    // Helper methods
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
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

    fn peek_keyword(&self) -> String {
        let start = self.pos;
        let mut end = start;
        for ch in self.input[start..].chars() {
            if ch.is_alphanumeric() || ch == '_' {
                end += ch.len_utf8();
            } else {
                break;
            }
        }
        self.input[start..end].to_string()
    }

    fn read_keyword(&mut self) -> Result<String> {
        let start = self.pos;
        let mut end = start;
        for ch in self.input[start..].chars() {
            if ch.is_alphanumeric() || ch == '_' {
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

    fn read_identifier(&mut self) -> Result<String> {
        self.read_keyword()
    }

    fn read_number(&mut self) -> Result<f64> {
        let (n, _) = self.read_number_with_decimal()?;
        Ok(n)
    }

    fn read_number_with_decimal(&mut self) -> Result<(f64, bool)> {
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

/// Parse a SQL-like command string
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
    fn test_parse_insert() {
        let sql = "INSERT INTO documents (embedding, title) VALUES ([0.1, 0.2, 0.3], 'Test Title');";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Insert { table, columns, values } => {
                assert_eq!(table, "documents");
                assert_eq!(columns, vec!["embedding", "title"]);
                assert_eq!(values.len(), 2);
            }
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_parse_select() {
        let sql = "SELECT * FROM documents WHERE id = 1 LIMIT 10;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Select { table, limit, .. } => {
                assert_eq!(table, "documents");
                assert_eq!(limit, Some(10));
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_delete() {
        let sql = "DELETE FROM documents WHERE id = 5;";
        let cmd = parse(sql).unwrap();

        match cmd {
            Command::Delete { table, where_clause } => {
                assert_eq!(table, "documents");
                assert!(where_clause.is_some());
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_parse_show_tables() {
        let sql = "SHOW TABLES;";
        let cmd = parse(sql).unwrap();
        assert!(matches!(cmd, Command::ShowTables));
    }
}
