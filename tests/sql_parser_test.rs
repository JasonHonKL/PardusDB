//! Integration tests for SQL parsing

use pardusdb::{parse, Command, ColumnType, Value, ComparisonOp};

#[test]
fn test_parse_create_table() {
    let sql = "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL);";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::CreateTable { name, columns } => {
            assert_eq!(name, "users");
            assert_eq!(columns.len(), 2);
            assert_eq!(columns[0].name, "id");
            assert!(columns[0].primary_key);
            assert_eq!(columns[1].name, "name");
            assert!(columns[1].not_null);
        }
        _ => panic!("Expected CreateTable"),
    }
}

#[test]
fn test_parse_create_table_with_vector() {
    let sql = "CREATE TABLE docs (embedding VECTOR(768), title TEXT);";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::CreateTable { name, columns } => {
            assert_eq!(name, "docs");
            assert_eq!(columns.len(), 2);
            assert_eq!(columns[0].data_type, ColumnType::Vector(768));
            assert_eq!(columns[1].data_type, ColumnType::Text);
        }
        _ => panic!("Expected CreateTable"),
    }
}

#[test]
fn test_parse_insert() {
    let sql = "INSERT INTO users (name, age) VALUES ('Alice', 30);";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Insert { table, columns, values } => {
            assert_eq!(table, "users");
            assert_eq!(columns, vec!["name", "age"]);
            assert_eq!(values.len(), 2);
            assert_eq!(values[0], Value::Text("Alice".to_string()));
            assert_eq!(values[1], Value::Integer(30));
        }
        _ => panic!("Expected Insert"),
    }
}

#[test]
fn test_parse_insert_with_vector() {
    let sql = "INSERT INTO docs (embedding, title) VALUES ([0.1, 0.2, 0.3], 'Test');";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Insert { table, columns, values } => {
            assert_eq!(table, "docs");
            assert_eq!(values[0], Value::Vector(vec![0.1, 0.2, 0.3]));
            assert_eq!(values[1], Value::Text("Test".to_string()));
        }
        _ => panic!("Expected Insert"),
    }
}

#[test]
fn test_parse_select_all() {
    let sql = "SELECT * FROM users;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Select { table, columns, where_clause, limit, .. } => {
            assert_eq!(table, "users");
            assert!(columns.is_empty()); // * means empty vec
            assert!(where_clause.is_none());
            assert!(limit.is_none());
        }
        _ => panic!("Expected Select"),
    }
}

#[test]
fn test_parse_select_columns() {
    let sql = "SELECT name, age FROM users LIMIT 10;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Select { table, columns, limit, .. } => {
            assert_eq!(table, "users");
            assert_eq!(columns, vec!["name", "age"]);
            assert_eq!(limit, Some(10));
        }
        _ => panic!("Expected Select"),
    }
}

#[test]
fn test_parse_select_where() {
    let sql = "SELECT * FROM users WHERE id = 1;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Select { table, where_clause, .. } => {
            assert_eq!(table, "users");
            let wc = where_clause.unwrap();
            assert_eq!(wc.conditions.len(), 1);
            assert_eq!(wc.conditions[0].column, "id");
            assert_eq!(wc.conditions[0].operator, ComparisonOp::Eq);
            assert_eq!(wc.conditions[0].value, Value::Integer(1));
        }
        _ => panic!("Expected Select"),
    }
}

#[test]
fn test_parse_select_where_multiple() {
    let sql = "SELECT * FROM users WHERE age > 18 AND age < 65;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Select { where_clause, .. } => {
            let wc = where_clause.unwrap();
            assert_eq!(wc.conditions.len(), 2);
            assert_eq!(wc.conditions[0].operator, ComparisonOp::Gt);
            assert_eq!(wc.conditions[1].operator, ComparisonOp::Lt);
        }
        _ => panic!("Expected Select"),
    }
}

#[test]
fn test_parse_select_similarity() {
    let sql = "SELECT * FROM docs WHERE embedding SIMILARITY [0.1, 0.2] LIMIT 5;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Select { where_clause, limit, .. } => {
            let wc = where_clause.unwrap();
            assert_eq!(wc.conditions[0].operator, ComparisonOp::Similar);
            assert_eq!(wc.conditions[0].value, Value::Vector(vec![0.1, 0.2]));
            assert_eq!(limit, Some(5));
        }
        _ => panic!("Expected Select"),
    }
}

#[test]
fn test_parse_update() {
    let sql = "UPDATE users SET name = 'Bob' WHERE id = 1;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Update { table, assignments, where_clause } => {
            assert_eq!(table, "users");
            assert_eq!(assignments.len(), 1);
            assert_eq!(assignments[0].0, "name");
            assert_eq!(assignments[0].1, Value::Text("Bob".to_string()));
            assert!(where_clause.is_some());
        }
        _ => panic!("Expected Update"),
    }
}

#[test]
fn test_parse_delete() {
    let sql = "DELETE FROM users WHERE id = 1;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::Delete { table, where_clause } => {
            assert_eq!(table, "users");
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

#[test]
fn test_parse_drop_table() {
    let sql = "DROP TABLE users;";
    let cmd = parse(sql).unwrap();

    match cmd {
        Command::DropTable { name } => {
            assert_eq!(name, "users");
        }
        _ => panic!("Expected DropTable"),
    }
}

#[test]
fn test_parse_value_types() {
    // Test various value types
    let cases = vec![
        ("'hello'", Value::Text("hello".to_string())),
        ("123", Value::Integer(123)),
        ("123.45", Value::Float(123.45)),
        ("true", Value::Boolean(true)),
        ("false", Value::Boolean(false)),
        ("null", Value::Null),
        ("[1.0, 2.0, 3.0]", Value::Vector(vec![1.0, 2.0, 3.0])),
    ];

    for (value_str, expected) in cases {
        let sql = format!("INSERT INTO t (v) VALUES ({});", value_str);
        let cmd = parse(&sql).unwrap();
        match cmd {
            Command::Insert { values, .. } => {
                assert_eq!(values[0], expected);
            }
            _ => panic!("Expected Insert"),
        }
    }
}

#[test]
fn test_parse_comparison_operators() {
    let cases = vec![
        ("id = 1", ComparisonOp::Eq),
        ("id != 1", ComparisonOp::Ne),
        ("id < 1", ComparisonOp::Lt),
        ("id <= 1", ComparisonOp::Le),
        ("id > 1", ComparisonOp::Gt),
        ("id >= 1", ComparisonOp::Ge),
    ];

    for (condition, expected_op) in cases {
        let sql = format!("SELECT * FROM t WHERE {};", condition);
        let cmd = parse(&sql).unwrap();
        match cmd {
            Command::Select { where_clause, .. } => {
                let wc = where_clause.unwrap();
                assert_eq!(wc.conditions[0].operator, expected_op);
            }
            _ => panic!("Expected Select"),
        }
    }
}
