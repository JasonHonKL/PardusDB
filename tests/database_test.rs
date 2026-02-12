//! Integration tests for database operations

use pardusdb::{Database, ExecuteResult, Value};

#[test]
fn test_create_table() {
    let mut db = Database::in_memory();

    let result = db.execute(
        "CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(128), title TEXT);"
    ).unwrap();

    match result {
        ExecuteResult::CreateTable { name } => assert_eq!(name, "docs"),
        _ => panic!("Expected CreateTable result"),
    }

    assert!(db.get_table("docs").is_some());
}

#[test]
fn test_show_tables() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();
    db.execute("CREATE TABLE users (embedding VECTOR(5), name TEXT);").unwrap();

    let result = db.execute("SHOW TABLES;").unwrap();

    match result {
        ExecuteResult::ShowTables { tables } => {
            assert_eq!(tables.len(), 2);
            let names: Vec<&str> = tables.iter().map(|t| t.name.as_str()).collect();
            assert!(names.contains(&"docs"));
            assert!(names.contains(&"users"));
        }
        _ => panic!("Expected ShowTables result"),
    }
}

#[test]
fn test_insert_and_select() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();

    db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'First');").unwrap();
    db.execute("INSERT INTO docs (embedding, title) VALUES ([0.0, 1.0, 0.0], 'Second');").unwrap();
    db.execute("INSERT INTO docs (embedding, title) VALUES ([0.0, 0.0, 1.0], 'Third');").unwrap();

    let result = db.execute("SELECT * FROM docs;").unwrap();

    match result {
        ExecuteResult::Select { rows } => assert_eq!(rows.len(), 3),
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_select_with_where() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE items (embedding VECTOR(2), category TEXT, score INTEGER);").unwrap();

    db.execute("INSERT INTO items (embedding, category, score) VALUES ([1.0, 0.0], 'A', 10);").unwrap();
    db.execute("INSERT INTO items (embedding, category, score) VALUES ([0.0, 1.0], 'B', 20);").unwrap();
    db.execute("INSERT INTO items (embedding, category, score) VALUES ([0.5, 0.5], 'A', 30);").unwrap();

    let result = db.execute("SELECT * FROM items WHERE category = 'A';").unwrap();

    match result {
        ExecuteResult::Select { rows } => assert_eq!(rows.len(), 2),
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_select_with_limit() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE items (embedding VECTOR(2), name TEXT);").unwrap();

    for i in 0..10 {
        db.execute(&format!(
            "INSERT INTO items (embedding, name) VALUES ([{}, 0.0], 'item{}');",
            i as f32 / 10.0, i
        )).unwrap();
    }

    let result = db.execute("SELECT * FROM items LIMIT 5;").unwrap();

    match result {
        ExecuteResult::Select { rows } => assert_eq!(rows.len(), 5),
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_similarity_search() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();

    // Insert vectors at different positions
    db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'X axis');").unwrap();
    db.execute("INSERT INTO docs (embedding, title) VALUES ([0.0, 1.0, 0.0], 'Y axis');").unwrap();
    db.execute("INSERT INTO docs (embedding, title) VALUES ([0.0, 0.0, 1.0], 'Z axis');").unwrap();
    db.execute("INSERT INTO docs (embedding, title) VALUES ([0.5, 0.5, 0.0], 'XY plane');").unwrap();

    // Query for vectors close to [1, 0, 0]
    let result = db.execute(
        "SELECT * FROM docs WHERE embedding SIMILARITY [0.9, 0.1, 0.0] LIMIT 2;"
    ).unwrap();

    match result {
        ExecuteResult::SelectSimilar { results } => {
            assert_eq!(results.len(), 2);
            // First result should be 'X axis' as it's closest
            let (first_row, first_dist) = &results[0];
            assert_eq!(first_row.values[1], Value::Text("X axis".to_string()));
            assert!(first_dist < &results[1].1, "Results should be ordered by distance");
        }
        _ => panic!("Expected SelectSimilar result"),
    }
}

#[test]
fn test_update() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE items (id INTEGER, embedding VECTOR(2), status TEXT);").unwrap();
    db.execute("INSERT INTO items (embedding, status) VALUES ([1.0, 0.0], 'pending');").unwrap();
    db.execute("INSERT INTO items (embedding, status) VALUES ([0.0, 1.0], 'pending');").unwrap();

    let result = db.execute("UPDATE items SET status = 'done' WHERE id = 1;").unwrap();

    match result {
        ExecuteResult::Update { count } => assert_eq!(count, 1),
        _ => panic!("Expected Update result"),
    }
}

#[test]
fn test_delete() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE items (id INTEGER, embedding VECTOR(2), name TEXT);").unwrap();
    db.execute("INSERT INTO items (embedding, name) VALUES ([1.0, 0.0], 'first');").unwrap();
    db.execute("INSERT INTO items (embedding, name) VALUES ([0.0, 1.0], 'second');").unwrap();

    let result = db.execute("DELETE FROM items WHERE id = 1;").unwrap();

    match result {
        ExecuteResult::Delete { count } => assert_eq!(count, 1),
        _ => panic!("Expected Delete result"),
    }

    // Verify deletion
    let result = db.execute("SELECT * FROM items;").unwrap();
    match result {
        ExecuteResult::Select { rows } => assert_eq!(rows.len(), 1),
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_drop_table() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE temp (embedding VECTOR(2));").unwrap();
    assert!(db.get_table("temp").is_some());

    let result = db.execute("DROP TABLE temp;").unwrap();

    match result {
        ExecuteResult::DropTable { name } => assert_eq!(name, "temp"),
        _ => panic!("Expected DropTable result"),
    }

    assert!(db.get_table("temp").is_none());
}

#[test]
fn test_direct_api_insert() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT, score FLOAT);").unwrap();

    // Use direct API for faster insert
    let id = db.insert_direct(
        "docs",
        vec![1.0, 2.0, 3.0],
        vec![
            ("title", Value::Text("Direct insert".to_string())),
            ("score", Value::Float(0.95)),
        ]
    ).unwrap();

    assert!(id > 0);

    // Verify the insert
    let result = db.execute("SELECT * FROM docs;").unwrap();
    match result {
        ExecuteResult::Select { rows } => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].values[1], Value::Text("Direct insert".to_string()));
        }
        _ => panic!("Expected Select result"),
    }
}

#[test]
fn test_direct_api_search() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (embedding VECTOR(3), title TEXT);").unwrap();

    // Insert some vectors
    db.insert_direct("docs", vec![1.0, 0.0, 0.0], vec![("title", Value::Text("X".to_string()))]).unwrap();
    db.insert_direct("docs", vec![0.0, 1.0, 0.0], vec![("title", Value::Text("Y".to_string()))]).unwrap();
    db.insert_direct("docs", vec![0.0, 0.0, 1.0], vec![("title", Value::Text("Z".to_string()))]).unwrap();

    // Search using direct API
    let results = db.search_similar("docs", &[0.9, 0.1, 0.0], 2, 10).unwrap();

    assert_eq!(results.len(), 2);
    // First should be "X" (closest to [0.9, 0.1, 0])
    assert_eq!(results[0].1[1], Value::Text("X".to_string()));
}

#[test]
fn test_persistence() {
    let temp_path = std::env::temp_dir().join("marsdb_test_persistence.mars");

    // Clean up any existing file
    let _ = std::fs::remove_file(&temp_path);

    // Create and populate database
    {
        let mut db = Database::open(&temp_path).unwrap();
        db.execute("CREATE TABLE docs (embedding VECTOR(2), title TEXT);").unwrap();
        db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0], 'Test');").unwrap();
        db.save().unwrap();
    }

    // Reopen and verify
    {
        let mut db = Database::open(&temp_path).unwrap();
        let result = db.execute("SELECT * FROM docs;").unwrap();
        match result {
            ExecuteResult::Select { rows } => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Select result"),
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_multiple_tables() {
    let mut db = Database::in_memory();

    // Create multiple tables
    db.execute("CREATE TABLE users (embedding VECTOR(4), name TEXT);").unwrap();
    db.execute("CREATE TABLE products (embedding VECTOR(4), sku TEXT);").unwrap();
    db.execute("CREATE TABLE orders (embedding VECTOR(4), order_id INTEGER);").unwrap();

    // Insert into each table
    db.execute("INSERT INTO users (embedding, name) VALUES ([1.0, 0.0, 0.0, 0.0], 'Alice');").unwrap();
    db.execute("INSERT INTO products (embedding, sku) VALUES ([0.0, 1.0, 0.0, 0.0], 'SKU123');").unwrap();
    db.execute("INSERT INTO orders (embedding, order_id) VALUES ([0.0, 0.0, 1.0, 0.0], 42);").unwrap();

    // Verify each table independently
    assert_eq!(db.get_table("users").unwrap().len(), 1);
    assert_eq!(db.get_table("products").unwrap().len(), 1);
    assert_eq!(db.get_table("orders").unwrap().len(), 1);
}

#[test]
fn test_comparison_operators() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE nums (embedding VECTOR(1), val INTEGER);").unwrap();
    db.execute("INSERT INTO nums (embedding, val) VALUES ([0.0], 10);").unwrap();
    db.execute("INSERT INTO nums (embedding, val) VALUES ([0.0], 20);").unwrap();
    db.execute("INSERT INTO nums (embedding, val) VALUES ([0.0], 30);").unwrap();

    // Test various operators
    let cases = vec![
        ("val = 20", 1),
        ("val != 20", 2),
        ("val < 25", 2),
        ("val <= 20", 2),
        ("val > 20", 1),
        ("val >= 20", 2),
    ];

    for (condition, expected_count) in cases {
        let result = db.execute(&format!("SELECT * FROM nums WHERE {};", condition)).unwrap();
        match result {
            ExecuteResult::Select { rows } => assert_eq!(rows.len(), expected_count, "Failed for: {}", condition),
            _ => panic!("Expected Select result"),
        }
    }
}
