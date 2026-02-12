//! Integration tests for concurrent database operations

use pardusdb::concurrent::{ConcurrentDatabase, DatabasePool};
use pardusdb::{ExecuteResult, Value};
use std::sync::Arc;
use std::thread;

#[test]
fn test_concurrent_multi_thread_insert() {
    let db = Arc::new(ConcurrentDatabase::in_memory());
    let mut conn = db.connect();

    conn.execute("CREATE TABLE items (embedding VECTOR(3), value INTEGER, name TEXT);")
        .unwrap();

    // Spawn 10 threads, each inserting 10 items
    let mut handles = vec![];

    for t in 0..10 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            for i in 0..10 {
                let mut conn = db_clone.connect();
                let idx = t * 10 + i;
                conn.execute(&format!(
                    "INSERT INTO items (embedding, value, name) VALUES ([{}, 0.0, 0.0], {}, 'item_{}');",
                    idx as f32 / 100.0, idx, idx
                )).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all 100 items were inserted
    let result = conn.execute("SELECT * FROM items;").unwrap();
    if let ExecuteResult::Select { rows } = result {
        assert_eq!(rows.len(), 100);
    } else {
        panic!("Expected Select result");
    }
}

#[test]
fn test_concurrent_read_while_writing() {
    let db = Arc::new(ConcurrentDatabase::in_memory());
    let mut conn = db.connect();

    conn.execute("CREATE TABLE data (embedding VECTOR(2), counter INTEGER);")
        .unwrap();

    // Initial data
    for i in 0..50 {
        conn.execute(&format!(
            "INSERT INTO data (embedding, counter) VALUES ([0.0, 0.0], {});",
            i
        )).unwrap();
    }

    let db_writer = Arc::clone(&db);
    let db_reader = Arc::clone(&db);

    // Writer thread - adds more items
    let writer = thread::spawn(move || {
        for i in 50..100 {
            let mut conn = db_writer.connect();
            conn.execute(&format!(
                "INSERT INTO data (embedding, counter) VALUES ([1.0, 0.0], {});",
                i
            )).unwrap();
        }
    });

    // Reader thread - reads items
    let reader = thread::spawn(move || {
        let mut conn = db_reader.connect();
        let result = conn.execute("SELECT * FROM data;").unwrap();
        if let ExecuteResult::Select { rows } = result {
            rows.len()
        } else {
            0
        }
    });

    writer.join().unwrap();
    let read_count = reader.join().unwrap();

    // Reader should have seen at least 50 items (initial data)
    assert!(read_count >= 50);
}

#[test]
fn test_transaction_atomicity() {
    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute("CREATE TABLE accounts (embedding VECTOR(2), balance INTEGER, name TEXT);")
        .unwrap();

    // Insert two accounts
    conn.execute("INSERT INTO accounts (embedding, balance, name) VALUES ([0.0, 0.0], 100, 'Alice');")
        .unwrap();
    conn.execute("INSERT INTO accounts (embedding, balance, name) VALUES ([1.0, 0.0], 50, 'Bob');")
        .unwrap();

    // Transaction: transfer 30 from Alice to Bob
    conn.begin().unwrap();
    conn.execute("UPDATE accounts SET balance = 70 WHERE name = 'Alice';").unwrap();
    conn.execute("UPDATE accounts SET balance = 80 WHERE name = 'Bob';").unwrap();
    conn.commit().unwrap();

    // Verify balances
    let result = conn.execute("SELECT * FROM accounts WHERE name = 'Alice';").unwrap();
    if let ExecuteResult::Select { rows } = result {
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], Value::Integer(70));
    }
}

#[test]
fn test_transaction_rollback_on_error() {
    let db = ConcurrentDatabase::in_memory();
    let mut conn = db.connect();

    conn.execute("CREATE TABLE test (embedding VECTOR(2), value INTEGER);")
        .unwrap();
    conn.execute("INSERT INTO test (embedding, value) VALUES ([0.0, 0.0], 10);")
        .unwrap();

    // Start transaction, make changes, then rollback
    conn.begin().unwrap();
    conn.execute("UPDATE test SET value = 999;").unwrap();
    conn.rollback().unwrap();

    // Verify original value is preserved
    let result = conn.execute("SELECT * FROM test;").unwrap();
    if let ExecuteResult::Select { rows } = result {
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values[1], Value::Integer(10));
    }
}

#[test]
fn test_database_pool_clone() {
    let pool = DatabasePool::in_memory();
    let mut conn = pool.connect();

    conn.execute("CREATE TABLE shared (embedding VECTOR(2), data TEXT);")
        .unwrap();

    // Clone pool and use from multiple threads
    let mut handles = vec![];

    for i in 0..5 {
        let pool_clone = pool.clone();
        let handle = thread::spawn(move || {
            let mut conn = pool_clone.connect();
            conn.execute(&format!(
                "INSERT INTO shared (embedding, data) VALUES ([0.0, 0.0], 'thread_{}');",
                i
            )).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all inserts
    let mut conn = pool.connect();
    let result = conn.execute("SELECT * FROM shared;").unwrap();
    if let ExecuteResult::Select { rows } = result {
        assert_eq!(rows.len(), 5);
    }
}

#[test]
fn test_direct_api_thread_safety() {
    let db = Arc::new(ConcurrentDatabase::in_memory());
    let mut conn = db.connect();

    conn.execute("CREATE TABLE vectors (embedding VECTOR(3), label TEXT);")
        .unwrap();

    // Use direct API from multiple threads
    let mut handles = vec![];

    for i in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let mut conn = db_clone.connect();
            let vec = vec![i as f32, 0.0, 0.0];
            conn.insert_direct("vectors", vec, vec![
                ("label", Value::Text(format!("vec_{}", i)))
            ]).unwrap();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify via direct search API
    let conn = db.connect();
    let results = conn.search_similar("vectors", &[0.0, 0.0, 0.0], 10, 100).unwrap();
    assert_eq!(results.len(), 4);
}

#[test]
fn test_concurrent_similarity_search() {
    let db = Arc::new(ConcurrentDatabase::in_memory());
    let mut conn = db.connect();

    conn.execute("CREATE TABLE docs (embedding VECTOR(3), content TEXT);")
        .unwrap();

    // Insert 100 documents
    for i in 0..100 {
        let x = (i % 10) as f32 / 10.0;
        let y = ((i / 10) % 10) as f32 / 10.0;
        let z = (i / 100) as f32 / 10.0;
        conn.execute(&format!(
            "INSERT INTO docs (embedding, content) VALUES ([{:.1}, {:.1}, {:.1}], 'doc_{}');",
            x, y, z, i
        )).unwrap();
    }

    // Run concurrent similarity searches
    let mut handles = vec![];

    for q in 0..5 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let conn = db_clone.connect();
            let query = vec![q as f32 / 10.0, 0.0, 0.0];
            conn.search_similar("docs", &query, 5, 100).unwrap()
        });
        handles.push(handle);
    }

    for handle in handles {
        let results = handle.join().unwrap();
        assert_eq!(results.len(), 5);
    }
}

#[test]
fn test_concurrent_persistence() {
    let temp_path = std::env::temp_dir().join("pardusdb_concurrent_integration_test.pardus");
    let _ = std::fs::remove_file(&temp_path);

    // Create and populate with concurrent access
    {
        let db = Arc::new(ConcurrentDatabase::open(&temp_path).unwrap());
        let mut conn = db.connect();

        conn.execute("CREATE TABLE items (embedding VECTOR(2), name TEXT);")
            .unwrap();

        let mut handles = vec![];
        for i in 0..10 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let mut conn = db_clone.connect();
                conn.execute(&format!(
                    "INSERT INTO items (embedding, name) VALUES ([{}, 0.0], 'item_{}');",
                    i as f32 / 10.0, i
                )).unwrap();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        db.save().unwrap();
    }

    // Reopen and verify
    {
        let db = ConcurrentDatabase::open(&temp_path).unwrap();
        let mut conn = db.connect();

        let result = conn.execute("SELECT * FROM items;").unwrap();
        if let ExecuteResult::Select { rows } = result {
            assert_eq!(rows.len(), 10);
        }
    }

    let _ = std::fs::remove_file(&temp_path);
}
