/// Test: Vector dimension validation prevents crash/UB from mismatched dimensions.

use pardusdb::Database;

#[test]
fn test_short_vector_rejected() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (id INTEGER, embedding VECTOR(8), title TEXT);")
        .unwrap();

    // Insert correct 8-dim vector
    db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 'correct');")
        .unwrap();

    // Insert 2-dim vector into 8-dim table -> should be rejected
    let result = db.execute(
        "INSERT INTO docs (embedding, title) VALUES ([1.0, 2.0], 'too short');"
    );
    assert!(result.is_err(), "Short vector should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("dimension mismatch"),
        "Error should mention dimension mismatch, got: {}",
        err_msg
    );
}

#[test]
fn test_long_vector_rejected() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (id INTEGER, embedding VECTOR(3), title TEXT);")
        .unwrap();

    db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'correct');")
        .unwrap();

    // Insert 10-dim vector into 3-dim table -> should be rejected
    let result = db.execute(
        "INSERT INTO docs (embedding, title) VALUES ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 'too long');"
    );
    assert!(result.is_err(), "Long vector should be rejected");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("dimension mismatch"),
        "Error should mention dimension mismatch, got: {}",
        err_msg
    );
}

#[test]
fn test_correct_dimension_still_works() {
    let mut db = Database::in_memory();

    db.execute("CREATE TABLE docs (id INTEGER, embedding VECTOR(3), title TEXT);")
        .unwrap();

    // Correct dimension should work fine
    db.execute("INSERT INTO docs (embedding, title) VALUES ([1.0, 0.0, 0.0], 'first');")
        .unwrap();
    db.execute("INSERT INTO docs (embedding, title) VALUES ([0.0, 1.0, 0.0], 'second');")
        .unwrap();

    // Query should also work
    let result = db.execute("SELECT * FROM docs WHERE embedding SIMILARITY [1.0, 0.0, 0.0] LIMIT 5;");
    assert!(result.is_ok(), "Query with correct dimension should work");
}
