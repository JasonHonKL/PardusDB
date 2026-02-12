use pardusdb::Database;

fn main() {
    let mut db = Database::in_memory();

    // Test 1: UNIQUE constraint
    println!("=== Testing UNIQUE constraint ===");
    db.execute("CREATE TABLE users (embedding VECTOR(3), id INTEGER PRIMARY KEY, email TEXT UNIQUE);").unwrap();
    db.execute("INSERT INTO users (embedding, id, email) VALUES ([0.1, 0.2, 0.3], 1, 'test@example.com');").unwrap();

    // This should fail because of UNIQUE constraint
    match db.execute("INSERT INTO users (embedding, id, email) VALUES ([0.4, 0.5, 0.6], 2, 'test@example.com');") {
        Ok(_) => println!("ERROR: UNIQUE constraint was violated but insert succeeded!"),
        Err(e) => println!("OK: UNIQUE constraint prevented duplicate: {}", e),
    }

    // This should succeed - different email
    db.execute("INSERT INTO users (embedding, id, email) VALUES ([0.4, 0.5, 0.6], 2, 'other@example.com');").unwrap();
    println!("OK: Insert with different email succeeded");

    // Test 2: GROUP BY with aggregates
    println!("\n=== Testing GROUP BY ===");
    db.execute("CREATE TABLE sales (embedding VECTOR(3), id INTEGER, category TEXT, amount FLOAT);").unwrap();
    db.execute("INSERT INTO sales (embedding, id, category, amount) VALUES ([0.1, 0.2, 0.3], 1, 'A', 100.0);").unwrap();
    db.execute("INSERT INTO sales (embedding, id, category, amount) VALUES ([0.1, 0.2, 0.3], 2, 'A', 200.0);").unwrap();
    db.execute("INSERT INTO sales (embedding, id, category, amount) VALUES ([0.1, 0.2, 0.3], 3, 'B', 150.0);").unwrap();
    db.execute("INSERT INTO sales (embedding, id, category, amount) VALUES ([0.1, 0.2, 0.3], 4, 'B', 250.0);").unwrap();
    db.execute("INSERT INTO sales (embedding, id, category, amount) VALUES ([0.1, 0.2, 0.3], 5, 'B', 300.0);").unwrap();

    // Test GROUP BY with COUNT
    match db.execute("SELECT category, COUNT(*) FROM sales GROUP BY category;") {
        Ok(result) => println!("OK: GROUP BY COUNT succeeded: {:?}", result),
        Err(e) => println!("ERROR: GROUP BY COUNT failed: {}", e),
    }

    // Test GROUP BY with SUM
    match db.execute("SELECT category, SUM(amount) FROM sales GROUP BY category;") {
        Ok(result) => println!("OK: GROUP BY SUM succeeded: {:?}", result),
        Err(e) => println!("ERROR: GROUP BY SUM failed: {}", e),
    }

    // Test GROUP BY with AVG
    match db.execute("SELECT category, AVG(amount) FROM sales GROUP BY category;") {
        Ok(result) => println!("OK: GROUP BY AVG succeeded: {:?}", result),
        Err(e) => println!("ERROR: GROUP BY AVG failed: {}", e),
    }

    // Test 3: INNER JOIN
    println!("\n=== Testing INNER JOIN ===");
    db.execute("CREATE TABLE orders (embedding VECTOR(3), order_id INTEGER, user_id INTEGER, product TEXT);").unwrap();
    db.execute("INSERT INTO orders (embedding, order_id, user_id, product) VALUES ([0.1, 0.2, 0.3], 101, 1, 'Widget');").unwrap();
    db.execute("INSERT INTO orders (embedding, order_id, user_id, product) VALUES ([0.1, 0.2, 0.3], 102, 1, 'Gadget');").unwrap();
    db.execute("INSERT INTO orders (embedding, order_id, user_id, product) VALUES ([0.1, 0.2, 0.3], 103, 2, 'Doohickey');").unwrap();
    db.execute("INSERT INTO orders (embedding, order_id, user_id, product) VALUES ([0.1, 0.2, 0.3], 104, 99, 'Orphan');").unwrap();

    // INNER JOIN - should match orders with users
    match db.execute("SELECT * FROM orders INNER JOIN users ON orders.user_id = users.id;") {
        Ok(result) => println!("OK: INNER JOIN succeeded: {:?}", result),
        Err(e) => println!("ERROR: INNER JOIN failed: {}", e),
    }

    // Test 4: LEFT JOIN
    println!("\n=== Testing LEFT JOIN ===");
    match db.execute("SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id;") {
        Ok(result) => println!("OK: LEFT JOIN succeeded: {:?}", result),
        Err(e) => println!("ERROR: LEFT JOIN failed: {}", e),
    }

    println!("\n=== All tests completed ===");
}
