/**
 * Tests for PardusDB TypeScript SDK
 */

import {
  PardusDB,
  VectorResult,
  PardusDBError,
  ConnectionError,
  QueryError,
  TableNotFoundError,
} from "./index";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";

// ==================== Test Utilities ====================

let tempDir: string;

function getTempDbPath(): string {
  return path.join(tempDir, `test-${Date.now()}.pardus`);
}

function cleanupTempDir(): void {
  if (tempDir && fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

// ==================== Setup & Teardown ====================

beforeAll(() => {
  tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "pardusdb-test-"));
});

afterAll(() => {
  cleanupTempDir();
});

// ==================== Connection Tests ====================

describe("Connection", () => {
  test("should create in-memory database", () => {
    const db = new PardusDB();
    expect(db).toBeDefined();
    db.close();
  });

  test("should create file-based database", async () => {
    const dbPath = getTempDbPath();
    const db = new PardusDB(dbPath);

    expect(db).toBeDefined();
    await db.close();

    // File should exist after operations
  });

  test("should throw error for invalid binary path", () => {
    expect(() => {
      new PardusDB(undefined, "/nonexistent/pardusdb");
    }).toThrow(ConnectionError);
  });
});

// ==================== Table Operations Tests ====================

describe("Table Operations", () => {
  let db: PardusDB;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = getTempDbPath();
    db = new PardusDB(dbPath);
  });

  afterEach(async () => {
    await db.close();
  });

  test("should create a table", async () => {
    await db.createTable("documents", 128);
    const tables = await db.listTables();
    expect(tables).toContain("documents");
  });

  test("should create table with metadata schema", async () => {
    await db.createTable("docs", 64, {
      title: "str",
      count: "int",
      active: "bool",
    });

    const tables = await db.listTables();
    expect(tables).toContain("docs");
  });

  test("should not error on IF NOT EXISTS", async () => {
    await db.createTable("test", 4);
    // Should not throw
    await db.createTable("test", 4, {}, true);
  });

  test("should use table", () => {
    const result = db.use("test");
    expect(result).toBe(db); // Should return self
  });

  test("should drop table", async () => {
    await db.createTable("to_delete", 4);
    await db.dropTable("to_delete");

    const tables = await db.listTables();
    expect(tables).not.toContain("to_delete");
  });
});

// ==================== Insert Tests ====================

describe("Insert Operations", () => {
  let db: PardusDB;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = getTempDbPath();
    db = new PardusDB(dbPath);
    await db.createTable("test_vectors", 4, { title: "str", score: "float" });
  });

  afterEach(async () => {
    await db.close();
  });

  test("should insert single vector", async () => {
    const vector = [0.1, 0.2, 0.3, 0.4];
    const metadata = { title: "Test Document", score: 0.95 };

    const rowId = await db.insert(vector, metadata, { table: "test_vectors" });

    expect(typeof rowId).toBe("number");
    expect(rowId).toBeGreaterThanOrEqual(0);
  });

  test("should insert vector without metadata", async () => {
    const vector = [0.5, 0.6, 0.7, 0.8];

    const rowId = await db.insert(vector, undefined, { table: "test_vectors" });

    expect(typeof rowId).toBe("number");
  });

  test("should insert using current table", async () => {
    db.use("test_vectors");

    const rowId = await db.insert([0.1, 0.2, 0.3, 0.4]);

    expect(typeof rowId).toBe("number");
  });

  test("should insert batch", async () => {
    const vectors = [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2],
    ];
    const metadataList = [
      { title: "Doc 1", score: 0.1 },
      { title: "Doc 2", score: 0.2 },
      { title: "Doc 3", score: 0.3 },
    ];

    const rowIds = await db.insertBatch(vectors, metadataList, { table: "test_vectors" });

    expect(rowIds).toHaveLength(3);
    expect(rowIds.every((id) => typeof id === "number")).toBe(true);
  });

  test("should insert batch without metadata", async () => {
    const vectors = [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
    ];

    const rowIds = await db.insertBatch(vectors, undefined, { table: "test_vectors" });

    expect(rowIds).toHaveLength(2);
  });

  test("should throw error when no table specified", async () => {
    await expect(db.insert([0.1, 0.2, 0.3, 0.4])).rejects.toThrow(QueryError);
  });

  test("should throw error on batch mismatch", async () => {
    const vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]];
    const metadataList = [{ title: "Only one" }];

    await expect(
      db.insertBatch(vectors, metadataList, { table: "test_vectors" })
    ).rejects.toThrow();
  });
});

// ==================== Search Tests ====================

describe("Search Operations", () => {
  let db: PardusDB;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = getTempDbPath();
    db = new PardusDB(dbPath);
    await db.createTable("test_vectors", 4, { title: "str", score: "float" });
    db.use("test_vectors");

    // Insert test data
    await db.insert([1.0, 0.0, 0.0, 0.0], { title: "Vector A", score: 1.0 });
    await db.insert([0.0, 1.0, 0.0, 0.0], { title: "Vector B", score: 2.0 });
    await db.insert([0.0, 0.0, 1.0, 0.0], { title: "Vector C", score: 3.0 });
    await db.insert([0.9, 0.1, 0.0, 0.0], { title: "Vector D", score: 4.0 });
  });

  afterEach(async () => {
    await db.close();
  });

  test("should perform basic search", async () => {
    const query = [1.0, 0.0, 0.0, 0.0];

    const results = await db.search(query, 2);

    expect(results.length).toBeLessThanOrEqual(2);
    results.forEach((r) => {
      expect(r).toHaveProperty("id");
      expect(r).toHaveProperty("distance");
    });
  });

  test("should return distance values", async () => {
    const query = [1.0, 0.0, 0.0, 0.0];

    const results = await db.search(query, 5);

    results.forEach((result) => {
      expect(typeof result.distance).toBe("number");
      expect(result.distance).toBeGreaterThanOrEqual(0);
    });
  });

  test("should search with options object", async () => {
    const query = [0.0, 1.0, 0.0, 0.0];

    const results = await db.search(query, { k: 3, table: "test_vectors" });

    expect(Array.isArray(results)).toBe(true);
  });

  test("should throw error when no table for search", async () => {
    const newDb = new PardusDB(getTempDbPath());
    await expect(newDb.search([0.1, 0.2, 0.3, 0.4], 5)).rejects.toThrow(QueryError);
    await newDb.close();
  });
});

// ==================== CRUD Tests ====================

describe("CRUD Operations", () => {
  let db: PardusDB;
  let dbPath: string;
  let rowId: number;

  beforeEach(async () => {
    dbPath = getTempDbPath();
    db = new PardusDB(dbPath);
    await db.createTable("test_vectors", 4, { title: "str", score: "float" });
    db.use("test_vectors");

    rowId = await db.insert([0.1, 0.2, 0.3, 0.4], { title: "Original", score: 1.0 });
  });

  afterEach(async () => {
    await db.close();
  });

  test("should get record", async () => {
    const result = await db.get(rowId);

    expect(result).not.toBeNull();
  });

  test("should return null for nonexistent record", async () => {
    const result = await db.get(999999, { table: "test_vectors" });

    expect(result).toBeNull();
  });

  test("should update record", async () => {
    const success = await db.update(rowId, { title: "Updated", score: 2.0 });

    expect(success).toBe(true);
  });

  test("should delete record", async () => {
    const success = await db.delete(rowId);

    expect(success).toBe(true);
  });

  test("should delete all records", async () => {
    await db.insert([0.1, 0.2, 0.3, 0.4]);
    await db.insert([0.5, 0.6, 0.7, 0.8]);

    const success = await db.deleteAll();

    expect(success).toBe(true);
  });
});

// ==================== Raw SQL Tests ====================

describe("Raw SQL", () => {
  let db: PardusDB;
  let dbPath: string;

  beforeEach(async () => {
    dbPath = getTempDbPath();
    db = new PardusDB(dbPath);
  });

  afterEach(async () => {
    await db.close();
  });

  test("should execute SHOW TABLES", async () => {
    await db.createTable("test", 4);

    const result = await db.rawSql("SHOW TABLES");

    expect(result).toContain("test");
  });

  test("should execute CREATE TABLE", async () => {
    const sql = "CREATE TABLE raw_test (embedding VECTOR(64), name TEXT)";
    const result = await db.rawSql(sql);

    expect(result).toBeDefined();
  });
});

// ==================== Integration Tests ====================

describe("Integration Tests", () => {
  test("should complete full workflow", async () => {
    const dbPath = getTempDbPath();
    const db = new PardusDB(dbPath);

    try {
      // Create table
      await db.createTable("embeddings", 8, { label: "str" });

      // Insert vectors
      const vectors = [
        { vec: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], meta: { label: "A" } },
        { vec: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], meta: { label: "B" } },
        { vec: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], meta: { label: "C" } },
      ];

      for (const { vec, meta } of vectors) {
        await db.insert(vec, meta);
      }

      // Search
      const results = await db.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2);

      expect(results.length).toBeGreaterThanOrEqual(1);

      // First result should be closest
      if (results.length > 0) {
        expect(results[0].distance).toBeLessThan(0.1);
      }
    } finally {
      await db.close();
    }
  });

  test("should support RAG pattern", async () => {
    const dbPath = getTempDbPath();
    const db = new PardusDB(dbPath);

    try {
      // Setup
      await db.createTable("documents", 4, { content: "str" });

      // Index documents
      const docs = [
        { embedding: [0.1, 0.2, 0.3, 0.4], content: "Hello world" },
        { embedding: [0.5, 0.6, 0.7, 0.8], content: "Goodbye world" },
        { embedding: [0.2, 0.3, 0.4, 0.5], content: "Hello there" },
      ];

      for (const doc of docs) {
        await db.insert(doc.embedding, { content: doc.content });
      }

      // Query
      const queryEmbedding = [0.15, 0.25, 0.35, 0.45];
      const results = await db.search(queryEmbedding, 2);

      expect(results.length).toBeGreaterThanOrEqual(1);
    } finally {
      await db.close();
    }
  });
});
