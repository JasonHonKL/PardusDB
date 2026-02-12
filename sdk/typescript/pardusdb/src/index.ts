/**
 * PardusDB TypeScript SDK
 *
 * A simple, type-safe interface for PardusDB vector database.
 */

import { exec, spawn } from "child_process";
import { promisify } from "util";
import * as path from "path";
import * as fs from "fs";

const execAsync = promisify(exec);

// ==================== Types ====================

export interface VectorResult {
  id: number;
  distance: number;
  metadata: Record<string, unknown>;
}

export interface TableSchema {
  name: string;
  columns: Record<string, string>;
  vectorDimension?: number;
}

export type ColumnType = "TEXT" | "INTEGER" | "FLOAT" | "BOOLEAN" | "VECTOR";

export interface MetadataSchema {
  [columnName: string]: ColumnType | "str" | "int" | "float" | "bool" | "string";
}

export interface SearchOptions {
  k?: number;
  table?: string;
}

export interface InsertOptions {
  table?: string;
}

// ==================== Errors ====================

export class PardusDBError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PardusDBError";
  }
}

export class ConnectionError extends PardusDBError {
  constructor(message: string) {
    super(message);
    this.name = "ConnectionError";
  }
}

export class QueryError extends PardusDBError {
  public readonly query?: string;

  constructor(message: string, query?: string) {
    super(message);
    this.name = "QueryError";
    this.query = query;
  }
}

export class TableNotFoundError extends PardusDBError {
  public readonly tableName: string;

  constructor(tableName: string) {
    super(`Table not found: ${tableName}`);
    this.name = "TableNotFoundError";
    this.tableName = tableName;
  }
}

// ==================== Main Client ====================

/**
 * PardusDB client for vector database operations.
 *
 * @example
 * ```typescript
 * const db = new PardusDB("mydb.pardus");
 * await db.createTable("documents", 768, { title: "str", content: "str" });
 * await db.insert([0.1, 0.2, ...], { title: "Doc 1", content: "Hello" });
 * const results = await db.search([0.1, 0.2, ...], 5);
 * ```
 */
export class PardusDB {
  private dbPath: string | null;
  private binaryPath: string;
  private tables: Map<string, TableSchema> = new Map();
  private currentTable: string | null = null;

  constructor(dbPath?: string, binaryPath?: string) {
    this.dbPath = dbPath || null;
    this.binaryPath = binaryPath || this.findBinary();

    if (!fs.existsSync(this.binaryPath)) {
      throw new ConnectionError(`PardusDB binary not found at: ${this.binaryPath}`);
    }
  }

  private findBinary(): string {
    // Try to find in PATH
    try {
      const result = require("child_process").spawnSync("which", ["pardusdb"], {
        encoding: "utf-8",
      });
      if (result.status === 0 && result.stdout.trim()) {
        return result.stdout.trim();
      }
    } catch {
      // Ignore
    }

    throw new ConnectionError(
      "pardusdb binary not found in PATH. Please install PardusDB first: " +
      "git clone https://github.com/pardus-ai/pardusdb && cd pardusdb && ./setup.sh"
    );
  }

  private async execute(command: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const dbArg = this.dbPath || "";
      const args = dbArg ? [dbArg] : [];

      const proc = spawn(this.binaryPath, args, {
        stdio: ["pipe", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      proc.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      proc.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      proc.on("close", () => {
        resolve(stdout + stderr);
      });

      proc.on("error", (err) => {
        reject(new QueryError(`Query failed: ${err.message}`, command));
      });

      // Send command and quit
      proc.stdin.write(command + "\n");
      proc.stdin.write("quit\n");
      proc.stdin.end();
    });
  }

  // ==================== Table Operations ====================

  /**
   * Create a new table for vector storage.
   *
   * @param name - Table name
   * @param vectorDim - Dimension of vectors
   * @param metadataSchema - Column definitions
   *
   * @example
   * ```typescript
   * await db.createTable("documents", 768, {
   *   title: "str",
   *   content: "str",
   *   score: "float"
   * });
   * ```
   */
  async createTable(
    name: string,
    vectorDim: number,
    metadataSchema?: MetadataSchema,
    ifNotExists: boolean = true
  ): Promise<void> {
    const columns: string[] = [`embedding VECTOR(${vectorDim})`];

    const typeMap: Record<string, string> = {
      str: "TEXT",
      string: "TEXT",
      int: "INTEGER",
      integer: "INTEGER",
      float: "FLOAT",
      bool: "BOOLEAN",
      text: "TEXT",
    };

    if (metadataSchema) {
      for (const [colName, colType] of Object.entries(metadataSchema)) {
        const sqlType = typeMap[colType.toLowerCase()] || colType.toUpperCase();
        columns.push(`${colName} ${sqlType}`);
      }
    }

    const ifNotExistsStr = ifNotExists ? "IF NOT EXISTS " : "";
    const sql = `CREATE TABLE ${ifNotExistsStr}${name} (${columns.join(", ")})`;
    await this.execute(sql);

    // Store schema
    this.tables.set(name, {
      name,
      columns: { embedding: `VECTOR(${vectorDim})` },
      vectorDimension: vectorDim,
    });
    this.currentTable = name;
  }

  /**
   * Set the current table for operations.
   */
  use(name: string): this {
    this.currentTable = name;
    return this;
  }

  /**
   * Drop a table.
   */
  async dropTable(name: string, ifExists: boolean = true): Promise<void> {
    const ifExistsStr = ifExists ? "IF EXISTS " : "";
    await this.execute(`DROP TABLE ${ifExistsStr}${name}`);
    this.tables.delete(name);
    if (this.currentTable === name) {
      this.currentTable = null;
    }
  }

  /**
   * List all tables.
   */
  async listTables(): Promise<string[]> {
    const result = await this.execute("SHOW TABLES");
    const tables: string[] = [];
    for (const line of result.split("\n")) {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith("-") && !trimmed.startsWith("|") && !trimmed.startsWith("Tables")) {
        tables.push(trimmed);
      }
    }
    return tables;
  }

  // ==================== Insert Operations ====================

  /**
   * Insert a single vector with metadata.
   *
   * @param vector - The embedding vector
   * @param metadata - Optional metadata fields
   * @returns The inserted row ID
   *
   * @example
   * ```typescript
   * const id = await db.insert([0.1, 0.2, 0.3], { title: "Doc 1" });
   * ```
   */
  async insert(
    vector: number[],
    metadata?: Record<string, unknown>,
    options?: InsertOptions
  ): Promise<number> {
    const tableName = options?.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified. Use .use() or pass table option.");
    }

    const columns = ["embedding"];
    const values: string[] = [`[${vector.join(", ")}]`];

    if (metadata) {
      for (const [key, val] of Object.entries(metadata)) {
        columns.push(key);
        values.push(this.formatValue(val));
      }
    }

    const sql = `INSERT INTO ${tableName} (${columns.join(", ")}) VALUES (${values.join(", ")})`;
    const result = await this.execute(sql);

    // Parse row ID from output
    const idMatch = result.match(/id=(\d+)/);
    return idMatch ? parseInt(idMatch[1], 10) : 0;
  }

  /**
   * Insert multiple vectors efficiently.
   *
   * @example
   * ```typescript
   * const ids = await db.insertBatch(
   *   [[0.1, 0.2], [0.3, 0.4]],
   *   [{ title: "A" }, { title: "B" }]
   * );
   * ```
   */
  async insertBatch(
    vectors: number[][],
    metadataList?: Record<string, unknown>[],
    options?: InsertOptions
  ): Promise<number[]> {
    const tableName = options?.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified.");
    }

    if (metadataList && vectors.length !== metadataList.length) {
      throw new Error("Number of vectors and metadata must match");
    }

    const ids: number[] = [];
    for (let i = 0; i < vectors.length; i++) {
      const metadata = metadataList?.[i];
      const id = await this.insert(vectors[i], metadata, { table: tableName });
      ids.push(id);
    }

    return ids;
  }

  // ==================== Search Operations ====================

  /**
   * Search for similar vectors.
   *
   * @param queryVector - The query embedding
   * @param kOrOptions - Number of results or options object
   * @returns List of VectorResult sorted by distance
   *
   * @example
   * ```typescript
   * const results = await db.search([0.1, 0.2, ...], 5);
   * for (const r of results) {
   *   console.log(`ID: ${r.id}, Distance: ${r.distance}`);
   * }
   * ```
   */
  async search(
    queryVector: number[],
    kOrOptions?: number | SearchOptions
  ): Promise<VectorResult[]> {
    const options: SearchOptions =
      typeof kOrOptions === "number" ? { k: kOrOptions } : kOrOptions || {};

    const tableName = options.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified.");
    }

    const k = options.k || 10;
    const vectorStr = `[${queryVector.join(", ")}]`;

    const sql = `SELECT * FROM ${tableName} WHERE embedding SIMILARITY ${vectorStr} LIMIT ${k}`;
    const result = await this.execute(sql);

    // Parse results
    const results: VectorResult[] = [];
    for (const line of result.split("\n")) {
      if (line.includes("id=") && line.includes("distance=")) {
        try {
          const idMatch = line.match(/id=(\d+)/);
          const distMatch = line.match(/distance=([\d.]+)/);

          if (idMatch && distMatch) {
            results.push({
              id: parseInt(idMatch[1], 10),
              distance: parseFloat(distMatch[1]),
              metadata: {},
            });
          }
        } catch {
          continue;
        }
      }
    }

    return results;
  }

  // ==================== CRUD Operations ====================

  /**
   * Get a single row by ID.
   */
  async get(rowId: number, options?: { table?: string }): Promise<Record<string, unknown> | null> {
    const tableName = options?.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified.");
    }

    const sql = `SELECT * FROM ${tableName} WHERE id = ${rowId}`;
    const result = await this.execute(sql);

    if (result.includes("id=")) {
      return { _raw: result };
    }
    return null;
  }

  /**
   * Update metadata for a row.
   */
  async update(
    rowId: number,
    metadata: Record<string, unknown>,
    options?: { table?: string }
  ): Promise<boolean> {
    const tableName = options?.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified.");
    }

    const setParts: string[] = [];
    for (const [key, val] of Object.entries(metadata)) {
      setParts.push(`${key} = ${this.formatValue(val)}`);
    }

    const sql = `UPDATE ${tableName} SET ${setParts.join(", ")} WHERE id = ${rowId}`;
    await this.execute(sql);
    return true;
  }

  /**
   * Delete a row by ID.
   */
  async delete(rowId: number, options?: { table?: string }): Promise<boolean> {
    const tableName = options?.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified.");
    }

    const sql = `DELETE FROM ${tableName} WHERE id = ${rowId}`;
    await this.execute(sql);
    return true;
  }

  /**
   * Delete all rows from a table.
   */
  async deleteAll(options?: { table?: string }): Promise<boolean> {
    const tableName = options?.table || this.currentTable;
    if (!tableName) {
      throw new QueryError("No table specified.");
    }

    await this.execute(`DELETE FROM ${tableName}`);
    return true;
  }

  // ==================== Utility Methods ====================

  /**
   * Execute raw SQL command.
   */
  async rawSql(sql: string): Promise<string> {
    return this.execute(sql);
  }

  /**
   * Close the database connection.
   */
  async close(): Promise<void> {
    if (this.dbPath) {
      await this.execute(".save");
    }
  }

  private formatValue(value: unknown): string {
    if (typeof value === "string") {
      return `'${value}'`;
    } else if (typeof value === "boolean") {
      return value ? "true" : "false";
    } else if (value === null || value === undefined) {
      return "NULL";
    } else {
      return String(value);
    }
  }
}

export default PardusDB;
