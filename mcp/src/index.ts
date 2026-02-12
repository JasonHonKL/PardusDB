#!/usr/bin/env node

/**
 * PardusDB MCP Server
 *
 * Model Context Protocol server for PardusDB vector database.
 * Enables AI agents to perform vector similarity search and manage vector data.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";

// ==================== Types ====================

interface PardusDBConfig {
  dbPath: string | null;
  currentTable: string | null;
}

interface ToolResult {
  content: Array<{ type: string; text: string }>;
  isError?: boolean;
}

// ==================== PardusDB Client ====================

class PardusDBClient {
  private config: PardusDBConfig = {
    dbPath: null,
    currentTable: null,
  };

  async execute(command: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const dbArg = this.config.dbPath || "";
      const args = dbArg ? [dbArg] : [];

      const proc = spawn("pardusdb", args, {
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
        reject(new Error(`Failed to execute command: ${err.message}`));
      });

      proc.stdin.write(command + "\n");
      proc.stdin.write("quit\n");
      proc.stdin.end();
    });
  }

  setDbPath(dbPath: string | null): void {
    this.config.dbPath = dbPath;
  }

  getDbPath(): string | null {
    return this.config.dbPath;
  }

  setCurrentTable(tableName: string | null): void {
    this.config.currentTable = tableName;
  }

  getCurrentTable(): string | null {
    return this.config.currentTable;
  }
}

const dbClient = new PardusDBClient();

// ==================== Tool Handlers ====================

async function handleCreateDatabase(args: Record<string, unknown>): Promise<ToolResult> {
  const dbPath = args.path as string;

  if (!dbPath) {
    return {
      content: [{ type: "text", text: "Error: Database path is required" }],
      isError: true,
    };
  }

  try {
    // Create parent directories if needed
    const dir = path.dirname(dbPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    dbClient.setDbPath(dbPath);
    await dbClient.execute(`.create ${dbPath}`);

    return {
      content: [{ type: "text", text: `Database created successfully at: ${dbPath}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error creating database: ${error}` }],
      isError: true,
    };
  }
}

async function handleOpenDatabase(args: Record<string, unknown>): Promise<ToolResult> {
  const dbPath = args.path as string;

  if (!dbPath) {
    return {
      content: [{ type: "text", text: "Error: Database path is required" }],
      isError: true,
    };
  }

  if (!fs.existsSync(dbPath)) {
    return {
      content: [{ type: "text", text: `Error: Database file not found: ${dbPath}` }],
      isError: true,
    };
  }

  try {
    dbClient.setDbPath(dbPath);
    await dbClient.execute(`.open ${dbPath}`);

    return {
      content: [{ type: "text", text: `Database opened successfully: ${dbPath}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error opening database: ${error}` }],
      isError: true,
    };
  }
}

async function handleCreateTable(args: Record<string, unknown>): Promise<ToolResult> {
  const name = args.name as string;
  const vectorDim = args.vector_dim as number;
  const metadataSchema = args.metadata_schema as Record<string, string> | undefined;

  if (!name || !vectorDim) {
    return {
      content: [{ type: "text", text: "Error: Table name and vector_dim are required" }],
      isError: true,
    };
  }

  try {
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

    const sql = `CREATE TABLE IF NOT EXISTS ${name} (${columns.join(", ")})`;
    const result = await dbClient.execute(sql);

    dbClient.setCurrentTable(name);

    return {
      content: [{ type: "text", text: `Table '${name}' created successfully with ${vectorDim}-dimensional vectors.\n\nSQL: ${sql}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error creating table: ${error}` }],
      isError: true,
    };
  }
}

async function handleInsertVector(args: Record<string, unknown>): Promise<ToolResult> {
  const vector = args.vector as number[];
  const metadata = args.metadata as Record<string, unknown> | undefined;
  const table = (args.table as string) || dbClient.getCurrentTable();

  if (!vector || !Array.isArray(vector)) {
    return {
      content: [{ type: "text", text: "Error: Vector array is required" }],
      isError: true,
    };
  }

  if (!table) {
    return {
      content: [{ type: "text", text: "Error: No table specified. Use 'use_table' first or provide 'table' parameter." }],
      isError: true,
    };
  }

  try {
    const columns = ["embedding"];
    const values: string[] = [`[${vector.join(", ")}]`];

    if (metadata) {
      for (const [key, val] of Object.entries(metadata)) {
        columns.push(key);
        if (typeof val === "string") {
          values.push(`'${val}'`);
        } else if (typeof val === "boolean") {
          values.push(val ? "true" : "false");
        } else {
          values.push(String(val));
        }
      }
    }

    const sql = `INSERT INTO ${table} (${columns.join(", ")}) VALUES (${values.join(", ")})`;
    const result = await dbClient.execute(sql);

    // Parse row ID from result
    const idMatch = result.match(/id=(\d+)/);
    const rowId = idMatch ? idMatch[1] : "unknown";

    return {
      content: [{ type: "text", text: `Vector inserted successfully with ID: ${rowId}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error inserting vector: ${error}` }],
      isError: true,
    };
  }
}

async function handleBatchInsert(args: Record<string, unknown>): Promise<ToolResult> {
  const vectors = args.vectors as number[][];
  const metadataList = args.metadata_list as Record<string, unknown>[] | undefined;
  const table = (args.table as string) || dbClient.getCurrentTable();

  if (!vectors || !Array.isArray(vectors)) {
    return {
      content: [{ type: "text", text: "Error: Vectors array is required" }],
      isError: true,
    };
  }

  if (!table) {
    return {
      content: [{ type: "text", text: "Error: No table specified" }],
      isError: true,
    };
  }

  try {
    const results: string[] = [];

    for (let i = 0; i < vectors.length; i++) {
      const vector = vectors[i];
      const metadata = metadataList?.[i];

      const columns = ["embedding"];
      const values: string[] = [`[${vector.join(", ")}]`];

      if (metadata) {
        for (const [key, val] of Object.entries(metadata)) {
          columns.push(key);
          if (typeof val === "string") {
            values.push(`'${val}'`);
          } else if (typeof val === "boolean") {
            values.push(val ? "true" : "false");
          } else {
            values.push(String(val));
          }
        }
      }

      const sql = `INSERT INTO ${table} (${columns.join(", ")}) VALUES (${values.join(", ")})`;
      const result = await dbClient.execute(sql);

      const idMatch = result.match(/id=(\d+)/);
      if (idMatch) {
        results.push(idMatch[1]);
      }
    }

    return {
      content: [{ type: "text", text: `Batch insert completed. Inserted ${results.length} vectors with IDs: ${results.join(", ")}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error during batch insert: ${error}` }],
      isError: true,
    };
  }
}

async function handleSearchSimilar(args: Record<string, unknown>): Promise<ToolResult> {
  const queryVector = args.query_vector as number[];
  const k = (args.k as number) || 10;
  const table = (args.table as string) || dbClient.getCurrentTable();

  if (!queryVector || !Array.isArray(queryVector)) {
    return {
      content: [{ type: "text", text: "Error: query_vector array is required" }],
      isError: true,
    };
  }

  if (!table) {
    return {
      content: [{ type: "text", text: "Error: No table specified" }],
      isError: true,
    };
  }

  try {
    const vectorStr = `[${queryVector.join(", ")}]`;
    const sql = `SELECT * FROM ${table} WHERE embedding SIMILARITY ${vectorStr} LIMIT ${k}`;
    const result = await dbClient.execute(sql);

    return {
      content: [{ type: "text", text: `Search Results:\n\n${result}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error searching: ${error}` }],
      isError: true,
    };
  }
}

async function handleExecuteSQL(args: Record<string, unknown>): Promise<ToolResult> {
  const sql = args.sql as string;

  if (!sql) {
    return {
      content: [{ type: "text", text: "Error: SQL query is required" }],
      isError: true,
    };
  }

  try {
    const result = await dbClient.execute(sql);

    return {
      content: [{ type: "text", text: `Query Result:\n\n${result}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error executing SQL: ${error}` }],
      isError: true,
    };
  }
}

async function handleListTables(): Promise<ToolResult> {
  try {
    const result = await dbClient.execute("SHOW TABLES");

    return {
      content: [{ type: "text", text: `Tables:\n\n${result}` }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error listing tables: ${error}` }],
      isError: true,
    };
  }
}

async function handleUseTable(args: Record<string, unknown>): Promise<ToolResult> {
  const table = args.table as string;

  if (!table) {
    return {
      content: [{ type: "text", text: "Error: Table name is required" }],
      isError: true,
    };
  }

  dbClient.setCurrentTable(table);

  return {
    content: [{ type: "text", text: `Now using table: ${table}` }],
  };
}

async function handleGetStatus(): Promise<ToolResult> {
  const dbPath = dbClient.getDbPath();
  const currentTable = dbClient.getCurrentTable();

  let status = "PardusDB Status:\n\n";
  status += `Database: ${dbPath || "Not opened (in-memory)"}\n`;
  status += `Current Table: ${currentTable || "None selected"}\n`;

  if (dbPath && fs.existsSync(dbPath)) {
    const stats = fs.statSync(dbPath);
    status += `Database Size: ${(stats.size / 1024).toFixed(2)} KB\n`;
  }

  return {
    content: [{ type: "text", text: status }],
  };
}

// ==================== Tool Definitions ====================

const TOOLS = [
  {
    name: "pardusdb_create_database",
    description: "Create a new PardusDB database file at the specified path",
    inputSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Path for the new .pardus database file (e.g., 'data/vectors.pardus')",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "pardusdb_open_database",
    description: "Open an existing PardusDB database file",
    inputSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Path to the existing .pardus database file",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "pardusdb_create_table",
    description: "Create a new table for storing vectors with optional metadata columns",
    inputSchema: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "Name of the table",
        },
        vector_dim: {
          type: "number",
          description: "Dimension of the vectors (e.g., 768 for sentence transformers)",
        },
        metadata_schema: {
          type: "object",
          description: "Optional metadata columns: {column_name: type}. Types: str, int, float, bool",
          additionalProperties: { type: "string" },
        },
      },
      required: ["name", "vector_dim"],
    },
  },
  {
    name: "pardusdb_insert_vector",
    description: "Insert a single vector with optional metadata into a table",
    inputSchema: {
      type: "object",
      properties: {
        vector: {
          type: "array",
          items: { type: "number" },
          description: "The embedding vector (array of floats)",
        },
        metadata: {
          type: "object",
          description: "Optional metadata to store with the vector",
        },
        table: {
          type: "string",
          description: "Table name (uses current table if not specified)",
        },
      },
      required: ["vector"],
    },
  },
  {
    name: "pardusdb_batch_insert",
    description: "Insert multiple vectors efficiently in a batch",
    inputSchema: {
      type: "object",
      properties: {
        vectors: {
          type: "array",
          items: {
            type: "array",
            items: { type: "number" },
          },
          description: "Array of embedding vectors",
        },
        metadata_list: {
          type: "array",
          items: { type: "object" },
          description: "Optional array of metadata objects (one per vector)",
        },
        table: {
          type: "string",
          description: "Table name (uses current table if not specified)",
        },
      },
      required: ["vectors"],
    },
  },
  {
    name: "pardusdb_search_similar",
    description: "Search for vectors similar to a query vector using cosine similarity",
    inputSchema: {
      type: "object",
      properties: {
        query_vector: {
          type: "array",
          items: { type: "number" },
          description: "The query embedding vector",
        },
        k: {
          type: "number",
          description: "Number of results to return (default: 10)",
        },
        table: {
          type: "string",
          description: "Table name (uses current table if not specified)",
        },
      },
      required: ["query_vector"],
    },
  },
  {
    name: "pardusdb_execute_sql",
    description: "Execute raw SQL commands on the database",
    inputSchema: {
      type: "object",
      properties: {
        sql: {
          type: "string",
          description: "SQL command to execute",
        },
      },
      required: ["sql"],
    },
  },
  {
    name: "pardusdb_list_tables",
    description: "List all tables in the current database",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "pardusdb_use_table",
    description: "Set the current table for subsequent operations",
    inputSchema: {
      type: "object",
      properties: {
        table: {
          type: "string",
          description: "Name of the table to use",
        },
      },
      required: ["table"],
    },
  },
  {
    name: "pardusdb_status",
    description: "Get the current status of the database connection",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
];

// ==================== Server Setup ====================

const server = new Server(
  {
    name: "pardusdb-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Handle list tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "pardusdb_create_database":
      return await handleCreateDatabase(args || {});
    case "pardusdb_open_database":
      return await handleOpenDatabase(args || {});
    case "pardusdb_create_table":
      return await handleCreateTable(args || {});
    case "pardusdb_insert_vector":
      return await handleInsertVector(args || {});
    case "pardusdb_batch_insert":
      return await handleBatchInsert(args || {});
    case "pardusdb_search_similar":
      return await handleSearchSimilar(args || {});
    case "pardusdb_execute_sql":
      return await handleExecuteSQL(args || {});
    case "pardusdb_list_tables":
      return await handleListTables();
    case "pardusdb_use_table":
      return await handleUseTable(args || {});
    case "pardusdb_status":
      return await handleGetStatus();
    default:
      return {
        content: [{ type: "text", text: `Unknown tool: ${name}` }],
        isError: true,
      };
  }
});

// Handle list resources
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return { resources: [] };
});

// Handle read resource
server.setRequestHandler(ReadResourceRequestSchema, async () => {
  throw new Error("No resources available");
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("PardusDB MCP server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
