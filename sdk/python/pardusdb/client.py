"""
PardusDB Python SDK Client

Provides a simple, Pythonic interface for vector database operations.
"""

from __future__ import annotations

import subprocess
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .errors import ConnectionError, QueryError, TableNotFoundError, DimensionMismatchError


@dataclass
class VectorResult:
    """Result from a vector similarity search."""

    id: int
    distance: float
    metadata: dict[str, Any]

    def __repr__(self) -> str:
        return f"VectorResult(id={self.id}, distance={self.distance:.4f}, metadata={self.metadata})"


@dataclass
class TableSchema:
    """Schema definition for a table."""

    name: str
    columns: dict[str, str]  # column_name -> type
    vector_dimension: Optional[int] = None

    @property
    def vector_column(self) -> Optional[str]:
        """Find the vector column name."""
        for col, dtype in self.columns.items():
            if dtype.startswith("VECTOR"):
                return col
        return None


class PardusDB:
    """
    PardusDB client for vector database operations.

    Example:
        >>> db = PardusDB("mydb.pardus")
        >>> db.create_table("documents", vector_dim=768, metadata_schema={"title": "str", "content": "str"})
        >>> db.insert([0.1, 0.2, ...], metadata={"title": "Doc 1", "content": "Hello"})
        >>> results = db.search([0.1, 0.2, ...], k=5)
    """

    def __init__(
        self,
        path: Optional[str] = None,
        binary_path: Optional[str] = None,
    ) -> None:
        """
        Initialize PardusDB client.

        Args:
            path: Path to .pardus database file. If None, creates in-memory database.
            binary_path: Path to pardusdb binary. If None, searches PATH.
        """
        self.path = Path(path) if path else None
        self._binary_path = binary_path or self._find_binary()
        self._tables: dict[str, TableSchema] = {}
        self._current_table: Optional[str] = None

        # Verify binary exists
        if not os.path.isfile(self._binary_path):
            raise ConnectionError(f"PardusDB binary not found at: {self._binary_path}")

        # Create database file if needed
        if self.path and not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._execute(f".create {self.path}")
        elif self.path:
            self._execute(f".open {self.path}")

    def _find_binary(self) -> str:
        """Find pardusdb binary in PATH."""
        result = subprocess.run(["which", "pardusdb"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()

        raise ConnectionError(
            "pardusdb binary not found in PATH. Please install PardusDB first: "
            "git clone https://github.com/pardus-ai/pardusdb && cd pardusdb && ./setup.sh"
        )

    def _execute(self, command: str) -> str:
        """Execute a command and return the output."""
        try:
            # Create a temp file with the command
            with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
                f.write(command + "\n")
                f.write("quit\n")
                temp_path = f.name

            try:
                db_arg = str(self.path) if self.path else ""
                result = subprocess.run(
                    [self._binary_path, db_arg],
                    stdin=open(temp_path),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.stdout + result.stderr
            finally:
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            raise QueryError("Query timed out", command)
        except Exception as e:
            raise QueryError(f"Query failed: {e}", command)

    def _parse_value(self, value: str) -> Any:
        """Parse a string value from the database."""
        value = value.strip()

        if value.startswith("Vector([") and value.endswith("])"):
            # Parse vector
            inner = value[8:-2]
            return [float(x.strip()) for x in inner.split(",") if x.strip()]
        elif value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        elif value == "true":
            return True
        elif value == "false":
            return False
        elif "." in value:
            try:
                return float(value)
            except ValueError:
                return value
        else:
            try:
                return int(value)
            except ValueError:
                return value

    # ==================== Table Operations ====================

    def create_table(
        self,
        name: str,
        vector_dim: int,
        metadata_schema: Optional[dict[str, str]] = None,
        if_not_exists: bool = True,
    ) -> None:
        """
        Create a new table for vector storage.

        Args:
            name: Table name
            vector_dim: Dimension of vectors
            metadata_schema: Column definitions, e.g., {"title": "str", "score": "float"}
            if_not_exists: Don't error if table already exists

        Example:
            >>> db.create_table("documents", vector_dim=768,
            ...                 metadata_schema={"title": "str", "content": "str"})
        """
        columns = [f"embedding VECTOR({vector_dim})"]

        type_map = {
            "str": "TEXT",
            "string": "TEXT",
            "int": "INTEGER",
            "integer": "INTEGER",
            "float": "FLOAT",
            "bool": "BOOLEAN",
            "text": "TEXT",
        }

        if metadata_schema:
            for col_name, col_type in metadata_schema.items():
                sql_type = type_map.get(col_type.lower(), col_type.upper())
                columns.append(f"{col_name} {sql_type}")

        sql = f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{name} ({', '.join(columns)})"
        self._execute(sql)

        # Store schema
        self._tables[name] = TableSchema(
            name=name,
            columns={"embedding": f"VECTOR({vector_dim})"} | {k: type_map.get(v.lower(), v.upper()) for k, v in (metadata_schema or {}).items()},
            vector_dimension=vector_dim,
        )
        self._current_table = name

    def use(self, name: str) -> "PardusDB":
        """
        Set the current table for operations.

        Args:
            name: Table name

        Returns:
            self for method chaining
        """
        if name not in self._tables:
            # Try to load schema
            result = self._execute("SHOW TABLES")
            if name not in result:
                raise TableNotFoundError(name)
        self._current_table = name
        return self

    def drop_table(self, name: str, if_exists: bool = True) -> None:
        """Drop a table."""
        self._execute(f"DROP TABLE {'IF EXISTS ' if if_exists else ''}{name}")
        self._tables.pop(name, None)
        if self._current_table == name:
            self._current_table = None

    def list_tables(self) -> list[str]:
        """List all tables in the database."""
        result = self._execute("SHOW TABLES")
        # Parse table names from output
        tables = []
        for line in result.split("\n"):
            line = line.strip()
            if line and not line.startswith(("-", "|", "Tables")):
                tables.append(line)
        return tables

    # ==================== Insert Operations ====================

    def insert(
        self,
        vector: Sequence[float],
        metadata: Optional[dict[str, Any]] = None,
        table: Optional[str] = None,
    ) -> int:
        """
        Insert a single vector with metadata.

        Args:
            vector: The embedding vector
            metadata: Optional metadata fields
            table: Table name (uses current table if not specified)

        Returns:
            The inserted row ID

        Example:
            >>> db.insert([0.1, 0.2, 0.3], metadata={"title": "Doc 1"})
        """
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified. Use .use() or pass table parameter.")

        # Format vector
        vector_str = "[" + ", ".join(str(x) for x in vector) + "]"

        # Format values
        columns = ["embedding"]
        values = [vector_str]

        if metadata:
            for key, val in metadata.items():
                columns.append(key)
                if isinstance(val, str):
                    values.append(f"'{val}'")
                elif isinstance(val, bool):
                    values.append("true" if val else "false")
                else:
                    values.append(str(val))

        sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(values)})"
        result = self._execute(sql)

        # Parse row ID from output
        for line in result.split("\n"):
            if "id=" in line:
                try:
                    return int(line.split("id=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        return 0

    def insert_batch(
        self,
        vectors: Sequence[Sequence[float]],
        metadata_list: Optional[Sequence[dict[str, Any]]] = None,
        table: Optional[str] = None,
    ) -> list[int]:
        """
        Insert multiple vectors efficiently.

        Args:
            vectors: List of embedding vectors
            metadata_list: Optional list of metadata dicts (one per vector)
            table: Table name

        Returns:
            List of inserted row IDs

        Example:
            >>> db.insert_batch(
            ...     [[0.1, 0.2], [0.3, 0.4]],
            ...     metadata_list=[{"title": "A"}, {"title": "B"}]
            ... )
        """
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified.")

        if metadata_list and len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors and metadata must match")

        ids = []
        for i, vector in enumerate(vectors):
            metadata = metadata_list[i] if metadata_list else None
            row_id = self.insert(vector, metadata, table_name)
            ids.append(row_id)

        return ids

    # ==================== Search Operations ====================

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 10,
        table: Optional[str] = None,
        filter_: Optional[dict[str, Any]] = None,
    ) -> list[VectorResult]:
        """
        Search for similar vectors.

        Args:
            query_vector: The query embedding
            k: Number of results to return
            table: Table name
            filter_: Optional metadata filters (not yet supported)

        Returns:
            List of VectorResult objects sorted by distance

        Example:
            >>> results = db.search([0.1, 0.2, ...], k=5)
            >>> for r in results:
            ...     print(f"ID: {r.id}, Distance: {r.distance}")
        """
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified.")

        # Format vector
        vector_str = "[" + ", ".join(str(x) for x in query_vector) + "]"

        sql = f"SELECT * FROM {table_name} WHERE embedding SIMILARITY {vector_str} LIMIT {k}"
        result = self._execute(sql)

        # Parse results
        results = []
        lines = result.split("\n")
        for line in lines:
            if "id=" in line and "distance=" in line:
                try:
                    # Parse id
                    id_part = line.split("id=")[1].split(",")[0]
                    row_id = int(id_part.strip())

                    # Parse distance
                    dist_part = line.split("distance=")[1].split(",")[0]
                    distance = float(dist_part.strip())

                    # Parse values (simplified)
                    metadata = {}
                    if "values=" in line:
                        values_part = line.split("values=")[1]
                        # Basic parsing - could be improved
                        metadata["_raw"] = values_part.strip()

                    results.append(VectorResult(id=row_id, distance=distance, metadata=metadata))
                except (IndexError, ValueError):
                    continue

        return results

    # ==================== CRUD Operations ====================

    def get(
        self,
        row_id: int,
        table: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Get a single row by ID.

        Args:
            row_id: The row ID
            table: Table name

        Returns:
            Row data or None if not found
        """
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified.")

        sql = f"SELECT * FROM {table_name} WHERE id = {row_id}"
        result = self._execute(sql)

        # Parse result
        if "id=" in result:
            return {"_raw": result}
        return None

    def update(
        self,
        row_id: int,
        metadata: dict[str, Any],
        table: Optional[str] = None,
    ) -> bool:
        """
        Update metadata for a row.

        Args:
            row_id: The row ID
            metadata: Fields to update
            table: Table name

        Returns:
            True if successful
        """
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified.")

        set_parts = []
        for key, val in metadata.items():
            if isinstance(val, str):
                set_parts.append(f"{key} = '{val}'")
            elif isinstance(val, bool):
                set_parts.append(f"{key} = {'true' if val else 'false'}")
            else:
                set_parts.append(f"{key} = {val}")

        sql = f"UPDATE {table_name} SET {', '.join(set_parts)} WHERE id = {row_id}"
        self._execute(sql)
        return True

    def delete(
        self,
        row_id: int,
        table: Optional[str] = None,
    ) -> bool:
        """
        Delete a row by ID.

        Args:
            row_id: The row ID
            table: Table name

        Returns:
            True if successful
        """
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified.")

        sql = f"DELETE FROM {table_name} WHERE id = {row_id}"
        self._execute(sql)
        return True

    def delete_all(self, table: Optional[str] = None) -> bool:
        """Delete all rows from a table."""
        table_name = table or self._current_table
        if not table_name:
            raise QueryError("No table specified.")

        self._execute(f"DELETE FROM {table_name}")
        return True

    # ==================== Utility Methods ====================

    def raw_sql(self, sql: str) -> str:
        """
        Execute raw SQL command.

        Args:
            sql: SQL command

        Returns:
            Raw output from the database
        """
        return self._execute(sql)

    def close(self) -> None:
        """Close the database connection."""
        if self.path:
            self._execute(".save")

    def __enter__(self) -> "PardusDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        path_str = str(self.path) if self.path else "memory"
        return f"PardusDB('{path_str}')"
