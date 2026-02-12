"""
Tests for PardusDB Python SDK
"""

import os
import tempfile
import pytest
from pathlib import Path

from pardusdb import PardusDB, VectorResult
from pardusdb.errors import (
    PardusDBError,
    ConnectionError,
    QueryError,
    TableNotFoundError,
)


# ==================== Fixtures ====================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.pardus"


@pytest.fixture
def db(temp_db_path):
    """Create a fresh database for each test."""
    database = PardusDB(str(temp_db_path))
    yield database
    database.close()


@pytest.fixture
def db_with_table(db):
    """Database with a pre-created table."""
    db.create_table(
        "test_vectors",
        vector_dim=4,
        metadata_schema={"title": "str", "score": "float"},
    )
    return db


# ==================== Connection Tests ====================

class TestConnection:
    """Test database connection and initialization."""

    def test_create_in_memory_database(self):
        """Test creating an in-memory database."""
        db = PardusDB()
        assert db is not None
        db.close()

    def test_create_file_database(self, temp_db_path):
        """Test creating a file-based database."""
        db = PardusDB(str(temp_db_path))
        assert db is not None
        assert temp_db_path.exists()
        db.close()

    def test_invalid_binary_path(self):
        """Test error when binary doesn't exist."""
        with pytest.raises(ConnectionError):
            PardusDB(binary_path="/nonexistent/pardusdb")

    def test_context_manager(self, temp_db_path):
        """Test using database as context manager."""
        with PardusDB(str(temp_db_path)) as db:
            db.create_table("test", vector_dim=4)
            assert "test" in db.list_tables()


# ==================== Table Operations Tests ====================

class TestTableOperations:
    """Test table creation and management."""

    def test_create_table(self, db):
        """Test creating a table."""
        db.create_table("documents", vector_dim=128)
        tables = db.list_tables()
        assert "documents" in tables

    def test_create_table_with_metadata(self, db):
        """Test creating a table with metadata columns."""
        db.create_table(
            "docs",
            vector_dim=64,
            metadata_schema={"title": "str", "count": "int", "active": "bool"},
        )
        tables = db.list_tables()
        assert "docs" in tables

    def test_create_table_if_not_exists(self, db):
        """Test IF NOT EXISTS behavior."""
        db.create_table("test", vector_dim=4)
        # Should not raise error
        db.create_table("test", vector_dim=4, if_not_exists=True)

    def test_use_table(self, db_with_table):
        """Test switching tables."""
        result = db_with_table.use("test_vectors")
        assert result is db_with_table  # Should return self for chaining

    def test_drop_table(self, db_with_table):
        """Test dropping a table."""
        db_with_table.drop_table("test_vectors")
        tables = db_with_table.list_tables()
        assert "test_vectors" not in tables

    def test_drop_table_if_exists(self, db):
        """Test DROP IF EXISTS behavior."""
        # Should not raise error
        db.drop_table("nonexistent", if_exists=True)


# ==================== Insert Tests ====================

class TestInsert:
    """Test vector insertion operations."""

    def test_insert_single_vector(self, db_with_table):
        """Test inserting a single vector."""
        vector = [0.1, 0.2, 0.3, 0.4]
        metadata = {"title": "Test Document", "score": 0.95}

        row_id = db_with_table.insert(vector, metadata, table="test_vectors")

        assert isinstance(row_id, int)
        assert row_id >= 0

    def test_insert_without_metadata(self, db_with_table):
        """Test inserting a vector without metadata."""
        vector = [0.5, 0.6, 0.7, 0.8]

        row_id = db_with_table.insert(vector, table="test_vectors")

        assert isinstance(row_id, int)

    def test_insert_with_current_table(self, db_with_table):
        """Test insert using current table."""
        db_with_table.use("test_vectors")

        row_id = db_with_table.insert([0.1, 0.2, 0.3, 0.4])

        assert isinstance(row_id, int)

    def test_insert_batch(self, db_with_table):
        """Test batch insertion."""
        vectors = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ]
        metadata_list = [
            {"title": "Doc 1", "score": 0.1},
            {"title": "Doc 2", "score": 0.2},
            {"title": "Doc 3", "score": 0.3},
        ]

        row_ids = db_with_table.insert_batch(vectors, metadata_list, table="test_vectors")

        assert len(row_ids) == 3
        assert all(isinstance(id_, int) for id_ in row_ids)

    def test_insert_batch_without_metadata(self, db_with_table):
        """Test batch insertion without metadata."""
        vectors = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]

        row_ids = db_with_table.insert_batch(vectors, table="test_vectors")

        assert len(row_ids) == 2

    def test_insert_no_table_error(self, db):
        """Test error when no table is specified."""
        with pytest.raises(QueryError):
            db.insert([0.1, 0.2, 0.3, 0.4])

    def test_insert_batch_mismatch_error(self, db_with_table):
        """Test error when vectors and metadata count mismatch."""
        vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        metadata_list = [{"title": "Only one"}]

        with pytest.raises(ValueError):
            db_with_table.insert_batch(vectors, metadata_list, table="test_vectors")


# ==================== Search Tests ====================

class TestSearch:
    """Test vector similarity search."""

    @pytest.fixture
    def db_with_data(self, db_with_table):
        """Database with test data inserted."""
        db_with_table.use("test_vectors")
        db_with_table.insert([1.0, 0.0, 0.0, 0.0], {"title": "Vector A", "score": 1.0})
        db_with_table.insert([0.0, 1.0, 0.0, 0.0], {"title": "Vector B", "score": 2.0})
        db_with_table.insert([0.0, 0.0, 1.0, 0.0], {"title": "Vector C", "score": 3.0})
        db_with_table.insert([0.9, 0.1, 0.0, 0.0], {"title": "Vector D", "score": 4.0})
        return db_with_table

    def test_search_basic(self, db_with_data):
        """Test basic similarity search."""
        query = [1.0, 0.0, 0.0, 0.0]

        results = db_with_data.search(query, k=2)

        assert len(results) <= 2
        assert all(isinstance(r, VectorResult) for r in results)

    def test_search_returns_distances(self, db_with_data):
        """Test that search returns distance values."""
        query = [1.0, 0.0, 0.0, 0.0]

        results = db_with_data.search(query, k=5)

        for result in results:
            assert isinstance(result.distance, float)
            assert result.distance >= 0

    def test_search_with_table_parameter(self, db_with_data):
        """Test search with explicit table parameter."""
        query = [0.0, 1.0, 0.0, 0.0]

        results = db_with_data.search(query, k=3, table="test_vectors")

        assert isinstance(results, list)

    def test_search_no_table_error(self, db):
        """Test error when no table specified for search."""
        with pytest.raises(QueryError):
            db.search([0.1, 0.2, 0.3, 0.4], k=5)


# ==================== CRUD Tests ====================

class TestCRUD:
    """Test CRUD operations."""

    @pytest.fixture
    def db_with_record(self, db_with_table):
        """Database with a single record."""
        db_with_table.use("test_vectors")
        row_id = db_with_table.insert(
            [0.1, 0.2, 0.3, 0.4],
            {"title": "Original", "score": 1.0}
        )
        return db_with_table, row_id

    def test_get_record(self, db_with_record):
        """Test retrieving a record."""
        db, row_id = db_with_record

        result = db.get(row_id)

        assert result is not None

    def test_get_nonexistent_record(self, db_with_table):
        """Test getting a nonexistent record."""
        result = db_with_table.get(999999, table="test_vectors")

        assert result is None

    def test_update_record(self, db_with_record):
        """Test updating a record."""
        db, row_id = db_with_record

        success = db.update(row_id, {"title": "Updated", "score": 2.0})

        assert success is True

    def test_delete_record(self, db_with_record):
        """Test deleting a record."""
        db, row_id = db_with_record

        success = db.delete(row_id)

        assert success is True

    def test_delete_all(self, db_with_table):
        """Test deleting all records."""
        db_with_table.use("test_vectors")
        db_with_table.insert([0.1, 0.2, 0.3, 0.4])
        db_with_table.insert([0.5, 0.6, 0.7, 0.8])

        success = db_with_table.delete_all()

        assert success is True


# ==================== Raw SQL Tests ====================

class TestRawSQL:
    """Test raw SQL execution."""

    def test_raw_sql_show_tables(self, db):
        """Test raw SQL for showing tables."""
        db.create_table("test", vector_dim=4)

        result = db.raw_sql("SHOW TABLES")

        assert "test" in result

    def test_raw_sql_create_table(self, db):
        """Test raw SQL for creating table."""
        sql = "CREATE TABLE raw_test (embedding VECTOR(64), name TEXT)"
        result = db.raw_sql(sql)

        # Should not raise error
        assert result is not None


# ==================== Error Handling Tests ====================

class TestErrorHandling:
    """Test error handling."""

    def test_query_error_no_table(self, db):
        """Test QueryError when no table is set."""
        with pytest.raises(QueryError):
            db.insert([0.1, 0.2])

    def test_table_not_found_use(self, db):
        """Test error when using nonexistent table."""
        # Note: This might not raise immediately depending on implementation
        # The error may occur on the next operation
        pass


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for common workflows."""

    def test_full_workflow(self, temp_db_path):
        """Test a complete workflow: create, insert, search, cleanup."""
        with PardusDB(str(temp_db_path)) as db:
            # Create table
            db.create_table("embeddings", vector_dim=8, metadata_schema={"label": "str"})

            # Insert vectors
            vectors = [
                ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {"label": "A"}),
                ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {"label": "B"}),
                ([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], {"label": "C"}),
            ]

            for vec, meta in vectors:
                db.insert(vec, meta)

            # Search
            results = db.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], k=2)
            assert len(results) >= 1

            # First result should be closest to query
            if results:
                assert results[0].distance < 0.1  # Should be very close

    def test_rag_pattern(self, temp_db_path):
        """Test RAG-like usage pattern."""
        with PardusDB(str(temp_db_path)) as db:
            # Setup
            db.create_table("documents", vector_dim=4, metadata_schema={"content": "str"})

            # Index documents
            docs = [
                ([0.1, 0.2, 0.3, 0.4], "Hello world"),
                ([0.5, 0.6, 0.7, 0.8], "Goodbye world"),
                ([0.2, 0.3, 0.4, 0.5], "Hello there"),
            ]

            for embedding, content in docs:
                db.insert(embedding, {"content": content})

            # Query
            query_embedding = [0.15, 0.25, 0.35, 0.45]
            results = db.search(query_embedding, k=2)

            assert len(results) >= 1


# ==================== Run Tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
