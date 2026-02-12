"""PardusDB exception classes."""


class PardusDBError(Exception):
    """Base exception for PardusDB errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class ConnectionError(PardusDBError):
    """Raised when connection to PardusDB fails."""

    pass


class QueryError(PardusDBError):
    """Raised when a query execution fails."""

    def __init__(self, message: str, query: str | None = None) -> None:
        self.query = query
        super().__init__(message)


class TableNotFoundError(PardusDBError):
    """Raised when a specified table doesn't exist."""

    def __init__(self, table_name: str) -> None:
        self.table_name = table_name
        super().__init__(f"Table not found: {table_name}")


class DimensionMismatchError(PardusDBError):
    """Raised when vector dimensions don't match table schema."""

    def __init__(self, expected: int, found: int) -> None:
        self.expected = expected
        self.found = found
        super().__init__(f"Vector dimension mismatch: expected {expected}, found {found}")
