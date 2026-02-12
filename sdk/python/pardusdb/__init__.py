"""
PardusDB Python SDK

A simple, Pythonic interface for PardusDB vector database.
"""

from .client import PardusDB, VectorResult
from .errors import PardusDBError, ConnectionError, QueryError

__version__ = "0.1.0"
__all__ = ["PardusDB", "VectorResult", "PardusDBError", "ConnectionError", "QueryError"]
