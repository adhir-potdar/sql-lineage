"""Base parser class with common utilities."""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import sqlglot
from sqlglot import Expression
from ...utils.logging_config import get_logger


class BaseParser(ABC):
    """Base class for all SQL component parsers."""
    
    def __init__(self, dialect: str = "trino"):
        self.dialect = dialect
        self.logger = get_logger('parsers.base')
    
    def parse_sql(self, sql: str) -> Expression:
        """Parse SQL string into AST."""
        try:
            self.logger.debug(f"Parsing SQL with dialect {self.dialect}")
            result = sqlglot.parse_one(sql, dialect=self.dialect)
            self.logger.debug("SQL parsing successful")
            return result
        except Exception as e:
            self.logger.error(f"Failed to parse SQL: {e}")
            raise ValueError(f"Failed to parse SQL: {e}")
    
    def extract_table_name(self, table_node) -> str:
        """Extract clean table name from table node."""
        if hasattr(table_node, 'name'):
            return table_node.name
        elif hasattr(table_node, 'this'):
            return str(table_node.this)
        return str(table_node)
    
    def extract_column_name(self, column_node) -> str:
        """Extract clean column name from column node."""
        if hasattr(column_node, 'name'):
            return column_node.name
        elif hasattr(column_node, 'this'):
            return str(column_node.this)
        return str(column_node)
    
    def clean_column_reference(self, column_ref: str) -> str:
        """Remove table prefixes from column references."""
        if '.' in column_ref:
            return column_ref.split('.')[-1]
        return column_ref
    
    @abstractmethod
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse the SQL component and return structured data."""
        pass