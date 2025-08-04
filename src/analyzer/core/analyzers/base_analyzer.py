"""Base analyzer class."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression
import json

from ..models import LineageResult, TableMetadata
from ..extractor import LineageExtractor
from ..parsers import SelectParser, TransformationParser, CTEParser, CTASParser, InsertParser, UpdateParser
from ...utils.validation import validate_sql_input
from ...utils.sql_parsing_utils import TableNameRegistry, CompatibilityMode


class BaseAnalyzer:
    """Base analyzer class for SQL lineage analysis."""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = CompatibilityMode.FULL, table_registry: TableNameRegistry = None):
        """
        Initialize the base analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
            compatibility_mode: Table name normalization mode
            table_registry: Shared table name registry (created if None)
        """
        self.dialect = dialect
        self.compatibility_mode = compatibility_mode
        self.table_registry = table_registry or TableNameRegistry(dialect, compatibility_mode)
        self.extractor = LineageExtractor(dialect, compatibility_mode)
        
        # Initialize modular parsers as core components
        self.select_parser = SelectParser(dialect)
        self.transformation_parser = TransformationParser(dialect)
        self.cte_parser = CTEParser(dialect)
        self.ctas_parser = CTASParser(dialect)
        self.insert_parser = InsertParser(dialect)
        self.update_parser = UpdateParser(dialect)
    
    def _determine_sql_type(self, sql: str) -> str:
        """Determine the type of SQL statement."""
        # Clean and normalize the SQL
        sql_cleaned = ' '.join(sql.strip().split()).upper()
        
        if sql_cleaned.startswith('WITH'):
            return 'CTE'
        elif sql_cleaned.startswith('CREATE TABLE') and 'AS SELECT' in sql_cleaned:
            return 'CTAS'
        elif sql_cleaned.startswith('SELECT'):
            return 'SELECT'
        elif sql_cleaned.startswith('INSERT'):
            return 'INSERT'
        elif sql_cleaned.startswith('UPDATE'):
            return 'UPDATE'
        elif sql_cleaned.startswith('DELETE'):
            return 'DELETE'
        else:
            return 'UNKNOWN'