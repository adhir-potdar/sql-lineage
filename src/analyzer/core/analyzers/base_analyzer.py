"""Base analyzer class with common functionality."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression

from ..models import LineageResult, TableMetadata
from ..extractor import LineageExtractor
from ...metadata.registry import MetadataRegistry
from ...utils.validation import validate_sql_input


class BaseAnalyzer:
    """Base analyzer class with common functionality for all SQL analyzers."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the base analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
        """
        self.dialect = dialect
        self.metadata_registry = MetadataRegistry()
        self.extractor = LineageExtractor()
    
    def _parse_sql(self, sql: str) -> Expression:
        """Parse SQL string into sqlglot Expression."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL: {str(e)}")
    
    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """
        Collect metadata for tables in the lineage.
        
        Args:
            table_lineage: Table lineage information
            
        Returns:
            Dictionary mapping table names to metadata
        """
        metadata = {}
        
        if hasattr(table_lineage, 'source_tables'):
            for table in table_lineage.source_tables:
                table_name = table.get('name') if isinstance(table, dict) else str(table)
                if table_name and self.metadata_registry:
                    try:
                        table_metadata = self.metadata_registry.get_table_metadata(table_name)
                        if table_metadata:
                            metadata[table_name] = table_metadata
                    except Exception:
                        # Skip if metadata not available
                        pass
        
        return metadata
    
    def set_metadata_registry(self, metadata_registry: MetadataRegistry) -> None:
        """
        Set the metadata registry for the analyzer.
        
        Args:
            metadata_registry: The metadata registry to use
        """
        self.metadata_registry = metadata_registry
    
    def add_metadata_provider(self, provider) -> None:
        """Add a metadata provider to the registry."""
        self.metadata_registry.add_provider(provider)
    
    def set_dialect(self, dialect: str) -> None:
        """Set the SQL dialect for parsing."""
        self.dialect = dialect
    
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
    
    def _validate_input(self, sql: str) -> None:
        """Validate SQL input."""
        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty")
        
        # Check for comment-only SQL
        sql_cleaned = sql.strip()
        # Remove single-line comments
        import re
        sql_no_comments = re.sub(r'--.*$', '', sql_cleaned, flags=re.MULTILINE)
        # Remove block comments
        sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)
        
        if not sql_no_comments.strip():
            raise ValueError("SQL query contains only comments")
        
        # Use validation utility if available
        try:
            validate_sql_input(sql)
        except ImportError:
            # Skip validation if utility not available
            pass
    
    def _handle_analysis_error(self, error: Exception, sql_type: str) -> Dict[str, Any]:
        """Handle analysis errors consistently."""
        return {
            'success': False,
            'error': f"Analysis error for {sql_type}: {str(error)}",
            'sql_type': sql_type
        }