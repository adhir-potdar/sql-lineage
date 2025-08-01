"""Main SQL lineage analyzer wrapper."""

from typing import Optional, Dict, Any, List
import json

from .models import LineageResult
from ..metadata.registry import MetadataRegistry
from .analyzers import (
    BaseAnalyzer, SelectAnalyzer, InsertAnalyzer, UpdateAnalyzer,
    CTASAnalyzer, CTEAnalyzer, TableAnalyzer, ColumnAnalyzer,
    TransformationAnalyzer, ChainBuilder
)


class SQLLineageAnalyzer:
    """Main SQL lineage analyzer class that orchestrates specialized analyzers."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the SQL lineage analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
        """
        self.dialect = dialect
        self.metadata_registry = MetadataRegistry()
        
        # Initialize specialized analyzers
        self.base_analyzer = BaseAnalyzer(dialect)
        self.select_analyzer = SelectAnalyzer(dialect)
        self.insert_analyzer = InsertAnalyzer(dialect)
        self.update_analyzer = UpdateAnalyzer(dialect)
        self.ctas_analyzer = CTASAnalyzer(dialect)
        self.cte_analyzer = CTEAnalyzer(dialect)
        self.table_analyzer = TableAnalyzer(dialect)
        self.column_analyzer = ColumnAnalyzer(dialect)
        self.transformation_analyzer = TransformationAnalyzer(dialect)
        self.chain_builder = ChainBuilder(dialect)
        
        # Set metadata registry for all analyzers
        self._set_metadata_registry_for_all(self.metadata_registry)
    
    def _set_metadata_registry_for_all(self, metadata_registry: MetadataRegistry) -> None:
        """Set metadata registry for all analyzers."""
        analyzers = [
            self.base_analyzer, self.select_analyzer, self.insert_analyzer,
            self.update_analyzer, self.ctas_analyzer, self.cte_analyzer,
            self.table_analyzer, self.column_analyzer, self.transformation_analyzer,
            self.chain_builder
        ]
        
        for analyzer in analyzers:
            analyzer.set_metadata_registry(metadata_registry)
    
    def analyze_comprehensive(self, sql: str) -> Dict[str, Any]:
        """
        Comprehensive analysis using modular analyzers.
        
        Args:
            sql: SQL query string to analyze
            
        Returns:
            Comprehensive analysis result
        """
        try:
            # Determine SQL type and route to appropriate analyzer
            sql_type = self._determine_sql_type(sql)
            
            analysis_result = {
                'sql': sql,
                'sql_type': sql_type,
                'dialect': self.dialect,
                'success': True
            }
            
            try:
                if sql_type == 'SELECT':
                    analysis_result.update(self.select_analyzer.analyze_select(sql))
                elif sql_type == 'CTE':
                    analysis_result.update(self.cte_analyzer.analyze_cte(sql))
                elif sql_type == 'CTAS':
                    analysis_result.update(self.ctas_analyzer.analyze_ctas(sql))
                elif sql_type == 'INSERT':
                    analysis_result.update(self.insert_analyzer.analyze_insert(sql))
                elif sql_type == 'UPDATE':
                    analysis_result.update(self.update_analyzer.analyze_update(sql))
                else:
                    analysis_result.update(self.base_analyzer._analyze_generic(sql))
            except Exception as analysis_error:
                analysis_result['success'] = False
                analysis_result['error'] = f"Analysis error for {sql_type}: {str(analysis_error)}"
            
            return analysis_result
            
        except Exception as e:
            return {
                'sql': sql,
                'error': str(e),
                'success': False
            }
    
    def _determine_sql_type(self, sql: str) -> str:
        """Determine the type of SQL statement."""
        return self.base_analyzer._determine_sql_type(sql)
    
    def analyze(self, sql: str, **kwargs) -> LineageResult:
        """
        Analyze SQL query for lineage information.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            LineageResult containing table and column lineage information
        """
        return self.base_analyzer.analyze(sql, **kwargs)
    
    def analyze_file(self, file_path: str, **kwargs) -> LineageResult:
        """
        Analyze SQL file and return lineage result.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Additional options
            
        Returns:
            LineageResult containing table and column lineage
        """
        return self.base_analyzer.analyze_file(file_path, **kwargs)
    
    def analyze_multiple(self, queries: list[str], **kwargs) -> list[LineageResult]:
        """
        Analyze multiple SQL queries.
        
        Args:
            queries: List of SQL query strings
            **kwargs: Additional options
            
        Returns:
            List of LineageResult objects
        """
        return self.base_analyzer.analyze_multiple(queries, **kwargs)
    
    def set_metadata_registry(self, metadata_registry: MetadataRegistry) -> None:
        """
        Set the metadata registry for the analyzer.
        
        Args:
            metadata_registry: Metadata registry instance
        """
        self.metadata_registry = metadata_registry
        self._set_metadata_registry_for_all(metadata_registry)
    
    def add_metadata_provider(self, provider) -> None:
        """Add a metadata provider to the registry."""
        self.metadata_registry.add_provider(provider)
    
    def set_dialect(self, dialect: str) -> None:
        """Set the SQL dialect for parsing."""
        self.dialect = dialect
        # Update dialect for all analyzers
        analyzers = [
            self.base_analyzer, self.select_analyzer, self.insert_analyzer,
            self.update_analyzer, self.ctas_analyzer, self.cte_analyzer,
            self.table_analyzer, self.column_analyzer, self.transformation_analyzer,
            self.chain_builder
        ]
        
        for analyzer in analyzers:
            analyzer.set_dialect(dialect)
    
    def get_lineage_result(self, sql: str, **kwargs) -> LineageResult:
        """
        Get lineage result for SQL query.
        
        Args:
            sql: SQL query string
            **kwargs: Additional options
            
        Returns:
            LineageResult object
        """
        return self.base_analyzer.get_lineage_result(sql, **kwargs)
    
    def get_lineage_json(self, sql: str, **kwargs) -> str:
        """
        Get lineage as JSON string.
        
        Args:
            sql: SQL query string
            **kwargs: Additional options
            
        Returns:
            JSON string representation of lineage
        """
        return self.base_analyzer.get_lineage_json(sql, **kwargs)
    
    # Table lineage chain methods
    def get_table_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the table lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the lineage chain information
        """
        return self.table_analyzer.get_table_lineage_chain(sql, chain_type, depth, **kwargs)
    
    def get_table_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """
        Get the table lineage chain as JSON string.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream" 
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the lineage chain
        """
        return self.table_analyzer.get_table_lineage_chain_json(sql, chain_type, depth, **kwargs)
    
    # Column lineage chain methods
    def get_column_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the column lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the column lineage chain information
        """
        return self.column_analyzer.get_column_lineage_chain(sql, chain_type, depth, **kwargs)
    
    def get_column_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """
        Get the JSON representation of column lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the column lineage chain
        """
        return self.column_analyzer.get_column_lineage_chain_json(sql, chain_type, depth, **kwargs)
    
    # Comprehensive lineage chain methods (the main new functionality)
    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive lineage chain combining table and column lineage with transformations.
        Supports CTAS, CTE, and other SQL block types with detailed metadata and transformations.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing comprehensive lineage chain information
        """
        return self.chain_builder.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
    
    def get_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """
        Get the JSON representation of comprehensive lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the comprehensive lineage chain
        """
        return self.chain_builder.get_lineage_chain_json(sql, chain_type, depth, target_entity, **kwargs)