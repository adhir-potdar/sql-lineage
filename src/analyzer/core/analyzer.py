"""Main SQL lineage analyzer - wrapper for refactored components."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression

from .models import LineageResult, TableMetadata
from .extractor import LineageExtractor
from .analyzers import (
    BaseAnalyzer,
    SelectAnalyzer,
    CTEAnalyzer,
    CTASAnalyzer,
    InsertAnalyzer,
    UpdateAnalyzer,
    TransformationAnalyzer,
    ChainBuilder
)
from ..metadata.registry import MetadataRegistry
from ..utils.validation import validate_sql_input


class SQLLineageAnalyzer(BaseAnalyzer):
    """Main SQL lineage analyzer class that wraps refactored analyzers."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the SQL lineage analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
        """
        super().__init__(dialect)
        
        # Initialize specialized analyzers
        self.select_analyzer = SelectAnalyzer(dialect)
        self.cte_analyzer = CTEAnalyzer(dialect)
        self.ctas_analyzer = CTASAnalyzer(dialect)
        self.insert_analyzer = InsertAnalyzer(dialect)
        self.update_analyzer = UpdateAnalyzer(dialect)
        self.transformation_analyzer = TransformationAnalyzer(dialect)
        self.chain_builder = ChainBuilder(dialect)
    
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
            
            if sql_type == "select":
                return self.select_analyzer.analyze(sql)
            elif sql_type == "cte":
                return self.cte_analyzer.analyze(sql)
            elif sql_type == "ctas":
                return self.ctas_analyzer.analyze(sql)
            elif sql_type == "insert":
                return self.insert_analyzer.analyze(sql)
            elif sql_type == "update":
                return self.update_analyzer.analyze(sql)
            else:
                return self._analyze_generic(sql)
                
        except Exception as e:
            return {
                'error': str(e),
                'query_structure': {},
                'transformations': {},
                'lineage': {},
                'result_columns': [],
                'source_tables': []
            }
    
    def _determine_sql_type(self, sql: str) -> str:
        """Determine the type of SQL statement."""
        sql_upper = sql.strip().upper()
        
        if sql_upper.startswith('WITH'):
            return "cte"
        elif sql_upper.startswith('CREATE TABLE') and 'AS' in sql_upper:
            return "ctas"
        elif sql_upper.startswith('INSERT'):
            return "insert"
        elif sql_upper.startswith('UPDATE'):
            return "update"
        elif sql_upper.startswith('SELECT'):
            return "select"
        else:
            return "generic"
    
    def _analyze_generic(self, sql: str) -> Dict[str, Any]:
        """Analyze generic SQL statements."""
        return {
            'query_structure': {'sql': sql, 'type': 'generic'},
            'transformations': {},
            'lineage': {},
            'result_columns': [],
            'source_tables': []
        }

    def analyze(self, sql: str, **kwargs) -> LineageResult:
        """
        Analyze SQL query and return lineage information.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional analysis options
            
        Returns:
            LineageResult object containing table and column lineage
        """
        try:
            # Validate input
            validate_sql_input(sql)
            
            # Parse SQL
            parsed = self._parse_sql(sql)
            if not parsed:
                raise ValueError("Unable to parse SQL query")
            
            # Extract lineage using the extractor
            table_lineage = self.extractor.extract_table_lineage(parsed)
            column_lineage = self.extractor.extract_column_lineage(parsed)
            
            # Collect metadata
            metadata = self._collect_metadata(table_lineage)
            
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=table_lineage,
                column_lineage=column_lineage,
                metadata=metadata
            )
            
        except Exception as e:
            from .models import TableLineage, ColumnLineage
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=TableLineage(),
                column_lineage=ColumnLineage(),
                metadata={},
                errors=[str(e)]
            )

    def analyze_file(self, file_path: str, **kwargs) -> LineageResult:
        """
        Analyze SQL file and return lineage information.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Additional analysis options
            
        Returns:
            LineageResult object
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql = f.read()
            return self.analyze(sql, **kwargs)
        except Exception as e:
            from .models import TableLineage, ColumnLineage
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=TableLineage(),
                column_lineage=ColumnLineage(),
                metadata={},
                errors=[f"File error: {str(e)}"]
            )

    def analyze_multiple(self, queries: list[str], **kwargs) -> list[LineageResult]:
        """
        Analyze multiple SQL queries.
        
        Args:
            queries: List of SQL query strings
            **kwargs: Additional analysis options
            
        Returns:
            List of LineageResult objects
        """
        return [self.analyze(query, **kwargs) for query in queries]

    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """Collect metadata for tables in lineage."""
        metadata = {}
        
        if not self.metadata_registry:
            return metadata
        
        # Collect metadata for upstream tables
        for target_table, source_tables in table_lineage.upstream.items():
            for source_table in source_tables:
                if source_table:
                    table_metadata = self.metadata_registry.get_table_metadata(source_table)
                    if table_metadata:
                        metadata[source_table] = table_metadata
        
        # Collect metadata for downstream tables  
        for source_table, target_tables in table_lineage.downstream.items():
            for target_table in target_tables:
                if target_table:
                    table_metadata = self.metadata_registry.get_table_metadata(target_table)
                    if table_metadata:
                        metadata[target_table] = table_metadata
        
        return metadata

    def set_metadata_registry(self, metadata_registry: MetadataRegistry) -> None:
        """
        Set metadata registry for table metadata lookup.
        
        Args:
            metadata_registry: MetadataRegistry instance
        """
        self.metadata_registry = metadata_registry
        # Also set for all specialized analyzers
        for analyzer in [self.select_analyzer, self.cte_analyzer, self.ctas_analyzer,
                        self.insert_analyzer, self.update_analyzer, self.transformation_analyzer]:
            analyzer.metadata_registry = metadata_registry

    def add_metadata_provider(self, provider) -> None:
        """
        Add metadata provider to registry.
        
        Args:
            provider: Metadata provider instance
        """
        if self.metadata_registry:
            self.metadata_registry.add_provider(provider)

    def get_lineage_json(self, sql: str, **kwargs) -> str:
        """
        Get the JSON representation of lineage analysis for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the lineage analysis
        """
        import json
        result = self.analyze(sql, **kwargs)
        return json.dumps(result.to_dict(), indent=2)

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
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        result = self.analyze(sql, **kwargs)
        
        # Get the appropriate lineage direction
        lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        
        # Build the chain starting from all tables at the current level
        chain = {}
        
        def build_chain(table_name: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth or if we have a circular dependency
            if current_depth > depth or table_name in visited_in_path:
                return {"table": table_name, "depth": current_depth - 1, "dependencies": []}
            
            # Add current table to the path to prevent cycles
            visited_in_path = visited_in_path | {table_name}
            
            dependencies = []
            if table_name in lineage_data:
                for dependent_table in lineage_data[table_name]:
                    dep_chain = build_chain(dependent_table, current_depth + 1, visited_in_path)
                    dependencies.append(dep_chain)
            
            return {
                "table": table_name,
                "depth": current_depth - 1,
                "dependencies": dependencies
            }
        
        # Start chain building from all root tables
        for table_name in lineage_data.keys():
            chain[table_name] = build_chain(table_name, 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": result.errors,
            "warnings": result.warnings
        }

    def get_table_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """
        Get the JSON representation of table lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the lineage chain
        """
        import json
        chain_data = self.get_table_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)

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
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        result = self.analyze(sql, **kwargs)
        
        # Get the appropriate lineage direction
        lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Build the chain starting from all columns at the current level
        chain = {}
        
        def build_column_chain(column_ref: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth or if we have a circular dependency
            if current_depth > depth or column_ref in visited_in_path:
                return {"column": column_ref, "depth": current_depth - 1, "dependencies": []}
            
            # Add current column to the path to prevent cycles
            visited_in_path = visited_in_path | {column_ref}
            
            dependencies = []
            if column_ref in lineage_data:
                for dependent_column in lineage_data[column_ref]:
                    dep_chain = build_column_chain(dependent_column, current_depth + 1, visited_in_path)
                    dependencies.append(dep_chain)
            
            return {
                "column": column_ref,
                "depth": current_depth - 1,
                "dependencies": dependencies
            }
        
        # Start chain building from all root columns
        for column_ref in lineage_data.keys():
            chain[column_ref] = build_column_chain(column_ref, 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": result.errors,
            "warnings": result.warnings
        }

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
        import json
        chain_data = self.get_column_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)