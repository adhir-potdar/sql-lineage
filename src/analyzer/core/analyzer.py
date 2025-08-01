"""Main SQL lineage analyzer - refactored to use modular analyzers."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression
import json

from .models import LineageResult, TableMetadata
from .extractor import LineageExtractor
from .parsers import SelectParser, TransformationParser, CTEParser, CTASParser, InsertParser, UpdateParser
from ..metadata.registry import MetadataRegistry
from ..utils.validation import validate_sql_input

# Import modular analyzers
from .analyzers import (
    BaseAnalyzer,
    SelectAnalyzer,
    CTEAnalyzer,
    CTASAnalyzer, 
    InsertAnalyzer,
    UpdateAnalyzer,
    ChainBuilder,
    TransformationExtractor
)


class SQLLineageAnalyzer(BaseAnalyzer):
    """Main SQL lineage analyzer class - refactored to use modular analyzers."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the SQL lineage analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
        """
        super().__init__(dialect)
        self.metadata_registry = MetadataRegistry()
        self.extractor = LineageExtractor()
        
        # Initialize modular parsers as core components
        self.select_parser = SelectParser(dialect)
        self.transformation_parser = TransformationParser(dialect)
        self.cte_parser = CTEParser(dialect)
        self.ctas_parser = CTASParser(dialect)
        self.insert_parser = InsertParser(dialect)
        self.update_parser = UpdateParser(dialect)
        
        # Initialize modular analyzers
        self.select_analyzer = SelectAnalyzer(dialect)
        self.cte_analyzer = CTEAnalyzer(dialect)
        self.ctas_analyzer = CTASAnalyzer(dialect)
        self.insert_analyzer = InsertAnalyzer(dialect)
        self.update_analyzer = UpdateAnalyzer(dialect)
        self.chain_builder = ChainBuilder(dialect)
        self.transformation_extractor = TransformationExtractor(dialect)
        
        # Inject dependencies into modular analyzers
        self._inject_dependencies()
    
    def _inject_dependencies(self):
        """Inject parser dependencies into modular analyzers."""
        # Inject parsers into analyzers
        self.select_analyzer.select_parser = self.select_parser
        self.select_analyzer.transformation_parser = self.transformation_parser
        
        self.cte_analyzer.cte_parser = self.cte_parser
        self.cte_analyzer.main_analyzer = self
        
        self.ctas_analyzer.ctas_parser = self.ctas_parser
        self.ctas_analyzer.transformation_parser = self.transformation_parser
        
        self.insert_analyzer.insert_parser = self.insert_parser
        
        self.update_analyzer.update_parser = self.update_parser
        
        self.chain_builder.transformation_extractor = self.transformation_extractor
        
        self.transformation_extractor.select_parser = self.select_parser
        self.transformation_extractor.cte_parser = self.cte_parser
        self.transformation_extractor.ctas_parser = self.ctas_parser
    
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
                    analysis_result.update(self._analyze_select(sql))
                elif sql_type == 'CTE':
                    analysis_result.update(self._analyze_cte(sql))
                elif sql_type == 'CTAS':
                    analysis_result.update(self._analyze_ctas(sql))
                elif sql_type == 'INSERT':
                    analysis_result.update(self._analyze_insert(sql))
                elif sql_type == 'UPDATE':
                    analysis_result.update(self._analyze_update(sql))
                else:
                    analysis_result.update(self._analyze_generic(sql))
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
    
    def _analyze_select(self, sql: str) -> Dict[str, Any]:
        """Analyze simple SELECT statement using SelectAnalyzer."""
        select_data = self.select_parser.parse(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'query_structure': select_data,
            'transformations': transformation_data,
            'lineage': self.select_analyzer.build_select_lineage(select_data, transformation_data),
            'result_columns': self.select_analyzer.extract_result_columns(select_data),
            'source_tables': self.select_analyzer.extract_source_tables(select_data)
        }
    
    def _analyze_cte(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE statement using CTEAnalyzer."""
        cte_data = self.cte_parser.parse(sql)
        cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'cte_structure': cte_data,
            'cte_lineage': cte_lineage,
            'transformations': transformation_data,
            'execution_order': cte_lineage.get('execution_order', []),
            'final_result': cte_lineage.get('final_result', {}),
            'cte_dependencies': cte_data.get('cte_dependencies', {})
        }
    
    def _analyze_ctas(self, sql: str) -> Dict[str, Any]:
        """Analyze CREATE TABLE AS SELECT statement using CTASAnalyzer."""
        return self.ctas_analyzer.analyze_ctas(sql)
    
    def _analyze_insert(self, sql: str) -> Dict[str, Any]:
        """Analyze INSERT statement using InsertAnalyzer."""
        return self.insert_analyzer.analyze_insert(sql)
    
    def _analyze_update(self, sql: str) -> Dict[str, Any]:
        """Analyze UPDATE statement using UpdateAnalyzer."""
        return self.update_analyzer.analyze_update(sql)
    
    def _analyze_generic(self, sql: str) -> Dict[str, Any]:
        """Analyze other types of SQL statements."""
        # For now, try to extract what we can using the select parser
        try:
            select_data = self.select_parser.parse(sql)
            return {
                'partial_analysis': select_data,
                'note': 'Generic analysis - limited information available'
            }
        except Exception:
            return {
                'error': 'Unable to parse this SQL type'
            }
    
    # Delegate helper methods to appropriate analyzers
    def _build_select_lineage(self, select_data: Dict[str, Any], transformation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build lineage chain for SELECT statement using SelectAnalyzer."""
        return self.select_analyzer.build_select_lineage(select_data, transformation_data)
    
    def _get_columns_used_from_table(self, table: Dict[str, Any], select_data: Dict[str, Any]) -> List[str]:
        """Get columns used from a specific table using SelectAnalyzer."""
        return self.select_analyzer.get_columns_used_from_table(table, select_data)
    
    def _extract_result_columns(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract result columns with their metadata using SelectAnalyzer."""
        return self.select_analyzer.extract_result_columns(select_data)
    
    def _extract_source_tables(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source tables with their metadata using SelectAnalyzer."""
        return self.select_analyzer.extract_source_tables(select_data)
    
    def analyze(self, sql: str, **kwargs) -> LineageResult:
        """
        Analyze SQL query for lineage information using the original implementation.
        
        This maintains full backward compatibility by using the original extractor.
        """
        # Validate input
        validation_error = validate_sql_input(sql)
        if validation_error:
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[validation_error]
            )
        
        try:
            # Use the legacy extractor approach to maintain exact compatibility
            # Parse SQL
            expression = self._parse_sql(sql)
            
            # Extract lineage using the original extractor
            table_lineage = self.extractor.extract_table_lineage(expression)
            column_lineage = self.extractor.extract_column_lineage(expression)
            
            # Get metadata for involved tables
            metadata = self._collect_metadata(table_lineage)
            
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=table_lineage,
                column_lineage=column_lineage,
                metadata=metadata
            )
            
        except Exception as e:
            # Return result with error
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[str(e)]
            )
    
    def analyze_file(self, file_path: str, **kwargs) -> LineageResult:
        """
        Analyze SQL file for lineage information.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Additional options
            
        Returns:
            LineageResult object containing analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                sql_content = file.read()
            return self.analyze(sql_content, **kwargs)
        except Exception as e:
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"File read error: {str(e)}"]
            )
    
    def analyze_multiple(self, queries: list[str], **kwargs) -> list[LineageResult]:
        """
        Analyze multiple SQL queries for lineage information.
        
        Args:
            queries: List of SQL query strings
            **kwargs: Additional options
            
        Returns:
            List of LineageResult objects
        """
        return [self.analyze(query, **kwargs) for query in queries]
    
    def _parse_sql(self, sql: str) -> Expression:
        """Parse SQL string into sqlglot expression."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL: {str(e)}")
    
    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """Collect metadata for tables involved in lineage."""
        metadata = {}
        
        # Collect all table names from upstream and downstream - following backup implementation
        all_tables = set()
        for target, sources in table_lineage.upstream.items():
            all_tables.add(target)
            all_tables.update(sources)
        
        # Get metadata for each table - only add if not None (following backup pattern)
        for table_name in all_tables:
            table_metadata = self.metadata_registry.get_table_metadata(table_name)
            if table_metadata:
                metadata[table_name] = table_metadata
        
        return metadata
    
    def set_metadata_registry(self, metadata_registry: MetadataRegistry) -> None:
        """
        Set the metadata registry for the analyzer.
        
        Args:
            metadata_registry: MetadataRegistry instance
        """
        self.metadata_registry = metadata_registry
    
    def add_metadata_provider(self, provider) -> None:
        """Add a metadata provider to the registry."""
        self.metadata_registry.add_provider(provider)
    
    def set_dialect(self, dialect: str) -> None:
        """Set the SQL dialect for parsing."""
        self.dialect = dialect
    
    def get_lineage_result(self, sql: str, **kwargs) -> LineageResult:
        """
        Get lineage result for SQL query.
        
        Args:
            sql: SQL query string
            **kwargs: Additional options
            
        Returns:
            LineageResult object
        """
        return self.analyze(sql, **kwargs)
    
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
        Get the table lineage chain for a SQL query using ChainBuilder.
        """
        result = self.analyze(sql, **kwargs)
        return self.chain_builder.get_table_lineage_chain(result, sql, chain_type, depth, **kwargs)
    
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
    
    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive lineage chain with unlimited depth support and metadata integration.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing comprehensive lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 0:
            raise ValueError("depth must be 0 or greater (0 = unlimited)")
        
        # Handle CTE queries differently using CTEAnalyzer
        if "WITH " in sql.upper():
            return self.cte_analyzer.build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        
        # Use ChainBuilder for other queries
        result = self.analyze(sql, **kwargs)
        return self.chain_builder.get_lineage_chain(result, sql, chain_type, depth, target_entity, **kwargs)
    
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
        return self.chain_builder.get_lineage_chain_json(self.analyze(sql, **kwargs), sql, chain_type, depth, target_entity, **kwargs)
    
    # Delegate parser methods to parsers for backward compatibility
    def parse_select(self, sql: str) -> Dict[str, Any]:
        """Parse SELECT statement using SelectParser."""
        return self.select_parser.parse(sql)
    
    def parse_transformations(self, sql: str) -> Dict[str, Any]:
        """Parse transformations using TransformationParser."""
        return self.transformation_parser.parse(sql)
    
    def parse_cte(self, sql: str) -> Dict[str, Any]:
        """Parse CTE using CTEParser."""
        return self.cte_parser.parse(sql)
    
    def parse_ctas(self, sql: str) -> Dict[str, Any]:
        """Parse CTAS using CTASParser."""
        return self.ctas_parser.parse(sql)
    
    def parse_insert(self, sql: str) -> Dict[str, Any]:
        """Parse INSERT using InsertParser."""
        return self.insert_parser.parse(sql)
    
    def parse_update(self, sql: str) -> Dict[str, Any]:
        """Parse UPDATE using UpdateParser."""
        return self.update_parser.parse(sql)
    
    # Add missing helper methods from backup file
    def _infer_query_result_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """Delegate to SelectAnalyzer for QUERY_RESULT column inference."""
        return self.select_analyzer.infer_query_result_columns(sql, column_lineage_data)
    
    def _filter_query_result_columns_by_parent(self, all_columns: List[Dict], parent_entity: str, column_lineage_data: Dict) -> List[Dict]:
        """Delegate to SelectAnalyzer for column filtering."""
        return self.select_analyzer.filter_query_result_columns_by_parent(all_columns, parent_entity, column_lineage_data)
    
    def _expand_star_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """Delegate to SelectAnalyzer for star column expansion."""
        return self.select_analyzer._expand_star_columns(sql, column_lineage_data)
    
    def _is_column_from_table(self, column_name: str, table_name: str, context_info: dict = None) -> bool:
        """Check if column belongs to table - inherited from BaseAnalyzer."""
        return super()._is_column_from_table(column_name, table_name, context_info)
    
    def _is_aggregate_function_for_table(self, column_expr: str, table_name: str) -> bool:
        """Check if expression is aggregate function for table - inherited from BaseAnalyzer."""
        return super()._is_aggregate_function_for_table(column_expr, table_name)