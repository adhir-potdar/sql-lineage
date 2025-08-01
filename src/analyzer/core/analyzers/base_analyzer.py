"""Base analyzer functionality."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression
import json

from ..models import LineageResult, TableMetadata
from ..extractor import LineageExtractor
from ..parsers import SelectParser, TransformationParser, CTEParser, CTASParser, InsertParser, UpdateParser
from ...metadata.registry import MetadataRegistry
from ...utils.validation import validate_sql_input


class BaseAnalyzer:
    """Base analyzer class with core functionality."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the base analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
        """
        self.dialect = dialect
        self.metadata_registry = MetadataRegistry()
        self.extractor = LineageExtractor()
        
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
    
    def analyze_comprehensive(self, sql: str) -> Dict[str, Any]:
        """
        Comprehensive analysis using modular parsers.
        
        Args:
            sql: SQL query string to analyze
            
        Returns:
            Comprehensive analysis result
        """
        try:
            # Determine SQL type and route to appropriate parser
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
        """Analyze simple SELECT statement."""
        select_data = self.select_parser.parse(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'query_structure': select_data,
            'transformations': transformation_data,
            'lineage': self._build_select_lineage(select_data, transformation_data),
            'result_columns': self._extract_result_columns(select_data),
            'source_tables': self._extract_source_tables(select_data)
        }
    
    def _analyze_cte(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE statement."""
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
        """Analyze CREATE TABLE AS SELECT statement."""
        ctas_data = self.ctas_parser.parse(sql)
        ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'ctas_structure': ctas_data,
            'ctas_lineage': ctas_lineage,
            'transformations': transformation_data,
            'target_table': ctas_data.get('target_table', {}),
            'source_analysis': ctas_lineage.get('source_analysis', {}),
            'ctas_transformations': ctas_lineage.get('transformations', [])
        }
    
    def _analyze_insert(self, sql: str) -> Dict[str, Any]:
        """Analyze INSERT statement."""
        insert_data = self.insert_parser.parse(sql)
        insert_lineage = self.insert_parser.get_insert_lineage(sql)
        
        return {
            'insert_structure': insert_data,
            'insert_lineage': insert_lineage,
            'target_table': insert_data.get('target_table', {}),
            'source_analysis': insert_lineage.get('source_analysis', {}),
            'data_flow': insert_lineage.get('data_flow', [])
        }
    
    def _analyze_update(self, sql: str) -> Dict[str, Any]:
        """Analyze UPDATE statement."""
        update_data = self.update_parser.parse(sql)
        update_lineage = self.update_parser.get_update_lineage(sql)
        
        return {
            'update_structure': update_data,
            'update_lineage': update_lineage,
            'target_table': update_data.get('target_table', {}),
            'source_analysis': update_lineage.get('source_analysis', {}),
            'column_updates': update_lineage.get('column_updates', {}),
            'data_flow': update_lineage.get('data_flow', [])
        }
    
    def _build_select_lineage(self, select_data: Dict[str, Any], transformation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build lineage chain for SELECT statement."""
        lineage = {
            'type': 'SELECT_LINEAGE',
            'flow': []
        }
        
        # Source tables
        source_tables = select_data.get('from_tables', [])
        for table in source_tables:
            lineage['flow'].append({
                'type': 'SOURCE',
                'entity': table.get('table_name'),
                'alias': table.get('alias'),
                'columns_used': self._get_columns_used_from_table(table, select_data)
            })
        
        # Transformations
        transformations = []
        
        # Add filters
        filters = transformation_data.get('filters', {})
        if filters.get('conditions'):
            transformations.append({
                'type': 'FILTER',
                'conditions': filters['conditions']
            })
        
        # Add joins
        joins = transformation_data.get('joins', [])
        for join in joins:
            transformations.append({
                'type': 'JOIN',
                'join_type': join.get('join_type'),
                'table': join.get('table_name'),
                'conditions': join.get('conditions', [])
            })
        
        # Add aggregations
        aggregations = transformation_data.get('aggregations', {})
        if aggregations.get('group_by_columns'):
            transformations.append({
                'type': 'GROUP_BY',
                'columns': aggregations['group_by_columns']
            })
        
        # Add transformations to flow
        for transform in transformations:
            lineage['flow'].append(transform)
        
        # Final result
        lineage['flow'].append({
            'type': 'RESULT',
            'entity': 'QUERY_RESULT',
            'columns': self._extract_result_columns(select_data)
        })
        
        return lineage
    
    def _get_columns_used_from_table(self, table: Dict[str, Any], select_data: Dict[str, Any]) -> List[str]:
        """Get columns used from a specific table."""
        table_name = table.get('table_name')
        table_alias = table.get('alias')
        used_columns = []
        
        # Check select columns
        select_columns = select_data.get('select_columns', [])
        for col in select_columns:
            source_table = col.get('source_table')
            if source_table == table_name or source_table == table_alias:
                used_columns.append(col.get('column_name'))
        
        # Check WHERE conditions
        where_conditions = select_data.get('where_conditions', [])
        for condition in where_conditions:
            column = condition.get('column', '')
            if '.' in column:
                col_table, col_name = column.split('.', 1)
                if col_table == table_name or col_table == table_alias:
                    used_columns.append(col_name)
        
        # Check JOIN conditions
        joins = select_data.get('joins', [])
        for join in joins:
            for condition in join.get('conditions', []):
                left_col = condition.get('left_column', '')
                right_col = condition.get('right_column', '')
                
                # Check both sides of join condition
                for col in [left_col, right_col]:
                    if '.' in col:
                        col_table, col_name = col.split('.', 1)
                        if col_table == table_name or col_table == table_alias:
                            used_columns.append(col_name)
        
        return list(set(used_columns))  # Remove duplicates
    
    def _extract_result_columns(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract result columns with their metadata."""
        result_columns = []
        
        select_columns = select_data.get('select_columns', [])
        for col in select_columns:
            result_columns.append({
                'name': col.get('column_name'),
                'alias': col.get('alias'),
                'source_table': col.get('source_table'),
                'expression': col.get('raw_expression'),
                'is_computed': col.get('is_aggregate') or col.get('is_window_function')
            })
        
        return result_columns
    
    def _extract_source_tables(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source tables with their metadata."""
        source_tables = []
        
        # FROM tables
        from_tables = select_data.get('from_tables', [])
        for table in from_tables:
            source_tables.append({
                'name': table.get('table_name'),
                'alias': table.get('alias'),
                'type': 'FROM'
            })
        
        # JOIN tables
        joins = select_data.get('joins', [])
        for join in joins:
            source_tables.append({
                'name': join.get('table_name'),
                'alias': join.get('alias'),
                'type': 'JOIN',
                'join_type': join.get('join_type')
            })
        
        return source_tables
    
    def analyze(self, sql: str, **kwargs) -> LineageResult:
        """
        Analyze SQL query for lineage information.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options (for future extensibility)
            
        Returns:
            LineageResult containing table and column lineage information
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
            # First try the legacy extractor approach for backward compatibility
            try:
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
            except Exception:
                # If legacy approach fails, fall back to modular parsers
                comprehensive_analysis = self.analyze_comprehensive(sql)
                
                if not comprehensive_analysis.get('success', False):
                    raise Exception(comprehensive_analysis.get('error', 'Analysis failed'))
                
                # Convert modular analysis to LineageResult format
                # For now, create basic lineage structures - this could be enhanced further
                from ..models import TableLineage, ColumnLineage
                
                table_lineage = TableLineage()
                column_lineage = ColumnLineage() 
                metadata = {}
                
                return LineageResult(
                    sql=sql,
                    dialect=self.dialect,
                    table_lineage=table_lineage,
                    column_lineage=column_lineage,
                    metadata=metadata
                )
            
        except Exception as e:
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"Failed to analyze SQL: {str(e)}"]
            )
    
    def analyze_file(self, file_path: str, **kwargs) -> LineageResult:
        """
        Analyze SQL file and return lineage result.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Additional options
            
        Returns:
            LineageResult containing table and column lineage
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                sql = file.read()
            return self.analyze(sql, **kwargs)
        except FileNotFoundError:
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=None,
                column_lineage=None,
                errors=[f"File not found: {file_path}"]
            )
        except Exception as e:
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=None,
                column_lineage=None,
                errors=[f"File reading error: {str(e)}"]
            )
    
    def analyze_multiple(self, queries: list[str], **kwargs) -> list[LineageResult]:
        """
        Analyze multiple SQL queries.
        
        Args:
            queries: List of SQL query strings
            **kwargs: Additional options
            
        Returns:
            List of LineageResult objects
        """
        results = []
        for sql in queries:
            results.append(self.analyze(sql, **kwargs))
        return results
    
    def _parse_sql(self, sql: str) -> Expression:
        """Parse SQL string into SQLGlot expression."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL: {str(e)}")
    
    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """Collect metadata for tables involved in lineage."""
        metadata = {}
        
        # Collect all table names from upstream and downstream
        table_names = set()
        if hasattr(table_lineage, 'upstream'):
            for tables in table_lineage.upstream.values():
                table_names.update(tables)
        if hasattr(table_lineage, 'downstream'):
            for tables in table_lineage.downstream.values():
                table_names.update(tables)
        
        # Get metadata for each table
        for table_name in table_names:
            try:
                table_metadata = self.metadata_registry.get_table_metadata(table_name)
                if table_metadata:
                    metadata[table_name] = table_metadata
            except Exception:
                # Skip tables that don't have metadata
                pass
        
        return metadata
    
    def set_metadata_registry(self, metadata_registry: MetadataRegistry) -> None:
        """
        Set the metadata registry for the analyzer.
        
        Args:
            metadata_registry: MetadataRegistry instance to use
        """
        self.metadata_registry = metadata_registry
        self.extractor = LineageExtractor()
    
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
        Get lineage as JSON string.
        
        Args:
            sql: SQL query string
            **kwargs: Additional options
            
        Returns:
            JSON string representation of lineage
        """
        result = self.analyze(sql, **kwargs)
        return json.dumps(result.to_dict(), indent=2)