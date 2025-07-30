"""Main SQL lineage analyzer."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression
import json

from .models import LineageResult, TableMetadata
from .extractor import LineageExtractor
from .parsers import SelectParser, TransformationParser, CTEParser, CTASParser, InsertParser, UpdateParser
from ..metadata.registry import MetadataRegistry
from ..utils.validation import validate_sql_input


class SQLLineageAnalyzer:
    """Main SQL lineage analyzer class."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the SQL lineage analyzer.
        
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
        
        return {
            'cte_structure': cte_data,
            'cte_lineage': cte_lineage,
            'execution_order': cte_lineage.get('execution_order', []),
            'final_result': cte_lineage.get('final_result', {}),
            'cte_dependencies': cte_data.get('cte_dependencies', {})
        }
    
    def _analyze_ctas(self, sql: str) -> Dict[str, Any]:
        """Analyze CREATE TABLE AS SELECT statement."""
        ctas_data = self.ctas_parser.parse(sql)
        ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
        
        return {
            'ctas_structure': ctas_data,
            'ctas_lineage': ctas_lineage,
            'target_table': ctas_data.get('target_table', {}),
            'source_analysis': ctas_lineage.get('source_analysis', {}),
            'transformations': ctas_lineage.get('transformations', [])
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
                from .models import TableLineage, ColumnLineage
                
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
        Analyze SQL from a file.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Additional options
            
        Returns:
            LineageResult containing analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql = f.read()
            return self.analyze(sql, **kwargs)
        except Exception as e:
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"Failed to read file {file_path}: {str(e)}"]
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
        return [self.analyze(query, **kwargs) for query in queries]
    
    def _parse_sql(self, sql: str) -> Expression:
        """Parse SQL string into SQLGlot expression."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL query: {str(e)}")
    
    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """Collect metadata for tables involved in lineage."""
        metadata = {}
        
        # Collect all table names from upstream and downstream
        all_tables = set()
        for target, sources in table_lineage.upstream.items():
            all_tables.add(target)
            all_tables.update(sources)
        
        # Get metadata for each table
        for table_name in all_tables:
            table_metadata = self.metadata_registry.get_table_metadata(table_name)
            if table_metadata:
                metadata[table_name] = table_metadata
        
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
        Get the LineageResult object for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            LineageResult object containing table and column lineage information
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
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 0:
            raise ValueError("depth must be 0 or greater (0 means unlimited depth)")
        
        result = self.analyze(sql, **kwargs)
        
        # Get both table and column lineage data
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Helper function to get table name from column reference
        def extract_table_from_column(column_ref: str) -> str:
            if '.' in column_ref:
                parts = column_ref.split('.')
                return '.'.join(parts[:-1])  # Everything except the last part (column name)
            return "unknown_table"
        
        # Helper function to get column name from column reference
        def extract_column_from_ref(column_ref: str) -> str:
            if '.' in column_ref:
                return column_ref.split('.')[-1]
            return column_ref
        
        # Build comprehensive lineage chain
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we have a circular dependency
            if entity_name in visited_in_path:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "metadata": {}
                }
            
            # Stop if we've exceeded max depth (only when depth > 0, meaning limited depth)
            if depth > 0 and current_depth > depth:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "metadata": {}
                }
            
            # Add current entity to the path to prevent cycles
            visited_in_path = visited_in_path | {entity_name}
            
            dependencies = []
            transformations = []
            metadata = {}
            
            if entity_type == "table":
                # Handle table-level lineage
                if entity_name in table_lineage_data:
                    for dependent_table in table_lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dependent_table, "table", current_depth + 1, visited_in_path)
                        dependencies.append(dep_chain)
                
                # Get table transformations (optimized - only include non-empty data)
                if entity_name in result.table_lineage.transformations:
                    for transformation in result.table_lineage.transformations[entity_name]:
                        trans_data = {
                            "type": "table_transformation",
                            "source_table": transformation.source_table,
                            "target_table": transformation.target_table
                        }
                        
                        # Only add non-null/non-empty values
                        if transformation.join_type:
                            trans_data["join_type"] = transformation.join_type.value
                        
                        if transformation.join_conditions:
                            trans_data["join_conditions"] = [
                                {
                                    "left_column": jc.left_column,
                                    "operator": jc.operator.value,
                                    "right_column": jc.right_column
                                }
                                for jc in transformation.join_conditions
                            ]
                        
                        if transformation.filter_conditions:
                            trans_data["filter_conditions"] = [
                                {
                                    "column": fc.column,
                                    "operator": fc.operator.value,
                                    "value": fc.value
                                }
                                for fc in transformation.filter_conditions
                            ]
                        
                        if transformation.group_by_columns:
                            trans_data["group_by_columns"] = transformation.group_by_columns
                        
                        if transformation.having_conditions:
                            trans_data["having_conditions"] = [
                                {
                                    "column": hc.column,
                                    "operator": hc.operator.value,
                                    "value": hc.value
                                }
                                for hc in transformation.having_conditions
                            ]
                        
                        if transformation.order_by_columns:
                            trans_data["order_by_columns"] = transformation.order_by_columns
                        
                        transformations.append(trans_data)
                
                # Get essential table metadata (excluding detailed column info)
                if entity_name in result.metadata:
                    table_meta = result.metadata[entity_name]
                    metadata = {
                        "table_type": table_meta.table_type.value
                    }
                    
                    # Only include non-null values to keep output clean
                    if table_meta.schema:
                        metadata["schema"] = table_meta.schema
                    if table_meta.description:
                        metadata["description"] = table_meta.description
                
                # Add simplified column-level information for this table
                table_columns = []
                for column_ref in column_lineage_data.keys():
                    if extract_table_from_column(column_ref) == entity_name:
                        column_name = extract_column_from_ref(column_ref)
                        upstream_columns = list(column_lineage_data.get(column_ref, set()))
                        
                        # Only include columns that have upstream relationships
                        if upstream_columns:
                            column_info = {
                                "name": column_name,
                                "upstream": upstream_columns
                            }
                            
                            # Add transformation type if present (simplified)
                            if column_ref in result.column_lineage.transformations:
                                column_transformations = result.column_lineage.transformations[column_ref]
                                if column_transformations:
                                    trans = column_transformations[0]  # Take first transformation
                                    if trans.aggregate_function:
                                        column_info["type"] = "AGGREGATE"
                                    elif trans.window_function:
                                        column_info["type"] = "WINDOW"
                                    elif trans.case_expression:
                                        column_info["type"] = "CASE"
                                    elif trans.expression and trans.expression != column_name:
                                        column_info["type"] = "COMPUTED"
                                    else:
                                        column_info["type"] = "DIRECT"
                                else:
                                    column_info["type"] = "DIRECT"
                            else:
                                column_info["type"] = "DIRECT"
                            
                            table_columns.append(column_info)
                
                # Special handling for QUERY_RESULT - infer result columns from SQL parsing
                if entity_name == "QUERY_RESULT" and not table_columns:
                    # For QUERY_RESULT, we should infer columns from the SELECT statement
                    table_columns = self._infer_query_result_columns(sql, column_lineage_data)
                
                # Only add table_columns if not empty
                if table_columns:
                    metadata["table_columns"] = table_columns
            
            elif entity_type == "column":
                # Handle column-level lineage (simplified)
                if entity_name in column_lineage_data:
                    for dependent_column in column_lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dependent_column, "column", current_depth + 1, visited_in_path)
                        dependencies.append(dep_chain)
                
                # Simplified column transformations
                if entity_name in result.column_lineage.transformations:
                    transformations_list = result.column_lineage.transformations[entity_name]
                    if transformations_list:
                        trans = transformations_list[0]  # Take first transformation
                        trans_data = {"type": "column_transformation"}
                        
                        if trans.aggregate_function:
                            trans_data["function_type"] = trans.aggregate_function.function_type.value
                        elif trans.window_function:
                            trans_data["function_type"] = "WINDOW"
                        elif trans.case_expression:
                            trans_data["function_type"] = "CASE"
                        elif trans.expression:
                            trans_data["expression"] = trans.expression
                        
                        transformations.append(trans_data)
                
                # Minimal column metadata
                parent_table = extract_table_from_column(entity_name)
                metadata = {"parent_table": parent_table}
            
            # Clean up empty arrays to reduce clutter
            result_dict = {
                "entity": entity_name,
                "entity_type": entity_type,
                "depth": current_depth - 1,
                "dependencies": dependencies,
                "metadata": metadata
            }
            
            # Only add transformations if not empty
            if transformations:
                result_dict["transformations"] = transformations
                
            return result_dict
        
        # Build chains starting from the target entity or all entities
        chains = {}
        
        if target_entity:
            # Focus on specific entity
            if target_entity in table_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "table", 1)
            elif target_entity in column_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "column", 1)
            else:
                # Try to find it as a partial match
                found = False
                for table_name in table_lineage_data.keys():
                    if target_entity in table_name:
                        chains[table_name] = build_comprehensive_chain(table_name, "table", 1)
                        found = True
                        break
                
                if not found:
                    for column_ref in column_lineage_data.keys():
                        if target_entity in column_ref:
                            chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1)
                            break
        else:
            # Build chains for all tables
            for table_name in table_lineage_data.keys():
                chains[table_name] = build_comprehensive_chain(table_name, "table", 1)
            
            # Build chains for all columns (only if no table chains exist to avoid redundancy)
            if not chains:
                for column_ref in column_lineage_data.keys():
                    chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1)
        
        # Calculate actual max depth achieved
        actual_max_depth = 0
        for chain_data in chains.values():
            if "depth" in chain_data:
                actual_max_depth = max(actual_max_depth, chain_data["depth"])
            # Also check nested dependencies for max depth
            def get_max_depth_from_chain(chain_obj, current_max=0):
                max_depth = current_max
                if isinstance(chain_obj, dict):
                    if "depth" in chain_obj:
                        max_depth = max(max_depth, chain_obj["depth"])
                    if "dependencies" in chain_obj:
                        for dep in chain_obj["dependencies"]:
                            max_depth = max(max_depth, get_max_depth_from_chain(dep, max_depth))
                return max_depth
            
            actual_max_depth = max(actual_max_depth, get_max_depth_from_chain(chain_data))
        
        # Calculate actually used columns from transformations and lineage
        # We need to normalize column references to avoid counting duplicates
        # (e.g., 'name' and 'QUERY_RESULT.name' should be treated as the same logical column)
        used_columns = set()
        
        def normalize_column_name(column_ref: str) -> str:
            """Normalize column reference to just the column name, ignoring table prefixes for counting."""
            if column_ref and '.' in column_ref:
                # Skip QUERY_RESULT columns as they are output columns, not source columns
                if column_ref.startswith('QUERY_RESULT.'):
                    return None
                return column_ref.split('.')[-1]  # Get just the column name
            return column_ref
        
        # Add columns from upstream lineage data (these are the actual source columns)
        for column_ref, upstream_columns in column_lineage_data.items():
            # Add upstream columns (source columns)
            for upstream_col in upstream_columns:
                normalized = normalize_column_name(upstream_col)
                if normalized:
                    used_columns.add(normalized)
        
        # Add columns from column transformations (focus on source columns)
        for transformation_list in result.column_lineage.transformations.values():
            for transformation in transformation_list:
                if transformation.source_column:
                    normalized = normalize_column_name(transformation.source_column)
                    if normalized:
                        used_columns.add(normalized)
        
        # Add columns from table transformations (join conditions, filters, etc.)
        for transformation_list in result.table_lineage.transformations.values():
            for transformation in transformation_list:
                # Join conditions
                for join_condition in transformation.join_conditions:
                    if join_condition.left_column:
                        normalized = normalize_column_name(join_condition.left_column)
                        if normalized:
                            used_columns.add(normalized)
                    if join_condition.right_column:
                        normalized = normalize_column_name(join_condition.right_column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Filter conditions
                for filter_condition in transformation.filter_conditions:
                    if filter_condition.column:
                        normalized = normalize_column_name(filter_condition.column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Group by columns
                for group_col in transformation.group_by_columns:
                    normalized = normalize_column_name(group_col)
                    if normalized:
                        used_columns.add(normalized)
                
                # Having conditions
                for having_condition in transformation.having_conditions:
                    if having_condition.column:
                        normalized = normalize_column_name(having_condition.column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Order by columns
                for order_col in transformation.order_by_columns:
                    normalized = normalize_column_name(order_col)
                    if normalized:
                        used_columns.add(normalized)

        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth if depth > 0 else "unlimited",
            "actual_max_depth": actual_max_depth,
            "target_entity": target_entity,
            "chains": chains,
            "summary": {
                "total_tables": len(set(table_lineage_data.keys()) | set().union(*table_lineage_data.values()) if table_lineage_data else set()),
                "total_columns": len(used_columns),
                "has_transformations": bool(result.table_lineage.transformations or result.column_lineage.transformations),
                "has_metadata": bool(result.metadata),
                "chain_count": len(chains)
            },
            "errors": result.errors,
            "warnings": result.warnings
        }
    
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
        import json
        chain_data = self.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def _infer_query_result_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Infer QUERY_RESULT columns from SQL query when column lineage doesn't provide them.
        
        Args:
            sql: The SQL query string
            column_lineage_data: Column lineage mapping
            
        Returns:
            List of column information dictionaries for QUERY_RESULT
        """
        import re
        
        result_columns = []
        
        # Try to extract SELECT columns from SQL
        # This is a simple approach - for more complex cases, proper SQL parsing would be needed
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            
            # Handle simple cases like "SELECT name, email FROM users"
            if select_clause != '*':
                # Split by comma and clean up
                columns = [col.strip() for col in select_clause.split(',')]
                
                for col in columns:
                    # Remove table aliases (u.name -> name)
                    if '.' in col:
                        col_name = col.split('.')[-1]
                    else:
                        col_name = col
                    
                    # Remove AS aliases (name AS user_name -> name)
                    if ' AS ' in col_name.upper():
                        col_name = col_name.split(' AS ')[0].strip()
                    elif ' as ' in col_name:
                        col_name = col_name.split(' as ')[0].strip()
                    
                    # Clean up any remaining whitespace and quotes
                    col_name = col_name.strip().strip('"').strip("'")
                    
                    if col_name:
                        # Try to find upstream columns from column lineage data
                        query_result_ref = f"QUERY_RESULT.{col_name}"
                        upstream_columns = []
                        
                        # Look for this column in the lineage data
                        for column_ref, upstream_cols in column_lineage_data.items():
                            if (column_ref == query_result_ref or 
                                column_ref.endswith(f".{col_name}") or
                                column_ref == col_name):
                                upstream_columns = list(upstream_cols)
                                break
                        
                        # If no upstream found, try to infer from table references in SQL
                        if not upstream_columns:
                            # Simple heuristic: if SQL has "FROM users" and column is "name", 
                            # assume it comes from users.name
                            from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                            if from_match:
                                table_name = from_match.group(1)
                                upstream_columns = [f"{table_name}.{col_name}"]
                        
                        # Use simplified format matching the optimized structure
                        result_columns.append({
                            "name": col_name,
                            "upstream": upstream_columns,
                            "type": "DIRECT"
                        })
        
        return result_columns
    
    # Modular parser convenience methods
    def parse_select(self, sql: str) -> Dict[str, Any]:
        """Parse SELECT statement using modular parser."""
        return self.select_parser.parse(sql)
    
    def parse_transformations(self, sql: str) -> Dict[str, Any]:
        """Parse transformations using modular parser.""" 
        return self.transformation_parser.parse(sql)
    
    def parse_cte(self, sql: str) -> Dict[str, Any]:
        """Parse CTE using modular parser."""
        return self.cte_parser.parse(sql)
    
    def parse_ctas(self, sql: str) -> Dict[str, Any]:
        """Parse CTAS using modular parser.""" 
        return self.ctas_parser.parse(sql)
    
    def parse_insert(self, sql: str) -> Dict[str, Any]:
        """Parse INSERT using modular parser."""
        return self.insert_parser.parse(sql)
    
    def parse_update(self, sql: str) -> Dict[str, Any]:
        """Parse UPDATE using modular parser."""
        return self.update_parser.parse(sql)
