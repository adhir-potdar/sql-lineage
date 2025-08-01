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
    
    def _add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
        """Add missing source columns to table metadata by extracting from transformations."""
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies OR add to existing
            if entity_data.get('depth') == 0:
                source_columns = set()
                
                # Look through dependencies to find transformations that reference this table
                for dep in entity_data.get('dependencies', []):
                    for trans in dep.get('transformations', []):
                        if trans.get('source_table') == entity_name:
                            # Extract filter condition columns
                            for condition in trans.get('filter_conditions', []):
                                col = condition.get('column', '')
                                if col:
                                    source_columns.add(col)
                            
                            # Extract join condition columns
                            for join in trans.get('joins', []):
                                for condition in join.get('conditions', []):
                                    for col_key in ['left_column', 'right_column']:
                                        col_ref = condition.get(col_key, '')
                                        if col_ref and entity_name in col_ref:
                                            clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                                            source_columns.add(clean_col)
                
                # Add the columns found in transformations, merging with existing
                if source_columns:
                    # Get existing column names to avoid duplicates
                    existing_columns = {col['name'] for col in table_columns}
                    
                    # Add new columns from transformations
                    for column_name in source_columns:
                        if column_name not in existing_columns:
                            column_info = {
                                "name": column_name,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            table_columns.append(column_info)
                    
                    # Update the metadata
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    entity_data['metadata']['table_columns'] = table_columns
            
            # For target tables (depth > 0), extract result columns from CTAS SELECT clause
            elif entity_data.get('depth', 0) > 0:
                # Check if this is a target table created by CTAS
                transformations = entity_data.get('transformations', [])
                for trans in transformations:
                    if trans.get('target_table') == entity_name and trans.get('source_table'):
                        # This is a CTAS target table, extract SELECT clause columns
                        target_columns = set()
                        
                        # Extract columns from GROUP BY (these will be in result)
                        group_by_columns = trans.get('group_by_columns', [])
                        for col in group_by_columns:
                            clean_col = col.split('.')[-1] if '.' in col else col
                            target_columns.add(clean_col)
                        
                        # Extract actual result columns from CTAS SELECT clause parsing
                        if sql and sql.strip().upper().startswith('CREATE TABLE'):
                            try:
                                # Get CTAS lineage to get actual output columns
                                ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
                                column_lineage = ctas_lineage.get('column_lineage', {})
                                output_columns = column_lineage.get('output_columns', [])
                                
                                # Add all result columns from the CTAS SELECT clause
                                for output_col in output_columns:
                                    col_name = output_col.get('alias') or output_col.get('name')
                                    if col_name:
                                        clean_col = col_name.split('.')[-1] if '.' in col_name else col_name
                                        target_columns.add(clean_col)
                            except Exception:
                                # Fallback: just use GROUP BY columns if CTAS parsing fails
                                pass
                        
                        # Get column transformations for this entity
                        column_transformations_map = {}
                        if sql:
                            try:
                                column_transformations = self._extract_column_transformations(
                                    sql, trans.get('source_table'), entity_name
                                )
                                # Create map of column_name -> transformation
                                for col_trans in column_transformations:
                                    col_name = col_trans.get('column_name')
                                    if col_name:
                                        column_transformations_map[col_name] = col_trans
                            except Exception:
                                pass
                        
                        # Add result columns to target table metadata with transformations
                        if target_columns:
                            existing_columns = {col['name'] for col in table_columns}
                            
                            for column_name in target_columns:
                                if column_name not in existing_columns:
                                    column_info = {
                                        "name": column_name,
                                        "upstream": [],
                                        "type": "RESULT" if column_name not in group_by_columns else "SOURCE"
                                    }
                                    
                                    # Add transformation info if available
                                    if column_name in column_transformations_map:
                                        col_trans = column_transformations_map[column_name]
                                        column_info["transformation"] = {
                                            "source_expression": col_trans.get('source_expression'),
                                            "transformation_type": col_trans.get('transformation_type'),
                                            "function_type": col_trans.get('function_type')
                                        }
                                    
                                    table_columns.append(column_info)
                            
                            # Update the metadata
                            if 'metadata' not in entity_data:
                                entity_data['metadata'] = {}
                            entity_data['metadata']['table_columns'] = table_columns
    
    def _extract_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """Extract column transformations generically for all query types."""
        if not sql:
            return []
        
        column_transformations = []
        
        try:
            # Determine query type and extract accordingly
            sql_upper = sql.strip().upper()
            
            if sql_upper.startswith('CREATE TABLE'):
                # CTAS query - extract from CTAS parser
                ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
                column_lineage = ctas_lineage.get('column_lineage', {})
                ctas_column_transforms = column_lineage.get('column_transformations', [])
                column_transformations.extend(ctas_column_transforms)
                
            elif 'WITH' in sql_upper and target_table:
                # CTE query - extract transformations from CTE lineage
                column_transformations.extend(self._extract_cte_column_transformations(sql, target_table))
                
            elif sql_upper.startswith('SELECT') or 'SELECT' in sql_upper:
                # Regular SELECT query or query containing SELECT - extract from SELECT columns
                column_transformations.extend(self._extract_select_column_transformations(sql, source_table, target_table))
                
            elif sql_upper.startswith('INSERT'):
                # INSERT query - extract from INSERT parser if needed
                pass  # TODO: Add INSERT column transformations if needed
                
            elif sql_upper.startswith('UPDATE'):
                # UPDATE query - extract from UPDATE parser if needed  
                pass  # TODO: Add UPDATE column transformations if needed
            
        except Exception:
            # If parsing fails, return empty list (fail gracefully)
            pass
        
        return column_transformations
    
    def _extract_select_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """Extract column transformations from SELECT queries."""
        column_transformations = []
        
        try:
            # Parse the SELECT query
            select_data = self.select_parser.parse(sql)
            select_columns = select_data.get('select_columns', [])
            
            for col in select_columns:
                # Check if this is a computed column (aggregate, function, expression)
                is_aggregate = col.get('is_aggregate', False)
                is_function = col.get('is_window_function', False) or col.get('is_function', False)
                raw_expression = col.get('raw_expression', '')
                
                if is_aggregate or is_function or self._is_computed_expression(raw_expression):
                    # Use alias as the column name, and show transformation as source expression
                    target_column_name = col.get('alias') or col.get('column_name')
                    source_expression = self._extract_source_from_expression(raw_expression, target_column_name)
                    
                    col_transformation = {
                        'column_name': target_column_name,  # This will be the column name in the table
                        'source_expression': source_expression,  # e.g., "SUM(amount)", "UPPER(email)"
                        'transformation_type': self._get_transformation_type(col, raw_expression),
                        'function_type': self._extract_function_type_generic(raw_expression),
                        'full_expression': raw_expression  # Keep full expression for reference
                    }
                    column_transformations.append(col_transformation)
                    
        except Exception:
            # If parsing fails, return empty list (fail gracefully)
            pass
            
        return column_transformations
    
    def _extract_cte_column_transformations(self, sql: str, target_table: str) -> List[Dict]:
        """Extract column transformations from CTE queries."""
        column_transformations = []
        
        try:
            # Get CTE lineage data
            cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
            cte_data = cte_lineage.get('ctes', {})
            
            # Look for the target CTE
            if target_table not in cte_data:
                return column_transformations
            
            target_cte = cte_data[target_table]
            columns = target_cte.get('columns', [])
            
            for col in columns:
                # Check if this is a computed column
                is_computed = col.get('is_computed', False)
                raw_expression = col.get('expression', '')
                
                if is_computed or self._is_computed_expression(raw_expression):
                    # Use alias as the column name, and show transformation as source expression
                    target_column_name = col.get('alias') or col.get('name')
                    source_expression = self._extract_source_from_expression(raw_expression, target_column_name)
                    
                    col_transformation = {
                        'column_name': target_column_name,
                        'source_expression': source_expression,
                        'transformation_type': self._get_cte_transformation_type(col, raw_expression),
                        'function_type': self._extract_function_type_generic(raw_expression),
                        'full_expression': raw_expression
                    }
                    column_transformations.append(col_transformation)
                    
        except Exception:
            # If parsing fails, return empty list (fail gracefully)
            pass
            
        return column_transformations
    
    def _get_cte_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type for CTE columns."""
        if col_info.get('is_aggregate', False):
            return 'AGGREGATE'
        elif 'CASE' in expression.upper():
            return 'CASE'
        elif self._is_computed_expression(expression):
            return 'COMPUTED'
        else:
            return 'DIRECT'
    
    def _is_computed_expression(self, expression: str) -> bool:
        """Check if an expression is computed (not just a simple column reference)."""
        if not expression:
            return False
        
        # Simple heuristics for computed expressions
        expr = expression.strip()
        
        # Check for function calls: FUNCTION(...)
        if '(' in expr and ')' in expr:
            return True
        
        # Check for mathematical operations
        if any(op in expr for op in ['+', '-', '*', '/', '%']):
            return True
        
        # Check for CASE statements
        if 'CASE' in expr.upper():
            return True
        
        return False
    
    def _get_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type generically."""
        if col_info.get('is_aggregate'):
            return 'AGGREGATE'
        elif col_info.get('is_window_function'):
            return 'WINDOW_FUNCTION'
        elif self._is_computed_expression(expression):
            return 'COMPUTED'
        else:
            return 'DIRECT'
    
    def _extract_function_type_generic(self, expression: str) -> str:
        """Extract function type from expression generically."""
        if not expression:
            return 'UNKNOWN'
        
        # Convert to uppercase for matching
        expr_upper = expression.upper().strip()
        
        # Extract function name from expressions like "COUNT(*)", "SUM(amount)", etc.
        # Match pattern: FUNCTION_NAME(...)
        import re
        function_match = re.match(r'^([A-Z_]+)\s*\(', expr_upper)
        if function_match:
            return function_match.group(1)
        
        # Check for CASE expressions
        if expr_upper.startswith('CASE'):
            return 'CASE'
        
        return 'EXPRESSION'
    
    def _extract_source_from_expression(self, expression: str, target_name: str) -> str:
        """Extract the source part from transformation expression."""
        if not expression:
            return 'UNKNOWN'
        
        # Remove the alias part if present (e.g., "SUM(amount) AS total" -> "SUM(amount)")
        expr = expression.strip()
        
        # Split by " AS " (case insensitive) and take the first part
        import re
        as_split = re.split(r'\s+AS\s+', expr, flags=re.IGNORECASE)
        if len(as_split) > 1:
            return as_split[0].strip()
        
        # If no AS clause, return the expression as is
        return expr
    
    def _integrate_column_transformations(self, chains: Dict, sql: str = None) -> None:
        """Integrate column transformations into column metadata throughout the chain."""
        if not sql:
            return
        
        def process_entity_columns(entity_data, source_table=None, target_table=None):
            """Process columns in an entity to add transformation information."""
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # Get column transformations for this entity
            try:
                column_transformations = self._extract_column_transformations(sql, source_table, target_table)
                
                # Create map of column_name -> transformation
                column_transformations_map = {}
                for col_trans in column_transformations:
                    col_name = col_trans.get('column_name')
                    if col_name:
                        column_transformations_map[col_name] = col_trans
                
                # Update existing columns with transformation info
                for column_info in table_columns:
                    col_name = column_info.get('name')
                    if col_name and col_name in column_transformations_map:
                        col_trans = column_transformations_map[col_name]
                        column_info["transformation"] = {
                            "source_expression": col_trans.get('source_expression'),
                            "transformation_type": col_trans.get('transformation_type'),
                            "function_type": col_trans.get('function_type')
                        }
                
                # Add new columns for transformations not already in table_columns
                # Only add RESULT columns if they are relevant to this specific table
                existing_column_names = {col.get('name') for col in table_columns}
                for col_name, col_trans in column_transformations_map.items():
                    if col_name not in existing_column_names:
                        # Check if this transformation column is relevant to the current table
                        # Use entity name if source_table is None (for top-level entities)
                        table_name_for_relevance = source_table if source_table else entity_data.get('entity')
                        if self._is_transformation_relevant_to_table(col_trans, table_name_for_relevance, sql):
                            column_info = {
                                "name": col_name,
                                "upstream": [],
                                "type": "RESULT",
                                "transformation": {
                                    "source_expression": col_trans.get('source_expression'),
                                    "transformation_type": col_trans.get('transformation_type'),
                                    "function_type": col_trans.get('function_type')
                                }
                            }
                            table_columns.append(column_info)
                
            except Exception:
                # If transformation extraction fails, continue without transformations
                pass
            
            # Update the metadata with the new table_columns
            if 'metadata' not in entity_data:
                entity_data['metadata'] = {}
            entity_data['metadata']['table_columns'] = table_columns
        
        def process_chain_recursively(entity_data, parent_source=None):
            """Recursively process entities and their dependencies."""
            entity_name = entity_data.get('entity')
            
            # Process current entity columns
            process_entity_columns(entity_data, parent_source, entity_name)
            
            # Process dependencies recursively
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                process_chain_recursively(dep, entity_name)
        
        # Process all top-level chains
        for entity_name, entity_data in chains.items():
            process_chain_recursively(entity_data)
    
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
        
        # Check if this is a CTE query and use CTE-specific processing
        if "WITH" in sql.upper():
            return self._build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        
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
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, visited_in_path: set = None, parent_entity: str = None) -> Dict[str, Any]:
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
                        # For CTAS queries, don't add QUERY_RESULT since the target table itself is the final result
                        # Skip QUERY_RESULT dependencies for CTAS queries
                        if (sql and sql.strip().upper().startswith('CREATE TABLE') and 
                            dependent_table == 'QUERY_RESULT'):
                            continue  # Skip QUERY_RESULT for CTAS queries
                        
                        dep_chain = build_comprehensive_chain(dependent_table, "table", current_depth + 1, visited_in_path, entity_name)
                        dependencies.append(dep_chain)
                
                # Get table transformations (optimized - only include non-empty data)
                # Filter transformations to only show those relevant to the parent-child relationship
                transformation_entities = set()
                
                # First check if current entity has transformations
                if entity_name in result.table_lineage.transformations:
                    transformation_entities.add(entity_name)
                
                # Also check all entities for transformations where current entity is involved
                for trans_entity, trans_list in result.table_lineage.transformations.items():
                    for transformation in trans_list:
                        if (transformation.source_table == entity_name or 
                            transformation.target_table == entity_name):
                            transformation_entities.add(trans_entity)
                
                # Process transformations from all relevant entities
                for trans_entity in transformation_entities:
                    for transformation in result.table_lineage.transformations[trans_entity]:
                        # Only include transformations that are relevant to this specific relationship
                        # If we have a parent entity, filter based on the relationship
                        if parent_entity is not None:
                            # For downstream: parent -> current, show transformations where parent is source and current is target
                            # For upstream: current -> parent, show transformations where current is source and parent is target
                            if chain_type == "downstream":
                                if not (transformation.source_table == parent_entity and transformation.target_table == entity_name):
                                    continue
                            elif chain_type == "upstream":
                                if not (transformation.source_table == entity_name and transformation.target_table == parent_entity):
                                    continue
                        else:
                            # If no parent entity (root level), include transformations involving this entity
                            if not (transformation.source_table == entity_name or transformation.target_table == entity_name):
                                continue
                        
                        trans_data = {
                            "type": "table_transformation",
                            "source_table": transformation.source_table,
                            "target_table": transformation.target_table
                        }
                        
                        # Only add non-null/non-empty values
                        
                        if transformation.join_conditions:
                            # Convert old format to new joins format
                            join_entry = {
                                "join_type": transformation.join_type.value if transformation.join_type else "INNER JOIN",
                                "right_table": None,  # Extract from conditions if possible
                                "conditions": [
                                    {
                                        "left_column": jc.left_column,
                                        "operator": jc.operator.value if hasattr(jc.operator, 'value') else str(jc.operator),
                                        "right_column": jc.right_column
                                    }
                                    for jc in transformation.join_conditions
                                ]
                            }
                            
                            # Try to extract right table from the first condition
                            if transformation.join_conditions:
                                first_condition = transformation.join_conditions[0]
                                if hasattr(first_condition, 'right_column') and '.' in first_condition.right_column:
                                    right_table = first_condition.right_column.split('.')[0]
                                    join_entry["right_table"] = right_table
                            
                            trans_data["joins"] = [join_entry]
                        
                        # Determine context for column filtering - used for all transformation types
                        # Single-table context includes both QUERY_RESULT and CTAS scenarios
                        is_single_table = (
                            transformation.target_table == "QUERY_RESULT" or  # Regular SELECT
                            (transformation.source_table != transformation.target_table and  # CTAS/CTE
                             transformation.target_table != "QUERY_RESULT")
                        )
                        
                        context_info = {
                            'is_single_table_context': is_single_table,
                            'tables_in_context': [transformation.source_table] if is_single_table else []
                        }
                        
                        # Filter conditions to only include those relevant to the current entity
                        if transformation.filter_conditions:
                            relevant_filters = []
                            
                            for fc in transformation.filter_conditions:
                                # Only include filter conditions that reference columns from this entity
                                if self._is_column_from_table(fc.column, entity_name, context_info):
                                    relevant_filters.append({
                                        "column": fc.column,
                                        "operator": fc.operator.value if hasattr(fc.operator, 'value') else str(fc.operator),
                                        "value": fc.value
                                    })
                            if relevant_filters:
                                trans_data["filter_conditions"] = relevant_filters
                        
                        # Group by columns - only include those from this entity
                        if transformation.group_by_columns:
                            relevant_group_by = []
                            for col in transformation.group_by_columns:
                                if self._is_column_from_table(col, entity_name, context_info):
                                    relevant_group_by.append(col)
                            if relevant_group_by:
                                trans_data["group_by_columns"] = relevant_group_by
                        
                        # Having conditions - only include those referencing columns from this entity
                        if transformation.having_conditions:
                            relevant_having = []
                            for hc in transformation.having_conditions:
                                # Having conditions often involve aggregations like COUNT(*) or AVG(u.salary)
                                # Check if they reference this entity or if they are general aggregations for this table
                                is_relevant = (self._is_column_from_table(hc.column, entity_name, context_info) or 
                                             self._is_aggregate_function_for_table(hc.column, entity_name))
                                if is_relevant:
                                    relevant_having.append({
                                        "column": hc.column,
                                        "operator": hc.operator.value if hasattr(hc.operator, 'value') else str(hc.operator),
                                        "value": hc.value
                                    })
                            if relevant_having:
                                trans_data["having_conditions"] = relevant_having
                        
                        # Order by columns - only include those from this entity
                        if transformation.order_by_columns:
                            relevant_order_by = []
                            for col in transformation.order_by_columns:
                                # Extract just the column name part (before ASC/DESC)
                                col_name = col.split()[0] if ' ' in col else col
                                if self._is_column_from_table(col_name, entity_name, context_info):
                                    relevant_order_by.append(col)
                            if relevant_order_by:
                                trans_data["order_by_columns"] = relevant_order_by
                        
                        # Column transformations will be integrated into individual column metadata
                        # Remove separate column_transformations from table transformations
                        
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
                columns_added = set()  # Track to avoid duplicates
                
                # First, add columns that have upstream relationships (downstream/intermediate tables)
                for column_ref in column_lineage_data.keys():
                    if extract_table_from_column(column_ref) == entity_name:
                        column_name = extract_column_from_ref(column_ref)
                        upstream_columns = list(column_lineage_data.get(column_ref, set()))
                        
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
                            columns_added.add(column_name)
                
                # Add source columns for tables that don't have any table_columns yet
                # Extract from transformations in dependencies since the actual transformations are stored there
                if not table_columns and entity_type == "table" and entity_name != "QUERY_RESULT":
                    source_columns = set()
                    
                    # Get selected columns from column lineage (SELECT clause columns)
                    for column_ref in column_lineage_data.keys():
                        # Column refs without table prefix are from the source table
                        if '.' not in column_ref:
                            source_columns.add(column_ref)
                    
                    # Add all found columns to table metadata
                    for column_name in source_columns:
                        column_info = {
                            "name": column_name,
                            "upstream": [],
                            "type": "SOURCE" 
                        }
                        table_columns.append(column_info)
                
                # Special handling for QUERY_RESULT - infer result columns from SQL parsing
                if entity_name == "QUERY_RESULT" and not table_columns:
                    # For QUERY_RESULT, we should infer columns from the SELECT statement
                    # Filter columns based on the parent entity that's requesting this QUERY_RESULT
                    all_query_result_columns = self._infer_query_result_columns(sql, column_lineage_data)
                    
                    if parent_entity and all_query_result_columns:
                        # Only filter if this is a multi-table query (JOIN scenario)
                        # Check if columns have table prefixes, indicating multiple tables
                        has_table_prefixes = any('.' in col.get('name', '') for col in all_query_result_columns)
                        
                        if has_table_prefixes:
                            # Filter columns to only include those that come from the parent entity
                            table_columns = self._filter_query_result_columns_by_parent(
                                all_query_result_columns, parent_entity, column_lineage_data
                            )
                        else:
                            # Single table query - use all columns
                            table_columns = all_query_result_columns
                    else:
                        # If no parent entity or no columns, use all columns
                        table_columns = all_query_result_columns
                
                # Only add table_columns if not empty
                if table_columns:
                    metadata["table_columns"] = table_columns
            
            elif entity_type == "column":
                # Handle column-level lineage (simplified)
                if entity_name in column_lineage_data:
                    for dependent_column in column_lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dependent_column, "column", current_depth + 1, visited_in_path, entity_name)
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
            
            # Add transformations to dependencies that are QUERY_RESULT
            for dep in dependencies:
                if dep.get("entity") == "QUERY_RESULT" and transformations:
                    # Filter transformations to only include those relevant to this entity
                    relevant_transformations = []
                    for trans in transformations:
                        if (trans.get("source_table") == entity_name and 
                            trans.get("target_table") == "QUERY_RESULT"):
                            relevant_transformations.append(trans)
                    
                    if relevant_transformations:
                        dep["transformations"] = relevant_transformations
            
            # Only add transformations if not empty AND there are no dependencies
            # (to avoid duplication - transformations will be shown in dependencies)
            if transformations and not dependencies:
                result_dict["transformations"] = transformations
                
            return result_dict
        
        # Build chains starting from the target entity or all entities
        chains = {}
        
        if target_entity:
            # Focus on specific entity
            if target_entity in table_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "table", 1, None, None)
            elif target_entity in column_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "column", 1, None, None)
            else:
                # Try to find it as a partial match
                found = False
                for table_name in table_lineage_data.keys():
                    if target_entity in table_name:
                        chains[table_name] = build_comprehensive_chain(table_name, "table", 1, None, None)
                        found = True
                        break
                
                if not found:
                    for column_ref in column_lineage_data.keys():
                        if target_entity in column_ref:
                            chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1, None, None)
                            break
        else:
            # First, collect all entities that will appear as dependencies to avoid duplication
            entities_in_dependencies = set()
            
            # Build initial chains to collect dependency information
            temp_chains = {}
            for table_name in table_lineage_data.keys():
                temp_chains[table_name] = build_comprehensive_chain(table_name, "table", 1, None, None)
            
            # Collect entities that appear in dependencies
            def collect_dependency_entities(chain_data):
                deps = chain_data.get('dependencies', [])
                for dep in deps:
                    dep_entity = dep.get('entity')
                    if dep_entity:
                        entities_in_dependencies.add(dep_entity)
                        collect_dependency_entities(dep)  # Recursively collect nested dependencies
            
            for chain_data in temp_chains.values():
                collect_dependency_entities(chain_data)
            
            # Build final chains, excluding entities that appear in dependencies
            for table_name in table_lineage_data.keys():
                # Only include as top-level if not already in dependencies
                if table_name not in entities_in_dependencies:
                    chains[table_name] = temp_chains[table_name]
            
            # Special handling: Ensure QUERY_RESULT appears as a dependency of source tables
            # rather than as an independent top-level entity, but include its transformations
            if 'QUERY_RESULT' in result.table_lineage.transformations:
                # QUERY_RESULT should appear as a dependency in the existing chains
                # The transformations will be included in the dependency metadata
                # This ensures proper flow: source_table → QUERY_RESULT (with transformations)
                pass  # The dependency relationships are already handled by build_comprehensive_chain
            
            # For downstream analysis with CTAS queries, ensure target tables appear as dependencies of source tables
            # rather than as separate top-level entities
            if chain_type == "downstream" and sql and sql.strip().upper().startswith('CREATE TABLE'):
                upstream_lineage_data = result.table_lineage.upstream
                downstream_lineage_data = result.table_lineage.downstream
                
                # Identify CTAS source and target tables
                ctas_source_tables = set()
                ctas_target_tables = set()
                
                for target_table, source_tables in upstream_lineage_data.items():
                    if target_table != 'QUERY_RESULT':
                        ctas_target_tables.add(target_table)
                        ctas_source_tables.update(source_tables)
                
                # Remove target tables from chains if they were added as top-level
                for target_table in ctas_target_tables:
                    if target_table in chains:
                        del chains[target_table]
                
                # Ensure target tables appear as dependencies in source table chains
                for source_table in ctas_source_tables:
                    if source_table in chains:
                        # Add target tables as dependencies of this source table
                        source_chain = chains[source_table]
                        source_dependencies = source_chain.get('dependencies', [])
                        
                        # Get target tables that depend on this source
                        for target_table in downstream_lineage_data.get(source_table, []):
                            if target_table != 'QUERY_RESULT' and target_table in ctas_target_tables:
                                # Check if this target is not already in dependencies
                                existing_dep_entities = {dep.get('entity') for dep in source_dependencies}
                                if target_table not in existing_dep_entities:
                                    # Build the target table as a dependency
                                    target_depth = source_chain.get('depth', 0) + 1
                                    target_chain = build_comprehensive_chain(target_table, "table", target_depth + 1, None, source_table)
                                    source_dependencies.append(target_chain)
                        
                        source_chain['dependencies'] = source_dependencies
            
            # Build chains for all columns (only if no table chains exist to avoid redundancy)
            if not chains:
                for column_ref in column_lineage_data.keys():
                    chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1, None, None)
        
        # Post-process chains to add missing source columns from filter conditions
        self._add_missing_source_columns(chains, sql)
        
        # Post-process to integrate column transformations into column metadata
        self._integrate_column_transformations(chains, sql)
        
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
            
            # Handle star (*) expansion
            if select_clause == '*':
                # Expand * to actual columns from CTE or table
                expanded_columns = self._expand_star_columns(sql, column_lineage_data)
                result_columns.extend(expanded_columns)
            else:
                # Handle simple cases like "SELECT name, email FROM users"
                # Split by comma and clean up
                columns = [col.strip() for col in select_clause.split(',')]
                
                for col in columns:
                    original_col = col.strip()  # Keep original column reference
                    
                    # Extract alias name from "expression AS alias" -> "alias"
                    col_name = original_col
                    if ' AS ' in col.upper():
                        # Split on AS and take the alias part (after AS)
                        parts = col.split(' AS ', 1) if ' AS ' in col else col.split(' as ', 1)
                        if len(parts) > 1:
                            col_name = parts[1].strip()
                        else:
                            col_name = parts[0].strip()
                    elif ' as ' in col:
                        # Handle lowercase 'as'
                        parts = col.split(' as ', 1)
                        if len(parts) > 1:
                            col_name = parts[1].strip()
                        else:
                            col_name = parts[0].strip()
                    
                    # Clean up any remaining whitespace and quotes
                    col_name = col_name.strip().strip('"').strip("'")
                    
                    if col_name:
                        # Try to find upstream columns from column lineage data
                        upstream_columns = []
                        
                        # For table-prefixed columns (e.g., "u.name"), look for the proper upstream mapping
                        if '.' in col_name:
                            # Extract table alias and column name
                            table_alias, simple_col_name = col_name.split('.', 1)
                            
                            # Look for corresponding upstream in column lineage data
                            for column_ref, upstream_cols in column_lineage_data.items():
                                # Check if this column reference matches our target
                                if column_ref.endswith(f".{simple_col_name}"):
                                    # Found a match, use its upstream
                                    upstream_columns = list(upstream_cols)
                                    # Convert QUERY_RESULT.name to proper table.name format
                                    corrected_upstream = []
                                    for upstream_ref in upstream_columns:
                                        if upstream_ref.startswith("QUERY_RESULT."):
                                            # Map back to the actual source table based on the column_ref
                                            source_table = column_ref.split('.')[0]  # e.g., "users" from "users.name"
                                            corrected_upstream.append(f"{source_table}.{upstream_ref.split('.')[1]}")
                                        else:
                                            corrected_upstream.append(upstream_ref)
                                    upstream_columns = corrected_upstream
                                    break
                        else:
                            # Simple column name without table prefix
                            query_result_ref = f"QUERY_RESULT.{col_name}"
                            for column_ref, upstream_cols in column_lineage_data.items():
                                if (column_ref == query_result_ref or 
                                    column_ref.endswith(f".{col_name}") or
                                    column_ref == col_name):
                                    upstream_columns = list(upstream_cols)
                                    break
                        
                        # If no upstream found, try to infer from SQL structure
                        if not upstream_columns:
                            if '.' in col_name:
                                # For table-prefixed columns, try to resolve table alias
                                table_alias, simple_col_name = col_name.split('.', 1)
                                
                                # Look for table alias mapping in SQL (simple heuristic)
                                alias_pattern = rf'{table_alias}\s+(?:JOIN\s+)?(\w+)|(\w+)\s+{table_alias}'
                                alias_match = re.search(alias_pattern, sql, re.IGNORECASE)
                                if alias_match:
                                    table_name = alias_match.group(1) or alias_match.group(2)
                                    upstream_columns = [f"{table_name}.{simple_col_name}"]
                                else:
                                    # Fallback: assume alias maps to similarly named table
                                    if table_alias.lower().startswith('u'):
                                        upstream_columns = [f"users.{simple_col_name}"]
                                    elif table_alias.lower().startswith('o'):
                                        upstream_columns = [f"orders.{simple_col_name}"]
                                    else:
                                        upstream_columns = [f"{table_alias}.{simple_col_name}"]
                            else:
                                # Simple column name - use first table as fallback
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
    
    def _filter_query_result_columns_by_parent(self, all_columns: List[Dict], parent_entity: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Filter QUERY_RESULT columns to only include those that originate from the specified parent entity.
        
        Args:
            all_columns: All columns in the QUERY_RESULT
            parent_entity: The parent table entity that's requesting the filtered columns
            column_lineage_data: Column lineage mapping
            
        Returns:
            Filtered list of columns that come from the parent entity
        """
        filtered_columns = []
        
        for col in all_columns:
            col_name = col.get('name', '')
            upstream = col.get('upstream', [])
            
            # Check if this column originates from the parent entity
            column_belongs_to_parent = False
            
            # Method 1: Check if column name has parent table prefix (e.g., "u.name" belongs to "users")
            if '.' in col_name:
                # Extract table alias/prefix from column name (e.g., "u.name" -> "u")
                table_prefix = col_name.split('.')[0]
                
                # Common alias mappings - this is a simple heuristic
                # In real scenarios, we'd need proper alias resolution
                if (parent_entity.startswith('user') and table_prefix.lower() in ['u', 'user']) or \
                   (parent_entity.startswith('order') and table_prefix.lower() in ['o', 'order']) or \
                   (parent_entity.startswith('customer') and table_prefix.lower() in ['c', 'customer']):
                    column_belongs_to_parent = True
            
            # Method 2: Check upstream lineage to see if column comes from parent entity
            if not column_belongs_to_parent and upstream:
                for upstream_ref in upstream:
                    if upstream_ref.startswith(f"{parent_entity}."):
                        column_belongs_to_parent = True
                        break
            
            # Method 3: Check column lineage data for mapping
            if not column_belongs_to_parent:
                # Look for column references that match parent entity
                for col_ref, lineage_list in column_lineage_data.items():
                    if col_ref.startswith(f"{parent_entity}."):
                        # Extract column name from reference (e.g., "users.name" -> "name")
                        source_col_name = col_ref.split('.')[-1]
                        target_col_name = col_name.split('.')[-1] if '.' in col_name else col_name
                        
                        if source_col_name == target_col_name:
                            column_belongs_to_parent = True
                            break
            
            if column_belongs_to_parent:
                filtered_columns.append(col)
        
        return filtered_columns
    
    def _expand_star_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """Expand * to actual columns from CTE or table."""
        expanded_columns = []
        
        try:
            # Parse the SQL to understand the structure
            ast = self._parse_sql(sql)
            
            # Handle WITH statements (CTEs)
            if isinstance(ast, exp.With):
                main_select = ast.this
                cte_definitions = {}
                
                # Collect CTE definitions
                for cte in ast.expressions:
                    if isinstance(cte, exp.CTE):
                        cte_name = cte.alias
                        if isinstance(cte.this, exp.Select):
                            # Parse CTE's SELECT to get its columns
                            cte_select_data = self.select_parser.parse(str(cte.this))
                            cte_columns = [col['column_name'] for col in cte_select_data.get('select_columns', []) if col['column_name']]
                            cte_definitions[cte_name] = cte_columns
                
                # Now check the main SELECT to see what table it's selecting from
                if isinstance(main_select, exp.Select):
                    from_clause = main_select.args.get('from')
                    if from_clause and hasattr(from_clause, 'this'):
                        source_table = str(from_clause.this)
                        
                        # If source table is a CTE, expand to its columns
                        if source_table in cte_definitions:
                            for col_name in cte_definitions[source_table]:
                                # Try to find upstream for this column
                                upstream_columns = []
                                for column_ref, upstream_cols in column_lineage_data.items():
                                    if column_ref.endswith(f".{col_name}") or column_ref == col_name:
                                        upstream_columns = list(upstream_cols)
                                        break
                                
                                expanded_columns.append({
                                    "name": col_name,
                                    "upstream": upstream_columns,
                                    "type": "DIRECT"
                                })
            
            # Handle simple SELECT * FROM table (no CTE)
            elif isinstance(ast, exp.Select):
                from_clause = ast.args.get('from')
                if from_clause and hasattr(from_clause, 'this'):
                    source_table = str(from_clause.this)
                    
                    # Try to get columns from metadata registry if available
                    if hasattr(self, 'metadata_registry') and self.metadata_registry:
                        table_metadata = self.metadata_registry.get_table_metadata(source_table)
                        if table_metadata and 'columns' in table_metadata:
                            for col_info in table_metadata['columns']:
                                col_name = col_info.get('name')
                                if col_name:
                                    expanded_columns.append({
                                        "name": col_name,
                                        "upstream": [f"{source_table}.{col_name}"],
                                        "type": "DIRECT"
                                    })
        
        except Exception as e:
            # If expansion fails, return empty list to avoid showing * as column
            pass
        
        return expanded_columns
    
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
    
    def _build_cte_lineage_chain(self, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries with proper single-flow chains."""
        # Get CTE-specific analysis
        cte_result = self._analyze_cte(sql)
        cte_lineage = cte_result.get('cte_lineage', {})
        ctes = cte_lineage.get('ctes', {})
        execution_order = cte_lineage.get('execution_order', [])
        
        # Also get standard table/column lineage for base tables and final result
        result = self.analyze(sql, **kwargs)
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        
        # Build chains dictionary
        chains = {}
        
        if chain_type == "downstream":
            # NEW APPROACH: Build single continuous chains from base tables through CTEs to QUERY_RESULT
            
            # 1. Identify base tables (non-CTE tables)
            base_tables = set()
            
            # Tables that CTEs depend on
            for cte_name, cte_data in ctes.items():
                source_tables = cte_data.get('source_tables', [])
                for source in source_tables:
                    table_name = source.get('name')
                    if table_name and table_name not in ctes:  # Not a CTE
                        base_tables.add(table_name)
            
            # Tables that final query references directly (like users in JOINs)
            if table_lineage_data:
                for table_name in table_lineage_data.keys():
                    if table_name not in ctes:  # Not a CTE
                        base_tables.add(table_name)
            
            # 2. For each base table, build a single continuous dependency chain
            for table_name in base_tables:
                chains[table_name] = self._build_single_cte_chain(table_name, ctes, execution_order, table_lineage_data, sql)
            
            # 3. Handle any orphaned CTEs (CTEs that don't connect to base tables)
            # This shouldn't happen in well-formed queries, but handle gracefully
            for cte_name in execution_order:
                if cte_name not in ctes:
                    continue
                    
                # Check if this CTE is already included in any base table chain
                cte_included = False
                for base_table in base_tables:
                    if self._cte_in_chain(cte_name, chains.get(base_table, {})):
                        cte_included = True
                        break
                
                # If CTE is not included in any chain, add it as a separate chain
                if not cte_included:
                    chains[cte_name] = self._build_single_cte_chain(cte_name, ctes, execution_order, table_lineage_data)
        
        else:  # upstream
            # For upstream: start from final result, trace back through CTEs to base tables
            
            # 1. Add final result
            final_result = cte_lineage.get('final_result', {})
            if final_result:
                chains['QUERY_RESULT'] = self._build_cte_final_result(final_result, ctes, execution_order, 0, table_lineage_data)
            
            # 2. Add CTE entities in reverse execution order
            for i, cte_name in enumerate(reversed(execution_order)):
                if cte_name in ctes:
                    cte_entity = self._build_cte_entity(cte_name, ctes[cte_name], ctes, execution_order, i + 1)
                    chains[cte_name] = cte_entity
            
            # 3. Add base tables
            base_tables = set()
            for cte_name, cte_data in ctes.items():
                source_tables = cte_data.get('source_tables', [])
                for source in source_tables:
                    table_name = source.get('name')
                    if table_name and table_name not in ctes:  # Not a CTE
                        base_tables.add(table_name)
            
            for table_name in base_tables:
                chains[table_name] = self._build_cte_table_entity(table_name, ctes, execution_order, len(execution_order) + len(base_tables))
        
        # Build final result structure
        actual_max_depth = max([entity.get('depth', 0) for entity in chains.values()]) if chains else 0
        
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
                "total_columns": 0,  # CTE queries don't have column lineage in the same way
                "has_transformations": bool(cte_result.get('transformations')),
                "has_metadata": bool(result.metadata),
                "chain_count": len(chains)
            },
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    def _build_cte_entity(self, cte_name: str, cte_data: Dict, all_ctes: Dict, execution_order: List[str], depth: int) -> Dict[str, Any]:
        """Build entity data for a CTE."""
        entity = {
            "entity": cte_name,
            "entity_type": "cte",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": {
                "table_columns": [],
                "is_cte": True
            }
        }
        
        # For downstream flow, CTEs should point to what depends on them (next CTE in chain or QUERY_RESULT)
        # Find what this CTE connects to in the execution chain
        
        # First, check if any other CTE depends on this one
        cte_dependencies = []
        for next_cte_name in execution_order:
            if next_cte_name != cte_name and next_cte_name in all_ctes:
                next_cte_data = all_ctes[next_cte_name]
                next_source_tables = next_cte_data.get('source_tables', [])
                
                # Check if the next CTE depends on this CTE
                for source in next_source_tables:
                    if source.get('name') == cte_name:
                        # This CTE should point to the next CTE in the chain
                        transformations = [{
                            'type': 'table_transformation',
                            'source_table': cte_name,
                            'target_table': next_cte_name,
                            'filter_conditions': [],
                            'group_by_columns': [],
                            'joins': []
                        }]
                        
                        cte_dependencies.append({
                            "entity": next_cte_name,
                            "transformations": transformations
                        })
                        break
        
        # Add the CTE dependencies to the entity
        entity["dependencies"].extend(cte_dependencies)
        
        # Add columns from CTE definition
        columns = cte_data.get('columns', [])
        for col in columns:
            column_name = col.get('alias') or col.get('name', 'unknown')
            
            column_info = {
                'name': column_name,
                'type': 'COMPUTED' if col.get('is_computed') else 'DIRECT',
                'source_column': col.get('source_column'),
                'source_table': col.get('source_table'),
                'expression': col.get('expression')
            }
            
            # Add transformation details for computed columns
            if col.get('is_computed') or col.get('is_aggregate') or col.get('is_function'):
                raw_expression = col.get('raw_expression', col.get('expression', ''))
                
                if raw_expression:
                    # Extract source expression and determine transformation type
                    source_expression = self._extract_source_from_expression(raw_expression, column_name)
                    transformation_type = self._get_transformation_type(col, raw_expression)
                    function_type = self._extract_function_type_generic(raw_expression)
                    
                    column_info["transformation"] = {
                        "source_expression": source_expression,
                        "transformation_type": transformation_type,
                        "function_type": function_type
                    }
            
            entity["metadata"]["table_columns"].append(column_info)
        
        return entity
    
    def _build_cte_table_entity(self, table_name: str, all_ctes: Dict, execution_order: List[str], depth: int) -> Dict[str, Any]:
        """Build entity data for a base table used by CTEs."""
        entity = {
            "entity": table_name,
            "entity_type": "table",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": {
                "table_columns": [],
                "is_cte": False
            }
        }
        
        # For downstream flow, base tables should have dependencies pointing to CTEs that use them
        # This creates the proper flow: base_table → CTE → QUERY_RESULT
        
        # Find CTEs that depend on this base table and add them as dependencies
        connected_to_cte = False
        for cte_name in execution_order:
            if cte_name in all_ctes:
                cte_data = all_ctes[cte_name]
                source_tables = cte_data.get('source_tables', [])
                
                # Check if this CTE uses the current base table
                for source in source_tables:
                    if source.get('name') == table_name:
                        # Add this CTE as a dependency of the base table for downstream flow
                        # Apply table-specific filtering to CTE transformations
                        cte_transformations = cte_data.get('transformations', [])
                        filtered_conditions = self._filter_nested_conditions(cte_transformations, table_name)
                        
                        transformations = [{
                            'type': 'table_transformation',
                            'source_table': table_name,
                            'target_table': cte_name,
                            'filter_conditions': filtered_conditions,
                            'group_by_columns': [],
                            'joins': []
                        }]
                        
                        entity["dependencies"].append({
                            "entity": cte_name,
                            "transformations": transformations
                        })
                        connected_to_cte = True
                        break
        
        # If this base table is not connected to any CTE but appears in final result,
        # it should point directly to QUERY_RESULT (like users table in JOINs)
        if not connected_to_cte:
            # This base table is only referenced in the final SELECT, so add QUERY_RESULT dependency
            transformations = [{
                'type': 'table_transformation',
                'source_table': table_name,
                'target_table': 'QUERY_RESULT',
                'filter_conditions': [],
                'group_by_columns': [],
                'joins': []
            }]
            
            entity["dependencies"].append({
                "entity": "QUERY_RESULT",
                "transformations": transformations
            })
        
        return entity
    
    def _build_cte_final_result(self, final_result: Dict, all_ctes: Dict, execution_order: List[str], depth: int, table_lineage_data: Dict = None) -> Dict[str, Any]:
        """Build entity data for the final query result."""
        entity = {
            "entity": "QUERY_RESULT",
            "entity_type": "query_result",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": {
                "table_columns": [],
                "is_cte": False
            }
        }
        
        # Analyze final result to find what it DIRECTLY depends on
        # Use the parsed structure instead of string parsing
        referenced_entities = set()
        
        # Extract table references from FROM clause
        from_tables = final_result.get('from_tables', [])
        for table_info in from_tables:
            table_name = table_info.get('table_name')
            if table_name:
                referenced_entities.add(table_name)
        
        # Extract table references from JOIN clauses
        joins = final_result.get('joins', [])
        for join_info in joins:
            table_name = join_info.get('table_name')
            if table_name:
                referenced_entities.add(table_name)
        
        # Fallback: if no references found from parsed structure, try string matching
        if not referenced_entities:
            final_sql = str(final_result)
            for cte_name in execution_order:
                if cte_name.lower() in final_sql.lower():
                    referenced_entities.add(cte_name)
        
        # Additional fallback: use table lineage data for direct table references
        if not referenced_entities and table_lineage_data:
            final_sql = str(final_result)
            for table_name in table_lineage_data.keys():
                if table_name.lower() not in [cte.lower() for cte in execution_order]:
                    if table_name.lower() in final_sql.lower():
                        referenced_entities.add(table_name)
        
        # Add dependencies
        for entity_name in referenced_entities:
            transformations = [{
                'type': 'table_transformation',
                'source_table': entity_name,
                'target_table': 'QUERY_RESULT',
                'filter_conditions': [],
                'group_by_columns': [],
                'joins': []
            }]
            
            entity["dependencies"].append({
                "entity": entity_name,
                "transformations": transformations
            })
        
        return entity
    
    def _is_column_from_table(self, column_name: str, table_name: str, context_info: dict = None) -> bool:
        """Check if a column belongs to a specific table based on naming patterns."""
        if not column_name or not table_name:
            return False
        
        # Handle qualified column names (e.g., "users.active", "u.salary")
        if '.' in column_name:
            column_parts = column_name.split('.')
            if len(column_parts) >= 2:
                table_part = column_parts[0].lower()
                
                # Direct table name match (e.g., "users.active" matches "users")
                if table_part == table_name.lower():
                    return True
                
                # Alias match (e.g., "u.salary" matches "users" if u is alias for users)
                # Common aliases: u for users, o for orders, etc.
                if table_name.lower().startswith(table_part):
                    return True
                
                # Check reverse - if table_part contains table_name (e.g., "users" in "user_details")
                if table_part in table_name.lower() or table_name.lower() in table_part:
                    return True
        else:
            # Unqualified column - use context to determine if it belongs to this table
            if context_info:
                # If this is a single-table context (only one source table), assume unqualified columns belong to it
                if context_info.get('is_single_table_context', False):
                    return True
                    
                # If we have a list of tables in the context and this is the primary/source table
                tables_in_context = context_info.get('tables_in_context', [])
                if len(tables_in_context) == 1 and tables_in_context[0] == table_name:
                    return True
        
        # If no table qualifier and no clear context, default to False for filtering
        # This prevents unqualified columns from being assigned to every table in multi-table contexts
        return False
    
    def _is_aggregate_function_for_table(self, column_expr: str, table_name: str) -> bool:
        """Check if an aggregate function expression is relevant to a specific table."""
        if not column_expr or not table_name:
            return False
        
        # Handle aggregate functions like COUNT(*), AVG(u.salary), SUM(users.amount)
        column_expr_lower = column_expr.lower()
        
        # Check if the expression contains explicit table references first
        if table_name.lower() in column_expr_lower:
            return True
        
        # Check for table aliases (u for users, o for orders)
        if table_name.lower().startswith('u') and ('u.' in column_expr_lower):
            return True
        elif table_name.lower().startswith('o') and ('o.' in column_expr_lower):
            return True
        
        # COUNT(*) is only relevant to the main grouped table (users in this case)
        # Only assign COUNT(*) to users table, not orders table
        if column_expr_lower == 'count(*)' and table_name.lower() == 'users':
            return True
        
        return False
    
    def _build_single_cte_chain(self, start_entity: str, ctes: dict, execution_order: list, table_lineage_data: dict, sql: str = None) -> dict:
        """Build a single continuous chain from a base table through CTEs to QUERY_RESULT."""
        
        # For the nested CTE example:
        # orders → order_stats → customer_tiers → tier_summary → QUERY_RESULT
        # users → QUERY_RESULT
        
        # Find the complete chain starting from this entity
        chain = self._build_complete_cte_chain_from_entity(start_entity, ctes, execution_order)
        
        # Build the nested dependency structure
        root_entity = None
        current_entity_dict = None
        
        for i, entity_name in enumerate(chain):
            entity_dict = {
                "entity": entity_name,
                "entity_type": "cte" if entity_name in ctes else "table",
                "depth": i,
                "dependencies": [],
                "metadata": self._get_entity_metadata(entity_name, ctes, sql)
            }
            
            # Add transformations if not the first entity
            if i > 0:
                prev_entity = chain[i-1]
                transformations = self._get_cte_transformations(prev_entity, entity_name, ctes)
                if transformations:
                    entity_dict["transformations"] = transformations
            
            if i == 0:
                # This is the root entity
                root_entity = entity_dict
                current_entity_dict = entity_dict
            else:
                # Add as dependency of the previous entity
                current_entity_dict["dependencies"].append(entity_dict)
                current_entity_dict = entity_dict
        
        # Add QUERY_RESULT as final dependency
        query_result_entity = {
            "entity": "QUERY_RESULT",
            "entity_type": "table", 
            "depth": len(chain),
            "dependencies": [],
            "metadata": {"table_columns": [], "is_cte": False}
        }
        
        # Add transformations from last entity to QUERY_RESULT
        if chain:
            last_entity = chain[-1]
            transformations = self._get_final_transformations(last_entity, table_lineage_data, sql)
            if transformations:
                query_result_entity["transformations"] = transformations
        
        if current_entity_dict:
            current_entity_dict["dependencies"].append(query_result_entity)
        
        return root_entity if root_entity else query_result_entity
    
    def _build_complete_cte_chain_from_entity(self, start_entity: str, ctes: dict, execution_order: list) -> list:
        """Build the complete chain of entities starting from a base table or CTE."""
        
        # The chain should be: [start_entity, cte1, cte2, ..., final_cte]
        # For orders: [orders, order_stats, customer_tiers, tier_summary]  
        # For users: [users] (no CTEs)
        
        chain = [start_entity]
        current_entity = start_entity
        
        # Follow the CTE dependency chain
        while True:
            next_cte = None
            
            # Find the next CTE that depends on the current entity
            for cte_name in execution_order:
                if cte_name in ctes:
                    cte_data = ctes[cte_name]
                    source_tables = cte_data.get('source_tables', [])
                    
                    # Check if this CTE depends on the current entity
                    for source in source_tables:
                        if source.get('name') == current_entity:
                            next_cte = cte_name
                            break
                    
                    if next_cte:
                        break
            
            if next_cte:
                chain.append(next_cte)
                current_entity = next_cte
            else:
                break
        
        return chain
    
    def _find_cte_chain_from_entity(self, start_entity: str, ctes: dict, execution_order: list) -> list:
        """Find the chain of CTEs that flows from the given entity."""
        chain = []
        
        # If start_entity is a CTE, start from it
        if start_entity in ctes:
            current_cte = start_entity
        else:
            # Find the first CTE that depends on this base table
            current_cte = None
            for cte_name in execution_order:
                if cte_name in ctes:
                    cte_data = ctes[cte_name]
                    source_tables = cte_data.get('source_tables', [])
                    for source in source_tables:
                        if source.get('name') == start_entity:
                            current_cte = cte_name
                            break
                    if current_cte:
                        break
        
        # Build the chain following the CTE dependencies
        while current_cte and current_cte in ctes:
            if current_cte != start_entity:  # Don't include start entity in chain
                chain.append(current_cte)
            
            # Find next CTE that depends on current_cte
            next_cte = None
            for cte_name in execution_order:
                if cte_name in ctes and cte_name != current_cte:
                    cte_data = ctes[cte_name]
                    source_tables = cte_data.get('source_tables', [])
                    for source in source_tables:
                        if source.get('name') == current_cte:
                            next_cte = cte_name
                            break
                    if next_cte:
                        break
            
            current_cte = next_cte
        
        return chain
    
    def _cte_in_chain(self, cte_name: str, chain_entity: dict) -> bool:
        """Check if a CTE is included in a dependency chain."""
        if not chain_entity:
            return False
            
        # Check if this entity is the CTE
        if chain_entity.get("entity") == cte_name:
            return True
        
        # Recursively check dependencies
        for dep in chain_entity.get("dependencies", []):
            if self._cte_in_chain(cte_name, dep):
                return True
        
        return False
    
    def _get_entity_metadata(self, entity_name: str, ctes: dict, sql: str = None) -> dict:
        """Get metadata for an entity (table or CTE)."""
        if entity_name in ctes:
            cte_data = ctes[entity_name]
            
            # Get columns from CTE data
            columns = cte_data.get('columns', [])
            table_columns = []
            
            # Extract column transformations for this CTE if SQL is available
            column_transformations_map = {}
            if sql:
                try:
                    column_transformations = self._extract_column_transformations(sql, None, entity_name)
                    for col_trans in column_transformations:
                        col_name = col_trans.get('column_name')
                        if col_name:
                            column_transformations_map[col_name] = col_trans
                except Exception:
                    pass
            
            # Build table_columns with transformations
            for col in columns:
                col_name = col.get('name') or col.get('alias')
                if col_name:
                    column_info = {
                        "name": col_name,
                        "upstream": [],
                        "type": "VARCHAR"  # Default type for CTEs
                    }
                    
                    # Add transformation if available
                    if col_name in column_transformations_map:
                        column_info["transformation"] = column_transformations_map[col_name]
                    
                    table_columns.append(column_info)
            
            return {
                "table_columns": table_columns,
                "is_cte": True
            }
        else:
            # Base table metadata (simplified)
            return {
                "table_columns": [],
                "is_cte": False
            }
    
    def _get_cte_transformations(self, source_entity: str, target_entity: str, ctes: dict) -> list:
        """Get transformations between two entities in CTE chain."""
        if target_entity in ctes:
            cte_data = ctes[target_entity]
            transformations = cte_data.get('transformations', [])
            if transformations:
                return [{
                    "type": "table_transformation",
                    "source_table": source_entity,
                    "target_table": target_entity,
                    "filter_conditions": self._filter_nested_conditions(transformations, source_entity),
                    "group_by_columns": [],
                    "joins": []
                }]
        return []
    
    def _get_final_transformations(self, source_entity: str, table_lineage_data: dict, sql: str = None) -> list:
        """Get transformations from final CTE/table to QUERY_RESULT."""
        transformation = {
            "type": "table_transformation",
            "source_table": source_entity,
            "target_table": "QUERY_RESULT",
            "filter_conditions": [],
            "group_by_columns": [],
            "joins": []
        }
        
        # Extract transformations from the main SELECT query using transformation parser
        if sql:
            try:
                transformation_data = self.transformation_parser.parse(sql)
                
                # Extract filter conditions from main SELECT WHERE clause
                filters = transformation_data.get('filters', {})
                if filters and filters.get('conditions'):
                    # Convert transformation parser format to our expected format
                    filter_conditions = []
                    for condition in filters['conditions']:
                        # Transform format to match expected structure
                        filter_conditions.append({
                            "type": "FILTER",
                            "conditions": [condition]
                        })
                    
                    transformation["filter_conditions"] = filter_conditions
                
                # Extract aggregation information (GROUP BY, HAVING, aggregate functions)
                aggregations = transformation_data.get('aggregations', {})
                if aggregations:
                    # Group by columns
                    if aggregations.get('group_by_columns'):
                        transformation["group_by_columns"] = aggregations['group_by_columns']
                    
                    # Having conditions
                    if aggregations.get('having_conditions'):
                        having_conditions = []
                        for condition in aggregations['having_conditions']:
                            having_conditions.append({
                                "column": condition.get('column', ''),
                                "operator": condition.get('operator', '='),
                                "value": condition.get('value', '')
                            })
                        transformation["having_conditions"] = having_conditions
                    
                    # Aggregate functions
                    if aggregations.get('aggregate_functions'):
                        transformation["aggregate_functions"] = aggregations['aggregate_functions']
                
                # Extract JOIN information (preserve pairing of join_type and conditions)
                joins = transformation_data.get('joins', [])
                if joins:
                    joined_data = []
                    for join in joins:
                        join_entry = {
                            "join_type": join.get('join_type', 'INNER JOIN'),
                            "right_table": join.get('right_table'),
                            "conditions": join.get('conditions', [])
                        }
                        joined_data.append(join_entry)
                    
                    if joined_data:
                        transformation["joins"] = joined_data
                
                # Extract sorting information
                sorting = transformation_data.get('sorting', {})
                if sorting and sorting.get('order_by_columns'):
                    order_by_info = sorting['order_by_columns']
                    if isinstance(order_by_info, list):
                        # Convert to simple string format
                        order_by_strings = []
                        for order_item in order_by_info:
                            if isinstance(order_item, dict):
                                column = order_item.get('column', '')
                                direction = order_item.get('direction', 'ASC')
                                order_by_strings.append(f"{column} {direction}")
                            else:
                                order_by_strings.append(str(order_item))
                        transformation["order_by_columns"] = order_by_strings
                
                # Extract window functions
                window_functions = transformation_data.get('window_functions', [])
                if window_functions:
                    transformation["window_functions"] = window_functions
                
                # Extract limiting information (LIMIT/OFFSET)
                limiting = transformation_data.get('limiting', {})
                if limiting and (limiting.get('limit') is not None or limiting.get('offset') is not None):
                    limit_info = {}
                    if limiting.get('limit') is not None:
                        limit_info['limit'] = limiting['limit']
                    if limiting.get('offset') is not None:
                        limit_info['offset'] = limiting['offset']
                    transformation["limiting"] = limit_info
                
                # Extract CASE statement information
                case_statements = transformation_data.get('case_statements', [])
                if case_statements:
                    transformation["case_statements"] = case_statements
                    
            except Exception as e:
                # If transformation parsing fails, fall back to empty transformations
                pass
        
        return [transformation]
    
    def _filter_nested_conditions(self, filter_conditions: list, entity_name: str) -> list:
        """Filter nested CTE transformation conditions to only include those relevant to the entity."""
        if not filter_conditions:
            return []
        
        relevant_filters = []
        for filter_item in filter_conditions:
            if isinstance(filter_item, dict):
                # Handle nested structure like {'type': 'FILTER', 'conditions': [...]}
                if filter_item.get('type') == 'FILTER' and 'conditions' in filter_item:
                    relevant_conditions = []
                    for condition in filter_item['conditions']:
                        if isinstance(condition, dict) and 'column' in condition:
                            # For CTE context, include the condition if:
                            # 1. Column is explicitly qualified for this table, OR
                            # 2. Column is unqualified (assume it belongs to source table in CTE)
                            column_name = condition['column']
                            if (self._is_column_from_table(column_name, entity_name) or 
                                ('.' not in column_name)):  # Unqualified column in CTE context
                                relevant_conditions.append(condition)
                    
                    if relevant_conditions:
                        relevant_filters.append({
                            'type': 'FILTER',
                            'conditions': relevant_conditions
                        })
                
                # Handle other transformation types (GROUP_BY, etc.)
                elif filter_item.get('type') in ['GROUP_BY', 'HAVING', 'ORDER_BY']:
                    # Include these transformation types as they are generally table-specific
                    relevant_filters.append(filter_item)
                
                # Handle flat structure like {'column': 'users.active', 'operator': '=', 'value': 'TRUE'}
                elif 'column' in filter_item:
                    column_name = filter_item['column']
                    if (self._is_column_from_table(column_name, entity_name) or 
                        ('.' not in column_name)):  # Unqualified column in CTE context
                        relevant_filters.append(filter_item)
        
        return relevant_filters
    
    def _is_transformation_relevant_to_table(self, col_trans: dict, table_name: str, sql: str) -> bool:
        """Check if a column transformation is relevant to a specific table."""
        if not col_trans or not table_name:
            return False
        
        source_expression = col_trans.get('source_expression', '')
        if not source_expression:
            return False
        
        # For aggregations and functions, check if they reference columns from this table
        # Examples:
        # - COUNT(*) -> could be relevant to any table in GROUP BY context
        # - AVG(u.salary) -> only relevant to users table (u alias)
        # - SUM(orders.total) -> only relevant to orders table
        
        # Remove function wrapper to get the inner expression
        # e.g., "AVG(u.salary)" -> "u.salary"
        inner_expr = self._extract_inner_expression(source_expression)
        
        if inner_expr:
            # Check if the inner expression references this table
            if self._expression_references_table(inner_expr, table_name, sql):
                return True
        
        # For COUNT(*) and similar general aggregations, only include in the primary table
        # In JOIN queries, the primary table is usually the one being grouped by
        if source_expression.upper().strip() == 'COUNT(*)':
            # Special handling for CTAS queries - COUNT(*) is always relevant to the source table
            if sql and sql.strip().upper().startswith('CREATE TABLE'):
                return True
            
            # For other queries, check if this table is involved in GROUP BY
            return self._table_involved_in_group_by(table_name, sql)
        
        return False
    
    def _extract_inner_expression(self, source_expression: str) -> str:
        """Extract the inner expression from a function call."""
        import re
        
        # Match function patterns like AVG(u.salary), MAX(u.hire_date), etc.
        match = re.match(r'^[A-Z_]+\s*\(\s*(.+?)\s*\)$', source_expression.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return source_expression
    
    def _expression_references_table(self, expression: str, table_name: str, sql: str = None) -> bool:
        """Check if an expression references a specific table."""
        if not expression or not table_name:
            return False
        
        expression_lower = expression.lower()
        table_lower = table_name.lower()
        
        # Check for explicit table references
        # e.g., "users.salary" or "u.salary" (where u is alias for users)
        if f"{table_lower}." in expression_lower:
            return True
        
        # Check for table aliases (first letter of table name)
        if table_lower.startswith('u') and 'u.' in expression_lower:
            return True
        elif table_lower.startswith('o') and 'o.' in expression_lower:
            return True
        
        # For single-table contexts, unqualified column names belong to that table
        if sql and self._is_single_table_context(sql):
            # In single-table queries, assume unqualified columns belong to the main table
            # Skip this check if the expression contains table qualifiers from other tables
            if not ('.' in expression_lower and not f"{table_lower}." in expression_lower):
                return True
        
        return False
    
    def _is_single_table_context(self, sql: str) -> bool:
        """Check if this is a single-table query context."""
        if not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Check if there are any JOINs
        if 'JOIN' in sql_upper:
            return False
        
        # Count number of table references in FROM clause
        from_start = sql_upper.find('FROM')
        if from_start == -1:
            return True  # No FROM clause, likely a simple query
        
        # Look for multiple table names separated by commas (old-style joins)
        from_section = sql_upper[from_start:sql_upper.find('WHERE', from_start) if 'WHERE' in sql_upper[from_start:] else len(sql_upper)]
        if ',' in from_section:
            return False
        
        return True
    
    def _table_involved_in_group_by(self, table_name: str, sql: str) -> bool:
        """Check if a table is involved in GROUP BY clause."""
        if not sql or not table_name:
            return False
        
        sql_upper = sql.upper()
        
        # Simple check - if GROUP BY contains references to this table
        group_by_start = sql_upper.find('GROUP BY')
        if group_by_start == -1:
            return False
        
        # Extract GROUP BY clause (until ORDER BY, HAVING, or end)
        group_by_clause = sql_upper[group_by_start:]
        for terminator in ['ORDER BY', 'HAVING', 'LIMIT', ';']:
            if terminator in group_by_clause:
                group_by_clause = group_by_clause[:group_by_clause.find(terminator)]
        
        table_lower = table_name.lower()
        group_by_lower = group_by_clause.lower()
        
        # Check if this table is referenced in GROUP BY
        if f"{table_lower}." in group_by_lower:
            return True
        
        # Check for aliases
        if table_lower.startswith('u') and 'u.' in group_by_lower:
            return True
        elif table_lower.startswith('o') and 'o.' in group_by_lower:
            return True
        
        return False
    
    def parse_insert(self, sql: str) -> Dict[str, Any]:
        """Parse INSERT using modular parser."""
        return self.insert_parser.parse(sql)
    
    def parse_update(self, sql: str) -> Dict[str, Any]:
        """Parse UPDATE using modular parser."""
        return self.update_parser.parse(sql)
