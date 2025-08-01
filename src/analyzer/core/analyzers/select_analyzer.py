"""SELECT-specific analyzer methods."""

from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer


class SelectAnalyzer(BaseAnalyzer):
    """Handles SELECT-specific analysis and lineage building."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize SELECT analyzer."""
        super().__init__(dialect)
        # These will be injected from the main analyzer
        self.select_parser = None
        self.transformation_parser = None
    
    def analyze_select(self, sql: str) -> Dict[str, Any]:
        """Analyze SELECT statement using select parser."""
        if self.select_parser:
            return self.select_parser.parse(sql)
        return {}
    
    def build_select_lineage(self, select_data: Dict[str, Any], transformation_data: Dict[str, Any]) -> Dict[str, Any]:
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
            'columns': self.extract_result_columns(select_data)
        })
        
        return lineage
    
    def get_columns_used_from_table(self, table: Dict[str, Any], select_data: Dict[str, Any]) -> List[str]:
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
    
    def extract_result_columns(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    
    def extract_source_tables(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
    
    def infer_query_result_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
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
                            # For non-prefixed columns, try to find direct matches in lineage data
                            potential_matches = []
                            for column_ref, upstream_cols in column_lineage_data.items():
                                if column_ref == col_name or column_ref.endswith(f".{col_name}"):
                                    potential_matches.extend(upstream_cols)
                            
                            upstream_columns = list(set(potential_matches))  # Remove duplicates
                        
                        # Create column entry
                        result_columns.append({
                            "name": col_name,
                            "upstream": upstream_columns,
                            "type": "DIRECT"
                        })
        
        return result_columns
    
    def filter_query_result_columns_by_parent(self, all_columns: List[Dict], parent_entity: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Filter QUERY_RESULT columns to only include those that are relevant to a specific parent entity.
        
        Args:
            all_columns: List of all QUERY_RESULT columns
            parent_entity: The parent entity (table) to filter by
            column_lineage_data: Column lineage mapping
            
        Returns:
            Filtered list of columns relevant to the parent entity
        """
        filtered_columns = []
        
        for column_info in all_columns:
            column_name = column_info.get("name", "")
            upstream_columns = column_info.get("upstream", [])
            
            # Check if any upstream column references this parent entity
            is_relevant = False
            for upstream_col in upstream_columns:
                # Check if upstream column references the parent entity
                # Handle cases like "users.name", "QUERY_RESULT.name", etc.
                if upstream_col.startswith(f"{parent_entity}."):
                    is_relevant = True
                    break
                elif '.' not in upstream_col and parent_entity in upstream_col:
                    # Handle unqualified columns that might belong to this entity
                    is_relevant = True
                    break
            
            # Also check if the column name itself suggests it belongs to this entity
            # (e.g., qualified column names like "u.name" where u might be an alias for users)
            if not is_relevant and '.' in column_name:
                col_prefix = column_name.split('.')[0]
                # Check if the prefix matches the parent entity or is a common alias
                if (col_prefix.lower() == parent_entity.lower() or 
                    (parent_entity.lower().startswith(col_prefix.lower()) and len(col_prefix) <= 2)):
                    is_relevant = True
            
            if is_relevant:
                filtered_columns.append(column_info)
        
        return filtered_columns
    
    def _expand_star_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Expand * (star) columns to actual column names.
        
        Args:
            sql: The SQL query string
            column_lineage_data: Column lineage mapping
            
        Returns:
            List of expanded column information
        """
        expanded_columns = []
        
        try:
            # Parse the SQL to identify the source of the star expansion
            import sqlglot
            from sqlglot import expressions as exp
            
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Get CTE definitions if this is a CTE query
            cte_definitions = {}
            if isinstance(ast, exp.With):
                for cte in ast.expressions:
                    if hasattr(cte, 'alias') and hasattr(cte, 'this'):
                        cte_name = str(cte.alias)
                        # Extract column names from the CTE definition
                        cte_select = cte.this
                        if isinstance(cte_select, exp.Select):
                            cte_columns = []
                            for expr in cte_select.expressions:
                                if hasattr(expr, 'alias') and expr.alias:
                                    cte_columns.append(str(expr.alias))
                                elif hasattr(expr, 'this'):
                                    cte_columns.append(str(expr.this))
                            cte_definitions[cte_name] = cte_columns
                
                # Get the main SELECT statement
                main_select = ast.this
                
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