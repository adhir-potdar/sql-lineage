"""Table and column transformations analyzer."""

from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer

# Import new utility modules
from ...utils.regex_patterns import extract_where_clause, extract_filter_conditions, extract_from_table
from ...utils.sqlglot_helpers import parse_sql_safely, get_where_conditions
from ...utils.metadata_utils import create_result_column_metadata
from ...utils.sql_parsing_utils import extract_function_type, extract_alias_from_expression
from ..transformation_engine import TransformationEngine


class TransformationAnalyzer(BaseAnalyzer):
    """Analyzer for table and column transformations."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize transformation analyzer with transformation engine."""
        super().__init__(dialect)
        self.transformation_engine = TransformationEngine(dialect)
    
    def parse_transformations(self, sql: str) -> Dict[str, Any]:
        """Parse transformations using modular parser.""" 
        return self.transformation_parser.parse(sql)
    
    def extract_filter_transformations(self, sql: str) -> List[Dict]:
        """Extract filter transformations from WHERE clause."""
        return self.transformation_engine.extract_filter_transformations(sql)
    
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
        
        return lineage
    
    def _get_columns_used_from_table(self, table: Dict[str, Any], select_data: Dict[str, Any]) -> List[str]:
        """Get columns used from a specific table in the SELECT query."""
        columns_used = []
        
        # This is a simplified implementation - could be enhanced
        select_columns = select_data.get('select_columns', [])
        table_name = table.get('table_name')
        table_alias = table.get('alias')
        
        for col in select_columns:
            col_name = col.get('column_name', '')
            # Check if column references this table
            if table_name and f"{table_name}." in col_name:
                columns_used.append(col_name)
            elif table_alias and f"{table_alias}." in col_name:
                columns_used.append(col_name)
        
        return columns_used
    
    def extract_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
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
        function_type = extract_function_type(expression)
        
        # Check for CASE expressions
        if expression and expression.upper().strip().startswith('CASE'):
            return 'CASE'
        
        return function_type if function_type != "UNKNOWN" else 'EXPRESSION'
    
    def _extract_source_from_expression(self, expression: str, target_name: str) -> str:
        """Extract the source part from transformation expression."""
        if not expression:
            return 'UNKNOWN'
        
        # Use utility function to extract alias and get the source part
        alias = extract_alias_from_expression(expression)
        if alias:
            # If alias found, remove the " AS alias" part
            alias_index = expression.upper().rfind(' AS ')
            if alias_index != -1:
                return expression[:alias_index].strip()
        
        # If no AS clause, return the expression as is
        return expression.strip()
    
    def integrate_column_transformations(self, chains: Dict, sql: str = None) -> None:
        """Integrate column transformations into column metadata throughout the chain."""
        if not sql:
            return
        
        def process_entity_columns(entity_data, source_table=None, target_table=None):
            """Process columns in an entity to add transformation information."""
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # Get column transformations for this entity
            try:
                column_transformations = self.extract_column_transformations(sql, source_table, target_table)
                
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
        from ...utils.regex_patterns import SQLPatterns
        
        # Match function patterns like AVG(u.salary), MAX(u.hire_date), etc.
        match = SQLPatterns.FUNCTION_WITH_ARGS.match(source_expression.strip())
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