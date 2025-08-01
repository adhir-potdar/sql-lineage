"""Table and column transformation analyzer."""

from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer


class TransformationAnalyzer(BaseAnalyzer):
    """Analyzer for table and column transformations."""
    
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
    
    def _extract_cte_column_transformations(self, sql: str, target_table: str) -> List[Dict]:
        """Extract column transformations from CTE queries."""
        column_transformations = []
        
        try:
            # Use CTE parser to get column transformations
            cte_result = self._analyze_cte(sql)
            cte_lineage = cte_result.get('cte_lineage', {})
            ctes = cte_lineage.get('ctes', {})
            
            # Look for the target CTE
            if target_table in ctes:
                cte_data = ctes[target_table]
                columns = cte_data.get('columns', [])
                
                for col in columns:
                    column_name = col.get('alias') or col.get('name', 'unknown')
                    
                    # Extract transformation details for computed columns
                    if col.get('is_computed') or col.get('is_aggregate') or col.get('is_function'):
                        raw_expression = col.get('raw_expression', col.get('expression', ''))
                        
                        if raw_expression:
                            # Extract source expression and determine transformation type
                            source_expression = self._extract_source_from_expression(raw_expression, column_name)
                            transformation_type = self._get_transformation_type(col, raw_expression)
                            function_type = self._extract_function_type_generic(raw_expression)
                            
                            column_transformations.append({
                                'column_name': column_name,
                                'source_expression': source_expression,
                                'transformation_type': transformation_type,
                                'function_type': function_type,
                                'source_table': col.get('source_table'),
                                'target_table': target_table
                            })
        except Exception:
            pass
        
        return column_transformations
    
    def _extract_select_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """Extract column transformations from SELECT queries."""
        column_transformations = []
        
        try:
            # Use transformation parser to get column transformations
            transformation_data = self.transformation_parser.parse(sql)
            column_transforms = transformation_data.get('column_transformations', [])
            
            for col_trans in column_transforms:
                if (col_trans.get('source_table') == source_table and 
                    col_trans.get('target_table') == target_table):
                    column_transformations.append(col_trans)
        except Exception:
            pass
        
        return column_transformations
    
    def get_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type generically."""
        if col_info.get('is_aggregate'):
            return 'AGGREGATE'
        elif col_info.get('is_window_function'):
            return 'WINDOW_FUNCTION'
        elif self._is_computed_expression(expression):
            return 'COMPUTED'
        else:
            return 'DIRECT'
    
    def extract_function_type_generic(self, expression: str) -> str:
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
        
        return 'UNKNOWN'
    
    def is_computed_expression(self, expression: str) -> bool:
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
    
    def extract_source_from_expression(self, expression: str, target_name: str) -> str:
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
    
    def is_transformation_relevant_to_table(self, col_trans: Dict, table_name: str, sql: str) -> bool:
        """Check if a column transformation is relevant to a specific table."""
        if not col_trans or not table_name:
            return False
        
        source_expression = col_trans.get('source_expression', '')
        source_table = col_trans.get('source_table', '')
        
        # Check if the transformation explicitly references this table
        if source_table == table_name:
            return True
        
        # Check if the source expression contains this table name
        if table_name.lower() in source_expression.lower():
            return True
        
        return False
    
    def filter_nested_conditions(self, filter_conditions: list, entity_name: str) -> list:
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
    
    # Helper methods (these are delegated to chain_builder for actual implementation)
    def _is_column_from_table(self, column_name: str, table_name: str, context_info: dict = None) -> bool:
        """Check if a column belongs to a specific table based on naming patterns."""
        from .chain_builder import ChainBuilder
        chain_builder = ChainBuilder(self.dialect)
        return chain_builder._is_column_from_table(column_name, table_name, context_info)
    
    def _get_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type generically."""
        return self.get_transformation_type(col_info, expression)
    
    def _extract_function_type_generic(self, expression: str) -> str:
        """Extract function type from expression generically."""
        return self.extract_function_type_generic(expression)
    
    def _is_computed_expression(self, expression: str) -> bool:
        """Check if an expression is computed."""
        return self.is_computed_expression(expression)
    
    def _extract_source_from_expression(self, expression: str, target_name: str) -> str:
        """Extract the source part from transformation expression."""
        return self.extract_source_from_expression(expression, target_name)
    
    def _analyze_cte(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE statement using CTE analyzer."""
        from .cte_analyzer import CTEAnalyzer
        cte_analyzer = CTEAnalyzer(self.dialect)
        cte_analyzer.set_metadata_registry(self.metadata_registry)
        return cte_analyzer.analyze_cte(sql)