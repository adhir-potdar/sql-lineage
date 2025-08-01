"""Column transformation analyzer."""

from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer
from ..parsers import SelectParser, CTEParser, CTASParser


class TransformationAnalyzer(BaseAnalyzer):
    """Analyzer for column transformations across different SQL types."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize transformation analyzer."""
        super().__init__(dialect)
        self.select_parser = SelectParser(dialect)
        self.cte_parser = CTEParser(dialect)
        self.ctas_parser = CTASParser(dialect)
    
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
                
                # Update each column with transformation information
                for column in table_columns:
                    col_name = column.get('name')
                    if col_name in column_transformations_map:
                        transformation = column_transformations_map[col_name]
                        column['transformation'] = {
                            'source_expression': transformation.get('source_expression'),
                            'transformation_type': transformation.get('transformation_type'),
                            'function_type': transformation.get('function_type'),
                            'full_expression': transformation.get('full_expression')
                        }
                        column['is_transformed'] = True
                    else:
                        column['is_transformed'] = False
                        
            except Exception:
                # If transformation extraction fails, continue without adding transformation info
                pass
        
        # Process all entities in the chain recursively
        def process_chain_recursive(chain_data):
            if isinstance(chain_data, dict):
                # Process current entity if it has table data
                table_name = chain_data.get('table_name')
                if table_name:
                    process_entity_columns(chain_data, target_table=table_name)
                
                # Process downstream entities
                downstream = chain_data.get('downstream', [])
                for downstream_entity in downstream:
                    process_chain_recursive(downstream_entity)
                
                # Process upstream entities  
                upstream = chain_data.get('upstream', [])
                for upstream_entity in upstream:
                    process_chain_recursive(upstream_entity)
        
        # Start processing from the root
        process_chain_recursive(chains)