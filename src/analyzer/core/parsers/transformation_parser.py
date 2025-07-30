"""Parser for SQL transformations (WHERE, JOIN, GROUP BY, etc.)."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser


class TransformationParser(BaseParser):
    """Parser for SQL transformation components."""
    
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse all transformations in SQL statement."""
        ast = self.parse_sql(sql)
        select_stmt = self._find_select(ast)
        
        if not select_stmt:
            return {}
        
        return {
            'filters': self.parse_transformation_filters(select_stmt),
            'joins': self.parse_transformation_joins(select_stmt),
            'aggregations': self.parse_transformation_aggregations(select_stmt),
            'window_functions': self.parse_transformation_window_functions(select_stmt),
            'sorting': self.parse_transformation_sorting(select_stmt),
            'limiting': self.parse_transformation_limiting(select_stmt)
        }
    
    def _find_select(self, ast) -> Optional[exp.Select]:
        """Find SELECT statement in AST."""
        if isinstance(ast, exp.Select):
            return ast
        
        for node in ast.find_all(exp.Select):
            return node
        
        return None
    
    def parse_transformation_filters(self, select_stmt: exp.Select) -> Dict[str, Any]:
        """Parse WHERE clause transformations."""
        transformation = {
            'type': 'FILTER',
            'conditions': [],
            'affected_tables': set(),
            'affected_columns': set()
        }
        
        where_clause = select_stmt.args.get('where')
        if not where_clause:
            return transformation
        
        # Extract filter conditions
        conditions = self._extract_filter_conditions(where_clause.this)
        transformation['conditions'] = conditions
        
        # Identify affected tables and columns
        for condition in conditions:
            column = condition.get('column', '')
            if '.' in column:
                table_name = column.split('.')[0]
                transformation['affected_tables'].add(table_name)
            
            transformation['affected_columns'].add(self.clean_column_reference(column))
        
        # Convert sets to lists for JSON serialization
        transformation['affected_tables'] = list(transformation['affected_tables'])
        transformation['affected_columns'] = list(transformation['affected_columns'])
        
        return transformation
    
    def _extract_filter_conditions(self, condition_expr) -> List[Dict[str, Any]]:
        """Extract individual filter conditions."""
        conditions = []
        
        # Handle different condition types
        condition_types = [
            (exp.EQ, '='),
            (exp.GT, '>'),
            (exp.LT, '<'),
            (exp.GTE, '>='),
            (exp.LTE, '<='),
            (exp.NEQ, '!='),
            (exp.Like, 'LIKE'),
            (exp.In, 'IN'),
            (exp.Is, 'IS')
        ]
        
        for condition_type, operator in condition_types:
            for node in condition_expr.find_all(condition_type):
                condition = self._create_condition(node, operator)
                if condition:
                    conditions.append(condition)
        
        return conditions
    
    def _create_condition(self, node, operator: str) -> Optional[Dict[str, Any]]:
        """Create condition dictionary from AST node."""
        try:
            if operator == 'IS NULL':
                return {
                    'column': str(node.this).strip(),
                    'operator': operator,
                    'value': None
                }
            elif operator == 'NOT':
                # Handle NOT conditions
                if isinstance(node.this, exp.Like):
                    return {
                        'column': str(node.this.left).strip(),
                        'operator': 'NOT LIKE',
                        'value': str(node.this.right).strip()
                    }
            elif hasattr(node, 'left') and hasattr(node, 'right'):
                return {
                    'column': str(node.left).strip(),
                    'operator': operator,
                    'value': str(node.right).strip()
                }
        except Exception:
            pass
        
        return None
    
    def parse_transformation_joins(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse JOIN transformations."""
        transformations = []
        
        for join in select_stmt.find_all(exp.Join):
            transformation = {
                'type': 'JOIN',
                'join_type': self._get_join_type(join),
                'left_table': None,
                'right_table': None,
                'conditions': [],
                'affected_tables': [],
                'affected_columns': []
            }
            
            # Get joined table
            if isinstance(join.this, exp.Alias):
                transformation['right_table'] = join.this.alias
            else:
                transformation['right_table'] = self.extract_table_name(join.this)
            
            # Parse join conditions
            if join.on:
                conditions = self._parse_join_conditions(join.on)
                transformation['conditions'] = conditions
                
                # Extract affected tables and columns
                for condition in conditions:
                    left_col = condition.get('left_column', '')
                    right_col = condition.get('right_column', '')
                    
                    if '.' in left_col:
                        transformation['affected_tables'].append(left_col.split('.')[0])
                    if '.' in right_col:
                        transformation['affected_tables'].append(right_col.split('.')[0])
                    
                    transformation['affected_columns'].extend([
                        self.clean_column_reference(left_col),
                        self.clean_column_reference(right_col)
                    ])
            
            transformations.append(transformation)
        
        return transformations
    
    def _get_join_type(self, join: exp.Join) -> str:
        """Get JOIN type string."""
        if join.side:
            return f"{join.side.upper()} JOIN"
        elif join.kind:
            return f"{join.kind.upper()} JOIN"
        else:
            return "INNER JOIN"
    
    def _parse_join_conditions(self, condition_expr) -> List[Dict[str, Any]]:
        """Parse JOIN conditions."""
        conditions = []
        
        for eq in condition_expr.find_all(exp.EQ):
            conditions.append({
                'left_column': str(eq.left).strip(),
                'operator': '=',
                'right_column': str(eq.right).strip()
            })
        
        return conditions
    
    def parse_transformation_aggregations(self, select_stmt: exp.Select) -> Dict[str, Any]:
        """Parse GROUP BY and aggregate function transformations."""
        transformation = {
            'type': 'AGGREGATION',
            'group_by_columns': [],
            'aggregate_functions': [],
            'having_conditions': [],
            'affected_tables': set(),
            'affected_columns': set()
        }
        
        # Parse GROUP BY
        if select_stmt.group:
            for expr in select_stmt.group.expressions:
                column = str(expr).strip()
                transformation['group_by_columns'].append(column)
                transformation['affected_columns'].add(self.clean_column_reference(column))
        
        # Parse aggregate functions in SELECT
        for expr in select_stmt.expressions:
            agg_funcs = self._extract_aggregate_functions(expr)
            transformation['aggregate_functions'].extend(agg_funcs)
        
        # Parse HAVING conditions
        if select_stmt.having:
            having_conditions = self._extract_filter_conditions(select_stmt.having.this)
            transformation['having_conditions'] = having_conditions
        
        # Convert sets to lists
        transformation['affected_tables'] = list(transformation['affected_tables'])
        transformation['affected_columns'] = list(transformation['affected_columns'])
        
        return transformation
    
    def _extract_aggregate_functions(self, expr) -> List[Dict[str, Any]]:
        """Extract aggregate functions from expression."""
        agg_functions = []
        
        for agg_node in expr.find_all(exp.AggFunc):
            agg_info = {
                'function': agg_node.__class__.__name__.upper(),
                'column': str(agg_node.this).strip() if agg_node.this else None,
                'distinct': getattr(agg_node, 'distinct', False)
            }
            agg_functions.append(agg_info)
        
        return agg_functions
    
    def parse_transformation_window_functions(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse window function transformations."""
        transformations = []
        
        for expr in select_stmt.expressions:
            for window in expr.find_all(exp.Window):
                transformation = {
                    'type': 'WINDOW_FUNCTION',
                    'function': str(window.this),
                    'partition_by': [],
                    'order_by': [],
                    'frame': None,
                    'affected_columns': []
                }
                
                # Parse PARTITION BY
                if window.partition_by:
                    for part_expr in window.partition_by:
                        column = str(part_expr).strip()
                        transformation['partition_by'].append(column)
                        transformation['affected_columns'].append(self.clean_column_reference(column))
                
                # Parse ORDER BY
                if window.order:
                    for order_expr in window.order.expressions:
                        column = str(order_expr.this).strip()
                        direction = 'DESC' if getattr(order_expr, 'desc', False) else 'ASC'
                        transformation['order_by'].append(f"{column} {direction}")
                        transformation['affected_columns'].append(self.clean_column_reference(column))
                
                transformations.append(transformation)
        
        return transformations
    
    def parse_transformation_sorting(self, select_stmt: exp.Select) -> Dict[str, Any]:
        """Parse ORDER BY transformation."""
        transformation = {
            'type': 'SORTING',
            'order_by_columns': [],
            'affected_columns': []
        }
        
        if not select_stmt.order:
            return transformation
        
        for expr in select_stmt.order.expressions:
            column = str(expr.this).strip()
            direction = 'DESC' if getattr(expr, 'desc', False) else 'ASC'
            
            transformation['order_by_columns'].append({
                'column': column,
                'direction': direction
            })
            transformation['affected_columns'].append(self.clean_column_reference(column))
        
        return transformation
    
    def parse_transformation_limiting(self, select_stmt: exp.Select) -> Dict[str, Any]:
        """Parse LIMIT/OFFSET transformation."""
        transformation = {
            'type': 'LIMITING',
            'limit': None,
            'offset': None
        }
        
        if select_stmt.limit:
            if select_stmt.limit.expression:
                transformation['limit'] = int(str(select_stmt.limit.expression))
            if select_stmt.limit.offset:
                transformation['offset'] = int(str(select_stmt.limit.offset))
        
        return transformation