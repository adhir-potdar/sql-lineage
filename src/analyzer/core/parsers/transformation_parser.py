"""Parser for SQL transformations (WHERE, JOIN, GROUP BY, etc.)."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser
from ...utils.condition_utils import GenericConditionHandler


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
            'limiting': self.parse_transformation_limiting(select_stmt),
            'case_statements': self.parse_transformation_case_statements(select_stmt)
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
        # Use generic condition handler with dict output format
        return GenericConditionHandler.extract_all_conditions(
            condition_expr, 
            column_resolver=None, 
            output_format="dict"
        )
    
    
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
        
        # Handle case where condition_expr might be a function
        if callable(condition_expr):
            try:
                condition_expr = condition_expr()
            except:
                return conditions
        
        # Skip if condition_expr is None or not an expression
        if not condition_expr or not hasattr(condition_expr, 'find_all'):
            return conditions
        
        # Use generic condition handler for join conditions (supports all operators)
        return GenericConditionHandler.extract_join_conditions(
            condition_expr, 
            output_format="dict"
        )
    
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
        group_by_clause = select_stmt.args.get('group')
        if group_by_clause:
            for expr in group_by_clause.expressions:
                column = str(expr).strip()
                transformation['group_by_columns'].append(column)
                transformation['affected_columns'].add(self.clean_column_reference(column))
        
        # Parse aggregate functions in SELECT
        for expr in select_stmt.expressions:
            agg_funcs = self._extract_aggregate_functions(expr)
            transformation['aggregate_functions'].extend(agg_funcs)
        
        # Parse HAVING conditions
        having_clause = select_stmt.args.get('having')
        if having_clause:
            having_conditions = self._extract_filter_conditions(having_clause.this)
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
                partition_by = window.args.get('partition_by', [])
                if partition_by:
                    for part_expr in partition_by:
                        column = str(part_expr).strip()
                        transformation['partition_by'].append(column)
                        transformation['affected_columns'].append(self.clean_column_reference(column))
                
                # Parse ORDER BY
                order_clause = window.args.get('order')
                if order_clause and hasattr(order_clause, 'expressions'):
                    for order_expr in order_clause.expressions:
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
        
        order_clause = select_stmt.args.get('order')
        if not order_clause:
            return transformation
        
        for expr in order_clause.expressions:
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
        
        # Parse LIMIT clause
        limit_clause = select_stmt.args.get('limit')
        if limit_clause and limit_clause.expression:
            transformation['limit'] = int(str(limit_clause.expression))
        
        # Parse OFFSET clause (separate from LIMIT in SQLGlot)
        offset_clause = select_stmt.args.get('offset')  
        if offset_clause and offset_clause.expression:
            transformation['offset'] = int(str(offset_clause.expression))
        
        return transformation
    
    def parse_transformation_case_statements(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse CASE statement transformations."""
        transformations = []
        
        # Look for CASE statements in SELECT expressions
        for expr in select_stmt.expressions:
            for case_node in expr.find_all(exp.Case):
                transformation = {
                    'type': 'CASE_STATEMENT',
                    'conditions': [],
                    'else_value': None,
                    'affected_columns': [],
                    'expression': str(case_node)
                }
                
                # Parse WHEN conditions
                ifs = case_node.args.get('ifs', [])
                if ifs:
                    for when_clause in ifs:
                        condition = {
                            'when_condition': str(when_clause.this) if when_clause.this else None,
                            'then_value': str(when_clause.args.get('true')) if when_clause.args.get('true') else None
                        }
                        transformation['conditions'].append(condition)
                        
                        # Extract affected columns from WHEN condition
                        if when_clause.this:
                            for col in when_clause.this.find_all(exp.Column):
                                col_name = self.clean_column_reference(str(col))
                                if col_name not in transformation['affected_columns']:
                                    transformation['affected_columns'].append(col_name)
                
                # Parse ELSE clause
                default_value = case_node.args.get('default')
                if default_value:
                    transformation['else_value'] = str(default_value)
                
                transformations.append(transformation)
        
        # Also look for CASE statements in WHERE, HAVING, and other clauses
        for clause_name in ['where', 'having']:
            clause = select_stmt.args.get(clause_name)
            if clause and clause.this:
                for case_node in clause.this.find_all(exp.Case):
                    transformation = {
                        'type': 'CASE_STATEMENT',
                        'conditions': [],
                        'else_value': None,
                        'affected_columns': [],
                        'expression': str(case_node),
                        'context': clause_name.upper()
                    }
                    
                    # Parse WHEN conditions
                    ifs = case_node.args.get('ifs', [])
                    if ifs:
                        for when_clause in ifs:
                            condition = {
                                'when_condition': str(when_clause.this) if when_clause.this else None,
                                'then_value': str(when_clause.args.get('true')) if when_clause.args.get('true') else None
                            }
                            transformation['conditions'].append(condition)
                            
                            # Extract affected columns from WHEN condition
                            if when_clause.this:
                                for col in when_clause.this.find_all(exp.Column):
                                    col_name = self.clean_column_reference(str(col))
                                    if col_name not in transformation['affected_columns']:
                                        transformation['affected_columns'].append(col_name)
                    
                    # Parse ELSE clause
                    default_value = case_node.args.get('default')
                    if default_value:
                        transformation['else_value'] = str(default_value)
                    
                    transformations.append(transformation)
        
        return transformations