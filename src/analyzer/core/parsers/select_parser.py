"""Parser for SELECT statements."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser
from ...utils.condition_utils import GenericConditionHandler
from ...utils.logging_config import get_logger


class SelectParser(BaseParser):
    """Parser for SELECT statement components."""
    
    def __init__(self, dialect: str = "trino"):
        super().__init__(dialect)
        self.logger = get_logger('parsers.select')
    
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse SELECT statement and extract all components."""
        self.logger.info(f"Parsing SELECT statement (length: {len(sql)})")
        self.logger.debug(f"SELECT SQL: {sql[:200]}..." if len(sql) > 200 else f"SELECT SQL: {sql}")
        
        try:
            ast = self.parse_sql(sql)
            
            # Find the main SELECT statement
            select_stmt = self._find_main_select(ast)
            if not select_stmt:
                self.logger.warning("No main SELECT statement found in SQL")
                return {}
            
            self.logger.debug("Main SELECT statement found")
        
            self.logger.debug("Parsing SELECT components")
            result = {
                'select_columns': self.parse_select_columns(select_stmt),
                'from_tables': self.parse_from_clause(select_stmt),
                'joins': self.parse_joins(select_stmt),
                'where_conditions': self.parse_where_clause(select_stmt),
                'group_by': self.parse_group_by(select_stmt),
                'having_conditions': self.parse_having_clause(select_stmt),
                'order_by': self.parse_order_by(select_stmt),
                'limit_clause': self.parse_limit_clause(select_stmt),
                'ctes': self.parse_ctes(ast)
            }
            
            self.logger.info(f"SELECT parsing completed - {len(result['select_columns'])} columns, {len(result['from_tables'])} tables, {len(result['joins'])} joins")
            return result
            
        except Exception as e:
            self.logger.error(f"SELECT parsing failed: {str(e)}", exc_info=True)
            raise
    
    def _find_main_select(self, ast) -> Optional[exp.Select]:
        """Find the main SELECT statement in the AST."""
        if isinstance(ast, exp.Select):
            return ast
        
        # Look for SELECT in WITH statements
        if isinstance(ast, exp.With):
            return self._find_main_select(ast.this)
        
        # Look for SELECT in subqueries
        for node in ast.find_all(exp.Select):
            return node
        
        return None
    
    def parse_select_columns(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse SELECT column list."""
        columns = []
        
        for expression in select_stmt.expressions:
            # Handle Star (*) expressions by expanding them
            if isinstance(expression, exp.Star):
                # Expand * to actual columns from the source
                expanded_columns = self._expand_star_expression(expression, select_stmt)
                columns.extend(expanded_columns)
                continue
            
            column_info = {
                'raw_expression': str(expression),
                'column_name': None,
                'alias': None,
                'source_table': None,
                'is_aggregate': False,
                'is_window_function': False,
                'is_computed': False
            }
            
            # Handle different expression types
            if isinstance(expression, exp.Alias):
                column_info['alias'] = expression.alias
                column_info['column_name'] = expression.alias
                actual_expr = expression.this
            else:
                actual_expr = expression
                column_info['column_name'] = self.extract_column_name(actual_expr)
            
            # Check if it's a column reference
            if isinstance(actual_expr, exp.Column):
                column_info['source_table'] = actual_expr.table if actual_expr.table else None
                if not column_info['column_name']:
                    column_info['column_name'] = actual_expr.name
            
            # Check for aggregate functions
            if any(isinstance(node, exp.AggFunc) for node in actual_expr.find_all(exp.AggFunc)):
                column_info['is_aggregate'] = True
            
            # Check for window functions
            if any(isinstance(node, exp.Window) for node in actual_expr.find_all(exp.Window)):
                column_info['is_window_function'] = True
            
            # Check for computed expressions (CASE, functions, etc.)
            # A column is computed if it's not a simple column reference
            if not isinstance(actual_expr, exp.Column):
                # Check for CASE expressions
                if any(isinstance(node, exp.Case) for node in actual_expr.find_all(exp.Case)):
                    column_info['is_computed'] = True
                # Check for function calls (excluding aggregates which are handled above)
                elif any(isinstance(node, exp.Func) for node in actual_expr.find_all(exp.Func)):
                    # Only mark as computed if it's not already marked as aggregate
                    if not column_info['is_aggregate']:
                        column_info['is_computed'] = True
                # Any other non-column expression is computed
                elif not column_info['is_aggregate'] and not column_info['is_window_function']:
                    column_info['is_computed'] = True
            
            columns.append(column_info)
        
        return columns
    
    def parse_from_clause(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse FROM clause tables."""
        tables = []
        
        # Access FROM clause through args
        from_clause = select_stmt.args.get('from')
        if from_clause:
            # Handle simple case where FROM has a single table in 'this'
            if hasattr(from_clause, 'this') and from_clause.this:
                table_info = {
                    'table_name': None,
                    'alias': None,
                    'is_subquery': False
                }
                
                table_expr = from_clause.this
                if isinstance(table_expr, exp.Alias):
                    table_info['alias'] = table_expr.alias
                    actual_table = table_expr.this
                else:
                    actual_table = table_expr
                
                if isinstance(actual_table, exp.Table):
                    table_info['table_name'] = self.extract_table_name(actual_table)
                elif isinstance(actual_table, exp.Subquery):
                    table_info['is_subquery'] = True
                    table_info['table_name'] = table_info['alias'] or 'subquery'
                
                tables.append(table_info)
            
            # Handle multiple tables in expressions (though this might be rare in FROM)
            if hasattr(from_clause, 'expressions') and from_clause.expressions:
                for table_expr in from_clause.expressions:
                    table_info = {
                        'table_name': None,
                        'alias': None,
                        'is_subquery': False
                    }
                    
                    if isinstance(table_expr, exp.Alias):
                        table_info['alias'] = table_expr.alias
                        actual_table = table_expr.this
                    else:
                        actual_table = table_expr
                    
                    if isinstance(actual_table, exp.Table):
                        table_info['table_name'] = self.extract_table_name(actual_table)
                    elif isinstance(actual_table, exp.Subquery):
                        table_info['is_subquery'] = True
                        table_info['table_name'] = table_info['alias'] or 'subquery'
                    
                    tables.append(table_info)
        
        return tables
    
    def parse_joins(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse JOIN clauses."""
        joins = []
        
        for join in select_stmt.find_all(exp.Join):
            join_info = {
                'join_type': self._get_join_type(join),
                'table_name': None,
                'alias': None,
                'conditions': []
            }
            
            # Get joined table
            if isinstance(join.this, exp.Alias):
                join_info['alias'] = join.this.alias
                join_info['table_name'] = self.extract_table_name(join.this.this)
            else:
                join_info['table_name'] = self.extract_table_name(join.this)
            
            # Get join conditions
            if join.on:
                join_info['conditions'] = self._parse_join_conditions(join.on)
            
            joins.append(join_info)
        
        return joins
    
    def _get_join_type(self, join: exp.Join) -> str:
        """Determine JOIN type."""
        if join.side:
            return f"{join.side.upper()} JOIN"
        elif join.kind:
            return f"{join.kind.upper()} JOIN"
        else:
            return "INNER JOIN"
    
    def _parse_join_conditions(self, condition_expr) -> List[Dict[str, Any]]:
        """Parse JOIN ON conditions."""
        conditions = []
        
        # Handle case where condition_expr might be a function or None
        if not condition_expr or callable(condition_expr):
            return conditions
        
        # Use generic condition handler for join conditions (supports all operators)
        return GenericConditionHandler.extract_join_conditions(
            condition_expr, 
            output_format="dict"
        )
    
    def parse_where_clause(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse WHERE clause conditions."""
        where_clause = select_stmt.args.get('where')
        if not where_clause:
            return []
        
        return self._parse_conditions(where_clause.this)
    
    def parse_having_clause(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse HAVING clause conditions."""
        having_clause = select_stmt.args.get('having')
        if not having_clause:
            return []
        
        return self._parse_conditions(having_clause.this)
    
    def _parse_conditions(self, condition_expr) -> List[Dict[str, Any]]:
        """Parse filter conditions from WHERE/HAVING clauses."""
        # Use generic condition handler with dict output format
        return GenericConditionHandler.extract_all_conditions(
            condition_expr, 
            column_resolver=None, 
            output_format="dict"
        )
    
    def parse_group_by(self, select_stmt: exp.Select) -> List[str]:
        """Parse GROUP BY clause."""
        group_clause = select_stmt.args.get('group')
        if not group_clause:
            return []
        
        columns = []
        for expr in group_clause.expressions:
            columns.append(str(expr).strip())
        
        return columns
    
    def parse_order_by(self, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Parse ORDER BY clause."""
        order_clause = select_stmt.args.get('order')
        if not order_clause:
            return []
        
        order_columns = []
        for expr in order_clause.expressions:
            order_info = {
                'column': str(expr.this).strip(),
                'direction': 'ASC'
            }
            
            if hasattr(expr, 'desc') and expr.desc:
                order_info['direction'] = 'DESC'
            
            order_columns.append(order_info)
        
        return order_columns
    
    def parse_limit_clause(self, select_stmt: exp.Select) -> Optional[Dict[str, Any]]:
        """Parse LIMIT clause."""
        limit_clause = select_stmt.args.get('limit')
        if not limit_clause:
            return None
        
        return {
            'limit': int(str(limit_clause.expression)) if limit_clause.expression else None,
            'offset': int(str(limit_clause.offset)) if hasattr(limit_clause, 'offset') and limit_clause.offset else None
        }
    
    def parse_ctes(self, ast) -> List[Dict[str, Any]]:
        """Parse Common Table Expressions (CTEs)."""
        ctes = []
        
        if isinstance(ast, exp.With):
            for cte in ast.expressions:
                if isinstance(cte, exp.CTE):
                    cte_info = {
                        'name': cte.alias,
                        'columns': [],
                        'sql': str(cte.this)
                    }
                    
                    # Parse the CTE's SELECT statement
                    if isinstance(cte.this, exp.Select):
                        cte_select_data = self.parse(str(cte.this))
                        cte_info['select_data'] = cte_select_data
                        cte_info['columns'] = [col['column_name'] for col in cte_select_data.get('select_columns', [])]
                    
                    ctes.append(cte_info)
        
        return ctes
    
    def _expand_star_expression(self, star_expr: exp.Star, select_stmt: exp.Select) -> List[Dict[str, Any]]:
        """Expand * expression to actual columns from source tables/CTEs."""
        expanded_columns = []
        
        # Get FROM clause to identify source tables/CTEs
        from_clause = select_stmt.args.get('from')
        if not from_clause:
            return expanded_columns
            
        # Handle FROM table or CTE
        if hasattr(from_clause, 'this') and from_clause.this:
            source_table = from_clause.this
            
            # If it's a table reference, we need to get columns from metadata or CTE
            if isinstance(source_table, exp.Table):
                table_name = str(source_table)
                
                # Check if this is a CTE by looking at available CTEs in the context
                cte_columns = self._get_cte_columns_from_context(table_name)
                if cte_columns:
                    # Expand to CTE columns
                    for col_name in cte_columns:
                        expanded_columns.append({
                            'raw_expression': col_name,
                            'column_name': col_name,
                            'alias': None,
                            'source_table': table_name,
                            'is_aggregate': False,
                            'is_window_function': False
                        })
                else:
                    # For regular tables, try to get columns from metadata registry
                    # This is a fallback - in real scenarios we'd need the metadata
                    # For now, return empty list to avoid showing * as column
                    pass
        
        return expanded_columns
    
    def _get_cte_columns_from_context(self, table_name: str) -> List[str]:
        """Get column names from a CTE definition using the current parsing context."""
        # This method will be enhanced to work with the analyzer's CTE context
        # For now, return empty list - the analyzer will handle this at a higher level
        return []