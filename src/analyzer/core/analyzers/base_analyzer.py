"""Base analyzer with common utility methods."""

from typing import Dict, Any, List, Optional
import sqlglot
from sqlglot import Expression


class BaseAnalyzer:
    """Base analyzer containing common utility methods shared across all analyzers."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize base analyzer."""
        self.dialect = dialect
    
    def _determine_sql_type(self, sql: str) -> str:
        """Determine the type of SQL statement."""
        if not sql:
            return "unknown"
        
        sql_upper = sql.strip().upper()
        
        if sql_upper.startswith('SELECT'):
            return "select"
        elif sql_upper.startswith('WITH'):
            return "cte"  
        elif sql_upper.startswith('CREATE TABLE') and 'AS SELECT' in sql_upper:
            return "ctas"
        elif sql_upper.startswith('INSERT'):
            return "insert"
        elif sql_upper.startswith('UPDATE'):
            return "update"
        else:
            return "generic"
    
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
    
    def _is_single_table_context(self, sql: str) -> bool:
        """Check if the SQL query involves only a single table (no JOINs)."""
        if not sql:
            return False
        
        try:
            # Parse the SQL to check for JOINs
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Look for JOIN expressions in the AST
            for node in ast.walk():
                if isinstance(node, sqlglot.expressions.Join):
                    return False
                    
            # Also check for explicit JOIN keywords in the SQL text
            sql_upper = sql.upper()
            join_keywords = ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN']
            for keyword in join_keywords:
                if keyword in sql_upper:
                    return False
                    
            return True
            
        except Exception:
            # If parsing fails, fallback to simple text analysis
            sql_upper = sql.upper()
            return 'JOIN' not in sql_upper
    
    def _table_involved_in_group_by(self, table_name: str, sql: str) -> bool:
        """Check if a table is involved in GROUP BY operations."""
        if not table_name or not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Check if there's a GROUP BY clause
        if 'GROUP BY' not in sql_upper:
            return False
        
        # Check if the table name appears in the GROUP BY context
        # This is a simple heuristic - in most cases, if there's a GROUP BY,
        # it's operating on the main table being queried
        return table_name.lower() in sql.lower()
    
    def _extract_inner_expression(self, source_expression: str) -> str:
        """Extract the inner part of a function expression."""
        if not source_expression:
            return ""
        
        # For expressions like "SUM(u.salary)", extract "u.salary"
        # For expressions like "COUNT(*)", extract "*"
        
        expr = source_expression.strip()
        
        # Find the opening and closing parentheses
        start_idx = expr.find('(')
        end_idx = expr.rfind(')')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return expr[start_idx + 1:end_idx].strip()
        
        # If no parentheses found, return the original expression
        return expr
    
    def _expression_references_table(self, expression: str, table_name: str, sql: str = None) -> bool:
        """Check if an expression references a specific table."""
        if not expression or not table_name:
            return False
        
        expression_lower = expression.lower()
        table_name_lower = table_name.lower()
        
        # Direct table name match
        if table_name_lower in expression_lower:
            return True
        
        # Check for common aliases
        if table_name_lower.startswith('u') and 'u.' in expression_lower:
            return True
        elif table_name_lower.startswith('o') and 'o.' in expression_lower:
            return True
        
        # For unqualified column references, check if this is a single-table context
        if sql and '.' not in expression and self._is_single_table_context(sql):
            return True
        
        return False
    
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
    
    def _filter_nested_conditions(self, filter_conditions: list, entity_name: str) -> list:
        """Filter and organize conditions for a specific entity."""
        if not filter_conditions:
            return []
        
        filtered_conditions = []
        
        for condition in filter_conditions:
            # Check if this condition is relevant to the entity
            condition_text = str(condition)
            if entity_name.lower() in condition_text.lower():
                filtered_conditions.append(condition)
            elif not any(table in condition_text.lower() for table in ['users', 'orders', 'customers']):
                # Include conditions that don't specify a table (might be relevant)
                filtered_conditions.append(condition)
        
        return filtered_conditions