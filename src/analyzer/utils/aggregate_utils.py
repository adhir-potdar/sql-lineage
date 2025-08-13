"""
Utility functions for handling aggregate functions in SQL queries.

This module provides utilities for:
- Detecting aggregate functions
- Extracting function types and aliases
- Processing aggregate expressions
- Determining aggregate relationships with tables
"""

import re
from typing import List, Dict, Any, Optional
from .sql_parsing_utils import clean_source_expression


def is_aggregate_function(expression: str) -> bool:
    """Check if expression contains an aggregate function."""
    aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT']
    pattern = r'\b(' + '|'.join(aggregate_functions) + r')\s*\('
    return bool(re.search(pattern, expression, re.IGNORECASE))


def query_has_aggregates(sql: str) -> bool:
    """Check if SQL query contains aggregate functions or GROUP BY."""
    sql_upper = sql.upper()
    aggregate_functions = ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(']
    has_aggregates = any(func in sql_upper for func in aggregate_functions)
    has_group_by = 'GROUP BY' in sql_upper
    return has_aggregates or has_group_by


def extract_alias_from_expression(expression: str) -> str:
    """Extract alias from expression like 'COUNT(*) as login_count'."""
    from .regex_patterns import extract_alias_from_expression
    return extract_alias_from_expression(expression)




def is_aggregate_function_for_table(column_expr: str, table_name: str, sql: str = None) -> bool:
    """Check if an aggregate function expression is relevant to a specific table."""
    if not column_expr or not table_name:
        return False
    
    if not is_aggregate_function(column_expr):
        return False
    
    # Handle aggregate functions like COUNT(*), AVG(u.salary), SUM(users.amount)
    column_expr_lower = column_expr.lower()
    
    # Check if the expression contains explicit table references first
    if table_name.lower() in column_expr_lower:
        return True
    
    # Check for table aliases dynamically by analyzing the SQL if SQL is provided
    if sql and column_expression_belongs_to_table(column_expr, table_name, sql):
        return True
    
    # COUNT(*) is only relevant to the main grouped table
    # Only assign COUNT(*) to the table that appears in GROUP BY
    if sql and 'count(*)' in column_expr_lower and is_main_aggregating_table(sql, table_name):
        return True
    
    return False


def column_expression_belongs_to_table(column_expr: str, table_name: str, sql: str) -> bool:
    """Check if a column expression belongs to a specific table by analyzing aliases in SQL."""
    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql, dialect='trino')
        
        # Build alias to table mapping
        alias_to_table = {}
        tables = list(parsed.find_all(sqlglot.exp.Table))
        for table in tables:
            if table.alias:
                alias_to_table[str(table.alias)] = str(table.name)
                
        # Find aliases for this table_name
        table_aliases = []
        for alias, actual_table in alias_to_table.items():
            if actual_table == table_name:
                table_aliases.append(alias)
        
        # Check if any of the table aliases appear in the column expression
        column_expr_lower = column_expr.lower()
        for alias in table_aliases:
            if f'{alias}.' in column_expr_lower:
                return True
                
        # Simple check: if table name appears in the expression
        if f"{table_name}." in column_expr:
            return True
                
        return False
        
    except Exception:
        # If parsing fails, fallback to simple heuristics
        # Check if table name starts with same letter as expression prefix
        if '.' in column_expr:
            prefix = column_expr.split('.')[0].lower()
            return table_name.lower().startswith(prefix)
        return False


def extract_aggregate_result_columns(sql: str, table_name: str) -> List[Dict]:
    """Extract aggregate result columns for a specific table."""
    result_columns = []
    
    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql, dialect='trino')
        
        # Find SELECT statement
        select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
        if not select_stmt:
            return result_columns
        
        # Process each SELECT expression
        for expr in select_stmt.expressions:
            expr_str = str(expr)
            
            # Check if this is an aggregate function
            if is_aggregate_function(expr_str):
                # Extract alias or use expression as name
                if hasattr(expr, 'alias') and expr.alias:
                    col_name = str(expr.alias)
                    source_expr = str(expr.this) if hasattr(expr, 'this') else expr_str
                else:
                    col_name = expr_str
                    source_expr = expr_str
                
                # Determine if this aggregate relates to the table
                if column_expression_belongs_to_table(expr_str, table_name, sql):
                    func_type = extract_function_type(expr_str)
                    
                    column_info = {
                        "name": col_name,
                        "upstream": [f"{table_name}.{col_name}"],
                        "type": "DIRECT",
                        "transformation": {
                            "source_expression": clean_source_expression(source_expr),
                            "transformation_type": "AGGREGATE",
                            "function_type": func_type
                        }
                    }
                    result_columns.append(column_info)
    
    except Exception:
        # Fallback to regex-based parsing
        pass
    
    return result_columns


def extract_aggregate_source_columns(sql: str, table_name: str) -> List[Dict]:
    """Extract source columns that are referenced in aggregate functions."""
    source_columns = []
    
    try:
        # Extract columns referenced inside aggregate functions
        aggregate_pattern = r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([^)]+)\s*\)'
        matches = re.findall(aggregate_pattern, sql, re.IGNORECASE)
        
        for match in matches:
            col_ref = match.strip()
            if col_ref == '*':
                continue  # COUNT(*) doesn't reference specific columns
            
            # Check if column belongs to this table
            if f"{table_name}." in col_ref or '.' not in col_ref:
                clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                
                column_info = {
                    "name": clean_col,
                    "upstream": [],
                    "type": "SOURCE"
                }
                
                # Avoid duplicates
                if not any(col["name"] == clean_col for col in source_columns):
                    source_columns.append(column_info)
    
    except Exception:
        pass
    
    return source_columns


def is_main_aggregating_table(sql: str, table_name: str) -> bool:
    """Check if a table is the main table being aggregated in the query."""
    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql, dialect='trino')
        
        # Find FROM clause to identify main table
        from_clause = parsed.find(sqlglot.exp.From)
        if from_clause:
            main_table = from_clause.this
            if hasattr(main_table, 'name') and str(main_table.name) == table_name:
                return True
            elif hasattr(main_table, 'alias') and str(main_table.alias) == table_name:
                return True
    
    except Exception:
        # Fallback: simple text search
        return table_name.lower() in sql.lower()
    
    return False


def extract_upstream_from_aggregate(aggregate_expr: str, sql: str) -> List[str]:
    """Extract upstream column references from aggregate function."""
    upstream = []
    
    # Pattern to match columns inside aggregate functions: FUNC(table.column) or FUNC(column)
    func_pattern = r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([^)]+)\s*\)'
    matches = re.findall(func_pattern, aggregate_expr, re.IGNORECASE)
    
    for match in matches:
        col_ref = match.strip()
        if col_ref == '*':
            # COUNT(*) - doesn't reference specific columns
            continue
        elif '.' in col_ref:
            # Qualified column reference (table.column)
            upstream.append(col_ref)
        else:
            # Unqualified column - try to infer table from SQL context
            # For now, just use the column name
            upstream.append(col_ref)
    
    return upstream if upstream else [f"QUERY_RESULT.{aggregate_expr}"]