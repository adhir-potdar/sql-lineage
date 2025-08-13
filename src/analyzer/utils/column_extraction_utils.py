"""Column extraction utility functions."""

import re
from typing import Set, List, Dict, Optional
import sqlglot
from .sql_parsing_utils import build_alias_to_table_mapping, clean_source_expression


def extract_all_referenced_columns(sql: str, table_name: str, dialect: str = "trino") -> Set[str]:
    """Extract all columns referenced in SQL for a specific table."""
    referenced_columns = set()
    
    # Check if this is a UNION query - use specialized extraction
    if sql and 'UNION' in sql.upper():
        from .sql_parsing_utils import extract_columns_referenced_by_table_in_union
        union_columns = extract_columns_referenced_by_table_in_union(sql, table_name, dialect)
        return set(union_columns)
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Build alias to table mapping
        alias_to_table = build_alias_to_table_mapping(sql, dialect)
                
        # Find table alias for this table_name
        table_aliases = []
        for alias, actual_table in alias_to_table.items():
            if actual_table == table_name:
                table_aliases.append(alias)
        
        # If no alias found, use the table name itself
        if not table_aliases:
            table_aliases = [table_name]
        
        # Extract columns from various parts of the SQL
        # 1. SELECT clause columns (including columns inside functions)
        select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
        if select_stmt:
            for expr in select_stmt.expressions:
                # Find all column references in the expression (including inside functions)
                for column in expr.find_all(sqlglot.exp.Column):
                    table_part = str(column.table) if column.table else None
                    column_name = str(column.name) if column.name else None
                    
                    # Check if this column belongs to our table
                    if table_part in table_aliases:
                        referenced_columns.add(column_name)
                    elif not table_part and len(table_aliases) == 1:
                        # Check if this column is inside a subquery first
                        # Find the parent node to see if we're inside a subquery
                        parent_node = column.parent
                        while parent_node:
                            if isinstance(parent_node, sqlglot.exp.Subquery):
                                # This column is inside a subquery, don't attribute it to outer table
                                break
                            parent_node = parent_node.parent
                        else:
                            # If no table prefix and this is a single-table context AND not in subquery, assume it belongs to this table
                            referenced_columns.add(column_name)
        
        # 2. WHERE clause columns
        where_clause = select_stmt.find(sqlglot.exp.Where) if select_stmt else None
        if where_clause:
            for column in where_clause.find_all(sqlglot.exp.Column):
                table_part = str(column.table) if column.table else None
                column_name = str(column.name) if column.name else None
                
                if table_part in table_aliases:
                    referenced_columns.add(column_name)
                elif not table_part and len(table_aliases) == 1:
                    # Check if this column is inside a subquery first
                    parent_node = column.parent
                    while parent_node:
                        if isinstance(parent_node, sqlglot.exp.Subquery):
                            # This column is inside a subquery, don't attribute it to outer table
                            break
                        parent_node = parent_node.parent
                    else:
                        # If no table prefix and this is a single-table context AND not in subquery, assume it belongs to this table
                        referenced_columns.add(column_name)
                    
        # 3. JOIN condition columns
        joins = list(parsed.find_all(sqlglot.exp.Join)) if parsed else []
        for join in joins:
            if join.on:
                for column in join.on.find_all(sqlglot.exp.Column):
                    table_part = str(column.table) if column.table else None
                    column_name = str(column.name) if column.name else None
                    
                    if table_part in table_aliases:
                        referenced_columns.add(column_name)
        
        # 4. GROUP BY columns
        group_by = select_stmt.find(sqlglot.exp.Group) if select_stmt else None
        if group_by:
            for expr in group_by.expressions:
                if isinstance(expr, sqlglot.exp.Column):
                    table_part = str(expr.table) if expr.table else None  
                    column_name = str(expr.name) if expr.name else None
                    
                    if table_part in table_aliases:
                        referenced_columns.add(column_name)
        
        # 5. ORDER BY columns  
        order_by = select_stmt.find(sqlglot.exp.Order) if select_stmt else None
        if order_by:
            for ordered in order_by.expressions:
                if hasattr(ordered, 'this') and isinstance(ordered.this, sqlglot.exp.Column):
                    column = ordered.this
                    table_part = str(column.table) if column.table else None
                    column_name = str(column.name) if column.name else None
                    
                    if table_part in table_aliases:
                        referenced_columns.add(column_name)
                        
    except Exception:
        # If parsing fails, return empty set
        pass
        
    return referenced_columns


def extract_columns_from_joins(sql: str, table_name: str) -> Set[str]:
    """Extract JOIN columns for a specific table from SQL query."""
    join_columns = set()
    
    # Extract the ON clause using regex
    on_match = re.search(r'ON\s+(.+?)(?:\s*$|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT)', sql, re.IGNORECASE)
    if on_match:
        on_clause = on_match.group(1)
        
        # Extract column references from ON clause
        # Look for patterns like "u.id = o.user_id"
        column_refs = re.findall(r'(\w+)\.(\w+)', on_clause)
        
        for table_alias, column_name in column_refs:
            # Check if this column belongs to our table using proper alias mapping
            # This requires parsing the FROM clause to get actual table-to-alias mappings
            alias_to_table = build_alias_to_table_mapping(sql, "trino")
            
            # Check if the alias maps to our table
            if alias_to_table.get(table_alias) == table_name:
                join_columns.add(column_name)
            elif table_alias == table_name:  # Direct table name match
                join_columns.add(column_name)
    
    return join_columns


def extract_columns_from_where(sql: str, table_name: str, dialect: str = "trino") -> Set[str]:
    """Extract WHERE clause columns for a specific table."""
    where_columns = set()
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Build alias to table mapping
        alias_to_table = build_alias_to_table_mapping(sql, dialect)
        
        # Find table aliases for this table_name
        table_aliases = []
        for alias, actual_table in alias_to_table.items():
            if actual_table == table_name:
                table_aliases.append(alias)
        
        # If no alias found, use the table name itself
        if not table_aliases:
            table_aliases = [table_name]
        
        # Extract WHERE clause columns
        select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
        if select_stmt:
            where_clause = select_stmt.find(sqlglot.exp.Where)
            if where_clause:
                for column in where_clause.find_all(sqlglot.exp.Column):
                    table_part = str(column.table) if column.table else None
                    column_name = str(column.name) if column.name else None
                    
                    if table_part in table_aliases:
                        where_columns.add(column_name)
                        
    except Exception:
        # If parsing fails, return empty set
        pass
        
    return where_columns


def get_columns_from_expression(expression: str, dialect: str = "trino") -> Set[str]:
    """Extract column names from a SQL expression."""
    columns = set()
    
    try:
        # Parse as a simple expression
        parsed_expr = sqlglot.parse_one(f"SELECT {expression}", dialect=dialect)
        
        if parsed_expr:
            # Find all column references in the expression
            for column in parsed_expr.find_all(sqlglot.exp.Column):
                column_name = str(column.name) if column.name else None
                if column_name:
                    columns.add(column_name)
                    
    except Exception:
        # Fallback to regex-based extraction
        column_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)
        # Filter out SQL keywords
        sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET', 'AND', 'OR',
            'NOT', 'IN', 'EXISTS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'ALL', 'TRUE', 'FALSE',
            'NULL', 'IS', 'LIKE', 'BETWEEN', 'ASC', 'DESC'
        }
        for match in column_matches:
            if match.upper() not in sql_keywords:
                columns.add(match)
    
    return columns


def extract_aggregate_columns(sql: str, table_name: str, dialect: str = "trino") -> List[Dict]:
    """Extract aggregate columns (SUM, COUNT, etc.) for a specific table."""
    aggregate_columns = []
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Build alias to table mapping
        alias_to_table = build_alias_to_table_mapping(sql, dialect)
        
        # Find table aliases for this table_name
        table_aliases = []
        for alias, actual_table in alias_to_table.items():
            if actual_table == table_name:
                table_aliases.append(alias)
        
        # If no alias found, use the table name itself
        if not table_aliases:
            table_aliases = [table_name]
            
        # Check if this is the main aggregating table by checking GROUP BY columns
        is_main_table = _is_main_aggregating_table(sql, table_name, table_aliases, dialect)
        if not is_main_table:
            return aggregate_columns
            
        select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
        if select_stmt:
            for expr in select_stmt.expressions:
                if not isinstance(expr, sqlglot.exp.Column):
                    # Check if this is an aggregate function
                    raw_expr = str(expr)
                    
                    # Extract alias if present
                    alias = None
                    if hasattr(expr, 'alias') and expr.alias:
                        alias = str(expr.alias)
                    
                    # Check for aggregate functions
                    if any(agg_func in raw_expr.upper() for agg_func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                        column_name = alias if alias else raw_expr
                        
                        # Clean source expression to remove AS alias part
                        clean_source_expr = raw_expr
                        if ' AS ' in raw_expr.upper():
                            clean_source_expr = raw_expr.split(' AS ')[0].strip()
                        elif ' as ' in raw_expr:
                            clean_source_expr = raw_expr.split(' as ')[0].strip()
                        
                        # Determine function type
                        function_type = None
                        if 'COUNT(' in raw_expr.upper():
                            function_type = "COUNT"
                        elif 'SUM(' in raw_expr.upper():
                            function_type = "SUM"
                        elif 'AVG(' in raw_expr.upper():
                            function_type = "AVG"
                        elif 'MAX(' in raw_expr.upper():
                            function_type = "MAX"
                        elif 'MIN(' in raw_expr.upper():
                            function_type = "MIN"
                        
                        if function_type:
                            result_col = {
                                "name": column_name,
                                "upstream": [],
                                "type": "RESULT",
                                "transformation": {
                                    "source_expression": clean_source_expr,
                                    "transformation_type": "AGGREGATE",
                                    "function_type": function_type
                                }
                            }
                            aggregate_columns.append(result_col)
                            
    except Exception:
        # If parsing fails, return empty list
        pass
        
    return aggregate_columns


def extract_qualified_filter_columns(sql: str, table_name: str, dialect: str = "trino") -> List[Dict]:
    """Extract qualified filter columns from WHERE clause for a specific table."""
    filter_columns = []
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Build alias to table mapping
        alias_to_table = build_alias_to_table_mapping(sql, dialect)
        
        # Find table aliases for this table_name
        table_aliases = []
        for alias, actual_table in alias_to_table.items():
            if actual_table == table_name:
                table_aliases.append(alias)
        
        # If no alias found, use the table name itself
        if not table_aliases:
            table_aliases = [table_name]
        
        select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
        if select_stmt:
            where_clause = select_stmt.find(sqlglot.exp.Where)
            if where_clause:
                # Find comparison operations in WHERE clause
                for binary_op in where_clause.find_all(sqlglot.exp.Binary):
                    # Check if left side is a column from our table
                    if isinstance(binary_op.left, sqlglot.exp.Column):
                        column = binary_op.left
                        table_part = str(column.table) if column.table else None
                        column_name = str(column.name) if column.name else None
                        
                        if table_part in table_aliases and column_name:
                            filter_col = {
                                "name": column_name,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            filter_columns.append(filter_col)
                            
    except Exception:
        # If parsing fails, return empty list
        pass
        
    return filter_columns


def _is_main_aggregating_table(sql: str, table_name: str, table_aliases: List[str], dialect: str = "trino") -> bool:
    """Determine if a table is the main table being aggregated (appears in GROUP BY)."""
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Check GROUP BY clause
        select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
        if select_stmt:
            group_by = select_stmt.find(sqlglot.exp.Group)
            if group_by:
                for expr in group_by.expressions:
                    if isinstance(expr, sqlglot.exp.Column):
                        table_part = str(expr.table) if expr.table else None
                        
                        if table_part in table_aliases:
                            return True
                        
                        # If no table prefix and we only have one table alias, assume it's this table
                        if not table_part and len(table_aliases) == 1:
                            return True
                            
    except Exception:
        pass
        
    return False


def choose_best_column(columns: List[Dict]) -> Dict:
    """Choose the best column representation from duplicates using scoring system."""
    if not columns:
        return {}
    
    if len(columns) == 1:
        return columns[0]
    
    # Scoring criteria (higher score = better column)
    def score_column(col):
        score = 0
        
        # Prefer columns that don't have SQL expressions as names
        name = col.get('name', '')
        has_sql_expression = any(keyword in name.upper() for keyword in ['AS ', 'COUNT', 'AVG'])
        if not has_sql_expression:
            score += 20
            
        # Prefer columns with transformation details
        if 'transformation' in col:
            score += 20
            
        # Prefer DIRECT type over SOURCE
        if col.get('type') == 'DIRECT':
            score += 10
        elif col.get('type') == 'SOURCE':
            score += 5
            
        # Prefer shorter names (likely cleaner)
        if len(name) < 20:
            score += 5
            
        return score
    
    # Find the column with the highest score
    best_column = max(columns, key=score_column)
    return best_column


def process_select_expression(expr, sql: str, dialect: str = "trino") -> Dict:
    """Process a SELECT expression and extract column information."""
    result = {
        "name": "",
        "upstream": [],
        "type": "SOURCE"
    }
    
    try:
        expr_str = str(expr)
        
        # Handle different expression types
        if isinstance(expr, sqlglot.exp.Column):
            # Simple column reference
            result["name"] = str(expr.name) if expr.name else expr_str
            result["upstream"] = [f"QUERY_RESULT.{result['name']}"]
            
        elif hasattr(expr, 'alias') and expr.alias:
            # Expression with alias
            result["name"] = str(expr.alias)
            result["upstream"] = [f"QUERY_RESULT.{result['name']}"]
            
            # Check if it's an aggregate function
            from .aggregate_utils import is_aggregate_function, extract_function_type
            if is_aggregate_function(expr_str):
                source_expr = str(expr.this) if hasattr(expr, 'this') else expr_str
                func_type = extract_function_type(expr_str)
                
                result["type"] = "DIRECT"
                result["transformation"] = {
                    "source_expression": clean_source_expression(source_expr),
                    "transformation_type": "AGGREGATE", 
                    "function_type": func_type
                }
        else:
            # Other expressions (literals, functions, etc.)
            result["name"] = expr_str
            result["upstream"] = [f"QUERY_RESULT.{expr_str}"]
            
    except Exception:
        # Fallback
        result["name"] = str(expr)
        result["upstream"] = [f"QUERY_RESULT.{result['name']}"]
    
    return result


