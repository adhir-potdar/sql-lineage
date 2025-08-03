"""SQL parsing utility functions."""

import re
from typing import Dict, Optional, List
import sqlglot


def normalize_table_name(table_name: str) -> str:
    """Normalize table name by removing quotes and standardizing format."""
    if not table_name:
        return ""
    
    # Remove quotes and extra whitespace
    normalized = table_name.strip().strip('"').strip("'").strip('`')
    return normalized.lower()


def clean_column_name(column_name: str) -> str:
    """Clean column name by removing quotes and normalizing format."""
    if not column_name:
        return ""
    
    # Remove quotes and extra whitespace
    cleaned = column_name.strip().strip('"').strip("'").strip('`')
    return cleaned


def parse_qualified_name(qualified_name: str) -> Dict[str, Optional[str]]:
    """
    Parse a qualified name into its components.
    
    Args:
        qualified_name: Name like 'schema.table.column' or 'table.column' or 'column'
        
    Returns:
        Dict with keys: schema, table, column
    """
    if not qualified_name:
        return {"schema": None, "table": None, "column": None}
    
    parts = qualified_name.split('.')
    
    if len(parts) == 3:
        return {
            "schema": clean_column_name(parts[0]),
            "table": clean_column_name(parts[1]), 
            "column": clean_column_name(parts[2])
        }
    elif len(parts) == 2:
        return {
            "schema": None,
            "table": clean_column_name(parts[0]),
            "column": clean_column_name(parts[1])
        }
    else:
        return {
            "schema": None, 
            "table": None,
            "column": clean_column_name(parts[0])
        }


def handle_schema_prefix(table_name: str, schema: Optional[str] = None) -> str:
    """Handle schema prefix for table names."""
    if not table_name:
        return ""
    
    # If table already has schema prefix, return as-is
    if '.' in table_name:
        return normalize_table_name(table_name)
    
    # Add schema prefix if provided
    if schema:
        return f"{normalize_table_name(schema)}.{normalize_table_name(table_name)}"
    
    return normalize_table_name(table_name)


def build_alias_to_table_mapping(sql: str, dialect: str = "trino") -> Dict[str, str]:
    """Build a mapping from table aliases to actual table names by parsing SQL."""
    alias_to_table = {}
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Find all table references with aliases
        tables = list(parsed.find_all(sqlglot.exp.Table))
        for table in tables:
            table_name = table.name
            if table.alias:
                alias = str(table.alias).lower()
                alias_to_table[alias] = table_name
                
    except Exception:
        # If parsing fails, return empty mapping to avoid incorrect assumptions
        pass
        
    return alias_to_table


def extract_alias_from_expression(expression: str) -> Optional[str]:
    """Extract alias from expression like 'COUNT(*) as login_count'."""
    if not expression:
        return None
    
    # For subqueries, we need to match the rightmost AS clause to get the outer alias
    # Use findall to get all matches and take the last one
    alias_matches = re.findall(r'\s+(?:as|AS)\s+([^\s,)]+)', expression)
    return alias_matches[-1] if alias_matches else None


def extract_function_type(expression: str) -> str:
    """Extract function type from aggregate expression."""
    if not expression:
        return "UNKNOWN"
        
    func_match = re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|ROW_NUMBER|RANK|DENSE_RANK)\s*\(', expression, re.IGNORECASE)
    return func_match.group(1).upper() if func_match else "UNKNOWN"


def is_column_from_table(column_name: str, table_name: str, sql: str = None, dialect: str = "trino") -> bool:
    """Check if a column belongs to a specific table using proper SQL parsing."""
    if not column_name or not table_name:
        return False
    
    # Handle qualified column names (e.g., "users.active", "u.salary")
    if '.' in column_name:
        parsed = parse_qualified_name(column_name)
        column_table = parsed["table"]
        
        if not column_table:
            return False
            
        # Direct table name match
        if column_table == table_name:
            return True
            
        # Check if it's an alias match using SQL context
        if sql:
            alias_mapping = build_alias_to_table_mapping(sql, dialect)
            actual_table = alias_mapping.get(column_table.lower())
            if actual_table and actual_table == table_name:
                return True
    
    # For unqualified column names, we can't definitively say without schema info
    # Default to True for backward compatibility
    return not ('.' in column_name)


def clean_source_expression(expression: str) -> str:
    """Clean source expression by removing AS alias part."""
    if not expression:
        return ""
        
    clean_expr = expression
    if ' AS ' in expression.upper():
        clean_expr = expression.split(' AS ')[0].strip()
    elif ' as ' in expression:
        clean_expr = expression.split(' as ')[0].strip()
        
    return clean_expr


def extract_table_references_from_sql(sql: str, dialect: str = "trino") -> List[Dict[str, str]]:
    """Extract all table references from SQL including aliases."""
    tables = []
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Find all table references
        table_nodes = list(parsed.find_all(sqlglot.exp.Table))
        for table in table_nodes:
            table_info = {
                "name": table.name,
                "alias": str(table.alias) if table.alias else None,
                "schema": str(table.db) if table.db else None
            }
            tables.append(table_info)
            
    except Exception:
        # If parsing fails, return empty list
        pass
        
    return tables


def get_union_columns_for_table(sql: str, table_name: str, dialect: str = "trino") -> List[str]:
    """Get the columns that a specific table contributes to a UNION query."""
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Find the UNION expression
        union_expr = parsed.find(sqlglot.exp.Union)
        if not union_expr:
            return []
        
        # Find the SELECT statement that references this table
        select_stmts = []
        collect_union_selects_helper(union_expr, select_stmts)
        
        for select_stmt in select_stmts:
            # Check if this SELECT uses the table
            tables = list(select_stmt.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.name == table_name:
                    # Extract column names from this SELECT
                    columns = []
                    for expr in select_stmt.expressions:
                        if hasattr(expr, 'alias') and expr.alias:
                            # Use the full expression with alias for clarity
                            columns.append(f"{str(expr.this)} as {str(expr.alias)}")
                        elif isinstance(expr, sqlglot.exp.Column):
                            # Use just the column name
                            columns.append(str(expr.name) if expr.name else str(expr))
                        else:
                            # Handle literals and expressions
                            expr_str = str(expr)
                            columns.append(expr_str)
                    return columns
        
        return []
    except Exception:
        # Fallback: return standard UNION column names
        return ['type', 'id', 'identifier']


def collect_union_selects_helper(union_expr, select_stmts: List) -> None:
    """Helper to collect all SELECT statements from a UNION."""
    try:
        if hasattr(union_expr, 'left') and union_expr.left:
            if isinstance(union_expr.left, sqlglot.exp.Union):
                collect_union_selects_helper(union_expr.left, select_stmts)
            elif isinstance(union_expr.left, sqlglot.exp.Select):
                select_stmts.append(union_expr.left)
        
        if hasattr(union_expr, 'right') and union_expr.right:
            if isinstance(union_expr.right, sqlglot.exp.Union):
                collect_union_selects_helper(union_expr.right, select_stmts)
            elif isinstance(union_expr.right, sqlglot.exp.Select):
                select_stmts.append(union_expr.right)
    except Exception:
        pass


def infer_query_result_columns_simple(sql: str, select_columns: List[Dict], dialect: str = "trino") -> List[Dict]:
    """
    Simple helper to infer query result columns similar to original analyzer-bkup.py approach.
    Returns columns with their names to check for table prefixes.
    """
    result_columns = []
    
    for sel_col in select_columns:
        raw_expression = sel_col.get('raw_expression', '')
        column_name = sel_col.get('column_name', raw_expression)
        
        # Try to extract a clean name, preferring aliases
        clean_name = extract_clean_column_name(raw_expression, column_name)
        
        column_info = {
            "name": clean_name,
            "upstream": [f"QUERY_RESULT.{clean_name}"],
            "type": "SOURCE"
        }
        result_columns.append(column_info)
    
    return result_columns


def extract_clean_column_name(raw_expression: str, fallback_name: str) -> str:
    """Extract clean column name from raw expression, preferring aliases."""
    if not raw_expression:
        return fallback_name or ""
    
    # Check for subqueries first - they should use their alias
    if '(SELECT' in raw_expression.upper():
        # For subqueries, extract the rightmost alias (outer alias, not internal table alias)
        alias_matches = re.findall(r'\s+(?:as|AS)\s+([^\s,)]+)', raw_expression)
        if alias_matches:
            return alias_matches[-1].strip()  # Take the last (rightmost) alias
        # If no alias found, return fallback
        return fallback_name or "subquery"
    
    # Check for alias (AS keyword) for non-subqueries
    alias_match = re.search(r'\s+(?:as|AS)\s+([^\s,)]+)', raw_expression)
    if alias_match:
        return alias_match.group(1).strip()
    
    # If no alias, try to extract clean column name
    # Remove table prefixes (e.g., "u.name" -> "name")
    if '.' in raw_expression and not any(func in raw_expression.upper() for func in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']):
        parts = raw_expression.split('.')
        if len(parts) == 2:
            return parts[1].strip()
    
    # For simple expressions, return as-is
    return raw_expression.strip()


def infer_query_result_columns_simple_fallback(sql: str) -> List[Dict]:
    """Simple fallback method using regex parsing."""
    result_columns = []
    
    # Try to extract SELECT columns from SQL
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1).strip()
        
        # Handle simple cases like "SELECT name, email FROM table_name"
        # Split by comma and clean up
        columns = [col.strip() for col in select_clause.split(',')]
        
        for col in columns:
            original_col = col.strip()  # Keep original column reference
            
            # Extract alias name from "expression AS alias" -> "alias"
            col_name = original_col
            if ' AS ' in col.upper():
                # Split on AS and take the alias part (after AS)
                parts = col.upper().split(' AS ')
                if len(parts) > 1:
                    col_name = parts[-1].strip()
            elif ' as ' in col:
                # Handle lowercase as
                parts = col.split(' as ')
                if len(parts) > 1:
                    col_name = parts[-1].strip()
            else:
                # No alias - use column name or expression
                # Clean table prefixes like "u.name" -> "name"
                if '.' in col_name and not any(func in col_name.upper() for func in ['COUNT', 'SUM', 'AVG']):
                    col_name = col_name.split('.')[-1]
            
            column_info = {
                "name": col_name,
                "upstream": [f"QUERY_RESULT.{col_name}"],
                "type": "SOURCE"
            }
            result_columns.append(column_info)
    
    return result_columns


def extract_table_columns_from_sql(sql: str, table_name: str, dialect: str = "trino") -> set:
    """Extract columns that are referenced for a specific table from SQL."""
    columns = set()
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Find all column references
        for column in parsed.find_all(sqlglot.exp.Column):
            # Check if column belongs to this table
            if column.table:
                table_ref = str(column.table)
                column_name = str(column.name)
                
                # Direct table name match or alias resolution
                if table_ref == table_name:
                    columns.add(column_name)
                else:
                    # Check if it's an alias
                    alias_mapping = build_alias_to_table_mapping(sql, dialect)
                    actual_table = alias_mapping.get(table_ref.lower())
                    if actual_table == table_name:
                        columns.add(column_name)
    
    except Exception:
        # Fallback to regex-based extraction
        pattern = rf'\b{re.escape(table_name)}\.(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        columns.update(matches)
    
    return columns


def extract_join_columns_from_sql(sql: str, table_name: str, dialect: str = "trino") -> set:
    """Extract columns used in JOIN conditions for a specific table."""
    columns = set()
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Find all JOIN expressions
        for join in parsed.find_all(sqlglot.exp.Join):
            if join.on:
                # Extract column references from JOIN condition
                for column in join.on.find_all(sqlglot.exp.Column):
                    if column.table:
                        table_ref = str(column.table)
                        column_name = str(column.name)
                        
                        # Check if this column belongs to our table
                        if table_ref == table_name:
                            columns.add(column_name)
                        else:
                            # Check if it's an alias
                            alias_mapping = build_alias_to_table_mapping(sql, dialect)
                            actual_table = alias_mapping.get(table_ref.lower())
                            if actual_table == table_name:
                                columns.add(column_name)
    
    except Exception:
        # Fallback: regex-based extraction from JOIN conditions
        join_pattern = r'JOIN\s+\w+\s+\w+\s+ON\s+([^)]+?)(?:\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|$))'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        for join_condition in join_matches:
            # Look for table.column references
            column_pattern = rf'\b{re.escape(table_name)}\.(\w+)'
            column_matches = re.findall(column_pattern, join_condition, re.IGNORECASE)
            columns.update(column_matches)
    
    return columns


def extract_all_referenced_columns(sql: str, table_name: str, dialect: str = "trino") -> set:
    """Extract all columns referenced for a specific table from SQL."""
    all_columns = set()
    
    # Get columns from various parts of the query
    all_columns.update(extract_table_columns_from_sql(sql, table_name, dialect))
    all_columns.update(extract_join_columns_from_sql(sql, table_name, dialect))
    
    # Additional extraction from WHERE, GROUP BY, ORDER BY, HAVING clauses
    try:
        # Simple regex-based extraction for WHERE, GROUP BY, etc.
        patterns = [
            r'WHERE\s+([^)]+?)(?:\s+(?:GROUP|ORDER|LIMIT|$))',
            r'GROUP\s+BY\s+([^)]+?)(?:\s+(?:HAVING|ORDER|LIMIT|$))',
            r'ORDER\s+BY\s+([^)]+?)(?:\s+(?:LIMIT|$))',
            r'HAVING\s+([^)]+?)(?:\s+(?:ORDER|LIMIT|$))'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Look for table.column references
                column_pattern = rf'\b{re.escape(table_name)}\.(\w+)'
                column_matches = re.findall(column_pattern, match, re.IGNORECASE)
                all_columns.update(column_matches)
    
    except Exception:
        pass
    
    return all_columns


def extract_qualified_filter_columns(sql: str, table_name: str, dialect: str = "trino") -> List[Dict]:
    """Extract qualified filter columns for a specific table."""
    result_columns = []
    
    try:
        # Find columns referenced in WHERE clause and other filter conditions
        referenced_columns = extract_all_referenced_columns(sql, table_name, dialect)
        
        # Create qualified column names
        qualified_columns = set()
        for column_name in referenced_columns:
            qualified_name = f"{table_name}.{column_name}"
            qualified_columns.add(qualified_name)
        
        # Create column info for each qualified column
        for qualified_col in qualified_columns:
            column_info = {
                "name": qualified_col,
                "upstream": [],
                "type": "SOURCE"
            }
            result_columns.append(column_info)
            
    except Exception:
        # If parsing fails, return empty list
        pass
        
    return result_columns