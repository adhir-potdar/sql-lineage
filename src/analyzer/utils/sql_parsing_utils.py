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
        
    alias_match = re.search(r'\s+(?:as|AS)\s+([^\s,)]+)', expression)
    return alias_match.group(1) if alias_match else None


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