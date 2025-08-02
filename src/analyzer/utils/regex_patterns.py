"""Centralized regex patterns for SQL parsing."""

import re
from typing import Dict, Pattern, List, Optional, Tuple


# Compiled regex patterns for better performance
class SQLPatterns:
    """Centralized SQL regex patterns with compiled objects."""
    
    # SQL clause extraction patterns
    SELECT_CLAUSE = re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE | re.DOTALL)
    WHERE_CLAUSE = re.compile(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*$)', re.IGNORECASE | re.DOTALL)
    FROM_CLAUSE = re.compile(r'FROM\s+(\w+)', re.IGNORECASE)
    GROUP_BY_CLAUSE = re.compile(r'GROUP\s+BY\s+([^)]+?)(?:\s+(?:HAVING|ORDER|LIMIT)|$)', re.IGNORECASE)
    ORDER_BY_CLAUSE = re.compile(r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|\s*$)', re.IGNORECASE | re.DOTALL)
    HAVING_CLAUSE = re.compile(r'HAVING\s+(.*?)(?:\s+ORDER\s+BY|\s+LIMIT|\s*$)', re.IGNORECASE | re.DOTALL)
    
    # JOIN patterns
    JOIN_ON_CLAUSE = re.compile(r'ON\s+(.+?)(?:\s*$|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT)', re.IGNORECASE)
    COLUMN_REFERENCES = re.compile(r'(\w+)\.(\w+)', re.IGNORECASE)
    
    # Function and expression patterns
    ALIAS_PATTERN = re.compile(r'\s+(?:as|AS)\s+([^\s,)]+)')
    FUNCTION_PATTERN = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT|ROW_NUMBER|RANK|DENSE_RANK|LEAD|LAG)\s*\(', re.IGNORECASE)
    FUNCTION_WITH_ARGS = re.compile(r'^[A-Z_]+\s*\(\s*(.+?)\s*\)$', re.IGNORECASE)
    AGGREGATE_FUNCTIONS = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', re.IGNORECASE)
    WINDOW_FUNCTIONS = re.compile(r'\b(ROW_NUMBER|RANK|DENSE_RANK|LEAD|LAG|FIRST_VALUE|LAST_VALUE)\s*\(', re.IGNORECASE)
    
    # Column and table name patterns
    QUALIFIED_COLUMN = re.compile(r'(\w+)\.(\w+)', re.IGNORECASE)
    TABLE_ALIAS = re.compile(r'(\w+)\s+(?:AS\s+)?(\w+)', re.IGNORECASE)
    QUOTED_IDENTIFIER = re.compile(r'["\'\`]([^"\'`]+)["\'\`]')
    
    # Filter condition patterns
    SIMPLE_CONDITION = re.compile(r'(\w+)\s*(>|<|>=|<=|=|!=)\s*([^\s]+)', re.IGNORECASE)
    QUALIFIED_CONDITION = re.compile(r'(\w+\.\w+)\s*(>|<|>=|<=|=|!=)\s*([^\s]+)', re.IGNORECASE)
    IN_CONDITION = re.compile(r'(\w+(?:\.\w+)?)\s+IN\s*\((.*?)\)', re.IGNORECASE)
    BETWEEN_CONDITION = re.compile(r'(\w+(?:\.\w+)?)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)', re.IGNORECASE)
    LIKE_CONDITION = re.compile(r'(\w+(?:\.\w+)?)\s+LIKE\s+(.+)', re.IGNORECASE)
    
    # CTAS patterns
    CTAS_PATTERN = re.compile(r'CREATE\s+TABLE\s+.+\s+AS\s+SELECT', re.IGNORECASE)
    
    # CTE patterns
    WITH_CLAUSE = re.compile(r'WITH\s+(.+?)\s+SELECT', re.IGNORECASE | re.DOTALL)
    CTE_DEFINITION = re.compile(r'(\w+)\s+AS\s*\((.+?)\)', re.IGNORECASE | re.DOTALL)
    
    # SQL keywords for filtering
    SQL_KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
        'ON', 'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'OFFSET', 'AND', 'OR',
        'NOT', 'IN', 'EXISTS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'AS',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'ALL', 'TRUE', 'FALSE',
        'NULL', 'IS', 'LIKE', 'BETWEEN', 'ASC', 'DESC', 'WITH', 'UNION', 'EXCEPT',
        'INTERSECT', 'CREATE', 'TABLE', 'INSERT', 'UPDATE', 'DELETE', 'DROP'
    }


def extract_select_clause(sql: str) -> Optional[str]:
    """Extract SELECT clause from SQL."""
    match = SQLPatterns.SELECT_CLAUSE.search(sql)
    return match.group(1).strip() if match else None


def extract_where_clause(sql: str) -> Optional[str]:
    """Extract WHERE clause from SQL."""
    match = SQLPatterns.WHERE_CLAUSE.search(sql)
    return match.group(1).strip() if match else None


def extract_from_table(sql: str) -> Optional[str]:
    """Extract main table from FROM clause."""
    match = SQLPatterns.FROM_CLAUSE.search(sql)
    return match.group(1) if match else None


def extract_group_by_columns(sql: str) -> List[str]:
    """Extract GROUP BY columns from SQL."""
    match = SQLPatterns.GROUP_BY_CLAUSE.search(sql)
    if match:
        columns_str = match.group(1)
        return [col.strip() for col in columns_str.split(',')]
    return []


def extract_order_by_columns(sql: str) -> List[str]:
    """Extract ORDER BY columns from SQL."""
    match = SQLPatterns.ORDER_BY_CLAUSE.search(sql)
    if match:
        columns_str = match.group(1)
        return [col.strip() for col in columns_str.split(',')]
    return []


def extract_join_conditions(sql: str) -> List[Tuple[str, str]]:
    """Extract JOIN ON conditions as (table.column, table.column) pairs."""
    conditions = []
    on_match = SQLPatterns.JOIN_ON_CLAUSE.search(sql)
    
    if on_match:
        on_clause = on_match.group(1)
        column_refs = SQLPatterns.COLUMN_REFERENCES.findall(on_clause)
        conditions.extend(column_refs)
    
    return conditions


def extract_alias_from_expression(expression: str) -> Optional[str]:
    """Extract alias from expression like 'COUNT(*) as login_count'."""
    match = SQLPatterns.ALIAS_PATTERN.search(expression)
    return match.group(1) if match else None


def extract_function_name(expression: str) -> Optional[str]:
    """Extract function name from expression."""
    match = SQLPatterns.FUNCTION_PATTERN.search(expression)
    return match.group(1).upper() if match else None


def is_aggregate_function(expression: str) -> bool:
    """Check if expression contains aggregate functions."""
    return bool(SQLPatterns.AGGREGATE_FUNCTIONS.search(expression))


def is_window_function(expression: str) -> bool:
    """Check if expression contains window functions."""
    return bool(SQLPatterns.WINDOW_FUNCTIONS.search(expression))


def extract_qualified_columns(text: str) -> List[Tuple[str, str]]:
    """Extract qualified column references as (table, column) pairs."""
    return SQLPatterns.QUALIFIED_COLUMN.findall(text)


def extract_table_aliases(from_clause: str) -> Dict[str, str]:
    """Extract table aliases from FROM clause."""
    aliases = {}
    # Simple pattern for "table_name alias" or "table_name AS alias"
    matches = SQLPatterns.TABLE_ALIAS.findall(from_clause)
    
    for table, alias in matches:
        aliases[alias.lower()] = table
    
    return aliases


def clean_identifier(identifier: str) -> str:
    """Remove quotes from SQL identifiers."""
    match = SQLPatterns.QUOTED_IDENTIFIER.search(identifier)
    return match.group(1) if match else identifier.strip()


def extract_filter_conditions(where_clause: str) -> List[Dict[str, str]]:
    """Extract filter conditions from WHERE clause."""
    conditions = []
    
    # Simple conditions (column operator value)
    simple_matches = SQLPatterns.SIMPLE_CONDITION.findall(where_clause)
    for column, operator, value in simple_matches:
        conditions.append({
            "column": column,
            "operator": operator,
            "value": clean_value(value)
        })
    
    # Qualified conditions (table.column operator value)
    qualified_matches = SQLPatterns.QUALIFIED_CONDITION.findall(where_clause)
    for column, operator, value in qualified_matches:
        conditions.append({
            "column": column,
            "operator": operator,
            "value": clean_value(value)
        })
    
    # IN conditions
    in_matches = SQLPatterns.IN_CONDITION.findall(where_clause)
    for column, values in in_matches:
        conditions.append({
            "column": column,
            "operator": "IN",
            "value": values.strip()
        })
    
    # BETWEEN conditions
    between_matches = SQLPatterns.BETWEEN_CONDITION.findall(where_clause)
    for column, start_val, end_val in between_matches:
        conditions.append({
            "column": column,
            "operator": "BETWEEN",
            "value": f"{start_val.strip()} AND {end_val.strip()}"
        })
    
    # LIKE conditions
    like_matches = SQLPatterns.LIKE_CONDITION.findall(where_clause)
    for column, pattern in like_matches:
        conditions.append({
            "column": column,
            "operator": "LIKE",
            "value": pattern.strip()
        })
    
    return conditions


def clean_value(value: str) -> str:
    """Clean up filter values by removing quotes."""
    return value.strip().strip("'").strip('"')


def is_ctas_query(sql: str) -> bool:
    """Check if SQL is a CREATE TABLE AS SELECT query."""
    return bool(SQLPatterns.CTAS_PATTERN.search(sql))


def extract_cte_definitions(sql: str) -> Dict[str, str]:
    """Extract CTE definitions from WITH clause."""
    ctes = {}
    with_match = SQLPatterns.WITH_CLAUSE.search(sql)
    
    if with_match:
        with_content = with_match.group(1)
        cte_matches = SQLPatterns.CTE_DEFINITION.findall(with_content)
        
        for cte_name, cte_sql in cte_matches:
            ctes[cte_name.strip()] = cte_sql.strip()
    
    return ctes


def is_sql_keyword(word: str) -> bool:
    """Check if a word is a SQL keyword."""
    return word.upper() in SQLPatterns.SQL_KEYWORDS


def extract_column_names_from_expression(expression: str) -> List[str]:
    """Extract column names from expression, filtering out SQL keywords."""
    # Simple word extraction
    words = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)
    
    # Filter out SQL keywords
    columns = []
    for word in words:
        if not is_sql_keyword(word):
            columns.append(word)
    
    return columns


def split_qualified_name(qualified_name: str) -> Dict[str, Optional[str]]:
    """Split qualified name into components."""
    parts = qualified_name.split('.')
    
    if len(parts) == 3:
        return {"schema": parts[0], "table": parts[1], "column": parts[2]}
    elif len(parts) == 2:
        return {"schema": None, "table": parts[0], "column": parts[1]}
    else:
        return {"schema": None, "table": None, "column": parts[0]}


def normalize_sql_formatting(sql: str) -> str:
    """Normalize SQL formatting for consistent parsing."""
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql.strip())
    
    # Ensure consistent spacing around keywords
    sql = re.sub(r'\b(SELECT|FROM|WHERE|JOIN|ON|GROUP BY|ORDER BY|HAVING)\b', r' \1 ', sql, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    sql = re.sub(r'\s+', ' ', sql)
    
    return sql.strip()