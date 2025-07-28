"""Input validation utilities."""

from typing import Optional


def validate_sql_input(sql: str) -> Optional[str]:
    """
    Validate SQL input.
    
    Args:
        sql: SQL string to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    if not sql:
        return "SQL query cannot be empty"
    
    if not isinstance(sql, str):
        return "SQL query must be a string"
    
    if len(sql.strip()) == 0:
        return "SQL query cannot be empty or whitespace only"
    
    if len(sql) > 1_000_000:  # 1MB limit
        return "SQL query is too large (max 1MB)"
    
    return None


def validate_dialect(dialect: str) -> Optional[str]:
    """
    Validate SQL dialect.
    
    Args:
        dialect: SQL dialect string
        
    Returns:
        Error message if invalid, None if valid
    """
    if not dialect:
        return "Dialect cannot be empty"
    
    if not isinstance(dialect, str):
        return "Dialect must be a string"
    
    # Common supported dialects
    supported_dialects = {
        "trino", "presto", "spark", "hive", "mysql", "postgres", "postgresql",
        "bigquery", "snowflake", "redshift", "sqlite", "oracle", "mssql",
        "clickhouse", "databricks", "duckdb"
    }
    
    if dialect.lower() not in supported_dialects:
        return f"Unsupported dialect '{dialect}'. Supported: {', '.join(sorted(supported_dialects))}"
    
    return None


def validate_file_path(file_path: str) -> Optional[str]:
    """
    Validate file path.
    
    Args:
        file_path: File path to validate
        
    Returns:
        Error message if invalid, None if valid
    """
    if not file_path:
        return "File path cannot be empty"
    
    if not isinstance(file_path, str):
        return "File path must be a string"
    
    if len(file_path.strip()) == 0:
        return "File path cannot be empty or whitespace only"
    
    # Basic path validation
    invalid_chars = ['<', '>', '|', '\0']
    if any(char in file_path for char in invalid_chars):
        return f"File path contains invalid characters: {invalid_chars}"
    
    return None