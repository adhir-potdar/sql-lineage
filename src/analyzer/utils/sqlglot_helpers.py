"""SQLGlot helper functions for consistent SQL parsing."""

from typing import List, Dict, Set, Optional, Any, Union
import sqlglot
import sqlglot.expressions as exp


def parse_sql_safely(sql: str, dialect: str = "trino") -> Optional[exp.Expression]:
    """Parse SQL safely with error handling."""
    try:
        return sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return None


def traverse_ast_nodes(parsed_sql: exp.Expression, node_type: type) -> List[exp.Expression]:
    """Traverse AST and find all nodes of a specific type."""
    if parsed_sql is None:
        return []
    
    try:
        return list(parsed_sql.find_all(node_type))
    except Exception:
        return []


def extract_table_references(sql: str, dialect: str = "trino") -> List[Dict[str, Optional[str]]]:
    """Extract all table references from SQL including aliases and schemas."""
    tables = []
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return tables
    
    table_nodes = traverse_ast_nodes(parsed, exp.Table)
    
    for table in table_nodes:
        table_info = {
            "name": table.name,
            "alias": str(table.alias) if table.alias else None,
            "schema": str(table.db) if table.db else None,
            "catalog": str(table.catalog) if table.catalog else None
        }
        tables.append(table_info)
    
    return tables


def extract_column_references(sql: str, dialect: str = "trino") -> List[Dict[str, Optional[str]]]:
    """Extract all column references from SQL."""
    columns = []
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return columns
    
    column_nodes = traverse_ast_nodes(parsed, exp.Column)
    
    for column in column_nodes:
        column_info = {
            "name": str(column.name) if column.name else None,
            "table": str(column.table) if column.table else None,
            "schema": str(column.db) if column.db else None
        }
        columns.append(column_info)
    
    return columns


def get_select_expressions(sql: str, dialect: str = "trino") -> List[exp.Expression]:
    """Get all expressions from SELECT clause."""
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return []
    
    select_stmt = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    
    if select_stmt and select_stmt.expressions:
        return select_stmt.expressions
    
    return []


def get_where_conditions(sql: str, dialect: str = "trino") -> List[exp.Expression]:
    """Extract WHERE clause conditions."""
    conditions = []
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return conditions
    
    select_stmt = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    
    if select_stmt:
        where_clause = select_stmt.find(exp.Where)
        if where_clause:
            # Find all binary operations (comparisons)
            conditions.extend(traverse_ast_nodes(where_clause, exp.Binary))
    
    return conditions


def get_join_conditions(sql: str, dialect: str = "trino") -> List[Dict[str, Any]]:
    """Extract JOIN conditions and types."""
    joins = []
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return joins
    
    join_nodes = traverse_ast_nodes(parsed, exp.Join)
    
    for join in join_nodes:
        join_info = {
            "join_type": str(join.kind) if join.kind else "INNER",
            "right_table": str(join.this) if join.this else None,
            "conditions": []
        }
        
        if join.on:
            # Extract join conditions
            binary_ops = traverse_ast_nodes(join.on, exp.Binary)
            for binary_op in binary_ops:
                condition = {
                    "left": str(binary_op.left) if binary_op.left else None,
                    "operator": str(binary_op.key) if binary_op.key else "=",
                    "right": str(binary_op.right) if binary_op.right else None
                }
                join_info["conditions"].append(condition)
        
        joins.append(join_info)
    
    return joins


def get_group_by_columns(sql: str, dialect: str = "trino") -> List[str]:
    """Extract GROUP BY columns."""
    columns = []
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return columns
    
    select_stmt = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    
    if select_stmt:
        group_by = select_stmt.find(exp.Group)
        if group_by and group_by.expressions:
            for expr in group_by.expressions:
                columns.append(str(expr))
    
    return columns


def get_order_by_columns(sql: str, dialect: str = "trino") -> List[str]:
    """Extract ORDER BY columns with direction."""
    columns = []
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return columns
    
    select_stmt = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    
    if select_stmt:
        order_by = select_stmt.find(exp.Order)
        if order_by and order_by.expressions:
            for ordered in order_by.expressions:
                if hasattr(ordered, 'this'):
                    column_name = str(ordered.this)
                    direction = "DESC" if ordered.desc else "ASC"
                    columns.append(f"{column_name} {direction}")
                else:
                    columns.append(str(ordered))
    
    return columns


def get_cte_definitions(sql: str, dialect: str = "trino") -> Dict[str, str]:
    """Extract CTE definitions from WITH clause."""
    ctes = {}
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return ctes
    
    # Look for WITH clause
    with_nodes = traverse_ast_nodes(parsed, exp.With)
    
    for with_node in with_nodes:
        if with_node.expressions:
            for cte in with_node.expressions:
                if hasattr(cte, 'alias') and hasattr(cte, 'this'):
                    cte_name = str(cte.alias)
                    cte_sql = str(cte.this)
                    ctes[cte_name] = cte_sql
    
    return ctes


def get_node_type(expression: exp.Expression) -> str:
    """Get the type of a SQLGlot expression node."""
    return type(expression).__name__


def is_aggregate_function(expression: exp.Expression) -> bool:
    """Check if an expression is an aggregate function."""
    aggregate_types = {
        exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max,
        exp.GroupConcat, exp.StdDev, exp.Variance
    }
    
    return any(isinstance(expression, agg_type) for agg_type in aggregate_types)


def is_window_function(expression: exp.Expression) -> bool:
    """Check if an expression is a window function."""
    window_types = {
        exp.RowNumber, exp.Rank, exp.DenseRank, exp.Lead, exp.Lag,
        exp.FirstValue, exp.LastValue, exp.NthValue
    }
    
    return any(isinstance(expression, win_type) for win_type in window_types) or \
           (hasattr(expression, 'over') and expression.over is not None)


def extract_function_name(expression: exp.Expression) -> Optional[str]:
    """Extract function name from a function expression."""
    if isinstance(expression, exp.Anonymous):
        return str(expression.this) if expression.this else None
    elif hasattr(expression, 'key'):
        return expression.key
    elif hasattr(expression, '__class__'):
        return expression.__class__.__name__.upper()
    
    return None


def get_table_from_column(column: exp.Column) -> Optional[str]:
    """Get table name from a column reference."""
    if column.table:
        return str(column.table)
    return None


def extract_literal_value(expression: exp.Expression) -> Any:
    """Extract literal value from an expression."""
    if isinstance(expression, exp.Literal):
        return expression.this
    elif isinstance(expression, exp.Boolean):
        return bool(expression.this)
    elif isinstance(expression, exp.Null):
        return None
    else:
        return str(expression)


def build_column_lineage_map(sql: str, dialect: str = "trino") -> Dict[str, Set[str]]:
    """Build a mapping of target columns to their source columns."""
    lineage_map = {}
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return lineage_map
    
    select_expressions = get_select_expressions(sql, dialect)
    
    for expr in select_expressions:
        # Get target column name (alias or expression)
        target_name = str(expr.alias) if expr.alias else str(expr)
        
        # Find source columns in the expression
        source_columns = set()
        column_refs = traverse_ast_nodes(expr, exp.Column)
        
        for col in column_refs:
            if col.name:
                source_columns.add(str(col.name))
        
        lineage_map[target_name] = source_columns
    
    return lineage_map


def validate_sql_syntax(sql: str, dialect: str = "trino") -> bool:
    """Validate SQL syntax using SQLGlot parser."""
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        return parsed is not None
    except Exception:
        return False


def get_sql_statement_type(sql: str, dialect: str = "trino") -> Optional[str]:
    """Determine the type of SQL statement."""
    parsed = parse_sql_safely(sql, dialect)
    
    if parsed is None:
        return None
    
    return get_node_type(parsed)