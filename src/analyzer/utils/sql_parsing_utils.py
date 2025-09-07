"""SQL parsing utility functions."""

import re
from typing import Dict, Optional, List, Set
import sqlglot


def clean_table_name_quotes(table_name: str) -> str:
    """Remove quotes from table name while preserving case and structure."""
    if not table_name:
        return table_name
    
    # Split by dots and remove quotes from each part
    parts = table_name.split('.')
    cleaned_parts = [part.strip('"').strip("'").strip('`') for part in parts]
    return '.'.join(cleaned_parts)


def normalize_entity_name(entity_name: str) -> str:
    """
    Normalize entity name by removing structural quotes while preserving semantics.
    
    This function handles the specific normalization pattern used for fixing 
    quoting inconsistencies in multi-database queries.
    
    Args:
        entity_name: The entity name to normalize (table, column, etc.)
        
    Returns:
        Normalized entity name with structural quotes removed
        
    Examples:
        '"oracle".schema.table' -> 'oracle.schema.table'
        '"database"."schema"."table"' -> 'database.schema.table'
        'database.schema."table"' -> 'database.schema.table'
    """
    if not entity_name:
        return entity_name
    
    return entity_name.strip('"').replace('".', '.').replace('"', '')


def normalize_table_name(table_name: str) -> str:
    """Normalize table name by removing quotes and standardizing format."""
    if not table_name:
        return ""
    
    # Remove quotes and extra whitespace
    normalized = table_name.strip().strip('"').strip("'").strip('`')
    return normalized.lower()


class CompatibilityMode:
    """Control normalization behavior for backward compatibility"""
    
    DISABLED = "disabled"           # No normalization (existing behavior)
    CONSERVATIVE = "conservative"   # Only normalize obvious duplicates
    FULL = "full"                  # Full normalization (new behavior)


class NamingPatternAnalysis:
    """Analyze table naming patterns to determine normalization strategy"""
    
    def __init__(self, table_names: set, dialect: str = "trino"):
        self.simple_names = set()      # users, orders
        self.schema_qualified = set()   # schema.table  
        self.catalog_qualified = set()  # catalog.schema.table
        self.quoted_names = set()       # "catalog"."schema"."table"
        self.needs_normalization = False
        self.dialect = dialect
        
        self._analyze_patterns(table_names)
    
    def _analyze_patterns(self, table_names: set):
        """Analyze naming patterns in the table set"""
        for name in table_names:
            if self._is_simple_name(name):
                self.simple_names.add(name)
            elif self._is_schema_qualified(name):
                self.schema_qualified.add(name)
            elif self._is_catalog_qualified(name):
                self.catalog_qualified.add(name)
            elif self._is_quoted_qualified(name):
                self.quoted_names.add(name)
        
        # Determine if normalization is needed
        pattern_types = sum([
            bool(self.simple_names),
            bool(self.schema_qualified), 
            bool(self.catalog_qualified),
            bool(self.quoted_names)
        ])
        
        # Only normalize if we have mixed patterns
        self.needs_normalization = pattern_types > 1
    
    def _is_simple_name(self, name: str) -> bool:
        """Check if name is simple (users, orders)"""
        return '.' not in name and '"' not in name and '`' not in name
    
    def _is_schema_qualified(self, name: str) -> bool:
        """Check if name is schema.table format"""
        return '.' in name and name.count('.') == 1 and '"' not in name
    
    def _is_catalog_qualified(self, name: str) -> bool:
        """Check if name is catalog.schema.table format (unquoted)"""
        return '.' in name and name.count('.') == 2 and '"' not in name
    
    def _is_quoted_qualified(self, name: str) -> bool:
        """Check if name uses quoted identifiers"""
        return '"' in name or '`' in name
    
    def has_obvious_duplicates(self) -> bool:
        """Check if there are obvious duplicates (same base name, different qualification)"""
        base_names = set()
        for name in self._get_all_names():
            base_name = self._extract_base_name(name)
            if base_name in base_names:
                return True
            base_names.add(base_name)
        return False
    
    def _get_all_names(self) -> set:
        """Get all table names from all categories"""
        return (self.simple_names | self.schema_qualified | 
                self.catalog_qualified | self.quoted_names)
    
    def _extract_base_name(self, table_name: str) -> str:
        """Extract base table name from any qualification format"""
        # Handle quoted names: "catalog"."schema"."table" → table
        if '"' in table_name:
            parts = [p.strip('"') for p in table_name.split('.')]
            return parts[-1].lower()
        
        # Handle backtick names: `catalog`.`schema`.`table` → table  
        if '`' in table_name:
            parts = [p.strip('`') for p in table_name.split('.')]
            return parts[-1].lower()
        
        # Handle unquoted names: catalog.schema.table → table
        return table_name.split('.')[-1].lower()


def normalize_table_references(table_names: set, dialect: str = "trino", 
                              compatibility_mode: str = CompatibilityMode.FULL) -> dict:
    """
    Normalize a set of table names to resolve duplicates with backward compatibility.
    
    Args:
        table_names: Set of table names to normalize
        dialect: SQL dialect for context-aware normalization
        compatibility_mode: Compatibility mode for backward compatibility
        
    Returns:
        Dict mapping original table names to their canonical forms
    """
    if not table_names:
        return {}
    
    # Handle compatibility modes
    if compatibility_mode == CompatibilityMode.DISABLED:
        return {name: name for name in table_names}
    
    # Analyze naming patterns
    analysis = NamingPatternAnalysis(table_names, dialect)
    
    if compatibility_mode == CompatibilityMode.CONSERVATIVE:
        # Only normalize if we detect clear duplicates
        if not analysis.has_obvious_duplicates():
            return {name: name for name in table_names}
    
    # If no normalization needed, return identity mapping
    if not analysis.needs_normalization:
        return {name: name for name in table_names}
    
    # Group tables by their normalized base name (without catalog/schema)
    base_name_groups = {}
    canonical_mapping = {}
    
    for table_name in table_names:
        # Extract base table name (last part after dots)
        base_name = analysis._extract_base_name(table_name)
        
        if base_name not in base_name_groups:
            base_name_groups[base_name] = []
        base_name_groups[base_name].append(table_name)
    
    # For each group, choose the most qualified name as canonical
    for base_name, table_list in base_name_groups.items():
        if len(table_list) == 1:
            # Only one table with this base name - use as-is
            canonical_mapping[table_list[0]] = table_list[0]
        else:
            # Multiple tables with same base name - choose most qualified
            canonical_name = choose_canonical_table_name(table_list, dialect)
            for table_name in table_list:
                canonical_mapping[table_name] = canonical_name
    
    return canonical_mapping


def choose_canonical_table_name(candidates: List[str], dialect: str = "trino") -> str:
    """
    Choose canonical name based on context and dialect preferences.
    
    Priority order:
    1. Most qualified name in current dialect format
    2. Preserve existing format if no conflicts
    3. Fallback to most explicit naming
    """
    if len(candidates) == 1:
        return candidates[0]
    
    # Group by qualification level
    simple = [c for c in candidates if '.' not in c]
    qualified = [c for c in candidates if '.' in c]
    
    # If we have both simple and qualified names for same table
    if simple and qualified:
        # For Trino/similar: prefer fully qualified
        if dialect.lower() in ['trino', 'presto', 'spark']:
            return max(qualified, key=lambda x: (x.count('.'), '"' in x, len(x)))
        # For others: prefer qualified but not overly complex
        else:
            schema_qualified = [q for q in qualified if q.count('.') == 1]
            if schema_qualified:
                return schema_qualified[0]
            return qualified[0]
    
    # If all same qualification level, prefer quoted/explicit
    return max(candidates, key=lambda x: ('"' in x, len(x)))


class TableNameRegistry:
    """Centralized registry for canonical table names with validation"""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = CompatibilityMode.FULL):
        self.dialect = dialect
        self.compatibility_mode = compatibility_mode
        self._canonical_mapping = {}
        self._reverse_mapping = {}  # canonical -> set of original names
        self._table_aliases = {}
    
    def register_tables(self, table_names: set, alias_mappings: dict = None):
        """Register a set of table names and create canonical mappings"""
        if alias_mappings is None:
            alias_mappings = {}
        
        self._table_aliases.update(alias_mappings)
        
        # Create canonical mapping
        canonical_mapping = normalize_table_references(
            table_names, self.dialect, self.compatibility_mode
        )
        
        # Update registries
        for original, canonical in canonical_mapping.items():
            self._canonical_mapping[original] = canonical
            
            if canonical not in self._reverse_mapping:
                self._reverse_mapping[canonical] = set()
            self._reverse_mapping[canonical].add(original)
    
    def get_canonical_name(self, table_name: str) -> str:
        """Get the canonical name for any table reference"""
        return self._canonical_mapping.get(table_name, table_name)
    
    def get_all_canonical_tables(self) -> set:
        """Get all unique canonical table names"""
        return set(self._canonical_mapping.values())
    
    def get_original_names(self, canonical_name: str) -> set:
        """Get all original names that map to this canonical name"""
        return self._reverse_mapping.get(canonical_name, {canonical_name})
    
    def is_canonical(self, table_name: str) -> bool:
        """Check if a table name is in canonical form"""
        canonical = self.get_canonical_name(table_name)
        return table_name == canonical
    
    def has_duplicates(self) -> bool:
        """Check if any table names had duplicates that were normalized"""
        return any(len(originals) > 1 for originals in self._reverse_mapping.values())
    
    def get_normalization_summary(self) -> dict:
        """Get summary of normalization actions taken"""
        return {
            'total_original_names': len(self._canonical_mapping),
            'total_canonical_names': len(self._reverse_mapping),
            'duplicates_found': self.has_duplicates(),
            'normalization_count': sum(
                len(originals) - 1 for originals in self._reverse_mapping.values()
            )
        }


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
    
    if len(parts) == 4:
        # Handle catalog.schema.table.column format
        return {
            "schema": f"{clean_column_name(parts[0])}.{clean_column_name(parts[1])}",
            "table": clean_column_name(parts[2]), 
            "column": clean_column_name(parts[3])
        }
    elif len(parts) == 3:
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


def normalize_quoted_identifier(identifier: str) -> str:
    """Normalize quoted identifiers for Trino/other dialects."""
    if not identifier:
        return ""
    
    # Remove outer quotes but preserve structure
    normalized = str(identifier).strip()
    
    # Handle different quote types: "name", 'name', `name`
    if ((normalized.startswith('"') and normalized.endswith('"')) or
        (normalized.startswith("'") and normalized.endswith("'")) or
        (normalized.startswith('`') and normalized.endswith('`'))):
        normalized = normalized[1:-1]
    
    return normalized


def build_alias_to_table_mapping(sql: str, dialect: str = "trino") -> Dict[str, str]:
    """Build a mapping from table aliases to actual table names by parsing SQL."""
    alias_to_table = {}
    
    try:
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Find all table references with aliases
        tables = list(parsed.find_all(sqlglot.exp.Table))
        for table in tables:
            # Extract full qualified table name with enhanced quoted identifier handling
            if table.catalog and table.db:
                # Handle three-part naming (e.g., "catalog"."schema"."table") 
                catalog = normalize_quoted_identifier(str(table.catalog))
                schema = normalize_quoted_identifier(str(table.db))  
                table_name_part = normalize_quoted_identifier(str(table.name))
                table_name = f'"{catalog}"."{schema}"."{table_name_part}"'
            elif table.db:
                # Handle database.table naming (e.g., "ecommerce.users")
                db = normalize_quoted_identifier(str(table.db))
                table_name_part = normalize_quoted_identifier(str(table.name))
                table_name = f"{db}.{table_name_part}"
            else:
                # Handle simple table naming (e.g., "users")
                table_name = normalize_quoted_identifier(str(table.name))
                
            if table.alias:
                alias = normalize_quoted_identifier(str(table.alias)).lower()
                alias_to_table[alias] = table_name
                
        # Handle subquery aliases (e.g., (SELECT * FROM table) t1)
        subqueries = list(parsed.find_all(sqlglot.exp.Subquery))
        for subquery in subqueries:
            if subquery.alias:
                # Find the table inside the subquery
                subquery_tables = list(subquery.find_all(sqlglot.exp.Table))
                if subquery_tables:
                    # Use the first (and usually only) table in the subquery
                    table = subquery_tables[0]
                    
                    # Extract full qualified table name 
                    if table.catalog and table.db:
                        catalog = normalize_quoted_identifier(str(table.catalog))
                        schema = normalize_quoted_identifier(str(table.db))  
                        table_name_part = normalize_quoted_identifier(str(table.name))
                        table_name = f'"{catalog}"."{schema}"."{table_name_part}"'
                    elif table.db:
                        db = normalize_quoted_identifier(str(table.db))
                        table_name_part = normalize_quoted_identifier(str(table.name))
                        table_name = f"{db}.{table_name_part}"
                    else:
                        table_name = normalize_quoted_identifier(str(table.name))
                    
                    alias = normalize_quoted_identifier(str(subquery.alias)).lower()
                    alias_to_table[alias] = table_name
        
        # Also find CTE aliases in WITH clauses for better CTE column resolution
        if 'WITH' in sql.upper():
            ctes = list(parsed.find_all(sqlglot.exp.CTE))
            for cte in ctes:
                if cte.alias:
                    cte_alias = normalize_quoted_identifier(str(cte.alias)).lower()
                    # CTE aliases map to themselves as they define new tables
                    alias_to_table[cte_alias] = cte_alias
                
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
    
    # Normalize both table_name and column_name to handle quoting inconsistencies
    # This fixes issues with multi-database quoted table names where entity names have different quoting than column references
    table_name_normalized = normalize_entity_name(table_name)
    column_name_normalized = normalize_entity_name(column_name)
    
    # Handle qualified column names (e.g., "users.active", "u.salary", "ecommerce.users.status")
    if '.' in column_name_normalized:
        parsed = parse_qualified_name(column_name_normalized)
        
        # For database.table.column format, reconstruct full table name
        if parsed["schema"] and parsed["table"] and '.' in table_name_normalized:
            # This is likely database.table.column format
            column_table = f"{parsed['schema']}.{parsed['table']}"
        else:
            column_table = parsed["table"]
        
        if not column_table:
            return False
            
        # Direct table name match using normalized names
        if column_table == table_name_normalized:
            return True
            
        # Check if it's an alias match using SQL context
        alias_mapping = {}
        if sql:
            alias_mapping = build_alias_to_table_mapping(sql, dialect)
            actual_table = alias_mapping.get(column_table.lower())
            if actual_table and actual_table == table_name_normalized:
                return True
        
        # Smart name matching for database.table vs simple table names
        # Handle cases where column references simple name but table is database.table
        if not alias_mapping:
            # Extract base table names for comparison
            column_table_base = column_table.split('.')[-1] if '.' in column_table else column_table
            table_name_base = table_name.split('.')[-1] if '.' in table_name else table_name
            
            # Remove quotes for comparison
            column_table_clean = column_table_base.strip('"').strip("'")
            table_name_clean = table_name_base.strip('"').strip("'")
            
            if column_table_clean.lower() == table_name_clean.lower():
                return True
            
            # Also handle cases where table_name contains full qualified name
            # and column_table is an alias (e.g. pmf vs ins_sql.ins_fin.policy_monthly_financials)
            if table_name.lower().endswith('.' + column_table_clean.lower()) or \
               column_table_clean.lower() in table_name.lower():
                return True
    
    # For unqualified column names, we need to check if they're in subqueries
    if not ('.' in column_name) and sql:
        # Use the same logic as column extraction utility to exclude subquery columns  
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=dialect)
            
            # Find all column nodes with this name
            for column_node in parsed.find_all(sqlglot.exp.Column):
                if str(column_node.name) == column_name:
                    # Check if this column is inside a subquery
                    parent_node = column_node.parent
                    subquery_node = None
                    while parent_node:
                        if isinstance(parent_node, sqlglot.exp.Subquery):
                            subquery_node = parent_node
                            break
                        parent_node = parent_node.parent
                    
                    if subquery_node:
                        # This column is inside a subquery - check if the subquery references our target table
                        subquery_select = subquery_node.this
                        if isinstance(subquery_select, sqlglot.exp.Select):
                            from_clause = subquery_select.find(sqlglot.exp.From)
                            if from_clause and hasattr(from_clause.this, 'name'):
                                subquery_table = str(from_clause.this.name)
                                # If subquery references the target table, the column belongs to that table
                                if subquery_table == table_name:
                                    return True
                        # Column is in subquery but doesn't reference our target table
                        return False
            
            # If no unqualified columns of this name are in subqueries, assume it belongs to this table
            return True
            
        except Exception:
            # Fallback to original behavior for backward compatibility
            return True
    
    return False


def clean_source_expression(expression: str) -> str:
    """Clean source expression by removing AS alias part and cleaning table name quotes."""
    if not expression:
        return ""
        
    clean_expr = expression
    
    # Smart AS alias detection - only split on AS that's at the end (for column aliases)
    # Don't split on AS within function calls like CAST(... AS DATE)
    import re
    # Pattern to match AS alias at the end of expression (after balanced parentheses)
    # This matches: "expression AS alias" but not "CAST(value AS type)"
    as_alias_pattern = r'^(.+?)\s+AS\s+(["\w]+)$'
    match = re.match(as_alias_pattern, expression, re.IGNORECASE)
    if match:
        clean_expr = match.group(1).strip()
    else:
        # Check for lowercase 'as' at the end
        as_alias_pattern_lower = r'^(.+?)\s+as\s+(["\w]+)$'
        match = re.match(as_alias_pattern_lower, expression)
        if match:
            clean_expr = match.group(1).strip()
    
    # Also clean table name quotes within the expression
    clean_expr = _clean_table_names_in_expression(clean_expr)
    return clean_expr


def _clean_table_names_in_expression(expression: str) -> str:
    """Clean quoted table names within SQL expressions."""
    import re
    
    # Pattern to match quoted table references like "schema"."table"."column"
    # This handles patterns like: "dbxadmin40test"."trino_demo"."orders"."o_orderkey"
    pattern = r'"([^"]+)"\."([^"]+)"\."([^"]+)"\."([^"]+)"'
    expression = re.sub(pattern, r'\1.\2.\3.\4', expression)
    
    # Pattern to match quoted table references like "schema"."table"."column" (3 parts)
    pattern = r'"([^"]+)"\."([^"]+)"\."([^"]+)"'
    expression = re.sub(pattern, r'\1.\2.\3', expression)
    
    # Pattern to match quoted table references like "schema"."table" (2 parts)
    pattern = r'"([^"]+)"\."([^"]+)"'
    expression = re.sub(pattern, r'\1.\2', expression)
    
    return expression


def is_subquery_expression(expression: str, dialect: str = "trino") -> bool:
    """Check if expression contains a subquery (SELECT statement)."""
    if not expression:
        return False
    
    try:
        import sqlglot
        from sqlglot import exp
        
        # Try to parse as an expression
        parsed = sqlglot.parse_one(expression, dialect=dialect)
        
        # Check if there are any Subquery nodes
        return bool(list(parsed.find_all(exp.Subquery)))
        
    except Exception:
        # Fallback to regex-based detection
        import re
        pattern = r"\(\s*SELECT\s+.*?\s+FROM\s+.*?\)"
        return bool(re.search(pattern, expression, re.IGNORECASE | re.DOTALL))


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
                    if actual_table:
                        # Remove quotes from actual_table for comparison
                        actual_table_clean = normalize_entity_name(actual_table)
                        if actual_table_clean == table_name or actual_table == table_name:
                            columns.add(column_name)
                    else:
                        # Handle three-part table names: if table_name ends with table_ref, it's a match
                        # Example: table_name="dbxadmin40test"."trino_demo"."orders" and table_ref="orders"
                        # Also handle cases like ins_sql.ins_fin.policy_monthly_financials matching pmf
                        if table_name.lower().endswith('.' + table_ref.lower()) or table_name.lower().endswith(table_ref.lower()):
                            columns.add(column_name)
                        if table_name.endswith(f'"."{table_ref}"') or table_name.endswith(f'.{table_ref}'):
                            columns.add(column_name)
    
    except Exception:
        # Fallback to regex-based extraction
        pattern = rf'\b{re.escape(table_name)}\.(\w+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)
        columns.update(matches)
    
    return columns


def extract_table_columns_from_sql_batch(sql: str, table_names: List[str], dialect: str = "trino") -> Dict[str, Set[str]]:
    """
    Extract columns for MULTIPLE tables from SQL in one parse.
    More efficient version of extract_table_columns_from_sql() for batch processing.
    
    Args:
        sql: SQL query to analyze
        table_names: List of table names to extract columns for
        dialect: SQL dialect (default: trino)
        
    Returns:
        Dict mapping table_name -> set of column names
    """
    all_columns = {table_name: set() for table_name in table_names}
    
    try:
        # Parse SQL once (instead of once per table)
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        
        # Build alias mapping once (instead of once per table) 
        alias_mapping = build_alias_to_table_mapping(sql, dialect)
        
        # Extract columns for ALL tables in one pass
        for column in parsed.find_all(sqlglot.exp.Column):
            if column.table:
                table_ref = str(column.table)
                column_name = str(column.name)
                
                # Check against ALL target tables (reusing existing logic)
                for table_name in table_names:
                    # Direct table name match or alias resolution
                    if table_ref == table_name:
                        all_columns[table_name].add(column_name)
                    else:
                        # Check if it's an alias (same logic as original function)
                        actual_table = alias_mapping.get(table_ref.lower())
                        if actual_table:
                            # Remove quotes from actual_table for comparison
                            actual_table_clean = normalize_entity_name(actual_table)
                            if actual_table_clean == table_name or actual_table == table_name:
                                all_columns[table_name].add(column_name)
                        else:
                            # Handle three-part table names: if table_name ends with table_ref, it's a match
                            # Example: table_name="dbxadmin40test"."trino_demo"."orders" and table_ref="orders"
                            # Also handle cases like ins_sql.ins_fin.policy_monthly_financials matching pmf
                            if table_name.lower().endswith('.' + table_ref.lower()) or table_name.lower().endswith(table_ref.lower()):
                                all_columns[table_name].add(column_name)
                            if table_name.endswith(f'"."{table_ref}"') or table_name.endswith(f'.{table_ref}'):
                                all_columns[table_name].add(column_name)
                                        
    except Exception:
        # Fallback to regex-based extraction for each table
        for table_name in table_names:
            pattern = rf'\b{re.escape(table_name)}\.(\w+)'
            matches = re.findall(pattern, sql, re.IGNORECASE)
            all_columns[table_name].update(matches)
    
    return all_columns


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


def extract_tables_from_subquery(subquery_expr: str, dialect: str = "trino") -> List[str]:
    """Extract table names from a subquery expression."""
    if not subquery_expr:
        return []
    
    try:
        import sqlglot
        import re
        
        # Extract just the subquery part (without outer alias)
        subquery_pattern = r'\(SELECT.*?\)'
        subquery_match = re.search(subquery_pattern, subquery_expr, re.IGNORECASE | re.DOTALL)
        if not subquery_match:
            return []
            
        subquery_sql = subquery_match.group(0)
        
        # Parse the subquery
        parsed_subquery = sqlglot.parse_one(subquery_sql, dialect=dialect)
        if not isinstance(parsed_subquery, sqlglot.exp.Select):
            return []
        
        # Get the FROM clause of the subquery
        from_clause = parsed_subquery.find(sqlglot.exp.From)
        if not from_clause:
            return []
        
        # Get tables from the subquery's FROM clause
        tables = []
        subquery_tables = list(from_clause.find_all(sqlglot.exp.Table))
        for table in subquery_tables:
            table_name = str(table.name)
            if table_name not in tables:
                tables.append(table_name)
                
        return tables
        
    except Exception:
        # Fallback: use regex to extract table names from FROM clause
        import re
        from_match = re.search(r'FROM\s+([^)\s]+)', subquery_expr, re.IGNORECASE)
        if from_match:
            return [from_match.group(1).strip()]
        return []


def is_subquery_relevant_to_table(subquery_expr: str, table_name: str, dialect: str = "trino") -> bool:
    """Check if a subquery expression is relevant to a specific table."""
    if not subquery_expr or not table_name:
        return False
    
    # Extract tables from the subquery
    subquery_tables = extract_tables_from_subquery(subquery_expr, dialect)
    return table_name in subquery_tables


def extract_columns_referenced_by_table_in_union(sql: str, table_name: str, dialect: str = "trino") -> List[str]:
    """Extract all columns referenced by a specific table in a UNION query (SELECT + WHERE clauses)."""
    columns = []
    
    try:
        import re
        
        # Split the SQL into individual SELECT statements
        union_statements = re.split(r'\bUNION(?:\s+ALL)?\b', sql, flags=re.IGNORECASE)
        
        for statement in union_statements:
            statement = statement.strip()
            # Check if this statement contains our table
            if f'FROM {table_name.upper()}' in statement.upper():
                # Extract columns from SELECT clause (only actual column references, not literals)
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', statement, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1)
                    # Find column references (skip string literals like 'user')
                    for item in select_clause.split(','):
                        item = item.strip()
                        # Skip string literals
                        if not item.startswith("'"):
                            # Extract column name (before 'as' if present)
                            col_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', item)
                            if col_match:
                                col_name = col_match.group(1)
                                if col_name not in columns:
                                    columns.append(col_name)
                
                # Extract columns from WHERE clause
                where_match = re.search(r'WHERE\s+(.*?)(?:\s*$)', statement, re.IGNORECASE | re.DOTALL)
                if where_match:
                    where_clause = where_match.group(1).strip()
                    # Extract column references from WHERE conditions (column names before operators)
                    col_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[><=!]', where_clause)
                    for col in col_matches:
                        if col not in columns and col.lower() not in ['and', 'or', 'where']:
                            columns.append(col)
                
                break  # Found the statement for this table
        
        return columns
        
    except Exception:
        return []# CTE validation functions



class CircularDependencyError(ValueError):
    """Exception raised when circular dependencies are detected in lineage analysis."""
    
    def __init__(self, cycle_path, dependency_type="CTE"):
        self.cycle_path = cycle_path
        self.dependency_type = dependency_type
        cycle_str = " -> ".join(cycle_path + [cycle_path[0]])
        super().__init__(f"CircularDependencyError: Circular {dependency_type} dependency detected: {cycle_str}")


def validate_cte_dependencies(sql, dialect="trino"):
    """Validate that CTEs do not have circular dependencies."""
    if not sql or "WITH" not in sql.upper():
        return True
    
    try:
        import sqlglot
        parsed = sqlglot.parse_one(sql, dialect=dialect)
        cte_deps = {}
        
        for with_stmt in parsed.find_all(sqlglot.exp.With):
            for cte in with_stmt.expressions:
                if not isinstance(cte, sqlglot.exp.CTE) or not cte.alias:
                    continue
                
                cte_name = str(cte.alias)
                defined_ctes = {str(c.alias) for c in with_stmt.expressions 
                               if isinstance(c, sqlglot.exp.CTE) and c.alias}
                
                referenced = []
                if cte.this:
                    for table_ref in cte.this.find_all(sqlglot.exp.Table):
                        table_name = str(table_ref.name) if table_ref.name else None
                        if table_name and table_name in defined_ctes:
                            referenced.append(table_name)
                
                if referenced:
                    cte_deps[cte_name] = referenced
        
        # Simple cycle detection
        def has_cycle(node, visited, path):
            if node in path:
                cycle_start = path.index(node)
                raise CircularDependencyError(path[cycle_start:], "CTE")
            if node in visited:
                return False
            visited.add(node)
            path.append(node)
            for dep in cte_deps.get(node, []):
                has_cycle(dep, visited, path)
            path.pop()
            return False
        
        visited = set()
        for node in cte_deps:
            if node not in visited:
                has_cycle(node, visited, [])
        
        return True
    except CircularDependencyError:
        raise
    except:
        return True
