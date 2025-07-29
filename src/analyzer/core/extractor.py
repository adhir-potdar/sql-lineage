"""Core lineage extraction logic."""

import re
from typing import Dict, Set, Optional, List, Tuple
from collections import defaultdict

import sqlglot
from sqlglot import Expression, exp

from .models import TableLineage, ColumnLineage


class LineageExtractor:
    """Extracts lineage information from parsed SQL expressions."""
    
    def __init__(self):
        pass
    
    def extract_table_lineage(self, expression: Expression) -> TableLineage:
        """Extract table-level lineage from SQL expression."""
        lineage = TableLineage()
        alias_mappings = self._extract_table_alias_mappings(expression)
        
        # First, collect all CTE names
        cte_mapping = {}
        for cte in expression.find_all(exp.CTE):
            cte_name = cte.alias
            cte_mapping[cte_name] = cte
        
        # Check if we should skip any CTEs that are simple pass-throughs
        ctes_to_skip = self._identify_passthrough_ctes(expression, cte_mapping)
        
        # Track dependencies of skipped CTEs so we can use them later
        skipped_cte_dependencies = {}
        
        # Process CTEs in order (they can reference earlier CTEs)
        processed_ctes = {}
        for cte in expression.find_all(exp.CTE):
            cte_name = cte.alias
            
            # Find source tables in this CTE, allowing references to previous CTEs
            source_tables = self._get_source_tables_from_node(cte.this, alias_mappings, processed_ctes, cte_name)
            
            # If this CTE should be skipped, store its dependencies but don't add to lineage
            if cte_name in ctes_to_skip:
                skipped_cte_dependencies[cte_name] = source_tables
                processed_ctes[cte_name] = cte
                continue
            
            # Always create an entry for this CTE, even if it has no sources
            if not source_tables:
                # Create empty entry
                if cte_name not in lineage.upstream:
                    lineage.upstream[cte_name] = set()
            else:
                for source in source_tables:
                    lineage.add_dependency(cte_name, source)
            
            # Add this CTE to the processed set
            processed_ctes[cte_name] = cte
        
        # Process main query
        if isinstance(expression, exp.Select):
            target_name = "QUERY_RESULT"
            source_tables = self._get_source_tables_from_node(expression, alias_mappings, cte_mapping)
            
            for source in source_tables:
                # If this source is a skipped CTE, replace it with its dependencies
                if source in ctes_to_skip and source in skipped_cte_dependencies:
                    # Add the dependencies of the skipped CTE instead
                    for skipped_cte_dependency in skipped_cte_dependencies[source]:
                        lineage.add_dependency(target_name, skipped_cte_dependency)
                else:
                    lineage.add_dependency(target_name, source)
        
        # Handle UNION queries
        elif isinstance(expression, exp.Union):
            target_name = "QUERY_RESULT"
            source_tables = self._get_source_tables_from_node(expression, alias_mappings, cte_mapping)
            
            for source in source_tables:
                # If this source is a skipped CTE, replace it with its dependencies
                if source in ctes_to_skip and source in skipped_cte_dependencies:
                    # Add the dependencies of the skipped CTE instead
                    for skipped_cte_dependency in skipped_cte_dependencies[source]:
                        lineage.add_dependency(target_name, skipped_cte_dependency)
                else:
                    lineage.add_dependency(target_name, source)
        
        # Handle CREATE TABLE AS SELECT
        elif isinstance(expression, exp.Create):
            if hasattr(expression, 'this') and expression.this:
                # For CREATE TABLE, use the raw table name without adding default schema
                target_name = str(expression.this)
                
                # Find the SELECT part
                for select in expression.find_all(exp.Select):
                    source_tables = self._get_source_tables_from_node(select, alias_mappings, cte_mapping)
                    for source in source_tables:
                        # If this source is a skipped CTE, replace it with its dependencies
                        if source in ctes_to_skip and source in skipped_cte_dependencies:
                            # Add the dependencies of the skipped CTE instead
                            for skipped_cte_dependency in skipped_cte_dependencies[source]:
                                lineage.add_dependency(target_name, skipped_cte_dependency)
                        else:
                            lineage.add_dependency(target_name, source)
        
        return lineage
    
    def extract_column_lineage(self, expression: Expression) -> ColumnLineage:
        """Extract column-level lineage from SQL expression."""
        lineage = ColumnLineage()
        alias_mappings = self._extract_table_alias_mappings(expression)
        
        # First, collect all CTE names
        cte_mapping = {}
        for cte in expression.find_all(exp.CTE):
            cte_name = cte.alias
            cte_mapping[cte_name] = cte
        
        # Check if we should skip any CTEs that are simple pass-throughs
        ctes_to_skip = self._identify_passthrough_ctes(expression, cte_mapping)
        
        # Process CTEs (skip the ones identified as pass-throughs)
        for cte in expression.find_all(exp.CTE):
            cte_name = cte.alias
            
            # Skip this CTE if it's a simple pass-through to the main query
            if cte_name in ctes_to_skip:
                continue
            
            self._process_select_for_column_lineage(
                cte, lineage, alias_mappings, target_prefix=cte_name
            )
        
        # Process main query
        if isinstance(expression, exp.Select):
            self._process_select_for_column_lineage(
                expression, lineage, alias_mappings, target_prefix="QUERY_RESULT"
            )
        elif isinstance(expression, exp.Create):
            # Handle CREATE TABLE AS SELECT
            if hasattr(expression, 'this') and expression.this:
                target_table = self._clean_table_reference(str(expression.this))
                
                for select in expression.find_all(exp.Select):
                    self._process_select_for_column_lineage(
                        select, lineage, alias_mappings, target_prefix=target_table
                    )
        
        return lineage
    
    def _extract_table_alias_mappings(self, expression: Expression) -> Dict[str, str]:
        """Extract mapping from table aliases to actual table names."""
        alias_mappings = {}
        
        # Extract CTE aliases
        for cte in expression.find_all(exp.CTE):
            cte_name = cte.alias
            alias_mappings[cte_name] = cte_name
        
        # Extract table aliases
        for alias in expression.find_all(exp.Alias):
            if isinstance(alias.expression, (exp.Table, exp.Identifier)):
                alias_name = alias.alias
                if isinstance(alias.expression, exp.Table):
                    table_name = str(alias.expression)
                else:
                    table_name = alias.expression.name
                
                # Clean up table name
                if " AS " in table_name:
                    table_name = table_name.split(" AS ")[0].strip()
                
                alias_mappings[alias_name] = table_name
        
        # Process SQL text for additional aliases
        sql_text = str(expression)
        from_patterns = [
            r'FROM\s+([^\s,()]+)\s+([^\s,()]+)',
            r'JOIN\s+([^\s,()]+)\s+([^\s,()]+)',
            r'FROM\s+([^\s,()]+)\s+AS\s+([^\s,()]+)',
            r'JOIN\s+([^\s,()]+)\s+AS\s+([^\s,()]+)'
        ]
        
        for pattern in from_patterns:
            matches = re.finditer(pattern, sql_text, re.IGNORECASE)
            for match in matches:
                table_name = match.group(1).strip('"')
                alias_name = match.group(2).strip('"')
                
                if alias_name not in alias_mappings:
                    alias_mappings[alias_name] = table_name
        
        return alias_mappings
    
    def _get_source_tables_from_node(
        self, 
        node: Expression, 
        alias_mappings: Dict[str, str],
        cte_mapping: Dict[str, Expression],
        current_cte_name: Optional[str] = None
    ) -> Set[str]:
        """Get source tables from a SQL node."""
        source_tables = set()
        
        # Determine if we should exclude CTE tables
        # We exclude tables inside OTHER CTEs when processing main query or other CTEs
        # But we include all tables when processing a specific CTE body
        exclude_cte_tables = len(cte_mapping) > 0
        
        tables_to_process = []
        for table in node.find_all(exp.Table):
            if exclude_cte_tables:
                # Check if this table is inside a DIFFERENT CTE definition
                parent = table.parent
                inside_different_cte = False
                while parent:
                    if isinstance(parent, exp.CTE):
                        # Check if this CTE is different from the current one
                        if current_cte_name is None or parent.alias != current_cte_name:
                            inside_different_cte = True
                        break
                    parent = parent.parent
                
                if not inside_different_cte:
                    tables_to_process.append(table)
            else:
                # Processing CTE body, include all tables
                tables_to_process.append(table)
        
        for table in tables_to_process:
            table_name = str(table)
            
            # Clean the table name first (remove " AS alias" part)
            clean_table_name = self._clean_table_reference(table_name)
            base_table_name = clean_table_name
            
            # If this is a CTE reference, add the CTE name directly
            if base_table_name in cte_mapping:
                source_tables.add(base_table_name)
            else:
                # Resolve alias
                resolved_table_name = clean_table_name
                for alias, actual_table in alias_mappings.items():
                    if base_table_name == alias:
                        resolved_table_name = self._clean_table_reference(actual_table)
                        break
                
                source_tables.add(resolved_table_name)
        
        return source_tables
    
    def _get_main_select_without_ctes(self, expression: Expression) -> Expression:
        """Get the main SELECT part excluding CTEs."""
        # If this is a regular select, return a copy without the WITH clause
        if isinstance(expression, exp.Select):
            # Create a copy of the select without WITH clause
            new_select = expression.copy()
            if hasattr(new_select, 'with_'):
                new_select.with_ = None
            return new_select
        return expression
    
    def _process_select_for_column_lineage(
        self,
        select_node: Expression,
        lineage: ColumnLineage,
        alias_mappings: Dict[str, str],
        target_prefix: str
    ) -> None:
        """Process a SELECT node for column lineage."""
        for select in select_node.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Alias):
                    target_column = f"{target_prefix}.{projection.alias}"
                    target_column = self._clean_column_reference(target_column)
                    
                    # Find source columns
                    source_columns = self._extract_source_columns(projection, alias_mappings)
                    
                    for source in source_columns:
                        lineage.add_dependency(target_column, source)
                
                elif isinstance(projection, exp.Column):
                    target_column = f"{target_prefix}.{projection.name}"
                    target_column = self._clean_column_reference(target_column)
                    
                    source_column = str(projection)
                    resolved_source = self._resolve_column_reference(source_column, alias_mappings)
                    resolved_source = self._clean_column_reference(resolved_source)
                    
                    lineage.add_dependency(target_column, resolved_source)
    
    def _extract_source_columns(
        self, 
        projection: Expression, 
        alias_mappings: Dict[str, str]
    ) -> Set[str]:
        """Extract source columns from a projection expression."""
        source_columns = set()
        
        for column in projection.find_all(exp.Column):
            source_column = str(column)
            resolved_source = self._resolve_column_reference(source_column, alias_mappings)
            resolved_source = self._clean_column_reference(resolved_source)
            source_columns.add(resolved_source)
        
        return source_columns
    
    def _resolve_column_reference(self, column_ref: str, alias_mappings: Dict[str, str]) -> str:
        """Resolve table aliases in column references."""
        if "." in column_ref:
            if '"' in column_ref and not column_ref.startswith('"'):
                # Handle quoted identifiers like 'alias."column_name"'
                parts = column_ref.split('.', 1)
                table_ref = parts[0]
                column_name = parts[1]
                
                if " AS " in table_ref:
                    table_ref = table_ref.split(" AS ")[1].strip()
                
                if table_ref in alias_mappings:
                    return f"{alias_mappings[table_ref]}.{column_name}"
            else:
                # Standard case: "alias.column_name"
                parts = column_ref.split(".", 1)
                table_ref = parts[0]
                column_name = parts[1]
                
                if " AS " in table_ref:
                    table_ref = table_ref.split(" AS ")[1].strip()
                
                if table_ref in alias_mappings:
                    return f"{alias_mappings[table_ref]}.{column_name}"
        
        return column_ref
    
    def _clean_table_reference(self, table_ref: str) -> str:
        """Clean up table reference by removing aliases."""
        if " AS " in table_ref:
            table_ref = table_ref.split(" AS ")[0].strip()
        
        # Return the table reference as-is without adding default schema
        return table_ref
    
    def _clean_column_reference(self, column_ref: str) -> str:
        """Clean up column reference."""
        # Handle fully qualified names with schema, table, and column
        if column_ref.count('.') == 2 and '"' in column_ref:
            parts = []
            current_part = ""
            in_quotes = False
            
            for char in column_ref:
                if char == '"':
                    in_quotes = not in_quotes
                    current_part += char
                elif char == '.' and not in_quotes:
                    parts.append(current_part)
                    current_part = ""
                else:
                    current_part += char
            
            if current_part:
                parts.append(current_part)
            
            if len(parts) == 3:
                schema, table, column = parts
                if column.startswith('"') and column.endswith('"'):
                    column = column[1:-1]
                return f"{schema}.{table}.{column}"
        
        # Handle table.column references
        elif "." in column_ref:
            parts = column_ref.split(".", 1)
            table_ref = parts[0]
            column_name = parts[1]
            
            table_ref = self._clean_table_reference(table_ref)
            
            if column_name.startswith('"') and column_name.endswith('"'):
                column_name = column_name[1:-1]
            
            return f"{table_ref}.{column_name}"
        
        return column_ref
    
    def _is_simple_cte_select(self, select_node: Expression, cte_name: str) -> bool:
        """
        Check if this is a simple "SELECT * FROM cte_name" or similar simple case.
        
        Args:
            select_node: The SELECT expression node
            cte_name: The CTE name to check against
            
        Returns:
            True if this is a simple select from the single CTE
        """
        try:
            # Check if the FROM clause references only the CTE
            from_clause = select_node.find(exp.From)
            if not from_clause:
                return False
            
            # Get the table referenced in FROM
            table_ref = from_clause.this
            if not table_ref:
                return False
            
            # Clean the table name
            table_name = str(table_ref)
            clean_name = self._clean_table_reference(table_name)
            
            # Check if it's referencing our CTE and nothing else complex
            return (clean_name == cte_name and 
                    not select_node.find(exp.Join) and  # No JOINs
                    not select_node.find(exp.Where) and # No complex WHERE
                    not select_node.find(exp.Group))    # No GROUP BY
        
        except Exception:
            # If we can't determine, err on the side of caution
            return False
    

    def _identify_passthrough_ctes(self, expression: Expression, cte_mapping: Dict[str, Expression]) -> Set[str]:
        """
        Identify CTEs that are simple pass-throughs to the main query and should be skipped.
        
        Args:
            expression: The main SQL expression
            cte_mapping: Dictionary mapping CTE names to their expressions
            
        Returns:
            Set of CTE names that should be skipped from lineage
        """
        ctes_to_skip = set()
        
        try:
            if not isinstance(expression, exp.Select):
                return ctes_to_skip
            
            # Convert to string and analyze the pattern
            sql_str = str(expression).strip()
            
            # Look for the pattern: WITH ... ) SELECT ... FROM cte_name
            # We want to identify when the main SELECT is just "SELECT * FROM cte_name" (possibly with ORDER BY)
            
            # Find the last ) before SELECT to identify where CTEs end and main query begins
            import re
            
            # Pattern to match: WITH ... ) SELECT ... FROM table_name
            # We're looking for the main SELECT after all CTEs
            cte_pattern = r'WITH\s+.*?\)\s*(SELECT\s+.*)'
            match = re.search(cte_pattern, sql_str, re.IGNORECASE | re.DOTALL)
            
            if match:
                main_query_part = match.group(1).strip()
                
                # Check if this is a simple "SELECT * FROM cte_name" possibly with ORDER BY
                simple_pattern = r'SELECT\s+\*\s+FROM\s+(\w+)(?:\s+ORDER\s+BY\s+.*)?$'
                simple_match = re.match(simple_pattern, main_query_part, re.IGNORECASE | re.DOTALL)
                
                if simple_match:
                    cte_name = simple_match.group(1)
                    
                    if cte_name in cte_mapping:
                        ctes_to_skip.add(cte_name)
                
        except Exception:
            # If we can't determine, err on the side of caution and don't skip
            pass
            
        return ctes_to_skip