"""Core lineage extraction logic."""

import re
from typing import Dict, Set, Optional, List, Tuple
from collections import defaultdict

import sqlglot
from sqlglot import Expression, exp

from .models import (
    TableLineage, ColumnLineage, TableTransformation, ColumnTransformation,
    JoinCondition, FilterCondition, AggregateFunction, WindowFunction, CaseExpression,
    JoinType, AggregateType, OperatorType, TransformationType
)
from ..utils.condition_utils import GenericConditionHandler
from ..utils.sql_parsing_utils import TableNameRegistry, CompatibilityMode
from ..utils.logging_config import get_logger


class LineageExtractor:
    """Extracts lineage information from parsed SQL expressions."""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = CompatibilityMode.FULL):
        self.dialect = dialect
        self.compatibility_mode = compatibility_mode
        self._table_registry = None
        self.logger = get_logger('core.extractor')
    
    def extract_table_lineage(self, expression: Expression) -> TableLineage:
        """Extract table-level lineage from SQL expression."""
        self.logger.info("Starting table lineage extraction")
        
        lineage = TableLineage()
        alias_mappings = self._extract_table_alias_mappings(expression)
        self.logger.debug(f"Found {len(alias_mappings)} table aliases")
        
        # Initialize table registry for this extraction
        self._table_registry = TableNameRegistry(self.dialect, self.compatibility_mode)
        raw_table_names = set()
        
        # Collect all raw table names first
        for table in expression.find_all(exp.Table):
            table_name = str(table)
            clean_table_name = self._clean_table_reference(table_name)
            raw_table_names.add(clean_table_name)
        
        # Register all table names to get canonical mappings
        if raw_table_names:
            self._table_registry.register_tables(raw_table_names)
        
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
            
            # Normalize table names using registry
            normalized_source_tables = self._normalize_table_names(source_tables)
            
            # If this CTE should be skipped, store its dependencies but don't add to lineage
            if cte_name in ctes_to_skip:
                skipped_cte_dependencies[cte_name] = normalized_source_tables
                processed_ctes[cte_name] = cte
                continue
            
            # Always create an entry for this CTE, even if it has no sources
            if not normalized_source_tables:
                # Create empty entry
                if cte_name not in lineage.upstream:
                    lineage.upstream[cte_name] = set()
            else:
                for source in normalized_source_tables:
                    lineage.add_dependency(cte_name, source)
                    
                    # Extract transformation details for this CTE (use original source for transformation analysis)
                    original_source = next((orig for orig in source_tables if self._table_registry.get_canonical_name(orig) == source), source)
                    transformation = self._extract_table_transformation(cte.this, original_source, cte_name, alias_mappings)
                    if transformation:
                        # Update transformation to use canonical names
                        transformation.source_table = source
                        lineage.add_transformation(cte_name, transformation)
            
            # Add this CTE to the processed set
            processed_ctes[cte_name] = cte
        
        # Process main query
        if isinstance(expression, exp.Select):
            target_name = "QUERY_RESULT"
            source_tables = self._get_source_tables_from_node(expression, alias_mappings, cte_mapping)
            normalized_source_tables = self._normalize_table_names(source_tables)
            
            # Check if this is a simple pass-through of a single CTE (SELECT * FROM cte [ORDER BY ...])
            passthrough_cte = self._detect_main_query_passthrough(expression, normalized_source_tables, cte_mapping)
            
            for source in normalized_source_tables:
                # If this source is a skipped CTE, replace it with its dependencies
                if source in ctes_to_skip and source in skipped_cte_dependencies:
                    # Add the dependencies of the skipped CTE instead
                    for skipped_cte_dependency in skipped_cte_dependencies[source]:
                        lineage.add_dependency(target_name, skipped_cte_dependency)
                # If this is a pass-through CTE, connect directly to its dependencies
                elif source == passthrough_cte and source in lineage.upstream:
                    for cte_dependency in lineage.upstream[source]:
                        lineage.add_dependency(target_name, cte_dependency)
                    # Remove the pass-through CTE from lineage
                    del lineage.upstream[source]
                else:
                    lineage.add_dependency(target_name, source)
                    
                    # Extract transformation details for main query (use original source for transformation analysis)
                    original_source = next((orig for orig in source_tables if self._table_registry.get_canonical_name(orig) == source), source)
                    transformation = self._extract_table_transformation(expression, original_source, target_name, alias_mappings)
                    if transformation:
                        # Update transformation to use canonical names
                        transformation.source_table = source
                        lineage.add_transformation(target_name, transformation)
        
        # Handle UNION queries
        elif isinstance(expression, exp.Union):
            target_name = "QUERY_RESULT"
            source_tables = self._get_source_tables_from_node(expression, alias_mappings, cte_mapping)
            normalized_source_tables = self._normalize_table_names(source_tables)
            
            # Determine UNION type (UNION or UNION ALL)
            # In SQLGlot, distinct=False means UNION ALL, distinct=True means UNION
            distinct_flag = expression.args.get('distinct', True)
            union_type = "UNION ALL" if distinct_flag is False else "UNION"
            
            # Add dependencies
            for source in normalized_source_tables:
                # If this source is a skipped CTE, replace it with its dependencies
                if source in ctes_to_skip and source in skipped_cte_dependencies:
                    # Add the dependencies of the skipped CTE instead
                    for skipped_cte_dependency in skipped_cte_dependencies[source]:
                        lineage.add_dependency(target_name, skipped_cte_dependency)
                else:
                    lineage.add_dependency(target_name, source)
            
            # Extract transformations from each SELECT in the UNION
            self._extract_union_transformations(expression, target_name, union_type, alias_mappings, ctes_to_skip, lineage, source_tables)
        
        # Handle CREATE TABLE AS SELECT
        elif isinstance(expression, exp.Create):
            if hasattr(expression, 'this') and expression.this:
                # For CREATE TABLE, use the raw table name without adding default schema
                target_name = str(expression.this)
                
                # Find the SELECT part
                for select in expression.find_all(exp.Select):
                    source_tables = self._get_source_tables_from_node(select, alias_mappings, cte_mapping)
                    normalized_source_tables = self._normalize_table_names(source_tables)
                    for source in normalized_source_tables:
                        # If this source is a skipped CTE, replace it with its dependencies
                        if source in ctes_to_skip and source in skipped_cte_dependencies:
                            # Add the dependencies of the skipped CTE instead
                            for skipped_cte_dependency in skipped_cte_dependencies[source]:
                                lineage.add_dependency(target_name, skipped_cte_dependency)
                        else:
                            lineage.add_dependency(target_name, source)
                            
                            # Extract transformation details for CREATE TABLE AS SELECT (use original source for transformation analysis)
                            original_source = next((orig for orig in source_tables if self._table_registry.get_canonical_name(orig) == source), source)
                            transformation = self._extract_table_transformation(select, original_source, target_name, alias_mappings)
                            if transformation:
                                # Update transformation to use canonical names
                                transformation.source_table = source
                                lineage.add_transformation(target_name, transformation)
        
        self.logger.info(f"Table lineage extraction completed - upstream: {len(lineage.upstream)} entries, downstream: {len(lineage.downstream)} entries")
        return lineage
    
    def _normalize_table_names(self, table_names: Set[str]) -> Set[str]:
        """Normalize table names using the registry to eliminate duplicates."""
        if not self._table_registry:
            return table_names
        
        normalized_names = set()
        for table_name in table_names:
            canonical_name = self._table_registry.get_canonical_name(table_name)
            normalized_names.add(canonical_name)
        
        return normalized_names
    
    def extract_column_lineage(self, expression: Expression) -> ColumnLineage:
        """Extract column-level lineage from SQL expression."""
        self.logger.info("Starting column lineage extraction")
        
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
        elif isinstance(expression, exp.Union):
            # Handle UNION queries
            self._process_union_for_column_lineage(
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
        
        self.logger.info(f"Column lineage extraction completed - upstream: {len(lineage.upstream)} entries, downstream: {len(lineage.downstream)} entries")
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
                        
                        # Extract column transformation details
                        transformation = self._extract_column_transformation(projection, source, target_column, alias_mappings)
                        if transformation:
                            lineage.add_transformation(target_column, transformation)
                
                elif isinstance(projection, exp.Column):
                    target_column = f"{target_prefix}.{projection.name}"
                    target_column = self._clean_column_reference(target_column)
                    
                    source_column = str(projection)
                    resolved_source = self._resolve_column_reference(source_column, alias_mappings)
                    resolved_source = self._clean_column_reference(resolved_source)
                    
                    lineage.add_dependency(target_column, resolved_source)
                    
                    # Create simple transformation for direct column reference
                    transformation = ColumnTransformation(
                        source_column=resolved_source,
                        target_column=target_column,
                        expression=str(projection)
                    )
                    lineage.add_transformation(target_column, transformation)
    
    def _process_union_for_column_lineage(
        self,
        union_node: Expression,
        lineage: ColumnLineage,
        alias_mappings: Dict[str, str],
        target_prefix: str
    ) -> None:
        """Process a UNION node for column lineage."""
        try:
            # Collect all SELECT statements in the UNION
            select_statements = []
            self._collect_union_selects(union_node, select_statements)
            
            # Process each SELECT statement and extract column mappings
            # For UNION, we need to map columns by position since they may have different names
            column_positions = {}  # position -> target_column_name
            
            for i, select_stmt in enumerate(select_statements):
                # Process each projection in this SELECT
                for pos, projection in enumerate(select_stmt.expressions):
                    if isinstance(projection, exp.Alias):
                        # For the first SELECT, establish the target column names
                        if i == 0:
                            target_column = f"{target_prefix}.{projection.alias}"
                            target_column = self._clean_column_reference(target_column)
                            column_positions[pos] = target_column
                        else:
                            # For subsequent SELECTs, use the column name established by the first SELECT
                            target_column = column_positions.get(pos)
                            if not target_column:
                                continue
                        
                        # Check if this is a literal alias (like 'user' as type)
                        if isinstance(projection.this, exp.Literal):
                            # Handle literal aliases
                            transformation = ColumnTransformation(
                                source_column="LITERAL",
                                target_column=target_column,
                                expression=str(projection)
                            )
                            lineage.add_transformation(target_column, transformation)
                        else:
                            # Find source columns for this projection (non-literal)
                            source_columns = self._extract_source_columns(projection, alias_mappings)
                            
                            for source in source_columns:
                                lineage.add_dependency(target_column, source)
                                
                                # Extract column transformation details
                                transformation = self._extract_column_transformation(
                                    projection, source, target_column, alias_mappings
                                )
                                if transformation:
                                    lineage.add_transformation(target_column, transformation)
                    
                    elif isinstance(projection, exp.Column):
                        # For the first SELECT, establish the target column names
                        if i == 0:
                            target_column = f"{target_prefix}.{projection.name}"
                            target_column = self._clean_column_reference(target_column)
                            column_positions[pos] = target_column
                        else:
                            # For subsequent SELECTs, use the column name established by the first SELECT
                            target_column = column_positions.get(pos)
                            if not target_column:
                                continue
                        
                        source_column = str(projection)
                        resolved_source = self._resolve_column_reference(source_column, alias_mappings)
                        resolved_source = self._clean_column_reference(resolved_source)
                        
                        lineage.add_dependency(target_column, resolved_source)
                        
                        # Create simple transformation for direct column reference
                        transformation = ColumnTransformation(
                            source_column=resolved_source,
                            target_column=target_column,
                            expression=str(projection)
                        )
                        lineage.add_transformation(target_column, transformation)
                    
                    elif isinstance(projection, exp.Literal):
                        # Handle literal values (like 'user', 'product', 'category')
                        if i == 0:
                            # For literal, we need to infer the column name from the context
                            # Use a generic name based on position if no alias
                            target_column = f"{target_prefix}._literal_{pos}"
                            target_column = self._clean_column_reference(target_column)
                            column_positions[pos] = target_column
                        else:
                            target_column = column_positions.get(pos)
                            if not target_column:
                                continue
                        
                        # For literals, create a computed transformation
                        transformation = ColumnTransformation(
                            source_column="LITERAL",
                            target_column=target_column,
                            expression=str(projection)
                        )
                        lineage.add_transformation(target_column, transformation)
        
        except Exception:
            # If UNION column processing fails, fall back to basic processing
            pass
    
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
        
        For nested CTE visualization, we want to show all CTEs in the chain, so we're more
        conservative about skipping CTEs.
        
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
            
            # For nested CTE visualization, we want to be much more conservative
            # Only skip CTEs in very specific cases where there's only ONE CTE
            # and it's a true passthrough with no transformations
            # 
            # However, we also want to skip CTEs that are simple pass-throughs even in multi-CTE scenarios
            # if they just add ORDER BY or other non-transformational operations
            
            # Check each CTE individually for pass-through patterns
            for cte_name, cte_node in cte_mapping.items():
                if self._is_simple_passthrough_cte(cte_node, expression, cte_name):
                    ctes_to_skip.add(cte_name)
            
            # If we have multiple CTEs but only found pass-throughs among the final ones, that's still valid
            # Example: WITH a AS (...), b AS (...) SELECT * FROM b ORDER BY col
            # Here 'b' could be a pass-through if it just does SELECT * FROM a with minor changes
            
                
        except Exception:
            # If we can't determine, err on the side of caution and don't skip
            pass
            
        return ctes_to_skip
    
    def _is_simple_passthrough_cte(self, cte_node: Expression, main_expression: Expression, cte_name: str) -> bool:
        """Check if a CTE and main query form a simple passthrough pattern."""
        try:
            # Check if CTE is simple (SELECT * FROM single_table with no transformations)
            cte_select = cte_node.this
            if not isinstance(cte_select, exp.Select):
                return False
            
            # CTE should have no JOINs, GROUP BY, complex WHERE, etc.
            if (cte_select.find(exp.Join) or 
                cte_select.find(exp.Group) or 
                cte_select.find(exp.Having) or
                cte_select.find(exp.Window)):
                return False
            
            # Check if main query is simple SELECT * FROM cte_name (with optional ORDER BY)
            main_sql = str(main_expression).strip()
            import re
            
            # Allow SELECT * FROM cte_name with optional ORDER BY (which is still a pass-through)
            simple_main_pattern = fr'SELECT\s+\*\s+FROM\s+{re.escape(cte_name)}(?:\s+ORDER\s+BY\s+[^\s]+(?:\s+(?:ASC|DESC))?)?(?:\s+(?:LIMIT|OFFSET)\s+\d+)*\s*$'
            if not re.search(simple_main_pattern, main_sql, re.IGNORECASE):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _detect_main_query_passthrough(self, expression: exp.Select, source_tables: Set[str], cte_mapping: Dict[str, exp.Expression]) -> Optional[str]:
        """
        Detect if the main query is a simple pass-through of a single CTE.
        
        Returns the CTE name if this is a pass-through, None otherwise.
        """
        try:
            # Must reference exactly one table and it must be a CTE
            if len(source_tables) != 1:
                return None
            
            source_table = list(source_tables)[0]
            if source_table not in cte_mapping:
                return None
            
            # Check if the main query is simple: SELECT * FROM cte [ORDER BY ...] [LIMIT ...]
            main_sql = str(expression).strip()
            
            # Remove any WITH clause from the main SQL for pattern matching
            import re
            # Find the main SELECT part (after WITH clause)
            with_match = re.search(r'WITH\s+.*?\s+SELECT', main_sql, re.IGNORECASE | re.DOTALL)
            if with_match:
                # Extract just the SELECT part
                select_part = main_sql[with_match.end()-6:].strip()  # -6 to include "SELECT"
            else:
                select_part = main_sql
            
            # Pattern for SELECT * FROM cte_name with optional ORDER BY and LIMIT
            passthrough_pattern = fr'SELECT\s+\*\s+FROM\s+{re.escape(source_table)}(?:\s+ORDER\s+BY\s+[^\s]+(?:\s+(?:ASC|DESC))?)?(?:\s+(?:LIMIT|OFFSET)\s+\d+)*\s*$'
            
            if re.search(passthrough_pattern, select_part, re.IGNORECASE):
                return source_table
                
            return None
            
        except Exception:
            return None
    
    def _extract_table_transformation(
        self, 
        select_node: Expression, 
        source_table: str, 
        target_table: str, 
        alias_mappings: Dict[str, str]
    ) -> Optional[TableTransformation]:
        """Extract table-level transformation details."""
        try:
            transformation = TableTransformation(
                source_table=source_table,
                target_table=target_table
            )
            
            # Extract JOIN information
            for join in select_node.find_all(exp.Join):
                join_type = self._get_join_type(join)
                join_conditions = self._extract_join_conditions(join, alias_mappings)
                
                transformation.join_type = join_type
                transformation.join_conditions.extend(join_conditions)
            
            # Extract WHERE conditions
            where_clause = select_node.find(exp.Where)
            if where_clause:
                filter_conditions = self._extract_filter_conditions(where_clause, alias_mappings)
                transformation.filter_conditions.extend(filter_conditions)
            
            # Extract GROUP BY
            group_by = select_node.find(exp.Group)
            if group_by:
                group_columns = [self._resolve_column_reference(str(expr), alias_mappings) 
                               for expr in group_by.expressions]
                transformation.group_by_columns = group_columns
            
            # Extract HAVING conditions
            having = select_node.find(exp.Having)
            if having:
                having_conditions = self._extract_filter_conditions(having, alias_mappings)
                transformation.having_conditions.extend(having_conditions)
            
            # Extract ORDER BY
            order_by = select_node.find(exp.Order)
            if order_by:
                order_columns = [self._resolve_column_reference(str(expr), alias_mappings) 
                               for expr in order_by.expressions]
                transformation.order_by_columns = order_columns
            
            return transformation
            
        except Exception:
            return None
    
    def _extract_column_transformation(
        self, 
        projection: Expression, 
        source_column: str, 
        target_column: str, 
        alias_mappings: Dict[str, str]
    ) -> Optional[ColumnTransformation]:
        """Extract column-level transformation details."""
        try:
            transformation = ColumnTransformation(
                source_column=source_column,
                target_column=target_column,
                expression=str(projection.this) if hasattr(projection, 'this') else str(projection)
            )
            
            # Check for aggregate functions
            for agg in projection.find_all(exp.AggFunc):
                agg_type = self._get_aggregate_type(agg)
                agg_column = None
                distinct = False
                
                if agg.expressions:
                    agg_column = str(agg.expressions[0])
                    agg_column = self._resolve_column_reference(agg_column, alias_mappings)
                
                if hasattr(agg, 'distinct') and agg.distinct:
                    distinct = True
                
                transformation.aggregate_function = AggregateFunction(
                    function_type=agg_type,
                    column=agg_column,
                    distinct=distinct
                )
                break  # Only handle first aggregate for now
            
            # Check for window functions
            for window in projection.find_all(exp.Window):
                window_func = WindowFunction(
                    function_name=str(window.this) if hasattr(window, 'this') else 'UNKNOWN'
                )
                
                # Extract window spec
                spec = window.args.get('spec')
                if spec:
                    if spec.partition_by:
                        window_func.partition_by = [
                            self._resolve_column_reference(str(p), alias_mappings) 
                            for p in spec.partition_by
                        ]
                    if spec.order:
                        window_func.order_by = [
                            self._resolve_column_reference(str(o), alias_mappings) 
                            for o in spec.order.expressions
                        ]
                
                transformation.window_function = window_func
                break  # Only handle first window function for now
            
            # Check for CASE expressions
            for case in projection.find_all(exp.Case):
                case_expr = CaseExpression()
                
                # Extract WHEN conditions (simplified)
                when_conditions = []
                then_values = []
                
                # Get the case conditions from args
                if hasattr(case, 'ifs') and case.ifs:
                    for if_clause in case.ifs:
                        # This is a simplified extraction - in reality, CASE conditions can be very complex
                        when_conditions.append(FilterCondition(
                            column="CASE_CONDITION",
                            operator=OperatorType.EQ,
                            value=str(if_clause.this)
                        ))
                        then_values.append(str(if_clause.expression))
                
                if case.default:
                    case_expr.else_value = str(case.default)
                
                case_expr.when_conditions = when_conditions
                case_expr.then_values = then_values
                
                transformation.case_expression = case_expr
                break  # Only handle first CASE for now
            
            return transformation
            
        except Exception:
            return None
    
    def _get_join_type(self, join: exp.Join) -> JoinType:
        """Extract JOIN type from JOIN expression.
        
        Preserves the exact syntax as written by the user:
        - LEFT JOIN → JoinType.LEFT  
        - LEFT OUTER JOIN → JoinType.LEFT_OUTER
        - RIGHT JOIN → JoinType.RIGHT
        - RIGHT OUTER JOIN → JoinType.RIGHT_OUTER
        - etc.
        """
        join_side = getattr(join, 'side', None)
        join_kind = getattr(join, 'kind', None)
        
        # Handle explicit OUTER JOIN variants
        if join_kind == 'OUTER':
            if join_side == 'LEFT':
                return JoinType.LEFT_OUTER
            elif join_side == 'RIGHT':
                return JoinType.RIGHT_OUTER
            elif join_side == 'FULL':
                return JoinType.FULL_OUTER
        
        # Handle other JOIN types
        if join_side == 'LEFT':
            return JoinType.LEFT
        elif join_side == 'RIGHT':
            return JoinType.RIGHT
        elif join_side == 'FULL':
            return JoinType.FULL
        elif join_side == 'CROSS':
            return JoinType.CROSS
        elif join_kind == 'INNER':
            return JoinType.INNER
        else:
            # Plain JOIN without qualifiers defaults to INNER
            return JoinType.INNER
    
    def _extract_join_conditions(self, join: exp.Join, alias_mappings: Dict[str, str]) -> List[JoinCondition]:
        """Extract JOIN conditions."""
        conditions = []
        
        on_condition = join.args.get('on')
        if on_condition:
            # Extract all join conditions using generic handler (not just EQ) in uniform dict format
            join_condition_dicts = GenericConditionHandler.extract_join_conditions(
                on_condition, 
                output_format="dict"
            )
            
            # Convert to JoinCondition objects for this caller
            for cond_dict in join_condition_dicts:
                # Convert string operator back to OperatorType enum
                op_enum = next((op for op in OperatorType if op.value == cond_dict["operator"]), OperatorType.EQ)
                condition = JoinCondition(
                    left_column=self._resolve_column_reference(cond_dict["left_column"], alias_mappings),
                    operator=op_enum,
                    right_column=self._resolve_column_reference(cond_dict["right_column"], alias_mappings)
                )
                conditions.append(condition)
        
        return conditions
    
    def _extract_filter_conditions(self, filter_node: Expression, alias_mappings: Dict[str, str]) -> List[FilterCondition]:
        """Extract filter conditions from WHERE or HAVING clause."""
        print(f"DEBUG: Extracting filter conditions from: {str(filter_node)[:100]}...")
        
        try:
            # Use generic condition handler with column resolver
            def column_resolver(column: str) -> str:
                return self._resolve_column_reference(column, alias_mappings)
            
            # Extract conditions using generic handler in uniform dict format
            condition_dicts = GenericConditionHandler.extract_all_conditions(
                filter_node, 
                column_resolver=column_resolver, 
                output_format="dict"
            )
            
            # Convert to FilterCondition objects for this caller
            conditions = []
            for cond_dict in condition_dicts:
                # Convert string operator back to OperatorType enum
                op_enum = next((op for op in OperatorType if op.value == cond_dict["operator"]), OperatorType.EQ)
                conditions.append(FilterCondition(
                    column=cond_dict["column"],
                    operator=op_enum,
                    value=cond_dict["value"]
                ))
            
            # Handle recursive subquery extraction for IN clauses
            for in_expr in filter_node.find_all(exp.In):
                for expr in in_expr.expressions:
                    if isinstance(expr, exp.Select):
                        # Found a subquery - recursively extract its WHERE conditions
                        subquery_where = expr.find(exp.Where)
                        if subquery_where:
                            subquery_conditions = self._extract_filter_conditions(subquery_where, alias_mappings)
                            conditions.extend(subquery_conditions)
        
        except Exception:
            conditions = []
        
        print(f"DEBUG: Found {len(conditions)} filter conditions: {[(c.column, c.operator, c.value) for c in conditions]}")
        return conditions
    
    def _get_aggregate_type(self, agg: exp.AggFunc) -> AggregateType:
        """Get aggregate function type."""
        agg_name = type(agg).__name__.upper()
        
        if agg_name == 'COUNT':
            return AggregateType.COUNT
        elif agg_name == 'SUM':
            return AggregateType.SUM
        elif agg_name == 'AVG':
            return AggregateType.AVG
        elif agg_name == 'MIN':
            return AggregateType.MIN
        elif agg_name == 'MAX':
            return AggregateType.MAX
        elif agg_name == 'STDDEV':
            return AggregateType.STDDEV
        elif agg_name == 'VARIANCE':
            return AggregateType.VARIANCE
        elif 'APPROX' in agg_name and 'DISTINCT' in agg_name:
            return AggregateType.APPROX_DISTINCT
        else:
            return AggregateType.OTHER
    
    def _create_union_transformation(self, source_table: str, target_table: str, 
                                   all_sources: List[str], union_type: str, 
                                   filter_conditions: List = None) -> Optional[TableTransformation]:
        """Create a UNION transformation."""
        try:
            transformation_type = TransformationType.UNION_ALL if union_type == "UNION ALL" else TransformationType.UNION
            
            return TableTransformation(
                source_table=source_table,
                target_table=target_table,
                transformation_type=transformation_type,
                union_sources=[source_table],  # Only include the specific source table
                union_type=union_type,
                filter_conditions=filter_conditions or []
            )
        except Exception:
            return None
    
    def _extract_union_transformations(self, union_expr: Expression, target_name: str, 
                                     union_type: str, alias_mappings: Dict[str, str], 
                                     ctes_to_skip: Set[str], lineage, original_source_tables: Set[str] = None) -> None:
        """Extract transformations from each SELECT statement in a UNION."""
        try:
            # Get all source tables in the entire UNION for context
            all_union_sources = list(self._get_source_tables_from_node(union_expr, alias_mappings, {}))
            
            # Collect all SELECT statements in the UNION
            select_statements = []
            self._collect_union_selects(union_expr, select_statements)
            
            # Extract transformation from each SELECT
            for select_stmt in select_statements:
                # Get source tables for this SELECT
                source_tables = self._get_source_tables_from_node(select_stmt, alias_mappings, {})
                normalized_source_tables = self._normalize_table_names(source_tables)
                
                for source in normalized_source_tables:
                    if source not in ctes_to_skip:
                        # Extract full transformation details including filter conditions (use original source for transformation analysis)
                        original_source = next((orig for orig in source_tables if self._table_registry.get_canonical_name(orig) == source), source)
                        transformation = self._extract_table_transformation(
                            select_stmt, original_source, target_name, alias_mappings
                        )
                        
                        if transformation:
                            # Convert to UNION transformation by updating type and adding union info
                            union_transformation = TableTransformation(
                                source_table=source,  # Use canonical name
                                target_table=transformation.target_table,
                                transformation_type=TransformationType.UNION_ALL if union_type == "UNION ALL" else TransformationType.UNION,
                                filter_conditions=transformation.filter_conditions,
                                group_by_columns=transformation.group_by_columns,
                                join_type=transformation.join_type,
                                join_conditions=transformation.join_conditions,
                                having_conditions=transformation.having_conditions,
                                order_by_columns=transformation.order_by_columns,
                                union_sources=[source],  # Use canonical name
                                union_type=union_type
                            )
                            lineage.add_transformation(target_name, union_transformation)
        except Exception as e:
            # If extraction fails, fall back to basic UNION transformations
            source_tables = self._get_source_tables_from_node(union_expr, alias_mappings, {}) if original_source_tables is None else original_source_tables
            normalized_source_tables = self._normalize_table_names(source_tables)
            for source in normalized_source_tables:
                if source not in ctes_to_skip:
                    union_transformation = self._create_union_transformation(
                        source, target_name, normalized_source_tables, union_type
                    )
                    if union_transformation:
                        lineage.add_transformation(target_name, union_transformation)
    
    def _collect_union_selects(self, union_expr: Expression, select_statements: List) -> None:
        """Recursively collect all SELECT statements from a UNION expression."""
        try:
            if hasattr(union_expr, 'left') and union_expr.left:
                if isinstance(union_expr.left, exp.Union):
                    self._collect_union_selects(union_expr.left, select_statements)
                elif isinstance(union_expr.left, exp.Select):
                    select_statements.append(union_expr.left)
            
            if hasattr(union_expr, 'right') and union_expr.right:
                if isinstance(union_expr.right, exp.Union):
                    self._collect_union_selects(union_expr.right, select_statements)
                elif isinstance(union_expr.right, exp.Select):
                    select_statements.append(union_expr.right)
        except Exception:
            pass