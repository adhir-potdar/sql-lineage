"""Core lineage extraction logic."""

import re
from typing import Dict, Set, Optional, List, Tuple
from collections import defaultdict

import sqlglot
from sqlglot import Expression, exp

from .models import (
    TableLineage, ColumnLineage, TableTransformation, ColumnTransformation,
    JoinCondition, FilterCondition, AggregateFunction, WindowFunction, CaseExpression,
    JoinType, AggregateType, OperatorType
)


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
                    
                    # Extract transformation details for this CTE
                    transformation = self._extract_table_transformation(cte.this, source, cte_name, alias_mappings)
                    if transformation:
                        lineage.add_transformation(cte_name, transformation)
            
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
                    
                    # Extract transformation details for main query
                    transformation = self._extract_table_transformation(expression, source, target_name, alias_mappings)
                    if transformation:
                        lineage.add_transformation(target_name, transformation)
        
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
                            
                            # Extract transformation details for CREATE TABLE AS SELECT
                            transformation = self._extract_table_transformation(select, source, target_name, alias_mappings)
                            if transformation:
                                lineage.add_transformation(target_name, transformation)
        
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
            
            if len(cte_mapping) > 1:
                # If there are multiple CTEs, don't skip any - we want to show the full chain
                return ctes_to_skip
            
            # Only consider skipping if there's exactly one CTE
            if len(cte_mapping) == 1:
                cte_name = list(cte_mapping.keys())[0]
                cte_node = cte_mapping[cte_name]
                
                # Check if the CTE is a simple passthrough (like SELECT * FROM table)
                # and the main query is also simple (SELECT * FROM cte)
                if self._is_simple_passthrough_cte(cte_node, expression, cte_name):
                    ctes_to_skip.add(cte_name)
                
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
            
            # Check if main query is simple SELECT * FROM cte_name
            main_sql = str(main_expression).strip()
            import re
            
            # Very restrictive pattern for main query
            simple_main_pattern = fr'SELECT\s+\*\s+FROM\s+{re.escape(cte_name)}\s*$'
            if not re.search(simple_main_pattern, main_sql, re.IGNORECASE):
                return False
            
            return True
            
        except Exception:
            return False
    
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
        """Extract JOIN type from JOIN expression."""
        join_side = getattr(join, 'side', None)
        if join_side == 'LEFT':
            return JoinType.LEFT
        elif join_side == 'RIGHT':
            return JoinType.RIGHT
        elif join_side == 'FULL':
            return JoinType.FULL
        elif join_side == 'CROSS':
            return JoinType.CROSS
        else:
            return JoinType.INNER
    
    def _extract_join_conditions(self, join: exp.Join, alias_mappings: Dict[str, str]) -> List[JoinCondition]:
        """Extract JOIN conditions."""
        conditions = []
        
        on_condition = join.args.get('on')
        if on_condition:
            # Extract equality conditions from JOIN ON clause
            for eq in on_condition.find_all(exp.EQ):
                left_col = self._resolve_column_reference(str(eq.this), alias_mappings)
                right_col = self._resolve_column_reference(str(eq.expression), alias_mappings)
                
                conditions.append(JoinCondition(
                    left_column=left_col,
                    operator=OperatorType.EQ,
                    right_column=right_col
                ))
        
        return conditions
    
    def _extract_filter_conditions(self, filter_node: Expression, alias_mappings: Dict[str, str]) -> List[FilterCondition]:
        """Extract filter conditions from WHERE or HAVING clause."""
        conditions = []
        
        try:
            # Extract various comparison operators
            comparison_types = [
                (exp.EQ, OperatorType.EQ),
                (exp.NEQ, OperatorType.NEQ),
                (exp.GT, OperatorType.GT),
                (exp.GTE, OperatorType.GTE),
                (exp.LT, OperatorType.LT),
                (exp.LTE, OperatorType.LTE),
                (exp.In, OperatorType.IN),
                (exp.Like, OperatorType.LIKE),
                (exp.Between, OperatorType.BETWEEN)
            ]
            
            for exp_type, op_type in comparison_types:
                for comp in filter_node.find_all(exp_type):
                    if exp_type == exp.Between:
                        # Handle BETWEEN specially
                        column = self._resolve_column_reference(str(comp.this), alias_mappings)
                        value = [str(comp.low), str(comp.high)]
                    elif exp_type == exp.In:
                        # Handle IN specially
                        column = self._resolve_column_reference(str(comp.this), alias_mappings)
                        value = [str(expr) for expr in comp.expressions]
                    else:
                        # Handle regular binary comparisons
                        column = self._resolve_column_reference(str(comp.this), alias_mappings)
                        value = str(comp.expression)
                    
                    conditions.append(FilterCondition(
                        column=column,
                        operator=op_type,
                        value=value
                    ))
        
        except Exception:
            pass
        
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