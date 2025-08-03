"""Chain building engine for SQL lineage analysis."""

from typing import Dict, List, Any, Optional, Set
import sqlglot
from ..utils.sql_parsing_utils import is_column_from_table, extract_function_type, extract_clean_column_name, is_subquery_expression, is_subquery_relevant_to_table, extract_columns_referenced_by_table_in_union
from ..utils.metadata_utils import create_metadata_entry
from ..utils.regex_patterns import is_aggregate_function
from ..utils.aggregate_utils import is_aggregate_function_for_table, extract_alias_from_expression
from .analyzers.derived_table_analyzer import DerivedTableAnalyzer
from .transformation_engine import TransformationEngine

class ChainBuilderEngine:
    """Engine for building lineage chains from dependencies."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize the chain builder engine."""
        self.dialect = dialect
        self.derived_table_analyzer = DerivedTableAnalyzer(dialect)
        self.transformation_engine = TransformationEngine(dialect)
    
    def build_chain_from_dependencies(self, entity_name: str, entity_type: str, 
                                    table_lineage_data: Dict, column_lineage_data: Dict,
                                    result, sql: str, current_depth: int = 0, 
                                    visited_in_path: Set = None, depth: int = 0,
                                    parent_entity: str = None) -> Dict[str, Any]:
        """
        Build comprehensive chain with metadata and transformations.
        Consolidated from lineage_chain_builder.py.
        """
        if visited_in_path is None:
            visited_in_path = set()
            
        # Standard chain building
        chain = {
            "entity": entity_name,
            "entity_type": entity_type,
            "depth": current_depth - 1,
            "dependencies": [],
            "metadata": {"table_columns": []}
        }
        
        # Add table metadata if available
        if hasattr(result, 'metadata') and entity_name in result.metadata:
            table_meta = result.metadata[entity_name]
            metadata = {
                "table_type": table_meta.table_type.value
            }
            
            # Only include non-null values to keep output clean
            if table_meta.schema:
                metadata["schema"] = table_meta.schema
            if table_meta.description:
                metadata["description"] = table_meta.description
            
            # Update existing metadata with table metadata
            chain["metadata"].update(metadata)
        
        # Process table dependencies
        if entity_type == "table" and entity_name in table_lineage_data:
            for dependent_table in table_lineage_data[entity_name]:
                # Skip if would cause circular dependency or exceed depth
                if (depth > 0 and current_depth > depth) or dependent_table in visited_in_path:
                    continue
                
                # Build dependency chain
                visited_in_path_new = visited_in_path | {entity_name}
                dep_chain = self.build_chain_from_dependencies(
                    dependent_table, "table", table_lineage_data, column_lineage_data,
                    result, sql, current_depth + 1, visited_in_path_new, depth, entity_name
                )
                
                # Add transformations to dependency
                dep_chain = self._add_transformations_to_chain(
                    dep_chain, entity_name, dependent_table, result, sql
                )
                
                chain["dependencies"].append(dep_chain)
        
        return chain
    
    def _add_transformations_to_chain(self, dep_chain: Dict, entity_name: str, 
                                    dependent_table: str, result, sql: str) -> Dict:
        """Add transformation information to a dependency chain."""
        if hasattr(result, 'table_lineage') and hasattr(result.table_lineage, 'transformations'):
            transformations = []
            for transformation_list in result.table_lineage.transformations.values():
                for trans in transformation_list:
                    # Create transformation data structure matching original format
                    trans_data = {
                        "type": "table_transformation",
                        "source_table": trans.source_table,
                        "target_table": trans.target_table
                    }
                    
                    # Add join information if present
                    if hasattr(trans, 'join_conditions') and trans.join_conditions:
                        join_entry = self._build_join_entry(trans)
                        trans_data["joins"] = [join_entry]
                    
                    # Add filter conditions
                    trans_data = self._add_filter_conditions(trans_data, trans, entity_name, sql)
                    
                    # Add group by columns
                    trans_data = self._add_group_by_columns(trans_data, trans, entity_name, sql)
                    
                    
                    # Add having conditions
                    trans_data = self._add_having_conditions(trans_data, trans, entity_name, sql)
                    
                    # Add order by columns
                    trans_data = self._add_order_by_columns(trans_data, trans, entity_name, sql)
                    
                    transformations.append(trans_data)
            
            # Filter transformations to only include those relevant to this entity
            relevant_transformations = []
            for trans in transformations:
                if (trans.get("source_table") == entity_name and 
                    trans.get("target_table") == dependent_table):
                    relevant_transformations.append(trans)
            
            if relevant_transformations:
                dep_chain["transformations"] = relevant_transformations
        
        return dep_chain
    
    def _build_join_entry(self, trans) -> Dict:
        """Build join entry from transformation data."""
        join_entry = {
            "join_type": trans.join_type.value if hasattr(trans, 'join_type') and trans.join_type else "INNER JOIN",
            "right_table": None,
            "conditions": [
                {
                    "left_column": jc.left_column,
                    "operator": jc.operator.value if hasattr(jc.operator, 'value') else str(jc.operator),
                    "right_column": jc.right_column
                }
                for jc in trans.join_conditions
            ]
        }
        
        # Extract right table from first condition
        if trans.join_conditions:
            first_condition = trans.join_conditions[0]
            if hasattr(first_condition, 'right_column') and '.' in first_condition.right_column:
                right_table = first_condition.right_column.split('.')[0]
                join_entry["right_table"] = right_table
        
        return join_entry
    
    def _add_filter_conditions(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add filter conditions to transformation data."""
        print(f"DEBUG: _add_filter_conditions called for {entity_name}, trans has filter_conditions: {hasattr(trans, 'filter_conditions')}")
        if hasattr(trans, 'filter_conditions'):
            print(f"DEBUG: Filter conditions count: {len(trans.filter_conditions) if trans.filter_conditions else 0}")
        
        if hasattr(trans, 'filter_conditions') and trans.filter_conditions:
            print(f"DEBUG: Processing filter conditions for {entity_name}: {[(fc.column, fc.operator, fc.value) for fc in trans.filter_conditions]}")
            # Determine context for column filtering
            is_single_table = (
                trans.target_table == "QUERY_RESULT" or  # Regular SELECT
                (trans.source_table != trans.target_table and  # CTAS/CTE
                 trans.target_table != "QUERY_RESULT")
            )
            
            context_info = {
                'is_single_table_context': is_single_table,
                'tables_in_context': [trans.source_table] if is_single_table else [],
                'sql': sql
            }
            
            relevant_filters = []
            for fc in trans.filter_conditions:
                # Only include filter conditions that reference columns from this entity
                column_belongs = is_column_from_table(fc.column, entity_name, sql, self.dialect)
                print(f"DEBUG: Filter condition '{fc.column}' belongs to {entity_name}? {column_belongs}")
                if column_belongs:
                    relevant_filters.append({
                        "column": fc.column,
                        "operator": fc.operator.value if hasattr(fc.operator, 'value') else str(fc.operator),
                        "value": fc.value
                    })
            
            if relevant_filters:
                trans_data["filter_conditions"] = relevant_filters
        
        return trans_data
    
    def _add_group_by_columns(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add group by columns to transformation data."""
        if hasattr(trans, 'group_by_columns') and trans.group_by_columns:
            # Determine context for column filtering
            is_single_table = (
                trans.target_table == "QUERY_RESULT" or
                (trans.source_table != trans.target_table and trans.target_table != "QUERY_RESULT")
            )
            
            context_info = {
                'is_single_table_context': is_single_table,
                'tables_in_context': [trans.source_table] if is_single_table else [],
                'sql': sql
            }
            
            relevant_group_by = []
            for col in trans.group_by_columns:
                if is_column_from_table(col, entity_name, sql, self.dialect):
                    relevant_group_by.append(col)
            
            if relevant_group_by:
                trans_data["group_by_columns"] = relevant_group_by
        
        return trans_data
    
    def _add_having_conditions(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add having conditions to transformation data."""
        if hasattr(trans, 'having_conditions') and trans.having_conditions:
            relevant_having = []
            for hc in trans.having_conditions:
                # Having conditions often involve aggregations like COUNT(*) or AVG(u.salary)
                # Check if they reference this entity or if they are general aggregations for this table
                is_relevant = (is_column_from_table(hc.column, entity_name, sql, self.dialect) or 
                             self._is_aggregate_function_for_table(hc.column, entity_name, sql))
                if is_relevant:
                    relevant_having.append({
                        "column": hc.column,
                        "operator": hc.operator.value if hasattr(hc.operator, 'value') else str(hc.operator),
                        "value": hc.value
                    })
            
            if relevant_having:
                trans_data["having_conditions"] = relevant_having
        
        return trans_data
    
    def _add_order_by_columns(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add order by columns to transformation data."""
        if hasattr(trans, 'order_by_columns') and trans.order_by_columns:
            # Determine context for column filtering
            is_single_table = (
                trans.target_table == "QUERY_RESULT" or
                (trans.source_table != trans.target_table and trans.target_table != "QUERY_RESULT")
            )
            
            context_info = {
                'is_single_table_context': is_single_table,
                'tables_in_context': [trans.source_table] if is_single_table else [],
                'sql': sql
            }
            
            relevant_order_by = []
            for col in trans.order_by_columns:
                # Extract just the column name part (before ASC/DESC)
                col_name = col.split()[0] if ' ' in col else col
                if is_column_from_table(col_name, entity_name, sql, self.dialect):
                    relevant_order_by.append(col)
            
            if relevant_order_by:
                trans_data["order_by_columns"] = relevant_order_by
        
        return trans_data
    
    def _is_aggregate_function_for_table(self, column_expr: str, table_name: str, sql: str = None) -> bool:
        """Check if an aggregate function expression is relevant to a specific table."""
        if not column_expr or not table_name:
            return False
        
        # Check if the expression contains aggregate functions
        aggregate_functions = ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'GROUP_CONCAT(']
        has_aggregate = any(func in column_expr.upper() for func in aggregate_functions)
        
        if not has_aggregate:
            return False
        
        # If it's a simple COUNT(*), it could apply to any table
        if 'COUNT(*)' in column_expr.upper():
            return True
        
        # Check if the expression references columns from this table
        if sql:
            return is_column_from_table(column_expr, table_name, sql, self.dialect)
        
        return False
    
    def create_dependency_chain(self, entity_name: str, entity_type: str, depth: int,
                              metadata: Dict = None, transformations: List = None) -> Dict[str, Any]:
        """Create a basic dependency chain structure."""
        chain = {
            "entity": entity_name,
            "entity_type": entity_type,
            "depth": depth,
            "dependencies": [],
            "metadata": metadata or {"table_columns": []}
        }
        
        if transformations:
            chain["transformations"] = transformations
        
        return chain
    
    def merge_chain_branches(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple chain branches into a single comprehensive chain."""
        if not chains:
            return {}
        
        if len(chains) == 1:
            return chains[0]
        
        # Use the first chain as the base
        merged_chain = chains[0].copy()
        
        # Merge dependencies from all chains
        all_dependencies = merged_chain.get("dependencies", [])
        
        for chain in chains[1:]:
            chain_deps = chain.get("dependencies", [])
            
            # Add dependencies that aren't already present
            existing_entities = {dep["entity"] for dep in all_dependencies}
            for dep in chain_deps:
                if dep["entity"] not in existing_entities:
                    all_dependencies.append(dep)
                    existing_entities.add(dep["entity"])
        
        merged_chain["dependencies"] = all_dependencies
        
        # Merge transformations if present
        all_transformations = merged_chain.get("transformations", [])
        for chain in chains[1:]:
            chain_transformations = chain.get("transformations", [])
            all_transformations.extend(chain_transformations)
        
        if all_transformations:
            merged_chain["transformations"] = all_transformations
        
        return merged_chain
    
    def calculate_max_depth(self, chains: Dict[str, Any]) -> int:
        """Calculate the maximum depth across all chains."""
        max_depth = 0
        
        def get_chain_depth(chain_entity: Dict[str, Any]) -> int:
            """Recursively calculate maximum depth of a chain."""
            current_depth = chain_entity.get('depth', 0)
            max_dep_depth = 0
            
            for dep in chain_entity.get('dependencies', []):
                dep_depth = get_chain_depth(dep)
                max_dep_depth = max(max_dep_depth, dep_depth)
            
            return current_depth + max_dep_depth
        
        for chain in chains.values():
            chain_depth = get_chain_depth(chain)
            max_depth = max(max_depth, chain_depth)
        
        return max_depth
    
    def process_derived_tables(self, sql: str) -> Dict[str, Any]:
        """Process derived tables and return 3-layer lineage structure."""
        if not sql:
            return {}
        
        return self.derived_table_analyzer.analyze_derived_tables(sql)
    
    def merge_derived_table_chains(self, base_chains: Dict, derived_chains: Dict) -> Dict:
        """Merge derived table chains with base chains, giving priority to derived table analysis."""
        if not derived_chains:
            return base_chains
        
        # If we have derived table chains, they provide more complete analysis
        # Merge them with base chains, with derived tables taking priority
        merged = base_chains.copy()
        
        for entity_name, chain_data in derived_chains.items():
            # If this entity exists in base chains, merge metadata
            if entity_name in merged:
                # Keep the derived table structure but merge any additional metadata
                base_metadata = merged[entity_name].get('metadata', {})
                derived_metadata = chain_data.get('metadata', {})
                
                # Merge table metadata (keep derived table columns as they're more complete)
                if 'table_type' in base_metadata and 'table_type' not in derived_metadata:
                    derived_metadata['table_type'] = base_metadata['table_type']
                if 'schema' in base_metadata and 'schema' not in derived_metadata:
                    derived_metadata['schema'] = base_metadata['schema']
                if 'description' in base_metadata and 'description' not in derived_metadata:
                    derived_metadata['description'] = base_metadata['description']
                
                # Update the derived chain with merged metadata
                chain_data['metadata'].update(derived_metadata)
            
            # Use the derived table chain (it has the complete 3-layer structure)
            merged[entity_name] = chain_data
        
        return merged
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None, column_lineage_data: Dict = None, column_transformations_data: Dict = None) -> None:
        """Add missing source columns and handle QUERY_RESULT dependencies - moved from LineageChainBuilder."""
        if not sql:
            return 
            
        try:
            # Parse SQL to get select columns and alias mapping for QUERY_RESULT columns
            import sqlglot
            from ..utils.sql_parsing_utils import is_subquery_expression
            parsed = sqlglot.parse_one(sql, dialect='trino')
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[str(table.alias)] = str(table.name)
            
            # Get select columns from parsing - handle all SELECT statements including UNIONs
            select_columns = []
            
            # Collect all SELECT statements (handles UNION queries properly)
            select_stmts = list(parsed.find_all(sqlglot.exp.Select))
            
            # Process each SELECT statement
            for select_stmt in select_stmts:
                # Get the FROM table for this SELECT to track source
                from_clause = select_stmt.find(sqlglot.exp.From)
                stmt_source_table = None
                if from_clause and isinstance(from_clause.this, sqlglot.exp.Table):
                    stmt_source_table = str(from_clause.this.name)
                
                for expr in select_stmt.expressions:
                    if isinstance(expr, sqlglot.exp.Column):
                        raw_expr = str(expr)
                        table_part = str(expr.table) if expr.table else stmt_source_table
                        column_name = str(expr.name) if expr.name else raw_expr
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': column_name,
                            'source_table': table_part,
                            'from_table': stmt_source_table  # Track which table this SELECT is from
                        })
                    else:
                        # Handle other expressions (including Alias expressions for subqueries)
                        raw_expr = str(expr)
                        
                        # For Alias expressions, extract the alias name
                        if isinstance(expr, sqlglot.exp.Alias) and expr.alias:
                            column_name = str(expr.alias)
                        else:
                            column_name = raw_expr
                        
                        # For subquery expressions, set source_table to the subquery's table
                        source_table_for_expr = stmt_source_table
                        if is_subquery_expression(raw_expr, self.dialect):
                            # Extract the actual source table from the subquery
                            from ..utils.sql_parsing_utils import extract_tables_from_subquery
                            subquery_tables = extract_tables_from_subquery(raw_expr, self.dialect)
                            if subquery_tables:
                                source_table_for_expr = subquery_tables[0]  # Use the first table from subquery
                            
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': column_name,
                            'source_table': source_table_for_expr,
                            'from_table': stmt_source_table  # Track which table this SELECT is from
                        })
        except:
            select_columns = []
            alias_to_table = {}
        
        # Import required utility functions
        from ..utils.sql_parsing_utils import extract_clean_column_name
        from ..utils.aggregate_utils import (
            is_aggregate_function, extract_alias_from_expression, 
            is_aggregate_function_for_table, query_has_aggregates
        )
        from ..utils.sql_parsing_utils import extract_function_type, infer_query_result_columns_simple
        from .analyzers.ctas_analyzer import (
            is_ctas_target_table, build_ctas_target_columns, add_group_by_to_ctas_transformations
        )
        
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Handle QUERY_RESULT and CTAS dependencies specially
            for dep in entity_data.get('dependencies', []):
                dep_entity = dep.get('entity')
                if dep_entity == 'QUERY_RESULT' or is_ctas_target_table(sql, dep_entity):
                    # Get existing columns to avoid duplication
                    existing_metadata = dep.get('metadata', {})
                    existing_columns = existing_metadata.get('table_columns', [])
                    existing_column_names = {col.get('name') for col in existing_columns}
                    
                    # Add qualified column names to QUERY_RESULT metadata
                    table_columns = list(existing_columns)  # Start with existing columns
                    # Check if this is a CTAS query
                    is_ctas = is_ctas_target_table(sql, dep_entity)
                    
                    if is_ctas:
                        # CTAS query: Add target table columns with special handling for aggregates
                        table_columns = build_ctas_target_columns(sql, select_columns)
                        
                        # Add GROUP BY information to transformations
                        add_group_by_to_ctas_transformations(dep, sql)
                    else:
                        # Check if this is a JOIN or UNION query first
                        # A JOIN query should have multiple different source tables or JOIN keywords
                        unique_source_tables = set(sel_col.get('source_table') for sel_col in select_columns if sel_col.get('source_table'))
                        is_join_query = len(unique_source_tables) > 1 or 'JOIN' in sql.upper()
                        is_union_query = 'UNION' in sql.upper()
                        
                        # Run aggregate-aware processing if needed, or for JOIN/UNION queries
                        needs_aggregate_processing = query_has_aggregates(sql)
                        
                        if needs_aggregate_processing or is_join_query or is_union_query:
                            # Use same logic as original analyzer
                            inferred_columns = infer_query_result_columns_simple(sql, select_columns)
                            has_table_prefixes = any('.' in col.get('name', '') for col in inferred_columns)
                            
                            # Special handling for QUERY_RESULT: Add all SELECT columns with transformation details
                            # Only add to the primary table (FROM clause) or the table that most SELECT expressions reference
                            if dep_entity == 'QUERY_RESULT':
                                # For UNION queries, filter select_columns to only those from this entity
                                if is_union_query:
                                    entity_select_columns = [col for col in select_columns if col.get('source_table') == entity_name]
                                else:
                                    entity_select_columns = select_columns
                                
                                # Determine primary table from FROM clause
                                primary_table = None
                                try:
                                    parsed_sql = sqlglot.parse_one(sql, dialect='trino')
                                    from_clause = parsed_sql.find(sqlglot.exp.From)
                                    if from_clause and hasattr(from_clause.this, 'name'):
                                        primary_table = str(from_clause.this.name)
                                except:
                                    pass
                                
                                # For aggregate queries, only add to primary table
                                # For JOIN/UNION queries, add to the main table that most SELECT expressions reference
                                should_add_columns = False
                                
                                if needs_aggregate_processing:
                                    # For aggregate queries, only add to primary table
                                    should_add_columns = (entity_name == primary_table)
                                elif is_join_query or is_union_query:
                                    # For JOIN/UNION queries, always process columns but filter by table relevance
                                    should_add_columns = True
                                else:
                                    # For simple queries, add to primary table
                                    should_add_columns = (entity_name == primary_table)
                                
                                if should_add_columns:
                                    for sel_col in entity_select_columns:
                                        raw_expression = sel_col.get('raw_expression')
                                        column_name = sel_col.get('column_name')
                                        source_table = sel_col.get('source_table')
                                        
                                        print(f"DEBUG: Processing column for {entity_name}: '{raw_expression}' (source_table: {source_table})")
                                        
                                        # Skip subquery columns for entities that don't match the subquery's source table
                                        if (is_subquery_expression(raw_expression, self.dialect) and 
                                            source_table != entity_name):
                                            print(f"DEBUG: Skipping subquery column '{raw_expression[:50]}...' for {entity_name} (belongs to {source_table})")
                                            continue
                                        
                                        # Skip individual aggregate functions that are part of subqueries
                                        if (is_aggregate_function(raw_expression) and 
                                            not is_subquery_expression(raw_expression, self.dialect) and
                                            source_table != entity_name):
                                            # Check if this aggregate function is part of a subquery by looking at other columns
                                            skip_aggregate = False
                                            for other_col in entity_select_columns:
                                                other_expr = other_col.get('raw_expression', '')
                                                if (is_subquery_expression(other_expr, self.dialect) and 
                                                    raw_expression in other_expr):
                                                    skip_aggregate = True
                                                    print(f"DEBUG: Skipping aggregate '{raw_expression}' as it's part of subquery '{other_expr[:50]}...'")
                                                    break
                                            if skip_aggregate:
                                                continue
                                        
                                        # For JOIN/UNION queries, only add columns that reference this specific table
                                        column_belongs_to_this_table = False
                                        
                                        if is_join_query or is_union_query:
                                            # FIRST: Check if this is a subquery expression - these should only belong to their source table
                                            if is_subquery_expression(raw_expression, self.dialect):
                                                from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                                subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                                column_belongs_to_this_table = entity_name in subquery_tables
                                            else:
                                                # SECOND: Check if this column expression references this entity's table alias
                                                for alias, table in alias_to_table.items():
                                                    if table == entity_name and f'{alias}.' in raw_expression:
                                                        column_belongs_to_this_table = True
                                                        break
                                            
                                            # Special handling for aggregate functions without table prefixes (like COUNT(*))
                                            # These should belong to the primary table
                                            if not column_belongs_to_this_table and is_aggregate_function(raw_expression):
                                                if entity_name == primary_table:
                                                    column_belongs_to_this_table = True
                                            
                                            # For UNION queries, use existing helper to get columns for this table
                                            if is_union_query and not column_belongs_to_this_table:
                                                from ..utils.sql_parsing_utils import get_union_columns_for_table
                                                union_columns = get_union_columns_for_table(sql, entity_name)
                                                # Check if this column expression matches any of the table's UNION columns
                                                for union_col in union_columns:
                                                    if (raw_expression in union_col or 
                                                        column_name in union_col or
                                                        raw_expression == union_col):
                                                        column_belongs_to_this_table = True
                                                        break
                                            
                                        else:
                                            # For aggregate queries or simple queries, add all columns to primary table
                                            column_belongs_to_this_table = True
                                        
                                        if not column_belongs_to_this_table:
                                            continue
                                        
                                        # Extract clean name (prefer alias over raw column name)
                                        clean_name = extract_clean_column_name(raw_expression, column_name)
                                        
                                        # For non-aggregate columns, use the raw expression format (e.g., "u.department")
                                        # For aggregate columns, use clean name (e.g., "employee_count")
                                        # For UNION queries, use the output column names (aliases)
                                        if is_aggregate_function(raw_expression):
                                            display_name = clean_name
                                        elif is_union_query:
                                            # For UNION queries, use the output column name (alias)
                                            display_name = column_name
                                        else:
                                            display_name = raw_expression if not ' AS ' in raw_expression.upper() else raw_expression.split(' AS ')[0].strip()
                                        
                                        if clean_name not in existing_column_names:
                                            # For UNION queries in QUERY_RESULT, use table-specific expressions and upstream
                                            if is_union_query and dep_entity == 'QUERY_RESULT':
                                                # Use the raw expression as the name and table-specific upstream
                                                column_name_to_use = raw_expression  # Full expression like "'product' as type"
                                                upstream_ref = f"{entity_name}.{raw_expression}"
                                            else:
                                                column_name_to_use = display_name
                                                upstream_ref = f"{entity_name}.{clean_name}"
                                                
                                            column_info = {
                                                "name": column_name_to_use,
                                                "upstream": [upstream_ref],
                                                "type": "DIRECT"
                                            }
                                            
                                            # Check if this is a subquery first (before checking for aggregates)
                                            if is_subquery_expression(raw_expression, self.dialect):
                                                from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                                subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                                if subquery_tables:
                                                    main_subquery_table = subquery_tables[0]
                                                    column_info["upstream"] = [f"{main_subquery_table}.{clean_name}"]
                                                column_info["transformation"] = {
                                                    "source_expression": raw_expression,
                                                    "transformation_type": "SUBQUERY",
                                                    "function_type": "SUBQUERY"
                                                }
                                            # Check if this is an aggregate function and add transformation details
                                            elif is_aggregate_function(raw_expression):
                                                function_type = extract_function_type(raw_expression)
                                                # Clean source expression: remove AS clause
                                                source_expr = raw_expression.split(' AS ')[0].strip() if ' AS ' in raw_expression.upper() else raw_expression
                                                column_info["transformation"] = {
                                                    "source_expression": source_expr,
                                                    "transformation_type": "AGGREGATE",
                                                    "function_type": function_type
                                                }
                                            
                                            table_columns.append(column_info)
                                            existing_column_names.add(clean_name)
                            
                            elif has_table_prefixes or is_join_query:
                                # JOIN query: Use qualified names from select columns
                                for sel_col in select_columns:
                                    source_table = sel_col.get('source_table')
                                    raw_expression = sel_col.get('raw_expression')
                                    column_name = sel_col.get('column_name')
                                    
                                    if (source_table in alias_to_table and 
                                        alias_to_table[source_table] == entity_name):
                                        # Regular table-prefixed column
                                        upstream_col = f"{entity_name}.{column_name}"
                                        
                                        if raw_expression not in existing_column_names:
                                            column_info = {
                                                "name": raw_expression,
                                                "upstream": [upstream_col],
                                                "type": "DIRECT"
                                            }
                                            table_columns.append(column_info)
                                    elif source_table is None and is_subquery_expression(raw_expression, self.dialect):
                                        # Subquery expression - use transformation engine for proper handling
                                        subquery_columns = self.transformation_engine.handle_subquery_functions(sql, entity_name)
                                        for subquery_col in subquery_columns:
                                            col_name = subquery_col.get("name")
                                            if col_name and col_name not in existing_column_names:
                                                table_columns.append(subquery_col)
                                                existing_column_names.add(col_name)
                                    elif source_table is None and is_aggregate_function(raw_expression):
                                        # Aggregate function column - only add if relevant to this entity
                                        if is_aggregate_function_for_table(raw_expression, entity_name, sql):
                                            alias = extract_alias_from_expression(raw_expression)
                                            func_type = extract_function_type(raw_expression)
                                            
                                            column_name_to_use = alias or column_name
                                            if column_name_to_use not in existing_column_names:
                                                column_info = {
                                                    "name": column_name_to_use,
                                                    "upstream": [f"{entity_name}.{column_name_to_use}"],
                                                    "type": "DIRECT",
                                                    "transformation": {
                                                        "source_expression": raw_expression.replace(f" as {alias}", "").replace(f" AS {alias}", "") if alias else raw_expression,
                                                        "transformation_type": "AGGREGATE",
                                                        "function_type": func_type
                                                    }
                                                }
                                                table_columns.append(column_info)
                            elif is_union_query:
                                # UNION query: Add unified output columns with proper upstream attribution
                                from ..utils.sql_parsing_utils import get_union_columns_for_table
                                union_columns = get_union_columns_for_table(sql, entity_name)
                                for column_expr in union_columns:
                                    # Extract clean column name (remove aliases and expressions)
                                    clean_name = extract_clean_column_name(column_expr, column_expr)
                                    
                                    if clean_name not in existing_column_names:
                                        column_info = {
                                            "name": clean_name,
                                            "upstream": [f"QUERY_RESULT.{clean_name}"],
                                            "type": "DIRECT"
                                        }
                                        table_columns.append(column_info)
                        else:
                            # Simple query: Add ALL columns from SELECT statement to QUERY_RESULT
                            # For simple queries, QUERY_RESULT should contain all SELECT columns
                            for sel_col in select_columns:
                                raw_expression = sel_col.get('raw_expression')
                                column_name = sel_col.get('column_name')
                                
                                # Extract clean name (prefer alias over raw column name)
                                clean_name = extract_clean_column_name(raw_expression, column_name)
                                
                                if clean_name not in existing_column_names:
                                    column_info = {
                                        "name": clean_name,
                                        "upstream": [f"QUERY_RESULT.{clean_name}"],
                                        "type": "DIRECT"
                                    }
                                    
                                    # Check if this is a subquery first (before checking for aggregates)
                                    if is_subquery_expression(raw_expression, self.dialect):
                                        # Use transformation engine for subquery handling
                                        from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                        subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                        if subquery_tables:
                                            main_subquery_table = subquery_tables[0]
                                            column_info["upstream"] = [f"{main_subquery_table}.{clean_name}"]
                                        column_info["transformation"] = {
                                            "source_expression": raw_expression,
                                            "transformation_type": "SUBQUERY",
                                            "function_type": "SUBQUERY"
                                        }
                                    # Check if this is an aggregate function and add transformation details
                                    elif is_aggregate_function(raw_expression):
                                        function_type = extract_function_type(raw_expression)
                                        column_info["transformation"] = {
                                            "source_expression": raw_expression,
                                            "transformation_type": "AGGREGATE",
                                            "function_type": function_type
                                        }
                                    
                                    table_columns.append(column_info)
                                    existing_column_names.add(clean_name)
                    
                    # Add subquery result columns to the QUERY_RESULT of tables referenced in subqueries
                    if dep_entity == 'QUERY_RESULT' and not table_columns:
                        # Check if any SELECT columns contain subqueries that reference this entity
                        for sel_col in select_columns:
                            raw_expression = sel_col.get('raw_expression', '')
                            if is_subquery_expression(raw_expression, self.dialect):
                                # Check if this subquery references the current entity
                                from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                if entity_name in subquery_tables:
                                    # Extract the alias/column name for this subquery result
                                    clean_name = extract_clean_column_name(raw_expression, f"subquery_{len(table_columns)}")
                                    if clean_name not in existing_column_names:
                                        column_info = {
                                            "name": clean_name,
                                            "upstream": [f"{entity_name}.{clean_name}"],
                                            "type": "DIRECT",
                                            "transformation": {
                                                "source_expression": raw_expression,
                                                "transformation_type": "SUBQUERY",
                                                "function_type": "SUBQUERY"
                                            }
                                        }
                                        table_columns.append(column_info)
                                        existing_column_names.add(clean_name)
                    
                    if table_columns:
                        if 'metadata' not in dep:
                            dep['metadata'] = {}
                        dep['metadata']['table_columns'] = table_columns
                
            # Handle source table missing columns
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies
            if entity_data.get('depth') == 0:
                source_columns = set()
                
                # Check if this is a UNION query - use specialized column extraction
                is_union_query = sql and 'UNION' in sql.upper()
                if is_union_query:
                    # For UNION queries, get table-specific columns directly
                    union_table_columns = extract_columns_referenced_by_table_in_union(sql, entity_name, self.dialect)
                    source_columns.update(union_table_columns)
                else:
                    # Look through dependencies to find transformations that reference this table
                    for dep in entity_data.get('dependencies', []):
                        for trans in dep.get('transformations', []):
                            if trans.get('source_table') == entity_name:
                                # Extract filter condition columns - only if they actually belong to this entity
                                for condition in trans.get('filter_conditions', []):
                                    col = condition.get('column', '')
                                    if col:
                                        # Only add columns that actually belong to this entity
                                        belongs = is_column_from_table(col, entity_name, sql, self.dialect)
                                        if belongs:
                                            # Clean column name by removing table prefix if present
                                            clean_col = col.split('.')[-1] if '.' in col else col
                                            source_columns.add(clean_col)
                                
                                # Extract group by columns - only if they actually belong to this entity
                                for group_col in trans.get('group_by_columns', []):
                                    if group_col:
                                        # Only add columns that actually belong to this entity
                                        if is_column_from_table(group_col, entity_name, sql, self.dialect):
                                            # Clean column name by removing table prefix if present
                                            clean_col = group_col.split('.')[-1] if '.' in group_col else group_col
                                            source_columns.add(clean_col)
                                
                                # Extract join condition columns
                                for join in trans.get('joins', []):
                                    for condition in join.get('conditions', []):
                                        for col_key in ['left_column', 'right_column']:
                                            col_ref = condition.get(col_key, '')
                                            if col_ref:
                                                # Only add columns that actually belong to this entity
                                                if is_column_from_table(col_ref, entity_name, sql, self.dialect):
                                                    clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                                                    source_columns.add(clean_col)
                
                # Add the columns found in transformations, merging with existing
                if source_columns:
                    # Get existing column names to avoid duplicates
                    existing_columns = {col['name'] for col in table_columns}
                    
                    # Add new columns from transformations
                    for column_name in source_columns:
                        if column_name not in existing_columns:
                            column_info = {
                                "name": column_name,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            table_columns.append(column_info)
                    
                    # Update the metadata
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    entity_data['metadata']['table_columns'] = table_columns
