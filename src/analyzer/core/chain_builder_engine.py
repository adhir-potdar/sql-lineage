"""Chain building engine for SQL lineage analysis."""

from typing import Dict, List, Any, Optional, Set
from ..utils.sql_parsing_utils import is_column_from_table
from ..utils.metadata_utils import create_metadata_entry


class ChainBuilderEngine:
    """Engine for building lineage chains from dependencies."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize the chain builder engine."""
        self.dialect = dialect
    
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
        if hasattr(trans, 'filter_conditions') and trans.filter_conditions:
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
                if is_column_from_table(fc.column, entity_name, sql, self.dialect):
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
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None, column_lineage_data: Dict = None, column_transformations_data: Dict = None) -> None:
        """Add missing source columns to table metadata by extracting from transformations."""
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies  
            if entity_data.get('depth') == 0:
                source_columns = set()
                
                # Look through dependencies to find transformations that reference this table
                for dep in entity_data.get('dependencies', []):
                    for trans in dep.get('transformations', []):
                        if trans.get('source_table') == entity_name:
                            # Extract filter condition columns
                            for condition in trans.get('filter_conditions', []):
                                col = condition.get('column', '')
                                if col:
                                    clean_col = col.split('.')[-1] if '.' in col else col
                                    source_columns.add(clean_col)
                            
                            # Extract group by columns  
                            for group_col in trans.get('group_by_columns', []):
                                if group_col:
                                    clean_col = group_col.split('.')[-1] if '.' in group_col else group_col
                                    source_columns.add(clean_col)
                            
                            # Extract join condition columns
                            for join in trans.get('joins', []):
                                for condition in join.get('conditions', []):
                                    for col_key in ['left_column', 'right_column']:
                                        col_ref = condition.get(col_key, '')
                                        if col_ref and entity_name in col_ref:
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
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None, column_lineage_data: Dict = None, column_transformations_data: Dict = None) -> None:
        """Add missing source columns and handle QUERY_RESULT dependencies - moved from LineageChainBuilder."""
        if not sql:
            return
            
        try:
            # Parse SQL to get select columns and alias mapping for QUERY_RESULT columns
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect='trino')
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[str(table.alias)] = str(table.name)
            
            # Get select columns from parsing
            select_columns = []
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                for expr in select_stmt.expressions:
                    if isinstance(expr, sqlglot.exp.Column):
                        raw_expr = str(expr)
                        table_part = str(expr.table) if expr.table else None
                        column_name = str(expr.name) if expr.name else raw_expr
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': column_name,
                            'source_table': table_part
                        })
                    else:
                        # Handle other expressions
                        raw_expr = str(expr)
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': raw_expr,
                            'source_table': None
                        })
        except:
            select_columns = []
            alias_to_table = {}
        
        # Import required utility functions
        from analyzer.utils.sql_parsing_utils import extract_clean_column_name
        from analyzer.utils.aggregate_utils import (
            is_aggregate_function, extract_alias_from_expression, 
            is_aggregate_function_for_table, query_has_aggregates
        )
        from analyzer.utils.sql_parsing_utils import extract_function_type, infer_query_result_columns_simple
        from analyzer.core.analyzers.ctas_analyzer import (
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
                        is_join_query = any(sel_col.get('source_table') is not None for sel_col in select_columns)
                        is_union_query = 'UNION' in sql.upper()
                        
                        # Run aggregate-aware processing if needed, or for JOIN/UNION queries
                        needs_aggregate_processing = query_has_aggregates(sql)
                        
                        if needs_aggregate_processing or is_join_query or is_union_query:
                            # Use same logic as original analyzer
                            inferred_columns = infer_query_result_columns_simple(sql, select_columns)
                            has_table_prefixes = any('.' in col.get('name', '') for col in inferred_columns)
                            
                            if has_table_prefixes or is_join_query:
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
                                # UNION query: Only add columns that belong to this specific table
                                from analyzer.utils.sql_parsing_utils import get_union_columns_for_table
                                union_columns = get_union_columns_for_table(sql, entity_name)
                                for column_name in union_columns:
                                    if column_name not in existing_column_names:
                                        column_info = {
                                            "name": column_name,
                                            "upstream": [f"{entity_name}.{column_name}"],
                                            "type": "DIRECT"
                                        }
                                        table_columns.append(column_info)
                        else:
                            # Simple query: Add columns from select statement to QUERY_RESULT
                            for sel_col in select_columns:
                                if sel_col.get('source_table') is None:  # Unprefixed columns
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
                                        table_columns.append(column_info)
                    
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
                
                # Look through dependencies to find transformations that reference this table
                for dep in entity_data.get('dependencies', []):
                    for trans in dep.get('transformations', []):
                        if trans.get('source_table') == entity_name:
                            # Extract filter condition columns
                            for condition in trans.get('filter_conditions', []):
                                col = condition.get('column', '')
                                if col:
                                    # Clean column name by removing table prefix if present
                                    clean_col = col.split('.')[-1] if '.' in col else col
                                    source_columns.add(clean_col)
                            
                            # Extract group by columns
                            for group_col in trans.get('group_by_columns', []):
                                if group_col:
                                    # Clean column name by removing table prefix if present
                                    clean_col = group_col.split('.')[-1] if '.' in group_col else group_col
                                    source_columns.add(clean_col)
                            
                            # Extract join condition columns
                            for join in trans.get('joins', []):
                                for condition in join.get('conditions', []):
                                    for col_key in ['left_column', 'right_column']:
                                        col_ref = condition.get(col_key, '')
                                        if col_ref and entity_name in col_ref:
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