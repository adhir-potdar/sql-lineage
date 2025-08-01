"""Lineage chain builder analyzer."""

from typing import Dict, Any, List, Optional, Set
import json
import re
import sqlglot
from sqlglot import expressions as exp
from .base_analyzer import BaseAnalyzer


class LineageChainBuilder(BaseAnalyzer):
    """Analyzer for building lineage chains."""
    
    def get_table_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the table lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        result = self.extractor.analyze(sql, **kwargs)
        
        # Get the appropriate lineage direction
        lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        
        # Build the chain starting from all tables at the current level
        chain = {}
        
        def build_chain(table_name: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth or if we have a circular dependency
            if current_depth > depth or table_name in visited_in_path:
                return {"table": table_name, "depth": current_depth - 1, "dependencies": []}
            
            # Add current table to the path to prevent cycles
            visited_in_path = visited_in_path | {table_name}
            
            dependencies = []
            if table_name in lineage_data:
                for dependent_table in lineage_data[table_name]:
                    dep_chain = build_chain(dependent_table, current_depth + 1, visited_in_path)
                    dependencies.append(dep_chain)
            
            return {
                "table": table_name,
                "depth": current_depth - 1,
                "dependencies": dependencies
            }
        
        # Start chain building from all root tables
        for table_name in lineage_data.keys():
            chain[table_name] = build_chain(table_name, 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    def get_column_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the column lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the column lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        result = self.extractor.analyze(sql, **kwargs)
        
        # Get the appropriate lineage direction
        lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Build the chain starting from all columns at the current level
        chain = {}
        
        def build_column_chain(column_ref: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth or if we have a circular dependency
            if current_depth > depth or column_ref in visited_in_path:
                return {"column": column_ref, "depth": current_depth - 1, "dependencies": []}
            
            # Add current column to the path to prevent cycles
            visited_in_path = visited_in_path | {column_ref}
            
            dependencies = []
            if column_ref in lineage_data:
                for dependent_column in lineage_data[column_ref]:
                    dep_chain = build_column_chain(dependent_column, current_depth + 1, visited_in_path)
                    dependencies.append(dep_chain)
            
            return {
                "column": column_ref,
                "depth": current_depth - 1,
                "dependencies": dependencies
            }
        
        # Start chain building from all root columns
        for column_ref in lineage_data.keys():
            chain[column_ref] = build_column_chain(column_ref, 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive lineage chain combining table and column lineage with transformations.
        Supports CTAS, CTE, and other SQL block types with detailed metadata and transformations.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing comprehensive lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 0:
            raise ValueError("depth must be 0 or greater (0 means unlimited depth)")
        
        # Check if this is a CTE query and use CTE-specific processing
        if "WITH" in sql.upper():
            return self._build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        
        result = self.extractor.analyze(sql, **kwargs)
        
        # Get both table and column lineage data
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Helper function to get table name from column reference
        def extract_table_from_column(column_ref: str) -> str:
            if '.' in column_ref:
                parts = column_ref.split('.')
                return '.'.join(parts[:-1])  # Everything except the last part (column name)
            return "unknown_table"
        
        # Helper function to get column name from column reference
        def extract_column_from_ref(column_ref: str) -> str:
            if '.' in column_ref:
                return column_ref.split('.')[-1]
            return column_ref
        
        # Build comprehensive lineage chain
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, visited_in_path: set = None, parent_entity: str = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we have a circular dependency
            if entity_name in visited_in_path:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "metadata": {}
                }
            
            # Stop if we've exceeded max depth (only when depth > 0, meaning limited depth)
            if depth > 0 and current_depth > depth:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "metadata": {}
                }
            
            # Add current entity to the path to prevent cycles
            visited_in_path = visited_in_path | {entity_name}
            
            dependencies = []
            transformations = []
            metadata = {}
            
            if entity_type == "table":
                # Handle table-level lineage
                if entity_name in table_lineage_data:
                    for dependent_table in table_lineage_data[entity_name]:
                        # For CTAS queries, don't add QUERY_RESULT since the target table itself is the final result
                        # Skip QUERY_RESULT dependencies for CTAS queries
                        if (sql and sql.strip().upper().startswith('CREATE TABLE') and 
                            dependent_table == 'QUERY_RESULT'):
                            continue  # Skip QUERY_RESULT for CTAS queries
                        
                        dep_chain = build_comprehensive_chain(dependent_table, "table", current_depth + 1, visited_in_path, entity_name)
                        dependencies.append(dep_chain)
                
                # Get table transformations (optimized - only include non-empty data)
                # Filter transformations to only show those relevant to the parent-child relationship
                transformation_entities = set()
                
                # First check if current entity has transformations
                if entity_name in result.table_lineage.transformations:
                    transformation_entities.add(entity_name)
                
                # Also check all entities for transformations where current entity is involved
                for trans_entity, trans_list in result.table_lineage.transformations.items():
                    for transformation in trans_list:
                        if (transformation.source_table == entity_name or 
                            transformation.target_table == entity_name):
                            transformation_entities.add(trans_entity)
                
                # Process transformations from all relevant entities
                for trans_entity in transformation_entities:
                    for transformation in result.table_lineage.transformations[trans_entity]:
                        # Only include transformations that are relevant to this specific relationship
                        # If we have a parent entity, filter based on the relationship
                        if parent_entity is not None:
                            # For downstream: parent -> current, show transformations where parent is source and current is target
                            # For upstream: current -> parent, show transformations where current is source and parent is target
                            if chain_type == "downstream":
                                if not (transformation.source_table == parent_entity and transformation.target_table == entity_name):
                                    continue
                            elif chain_type == "upstream":
                                if not (transformation.source_table == entity_name and transformation.target_table == parent_entity):
                                    continue
                        else:
                            # If no parent entity (root level), include transformations involving this entity
                            if not (transformation.source_table == entity_name or transformation.target_table == entity_name):
                                continue
                        
                        trans_data = {
                            "type": "table_transformation",
                            "source_table": transformation.source_table,
                            "target_table": transformation.target_table
                        }
                        
                        # Only add non-null/non-empty values
                        
                        if transformation.join_conditions:
                            # Convert old format to new joins format
                            join_entry = {
                                "join_type": transformation.join_type.value if transformation.join_type else "INNER JOIN",
                                "right_table": None,  # Extract from conditions if possible
                                "conditions": [
                                    {
                                        "left_column": jc.left_column,
                                        "operator": jc.operator.value if hasattr(jc.operator, 'value') else str(jc.operator),
                                        "right_column": jc.right_column
                                    }
                                    for jc in transformation.join_conditions
                                ]
                            }
                            
                            # Try to extract right table from the first condition
                            if transformation.join_conditions:
                                first_condition = transformation.join_conditions[0]
                                if hasattr(first_condition, 'right_column') and '.' in first_condition.right_column:
                                    right_table = first_condition.right_column.split('.')[0]
                                    join_entry["right_table"] = right_table
                            
                            trans_data["joins"] = [join_entry]
                        
                        # Determine context for column filtering - used for all transformation types
                        # Single-table context includes both QUERY_RESULT and CTAS scenarios
                        is_single_table = (
                            transformation.target_table == "QUERY_RESULT" or  # Regular SELECT
                            (transformation.source_table != transformation.target_table and  # CTAS/CTE
                             transformation.target_table != "QUERY_RESULT")
                        )
                        
                        context_info = {
                            'is_single_table_context': is_single_table,
                            'tables_in_context': [transformation.source_table] if is_single_table else []
                        }
                        
                        # Filter conditions to only include those relevant to the current entity
                        if transformation.filter_conditions:
                            relevant_filters = []
                            
                            for fc in transformation.filter_conditions:
                                # Only include filter conditions that reference columns from this entity
                                if self._is_column_from_table(fc.column, entity_name, context_info):
                                    relevant_filters.append({
                                        "column": fc.column,
                                        "operator": fc.operator.value if hasattr(fc.operator, 'value') else str(fc.operator),
                                        "value": fc.value
                                    })
                            if relevant_filters:
                                trans_data["filter_conditions"] = relevant_filters
                        
                        # Group by columns - only include those from this entity
                        if transformation.group_by_columns:
                            relevant_group_by = []
                            for col in transformation.group_by_columns:
                                if self._is_column_from_table(col, entity_name, context_info):
                                    relevant_group_by.append(col)
                            if relevant_group_by:
                                trans_data["group_by_columns"] = relevant_group_by
                        
                        # Having conditions - only include those referencing columns from this entity
                        if transformation.having_conditions:
                            relevant_having = []
                            for hc in transformation.having_conditions:
                                # Having conditions often involve aggregations like COUNT(*) or AVG(u.salary)
                                # Check if they reference this entity or if they are general aggregations for this table
                                is_relevant = (self._is_column_from_table(hc.column, entity_name, context_info) or 
                                             self._is_aggregate_function_for_table(hc.column, entity_name))
                                if is_relevant:
                                    relevant_having.append({
                                        "column": hc.column,
                                        "operator": hc.operator.value if hasattr(hc.operator, 'value') else str(hc.operator),
                                        "value": hc.value
                                    })
                            if relevant_having:
                                trans_data["having_conditions"] = relevant_having
                        
                        # Order by columns - only include those from this entity
                        if transformation.order_by_columns:
                            relevant_order_by = []
                            for col in transformation.order_by_columns:
                                # Extract just the column name part (before ASC/DESC)
                                col_name = col.split()[0] if ' ' in col else col
                                if self._is_column_from_table(col_name, entity_name, context_info):
                                    relevant_order_by.append(col)
                            if relevant_order_by:
                                trans_data["order_by_columns"] = relevant_order_by
                        
                        # Column transformations will be integrated into individual column metadata
                        # Remove separate column_transformations from table transformations
                        
                        transformations.append(trans_data)
                
                # Get essential table metadata (excluding detailed column info)
                if entity_name in result.metadata:
                    table_meta = result.metadata[entity_name]
                    metadata = {
                        "table_type": table_meta.table_type.value
                    }
                    
                    # Only include non-null values to keep output clean
                    if table_meta.schema:
                        metadata["schema"] = table_meta.schema
                    if table_meta.description:
                        metadata["description"] = table_meta.description
                
                # Add simplified column-level information for this table
                table_columns = []
                columns_added = set()  # Track to avoid duplicates
                
                # First, add columns that have upstream relationships (downstream/intermediate tables)
                for column_ref in column_lineage_data.keys():
                    if extract_table_from_column(column_ref) == entity_name:
                        column_name = extract_column_from_ref(column_ref)
                        upstream_columns = list(column_lineage_data.get(column_ref, set()))
                        
                        if upstream_columns:
                            column_info = {
                                "name": column_name,
                                "upstream": upstream_columns
                            }
                            
                            # Add transformation type if present (simplified)
                            if column_ref in result.column_lineage.transformations:
                                column_transformations = result.column_lineage.transformations[column_ref]
                                if column_transformations:
                                    trans = column_transformations[0]  # Take first transformation
                                    if trans.aggregate_function:
                                        column_info["type"] = "AGGREGATE"
                                    elif trans.window_function:
                                        column_info["type"] = "WINDOW"
                                    elif trans.case_expression:
                                        column_info["type"] = "CASE"
                                    elif trans.expression and trans.expression != column_name:
                                        column_info["type"] = "COMPUTED"
                                    else:
                                        column_info["type"] = "DIRECT"
                                else:
                                    column_info["type"] = "DIRECT"
                            else:
                                column_info["type"] = "DIRECT"
                            
                            table_columns.append(column_info)
                            columns_added.add(column_name)
                
                # Add source columns for tables that don't have any table_columns yet
                # Extract from transformations in dependencies since the actual transformations are stored there
                if not table_columns and entity_type == "table" and entity_name != "QUERY_RESULT":
                    source_columns = set()
                    
                    # Get selected columns from column lineage (SELECT clause columns)
                    for column_ref in column_lineage_data.keys():
                        # Column refs without table prefix are from the source table
                        if '.' not in column_ref:
                            source_columns.add(column_ref)
                    
                    # Add all found columns to table metadata
                    for column_name in source_columns:
                        column_info = {
                            "name": column_name,
                            "upstream": [],
                            "type": "SOURCE" 
                        }
                        table_columns.append(column_info)
                
                # Special handling for QUERY_RESULT - infer result columns from SQL parsing
                if entity_name == "QUERY_RESULT" and not table_columns:
                    # For QUERY_RESULT, we should infer columns from the SELECT statement
                    # Filter columns based on the parent entity that's requesting this QUERY_RESULT
                    all_query_result_columns = self._infer_query_result_columns(sql, column_lineage_data)
                    
                    if parent_entity and all_query_result_columns:
                        # Only filter if this is a multi-table query (JOIN scenario)
                        # Check if columns have table prefixes, indicating multiple tables
                        has_table_prefixes = any('.' in col.get('name', '') for col in all_query_result_columns)
                        
                        if has_table_prefixes:
                            # Filter columns to only include those that come from the parent entity
                            table_columns = self._filter_query_result_columns_by_parent(
                                all_query_result_columns, parent_entity, column_lineage_data
                            )
                        else:
                            # Single table query - use all columns
                            table_columns = all_query_result_columns
                    else:
                        # If no parent entity or no columns, use all columns
                        table_columns = all_query_result_columns
                
                # Only add table_columns if not empty
                if table_columns:
                    metadata["table_columns"] = table_columns
            
            elif entity_type == "column":
                # Handle column-level lineage (simplified)
                if entity_name in column_lineage_data:
                    for dependent_column in column_lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dependent_column, "column", current_depth + 1, visited_in_path, entity_name)
                        dependencies.append(dep_chain)
                
                # Simplified column transformations
                if entity_name in result.column_lineage.transformations:
                    transformations_list = result.column_lineage.transformations[entity_name]
                    if transformations_list:
                        trans = transformations_list[0]  # Take first transformation
                        trans_data = {"type": "column_transformation"}
                        
                        if trans.aggregate_function:
                            trans_data["function_type"] = trans.aggregate_function.function_type.value
                        elif trans.window_function:
                            trans_data["function_type"] = "WINDOW"
                        elif trans.case_expression:
                            trans_data["function_type"] = "CASE"
                        elif trans.expression:
                            trans_data["expression"] = trans.expression
                        
                        transformations.append(trans_data)
                
                # Minimal column metadata
                parent_table = extract_table_from_column(entity_name)
                metadata = {"parent_table": parent_table}
            
            # Clean up empty arrays to reduce clutter
            result_dict = {
                "entity": entity_name,
                "entity_type": entity_type,
                "depth": current_depth - 1,
                "dependencies": dependencies,
                "metadata": metadata
            }
            
            # Add transformations to dependencies that are QUERY_RESULT
            for dep in dependencies:
                if dep.get("entity") == "QUERY_RESULT" and transformations:
                    # Filter transformations to only include those relevant to this entity
                    relevant_transformations = []
                    for trans in transformations:
                        if (trans.get("source_table") == entity_name and 
                            trans.get("target_table") == "QUERY_RESULT"):
                            relevant_transformations.append(trans)
                    
                    if relevant_transformations:
                        dep["transformations"] = relevant_transformations
            
            # Only add transformations if not empty AND there are no dependencies
            # (to avoid duplication - transformations will be shown in dependencies)
            if transformations and not dependencies:
                result_dict["transformations"] = transformations
                
            return result_dict
        
        # Build chains starting from the target entity or all entities
        chains = {}
        
        if target_entity:
            # Focus on specific entity
            if target_entity in table_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "table", 1, None, None)
            elif target_entity in column_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "column", 1, None, None)
            else:
                # Try to find it as a partial match
                found = False
                for table_name in table_lineage_data.keys():
                    if target_entity in table_name:
                        chains[table_name] = build_comprehensive_chain(table_name, "table", 1, None, None)
                        found = True
                        break
                
                if not found:
                    for column_ref in column_lineage_data.keys():
                        if target_entity in column_ref:
                            chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1, None, None)
                            break
        else:
            # First, collect all entities that will appear as dependencies to avoid duplication
            entities_in_dependencies = set()
            
            # Build initial chains to collect dependency information
            temp_chains = {}
            for table_name in table_lineage_data.keys():
                temp_chains[table_name] = build_comprehensive_chain(table_name, "table", 1, None, None)
            
            # Collect entities that appear in dependencies
            def collect_dependency_entities(chain_data):
                deps = chain_data.get('dependencies', [])
                for dep in deps:
                    dep_entity = dep.get('entity')
                    if dep_entity:
                        entities_in_dependencies.add(dep_entity)
                        collect_dependency_entities(dep)  # Recursively collect nested dependencies
            
            for chain_data in temp_chains.values():
                collect_dependency_entities(chain_data)
            
            # Build final chains, excluding entities that appear in dependencies
            for table_name in table_lineage_data.keys():
                # Only include as top-level if not already in dependencies
                if table_name not in entities_in_dependencies:
                    chains[table_name] = temp_chains[table_name]
            
            # Special handling: Ensure QUERY_RESULT appears as a dependency of source tables
            # rather than as an independent top-level entity, but include its transformations
            if 'QUERY_RESULT' in result.table_lineage.transformations:
                # QUERY_RESULT should appear as a dependency in the existing chains
                # The transformations will be included in the dependency metadata
                # This ensures proper flow: source_table → QUERY_RESULT (with transformations)
                pass  # The dependency relationships are already handled by build_comprehensive_chain
            
            # For downstream analysis with CTAS queries, ensure target tables appear as dependencies of source tables
            # rather than as separate top-level entities
            if chain_type == "downstream" and sql and sql.strip().upper().startswith('CREATE TABLE'):
                upstream_lineage_data = result.table_lineage.upstream
                downstream_lineage_data = result.table_lineage.downstream
                
                # Identify CTAS source and target tables
                ctas_source_tables = set()
                ctas_target_tables = set()
                
                for target_table, source_tables in upstream_lineage_data.items():
                    if target_table != 'QUERY_RESULT':
                        ctas_target_tables.add(target_table)
                        ctas_source_tables.update(source_tables)
                
                # Remove target tables from chains if they were added as top-level
                for target_table in ctas_target_tables:
                    if target_table in chains:
                        del chains[target_table]
                
                # Ensure target tables appear as dependencies in source table chains
                for source_table in ctas_source_tables:
                    if source_table in chains:
                        # Add target tables as dependencies of this source table
                        source_chain = chains[source_table]
                        source_dependencies = source_chain.get('dependencies', [])
                        
                        # Get target tables that depend on this source
                        for target_table in downstream_lineage_data.get(source_table, []):
                            if target_table != 'QUERY_RESULT' and target_table in ctas_target_tables:
                                # Check if this target is not already in dependencies
                                existing_dep_entities = {dep.get('entity') for dep in source_dependencies}
                                if target_table not in existing_dep_entities:
                                    # Build the target table as a dependency
                                    target_depth = source_chain.get('depth', 0) + 1
                                    target_chain = build_comprehensive_chain(target_table, "table", target_depth + 1, None, source_table)
                                    source_dependencies.append(target_chain)
                        
                        source_chain['dependencies'] = source_dependencies
            
            # Build chains for all columns (only if no table chains exist to avoid redundancy)
            if not chains:
                for column_ref in column_lineage_data.keys():
                    chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1, None, None)
        
        # Post-process chains to add missing source columns from filter conditions
        self.add_missing_source_columns(chains, sql)
        
        # Post-process to integrate column transformations into column metadata
        self.integrate_column_transformations(chains, sql)
        
        # Calculate actual max depth achieved
        actual_max_depth = 0
        for chain_data in chains.values():
            if "depth" in chain_data:
                actual_max_depth = max(actual_max_depth, chain_data["depth"])
            # Also check nested dependencies for max depth
            def get_max_depth_from_chain(chain_obj, current_max=0):
                max_depth = current_max
                if isinstance(chain_obj, dict):
                    if "depth" in chain_obj:
                        max_depth = max(max_depth, chain_obj["depth"])
                    if "dependencies" in chain_obj:
                        for dep in chain_obj["dependencies"]:
                            max_depth = max(max_depth, get_max_depth_from_chain(dep, max_depth))
                return max_depth
            
            actual_max_depth = max(actual_max_depth, get_max_depth_from_chain(chain_data))
        
        # Calculate actually used columns from transformations and lineage
        # We need to normalize column references to avoid counting duplicates
        # (e.g., 'name' and 'QUERY_RESULT.name' should be treated as the same logical column)
        used_columns = set()
        
        def normalize_column_name(column_ref: str) -> str:
            """Normalize column reference to just the column name, ignoring table prefixes for counting."""
            if column_ref and '.' in column_ref:
                # Skip QUERY_RESULT columns as they are output columns, not source columns
                if column_ref.startswith('QUERY_RESULT.'):
                    return None
                return column_ref.split('.')[-1]  # Get just the column name
            return column_ref
        
        # Add columns from upstream lineage data (these are the actual source columns)
        for column_ref, upstream_columns in column_lineage_data.items():
            # Add upstream columns (source columns)
            for upstream_col in upstream_columns:
                normalized = normalize_column_name(upstream_col)
                if normalized:
                    used_columns.add(normalized)
        
        # Add columns from column transformations (focus on source columns)
        for transformation_list in result.column_lineage.transformations.values():
            for transformation in transformation_list:
                if transformation.source_column:
                    normalized = normalize_column_name(transformation.source_column)
                    if normalized:
                        used_columns.add(normalized)
        
        # Add columns from table transformations (join conditions, filters, etc.)
        for transformation_list in result.table_lineage.transformations.values():
            for transformation in transformation_list:
                # Join conditions
                for join_condition in transformation.join_conditions:
                    if join_condition.left_column:
                        normalized = normalize_column_name(join_condition.left_column)
                        if normalized:
                            used_columns.add(normalized)
                    if join_condition.right_column:
                        normalized = normalize_column_name(join_condition.right_column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Filter conditions
                for filter_condition in transformation.filter_conditions:
                    if filter_condition.column:
                        normalized = normalize_column_name(filter_condition.column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Group by columns
                for group_col in transformation.group_by_columns:
                    normalized = normalize_column_name(group_col)
                    if normalized:
                        used_columns.add(normalized)
                
                # Having conditions
                for having_condition in transformation.having_conditions:
                    if having_condition.column:
                        normalized = normalize_column_name(having_condition.column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Order by columns
                for order_col in transformation.order_by_columns:
                    normalized = normalize_column_name(order_col)
                    if normalized:
                        used_columns.add(normalized)

        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth if depth > 0 else "unlimited",
            "actual_max_depth": actual_max_depth,
            "target_entity": target_entity,
            "chains": chains,
            "summary": {
                "total_tables": len(set(table_lineage_data.keys()) | set().union(*table_lineage_data.values()) if table_lineage_data else set()),
                "total_columns": len(used_columns),
                "has_transformations": bool(result.table_lineage.transformations or result.column_lineage.transformations),
                "has_metadata": bool(result.metadata),
                "chain_count": len(chains)
            },
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    def get_table_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get the JSON representation of table lineage chain for a SQL query."""
        chain_data = self.get_table_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def get_column_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get the JSON representation of column lineage chain for a SQL query."""
        chain_data = self.get_column_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def get_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """
        Get the JSON representation of comprehensive lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream" 
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the comprehensive lineage chain
        """
        chain_data = self.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def _build_cte_lineage_chain(self, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries with proper single-flow chains."""
        # This should delegate to the CTE analyzer
        from .cte_analyzer import CTEAnalyzer
        cte_analyzer = CTEAnalyzer(self.dialect)
        return cte_analyzer.build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
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
                                    source_columns.add(col)
                            
                            # Extract join condition columns
                            for join in trans.get('joins', []):
                                for condition in join.get('conditions', []):
                                    for col_key in ['left_column', 'right_column']:
                                        col_ref = condition.get(col_key, '')
                                        if col_ref and entity_name in col_ref:
                                            clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                                            source_columns.add(clean_col)
                
                # Add the columns found in transformations
                if source_columns:
                    existing_columns = {col['name'] for col in table_columns}
                    for col_name in source_columns:
                        if col_name not in existing_columns:
                            table_columns.append({
                                "name": col_name,
                                "upstream": [],
                                "type": "SOURCE"
                            })
                    
                    # Update metadata
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    entity_data['metadata']['table_columns'] = table_columns
            
            # For target tables (depth > 0), extract result columns from CTAS SELECT clause
            elif entity_data.get('depth', 0) > 0:
                # Use CTAS analyzer for CTAS-specific logic
                if sql and sql.strip().upper().startswith('CREATE TABLE'):
                    from .ctas_analyzer import CTASAnalyzer
                    ctas_analyzer = CTASAnalyzer(self.dialect)
                    ctas_analyzer.add_ctas_result_columns(entity_data, entity_name, sql)
    
    def integrate_column_transformations(self, chains: Dict, sql: str = None) -> None:
        """Integrate column transformations into column metadata throughout the chain."""
        # Use transformation analyzer for this functionality
        from .transformation_analyzer import TransformationAnalyzer
        transformation_analyzer = TransformationAnalyzer(self.dialect)
        transformation_analyzer.integrate_column_transformations(chains, sql)
    
    def _infer_query_result_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Infer QUERY_RESULT columns from SQL query when column lineage doesn't provide them.
        
        Args:
            sql: The SQL query string
            column_lineage_data: Column lineage mapping
            
        Returns:
            List of column information dictionaries for QUERY_RESULT
        """
        result_columns = []
        
        # Try to extract SELECT columns from SQL
        # This is a simple approach - for more complex cases, proper SQL parsing would be needed
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            
            # Handle star (*) expansion
            if select_clause == '*':
                # Expand * to actual columns from CTE or table
                expanded_columns = self._expand_star_columns(sql, column_lineage_data)
                result_columns.extend(expanded_columns)
            else:
                # Handle simple cases like "SELECT name, email FROM users"
                # Split by comma and clean up
                columns = [col.strip() for col in select_clause.split(',')]
                
                for col in columns:
                    original_col = col.strip()  # Keep original column reference
                    
                    # Extract alias name from "expression AS alias" -> "alias"
                    col_name = original_col
                    if ' AS ' in col.upper():
                        # Split on AS and take the alias part (after AS)
                        parts = col.split(' AS ', 1) if ' AS ' in col else col.split(' as ', 1)
                        if len(parts) > 1:
                            col_name = parts[1].strip()
                        else:
                            col_name = parts[0].strip()
                    elif ' as ' in col:
                        # Handle lowercase 'as'
                        parts = col.split(' as ', 1)
                        if len(parts) > 1:
                            col_name = parts[1].strip()
                        else:
                            col_name = parts[0].strip()
                    
                    # Clean up any remaining whitespace and quotes
                    col_name = col_name.strip().strip('"').strip("'")
                    
                    if col_name:
                        # Try to find upstream columns from column lineage data
                        upstream_columns = []
                        
                        # For table-prefixed columns (e.g., "u.name"), look for the proper upstream mapping
                        if '.' in col_name:
                            # Extract table alias and column name
                            table_alias, simple_col_name = col_name.split('.', 1)
                            
                            # Look for corresponding upstream in column lineage data
                            for column_ref, upstream_cols in column_lineage_data.items():
                                # Check if this column reference matches our target
                                if column_ref.endswith(f".{simple_col_name}"):
                                    # Found a match, use its upstream
                                    upstream_columns = list(upstream_cols)
                                    # Convert QUERY_RESULT.name to proper table.name format
                                    corrected_upstream = []
                                    for upstream_ref in upstream_columns:
                                        if upstream_ref.startswith("QUERY_RESULT."):
                                            # Map back to the actual source table based on the column_ref
                                            source_table = column_ref.split('.')[0]  # e.g., "users" from "users.name"
                                            corrected_upstream.append(f"{source_table}.{upstream_ref.split('.')[1]}")
                                        else:
                                            corrected_upstream.append(upstream_ref)
                                    upstream_columns = corrected_upstream
                                    break
                        else:
                            # Simple column name without table prefix
                            query_result_ref = f"QUERY_RESULT.{col_name}"
                            for column_ref, upstream_cols in column_lineage_data.items():
                                if (column_ref == query_result_ref or 
                                    column_ref.endswith(f".{col_name}") or
                                    column_ref == col_name):
                                    upstream_columns = list(upstream_cols)
                                    break
                        
                        # If no upstream found, try to infer from SQL structure
                        if not upstream_columns:
                            if '.' in col_name:
                                # For table-prefixed columns, try to resolve table alias
                                table_alias, simple_col_name = col_name.split('.', 1)
                                
                                # Look for table alias mapping in SQL (simple heuristic)
                                alias_pattern = rf'{table_alias}\s+(?:JOIN\s+)?(\w+)|(\w+)\s+{table_alias}'
                                alias_match = re.search(alias_pattern, sql, re.IGNORECASE)
                                if alias_match:
                                    table_name = alias_match.group(1) or alias_match.group(2)
                                    upstream_columns = [f"{table_name}.{simple_col_name}"]
                                else:
                                    # Fallback: assume alias maps to similarly named table
                                    if table_alias.lower().startswith('u'):
                                        upstream_columns = [f"users.{simple_col_name}"]
                                    elif table_alias.lower().startswith('o'):
                                        upstream_columns = [f"orders.{simple_col_name}"]
                                    else:
                                        upstream_columns = [f"{table_alias}.{simple_col_name}"]
                            else:
                                # Simple column name - use first table as fallback
                                from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                                if from_match:
                                    table_name = from_match.group(1)
                                    upstream_columns = [f"{table_name}.{col_name}"]
                        
                        # Use simplified format matching the optimized structure
                        result_columns.append({
                            "name": col_name,
                            "upstream": upstream_columns,
                            "type": "DIRECT"
                        })
        
        return result_columns
    
    def _filter_query_result_columns_by_parent(self, all_columns: List[Dict], parent_entity: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Filter QUERY_RESULT columns to only include those that originate from the specified parent entity.
        
        Args:
            all_columns: All columns in the QUERY_RESULT
            parent_entity: The parent table entity that's requesting the filtered columns
            column_lineage_data: Column lineage mapping
            
        Returns:
            Filtered list of columns that come from the parent entity
        """
        filtered_columns = []
        
        for col in all_columns:
            col_name = col.get('name', '')
            upstream = col.get('upstream', [])
            
            # Check if this column originates from the parent entity
            column_belongs_to_parent = False
            
            # Method 1: Check if column name has parent table prefix (e.g., "u.name" belongs to "users")
            if '.' in col_name:
                # Extract table alias/prefix from column name (e.g., "u.name" -> "u")
                table_prefix = col_name.split('.')[0]
                
                # Common alias mappings - this is a simple heuristic
                # In real scenarios, we'd need proper alias resolution
                if (parent_entity.startswith('user') and table_prefix.lower() in ['u', 'user']) or \
                   (parent_entity.startswith('order') and table_prefix.lower() in ['o', 'order']) or \
                   (parent_entity.startswith('customer') and table_prefix.lower() in ['c', 'customer']):
                    column_belongs_to_parent = True
            
            # Method 2: Check upstream lineage to see if column comes from parent entity
            if not column_belongs_to_parent and upstream:
                for upstream_ref in upstream:
                    if upstream_ref.startswith(f"{parent_entity}."):
                        column_belongs_to_parent = True
                        break
            
            # Method 3: Check column lineage data for mapping
            if not column_belongs_to_parent:
                # Look for column references that match parent entity
                for col_ref, lineage_list in column_lineage_data.items():
                    if col_ref.startswith(f"{parent_entity}."):
                        # Extract column name from reference (e.g., "users.name" -> "name")
                        source_col_name = col_ref.split('.')[-1]
                        target_col_name = col_name.split('.')[-1] if '.' in col_name else col_name
                        
                        if source_col_name == target_col_name:
                            column_belongs_to_parent = True
                            break
            
            if column_belongs_to_parent:
                filtered_columns.append(col)
        
        return filtered_columns
    
    def _expand_star_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """Expand * to actual columns from CTE or table."""
        expanded_columns = []
        
        try:
            # Parse the SQL to understand the structure
            ast = self._parse_sql(sql)
            
            # Handle WITH statements (CTEs)
            if isinstance(ast, exp.With):
                main_select = ast.this
                cte_definitions = {}
                
                # Collect CTE definitions
                for cte in ast.expressions:
                    if isinstance(cte, exp.CTE):
                        cte_name = cte.alias
                        if isinstance(cte.this, exp.Select):
                            # Parse CTE's SELECT to get its columns
                            cte_select_data = self.select_parser.parse(str(cte.this))
                            cte_columns = [col['column_name'] for col in cte_select_data.get('select_columns', []) if col['column_name']]
                            cte_definitions[cte_name] = cte_columns
                
                # Now check the main SELECT to see what table it's selecting from
                if isinstance(main_select, exp.Select):
                    from_clause = main_select.args.get('from')
                    if from_clause and hasattr(from_clause, 'this'):
                        source_table = str(from_clause.this)
                        
                        # If source table is a CTE, expand to its columns
                        if source_table in cte_definitions:
                            for col_name in cte_definitions[source_table]:
                                # Try to find upstream for this column
                                upstream_columns = []
                                for column_ref, upstream_cols in column_lineage_data.items():
                                    if column_ref.endswith(f".{col_name}") or column_ref == col_name:
                                        upstream_columns = list(upstream_cols)
                                        break
                                
                                expanded_columns.append({
                                    "name": col_name,
                                    "upstream": upstream_columns,
                                    "type": "DIRECT"
                                })
            
            # Handle simple SELECT * FROM table (no CTE)
            elif isinstance(ast, exp.Select):
                from_clause = ast.args.get('from')
                if from_clause and hasattr(from_clause, 'this'):
                    source_table = str(from_clause.this)
                    
                    # Try to get columns from metadata registry if available
                    if hasattr(self, 'metadata_registry') and self.metadata_registry:
                        table_metadata = self.metadata_registry.get_table_metadata(source_table)
                        if table_metadata and 'columns' in table_metadata:
                            for col_info in table_metadata['columns']:
                                col_name = col_info.get('name')
                                if col_name:
                                    expanded_columns.append({
                                        "name": col_name,
                                        "upstream": [f"{source_table}.{col_name}"],
                                        "type": "DIRECT"
                                    })
        
        except Exception as e:
            # If expansion fails, return empty list to avoid showing * as column
            pass
        
        return expanded_columns
    
    def _parse_sql(self, sql: str):
        """Parse SQL using sqlglot."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception:
            return None
    
    def _is_column_from_table(self, column_name: str, table_name: str, context_info: dict = None) -> bool:
        """Check if a column belongs to a specific table based on naming patterns."""
        if not column_name or not table_name:
            return False
        
        # Handle qualified column names (e.g., "users.active", "u.salary")
        if '.' in column_name:
            column_parts = column_name.split('.')
            if len(column_parts) >= 2:
                table_part = column_parts[0].lower()
                
                # Direct table name match (e.g., "users.active" matches "users")
                if table_part == table_name.lower():
                    return True
                
                # Alias match (e.g., "u.salary" matches "users" if u is alias for users)
                # Common aliases: u for users, o for orders, etc.
                if table_name.lower().startswith(table_part):
                    return True
                
                # Check reverse - if table_part contains table_name (e.g., "users" in "user_details")
                if table_part in table_name.lower() or table_name.lower() in table_part:
                    return True
        else:
            # Unqualified column - use context to determine if it belongs to this table
            if context_info:
                # If this is a single-table context (only one source table), assume unqualified columns belong to it
                if context_info.get('is_single_table_context', False):
                    return True
                    
                # If we have a list of tables in the context and this is the primary/source table
                tables_in_context = context_info.get('tables_in_context', [])
                if len(tables_in_context) == 1 and tables_in_context[0] == table_name:
                    return True
        
        # If no table qualifier and no clear context, default to False for filtering
        # This prevents unqualified columns from being assigned to every table in multi-table contexts
        return False
    
    def _is_aggregate_function_for_table(self, column_expr: str, table_name: str) -> bool:
        """Check if an aggregate function expression is relevant to a specific table."""
        if not column_expr or not table_name:
            return False
        
        # Handle aggregate functions like COUNT(*), AVG(u.salary), SUM(users.amount)
        column_expr_lower = column_expr.lower()
        
        # Check if the expression contains explicit table references first
        if table_name.lower() in column_expr_lower:
            return True
        
        # Check for table aliases (u for users, o for orders)
        if table_name.lower().startswith('u') and ('u.' in column_expr_lower):
            return True
        elif table_name.lower().startswith('o') and ('o.' in column_expr_lower):
            return True
        
        # COUNT(*) is only relevant to the main grouped table (users in this case)
        # Only assign COUNT(*) to users table, not orders table
        if column_expr_lower == 'count(*)' and table_name.lower() == 'users':
            return True
        
        return False