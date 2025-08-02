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