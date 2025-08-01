"""Chain builder for SQL lineage analysis."""

from typing import Dict, Any, List, Optional, Set
from .base_analyzer import BaseAnalyzer


class ChainBuilder(BaseAnalyzer):
    """Handles building lineage chains from analysis results."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize chain builder."""
        super().__init__(dialect)
        # These will be injected from the main analyzer
        self.transformation_extractor = None
    
    def get_table_lineage_chain(self, result, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the table lineage chain for a SQL query with specified direction and depth.
        
        Args:
            result: Analysis result from main analyzer
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
    
    def get_lineage_chain(self, result, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive lineage chain with unlimited depth support and metadata integration.
        
        Args:
            result: Analysis result from main analyzer
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing comprehensive lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 0:
            raise ValueError("depth must be 0 or greater (0 = unlimited)")
        
        # Handle CTE queries differently
        if "WITH " in sql.upper():
            return self._build_cte_lineage_chain(result, sql, chain_type, depth, target_entity, **kwargs)
        
        # Get table and column lineage data
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Build comprehensive chains with unlimited depth support
        chains = {}
        
        def build_comprehensive_chain(table_name: str, current_depth: int, visited_in_path: Set[str] = None) -> Dict[str, Any]:
            """Build a comprehensive chain with metadata and transformations."""
            if visited_in_path is None:
                visited_in_path = set()
            
            # Prevent cycles but allow unlimited depth if depth=0
            if table_name in visited_in_path:
                return {
                    "entity": table_name,
                    "entity_type": "table",
                    "depth": current_depth,
                    "dependencies": [],
                    "metadata": self._get_table_metadata_from_result(result, table_name)
                }
            
            # Check depth limit (only if depth > 0, otherwise unlimited)
            if depth > 0 and current_depth >= depth:
                return {
                    "entity": table_name,
                    "entity_type": "table", 
                    "depth": current_depth,
                    "dependencies": [],
                    "metadata": self._get_table_metadata_from_result(result, table_name)
                }
            
            visited_in_path = visited_in_path | {table_name}
            
            # Build dependencies
            dependencies = []
            if table_name in table_lineage_data:
                for dependent_table in table_lineage_data[table_name]:
                    dep_chain = build_comprehensive_chain(dependent_table, current_depth + 1, visited_in_path)
                    
                    # Add transformations from result data
                    transformations = self._get_transformations_from_result(result, table_name, dependent_table)
                    if transformations or chain_type == "downstream":
                        # For downstream, we always include transformation information
                        dep_chain["transformations"] = transformations
                    
                    dependencies.append(dep_chain)
            
            # Build entity with metadata
            entity = {
                "entity": table_name,
                "entity_type": "table",
                "depth": current_depth,
                "dependencies": dependencies,
                "metadata": self._get_table_metadata_from_result(result, table_name)
            }
            
            return entity
        
        # Build chains from root tables
        entities_in_dependencies = set()
        temp_chains = {}
        
        # First pass: build all chains 
        for table_name in table_lineage_data.keys():
            temp_chains[table_name] = build_comprehensive_chain(table_name, 0)
        
        # Collect all entities that appear as dependencies 
        def collect_dependency_entities(chain_data):
            for dep in chain_data.get("dependencies", []):
                entities_in_dependencies.add(dep.get("entity"))
                collect_dependency_entities(dep)
        
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
            
            # Remove target tables from top-level chains - they should appear as dependencies
            for target_table in ctas_target_tables:
                if target_table in chains:
                    del chains[target_table]
        
        # Add missing source columns based on transformations
        self._add_missing_source_columns(chains, sql)
        
        # Integrate column transformations 
        if self.transformation_extractor:
            self.transformation_extractor.integrate_column_transformations(chains, sql)
        
        # Calculate actual max depth used in the chains
        actual_max_depth = 0
        for chain_data in chains.values():
            # Get max depth from direct chain
            if isinstance(chain_data, dict) and "depth" in chain_data:
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
    
    def get_lineage_chain_json(self, result, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """
        Get the JSON representation of comprehensive lineage chain for a SQL query.
        
        Args:
            result: Analysis result from main analyzer
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream" 
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the comprehensive lineage chain
        """
        import json
        chain_data = self.get_lineage_chain(result, sql, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def _add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
        """Add missing source columns to table metadata by extracting from transformations."""
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies OR add to existing
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
                
                # Add the columns found in transformations, merging with existing
                if source_columns:
                    # Get existing column names to avoid duplicates
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
    
    def _get_table_metadata_from_result(self, result, table_name: str) -> Dict[str, Any]:
        """Extract table metadata from analysis result."""
        metadata = {}
        
        # Get essential table metadata (excluding detailed column info) - following backup implementation
        if hasattr(result, 'metadata') and result.metadata and table_name in result.metadata:
            table_meta = result.metadata[table_name]
            if table_meta:  # Check if table_meta is not None
                # Safely get table_type
                if hasattr(table_meta, 'table_type') and table_meta.table_type:
                    table_type_value = table_meta.table_type.value if hasattr(table_meta.table_type, 'value') else str(table_meta.table_type)
                    metadata["table_type"] = table_type_value
                
                # Only include non-null values to keep output clean
                if hasattr(table_meta, 'schema') and table_meta.schema:
                    metadata["schema"] = table_meta.schema
                if hasattr(table_meta, 'description') and table_meta.description:
                    metadata["description"] = table_meta.description
        
        # Column information will be added separately by the main lineage building logic
        # following the pattern in backup file where columns come from column_lineage_data
        if not metadata:
            metadata = {
                "table_columns": [],
                "is_cte": False
            }
        
        return metadata
    
    def _get_transformations_from_result(self, result, source_table: str, target_table: str) -> List[Dict[str, Any]]:
        """Extract transformations from analysis result."""
        transformations = []
        
        # Get transformations from result
        if hasattr(result, 'table_lineage') and result.table_lineage.transformations:
            for trans_list in result.table_lineage.transformations.values():
                for trans in trans_list:
                    if trans.source_table == source_table and trans.target_table == target_table:
                        trans_dict = {
                            "type": "table_transformation",
                            "source_table": trans.source_table,
                            "target_table": trans.target_table
                        }
                        
                        # Add various transformation details
                        if trans.filter_conditions:
                            trans_dict["filter_conditions"] = [
                                {
                                    "column": fc.column,
                                    "operator": fc.operator,
                                    "value": fc.value
                                }
                                for fc in trans.filter_conditions
                            ]
                        
                        if trans.join_conditions:
                            trans_dict["joins"] = [
                                {
                                    "join_type": trans.join_type.value if trans.join_type else "INNER",
                                    "right_table": jc.right_column.split('.')[0] if '.' in jc.right_column else None,
                                    "conditions": [
                                        {
                                            "left_column": jc.left_column,
                                            "operator": jc.operator.value if hasattr(jc.operator, 'value') else str(jc.operator),
                                            "right_column": jc.right_column
                                        }
                                    ]
                                }
                                for jc in trans.join_conditions
                            ]
                        
                        if trans.group_by_columns:
                            trans_dict["group_by_columns"] = trans.group_by_columns
                        
                        if trans.having_conditions:
                            trans_dict["having_conditions"] = [
                                {
                                    "column": hc.column,
                                    "operator": hc.operator,
                                    "value": hc.value
                                }
                                for hc in trans.having_conditions
                            ]
                        
                        if trans.order_by_columns:
                            trans_dict["order_by_columns"] = trans.order_by_columns
                        
                        transformations.append(trans_dict)
        
        return transformations
    
    def _build_cte_lineage_chain(self, result, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries - this will be implemented by CTEAnalyzer."""
        # This method will be delegated to CTEAnalyzer
        raise NotImplementedError("CTE lineage chain building should be handled by CTEAnalyzer")
    
    def _add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
        """Add missing source columns to table metadata by extracting from transformations."""
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies OR add to existing
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
                
                # Add the columns found in transformations, merging with existing
                if source_columns:
                    # Get existing column names to avoid duplicates
                    existing_columns = {col.get('name') for col in table_columns if isinstance(col, dict)}
                    
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