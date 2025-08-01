"""CTE-specific analyzer methods."""

from typing import Dict, Any, List, Optional, Set
from .base_analyzer import BaseAnalyzer


class CTEAnalyzer(BaseAnalyzer):
    """Handles CTE-specific analysis and chain building."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize CTE analyzer."""
        super().__init__(dialect)
        # These will be injected from the main analyzer
        self.cte_parser = None
        self.main_analyzer = None
    
    def build_cte_lineage_chain(self, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries with proper single-flow chains."""
        # Get CTE-specific analysis
        cte_result = self._analyze_cte(sql)
        cte_lineage = cte_result.get('cte_lineage', {})
        ctes = cte_lineage.get('ctes', {})
        execution_order = cte_lineage.get('execution_order', [])
        
        # Also get standard table/column lineage for base tables and final result
        result = self.main_analyzer.analyze(sql, **kwargs)
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        
        # Build chains dictionary
        chains = {}
        
        if chain_type == "downstream":
            # NEW APPROACH: Build single continuous chains from base tables through CTEs to QUERY_RESULT
            
            # 1. Identify base tables (non-CTE tables)
            base_tables = set()
            
            # Tables that CTEs depend on
            for cte_name, cte_data in ctes.items():
                source_tables = cte_data.get('source_tables', [])
                for source in source_tables:
                    table_name = source.get('name')
                    if table_name and table_name not in ctes:  # Not a CTE
                        base_tables.add(table_name)
            
            # Tables that final query references directly (like users in JOINs)
            if table_lineage_data:
                for table_name in table_lineage_data.keys():
                    if table_name not in ctes:  # Not a CTE
                        base_tables.add(table_name)
            
            # 2. For each base table, build a single continuous dependency chain
            for table_name in base_tables:
                chains[table_name] = self._build_single_cte_chain(table_name, ctes, execution_order, table_lineage_data, sql)
            
            # 3. Handle any orphaned CTEs (CTEs that don't connect to base tables)
            # This shouldn't happen in well-formed queries, but handle gracefully
            for cte_name in execution_order:
                if cte_name not in ctes:
                    continue
                    
                # Check if this CTE is already included in any base table chain
                cte_included = False
                for base_table in base_tables:
                    if self._cte_in_chain(cte_name, chains.get(base_table, {})):
                        cte_included = True
                        break
                
                # If CTE is not included in any chain, add it as a separate chain
                if not cte_included:
                    chains[cte_name] = self._build_single_cte_chain(cte_name, ctes, execution_order, table_lineage_data)
        
        else:  # upstream
            # For upstream: start from final result, trace back through CTEs to base tables
            
            # 1. Add final result
            final_result = cte_lineage.get('final_result', {})
            if final_result:
                chains['QUERY_RESULT'] = self._build_cte_final_result(final_result, ctes, execution_order, 0, table_lineage_data)
            
            # 2. Add CTE entities in reverse execution order
            for i, cte_name in enumerate(reversed(execution_order)):
                if cte_name in ctes:
                    cte_entity = self._build_cte_entity(cte_name, ctes[cte_name], ctes, execution_order, i + 1)
                    chains[cte_name] = cte_entity
            
            # 3. Add base tables
            base_tables = set()
            for cte_name, cte_data in ctes.items():
                source_tables = cte_data.get('source_tables', [])
                for source in source_tables:
                    table_name = source.get('name')
                    if table_name and table_name not in ctes:  # Not a CTE
                        base_tables.add(table_name)
            
            for table_name in base_tables:
                chains[table_name] = self._build_cte_table_entity(table_name, ctes, execution_order, len(execution_order) + len(base_tables))
        
        # Build final result structure
        actual_max_depth = max([entity.get('depth', 0) for entity in chains.values()]) if chains else 0
        
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
                "total_columns": 0,  # CTE queries don't have column lineage in the same way
                "has_transformations": bool(cte_result.get('transformations')),
                "has_metadata": bool(result.metadata),
                "chain_count": len(chains)
            },
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    def _analyze_cte(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE query using CTE parser."""
        if self.cte_parser:
            return self.cte_parser.get_cte_lineage_chain(sql)
        return {}
    
    def _build_cte_entity(self, cte_name: str, cte_data: Dict, all_ctes: Dict, execution_order: List[str], depth: int) -> Dict[str, Any]:
        """Build entity data for a CTE."""
        entity = {
            "entity": cte_name,
            "entity_type": "cte",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": self._get_entity_metadata(cte_name, all_ctes)
        }
        
        # For downstream flow, CTEs should point to what depends on them (next CTE in chain or QUERY_RESULT)
        # Find what this CTE connects to in the execution chain
        
        # First, check if any other CTE depends on this one
        cte_dependencies = []
        for next_cte_name in execution_order:
            if next_cte_name != cte_name and next_cte_name in all_ctes:
                next_cte_data = all_ctes[next_cte_name]
                next_source_tables = next_cte_data.get('source_tables', [])
                
                # Check if the next CTE depends on this CTE
                for source in next_source_tables:
                    if source.get('name') == cte_name:
                        # This CTE should point to the next CTE in the chain
                        transformations = [{
                            'type': 'table_transformation',
                            'source_table': cte_name,
                            'target_table': next_cte_name,
                            'filter_conditions': [],
                            'group_by_columns': [],
                            'joins': []
                        }]
                        
                        cte_dependencies.append({
                            "entity": next_cte_name,
                            "transformations": transformations
                        })
                        break
        
        # Add the CTE dependencies to the entity
        entity["dependencies"].extend(cte_dependencies)
        
        return entity
    
    def _build_cte_table_entity(self, table_name: str, all_ctes: Dict, execution_order: List[str], depth: int) -> Dict[str, Any]:
        """Build entity data for a base table used by CTEs."""
        entity = {
            "entity": table_name,
            "entity_type": "table",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": {
                "table_columns": [],
                "is_cte": False
            }
        }
        
        # For downstream flow, base tables should have dependencies pointing to CTEs that use them
        # This creates the proper flow: base_table → CTE → QUERY_RESULT
        
        # Find CTEs that depend on this base table and add them as dependencies
        connected_to_cte = False
        for cte_name in execution_order:
            if cte_name in all_ctes:
                cte_data = all_ctes[cte_name]
                source_tables = cte_data.get('source_tables', [])
                
                # Check if this CTE uses the current base table
                for source in source_tables:
                    if source.get('name') == table_name:
                        # Add this CTE as a dependency of the base table for downstream flow
                        # Apply table-specific filtering to CTE transformations
                        cte_transformations = cte_data.get('transformations', [])
                        filtered_conditions = self._filter_nested_conditions(cte_transformations, table_name)
                        
                        transformations = [{
                            'type': 'table_transformation',
                            'source_table': table_name,
                            'target_table': cte_name,
                            'filter_conditions': filtered_conditions,
                            'group_by_columns': [],
                            'joins': []
                        }]
                        
                        entity["dependencies"].append({
                            "entity": cte_name,
                            "transformations": transformations
                        })
                        connected_to_cte = True
                        break
        
        # If this base table is not connected to any CTE but appears in final result,
        # it should point directly to QUERY_RESULT (like users table in JOINs)
        if not connected_to_cte:
            # This base table is only referenced in the final SELECT, so add QUERY_RESULT dependency
            transformations = [{
                'type': 'table_transformation',
                'source_table': table_name,
                'target_table': 'QUERY_RESULT',
                'filter_conditions': [],
                'group_by_columns': [],
                'joins': []
            }]
            
            entity["dependencies"].append({
                "entity": "QUERY_RESULT",
                "transformations": transformations
            })
        
        return entity
    
    def _build_cte_final_result(self, final_result: Dict, all_ctes: Dict, execution_order: List[str], depth: int, table_lineage_data: Dict = None) -> Dict[str, Any]:
        """Build entity data for the final query result."""
        entity = {
            "entity": "QUERY_RESULT",
            "entity_type": "query_result",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": {
                "table_columns": [],
                "is_cte": False
            }
        }
        
        # Analyze final result to find what it DIRECTLY depends on
        # Use the parsed structure instead of string parsing
        referenced_entities = set()
        
        # Extract table references from FROM clause
        from_tables = final_result.get('from_tables', [])
        for table_info in from_tables:
            table_name = table_info.get('table_name')
            if table_name:
                referenced_entities.add(table_name)
        
        # Extract table references from JOIN clauses
        joins = final_result.get('joins', [])
        for join_info in joins:
            table_name = join_info.get('table_name')
            if table_name:
                referenced_entities.add(table_name)
        
        # Fallback: if no references found from parsed structure, try string matching
        if not referenced_entities:
            final_sql = str(final_result)
            for cte_name in execution_order:
                if cte_name.lower() in final_sql.lower():
                    referenced_entities.add(cte_name)
        
        # Additional fallback: use table lineage data for direct table references
        if not referenced_entities and table_lineage_data:
            final_sql = str(final_result)
            for table_name in table_lineage_data.keys():
                if table_name.lower() not in [cte.lower() for cte in execution_order]:
                    if table_name.lower() in final_sql.lower():
                        referenced_entities.add(table_name)
        
        # Add dependencies
        for entity_name in referenced_entities:
            entity_type = "cte" if entity_name in all_ctes else "table"
            transformations = self._get_final_transformations(entity_name, table_lineage_data)
            
            entity["dependencies"].append({
                "entity": entity_name,
                "entity_type": entity_type,
                "depth": depth + 1,
                "dependencies": [],
                "transformations": transformations,
                "metadata": self._get_entity_metadata(entity_name, all_ctes) if entity_name in all_ctes else {"table_columns": [], "is_cte": False}
            })
        
        return entity
    
    def _build_single_cte_chain(self, start_entity: str, ctes: dict, execution_order: list, table_lineage_data: dict, sql: str = None) -> dict:
        """Build a single continuous chain from a base table through CTEs to QUERY_RESULT."""
        
        # For the nested CTE example:
        # orders → order_stats → customer_tiers → tier_summary → QUERY_RESULT
        # users → QUERY_RESULT
        
        # Find the complete chain starting from this entity
        chain = self._build_complete_cte_chain_from_entity(start_entity, ctes, execution_order)
        
        # Build the nested dependency structure
        root_entity = None
        current_entity_dict = None
        
        for i, entity_name in enumerate(chain):
            entity_dict = {
                "entity": entity_name,
                "entity_type": "cte" if entity_name in ctes else "table",
                "depth": i,
                "dependencies": [],
                "metadata": self._get_entity_metadata(entity_name, ctes, sql)
            }
            
            # Add transformations if not the first entity
            if i > 0:
                prev_entity = chain[i-1]
                transformations = self._get_cte_transformations(prev_entity, entity_name, ctes)
                if transformations:
                    entity_dict["transformations"] = transformations
            
            if i == 0:
                # This is the root entity
                root_entity = entity_dict
                current_entity_dict = entity_dict
            else:
                # Add as dependency of the previous entity
                current_entity_dict["dependencies"].append(entity_dict)
                current_entity_dict = entity_dict
        
        # Add QUERY_RESULT as final dependency
        query_result_entity = {
            "entity": "QUERY_RESULT",
            "entity_type": "table", 
            "depth": len(chain),
            "dependencies": [],
            "metadata": {"table_columns": [], "is_cte": False}
        }
        
        # Add transformations from last entity to QUERY_RESULT
        if chain:
            last_entity = chain[-1]
            transformations = self._get_final_transformations(last_entity, table_lineage_data, sql)
            if transformations:
                query_result_entity["transformations"] = transformations
        
        if current_entity_dict:
            current_entity_dict["dependencies"].append(query_result_entity)
        
        return root_entity if root_entity else query_result_entity
    
    def _build_complete_cte_chain_from_entity(self, start_entity: str, ctes: dict, execution_order: list) -> list:
        """Build the complete chain of entities starting from a base table or CTE."""
        
        # The chain should be: [start_entity, cte1, cte2, ..., final_cte]
        # For orders: [orders, order_stats, customer_tiers, tier_summary]  
        # For users: [users] (no CTEs)
        
        chain = [start_entity]
        current_entity = start_entity
        
        # Follow the CTE dependency chain
        while True:
            next_cte = None
            
            # Find the next CTE that depends on the current entity
            for cte_name in execution_order:
                if cte_name in ctes:
                    cte_data = ctes[cte_name]
                    source_tables = cte_data.get('source_tables', [])
                    
                    # Check if this CTE depends on the current entity
                    for source in source_tables:
                        if source.get('name') == current_entity:
                            next_cte = cte_name
                            break
                    
                    if next_cte:
                        break
            
            if next_cte:
                chain.append(next_cte)
                current_entity = next_cte
            else:
                break
        
        return chain
    
    def _find_cte_chain_from_entity(self, start_entity: str, ctes: dict, execution_order: list) -> list:
        """Find the chain of CTEs that flows from the given entity."""
        chain = []
        
        # If start_entity is a CTE, start from it
        if start_entity in ctes:
            current_cte = start_entity
        else:
            # Find the first CTE that depends on this base table
            current_cte = None
            for cte_name in execution_order:
                if cte_name in ctes:
                    cte_data = ctes[cte_name]
                    source_tables = cte_data.get('source_tables', [])
                    for source in source_tables:
                        if source.get('name') == start_entity:
                            current_cte = cte_name
                            break
                    if current_cte:
                        break
        
        # Build the chain following the CTE dependencies
        while current_cte and current_cte in ctes:
            if current_cte != start_entity:  # Don't include start entity in chain
                chain.append(current_cte)
            
            # Find next CTE that depends on current_cte
            next_cte = None
            for cte_name in execution_order:
                if cte_name in ctes and cte_name != current_cte:
                    cte_data = ctes[cte_name]
                    source_tables = cte_data.get('source_tables', [])
                    for source in source_tables:
                        if source.get('name') == current_cte:
                            next_cte = cte_name
                            break
                    if next_cte:
                        break
            
            current_cte = next_cte
        
        return chain
    
    def _get_entity_metadata(self, entity_name: str, ctes: dict, sql: str = None) -> dict:
        """Get metadata for an entity (table or CTE)."""
        if entity_name in ctes:
            cte_data = ctes[entity_name]
            
            # Get columns from CTE data
            columns = cte_data.get('columns', [])
            table_columns = []
            
            # Extract column transformations for this CTE if SQL is available
            column_transformations_map = {}
            if sql and self.main_analyzer and hasattr(self.main_analyzer, 'transformation_extractor'):
                try:
                    column_transformations = self.main_analyzer.transformation_extractor.extract_cte_column_transformations(sql, entity_name)
                    for col_trans in column_transformations:
                        col_name = col_trans.get('column_name')
                        if col_name:
                            column_transformations_map[col_name] = col_trans
                except Exception:
                    pass
            
            # Build table_columns with transformations
            for col in columns:
                col_name = col.get('name') or col.get('alias')
                if col_name:
                    column_info = {
                        "name": col_name,
                        "upstream": [],
                        "type": "VARCHAR"  # Default type for CTEs
                    }
                    
                    # Add transformation if available
                    if col_name in column_transformations_map:
                        col_trans = column_transformations_map[col_name]
                        column_info["transformation"] = {
                            "column_name": col_trans.get('column_name'),
                            "source_expression": col_trans.get('source_expression'),
                            "transformation_type": col_trans.get('transformation_type'),
                            "function_type": col_trans.get('function_type'),
                            "full_expression": col_trans.get('full_expression')
                        }
                    
                    table_columns.append(column_info)
            
            return {
                "table_columns": table_columns,
                "is_cte": True
            }
        else:
            # Base table metadata (simplified)
            return {
                "table_columns": [],
                "is_cte": False
            }
    
    def _get_cte_transformations(self, source_entity: str, target_entity: str, ctes: dict) -> list:
        """Get transformations between two entities in CTE chain."""
        if target_entity in ctes:
            cte_data = ctes[target_entity]
            transformations = cte_data.get('transformations', [])
            
            # Filter transformations that are relevant to the source entity
            return [
                {
                    'type': 'table_transformation',
                    'source_table': source_entity,
                    'target_table': target_entity,
                    'filter_conditions': [
                        {
                            'type': 'FILTER',
                            'conditions': [
                                {
                                    'column': condition.get('column', ''),
                                    'operator': condition.get('operator', ''),
                                    'value': condition.get('value', '')
                                }
                            ]
                        }
                        for condition in transformations if condition.get('type') == 'filter'
                    ],
                    'group_by_columns': [
                        group.get('column') for group in transformations if group.get('type') == 'group_by'
                    ],
                    'joins': []
                }
            ]
        
        return []
    
    def _get_final_transformations(self, source_entity: str, table_lineage_data: dict, sql: str = None) -> list:
        """Get transformations from source entity to QUERY_RESULT."""
        # Extract transformations from main analyzer result
        if self.main_analyzer and hasattr(self.main_analyzer, 'transformation_extractor'):
            try:
                # Get transformations from the main analyzer for this entity
                return [
                    {
                        'type': 'table_transformation',
                        'source_table': source_entity,
                        'target_table': 'QUERY_RESULT',
                        'filter_conditions': [],
                        'group_by_columns': [],
                        'joins': []
                    }
                ]
            except Exception:
                pass
        
        return []