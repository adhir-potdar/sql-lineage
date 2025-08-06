"""CTE (Common Table Expression) analyzer."""

from typing import Dict, Any, List, Optional
import sqlglot
from .base_analyzer import BaseAnalyzer

# Import new utility modules
from ...utils.sqlglot_helpers import (
    parse_sql_safely, get_cte_definitions, traverse_ast_nodes
)
from ...utils.column_extraction_utils import extract_all_referenced_columns
from ...utils.metadata_utils import create_cte_metadata, merge_metadata_entries
from ...utils.sql_parsing_utils import extract_function_type, extract_alias_from_expression, TableNameRegistry, CompatibilityMode
from ...utils.regex_patterns import is_aggregate_function
from ..chain_builder_engine import ChainBuilderEngine


class CTEAnalyzer(BaseAnalyzer):
    """Analyzer for CTE statements."""
    
    def __init__(self, dialect: str = "trino", main_analyzer=None, table_registry: TableNameRegistry = None):
        """Initialize CTE analyzer with chain builder engine."""
        # Get compatibility mode from main analyzer if available
        compatibility_mode = getattr(main_analyzer, 'compatibility_mode', CompatibilityMode.FULL) if main_analyzer else CompatibilityMode.FULL
        registry = table_registry or getattr(main_analyzer, 'table_registry', None)
        
        super().__init__(dialect, compatibility_mode, registry)
        self.main_analyzer = main_analyzer
        self.chain_builder_engine = ChainBuilderEngine(dialect)
    
    def analyze_cte(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE statement."""
        cte_data = self.cte_parser.parse(sql)
        cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'cte_structure': cte_data,
            'cte_lineage': cte_lineage,
            'transformations': transformation_data,
            'execution_order': cte_lineage.get('execution_order', []),
            'final_result': cte_lineage.get('final_result', {}),
            'cte_dependencies': cte_data.get('cte_dependencies', {})
        }
    
    def build_cte_lineage_chain(self, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries with proper single-flow chains."""
        # Get CTE-specific analysis
        cte_result = self.analyze_cte(sql)
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
            
            # Normalize base table names using registry to eliminate duplicates
            original_to_canonical = {}
            canonical_to_original = {}
            
            if self.table_registry and base_tables:
                self.table_registry.register_tables(base_tables)
                
                for table_name in base_tables:
                    canonical_name = self.table_registry.get_canonical_name(table_name)
                    original_to_canonical[table_name] = canonical_name
                    canonical_to_original[canonical_name] = table_name
            else:
                # No normalization - identity mapping
                for table_name in base_tables:
                    original_to_canonical[table_name] = table_name
                    canonical_to_original[table_name] = table_name
            
            # 2. For each canonical base table, build a single continuous dependency chain
            # Use original names for CTE lookups but store with canonical names
            canonical_base_tables = set(canonical_to_original.keys())
            
            for canonical_name in canonical_base_tables:
                original_name = canonical_to_original[canonical_name]
                # Build chain using original name for CTE lookups but canonical name for output
                chain = self._build_single_cte_chain(original_name, ctes, execution_order, table_lineage_data, sql, canonical_name)
                # Store with canonical name
                chains[canonical_name] = chain
            
            # 3. Handle any orphaned CTEs (CTEs that don't connect to base tables)
            # This shouldn't happen in well-formed queries, but handle gracefully
            for cte_name in execution_order:
                if cte_name not in ctes:
                    continue
                    
                # Check if this CTE is already included in any base table chain
                cte_included = False
                for canonical_name in canonical_base_tables:
                    if self._cte_in_chain(cte_name, chains.get(canonical_name, {})):
                        cte_included = True
                        break
                
                # If CTE is not included in any chain, add it as a separate chain
                if not cte_included:
                    chains[cte_name] = self._build_single_cte_chain(cte_name, ctes, execution_order, table_lineage_data, sql)
        
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
        
        # Add missing column information to QUERY_RESULT entities  
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        column_transformations_data = result.column_lineage.transformations if hasattr(result.column_lineage, 'transformations') else {}
        
        # First call regular method for basic population
        self.chain_builder_engine.add_missing_source_columns(chains, sql, column_lineage_data, column_transformations_data)
        
        # CTE-specific enhancements: populate both source tables AND QUERY_RESULTs
        self._populate_cte_source_tables(chains, sql, result)
        self._populate_cte_query_results(chains, sql, column_lineage_data, column_transformations_data)
        
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
    
    def _build_cte_entity(self, cte_name: str, cte_data: Dict, all_ctes: Dict, execution_order: List[str], depth: int) -> Dict[str, Any]:
        """Build entity data for a CTE."""
        entity = {
            "entity": cte_name,
            "entity_type": "cte",
            "depth": depth,
            "dependencies": [],
            "transformations": [],
            "metadata": {
                "table_columns": [],
                "is_cte": True
            }
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
        
        # Add columns from CTE definition
        columns = cte_data.get('columns', [])
        for col in columns:
            column_name = col.get('alias') or col.get('name', 'unknown')
            
            column_info = {
                'name': column_name,
                'type': 'COMPUTED' if col.get('is_computed') else 'DIRECT',
                'source_column': col.get('source_column'),
                'source_table': col.get('source_table'),
                'expression': col.get('expression')
            }
            
            # Add transformation details for computed columns
            if col.get('is_computed') or col.get('is_aggregate') or col.get('is_function'):
                raw_expression = col.get('raw_expression', col.get('expression', ''))
                
                if raw_expression:
                    # Extract source expression and determine transformation type
                    source_expression = self._extract_source_from_expression(raw_expression, column_name)
                    transformation_type = self._get_transformation_type(col, raw_expression)
                    function_type = self._extract_function_type_generic(raw_expression)
                    
                    column_info["transformation"] = {
                        "source_expression": source_expression,
                        "transformation_type": transformation_type,
                        "function_type": function_type
                    }
            
            entity["metadata"]["table_columns"].append(column_info)
        
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
            transformations = [{
                'type': 'table_transformation',
                'source_table': entity_name,
                'target_table': 'QUERY_RESULT',
                'filter_conditions': [],
                'group_by_columns': [],
                'joins': []
            }]
            
            entity["dependencies"].append({
                "entity": entity_name,
                "transformations": transformations
            })
        
        return entity
    
    def _build_single_cte_chain(self, start_entity: str, ctes: dict, execution_order: list, table_lineage_data: dict, sql: str = None, canonical_name: str = None) -> dict:
        """Build a single continuous chain from a base table through CTEs to QUERY_RESULT."""
        
        # For the nested CTE example:
        # orders → order_stats → customer_tiers → tier_summary → QUERY_RESULT
        # users → QUERY_RESULT
        
        # Find the complete chain starting from this entity (using original name for lookups)
        chain = self._build_complete_cte_chain_from_entity(start_entity, ctes, execution_order)
        
        # Build the nested dependency structure
        root_entity = None
        current_entity_dict = None
        
        for i, entity_name in enumerate(chain):
            # Use canonical name for the root entity (base table), original names for CTEs
            display_name = canonical_name if (i == 0 and canonical_name) else entity_name
            
            entity_dict = {
                "entity": display_name,
                "entity_type": "cte" if entity_name in ctes else "table",
                "depth": i,
                "dependencies": [],
                "metadata": self._get_entity_metadata(entity_name, ctes, sql)
            }
            
            # Add transformations if not the first entity
            if i > 0:
                prev_entity = chain[i-1]
                # Use canonical name for source table in transformations
                prev_display_name = canonical_name if (i == 1 and canonical_name) else prev_entity
                transformations = self._get_cte_transformations(prev_display_name, entity_name, ctes)
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
                        source_name = source.get('name')
                        if self._table_names_match(source_name, current_entity):
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
    
    def _table_names_match(self, source_name: str, target_name: str) -> bool:
        """Check if two table names refer to the same table, accounting for canonical vs. original names."""
        if not source_name or not target_name:
            return False
        
        # Direct match
        if source_name == target_name:
            return True
        
        # If we have a table registry, check if they normalize to the same canonical name
        if self.table_registry:
            source_canonical = self.table_registry.get_canonical_name(source_name)
            target_canonical = self.table_registry.get_canonical_name(target_name)
            if source_canonical == target_canonical:
                return True
        
        # Check if one is the base name of the other (e.g., "orders" vs "catalog.schema.orders")
        source_base = source_name.split('.')[-1].strip('"')
        target_base = target_name.split('.')[-1].strip('"')
        
        return source_base.lower() == target_base.lower()
    
    def _cte_in_chain(self, cte_name: str, chain_entity: dict) -> bool:
        """Check if a CTE is included in a dependency chain."""
        if not chain_entity:
            return False
            
        # Check if this entity is the CTE
        if chain_entity.get("entity") == cte_name:
            return True
        
        # Recursively check dependencies
        for dep in chain_entity.get("dependencies", []):
            if self._cte_in_chain(cte_name, dep):
                return True
        
        return False
    
    def _get_entity_metadata(self, entity_name: str, ctes: dict, sql: str = None) -> dict:
        """Get metadata for an entity (table or CTE)."""
        if entity_name in ctes:
            cte_data = ctes[entity_name]
            
            # Get columns from CTE data
            columns = cte_data.get('columns', [])
            table_columns = []
            
            # Extract column transformations for this CTE if SQL is available
            column_transformations_map = {}
            if sql:
                try:
                    column_transformations = self._extract_column_transformations(sql, None, entity_name)
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
                        column_info["transformation"] = column_transformations_map[col_name]
                    
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
            if transformations:
                return [{
                    "type": "table_transformation",
                    "source_table": source_entity,
                    "target_table": target_entity,
                    "filter_conditions": self._filter_nested_conditions(transformations, source_entity),
                    "group_by_columns": [],
                    "joins": []
                }]
        return []
    
    def _get_final_transformations(self, source_entity: str, table_lineage_data: dict, sql: str = None) -> list:
        """Get transformations from final CTE/table to QUERY_RESULT."""
        transformation = {
            "type": "table_transformation",
            "source_table": source_entity,
            "target_table": "QUERY_RESULT",
            "filter_conditions": [],
            "group_by_columns": [],
            "joins": []
        }
        
        # Extract transformations from the main SELECT query using transformation parser
        if sql:
            try:
                transformation_data = self.transformation_parser.parse(sql)
                
                # Extract filter conditions from main SELECT WHERE clause
                filters = transformation_data.get('filters', {})
                if filters and filters.get('conditions'):
                    # Convert transformation parser format to our expected format
                    filter_conditions = []
                    for condition in filters['conditions']:
                        # Transform format to match expected structure
                        filter_conditions.append({
                            "type": "FILTER",
                            "conditions": [condition]
                        })
                    
                    transformation["filter_conditions"] = filter_conditions
                
                # Extract aggregation information (GROUP BY, HAVING, aggregate functions)
                aggregations = transformation_data.get('aggregations', {})
                if aggregations:
                    # Group by columns
                    if aggregations.get('group_by_columns'):
                        transformation["group_by_columns"] = aggregations['group_by_columns']
                    
                    # Having conditions
                    if aggregations.get('having_conditions'):
                        having_conditions = []
                        for condition in aggregations['having_conditions']:
                            having_conditions.append({
                                "column": condition.get('column', ''),
                                "operator": condition.get('operator', '='),
                                "value": condition.get('value', '')
                            })
                        transformation["having_conditions"] = having_conditions
                    
                    # Aggregate functions
                    if aggregations.get('aggregate_functions'):
                        transformation["aggregate_functions"] = aggregations['aggregate_functions']
                
                # Extract JOIN information (preserve pairing of join_type and conditions)
                joins = transformation_data.get('joins', [])
                if joins:
                    joined_data = []
                    for join in joins:
                        join_entry = {
                            "join_type": join.get('join_type', 'INNER JOIN'),
                            "right_table": join.get('right_table'),
                            "conditions": join.get('conditions', [])
                        }
                        joined_data.append(join_entry)
                    
                    if joined_data:
                        transformation["joins"] = joined_data
                
                # Extract sorting information
                sorting = transformation_data.get('sorting', {})
                if sorting and sorting.get('order_by_columns'):
                    order_by_info = sorting['order_by_columns']
                    if isinstance(order_by_info, list):
                        # Convert to simple string format
                        order_by_strings = []
                        for order_item in order_by_info:
                            if isinstance(order_item, dict):
                                column = order_item.get('column', '')
                                direction = order_item.get('direction', 'ASC')
                                order_by_strings.append(f"{column} {direction}")
                            else:
                                order_by_strings.append(str(order_item))
                        transformation["order_by_columns"] = order_by_strings
                
                # Extract window functions
                window_functions = transformation_data.get('window_functions', [])
                if window_functions:
                    transformation["window_functions"] = window_functions
                
                # Extract limiting information (LIMIT/OFFSET)
                limiting = transformation_data.get('limiting', {})
                if limiting and (limiting.get('limit') is not None or limiting.get('offset') is not None):
                    limit_info = {}
                    if limiting.get('limit') is not None:
                        limit_info['limit'] = limiting['limit']
                    if limiting.get('offset') is not None:
                        limit_info['offset'] = limiting['offset']
                    transformation["limiting"] = limit_info
                
                # Extract CASE statement information
                case_statements = transformation_data.get('case_statements', [])
                if case_statements:
                    transformation["case_statements"] = case_statements
                    
            except Exception as e:
                # If transformation parsing fails, fall back to empty transformations
                pass
        
        return [transformation]
    
    def _filter_nested_conditions(self, filter_conditions: list, entity_name: str) -> list:
        """Filter nested CTE transformation conditions to only include those relevant to the entity."""
        if not filter_conditions:
            return []
        
        relevant_filters = []
        for filter_item in filter_conditions:
            if isinstance(filter_item, dict):
                # Handle nested structure like {'type': 'FILTER', 'conditions': [...]}
                if filter_item.get('type') == 'FILTER' and 'conditions' in filter_item:
                    relevant_conditions = []
                    for condition in filter_item['conditions']:
                        if isinstance(condition, dict) and 'column' in condition:
                            # For CTE context, include the condition if:
                            # 1. Column is explicitly qualified for this table, OR
                            # 2. Column is unqualified (assume it belongs to source table in CTE)
                            column_name = condition['column']
                            if (self._is_column_from_table(column_name, entity_name) or 
                                ('.' not in column_name)):  # Unqualified column in CTE context
                                relevant_conditions.append(condition)
                    
                    if relevant_conditions:
                        relevant_filters.append({
                            'type': 'FILTER',
                            'conditions': relevant_conditions
                        })
                
                # Handle other transformation types (GROUP_BY, etc.)
                elif filter_item.get('type') in ['GROUP_BY', 'HAVING', 'ORDER_BY']:
                    # Include these transformation types as they are generally table-specific
                    relevant_filters.append(filter_item)
                
                # Handle flat structure like {'column': 'users.active', 'operator': '=', 'value': 'TRUE'}
                elif 'column' in filter_item:
                    column_name = filter_item['column']
                    if (self._is_column_from_table(column_name, entity_name) or 
                        ('.' not in column_name)):  # Unqualified column in CTE context
                        relevant_filters.append(filter_item)
        
        return relevant_filters
    
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
    
    def _extract_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """Extract column transformations from CTE queries."""
        column_transformations = []
        
        try:
            # Get CTE lineage data
            cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
            cte_data = cte_lineage.get('ctes', {})
            
            # Look for the target CTE
            if target_table not in cte_data:
                return column_transformations
            
            target_cte = cte_data[target_table]
            columns = target_cte.get('columns', [])
            
            for col in columns:
                # Check if this is a computed column
                is_computed = col.get('is_computed', False)
                raw_expression = col.get('expression', '')
                
                if is_computed or self._is_computed_expression(raw_expression):
                    # Use alias as the column name, and show transformation as source expression
                    target_column_name = col.get('alias') or col.get('name')
                    source_expression = self._extract_source_from_expression(raw_expression, target_column_name)
                    
                    col_transformation = {
                        'column_name': target_column_name,
                        'source_expression': source_expression,
                        'transformation_type': self._get_cte_transformation_type(col, raw_expression),
                        'function_type': self._extract_function_type_generic(raw_expression),
                        'full_expression': raw_expression
                    }
                    column_transformations.append(col_transformation)
                    
        except Exception:
            # If parsing fails, return empty list (fail gracefully)
            pass
            
        return column_transformations
    
    def _get_cte_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type for CTE columns."""
        if col_info.get('is_aggregate', False) or is_aggregate_function(expression):
            return 'AGGREGATE'
        elif 'CASE' in expression.upper():
            return 'CASE'
        elif self._is_computed_expression(expression):
            return 'COMPUTED'
        else:
            return 'DIRECT'
    
    def _is_computed_expression(self, expression: str) -> bool:
        """Check if an expression is computed (not just a simple column reference)."""
        if not expression:
            return False
        
        # Simple heuristics for computed expressions
        expr = expression.strip()
        
        # Check for function calls: FUNCTION(...)
        if '(' in expr and ')' in expr:
            return True
        
        # Check for mathematical operations
        if any(op in expr for op in ['+', '-', '*', '/', '%']):
            return True
        
        # Check for CASE statements
        if 'CASE' in expr.upper():
            return True
        
        return False
    
    def _extract_source_from_expression(self, expression: str, target_name: str) -> str:
        """Extract the source part from transformation expression."""
        if not expression:
            return 'UNKNOWN'
        
        # Use utility function to extract alias and get the source part
        alias = extract_alias_from_expression(expression)
        if alias:
            # If alias found, remove the " AS alias" part
            alias_index = expression.upper().rfind(' AS ')
            if alias_index != -1:
                return expression[:alias_index].strip()
        
        # If no AS clause, return the expression as is
        return expression.strip()
    
    def _extract_function_type_generic(self, expression: str) -> str:
        """Extract function type from expression generically."""
        function_type = extract_function_type(expression)
        
        # Check for CASE expressions
        if expression and expression.upper().strip().startswith('CASE'):
            return 'CASE'
        
        return function_type if function_type != "UNKNOWN" else 'EXPRESSION'
    
    def _populate_cte_source_tables(self, chains: dict, sql: str, result):
        """Populate source table columns for CTE queries."""
        try:
            # Get table lineage to understand which tables are source tables  
            table_lineage_data = result.table_lineage.downstream
            
            # For each source table, add basic column metadata
            for table_name, chain_data in chains.items():
                if isinstance(chain_data, dict) and chain_data.get("entity_type") == "table":
                    metadata = chain_data.get("metadata", {})
                    existing_columns = metadata.get("table_columns", [])
                    
                    # If source table has no columns, add basic ones based on usage
                    if not existing_columns:
                        # Extract columns that are used from this table
                        table_columns = self._extract_source_table_columns(table_name, sql, result)
                        if table_columns:
                            metadata["table_columns"] = table_columns
                            chain_data["metadata"] = metadata
        except Exception:
            pass
            
    def _extract_source_table_columns(self, table_name: str, sql: str, result) -> list:
        """Extract columns that should belong to a source table using CTE tracing."""
        columns = []
        
        try:
            # Strategy 1: Look for direct references in downstream lineage
            column_lineage_downstream = result.column_lineage.downstream 
            for target_col, source_cols in column_lineage_downstream.items():
                for source_col in source_cols:
                    if self._source_column_belongs_to_table(source_col, table_name):
                        col_name = source_col.split(".")[-1] if "." in source_col else source_col
                        if not any(col.get("name") == col_name for col in columns):
                            columns.append({
                                "name": col_name,
                                "upstream": [],
                                "type": "SOURCE"
                            })
            
            # Strategy 2: Trace back through CTE dependencies using upstream lineage
            if not columns:
                columns = self._trace_cte_columns_to_source(table_name, result)
            
            # Strategy 3: Parse SQL directly to find columns used from this table
            if not columns:
                columns = self._parse_sql_for_source_columns(table_name, sql)
                            
        except Exception:
            pass
            
        return columns
        
    def _trace_cte_columns_to_source(self, table_name: str, result) -> list:
        """Trace CTE columns back to source table."""
        columns = []
        
        try:
            # Use UPSTREAM table lineage to find what depends on this table
            table_lineage_upstream = result.table_lineage.upstream
            column_lineage_upstream = result.column_lineage.upstream
            
            # Find CTEs that depend on this table (table is in sources)
            dependent_ctes = []
            for target, sources in table_lineage_upstream.items():
                if table_name in sources:
                    dependent_ctes.append(target)
            
            # For each dependent CTE, find what columns it takes from the source
            for cte_name in dependent_ctes:
                for cte_col, source_cols in column_lineage_upstream.items():
                    if cte_col.startswith(f"{cte_name}."):
                        # This CTE column comes from source columns
                        for source_col in source_cols:
                            # These are the original column names from the source table
                            if not any(col.get("name") == source_col for col in columns):
                                columns.append({
                                    "name": source_col,
                                    "upstream": [],
                                    "type": "SOURCE"
                                })
                                
        except Exception:
            pass
            
        return columns
        
    def _parse_sql_for_source_columns(self, table_name: str, sql: str) -> list:
        """Parse SQL to find columns referenced from a source table."""
        columns = []
        
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Find all column references in the SQL
            for column in parsed.find_all(sqlglot.exp.Column):
                if column.table:
                    col_table = str(column.table)
                    if self._tables_match(col_table, table_name):
                        col_name = str(column.name) if column.name else str(column)
                        if not any(col.get("name") == col_name for col in columns):
                            columns.append({
                                "name": col_name,
                                "upstream": [],
                                "type": "SOURCE"
                            })
                            
        except Exception:
            pass
            
        return columns
        
    def _source_column_belongs_to_table(self, source_col: str, table_name: str) -> bool:
        """Check if a source column belongs to a specific table."""
        try:
            # Handle qualified column names like "users.id", "ecommerce.users.name", "hive.default.users.name"
            if "." in source_col and not source_col.startswith("QUERY_RESULT."):
                # Extract table part from column name
                parts = source_col.split(".")
                if len(parts) >= 2:
                    # Could be "users.id", "ecommerce.users.name", or "hive.default.users.name"
                    if len(parts) == 2:
                        # Simple case: "users.id"
                        col_table = parts[0]
                    elif len(parts) == 3:
                        # "ecommerce.users.name" or "schema.table.column"
                        col_table = f"{parts[0]}.{parts[1]}"
                    elif len(parts) == 4:
                        # "hive.default.users.name"
                        col_table = f"{parts[0]}.{parts[1]}.{parts[2]}"
                    else:
                        col_table = ".".join(parts[:-1])
                    
                    return self._tables_match(col_table, table_name)
        except Exception:
            pass
            
        return False
    
    def _populate_cte_query_results(self, chains: dict, sql: str, column_lineage_data: dict, column_transformations_data: dict):
        """CTE-specific method to populate QUERY_RESULT entities with proper column attribution."""
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Get the main SELECT (not in CTE)
            main_select = self._get_main_select_from_cte(parsed)
            if not main_select:
                return
                
            # Build alias mapping for CTE resolution
            alias_to_table_map, cte_to_source_map = self._build_alias_mappings(main_select, parsed)
            
            
            # Process each source table's QUERY_RESULT
            for source_table, chain_data in chains.items():
                self._populate_single_query_result_for_source(
                    chain_data, main_select, source_table, 
                    column_lineage_data, alias_to_table_map, cte_to_source_map
                )
        except Exception:
            pass  # Fallback to regular method
    
    def _get_main_select_from_cte(self, parsed):
        """Get main SELECT statement (not in CTE)."""
        import sqlglot
        
        if parsed.find(sqlglot.exp.With):
            all_selects = list(parsed.find_all(sqlglot.exp.Select))
            cte_selects = []
            
            for cte in parsed.find_all(sqlglot.exp.CTE):
                cte_selects.extend(list(cte.find_all(sqlglot.exp.Select)))
            
            for select_stmt in all_selects:
                if select_stmt not in cte_selects:
                    return select_stmt
        else:
            return parsed.find(sqlglot.exp.Select)
        return None
    
    def _build_alias_mappings(self, main_select, parsed):
        """Build minimal alias-to-table and CTE-to-source mappings."""
        import sqlglot
        
        alias_to_table_map = {}
        cte_to_source_map = {}
        
        try:
            # Build alias mapping from FROM/JOINs
            from_clause = main_select.find(sqlglot.exp.From)
            if from_clause and isinstance(from_clause.this, sqlglot.exp.Table):
                if from_clause.this.alias:
                    alias_to_table_map[str(from_clause.this.alias)] = str(from_clause.this.this)
            
            for join in main_select.find_all(sqlglot.exp.Join):
                if isinstance(join.this, sqlglot.exp.Table) and join.this.alias:
                    alias_to_table_map[str(join.this.alias)] = str(join.this.this)
            
            # Build CTE-to-source mapping
            cte_names = {str(c.alias) for c in parsed.find_all(sqlglot.exp.CTE)}
            
            for cte in parsed.find_all(sqlglot.exp.CTE):
                cte_name = str(cte.alias)
                source_tables = [str(table) for table in cte.this.find_all(sqlglot.exp.Table)]
                
                if source_tables:
                    # Use first non-CTE table as source, or trace through CTE chain
                    for table in source_tables:
                        clean_table = table.split('.')[-1].split(' ')[0]  # Handle "table AS alias"
                        
                        if clean_table not in cte_names:
                            # Direct base table
                            cte_to_source_map[cte_name] = clean_table
                            break
                        elif clean_table in cte_to_source_map:
                            # Trace through CTE chain
                            final_source = cte_to_source_map[clean_table]
                            cte_to_source_map[cte_name] = final_source  
                            break
                    else:
                        # If no base table found, try to resolve later
                        pass
            
            # Second pass to resolve any remaining CTEs
            max_iterations = 3
            for iteration in range(max_iterations):
                unresolved = []
                for cte in parsed.find_all(sqlglot.exp.CTE):
                    cte_name = str(cte.alias)
                    if cte_name not in cte_to_source_map:
                        source_tables = [str(table) for table in cte.this.find_all(sqlglot.exp.Table)]
                        for table in source_tables:
                            clean_table = table.split('.')[-1].split(' ')[0]
                            if clean_table in cte_to_source_map:
                                final_source = cte_to_source_map[clean_table]
                                cte_to_source_map[cte_name] = final_source
                                break
                        else:
                            unresolved.append(cte_name)
                
                if not unresolved:
                    break
            
        except Exception:
            pass
            
        return alias_to_table_map, cte_to_source_map
    
    def _populate_single_query_result_for_source(self, chain_data, main_select, source_table, 
                                                column_lineage_data, alias_to_table_map, cte_to_source_map):
        """Populate QUERY_RESULT for a single source table."""
        try:
            # Find all QUERY_RESULT entities in chain (including deeply nested ones)
            query_results = self._find_query_result_entity(chain_data)
            if not query_results:
                return
                
            table_columns = []
            
            # Process each SELECT expression
            for expr in main_select.expressions:
                column_name = str(expr)
                
                # Handle SELECT * expansion  
                if column_name == "*":
                    # For SELECT *, expand to all available columns from column lineage
                    # For simple cases like "SELECT * FROM cte", all columns belong to the source
                    from_table = None
                    from_clause = main_select.find(__import__('sqlglot').exp.From)
                    if from_clause:
                        from_table = str(from_clause.this)
                        
                    # If SELECT * FROM single_table (no JOINs), all columns go to mapped source
                    joins = list(main_select.find_all(__import__('sqlglot').exp.Join))
                    if not joins and from_table:
                        # Simple SELECT * FROM table case
                        final_source = cte_to_source_map.get(from_table, from_table)
                        if self._tables_match(final_source, source_table):
                            # All columns belong to this source
                            for col_name in column_lineage_data.keys():
                                table_columns.append({
                                    "name": col_name,
                                    "upstream": list(column_lineage_data[col_name]),
                                    "type": "DIRECT"
                                })
                    else:
                        # Complex SELECT * with JOINs - use original logic
                        for key, values in column_lineage_data.items():
                            if "QUERY_RESULT." in key:
                                bare_col_name = key.replace("QUERY_RESULT.", "")
                                if self._column_belongs_to_source(bare_col_name, source_table, 
                                                               alias_to_table_map, cte_to_source_map, column_lineage_data):
                                    table_columns.append({
                                        "name": bare_col_name,
                                        "upstream": list(values),
                                        "type": "DIRECT"
                                    })
                else:
                    # Check if column belongs to this source table
                    belongs = self._column_belongs_to_source(column_name, source_table, 
                                                           alias_to_table_map, cte_to_source_map, column_lineage_data)
                    
                    if belongs:
                        # Get upstream info
                        bare_name = column_name.split('.')[-1] if '.' in column_name else column_name
                        upstream = set()
                        for key, values in column_lineage_data.items():
                            if bare_name in key:
                                upstream.update(values)
                        
                        table_columns.append({
                            "name": column_name,
                            "upstream": list(upstream),
                            "type": "DIRECT"
                        })
            
            # If no columns were found but source table is involved, add fallback columns
            if not table_columns:
                # Look for any columns in column_lineage_data that reference this source table
                for col_name, sources in column_lineage_data.items():
                    for source in sources:
                        # Try different matching strategies
                        source_table_part = source.split('.')[0] if '.' in source else source
                        # Remove quotes for comparison
                        source_clean = source_table_part.replace('"', '').replace("'", "")
                        target_clean = source_table.replace('"', '').replace("'", "")
                        
                        # Check if this source belongs to our target table
                        if (source_clean == target_clean or 
                            source_clean.endswith('.' + target_clean) or 
                            target_clean.endswith('.' + source_clean) or
                            source.startswith(source_table) or
                            source_table.startswith(source_clean)):
                            table_columns.append({
                                "name": col_name.split('.')[-1],  # Use bare column name
                                "upstream": [source],
                                "type": "INDIRECT"
                            })
                            break  # Only add one column per source for fallback
                    if table_columns:  # Found at least one, stop looking
                        break
            
            # Update all QUERY_RESULT entities that belong to this source
            for query_result in query_results:
                metadata = query_result.get("metadata", {})
                existing_columns = metadata.get("table_columns", [])
                existing_names = {col.get("name") for col in existing_columns}
                
                # Only add new columns that don't already exist
                for new_col in table_columns:
                    if new_col["name"] not in existing_names:
                        existing_columns.append(new_col)
                        existing_names.add(new_col["name"])
                
                metadata["table_columns"] = existing_columns
                query_result["metadata"] = metadata
            
        except Exception:
            pass
    
    def _find_query_result_entity(self, chain_data):
        """Find all QUERY_RESULT entities in chain data (including deeply nested ones)."""
        results = []
        
        def _recursive_find(data):
            if isinstance(data, dict):
                if data.get("entity") == "QUERY_RESULT":
                    results.append(data)
                if "dependencies" in data:
                    for dep in data["dependencies"]:
                        _recursive_find(dep)
        
        _recursive_find(chain_data)
        return results  # Return all found QUERY_RESULT entities
    
    def _column_belongs_to_source(self, column_name, source_table, alias_to_table_map, cte_to_source_map, column_lineage_data):
        """Check if column belongs to source table."""
        try:
            # Handle qualified names like "ts.tier"
            if "." in column_name:
                alias = column_name.split(".")[0]
                bare_column = column_name.split(".")[1]
                
                # Check alias mapping: ts -> tier_summary
                actual_table = alias_to_table_map.get(alias, alias)
                
                # Check CTE mapping: tier_summary -> orders  
                final_source = cte_to_source_map.get(actual_table, actual_table)
                
                if self._tables_match(final_source, source_table):
                    # Double-check by looking at column lineage for the bare column
                    return self._verify_column_lineage_trace(bare_column, actual_table, source_table, column_lineage_data, cte_to_source_map)
                
                return False
            else:
                # Handle unqualified names - check upstream
                for key, values in column_lineage_data.items():
                    if column_name in key:
                        for upstream_col in values:
                            if source_table in upstream_col:
                                return True
                            # Check CTE tracing
                            if "." in upstream_col:
                                upstream_table = upstream_col.split(".")[0] 
                                final_source = cte_to_source_map.get(upstream_table, upstream_table)
                                if self._tables_match(final_source, source_table):
                                    return True
        except Exception:
            pass
        return False
        
    def _verify_column_lineage_trace(self, bare_column, cte_table, source_table, column_lineage_data, cte_to_source_map):
        """Verify that the column lineage supports the CTE-to-source mapping."""
        try:
            # Look for entries like "tier_summary.tier" or just "tier" 
            for key, values in column_lineage_data.items():
                # Check both qualified and unqualified forms
                if (key == bare_column or 
                    key == f"{cte_table}.{bare_column}" or
                    bare_column in key.lower()):
                    
                    # Check if any upstream values trace back to our source
                    for upstream_col in values:
                        if "QUERY_RESULT." in upstream_col:
                            # This confirms it's part of the final result
                            return True
                        if "." in upstream_col:
                            upstream_table = upstream_col.split(".")[0]
                            final_source = cte_to_source_map.get(upstream_table, upstream_table)
                            if self._tables_match(final_source, source_table):
                                return True
            
            return True  # Default to True if we can't disprove it
        except Exception:
            return True
    
    def _tables_match(self, table1, table2):
        """Check if two table names match."""
        if not table1 or not table2:
            return False
        
        # Remove schema/catalog prefixes for comparison
        clean1 = table1.split('.')[-1]
        clean2 = table2.split('.')[-1]
        
        return clean1 == clean2
    
    def _get_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type generically."""
        if col_info.get('is_aggregate') or is_aggregate_function(expression):
            return 'AGGREGATE'
        elif col_info.get('is_window_function'):
            return 'WINDOW_FUNCTION'
        elif self._is_computed_expression(expression):
            return 'COMPUTED'
        else:
            return 'DIRECT'