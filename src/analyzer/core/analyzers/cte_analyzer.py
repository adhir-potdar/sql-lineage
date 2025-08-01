"""CTE (Common Table Expression) analyzer."""

from typing import Dict, Any, List, Optional
import sqlglot
from .base_analyzer import BaseAnalyzer


class CTEAnalyzer(BaseAnalyzer):
    """Analyzer for CTE statements."""
    
    def __init__(self, dialect: str = "trino", main_analyzer=None):
        """Initialize CTE analyzer with optional reference to main analyzer."""
        super().__init__(dialect)
        self.main_analyzer = main_analyzer
    
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
        if col_info.get('is_aggregate', False):
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
        
        # Remove the alias part if present (e.g., "SUM(amount) AS total" -> "SUM(amount)")
        expr = expression.strip()
        
        # Split by " AS " (case insensitive) and take the first part
        import re
        as_split = re.split(r'\s+AS\s+', expr, flags=re.IGNORECASE)
        if len(as_split) > 1:
            return as_split[0].strip()
        
        # If no AS clause, return the expression as is
        return expr
    
    def _extract_function_type_generic(self, expression: str) -> str:
        """Extract function type from expression generically."""
        if not expression:
            return 'UNKNOWN'
        
        # Convert to uppercase for matching
        expr_upper = expression.upper().strip()
        
        # Extract function name from expressions like "COUNT(*)", "SUM(amount)", etc.
        # Match pattern: FUNCTION_NAME(...)
        import re
        function_match = re.match(r'^([A-Z_]+)\s*\(', expr_upper)
        if function_match:
            return function_match.group(1)
        
        # Check for CASE expressions
        if expr_upper.startswith('CASE'):
            return 'CASE'
        
        return 'EXPRESSION'
    
    def _get_transformation_type(self, col_info: Dict, expression: str) -> str:
        """Determine the transformation type generically."""
        if col_info.get('is_aggregate'):
            return 'AGGREGATE'
        elif col_info.get('is_window_function'):
            return 'WINDOW_FUNCTION'
        elif self._is_computed_expression(expression):
            return 'COMPUTED'
        else:
            return 'DIRECT'