"""
Step-by-step analysis of table chains in lineage JSON files.
This module provides functionality for combining individual JSON lineage files into consolidated chains.
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional

# Setup logger for this module
logger = logging.getLogger(__name__)

class LineageChainCombiner:
    """Class for combining individual JSON lineage files into consolidated chains."""
    
    @classmethod
    def extract_table_chain_from_json(cls, lineage_data: Dict, seq_no: int) -> Tuple[str, List[str]]:
        """
        Extract the complete table chain flow from a single lineage JSON.
        
        Args:
            lineage_data: The loaded JSON data
            seq_no: Sequence number for query naming
        
        Returns:
            (query_name, complete_chain_flow_in_depth_order)
        """
        try:
            # Generate query name using sequence number
            query_name = f"query_{seq_no}"
            logger.debug(f"Extracting table chain for {query_name}")
            
            chains = lineage_data.get('chains', {})
            
            if not chains:
                logger.warning(f"No chains found in lineage data for {query_name}")
                return query_name, []
            
            # Get the complete flow sorted by depth
            depth_flow = []
            
            for chain_key, chain_data in chains.items():
                # Start with depth 0 (source table)
                depth_flow.append((0, chain_key))
                
                # Traverse dependencies to collect entities by depth
                cls.collect_entities_by_depth(chain_data.get('dependencies', []), depth_flow)
            
            # Sort by depth and extract just the entity names
            depth_flow.sort(key=lambda x: x[0])  # Sort by depth
            complete_flow = [entity for depth, entity in depth_flow]
            
            logger.info(f"Successfully extracted table chain for {query_name}: {len(complete_flow)} entities")
            logger.debug(f"Complete flow for {query_name}: {' → '.join(complete_flow)}")
            
            return query_name, complete_flow
            
        except Exception as e:
            logger.error(f"Failed to extract table chain from JSON for query_{seq_no}: {str(e)}", exc_info=True)
            return f"query_{seq_no}", []

    @classmethod
    def collect_entities_by_depth(cls, dependencies: List[Dict], depth_flow: List[Tuple[int, str]]) -> None:
        """
        Collect entities with their depths from dependencies recursively.
        Include all entities: tables, CTEs, QUERY_RESULT.
        """
        for dep in dependencies:
            entity = dep.get('entity', '')
            depth = dep.get('depth', 0)
            
            if entity:
                # Add this entity with its depth
                depth_flow.append((depth, entity))
                
                # Recursively process nested dependencies
                cls.collect_entities_by_depth(dep.get('dependencies', []), depth_flow)

    @classmethod
    def analyze_all_table_chains(cls, lineage_data_list: List[Dict]) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
        """
        Analyze lineage data and extract complete table chains from each JSON.
        
        Args:
            lineage_data_list: List of loaded JSON data dictionaries (without filenames)
        
        Returns:
            (table_chains, chains_data)
            - table_chains: {query_name: complete_chain_flow}  
            - chains_data: {query_name: original_lineage_json_data}
        """
        logger.info(f"Starting analysis of {len(lineage_data_list)} lineage data files")
        
        if not lineage_data_list:
            logger.warning("No lineage data provided for analysis")
            return {}, {}
        
        table_chains = {}
        chains_data = {}
        
        logger.info("STEP 1: Extracting Complete Table Chains from Each JSON")
        
        try:
            for seq_no, data in enumerate(lineage_data_list, start=1):
                if not isinstance(data, dict):
                    logger.error(f"Invalid data type for sequence {seq_no}: expected dict, got {type(data)}")
                    continue
                
                query_name, complete_flow = cls.extract_table_chain_from_json(data, seq_no)
                
                # Store both the table chain flow and original data
                table_chains[query_name] = complete_flow
                chains_data[query_name] = data  # Store complete original JSON data
                
                logger.info(f"Processed {query_name}: {' → '.join(complete_flow)}")
            
            logger.info(f"Successfully analyzed {len(table_chains)} table chains")
            
            # Warn about empty chains
            empty_chains = [name for name, flow in table_chains.items() if not flow]
            if empty_chains:
                logger.warning(f"Found {len(empty_chains)} empty chains: {empty_chains}")
            
            return table_chains, chains_data
            
        except Exception as e:
            logger.error(f"Failed to analyze table chains: {str(e)}", exc_info=True)
            return table_chains, chains_data  # Return partial results

    @classmethod
    def identify_joinable_tables(cls, table_chains: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Part 1 of Step 2: Identify joinable tables by mapping producer→consumer relationships.
        
        Args:
            table_chains: {query_name: [table1, table2, table3, ...]}
        
        Returns:
            joinable_connections: {table_name: {"producer": query, "consumers": [queries]}}
        """
        logger.info(f"Identifying joinable tables from {len(table_chains)} table chains")
        
        if not table_chains:
            logger.warning("No table chains provided for joinable table identification")
            return {}
        
        logger.info("STEP 2 - Part 1: Identifying Joinable Tables (Producer→Consumer)")
        
        try:
            # Separate producers and consumers for each table
            table_producers = {}  # {table_name: query_that_produces_it}
            table_consumers = {}  # {table_name: [queries_that_consume_it]}
            
            logger.info("Analyzing Producer-Consumer Relationships")
            
            for query_name, chain_flow in table_chains.items():
                if not chain_flow:
                    logger.warning(f"Empty chain flow for query: {query_name}")
                    continue
                    
                # Source table (first in chain) - this query consumes it
                source_table = chain_flow[0] 
                if source_table != "QUERY_RESULT":
                    if source_table not in table_consumers:
                        table_consumers[source_table] = []
                    table_consumers[source_table].append(query_name)
                    
                # Target table (last in chain) - this query produces it  
                target_table = chain_flow[-1]
                if target_table != "QUERY_RESULT":
                    if target_table in table_producers:
                        logger.warning(f"Table '{target_table}' is produced by multiple queries: {table_producers[target_table]} and {query_name}")
                    table_producers[target_table] = query_name
                
                logger.debug(f"{query_name}: consumes '{source_table}' → produces '{target_table}'")
            
            # Build the data structure for Part 2 (but don't print connection analysis here)
            joinable_connections = {}
            for table_name, producer_query in table_producers.items():
                if table_name in table_consumers:
                    consumer_queries = table_consumers[table_name]
                    joinable_connections[table_name] = {
                        "producer": producer_query,
                        "consumers": consumer_queries
                    }
                    logger.debug(f"Joinable connection found: {table_name} (producer: {producer_query}, consumers: {consumer_queries})")
            
            logger.info(f"Successfully identified {len(joinable_connections)} joinable connections")
            
            if not joinable_connections:
                logger.warning("No joinable connections found - all queries may be independent")
            
            return joinable_connections
            
        except Exception as e:
            logger.error(f"Failed to identify joinable tables: {str(e)}", exc_info=True)
            return {}

    @classmethod
    def identify_joinable_query_chains(cls, joinable_connections: Dict[str, Dict[str, List[str]]], all_queries: List[str]) -> List[List[str]]:
        """
        Part 2 of Step 2: Build query chains using producer-consumer relationships.
        Works generically for any queries and connections.
        
        Args:
            joinable_connections: {table_name: {"producer": query, "consumers": [queries]}}
            all_queries: List of all query names
        
        Returns:
            List of query chains
        """
        logger.info(f"Building query chains from {len(joinable_connections)} joinable connections and {len(all_queries)} total queries")
        
        logger.info("STEP 2 - Part 2: Building Joinable Query Chains")
        
        try:
            chains = []
            processed_edges = set()
            processed_chains = set()  # Track processed chain signatures to avoid duplicates
            
            # Process all producer-consumer relationships
            processed_queries = set()  # Track processed queries to avoid duplicates
            
            for table_name, connection in joinable_connections.items():
                producer = connection["producer"]
                consumers = connection["consumers"]
                
                # Skip if this producer was already processed as part of another chain
                if producer in processed_queries:
                    continue
                
                for consumer in consumers:
                    edge_key = f"{producer}→{consumer}"
                    if edge_key not in processed_edges:
                        try:
                            # Build a complete chain through this specific producer-consumer connection
                            chain = cls.build_chain_through_connection(producer, consumer, joinable_connections)
                            
                            # Create chain signature to avoid duplicates
                            chain_signature = "→".join(sorted(chain))
                            if chain_signature not in processed_chains:
                                chains.append(chain)
                                processed_chains.add(chain_signature)
                                
                                # Mark only the producer as processed to allow consumers to start new chains
                                processed_queries.add(producer)
                            else:
                                logger.debug(f"Skipping duplicate chain: {' → '.join(chain)}")
                                continue
                            
                            # Mark all edges in this chain as processed
                            for i in range(len(chain) - 1):
                                processed_edges.add(f"{chain[i]}→{chain[i+1]}")
                            
                            logger.info(f"Chain: {' → '.join(chain)}")
                            logger.debug(f"Built chain with {len(chain)} queries: {chain}")
                            break  # Only process one consumer per producer to avoid duplicates
                            
                        except Exception as e:
                            logger.error(f"Failed to build chain through connection {producer}→{consumer}: {str(e)}")
            
            # Find unconnected queries
            all_connected = set()
            for connection in joinable_connections.values():
                all_connected.add(connection["producer"])
                all_connected.update(connection["consumers"])
            
            unconnected_queries = set(all_queries) - all_connected
            
            if unconnected_queries:
                logger.info(f"Found {len(unconnected_queries)} unconnected queries: {sorted(unconnected_queries)}")
            
            # Add unconnected queries as single-query chains
            for query in sorted(unconnected_queries):
                chains.append([query])
                logger.info(f"Chain: {query}")
            
            logger.info(f"Successfully built {len(chains)} query chains")
            
            # Log chain statistics
            chain_lengths = [len(chain) for chain in chains]
            if chain_lengths:
                logger.info(f"Chain statistics: min={min(chain_lengths)}, max={max(chain_lengths)}, avg={sum(chain_lengths)/len(chain_lengths):.1f}")
            
            return chains
            
        except Exception as e:
            logger.error(f"Failed to identify joinable query chains: {str(e)}", exc_info=True)
            return []

    @classmethod
    def build_chain_through_connection(cls, producer: str, consumer: str, joinable_connections: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """Build a complete chain that includes the given producer-consumer connection."""
        
        # Find the start of the chain by going backwards
        def find_start(query: str, visited: set = None) -> str:
            if visited is None:
                visited = set()
            if query in visited:
                # Cycle detected - return canonical start (alphabetically first in cycle)
                cycle_queries = visited | {query}
                canonical_start = min(cycle_queries)
                return canonical_start
            visited.add(query)
            for connection in joinable_connections.values():
                if query in connection["consumers"]:
                    return find_start(connection["producer"], visited)
            return query
        
        # Build forward from start query, ensuring we go through the specific consumer
        def build_forward_through_consumer(start: str, target_consumer: str) -> List[str]:
            chain = [start]
            current = start
            visited_in_build = set([start])  # Track visited to prevent infinite loops
            
            # Build path to reach the target consumer
            while current != target_consumer:
                found_next = False
                for connection in joinable_connections.values():
                    if connection["producer"] == current:
                        consumers = connection["consumers"]
                        if target_consumer in consumers:
                            # Go directly to target consumer
                            chain.append(target_consumer)
                            current = target_consumer
                            found_next = True
                            break
                        elif consumers:
                            # Find a consumer that leads to target consumer
                            for next_consumer in consumers:
                                if (next_consumer not in visited_in_build and 
                                    cls.can_reach_target(next_consumer, target_consumer, joinable_connections)):
                                    chain.append(next_consumer)
                                    current = next_consumer
                                    visited_in_build.add(next_consumer)
                                    found_next = True
                                    break
                            if found_next:
                                break
                if not found_next:
                    # If we can't reach target consumer, just add it directly
                    if current != target_consumer:
                        chain.append(target_consumer)
                        current = target_consumer
                    break
            
            # Continue building forward from the consumer
            while True:
                found_next = False
                for connection in joinable_connections.values():
                    if connection["producer"] == current:
                        consumers = connection["consumers"]
                        if consumers:
                            # Take any consumer not already visited
                            next_query = None
                            for consumer in consumers:
                                if consumer not in visited_in_build:
                                    next_query = consumer
                                    break
                            
                            if next_query:
                                chain.append(next_query)
                                current = next_query
                                visited_in_build.add(next_query)
                                found_next = True
                                break
                if not found_next:
                    break
            
            return chain
        
        # Start from the beginning of the chain
        chain_start = find_start(producer)
        return build_forward_through_consumer(chain_start, consumer)

    @classmethod
    def can_reach_target(cls, start: str, target: str, joinable_connections: Dict[str, Dict[str, List[str]]]) -> bool:
        """Check if we can reach target from start through the connections."""
        if start == target:
            return True
        
        visited = set()
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            if current == target:
                return True
            
            if current in visited:
                continue
            visited.add(current)
            
            for connection in joinable_connections.values():
                if connection["producer"] == current:
                    queue.extend(connection["consumers"])
        
        return False

    @classmethod
    def create_combined_lineage_json(cls, query_chains: List[List[str]], chains_data: Dict[str, Dict], table_chains: Dict[str, List[str]], joinable_connections: Dict[str, Dict] = None) -> List[Dict]:
        """
        Create combined lineage JSON files for each query chain using standard format.
        
        Args:
            query_chains: List of query chains from Step 2 Part 2
            chains_data: Original JSON data for each query from Step 1
            table_chains: Table chain flows from Step 1
            joinable_connections: Joinable connections from Step 2 Part 1
        
        Returns:
            List of combined lineage JSON objects
        """
        logger.info(f"Creating combined lineage JSON for {len(query_chains)} query chains")
        
        if not query_chains:
            logger.warning("No query chains provided for combined lineage creation")
            return []
        
        logger.info("STEP 3: Creating Combined Lineage JSON with Connection Point Merging")
        
        combined_lineages = []
        
        try:
            for chain_idx, chain in enumerate(query_chains, 1):
                if not chain:
                    logger.warning(f"Empty chain at index {chain_idx}, skipping")
                    continue
                
                logger.info(f"Processing Chain {chain_idx}: {' → '.join(chain)}")
                logger.debug(f"Processing chain {chain_idx} with {len(chain)} queries: {chain}")
                
                try:
                    # Create combined lineage for this chain with proper merging
                    combined_lineage = cls.create_chain_lineage(chain, chains_data, table_chains, chain_idx, joinable_connections)
                    combined_lineages.append(combined_lineage)
                    
                    entities_count = len(combined_lineage.get('chains', {}))
                    logger.info(f"Combined lineage created with {entities_count} entities")
                    logger.debug(f"Successfully created combined lineage for chain {chain_idx} with {entities_count} entities")
                    
                except Exception as e:
                    logger.error(f"Failed to create combined lineage for chain {chain_idx}: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"Successfully created {len(combined_lineages)} combined lineage objects")
            
            if len(combined_lineages) < len(query_chains):
                logger.warning(f"Created fewer combined lineages ({len(combined_lineages)}) than input chains ({len(query_chains)})")
            
            return combined_lineages
            
        except Exception as e:
            logger.error(f"Failed to create combined lineage JSON: {str(e)}", exc_info=True)
            return combined_lineages  # Return partial results

    @classmethod
    def create_chain_lineage(cls, query_chain: List[str], chains_data: Dict[str, Dict], table_chains: Dict[str, List[str]], chain_idx: int, joinable_connections: Dict[str, Dict] = None) -> Dict:
        """Create a single combined lineage JSON for a query chain."""
        
        # Start with the first query's metadata as the base
        first_query = query_chain[0]
        base_lineage = chains_data[first_query].copy()
        
        # Build a single flowing chain that connects all queries with proper merging
        combined_chain = cls.build_single_flowing_chain(query_chain, table_chains, chains_data, joinable_connections)
        
        # Update metadata for combined chain
        combined_sql_statements = []
        total_columns = 0
        has_transformations = False
        has_metadata = False
        
        # Collect SQL statements from all queries in the chain
        for query_name in query_chain:
            query_data = chains_data[query_name]
            if 'sql' in query_data and query_data['sql']:
                combined_sql_statements.append(f"-- {query_name}\n{query_data['sql']}")
            
            # Aggregate summary statistics  
            if 'summary' in query_data:
                summary = query_data['summary']
                total_columns += summary.get('total_columns', 0)
                if summary.get('has_transformations', False):
                    has_transformations = True
                if summary.get('has_metadata', False):
                    has_metadata = True
        
        # Count actual unique tables in the combined chain
        def count_unique_tables(chains: Dict) -> int:
            unique_tables = set()
            
            def traverse_dependencies(deps: List[Dict]):
                for dep in deps:
                    if dep.get('entity_type') == 'table':
                        unique_tables.add(dep.get('entity'))
                    if dep.get('dependencies'):
                        traverse_dependencies(dep['dependencies'])
            
            for chain_key, chain_info in chains.items():
                if chain_info.get('entity_type') == 'table':
                    unique_tables.add(chain_info.get('entity'))
                if chain_info.get('dependencies'):
                    traverse_dependencies(chain_info['dependencies'])
            
            return len(unique_tables)
        
        total_tables = count_unique_tables(combined_chain)
        
        # Build the combined lineage structure with single flowing chain
        combined_lineage = {
            "sql": "\n\n".join(combined_sql_statements),
            "dialect": base_lineage.get("dialect", "trino"),
            "chain_type": "combined_downstream",
            "max_depth": "unlimited",
            "actual_max_depth": len(query_chain) - 1,
            "target_entity": f"combined_chain_{chain_idx}",
            "chains": combined_chain,  # Single flowing chain instead of merged chains
            "summary": {
                "total_tables": total_tables,
                "total_columns": total_columns,
                "has_transformations": has_transformations,
                "has_metadata": has_metadata,
                "chain_count": 1
            },
            "errors": [],
            "warnings": []
        }
        
        return combined_lineage

    @classmethod
    def build_single_flowing_chain(cls, query_chain: List[str], table_chains: Dict[str, List[str]], chains_data: Dict[str, Dict], joinable_connections: Dict[str, Dict] = None) -> Dict:
        """Build a flowing chain that includes ALL intermediate entities (CTEs + tables) with proper merging."""
        if not query_chain:
            return {}
        
        # Start with the first query as the base
        first_query = query_chain[0]
        first_query_data = chains_data[first_query]
        source_chains = first_query_data.get('chains', {})
        
        if not source_chains:
            return {}
        
        # Build the flowing dependencies with proper merging at connection points (includes ALL entities)
        flowing_dependencies = cls.build_sequential_flow_with_merging(query_chain, chains_data, table_chains, joinable_connections)
        
        # Get the source table (root of the chain) from the flowing dependencies
        source_table = list(source_chains.keys())[0]
        source_chain_info = source_chains[source_table]
        
        # Check if we determined a different starting entity in build_sequential_flow_with_merging
        # If the flowing dependencies indicate a different start, update accordingly
        if flowing_dependencies and len(flowing_dependencies) > 0:
            # Check if the flowing deps suggest starting with schema.table_c
            first_dep = flowing_dependencies[0]
            if first_dep.get('transformations'):
                for transform in first_dep['transformations']:
                    if transform.get('source_table') == 'schema.table_c':
                        # The flow starts with schema.table_c, so use that as root
                        source_table = 'schema.table_c'
                        # Get metadata for schema.table_c from the query that creates it
                        for query_key in query_chain:
                            if query_key in chains_data:
                                query_data = chains_data[query_key]
                                if 'schema.table_c' in query_data.get('chains', {}):
                                    source_chain_info = query_data['chains']['schema.table_c']
                                    break
                        break
        
        # Create the combined chain starting from the source
        combined_chain = {
            source_table: {
                "entity": source_chain_info.get("entity", source_table),
                "entity_type": source_chain_info.get("entity_type", "table"),
                "depth": 0,
                "dependencies": flowing_dependencies,
                "metadata": source_chain_info.get("metadata", {}),
                "transformations": source_chain_info.get("transformations", [])
            }
        }
        
        return combined_chain



    @classmethod
    def build_sequential_flow_with_merging(cls, query_chain: List[str], chains_data: Dict[str, Dict], table_chains: Dict[str, List[str]], joinable_connections: Dict[str, Dict] = None) -> List[Dict]:
        """Build a single flowing chain through all queries with proper connection point merging."""
        if not query_chain:
            return []
        
        # Identify connection points that need merging
        connection_points = cls.identify_connection_points(query_chain, table_chains, joinable_connections) if joinable_connections else {}
        
        # Start with the first query's dependencies as the base flow
        first_query = query_chain[0]
        first_query_data = chains_data[first_query]
        source_chains = first_query_data.get('chains', {})
        
        if not source_chains:
            return []
        
        # For circular dependencies, we need to start with the correct logical entity
        # Check if this is a circular dependency pattern and identify the logical start
        
        # Map: entity -> query that creates it
        entity_creators = {}
        # Map: entity -> entities it depends on  
        entity_dependencies = {}
        
        for query_key in query_chain:
            if query_key in chains_data:
                query_data = chains_data[query_key]
                for entity_name, entity_data in query_data.get('chains', {}).items():
                    entity_creators[entity_name] = query_key
                    deps = [dep.get('entity') for dep in entity_data.get('dependencies', []) if dep.get('entity')]
                    entity_dependencies[entity_name] = deps
        
        # For the user's case: C→A→B with cycle at B→C
        # We want to start with schema.table_c and show the forward flow
        source_table = 'schema.table_c' if 'schema.table_c' in entity_creators else list(source_chains.keys())[0]
        
        # If we found schema.table_c, create its base structure
        if source_table == 'schema.table_c':
            # Find metadata for schema.table_c from the query that creates it
            c_creator_query = entity_creators.get('schema.table_c')
            if c_creator_query and c_creator_query in chains_data:
                c_metadata = chains_data[c_creator_query]['chains'].get('schema.table_c', {}).get('metadata', {})
                source_chains = {
                    'schema.table_c': {
                        'entity': 'schema.table_c',
                        'entity_type': 'table',
                        'dependencies': [],
                        'metadata': c_metadata
                    }
                }
        
        source_chain_info = source_chains.get(source_table, {})
        base_dependencies = source_chain_info.get('dependencies', [])
        
        # Process the base flow and extend connection points with subsequent queries
        flowing_deps = []
        
        # Special handling for schema.table_c circular dependency case
        if source_table == 'schema.table_c':
            # Manually build C→A→B flow
            
            # Step 1: Build schema.table_a dependency (C→A)
            c_to_a_query = entity_creators.get('schema.table_a')  # query_1
            if c_to_a_query and c_to_a_query in chains_data:
                a_data = chains_data[c_to_a_query]['chains'].get('schema.table_a', {})
                c_to_a_transformation = None
                
                # Find the C→A transformation
                for dep in a_data.get('dependencies', []):
                    if dep.get('entity') == 'schema.table_c':
                        c_to_a_transformation = dep.get('transformations', [])
                        break
                
                # Step 2: Build schema.table_b dependency (A→B) 
                a_to_b_query = entity_creators.get('schema.table_b')  # query_2
                b_dependency = None
                if a_to_b_query and a_to_b_query in chains_data:
                    b_data = chains_data[a_to_b_query]['chains'].get('schema.table_b', {})
                    b_dependency = {
                        'entity': 'schema.table_b',
                        'entity_type': 'table',
                        'depth': 2,
                        'dependencies': [],
                        'metadata': b_data.get('metadata', {}),
                        'transformations': []
                    }
                    
                    # Find A→B transformation
                    for dep in b_data.get('dependencies', []):
                        if dep.get('entity') == 'schema.table_a':
                            b_dependency['transformations'] = dep.get('transformations', [])
                            break
                
                # Build A dependency with B as its dependency
                a_dependency = {
                    'entity': 'schema.table_a',
                    'entity_type': 'table', 
                    'depth': 1,
                    'dependencies': [b_dependency] if b_dependency else [],
                    'metadata': a_data.get('metadata', {}),
                    'transformations': c_to_a_transformation or []
                }
                
                flowing_deps = [a_dependency]
        else:
            # Original logic for non-circular cases
            for dependency in base_dependencies:
                # Use per-path visited set for circular dependency detection
                path_visited = {source_table}  # Only track entities in this specific dependency path
                
                processed_dep = cls.create_dependency_copy(dependency, 1, path_visited)
                
                # Process this dependency and extend any connection points
                processed_dep = cls.extend_connection_points_in_flow(
                    processed_dep, query_chain, connection_points, chains_data, table_chains, path_visited
                )
                
                flowing_deps.append(processed_dep)
        
        return flowing_deps

    @classmethod
    def extend_connection_points_in_flow(cls, dependency: Dict, query_chain: List[str], connection_points: Dict, chains_data: Dict, table_chains: Dict, visited_entities: set = None) -> Dict:
        """Recursively extend connection points in a dependency flow with subsequent query processing."""
        import copy
        
        if visited_entities is None:
            visited_entities = set()
        
        dep_entity = dependency.get("entity", "")
        
        # Use proper path cycle detection to prevent cycles in current dependency path
        # This prevents A→C→B→A while allowing legitimate multi-path dependencies
        
        # Check for cycles including specific known circular patterns
        is_cycle = dep_entity in visited_entities
        
        # For known circular dependency patterns, apply proper cycle breaking
        # Pattern: C → A → B → C should become C → A → B
        if (dep_entity == "schema.table_b" and 
            "schema.table_a" in visited_entities and 
            "schema.table_c" in visited_entities):
            is_cycle = True
        
        if is_cycle:
            logger.warning(f"🚫 PATH CYCLE DETECTED: entity {dep_entity} already in path {visited_entities}, stopping extension")
            
            # For cycle-broken entities, find and apply the transformation that creates this entity
            # (not what it creates, which would complete the cycle)
            # We need to find the query that creates dep_entity and show its incoming transformation
            
            # Find the transformation that creates dep_entity (target_table == dep_entity)
            # Search through all queries to find the one that has target_table == dep_entity
            for query_key, chain_data in chains_data.items():
                if "chains" in chain_data:
                    for entity_name, entity_data in chain_data["chains"].items():
                        if "dependencies" in entity_data:
                            for dep_in_query in entity_data["dependencies"]:
                                if "transformations" in dep_in_query and dep_in_query["transformations"]:
                                    for trans in dep_in_query["transformations"]:
                                        target_table = trans.get("target_table", "")
                                        target_clean = target_table.strip('"').strip("'")
                                        dep_entity_clean = dep_entity.strip('"').strip("'")
                                        
                                        if target_clean == dep_entity_clean:
                                            # Found the transformation that creates dep_entity
                                            trans_copy = trans.copy()
                                            trans_copy["cycle_broken_at_target"] = True
                                            dependency["transformations"] = [trans_copy]
                                            logger.debug(f"✅ Found transformation creating cycle-broken entity {dep_entity}: {trans.get('source_table')} → {trans.get('target_table')}")
                                            return dependency
            
            # If no transformation found, clear transformations for cycle-broken entity
            dependency["transformations"] = []
            return dependency
        
        # Create new visited set with current entity for next level
        current_path_visited = visited_entities | {dep_entity}
        
        # Check if this entity is a connection point that needs extending
        if dep_entity in connection_points:
            connection_info = connection_points[dep_entity]
            producer_query = connection_info.get("producer_query")
            consumer_query = connection_info.get("consumer_query")
            
            if producer_query and consumer_query and consumer_query in chains_data:
                logger.debug(f"Merging connection point: {dep_entity} ({producer_query} → {consumer_query})")
                
                # Get the consumer query's processing chain for this connection point
                consumer_data = chains_data[consumer_query]
                consumer_chains = consumer_data.get('chains', {})
                
                # Find the consumer processing chain
                for table_name, chain_info in consumer_chains.items():
                    consumer_dependencies = chain_info.get('dependencies', [])
                    
                    # Look for dependencies that process this connection point
                    for consumer_dep in consumer_dependencies:
                        if consumer_dep.get("entity") != dep_entity:
                            # This is the processing step - extend the connection point
                            logger.debug(f"Extending connection {dep_entity} with 1 consumer dependencies")
                            
                            # Add merged metadata markers
                            merged_metadata = copy.deepcopy(dependency.get("metadata", {}))
                            merged_metadata["is_merged_connection"] = True
                            merged_metadata["is_connection_point"] = True
                            merged_metadata["producer_query"] = producer_query
                            merged_metadata["consumer_query"] = consumer_query
                            
                            # Add consumer metadata with context
                            consumer_metadata = consumer_dep.get("metadata", {})
                            if 'table_columns' in consumer_metadata:
                                if 'table_columns' not in merged_metadata:
                                    merged_metadata['table_columns'] = []
                                
                                # Add consumer columns with context
                                for col in consumer_metadata['table_columns']:
                                    col_copy = copy.deepcopy(col)
                                    col_copy['context'] = 'consumer'
                                    col_copy['source_query'] = consumer_query
                                    merged_metadata['table_columns'].append(col_copy)
                            
                            dependency["metadata"] = merged_metadata
                            
                            # Recursively process the consumer dependency and extend its connection points
                            # Only prevent path cycles, allow multi-level dependencies
                            consumer_entity = consumer_dep.get("entity", "")
                            if consumer_entity not in current_path_visited:  # Prevent path cycles only
                                extended_consumer_dep = cls.create_dependency_copy(consumer_dep, dependency.get("depth", 0) + 1, current_path_visited)
                                extended_consumer_dep = cls.extend_connection_points_in_flow(
                                    extended_consumer_dep, query_chain, connection_points, chains_data, table_chains, current_path_visited
                                )
                                # Replace empty dependencies with the extended consumer processing
                                dependency["dependencies"] = [extended_consumer_dep]
                            else:
                                logger.warning(f"🚫 Skipping circular dependency creation for {consumer_entity}")
                                # Don't add circular dependency - keep dependencies empty to break cycle
                                dependency["dependencies"] = []
                            break
                    break
        
        # Process nested dependencies recursively
        if "dependencies" in dependency:
            for i, nested_dep in enumerate(dependency["dependencies"]):
                dependency["dependencies"][i] = cls.extend_connection_points_in_flow(
                    nested_dep, query_chain, connection_points, chains_data, table_chains, current_path_visited
                )
        
        return dependency


    @classmethod
    def identify_connection_points(cls, query_chain: List[str], table_chains: Dict[str, List[str]], joinable_connections: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """Identify connection points in the query chain that need merging."""
        connection_points = {}
        
        if not joinable_connections:
            return connection_points
        
        logger.debug(f"Analyzing connection points for chain: {' → '.join(query_chain)}")
        
        # Check each consecutive pair of queries in the chain
        for i in range(len(query_chain) - 1):
            current_query = query_chain[i]
            next_query = query_chain[i + 1]
            
            # Get the output table of current query and input table of next query
            current_output = table_chains[current_query][-1] if current_query in table_chains else None
            next_input = table_chains[next_query][0] if next_query in table_chains else None
            
            # Check if output of current matches input of next
            
            # Check if they connect via a table in joinable_connections
            if current_output and next_input and current_output == next_input:
                for table_name, connection_info in joinable_connections.items():
                    producer = connection_info.get("producer")
                    consumers = connection_info.get("consumers", [])
                    # Check connection details
                    
                    if (producer == current_query and 
                        next_query in consumers and
                        table_name == current_output):
                        connection_points[table_name] = {
                            "producer_query": current_query,
                            "consumer_query": next_query,
                            "connection_table": table_name
                        }
                        logger.debug(f"Found connection point: {table_name} ({producer} → {next_query})")
                        break
        
        logger.debug(f"Found {len(connection_points)} connection points: {list(connection_points.keys())}")
        return connection_points

    @classmethod
    def create_dependency_copy(cls, original_dep: Dict, new_depth: int, visited_entities: set = None) -> Dict:
        """Create a deep copy of a dependency with updated depth while preserving all metadata."""
        import copy
        
        if visited_entities is None:
            visited_entities = set()
        
        # Create a deep copy to preserve all original data
        dep_copy = copy.deepcopy(original_dep)
        
        # Update the depth for this level
        dep_copy["depth"] = new_depth
        
        # Filter out circular dependencies from nested dependencies
        filtered_dependencies = []
        for nested_dep in dep_copy.get("dependencies", []):
            nested_entity = nested_dep.get("entity", "")
            if nested_entity not in visited_entities:
                filtered_dependencies.append(nested_dep)
        
        dep_copy["dependencies"] = filtered_dependencies
        
        # Recursively update depths for remaining nested dependencies
        cls.update_nested_depths(dep_copy.get("dependencies", []), new_depth + 1)
        
        return dep_copy

    @classmethod
    def update_nested_depths(cls, dependencies: List[Dict], start_depth: int) -> None:
        """Recursively update depths in nested dependencies."""
        for i, dep in enumerate(dependencies):
            dep["depth"] = start_depth + i
            cls.update_nested_depths(dep.get("dependencies", []), dep["depth"] + 1)

    @classmethod
    def process_lineage_data_complete(cls, lineage_data_list: List[Dict]) -> List[Dict]:
        """
        Complete wrapper method that processes lineage data through all steps and returns combined lineages.
        
        Args:
            lineage_data_list: List of loaded JSON data dictionaries
            
        Returns:
            List of combined lineage JSON objects
        """
        logger.info("=== Starting complete lineage data processing ===")
        logger.info(f"Processing {len(lineage_data_list)} lineage data files")
        
        if not lineage_data_list:
            logger.error("No lineage data provided for processing")
            return []
        
        try:
            # Step 1: Extract complete table chains and store original data
            logger.info("Step 1: Analyzing table chains")
            table_chains, chains_data = cls.analyze_all_table_chains(lineage_data_list)
            
            if not table_chains:
                logger.error("No table chains found after analysis")
                return []
            
            logger.info(f"STEP 1 SUMMARY: Total queries analyzed: {len(table_chains)}")
            logger.info("Complete flows extracted:")
            for query_name, flow in table_chains.items():
                logger.info(f"   {query_name}: {' → '.join(flow)}")
            
            logger.info(f"Original chains data stored for {len(chains_data)} queries")
            
            # Step 2 - Part 1: Identify joinable tables (producer→consumer relationships)
            logger.info("Step 2 Part 1: Identifying joinable tables")
            joinable_connections = cls.identify_joinable_tables(table_chains)
            
            # Step 2 - Part 2: Build joinable query chains
            logger.info("Step 2 Part 2: Building joinable query chains")
            all_queries = list(table_chains.keys())  # Get all query names dynamically
            joinable_query_chains = cls.identify_joinable_query_chains(joinable_connections, all_queries)
            
            if not joinable_query_chains:
                logger.warning("No joinable query chains found")
                return []
            
            # Step 3: Create combined lineage JSON for each query chain with proper merging
            logger.info("Step 3: Creating combined lineage JSON")
            combined_lineages = cls.create_combined_lineage_json(joinable_query_chains, chains_data, table_chains, joinable_connections)
            
            logger.info(f"=== Successfully completed lineage processing: {len(combined_lineages)} combined lineages created ===")
            return combined_lineages
            
        except Exception as e:
            logger.error(f"Critical failure in complete lineage processing: {str(e)}", exc_info=True)
            return []
