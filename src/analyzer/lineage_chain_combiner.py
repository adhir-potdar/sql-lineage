"""
Step-by-step analysis of table chains in lineage JSON files.
This module provides functionality for combining individual JSON lineage files into consolidated chains.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

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
        # Generate query name using sequence number
        query_name = f"query_{seq_no}"
        
        chains = lineage_data.get('chains', {})
        
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
        
        return query_name, complete_flow

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
        table_chains = {}
        chains_data = {}
        
        print("🔍 STEP 1: Extracting Complete Table Chains from Each JSON")
        print("=" * 70)
        
        for seq_no, data in enumerate(lineage_data_list, start=1):
            query_name, complete_flow = cls.extract_table_chain_from_json(data, seq_no)
            
            # Store both the table chain flow and original data
            table_chains[query_name] = complete_flow
            chains_data[query_name] = data  # Store complete original JSON data
            
            print(f"📄 {query_name}")
            print(f"   Complete Flow: {' → '.join(complete_flow)}")
            print()
        
        return table_chains, chains_data

    @classmethod
    def identify_joinable_tables(cls, table_chains: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Part 1 of Step 2: Identify joinable tables by mapping producer→consumer relationships.
        
        Args:
            table_chains: {query_name: [table1, table2, table3, ...]}
        
        Returns:
            joinable_connections: {table_name: {"producer": query, "consumers": [queries]}}
        """
        print("🔗 STEP 2 - Part 1: Identifying Joinable Tables (Producer→Consumer)")
        print("=" * 70)
        
        # Separate producers and consumers for each table
        table_producers = {}  # {table_name: query_that_produces_it}
        table_consumers = {}  # {table_name: [queries_that_consume_it]}
        
        print("📊 Analyzing Producer-Consumer Relationships:")
        
        for query_name, chain_flow in table_chains.items():
            if not chain_flow:
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
                table_producers[target_table] = query_name
            
            print(f"📄 {query_name}: consumes '{source_table}' → produces '{target_table}'")
        
        # Build the data structure for Part 2 (but don't print connection analysis here)
        joinable_connections = {}
        for table_name, producer_query in table_producers.items():
            if table_name in table_consumers:
                consumer_queries = table_consumers[table_name]
                joinable_connections[table_name] = {
                    "producer": producer_query,
                    "consumers": consumer_queries
                }
        
        return joinable_connections

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
        print(f"\n🔗 STEP 2 - Part 2: Building Joinable Query Chains")
        print("=" * 60)
        
        chains = []
        processed_edges = set()
        
        # Process all producer-consumer relationships  
        for table_name, connection in joinable_connections.items():
            producer = connection["producer"]
            consumers = connection["consumers"]
            
            for consumer in consumers:
                edge_key = f"{producer}→{consumer}"
                if edge_key not in processed_edges:
                    # Build a complete chain through this specific producer-consumer connection
                    chain = cls.build_chain_through_connection(producer, consumer, joinable_connections)
                    chains.append(chain)
                    
                    # Mark all edges in this chain as processed
                    for i in range(len(chain) - 1):
                        processed_edges.add(f"{chain[i]}→{chain[i+1]}")
                    
                    print(f"📋 Chain: {' → '.join(chain)}")
        
        # Find unconnected queries
        all_connected = set()
        for connection in joinable_connections.values():
            all_connected.add(connection["producer"])
            all_connected.update(connection["consumers"])
        
        unconnected_queries = set(all_queries) - all_connected
        
        # Add unconnected queries as single-query chains
        for query in sorted(unconnected_queries):
            chains.append([query])
            print(f"📋 Chain: {query}")
        
        return chains

    @classmethod
    def build_chain_through_connection(cls, producer: str, consumer: str, joinable_connections: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """Build a complete chain that includes the given producer-consumer connection."""
        
        # Find the start of the chain by going backwards
        def find_start(query: str) -> str:
            for connection in joinable_connections.values():
                if query in connection["consumers"]:
                    return find_start(connection["producer"])
            return query
        
        # Build forward from start query, ensuring we go through the specific consumer
        def build_forward_through_consumer(start: str, target_consumer: str) -> List[str]:
            chain = [start]
            current = start
            
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
                                if cls.can_reach_target(next_consumer, target_consumer, joinable_connections):
                                    chain.append(next_consumer)
                                    current = next_consumer
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
                            # Take any consumer (first one found)
                            next_query = consumers[0]
                            chain.append(next_query)
                            current = next_query
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
        print(f"\n🔗 STEP 3: Creating Combined Lineage JSON with Connection Point Merging")
        print("=" * 60)
        
        combined_lineages = []
        
        for chain_idx, chain in enumerate(query_chains, 1):
            print(f"📋 Processing Chain {chain_idx}: {' → '.join(chain)}")
            
            # Create combined lineage for this chain with proper merging
            combined_lineage = cls.create_chain_lineage(chain, chains_data, table_chains, chain_idx, joinable_connections)
            combined_lineages.append(combined_lineage)
            
            print(f"   ✅ Combined lineage created with {len(combined_lineage['chains'])} entities")
        
        return combined_lineages

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
        
        # Get the source table (root of the chain)
        source_table = list(source_chains.keys())[0]
        source_chain_info = source_chains[source_table]
        
        # Build the flowing dependencies with proper merging at connection points (includes ALL entities)
        flowing_dependencies = cls.build_sequential_flow_with_merging(query_chain, chains_data, table_chains, joinable_connections)
        
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
        
        # Get the main dependency chain from the first query (source table)
        source_table = list(source_chains.keys())[0]
        source_chain_info = source_chains[source_table]
        base_dependencies = source_chain_info.get('dependencies', [])
        
        if not base_dependencies:
            return []
        
        # Process the base flow and extend connection points with subsequent queries
        flowing_deps = []
        for dependency in base_dependencies:
            processed_dep = cls.create_dependency_copy(dependency, 1)
            
            # Process this dependency and extend any connection points
            processed_dep = cls.extend_connection_points_in_flow(
                processed_dep, query_chain, connection_points, chains_data, table_chains
            )
            
            flowing_deps.append(processed_dep)
        
        return flowing_deps

    @classmethod
    def extend_connection_points_in_flow(cls, dependency: Dict, query_chain: List[str], connection_points: Dict, chains_data: Dict, table_chains: Dict) -> Dict:
        """Recursively extend connection points in a dependency flow with subsequent query processing."""
        import copy
        
        dep_entity = dependency.get("entity", "")
        
        # Check if this entity is a connection point that needs extending
        if dep_entity in connection_points:
            connection_info = connection_points[dep_entity]
            producer_query = connection_info.get("producer_query")
            consumer_query = connection_info.get("consumer_query")
            
            if producer_query and consumer_query and consumer_query in chains_data:
                print(f"      🔀 Merging connection point: {dep_entity} ({producer_query} → {consumer_query})")
                
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
                            print(f"        Extending connection {dep_entity} with 1 consumer dependencies")
                            
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
                            extended_consumer_dep = cls.create_dependency_copy(consumer_dep, dependency.get("depth", 0) + 1)
                            extended_consumer_dep = cls.extend_connection_points_in_flow(
                                extended_consumer_dep, query_chain, connection_points, chains_data, table_chains
                            )
                            
                            # Replace empty dependencies with the extended consumer processing
                            dependency["dependencies"] = [extended_consumer_dep]
                            break
                    break
        
        # Process nested dependencies recursively
        if "dependencies" in dependency:
            for i, nested_dep in enumerate(dependency["dependencies"]):
                dependency["dependencies"][i] = cls.extend_connection_points_in_flow(
                    nested_dep, query_chain, connection_points, chains_data, table_chains
                )
        
        return dependency


    @classmethod
    def identify_connection_points(cls, query_chain: List[str], table_chains: Dict[str, List[str]], joinable_connections: Dict[str, Dict] = None) -> Dict[str, Dict]:
        """Identify connection points in the query chain that need merging."""
        connection_points = {}
        
        if not joinable_connections:
            return connection_points
        
        print(f"    Analyzing connection points for chain: {' → '.join(query_chain)}")
        
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
                        print(f"      ✅ Found connection point: {table_name} ({producer} → {next_query})")
                        break
        
        print(f"    Found {len(connection_points)} connection points: {list(connection_points.keys())}")
        return connection_points

    @classmethod
    def create_dependency_copy(cls, original_dep: Dict, new_depth: int) -> Dict:
        """Create a deep copy of a dependency with updated depth while preserving all metadata."""
        import copy
        
        # Create a deep copy to preserve all original data
        dep_copy = copy.deepcopy(original_dep)
        
        # Update the depth for this level
        dep_copy["depth"] = new_depth
        
        # Recursively update depths for nested dependencies
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
        # Step 1: Extract complete table chains and store original data
        table_chains, chains_data = cls.analyze_all_table_chains(lineage_data_list)
        
        if not table_chains:
            print("❌ No table chains found.")
            return []
        
        print("\n📋 STEP 1 SUMMARY:")
        print("=" * 70)
        print(f"Total queries analyzed: {len(table_chains)}")
        print("\nComplete flows extracted:")
        for query_name, flow in table_chains.items():
            print(f"   {query_name}: {' → '.join(flow)}")
        
        print(f"\n📦 Original chains data stored for {len(chains_data)} queries")
        
        # Step 2 - Part 1: Identify joinable tables (producer→consumer relationships)
        joinable_connections = cls.identify_joinable_tables(table_chains)
        
        # Step 2 - Part 2: Build joinable query chains
        all_queries = list(table_chains.keys())  # Get all query names dynamically
        joinable_query_chains = cls.identify_joinable_query_chains(joinable_connections, all_queries)
        
        # Step 3: Create combined lineage JSON for each query chain with proper merging
        combined_lineages = cls.create_combined_lineage_json(joinable_query_chains, chains_data, table_chains, joinable_connections)
        
        return combined_lineages
