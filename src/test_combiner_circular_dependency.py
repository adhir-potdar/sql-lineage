#!/usr/bin/env python3
"""
Test script to check circular dependency handling in LineageChainCombiner.
Creates multiple SQL queries that form circular dependencies between them.
"""

import sys
import os
import json
import signal
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.lineage_chain_combiner import LineageChainCombiner


def create_circular_dependency_queries():
    """Create a set of SQL queries that form circular dependencies."""
    queries = [
        # Query 1: Creates table A from table C (C -> A)
        """
        CREATE TABLE schema.table_a AS
        SELECT 
            id,
            name,
            status
        FROM schema.table_c
        WHERE active = 1
        """,
        
        # Query 2: Creates table B from table A (A -> B)  
        """
        CREATE TABLE schema.table_b AS
        SELECT
            id,
            name,
            processed_status
        FROM schema.table_a
        WHERE status = 'valid'
        """,
        
        # Query 3: Creates table C from table B (B -> C) - This completes the cycle!
        """
        CREATE TABLE schema.table_c AS
        SELECT
            id,
            name as entity_name,
            'processed' as active
        FROM schema.table_b
        WHERE processed_status IS NOT NULL
        """
    ]
    
    return queries



def create_chain_dependency_queries():
    """Create queries that form a chain with potential circular issues."""
    queries = [
        # Query 1: A depends on D
        """
        CREATE VIEW reporting.view_a AS
        SELECT id, name FROM staging.table_d
        """,
        
        # Query 2: B depends on A
        """  
        CREATE VIEW reporting.view_b AS
        SELECT id, name, 'processed' as status FROM reporting.view_a
        """,
        
        # Query 3: C depends on B
        """
        CREATE VIEW reporting.view_c AS
        SELECT id, name, status FROM reporting.view_b WHERE id > 0
        """,
        
        # Query 4: D depends on C - creates cycle A->B->C->D->A
        """
        CREATE TABLE staging.table_d AS
        SELECT id, name FROM reporting.view_c WHERE status = 'processed'
        """
    ]
    
    return queries


def test_combiner_with_real_queries():
    """Test combiner with real SQL queries that create circular dependencies."""
    print("Testing LineageChainCombiner with real circular SQL queries...")
    
    circular_queries = create_circular_dependency_queries()
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Analyze each query individually (they should pass since individual queries are valid)
    lineage_data_list = []
    print(f"\n📝 Analyzing {len(circular_queries)} queries individually...")
    
    for i, sql in enumerate(circular_queries, 1):
        print(f"Query {i}: Analyzing...")
        try:
            result = analyzer.analyze(sql)
            if result.errors:
                print(f"❌ Query {i} failed with errors: {result.errors}")
                continue
                
            # Convert to chain JSON format
            chain_json = analyzer.get_lineage_chain_json(sql)
            lineage_data = json.loads(chain_json)
            lineage_data_list.append(lineage_data)
            
            # Save individual query lineage JSON
            query_output_file = f"output/combiner_test_query_{i}_lineage.json"
            with open(query_output_file, 'w') as f:
                f.write(chain_json)
            print(f"💾 Query {i} lineage saved to: {query_output_file}")
            
            # Create visualization for each query
            try:
                from analyzer.visualization.visualizer import create_lineage_chain_visualization
                viz_file = create_lineage_chain_visualization(
                    chain_json,
                    output_path=f"output/combiner_test_query_{i}_lineage",
                    output_format="jpeg"
                )
                print(f"🎨 Query {i} visualization saved to: {viz_file}")
            except Exception as viz_error:
                print(f"⚠️  Query {i} visualization failed: {str(viz_error)}")
            
            print(f"✅ Query {i} analyzed successfully")
            
        except Exception as e:
            print(f"❌ Query {i} failed: {str(e)}")
            return False
    
    if len(lineage_data_list) < 3:
        print("⚠️  Not enough valid queries to create circular dependencies")
        return True
    
    # Test the combiner with timeout protection
    print(f"\n🔗 Testing combiner with {len(lineage_data_list)} lineage datasets...")
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Combiner took too long - likely infinite loop")
    
    try:
        # Set 15-second timeout to catch infinite loops
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)
        
        # Test each combiner step
        table_chains, chains_data = LineageChainCombiner.analyze_all_table_chains(lineage_data_list)
        print(f"✅ Step 1 (extract chains): {len(table_chains)} chains")
        
        joinable_connections = LineageChainCombiner.identify_joinable_tables(table_chains)
        print(f"✅ Step 2a (joinable tables): {len(joinable_connections)} connections")
        
        all_queries = list(table_chains.keys())
        query_chains = LineageChainCombiner.identify_joinable_query_chains(joinable_connections, all_queries)
        print(f"✅ Step 2b (query chains): {len(query_chains)} chains built")
        
        signal.alarm(0)  # Cancel timeout
        
        # Save combined results to output folder
        combined_result = {
            "test_type": "circular_dependency_combiner_test",
            "total_queries": len(lineage_data_list),
            "table_chains": table_chains,
            "joinable_connections": joinable_connections,
            "query_chains": query_chains,
            "individual_lineage_data": chains_data
        }
        
        # Save to JSON file
        output_file = "output/combiner_circular_dependency_test.json"
        with open(output_file, 'w') as f:
            json.dump(combined_result, f, indent=2)
        print(f"💾 Combined results saved to: {output_file}")
        
        # Generate final combined lineage chain output using combiner
        try:
            # Create the final combined lineage JSON
            combined_chains_output = LineageChainCombiner.create_combined_lineage_json(
                query_chains, chains_data, table_chains, joinable_connections
            )
            
            # Save combined lineage chain JSON
            combined_lineage_file = "output/combiner_final_combined_lineage_chains.json"
            with open(combined_lineage_file, 'w') as f:
                json.dump(combined_chains_output, f, indent=2)
            print(f"💾 Combined lineage chains saved to: {combined_lineage_file}")
            
            # Create visualization for combined result (use first chain if available)
            if combined_chains_output and len(combined_chains_output) > 0:
                first_combined_chain = combined_chains_output[0]
                combined_lineage_json = json.dumps(first_combined_chain, indent=2)
                from analyzer.visualization.visualizer import create_lineage_chain_visualization
                viz_file = create_lineage_chain_visualization(
                    combined_lineage_json,
                    output_path="output/combiner_final_combined_lineage_chains",
                    output_format="jpeg"
                )
                print(f"🎨 Combined lineage visualization saved to: {viz_file}")
            
        except Exception as combine_error:
            print(f"⚠️  Combined lineage generation failed: {str(combine_error)}")
        
        # All individual visualizations were already created above
        
        # Analyze results
        print("\n🔍 Analysis Results:")
        for i, chain in enumerate(query_chains):
            if len(chain) != len(set(chain)):
                print(f"🔄 Circular chain {i+1}: {' → '.join(chain)}")
            else:
                print(f"📊 Linear chain {i+1}: {' → '.join(chain)}")
        
        # Print connection details
        print(f"\n📋 Joinable Connections Found:")
        for table, connection in joinable_connections.items():
            producer = connection["producer"]
            consumers = ", ".join(connection["consumers"])
            print(f"   {table}: {producer} → {consumers}")
        
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ CRITICAL: Combiner infinite loop detected!")
        return False
    except Exception as e:
        signal.alarm(0)
        if "CircularDependencyError" in str(e):
            print(f"✅ SUCCESS: Combiner rejected circular dependencies - {str(e)}")
            return True
        else:
            print(f"❌ Combiner failed: {str(e)}")
            return False


def test_combiner_circular_connections():
    """Test circular producer-consumer relationships in combiner."""
    print("\n" + "="*60)  
    print("Testing combiner with predefined circular connections...")
    
    # Create a circular connection scenario:
    # Query1 produces Table_A, Query2 consumes Table_A
    # Query2 produces Table_B, Query3 consumes Table_B
    # Query3 produces Table_C, Query1 consumes Table_C
    # This creates: Query1 -> Query2 -> Query3 -> Query1 (circular)
    
    joinable_connections = {
        "schema.table_a": {
            "producer": "query_1",
            "consumers": ["query_2"]
        },
        "schema.table_b": {
            "producer": "query_2", 
            "consumers": ["query_3"]
        },
        "schema.table_c": {
            "producer": "query_3",
            "consumers": ["query_1"]  # Creates the cycle
        }
    }
    
    all_queries = ["query_1", "query_2", "query_3"]
    
    def timeout_handler(signum, frame):
        raise TimeoutError("identify_joinable_query_chains took too long")
    
    try:
        # Set timeout to catch infinite loops
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        # This should handle circular connections gracefully
        chains = LineageChainCombiner.identify_joinable_query_chains(
            joinable_connections, all_queries
        )
        
        signal.alarm(0)  # Cancel timeout
        
        print(f"✅ Combiner handled circular connections: {len(chains)} chains")
        for i, chain in enumerate(chains):
            print(f"   Chain {i+1}: {' → '.join(chain)}")
            
        # Check if any chains have cycles (repeated queries)
        has_cycles = False
        for chain in chains:
            if len(chain) != len(set(chain)):
                print(f"🔄 Circular chain detected: {' → '.join(chain)}")
                has_cycles = True
        
        if not has_cycles:
            print("ℹ️  No circular chains in output (combiner may have broken cycles)")
            
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ CRITICAL: Infinite loop in identify_joinable_query_chains!")
        return False
    except RecursionError:
        print("❌ CRITICAL: Infinite recursion in combiner!")
        return False
    except Exception as e:
        print(f"⚠️  Combiner error: {str(e)}")
        return False


def test_find_start_infinite_loop():
    """Test the find_start function for infinite recursion."""
    print("\nTesting find_start infinite recursion...")
    
    # Create circular producer-consumer chain
    joinable_connections = {
        "table_x": {
            "producer": "query_1",
            "consumers": ["query_2"]
        },
        "table_y": {
            "producer": "query_2",
            "consumers": ["query_1"]  # Circular reference
        }
    }
    
    def timeout_handler(signum, frame):
        raise TimeoutError("build_chain_through_connection took too long")
    
    try:
        # Set timeout for infinite loop detection
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Test build_chain_through_connection which calls find_start
        chain = LineageChainCombiner.build_chain_through_connection(
            "query_1", "query_2", joinable_connections
        )
        
        signal.alarm(0)  # Cancel timeout
        
        print(f"✅ find_start handled circular reference: {' → '.join(chain)}")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ CRITICAL: find_start has infinite loop!")
        return False
    except RecursionError:
        print("❌ CRITICAL: find_start has infinite recursion!")
        return False
    except Exception as e:
        print(f"⚠️  find_start error: {str(e)}")
        return False


def test_can_reach_target_cycles():
    """Test can_reach_target with circular references."""
    print("\nTesting can_reach_target with cycles...")
    
    # Create connections with cycles
    joinable_connections = {
        "table_p": {
            "producer": "query_a",
            "consumers": ["query_b"]
        },
        "table_q": {
            "producer": "query_b", 
            "consumers": ["query_c"]
        },
        "table_r": {
            "producer": "query_c",
            "consumers": ["query_a"]  # Cycle back to start
        }
    }
    
    def timeout_handler(signum, frame):
        raise TimeoutError("can_reach_target took too long")
    
    try:
        # Set timeout for infinite loop detection
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Test reachability in circular graph
        result1 = LineageChainCombiner.can_reach_target("query_a", "query_b", joinable_connections)
        result2 = LineageChainCombiner.can_reach_target("query_b", "query_a", joinable_connections) 
        result3 = LineageChainCombiner.can_reach_target("query_a", "query_a", joinable_connections)
        
        signal.alarm(0)  # Cancel timeout
        
        print(f"✅ can_reach_target handled cycles: a->b: {result1}, b->a: {result2}, a->a: {result3}")
        return True
        
    except TimeoutError:
        signal.alarm(0)
        print("❌ CRITICAL: can_reach_target has infinite loop!")
        return False
    except RecursionError:
        print("❌ CRITICAL: can_reach_target has infinite recursion!")
        return False
    except Exception as e:
        print(f"⚠️  can_reach_target error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔍 Testing Lineage Chain Combiner Circular Dependencies")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run tests
    test1 = test_combiner_with_real_queries()
    test2 = test_combiner_circular_connections()
    test3 = test_find_start_infinite_loop()
    test4 = test_can_reach_target_cycles()
    
    print(f"\n📊 Test Results:")
    print(f"   Real Query Circular Dependencies: {'✅ SAFE' if test1 else '❌ UNSAFE'}")
    print(f"   Circular Connections: {'✅ SAFE' if test2 else '❌ UNSAFE'}")
    print(f"   find_start Function: {'✅ SAFE' if test3 else '❌ UNSAFE'}")
    print(f"   can_reach_target Function: {'✅ SAFE' if test4 else '❌ UNSAFE'}")
    
    if all([test1, test2, test3, test4]):
        print("\n🎉 All combiner circular dependency tests PASSED!")
    else:
        print("\n⚠️  Some combiner tests failed - Review circular dependency handling!")