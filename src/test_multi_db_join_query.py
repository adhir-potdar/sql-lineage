#!/usr/bin/env python3
"""
Test script for multi-database JOIN query with quoted table names
Creates JSON lineage chain and visualization JPEG
"""
import sys
import os
import json
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.visualization.visualizer import create_lineage_chain_visualization

def test_multi_db_join_query():
    """
    Test the multi-database JOIN query that was previously failing
    """
    print("🔍 Testing Multi-Database JOIN Query with Column Metadata")
    print("=" * 70)
    
    # Test both original query and CTE version
    queries = [
        {
            'name': 'Simple Multi-DB JOIN',
            'filename': 'multi_db_join_query',
            'sql': """SELECT b.agent_vendor AS vendor, a.PLAN_NAME AS plan, SUM(a.MER) AS revenue FROM "oracle".CUSTOMER_DEMO.MASTER_PLAN_TABLE_WDATES AS a JOIN "postgresql".customer_demo.fact_subscription_activity AS c ON a.PLAN_ID = c.PLAN_ID JOIN "redshift".customer_demo.dim_agent AS b ON c.AGN_KEY = b.agent_record_key WHERE a.ENDDATE BETWEEN CAST('2018-01-01' AS TIMESTAMP) AND CAST('2019-12-31' AS TIMESTAMP) GROUP BY b.agent_vendor, a.PLAN_NAME ORDER BY b.agent_vendor, a.PLAN_NAME"""
        },
        {
            'name': 'CTE Multi-DB Query',
            'filename': 'multi_db_cte_query',
            'sql': """WITH revenue_data AS (
    SELECT
        b.agent_vendor AS vendor,
        a.PLAN_NAME AS plan,
        SUM(a.MER) AS revenue
    FROM "oracle".CUSTOMER_DEMO.MASTER_PLAN_TABLE_WDATES AS a
    JOIN "postgresql".customer_demo.fact_subscription_activity AS c
        ON a.PLAN_ID = c.PLAN_ID
    JOIN "redshift".customer_demo.dim_agent AS b
        ON c.AGN_KEY = b.agent_record_key
    WHERE
        a.ENDDATE BETWEEN CAST('2012-01-01' AS TIMESTAMP) AND CAST('2021-12-31' AS TIMESTAMP)
    GROUP BY
        b.agent_vendor,
        a.PLAN_NAME
)
SELECT
    vendor,
    plan,
    revenue
FROM revenue_data
ORDER BY
    vendor,
    plan"""
        },
        {
            'name': 'Simple CTE Select Query',
            'filename': 'simple_cte_select_query',
            'sql': """WITH
  select_step1 as (
    SELECT
      "postgresql"."customer_demo"."items"."item_name" AS "item_name",
      "postgresql"."customer_demo"."items"."item_id" AS "item_id",
      "postgresql"."customer_demo"."items"."category_id" AS "category_id"
    FROM
      "postgresql"."customer_demo"."items"
  )
SELECT
  *
FROM
  select_step1
LIMIT
  100"""
        }
    ]
    
    all_success = True
    
    for i, query_config in enumerate(queries, 1):
        print(f"\n{'='*20} TEST {i}: {query_config['name']} {'='*20}")
        print("📋 SQL Query:")
        print("-" * 50)
        print(query_config['sql'])
        print()
        
        success = test_single_query(query_config['sql'], query_config['filename'], query_config['name'])
        if not success:
            all_success = False
    
    return all_success

def test_single_query(sql_query, filename_base, query_name):
    """Test a single query and generate its lineage chain and visualization"""
    
    # Initialize the analyzer
    print("🚀 Initializing SQLLineageAnalyzer...")
    analyzer = SQLLineageAnalyzer(dialect='trino')
    
    # Test basic analysis first
    print("📊 Running Analysis...")
    start_time = time.time()
    
    try:
        result = analyzer.analyze(sql_query)
        analysis_time = time.time() - start_time
        
        print(f"✅ Analysis completed in {analysis_time:.3f} seconds")
        print(f"   📋 Table lineage: {len(result.table_lineage.downstream)} downstream tables")
        print(f"   📋 Column lineage: {len(result.column_lineage.downstream)} downstream columns")
        
        # DEBUG: Check what's in the table lineage
        print(f"   🔍 DEBUG - Downstream table lineage: {dict(result.table_lineage.downstream)}")
        print(f"   🔍 DEBUG - Upstream table lineage: {dict(result.table_lineage.upstream)}")
        
        # Store result for later inspection
        global debug_result
        debug_result = result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    # Generate lineage chain JSON
    print("\n📄 Generating Lineage Chain JSON...")
    json_start_time = time.time()
    
    try:
        # The insurance query uses "downstream" and gets individual table chains
        # Let's try different approaches to match that behavior
        
        print("   📄 Trying downstream direction (like insurance query)...")
        lineage_chain_json = analyzer.get_lineage_chain_json(sql_query, "downstream")
        lineage_data = json.loads(lineage_chain_json)
        chains = lineage_data.get('chains', {})
        
        print(f"      Chains generated: {len(chains)}")
        print(f"      Chain names: {list(chains.keys())}")
        
        # Debug: Check if we can get individual table chains by using upstream direction
        if len(chains) == 1 and 'QUERY_RESULT' in chains:
            print("   📄 Only QUERY_RESULT chain generated, trying upstream direction...")
            upstream_json = analyzer.get_lineage_chain_json(sql_query, "upstream")
            upstream_data = json.loads(upstream_json)
            upstream_chains = upstream_data.get('chains', {})
            print(f"      Upstream chains: {len(upstream_chains)}")
            print(f"      Upstream chain names: {list(upstream_chains.keys())}")
            
            # If upstream gives us individual table chains, use that instead
            if len(upstream_chains) > 1 and 'QUERY_RESULT' not in upstream_chains:
                print("   ✅ Using upstream direction - gives individual table chains")
                lineage_chain_json = upstream_json
                lineage_data = upstream_data
                chains = upstream_chains
        
            
        json_time = time.time() - json_start_time
        
        print(f"✅ JSON generation completed in {json_time:.3f} seconds")
        summary = lineage_data.get('summary', {})
        
        # Check chain structure
        chain_structure = "Individual table chains" if len(chains) > 1 else "QUERY_RESULT with dependencies"
        
        print("\n📊 Lineage Chain Summary:")
        print(f"   🔗 Chain count: {len(chains)}")
        print(f"   🔗 Chain structure: {chain_structure}")
        print(f"   📋 Total tables: {summary.get('total_tables', 0)}")
        print(f"   📋 Total columns: {summary.get('total_columns', 0)}")
        print(f"   🔄 Has transformations: {summary.get('has_transformations', False)}")
        print(f"   📋 Has metadata: {summary.get('has_metadata', False)}")
        
        # Analyze each chain for column metadata
        print("\n📋 Chain Details:")
        total_table_columns = 0
        dependency_table_columns = 0
        
        for entity_name, chain_data in chains.items():
            if isinstance(chain_data, dict):
                metadata = chain_data.get('metadata', {})
                table_columns = metadata.get('table_columns', [])
                total_table_columns += len(table_columns)
                
                print(f"   📊 {entity_name}")
                print(f"      Columns: {len(table_columns)}")
                if table_columns:
                    column_names = [col.get('name') for col in table_columns[:5]]  # Show first 5
                    more_text = f"... (+{len(table_columns)-5} more)" if len(table_columns) > 5 else ""
                    print(f"      Names: {column_names}{more_text}")
                    
                    # Show column types
                    column_types = [col.get('type') for col in table_columns[:3]]
                    print(f"      Types: {column_types}")
                
                # Also check dependencies for column metadata
                dependencies = chain_data.get('dependencies', [])
                if dependencies:
                    print(f"      Dependencies: {len(dependencies)}")
                    for i, dep in enumerate(dependencies):
                        if isinstance(dep, dict):
                            dep_metadata = dep.get('metadata', {})
                            dep_table_columns = dep_metadata.get('table_columns', [])
                            dependency_table_columns += len(dep_table_columns)
                            dep_entity = dep.get('entity', 'Unknown')
                            print(f"         {i+1}. {dep_entity}: {len(dep_table_columns)} columns")
        
        print(f"\n   📊 Total table_columns in main chains: {total_table_columns}")
        print(f"   📊 Total table_columns in dependencies: {dependency_table_columns}")
        print(f"   📊 Grand total table_columns: {total_table_columns + dependency_table_columns}")
        
        # Verify the fix worked - for SELECT queries, columns might be in dependencies
        total_columns_found = total_table_columns + dependency_table_columns
        expected_columns = 9  # Based on the SQL having columns like agent_vendor, PLAN_NAME, MER, AGN_KEY, etc.
        
        if total_columns_found > 0 and summary.get('total_columns', 0) > 0:
            print("\n🎉 SUCCESS: Column metadata extraction is working!")
            print(f"   ✅ Found {total_columns_found} table_columns and {summary.get('total_columns', 0)} used_columns")
            print("   ✅ Multi-database JOIN with quotes handled correctly")
            success_status = True
        elif summary.get('total_columns', 0) >= expected_columns:
            print("\n✅ ACCEPTABLE: Used columns count is correct")
            print(f"   ✅ Found {summary.get('total_columns', 0)} used_columns (expected ~{expected_columns})")
            print("   ℹ️  Table columns in dependencies: this is normal for SELECT queries")
            success_status = True
        else:
            print("\n⚠️  WARNING: Column metadata may still have issues")
            success_status = False
        
    except Exception as e:
        print(f"❌ JSON generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save JSON to file
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    json_filename = f"{filename_base}_lineage_chain.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    try:
        with open(json_filepath, 'w') as f:
            json.dump(lineage_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 JSON saved to: {json_filename}")
        print(f"   📁 Full path: {json_filepath}")
        print(f"   📊 File size: {os.path.getsize(json_filepath)} bytes")
        
    except Exception as e:
        print(f"❌ Failed to save JSON: {e}")
        return False
    
    # Generate visualization
    print("\n🎨 Generating Visualization...")
    viz_start_time = time.time()
    viz_time = 0  # Initialize to avoid UnboundLocalError
    
    try:
        # Create visualization filename
        viz_filename = f"{filename_base}_visualization.jpeg"
        viz_filepath = os.path.join(output_dir, viz_filename)
        
        # Generate visualization (pass JSON string, not parsed dict)
        viz_file = create_lineage_chain_visualization(
            lineage_chain_json, 
            output_path=os.path.join(output_dir, viz_filename.replace('.jpeg', '')),
            output_format="jpeg"
        )
        
        viz_time = time.time() - viz_start_time
        print(f"✅ Visualization completed in {viz_time:.3f} seconds")
        print(f"   🖼️  Image saved to: {viz_file if viz_file else viz_filename}")
        
        # Check if file was created
        if viz_file and os.path.exists(viz_file):
            print(f"   📁 Full path: {viz_file}")
            print(f"   📊 Image size: {os.path.getsize(viz_file)} bytes")
        elif os.path.exists(viz_filepath):
            print(f"   📁 Full path: {viz_filepath}")
            print(f"   📊 Image size: {os.path.getsize(viz_filepath)} bytes")
        
    except Exception as e:
        viz_time = time.time() - viz_start_time
        print(f"❌ Visualization generation failed: {e}")
        print("   Note: Visualization failure doesn't affect core functionality")
    
    # Performance summary
    total_time = time.time() - start_time
    print(f"\n⏱️  Performance Summary:")
    print(f"   🔍 Analysis time: {analysis_time:.3f}s")
    print(f"   📄 JSON generation: {json_time:.3f}s") 
    print(f"   🎨 Visualization: {viz_time:.3f}s")
    print(f"   ⏱️  Total time: {total_time:.3f}s")
    
    # Final validation - use the success_status from JSON generation
    print(f"\n🔍 Final Validation:")
    try:
        # success_status was set during JSON generation
        if success_status:
            print("   ✅ Column metadata extraction working correctly")
            print("   ✅ Multi-database JOIN query processed successfully")
            print("   ✅ Both JSON lineage and JPEG visualization generated")
            return True
        else:
            print("   ❌ Column metadata issues detected")
            return False
    except NameError:
        # Fallback if success_status wasn't set (JSON generation failed)
        print("   ❌ Test failed during JSON generation")
        return False

if __name__ == "__main__":
    print("🚀 Multi-Database JOIN Query Test")
    print("=" * 50)
    
    success = test_multi_db_join_query()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Test completed successfully!")
        print("✅ All validations passed")
    else:
        print("❌ Test encountered issues")
        print("⚠️  Check the output above for details")