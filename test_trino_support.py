#!/usr/bin/env python3

"""Test script to check Trino dialect support for catalog.schema.table naming."""

import sys
import os
import json
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.visualization.visualizer import SQLLineageVisualizer

def create_output_directory():
    """Create output directory for generated files."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def save_lineage_chain_json(json_output, test_name, query_name, output_dir):
    """Save lineage chain JSON output to file."""
    filename = f"{output_dir}/{test_name}_{query_name}_lineage_chain.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"📁 Saved lineage chain JSON to: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")
        return None


def generate_jpeg_visualization(json_output, test_name, query_name, output_dir):
    """Generate JPEG visualization from lineage chain JSON."""
    try:
        visualizer = SQLLineageVisualizer()
        output_path = f"{output_dir}/{test_name}_{query_name}_visualization"
        
        # Generate JPEG visualization using lineage chain data
        jpeg_file = visualizer.create_lineage_chain_diagram(
            lineage_chain_json=json_output,
            output_path=output_path,
            output_format="jpeg",
            layout="horizontal"
        )
        
        print(f"🖼️  Generated JPEG visualization: {jpeg_file}")
        return jpeg_file
        
    except Exception as e:
        print(f"❌ Failed to generate visualization: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return None


def test_trino_simple_query():
    """Test simple Trino three-part naming convention support."""
    
    sql = '''SELECT
      "dbxadmin40test"."trino_demo"."partsupp"."ps_partkey" AS "ps_partkey",
      "dbxadmin40test"."trino_demo"."partsupp"."ps_suppkey" AS "ps_suppkey",
      "dbxadmin40test"."trino_demo"."partsupp"."ps_availqty" AS "ps_availqty",
      "dbxadmin40test"."trino_demo"."partsupp"."ps_supplycost" AS "ps_supplycost",
      "dbxadmin40test"."trino_demo"."partsupp"."ps_comment" AS "ps_comment"
    FROM
      "dbxadmin40test"."trino_demo"."partsupp"'''
    
    return test_trino_query(sql, "simple_partsupp")


def test_trino_orders_simple_query():
    """Test simple Trino orders query."""
    
    sql = '''SELECT
      "dbxadmin40test"."trino_demo"."orders"."o_orderkey" AS "o_orderkey",
      "dbxadmin40test"."trino_demo"."orders"."o_custkey" AS "o_custkey",
      "dbxadmin40test"."trino_demo"."orders"."o_orderstatus" AS "o_orderstatus",
      "dbxadmin40test"."trino_demo"."orders"."o_totalprice" AS "o_totalprice",
      CAST(
        "dbxadmin40test"."trino_demo"."orders"."o_orderdate" AS DATE
      ) AS "o_orderdate",
      "dbxadmin40test"."trino_demo"."orders"."o_orderpriority" AS "o_orderpriority",
      "dbxadmin40test"."trino_demo"."orders"."o_clerk" AS "o_clerk",
      "dbxadmin40test"."trino_demo"."orders"."o_shippriority" AS "o_shippriority",
      "dbxadmin40test"."trino_demo"."orders"."o_comment" AS "o_comment"
    FROM
      "dbxadmin40test"."trino_demo"."orders"'''
    
    return test_trino_query(sql, "simple_orders")


def test_trino_complex_cte_query():
    """Test complex Trino CTE query with joins and multiple tables."""
    
    sql = '''WITH
  select_step1 as (
    SELECT
      "dbxadmin40test"."trino_demo"."orders"."o_orderkey" AS "o_orderkey",
      "dbxadmin40test"."trino_demo"."orders"."o_custkey" AS "o_custkey",
      "dbxadmin40test"."trino_demo"."orders"."o_orderstatus" AS "o_orderstatus",
      "dbxadmin40test"."trino_demo"."orders"."o_totalprice" AS "o_totalprice",
      CAST(
        "dbxadmin40test"."trino_demo"."orders"."o_orderdate" AS DATE
      ) AS "o_orderdate",
      "dbxadmin40test"."trino_demo"."orders"."o_orderpriority" AS "o_orderpriority",
      "dbxadmin40test"."trino_demo"."orders"."o_clerk" AS "o_clerk",
      "dbxadmin40test"."trino_demo"."orders"."o_shippriority" AS "o_shippriority",
      "dbxadmin40test"."trino_demo"."orders"."o_comment" AS "o_comment"
    FROM
      "dbxadmin40test"."trino_demo"."orders"
  ),
  join_step2 as (
    SELECT
      select_step1."o_orderkey" AS "o_orderkey",
      select_step1."o_custkey" AS "o_custkey",
      select_step1."o_orderstatus" AS "o_orderstatus",
      select_step1."o_totalprice" AS "o_totalprice",
      CAST(select_step1."o_orderdate" as DATE) AS "o_orderdate",
      select_step1."o_orderpriority" AS "o_orderpriority",
      select_step1."o_clerk" AS "o_clerk",
      select_step1."o_shippriority" AS "o_shippriority",
      select_step1."o_comment" AS "o_comment",
      "dbxadmin40test"."trino_demo"."lineitem"."l_orderkey" AS "l_orderkey",
      "dbxadmin40test"."trino_demo"."lineitem"."l_partkey" AS "l_partkey",
      "dbxadmin40test"."trino_demo"."lineitem"."l_suppkey" AS "l_suppkey",
      "dbxadmin40test"."trino_demo"."lineitem"."l_linenumber" AS "l_linenumber",
      "dbxadmin40test"."trino_demo"."lineitem"."l_quantity" AS "l_quantity",
      "dbxadmin40test"."trino_demo"."lineitem"."l_extendedprice" AS "l_extendedprice",
      "dbxadmin40test"."trino_demo"."lineitem"."l_discount" AS "l_discount",
      "dbxadmin40test"."trino_demo"."lineitem"."l_tax" AS "l_tax",
      "dbxadmin40test"."trino_demo"."lineitem"."l_returnflag" AS "l_returnflag",
      "dbxadmin40test"."trino_demo"."lineitem"."l_linestatus" AS "l_linestatus",
      CAST(
        "dbxadmin40test"."trino_demo"."lineitem"."l_shipdate" as DATE
      ) AS "l_shipdate",
      CAST(
        "dbxadmin40test"."trino_demo"."lineitem"."l_commitdate" as DATE
      ) AS "l_commitdate",
      CAST(
        "dbxadmin40test"."trino_demo"."lineitem"."l_receiptdate" as DATE
      ) AS "l_receiptdate",
      "dbxadmin40test"."trino_demo"."lineitem"."l_shipinstruct" AS "l_shipinstruct",
      "dbxadmin40test"."trino_demo"."lineitem"."l_shipmode" AS "l_shipmode",
      "dbxadmin40test"."trino_demo"."lineitem"."l_comment" AS "l_comment"
    FROM
      select_step1
      INNER JOIN "dbxadmin40test"."trino_demo"."lineitem" ON (
        select_step1."o_orderkey" = "dbxadmin40test"."trino_demo"."lineitem"."l_orderkey"
      )
  )
SELECT
  *
FROM
  join_step2
LIMIT
  100'''
    
    return test_trino_query(sql, "complex_cte_orders_lineitem")


def test_trino_query(sql, test_name):
    """Common function to test a Trino query with lineage chain and visualization generation."""
    
    print(f"Testing Trino query: {test_name}")
    print(f"SQL: {sql[:200]}..." if len(sql) > 200 else f"SQL: {sql}")
    print()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Test with Trino dialect
    analyzer = SQLLineageAnalyzer(dialect="trino")
    result = analyzer.analyze(sql, dialect="trino")
    
    print("=== Analysis Results ===")
    print(f"Dialect: {result.dialect}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print()
    
    print("=== Table Lineage ===")
    print(f"Upstream tables: {result.table_lineage.upstream}")
    print(f"Downstream tables: {result.table_lineage.downstream}")
    print()
    
    print("=== Column Lineage ===")
    print("Upstream columns (first 10):")
    count = 0
    for target, sources in result.column_lineage.upstream.items():
        if count < 10:
            print(f"  {target} <- {sources}")
            count += 1
        else:
            remaining = len(result.column_lineage.upstream) - count
            if remaining > 0:
                print(f"  ... and {remaining} more columns")
            break
    print()
    
    print("=== Metadata ===")
    for table, metadata in result.metadata.items():
        print(f"Table: {table}")
        if metadata is not None:
            print(f"  Catalog: {metadata.catalog}")
            print(f"  Schema: {metadata.schema}")
            print(f"  Table: {metadata.table}")
        else:
            print("  No metadata available")
        print()
    
    # Check if three-part naming is properly recognized
    print("=== Validation ===")
    
    # Get all tables from upstream
    upstream_tables = []
    for target, sources in result.table_lineage.upstream.items():
        upstream_tables.extend(sources)
    
    print(f"Actual upstream tables found: {upstream_tables}")
    
    # Check if the full qualified name is recognized (in various formats)
    found_trino_tables = []
    for table in upstream_tables:
        if "dbxadmin40test" in str(table) and "trino_demo" in str(table):
            found_trino_tables.append(table)
    
    success = False
    if found_trino_tables:
        print(f"✅ Three-part table naming is supported! Found {len(found_trino_tables)} tables:")
        for table in found_trino_tables:
            print(f"  • {table}")
        # Check if quotes are preserved
        if any('"' in str(table) for table in found_trino_tables):
            print("✅ Quotes are preserved in table names")  
        success = True
    else:
        print("❌ No Trino three-part tables found")
        success = False
    
    print()
    
    # Generate lineage chain JSON
    print("=== Generating Lineage Chain JSON ===")    
    try:
        # Generate comprehensive lineage chain JSON with downstream analysis
        chain_json = analyzer.get_lineage_chain_json(sql, "downstream")
        
        # Parse and validate JSON structure
        parsed_json = json.loads(chain_json)
        
        required_keys = ["sql", "dialect", "chain_type", "max_depth", "actual_max_depth", "chains", "summary"]
        missing_keys = [key for key in required_keys if key not in parsed_json]
        
        if missing_keys:
            print(f"❌ Invalid JSON structure. Missing keys: {missing_keys}")
        else:
            chains_count = len(parsed_json.get("chains", {}))
            summary = parsed_json.get("summary", {})
            max_depth = parsed_json.get("max_depth", "unknown")
            actual_depth = parsed_json.get("actual_max_depth", 0)
            
            print(f"✅ Generated lineage chain JSON: {len(chain_json)} characters")
            print(f"   📊 Chains: {chains_count}, Max depth: {max_depth}, Actual depth: {actual_depth}")
            print(f"   📊 Tables: {summary.get('total_tables', 0)}, Columns: {summary.get('total_columns', 0)}")
            print(f"   🔄 Has transformations: {summary.get('has_transformations', False)}")
            print(f"   📋 Has metadata: {summary.get('has_metadata', False)}")
            
            # Save lineage chain JSON
            json_file = save_lineage_chain_json(chain_json, "trino_support_test", test_name, output_dir)
            
            if json_file:
                print(f"✅ Lineage chain JSON saved successfully")
            
            print()
            
            # Generate JPEG visualization
            print("=== Generating JPEG Visualization ===")
            jpeg_file = generate_jpeg_visualization(chain_json, "trino_support_test", test_name, output_dir)
            
            if jpeg_file:
                print(f"✅ JPEG visualization generated successfully")
                
                # Get file size
                try:
                    jpeg_size = os.path.getsize(jpeg_file)
                    print(f"   📏 File size: {jpeg_size:,} bytes")
                except:
                    pass
            
            print()
            
    except Exception as e:
        print(f"❌ Failed to generate lineage chain JSON or visualization: {e}")
        print(f"   Error details: {traceback.format_exc()}")
    
    print("=== Test Summary ===")
    print(f"✅ Trino dialect support: {'Passed' if success else 'Failed'}")
    print(f"📁 Output directory: {output_dir}")
    
    # List generated files
    try:
        files = os.listdir(output_dir)
        trino_files = [f for f in files if 'trino_support_test' in f]
        if trino_files:
            print(f"📄 Generated files:")
            for file in sorted(trino_files):
                file_path = os.path.join(output_dir, file)
                try:
                    size = os.path.getsize(file_path)
                    print(f"   • {file} ({size:,} bytes)")
                except:
                    print(f"   • {file}")
        else:
            print("⚠️  No files were generated")
    except Exception as e:
        print(f"⚠️  Could not list output files: {e}")
    
    return success


def main():
    """Main function to run all Trino support tests."""
    print("🚀 Trino Support Test Suite")
    print("=" * 60)
    print("Testing Trino catalog.schema.table naming convention support")
    print("with lineage chain JSON generation and JPEG visualization")
    print()
    
    results = []
    
    # Test 1: Simple partsupp query
    print("📋 Test 1: Simple Partsupp Query")
    print("-" * 40)
    results.append(test_trino_simple_query())
    print()
    
    # Test 2: Simple orders query
    print("📋 Test 2: Simple Orders Query")
    print("-" * 40)
    results.append(test_trino_orders_simple_query())
    print()
    
    # Test 3: Complex CTE query with joins
    print("📋 Test 3: Complex CTE with Orders and LineItem Join")
    print("-" * 40)
    results.append(test_trino_complex_cte_query())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("🎉 Trino Support Test Suite Completed!")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Trino three-part naming is fully supported.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print(f"\n📁 Check the 'output/' directory for generated JSON and JPEG files.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())