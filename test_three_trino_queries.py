#!/usr/bin/env python3
"""
Test code to generate lineage chain JSONs and JPEG visualizations for three Trino queries.
"""

import sys
import os
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.visualization.visualizer import SQLLineageVisualizer


def test_query_1():
    """Test Query 1: Simple CTE with filter and limit."""
    print("📊 Testing Query 1: Simple CTE with filter")
    print("=" * 60)
    
    sql_query_1 = """
    WITH
      select_step1 as (
        SELECT
          "postgresqlyauhentest1111111111"."public"."sales_retail"."row_id" AS "row_id",
          "postgresqlyauhentest1111111111"."public"."sales_retail"."order_id" AS "order_id",
          "postgresqlyauhentest1111111111"."public"."sales_retail"."order_date" AS "order_date"
        FROM
          "postgresqlyauhentest1111111111"."public"."sales_retail"
        WHERE
          (
            "postgresqlyauhentest1111111111"."public"."sales_retail"."row_id" IS NOT NULL
          )
      )
    SELECT
      *
    FROM
      select_step1
    LIMIT
      100
    """
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Generate lineage chain JSON
    result_json = analyzer.get_lineage_chain_json(sql_query_1, "downstream")
    
    # Save JSON to file
    json_filename = "output/trino_query_1_lineage_chain.json"
    with open(json_filename, 'w') as f:
        json.dump(json.loads(result_json), f, indent=2)
    
    print(f"✅ Lineage JSON saved: {json_filename}")
    
    # Generate JPEG visualization
    try:
        visualizer = SQLLineageVisualizer()
        output_path = "output/trino_query_1_lineage_visualization"
        jpeg_file = visualizer.create_lineage_chain_diagram(
            lineage_chain_json=result_json,
            output_path=output_path,
            output_format="jpeg",
            layout="horizontal"
        )
        print(f"✅ JPEG visualization saved: {jpeg_file}")
    except Exception as e:
        print(f"❌ JPEG generation failed: {e}")
    
    # Print summary
    parsed = json.loads(result_json)
    print(f"\n📈 Summary:")
    print(f"   Tables: {parsed['summary']['total_tables']}")
    print(f"   Columns: {parsed['summary']['total_columns']}")
    print(f"   Has Transformations: {parsed['summary']['has_transformations']}")
    print()


def test_query_2():
    """Test Query 2: CTE with LEFT OUTER JOIN."""
    print("📊 Testing Query 2: CTE with LEFT OUTER JOIN")
    print("=" * 60)
    
    sql_query_2 = """
    WITH
      select_step1 as (
        SELECT
          "postgresqlyauhentest1111111111"."public"."sales_retail"."row_id" AS "row_id",
          "postgresqlyauhentest1111111111"."public"."sales_retail"."order_id" AS "order_id",
          "postgresqlyauhentest1111111111"."public"."sales_retail"."order_date" AS "order_date"
        FROM
          "postgresqlyauhentest1111111111"."public"."sales_retail"
      ),
      join_step2 as (
        SELECT
          select_step1."row_id" AS "row_id",
          select_step1."order_id" AS "order_id",
          select_step1."order_date" AS "order_date",
          "postgresqlyauhentest1111111111"."public"."machine_stats"."product_id" AS "product_id",
          "postgresqlyauhentest1111111111"."public"."machine_stats"."rotational_speed_rpm" AS "rotational_speed_rpm",
          "postgresqlyauhentest1111111111"."public"."machine_stats"."torque_nm" AS "torque_nm",
          "postgresqlyauhentest1111111111"."public"."machine_stats"."tool_wear_min" AS "tool_wear_min"
        FROM
          select_step1
          LEFT OUTER JOIN "postgresqlyauhentest1111111111"."public"."machine_stats" ON (
            select_step1."order_id" = "postgresqlyauhentest1111111111"."public"."machine_stats"."product_id"
          )
      )
    SELECT
      *
    FROM
      join_step2
    LIMIT
      100
    """
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Generate lineage chain JSON
    result_json = analyzer.get_lineage_chain_json(sql_query_2, "downstream")
    
    # Save JSON to file
    json_filename = "output/trino_query_2_lineage_chain.json"
    with open(json_filename, 'w') as f:
        json.dump(json.loads(result_json), f, indent=2)
    
    print(f"✅ Lineage JSON saved: {json_filename}")
    
    # Generate JPEG visualization
    try:
        visualizer = SQLLineageVisualizer()
        output_path = "output/trino_query_2_lineage_visualization"
        jpeg_file = visualizer.create_lineage_chain_diagram(
            lineage_chain_json=result_json,
            output_path=output_path,
            output_format="jpeg",
            layout="horizontal"
        )
        print(f"✅ JPEG visualization saved: {jpeg_file}")
    except Exception as e:
        print(f"❌ JPEG generation failed: {e}")
    
    # Print summary
    parsed = json.loads(result_json)
    print(f"\n📈 Summary:")
    print(f"   Tables: {parsed['summary']['total_tables']}")
    print(f"   Columns: {parsed['summary']['total_columns']}")
    print(f"   Has Transformations: {parsed['summary']['has_transformations']}")
    print()


def test_query_3():
    """Test Query 3: Complex CTE with multiple LEFT OUTER JOINs."""
    print("📊 Testing Query 3: Complex CTE with multiple LEFT OUTER JOINs")
    print("=" * 60)
    
    sql_query_3 = """
    WITH
      select_step1 as (
        SELECT
          "machine_stats"."product_id" AS "product_id",
          "machine_stats"."rotational_speed_rpm" AS "rotational_speed_rpm",
          "machine_stats"."torque_nm" AS "torque_nm",
          "machine_stats"."tool_wear_min" AS "tool_wear_min"
        FROM
          "public"."machine_stats"
      ),
      join_step2 as (
        SELECT
          select_step1."product_id" AS "product_id",
          select_step1."rotational_speed_rpm" AS "rotational_speed_rpm",
          select_step1."torque_nm" AS "torque_nm",
          select_step1."tool_wear_min" AS "tool_wear_min",
          "sales_retail"."row_id" AS "row_id",
          "sales_retail"."order_id" AS "order_id",
          "sales_retail"."order_date" AS "order_date",
          "sales_retail"."ship_date" AS "ship_date",
          "sales_retail"."ship_mode" AS "ship_mode",
          "sales_retail"."customer_id" AS "customer_id",
          "sales_retail"."customer_name" AS "customer_name",
          "sales_retail"."segment" AS "segment",
          "sales_retail"."city" AS "city",
          "sales_retail"."state" AS "state",
          "sales_retail"."country" AS "country",
          "sales_retail"."postal code" AS "postal code",
          "sales_retail"."market" AS "market",
          "sales_retail"."region" AS "region",
          "sales_retail"."product id" AS "product id",
          "sales_retail"."sales" AS "sales",
          "sales_retail"."quantity" AS "quantity",
          "sales_retail"."discount" AS "discount",
          "sales_retail"."profit" AS "profit",
          "sales_retail"."shipping_cost" AS "shipping_cost",
          "sales_retail"."order_priority" AS "order_priority"
        FROM
          select_step1
          LEFT OUTER JOIN "public"."sales_retail" ON (
            select_step1."product_id" = "sales_retail"."order_id"
          )
      ),
      join_step3 as (
        SELECT
          join_step2."product_id" AS "product_id",
          join_step2."rotational_speed_rpm" AS "rotational_speed_rpm",
          join_step2."torque_nm" AS "torque_nm",
          join_step2."tool_wear_min" AS "tool_wear_min",
          join_step2."row_id" AS "row_id",
          join_step2."order_id" AS "order_id",
          join_step2."order_date" AS "order_date",
          join_step2."ship_date" AS "ship_date",
          join_step2."ship_mode" AS "ship_mode",
          join_step2."customer_id" AS "customer_id",
          join_step2."customer_name" AS "customer_name",
          join_step2."segment" AS "segment",
          join_step2."city" AS "city",
          join_step2."state" AS "state",
          join_step2."country" AS "country",
          join_step2."postal code" AS "postal code",
          join_step2."market" AS "market",
          join_step2."region" AS "region",
          join_step2."product id" AS "product id",
          join_step2."sales" AS "sales",
          join_step2."quantity" AS "quantity",
          join_step2."discount" AS "discount",
          join_step2."profit" AS "profit",
          join_step2."shipping_cost" AS "shipping_cost",
          join_step2."order_priority" AS "order_priority",
          "little_orange_trip"."id" AS "id",
          "little_orange_trip"."driver_id" AS "driver_id",
          "little_orange_trip"."passenger_id" AS "passenger_id",
          "little_orange_trip"."city_id" AS "city_id",
          "little_orange_trip"."call_time" AS "call_time",
          "little_orange_trip"."finish_time" AS "finish_time",
          "little_orange_trip"."surge_rate" AS "surge_rate",
          "little_orange_trip"."trip_distance" AS "trip_distance",
          "little_orange_trip"."trip_fare" AS "trip_fare"
        FROM
          join_step2
          LEFT OUTER JOIN "public"."little_orange_trip" ON (
            join_step2."product_id" = "little_orange_trip"."driver_id"
          )
      )
    SELECT
      *
    FROM
      join_step3
    LIMIT
      100
    """
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Generate lineage chain JSON
    result_json = analyzer.get_lineage_chain_json(sql_query_3, "downstream")
    
    # Save JSON to file
    json_filename = "output/trino_query_3_lineage_chain.json"
    with open(json_filename, 'w') as f:
        json.dump(json.loads(result_json), f, indent=2)
    
    print(f"✅ Lineage JSON saved: {json_filename}")
    
    # Generate JPEG visualization
    try:
        visualizer = SQLLineageVisualizer()
        output_path = "output/trino_query_3_lineage_visualization"
        jpeg_file = visualizer.create_lineage_chain_diagram(
            lineage_chain_json=result_json,
            output_path=output_path,
            output_format="jpeg",
            layout="horizontal"
        )
        print(f"✅ JPEG visualization saved: {jpeg_file}")
    except Exception as e:
        print(f"❌ JPEG generation failed: {e}")
    
    # Print summary
    parsed = json.loads(result_json)
    print(f"\n📈 Summary:")
    print(f"   Tables: {parsed['summary']['total_tables']}")
    print(f"   Columns: {parsed['summary']['total_columns']}")
    print(f"   Has Transformations: {parsed['summary']['has_transformations']}")
    print()


def main():
    """Run all three query tests."""
    print("🚀 SQL Lineage Analysis - Three Trino Queries Test")
    print("=" * 70)
    print("Testing three complex Trino queries with CTEs and JOINs")
    print()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Test all three queries
    test_query_1()
    test_query_2()
    test_query_3()
    
    print("🎉 All tests completed!")
    print("📁 Check the 'output/' directory for generated files:")
    print("   - trino_query_1_lineage_chain.json")
    print("   - trino_query_1_lineage_visualization.jpeg")
    print("   - trino_query_2_lineage_chain.json") 
    print("   - trino_query_2_lineage_visualization.jpeg")
    print("   - trino_query_3_lineage_chain.json")
    print("   - trino_query_3_lineage_visualization.jpeg")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())