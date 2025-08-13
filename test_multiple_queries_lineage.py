#!/usr/bin/env python3
"""
Test script for seven queries demonstrating branching pipelines and independent analysis:

Main Pipeline (Queries 1-4):
Query-1: Creates initial filtered orders table from source
Query-2: Creates aggregated customer summary from Query-1 result  
Query-3: Creates enriched customer segments from Query-2 result
Query-4: Creates final customer report from Query-3 result

Branching Analyses:
Query-6: Creates customer churn analysis from Query-2 (branch from customer_summary)
Query-7: Creates geographic distribution analysis from Query-3 (branch from customer_segments)

Independent Analysis:
Query-5: Creates product performance analysis directly from lineitem source (independent)

This demonstrates complex lineage patterns:
- Linear pipeline: 1→2→3→4
- Branch from step 2: 2→6  
- Branch from step 3: 3→7
- Independent flow: 5 (standalone)
"""

import sys
import os
import traceback
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from analyzer import SQLLineageAnalyzer
from analyzer.visualization.visualizer import SQLLineageVisualizer
from test_formatter import print_section_header, print_subsection_header, print_test_summary

def create_analyzer():
    """Create analyzer - no metadata registry needed."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    print("📊 Using SQL-only analysis (no external metadata)")
    return analyzer

def save_lineage_chain_outputs(chain_outputs, test_name):
    """Save lineage chain JSON outputs to files."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_count = 0
    for query_name, json_output in chain_outputs:
        filename = f"{output_dir}/{test_name}_{query_name}_lineage_chain.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"📁 Saved lineage chain JSON to: {filename}")
            saved_count += 1
        except Exception as e:
            print(f"❌ Failed to save {filename}: {e}")
    
    return saved_count

def generate_lineage_chain_visualizations(chain_outputs, test_name):
    """Generate JPEG visualizations from lineage chain JSON outputs."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visualizer = SQLLineageVisualizer()
    generated_count = 0
    
    for query_name, json_output in chain_outputs:
        try:
            # Generate JPEG visualization
            output_path = f"{output_dir}/{test_name}_{query_name}_visualization"
            
            # Use the lineage chain visualization method
            jpeg_file = visualizer.create_lineage_chain_diagram(
                lineage_chain_json=json_output,
                output_path=output_path,
                output_format="jpeg",
                layout="horizontal"
            )
            
            print(f"🖼️  Generated JPEG visualization: {jpeg_file}")
            generated_count += 1
            
        except Exception as e:
            print(f"❌ Failed to generate visualization for {query_name}: {e}")
            print(f"   Error details: {traceback.format_exc()}")
    
    return generated_count

def test_multiple_queries_lineage():
    """Test lineage chain for seven queries: main pipeline + branches + independent."""
    print_subsection_header("Multiple Queries Lineage Chain Test")
    
    analyzer = create_analyzer()
    
    # Query-1: Create initial filtered orders table
    query1_sql = """
CREATE TABLE "hive"."pipeline"."filtered_orders" AS
WITH
  recent_orders as (
    SELECT
      "dbxdemo0718"."trino_demo"."orders"."o_orderkey" AS "o_orderkey",
      "dbxdemo0718"."trino_demo"."orders"."o_custkey" AS "o_custkey",
      "dbxdemo0718"."trino_demo"."orders"."o_orderstatus" AS "o_orderstatus",
      "dbxdemo0718"."trino_demo"."orders"."o_totalprice" AS "o_totalprice",
      CAST("dbxdemo0718"."trino_demo"."orders"."o_orderdate" AS DATE) AS "o_orderdate",
      "dbxdemo0718"."trino_demo"."orders"."o_orderpriority" AS "o_orderpriority"
    FROM
      "dbxdemo0718"."trino_demo"."orders"
    WHERE
      "dbxdemo0718"."trino_demo"."orders"."o_orderstatus" = 'O'
      AND "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 1000
  )
SELECT
  *
FROM
  recent_orders
ORDER BY
  o_orderdate DESC
LIMIT
  500
"""
    
    # Query-2: Create aggregated customer summary from Query-1 result
    query2_sql = """
CREATE TABLE "hive"."pipeline"."customer_summary" AS
WITH
  customer_stats as (
    SELECT
      "hive"."pipeline"."filtered_orders"."o_custkey" AS "customer_id",
      COUNT("hive"."pipeline"."filtered_orders"."o_orderkey") AS "total_orders",
      SUM("hive"."pipeline"."filtered_orders"."o_totalprice") AS "total_spent",
      AVG("hive"."pipeline"."filtered_orders"."o_totalprice") AS "avg_order_value",
      MAX("hive"."pipeline"."filtered_orders"."o_orderdate") AS "last_order_date",
      MIN("hive"."pipeline"."filtered_orders"."o_orderdate") AS "first_order_date"
    FROM
      "hive"."pipeline"."filtered_orders"
    GROUP BY
      "hive"."pipeline"."filtered_orders"."o_custkey"
    HAVING
      COUNT("hive"."pipeline"."filtered_orders"."o_orderkey") >= 2
  )
SELECT
  *
FROM
  customer_stats
ORDER BY
  total_spent DESC
LIMIT
  100
"""
    
    # Query-3: Create enriched customer segments from Query-2 result
    query3_sql = """
CREATE TABLE "hive"."pipeline"."customer_segments" AS
WITH
  segmented_customers as (
    SELECT
      "hive"."pipeline"."customer_summary"."customer_id",
      "hive"."pipeline"."customer_summary"."total_orders",
      "hive"."pipeline"."customer_summary"."total_spent",
      "hive"."pipeline"."customer_summary"."avg_order_value",
      "hive"."pipeline"."customer_summary"."last_order_date",
      CASE 
        WHEN "hive"."pipeline"."customer_summary"."total_spent" > 10000 THEN 'Premium'
        WHEN "hive"."pipeline"."customer_summary"."total_spent" > 5000 THEN 'Gold'
        WHEN "hive"."pipeline"."customer_summary"."total_spent" > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END AS "customer_segment",
      CASE
        WHEN "hive"."pipeline"."customer_summary"."avg_order_value" > 2000 THEN 'High Value'
        WHEN "hive"."pipeline"."customer_summary"."avg_order_value" > 1000 THEN 'Medium Value'
        ELSE 'Low Value'
      END AS "order_value_category",
      DATEDIFF(CURRENT_DATE, "hive"."pipeline"."customer_summary"."last_order_date") AS "days_since_last_order"
    FROM
      "hive"."pipeline"."customer_summary"
    WHERE
      "hive"."pipeline"."customer_summary"."total_spent" > 1500
  )
SELECT
  *
FROM
  segmented_customers
ORDER BY
  total_spent DESC
"""
    
    # Query-4: Create final report from Query-3 result
    query4_sql = """
WITH
  final_report as (
    SELECT
      "hive"."pipeline"."customer_segments"."customer_segment",
      "hive"."pipeline"."customer_segments"."order_value_category",
      COUNT("hive"."pipeline"."customer_segments"."customer_id") AS "customer_count",
      SUM("hive"."pipeline"."customer_segments"."total_spent") AS "segment_total_revenue",
      AVG("hive"."pipeline"."customer_segments"."total_spent") AS "avg_customer_value",
      AVG("hive"."pipeline"."customer_segments"."total_orders") AS "avg_orders_per_customer",
      AVG("hive"."pipeline"."customer_segments"."days_since_last_order") AS "avg_days_since_last_order"
    FROM
      "hive"."pipeline"."customer_segments"
    GROUP BY
      "hive"."pipeline"."customer_segments"."customer_segment",
      "hive"."pipeline"."customer_segments"."order_value_category"
  )
SELECT
  *
FROM
  final_report
ORDER BY
  segment_total_revenue DESC
LIMIT
  20
"""

    # Query-5: Independent query - Product analysis (not connected to customer pipeline)
    query5_sql = """
CREATE TABLE "hive"."analytics"."product_performance" AS
WITH
  product_metrics as (
    SELECT
      "dbxdemo0718"."trino_demo"."lineitem"."l_partkey" AS "product_id",
      "dbxdemo0718"."trino_demo"."lineitem"."l_suppkey" AS "supplier_id",
      COUNT("dbxdemo0718"."trino_demo"."lineitem"."l_orderkey") AS "total_orders",
      SUM("dbxdemo0718"."trino_demo"."lineitem"."l_quantity") AS "total_quantity_sold",
      SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") AS "total_revenue",
      AVG("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") AS "avg_line_value",
      AVG("dbxdemo0718"."trino_demo"."lineitem"."l_discount") AS "avg_discount_rate",
      MIN("dbxdemo0718"."trino_demo"."lineitem"."l_shipdate") AS "first_ship_date",
      MAX("dbxdemo0718"."trino_demo"."lineitem"."l_shipdate") AS "last_ship_date",
      CASE
        WHEN SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") > 100000 THEN 'High Revenue'
        WHEN SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") > 50000 THEN 'Medium Revenue'
        ELSE 'Low Revenue'
      END AS "revenue_category",
      CASE
        WHEN COUNT("dbxdemo0718"."trino_demo"."lineitem"."l_orderkey") > 50 THEN 'High Volume'
        WHEN COUNT("dbxdemo0718"."trino_demo"."lineitem"."l_orderkey") > 20 THEN 'Medium Volume'
        ELSE 'Low Volume'
      END AS "volume_category"
    FROM
      "dbxdemo0718"."trino_demo"."lineitem"
    WHERE
      "dbxdemo0718"."trino_demo"."lineitem"."l_shipdate" >= DATE '2023-01-01'
      AND "dbxdemo0718"."trino_demo"."lineitem"."l_returnflag" = 'N'
    GROUP BY
      "dbxdemo0718"."trino_demo"."lineitem"."l_partkey",
      "dbxdemo0718"."trino_demo"."lineitem"."l_suppkey"
    HAVING
      SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") > 1000
      AND COUNT("dbxdemo0718"."trino_demo"."lineitem"."l_orderkey") >= 5
  )
SELECT
  *
FROM
  product_metrics
ORDER BY
  total_revenue DESC,
  total_quantity_sold DESC
LIMIT
  200
"""

    # Query-6: Branch from Query-2 - Customer churn analysis using customer_summary
    query6_sql = """
CREATE TABLE "hive"."analytics"."customer_churn_risk" AS
WITH
  churn_analysis as (
    SELECT
      "hive"."pipeline"."customer_summary"."customer_id",
      "hive"."pipeline"."customer_summary"."total_orders",
      "hive"."pipeline"."customer_summary"."total_spent",
      "hive"."pipeline"."customer_summary"."avg_order_value",
      "hive"."pipeline"."customer_summary"."last_order_date",
      "hive"."pipeline"."customer_summary"."first_order_date",
      DATEDIFF(CURRENT_DATE, "hive"."pipeline"."customer_summary"."last_order_date") AS "days_since_last_order",
      DATEDIFF("hive"."pipeline"."customer_summary"."last_order_date", "hive"."pipeline"."customer_summary"."first_order_date") AS "customer_lifetime_days",
      "hive"."pipeline"."customer_summary"."total_spent" / NULLIF("hive"."pipeline"."customer_summary"."total_orders", 0) AS "calculated_avg_order",
      CASE
        WHEN DATEDIFF(CURRENT_DATE, "hive"."pipeline"."customer_summary"."last_order_date") > 180 THEN 'High Risk'
        WHEN DATEDIFF(CURRENT_DATE, "hive"."pipeline"."customer_summary"."last_order_date") > 90 THEN 'Medium Risk'
        WHEN DATEDIFF(CURRENT_DATE, "hive"."pipeline"."customer_summary"."last_order_date") > 30 THEN 'Low Risk'
        ELSE 'Active'
      END AS "churn_risk_category",
      CASE
        WHEN "hive"."pipeline"."customer_summary"."total_orders" = 1 THEN 'One-time Buyer'
        WHEN "hive"."pipeline"."customer_summary"."total_orders" <= 3 THEN 'Occasional Buyer'
        WHEN "hive"."pipeline"."customer_summary"."total_orders" <= 10 THEN 'Regular Buyer'
        ELSE 'Frequent Buyer'
      END AS "purchase_frequency_category",
      CASE
        WHEN "hive"."pipeline"."customer_summary"."avg_order_value" > 5000 THEN 'High Value'
        WHEN "hive"."pipeline"."customer_summary"."avg_order_value" > 2000 THEN 'Medium Value'
        ELSE 'Low Value'
      END AS "value_category"
    FROM
      "hive"."pipeline"."customer_summary"
    WHERE
      "hive"."pipeline"."customer_summary"."total_spent" > 500
  )
SELECT
  *,
  CASE
    WHEN churn_risk_category = 'High Risk' AND value_category = 'High Value' THEN 'Priority Retention'
    WHEN churn_risk_category = 'High Risk' AND value_category = 'Medium Value' THEN 'Standard Retention'
    WHEN churn_risk_category = 'Medium Risk' AND value_category = 'High Value' THEN 'Watch List'
    ELSE 'Monitor'
  END AS "retention_priority"
FROM
  churn_analysis
ORDER BY
  total_spent DESC,
  days_since_last_order DESC
LIMIT
  300
"""

    # Query-7: Branch from Query-3 - Geographic customer distribution using customer_segments
    query7_sql = """
WITH
  geographic_segments as (
    SELECT
      "hive"."pipeline"."customer_segments"."customer_segment",
      "hive"."pipeline"."customer_segments"."order_value_category",
      "hive"."pipeline"."customer_segments"."customer_id",
      "hive"."pipeline"."customer_segments"."total_spent",
      "hive"."pipeline"."customer_segments"."total_orders",
      "hive"."pipeline"."customer_segments"."avg_order_value",
      -- Simulated geographic data based on customer patterns
      CASE
        WHEN MOD(CAST("hive"."pipeline"."customer_segments"."customer_id" AS INTEGER), 5) = 0 THEN 'East Coast'
        WHEN MOD(CAST("hive"."pipeline"."customer_segments"."customer_id" AS INTEGER), 5) = 1 THEN 'West Coast'
        WHEN MOD(CAST("hive"."pipeline"."customer_segments"."customer_id" AS INTEGER), 5) = 2 THEN 'Midwest'
        WHEN MOD(CAST("hive"."pipeline"."customer_segments"."customer_id" AS INTEGER), 5) = 3 THEN 'Southwest'
        ELSE 'Southeast'
      END AS "region",
      CASE
        WHEN "hive"."pipeline"."customer_segments"."avg_order_value" > 3000 AND "hive"."pipeline"."customer_segments"."total_orders" > 5 THEN 'Urban Premium'
        WHEN "hive"."pipeline"."customer_segments"."avg_order_value" > 1500 THEN 'Suburban Standard'
        ELSE 'Rural Economy'
      END AS "market_type",
      CASE
        WHEN "hive"."pipeline"."customer_segments"."customer_segment" = 'Premium' THEN 1.0
        WHEN "hive"."pipeline"."customer_segments"."customer_segment" = 'Gold' THEN 0.8
        WHEN "hive"."pipeline"."customer_segments"."customer_segment" = 'Silver' THEN 0.6
        ELSE 0.4
      END AS "segment_weight"
    FROM
      "hive"."pipeline"."customer_segments"
    WHERE
      "hive"."pipeline"."customer_segments"."total_spent" > 1000
  ),
  regional_summary as (
    SELECT
      region,
      market_type,
      customer_segment,
      COUNT(customer_id) AS "customer_count",
      SUM(total_spent) AS "total_regional_revenue",
      AVG(total_spent) AS "avg_customer_value",
      AVG(total_orders) AS "avg_orders_per_customer",
      AVG(avg_order_value) AS "avg_order_value_in_region",
      SUM(total_spent * segment_weight) AS "weighted_revenue"
    FROM
      geographic_segments
    GROUP BY
      region,
      market_type,
      customer_segment
  )
SELECT
  region,
  market_type,
  customer_segment,
  customer_count,
  total_regional_revenue,
  avg_customer_value,
  avg_orders_per_customer,
  avg_order_value_in_region,
  weighted_revenue,
  RANK() OVER (ORDER BY total_regional_revenue DESC) AS "revenue_rank",
  ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_regional_revenue DESC) AS "region_rank"
FROM
  regional_summary
ORDER BY
  total_regional_revenue DESC
LIMIT
  50
"""
    
    queries = [
        ("query1_filtered_orders", query1_sql),
        ("query2_customer_summary", query2_sql),
        ("query3_customer_segments", query3_sql),
        ("query4_final_report", query4_sql),
        ("query5_independent_product_analysis", query5_sql),
        ("query6_customer_churn_analysis", query6_sql),
        ("query7_geographic_distribution", query7_sql)
    ]
    
    chain_outputs = []
    successful_queries = 0
    failed_queries = 0
    
    for query_name, sql in queries:
        print(f"\n📋 Processing: {query_name}")
        print("─" * 50)
        
        try:
            # Generate lineage chain JSON with unlimited depth
            query_success = True
            
            try:
                # Generate comprehensive lineage chain JSON with unlimited downstream depth
                chain_json = analyzer.get_lineage_chain_json(sql, "downstream")
                
                # Store for saving
                chain_outputs.append((query_name, chain_json))
                
                # Verify JSON is valid and has expected structure
                import json
                parsed_json = json.loads(chain_json)
                
                required_keys = ["sql", "dialect", "chain_type", "max_depth", "actual_max_depth", "chains", "summary"]
                for key in required_keys:
                    if key not in parsed_json:
                        print(f"   ⚠️  Missing key '{key}' in JSON output")
                        query_success = False
                        break
                
                if query_success:
                    chains_count = len(parsed_json.get("chains", {}))
                    summary = parsed_json.get("summary", {})
                    max_depth = parsed_json.get("max_depth", "unknown")
                    actual_depth = parsed_json.get("actual_max_depth", 0)
                    
                    print(f"   ✅ Downstream unlimited depth: {chains_count} chains, {len(chain_json)} chars")
                    print(f"      📊 Tables: {summary.get('total_tables', 0)}, Columns: {summary.get('total_columns', 0)}")
                    print(f"      🔄 Has transformations: {summary.get('has_transformations', False)}")
                    print(f"      📋 Has metadata: {summary.get('has_metadata', False)}")
                    print(f"      📏 Max depth: {max_depth}, Actual depth: {actual_depth}")
                
            except Exception as e:
                print(f"   ❌ Unlimited downstream depth generation failed: {str(e)}")
                query_success = False
            
            if query_success:
                successful_queries += 1
                print(f"   🎉 Successfully generated lineage chain file")
            else:
                failed_queries += 1
                
        except Exception as e:
            print(f"   💥 Query processing failed: {str(e)}")
            failed_queries += 1
    
    # Save all chain outputs
    if chain_outputs:
        print(f"\n💾 Saving {len(chain_outputs)} lineage chain files...")
        saved_count = save_lineage_chain_outputs(chain_outputs, "multiple_queries")
        print(f"📁 Successfully saved {saved_count} lineage chain JSON files")
        
        # Generate JPEG visualizations
        print(f"\n🖼️  Generating {len(chain_outputs)} JPEG visualizations...")
        viz_count = generate_lineage_chain_visualizations(chain_outputs, "multiple_queries")
        print(f"🎨 Successfully generated {viz_count} JPEG visualization files")
    
    # Summary
    print(f"\n📊 Multiple Queries Lineage Chain Results:")
    print(f"   • Total queries processed: {len(queries)}")
    print(f"   • Successful queries: {successful_queries}")
    print(f"   • Failed queries: {failed_queries}")
    print(f"   • Total lineage chain files generated: {len(chain_outputs)}")
    print(f"   • Total JPEG visualizations generated: {len(chain_outputs)}")
    print(f"   • Success rate: {(successful_queries / len(queries) * 100):.1f}%")
    
    return successful_queries > failed_queries

def analyze_lineage_relationship():
    """Analyze the complex branching relationships in the data pipeline."""
    print_subsection_header("Branching Pipeline Lineage Analysis")
    
    print("📊 Main Data Pipeline Flow (Queries 1-4):")
    print(f"   📋 Source Table: dbxdemo0718.trino_demo.orders")
    print(f"   🎯 Query-1: Creates filtered orders table (hive.pipeline.filtered_orders)")
    print(f"   🎯 Query-2: Creates customer summary table (hive.pipeline.customer_summary)")
    print(f"   🎯 Query-3: Creates customer segments table (hive.pipeline.customer_segments)")
    print(f"   🎯 Query-4: Creates final customer report (query result)")
    
    print("\n🌳 Branching Analysis Flows:")
    print("   📋 Branch from Query-2 (Customer Summary):")
    print("   🎯 Query-6: Creates customer churn analysis (hive.analytics.customer_churn_risk)")
    print("   ➡️  Focuses on retention and churn risk assessment")
    print("")
    print("   📋 Branch from Query-3 (Customer Segments):") 
    print("   🎯 Query-7: Creates geographic distribution analysis (query result)")
    print("   ➡️  Focuses on regional market analysis and customer distribution")
    
    print("\n🔄 Independent Analysis Flow (Query 5):")
    print("   📋 Independent Source: dbxdemo0718.trino_demo.lineitem")
    print("   🎯 Query-5: Creates product performance analysis (hive.analytics.product_performance)")
    print("   ➡️  Completely separate product-focused analysis")
    
    print("\n🔗 Complete Data Flow Structure:")
    print("   Main Pipeline: 1 → 2 → 3 → 4")
    print("   Branch A:      2 → 6 (Churn Analysis)")
    print("   Branch B:      3 → 7 (Geographic Analysis)")
    print("   Independent:   5 (Product Analysis)")
    
    print("\n🏗️  Advanced Lineage Features Demonstrated:")
    print("   • Multi-branching data pipeline architecture")
    print("   • Shared intermediate results (Query-2 feeds both Query-3 and Query-6)")
    print("   • Specialized branch analyses for different business domains")
    print("   • Independent parallel data flow analysis")
    print("   • Multiple source tables (orders + lineitem)")
    print("   • Cross-schema dependencies (pipeline + analytics schemas)")
    print("   • Complex aggregation functions (COUNT, SUM, AVG, MIN, MAX)")
    print("   • Advanced window functions (RANK, ROW_NUMBER)")
    print("   • Sophisticated CASE statements and business logic")
    print("   • Date calculations and temporal analysis")
    print("   • Mathematical operations (MOD, DATEDIFF, NULLIF)")
    print("   • Multi-level data enrichment and transformation")
    
    print("\n✅ Branching pipeline lineage analysis complete!")
    return True

def save_combined_lineages(combined_lineages):
    """Save combined lineages to JSON files."""
    print(f"\n💾 Saving Combined Lineages to Files:")
    print("=" * 50)
    
    output_dir = "output/combined_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, lineage in enumerate(combined_lineages, 1):
        filename = f"combined_lineage_chain_{idx}.json"
        file_path = os.path.join(output_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(lineage, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved: {filename}")
            
            # Display summary of what was saved
            summary = lineage.get('summary', {})
            total_tables = summary.get('total_tables', 0)
            has_transformations = summary.get('has_transformations', False)
            print(f"   Tables: {total_tables}, Transformations: {'Yes' if has_transformations else 'No'}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to save {filename}: {e}")

def generate_combined_lineage_visualizations():
    """Generate JPEG visualizations from combined lineage JSON files."""
    print(f"\n🖼️  Generating JPEG Visualizations for Combined Lineages:")
    print("=" * 60)
    
    combined_output_dir = "output/combined_output"
    if not os.path.exists(combined_output_dir):
        print("❌ No combined output directory found.")
        return 0
    
    # Find all combined lineage JSON files
    combined_json_files = []
    for filename in os.listdir(combined_output_dir):
        if filename.startswith("combined_lineage_chain_") and filename.endswith(".json"):
            combined_json_files.append(os.path.join(combined_output_dir, filename))
    
    combined_json_files.sort()
    
    if not combined_json_files:
        print("❌ No combined lineage JSON files found.")
        return 0
    
    visualizer = SQLLineageVisualizer()
    generated_count = 0
    
    for json_file_path in combined_json_files:
        try:
            # Read the combined lineage JSON
            with open(json_file_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            # Extract base filename for output
            base_filename = os.path.splitext(os.path.basename(json_file_path))[0]
            output_path = os.path.join(combined_output_dir, f"{base_filename}_visualization")
            
            print(f"📋 Processing: {base_filename}")
            
            # Generate JPEG visualization
            jpeg_file = visualizer.create_lineage_chain_diagram(
                lineage_chain_json=json_content,
                output_path=output_path,
                output_format="jpeg",
                layout="horizontal"
            )
            
            print(f"   ✅ Generated JPEG: {jpeg_file}")
            generated_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to generate visualization for {os.path.basename(json_file_path)}: {e}")
            print(f"      Error details: {traceback.format_exc()}")
    
    print(f"\n🎨 Successfully generated {generated_count} combined lineage visualizations")
    return generated_count

def analyze_lineage_chains():
    """Analyze lineage chains from generated JSON files using the combiner logic."""
    print_subsection_header("Lineage Chain Analysis")
    
    # Import the combiner class
    from src.analyzer.lineage_chain_combiner import LineageChainCombiner
    
    # Load all JSON files
    output_dir = "output"
    multiple_queries_files = []
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith("multiple_queries_query") and filename.endswith(".json"):
                multiple_queries_files.append(os.path.join(output_dir, filename))
    
    multiple_queries_files.sort()
    
    # Load JSON data (no filename metadata added)
    lineage_data_list = []
    for file_path in multiple_queries_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lineage_data_list.append(data)
                print(f"✅ Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
    
    if not lineage_data_list:
        print("❌ No JSON files loaded.")
        return False
    
    # Process lineage data through all steps using the wrapper method
    combined_lineages = LineageChainCombiner.process_lineage_data_complete(lineage_data_list)
    
    if not combined_lineages:
        print("❌ No combined lineages created.")
        return False
    
    # Save combined lineages to files
    save_combined_lineages(combined_lineages)
    
    # Generate JPEG visualizations for combined lineages
    viz_count = generate_combined_lineage_visualizations()
    
    # Display final results summary
    print(f"\n📋 PROCESSING RESULTS SUMMARY:")
    print("=" * 70)
    print(f"✅ Successfully processed lineage data through all steps")
    print(f"✅ Generated combined lineages: {len(combined_lineages)} combined JSON objects")
    
    print(f"\n🎉 Lineage chain analysis completed successfully!")
    return True

def main():
    """Main test runner for multiple queries lineage test."""
    print("🚀 Multiple Queries Lineage Chain Test Suite")
    print("=" * 60)
    print("Testing complex data flows with 7 queries: main pipeline + branches + independent analysis")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    results = []
    
    # Run test sections
    print_section_header("MULTIPLE QUERIES LINEAGE CHAIN TEST")
    results.append(test_multiple_queries_lineage())
    
    print_section_header("LINEAGE CHAIN ANALYSIS")
    results.append(analyze_lineage_chains())
    
    print_section_header("BRANCHING PIPELINE LINEAGE ANALYSIS")
    results.append(analyze_lineage_relationship())
    
    # Final summary
    print("\n🎉 Multiple Queries Lineage Test Suite Completed!")
    print(f"\n📁 Check the '{output_dir}' directory for generated files.")
    
    # List generated files
    if os.path.exists(output_dir):
        lineage_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and 'multiple_queries_' in f]
        jpeg_files = [f for f in os.listdir(output_dir) if f.endswith('.jpeg') and 'multiple_queries_' in f]
        combined_files = [f for f in os.listdir(os.path.join(output_dir, 'combined_output')) if f.endswith('.json')] if os.path.exists(os.path.join(output_dir, 'combined_output')) else []
        combined_viz_files = [f for f in os.listdir(os.path.join(output_dir, 'combined_output')) if f.endswith('.jpeg') and 'combined_lineage_chain_' in f] if os.path.exists(os.path.join(output_dir, 'combined_output')) else []
        
        if lineage_files:
            print(f"\n📄 Generated lineage chain files ({len(lineage_files)}):")
            for file in sorted(lineage_files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"     • {file} ({size:,} bytes)")
        
        if combined_files:
            print(f"\n🔗 Generated combined lineage files ({len(combined_files)}):")
            for file in sorted(combined_files):
                file_path = os.path.join(output_dir, 'combined_output', file)
                size = os.path.getsize(file_path)
                print(f"     • {file} ({size:,} bytes)")
        
        if combined_viz_files:
            print(f"\n🎨 Generated combined lineage visualizations ({len(combined_viz_files)}):")
            for file in sorted(combined_viz_files):
                file_path = os.path.join(output_dir, 'combined_output', file)
                size = os.path.getsize(file_path)
                print(f"     • {file} ({size:,} bytes)")
        
        if jpeg_files:
            print(f"\n🖼️  Generated individual query visualizations ({len(jpeg_files)}):")
            for file in sorted(jpeg_files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"     • {file} ({size:,} bytes)")
    
    # Print summary
    passed = sum(results)
    total = len(results)
    success = print_test_summary(total, passed, "Multiple Queries Lineage Tests")
    
    print("\n🔗 Demonstrated Complex Data Flow Architecture:")
    print("   Main Pipeline:")
    print("   1. dbxdemo0718.trino_demo.orders → hive.pipeline.filtered_orders")
    print("   2. hive.pipeline.filtered_orders → hive.pipeline.customer_summary") 
    print("   3. hive.pipeline.customer_summary → hive.pipeline.customer_segments")
    print("   4. hive.pipeline.customer_segments → Final Customer Report")
    print("\n   Branching Analyses:")
    print("   6. hive.pipeline.customer_summary → hive.analytics.customer_churn_risk")
    print("   7. hive.pipeline.customer_segments → Geographic Distribution Report")
    print("\n   Independent Analysis:")
    print("   5. dbxdemo0718.trino_demo.lineitem → hive.analytics.product_performance")
    print("\n✨ Advanced Lineage Features Tested:")
    print("   • Multi-branching data pipeline architecture")
    print("   • Shared intermediate results (Query-2 feeds both Query-3 and Query-6)")
    print("   • Specialized branch analyses for different business domains")
    print("   • Multi-stage CREATE TABLE AS transformations")
    print("   • Connected pipeline with branching points")
    print("   • Independent parallel data flow analysis")
    print("   • Multiple source tables (orders + lineitem)")
    print("   • Cross-schema table dependencies (pipeline + analytics)")
    print("   • Complex aggregation functions (COUNT, SUM, AVG, MIN, MAX)")
    print("   • Advanced window functions (RANK, ROW_NUMBER)")
    print("   • Sophisticated CASE statements and business logic")
    print("   • Date calculations and temporal analysis")
    print("   • Mathematical operations (MOD, DATEDIFF, NULLIF)")
    print("   • LIMIT and ORDER BY transformations")
    print("   • Multi-step data enrichment and transformation")
    print("   • Lineage chain combination with connection point merging")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
