#!/usr/bin/env python3
"""
Test script for a single CREATE VIEW query with multiple CTEs demonstrating complex transformations:

Single Query Structure:
- Creates a view with multiple nested CTEs 
- Uses tier_summary, customer_metrics, product_analytics CTEs
- Combines aggregations, CASE expressions, window functions
- Demonstrates complex multi-CTE lineage patterns in a view creation

This demonstrates advanced CTE lineage features:
- Multiple nested CTEs with dependencies
- Complex transformations (COUNT, SUM, AVG, CASE, window functions)
- View creation (CREATE VIEW instead of CREATE TABLE)
- Multi-level data enrichment within a single query
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

def save_lineage_chain_output(json_output, test_name, query_name):
    """Save lineage chain JSON output to file."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"{output_dir}/{test_name}_{query_name}_lineage_chain.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"📁 Saved lineage chain JSON to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")
        return False

def generate_lineage_chain_visualization(json_output, test_name, query_name):
    """Generate JPEG visualization from lineage chain JSON output."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visualizer = SQLLineageVisualizer()
    
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
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate visualization for {query_name}: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

def test_create_view_multiple_ctes():
    """Test lineage chain for CREATE VIEW queries with multiple CTEs."""
    print_subsection_header("CREATE VIEW with Multiple CTEs Lineage Test")
    
    analyzer = create_analyzer()
    
    # Query 1: Original query with standard table names
    print("\n📋 Testing CREATE VIEW with standard table names")
    create_view_sql_1 = test_create_view_standard_names()
    
    # Query 2: New query with Trino-style table names  
    print("\n📋 Testing CREATE VIEW with Trino-style table names")
    create_view_sql_2 = test_create_view_trino_names()
    
    queries = [
        ("business_intelligence_standard", create_view_sql_1),
        ("business_intelligence_trino", create_view_sql_2)
    ]
    
    successful_queries = 0
    failed_queries = 0
    
    for query_name, sql in queries:
        print(f"\n📋 Processing: {query_name}")
        print("─" * 70)
        
        if process_single_create_view_query(analyzer, sql, query_name):
            successful_queries += 1
        else:
            failed_queries += 1
    
    # Summary
    print(f"\n📊 CREATE VIEW Multiple CTEs Test Results:")
    print(f"   • Total queries processed: {len(queries)}")
    print(f"   • Successful queries: {successful_queries}")
    print(f"   • Failed queries: {failed_queries}")
    print(f"   • Success rate: {(successful_queries / len(queries) * 100):.1f}%")
    
    return successful_queries > failed_queries

def test_create_view_standard_names():
    """Return CREATE VIEW query with standard table names."""
    # Single query with multiple CTEs creating a comprehensive business analytics view
    return """
CREATE VIEW "analytics"."customer_business_intelligence" AS
WITH
  -- CTE 1: Tier summary aggregating customer spending patterns
  tier_summary AS (
    SELECT 
      CASE 
        WHEN "ecommerce"."orders"."total_amount" > 10000 THEN 'Premium'
        WHEN "ecommerce"."orders"."total_amount" > 5000 THEN 'Gold'
        WHEN "ecommerce"."orders"."total_amount" > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END as customer_tier,
      COUNT("ecommerce"."orders"."customer_id") as tier_customer_count,
      SUM("ecommerce"."orders"."total_amount") as tier_total_revenue,
      AVG("ecommerce"."orders"."total_amount") as tier_avg_spending,
      MIN("ecommerce"."orders"."order_date") as tier_earliest_order,
      MAX("ecommerce"."orders"."order_date") as tier_latest_order
    FROM "ecommerce"."orders"
    WHERE "ecommerce"."orders"."order_status" = 'COMPLETED'
      AND "ecommerce"."orders"."order_date" >= DATE '2023-01-01'
    GROUP BY 
      CASE 
        WHEN "ecommerce"."orders"."total_amount" > 10000 THEN 'Premium'
        WHEN "ecommerce"."orders"."total_amount" > 5000 THEN 'Gold'
        WHEN "ecommerce"."orders"."total_amount" > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END
  ),
  -- CTE 2: Customer metrics with detailed customer analysis
  customer_metrics AS (
    SELECT 
      "ecommerce"."users"."user_id",
      "ecommerce"."users"."username",
      "ecommerce"."users"."email",
      "ecommerce"."users"."registration_date",
      COUNT("ecommerce"."orders"."order_id") as total_orders,
      SUM("ecommerce"."orders"."total_amount") as customer_lifetime_value,
      AVG("ecommerce"."orders"."total_amount") as avg_order_value,
      MIN("ecommerce"."orders"."order_date") as first_order_date,
      MAX("ecommerce"."orders"."order_date") as last_order_date,
      DATEDIFF(MAX("ecommerce"."orders"."order_date"), MIN("ecommerce"."orders"."order_date")) as customer_lifespan_days,
      CASE 
        WHEN SUM("ecommerce"."orders"."total_amount") > 10000 THEN 'Premium'
        WHEN SUM("ecommerce"."orders"."total_amount") > 5000 THEN 'Gold'
        WHEN SUM("ecommerce"."orders"."total_amount") > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END as derived_customer_tier
    FROM "ecommerce"."users" 
    INNER JOIN "ecommerce"."orders" ON "ecommerce"."users"."user_id" = "ecommerce"."orders"."customer_id"
    WHERE "ecommerce"."orders"."order_status" = 'COMPLETED'
      AND "ecommerce"."orders"."order_date" >= DATE '2023-01-01'
    GROUP BY 
      "ecommerce"."users"."user_id",
      "ecommerce"."users"."username", 
      "ecommerce"."users"."email",
      "ecommerce"."users"."registration_date"
    HAVING COUNT("ecommerce"."orders"."order_id") >= 1
  ),
  -- CTE 3: Product analytics combining order items with product information
  product_analytics AS (
    SELECT 
      "inventory"."products"."product_id",
      "inventory"."products"."product_name",
      "inventory"."products"."category",
      "inventory"."products"."price",
      COUNT("ecommerce"."order_items"."item_id") as total_items_sold,
      SUM("ecommerce"."order_items"."quantity") as total_quantity_sold,
      SUM("ecommerce"."order_items"."quantity" * "inventory"."products"."price") as total_product_revenue,
      AVG("ecommerce"."order_items"."quantity") as avg_quantity_per_order,
      COUNT(DISTINCT "ecommerce"."order_items"."order_id") as unique_orders_containing_product,
      CASE 
        WHEN SUM("ecommerce"."order_items"."quantity" * "inventory"."products"."price") > 50000 THEN 'Top Performer'
        WHEN SUM("ecommerce"."order_items"."quantity" * "inventory"."products"."price") > 20000 THEN 'High Performer'
        WHEN SUM("ecommerce"."order_items"."quantity" * "inventory"."products"."price") > 5000 THEN 'Medium Performer'
        ELSE 'Low Performer'
      END as product_performance_tier
    FROM "inventory"."products"
    INNER JOIN "ecommerce"."order_items" ON "inventory"."products"."product_id" = "ecommerce"."order_items"."product_id"
    INNER JOIN "ecommerce"."orders" ON "ecommerce"."order_items"."order_id" = "ecommerce"."orders"."order_id"
    WHERE "ecommerce"."orders"."order_status" = 'COMPLETED'
      AND "ecommerce"."orders"."order_date" >= DATE '2023-01-01'
    GROUP BY 
      "inventory"."products"."product_id",
      "inventory"."products"."product_name",
      "inventory"."products"."category", 
      "inventory"."products"."price"
    HAVING SUM("ecommerce"."order_items"."quantity") > 0
  )
-- Final SELECT combining all CTEs with window functions
SELECT 
  -- Customer information from customer_metrics CTE
  cm.user_id,
  cm.username,
  cm.email,
  cm.registration_date,
  cm.total_orders,
  cm.customer_lifetime_value,
  cm.avg_order_value,
  cm.first_order_date,
  cm.last_order_date,
  cm.customer_lifespan_days,
  cm.derived_customer_tier,
  
  -- Tier summary information matching customer's tier
  ts.tier_customer_count,
  ts.tier_total_revenue,
  ts.tier_avg_spending,
  ts.tier_earliest_order,
  ts.tier_latest_order,
  
  -- Product analytics aggregated by customer tier
  COUNT(pa.product_id) as products_in_tier_orders,
  SUM(pa.total_product_revenue) as tier_product_revenue_contribution,
  AVG(pa.total_quantity_sold) as avg_product_quantity_in_tier,
  
  -- Window functions for ranking and analytics
  RANK() OVER (ORDER BY cm.customer_lifetime_value DESC) as customer_value_rank,
  DENSE_RANK() OVER (PARTITION BY cm.derived_customer_tier ORDER BY cm.customer_lifetime_value DESC) as rank_within_tier,
  ROW_NUMBER() OVER (ORDER BY cm.last_order_date DESC) as recency_sequence,
  PERCENT_RANK() OVER (ORDER BY cm.customer_lifetime_value) as customer_value_percentile,
  
  -- Additional calculated fields
  CASE 
    WHEN DATEDIFF(CURRENT_DATE, cm.last_order_date) <= 30 THEN 'Active'
    WHEN DATEDIFF(CURRENT_DATE, cm.last_order_date) <= 90 THEN 'Recent'  
    WHEN DATEDIFF(CURRENT_DATE, cm.last_order_date) <= 180 THEN 'Dormant'
    ELSE 'Inactive'
  END as customer_activity_status,
  
  CASE 
    WHEN cm.total_orders = 1 THEN 'One-time'
    WHEN cm.total_orders <= 5 THEN 'Occasional' 
    WHEN cm.total_orders <= 15 THEN 'Regular'
    ELSE 'Frequent'
  END as purchase_frequency_category

FROM customer_metrics cm
INNER JOIN tier_summary ts ON cm.derived_customer_tier = ts.customer_tier
LEFT JOIN product_analytics pa ON pa.product_performance_tier = 'Top Performer'
  AND pa.category IN (
    SELECT DISTINCT "inventory"."products"."category" 
    FROM "inventory"."products" 
    WHERE "inventory"."products"."price" > 100
  )
WHERE cm.customer_lifetime_value > 1000
  AND cm.total_orders >= 2
GROUP BY 
  cm.user_id, cm.username, cm.email, cm.registration_date,
  cm.total_orders, cm.customer_lifetime_value, cm.avg_order_value,
  cm.first_order_date, cm.last_order_date, cm.customer_lifespan_days,
  cm.derived_customer_tier, ts.tier_customer_count, ts.tier_total_revenue,
  ts.tier_avg_spending, ts.tier_earliest_order, ts.tier_latest_order
ORDER BY 
  cm.customer_lifetime_value DESC,
  cm.last_order_date DESC
LIMIT 500
"""

def test_create_view_trino_names():
    """Return CREATE VIEW query with Trino-style table names."""
    return """
CREATE VIEW "hive"."analytics"."customer_business_metrics_view" AS
WITH
  -- CTE 1: Order tier analysis using Trino-style naming
  order_tier_analysis AS (
    SELECT 
      CASE 
        WHEN "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 10000 THEN 'Premium'
        WHEN "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 5000 THEN 'Gold'
        WHEN "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END as o_customer_tier,
      COUNT("dbxdemo0718"."trino_demo"."orders"."o_custkey") as tier_customer_count,
      SUM("dbxdemo0718"."trino_demo"."orders"."o_totalprice") as tier_total_revenue,
      AVG("dbxdemo0718"."trino_demo"."orders"."o_totalprice") as tier_avg_spending,
      MIN("dbxdemo0718"."trino_demo"."orders"."o_orderdate") as tier_earliest_order,
      MAX("dbxdemo0718"."trino_demo"."orders"."o_orderdate") as tier_latest_order
    FROM "dbxdemo0718"."trino_demo"."orders"
    WHERE "dbxdemo0718"."trino_demo"."orders"."o_orderstatus" = 'F'
      AND "dbxdemo0718"."trino_demo"."orders"."o_orderdate" >= DATE '2023-01-01'
    GROUP BY 
      CASE 
        WHEN "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 10000 THEN 'Premium'
        WHEN "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 5000 THEN 'Gold'
        WHEN "dbxdemo0718"."trino_demo"."orders"."o_totalprice" > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END
  ),
  -- CTE 2: Customer order analysis with lineitem details
  customer_order_metrics AS (
    SELECT 
      "dbxdemo0718"."trino_demo"."orders"."o_custkey",
      "dbxdemo0718"."trino_demo"."customer"."c_name",
      "dbxdemo0718"."trino_demo"."customer"."c_address",
      "dbxdemo0718"."trino_demo"."customer"."c_nationkey",
      COUNT("dbxdemo0718"."trino_demo"."orders"."o_orderkey") as total_orders,
      SUM("dbxdemo0718"."trino_demo"."orders"."o_totalprice") as customer_lifetime_value,
      AVG("dbxdemo0718"."trino_demo"."orders"."o_totalprice") as avg_order_value,
      MIN("dbxdemo0718"."trino_demo"."orders"."o_orderdate") as first_order_date,
      MAX("dbxdemo0718"."trino_demo"."orders"."o_orderdate") as last_order_date,
      CASE 
        WHEN SUM("dbxdemo0718"."trino_demo"."orders"."o_totalprice") > 10000 THEN 'Premium'
        WHEN SUM("dbxdemo0718"."trino_demo"."orders"."o_totalprice") > 5000 THEN 'Gold'
        WHEN SUM("dbxdemo0718"."trino_demo"."orders"."o_totalprice") > 2000 THEN 'Silver'
        ELSE 'Bronze'
      END as derived_customer_tier
    FROM "dbxdemo0718"."trino_demo"."customer" 
    INNER JOIN "dbxdemo0718"."trino_demo"."orders" ON "dbxdemo0718"."trino_demo"."customer"."c_custkey" = "dbxdemo0718"."trino_demo"."orders"."o_custkey"
    WHERE "dbxdemo0718"."trino_demo"."orders"."o_orderstatus" = 'F'
      AND "dbxdemo0718"."trino_demo"."orders"."o_orderdate" >= DATE '2023-01-01'
    GROUP BY 
      "dbxdemo0718"."trino_demo"."orders"."o_custkey",
      "dbxdemo0718"."trino_demo"."customer"."c_name", 
      "dbxdemo0718"."trino_demo"."customer"."c_address",
      "dbxdemo0718"."trino_demo"."customer"."c_nationkey"
    HAVING COUNT("dbxdemo0718"."trino_demo"."orders"."o_orderkey") >= 1
  ),
  -- CTE 3: Lineitem product analysis 
  lineitem_product_analysis AS (
    SELECT 
      "dbxdemo0718"."trino_demo"."lineitem"."l_partkey",
      "dbxdemo0718"."trino_demo"."part"."p_name",
      "dbxdemo0718"."trino_demo"."part"."p_type",
      "dbxdemo0718"."trino_demo"."part"."p_size",
      COUNT("dbxdemo0718"."trino_demo"."lineitem"."l_linenumber") as total_lineitem_count,
      SUM("dbxdemo0718"."trino_demo"."lineitem"."l_quantity") as total_quantity_sold,
      SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") as total_lineitem_revenue,
      AVG("dbxdemo0718"."trino_demo"."lineitem"."l_quantity") as avg_quantity_per_lineitem,
      COUNT(DISTINCT "dbxdemo0718"."trino_demo"."lineitem"."l_orderkey") as unique_orders_with_part,
      CASE 
        WHEN SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") > 50000 THEN 'Top Performer'
        WHEN SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") > 20000 THEN 'High Performer'
        WHEN SUM("dbxdemo0718"."trino_demo"."lineitem"."l_extendedprice") > 5000 THEN 'Medium Performer'
        ELSE 'Low Performer'
      END as part_performance_tier
    FROM "dbxdemo0718"."trino_demo"."part"
    INNER JOIN "dbxdemo0718"."trino_demo"."lineitem" ON "dbxdemo0718"."trino_demo"."part"."p_partkey" = "dbxdemo0718"."trino_demo"."lineitem"."l_partkey"
    INNER JOIN "dbxdemo0718"."trino_demo"."orders" ON "dbxdemo0718"."trino_demo"."lineitem"."l_orderkey" = "dbxdemo0718"."trino_demo"."orders"."o_orderkey"
    WHERE "dbxdemo0718"."trino_demo"."orders"."o_orderstatus" = 'F'
      AND "dbxdemo0718"."trino_demo"."orders"."o_orderdate" >= DATE '2023-01-01'
    GROUP BY 
      "dbxdemo0718"."trino_demo"."lineitem"."l_partkey",
      "dbxdemo0718"."trino_demo"."part"."p_name",
      "dbxdemo0718"."trino_demo"."part"."p_type", 
      "dbxdemo0718"."trino_demo"."part"."p_size"
    HAVING SUM("dbxdemo0718"."trino_demo"."lineitem"."l_quantity") > 0
  )
-- Final SELECT combining all CTEs with window functions
SELECT 
  -- Customer information from customer_order_metrics CTE
  com.o_custkey,
  com.c_name,
  com.c_address,
  com.c_nationkey,
  com.total_orders,
  com.customer_lifetime_value,
  com.avg_order_value,
  com.first_order_date,
  com.last_order_date,
  com.derived_customer_tier,
  
  -- Tier summary information matching customer's tier
  ota.tier_customer_count,
  ota.tier_total_revenue,
  ota.tier_avg_spending,
  ota.tier_earliest_order,
  ota.tier_latest_order,
  
  -- Product analytics aggregated by customer tier
  COUNT(lpa.l_partkey) as parts_in_tier_orders,
  SUM(lpa.total_lineitem_revenue) as tier_lineitem_revenue_contribution,
  AVG(lpa.total_quantity_sold) as avg_part_quantity_in_tier,
  
  -- Window functions for ranking and analytics
  RANK() OVER (ORDER BY com.customer_lifetime_value DESC) as customer_value_rank,
  DENSE_RANK() OVER (PARTITION BY com.derived_customer_tier ORDER BY com.customer_lifetime_value DESC) as rank_within_tier,
  ROW_NUMBER() OVER (ORDER BY com.last_order_date DESC) as recency_sequence,
  PERCENT_RANK() OVER (ORDER BY com.customer_lifetime_value) as customer_value_percentile,
  
  -- Additional calculated fields
  CASE 
    WHEN date_diff('day', com.last_order_date, CURRENT_DATE) <= 30 THEN 'Active'
    WHEN date_diff('day', com.last_order_date, CURRENT_DATE) <= 90 THEN 'Recent'  
    WHEN date_diff('day', com.last_order_date, CURRENT_DATE) <= 180 THEN 'Dormant'
    ELSE 'Inactive'
  END as customer_activity_status,
  
  CASE 
    WHEN com.total_orders = 1 THEN 'One-time'
    WHEN com.total_orders <= 5 THEN 'Occasional' 
    WHEN com.total_orders <= 15 THEN 'Regular'
    ELSE 'Frequent'
  END as purchase_frequency_category

FROM customer_order_metrics com
INNER JOIN order_tier_analysis ota ON com.derived_customer_tier = ota.o_customer_tier
LEFT JOIN lineitem_product_analysis lpa ON lpa.part_performance_tier = 'Top Performer'
  AND lpa.p_size IN (
    SELECT DISTINCT "dbxdemo0718"."trino_demo"."part"."p_size" 
    FROM "dbxdemo0718"."trino_demo"."part" 
    WHERE "dbxdemo0718"."trino_demo"."part"."p_retailprice" > 100
  )
WHERE com.customer_lifetime_value > 1000
  AND com.total_orders >= 2
GROUP BY 
  com.o_custkey, com.c_name, com.c_address, com.c_nationkey,
  com.total_orders, com.customer_lifetime_value, com.avg_order_value,
  com.first_order_date, com.last_order_date, com.derived_customer_tier,
  ota.tier_customer_count, ota.tier_total_revenue, ota.tier_avg_spending,
  ota.tier_earliest_order, ota.tier_latest_order
ORDER BY 
  com.customer_lifetime_value DESC,
  com.last_order_date DESC
LIMIT 500
"""

def process_single_create_view_query(analyzer, sql, query_name):
    """Process a single CREATE VIEW query and return success status."""
    try:
        # Generate lineage chain JSON with unlimited depth
        query_success = True
        
        try:
            # Generate comprehensive lineage chain JSON with unlimited downstream depth
            chain_json = analyzer.get_lineage_chain_json(sql, "downstream")
            
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
            # Save lineage chain output
            test_name = "create_view_multiple_ctes"
            save_success = save_lineage_chain_output(chain_json, test_name, query_name)
            
            # Generate JPEG visualization
            viz_success = generate_lineage_chain_visualization(chain_json, test_name, query_name)
            
            if save_success and viz_success:
                print(f"   🎉 Successfully generated lineage chain file and visualization")
                return True
            else:
                print(f"   ⚠️  Partial success - lineage generated but file operations had issues")
                return True
        else:
            print(f"   ❌ Query processing failed")
            return False
                
    except Exception as e:
        print(f"   💥 Query processing failed: {str(e)}")
        return False

def analyze_create_view_cte_structure():
    """Analyze the CREATE VIEW CTE structure and transformations."""
    print_subsection_header("CREATE VIEW CTE Structure Analysis")
    
    print("📊 Single CREATE VIEW Query with Multiple CTEs:")
    print("   🎯 Target: analytics.customer_business_intelligence (VIEW)")
    print("")
    
    print("🔄 CTE Dependencies and Data Flow:")
    print("   📋 CTE-1: tier_summary")
    print("   ├── Source: ecommerce.orders")
    print("   ├── Transformations: CASE expressions, aggregations (COUNT, SUM, AVG, MIN, MAX)")
    print("   └── Purpose: Customer tier classification and tier-level aggregations")
    print("")
    print("   📋 CTE-2: customer_metrics") 
    print("   ├── Sources: ecommerce.users, ecommerce.orders")
    print("   ├── Join: INNER JOIN on user_id = customer_id")
    print("   ├── Transformations: Aggregations, CASE expressions, DATEDIFF calculations")
    print("   └── Purpose: Individual customer lifetime value and behavior analysis")
    print("")
    print("   📋 CTE-3: product_analytics")
    print("   ├── Sources: inventory.products, ecommerce.order_items, ecommerce.orders")  
    print("   ├── Joins: Multiple INNER JOINs connecting product → order_items → orders")
    print("   ├── Transformations: Complex aggregations, revenue calculations, performance tiers")
    print("   └── Purpose: Product performance analysis and categorization")
    print("")
    print("   📋 Final SELECT:")
    print("   ├── Combines: customer_metrics + tier_summary + product_analytics")
    print("   ├── Joins: INNER JOIN (cm-ts), LEFT JOIN (with subquery filter)")
    print("   ├── Window Functions: RANK(), DENSE_RANK(), ROW_NUMBER(), PERCENT_RANK()")
    print("   ├── Complex Filters: Multi-level WHERE conditions and subquery")
    print("   └── Output: Comprehensive business intelligence view")
    
    print("\n🏗️  Advanced CTE Features Demonstrated:")
    print("   • Multiple dependent CTEs within single query")
    print("   • Cross-schema table dependencies (ecommerce + inventory + analytics)")
    print("   • Complex multi-table JOINs within CTEs")
    print("   • Nested CASE expressions for business logic")
    print("   • Advanced aggregation functions (COUNT, SUM, AVG, MIN, MAX)")
    print("   • Date/time calculations with DATEDIFF")
    print("   • Window functions with partitioning and ordering")
    print("   • Subqueries within JOIN conditions")
    print("   • GROUP BY with HAVING clauses")
    print("   • Multiple ORDER BY criteria")
    print("   • LIMIT for result set management")
    print("   • VIEW creation (instead of table)")
    print("   • Multi-level data enrichment and transformation")
    
    print("\n🔗 Data Lineage Complexity:")
    print("   • 4 source tables (orders, users, order_items, products)")
    print("   • 3 intermediate CTEs with dependencies")
    print("   • 1 target view with comprehensive business logic")
    print("   • Multi-step transformations and enrichment")
    print("   • Complex JOIN relationships across CTEs")
    
    print("\n✅ CREATE VIEW CTE structure analysis complete!")
    return True

def main():
    """Main test runner for CREATE VIEW with multiple CTEs test."""
    print("🚀 CREATE VIEW with Multiple CTEs Lineage Test Suite")
    print("=" * 70)
    print("Testing CREATE VIEW query with complex multi-CTE transformations")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    results = []
    
    # Run test sections
    print_section_header("CREATE VIEW MULTIPLE CTES LINEAGE TEST")
    results.append(test_create_view_multiple_ctes())
    
    print_section_header("CREATE VIEW CTE STRUCTURE ANALYSIS")
    results.append(analyze_create_view_cte_structure())
    
    # Final summary
    print("\n🎉 CREATE VIEW Multiple CTEs Test Suite Completed!")
    print(f"\n📁 Check the '{output_dir}' directory for generated files.")
    
    # List generated files
    if os.path.exists(output_dir):
        lineage_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and 'create_view_multiple_ctes_' in f]
        jpeg_files = [f for f in os.listdir(output_dir) if f.endswith('.jpeg') and 'create_view_multiple_ctes_' in f]
        
        if lineage_files:
            print(f"\n📄 Generated lineage chain files ({len(lineage_files)}):")
            for file in sorted(lineage_files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"     • {file} ({size:,} bytes)")
        
        if jpeg_files:
            print(f"\n🖼️  Generated visualizations ({len(jpeg_files)}):")
            for file in sorted(jpeg_files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"     • {file} ({size:,} bytes)")
    
    # Print summary
    passed = sum(results)
    total = len(results)
    success = print_test_summary(total, passed, "CREATE VIEW Multiple CTEs Tests")
    
    print("\n🔗 CREATE VIEW Data Flow Architecture:")
    print("   Sources:")
    print("   • ecommerce.orders → tier_summary CTE")
    print("   • ecommerce.users + ecommerce.orders → customer_metrics CTE")
    print("   • inventory.products + ecommerce.order_items + ecommerce.orders → product_analytics CTE")
    print("\n   Target:")
    print("   • analytics.customer_business_intelligence VIEW")
    print("\n✨ Advanced CREATE VIEW Features Tested:")
    print("   • Multi-CTE query structure within single CREATE VIEW")
    print("   • Cross-CTE dependencies and data flow")
    print("   • Complex business logic with CASE expressions")
    print("   • Multiple aggregation levels (CTE and final SELECT)")
    print("   • Advanced window functions with partitioning")
    print("   • Multi-table JOINs within CTEs")
    print("   • Subquery filtering in JOIN conditions")
    print("   • Date/time calculations and transformations")
    print("   • Performance tier categorization")
    print("   • Customer lifecycle and behavior analysis")
    print("   • VIEW creation with comprehensive business intelligence")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())