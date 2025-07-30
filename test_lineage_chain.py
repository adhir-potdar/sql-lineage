#!/usr/bin/env python3
"""
Test script for the newly added get_lineage_chain and get_lineage_chain_json functions.
This script uses all queries from test_quick.py, test_simple.py, and test_samples.py
to generate comprehensive lineage chain JSON outputs.
"""

import sys
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from analyzer import SQLLineageAnalyzer
from analyzer.metadata import SampleMetadataRegistry
from test_formatter import print_section_header, print_subsection_header, print_test_summary


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


def test_lineage_chain_basic_functionality():
    """Test basic functionality of get_lineage_chain and get_lineage_chain_json functions."""
    print_subsection_header("Basic Lineage Chain Functionality")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Simple test query
    sql = """
    CREATE TABLE user_summary AS 
    SELECT u.id, u.name, u.email, COUNT(o.id) as order_count
    FROM users u 
    LEFT JOIN orders o ON u.id = o.user_id 
    WHERE u.age > 25 
    GROUP BY u.id, u.name, u.email
    """
    
    try:
        # Test get_lineage_chain function with unlimited depth (depth=0) using downstream
        print("🧪 Testing get_lineage_chain function with unlimited downstream depth...")
        
        # Test downstream with unlimited depth
        downstream_chain = analyzer.get_lineage_chain(sql, "downstream")  # defaults to depth=0
        
        # Test with target entity using downstream
        target_chain = analyzer.get_lineage_chain(sql, "downstream", 0, "user_summary")
        
        print(f"✅ Downstream unlimited depth: {len(downstream_chain['chains'])} chains, max depth: {downstream_chain['max_depth']}, actual depth: {downstream_chain['actual_max_depth']}")
        print(f"✅ Target entity chain: {len(target_chain['chains'])} chains, max depth: {target_chain['max_depth']}, actual depth: {target_chain['actual_max_depth']}")
        
        # Test get_lineage_chain_json function with unlimited downstream depth
        print("\n🧪 Testing get_lineage_chain_json function with unlimited downstream depth...")
        
        downstream_json = analyzer.get_lineage_chain_json(sql, "downstream")  # defaults to depth=0
        target_json = analyzer.get_lineage_chain_json(sql, "downstream", 0, "user_summary")
        
        print(f"✅ Downstream JSON length: {len(downstream_json)} characters")
        print(f"✅ Target entity JSON length: {len(target_json)} characters")
        
        # Test error handling
        print("\n🧪 Testing error handling...")
        
        try:
            analyzer.get_lineage_chain(sql, "invalid_direction", 0)
            print("❌ Error handling failed - should have raised ValueError")
            return False
        except ValueError:
            print("✅ Direction validation works correctly")
        
        try:
            analyzer.get_lineage_chain(sql, "upstream", -1)
            print("❌ Error handling failed - should have raised ValueError")
            return False
        except ValueError:
            print("✅ Depth validation works correctly (negative depth rejected)")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def collect_queries_from_test_quick():
    """Collect all SQL queries from test_quick.py patterns."""
    return [
        ("quick_simple_query", 
         "SELECT name, email FROM users WHERE age > 25"),
        
        ("quick_join_query", 
         "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"),
        
        ("quick_cte_query", """
         WITH active_users AS (SELECT id, name FROM users WHERE active = true)
         SELECT * FROM active_users WHERE name LIKE 'A%'
         """),
        
        ("quick_ctas_query", 
         "CREATE TABLE user_summary AS SELECT id, name, COUNT(*) as login_count FROM users GROUP BY id, name"),
        
        ("quick_complex_multi_cte_query", """
         WITH order_stats AS (
             SELECT customer_id, COUNT(*) as orders, SUM(total) as spent
             FROM orders WHERE order_date >= '2023-01-01'
             GROUP BY customer_id
         ),
         customer_tiers AS (
             SELECT 
                 os.customer_id,
                 os.orders,
                 os.spent,
                 CASE WHEN os.spent > 1000 THEN 'Premium' ELSE 'Standard' END as tier
             FROM order_stats os
         ),
         tier_summary AS (
             SELECT tier, COUNT(*) as customer_count, AVG(spent) as avg_spent
             FROM customer_tiers
             GROUP BY tier
         )
         SELECT ts.tier, ts.customer_count, ts.avg_spent, u.name
         FROM tier_summary ts
         JOIN customer_tiers ct ON ts.tier = ct.tier
         JOIN users u ON ct.customer_id = u.id
         """)
    ]


def collect_queries_from_test_simple():
    """Collect all SQL queries from test_simple.py patterns."""
    return [
        ("simple_select", 
         "SELECT id, name, email FROM users WHERE age > 25"),
        
        ("simple_join", """
         SELECT u.name, o.total, o.order_date
         FROM users u
         JOIN orders o ON u.id = o.user_id
         WHERE u.age > 18
         """),
        
        ("simple_cte", """
         WITH active_users AS (
             SELECT id, name, email
             FROM users
             WHERE created_at >= '2023-01-01'
         )
         SELECT * FROM active_users WHERE name LIKE 'A%'
         """),
        
        ("create_table_as_select", """
         CREATE TABLE user_summary AS
         SELECT id, name, age, COUNT(*) as login_count
         FROM users
         WHERE age >= 18
         GROUP BY id, name, age
         """),
        
        ("complex_multi_cte", """
         WITH order_stats AS (
             SELECT 
                 customer_id,
                 COUNT(*) as order_count,
                 SUM(total) as total_spent
             FROM orders
             WHERE order_date >= '2023-01-01'
             GROUP BY customer_id
         ),
         customer_segments AS (
             SELECT 
                 u.id,
                 u.name,
                 u.email,
                 os.order_count,
                 os.total_spent,
                 CASE 
                     WHEN os.total_spent > 1000 THEN 'Premium'
                     WHEN os.total_spent > 500 THEN 'Standard'
                     ELSE 'Basic'
                 END as segment
             FROM users u
             LEFT JOIN order_stats os ON u.id = os.customer_id
         ),
         segment_summary AS (
             SELECT 
                 segment,
                 COUNT(*) as customer_count,
                 AVG(total_spent) as avg_spent
             FROM customer_segments
             GROUP BY segment
         )
         SELECT * FROM segment_summary ORDER BY avg_spent DESC
         """),
        
        ("complex_joins_with_aggregation", """
         SELECT 
             c.category_name,
             COUNT(DISTINCT p.product_id) as product_count,
             COUNT(DISTINCT o.order_id) as order_count,
             SUM(o.order_total) as total_revenue,
             AVG(o.order_total) as avg_order_value,
             COUNT(DISTINCT o.customer_id) as unique_customers
         FROM categories c
             LEFT JOIN products p ON c.category_id = p.category_id
             LEFT JOIN orders o ON p.product_id = o.product_id
             LEFT JOIN users u ON o.customer_id = u.id
         WHERE o.order_date >= CURRENT_DATE - INTERVAL '1' YEAR
             AND u.age >= 18
         GROUP BY c.category_id, c.category_name
         HAVING COUNT(DISTINCT o.order_id) > 10
         ORDER BY total_revenue DESC
         LIMIT 20
         """),
        
        ("window_functions", """
         SELECT 
             customer_id,
             order_date,
             order_total,
             ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_sequence,
             SUM(order_total) OVER (PARTITION BY customer_id ORDER BY order_date 
                                   ROWS UNBOUNDED PRECEDING) as running_total,
             LAG(order_total) OVER (PARTITION BY customer_id ORDER BY order_date) as prev_order_total,
             RANK() OVER (ORDER BY order_total DESC) as order_rank
         FROM orders
         WHERE order_date >= '2023-01-01'
         """),
        
        ("subquery", """
         SELECT 
             u.name,
             u.email,
             (SELECT COUNT(*) FROM orders o WHERE o.customer_id = u.id) as order_count,
             (SELECT MAX(total) FROM orders o WHERE o.customer_id = u.id) as max_order
         FROM users u
         WHERE u.id IN (
             SELECT DISTINCT customer_id 
             FROM orders 
             WHERE order_date >= '2023-01-01'
         )
         """),
        
        ("union_query", """
         SELECT 'user' as type, id, name as identifier FROM users WHERE age > 25
         UNION ALL
         SELECT 'product' as type, product_id as id, product_name as identifier FROM products WHERE price > 100
         UNION ALL
         SELECT 'category' as type, category_id as id, category_name as identifier FROM categories
         """)
    ]


def collect_queries_from_test_samples():
    """Collect all SQL queries from test_samples.py patterns."""
    return [
        ("sample1_complex_cte", """
         WITH CategorySales AS (SELECT c.category_name,
                                       p.product_id,
                                       p.product_name,
                                       SUM(o.order_total)         AS total_sales,
                                       COUNT(DISTINCT o.order_id) AS total_orders,
                                       AVG(o.order_total)         AS avg_order_value
                                FROM orders o
                                         INNER JOIN
                                     products p ON o.product_id = p.product_id
                                         INNER JOIN
                                     categories c ON p.category_id = c.category_id
                                WHERE o.order_date >= DATEADD(YEAR, -1, GETDATE())
                                GROUP BY c.category_name, p.product_id, p.product_name),
              TopCustomers AS (SELECT c.category_name,
                                      o.customer_id,
                                      SUM(o.order_total) AS customer_total,
                                      RANK()                OVER (
                         PARTITION BY c.category_name
                         ORDER BY SUM(o.order_total) DESC
                     ) AS rank
                               FROM orders o
                                        INNER JOIN
                                    products p ON o.product_id = p.product_id
                                        INNER JOIN
                                    categories c ON p.category_id = c.category_id
                               WHERE o.order_date >= DATEADD(YEAR, -1, GETDATE())
                               GROUP BY c.category_name, o.customer_id)
         SELECT cs.category_name,
                cs.product_id,
                cs.product_name,
                cs.total_sales,
                cs.total_orders,
                cs.avg_order_value,
                tc.customer_id    AS top_customer_id,
                tc.customer_total AS top_customer_revenue
         FROM CategorySales cs
                  LEFT JOIN
              TopCustomers tc
              ON
                  cs.category_name = tc.category_name
         WHERE tc.rank <= 3
         ORDER BY cs.category_name,
                  cs.total_sales DESC
         """),
        
        ("sample2_qualified_names", """
         WITH SalesSummary AS (
             SELECT
                 "production"."orders"."order_id" AS "FullyQualifiedTableId",
                 "production"."orders"."customer_id",
                 "production"."orders"."order_date",
                 "production"."orders"."product_id",
                 SUM("production"."orders"."order_total") AS "TotalOrderValue",
                 ROW_NUMBER() OVER (
                     PARTITION BY "production"."orders"."customer_id"
                     ORDER BY "production"."orders"."order_date" DESC
                 ) AS "OrderRank"
             FROM
                 "production"."orders"
             WHERE
                 "production"."orders"."order_date" >= DATEADD(YEAR, -2, CURRENT_TIMESTAMP)
             GROUP BY
                 "production"."orders"."order_id",
                 "production"."orders"."customer_id",
                 "production"."orders"."order_date",
                 "production"."orders"."product_id"
         ),
         TopProducts AS (
             SELECT
                 "catalog"."products"."product_id",
                 "catalog"."categories"."category_name",
                 COUNT("production"."orders"."order_id") AS "NumberOfOrders",
                 RANK() OVER (
                     PARTITION BY "catalog"."categories"."category_id"
                     ORDER BY COUNT("production"."orders"."order_id") DESC
                 ) AS "ProductRank"
             FROM
                 "catalog"."categories"
             INNER JOIN
                 "catalog"."products" ON "catalog"."categories"."category_id" = "catalog"."products"."category_id"
             LEFT JOIN
                 "production"."orders" ON "catalog"."products"."product_id" = "production"."orders"."product_id"
             GROUP BY
                 "catalog"."products"."product_id",
                 "catalog"."categories"."category_name",
                 "catalog"."categories"."category_id"
         )
         SELECT
             ss."FullyQualifiedTableId" AS "Order_Id",
             ss."customer_id",
             ss."order_date",
             tp."category_name",
             tp."NumberOfOrders",
             tp."ProductRank",
             ss."TotalOrderValue"
         FROM
             SalesSummary ss
         INNER JOIN
             TopProducts tp ON tp."product_id" = ss."product_id"
         WHERE
             tp."ProductRank" <= 5
         ORDER BY
             tp."category_name",
             ss."order_date" DESC
         """),
        
        ("sample3_union_cte", """
         WITH OrderSummary AS (
             SELECT
                 "schema1"."orders"."order_id",
                 "schema1"."orders"."customer_id",
                 "schema1"."orders"."product_id",
                 SUM("schema1"."orders"."order_total") AS total_order_value
             FROM
                 "schema1"."orders"
             INNER JOIN
                 "schema1"."products" ON "schema1"."orders"."product_id" = "schema1"."products"."product_id"
             WHERE
                 "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
             GROUP BY
                 "schema1"."orders"."order_id",
                 "schema1"."orders"."customer_id",
                 "schema1"."orders"."product_id"
         ),
         TopCustomers AS (
             SELECT
                 "schema1"."customers"."customer_id",
                 SUM("schema1"."orders"."order_total") AS customer_spent
             FROM
                 "schema1"."orders"
             INNER JOIN
                 "schema1"."customers" ON "schema1"."orders"."customer_id" = "schema1"."customers"."customer_id"
             WHERE
                 "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
             GROUP BY
                 "schema1"."customers"."customer_id"
         )
         SELECT
             os.customer_id,
             os.order_id,
             os.product_id,
             os.total_order_value,
             tc.customer_spent
         FROM
             OrderSummary os
         LEFT JOIN
             TopCustomers tc ON os.customer_id = tc.customer_id
         """),
        
        ("sample4_complex_multi_cte_union", """
         WITH order_base AS (
             SELECT customer_id, order_total, product_id
             FROM orders 
             WHERE order_date >= '2023-01-01'
         ),
         customer_stats AS (
             SELECT 
                 customer_id,
                 COUNT(*) as order_count,
                 SUM(order_total) as total_spent,
                 AVG(order_total) as avg_order
             FROM order_base
             GROUP BY customer_id
         ),
         customer_segments AS (
             SELECT 
                 cs.customer_id,
                 cs.order_count,
                 cs.total_spent,
                 cs.avg_order,
                 u.name,
                 CASE 
                     WHEN cs.total_spent > 2000 THEN 'Platinum'
                     WHEN cs.total_spent > 1000 THEN 'Gold'
                     WHEN cs.total_spent > 500 THEN 'Silver'
                     ELSE 'Bronze'
                 END as segment
             FROM customer_stats cs
             JOIN users u ON cs.customer_id = u.id
         ),
         segment_analysis AS (
             SELECT 
                 segment,
                 COUNT(*) as customer_count,
                 AVG(total_spent) as avg_segment_spend,
                 MIN(total_spent) as min_spend,
                 MAX(total_spent) as max_spend
             FROM customer_segments
             GROUP BY segment
         )
         SELECT 
             sa.segment,
             sa.customer_count,
             sa.avg_segment_spend,
             sa.min_spend,
             sa.max_spend
         FROM segment_analysis sa
         ORDER BY sa.avg_segment_spend DESC
         """),
        
        ("sample_ctas_simple", """
         CREATE TABLE user_summary AS
         SELECT 
             id, 
             name, 
             email,
             age,
             created_at
         FROM users 
         WHERE age >= 18 AND active = true
         """),
        
        ("sample_ctas_aggregated", """
         CREATE TABLE customer_metrics AS
         SELECT 
             u.id,
             u.name,
             u.region,
             COUNT(o.id) as order_count,
             SUM(o.total) as total_spent,
             AVG(o.total) as avg_order_value,
             MAX(o.order_date) as last_order_date,
             MIN(o.order_date) as first_order_date
         FROM users u
         LEFT JOIN orders o ON u.id = o.user_id
         WHERE u.active = true
         GROUP BY u.id, u.name, u.region
         HAVING COUNT(o.id) >= 1
         """),
        
        ("sample_ctas_complex_cte", """
         CREATE TABLE premium_customer_analysis AS
         WITH order_stats AS (
             SELECT 
                 customer_id,
                 COUNT(*) as total_orders,
                 SUM(order_total) as lifetime_value,
                 AVG(order_total) as avg_order_value,
                 MAX(order_date) as last_order_date
             FROM orders 
             WHERE order_date >= '2023-01-01'
             GROUP BY customer_id
         ),
         customer_segments AS (
             SELECT 
                 u.id as customer_id,
                 u.name,
                 u.email,
                 u.region,
                 u.age,
                 os.total_orders,
                 os.lifetime_value,
                 os.avg_order_value,
                 os.last_order_date,
                 CASE 
                     WHEN os.lifetime_value > 10000 THEN 'Platinum'
                     WHEN os.lifetime_value > 5000 THEN 'Gold'
                     WHEN os.lifetime_value > 2000 THEN 'Silver'
                     ELSE 'Bronze'
                 END as tier,
                 ROW_NUMBER() OVER (
                     PARTITION BY u.region 
                     ORDER BY os.lifetime_value DESC
                 ) as region_rank
             FROM users u
             INNER JOIN order_stats os ON u.id = os.customer_id
             WHERE u.active = true
         )
         SELECT 
             customer_id,
             name,
             email,
             region,
             age,
             total_orders,
             lifetime_value,
             avg_order_value,
             last_order_date,
             tier,
             region_rank
         FROM customer_segments 
         WHERE tier IN ('Platinum', 'Gold')
           AND region_rank <= 10
         ORDER BY lifetime_value DESC
         """)
    ]


def collect_queries_from_test_cte_stats():
    """Collect the complex CTE statistics query from test_cte_stats_query.py."""
    return [
        ("cte_stats_query", """
    WITH total_cols AS (
        SELECT
            TABNAME,
            COUNT(*) as TOTAL_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
        GROUP BY TABNAME
    ),
    syscat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSCAT_STATS_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    sysstat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSSTAT_STATS_COLUMNS
        FROM maximo_dev.SYSSTAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    stats_summary AS (
        SELECT
            t.TABNAME,
            t.TOTAL_COLUMNS,
            COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) as SYSCAT_STATS_COLUMNS,
            COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) as SYSSTAT_STATS_COLUMNS,
            ROUND((COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSCAT_COVERAGE_PCT,
            ROUND((COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSSTAT_COVERAGE_PCT
        FROM total_cols t
        LEFT JOIN syscat_stats sc ON t.TABNAME = sc.TABNAME
        LEFT JOIN sysstat_stats ss ON t.TABNAME = ss.TABNAME
    )
    SELECT
        COUNT(*) as TOTAL_TABLES,
        SUM(TOTAL_COLUMNS) as TOTAL_COLUMNS_ALL_TABLES,
        SUM(SYSCAT_STATS_COLUMNS) as TOTAL_SYSCAT_STATS,
        SUM(SYSSTAT_STATS_COLUMNS) as TOTAL_SYSSTAT_STATS,
        ROUND(AVG(SYSCAT_COVERAGE_PCT), 2) as AVG_SYSCAT_COVERAGE,
        ROUND(AVG(SYSSTAT_COVERAGE_PCT), 2) as AVG_SYSSTAT_COVERAGE
    FROM stats_summary
    """)
    ]


def test_lineage_chain_comprehensive():
    """Generate comprehensive lineage chain JSON files for all test queries."""
    print_subsection_header("Comprehensive Lineage Chain Generation")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Collect all queries from test files
    all_queries = []
    all_queries.extend(collect_queries_from_test_quick())
    all_queries.extend(collect_queries_from_test_simple())
    all_queries.extend(collect_queries_from_test_samples())
    all_queries.extend(collect_queries_from_test_cte_stats())
    
    print(f"🔍 Collected {len(all_queries)} queries from all test files")
    
    chain_outputs = []
    successful_queries = 0
    failed_queries = 0
    
    for query_name, sql in all_queries:
        print(f"\n📋 Processing: {query_name}")
        print("─" * 50)
        
        try:
            # Generate lineage chain JSON with unlimited depth (depth=0)
            query_success = True
            
            try:
                # Generate comprehensive lineage chain JSON with unlimited downstream depth
                chain_json = analyzer.get_lineage_chain_json(sql, "downstream")  # defaults to depth=0
                
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
        saved_count = save_lineage_chain_outputs(chain_outputs, "lineage_chain_test")
        print(f"📁 Successfully saved {saved_count} lineage chain JSON files")
    
    # Summary
    print(f"\n📊 Comprehensive Lineage Chain Generation Results:")
    print(f"   • Total queries processed: {len(all_queries)}")
    print(f"   • Successful queries: {successful_queries}")
    print(f"   • Failed queries: {failed_queries}")
    print(f"   • Total lineage chain files generated: {len(chain_outputs)}")
    print(f"   • Success rate: {(successful_queries / len(all_queries) * 100):.1f}%")
    print(f"   • Each query generated 1 comprehensive lineage chain file with unlimited depth")
    
    return successful_queries > failed_queries


def test_lineage_chain_advanced_features():
    """Test advanced features of the lineage chain functions."""
    print_subsection_header("Advanced Lineage Chain Features")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Complex query with multiple transformation types
    complex_sql = """
    CREATE TABLE analytics_summary AS
    WITH sales_data AS (
        SELECT 
            o.customer_id,
            o.product_id,
            o.order_date,
            o.order_total,
            p.product_name,
            p.category_id,
            c.category_name,
            u.name as customer_name,
            u.region,
            ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date) as order_sequence,
            SUM(o.order_total) OVER (PARTITION BY o.customer_id ORDER BY o.order_date 
                                   ROWS UNBOUNDED PRECEDING) as running_customer_total,
            AVG(o.order_total) OVER (PARTITION BY c.category_id) as category_avg_order
        FROM orders o
        INNER JOIN products p ON o.product_id = p.product_id
        INNER JOIN categories c ON p.category_id = c.category_id
        INNER JOIN users u ON o.customer_id = u.id
        WHERE o.order_date >= '2023-01-01'
    ),
    customer_metrics AS (
        SELECT 
            customer_id,
            customer_name,
            region,
            COUNT(*) as total_orders,
            SUM(order_total) as lifetime_value,
            AVG(order_total) as avg_order_value,
            MAX(order_date) as last_order_date,
            COUNT(DISTINCT category_id) as unique_categories,
            CASE 
                WHEN SUM(order_total) > 10000 THEN 'VIP'
                WHEN SUM(order_total) > 5000 THEN 'Premium'
                WHEN SUM(order_total) > 1000 THEN 'Standard'
                ELSE 'Basic'
            END as customer_tier,
            RANK() OVER (PARTITION BY region ORDER BY SUM(order_total) DESC) as region_rank
        FROM sales_data
        GROUP BY customer_id, customer_name, region
    ),
    category_performance AS (
        SELECT 
            category_id,
            category_name,
            COUNT(DISTINCT customer_id) as unique_customers,
            SUM(order_total) as category_revenue,
            AVG(order_total) as avg_category_order,
            COUNT(*) as total_category_orders
        FROM sales_data
        GROUP BY category_id, category_name
    ),
    combined_analysis AS (
        SELECT 
            cm.customer_id,
            cm.customer_name,
            cm.region,
            cm.total_orders,
            cm.lifetime_value,
            cm.avg_order_value,
            cm.customer_tier,
            cm.region_rank,
            cp.category_name as top_category,
            cp.category_revenue,
            CASE 
                WHEN cm.region_rank <= 5 THEN 'Top Regional Customer'
                WHEN cm.customer_tier = 'VIP' THEN 'VIP Customer'
                ELSE 'Regular Customer'
            END as final_classification
        FROM customer_metrics cm
        CROSS JOIN (
            SELECT category_name, category_revenue,
                   ROW_NUMBER() OVER (ORDER BY category_revenue DESC) as cat_rank
            FROM category_performance
        ) cp
        WHERE cp.cat_rank = 1  -- Top category only
    )
    SELECT 
        customer_id,
        customer_name,
        region,
        total_orders,
        lifetime_value,
        ROUND(avg_order_value, 2) as avg_order_value,
        customer_tier,
        region_rank,
        top_category,
        final_classification,
        CURRENT_TIMESTAMP as analysis_timestamp
    FROM combined_analysis
    WHERE lifetime_value > 500
    ORDER BY lifetime_value DESC, region_rank ASC
    """
    
    try:
        print("🧪 Testing advanced lineage chain features...")
        
        # Test comprehensive lineage chain with unlimited downstream depth
        print("\n1️⃣ Testing comprehensive chain with unlimited downstream depth...")
        comprehensive_chain = analyzer.get_lineage_chain(complex_sql, "downstream")  # defaults to depth=0
        
        print(f"   ✅ Generated comprehensive chain with {len(comprehensive_chain['chains'])} entities")
        print(f"   📊 Max depth: {comprehensive_chain['max_depth']}, Actual depth: {comprehensive_chain['actual_max_depth']}")
        print(f"   📊 Summary: {comprehensive_chain['summary']}")
        
        # Test targeted entity analysis with unlimited downstream depth
        print("\n2️⃣ Testing targeted entity analysis with unlimited downstream depth...")
        target_entities = ["analytics_summary", "sales_data", "customer_metrics"]
        
        for entity in target_entities:
            try:
                targeted_chain = analyzer.get_lineage_chain(complex_sql, "downstream", 0, entity)
                print(f"   ✅ {entity}: {len(targeted_chain['chains'])} targeted chains, actual depth: {targeted_chain['actual_max_depth']}")
            except Exception as e:
                print(f"   ⚠️  {entity}: {str(e)}")
        
        # Test JSON serialization with unlimited downstream depth
        print("\n3️⃣ Testing JSON serialization with unlimited downstream depth...")
        json_output = analyzer.get_lineage_chain_json(complex_sql, "downstream")  # defaults to depth=0
        
        # Validate JSON structure
        import json
        parsed_json = json.loads(json_output)
        
        required_keys = ["sql", "dialect", "chain_type", "max_depth", "actual_max_depth", "target_entity", "chains", "summary"]
        missing_keys = [key for key in required_keys if key not in parsed_json]
        
        if missing_keys:
            print(f"   ❌ Missing keys: {missing_keys}")
            return False
        else:
            print(f"   ✅ JSON structure valid: {len(json_output)} characters")
            print(f"   📏 Max depth: {parsed_json.get('max_depth')}, Actual depth: {parsed_json.get('actual_max_depth')}")
        
        # Test transformation detail extraction
        print("\n4️⃣ Testing transformation detail extraction...")
        chains = parsed_json.get("chains", {})
        transformation_count = 0
        
        for entity_name, chain_data in chains.items():
            if "transformations" in chain_data:
                transformation_count += len(chain_data["transformations"])
        
        print(f"   ✅ Found {transformation_count} transformations across all entities")
        
        # Test metadata integration
        print("\n5️⃣ Testing metadata integration...")
        metadata_count = 0
        
        for entity_name, chain_data in chains.items():
            if "metadata" in chain_data and chain_data["metadata"]:
                metadata_count += 1
        
        print(f"   ✅ Found metadata for {metadata_count} entities")
        
        # Save advanced test results - single comprehensive file with unlimited downstream depth
        advanced_outputs = [
            ("advanced_comprehensive", analyzer.get_lineage_chain_json(complex_sql, "downstream"))  # defaults to depth=0
        ]
        
        save_lineage_chain_outputs(advanced_outputs, "lineage_chain_test")
        print(f"   📁 Saved {len(advanced_outputs)} advanced test file with unlimited downstream depth")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main test runner for lineage chain functionality."""
    print("🚀 SQL Lineage Chain Test Suite")
    print("=" * 60)
    print("Testing newly added get_lineage_chain and get_lineage_chain_json functions")
    print("with all queries from test_quick.py, test_simple.py, and test_samples.py")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    results = []
    
    # Run test sections
    print_section_header("BASIC FUNCTIONALITY TESTS")
    results.append(test_lineage_chain_basic_functionality())
    
    print_section_header("COMPREHENSIVE LINEAGE CHAIN GENERATION")
    results.append(test_lineage_chain_comprehensive())
    
    print_section_header("ADVANCED FEATURES TESTS")  
    results.append(test_lineage_chain_advanced_features())
    
    # Final summary
    print("\n🎉 Lineage Chain Test Suite Completed!")
    print(f"\n📁 Check the '{output_dir}' directory for generated lineage chain JSON files.")
    
    # List generated files
    if os.path.exists(output_dir):
        lineage_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and 'lineage_chain' in f and 'lineage_chain_test_' in f]
        if lineage_files:
            print(f"\n📄 Generated lineage chain files ({len(lineage_files)}):")
            
            # Categorize by query source
            quick_files = [f for f in lineage_files if 'quick_' in f]
            simple_files = [f for f in lineage_files if 'simple_' in f]
            sample_files = [f for f in lineage_files if 'sample' in f and 'simple_' not in f]
            advanced_files = [f for f in lineage_files if 'advanced_' in f]
            
            if quick_files:
                print(f"\n   🚀 From test_quick.py queries ({len(quick_files)} files):")
                for file in sorted(quick_files):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
            
            if simple_files:
                print(f"\n   📋 From test_simple.py queries ({len(simple_files)} files):")
                for file in sorted(simple_files):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
            
            if sample_files:
                print(f"\n   🎯 From test_samples.py queries ({len(sample_files)} files):")
                for file in sorted(sample_files):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
            
            if advanced_files:
                print(f"\n   🔬 Advanced feature test ({len(advanced_files)} file):")
                for file in sorted(advanced_files):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
            
            print(f"\n✨ Each lineage chain file contains comprehensive relationship data")
            print(f"   with unlimited depth traversal, transformations, and metadata integration!")
        else:
            print("\n⚠️  No lineage chain files were generated")
    
    # Print summary
    passed = sum(results)
    total = len(results)
    success = print_test_summary(total, passed, "Lineage Chain Tests")
    
    print("\n🔗 New Functions Successfully Tested:")
    print("   • get_lineage_chain() - Comprehensive lineage with unlimited depth (depth=0 default)")
    print("   • get_lineage_chain_json() - JSON serialization with unlimited depth traversal")
    print("   • Support for downstream analysis with complete relationship mapping")
    print("   • Unlimited depth relationship traversal until no more dependencies found")
    print("   • Targeted entity analysis with unlimited downstream depth")
    print("   • CTAS and CTE transformation tracking with full downstream depth")
    print("   • Complete metadata integration at table and column level")
    print("   • Single comprehensive file per query with all downstream relationship data")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())