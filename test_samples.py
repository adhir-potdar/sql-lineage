#!/usr/bin/env python3
"""
Test script for running all original sample queries without pytest.
This script tests all the sample queries from the original folders to ensure compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from analyzer import SQLLineageAnalyzer
from analyzer.metadata import SampleMetadataRegistry
from analyzer.formatters import ConsoleFormatter
from analyzer.visualization import SQLLineageVisualizer
from test_formatter import print_lineage_analysis, print_section_header, print_subsection_header, print_test_summary


# Use the enhanced formatters instead of local ones

def save_json_outputs(json_outputs, test_name):
    """Save JSON outputs to files."""
    import os
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for query_name, json_output in json_outputs:
        filename = f"{output_dir}/{test_name}_{query_name}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"📁 Saved JSON output to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save {filename}: {e}")

def save_chain_outputs(chain_outputs, test_name):
    """Save chain JSON outputs to files."""
    import os
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for query_name, lineage_type, chain_type, depth, json_output in chain_outputs:
        filename = f"{output_dir}/{test_name}_{query_name}_{lineage_type}_chain_{chain_type}_depth{depth}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"📁 Saved {lineage_type} chain JSON to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save {lineage_type} chain {filename}: {e}")

def create_visualizations(analyzer, test_name):
    """Create visualization outputs for key sample queries."""
    try:
        visualizer = SQLLineageVisualizer()
    except ImportError as e:
        print(f"⚠️  Visualization not available: {e}")
        return
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Key sample queries for visualization
    visualization_queries = [
        ("sample1_complex_cte", """
        WITH order_summary AS (
            SELECT 
                customer_id,
                COUNT(*) as total_orders,
                SUM(order_total) as total_spent,
                AVG(order_total) as avg_order_value
            FROM orders 
            WHERE order_date >= '2023-01-01'
            GROUP BY customer_id
        ),
        customer_tiers AS (
            SELECT 
                os.customer_id,
                u.name as customer_name,
                os.total_orders,
                os.total_spent,
                os.avg_order_value,
                CASE 
                    WHEN os.total_spent > 5000 THEN 'Platinum'
                    WHEN os.total_spent > 2000 THEN 'Gold'
                    WHEN os.total_spent > 500 THEN 'Silver'
                    ELSE 'Bronze'
                END as tier
            FROM order_summary os
            JOIN users u ON os.customer_id = u.id
            WHERE u.active = true
        ),
        tier_analytics AS (
            SELECT 
                tier,
                COUNT(*) as customer_count,
                AVG(total_spent) as avg_lifetime_value,
                MIN(total_spent) as min_spent,
                MAX(total_spent) as max_spent
            FROM customer_tiers
            GROUP BY tier
        )
        SELECT 
            ta.tier,
            ta.customer_count,
            ta.avg_lifetime_value,
            ROUND(ta.customer_count * 100.0 / SUM(ta.customer_count) OVER (), 2) as percentage
        FROM tier_analytics ta
        ORDER BY ta.avg_lifetime_value DESC
        """),
        ("sample4_complex_multi_cte_union", """
        WITH regional_sales AS (
            SELECT 
                u.region,
                DATE_TRUNC('month', o.order_date) as month,
                SUM(o.order_total) as monthly_sales,
                COUNT(DISTINCT o.customer_id) as unique_customers
            FROM orders o
            JOIN users u ON o.customer_id = u.id
            WHERE o.order_date >= '2023-01-01'
            GROUP BY u.region, DATE_TRUNC('month', o.order_date)
        ),
        product_performance AS (
            SELECT 
                p.category,
                p.name as product_name,
                SUM(oi.quantity * oi.price) as product_revenue,
                COUNT(DISTINCT oi.order_id) as order_frequency
            FROM order_items oi
            JOIN products p ON oi.product_id = p.id
            JOIN orders o ON oi.order_id = o.id
            WHERE o.order_date >= '2023-01-01'
            GROUP BY p.category, p.name
        ),
        combined_metrics AS (
            SELECT 
                'Regional' as metric_type,
                region as dimension,
                month as time_period,
                monthly_sales as value,
                unique_customers as secondary_metric
            FROM regional_sales
            
            UNION ALL
            
            SELECT 
                'Product' as metric_type,
                category as dimension,
                DATE_TRUNC('month', CURRENT_DATE) as time_period,
                product_revenue as value,
                order_frequency as secondary_metric
            FROM product_performance
            WHERE product_revenue > 1000
        )
        SELECT 
            metric_type,
            dimension,
            time_period,
            SUM(value) as total_value,
            AVG(secondary_metric) as avg_secondary
        FROM combined_metrics
        GROUP BY metric_type, dimension, time_period
        ORDER BY total_value DESC
        """)
    ]
    
    print_section_header("Creating Sample Visualizations", 50)
    
    for query_name, sql in visualization_queries:
        print(f"\n📊 Creating visualizations for {query_name}...")
        
        try:
            # Create visualizations for both upstream and downstream
            for chain_type in ["upstream", "downstream"]:
                # Get chain data with depth 4 for complex samples
                table_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 4)
                column_json = analyzer.get_column_lineage_chain_json(sql, chain_type, 3)
                
                # Create table-only visualization (JPEG only)
                table_output = visualizer.create_table_only_diagram(
                    table_chain_json=table_json,
                    output_path=f"{output_dir}/{test_name}_{query_name}_{chain_type}_table",
                    output_format="jpeg",
                    layout="horizontal",
                    sql_query=sql
                )
                print(f"   ✅ {chain_type.title()} table diagram: {os.path.basename(table_output)}")
                
                # Create integrated table + column visualization (JPEG only)
                integrated_output = visualizer.create_lineage_diagram(
                    table_chain_json=table_json,
                    column_chain_json=column_json,
                    output_path=f"{output_dir}/{test_name}_{query_name}_{chain_type}_integrated",
                    output_format="jpeg",
                    show_columns=True,
                    layout="horizontal",
                    sql_query=sql
                )
                print(f"   ✅ {chain_type.title()} integrated diagram: {os.path.basename(integrated_output)}")
                
        except Exception as e:
            print(f"   ❌ Failed to create visualizations for {query_name}: {e}")
    
    print("\n🎨 Sample visualization creation completed!")


def analyze_and_display(analyzer, sql, title, show_details=True, show_column_lineage=True):
    """Analyze SQL and display results."""
    try:
        result = analyzer.analyze(sql)
        
        if show_details:
            return print_lineage_analysis(result, sql, title, show_column_lineage=show_column_lineage)
        else:
            # Simple success/failure for non-detailed tests
            if result.has_errors():
                print(f"❌ {title}: {result.errors[0]}")
                return False
            else:
                print(f"✅ {title}: Analysis successful")
                return True
        
    except Exception as e:
        print(f"💥 {title} - EXCEPTION: {str(e)}")
        return False


def test_original_sample1():
    """Test sample1.sql - Complex CTE with CategorySales and TopCustomers."""
    print_subsection_header("Sample 1: Complex CTE Query")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = """
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
    """
    
    return analyze_and_display(analyzer, sql, "CategorySales and TopCustomers CTE")


def test_original_sample2():
    """Test sample2.sql - Fully qualified table names."""
    print_subsection_header("Sample 2: Fully Qualified Names")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = """
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
    """
    
    return analyze_and_display(analyzer, sql, "Schema-qualified table references")


def test_original_sample3():
    """Test sample3.sql - Complex UNION with multiple CTEs."""
    print_subsection_header("Sample 3: UNION with Multiple CTEs")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry()) 
    
    # Test with a simplified version first due to complexity
    sql = """
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
    """
    
    return analyze_and_display(analyzer, sql, "Multi-CTE with schema qualification")


def test_sqlglot_test_queries():
    """Test queries from sqlglot-test folder patterns."""
    print_subsection_header("SQLGlot Test Queries")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    queries = [
        ("SELECT id, name, age FROM users WHERE age > 21", "Basic SELECT"),
        ("SELECT DATE_FORMAT(created_at, '%Y-%m') AS month FROM orders", "Date formatting"),
        ("""SELECT u.id, u.name, COUNT(o.id) as order_count, AVG(o.total) as avg_order_value
            FROM users u LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.created_at >= '2023-01-01'
            GROUP BY u.id, u.name HAVING COUNT(o.id) > 5""", "Complex JOIN with aggregation"),
        ("CREATE TABLE user_summary AS SELECT id, name, age FROM users WHERE age >= 18", "CTAS"),
        ("SELECT customer_id, ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) as rank FROM sales", "Window function"),
        ("SELECT APPROX_DISTINCT(user_id) FROM events", "Trino function"),
        ("SELECT JSON_EXTRACT(metadata, '$.user.id') FROM logs", "JSON function")
    ]
    
    success_count = 0
    for sql, description in queries:
        if analyze_and_display(analyzer, sql, description, show_details=False):
            success_count += 1
    
    print(f"\n📊 SQLGlot Test Results: {success_count}/{len(queries)} queries successful")
    return success_count == len(queries)


def test_trino_specific():
    """Test Trino-specific queries."""
    print_subsection_header("Trino-Specific Queries")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    queries = [
        ("SELECT * FROM hive.default.users", "Hive catalog reference"),
        ("SELECT * FROM mysql.sales_db.orders", "MySQL catalog reference"),
        ("""CREATE TABLE hive.analytics.user_summary AS 
            SELECT user_id, name, age FROM hive.default.users WHERE age >= 18""", "Cross-catalog CTAS"),
        ("""WITH user_metrics AS (
                SELECT user_id, COUNT(*) as event_count, 
                       APPROX_DISTINCT(session_id) as session_count,
                       JSON_EXTRACT(properties, '$.category') as category
                FROM events WHERE event_date >= DATE '2023-01-01'
                GROUP BY user_id, JSON_EXTRACT(properties, '$.category')
            ),
            ranked_users AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY event_count DESC) as category_rank
                FROM user_metrics
            )
            SELECT category, user_id, event_count, session_count, category_rank
            FROM ranked_users WHERE category_rank <= 10""", "Complex Trino CTE with functions")
    ]
    
    success_count = 0
    for sql, description in queries:
        if analyze_and_display(analyzer, sql, description, show_details=False):
            success_count += 1
    
    print(f"\n📊 Trino Test Results: {success_count}/{len(queries)} queries successful")
    return success_count == len(queries)


def test_edge_cases():
    """Test edge cases and error conditions."""
    print_subsection_header("Edge Cases and Error Handling")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    test_cases = [
        ("", "Empty SQL", True),  # Should error
        ("SELECT FROM WHERE", "Invalid syntax", True),  # Should error
        ("SELECT * FROM nonexistent_table", "Unknown table", False),  # Should not error
        ("SELECT 1", "No tables", False),  # Should not error
        ("/* Comment only */", "Comment only", True),  # Should error
        ("SELECT id, name FROM users; SELECT * FROM orders;", "Multiple statements", False)  # Should not error
    ]
    
    success_count = 0
    for sql, description, should_error in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"SQL: {repr(sql)}")
        
        try:
            result = analyzer.analyze(sql)
            has_errors = result.has_errors()
            
            if should_error and has_errors:
                print("✅ Expected error occurred")
                success_count += 1
            elif not should_error and not has_errors:
                print("✅ No unexpected errors")
                success_count += 1
            elif should_error and not has_errors:
                print("❌ Expected error but none occurred")
            else:
                print("❌ Unexpected error occurred")
                for error in result.errors:
                    print(f"   Error: {error}")
                    
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
    
    print(f"\n📊 Edge Case Results: {success_count}/{len(test_cases)} tests successful")
    return success_count == len(test_cases)


def test_output_formats():
    """Test different output formats."""
    print_subsection_header("Output Format Tests")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    sql = "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name"
    
    try:
        result = analyzer.analyze(sql)
        if result.has_errors():
            print("❌ Query failed, cannot test output formats")
            return False
        
        # Test JSON output
        from analyzer.formatters import JSONFormatter
        json_formatter = JSONFormatter()
        json_output = json_formatter.format(result)
        print(f"✅ JSON format: {len(json_output)} characters")
        
        # Test table-only JSON
        table_json = json_formatter.format_table_lineage_only(result)
        print(f"✅ Table-only JSON: {len(table_json)} characters")
        
        # Test column-only JSON  
        column_json = json_formatter.format_column_lineage_only(result)
        print(f"✅ Column-only JSON: {len(column_json)} characters")
        
        # Test console format
        from analyzer.formatters import ConsoleFormatter
        console_formatter = ConsoleFormatter()
        print("✅ Console format: Available")
        
        # Test to_dict
        dict_output = result.to_dict()
        print(f"✅ Dictionary format: {len(dict_output)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ Output format test failed: {str(e)}")
        return False


def test_complex_multi_cte_union():
    """Test sample3.sql - Complex multi-CTE query with UNION and advanced analytics."""
    print_subsection_header("Sample 3: Complex Multi-CTE with UNION")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = """
    WITH OrderSummary AS (
        SELECT
            "schema1"."orders"."order_id",
            "schema1"."orders"."customer_id",
            "schema1"."orders"."product_id",
            SUM("schema1"."orders"."order_total") AS total_order_value,
            COUNT(DISTINCT "schema1"."orders"."order_id") AS order_count,
            CASE
                WHEN SUM("schema1"."orders"."order_total") > 1000 THEN 'High'
                WHEN SUM("schema1"."orders"."order_total") BETWEEN 500 AND 1000 THEN 'Medium'
                ELSE 'Low'
            END AS order_priority
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
            CONCAT("schema1"."customers"."first_name", ' ', "schema1"."customers"."last_name") AS customer_name,
            SUM("schema1"."orders"."order_total") AS customer_spent,
            RANK() OVER (
                ORDER BY SUM("schema1"."orders"."order_total") DESC
            ) AS spending_rank
        FROM
            "schema1"."orders"
        INNER JOIN
            "schema1"."customers" ON "schema1"."orders"."customer_id" = "schema1"."customers"."customer_id"
        WHERE
            "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
        GROUP BY
            "schema1"."customers"."customer_id", "schema1"."customers"."first_name", "schema1"."customers"."last_name"
    ),
    TopProductsInCategory AS (
        SELECT
            "schema1"."categories"."category_name",
            "schema1"."products"."product_id",
            COUNT("schema1"."orders"."order_id") AS total_orders,
            RANK() OVER (
                PARTITION BY "schema1"."categories"."category_name"
                ORDER BY COUNT("schema1"."orders"."order_id") DESC
            ) AS product_rank
        FROM
            "schema1"."products"
        INNER JOIN
            "schema1"."categories" ON "schema1"."products"."category_id" = "schema1"."categories"."category_id"
        LEFT JOIN
            "schema1"."orders" ON "schema1"."products"."product_id" = "schema1"."orders"."product_id"
        WHERE
            "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
        GROUP BY
            "schema1"."categories"."category_name", "schema1"."products"."product_id"
    ),
    CombinedResults AS (
        SELECT
            os.customer_id,
            os.order_id,
            os.product_id,
            os.total_order_value,
            os.order_priority,
            NULL AS product_rank,
            NULL AS category_name,
            tc.customer_spent,
            tc.spending_rank
        FROM
            OrderSummary os
        LEFT JOIN
            TopCustomers tc ON os.customer_id = tc.customer_id

        UNION ALL

        SELECT
            NULL AS customer_id,
            NULL AS order_id,
            tp.product_id,
            NULL AS total_order_value,
            NULL AS order_priority,
            tp.product_rank,
            tp.category_name,
            NULL AS customer_spent,
            NULL AS spending_rank
        FROM
            TopProductsInCategory tp
    ),
    FinalResults AS (
        SELECT
            cr.customer_id,
            cr.order_id,
            cr.product_id,
            cr.total_order_value,
            cr.order_priority,
            cr.product_rank,
            cr.category_name,
            cr.customer_spent,
            cr.spending_rank,
            CASE
                WHEN cr.spending_rank IS NOT NULL AND cr.spending_rank <= 5 THEN 'VIP'
                WHEN cr.customer_spent IS NOT NULL THEN 'Premium'
                WHEN cr.product_rank IS NOT NULL AND cr.product_rank <= 3 THEN 'Top Product'
                ELSE 'Standard'
            END AS tag
        FROM
            CombinedResults cr
    )
    SELECT
        fr.customer_id,
        fr.order_id,
        fr.product_id,
        fr.category_name,
        fr.total_order_value,
        fr.customer_spent,
        fr.spending_rank,
        fr.product_rank,
        fr.order_priority,
        fr.tag
    FROM
        FinalResults fr
    ORDER BY
        fr.spending_rank ASC,
        fr.product_rank ASC,
        fr.total_order_value DESC
    """
    
    return analyze_and_display(analyzer, sql, "Complex Multi-CTE with UNION and Analytics")


def test_column_lineage_flag_control():
    """Test column lineage flag control in formatters."""
    print_subsection_header("Column Lineage Flag Control")
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    
    print("\n🧪 Testing Column Lineage Flag Control:")
    print("─" * 50)
    
    # Test with column lineage enabled
    print("\n✅ With Column Lineage (show_column_lineage=True):")
    success1 = analyze_and_display(analyzer, sql, "Sample with Columns", show_column_lineage=True)
    
    print("\n" + "="*80)
    print("❌ Without Column Lineage (show_column_lineage=False):")
    success2 = analyze_and_display(analyzer, sql, "Sample without Columns", show_column_lineage=False)
    
    print("\n📊 Column Lineage Flag Test Results:")
    print(f"  • With column lineage: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"  • Without column lineage: {'✅ Success' if success2 else '❌ Failed'}")
    
    return success1 and success2


def main():
    """Main test runner."""
    print("🚀 SQL Lineage Sample Queries Test Suite")
    print("=" * 60)
    
    # Create output directory with absolute path
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print("Testing all original sample queries and patterns...")
    
    results = []
    json_outputs = []
    
    # Initialize analyzer for JSON collection
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Collect JSON outputs for key sample queries
    sample_queries = [
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
        ("sample2_qualified_names", """SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name"""),
        ("sample3_union_cte", """
        WITH OrderSummary AS (
            SELECT order_id, customer_id, product_id, SUM(order_total) AS total_order_value
            FROM orders WHERE order_date >= '2023-01-01'
            GROUP BY order_id, customer_id, product_id
        ),
        TopCustomers AS (
            SELECT customer_id, SUM(order_total) AS customer_spent
            FROM orders WHERE order_date >= '2023-01-01'
            GROUP BY customer_id
        )
        SELECT os.customer_id, os.order_id, os.product_id, os.total_order_value, tc.customer_spent
        FROM OrderSummary os LEFT JOIN TopCustomers tc ON os.customer_id = tc.customer_id
        """),
        ("sample4_complex_multi_cte_union", """
        WITH OrderSummary AS (
            SELECT
                "schema1"."orders"."order_id",
                "schema1"."orders"."customer_id",
                "schema1"."orders"."product_id",
                SUM("schema1"."orders"."order_total") AS total_order_value,
                COUNT(DISTINCT "schema1"."orders"."order_id") AS order_count,
                CASE
                    WHEN SUM("schema1"."orders"."order_total") > 1000 THEN 'High'
                    WHEN SUM("schema1"."orders"."order_total") BETWEEN 500 AND 1000 THEN 'Medium'
                    ELSE 'Low'
                END AS order_priority
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
                CONCAT("schema1"."customers"."first_name", ' ', "schema1"."customers"."last_name") AS customer_name,
                SUM("schema1"."orders"."order_total") AS customer_spent,
                RANK() OVER (
                    ORDER BY SUM("schema1"."orders"."order_total") DESC
                ) AS spending_rank
            FROM
                "schema1"."orders"
            INNER JOIN
                "schema1"."customers" ON "schema1"."orders"."customer_id" = "schema1"."customers"."customer_id"
            WHERE
                "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
            GROUP BY
                "schema1"."customers"."customer_id", "schema1"."customers"."first_name", "schema1"."customers"."last_name"
        ),
        TopProductsInCategory AS (
            SELECT
                "schema1"."categories"."category_name",
                "schema1"."products"."product_id",
                COUNT("schema1"."orders"."order_id") AS total_orders,
                RANK() OVER (
                    PARTITION BY "schema1"."categories"."category_name"
                    ORDER BY COUNT("schema1"."orders"."order_id") DESC
                ) AS product_rank
            FROM
                "schema1"."products"
            INNER JOIN
                "schema1"."categories" ON "schema1"."products"."category_id" = "schema1"."categories"."category_id"
            LEFT JOIN
                "schema1"."orders" ON "schema1"."products"."product_id" = "schema1"."orders"."product_id"
            WHERE
                "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
            GROUP BY
                "schema1"."categories"."category_name", "schema1"."products"."product_id"
        ),
        CombinedResults AS (
            SELECT
                os.customer_id,
                os.order_id,
                os.product_id,
                os.total_order_value,
                os.order_priority,
                NULL AS product_rank,
                NULL AS category_name,
                tc.customer_spent,
                tc.spending_rank
            FROM
                OrderSummary os
            LEFT JOIN
                TopCustomers tc ON os.customer_id = tc.customer_id
            UNION ALL
            SELECT
                NULL AS customer_id,
                NULL AS order_id,
                tp.product_id,
                NULL AS total_order_value,
                NULL AS order_priority,
                tp.product_rank,
                tp.category_name,
                NULL AS customer_spent,
                NULL AS spending_rank
            FROM
                TopProductsInCategory tp
        ),
        FinalResults AS (
            SELECT
                cr.customer_id,
                cr.order_id,
                cr.product_id,
                cr.total_order_value,
                cr.order_priority,
                cr.product_rank,
                cr.category_name,
                cr.customer_spent,
                cr.spending_rank,
                CASE
                    WHEN cr.spending_rank IS NOT NULL AND cr.spending_rank <= 5 THEN 'VIP'
                    WHEN cr.customer_spent IS NOT NULL THEN 'Premium'
                    WHEN cr.product_rank IS NOT NULL AND cr.product_rank <= 3 THEN 'Top Product'
                    ELSE 'Standard'
                END AS tag
            FROM
                CombinedResults cr
        )
        SELECT
            fr.customer_id,
            fr.order_id,
            fr.product_id,
            fr.category_name,
            fr.total_order_value,
            fr.customer_spent,
            fr.spending_rank,
            fr.product_rank,
            fr.order_priority,
            fr.tag
        FROM
            FinalResults fr
        ORDER BY
            fr.spending_rank ASC,
            fr.product_rank ASC,
            fr.total_order_value DESC
        """)
    ]
    
    for query_name, sql in sample_queries:
        try:
            json_output = analyzer.get_lineage_json(sql)
            json_outputs.append((query_name, json_output))
        except Exception as e:
            print(f"❌ Failed to generate JSON for {query_name}: {e}")
    
    # Run all test sections
    print_section_header("ORIGINAL SAMPLE FILES")
    results.append(test_original_sample1())
    results.append(test_original_sample2()) 
    results.append(test_original_sample3())
    results.append(test_complex_multi_cte_union())
    
    print_section_header("SQLGLOT-TEST PATTERNS")
    results.append(test_sqlglot_test_queries())
    
    print_section_header("TRINO-SPECIFIC FEATURES")
    results.append(test_trino_specific())
    
    print_section_header("EDGE CASES")
    results.append(test_edge_cases())
    
    print_section_header("OUTPUT FORMATS")
    results.append(test_output_formats())
    
    print_section_header("COLUMN LINEAGE FLAG CONTROL")
    results.append(test_column_lineage_flag_control())
    
    # Generate chain outputs for complex sample queries
    chain_outputs = []
    chain_test_queries = [
        ("sample1_complex_cte", sample_queries[0][1]),
        ("sample4_complex_multi_cte_union", sample_queries[3][1])
    ]
    
    for query_name, sql in chain_test_queries:
        try:
            # Test table chains with different depths
            for depth in [1, 2, 3, 4]:
                upstream_table_chain = analyzer.get_table_lineage_chain_json(sql, "upstream", depth)
                chain_outputs.append((query_name, "table", "upstream", depth, upstream_table_chain))
            
            for depth in [1, 2, 3]:
                downstream_table_chain = analyzer.get_table_lineage_chain_json(sql, "downstream", depth)
                chain_outputs.append((query_name, "table", "downstream", depth, downstream_table_chain))
            
            # Test column chains with different depths
            for depth in [1, 2, 3]:
                upstream_column_chain = analyzer.get_column_lineage_chain_json(sql, "upstream", depth)
                chain_outputs.append((query_name, "column", "upstream", depth, upstream_column_chain))
                
                downstream_column_chain = analyzer.get_column_lineage_chain_json(sql, "downstream", depth)
                chain_outputs.append((query_name, "column", "downstream", depth, downstream_column_chain))
        except Exception as e:
            print(f"❌ Failed to generate chain JSON for {query_name}: {e}")
    
    # Save JSON outputs to files
    if json_outputs:
        save_json_outputs(json_outputs, "sample_test")
    
    # Save chain outputs to files
    if chain_outputs:
        save_chain_outputs(chain_outputs, "sample_test")
    
    # Create visualizations for key sample queries
    create_visualizations(analyzer, "sample_test")
    
    print("\n🎉 All sample tests completed!")
    print(f"\n📁 Check the '{output_dir}' directory for generated files.")
    
    # List generated files with better organization
    if os.path.exists(output_dir):
        files = [f for f in os.listdir(output_dir) if f.endswith(('.jpeg', '.json'))]
        if files:
            print(f"\n📄 Generated files ({len(files)}):")
            
            # Categorize files
            visualization_files = [f for f in files if f.endswith('.jpeg')]
            json_files = [f for f in files if f.endswith('.json')]
            
            # Further categorize visualization files
            table_diagrams = [f for f in visualization_files if 'table' in f and 'integrated' not in f]
            integrated_diagrams = [f for f in visualization_files if 'integrated' in f]
            other_visualizations = [f for f in visualization_files if f not in table_diagrams and f not in integrated_diagrams]
            
            if integrated_diagrams:
                print(f"\n   🎨 Integrated Column+Table Diagrams ({len(integrated_diagrams)}):")
                for file in sorted(integrated_diagrams):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
            
            if table_diagrams:
                print(f"\n   📊 Table-Only Diagrams ({len(table_diagrams)}):")
                for file in sorted(table_diagrams)[:5]:  # Show first 5
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
                if len(table_diagrams) > 5:
                    print(f"     ... and {len(table_diagrams) - 5} more table diagrams")
            
            if other_visualizations:
                print(f"\n   🔧 Other Visualization Files ({len(other_visualizations)}):")
                for file in sorted(other_visualizations)[:3]:  # Show first 3
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"     • {file} ({size:,} bytes)")
                if len(other_visualizations) > 3:
                    print(f"     ... and {len(other_visualizations) - 3} more files")
            
            if json_files:
                print(f"\n   📋 JSON Data Files ({len(json_files)}):")
                chain_json_files = [f for f in json_files if 'chain' in f]
                regular_json_files = [f for f in json_files if 'chain' not in f]
                
                if regular_json_files:
                    print(f"     📊 Sample Query Results ({len(regular_json_files)}):")
                    for file in sorted(regular_json_files):
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"       • {file} ({size:,} bytes)")
                
                if chain_json_files:
                    print(f"     🔗 Lineage Chain Data ({len(chain_json_files)}):")
                    for file in sorted(chain_json_files)[:6]:
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"       • {file} ({size:,} bytes)")
                    if len(chain_json_files) > 6:
                        print(f"       ... and {len(chain_json_files) - 6} more chain files")
            
            print(f"\n✨ Sample tests demonstrate real-world SQL patterns with")
            print(f"   complex CTEs, UNION operations, and query-contextualized visualizations!")
        else:
            print("\n⚠️  No files were generated")
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    success = print_test_summary(total, passed, "Sample Queries Test Suite")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())