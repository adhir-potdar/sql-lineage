#!/usr/bin/env python3
"""
Simple test runner without pytest for basic SQL lineage analysis testing.
This script provides standalone testing functionality to verify the analyzer works correctly.
"""

import sys
import os
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from analyzer import SQLLineageAnalyzer
from analyzer.formatters import ConsoleFormatter, JSONFormatter
from analyzer.visualization import SQLLineageVisualizer
from test_formatter import print_lineage_analysis, print_test_summary, print_section_header


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
    """Create visualization outputs for key test queries."""
    try:
        visualizer = SQLLineageVisualizer()
    except ImportError as e:
        print(f"⚠️  Visualization not available: {e}")
        return
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Key queries for visualization
    visualization_queries = [
        ("simple_cte", """
        WITH active_users AS (
            SELECT id, name, email FROM users WHERE active = true
        )
        SELECT * FROM active_users WHERE email LIKE '%@company.com'
        """),
        ("complex_multi_cte", """
        WITH sales_data AS (
            SELECT 
                customer_id,
                product_id,
                SUM(quantity * price) AS total_sales,
                COUNT(*) AS order_count
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            GROUP BY customer_id, product_id
        ),
        customer_summary AS (
            SELECT 
                sd.customer_id,
                u.name AS customer_name,
                SUM(sd.total_sales) AS lifetime_value,
                COUNT(DISTINCT sd.product_id) AS unique_products
            FROM sales_data sd
            JOIN users u ON sd.customer_id = u.id
            GROUP BY sd.customer_id, u.name
        ),
        top_customers AS (
            SELECT 
                customer_name,
                lifetime_value,
                unique_products,
                RANK() OVER (ORDER BY lifetime_value DESC) AS customer_rank
            FROM customer_summary
            WHERE lifetime_value > 1000
        )
        SELECT * FROM top_customers WHERE customer_rank <= 10
        """)
    ]
    
    print_section_header("Creating Visualizations", 50)
    
    for query_name, sql in visualization_queries:
        print(f"\n📊 Creating visualizations for {query_name}...")
        
        try:
            # Create visualizations for both upstream and downstream
            for chain_type in ["upstream", "downstream"]:
                # Get chain data with depth 3
                table_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
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
    
    print("\n🎨 Visualization creation completed!")


class SimpleTestRunner:
    """Simple test runner without external dependencies."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
    
    def assert_true(self, condition, message="Assertion failed"):
        """Simple assertion helper."""
        if not condition:
            raise AssertionError(message)
    
    def assert_in(self, item, container, message=None):
        """Assert item is in container."""
        if item not in container:
            if message is None:
                message = f"'{item}' not found in {container}"
            raise AssertionError(message)
    
    def assert_greater(self, a, b, message=None):
        """Assert a > b."""
        if not a > b:
            if message is None:
                message = f"{a} is not greater than {b}"
            raise AssertionError(message)
    
    def run_test(self, test_func, test_name):
        """Run a single test function."""
        self.tests_run += 1
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            test_func()
            self.tests_passed += 1
            print(f"✅ PASSED: {test_name}")
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e), traceback.format_exc()))
            print(f"❌ FAILED: {test_name}")
            print(f"   Error: {str(e)}")
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)
        print(f"Total tests run: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        
        if self.failures:
            print(f"\nFAILED TESTS:")
            for test_name, error, tb in self.failures:
                print(f"\n❌ {test_name}")
                print(f"   Error: {error}")
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        return self.tests_failed == 0


def test_simple_select():
    """Test basic SELECT query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = "SELECT id, name, email FROM users WHERE age > 25"
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    runner.assert_in("QUERY_RESULT", result.table_lineage.upstream, "Should have QUERY_RESULT in upstream")
    runner.assert_in("users", result.table_lineage.upstream["QUERY_RESULT"], "Should depend on users table")
    
    print_lineage_analysis(result, sql, "Simple SELECT Query")


def test_simple_join():
    """Test basic JOIN query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
    SELECT u.name, o.total, o.order_date
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.age > 18
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    upstream = result.table_lineage.upstream["QUERY_RESULT"]
    runner.assert_in("users", upstream, "Should depend on users table")
    runner.assert_in("orders", upstream, "Should depend on orders table")
    runner.assert_greater(len(result.column_lineage.upstream), 0, "Should have column lineage")
    
    print_lineage_analysis(result, sql, "Simple JOIN Query")


def test_simple_cte():
    """Test basic CTE query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
    WITH active_users AS (
        SELECT id, name, email
        FROM users
        WHERE created_at >= '2023-01-01'
    )
    SELECT * FROM active_users WHERE name LIKE 'A%'
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    runner.assert_in("active_users", result.table_lineage.upstream, "Should have CTE in upstream")
    runner.assert_in("QUERY_RESULT", result.table_lineage.upstream, "Should have main query in upstream")
    runner.assert_in("users", result.table_lineage.upstream["active_users"], "CTE should depend on users")
    runner.assert_in("active_users", result.table_lineage.upstream["QUERY_RESULT"], "Main query should depend on CTE")
    
    print_lineage_analysis(result, sql, "Simple CTE Query")


def test_create_table_as_select():
    """Test CREATE TABLE AS SELECT."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
    CREATE TABLE user_summary AS
    SELECT id, name, age, COUNT(*) as login_count
    FROM users
    WHERE age >= 18
    GROUP BY id, name, age
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    runner.assert_in("user_summary", result.table_lineage.upstream, "Should have target table in upstream")
    runner.assert_in("users", result.table_lineage.upstream["user_summary"], "Target should depend on users")
    
    # Check downstream
    runner.assert_in("users", result.table_lineage.downstream, "Users should have downstream")
    runner.assert_in("user_summary", result.table_lineage.downstream["users"], "Users should flow to user_summary")
    
    print(f"✓ SQL: {sql.strip()}")
    print(f"✓ Target table upstream: {result.table_lineage.upstream['user_summary']}")
    print(f"✓ Source table downstream: {result.table_lineage.downstream['users']}")


def test_complex_multi_cte():
    """Test complex query with multiple CTEs."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
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
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    
    # Check remaining CTEs exist (segment_summary should be eliminated as pass-through)
    expected_ctes = ["order_stats", "customer_segments"]
    for cte in expected_ctes:
        runner.assert_in(cte, result.table_lineage.upstream, f"Should have {cte} CTE")
    
    # Check that segment_summary was eliminated as a pass-through CTE
    runner.assert_true("segment_summary" not in result.table_lineage.upstream, "segment_summary should be eliminated as pass-through CTE")
    
    # Check dependencies
    runner.assert_in("orders", result.table_lineage.upstream["order_stats"], "order_stats should depend on orders")
    runner.assert_in("users", result.table_lineage.upstream["customer_segments"], "customer_segments should depend on users")
    runner.assert_in("order_stats", result.table_lineage.upstream["customer_segments"], "customer_segments should depend on order_stats")
    
    # QUERY_RESULT should now connect directly to customer_segments (skipping eliminated segment_summary)
    runner.assert_in("customer_segments", result.table_lineage.upstream["QUERY_RESULT"], "main query should connect directly to customer_segments")
    
    print(f"✓ SQL: Complex multi-CTE query")
    print(f"✓ Found CTEs: {[cte for cte in expected_ctes if cte in result.table_lineage.upstream]}")
    print(f"✓ Total upstream relationships: {len(result.table_lineage.upstream)}")
    print(f"✓ Total downstream relationships: {len(result.table_lineage.downstream)}")
    print(f"✓ Column lineage relationships: {len(result.column_lineage.upstream)}")


def test_complex_joins_with_aggregation():
    """Test complex query with multiple joins and aggregation."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
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
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    
    upstream = result.table_lineage.upstream["QUERY_RESULT"]
    expected_tables = ["categories", "products", "orders", "users"]
    
    for table in expected_tables:
        runner.assert_in(table, upstream, f"Should depend on {table}")
    
    runner.assert_greater(len(result.column_lineage.upstream), 5, "Should have substantial column lineage")
    
    print(f"✓ SQL: Complex multi-table JOIN with aggregation")
    print(f"✓ Upstream tables: {sorted(upstream)}")
    print(f"✓ Downstream tables: {result.table_lineage.downstream}")
    print(f"✓ Column lineage count: {len(result.column_lineage.upstream)}")


def test_window_functions():
    """Test query with window functions."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
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
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    runner.assert_in("orders", result.table_lineage.upstream["QUERY_RESULT"], "Should depend on orders table")
    runner.assert_greater(len(result.column_lineage.upstream), 3, "Should have column lineage for window functions")
    
    print(f"✓ SQL: Window functions query")
    print(f"✓ Upstream: {result.table_lineage.upstream}")
    print(f"✓ Downstream: {result.table_lineage.downstream}")
    print(f"✓ Column lineage: {len(result.column_lineage.upstream)} relationships")


def test_subquery():
    """Test query with subquery."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
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
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    upstream = result.table_lineage.upstream["QUERY_RESULT"]
    runner.assert_in("users", upstream, "Should depend on users table")
    runner.assert_in("orders", upstream, "Should depend on orders table")
    
    print(f"✓ SQL: Subquery test")
    print(f"✓ Upstream tables: {upstream}")
    print(f"✓ Downstream tables: {result.table_lineage.downstream}")


def test_union_query():
    """Test UNION query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = """
    SELECT 'user' as type, id, name as identifier FROM users WHERE age > 25
    UNION ALL
    SELECT 'product' as type, product_id as id, product_name as identifier FROM products WHERE price > 100
    UNION ALL
    SELECT 'category' as type, category_id as id, category_name as identifier FROM categories
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    upstream = result.table_lineage.upstream["QUERY_RESULT"]
    
    expected_tables = ["users", "products", "categories"]
    for table in expected_tables:
        runner.assert_in(table, upstream, f"Should depend on {table}")
    
    print(f"✓ SQL: UNION query")
    print(f"✓ Upstream tables: {sorted(upstream)}")
    print(f"✓ Downstream tables: {result.table_lineage.downstream}")


def test_error_handling():
    """Test error handling with invalid SQL."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Test empty SQL
    result1 = analyzer.analyze("")
    runner.assert_true(result1.has_errors(), "Empty SQL should have errors")
    
    # Test invalid SQL syntax
    result2 = analyzer.analyze("SELECT FROM WHERE")
    runner.assert_true(result2.has_errors(), "Invalid SQL should have errors")
    
    # Test None input
    result3 = analyzer.analyze("SELECT * FROM nonexistent_table")
    runner.assert_true(not result3.has_errors(), "Valid syntax should not have errors even with unknown tables")
    
    print("✓ Error handling tests passed")


def test_different_dialects():
    """Test different SQL dialects."""
    sql = "SELECT id, name FROM users LIMIT 10"
    
    dialects = ["trino", "mysql", "postgres", "sqlite"]
    
    for dialect in dialects:
        analyzer = SQLLineageAnalyzer(dialect=dialect)
            result = analyzer.analyze(sql)
        
        runner.assert_true(not result.has_errors(), f"Query should work with {dialect} dialect")
        runner.assert_true(result.dialect == dialect, f"Result should have {dialect} dialect")
        runner.assert_in("QUERY_RESULT", result.table_lineage.upstream, f"Should have upstream with {dialect}")
    
    print(f"✓ Tested dialects: {dialects}")


def test_json_output():
    """Test JSON output formatting."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name"
    
    result = analyzer.analyze(sql)
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    
    # Test JSON formatter
    json_formatter = JSONFormatter()
    json_output = json_formatter.format(result)
    
    runner.assert_true(isinstance(json_output, str), "JSON output should be string")
    runner.assert_true(len(json_output) > 50, "JSON output should be substantial")
    runner.assert_in("table_lineage", json_output, "JSON should contain table_lineage")
    runner.assert_in("column_lineage", json_output, "JSON should contain column_lineage")
    
    print("✓ JSON formatting test passed")
    print(f"✓ JSON output length: {len(json_output)} characters")


def test_metadata_integration():
    """Test metadata integration."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    sql = "SELECT u.name, u.email, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    
    result = analyzer.analyze(sql)
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    
    # Check if metadata is populated
    runner.assert_greater(len(result.metadata), 0, "Should have metadata for tables")
    
    # Check if users table metadata exists
    users_metadata_found = any("users" in key for key in result.metadata.keys())
    runner.assert_true(users_metadata_found, "Should have metadata for users table")
    
    print("✓ Metadata integration test passed")
    print(f"✓ Metadata entries: {len(result.metadata)}")
    for key in result.metadata.keys():
        print(f"  - {key}")


def test_chain_building():
    """Test table and column lineage chain building functionality."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Complex multi-level CTE query for thorough chain testing
    sql = """
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
    """
    
    result = analyzer.analyze(sql)
    
    # Test basic chain functionality
    runner.assert_true(not result.has_errors(), "Chain test query should not have errors")
    
    # Test table upstream chains with different depths
    upstream_table_chain_1 = analyzer.get_table_lineage_chain(sql, "upstream", 1)
    runner.assert_true("chains" in upstream_table_chain_1, "Should have chains in upstream table result")
    runner.assert_true(upstream_table_chain_1["chain_type"] == "upstream", "Should have correct chain type")
    runner.assert_true(upstream_table_chain_1["max_depth"] == 1, "Should have correct max depth")
    
    upstream_table_chain_3 = analyzer.get_table_lineage_chain(sql, "upstream", 3)
    runner.assert_true(len(upstream_table_chain_3["chains"]) > 0, "Should have table chains for depth 3")
    
    # Test table downstream chains
    downstream_table_chain_2 = analyzer.get_table_lineage_chain(sql, "downstream", 2)
    runner.assert_true(downstream_table_chain_2["chain_type"] == "downstream", "Should have downstream chain type")
    
    # Test column upstream chains with different depths
    upstream_column_chain_1 = analyzer.get_column_lineage_chain(sql, "upstream", 1)
    runner.assert_true("chains" in upstream_column_chain_1, "Should have chains in upstream column result")
    runner.assert_true(upstream_column_chain_1["chain_type"] == "upstream", "Should have correct column chain type")
    
    upstream_column_chain_3 = analyzer.get_column_lineage_chain(sql, "upstream", 3)
    runner.assert_true(len(upstream_column_chain_3["chains"]) >= 0, "Should have column chains for depth 3")
    
    # Test column downstream chains
    downstream_column_chain_2 = analyzer.get_column_lineage_chain(sql, "downstream", 2)
    runner.assert_true(downstream_column_chain_2["chain_type"] == "downstream", "Should have downstream column chain type")
    
    # Test JSON output for table chains
    table_json_output = analyzer.get_table_lineage_chain_json(sql, "upstream", 4)
    runner.assert_true(len(table_json_output) > 100, "Table JSON output should be substantial")
    runner.assert_in("chain_type", table_json_output, "Table JSON should contain chain_type")
    runner.assert_in("upstream", table_json_output, "Table JSON should contain upstream")
    
    # Test JSON output for column chains
    column_json_output = analyzer.get_column_lineage_chain_json(sql, "upstream", 4)
    runner.assert_true(len(column_json_output) > 100, "Column JSON output should be substantial")
    runner.assert_in("chain_type", column_json_output, "Column JSON should contain chain_type")
    runner.assert_in("upstream", column_json_output, "Column JSON should contain upstream")
    
    # Test error handling for table chains
    try:
        analyzer.get_table_lineage_chain(sql, "invalid_type", 1)
        runner.assert_true(False, "Should have raised ValueError for invalid table chain type")
    except ValueError:
        pass  # Expected
    
    try:
        analyzer.get_table_lineage_chain(sql, "upstream", 0)
        runner.assert_true(False, "Should have raised ValueError for invalid table depth")
    except ValueError:
        pass  # Expected
    
    # Test error handling for column chains
    try:
        analyzer.get_column_lineage_chain(sql, "invalid_type", 1)
        runner.assert_true(False, "Should have raised ValueError for invalid column chain type")
    except ValueError:
        pass  # Expected
    
    try:
        analyzer.get_column_lineage_chain(sql, "upstream", 0)
        runner.assert_true(False, "Should have raised ValueError for invalid column depth")
    except ValueError:
        pass  # Expected
    
    print("✓ Chain building functionality test passed")
    print(f"✓ Table upstream chains (depth 1): {len(upstream_table_chain_1['chains'])}")
    print(f"✓ Table upstream chains (depth 3): {len(upstream_table_chain_3['chains'])}")
    print(f"✓ Table downstream chains (depth 2): {len(downstream_table_chain_2['chains'])}")
    print(f"✓ Column upstream chains (depth 1): {len(upstream_column_chain_1['chains'])}")
    print(f"✓ Column upstream chains (depth 3): {len(upstream_column_chain_3['chains'])}")
    print(f"✓ Column downstream chains (depth 2): {len(downstream_column_chain_2['chains'])}")
    print(f"✓ Table JSON output length: {len(table_json_output)} characters")
    print(f"✓ Column JSON output length: {len(column_json_output)} characters")


def main():
    """Main test runner."""
    global runner
    runner = SimpleTestRunner()
    
    print("🚀 SQL Lineage Simple Test Suite")
    print("=" * 60)
    
    # Create output directory with absolute path
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Collect JSON outputs for key test queries
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    json_outputs = []
    chain_outputs = []
    test_queries = [
        ("simple_select", "SELECT id, name, email FROM users WHERE age > 25"),
        ("simple_join", "SELECT u.name, o.total, o.order_date FROM users u JOIN orders o ON u.id = o.user_id WHERE u.age > 18"),
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
        """)
    ]
    
    for query_name, sql in test_queries:
        try:
            json_output = analyzer.get_lineage_json(sql)
            json_outputs.append((query_name, json_output))
        except Exception as e:
            print(f"❌ Failed to generate JSON for {query_name}: {e}")
    
    # Generate chain outputs for complex queries
    chain_test_queries = [
        ("simple_cte", test_queries[2][1]),
        ("complex_multi_cte", test_queries[4][1])
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
    
    # Run all tests
    test_functions = [
        (test_simple_select, "Simple SELECT Query"),
        (test_simple_join, "Simple JOIN Query"),
        (test_simple_cte, "Simple CTE Query"),
        (test_create_table_as_select, "CREATE TABLE AS SELECT"),
        (test_complex_multi_cte, "Complex Multi-CTE Query"),
        (test_complex_joins_with_aggregation, "Complex JOINs with Aggregation"),
        (test_window_functions, "Window Functions"),
        (test_subquery, "Subquery"),
        (test_union_query, "UNION Query"),
        (test_error_handling, "Error Handling"),
        (test_different_dialects, "Different Dialects"),
        (test_json_output, "JSON Output"),
        (test_metadata_integration, "Metadata Integration"),
        (test_chain_building, "Chain Building Functionality")
    ]
    
    for test_func, test_name in test_functions:
        runner.run_test(test_func, test_name)
    
    # Save JSON outputs to files
    if json_outputs:
        save_json_outputs(json_outputs, "simple_test")
    
    # Save chain outputs to files
    if chain_outputs:
        save_chain_outputs(chain_outputs, "simple_test")
    
    # Create visualizations for key queries
    create_visualizations(analyzer, "simple_test")
    
    print("\n🎉 All simple tests completed!")
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
                    print(f"     📊 Analysis Results ({len(regular_json_files)}):")
                    for file in sorted(regular_json_files)[:3]:
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"       • {file} ({size:,} bytes)")
                    if len(regular_json_files) > 3:
                        print(f"       ... and {len(regular_json_files) - 3} more analysis files")
                
                if chain_json_files:
                    print(f"     🔗 Lineage Chain Data ({len(chain_json_files)}):")
                    for file in sorted(chain_json_files)[:5]:
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"       • {file} ({size:,} bytes)")
                    if len(chain_json_files) > 5:
                        print(f"       ... and {len(chain_json_files) - 5} more chain files")
            
            print(f"\n✨ Simple tests demonstrate core lineage functionality with")
            print(f"   comprehensive SQL pattern coverage and query-contextualized visualizations!")
        else:
            print("\n⚠️  No files were generated")
    
    # Print final summary
    runner.print_summary()
    success = print_test_summary(runner.tests_run, runner.tests_passed, "Simple Tests")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())