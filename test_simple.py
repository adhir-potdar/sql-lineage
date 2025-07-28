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
from analyzer.metadata import SampleMetadataRegistry
from analyzer.formatters import ConsoleFormatter, JSONFormatter
from test_formatter import print_lineage_analysis, print_test_summary, print_section_header


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
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    sql = "SELECT id, name, email FROM users WHERE age > 25"
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    runner.assert_in("MAIN_QUERY", result.table_lineage.upstream, "Should have MAIN_QUERY in upstream")
    runner.assert_in("default.users", result.table_lineage.upstream["MAIN_QUERY"], "Should depend on users table")
    
    print_lineage_analysis(result, sql, "Simple SELECT Query")


def test_simple_join():
    """Test basic JOIN query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    sql = """
    SELECT u.name, o.total, o.order_date
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE u.age > 18
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    upstream = result.table_lineage.upstream["MAIN_QUERY"]
    runner.assert_in("default.users", upstream, "Should depend on users table")
    runner.assert_in("default.orders", upstream, "Should depend on orders table")
    runner.assert_greater(len(result.column_lineage.upstream), 0, "Should have column lineage")
    
    print_lineage_analysis(result, sql, "Simple JOIN Query")


def test_simple_cte():
    """Test basic CTE query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    runner.assert_in("MAIN_QUERY", result.table_lineage.upstream, "Should have main query in upstream")
    runner.assert_in("default.users", result.table_lineage.upstream["active_users"], "CTE should depend on users")
    runner.assert_in("active_users", result.table_lineage.upstream["MAIN_QUERY"], "Main query should depend on CTE")
    
    print_lineage_analysis(result, sql, "Simple CTE Query")


def test_create_table_as_select():
    """Test CREATE TABLE AS SELECT."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    runner.assert_in("default.users", result.table_lineage.upstream["user_summary"], "Target should depend on users")
    
    # Check downstream
    runner.assert_in("default.users", result.table_lineage.downstream, "Users should have downstream")
    runner.assert_in("user_summary", result.table_lineage.downstream["default.users"], "Users should flow to user_summary")
    
    print(f"✓ SQL: {sql.strip()}")
    print(f"✓ Target table upstream: {result.table_lineage.upstream['user_summary']}")
    print(f"✓ Source table downstream: {result.table_lineage.downstream['default.users']}")


def test_complex_multi_cte():
    """Test complex query with multiple CTEs."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    
    # Check all CTEs exist
    expected_ctes = ["order_stats", "customer_segments", "segment_summary"]
    for cte in expected_ctes:
        runner.assert_in(cte, result.table_lineage.upstream, f"Should have {cte} CTE")
    
    # Check dependencies
    runner.assert_in("default.orders", result.table_lineage.upstream["order_stats"], "order_stats should depend on orders")
    runner.assert_in("default.users", result.table_lineage.upstream["customer_segments"], "customer_segments should depend on users")
    runner.assert_in("order_stats", result.table_lineage.upstream["customer_segments"], "customer_segments should depend on order_stats")
    runner.assert_in("customer_segments", result.table_lineage.upstream["segment_summary"], "segment_summary should depend on customer_segments")
    runner.assert_in("segment_summary", result.table_lineage.upstream["MAIN_QUERY"], "main query should depend on segment_summary")
    
    print(f"✓ SQL: Complex multi-CTE query")
    print(f"✓ Found CTEs: {[cte for cte in expected_ctes if cte in result.table_lineage.upstream]}")
    print(f"✓ Total upstream relationships: {len(result.table_lineage.upstream)}")
    print(f"✓ Total downstream relationships: {len(result.table_lineage.downstream)}")
    print(f"✓ Column lineage relationships: {len(result.column_lineage.upstream)}")


def test_complex_joins_with_aggregation():
    """Test complex query with multiple joins and aggregation."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    
    upstream = result.table_lineage.upstream["MAIN_QUERY"]
    expected_tables = ["default.categories", "default.products", "default.orders", "default.users"]
    
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
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    runner.assert_in("default.orders", result.table_lineage.upstream["MAIN_QUERY"], "Should depend on orders table")
    runner.assert_greater(len(result.column_lineage.upstream), 3, "Should have column lineage for window functions")
    
    print(f"✓ SQL: Window functions query")
    print(f"✓ Upstream: {result.table_lineage.upstream}")
    print(f"✓ Downstream: {result.table_lineage.downstream}")
    print(f"✓ Column lineage: {len(result.column_lineage.upstream)} relationships")


def test_subquery():
    """Test query with subquery."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    upstream = result.table_lineage.upstream["MAIN_QUERY"]
    runner.assert_in("default.users", upstream, "Should depend on users table")
    runner.assert_in("default.orders", upstream, "Should depend on orders table")
    
    print(f"✓ SQL: Subquery test")
    print(f"✓ Upstream tables: {upstream}")
    print(f"✓ Downstream tables: {result.table_lineage.downstream}")


def test_union_query():
    """Test UNION query."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    sql = """
    SELECT 'user' as type, id, name as identifier FROM users WHERE age > 25
    UNION ALL
    SELECT 'product' as type, product_id as id, product_name as identifier FROM products WHERE price > 100
    UNION ALL
    SELECT 'category' as type, category_id as id, category_name as identifier FROM categories
    """
    
    result = analyzer.analyze(sql)
    
    runner.assert_true(not result.has_errors(), "Query should not have errors")
    upstream = result.table_lineage.upstream["MAIN_QUERY"]
    
    expected_tables = ["default.users", "default.products", "default.categories"]
    for table in expected_tables:
        runner.assert_in(table, upstream, f"Should depend on {table}")
    
    print(f"✓ SQL: UNION query")
    print(f"✓ Upstream tables: {sorted(upstream)}")
    print(f"✓ Downstream tables: {result.table_lineage.downstream}")


def test_error_handling():
    """Test error handling with invalid SQL."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
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
        analyzer.set_metadata_registry(SampleMetadataRegistry())
        result = analyzer.analyze(sql)
        
        runner.assert_true(not result.has_errors(), f"Query should work with {dialect} dialect")
        runner.assert_true(result.dialect == dialect, f"Result should have {dialect} dialect")
        runner.assert_in("MAIN_QUERY", result.table_lineage.upstream, f"Should have upstream with {dialect}")
    
    print(f"✓ Tested dialects: {dialects}")


def test_json_output():
    """Test JSON output formatting."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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
    analyzer.set_metadata_registry(SampleMetadataRegistry())
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


def main():
    """Main test runner."""
    global runner
    runner = SimpleTestRunner()
    
    print_section_header("SQL Lineage Analyzer - Simple Tests")
    
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
        (test_metadata_integration, "Metadata Integration")
    ]
    
    for test_func, test_name in test_functions:
        runner.run_test(test_func, test_name)
    
    # Print final summary
    runner.print_summary()
    success = print_test_summary(runner.tests_run, runner.tests_passed, "Simple Tests")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())