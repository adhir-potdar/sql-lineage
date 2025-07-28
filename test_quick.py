#!/usr/bin/env python3
"""
Quick test script for basic functionality verification.
Use this for rapid testing of core lineage analysis features.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from analyzer import SQLLineageAnalyzer
from analyzer.metadata import SampleMetadataRegistry
from test_formatter import print_quick_result, print_section_header, print_test_summary, print_lineage_analysis


def quick_test():
    """Quick functionality test."""
    print_section_header("Quick SQL Lineage Test", 60)
    
    # Initialize analyzer with sample metadata
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Test 1: Simple query
    sql1 = "SELECT name, email FROM users WHERE age > 25"
    result1 = analyzer.analyze(sql1)
    
    if not print_lineage_analysis(result1, sql1, "1. Simple Query"):
        return False
    
    # Test 2: JOIN query  
    sql2 = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    result2 = analyzer.analyze(sql2)
    
    if not print_lineage_analysis(result2, sql2, "2. JOIN Query"):
        return False
    
    # Test 3: CTE query
    sql3 = """
    WITH active_users AS (SELECT id, name FROM users WHERE active = true)
    SELECT * FROM active_users WHERE name LIKE 'A%'
    """
    result3 = analyzer.analyze(sql3)
    
    if not print_lineage_analysis(result3, sql3, "3. CTE Query"):
        return False
    
    # Test 4: CREATE TABLE AS SELECT
    sql4 = "CREATE TABLE user_summary AS SELECT id, name, COUNT(*) as login_count FROM users GROUP BY id, name"
    result4 = analyzer.analyze(sql4)
    
    if not print_lineage_analysis(result4, sql4, "4. CREATE TABLE AS SELECT Query"):
        return False
    
    # Test 5: Complex query
    sql5 = """
    WITH order_stats AS (
        SELECT customer_id, COUNT(*) as orders, SUM(total) as spent
        FROM orders WHERE order_date >= '2023-01-01'
        GROUP BY customer_id
    )
    SELECT u.name, os.orders, os.spent
    FROM users u JOIN order_stats os ON u.id = os.customer_id
    WHERE os.spent > 1000
    """
    result5 = analyzer.analyze(sql5)
    
    if not print_lineage_analysis(result5, sql5, "5. Complex Multi-CTE Query"):
        return False
    
    print("\n🎉 All quick tests passed!")
    return True


def performance_test():
    """Quick performance test."""
    print("\n⚡ Performance Test")
    print("─" * 40)
    
    import time
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Complex query for performance testing
    complex_sql = """
    WITH sales_by_month AS (
        SELECT 
            DATE_TRUNC('month', o.order_date) as month,
            p.category_id,
            SUM(o.total) as monthly_sales,
            COUNT(DISTINCT o.customer_id) as unique_customers
        FROM orders o
        JOIN products p ON o.product_id = p.product_id
        WHERE o.order_date >= '2022-01-01'
        GROUP BY DATE_TRUNC('month', o.order_date), p.category_id
    ),
    category_performance AS (
        SELECT 
            c.category_name,
            sbm.month,
            sbm.monthly_sales,
            sbm.unique_customers,
            LAG(sbm.monthly_sales) OVER (PARTITION BY c.category_id ORDER BY sbm.month) as prev_month_sales
        FROM sales_by_month sbm
        JOIN categories c ON sbm.category_id = c.category_id
    ),
    growth_analysis AS (
        SELECT 
            *,
            CASE 
                WHEN prev_month_sales IS NOT NULL 
                THEN (monthly_sales - prev_month_sales) / prev_month_sales * 100
                ELSE NULL 
            END as growth_rate
        FROM category_performance
    )
    SELECT 
        category_name,
        month,
        monthly_sales,
        unique_customers,
        growth_rate,
        RANK() OVER (PARTITION BY month ORDER BY monthly_sales DESC) as sales_rank
    FROM growth_analysis
    WHERE growth_rate > 10 OR growth_rate IS NULL
    ORDER BY month DESC, monthly_sales DESC
    """
    
    start_time = time.time()
    result = analyzer.analyze(complex_sql)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.has_errors():
        print("❌ Performance test failed:", result.errors[0])
        return False
    
    print(f"✅ Analysis time: {duration:.3f} seconds")
    print(f"✅ Upstream relationships: {len(result.table_lineage.upstream)}")
    print(f"✅ Downstream relationships: {len(result.table_lineage.downstream)}")
    print(f"✅ Column relationships: {len(result.column_lineage.upstream)}")
    print(f"✅ Metadata tables: {len(result.metadata)}")
    
    return True


def dialect_test():
    """Test multiple SQL dialects."""
    print("\n🌐 Dialect Test")
    print("─" * 30)
    
    sql = "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name LIMIT 10"
    
    dialects = ["trino", "mysql", "postgres", "sqlite"]
    
    for dialect in dialects:
        analyzer = SQLLineageAnalyzer(dialect=dialect)
        analyzer.set_metadata_registry(SampleMetadataRegistry())
        result = analyzer.analyze(sql)
        
        if result.has_errors():
            print(f"❌ {dialect}: {result.errors[0]}")
        else:
            upstream_count = len(result.table_lineage.upstream.get("MAIN_QUERY", []))
            downstream_count = len(result.table_lineage.downstream)
            print(f"✅ {dialect}: {upstream_count} upstream, {downstream_count} downstream")
    
    return True


def test_column_lineage_flag():
    """Test column lineage flag functionality."""
    print("\n🔧 Column Lineage Flag Test")
    print("─" * 40)
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    result = analyzer.analyze(sql)
    
    if result.has_errors():
        print("❌ Column lineage flag test failed: Query has errors")
        return False
    
    print("Testing with column lineage ENABLED:")
    print_lineage_analysis(result, sql, "With Column Details", show_column_lineage=True)
    
    print("\n" + "="*80)
    print("Testing with column lineage DISABLED:")
    print_lineage_analysis(result, sql, "Without Column Details", show_column_lineage=False)
    
    print("✅ Column lineage flag functionality demonstrated")
    return True


def main():
    """Main test runner."""
    success = True
    
    try:
        # Run quick tests
        success_count = 0
        total_tests = 4
        
        # Run quick tests
        if quick_test():
            success_count += 1
        
        # Run performance test
        if performance_test():
            success_count += 1
        
        # Run dialect test
        if dialect_test():
            success_count += 1
        
        # Run column lineage flag test
        if test_column_lineage_flag():
            success_count += 1
        
        # Print summary
        print_test_summary(total_tests, success_count, "Quick Tests")
        return 0 if success_count == total_tests else 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the analyzer package:")
        print("   cd /Users/adhirpotdar/Work/git-repos/sql-lineage")
        print("   ./dev.sh install")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())