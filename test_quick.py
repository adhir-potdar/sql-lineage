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
from analyzer.visualization import SQLLineageVisualizer
from test_formatter import print_quick_result, print_section_header, print_test_summary, print_lineage_analysis


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
        from analyzer.visualization import SQLLineageVisualizer
        visualizer = SQLLineageVisualizer()
    except ImportError as e:
        print(f"⚠️  Visualization not available: {e}")
        return
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Key queries for visualization
    visualization_queries = [
        ("cte_query", """
        WITH active_users AS (SELECT id, name FROM users WHERE active = true)
        SELECT * FROM active_users WHERE name LIKE 'A%'
        """),
        ("complex_multi_cte_query", """
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
    
    print_section_header("Creating Visualizations", 50)
    
    for query_name, sql in visualization_queries:
        print(f"\n📊 Creating visualizations for {query_name}...")
        
        try:
            # Create visualizations for both upstream and downstream with depth 3
            for chain_type in ["upstream", "downstream"]:
                # Get chain data
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


def quick_test():
    """Quick functionality test."""
    print_section_header("Quick SQL Lineage Test", 60)
    
    # Initialize analyzer with sample metadata
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Store all JSON outputs and chain outputs
    json_outputs = []
    chain_outputs = []
    
    # Test 1: Simple query
    sql1 = "SELECT name, email FROM users WHERE age > 25"
    result1 = analyzer.analyze(sql1)
    json_outputs.append(("simple_query", analyzer.get_lineage_json(sql1)))
    
    if not print_lineage_analysis(result1, sql1, "1. Simple Query"):
        return False
    
    # Test 2: JOIN query  
    sql2 = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    result2 = analyzer.analyze(sql2)
    json_outputs.append(("join_query", analyzer.get_lineage_json(sql2)))
    
    if not print_lineage_analysis(result2, sql2, "2. JOIN Query"):
        return False
    
    # Test 3: CTE query
    sql3 = """
    WITH active_users AS (SELECT id, name FROM users WHERE active = true)
    SELECT * FROM active_users WHERE name LIKE 'A%'
    """
    result3 = analyzer.analyze(sql3)
    json_outputs.append(("cte_query", analyzer.get_lineage_json(sql3)))
    
    if not print_lineage_analysis(result3, sql3, "3. CTE Query"):
        return False
    
    # Test 4: CREATE TABLE AS SELECT
    sql4 = "CREATE TABLE user_summary AS SELECT id, name, COUNT(*) as login_count FROM users GROUP BY id, name"
    result4 = analyzer.analyze(sql4)
    json_outputs.append(("ctas_query", analyzer.get_lineage_json(sql4)))
    
    if not print_lineage_analysis(result4, sql4, "4. CREATE TABLE AS SELECT Query"):
        return False
    
    # Test 5: Complex query with deeper chain for testing
    sql5 = """
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
    """
    result5 = analyzer.analyze(sql5)
    json_outputs.append(("complex_multi_cte_query", analyzer.get_lineage_json(sql5)))
    
    if not print_lineage_analysis(result5, sql5, "5. Complex Multi-CTE Query"):
        return False
    
    # Generate chain outputs for key queries
    chain_test_queries = [
        ("cte_query", sql3),
        ("complex_multi_cte_query", sql5)
    ]
    
    for query_name, sql in chain_test_queries:
        # Test table chains with different depths
        for depth in [1, 2, 3]:
            upstream_table_chain = analyzer.get_table_lineage_chain_json(sql, "upstream", depth)
            chain_outputs.append((query_name, "table", "upstream", depth, upstream_table_chain))
            
            downstream_table_chain = analyzer.get_table_lineage_chain_json(sql, "downstream", depth)
            chain_outputs.append((query_name, "table", "downstream", depth, downstream_table_chain))
        
        # Test column chains with different depths
        for depth in [1, 2, 3]:
            upstream_column_chain = analyzer.get_column_lineage_chain_json(sql, "upstream", depth)
            chain_outputs.append((query_name, "column", "upstream", depth, upstream_column_chain))
            
            downstream_column_chain = analyzer.get_column_lineage_chain_json(sql, "downstream", depth)
            chain_outputs.append((query_name, "column", "downstream", depth, downstream_column_chain))
    
    # Save JSON outputs to files
    save_json_outputs(json_outputs, "quick_test")
    save_chain_outputs(chain_outputs, "quick_test")
    
    # Create visualizations for key queries
    create_visualizations(analyzer, "quick_test")
    
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
            upstream_count = len(result.table_lineage.upstream.get("QUERY_RESULT", []))
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


def test_chain_functionality():
    """Test table and column lineage chain functionality."""
    print("\n🔗 Chain Functionality Test")
    print("─" * 40)
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # Complex multi-level CTE for chain testing
    sql = """
    WITH base_data AS (
        SELECT customer_id, order_total FROM orders WHERE order_date >= '2023-01-01'
    ),
    aggregated AS (
        SELECT customer_id, SUM(order_total) as total_spent FROM base_data GROUP BY customer_id
    ),
    classified AS (
        SELECT customer_id, total_spent,
               CASE WHEN total_spent > 1000 THEN 'VIP' ELSE 'Regular' END as tier
        FROM aggregated
    ),
    final_report AS (
        SELECT tier, COUNT(*) as customer_count, AVG(total_spent) as avg_spent
        FROM classified GROUP BY tier
    )
    SELECT * FROM final_report
    """
    
    try:
        # Test table upstream chain with depth 3
        upstream_table_chain = analyzer.get_table_lineage_chain(sql, "upstream", 3)
        print(f"✅ Table upstream chain (depth 3): {len(upstream_table_chain['chains'])} chains")
        
        # Test table downstream chain with depth 2
        downstream_table_chain = analyzer.get_table_lineage_chain(sql, "downstream", 2)
        print(f"✅ Table downstream chain (depth 2): {len(downstream_table_chain['chains'])} chains")
        
        # Test column upstream chain with depth 3
        upstream_column_chain = analyzer.get_column_lineage_chain(sql, "upstream", 3)
        print(f"✅ Column upstream chain (depth 3): {len(upstream_column_chain['chains'])} chains")
        
        # Test column downstream chain with depth 2
        downstream_column_chain = analyzer.get_column_lineage_chain(sql, "downstream", 2)
        print(f"✅ Column downstream chain (depth 2): {len(downstream_column_chain['chains'])} chains")
        
        # Test JSON output for both table and column chains
        table_json_output = analyzer.get_table_lineage_chain_json(sql, "upstream", 4)
        print(f"✅ Table chain JSON output: {len(table_json_output)} characters")
        
        column_json_output = analyzer.get_column_lineage_chain_json(sql, "upstream", 4)
        print(f"✅ Column chain JSON output: {len(column_json_output)} characters")
        
        # Test error handling for table chains
        try:
            analyzer.get_table_lineage_chain(sql, "invalid", 1)
            print("❌ Table chain error handling failed")
            return False
        except ValueError:
            print("✅ Table chain error handling works correctly")
        
        # Test error handling for column chains
        try:
            analyzer.get_column_lineage_chain(sql, "invalid", 1)
            print("❌ Column chain error handling failed")
            return False
        except ValueError:
            print("✅ Column chain error handling works correctly")
        
        print("✅ Chain functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Chain functionality test failed: {e}")
        return False


def main():
    """Main test runner."""
    print("🚀 SQL Lineage Quick Test Suite")
    print("=" * 60)
    
    # Create output directory with absolute path
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Run quick tests
        success_count = 0
        total_tests = 5
        
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
        
        # Run chain functionality test
        if test_chain_functionality():
            success_count += 1
        
        print("\n🎉 All quick tests completed!")
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
                
                print(f"\n✨ Quick tests generate integrated visualizations showing both")
                print(f"   table relationships and column-level lineage with SQL query context!")
            else:
                print("\n⚠️  No files were generated")
        
        # Print summary
        success = print_test_summary(total_tests, success_count, "Quick Tests")
        return 0 if success else 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the analyzer package:")
        print("   cd /Users/adhirpotdar/Work/git-repos/sql-lineage")
        print("   ./dev.sh install")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())