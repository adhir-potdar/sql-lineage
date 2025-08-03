#!/usr/bin/env python3
"""
Test script for SQLLineageVisualizer.
Demonstrates creating visual lineage diagrams from JSON chain data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.visualization import SQLLineageVisualizer, create_lineage_visualization


def test_basic_visualization():
    """Test basic visualization functionality."""
    print("🎨 Testing SQLLineageVisualizer")
    print("=" * 50)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Complex CTE query for testing
    sql = """
    WITH order_stats AS (
        SELECT 
            customer_id,
            COUNT(*) as order_count,
            SUM(order_total) as total_spent
        FROM orders
        WHERE order_date >= '2023-01-01'
        GROUP BY customer_id
    ),
    customer_segments AS (
        SELECT 
            os.customer_id,
            os.order_count,
            os.total_spent,
            u.name,
            CASE 
                WHEN os.total_spent > 1000 THEN 'Premium'
                WHEN os.total_spent > 500 THEN 'Standard'
                ELSE 'Basic'
            END as segment
        FROM order_stats os
        JOIN users u ON os.customer_id = u.id
    ),
    final_report AS (
        SELECT 
            segment,
            COUNT(*) as customer_count,
            AVG(total_spent) as avg_spent
        FROM customer_segments
        GROUP BY segment
    )
    SELECT * FROM final_report ORDER BY avg_spent DESC
    """
    
    # Create visualizer
    visualizer = SQLLineageVisualizer()
    
    # Test comprehensive JPEG generation for table and column chains
    chain_types = ['upstream', 'downstream']
    lineage_types = ['table', 'column', 'combined']
    
    for chain_type in chain_types:
        for lineage_type in lineage_types:
            print(f"\n📊 Creating {chain_type} {lineage_type} chain visualization (JPEG)...")
            try:
                if lineage_type == 'table':
                    # Table-only visualization
                    chain_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
                    output_file = visualizer.create_table_only_diagram(
                        table_chain_json=chain_json,
                        output_path=os.path.join(output_dir, f"{chain_type}_table_lineage_jpeg"),
                        output_format="jpeg",
                        sql_query=sql
                    )
                elif lineage_type == 'column':
                    # Column-focused visualization
                    table_chain_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
                    column_chain_json = analyzer.get_column_lineage_chain_json(sql, chain_type, 3)
                    output_file = visualizer.create_column_focused_diagram(
                        table_chain_json=table_chain_json,
                        column_chain_json=column_chain_json,
                        output_path=os.path.join(output_dir, f"{chain_type}_column_lineage_jpeg"),
                        output_format="jpeg",
                        sql_query=sql
                    )
                else:  # combined
                    # Combined table + column visualization
                    table_chain_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
                    column_chain_json = analyzer.get_column_lineage_chain_json(sql, chain_type, 3)
                    output_file = visualizer.create_lineage_diagram(
                        table_chain_json=table_chain_json,
                        column_chain_json=column_chain_json,
                        output_path=os.path.join(output_dir, f"{chain_type}_combined_lineage_jpeg"),
                        output_format="jpeg",
                        show_columns=True,
                        sql_query=sql
                    )
                
                print(f"✅ Created {chain_type} {lineage_type} JPEG: {output_file}")
            except Exception as e:
                print(f"❌ Failed to create {chain_type} {lineage_type} JPEG: {e}")
    
    # Test convenience function
    print("\n📊 Testing convenience function...")
    try:
        upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 2)
        
        output_file = create_lineage_visualization(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "convenience_test"),
            output_format="jpeg",
            show_columns=False,
            sql_query=sql
        )
        print(f"✅ Created diagram using convenience function: {output_file}")
    except Exception as e:
        print(f"❌ Failed with convenience function: {e}")
    
    # Test horizontal vs vertical layout comparison
    print("\n📊 Creating layout comparison (horizontal vs vertical)...")
    try:
        upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 3)
        
        # Horizontal layout (default)
        horizontal_file = visualizer.create_table_only_diagram(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "layout_horizontal"),
            output_format="jpeg",
            layout="horizontal",
            sql_query=sql
        )
        print(f"✅ Created horizontal layout: {horizontal_file}")
        
        # Vertical layout
        vertical_file = visualizer.create_table_only_diagram(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "layout_vertical"),
            output_format="jpeg",
            layout="vertical",
            sql_query=sql
        )
        print(f"✅ Created vertical layout: {vertical_file}")
        
    except Exception as e:
        print(f"❌ Failed to create layout comparison: {e}")
    
    # Test integrated column lineage visualization
    print("\n📊 Creating integrated column lineage visualization...")
    try:
        upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 3)
        upstream_column_json = analyzer.get_column_lineage_chain_json(sql, "upstream", 3)
        
        # Integrated visualization (new approach)
        integrated_file = visualizer.create_lineage_diagram(
            table_chain_json=upstream_table_json,
            column_chain_json=upstream_column_json,
            output_path=os.path.join(output_dir, "integrated_column_lineage"),
            output_format="jpeg",
            show_columns=True,
            layout="horizontal",
            sql_query=sql
        )
        print(f"✅ Created integrated column lineage: {integrated_file}")
        
    except Exception as e:
        print(f"❌ Failed to create integrated column lineage: {e}")
    
    # Test combined column lineage with different complexity levels
    print("\n📊 Testing combined column lineage with different SQL complexities...")
    
    combined_test_queries = [
        ("simple_combined", """
        SELECT 
            u.name as customer_name,
            u.email,
            o.order_total,
            o.order_date
        FROM users u
        JOIN orders o ON u.id = o.customer_id
        WHERE u.active = true
        """),
        ("cte_combined", """
        WITH customer_stats AS (
            SELECT 
                customer_id,
                COUNT(*) as order_count,
                SUM(order_total) as total_spent
            FROM orders 
            GROUP BY customer_id
        )
        SELECT 
            u.name,
            cs.order_count,
            cs.total_spent,
            CASE WHEN cs.total_spent > 1000 THEN 'VIP' ELSE 'Regular' END as tier
        FROM users u
        JOIN customer_stats cs ON u.id = cs.customer_id
        """),
        ("complex_combined", """
        WITH user_profiles AS (
            SELECT id, name, email, age FROM users WHERE active = true
        ),
        order_metrics AS (
            SELECT 
                customer_id,
                COUNT(*) as order_count,
                SUM(order_total) as total_revenue,
                AVG(order_total) as avg_order_value
            FROM orders 
            GROUP BY customer_id
        ),
        customer_segments AS (
            SELECT 
                up.id,
                up.name,
                up.email,
                om.order_count,
                om.total_revenue,
                CASE 
                    WHEN om.total_revenue > 2000 THEN 'Premium'
                    WHEN om.total_revenue > 1000 THEN 'Gold'
                    ELSE 'Standard'
                END as segment
            FROM user_profiles up
            LEFT JOIN order_metrics om ON up.id = om.customer_id
        )
        SELECT 
            name as customer_name,
            email,
            segment,
            order_count,
            total_revenue,
            ROUND(total_revenue / NULLIF(order_count, 0), 2) as revenue_per_order
        FROM customer_segments
        WHERE order_count > 0
        """)
    ]
    
    for query_name, test_sql in combined_test_queries:
        print(f"\n📊 Creating combined diagram for {query_name}...")
        try:
            # Get chain data
            table_json = analyzer.get_table_lineage_chain_json(test_sql, "upstream", 3)
            column_json = analyzer.get_column_lineage_chain_json(test_sql, "upstream", 3)
            
            # Create combined column lineage diagram (like the sample image) - JPEG only
            combined_output = visualizer.create_lineage_diagram(
                table_chain_json=table_json,
                column_chain_json=column_json,
                output_path=os.path.join(output_dir, f"combined_column_lineage_{query_name}"),
                output_format="jpeg",
                show_columns=True,  # This creates the integrated column+table view
                layout="horizontal",  # Tables flow horizontally, columns vertically within tables
                sql_query=test_sql
            )
            print(f"   ✅ Created combined column lineage: {os.path.basename(combined_output)}")
            
        except Exception as e:
            print(f"   ❌ Failed to create {query_name}: {e}")


def test_custom_styling():
    """Test custom styling options."""
    print("\n🎨 Testing custom styling...")
    print("=" * 30)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Simple query for styling test
    sql = """
    WITH active_users AS (
        SELECT id, name FROM users WHERE active = true
    )
    SELECT au.name, COUNT(o.id) as order_count
    FROM active_users au
    LEFT JOIN orders o ON au.id = o.customer_id
    GROUP BY au.name
    """
    
    # Custom configuration
    custom_config = {
        'node_style': {
            'table': {
                'fillcolor': '#FFE0B2',
                'color': '#FF8F00',
                'fontcolor': '#E65100'
            },
            'cte': {
                'fillcolor': '#C8E6C9',
                'color': '#388E3C',
                'fontcolor': '#1B5E20'
            },
            'query_result': {
                'fillcolor': '#FFCDD2',
                'color': '#D32F2F',
                'fontcolor': '#B71C1C'
            }
        },
        'graph_attributes': {
            'size': '10,8',
            'bgcolor': '#F5F5F5'
        }
    }
    
    try:
        visualizer = SQLLineageVisualizer()
        upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 2)
        
        output_file = visualizer.create_table_only_diagram(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "custom_styled_lineage"),
            output_format="jpeg",
            config=custom_config,
            sql_query=sql
        )
        print(f"✅ Created custom styled diagram: {output_file}")
    except Exception as e:
        print(f"❌ Failed to create custom styled diagram: {e}")


def test_different_formats():
    """Test different output formats."""
    print("\n📁 Testing different output formats...")
    print("=" * 40)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    
    visualizer = SQLLineageVisualizer()
    upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 2)
    
    # Test only JPEG format now
    try:
        output_file = visualizer.create_table_only_diagram(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "format_test_jpeg"),
            output_format="jpeg",
            sql_query=sql
        )
        print(f"✅ Created JPEG format: {output_file}")
    except Exception as e:
        print(f"❌ Failed to create JPEG format: {e}")


def main():
    """Main test runner."""
    print("🚀 SQL Lineage Visualizer Test Suite")
    print("=" * 60)
    
    # Create output directory with absolute path
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Check if graphviz is available
        try:
            from graphviz import Digraph
            print("✅ Graphviz Python package is available")
        except ImportError:
            raise ImportError("graphviz package not found. Install with: pip install graphviz")
        
        # Run tests
        test_basic_visualization()
        test_custom_styling()
        test_different_formats()
        
        print("\n🎉 All visualization tests completed!")
        print(f"\n📁 Check the '{output_dir}' directory for generated diagrams.")
        
        # List generated files with categories
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.jpeg')]
            if files:
                print(f"\n📄 Generated files ({len(files)}):")
                
                # Categorize files
                combined_files = [f for f in files if 'combined' in f]
                table_files = [f for f in files if 'table' in f and 'combined' not in f]
                other_files = [f for f in files if f not in combined_files and f not in table_files]
                
                if combined_files:
                    print(f"\n   🎨 Combined Column Lineage Diagrams ({len(combined_files)}):")
                    for file in sorted(combined_files):
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"     • {file} ({size:,} bytes)")
                
                if table_files:
                    print(f"\n   📊 Table-Only Diagrams ({len(table_files)}):")
                    for file in sorted(table_files)[:5]:  # Show first 5
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"     • {file} ({size:,} bytes)")
                    if len(table_files) > 5:
                        print(f"     ... and {len(table_files) - 5} more table diagrams")
                
                if other_files:
                    print(f"\n   🔧 Other Test Files ({len(other_files)}):")
                    for file in sorted(other_files)[:3]:  # Show first 3
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"     • {file} ({size:,} bytes)")
                    if len(other_files) > 3:
                        print(f"     ... and {len(other_files) - 3} more files")
                
                print(f"\n✨ **ENHANCED**: Combined column lineage diagrams show SQL query context")
                print(f"   at the top, with clear distinction between CTEs, source tables, and")
                print(f"   'Final Query Result' for complete lineage traceability!")
            else:
                print("\n⚠️  No visualization files were generated")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you have installed graphviz:")
        print("   pip install graphviz")
        print("   # And install system graphviz:")
        print("   # Ubuntu: sudo apt-get install graphviz")
        print("   # macOS: brew install graphviz")
        print("   # Windows: choco install graphviz")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()