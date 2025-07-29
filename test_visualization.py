#!/usr/bin/env python3
"""
Test script for SQLLineageVisualizer.
Demonstrates creating visual lineage diagrams from JSON chain data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.metadata import SampleMetadataRegistry
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
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
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
    
    # Test comprehensive format generation for table and column chains
    formats = ['png', 'jpeg', 'pdf']
    chain_types = ['upstream', 'downstream']
    lineage_types = ['table', 'column', 'combined']
    
    for chain_type in chain_types:
        for lineage_type in lineage_types:
            for fmt in formats:
                print(f"\n📊 Creating {chain_type} {lineage_type} chain visualization ({fmt.upper()})...")
                try:
                    if lineage_type == 'table':
                        # Table-only visualization
                        chain_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
                        output_file = visualizer.create_table_only_diagram(
                            table_chain_json=chain_json,
                            output_path=os.path.join(output_dir, f"{chain_type}_table_lineage_{fmt}"),
                            output_format=fmt
                        )
                    elif lineage_type == 'column':
                        # Column-focused visualization
                        table_chain_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
                        column_chain_json = analyzer.get_column_lineage_chain_json(sql, chain_type, 3)
                        output_file = visualizer.create_column_focused_diagram(
                            table_chain_json=table_chain_json,
                            column_chain_json=column_chain_json,
                            output_path=os.path.join(output_dir, f"{chain_type}_column_lineage_{fmt}"),
                            output_format=fmt
                        )
                    else:  # combined
                        # Combined table + column visualization
                        table_chain_json = analyzer.get_table_lineage_chain_json(sql, chain_type, 3)
                        column_chain_json = analyzer.get_column_lineage_chain_json(sql, chain_type, 3)
                        output_file = visualizer.create_lineage_diagram(
                            table_chain_json=table_chain_json,
                            column_chain_json=column_chain_json,
                            output_path=os.path.join(output_dir, f"{chain_type}_combined_lineage_{fmt}"),
                            output_format=fmt,
                            show_columns=True
                        )
                    
                    print(f"✅ Created {chain_type} {lineage_type} {fmt.upper()}: {output_file}")
                except Exception as e:
                    print(f"❌ Failed to create {chain_type} {lineage_type} {fmt.upper()}: {e}")
    
    # Test convenience function
    print("\n📊 Testing convenience function...")
    try:
        upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 2)
        
        output_file = create_lineage_visualization(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "convenience_test"),
            output_format="svg",
            show_columns=False
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
            output_format="png",
            layout="horizontal"
        )
        print(f"✅ Created horizontal layout: {horizontal_file}")
        
        # Vertical layout
        vertical_file = visualizer.create_table_only_diagram(
            table_chain_json=upstream_table_json,
            output_path=os.path.join(output_dir, "layout_vertical"),
            output_format="png",
            layout="vertical"
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
            output_format="png",
            show_columns=True,
            layout="horizontal"
        )
        print(f"✅ Created integrated column lineage: {integrated_file}")
        
    except Exception as e:
        print(f"❌ Failed to create integrated column lineage: {e}")


def test_custom_styling():
    """Test custom styling options."""
    print("\n🎨 Testing custom styling...")
    print("=" * 30)
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
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
            output_format="png",
            config=custom_config
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
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    
    visualizer = SQLLineageVisualizer()
    upstream_table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", 2)
    
    formats = ['png', 'svg', 'pdf', 'jpg']
    
    for fmt in formats:
        try:
            output_file = visualizer.create_table_only_diagram(
                table_chain_json=upstream_table_json,
                output_path=os.path.join(output_dir, f"format_test_{fmt}"),
                output_format=fmt
            )
            print(f"✅ Created {fmt.upper()} format: {output_file}")
        except Exception as e:
            print(f"❌ Failed to create {fmt.upper()} format: {e}")


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
        
        # List generated files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.svg', '.pdf', '.jpg', '.jpeg'))]
            if files:
                print(f"\n📄 Generated files ({len(files)}):")
                for file in sorted(files):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"   • {file} ({size:,} bytes)")
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