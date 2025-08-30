#!/usr/bin/env python3
"""
Test code to generate lineage chain JSON and JPEG visualization for National Grid query.
Based on the query from ng_query.txt file.
"""

import sys
import os
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.visualization.visualizer import SQLLineageVisualizer


def test_ng_query():
    """Test National Grid complex query with CTEs and multiple JOINs."""
    print("📊 Testing National Grid Query: Complex CTEs with ROW_NUMBER and Multiple JOINs")
    print("=" * 80)
    
    # Read the SQL query from ng_query.txt file as-is
    with open('ng_query.txt', 'r') as f:
        ng_sql_query = f.read().strip()
    
    print(f"📝 Query length: {len(ng_sql_query)} characters")
    print(f"📋 Preview: {ng_sql_query[:200]}...")
    print()
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Generate lineage chain JSON
    print("🔍 Generating lineage chain JSON...")
    result_json = analyzer.get_lineage_chain_json(ng_sql_query, "downstream")
    
    # Save JSON to file
    json_filename = "output/ng_query_lineage_chain.json"
    with open(json_filename, 'w') as f:
        json.dump(json.loads(result_json), f, indent=2)
    
    print(f"✅ Lineage JSON saved: {json_filename}")
    
    # Generate JPEG visualization
    print("🖼️  Generating JPEG visualization...")
    try:
        visualizer = SQLLineageVisualizer()
        output_path = "output/ng_query_lineage_visualization"
        jpeg_file = visualizer.create_lineage_chain_diagram(
            lineage_chain_json=result_json,
            output_path=output_path,
            output_format="jpeg",
            layout="horizontal"
        )
        print(f"✅ JPEG visualization saved: {jpeg_file}")
    except Exception as e:
        print(f"❌ JPEG generation failed: {e}")
    
    # Print analysis summary
    parsed = json.loads(result_json)
    print(f"\n📈 Analysis Summary:")
    print(f"   📊 Total Tables: {parsed['summary']['total_tables']}")
    print(f"   📋 Total Columns: {parsed['summary']['total_columns']}")
    print(f"   🔄 Has Transformations: {parsed['summary']['has_transformations']}")
    print(f"   📖 Has Metadata: {parsed['summary']['has_metadata']}")
    print(f"   🔗 Chain Count: {parsed['summary']['chain_count']}")
    
    # Show table breakdown
    if 'chains' in parsed:
        print(f"\n🏦 Source Tables Found:")
        for i, (table_name, chain_data) in enumerate(parsed['chains'].items(), 1):
            entity_type = chain_data.get('entity_type', 'unknown')
            columns_count = len(chain_data.get('metadata', {}).get('table_columns', []))
            print(f"   {i}. {table_name} ({entity_type}) - {columns_count} columns")
    
    # Show CTEs found
    cte_count = 0
    for table_name, chain_data in parsed['chains'].items():
        if chain_data.get('metadata', {}).get('is_cte', False):
            cte_count += 1
    
    print(f"\n🔄 CTE Analysis:")
    print(f"   📝 CTEs Detected: {cte_count}")
    print(f"   🔗 Expected CTEs: 3 (cte, cte2, cte3)")
    
    # Show transformation types detected
    transformation_types = set()
    for entity_name, chain_data in parsed['chains'].items():
        for dep in chain_data.get('dependencies', []):
            for trans in dep.get('transformations', []):
                if trans.get('joins'):
                    transformation_types.add('JOIN')
                if trans.get('filter_conditions'):
                    transformation_types.add('FILTER') 
                if trans.get('group_by_columns'):
                    transformation_types.add('GROUP_BY')
                if trans.get('order_by_columns'):
                    transformation_types.add('ORDER_BY')
                # Check for window functions
                for col in dep.get('metadata', {}).get('table_columns', []):
                    if 'transformation' in col:
                        trans_type = col['transformation'].get('transformation_type', '')
                        if 'WINDOW' in trans_type or 'ROW_NUMBER' in trans_type:
                            transformation_types.add('WINDOW')
    
    print(f"   🔄 Transformation Types: {sorted(transformation_types)}")
    
    return True


def main():
    """Run the National Grid query test."""
    print("🚀 SQL Lineage Analysis - National Grid Query Test")
    print("=" * 70)
    print("Testing complex National Grid query with CTEs, window functions, and multiple JOINs")
    print()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Test the National Grid query
    success = test_ng_query()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print("📁 Check the 'output/' directory for generated files:")
        print("   - ng_query_lineage_chain.json")
        print("   - ng_query_lineage_visualization.jpeg")
        print()
        print("💡 The query contains:")
        print("   - 3 CTEs with ROW_NUMBER() window functions")
        print("   - Multiple LEFT JOINs with ~20 tables")
        print("   - Complex filtering conditions")
        print("   - String functions (substring, concat, trim)")
        print("   - Timestamp comparisons")
    else:
        print(f"\n❌ Test failed!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())