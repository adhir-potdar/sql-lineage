#!/usr/bin/env python3
"""
Test code to generate lineage JSON and JPEG visualization for CTE query with TOP clause.

This script analyzes the following query which causes sqlglot parsing issues:
WITH select_step1 as (
    SELECT
      "orders"."order_id" AS "order_id",
        CAST(YEAR("order_date") AS varchar) AS "year of order"
       FROM
      "customer_demo"."orders"
)
SELECT
       TOP 100 * FROM
       select_step1
"""

import sys
import os
import json
import traceback

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.visualization.visualizer import SQLLineageVisualizer

def create_analyzer():
    """Create analyzer for CTE query analysis."""
    analyzer = SQLLineageAnalyzer(dialect="trino")  # Start with Trino to test auto-detection
    print("📊 Using analyzer.get_lineage_chain_json for CTE query with TOP clause (Trino dialect - should auto-detect to TSQL)")
    return analyzer

def save_json_output(json_output, filename):
    """Save JSON output to file."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"📁 Saved JSON output to: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to save {filepath}: {e}")
        return None

def generate_jpeg_visualization(json_output, output_name):
    """Generate JPEG visualization from lineage chain JSON."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    visualizer = SQLLineageVisualizer()
    
    try:
        output_path = os.path.join(output_dir, output_name)
        
        # Use the lineage chain visualization method
        jpeg_file = visualizer.create_lineage_chain_diagram(
            lineage_chain_json=json_output,
            output_path=output_path,
            output_format="jpeg",
            layout="horizontal"
        )
        
        print(f"🖼️  Generated JPEG visualization: {jpeg_file}")
        return jpeg_file
        
    except Exception as e:
        print(f"❌ Failed to generate visualization: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        return None

def test_cte_top_query():
    """Test the CTE query with TOP clause for lineage analysis."""
    print("📋 CTE Query with TOP Clause Analysis")
    print("=" * 60)
    
    # The CTE query with TOP clause that causes sqlglot issues
    cte_sql = """
    WITH select_step1 as (
        SELECT
          "orders"."order_id" AS "order_id",
            CAST(YEAR("order_date") AS varchar) AS "year of order"
           FROM
          "customer_demo"."orders"
    )
    SELECT
           TOP 100 * FROM
           select_step1
    """
    
    analyzer = create_analyzer()
    
    try:
        print("\n🔍 Analyzing CTE query with TOP clause...")
        
        # Step 1: Generate lineage chain JSON directly using analyzer
        lineage_json = analyzer.get_lineage_chain_json(cte_sql, "downstream", 0)
        parsed_json = json.loads(lineage_json)
        
        print("✅ Lineage chain analysis completed successfully")
        print(f"   • Chains generated: {len(parsed_json.get('chains', {}))}")
        print(f"   • Final dialect used: {analyzer.dialect}")
        print(f"   • Chain type: {parsed_json.get('chain_type')}")
        print(f"   • Max depth: {parsed_json.get('max_depth')}")
        
        # Check for dialect auto-correction by comparing initial vs final dialects
        if analyzer.dialect == 'tsql':
            print("\n🔄 Dialect Auto-Correction Occurred:")
            print(f"   • trino → {analyzer.dialect}")
            print(f"   • Analyzer dialect updated via lineage_chain_builder")
            print(f"   • Reason: Detected TOP clause in CTE query")
        else:
            print(f"\n   • Final analyzer dialect: {analyzer.dialect}")
        
        # Step 2: Validate the lineage JSON structure
        print(f"\n📋 Validating lineage chain JSON structure...")
        print(f"   • Current analyzer dialect: {analyzer.dialect}")
        
        # lineage_json already generated in Step 1
        
        # JSON structure already parsed in Step 1
        
        required_keys = ["sql", "dialect", "chain_type", "max_depth", "actual_max_depth", "chains", "summary"]
        for key in required_keys:
            if key not in parsed_json:
                print(f"   ⚠️  Missing key '{key}' in JSON output")
                return False
        
        chains_count = len(parsed_json.get("chains", {}))
        summary = parsed_json.get("summary", {})
        max_depth = parsed_json.get("max_depth", "unknown")
        actual_depth = parsed_json.get("actual_max_depth", 0)
        
        print(f"✅ Lineage chain JSON generated successfully")
        print(f"   • Chains: {chains_count}")
        print(f"   • JSON size: {len(lineage_json):,} characters")
        print(f"   • Tables: {summary.get('total_tables', 0)}")
        print(f"   • Columns: {summary.get('total_columns', 0)}")
        print(f"   • Has transformations: {summary.get('has_transformations', False)}")
        print(f"   • Has metadata: {summary.get('has_metadata', False)}")
        print(f"   • Max depth: {max_depth}, Actual depth: {actual_depth}")
        
        # Step 3: Save JSON output
        print("\n💾 Saving lineage JSON...")
        json_file = save_json_output(lineage_json, "cte_top_query_lineage_chain.json")
        
        if not json_file:
            print("❌ Failed to save JSON output")
            return False
        
        # Step 4: Generate JPEG visualization
        print("\n🎨 Generating JPEG visualization...")
        jpeg_file = generate_jpeg_visualization(lineage_json, "cte_top_query_visualization")
        
        if not jpeg_file:
            print("❌ Failed to generate JPEG visualization")
            return False
        
        # Step 5: Print detailed analysis
        print("\n📊 Detailed Analysis Results:")
        print("-" * 40)
        
        # Show source tables
        if parsed_json.get("chains"):
            print("\n🏦 Source Tables and CTEs Identified:")
            for entity_name, chain_data in parsed_json["chains"].items():
                entity_type = chain_data.get("entity_type", "unknown")
                depth = chain_data.get("depth", 0)
                print(f"   • {entity_name} ({entity_type}, depth: {depth})")
                
                if "metadata" in chain_data and chain_data["metadata"]:
                    table_type = chain_data["metadata"].get("table_type", "unknown")
                    if table_type != "unknown":
                        print(f"     Table type: {table_type}")
                    
                    # Show columns if available
                    table_columns = chain_data["metadata"].get("table_columns", [])
                    if table_columns:
                        column_names = [col.get("name", "unknown") for col in table_columns]
                        if len(column_names) > 5:
                            print(f"     Columns: {', '.join(column_names[:5])}... ({len(column_names)} total)")
                        else:
                            print(f"     Columns: {', '.join(column_names)}")
        
        # Show transformations and dependencies
        transformation_count = 0
        cte_count = 0
        for entity_name, chain_data in parsed_json["chains"].items():
            if chain_data.get("entity_type") == "cte":
                cte_count += 1
            if "dependencies" in chain_data:
                for dep in chain_data["dependencies"]:
                    if "transformations" in dep:
                        transformation_count += len(dep["transformations"])
        
        if cte_count > 0:
            print(f"\n📋 CTEs Found: {cte_count}")
        
        if transformation_count > 0:
            print(f"\n🔄 Transformations Found: {transformation_count}")
            for entity_name, chain_data in parsed_json["chains"].items():
                if "dependencies" in chain_data:
                    for dep in chain_data["dependencies"]:
                        if "transformations" in dep and dep["transformations"]:
                            for transform in dep["transformations"]:
                                transform_type = transform.get("type", "unknown")
                                source = transform.get("source_table", "unknown")
                                target = transform.get("target_table", "unknown")
                                print(f"   • {source} → {target}: {transform_type}")
        
        # Check for TOP clause handling
        print("\n🔝 TOP Clause Analysis:")
        top_found = False
        for entity_name, chain_data in parsed_json["chains"].items():
            if "dependencies" in chain_data:
                for dep in chain_data["dependencies"]:
                    if "transformations" in dep:
                        for transform in dep["transformations"]:
                            if "limiting" in transform:
                                limiting = transform["limiting"]
                                if limiting.get("limit"):
                                    print(f"   • Limit found: {limiting['limit']}")
                                    top_found = True
                            elif "TOP" in str(transform):
                                print(f"   • TOP clause detected in transformation")
                                top_found = True
        
        if not top_found:
            print("   • No TOP/LIMIT clause found in transformations")
        
        print(f"\n🎉 CTE query with TOP clause analysis completed successfully!")
        print(f"📁 Files generated:")
        print(f"   • JSON: {os.path.basename(json_file)}")
        print(f"   • JPEG: {os.path.basename(jpeg_file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test runner."""
    print("🚀 CTE Query with TOP Clause Lineage Test")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Run the test
    success = test_cte_top_query()
    
    if success:
        print("\n✅ Test completed successfully!")
        
        # List generated files
        output_files = [f for f in os.listdir(output_dir) if f.startswith('cte_top_query')]
        if output_files:
            print(f"\n📄 Generated files:")
            for file in sorted(output_files):
                filepath = os.path.join(output_dir, file)
                size = os.path.getsize(filepath)
                print(f"   • {file} ({size:,} bytes)")
        
        print(f"\n💡 The lineage JSON contains comprehensive table and column relationships")
        print(f"   including CTE (Common Table Expression) handling and TOP clause analysis.")
        print(f"   The JPEG visualization provides a visual representation of the CTE dependencies.")
        
        return 0
    else:
        print("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
