#!/usr/bin/env python3
"""
Test code to generate lineage JSON and JPEG visualization for insurance profitability query.

This script analyzes the following query:
SELECT pmf.policy_id, c.first_name, c.last_name, c.company_name, 
       SUM(pmf.earned_premium - pmf.paid_claims - pmf.underwriting_expense - pmf.commission_expense) AS total_profit, 
       AVG(pmf.combined_ratio) AS avg_combined_ratio 
FROM ins_sql.ins_fin.policy_monthly_financials pmf 
JOIN sf_ins.INSURANCE_OPS.POLICIES p ON pmf.policy_id = p.POLICY_ID 
JOIN sf_ins.INSURANCE_OPS.CUSTOMERS c ON p.CUSTOMER_ID = c.CUSTOMER_ID 
WHERE pmf.combined_ratio < 1.0 
GROUP BY pmf.policy_id, c.first_name, c.last_name, c.company_name 
ORDER BY total_profit DESC 
LIMIT 10
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
    """Create analyzer for insurance query analysis."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    print("📊 Using SQL-only analysis for insurance profitability query")
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


def test_insurance_profitability_query():
    """Test the insurance profitability query for lineage analysis."""
    print("🏢 Insurance Profitability Query Analysis")
    print("=" * 60)
    
    # The insurance profitability query
    insurance_sql = """
    SELECT pmf.policy_id, c.first_name, c.last_name, c.company_name, 
           SUM(pmf.earned_premium - pmf.paid_claims - pmf.underwriting_expense - pmf.commission_expense) AS total_profit, 
           AVG(pmf.combined_ratio) AS avg_combined_ratio 
    FROM ins_sql.ins_fin.policy_monthly_financials pmf 
    JOIN sf_ins.INSURANCE_OPS.POLICIES p ON pmf.policy_id = p.POLICY_ID 
    JOIN sf_ins.INSURANCE_OPS.CUSTOMERS c ON p.CUSTOMER_ID = c.CUSTOMER_ID 
    WHERE pmf.combined_ratio < 1.0 
    GROUP BY pmf.policy_id, c.first_name, c.last_name, c.company_name 
    ORDER BY total_profit DESC 
    LIMIT 10
    """
    
    analyzer = create_analyzer()
    
    try:
        print("\n🔍 Analyzing insurance profitability query...")
        
        # Step 1: Basic analysis
        result = analyzer.analyze(insurance_sql)
        
        if result.has_errors():
            print("❌ Query analysis failed:")
            for error in result.errors:
                print(f"   • {error}")
            return False
        
        print("✅ Query analysis completed successfully")
        print(f"   • Upstream tables: {len(result.table_lineage.upstream.get('QUERY_RESULT', []))}")
        print(f"   • Downstream relationships: {len(result.table_lineage.downstream)}")
        print(f"   • Column relationships: {len(result.column_lineage.upstream)}")
        
        # Step 2: Generate comprehensive lineage JSON
        print("\n📋 Generating comprehensive lineage chain JSON...")
        
        lineage_json = analyzer.get_lineage_chain_json(insurance_sql, "downstream")
        
        # Validate JSON structure
        parsed_json = json.loads(lineage_json)
        
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
        json_file = save_json_output(lineage_json, "insurance_profitability_lineage_chain.json")
        
        if not json_file:
            print("❌ Failed to save JSON output")
            return False
        
        # Step 4: Generate JPEG visualization
        print("\n🎨 Generating JPEG visualization...")
        jpeg_file = generate_jpeg_visualization(lineage_json, "insurance_profitability_visualization")
        
        if not jpeg_file:
            print("❌ Failed to generate JPEG visualization")
            return False
        
        # Step 5: Print detailed analysis
        print("\n📊 Detailed Analysis Results:")
        print("-" * 40)
        
        # Show source tables
        if parsed_json.get("chains"):
            print("\n🏦 Source Tables Identified:")
            for entity_name, chain_data in parsed_json["chains"].items():
                if "metadata" in chain_data and chain_data["metadata"]:
                    table_type = chain_data["metadata"].get("table_type", "unknown")
                    print(f"   • {entity_name} ({table_type})")
                    
                    # Show columns if available
                    if "columns" in chain_data["metadata"]:
                        columns = chain_data["metadata"]["columns"]
                        if len(columns) > 5:
                            print(f"     Columns: {', '.join(columns[:5])}... ({len(columns)} total)")
                        else:
                            print(f"     Columns: {', '.join(columns)}")
        
        # Show transformations
        transformation_count = 0
        for entity_name, chain_data in parsed_json["chains"].items():
            if "transformations" in chain_data:
                transformation_count += len(chain_data["transformations"])
        
        if transformation_count > 0:
            print(f"\n🔄 Transformations Found: {transformation_count}")
            for entity_name, chain_data in parsed_json["chains"].items():
                if "transformations" in chain_data and chain_data["transformations"]:
                    for transform in chain_data["transformations"]:
                        transform_type = transform.get("type", "unknown")
                        transform_desc = transform.get("description", "")
                        print(f"   • {entity_name}: {transform_type} - {transform_desc}")
        
        print(f"\n🎉 Insurance profitability query analysis completed successfully!")
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
    print("🚀 Insurance Profitability Query Lineage Test")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Run the test
    success = test_insurance_profitability_query()
    
    if success:
        print("\n✅ Test completed successfully!")
        
        # List generated files
        output_files = [f for f in os.listdir(output_dir) if f.startswith('insurance_profitability')]
        if output_files:
            print(f"\n📄 Generated files:")
            for file in sorted(output_files):
                filepath = os.path.join(output_dir, file)
                size = os.path.getsize(filepath)
                print(f"   • {file} ({size:,} bytes)")
        
        print(f"\n💡 The lineage JSON contains comprehensive table and column relationships")
        print(f"   showing how data flows from source tables to the final result.")
        print(f"   The JPEG visualization provides a visual representation of these relationships.")
        
        return 0
    else:
        print("\n❌ Test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())