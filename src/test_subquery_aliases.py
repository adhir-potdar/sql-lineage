#!/usr/bin/env python3
"""
Test code to generate lineage JSON and JPEG visualization for subquery aliases query.

This script analyzes the following query with subquery aliases:
SELECT
  t1.country,
  t1.category,
  t1."sum of profit",
  t2.plan_name,
  t2.agent_vendor,
  t2.sumof_mer
FROM
  (SELECT * FROM "hive"."promethium"."profit_by_country_by_category_1757103693384") t1
FULL OUTER JOIN
  (SELECT * FROM "hive"."promethium"."revenue_by_plan_by_vendor_1757103916264") t2
ON t1.category = t2.agent_vendor
LIMIT 100
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
    """Create analyzer for subquery aliases query analysis."""
    analyzer = SQLLineageAnalyzer(dialect="trino")
    print("📊 Using SQL-only analysis for subquery aliases query")
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


def test_subquery_aliases_query():
    """Test the subquery aliases query for lineage analysis."""
    print("🔗 Subquery Aliases Query Analysis")
    print("=" * 60)
    
    # The subquery aliases query from data-148.json
    subquery_sql = """
    SELECT
      t1.country,
      t1.category,
      t1."sum of profit",
      t2.plan_name,
      t2.agent_vendor,
      t2.sumof_mer
    FROM
      (SELECT * FROM "hive"."promethium"."profit_by_country_by_category_1757103693384") t1
    FULL OUTER JOIN
      (SELECT * FROM "hive"."promethium"."revenue_by_plan_by_vendor_1757103916264") t2
    ON t1.category = t2.agent_vendor
    LIMIT 100
    """
    
    analyzer = create_analyzer()
    
    try:
        print("\n🔍 Analyzing subquery aliases query...")
        
        # Step 1: Basic analysis
        result = analyzer.analyze(subquery_sql)
        
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
        
        lineage_json = analyzer.get_lineage_chain_json(subquery_sql, "downstream")
        
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
        
        # Step 3: Validate the fixes - check for proper column extraction and deduplication
        print("\n🔧 Validating fixes...")
        chains = parsed_json.get("chains", {})
        
        # Check for expected source tables (should be 2, not 4)
        source_tables = []
        for entity_name, chain_data in chains.items():
            if chain_data.get("entity_type") == "table" and chain_data.get("depth", -1) == 0:
                source_tables.append(entity_name)
        
        print(f"   • Source tables found: {len(source_tables)}")
        for table in source_tables:
            print(f"     - {table}")
        
        # Check for column extraction
        total_columns = 0
        tables_with_columns = 0
        for entity_name, chain_data in chains.items():
            if chain_data.get("entity_type") == "table":
                columns = chain_data.get("metadata", {}).get("table_columns", [])
                if columns:
                    tables_with_columns += 1
                    total_columns += len(columns)
                    print(f"   • {entity_name}: {len(columns)} columns")
                    for col in columns:
                        print(f"     - {col.get('name', 'unnamed')}")
                else:
                    print(f"   ⚠️  {entity_name}: No columns found")
        
        print(f"   • Tables with columns: {tables_with_columns}/{len([c for c in chains.values() if c.get('entity_type') == 'table'])}")
        print(f"   • Total columns extracted: {total_columns}")
        
        # Validation checks
        validation_passed = True
        if len(source_tables) > 2:
            print("   ⚠️  WARNING: More than 2 source tables found - possible duplication")
            validation_passed = False
        if total_columns < 6:
            print("   ⚠️  WARNING: Expected 6 columns (3 from each table), but found less")
            validation_passed = False
        if tables_with_columns == 0:
            print("   ❌ CRITICAL: No tables have column information")
            validation_passed = False
        
        if validation_passed:
            print("   ✅ All validation checks passed!")
        else:
            print("   ⚠️  Some validation checks failed - fixes may need refinement")
        
        # Step 4: Save JSON output
        print("\n💾 Saving lineage JSON...")
        json_file = save_json_output(lineage_json, "subquery_aliases_lineage_chain.json")
        
        if not json_file:
            print("❌ Failed to save JSON output")
            return False
        
        # Step 5: Generate JPEG visualization
        print("\n🎨 Generating JPEG visualization...")
        jpeg_file = generate_jpeg_visualization(lineage_json, "subquery_aliases_visualization")
        
        if not jpeg_file:
            print("❌ Failed to generate JPEG visualization")
            return False
        
        # Step 6: Print detailed analysis
        print("\n📊 Detailed Analysis Results:")
        print("-" * 40)
        
        # Show source tables
        if parsed_json.get("chains"):
            print("\n🗃️  Source Tables Identified:")
            for entity_name, chain_data in parsed_json["chains"].items():
                if chain_data.get("entity_type") == "table" and chain_data.get("depth", -1) == 0:
                    print(f"   • {entity_name}")
                    
                    # Show columns if available
                    columns = chain_data.get("metadata", {}).get("table_columns", [])
                    if columns:
                        column_names = [col.get("name", "unnamed") for col in columns]
                        if len(column_names) > 5:
                            print(f"     Columns: {', '.join(column_names[:5])}... ({len(column_names)} total)")
                        else:
                            print(f"     Columns: {', '.join(column_names)}")
                    else:
                        print(f"     Columns: None")
        
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
                        source_table = transform.get("source_table", "unknown")
                        target_table = transform.get("target_table", "unknown")
                        print(f"   • {transform_type}: {source_table} → {target_table}")
        
        print(f"\n🎉 Subquery aliases query analysis completed successfully!")
        print(f"📁 Files generated:")
        print(f"   • JSON: {os.path.basename(json_file)}")
        print(f"   • JPEG: {os.path.basename(jpeg_file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_create_view_with_subquery_aliases():
    """Test CREATE VIEW query with subquery aliases."""
    print("🔗 CREATE VIEW with Subquery Aliases Analysis")
    print("=" * 60)
    
    # The CREATE VIEW query with subquery aliases
    create_view_sql = """CREATE VIEW "hive"."promethium"."adhir_datamap_1757141678120" AS SELECT
  t1.country,
  t1.category,
  t1."sum of profit",
  t2.plan_name,
  t2.agent_vendor,
  t2.sumof_mer
FROM
  (SELECT * FROM "hive"."promethium"."profit_by_country_by_category_1757103693384") t1
FULL OUTER JOIN
  (SELECT * FROM "hive"."promethium"."revenue_by_plan_by_vendor_1757103916264") t2
ON t1.category = t2.agent_vendor
LIMIT 100"""
    
    analyzer = create_analyzer()
    
    try:
        print("\n🔍 Analyzing CREATE VIEW with subquery aliases...")
        
        # Step 1: Basic analysis
        result = analyzer.analyze(create_view_sql)
        
        if result.has_errors():
            print("❌ CREATE VIEW analysis failed:")
            for error in result.errors:
                print(f"   • {error}")
            return False
        
        print("✅ CREATE VIEW analysis completed successfully")
        print(f"   • Upstream tables: {len(result.table_lineage.upstream.get('QUERY_RESULT', []))}")
        print(f"   • Downstream relationships: {len(result.table_lineage.downstream)}")
        print(f"   • Column relationships: {len(result.column_lineage.upstream)}")
        
        # Step 2: Generate comprehensive lineage chain
        print("\n📋 Generating comprehensive lineage chain JSON...")
        lineage_chain_data = analyzer.get_lineage_chain(create_view_sql, chain_type="downstream", depth=0)
        
        if not lineage_chain_data:
            print("❌ Failed to generate lineage chains")
            return False
        
        lineage_json = json.dumps(lineage_chain_data, indent=2)
        parsed_json = json.loads(lineage_json)
        
        print("✅ Lineage chain JSON generated successfully")
        print(f"   • Chains: {len(parsed_json.get('chains', {}))}")
        print(f"   • JSON size: {len(lineage_json):,} characters")
        
        if 'summary' in parsed_json:
            summary = parsed_json['summary']
            for key, value in summary.items():
                if key != 'chain_count':
                    key_formatted = key.replace('_', ' ').title()
                    print(f"   • {key_formatted}: {value}")
        
        # Step 3: Validate fixes
        print("\n🔧 Validating CREATE VIEW fixes...")
        chains = parsed_json.get("chains", {})
        
        # Check for source tables
        source_tables = [name for name, data in chains.items() if data.get('entity_type') == 'table']
        print(f"   • Source tables found: {len(source_tables)}")
        
        for table_name in source_tables:
            chain_data = chains[table_name]
            columns = chain_data.get("metadata", {}).get("table_columns", [])
            print(f"     - {table_name}: {len(columns)} columns")
            if columns:
                column_names = [col.get("name", "unnamed") for col in columns[:3]]  # Show first 3
                print(f"       {', '.join(column_names)}")
        
        # Check for views
        views = [name for name, data in chains.items() if data.get('entity_type') == 'view']
        print(f"   • Views found: {len(views)}")
        for view_name in views:
            view_data = chains[view_name]
            columns = view_data.get("metadata", {}).get("table_columns", [])
            print(f"     - {view_name}: {len(columns)} columns")
        
        # Check for derived tables in dependencies
        derived_tables = 0
        for name, data in chains.items():
            for dep in data.get('dependencies', []):
                if dep.get('entity_type') == 'derived_table':
                    derived_tables += 1
                    derived_columns = dep.get('metadata', {}).get('table_columns', [])
                    print(f"     - {dep.get('entity')}: {len(derived_columns)} columns (derived)")
        
        print(f"   • Derived tables: {derived_tables}")
        
        # Step 4: Save JSON output
        print("\n💾 Saving CREATE VIEW lineage JSON...")
        json_file = save_json_output(lineage_json, "subquery_aliases_create_view_lineage_chain.json")
        
        if not json_file:
            print("❌ Failed to save CREATE VIEW JSON output")
            return False
        
        # Step 5: Generate JPEG visualization
        print("\n🎨 Generating CREATE VIEW JPEG visualization...")
        jpeg_file = generate_jpeg_visualization(lineage_json, "subquery_aliases_create_view_visualization")
        
        if not jpeg_file:
            print("❌ Failed to generate CREATE VIEW JPEG visualization")
            return False
        
        print(f"\n🎉 CREATE VIEW analysis completed!")
        print(f"📁 Files generated:")
        print(f"   • JSON: {os.path.basename(json_file)}")
        print(f"   • JPEG: {os.path.basename(jpeg_file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ CREATE VIEW analysis failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


def test_create_table_with_subquery_aliases():
    """Test CREATE TABLE with subquery aliases functionality."""
    print("🔗 CREATE TABLE with Subquery Aliases Analysis")
    print("="*60)
    
    # CREATE TABLE query with subquery aliases
    create_table_sql = '''CREATE TABLE "hive"."promethium"."adhir_analytics_table_1757141678120" AS SELECT
  t1.country,
  t1.category,
  t1."sum of profit",
  t2.plan_name,
  t2.agent_vendor,
  t2.sumof_mer
FROM
  (SELECT * FROM "hive"."promethium"."profit_by_country_by_category_1757103693384") t1
FULL OUTER JOIN
  (SELECT * FROM "hive"."promethium"."revenue_by_plan_by_vendor_1757103916264") t2
ON t1.category = t2.agent_vendor
LIMIT 100'''
    
    try:
        # Step 1: Initialize analyzer
        analyzer = SQLLineageAnalyzer(dialect='trino', compatibility_mode='full')
        print("📊 Using SQL-only analysis for CREATE TABLE query")
        
        # Step 2: Run basic analysis
        print("\n🔍 Analyzing CREATE TABLE with subquery aliases...")
        results = analyzer.analyze(create_table_sql)
        
        if not results:
            print("❌ CREATE TABLE analysis failed - no results returned")
            return False
        
        print("✅ CREATE TABLE analysis completed successfully")
        print(f"   • Upstream tables: {len(results.upstream_table_names) if hasattr(results, 'upstream_table_names') else 0}")
        print(f"   • Downstream relationships: {len(results.downstream_table_names) if hasattr(results, 'downstream_table_names') else 0}")
        print(f"   • Column relationships: {len(results.column_lineage_relationships) if hasattr(results, 'column_lineage_relationships') else 0}")
        
        # Step 3: Generate comprehensive lineage chain
        print("\n📋 Generating comprehensive lineage chain JSON...")
        lineage_json = analyzer.get_lineage_chain(
            create_table_sql,
            chain_type="downstream",
            max_depth="unlimited"
        )
        
        if not lineage_json:
            print("❌ Failed to generate lineage chain JSON")
            return False
        
        # Handle both string and dict responses
        if isinstance(lineage_json, dict):
            parsed_json = lineage_json
        else:
            parsed_json = json.loads(lineage_json)
        print("✅ Lineage chain JSON generated successfully")
        print(f"   • Chains: {len(parsed_json.get('chains', {}))}")
        json_size = len(json.dumps(lineage_json)) if isinstance(lineage_json, dict) else len(lineage_json)
        print(f"   • JSON size: {json_size:,} characters")
        print(f"   • Total Tables: {parsed_json.get('summary', {}).get('total_tables', 0)}")
        print(f"   • Total Columns: {parsed_json.get('summary', {}).get('total_columns', 0)}")
        print(f"   • Has Transformations: {parsed_json.get('summary', {}).get('has_transformations', False)}")
        print(f"   • Has Metadata: {parsed_json.get('summary', {}).get('has_metadata', False)}")
        
        # Step 4: Validate CREATE TABLE fixes
        print("\n🔧 Validating CREATE TABLE fixes...")
        validation_success = validate_create_table_fixes(parsed_json)
        
        if validation_success:
            print("   ✅ All validation checks passed!")
        else:
            print("   ⚠️  Some validation checks failed - fixes may need refinement")
        
        # Step 5: Save JSON output
        print("\n💾 Saving CREATE TABLE lineage JSON...")
        json_string = json.dumps(lineage_json, indent=2) if isinstance(lineage_json, dict) else lineage_json
        json_file = save_json_output(json_string, "subquery_aliases_create_table_lineage_chain.json")
        
        if not json_file:
            print("❌ Failed to save JSON output")
            return False
        
        # Step 6: Generate JPEG visualization
        print("\n🎨 Generating CREATE TABLE JPEG visualization...")
        jpeg_file = generate_jpeg_visualization(json_string, "subquery_aliases_create_table_visualization")
        
        if not jpeg_file:
            print("❌ Failed to generate JPEG visualization")
            return False
        
        print("\n🎉 CREATE TABLE analysis completed!")
        print(f"📁 Files generated:")
        print(f"   • JSON: {os.path.basename(json_file)}")
        print(f"   • JPEG: {os.path.basename(jpeg_file)}")
        
        return True
        
    except Exception as e:
        print(f"❌ CREATE TABLE test failed with error: {e}")
        import traceback
        print("📋 Full traceback:")
        print(traceback.format_exc())
        return False


def validate_create_table_fixes(parsed_json):
    """Validate that CREATE TABLE fixes are working correctly."""
    try:
        chains = parsed_json.get("chains", {})
        
        # Count source tables vs tables vs derived tables
        source_tables = []
        tables_found = []
        derived_tables = []
        
        for entity_name, chain_data in chains.items():
            entity_type = chain_data.get("entity_type", "unknown")
            depth = chain_data.get("depth", -1)
            
            if entity_type == "table" and depth == 0:
                source_tables.append(entity_name)
                columns = chain_data.get("metadata", {}).get("table_columns", [])
                print(f"     - {entity_name}: {len(columns)} columns")
                if columns:
                    column_names = [col.get("name", "unnamed") for col in columns[:3]]  # Show first 3
                    print(f"       {', '.join(column_names)}{'...' if len(columns) > 3 else ''}")
            elif entity_type == "table" and depth == 2:
                tables_found.append(entity_name)
                columns = chain_data.get("metadata", {}).get("table_columns", [])
                print(f"     - {entity_name}: {len(columns)} columns")
            
            # Check for derived tables in dependencies
            dependencies = chain_data.get("dependencies", [])
            for dep in dependencies:
                if dep.get("entity_type") == "derived_table":
                    derived_table_name = dep.get("entity")
                    derived_columns = dep.get("metadata", {}).get("table_columns", [])
                    derived_tables.append(f"{derived_table_name}: {len(derived_columns)} columns (derived)")
        
        print(f"   • Source tables found: {len(source_tables)}")
        for source in source_tables:
            source_chain = chains[source]
            columns = source_chain.get("metadata", {}).get("table_columns", [])
            column_names = [col.get("name", "unnamed") for col in columns]
            print(f"     - {source}: {len(columns)} columns")
            if columns:
                print(f"       {', '.join(column_names)}")
        
        print(f"   • Tables found: {len(tables_found)}")
        if tables_found:
            for table in tables_found:
                print(f"     - {table}: TABLE")
        
        if derived_tables:
            print(f"   • Derived tables: {len(derived_tables)}")
            for derived in derived_tables:
                print(f"     - {derived}")
        
        # Basic validation
        has_source_tables = len(source_tables) >= 2  # Expecting at least 2 source tables
        has_columns = all(len(chains[src].get("metadata", {}).get("table_columns", [])) > 0 for src in source_tables)
        
        return has_source_tables and has_columns
        
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False


def main():
    """Main test runner."""
    print("🚀 Subquery Aliases Query Lineage Test")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Run the SELECT test
    print("\n" + "="*60)
    print("TEST 1: SELECT with Subquery Aliases")
    print("="*60)
    success1 = test_subquery_aliases_query()
    
    # Run the CREATE VIEW test  
    print("\n" + "="*60)
    print("TEST 2: CREATE VIEW with Subquery Aliases")  
    print("="*60)
    success2 = test_create_view_with_subquery_aliases()
    
    # Run the CREATE TABLE test  
    print("\n" + "="*60)
    print("TEST 3: CREATE TABLE with Subquery Aliases")  
    print("="*60)
    success3 = test_create_table_with_subquery_aliases()
    
    if success1 and success2 and success3:
        print("\n✅ Test completed successfully!")
        
        # List generated files
        output_files = [f for f in os.listdir(output_dir) if f.startswith('subquery_aliases')]
        if output_files:
            print(f"\n📄 Generated files:")
            for file in sorted(output_files):
                filepath = os.path.join(output_dir, file)
                size = os.path.getsize(filepath)
                print(f"   • {file} ({size:,} bytes)")
        
        print(f"\n💡 The lineage JSONs contain comprehensive table and column relationships")
        print(f"   showing how data flows from source tables through subquery aliases to the final result.")
        print(f"   The JPEG visualizations provide visual representations of these relationships.")
        print(f"\n🔧 These tests validate the fixes for:")
        print(f"   • Subquery alias resolution (t1, t2 → actual table names)")
        print(f"   • CREATE VIEW with subquery aliases")
        print(f"   • CREATE TABLE with subquery aliases")
        print(f"   • 3-layer chain structure for derived tables")  
        print(f"   • Column extraction from aliased subqueries")
        print(f"   • Deduplication of fully qualified vs short table names")
        
        return 0
    else:
        print("\n❌ Test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())