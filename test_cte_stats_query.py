#!/usr/bin/env python3
"""
Test program for CTE statistics query - table and column level chain creation and diagram generation.
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


def create_visualizations(analyzer, sql, test_name, query_name):
    """Create visualization outputs for the test query."""
    try:
        visualizer = SQLLineageVisualizer()
    except ImportError as e:
        print(f"⚠️  Visualization not available: {e}")
        return
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print_section_header("Creating Visualizations", 50)
    
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


def test_cte_stats_query():
    """Test the complex CTE statistics query for table and column lineage."""
    print_section_header("CTE Statistics Query Lineage Test", 60)
    
    # Initialize analyzer with sample metadata
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # The complex CTE query provided by the user
    sql = """
    WITH total_cols AS (
        SELECT
            TABNAME,
            COUNT(*) as TOTAL_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
        GROUP BY TABNAME
    ),
    syscat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSCAT_STATS_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    sysstat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSSTAT_STATS_COLUMNS
        FROM maximo_dev.SYSSTAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    stats_summary AS (
        SELECT
            t.TABNAME,
            t.TOTAL_COLUMNS,
            COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) as SYSCAT_STATS_COLUMNS,
            COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) as SYSSTAT_STATS_COLUMNS,
            ROUND((COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSCAT_COVERAGE_PCT,
            ROUND((COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSSTAT_COVERAGE_PCT
        FROM total_cols t
        LEFT JOIN syscat_stats sc ON t.TABNAME = sc.TABNAME
        LEFT JOIN sysstat_stats ss ON t.TABNAME = ss.TABNAME
    )
    SELECT
        COUNT(*) as TOTAL_TABLES,
        SUM(TOTAL_COLUMNS) as TOTAL_COLUMNS_ALL_TABLES,
        SUM(SYSCAT_STATS_COLUMNS) as TOTAL_SYSCAT_STATS,
        SUM(SYSSTAT_STATS_COLUMNS) as TOTAL_SYSSTAT_STATS,
        ROUND(AVG(SYSCAT_COVERAGE_PCT), 2) as AVG_SYSCAT_COVERAGE,
        ROUND(AVG(SYSSTAT_COVERAGE_PCT), 2) as AVG_SYSSTAT_COVERAGE
    FROM stats_summary
    """
    
    query_name = "cte_stats_query"
    
    # Store all JSON outputs and chain outputs
    json_outputs = []
    chain_outputs = []
    
    # Analyze the query
    print(f"\n🔍 Analyzing CTE Statistics Query...")
    result = analyzer.analyze(sql)
    json_outputs.append((query_name, analyzer.get_lineage_json(sql)))
    
    if not print_lineage_analysis(result, sql, "CTE Statistics Query Analysis"):
        return False
    
    # Test table lineage chains with different depths
    print("\n📊 Testing Table Lineage Chains...")
    for depth in [1, 2, 3, 4]:
        try:
            # Upstream table chain
            upstream_table_chain = analyzer.get_table_lineage_chain_json(sql, "upstream", depth)
            chain_outputs.append((query_name, "table", "upstream", depth, upstream_table_chain))
            print(f"✅ Table upstream chain (depth {depth}): Generated")
            
            # Downstream table chain
            downstream_table_chain = analyzer.get_table_lineage_chain_json(sql, "downstream", depth)
            chain_outputs.append((query_name, "table", "downstream", depth, downstream_table_chain))
            print(f"✅ Table downstream chain (depth {depth}): Generated")
            
        except Exception as e:
            print(f"❌ Failed to generate table chains at depth {depth}: {e}")
    
    # Test column lineage chains with different depths
    print("\n📋 Testing Column Lineage Chains...")
    for depth in [1, 2, 3, 4]:
        try:
            # Upstream column chain
            upstream_column_chain = analyzer.get_column_lineage_chain_json(sql, "upstream", depth)
            chain_outputs.append((query_name, "column", "upstream", depth, upstream_column_chain))
            print(f"✅ Column upstream chain (depth {depth}): Generated")
            
            # Downstream column chain
            downstream_column_chain = analyzer.get_column_lineage_chain_json(sql, "downstream", depth)
            chain_outputs.append((query_name, "column", "downstream", depth, downstream_column_chain))
            print(f"✅ Column downstream chain (depth {depth}): Generated")
            
        except Exception as e:
            print(f"❌ Failed to generate column chains at depth {depth}: {e}")
    
    # Save JSON outputs to files
    save_json_outputs(json_outputs, "cte_stats_test")
    save_chain_outputs(chain_outputs, "cte_stats_test")
    
    # Create visualizations
    create_visualizations(analyzer, sql, "cte_stats_test", query_name)
    
    print("\n🎉 CTE Statistics Query test completed successfully!")
    return True


def test_chain_details():
    """Test detailed chain functionality for the CTE stats query."""
    print_section_header("Chain Details Analysis", 50)
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = """
    WITH total_cols AS (
        SELECT
            TABNAME,
            COUNT(*) as TOTAL_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
        GROUP BY TABNAME
    ),
    syscat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSCAT_STATS_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    sysstat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSSTAT_STATS_COLUMNS
        FROM maximo_dev.SYSSTAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    stats_summary AS (
        SELECT
            t.TABNAME,
            t.TOTAL_COLUMNS,
            COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) as SYSCAT_STATS_COLUMNS,
            COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) as SYSSTAT_STATS_COLUMNS,
            ROUND((COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSCAT_COVERAGE_PCT,
            ROUND((COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSSTAT_COVERAGE_PCT
        FROM total_cols t
        LEFT JOIN syscat_stats sc ON t.TABNAME = sc.TABNAME
        LEFT JOIN sysstat_stats ss ON t.TABNAME = ss.TABNAME
    )
    SELECT
        COUNT(*) as TOTAL_TABLES,
        SUM(TOTAL_COLUMNS) as TOTAL_COLUMNS_ALL_TABLES,
        SUM(SYSCAT_STATS_COLUMNS) as TOTAL_SYSCAT_STATS,
        SUM(SYSSTAT_STATS_COLUMNS) as TOTAL_SYSSTAT_STATS,
        ROUND(AVG(SYSCAT_COVERAGE_PCT), 2) as AVG_SYSCAT_COVERAGE,
        ROUND(AVG(SYSSTAT_COVERAGE_PCT), 2) as AVG_SYSSTAT_COVERAGE
    FROM stats_summary
    """
    
    try:
        # Get table chain objects (not JSON) for detailed analysis
        print("\n🔍 Analyzing Table Chain Structure (depth 3):")
        upstream_table_chain = analyzer.get_table_lineage_chain(sql, "upstream", 3)
        print(f"   • Upstream table chains: {len(upstream_table_chain['chains'])}")
        print(f"   • Chain type: {upstream_table_chain['chain_type']}")
        print(f"   • Max depth: {upstream_table_chain['max_depth']}")
        
        downstream_table_chain = analyzer.get_table_lineage_chain(sql, "downstream", 3)
        print(f"   • Downstream table chains: {len(downstream_table_chain['chains'])}")
        
        # Get column chain objects for detailed analysis
        print("\n🔍 Analyzing Column Chain Structure (depth 3):")
        upstream_column_chain = analyzer.get_column_lineage_chain(sql, "upstream", 3)
        print(f"   • Upstream column chains: {len(upstream_column_chain['chains'])}")
        print(f"   • Chain type: {upstream_column_chain['chain_type']}")
        print(f"   • Max depth: {upstream_column_chain['max_depth']}")
        
        downstream_column_chain = analyzer.get_column_lineage_chain(sql, "downstream", 3)
        print(f"   • Downstream column chains: {len(downstream_column_chain['chains'])}")
        
        # Print first few chains for inspection
        if upstream_table_chain['chains']:
            print("\n📋 First Table Chain Details:")
            first_table = list(upstream_table_chain['chains'].keys())[0]
            first_chain = upstream_table_chain['chains'][first_table]
            print(f"   • Start table: {first_table}")
            print(f"   • Chain depth: {first_chain['depth']}")
            print(f"   • Dependencies: {len(first_chain['dependencies'])}")
        
        if upstream_column_chain['chains']:
            print("\n📋 First Column Chain Details:")
            first_column = list(upstream_column_chain['chains'].keys())[0]
            first_chain = upstream_column_chain['chains'][first_column]
            print(f"   • Start column: {first_column}")
            print(f"   • Chain depth: {first_chain['depth']}")
            print(f"   • Dependencies: {len(first_chain['dependencies'])}")
        
        print("\n✅ Chain details analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Chain details analysis failed: {e}")
        return False


def performance_test():
    """Test performance of the CTE stats query analysis."""
    print_section_header("Performance Test", 40)
    
    import time
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = """
    WITH total_cols AS (
        SELECT
            TABNAME,
            COUNT(*) as TOTAL_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
        GROUP BY TABNAME
    ),
    syscat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSCAT_STATS_COLUMNS
        FROM maximo_dev.SYSCAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    sysstat_stats AS (
        SELECT
            TABNAME,
            COUNT(*) as SYSSTAT_STATS_COLUMNS
        FROM maximo_dev.SYSSTAT.COLUMNS
        WHERE TABSCHEMA = 'MAXIMO'
          AND (COLCARD IS NOT NULL AND COLCARD >= 0)
        GROUP BY TABNAME
    ),
    stats_summary AS (
        SELECT
            t.TABNAME,
            t.TOTAL_COLUMNS,
            COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) as SYSCAT_STATS_COLUMNS,
            COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) as SYSSTAT_STATS_COLUMNS,
            ROUND((COALESCE(sc.SYSCAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSCAT_COVERAGE_PCT,
            ROUND((COALESCE(ss.SYSSTAT_STATS_COLUMNS, 0) * 100.0 / t.TOTAL_COLUMNS), 2) as SYSSTAT_COVERAGE_PCT
        FROM total_cols t
        LEFT JOIN syscat_stats sc ON t.TABNAME = sc.TABNAME
        LEFT JOIN sysstat_stats ss ON t.TABNAME = ss.TABNAME
    )
    SELECT
        COUNT(*) as TOTAL_TABLES,
        SUM(TOTAL_COLUMNS) as TOTAL_COLUMNS_ALL_TABLES,
        SUM(SYSCAT_STATS_COLUMNS) as TOTAL_SYSCAT_STATS,
        SUM(SYSSTAT_STATS_COLUMNS) as TOTAL_SYSSTAT_STATS,
        ROUND(AVG(SYSCAT_COVERAGE_PCT), 2) as AVG_SYSCAT_COVERAGE,
        ROUND(AVG(SYSSTAT_COVERAGE_PCT), 2) as AVG_SYSSTAT_COVERAGE
    FROM stats_summary
    """
    
    start_time = time.time()
    result = analyzer.analyze(sql)
    end_time = time.time()
    
    analysis_duration = end_time - start_time
    
    if result.has_errors():
        print(f"❌ Performance test failed: {result.errors[0]}")
        return False
    
    print(f"✅ Analysis time: {analysis_duration:.3f} seconds")
    print(f"✅ Upstream relationships: {len(result.table_lineage.upstream)}")
    print(f"✅ Downstream relationships: {len(result.table_lineage.downstream)}")
    print(f"✅ Column relationships: {len(result.column_lineage.upstream)}")
    
    # Test chain generation performance
    print("\n⚡ Chain Generation Performance:")
    
    # Table chains
    start_time = time.time()
    table_chain = analyzer.get_table_lineage_chain(sql, "upstream", 3)
    end_time = time.time()
    table_duration = end_time - start_time
    print(f"   • Table chain (depth 3): {table_duration:.3f} seconds, {len(table_chain['chains'])} chains")
    
    # Column chains
    start_time = time.time()
    column_chain = analyzer.get_column_lineage_chain(sql, "upstream", 3)
    end_time = time.time()
    column_duration = end_time - start_time
    print(f"   • Column chain (depth 3): {column_duration:.3f} seconds, {len(column_chain['chains'])} chains")
    
    total_time = analysis_duration + table_duration + column_duration
    print(f"\n⏱️  Total processing time: {total_time:.3f} seconds")
    
    return True


def main():
    """Main test runner for CTE statistics query."""
    print("🚀 CTE Statistics Query Lineage Test Suite")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        success_count = 0
        total_tests = 3
        
        # Run main CTE stats query test
        if test_cte_stats_query():
            success_count += 1
        
        # Run chain details test
        if test_chain_details():
            success_count += 1
        
        # Run performance test
        if performance_test():
            success_count += 1
        
        print(f"\n📁 Check the '{output_dir}' directory for generated files.")
        
        # List generated files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.startswith('cte_stats_test') and f.endswith(('.jpeg', '.json'))]
            if files:
                print(f"\n📄 Generated CTE Stats Test files ({len(files)}):")
                
                visualization_files = [f for f in files if f.endswith('.jpeg')]
                json_files = [f for f in files if f.endswith('.json')]
                
                if visualization_files:
                    print(f"\n   🎨 Visualization Files ({len(visualization_files)}):")
                    for file in sorted(visualization_files):
                        file_path = os.path.join(output_dir, file)
                        size = os.path.getsize(file_path)
                        print(f"     • {file} ({size:,} bytes)")
                
                if json_files:
                    print(f"\n   📋 JSON Data Files ({len(json_files)}):")
                    regular_json = [f for f in json_files if 'chain' not in f]
                    chain_json = [f for f in json_files if 'chain' in f]
                    
                    if regular_json:
                        print(f"     📊 Analysis Results ({len(regular_json)}):")
                        for file in sorted(regular_json):
                            file_path = os.path.join(output_dir, file)
                            size = os.path.getsize(file_path)
                            print(f"       • {file} ({size:,} bytes)")
                    
                    if chain_json:
                        print(f"     🔗 Lineage Chain Data ({len(chain_json)}):")
                        for file in sorted(chain_json)[:10]:  # Show first 10
                            file_path = os.path.join(output_dir, file)
                            size = os.path.getsize(file_path)
                            print(f"       • {file} ({size:,} bytes)")
                        if len(chain_json) > 10:
                            print(f"       ... and {len(chain_json) - 10} more chain files")
            else:
                print("\n⚠️  No CTE stats test files were generated")
        
        # Print summary
        success = print_test_summary(total_tests, success_count, "CTE Statistics Query Tests")
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