#!/usr/bin/env python3
"""
Enhanced output formatter for SQL lineage test scripts.
Provides user-friendly formatting similar to enhanced_lineage_analysis.py
"""

def print_section_header(title, width=80):
    """Print a formatted section header."""
    print(f"\n{'='*width}")
    print(f" {title}")
    print('='*width)

def print_subsection_header(title, width=60):
    """Print a formatted subsection header."""
    print(f"\n{'-'*width}")
    print(f" {title}")
    print('-'*width)

def format_table_info(table_name, metadata=None):
    """Format table information with metadata if available."""
    # Handle special logical targets
    if table_name == "QUERY_RESULT":
        return "Query Result Set (Main Output)"
    elif table_name.endswith("_QUERY"):
        return f"Query Result Set ({table_name.replace('_QUERY', '')})"
    
    # Handle actual table names
    if "." in table_name:
        parts = table_name.split(".")
        if len(parts) == 3:
            return f"Catalog: {parts[0]}, Schema: {parts[1]}, Table: {parts[2]}"
        elif len(parts) == 2:
            return f"Database: {parts[0]}, Table: {parts[1]}"
    return f"Table: {table_name}"

def get_table_columns_mapping(result):
    """Extract column mappings from lineage result."""
    table_columns = {}
    
    # Get columns from column lineage
    for target_col, source_cols in result.column_lineage.upstream.items():
        for source_col in source_cols:
            # Extract table name from column reference
            if "." in source_col:
                table_part = ".".join(source_col.split(".")[:-1])  # Everything except column name
                col_name = source_col.split(".")[-1]  # Just the column name
                
                if table_part not in table_columns:
                    table_columns[table_part] = set()
                table_columns[table_part].add(col_name)
    
    return table_columns

def print_lineage_analysis(result, sql, test_name, show_column_lineage=True):
    """Print comprehensive lineage analysis with user-friendly formatting."""
    print(f"\n🔍 {test_name}")
    print("─" * 80)
    
    # Show FULL SQL query (not truncated)
    print(f"📝 Query:")
    print(sql.strip())
    print(f"📊 Dialect: {result.dialect}")
    print("─" * 80)
    
    if result.has_errors():
        print("\n❌ ERRORS:")
        for error in result.errors:
            print(f"   • {error}")
        return False
    
    if result.has_warnings():
        print("\n⚠️  WARNINGS:")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    print("\n✅ ANALYSIS SUCCESSFUL")
    
    # Get table-column mappings
    table_columns = get_table_columns_mapping(result)
    
    # UPSTREAM LINEAGE with detailed column information
    print("\n📊 UPSTREAM LINEAGE (Dependencies):")
    if result.table_lineage.upstream:
        for target, sources in result.table_lineage.upstream.items():
            target_display = format_table_info(target)
            print(f"\n  🎯 {target_display}")
            
            if sources:
                print(f"    Dependencies:")
                for source in sorted(sources):
                    source_display = format_table_info(source)
                    print(f"      ← {source_display}")
                    
                    # Add metadata if available
                    if source in result.metadata:
                        meta = result.metadata[source]
                        print(f"        📋 Description: {meta.description or 'N/A'}")
                        print(f"        👤 Owner: {meta.owner or 'N/A'}")
                        print(f"        💾 Storage Format: {meta.storage_format or 'N/A'}")
                        if hasattr(meta, 'row_count') and meta.row_count:
                            print(f"        📊 Row Count: {meta.row_count:,}")
                    
                    # Show referenced columns for this table
                    referenced_columns = set()
                    for table_key, cols in table_columns.items():
                        if (source == table_key or 
                            table_key.endswith(f".{source}") or 
                            source.endswith(f".{table_key}") or
                            source.split(".")[-1] == table_key):
                            referenced_columns.update(cols)
                    
                    if referenced_columns:
                        print(f"        🔍 Referenced Columns:")
                        for col in sorted(referenced_columns):
                            # Get column metadata if available
                            if source in result.metadata:
                                meta = result.metadata[source]
                                col_meta = None
                                for col_info in meta.columns:
                                    if col_info.name == col:
                                        col_meta = col_info
                                        break
                                
                                if col_meta:
                                    pk_info = " (PK)" if col_meta.primary_key else ""
                                    fk_info = f" -> {col_meta.foreign_key}" if col_meta.foreign_key else ""
                                    print(f"          • {col}: {col_meta.data_type}{pk_info}{fk_info}")
                                    if col_meta.description:
                                        print(f"            {col_meta.description}")
                                else:
                                    print(f"          • {col}: UNKNOWN_TYPE")
                            else:
                                print(f"          • {col}")
                    else:
                        # Show all available columns if SELECT * or no specific columns found
                        if source in result.metadata:
                            meta = result.metadata[source]
                            print(f"        🔍 Available Columns:")
                            for col_meta in meta.columns[:10]:  # Show first 10 columns
                                pk_info = " (PK)" if col_meta.primary_key else ""
                                fk_info = f" -> {col_meta.foreign_key}" if col_meta.foreign_key else ""
                                print(f"          • {col_meta.name}: {col_meta.data_type}{pk_info}{fk_info}")
                                if col_meta.description:
                                    print(f"            {col_meta.description}")
                            if len(meta.columns) > 10:
                                print(f"          ... and {len(meta.columns) - 10} more columns")
            else:
                print(f"    ← No dependencies")
    else:
        print("  No upstream dependencies found")
    
    # DOWNSTREAM LINEAGE (reorganized to show from target perspective)
    print(f"\n📈 DOWNSTREAM LINEAGE (Data Flow):")
    if result.table_lineage.downstream:
        # Reorganize to show targets first, then their sources
        target_to_sources = {}
        for source, targets in result.table_lineage.downstream.items():
            for target in targets:
                if target not in target_to_sources:
                    target_to_sources[target] = set()
                target_to_sources[target].add(source)
        
        for target, sources in target_to_sources.items():
            target_display = format_table_info(target)
            print(f"\n  📤 {target_display}")
            
            if sources:
                print(f"    Receives data from:")
                for source in sorted(sources):
                    source_display = format_table_info(source)
                    print(f"      ← {source_display}")
            else:
                print(f"    ← No data sources")
    else:
        print("  No downstream dependencies found")
    
    # DETAILED COLUMN LINEAGE (only if flag is enabled)
    if show_column_lineage and result.column_lineage.upstream:
        print(f"\n🔍 COLUMN LINEAGE (Detailed):")
        print(f"    Total relationships: {len(result.column_lineage.upstream)}")
        print(f"    Column Dependencies:")
        
        for target_col, source_cols in result.column_lineage.upstream.items():
            # Clean up target column display
            if target_col.startswith("QUERY_RESULT."):
                target_display = target_col.replace("QUERY_RESULT.", "result.")
            elif "QUERY_RESULT" in target_col:
                target_display = target_col.replace("QUERY_RESULT", "result")
            else:
                target_display = target_col
            
            sources_display = ", ".join(sorted(source_cols))
            print(f"      • {target_display}")
            print(f"        ← {sources_display}")
    
    # METADATA SUMMARY
    if result.metadata:
        print(f"\n📋 METADATA SUMMARY:")
        print(f"    Tables with metadata: {len(result.metadata)}")
        for table_name, meta in result.metadata.items():
            table_display = format_table_info(table_name)
            print(f"      • {table_display}")
            if meta.description:
                print(f"        {meta.description}")
            print(f"        Columns: {len(meta.columns)}, Owner: {meta.owner or 'N/A'}")
    
    print("─" * 80)
    return True

def print_test_summary(total_tests, passed_tests, test_name="Test Suite"):
    """Print a formatted test summary."""
    print_section_header(f"{test_name} - SUMMARY")
    
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Results: {passed_tests}/{total_tests} tests passed")
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("The SQL Lineage Analyzer is working correctly!")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed")
        print("Please review the output above for details.")
    
    return failed_tests == 0

def print_quick_result(result, label, show_details=True, show_column_lineage=True):
    """Print quick test result in compact format."""
    if result.has_errors():
        print(f"❌ {label}: {result.errors[0]}")
        return False
    
    upstream_count = len(result.table_lineage.upstream.get("QUERY_RESULT", []))
    downstream_count = len(result.table_lineage.downstream)
    
    print(f"✅ {label}:")
    if show_details:
        if upstream_count > 0:
            upstream_tables = list(result.table_lineage.upstream.get("QUERY_RESULT", []))
            print(f"    📊 Upstream: {', '.join(sorted(upstream_tables))}")
        
        if downstream_count > 0:
            print(f"    📈 Downstream: {downstream_count} relationships")
        
        if show_column_lineage and result.column_lineage.upstream:
            print(f"    🔍 Columns: {len(result.column_lineage.upstream)} relationships")
    else:
        print(f"    {upstream_count} upstream, {downstream_count} downstream")
    
    return True