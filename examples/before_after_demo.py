#!/usr/bin/env python3
"""
Before/After demonstration showing the difference between
sample metadata and external metadata integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer import SQLLineageAnalyzer
from analyzer.metadata import MetadataRegistry, SampleMetadataRegistry
from external_metadata_providers import DatabaseMetadataProvider, create_sample_json_metadata, JSONFileMetadataProvider
from test_formatter import print_lineage_analysis


def demo_before_external_metadata():
    """Show what happens with default sample metadata."""
    print("🔍 BEFORE: Using Default Sample Metadata")
    print("=" * 80)
    
    # Default analyzer with sample metadata
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
    result = analyzer.analyze(sql)
    
    print_lineage_analysis(result, sql, "Default Sample Metadata", show_column_lineage=False)


def demo_after_external_metadata():
    """Show what happens with external metadata."""
    print("\n\n🚀 AFTER: Using External Production Metadata")
    print("=" * 80)
    
    # Create custom registry with external provider
    custom_registry = MetadataRegistry()  # Already empty by default
    
    # Add external provider (simulating production database)
    custom_registry.add_provider(DatabaseMetadataProvider("postgresql://prod-db/metadata"))
    
    # Create analyzer with external metadata
    analyzer = SQLLineageAnalyzer(dialect="postgres")
    analyzer.set_metadata_registry(custom_registry)
    
    sql = "SELECT u.username, o.amount FROM users u JOIN orders o ON u.id = o.user_id"
    result = analyzer.analyze(sql)
    
    print_lineage_analysis(result, sql, "External Production Metadata", show_column_lineage=False)


def demo_comparison():
    """Show side-by-side comparison of key differences."""
    print("\n\n📊 KEY DIFFERENCES COMPARISON")
    print("=" * 80)
    
    # Default analyzer
    default_analyzer = SQLLineageAnalyzer(dialect="trino")
    default_analyzer.set_metadata_registry(SampleMetadataRegistry())
    
    # External analyzer
    custom_registry = MetadataRegistry()  # Already empty by default
    custom_registry.add_provider(DatabaseMetadataProvider("postgresql://prod-db/metadata"))
    external_analyzer = SQLLineageAnalyzer(dialect="postgres")
    external_analyzer.set_metadata_registry(custom_registry)
    
    sql = "SELECT name FROM users"
    
    default_result = default_analyzer.analyze(sql)
    external_result = external_analyzer.analyze(sql)
    
    print("COMPARISON TABLE:")
    print("─" * 80)
    print(f"{'Aspect':<20} {'Sample Metadata':<30} {'External Metadata':<30}")
    print("─" * 80)
    
    # Get metadata for comparison
    default_meta = default_result.metadata.get('default.users')
    external_meta = external_result.metadata.get('users')
    
    if default_meta and external_meta:
        print(f"{'Description':<20} {default_meta.description:<30} {external_meta.description:<30}")
        print(f"{'Owner':<20} {default_meta.owner:<30} {external_meta.owner:<30}")
        print(f"{'Row Count':<20} {default_meta.row_count:<30,} {external_meta.row_count:<30,}")
        print(f"{'Storage Format':<20} {default_meta.storage_format:<30} {external_meta.storage_format:<30}")
        print(f"{'Columns':<20} {len(default_meta.columns):<30} {len(external_meta.columns):<30}")
    
    print("─" * 80)
    print("\n✨ Benefits of External Metadata:")
    print("  • Real production data (accurate row counts, owners)")
    print("  • Up-to-date column information")
    print("  • Actual storage formats and locations") 
    print("  • Integration with your data governance tools")
    print("  • Automatic updates when schema changes")


if __name__ == "__main__":
    print("SQL Lineage Analyzer - External Metadata Integration Demo")
    print("=" * 80)
    print("This demo shows the difference between using sample metadata")
    print("vs. integrating with external metadata sources.")
    print("=" * 80)
    
    demo_before_external_metadata()
    demo_after_external_metadata()
    demo_comparison()