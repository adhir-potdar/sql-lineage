#!/usr/bin/env python3
"""
Integration examples showing how to use external metadata providers
with the SQL Lineage Analyzer in different scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyzer import SQLLineageAnalyzer
from analyzer.metadata.registry import MetadataRegistry
from external_metadata_providers import (
    DatabaseMetadataProvider,
    JSONFileMetadataProvider,
    HiveMetastoreProvider,
    AWSGlueMetadataProvider,
    create_sample_json_metadata
)


def example_1_replace_default_metadata():
    """Example 1: Completely replace default metadata with external source."""
    print("=" * 80)
    print("EXAMPLE 1: Replace Default Metadata with JSON File")
    print("=" * 80)
    
    # Create JSON metadata file
    json_file = create_sample_json_metadata()
    
    # Method 1: Create empty registry and add provider
    custom_registry = MetadataRegistry()  # Already empty by default
    custom_registry.add_provider(JSONFileMetadataProvider(json_file))
    
    # Create analyzer with custom registry
    analyzer = SQLLineageAnalyzer(dialect="postgres")
    analyzer.set_metadata_registry(custom_registry)
    
    # Test with external metadata
    sql = "SELECT u.email, o.total_amount FROM production.users u JOIN production.orders o ON u.id = o.user_id"
    result = analyzer.analyze(sql)
    
    print(f"Query: {sql}")
    print(f"Metadata sources: {len(result.metadata)} tables")
    for table_name, metadata in result.metadata.items():
        print(f"  - {table_name}: {metadata.description} (Owner: {metadata.owner})")
        print(f"    Row Count: {metadata.row_count:,}, Format: {metadata.storage_format}")


def example_2_multiple_providers():
    """Example 2: Use multiple metadata providers with fallback."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Metadata Providers with Fallback")
    print("=" * 80)
    
    # Create analyzer with multiple providers
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Add multiple providers in priority order
    # 1. Database provider (highest priority)
    analyzer.add_metadata_provider(DatabaseMetadataProvider("postgresql://localhost/mydb"))
    
    # 2. JSON file provider (fallback)
    json_file = create_sample_json_metadata()
    analyzer.add_metadata_provider(JSONFileMetadataProvider(json_file))
    
    # 3. Hive provider (for Hive tables)
    analyzer.add_metadata_provider(HiveMetastoreProvider("hive-metastore.company.com"))
    
    # Test query that might hit different providers
    sql = """
    WITH user_events AS (
        SELECT user_id, COUNT(*) as event_count
        FROM hive.analytics.events  -- Will use Hive provider
        GROUP BY user_id
    ),
    order_summary AS (
        SELECT user_id, SUM(total_amount) as total_spent
        FROM production.orders  -- Will use JSON/Database provider
        GROUP BY user_id
    )
    SELECT u.email, ue.event_count, os.total_spent
    FROM production.users u  -- Will use JSON/Database provider
    JOIN user_events ue ON u.id = ue.user_id
    JOIN order_summary os ON u.id = os.user_id
    """
    
    result = analyzer.analyze(sql)
    
    print(f"Query involves {len(result.metadata)} tables from different sources:")
    for table_name, metadata in result.metadata.items():
        provider_type = "Unknown"
        if "hive" in table_name:
            provider_type = "Hive Metastore"
        elif "production" in table_name:
            provider_type = "Database/JSON"
        
        print(f"  - {table_name}: {metadata.description}")
        print(f"    Provider: {provider_type}, Owner: {metadata.owner}")


def example_3_aws_glue_integration():
    """Example 3: AWS Glue Data Catalog integration."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: AWS Glue Data Catalog Integration")
    print("=" * 80)
    
    # Custom registry with only AWS Glue
    glue_registry = MetadataRegistry()  # Already empty by default
    glue_registry.add_provider(AWSGlueMetadataProvider("us-east-1"))
    
    analyzer = SQLLineageAnalyzer(dialect="trino")
    analyzer.set_metadata_registry(glue_registry)
    
    # Test with Glue table
    sql = "SELECT event_type, COUNT(*) FROM glue.analytics.events GROUP BY event_type"
    result = analyzer.analyze(sql)
    
    print(f"Query: {sql}")
    if result.metadata:
        for table_name, metadata in result.metadata.items():
            print(f"AWS Glue Table: {table_name}")
            print(f"  Description: {metadata.description}")
            print(f"  Columns: {len(metadata.columns)}")
            for col in metadata.columns[:3]:  # Show first 3 columns
                print(f"    - {col.name}: {col.data_type} - {col.description}")


def example_4_hybrid_approach():
    """Example 4: Hybrid approach - keep some sample data, add external sources."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Hybrid Approach - Sample + External Data")
    print("=" * 80)
    
    # Use default registry (has sample data) and add external providers
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Add external providers for specific tables/schemas
    json_file = create_sample_json_metadata()
    analyzer.add_metadata_provider(JSONFileMetadataProvider(json_file))
    analyzer.add_metadata_provider(HiveMetastoreProvider("hive-metastore.company.com"))
    
    # Query that uses both sample and external metadata
    sql = """
    SELECT 
        default_users.name,  -- Uses sample metadata
        prod_users.email,    -- Uses JSON provider
        hive_events.event_type  -- Uses Hive provider
    FROM default.users default_users
    JOIN production.users prod_users ON default_users.id = prod_users.id
    JOIN hive.analytics.events hive_events ON prod_users.id = hive_events.user_id
    """
    
    result = analyzer.analyze(sql)
    
    print(f"Query combines data from multiple sources:")
    for table_name, metadata in result.metadata.items():
        source_type = "Sample Data"
        if "production" in table_name:
            source_type = "JSON Provider"
        elif "hive" in table_name:
            source_type = "Hive Provider"
        
        print(f"  - {table_name}: {source_type}")
        print(f"    {metadata.description} (Format: {metadata.storage_format})")


def example_5_custom_provider_class():
    """Example 5: Create a custom provider for your specific needs."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom Provider for REST API")
    print("=" * 80)
    
    from analyzer.metadata.registry import MetadataProvider
    from analyzer.core.models import TableMetadata, ColumnMetadata
    from typing import Optional
    
    class RestAPIMetadataProvider(MetadataProvider):
        """Custom provider that fetches metadata from a REST API."""
        
        def __init__(self, api_base_url: str, api_key: str):
            self.api_base_url = api_base_url
            self.api_key = api_key
        
        def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
            """Fetch metadata from REST API."""
            # In real implementation:
            # import requests
            # response = requests.get(
            #     f"{self.api_base_url}/tables/{table_identifier}",
            #     headers={"Authorization": f"Bearer {self.api_key}"}
            # )
            # if response.status_code == 200:
            #     data = response.json()
            #     return self._parse_api_response(data)
            
            # Simulate API response
            if table_identifier == "api.customer_data":
                columns = [
                    ColumnMetadata("customer_id", "BIGINT", False, True, description="API customer ID"),
                    ColumnMetadata("customer_name", "VARCHAR(200)", False, description="Customer name"),
                    ColumnMetadata("subscription_tier", "VARCHAR(50)", True, description="Subscription level"),
                ]
                
                return TableMetadata(
                    catalog="api",
                    schema="customer_data",
                    table="customers",
                    columns=columns,
                    description="Customer data from CRM API",
                    owner="crm_team",
                    row_count=500000,
                    storage_format="REST_API"
                )
            
            return None
        
        def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
            table_meta = self.get_table_metadata(table_identifier)
            if table_meta:
                for col in table_meta.columns:
                    if col.name == column_name:
                        return col
            return None
    
    # Use the custom provider
    custom_registry = MetadataRegistry()  # Already empty by default
    custom_registry.add_provider(RestAPIMetadataProvider("https://api.company.com/metadata", "your-api-key"))
    
    analyzer = SQLLineageAnalyzer()
    analyzer.set_metadata_registry(custom_registry)
    
    sql = "SELECT customer_name, subscription_tier FROM api.customer_data WHERE subscription_tier = 'premium'"
    result = analyzer.analyze(sql)
    
    print(f"Query: {sql}")
    if result.metadata:
        for table_name, metadata in result.metadata.items():
            print(f"REST API Table: {table_name}")
            print(f"  Source: {metadata.storage_format}")
            print(f"  Owner: {metadata.owner}")


def example_6_configuration_based_setup():
    """Example 6: Configuration-based metadata setup."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Configuration-Based Setup")
    print("=" * 80)
    
    # Configuration for different environments
    config = {
        "development": {
            "use_sample_data": True,
            "providers": []
        },
        "staging": {
            "use_sample_data": False,
            "providers": [
                {"type": "json", "config": {"file_path": "/config/staging_metadata.json"}},
                {"type": "database", "config": {"connection": "postgresql://staging-db/metadata"}}
            ]
        },
        "production": {
            "use_sample_data": False,
            "providers": [
                {"type": "glue", "config": {"region": "us-east-1"}},
                {"type": "hive", "config": {"host": "hive-prod.company.com", "port": 9083}},
                {"type": "database", "config": {"connection": "postgresql://prod-db/metadata"}}
            ]
        }
    }
    
    def create_analyzer_from_config(environment: str) -> SQLLineageAnalyzer:
        """Create analyzer based on configuration."""
        env_config = config[environment]
        
        if env_config["use_sample_data"]:
            # Use default registry with sample data
            return SQLLineageAnalyzer(dialect="trino")
        else:
            # Create custom registry
            registry = MetadataRegistry()  # Already empty by default
            
            # Add providers based on configuration
            for provider_config in env_config["providers"]:
                provider_type = provider_config["type"]
                provider_cfg = provider_config["config"]
                
                if provider_type == "json":
                    registry.add_provider(JSONFileMetadataProvider(provider_cfg["file_path"]))
                elif provider_type == "database":
                    registry.add_provider(DatabaseMetadataProvider(provider_cfg["connection"]))
                elif provider_type == "glue":
                    registry.add_provider(AWSGlueMetadataProvider(provider_cfg["region"]))
                elif provider_type == "hive":
                    registry.add_provider(HiveMetastoreProvider(provider_cfg["host"], provider_cfg["port"]))
            
            analyzer = SQLLineageAnalyzer(dialect="trino")
            analyzer.set_metadata_registry(registry)
            return analyzer
    
    # Test different environments
    for env in ["development", "staging", "production"]:
        print(f"\n--- {env.upper()} Environment ---")
        analyzer = create_analyzer_from_config(env)
        
        # Simple test query
        sql = "SELECT name FROM users LIMIT 5"
        result = analyzer.analyze(sql)
        
        if result.metadata:
            table_name = list(result.metadata.keys())[0]
            metadata = result.metadata[table_name]
            print(f"  Metadata source: {metadata.storage_format}")
            print(f"  Table owner: {metadata.owner}")
        else:
            print(f"  No metadata found (using {env} configuration)")


if __name__ == "__main__":
    print("SQL Lineage Analyzer - External Metadata Integration Examples")
    print("=" * 80)
    
    # Run all examples
    example_1_replace_default_metadata()
    example_2_multiple_providers()
    example_3_aws_glue_integration()
    example_4_hybrid_approach()
    example_5_custom_provider_class()
    example_6_configuration_based_setup()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("These examples show different ways to integrate external metadata:")
    print("1. Complete replacement of sample data")
    print("2. Multiple providers with fallback")
    print("3. Cloud-specific integrations (AWS Glue)")
    print("4. Hybrid approach (sample + external)")
    print("5. Custom provider implementation")
    print("6. Configuration-driven setup")
    print("\nChoose the approach that best fits your architecture and requirements.")