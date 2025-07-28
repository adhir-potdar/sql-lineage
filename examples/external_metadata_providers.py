#!/usr/bin/env python3
"""
Example external metadata providers showing how to integrate with external systems.
These examples demonstrate various integration patterns for real-world usage.
"""

import json
import sqlite3
from typing import Optional, Dict, Any
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyzer.metadata.registry import MetadataProvider
from analyzer.core.models import TableMetadata, ColumnMetadata


class DatabaseMetadataProvider(MetadataProvider):
    """Metadata provider that reads from a database information schema."""
    
    def __init__(self, connection_string: str):
        """
        Initialize with database connection.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self._cache = {}
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        """Get table metadata from database information schema."""
        if table_identifier in self._cache:
            return self._cache[table_identifier]
        
        # Parse table identifier
        parts = table_identifier.split('.')
        if len(parts) == 3:
            catalog, schema, table = parts
        elif len(parts) == 2:
            catalog, schema, table = None, parts[0], parts[1]
        else:
            catalog, schema, table = None, 'default', parts[0]
        
        try:
            # Example using SQLite (in real scenario, use your actual database)
            conn = sqlite3.connect(':memory:')  # Replace with real connection
            
            # Simulate metadata query results
            metadata = self._simulate_database_metadata(table)
            
            if metadata:
                self._cache[table_identifier] = metadata
                return metadata
                
        except Exception as e:
            print(f"Error fetching metadata for {table_identifier}: {e}")
        
        return None
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        """Get column metadata from table metadata."""
        table_meta = self.get_table_metadata(table_identifier)
        if table_meta:
            for col in table_meta.columns:
                if col.name == column_name:
                    return col
        return None
    
    def _simulate_database_metadata(self, table_name: str) -> Optional[TableMetadata]:
        """Simulate fetching metadata from database (replace with real implementation)."""
        # This simulates what you'd get from information_schema queries
        sample_tables = {
            'users': {
                'description': 'Production user table',
                'owner': 'user_service_team',
                'row_count': 1500000,
                'storage_format': 'POSTGRES',
                'columns': [
                    {'name': 'id', 'data_type': 'BIGINT', 'nullable': False, 'primary_key': True},
                    {'name': 'username', 'data_type': 'VARCHAR(100)', 'nullable': False},
                    {'name': 'email', 'data_type': 'VARCHAR(255)', 'nullable': False},
                    {'name': 'created_at', 'data_type': 'TIMESTAMP', 'nullable': False},
                    {'name': 'last_login', 'data_type': 'TIMESTAMP', 'nullable': True},
                ]
            },
            'orders': {
                'description': 'Production order transactions',
                'owner': 'order_service_team',
                'row_count': 5000000,
                'storage_format': 'POSTGRES',
                'columns': [
                    {'name': 'order_id', 'data_type': 'BIGINT', 'nullable': False, 'primary_key': True},
                    {'name': 'user_id', 'data_type': 'BIGINT', 'nullable': False, 'foreign_key': 'users.id'},
                    {'name': 'amount', 'data_type': 'DECIMAL(12,2)', 'nullable': False},
                    {'name': 'status', 'data_type': 'VARCHAR(50)', 'nullable': False},
                    {'name': 'created_at', 'data_type': 'TIMESTAMP', 'nullable': False},
                ]
            }
        }
        
        if table_name in sample_tables:
            table_info = sample_tables[table_name]
            columns = [
                ColumnMetadata(
                    name=col['name'],
                    data_type=col['data_type'],
                    nullable=col.get('nullable', True),
                    primary_key=col.get('primary_key', False),
                    foreign_key=col.get('foreign_key'),
                    description=f"Production column: {col['name']}"
                )
                for col in table_info['columns']
            ]
            
            return TableMetadata(
                catalog=None,
                schema='production',
                table=table_name,
                columns=columns,
                description=table_info['description'],
                owner=table_info['owner'],
                row_count=table_info['row_count'],
                storage_format=table_info['storage_format']
            )
        
        return None


class JSONFileMetadataProvider(MetadataProvider):
    """Metadata provider that reads from JSON configuration files."""
    
    def __init__(self, metadata_file_path: str):
        """
        Initialize with JSON metadata file.
        
        Args:
            metadata_file_path: Path to JSON file containing metadata
        """
        self.metadata_file_path = Path(metadata_file_path)
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if not self.metadata_file_path.exists():
            print(f"Metadata file not found: {self.metadata_file_path}")
            return {}
        
        try:
            with open(self.metadata_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata file: {e}")
            return {}
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        """Get table metadata from JSON configuration."""
        # Try exact match first
        if table_identifier in self._metadata:
            return self._parse_table_metadata(table_identifier, self._metadata[table_identifier])
        
        # Try partial matches
        for key, metadata in self._metadata.items():
            if key.endswith(f".{table_identifier}") or table_identifier.endswith(f".{key}"):
                return self._parse_table_metadata(table_identifier, metadata)
        
        return None
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        """Get column metadata from table metadata."""
        table_meta = self.get_table_metadata(table_identifier)
        if table_meta:
            for col in table_meta.columns:
                if col.name == column_name:
                    return col
        return None
    
    def _parse_table_metadata(self, table_identifier: str, metadata: Dict[str, Any]) -> TableMetadata:
        """Parse JSON metadata into TableMetadata object."""
        parts = table_identifier.split('.')
        catalog = parts[0] if len(parts) == 3 else None
        schema = parts[1] if len(parts) == 3 else (parts[0] if len(parts) == 2 else 'default')
        table = parts[-1]
        
        columns = [
            ColumnMetadata(
                name=col['name'],
                data_type=col['data_type'],
                nullable=col.get('nullable', True),
                primary_key=col.get('primary_key', False),
                foreign_key=col.get('foreign_key'),
                description=col.get('description')
            )
            for col in metadata.get('columns', [])
        ]
        
        return TableMetadata(
            catalog=catalog,
            schema=schema,
            table=table,
            columns=columns,
            description=metadata.get('description'),
            owner=metadata.get('owner'),
            row_count=metadata.get('row_count'),
            storage_format=metadata.get('storage_format')
        )


class HiveMetastoreProvider(MetadataProvider):
    """Metadata provider for Hive Metastore (example implementation)."""
    
    def __init__(self, hive_host: str, hive_port: int = 9083):
        """
        Initialize Hive Metastore connection.
        
        Args:
            hive_host: Hive Metastore host
            hive_port: Hive Metastore port
        """
        self.hive_host = hive_host
        self.hive_port = hive_port
        # In real implementation, initialize Hive Metastore client
        # from pyhive import hive
        # self.client = hive.connect(host=hive_host, port=hive_port)
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        """Get table metadata from Hive Metastore."""
        # Parse table identifier
        parts = table_identifier.split('.')
        if len(parts) >= 2:
            database = parts[-2]
            table = parts[-1]
        else:
            database = 'default'
            table = parts[0]
        
        try:
            # In real implementation:
            # table_info = self.client.get_table(database, table)
            # columns_info = self.client.get_columns(database, table)
            
            # Simulate Hive Metastore response
            return self._simulate_hive_metadata(database, table)
            
        except Exception as e:
            print(f"Error fetching Hive metadata for {table_identifier}: {e}")
            return None
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        """Get column metadata from table metadata."""
        table_meta = self.get_table_metadata(table_identifier)
        if table_meta:
            for col in table_meta.columns:
                if col.name == column_name:
                    return col
        return None
    
    def _simulate_hive_metadata(self, database: str, table: str) -> Optional[TableMetadata]:
        """Simulate Hive Metastore response."""
        # This simulates what you'd get from Hive Metastore
        if table == 'users':
            columns = [
                ColumnMetadata('user_id', 'BIGINT', False, True, description='Hive user ID'),
                ColumnMetadata('username', 'STRING', False, description='Hive username'),
                ColumnMetadata('registration_date', 'DATE', True, description='User registration date'),
                ColumnMetadata('last_active', 'TIMESTAMP', True, description='Last activity timestamp'),
            ]
            
            return TableMetadata(
                catalog='hive',
                schema=database,
                table=table,
                columns=columns,
                description=f'Hive table: {database}.{table}',
                owner='hive_admin',
                row_count=2000000,
                storage_format='PARQUET'
            )
        
        return None


class AWSGlueMetadataProvider(MetadataProvider):
    """Metadata provider for AWS Glue Data Catalog (example implementation)."""
    
    def __init__(self, region: str = 'us-east-1'):
        """
        Initialize AWS Glue client.
        
        Args:
            region: AWS region
        """
        self.region = region
        # In real implementation:
        # import boto3
        # self.glue_client = boto3.client('glue', region_name=region)
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        """Get table metadata from AWS Glue Data Catalog."""
        parts = table_identifier.split('.')
        if len(parts) >= 2:
            database = parts[-2]
            table = parts[-1]
        else:
            database = 'default'
            table = parts[0]
        
        try:
            # In real implementation:
            # response = self.glue_client.get_table(DatabaseName=database, Name=table)
            # table_info = response['Table']
            
            # Simulate Glue response
            return self._simulate_glue_metadata(database, table)
            
        except Exception as e:
            print(f"Error fetching Glue metadata for {table_identifier}: {e}")
            return None
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        """Get column metadata from table metadata."""
        table_meta = self.get_table_metadata(table_identifier)
        if table_meta:
            for col in table_meta.columns:
                if col.name == column_name:
                    return col
        return None
    
    def _simulate_glue_metadata(self, database: str, table: str) -> Optional[TableMetadata]:
        """Simulate AWS Glue Data Catalog response."""
        if table == 'events':
            columns = [
                ColumnMetadata('event_id', 'STRING', False, True, description='Glue event ID'),
                ColumnMetadata('user_id', 'BIGINT', False, description='User identifier'),
                ColumnMetadata('event_type', 'STRING', False, description='Type of event'),
                ColumnMetadata('event_data', 'STRING', True, description='JSON event payload'),
                ColumnMetadata('timestamp', 'TIMESTAMP', False, description='Event timestamp'),
            ]
            
            return TableMetadata(
                catalog='glue',
                schema=database,
                table=table,
                columns=columns,
                description=f'Glue table: {database}.{table}',
                owner='data_team',
                row_count=10000000,
                storage_format='PARQUET'
            )
        
        return None


def create_sample_json_metadata():
    """Create a sample JSON metadata file for demonstration."""
    metadata = {
        "production.users": {
            "description": "Production user accounts",
            "owner": "user_service",
            "row_count": 1500000,
            "storage_format": "POSTGRESQL",
            "columns": [
                {
                    "name": "id",
                    "data_type": "BIGINT",
                    "nullable": False,
                    "primary_key": True,
                    "description": "Primary key for users"
                },
                {
                    "name": "email",
                    "data_type": "VARCHAR(255)",
                    "nullable": False,
                    "description": "User email address"
                },
                {
                    "name": "created_at",
                    "data_type": "TIMESTAMP",
                    "nullable": False,
                    "description": "Account creation timestamp"
                }
            ]
        },
        "production.orders": {
            "description": "Production order transactions",
            "owner": "order_service",
            "row_count": 8000000,
            "storage_format": "POSTGRESQL",
            "columns": [
                {
                    "name": "order_id",
                    "data_type": "BIGINT",
                    "nullable": False,
                    "primary_key": True,
                    "description": "Primary key for orders"
                },
                {
                    "name": "user_id",
                    "data_type": "BIGINT",
                    "nullable": False,
                    "foreign_key": "users.id",
                    "description": "Reference to user account"
                },
                {
                    "name": "total_amount",
                    "data_type": "DECIMAL(12,2)",
                    "nullable": False,
                    "description": "Total order amount"
                }
            ]
        }
    }
    
    with open('/tmp/sample_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Sample metadata file created at /tmp/sample_metadata.json")
    return '/tmp/sample_metadata.json'


if __name__ == "__main__":
    print("External Metadata Providers Examples")
    print("=" * 50)
    
    # Create sample JSON file
    json_file = create_sample_json_metadata()
    
    # Test each provider
    providers = [
        ("Database Provider", DatabaseMetadataProvider("postgresql://localhost/mydb")),
        ("JSON File Provider", JSONFileMetadataProvider(json_file)),
        ("Hive Metastore Provider", HiveMetastoreProvider("hive-metastore.company.com")),
        ("AWS Glue Provider", AWSGlueMetadataProvider("us-east-1"))
    ]
    
    for provider_name, provider in providers:
        print(f"\n--- {provider_name} ---")
        
        # Test getting metadata
        tables_to_test = ["users", "orders", "events"]
        for table in tables_to_test:
            metadata = provider.get_table_metadata(table)
            if metadata:
                print(f"✅ {table}: {metadata.description} ({len(metadata.columns)} columns)")
            else:
                print(f"❌ {table}: No metadata found")