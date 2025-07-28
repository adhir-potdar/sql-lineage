# External Metadata Integration Guide

This guide explains how to integrate external metadata sources with the SQL Lineage Analyzer, replacing or supplementing the built-in sample metadata with real production metadata.

## Why External Metadata Integration?

The built-in `registry.py` contains sample metadata for testing, but in production you need:
- **Real table and column information** from your data catalog
- **Accurate row counts and ownership** information
- **Up-to-date schema information** that reflects current state
- **Integration with your data governance tools**

## Integration Approaches

### 1. Complete Replacement (Recommended for Production)

Replace all sample metadata with external sources:

```python
from analyzer import SQLLineageAnalyzer
from analyzer.metadata.registry import MetadataRegistry
from your_metadata_providers import DatabaseMetadataProvider

# Create empty registry
custom_registry = MetadataRegistry()
custom_registry.tables.clear()  # Remove sample data

# Add your external provider
custom_registry.add_provider(DatabaseMetadataProvider("your-connection-string"))

# Create analyzer with external metadata only
analyzer = SQLLineageAnalyzer(
    dialect="your-dialect",
    metadata_registry=custom_registry
)
```

### 2. Multiple Providers with Fallback

Use multiple metadata sources with priority ordering:

```python
analyzer = SQLLineageAnalyzer(dialect="trino")

# Add providers in priority order (first found wins)
analyzer.add_metadata_provider(DatabaseMetadataProvider("primary-db"))
analyzer.add_metadata_provider(JSONFileMetadataProvider("backup-metadata.json"))
analyzer.add_metadata_provider(HiveMetastoreProvider("hive-host"))
```

### 3. Hybrid Approach

Keep sample data for testing, add external sources for production tables:

```python
# Use default registry (includes sample data)
analyzer = SQLLineageAnalyzer(dialect="trino")

# Add external providers for specific schemas/catalogs
analyzer.add_metadata_provider(DatabaseMetadataProvider("production-db"))
analyzer.add_metadata_provider(HiveMetastoreProvider("hive-metastore"))
```

## Available Provider Types

### 1. Database Information Schema Provider

For databases that support information_schema:

```python
class DatabaseMetadataProvider(MetadataProvider):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        # Query information_schema.tables and information_schema.columns
        # Return TableMetadata object
```

**Use Cases:**
- PostgreSQL, MySQL, SQL Server metadata
- Any database with ANSI information_schema
- Direct database connectivity

### 2. JSON File Provider

For file-based metadata configuration:

```python
# metadata.json
{
  "production.users": {
    "description": "Production user accounts",
    "owner": "user_service",
    "row_count": 1500000,
    "storage_format": "POSTGRESQL",
    "columns": [...]
  }
}

provider = JSONFileMetadataProvider("metadata.json")
```

**Use Cases:**
- Configuration-driven metadata
- Small to medium deployments
- Quick prototyping and testing

### 3. Hive Metastore Provider

For Hadoop ecosystem integration:

```python
class HiveMetastoreProvider(MetadataProvider):
    def __init__(self, hive_host: str, hive_port: int = 9083):
        # Initialize Hive Metastore client
        # from pyhive import hive
        # self.client = hive.connect(host=hive_host, port=hive_port)
```

**Use Cases:**
- Hive data warehouses
- Spark/Hadoop environments
- Data lake architectures

### 4. AWS Glue Data Catalog Provider

For AWS cloud deployments:

```python
class AWSGlueMetadataProvider(MetadataProvider):
    def __init__(self, region: str = 'us-east-1'):
        # import boto3
        # self.glue_client = boto3.client('glue', region_name=region)
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        # response = self.glue_client.get_table(DatabaseName=db, Name=table)
        # Parse response into TableMetadata
```

**Use Cases:**
- AWS data lakes
- Glue ETL workflows
- S3-based data storage

### 5. Custom REST API Provider

For custom metadata services:

```python
class RestAPIMetadataProvider(MetadataProvider):
    def __init__(self, api_base_url: str, api_key: str):
        self.api_base_url = api_base_url
        self.api_key = api_key
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        # Make HTTP request to your metadata API
        # Parse response into TableMetadata
```

**Use Cases:**
- Custom data catalogs
- Microservice architectures
- Enterprise metadata management systems

## Implementation Steps

### Step 1: Identify Your Metadata Sources

Determine where your metadata currently lives:
- Database information schemas
- Data catalog tools (Alation, Collibra, etc.)
- Configuration files
- Cloud services (AWS Glue, Azure Purview, GCP Data Catalog)

### Step 2: Implement MetadataProvider

Create a class that implements the `MetadataProvider` protocol:

```python
from analyzer.metadata.registry import MetadataProvider
from analyzer.core.models import TableMetadata, ColumnMetadata
from typing import Optional

class YourMetadataProvider(MetadataProvider):
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        # Your implementation here
        pass
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        # Your implementation here
        pass
```

### Step 3: Handle Table Identifier Mapping

Map SQL table references to your metadata system:

```python
def parse_table_identifier(self, table_identifier: str):
    # Handle different formats:
    # - "users" -> lookup in default schema
    # - "schema.users" -> lookup in specific schema
    # - "catalog.schema.users" -> full three-part name
    
    parts = table_identifier.split('.')
    if len(parts) == 3:
        catalog, schema, table = parts
    elif len(parts) == 2:
        catalog, schema, table = None, parts[0], parts[1]
    else:
        catalog, schema, table = None, 'default', parts[0]
    
    return catalog, schema, table
```

### Step 4: Implement Caching (Recommended)

Add caching to avoid repeated metadata lookups:

```python
from functools import lru_cache

class YourMetadataProvider(MetadataProvider):
    @lru_cache(maxsize=1000)
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        # Cached implementation
        pass
```

### Step 5: Error Handling

Handle metadata source failures gracefully:

```python
def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
    try:
        # Fetch metadata from source
        return self._fetch_metadata(table_identifier)
    except ConnectionError:
        logger.warning(f"Metadata source unavailable for {table_identifier}")
        return None
    except Exception as e:
        logger.error(f"Error fetching metadata for {table_identifier}: {e}")
        return None
```

## Configuration Examples

### Environment-Based Configuration

```python
def create_analyzer_for_environment(env: str) -> SQLLineageAnalyzer:
    if env == "development":
        # Use sample data for development
        return SQLLineageAnalyzer(dialect="trino")
    
    elif env == "staging":
        registry = MetadataRegistry()
        registry.tables.clear()
        registry.add_provider(JSONFileMetadataProvider("staging-metadata.json"))
        return SQLLineageAnalyzer(dialect="postgres", metadata_registry=registry)
    
    elif env == "production":
        registry = MetadataRegistry()
        registry.tables.clear()
        registry.add_provider(DatabaseMetadataProvider("prod-connection"))
        registry.add_provider(HiveMetastoreProvider("hive-prod"))
        return SQLLineageAnalyzer(dialect="trino", metadata_registry=registry)
```

### YAML Configuration

```yaml
# config.yaml
metadata:
  providers:
    - type: database
      connection: "postgresql://prod-db/metadata"
      priority: 1
    - type: glue
      region: "us-east-1"
      priority: 2
    - type: json
      file_path: "fallback-metadata.json"
      priority: 3
```

```python
def create_analyzer_from_config(config_path: str) -> SQLLineageAnalyzer:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    registry = MetadataRegistry()
    registry.tables.clear()
    
    # Sort providers by priority
    providers = sorted(config['metadata']['providers'], key=lambda p: p['priority'])
    
    for provider_config in providers:
        provider = create_provider_from_config(provider_config)
        registry.add_provider(provider)
    
    return SQLLineageAnalyzer(metadata_registry=registry)
```

## Best Practices

### 1. Provider Priority
- Put most reliable/complete sources first
- Use fallback providers for missing metadata
- Consider performance implications of provider order

### 2. Caching Strategy
- Cache metadata to avoid repeated lookups
- Implement cache invalidation for schema changes
- Consider memory usage for large metadata sets

### 3. Error Handling
- Handle metadata source unavailability gracefully
- Log metadata lookup failures for debugging
- Provide meaningful error messages to users

### 4. Performance Optimization
- Batch metadata requests when possible
- Use connection pooling for database providers
- Implement async providers for high-throughput scenarios

### 5. Schema Evolution
- Handle schema changes in metadata sources
- Version your metadata formats
- Implement migration strategies for metadata updates

## Testing Your Integration

### Unit Tests

```python
def test_your_metadata_provider():
    provider = YourMetadataProvider("test-connection")
    
    # Test successful lookup
    metadata = provider.get_table_metadata("test_table")
    assert metadata is not None
    assert metadata.description == "Expected description"
    
    # Test missing table
    metadata = provider.get_table_metadata("nonexistent_table")
    assert metadata is None
```

### Integration Tests

```python
def test_analyzer_with_external_metadata():
    registry = MetadataRegistry()
    registry.tables.clear()
    registry.add_provider(YourMetadataProvider("test-connection"))
    
    analyzer = SQLLineageAnalyzer(metadata_registry=registry)
    result = analyzer.analyze("SELECT * FROM test_table")
    
    assert not result.has_errors()
    assert "test_table" in result.metadata
    assert result.metadata["test_table"].description == "Expected description"
```

## Migration from Sample Data

### Step-by-Step Migration

1. **Identify Current Usage**: Find where sample metadata is being used
2. **Implement Provider**: Create your MetadataProvider implementation
3. **Test with Subset**: Test with a small subset of tables first
4. **Gradual Rollout**: Migrate table by table or schema by schema
5. **Monitor and Validate**: Ensure metadata accuracy and completeness

### Migration Script Example

```python
def migrate_to_external_metadata():
    # Compare sample vs external metadata
    sample_analyzer = SQLLineageAnalyzer()  # Default with sample data
    
    external_registry = MetadataRegistry()
    external_registry.tables.clear()
    external_registry.add_provider(YourMetadataProvider("connection"))
    external_analyzer = SQLLineageAnalyzer(metadata_registry=external_registry)
    
    test_queries = ["SELECT * FROM users", "SELECT * FROM orders"]
    
    for sql in test_queries:
        sample_result = sample_analyzer.analyze(sql)
        external_result = external_analyzer.analyze(sql)
        
        print(f"Query: {sql}")
        print(f"Sample metadata tables: {len(sample_result.metadata)}")
        print(f"External metadata tables: {len(external_result.metadata)}")
        print("---")
```

This comprehensive guide should help you integrate external metadata sources with the SQL Lineage Analyzer. The key is implementing the `MetadataProvider` protocol to connect to your specific metadata sources.