"""Metadata registry for table and column information."""

from typing import Dict, Optional, List, Protocol
from ..core.models import TableMetadata, ColumnMetadata, TableType


class MetadataProvider(Protocol):
    """Protocol for metadata providers."""
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        """Get table metadata by identifier."""
        ...
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        """Get column metadata."""
        ...


class MetadataRegistry:
    """Registry for table and column metadata."""
    
    def __init__(self):
        self.tables: Dict[str, TableMetadata] = {}
        self.providers: List[MetadataProvider] = []
    
    def add_provider(self, provider: MetadataProvider) -> None:
        """Add a metadata provider."""
        self.providers.append(provider)
    
    def register_table(self, table_identifier: str, metadata: TableMetadata) -> None:
        """Register table metadata."""
        self.tables[table_identifier] = metadata
    
    def get_table_metadata(self, table_identifier: str) -> Optional[TableMetadata]:
        """Get table metadata by identifier."""
        # Try exact match first
        if table_identifier in self.tables:
            return self.tables[table_identifier]
        
        # Try with default schema
        if f"default.{table_identifier}" in self.tables:
            return self.tables[f"default.{table_identifier}"]
        
        # Try partial matches
        for key, metadata in self.tables.items():
            if key.endswith(f".{table_identifier}") or key == table_identifier:
                return metadata
        
        # Try providers
        for provider in self.providers:
            metadata = provider.get_table_metadata(table_identifier)
            if metadata:
                return metadata
        
        return None
    
    def get_column_metadata(self, table_identifier: str, column_name: str) -> Optional[ColumnMetadata]:
        """Get column metadata."""
        table_meta = self.get_table_metadata(table_identifier)
        if table_meta:
            for col in table_meta.columns:
                if col.name == column_name:
                    return col
        
        # Try providers
        for provider in self.providers:
            metadata = provider.get_column_metadata(table_identifier, column_name)
            if metadata:
                return metadata
        
        return None
    
    def get_table_key(self, catalog: Optional[str], schema: Optional[str], table: str) -> str:
        """Generate standardized table key."""
        if catalog and schema:
            return f"{catalog}.{schema}.{table}"
        elif schema:
            return f"{schema}.{table}"
        else:
            return f"default.{table}"
    
