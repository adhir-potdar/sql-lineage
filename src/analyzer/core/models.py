"""Data models for SQL lineage analysis."""

from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TableType(str, Enum):
    """Table type enumeration."""
    TABLE = "TABLE"
    VIEW = "VIEW"
    EXTERNAL_TABLE = "EXTERNAL_TABLE"
    MATERIALIZED_VIEW = "MATERIALIZED_VIEW"
    CTE = "CTE"

@dataclass
class ColumnMetadata:
    """Metadata for a database column."""
    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: Optional[str] = None
    description: Optional[str] = None
    default_value: Optional[str] = None

@dataclass
class TableMetadata:
    """Metadata for a database table."""
    catalog: Optional[str] = None
    schema: Optional[str] = None
    table: str = ""
    columns: List[ColumnMetadata] = field(default_factory=list)
    table_type: TableType = TableType.TABLE
    description: Optional[str] = None
    owner: Optional[str] = None
    created_date: Optional[datetime] = None
    row_count: Optional[int] = None
    storage_format: Optional[str] = None

@dataclass
class TableLineage:
    """Table-level lineage information."""
    upstream: Dict[str, Set[str]] = field(default_factory=dict)
    downstream: Dict[str, Set[str]] = field(default_factory=dict)
    
    def add_dependency(self, target: str, source: str) -> None:
        """Add a table dependency."""
        if target not in self.upstream:
            self.upstream[target] = set()
        if source not in self.downstream:
            self.downstream[source] = set()
        
        self.upstream[target].add(source)
        self.downstream[source].add(target)

@dataclass
class ColumnLineage:
    """Column-level lineage information."""
    upstream: Dict[str, Set[str]] = field(default_factory=dict)
    downstream: Dict[str, Set[str]] = field(default_factory=dict)
    
    def add_dependency(self, target_column: str, source_column: str) -> None:
        """Add a column dependency."""
        if target_column not in self.upstream:
            self.upstream[target_column] = set()
        if source_column not in self.downstream:
            self.downstream[source_column] = set()
        
        self.upstream[target_column].add(source_column)
        self.downstream[source_column].add(target_column)

@dataclass
class LineageResult:
    """Complete lineage analysis result."""
    sql: str
    dialect: str
    table_lineage: TableLineage
    column_lineage: ColumnLineage
    metadata: Dict[str, TableMetadata] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if analysis had errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if analysis had warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sql": self.sql,
            "dialect": self.dialect,
            "table_lineage": {
                "upstream": {k: list(v) for k, v in self.table_lineage.upstream.items()},
                "downstream": {k: list(v) for k, v in self.table_lineage.downstream.items()}
            },
            "column_lineage": {
                "upstream": {k: list(v) for k, v in self.column_lineage.upstream.items()},
                "downstream": {k: list(v) for k, v in self.column_lineage.downstream.items()}
            },
            "metadata": {k: self._serialize_metadata(v) for k, v in self.metadata.items()},
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def _serialize_metadata(self, metadata: TableMetadata) -> Dict[str, Any]:
        """Serialize table metadata for JSON."""
        return {
            "catalog": metadata.catalog,
            "schema": metadata.schema,
            "table": metadata.table,
            "table_type": metadata.table_type.value,
            "description": metadata.description,
            "owner": metadata.owner,
            "created_date": metadata.created_date.isoformat() if metadata.created_date else None,
            "row_count": metadata.row_count,
            "storage_format": metadata.storage_format,
            "columns": [
                {
                    "name": col.name,
                    "data_type": col.data_type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "foreign_key": col.foreign_key,
                    "description": col.description,
                    "default_value": col.default_value
                }
                for col in metadata.columns
            ]
        }