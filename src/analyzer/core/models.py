"""Data models for SQL lineage analysis."""

from typing import Dict, Set, Optional, List, Any, Union
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

class JoinType(str, Enum):
    """JOIN type enumeration."""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"
    CROSS = "CROSS"

class AggregateType(str, Enum):
    """Aggregate function type enumeration."""
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    STDDEV = "STDDEV"
    VARIANCE = "VARIANCE"
    APPROX_DISTINCT = "APPROX_DISTINCT"
    OTHER = "OTHER"

class OperatorType(str, Enum):
    """SQL operator type enumeration."""
    EQ = "="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "IN"
    NOT_IN = "NOT IN"
    LIKE = "LIKE"
    NOT_LIKE = "NOT LIKE"
    BETWEEN = "BETWEEN"
    NOT_BETWEEN = "NOT BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

@dataclass
class JoinCondition:
    """Represents a JOIN condition."""
    left_column: str
    operator: OperatorType
    right_column: str
    
@dataclass
class FilterCondition:
    """Represents a WHERE/HAVING filter condition."""
    column: str
    operator: OperatorType
    value: Union[str, int, float, List[Any]]
    
@dataclass
class AggregateFunction:
    """Represents an aggregate function."""
    function_type: AggregateType
    column: Optional[str] = None
    distinct: bool = False
    
@dataclass
class WindowFunction:
    """Represents a window function."""
    function_name: str
    partition_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    
@dataclass
class CaseExpression:
    """Represents a CASE expression."""
    when_conditions: List[FilterCondition] = field(default_factory=list)
    then_values: List[Any] = field(default_factory=list)
    else_value: Optional[Any] = None
    
@dataclass
class TableTransformation:
    """Transformation information at table level."""
    source_table: str
    target_table: str
    join_type: Optional[JoinType] = None
    join_conditions: List[JoinCondition] = field(default_factory=list)
    filter_conditions: List[FilterCondition] = field(default_factory=list)
    group_by_columns: List[str] = field(default_factory=list)
    having_conditions: List[FilterCondition] = field(default_factory=list)
    order_by_columns: List[str] = field(default_factory=list)
    
@dataclass
class ColumnTransformation:
    """Transformation information at column level."""
    source_column: str
    target_column: str
    aggregate_function: Optional[AggregateFunction] = None
    window_function: Optional[WindowFunction] = None
    case_expression: Optional[CaseExpression] = None
    expression: Optional[str] = None  # For complex expressions
    
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
    transformations: Dict[str, List[TableTransformation]] = field(default_factory=dict)
    
    def add_dependency(self, target: str, source: str) -> None:
        """Add a table dependency."""
        if target not in self.upstream:
            self.upstream[target] = set()
        if source not in self.downstream:
            self.downstream[source] = set()
        
        self.upstream[target].add(source)
        self.downstream[source].add(target)
        
    def add_transformation(self, target: str, transformation: TableTransformation) -> None:
        """Add a table transformation."""
        if target not in self.transformations:
            self.transformations[target] = []
        self.transformations[target].append(transformation)

@dataclass
class ColumnLineage:
    """Column-level lineage information."""
    upstream: Dict[str, Set[str]] = field(default_factory=dict)
    downstream: Dict[str, Set[str]] = field(default_factory=dict)
    transformations: Dict[str, List[ColumnTransformation]] = field(default_factory=dict)
    
    def add_dependency(self, target_column: str, source_column: str) -> None:
        """Add a column dependency."""
        if target_column not in self.upstream:
            self.upstream[target_column] = set()
        if source_column not in self.downstream:
            self.downstream[source_column] = set()
        
        self.upstream[target_column].add(source_column)
        self.downstream[source_column].add(target_column)
        
    def add_transformation(self, target_column: str, transformation: ColumnTransformation) -> None:
        """Add a column transformation."""
        if target_column not in self.transformations:
            self.transformations[target_column] = []
        self.transformations[target_column].append(transformation)

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
                "downstream": {k: list(v) for k, v in self.table_lineage.downstream.items()},
                "transformations": {k: [self._serialize_table_transformation(t) for t in v] 
                                  for k, v in self.table_lineage.transformations.items()}
            },
            "column_lineage": {
                "upstream": {k: list(v) for k, v in self.column_lineage.upstream.items()},
                "downstream": {k: list(v) for k, v in self.column_lineage.downstream.items()},
                "transformations": {k: [self._serialize_column_transformation(t) for t in v] 
                                   for k, v in self.column_lineage.transformations.items()}
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
    
    def _serialize_table_transformation(self, transformation: TableTransformation) -> Dict[str, Any]:
        """Serialize table transformation for JSON."""
        return {
            "source_table": transformation.source_table,
            "target_table": transformation.target_table,
            "join_type": transformation.join_type.value if transformation.join_type else None,
            "join_conditions": [{
                "left_column": jc.left_column,
                "operator": jc.operator.value,
                "right_column": jc.right_column
            } for jc in transformation.join_conditions],
            "filter_conditions": [{
                "column": fc.column,
                "operator": fc.operator.value,
                "value": fc.value
            } for fc in transformation.filter_conditions],
            "group_by_columns": transformation.group_by_columns,
            "having_conditions": [{
                "column": hc.column,
                "operator": hc.operator.value,
                "value": hc.value
            } for hc in transformation.having_conditions],
            "order_by_columns": transformation.order_by_columns
        }
    
    def _serialize_column_transformation(self, transformation: ColumnTransformation) -> Dict[str, Any]:
        """Serialize column transformation for JSON."""
        result = {
            "source_column": transformation.source_column,
            "target_column": transformation.target_column,
            "expression": transformation.expression
        }
        
        if transformation.aggregate_function:
            result["aggregate_function"] = {
                "function_type": transformation.aggregate_function.function_type.value,
                "column": transformation.aggregate_function.column,
                "distinct": transformation.aggregate_function.distinct
            }
        
        if transformation.window_function:
            result["window_function"] = {
                "function_name": transformation.window_function.function_name,
                "partition_by": transformation.window_function.partition_by,
                "order_by": transformation.window_function.order_by
            }
        
        if transformation.case_expression:
            result["case_expression"] = {
                "when_conditions": [{
                    "column": wc.column,
                    "operator": wc.operator.value,
                    "value": wc.value
                } for wc in transformation.case_expression.when_conditions],
                "then_values": transformation.case_expression.then_values,
                "else_value": transformation.case_expression.else_value
            }
        
        return result