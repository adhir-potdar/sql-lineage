"""SQL Lineage Analyzer - Production-quality SQL lineage analysis tool."""

from .core.analyzer import SQLLineageAnalyzer
from .core.models import LineageResult, TableLineage, ColumnLineage

__version__ = "1.0.0"
__all__ = ["SQLLineageAnalyzer", "LineageResult", "TableLineage", "ColumnLineage"]