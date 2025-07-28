"""JSON output formatter."""

import json
from typing import Dict, Any, Optional
from ..core.models import LineageResult


class JSONFormatter:
    """Formats lineage results as JSON."""
    
    def __init__(self, indent: Optional[int] = 2):
        """
        Initialize JSON formatter.
        
        Args:
            indent: JSON indentation level (None for compact output)
        """
        self.indent = indent
    
    def format(self, result: LineageResult) -> str:
        """
        Format lineage result as JSON string.
        
        Args:
            result: LineageResult to format
            
        Returns:
            JSON string representation
        """
        data = result.to_dict()
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def format_to_file(self, result: LineageResult, file_path: str) -> None:
        """
        Format lineage result and write to file.
        
        Args:
            result: LineageResult to format
            file_path: Output file path
        """
        json_str = self.format(result)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
    
    def format_table_lineage_only(self, result: LineageResult) -> str:
        """
        Format only table lineage as JSON.
        
        Args:
            result: LineageResult to format
            
        Returns:
            JSON string with only table lineage
        """
        data = {
            "table_lineage": {
                "upstream": {k: list(v) for k, v in result.table_lineage.upstream.items()},
                "downstream": {k: list(v) for k, v in result.table_lineage.downstream.items()}
            }
        }
        return json.dumps(data, indent=self.indent, ensure_ascii=False)
    
    def format_column_lineage_only(self, result: LineageResult) -> str:
        """
        Format only column lineage as JSON.
        
        Args:
            result: LineageResult to format
            
        Returns:
            JSON string with only column lineage
        """
        data = {
            "column_lineage": {
                "upstream": {k: list(v) for k, v in result.column_lineage.upstream.items()},
                "downstream": {k: list(v) for k, v in result.column_lineage.downstream.items()}
            }
        }
        return json.dumps(data, indent=self.indent, ensure_ascii=False)