"""INSERT-specific analyzer methods."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class InsertAnalyzer(BaseAnalyzer):
    """Handles INSERT-specific analysis and lineage building."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize INSERT analyzer."""
        super().__init__(dialect)
        # These will be injected from the main analyzer
        self.insert_parser = None
    
    def analyze_insert(self, sql: str) -> Dict[str, Any]:
        """Analyze INSERT statement."""
        insert_data = self.insert_parser.parse(sql) if self.insert_parser else {}
        insert_lineage = self.insert_parser.get_insert_lineage(sql) if self.insert_parser else {}
        
        return {
            'insert_structure': insert_data,
            'insert_lineage': insert_lineage,
            'target_table': insert_data.get('target_table', {}),
            'source_analysis': insert_lineage.get('source_analysis', {}),
            'data_flow': insert_lineage.get('data_flow', [])
        }