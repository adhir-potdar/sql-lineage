"""UPDATE-specific analyzer methods."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class UpdateAnalyzer(BaseAnalyzer):
    """Handles UPDATE-specific analysis and lineage building."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize UPDATE analyzer."""
        super().__init__(dialect)
        # These will be injected from the main analyzer
        self.update_parser = None
    
    def analyze_update(self, sql: str) -> Dict[str, Any]:
        """Analyze UPDATE statement."""
        update_data = self.update_parser.parse(sql) if self.update_parser else {}
        update_lineage = self.update_parser.get_update_lineage(sql) if self.update_parser else {}
        
        return {
            'update_structure': update_data,
            'update_lineage': update_lineage,
            'target_table': update_data.get('target_table', {}),
            'source_analysis': update_lineage.get('source_analysis', {}),
            'column_updates': update_lineage.get('column_updates', {}),
            'data_flow': update_lineage.get('data_flow', [])
        }