"""INSERT statement analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class InsertAnalyzer(BaseAnalyzer):
    """Analyzer for INSERT statements."""
    
    def analyze_insert(self, sql: str) -> Dict[str, Any]:
        """Analyze INSERT statement."""
        insert_data = self.insert_parser.parse(sql)
        insert_lineage = self.insert_parser.get_insert_lineage(sql)
        
        return {
            'insert_structure': insert_data,
            'insert_lineage': insert_lineage,
            'target_table': insert_data.get('target_table', {}),
            'source_analysis': insert_lineage.get('source_analysis', {}),
            'data_flow': insert_lineage.get('data_flow', [])
        }