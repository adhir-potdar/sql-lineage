"""UPDATE statement analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class UpdateAnalyzer(BaseAnalyzer):
    """Analyzer for UPDATE statements."""
    
    def analyze_update(self, sql: str) -> Dict[str, Any]:
        """Analyze UPDATE statement."""
        update_data = self.update_parser.parse(sql)
        update_lineage = self.update_parser.get_update_lineage(sql)
        
        return {
            'update_structure': update_data,
            'update_lineage': update_lineage,
            'target_table': update_data.get('target_table', {}),
            'source_analysis': update_lineage.get('source_analysis', {}),
            'column_updates': update_lineage.get('column_updates', {}),
            'data_flow': update_lineage.get('data_flow', [])
        }