"""UPDATE statement analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer
from ..parsers import UpdateParser


class UpdateAnalyzer(BaseAnalyzer):
    """Analyzer for UPDATE statements."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize UPDATE analyzer."""
        super().__init__(dialect)
        self.update_parser = UpdateParser(dialect)
    
    def analyze(self, sql: str) -> Dict[str, Any]:
        """Analyze UPDATE statement."""
        try:
            self._validate_input(sql)
            
            update_data = self.update_parser.parse(sql)
            update_lineage = self.update_parser.get_update_lineage(sql)
            
            return {
                'update_structure': update_data,
                'update_lineage': update_lineage,
                'target_table': update_data.get('target_table', {}),
                'source_analysis': update_lineage.get('source_analysis', {}),
                'column_updates': update_lineage.get('column_updates', {}),
                'data_flow': update_lineage.get('data_flow', []),
                'success': True
            }
        except Exception as e:
            return self._handle_analysis_error(e, 'UPDATE')