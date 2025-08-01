"""CTAS (CREATE TABLE AS SELECT) statement analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class CTASAnalyzer(BaseAnalyzer):
    """Analyzer for CREATE TABLE AS SELECT statements."""
    
    def analyze_ctas(self, sql: str) -> Dict[str, Any]:
        """Analyze CREATE TABLE AS SELECT statement."""
        ctas_data = self.ctas_parser.parse(sql)
        ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'ctas_structure': ctas_data,
            'ctas_lineage': ctas_lineage,
            'transformations': transformation_data,
            'target_table': ctas_data.get('target_table', {}),
            'source_analysis': ctas_lineage.get('source_analysis', {}),
            'ctas_transformations': ctas_lineage.get('transformations', [])
        }