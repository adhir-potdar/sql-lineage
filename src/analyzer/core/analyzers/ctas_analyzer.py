"""CTAS-specific analyzer methods."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class CTASAnalyzer(BaseAnalyzer):
    """Handles CTAS-specific analysis and lineage building."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize CTAS analyzer."""
        super().__init__(dialect)
        # These will be injected from the main analyzer
        self.ctas_parser = None
        self.transformation_parser = None
    
    def analyze_ctas(self, sql: str) -> Dict[str, Any]:
        """Analyze CREATE TABLE AS SELECT statement."""
        ctas_data = self.ctas_parser.parse(sql) if self.ctas_parser else {}
        ctas_lineage = self.ctas_parser.get_ctas_lineage(sql) if self.ctas_parser else {}
        transformation_data = self.transformation_parser.parse(sql) if self.transformation_parser else {}
        
        return {
            'ctas_structure': ctas_data,
            'ctas_lineage': ctas_lineage,
            'transformations': transformation_data,
            'target_table': ctas_data.get('target_table', {}),
            'source_analysis': ctas_lineage.get('source_analysis', {}),
            'ctas_transformations': ctas_lineage.get('transformations', [])
        }