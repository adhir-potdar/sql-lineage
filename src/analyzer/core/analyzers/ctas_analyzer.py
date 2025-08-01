"""CTAS (CREATE TABLE AS SELECT) analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer
from ..parsers import CTASParser, TransformationParser


class CTASAnalyzer(BaseAnalyzer):
    """Analyzer for CREATE TABLE AS SELECT statements."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize CTAS analyzer."""
        super().__init__(dialect)
        self.ctas_parser = CTASParser(dialect)
        self.transformation_parser = TransformationParser(dialect)
    
    def analyze(self, sql: str) -> Dict[str, Any]:
        """Analyze CTAS statement."""
        try:
            self._validate_input(sql)
            
            ctas_data = self.ctas_parser.parse(sql)
            ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
            transformation_data = self.transformation_parser.parse(sql)
            
            return {
                'ctas_structure': ctas_data,
                'ctas_lineage': ctas_lineage,
                'transformations': transformation_data,
                'target_table': ctas_data.get('target_table', {}),
                'source_analysis': ctas_lineage.get('source_analysis', {}),
                'ctas_transformations': ctas_lineage.get('transformations', []),
                'success': True
            }
        except Exception as e:
            return self._handle_analysis_error(e, 'CTAS')