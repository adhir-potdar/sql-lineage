"""CTE (Common Table Expression) analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer
from ..parsers import CTEParser, TransformationParser


class CTEAnalyzer(BaseAnalyzer):
    """Analyzer for CTE (WITH clause) statements."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize CTE analyzer."""
        super().__init__(dialect)
        self.cte_parser = CTEParser(dialect)
        self.transformation_parser = TransformationParser(dialect)
    
    def analyze(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE statement."""
        try:
            self._validate_input(sql)
            
            cte_data = self.cte_parser.parse(sql)
            cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
            transformation_data = self.transformation_parser.parse(sql)
            
            return {
                'cte_structure': cte_data,
                'cte_lineage': cte_lineage,
                'transformations': transformation_data,
                'execution_order': cte_lineage.get('execution_order', []),
                'final_result': cte_lineage.get('final_result', {}),
                'cte_dependencies': cte_data.get('cte_dependencies', {}),
                'success': True
            }
        except Exception as e:
            return self._handle_analysis_error(e, 'CTE')