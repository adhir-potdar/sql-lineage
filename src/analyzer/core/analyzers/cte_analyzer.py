"""CTE (Common Table Expression) analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer


class CTEAnalyzer(BaseAnalyzer):
    """Analyzer for CTE statements."""
    
    def analyze_cte(self, sql: str) -> Dict[str, Any]:
        """Analyze CTE statement."""
        cte_data = self.cte_parser.parse(sql)
        cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
        transformation_data = self.transformation_parser.parse(sql)
        
        return {
            'cte_structure': cte_data,
            'cte_lineage': cte_lineage,
            'transformations': transformation_data,
            'execution_order': cte_lineage.get('execution_order', []),
            'final_result': cte_lineage.get('final_result', {}),
            'cte_dependencies': cte_data.get('cte_dependencies', {})
        }