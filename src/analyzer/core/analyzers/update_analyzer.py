"""UPDATE statement analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer
from ...utils.logging_config import get_logger


class UpdateAnalyzer(BaseAnalyzer):
    """Analyzer for UPDATE statements."""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = None, table_registry = None):
        super().__init__(dialect, compatibility_mode, table_registry)
        self.logger = get_logger('analyzers.update')
    
    def analyze_update(self, sql: str) -> Dict[str, Any]:
        """Analyze UPDATE statement."""
        self.logger.info(f"Analyzing UPDATE statement (length: {len(sql)})")
        self.logger.debug(f"UPDATE SQL: {sql[:200]}..." if len(sql) > 200 else f"UPDATE SQL: {sql}")
        
        try:
            self.logger.debug("Parsing UPDATE structure")
            update_data = self.update_parser.parse(sql)
            self.logger.debug("Building UPDATE lineage")
            update_lineage = self.update_parser.get_update_lineage(sql)
            self.logger.info("UPDATE parsing completed successfully")
        
            result = {
                'update_structure': update_data,
                'update_lineage': update_lineage,
                'target_table': update_data.get('target_table', {}),
                'source_analysis': update_lineage.get('source_analysis', {}),
                'column_updates': update_lineage.get('column_updates', {}),
                'data_flow': update_lineage.get('data_flow', [])
            }
            
            target_table = update_data.get('target_table', {}).get('name', 'unknown')
            column_count = len(update_lineage.get('column_updates', {}))
            self.logger.info(f"UPDATE analysis completed - target: {target_table}, {column_count} columns updated")
            return result
            
        except Exception as e:
            self.logger.error(f"UPDATE analysis failed: {str(e)}", exc_info=True)
            raise