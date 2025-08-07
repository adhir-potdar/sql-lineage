"""INSERT statement analyzer."""

from typing import Dict, Any
from .base_analyzer import BaseAnalyzer
from ...utils.logging_config import get_logger


class InsertAnalyzer(BaseAnalyzer):
    """Analyzer for INSERT statements."""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = None, table_registry = None):
        super().__init__(dialect, compatibility_mode, table_registry)
        self.logger = get_logger('analyzers.insert')
    
    def analyze_insert(self, sql: str) -> Dict[str, Any]:
        """Analyze INSERT statement."""
        self.logger.info(f"Analyzing INSERT statement (length: {len(sql)})")
        self.logger.debug(f"INSERT SQL: {sql[:200]}..." if len(sql) > 200 else f"INSERT SQL: {sql}")
        
        try:
            self.logger.debug("Parsing INSERT structure")
            insert_data = self.insert_parser.parse(sql)
            self.logger.debug("Building INSERT lineage")
            insert_lineage = self.insert_parser.get_insert_lineage(sql)
            self.logger.info("INSERT parsing completed successfully")
        
            result = {
                'insert_structure': insert_data,
                'insert_lineage': insert_lineage,
                'target_table': insert_data.get('target_table', {}),
                'source_analysis': insert_lineage.get('source_analysis', {}),
                'data_flow': insert_lineage.get('data_flow', [])
            }
            
            target_table = insert_data.get('target_table', {}).get('name', 'unknown')
            source_count = len(insert_lineage.get('source_analysis', {}).get('source_tables', []))
            self.logger.info(f"INSERT analysis completed - target: {target_table}, {source_count} source tables")
            return result
            
        except Exception as e:
            self.logger.error(f"INSERT analysis failed: {str(e)}", exc_info=True)
            raise