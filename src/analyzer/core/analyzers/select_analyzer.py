"""SELECT statement analyzer."""

from typing import Dict, Any, List
from .base_analyzer import BaseAnalyzer
from ..parsers import SelectParser, TransformationParser


class SelectAnalyzer(BaseAnalyzer):
    """Analyzer for SELECT statements."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize SELECT analyzer."""
        super().__init__(dialect)
        self.select_parser = SelectParser(dialect)
        self.transformation_parser = TransformationParser(dialect)
    
    def analyze(self, sql: str) -> Dict[str, Any]:
        """Analyze SELECT statement."""
        try:
            self._validate_input(sql)
            
            select_data = self.select_parser.parse(sql)
            transformation_data = self.transformation_parser.parse(sql)
            
            return {
                'query_structure': select_data,
                'transformations': transformation_data,
                'lineage': self._build_select_lineage(select_data, transformation_data),
                'result_columns': self._extract_result_columns(select_data),
                'source_tables': self._extract_source_tables(select_data),
                'success': True
            }
        except Exception as e:
            return self._handle_analysis_error(e, 'SELECT')
    
    def _build_select_lineage(self, select_data: Dict[str, Any], transformation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build lineage chain for SELECT statement."""
        lineage = {
            'type': 'SELECT_LINEAGE',
            'flow': []
        }
        
        # Source tables
        source_tables = select_data.get('from_tables', [])
        for table in source_tables:
            lineage['flow'].append({
                'type': 'SOURCE',
                'entity': table.get('table_name'),
                'alias': table.get('alias'),
                'columns_used': self._get_columns_used_from_table(table, select_data)
            })
        
        # Transformations
        transformations = []
        
        # Add filters
        filters = transformation_data.get('filters', {})
        if filters.get('conditions'):
            transformations.append({
                'type': 'FILTER',
                'conditions': filters['conditions']
            })
        
        # Add joins
        joins = transformation_data.get('joins', [])
        for join in joins:
            transformations.append({
                'type': 'JOIN',
                'join_type': join.get('join_type'),
                'table': join.get('table_name'),
                'conditions': join.get('conditions', [])
            })
        
        # Add aggregations
        aggregations = transformation_data.get('aggregations', {})
        if aggregations.get('group_by_columns'):
            transformations.append({
                'type': 'GROUP_BY',
                'columns': aggregations['group_by_columns']
            })
        
        # Add transformations to flow
        for transform in transformations:
            lineage['flow'].append(transform)
        
        # Final result
        lineage['flow'].append({
            'type': 'RESULT',
            'entity': 'QUERY_RESULT',
            'columns': self._extract_result_columns(select_data)
        })
        
        return lineage
    
    def _get_columns_used_from_table(self, table: Dict[str, Any], select_data: Dict[str, Any]) -> List[str]:
        """Get columns used from a specific table."""
        table_name = table.get('table_name')
        table_alias = table.get('alias')
        used_columns = []
        
        # Check select columns
        select_columns = select_data.get('select_columns', [])
        for col in select_columns:
            source_table = col.get('source_table')
            if source_table == table_name or source_table == table_alias:
                used_columns.append(col.get('column_name'))
        
        # Check WHERE conditions
        where_conditions = select_data.get('where_conditions', [])
        for condition in where_conditions:
            column = condition.get('column', '')
            if '.' in column:
                col_table, col_name = column.split('.', 1)
                if col_table == table_name or col_table == table_alias:
                    used_columns.append(col_name)
        
        # Check JOIN conditions
        joins = select_data.get('joins', [])
        for join in joins:
            for condition in join.get('conditions', []):
                left_col = condition.get('left_column', '')
                right_col = condition.get('right_column', '')
                
                # Check both sides of join condition
                for col in [left_col, right_col]:
                    if '.' in col:
                        col_table, col_name = col.split('.', 1)
                        if col_table == table_name or col_table == table_alias:
                            used_columns.append(col_name)
        
        return list(set(used_columns))  # Remove duplicates
    
    def _extract_result_columns(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract result columns with their metadata."""
        result_columns = []
        
        select_columns = select_data.get('select_columns', [])
        for col in select_columns:
            result_columns.append({
                'name': col.get('column_name'),
                'alias': col.get('alias'),
                'source_table': col.get('source_table'),
                'expression': col.get('raw_expression'),
                'is_computed': col.get('is_aggregate') or col.get('is_window_function')
            })
        
        return result_columns
    
    def _extract_source_tables(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source tables with their metadata."""
        source_tables = []
        
        # FROM tables
        from_tables = select_data.get('from_tables', [])
        for table in from_tables:
            source_tables.append({
                'name': table.get('table_name'),
                'alias': table.get('alias'),
                'type': 'FROM'
            })
        
        # JOIN tables
        joins = select_data.get('joins', [])
        for join in joins:
            source_tables.append({
                'name': join.get('table_name'),
                'alias': join.get('alias'),
                'type': 'JOIN',
                'join_type': join.get('join_type')
            })
        
        return source_tables