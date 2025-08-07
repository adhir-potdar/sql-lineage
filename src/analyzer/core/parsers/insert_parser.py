"""Parser for INSERT statements."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser
from .select_parser import SelectParser
from ...utils.logging_config import get_logger


class InsertParser(BaseParser):
    """Parser for INSERT statement components."""
    
    def __init__(self, dialect: str = "trino"):
        super().__init__(dialect)
        self.select_parser = SelectParser(dialect)
        self.logger = get_logger('parsers.insert')
    
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse INSERT statement and extract all components."""
        self.logger.info(f"Parsing INSERT statement (length: {len(sql)})")
        self.logger.debug(f"INSERT SQL: {sql[:200]}..." if len(sql) > 200 else f"INSERT SQL: {sql}")
        
        try:
            ast = self.parse_sql(sql)
            
            # Find the INSERT statement
            insert_stmt = self._find_insert_statement(ast)
            if not insert_stmt:
                self.logger.warning("No INSERT statement found in SQL")
                return {}
            
            self.logger.debug("INSERT statement found")
            
            result = {
                'target_table': self.parse_target_table(insert_stmt),
                'insert_columns': self.parse_insert_columns(insert_stmt),
                'source_data': self.parse_source_data(insert_stmt),
                'insert_type': self.determine_insert_type(insert_stmt),
                'on_conflict': self.parse_on_conflict(insert_stmt),
                'returning_clause': self.parse_returning_clause(insert_stmt)
            }
            
            target_table = result.get('target_table', {}).get('name', 'unknown')
            self.logger.info(f"INSERT parsing completed - target: {target_table}")
            return result
            
        except Exception as e:
            self.logger.error(f"INSERT parsing failed: {str(e)}", exc_info=True)
            raise
    
    def _find_insert_statement(self, ast) -> Optional[exp.Insert]:
        """Find INSERT statement in AST."""
        if isinstance(ast, exp.Insert):
            return ast
        
        for node in ast.find_all(exp.Insert):
            return node
        
        return None
    
    def parse_target_table(self, insert_stmt: exp.Insert) -> Dict[str, Any]:
        """Parse target table information."""
        target_info = {
            'name': None,
            'schema': None,
            'catalog': None,
            'full_name': None,
            'alias': None
        }
        
        if insert_stmt.this:
            table_ref = insert_stmt.this
            
            if isinstance(table_ref, exp.Alias):
                target_info['alias'] = table_ref.alias
                table_ref = table_ref.this
            
            target_info['name'] = self.extract_table_name(table_ref)
            target_info['full_name'] = str(table_ref)
            
            # Extract schema and catalog
            parts = str(table_ref).split('.')
            if len(parts) == 3:
                target_info['catalog'] = parts[0]
                target_info['schema'] = parts[1]
                target_info['name'] = parts[2]
            elif len(parts) == 2:
                target_info['schema'] = parts[0]
                target_info['name'] = parts[1]
        
        return target_info
    
    def parse_insert_columns(self, insert_stmt: exp.Insert) -> List[Dict[str, Any]]:
        """Parse columns specified in INSERT clause."""
        columns = []
        
        # Look for column list in INSERT statement
        if hasattr(insert_stmt, 'columns') and insert_stmt.columns:
            for col in insert_stmt.columns:
                column_info = {
                    'name': self.extract_column_name(col),
                    'position': len(columns)
                }
                columns.append(column_info)
        
        return columns
    
    def parse_source_data(self, insert_stmt: exp.Insert) -> Dict[str, Any]:
        """Parse source data for INSERT."""
        source_data = {
            'type': None,
            'values': [],
            'select_query': None,
            'source_tables': [],
            'transformations': []
        }
        
        if insert_stmt.expression:
            expr = insert_stmt.expression
            
            if isinstance(expr, exp.Values):
                # INSERT ... VALUES
                source_data['type'] = 'VALUES'
                source_data['values'] = self._parse_values_clause(expr)
                
            elif isinstance(expr, exp.Select):
                # INSERT ... SELECT
                source_data['type'] = 'SELECT'
                select_data = self.select_parser.parse(str(expr))
                source_data['select_query'] = select_data
                source_data['source_tables'] = self._extract_source_tables_from_select(select_data)
                source_data['transformations'] = self._extract_transformations_from_select(select_data)
                
            elif isinstance(expr, exp.With):
                # INSERT ... WITH ... SELECT
                source_data['type'] = 'CTE_SELECT'
                select_data = self.select_parser.parse(str(expr))
                source_data['select_query'] = select_data
                source_data['source_tables'] = self._extract_source_tables_from_select(select_data)
                source_data['transformations'] = self._extract_transformations_from_select(select_data)
        
        return source_data
    
    def _parse_values_clause(self, values_expr: exp.Values) -> List[List[Any]]:
        """Parse VALUES clause."""
        value_lists = []
        
        for tuple_expr in values_expr.expressions:
            if isinstance(tuple_expr, exp.Tuple):
                values = []
                for value in tuple_expr.expressions:
                    values.append(str(value))
                value_lists.append(values)
        
        return value_lists
    
    def _extract_source_tables_from_select(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source tables from SELECT data."""
        source_tables = []
        
        # FROM tables
        from_tables = select_data.get('from_tables', [])
        for table in from_tables:
            source_tables.append({
                'name': table.get('table_name'),
                'alias': table.get('alias'),
                'type': 'FROM',
                'is_subquery': table.get('is_subquery', False)
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
    
    def _extract_transformations_from_select(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transformations from SELECT data."""
        transformations = []
        
        # WHERE filters
        where_conditions = select_data.get('where_conditions', [])
        if where_conditions:
            transformations.append({
                'type': 'FILTER',
                'conditions': where_conditions
            })
        
        # JOINs
        joins = select_data.get('joins', [])
        for join in joins:
            transformations.append({
                'type': 'JOIN',
                'join_type': join.get('join_type'),
                'table': join.get('table_name'),
                'conditions': join.get('conditions', [])
            })
        
        # GROUP BY
        group_by = select_data.get('group_by', [])
        if group_by:
            transformations.append({
                'type': 'GROUP_BY',
                'columns': group_by
            })
        
        # HAVING
        having_conditions = select_data.get('having_conditions', [])
        if having_conditions:
            transformations.append({
                'type': 'HAVING',
                'conditions': having_conditions
            })
        
        # ORDER BY
        order_by = select_data.get('order_by', [])
        if order_by:
            transformations.append({
                'type': 'ORDER_BY',
                'columns': order_by
            })
        
        return transformations
    
    def determine_insert_type(self, insert_stmt: exp.Insert) -> str:
        """Determine the type of INSERT operation."""
        if insert_stmt.expression:
            expr = insert_stmt.expression
            
            if isinstance(expr, exp.Values):
                return 'INSERT_VALUES'
            elif isinstance(expr, exp.Select):
                return 'INSERT_SELECT'
            elif isinstance(expr, exp.With):
                return 'INSERT_CTE_SELECT'
        
        # Check for specific INSERT variants
        if hasattr(insert_stmt, 'replace') and insert_stmt.replace:
            return 'REPLACE'
        elif hasattr(insert_stmt, 'ignore') and insert_stmt.ignore:
            return 'INSERT_IGNORE'
        
        return 'INSERT'
    
    def parse_on_conflict(self, insert_stmt: exp.Insert) -> Optional[Dict[str, Any]]:
        """Parse ON CONFLICT clause (PostgreSQL) or similar constructs."""
        # This would need dialect-specific implementation
        # For now, return None as it's not commonly used in all dialects
        return None
    
    def parse_returning_clause(self, insert_stmt: exp.Insert) -> Optional[List[Dict[str, Any]]]:
        """Parse RETURNING clause if present."""
        # Look for RETURNING clause
        if hasattr(insert_stmt, 'returning') and insert_stmt.returning:
            returning_columns = []
            
            for expr in insert_stmt.returning.expressions:
                column_info = {
                    'expression': str(expr),
                    'alias': None
                }
                
                if isinstance(expr, exp.Alias):
                    column_info['alias'] = expr.alias
                    column_info['column'] = str(expr.this)
                else:
                    column_info['column'] = str(expr)
                
                returning_columns.append(column_info)
            
            return returning_columns
        
        return None
    
    def get_insert_lineage(self, sql: str) -> Dict[str, Any]:
        """Get complete INSERT lineage information."""
        insert_data = self.parse(sql)
        
        if not insert_data:
            return {}
        
        lineage = {
            'type': 'INSERT_LINEAGE',
            'target_table': insert_data.get('target_table', {}),
            'source_analysis': self._analyze_insert_sources(insert_data),
            'column_mapping': self._analyze_column_mapping(insert_data),
            'transformations': insert_data.get('source_data', {}).get('transformations', []),
            'data_flow': self._build_insert_data_flow(insert_data)
        }
        
        return lineage
    
    def _analyze_insert_sources(self, insert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze source data for INSERT."""
        source_data = insert_data.get('source_data', {})
        
        analysis = {
            'type': source_data.get('type'),
            'source_tables': source_data.get('source_tables', []),
            'has_transformations': len(source_data.get('transformations', [])) > 0,
            'is_static_values': source_data.get('type') == 'VALUES'
        }
        
        if source_data.get('type') == 'VALUES':
            values = source_data.get('values', [])
            analysis['value_rows'] = len(values)
            analysis['value_columns'] = len(values[0]) if values else 0
        
        return analysis
    
    def _analyze_column_mapping(self, insert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze column mapping for INSERT."""
        target_table = insert_data.get('target_table', {})
        insert_columns = insert_data.get('insert_columns', [])
        source_data = insert_data.get('source_data', {})
        
        mapping = {
            'target_table': target_table.get('name'),
            'explicit_columns': len(insert_columns) > 0,
            'column_mappings': []
        }
        
        if source_data.get('type') == 'SELECT':
            select_query = source_data.get('select_query', {})
            select_columns = select_query.get('select_columns', [])
            
            # Map INSERT columns to SELECT columns
            for i, insert_col in enumerate(insert_columns):
                source_col = select_columns[i] if i < len(select_columns) else None
                
                mapping['column_mappings'].append({
                    'target_column': insert_col.get('name'),
                    'source_expression': source_col.get('raw_expression') if source_col else None,
                    'source_table': source_col.get('source_table') if source_col else None,
                    'position': i
                })
        
        return mapping
    
    def _build_insert_data_flow(self, insert_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build data flow for INSERT operation."""
        flow = []
        
        source_data = insert_data.get('source_data', {})
        target_table = insert_data.get('target_table', {})
        
        # Add source tables
        for table in source_data.get('source_tables', []):
            flow.append({
                'type': 'SOURCE',
                'entity': table.get('name'),
                'alias': table.get('alias'),
                'table_type': table.get('type')
            })
        
        # Add transformations
        for transform in source_data.get('transformations', []):
            flow.append({
                'type': 'TRANSFORMATION',
                'transformation_type': transform.get('type'),
                'details': transform
            })
        
        # Add target
        flow.append({
            'type': 'TARGET',
            'entity': target_table.get('name'),
            'operation': 'INSERT',
            'columns': [col.get('name') for col in insert_data.get('insert_columns', [])]
        })
        
        return flow