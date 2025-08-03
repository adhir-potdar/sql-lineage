"""Parser for UPDATE statements."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser
from .select_parser import SelectParser
from ...utils.condition_utils import GenericConditionHandler


class UpdateParser(BaseParser):
    """Parser for UPDATE statement components."""
    
    def __init__(self, dialect: str = "trino"):
        super().__init__(dialect)
        self.select_parser = SelectParser(dialect)
    
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse UPDATE statement and extract all components."""
        ast = self.parse_sql(sql)
        
        # Find the UPDATE statement
        update_stmt = self._find_update_statement(ast)
        if not update_stmt:
            return {}
        
        return {
            'target_table': self.parse_target_table(update_stmt),
            'set_clauses': self.parse_set_clauses(update_stmt),
            'from_tables': self.parse_from_tables(update_stmt),
            'joins': self.parse_joins(update_stmt),
            'where_conditions': self.parse_where_conditions(update_stmt),
            'returning_clause': self.parse_returning_clause(update_stmt),
            'update_type': self.determine_update_type(update_stmt)
        }
    
    def _find_update_statement(self, ast) -> Optional[exp.Update]:
        """Find UPDATE statement in AST."""
        if isinstance(ast, exp.Update):
            return ast
        
        for node in ast.find_all(exp.Update):
            return node
        
        return None
    
    def parse_target_table(self, update_stmt: exp.Update) -> Dict[str, Any]:
        """Parse target table information."""
        target_info = {
            'name': None,
            'schema': None,
            'catalog': None,
            'full_name': None,
            'alias': None
        }
        
        if update_stmt.this:
            table_ref = update_stmt.this
            
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
    
    def parse_set_clauses(self, update_stmt: exp.Update) -> List[Dict[str, Any]]:
        """Parse SET clauses in UPDATE statement."""
        set_clauses = []
        
        if hasattr(update_stmt, 'expressions') and update_stmt.expressions:
            for expr in update_stmt.expressions:
                if isinstance(expr, exp.EQ):
                    # Standard SET column = value
                    set_info = {
                        'column': str(expr.left).strip(),
                        'value_expression': str(expr.right).strip(),
                        'value_type': self._determine_value_type(expr.right),
                        'is_subquery': self._is_subquery(expr.right),
                        'source_columns': self._extract_source_columns(expr.right)
                    }
                    set_clauses.append(set_info)
        
        return set_clauses
    
    def _determine_value_type(self, value_expr) -> str:
        """Determine the type of value in SET clause."""
        if isinstance(value_expr, exp.Literal):
            return 'LITERAL'
        elif isinstance(value_expr, exp.Column):
            return 'COLUMN_REFERENCE'
        elif isinstance(value_expr, exp.Subquery):
            return 'SUBQUERY'
        elif isinstance(value_expr, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            return 'ARITHMETIC'
        elif isinstance(value_expr, exp.Function):
            return 'FUNCTION'
        elif isinstance(value_expr, exp.Case):
            return 'CASE_EXPRESSION'
        else:
            return 'EXPRESSION'
    
    def _is_subquery(self, value_expr) -> bool:
        """Check if value expression contains a subquery."""
        return bool(list(value_expr.find_all(exp.Subquery)))
    
    def _extract_source_columns(self, value_expr) -> List[str]:
        """Extract source columns referenced in value expression."""
        columns = []
        
        for col in value_expr.find_all(exp.Column):
            col_name = str(col)
            if col_name not in columns:
                columns.append(col_name)
        
        return columns
    
    def parse_from_tables(self, update_stmt: exp.Update) -> List[Dict[str, Any]]:
        """Parse FROM clause tables (for UPDATE with FROM)."""
        tables = []
        
        # Some dialects support UPDATE ... FROM syntax
        if hasattr(update_stmt, 'from_') and update_stmt.from_:
            for table_expr in update_stmt.from_.expressions:
                table_info = {
                    'table_name': None,
                    'alias': None,
                    'is_subquery': False
                }
                
                if isinstance(table_expr, exp.Alias):
                    table_info['alias'] = table_expr.alias
                    actual_table = table_expr.this
                else:
                    actual_table = table_expr
                
                if isinstance(actual_table, exp.Table):
                    table_info['table_name'] = self.extract_table_name(actual_table)
                elif isinstance(actual_table, exp.Subquery):
                    table_info['is_subquery'] = True
                    table_info['table_name'] = table_info['alias'] or 'subquery'
                
                tables.append(table_info)
        
        return tables
    
    def parse_joins(self, update_stmt: exp.Update) -> List[Dict[str, Any]]:
        """Parse JOIN clauses in UPDATE statement."""
        joins = []
        
        for join in update_stmt.find_all(exp.Join):
            join_info = {
                'join_type': self._get_join_type(join),
                'table_name': None,
                'alias': None,
                'conditions': []
            }
            
            # Get joined table
            if isinstance(join.this, exp.Alias):
                join_info['alias'] = join.this.alias
                join_info['table_name'] = self.extract_table_name(join.this.this)
            else:
                join_info['table_name'] = self.extract_table_name(join.this)
            
            # Get join conditions
            if join.on:
                join_info['conditions'] = self._parse_join_conditions(join.on)
            
            joins.append(join_info)
        
        return joins
    
    def _get_join_type(self, join: exp.Join) -> str:
        """Get JOIN type string."""
        if join.side:
            return f"{join.side.upper()} JOIN"
        elif join.kind:
            return f"{join.kind.upper()} JOIN"
        else:
            return "INNER JOIN"
    
    def _parse_join_conditions(self, condition_expr) -> List[Dict[str, Any]]:
        """Parse JOIN conditions."""
        # Use generic condition handler for join conditions (supports all operators)
        return GenericConditionHandler.extract_join_conditions(
            condition_expr, 
            output_format="dict"
        )
    
    def parse_where_conditions(self, update_stmt: exp.Update) -> List[Dict[str, Any]]:
        """Parse WHERE clause conditions."""
        if not update_stmt.where:
            return []
        
        return self._parse_conditions(update_stmt.where.this)
    
    def _parse_conditions(self, condition_expr) -> List[Dict[str, Any]]:
        """Parse filter conditions from WHERE clause."""
        # Use generic condition handler with dict output format
        return GenericConditionHandler.extract_all_conditions(
            condition_expr, 
            column_resolver=None, 
            output_format="dict"
        )
    
    
    def parse_returning_clause(self, update_stmt: exp.Update) -> Optional[List[Dict[str, Any]]]:
        """Parse RETURNING clause if present."""
        if hasattr(update_stmt, 'returning') and update_stmt.returning:
            returning_columns = []
            
            for expr in update_stmt.returning.expressions:
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
    
    def determine_update_type(self, update_stmt: exp.Update) -> str:
        """Determine the type of UPDATE operation."""
        # Check for different UPDATE variants
        has_joins = bool(list(update_stmt.find_all(exp.Join)))
        has_from = hasattr(update_stmt, 'from_') and update_stmt.from_
        has_subqueries = bool([
            set_clause for set_clause in self.parse_set_clauses(update_stmt)
            if set_clause.get('is_subquery')
        ])
        
        if has_joins:
            return 'UPDATE_WITH_JOIN'
        elif has_from:
            return 'UPDATE_WITH_FROM'
        elif has_subqueries:
            return 'UPDATE_WITH_SUBQUERY'
        else:
            return 'SIMPLE_UPDATE'
    
    def get_update_lineage(self, sql: str) -> Dict[str, Any]:
        """Get complete UPDATE lineage information."""
        update_data = self.parse(sql)
        
        if not update_data:
            return {}
        
        lineage = {
            'type': 'UPDATE_LINEAGE',
            'target_table': update_data.get('target_table', {}),
            'source_analysis': self._analyze_update_sources(update_data),
            'column_updates': self._analyze_column_updates(update_data),
            'transformations': self._analyze_update_transformations(update_data),
            'data_flow': self._build_update_data_flow(update_data)
        }
        
        return lineage
    
    def _analyze_update_sources(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze source data for UPDATE."""
        analysis = {
            'source_tables': [],
            'has_joins': False,
            'has_subqueries': False,
            'self_referential': False
        }
        
        target_table = update_data.get('target_table', {}).get('name')
        
        # Analyze FROM tables
        from_tables = update_data.get('from_tables', [])
        for table in from_tables:
            analysis['source_tables'].append({
                'name': table.get('table_name'),
                'alias': table.get('alias'),
                'type': 'FROM',
                'is_subquery': table.get('is_subquery', False)
            })
            
            if table.get('is_subquery'):
                analysis['has_subqueries'] = True
        
        # Analyze JOIN tables
        joins = update_data.get('joins', [])
        if joins:
            analysis['has_joins'] = True
            for join in joins:
                analysis['source_tables'].append({
                    'name': join.get('table_name'),
                    'alias': join.get('alias'),
                    'type': 'JOIN',
                    'join_type': join.get('join_type')
                })
        
        # Check for self-referential updates
        set_clauses = update_data.get('set_clauses', [])
        for set_clause in set_clauses:
            source_columns = set_clause.get('source_columns', [])
            for col in source_columns:
                if '.' in col:
                    table_part = col.split('.')[0]
                    if table_part == target_table:
                        analysis['self_referential'] = True
                        break
        
        return analysis
    
    def _analyze_column_updates(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze column updates."""
        analysis = {
            'updated_columns': [],
            'update_patterns': {},
            'source_dependencies': []
        }
        
        set_clauses = update_data.get('set_clauses', [])
        for set_clause in set_clauses:
            column_update = {
                'column': set_clause.get('column'),
                'value_type': set_clause.get('value_type'),
                'expression': set_clause.get('value_expression'),
                'source_columns': set_clause.get('source_columns', [])
            }
            analysis['updated_columns'].append(column_update)
            
            # Track update patterns
            value_type = set_clause.get('value_type')
            if value_type not in analysis['update_patterns']:
                analysis['update_patterns'][value_type] = 0
            analysis['update_patterns'][value_type] += 1
            
            # Track source dependencies
            analysis['source_dependencies'].extend(set_clause.get('source_columns', []))
        
        # Remove duplicates from source dependencies
        analysis['source_dependencies'] = list(set(analysis['source_dependencies']))
        
        return analysis
    
    def _analyze_update_transformations(self, update_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze transformations in UPDATE."""
        transformations = []
        
        # WHERE filters
        where_conditions = update_data.get('where_conditions', [])
        if where_conditions:
            transformations.append({
                'type': 'FILTER',
                'conditions': where_conditions,
                'purpose': 'row_selection'
            })
        
        # JOINs
        joins = update_data.get('joins', [])
        for join in joins:
            transformations.append({
                'type': 'JOIN',
                'join_type': join.get('join_type'),
                'table': join.get('table_name'),
                'conditions': join.get('conditions', []),
                'purpose': 'data_enrichment'
            })
        
        # SET clause transformations
        set_clauses = update_data.get('set_clauses', [])
        for set_clause in set_clauses:
            if set_clause.get('value_type') in ['ARITHMETIC', 'FUNCTION', 'CASE_EXPRESSION']:
                transformations.append({
                    'type': 'COLUMN_TRANSFORMATION',
                    'column': set_clause.get('column'),
                    'transformation_type': set_clause.get('value_type'),
                    'expression': set_clause.get('value_expression'),
                    'purpose': 'value_computation'
                })
        
        return transformations
    
    def _build_update_data_flow(self, update_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build data flow for UPDATE operation."""
        flow = []
        
        target_table = update_data.get('target_table', {})
        
        # Add source tables
        source_analysis = self._analyze_update_sources(update_data)
        for table in source_analysis.get('source_tables', []):
            flow.append({
                'type': 'SOURCE',
                'entity': table.get('name'),
                'alias': table.get('alias'),
                'table_type': table.get('type'),
                'join_type': table.get('join_type')
            })
        
        # Add transformations
        transformations = self._analyze_update_transformations(update_data)
        for transform in transformations:
            flow.append({
                'type': 'TRANSFORMATION',
                'transformation_type': transform.get('type'),
                'purpose': transform.get('purpose'),
                'details': transform
            })
        
        # Add target (the table being updated)
        flow.append({
            'type': 'TARGET',
            'entity': target_table.get('name'),
            'operation': 'UPDATE',
            'updated_columns': [
                set_clause.get('column') 
                for set_clause in update_data.get('set_clauses', [])
            ]
        })
        
        return flow