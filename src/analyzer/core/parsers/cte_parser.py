"""Parser for Common Table Expressions (CTEs)."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser
from .select_parser import SelectParser
from ...utils.logging_config import get_logger


class CTEParser(BaseParser):
    """Parser for Common Table Expressions."""
    
    def __init__(self, dialect: str = "trino"):
        super().__init__(dialect)
        self.select_parser = SelectParser(dialect)
        self.logger = get_logger('parsers.cte')
    
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse CTE structure and dependencies."""
        self.logger.info(f"Parsing CTE statement (length: {len(sql)})")
        self.logger.debug(f"CTE SQL: {sql[:200]}..." if len(sql) > 200 else f"CTE SQL: {sql}")
        
        try:
            ast = self.parse_sql(sql)
            self.logger.debug("AST parsed successfully")
        
            # Look for WITH statement - it might be at root level or nested
            with_stmt = None
            main_query = None
            
            if isinstance(ast, exp.With):
                with_stmt = ast
                main_query = ast.this
                self.logger.debug("Found WITH statement at root level")
            else:
                # Look for WITH statements in the AST
                with_nodes = list(ast.find_all(exp.With))
                self.logger.debug(f"Found {len(with_nodes)} WITH statements in AST")
                if with_nodes:
                    with_stmt = with_nodes[0]  # Use the first WITH statement found
                    # For nested WITH, the main query is the parent of the WITH
                    main_query = ast
            
            if not with_stmt:
                self.logger.warning("No WITH statement found in CTE SQL")
                return {}
        
            result = {
                'ctes': self.parse_cte_definitions(with_stmt),
                'main_query': self.parse_main_query(main_query),
                'cte_dependencies': self.analyze_cte_dependencies(with_stmt, main_query)
            }
            
            self.logger.info(f"CTE parsing completed - found {len(result['ctes'])} CTEs")
            return result
            
        except Exception as e:
            self.logger.error(f"CTE parsing failed: {str(e)}", exc_info=True)
            raise
    
    def parse_cte_definitions(self, with_stmt: exp.With) -> List[Dict[str, Any]]:
        """Parse all CTE definitions."""
        ctes = []
        
        for cte in with_stmt.expressions:
            if isinstance(cte, exp.CTE):
                cte_data = self._parse_single_cte(cte)
                ctes.append(cte_data)
        
        return ctes
    
    def _parse_single_cte(self, cte: exp.CTE) -> Dict[str, Any]:
        """Parse a single CTE definition."""
        cte_data = {
            'name': cte.alias,
            'columns': [],
            'source_tables': [],
            'transformations': [],
            'sql': str(cte.this),
            'is_recursive': False
        }
        
        # Parse the CTE's SELECT statement
        if isinstance(cte.this, exp.Select):
            select_data = self.select_parser.parse(str(cte.this))
            
            # Extract output columns
            cte_data['columns'] = self._extract_cte_output_columns(select_data)
            
            # Extract source tables
            cte_data['source_tables'] = self._extract_source_tables(select_data)
            
            # Extract transformations
            cte_data['transformations'] = self._extract_cte_transformations(select_data)
        
        return cte_data
    
    def _extract_cte_output_columns(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract output columns from CTE SELECT data."""
        output_columns = []
        
        select_columns = select_data.get('select_columns', [])
        for col in select_columns:
            column_info = {
                'name': col.get('column_name'),
                'alias': col.get('alias'),
                'source_column': None,
                'source_table': col.get('source_table'),
                'expression': col.get('raw_expression'),
                'is_computed': col.get('is_computed') or col.get('is_aggregate') or col.get('is_window_function')
            }
            
            # Determine source column
            if col.get('source_table') and not column_info['is_computed']:
                column_info['source_column'] = f"{col['source_table']}.{col['column_name']}"
            
            output_columns.append(column_info)
        
        return output_columns
    
    def _extract_source_tables(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source tables from CTE SELECT data."""
        source_tables = []
        
        # Add tables from FROM clause
        from_tables = select_data.get('from_tables', [])
        for table in from_tables:
            source_tables.append({
                'name': table.get('table_name'),
                'alias': table.get('alias'),
                'type': 'subquery' if table.get('is_subquery') else 'table'
            })
        
        # Add tables from JOINs
        joins = select_data.get('joins', [])
        for join in joins:
            source_tables.append({
                'name': join.get('table_name'),
                'alias': join.get('alias'),
                'type': 'table',
                'join_type': join.get('join_type')
            })
        
        return source_tables
    
    def _extract_cte_transformations(self, select_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transformations applied in the CTE."""
        transformations = []
        
        # Add WHERE filters
        where_conditions = select_data.get('where_conditions', [])
        if where_conditions:
            transformations.append({
                'type': 'FILTER',
                'conditions': where_conditions
            })
        
        # Add JOINs
        joins = select_data.get('joins', [])
        for join in joins:
            transformations.append({
                'type': 'JOIN',
                'join_type': join.get('join_type'),
                'table': join.get('table_name'),
                'conditions': join.get('conditions', [])
            })
        
        # Add GROUP BY
        group_by = select_data.get('group_by', [])
        if group_by:
            transformations.append({
                'type': 'GROUP_BY',
                'columns': group_by
            })
        
        # Add HAVING
        having_conditions = select_data.get('having_conditions', [])
        if having_conditions:
            transformations.append({
                'type': 'HAVING',
                'conditions': having_conditions
            })
        
        # Add ORDER BY
        order_by = select_data.get('order_by', [])
        if order_by:
            transformations.append({
                'type': 'ORDER_BY',
                'columns': order_by
            })
        
        # Add LIMIT
        limit_clause = select_data.get('limit_clause')
        if limit_clause:
            transformations.append({
                'type': 'LIMIT',
                'limit': limit_clause.get('limit'),
                'offset': limit_clause.get('offset')
            })
        
        return transformations
    
    def parse_main_query(self, main_query) -> Dict[str, Any]:
        """Parse the main query that uses CTEs."""
        if main_query is None:
            return {}
        main_query_sql = str(main_query)
        return self.select_parser.parse(main_query_sql)
    
    def analyze_cte_dependencies(self, with_stmt: exp.With, main_query=None) -> Dict[str, List[str]]:
        """Analyze dependencies between CTEs and main query."""
        dependencies = {}
        cte_names = set()
        
        # Collect CTE names
        for cte in with_stmt.expressions:
            if isinstance(cte, exp.CTE):
                cte_names.add(cte.alias)
        
        # Analyze each CTE's dependencies
        for cte in with_stmt.expressions:
            if isinstance(cte, exp.CTE):
                deps = self._find_cte_dependencies(cte.this, cte_names)
                dependencies[cte.alias] = deps
        
        # Analyze main query dependencies
        if main_query is not None:
            main_deps = self._find_cte_dependencies(main_query, cte_names)
            dependencies['main_query'] = main_deps
        else:
            dependencies['main_query'] = []
        
        return dependencies
    
    def _find_cte_dependencies(self, query_node, cte_names: set) -> List[str]:
        """Find which CTEs a query depends on."""
        dependencies = []
        
        # Find table references in the query
        for table in query_node.find_all(exp.Table):
            table_name = self.extract_table_name(table)
            if table_name in cte_names:
                dependencies.append(table_name)
        
        return list(set(dependencies))  # Remove duplicates
    
    def get_cte_lineage_chain(self, sql: str) -> Dict[str, Any]:
        """Get complete CTE lineage chain."""
        cte_data = self.parse(sql)
        
        if not cte_data:
            return {}
        
        lineage_chain = {
            'type': 'CTE_CHAIN',
            'ctes': {},
            'execution_order': [],
            'final_result': None
        }
        
        # Build execution order based on dependencies
        cte_deps = cte_data.get('cte_dependencies', {})
        ctes = cte_data.get('ctes', [])
        
        # Simple topological sort for execution order
        remaining_ctes = {cte['name']: cte for cte in ctes}
        execution_order = []
        
        while remaining_ctes:
            # Find CTEs with no unresolved dependencies
            ready_ctes = []
            for cte_name, cte in remaining_ctes.items():
                deps = cte_deps.get(cte_name, [])
                if all(dep in execution_order for dep in deps):
                    ready_ctes.append(cte_name)
            
            if not ready_ctes:
                # Circular dependency or error - add remaining arbitrarily
                ready_ctes = list(remaining_ctes.keys())
            
            for cte_name in ready_ctes:
                execution_order.append(cte_name)
                lineage_chain['ctes'][cte_name] = remaining_ctes.pop(cte_name)
        
        lineage_chain['execution_order'] = execution_order
        lineage_chain['final_result'] = cte_data.get('main_query', {})
        
        return lineage_chain