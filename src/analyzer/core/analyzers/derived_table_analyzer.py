"""Derived Table Analyzer for FROM clause subqueries.

This module provides specialized analysis for derived tables (subqueries in FROM clauses)
to create comprehensive 3-layer lineage: source_table → derived_table → query_result
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import sqlglot
from sqlglot import exp
from ...utils.logging_config import get_logger
from ...utils.sql_parsing_utils import clean_table_name_quotes


class DerivedTableAnalyzer:
    """Specialized analyzer for FROM clause derived tables (subqueries)."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize the derived table analyzer."""
        self.dialect = dialect
        self.logger = get_logger('analyzers.derived_table')
    
    def analyze_derived_tables(self, sql: str) -> Dict[str, Any]:
        """
        Analyze derived tables in FROM clauses and create 3-layer lineage structure.
        
        Args:
            sql: SQL query string to analyze
            
        Returns:
            Dict containing derived table analysis with 3-layer flow
        """
        self.logger.info(f"Analyzing derived tables in SQL (length: {len(sql)})")
        self.logger.debug(f"Derived table SQL: {sql[:200]}..." if len(sql) > 200 else f"Derived table SQL: {sql}")
        
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Find derived tables in FROM clauses
            derived_tables = self._extract_from_subqueries(parsed)
            
            if not derived_tables:
                return {}
            
            # Process each derived table
            analysis_result = {}
            for derived_table in derived_tables:
                table_analysis = self._process_derived_table(derived_table, parsed)
                if table_analysis:
                    analysis_result.update(table_analysis)
            
            return analysis_result
            
        except Exception as e:
            # Return empty result on parsing errors
            return {}
    
    def _extract_from_subqueries(self, parsed_sql) -> List[Dict]:
        """Find all subqueries in FROM clauses and JOIN clauses."""
        derived_tables = []
        
        # Find FROM clauses
        for from_clause in parsed_sql.find_all(exp.From):
            if isinstance(from_clause.this, exp.Subquery):
                subquery = from_clause.this
                alias = str(subquery.alias) if subquery.alias else f"derived_table_{len(derived_tables)}"
                
                derived_tables.append({
                    'subquery': subquery.this,  # The inner SELECT
                    'alias': alias,
                    'full_subquery': subquery
                })
        
        # Find JOIN clauses with subqueries
        for join_clause in parsed_sql.find_all(exp.Join):
            if isinstance(join_clause.this, exp.Subquery):
                subquery = join_clause.this
                alias = str(subquery.alias) if subquery.alias else f"derived_table_{len(derived_tables)}"
                
                derived_tables.append({
                    'subquery': subquery.this,  # The inner SELECT
                    'alias': alias,
                    'full_subquery': subquery
                })
        
        return derived_tables
    
    def _process_derived_table(self, derived_table: Dict, outer_query) -> Dict[str, Any]:
        """Process a single derived table and create 3-layer lineage."""
        subquery = derived_table['subquery']
        alias = derived_table['alias']
        
        # Extract source table from the subquery
        source_table = self._extract_source_table(subquery)
        if not source_table:
            return {}
        
        # Create derived table entity name
        derived_table_entity = f"derived_table_{alias}"
        
        # Analyze columns and transformations in each layer
        layer_analysis = self._analyze_three_layers(source_table, derived_table_entity, subquery, outer_query, alias)
        
        return layer_analysis
    
    def _extract_source_table(self, subquery) -> Optional[str]:
        """Extract the source table name from the subquery."""
        from_clause = subquery.find(exp.From)
        if from_clause and isinstance(from_clause.this, exp.Table):
            return str(from_clause.this.name)
        return None
    
    def _analyze_three_layers(self, source_table: str, derived_table_entity: str, 
                            subquery, outer_query, alias: str) -> Dict[str, Any]:
        """Analyze the complete 3-layer flow: source → derived → result."""
        
        # Layer 1: Source table columns (from subquery SELECT and aggregations)
        source_columns = self._extract_source_table_columns(subquery, source_table)
        
        # Layer 2: Derived table columns (subquery SELECT expressions)
        derived_columns = self._extract_derived_table_columns(subquery)
        
        # Layer 3: Query result columns (outer SELECT expressions)
        result_columns = self._extract_query_result_columns(outer_query, alias, derived_columns)
        
        # Layer transformations
        source_to_derived_trans = self._extract_source_to_derived_transformations(subquery, source_table, derived_table_entity)
        derived_to_result_trans = self._extract_derived_to_result_transformations(outer_query, derived_table_entity, alias)
        
        # Build the 3-layer structure
        return self._build_three_layer_structure(
            source_table, derived_table_entity, source_columns, derived_columns, result_columns,
            source_to_derived_trans, derived_to_result_trans
        )
    
    def _extract_source_table_columns(self, subquery, source_table: str) -> List[Dict]:
        """Extract columns needed from the source table."""
        columns = []
        
        # Get columns from SELECT expressions
        for expr in subquery.expressions:
            if isinstance(expr, exp.Column):
                # Direct column reference
                col_name = str(expr.name)
                columns.append({
                    "name": col_name,
                    "upstream": [],
                    "type": "SOURCE"
                })
            elif isinstance(expr, exp.Alias):
                # Check if the alias contains column references
                referenced_cols = self._extract_column_references(expr.this, source_table)
                for col_name in referenced_cols:
                    if not any(c["name"] == col_name for c in columns):
                        columns.append({
                            "name": col_name,
                            "upstream": [],
                            "type": "SOURCE"
                        })
        
        # Get columns from GROUP BY
        group_by = subquery.find(exp.Group)
        if group_by:
            for group_expr in group_by.expressions:
                if isinstance(group_expr, exp.Column):
                    col_name = str(group_expr.name)
                    if not any(c["name"] == col_name for c in columns):
                        columns.append({
                            "name": col_name,
                            "upstream": [],
                            "type": "SOURCE"
                        })
        
        return columns
    
    def _extract_column_references(self, expression, source_table: str) -> List[str]:
        """Extract column references from an expression."""
        columns = []
        
        # Find all column references in the expression
        for col in expression.find_all(exp.Column):
            col_name = str(col.name)
            columns.append(col_name)
        
        return columns
    
    def _extract_derived_table_columns(self, subquery) -> List[Dict]:
        """Extract columns produced by the derived table."""
        columns = []
        
        for expr in subquery.expressions:
            if isinstance(expr, exp.Column):
                # Direct column
                col_name = str(expr.name)
                columns.append({
                    "name": col_name,
                    "upstream": [f"SOURCE.{col_name}"],  # Will be updated with actual source table
                    "type": "DIRECT"
                })
            elif isinstance(expr, exp.Alias):
                # Aliased expression (could be aggregate)
                alias_name = str(expr.alias)
                source_expr = str(expr.this)
                
                # Check if it's an aggregate function
                if self._is_aggregate_expression(expr.this):
                    func_type = self._extract_function_type(source_expr)
                    referenced_cols = self._extract_column_references(expr.this, "")
                    
                    columns.append({
                        "name": alias_name,
                        "upstream": [f"SOURCE.{col}" for col in referenced_cols],
                        "type": "DIRECT",
                        "transformation": {
                            "source_expression": source_expr,
                            "transformation_type": "AGGREGATE",
                            "function_type": func_type
                        }
                    })
                else:
                    columns.append({
                        "name": alias_name,
                        "upstream": [f"SOURCE.{alias_name}"],
                        "type": "DIRECT"
                    })
        
        return columns
    
    def _extract_query_result_columns(self, outer_query, alias: str, derived_columns: List[Dict]) -> List[Dict]:
        """Extract columns in the final query result."""
        columns = []
        
        for expr in outer_query.expressions:
            if isinstance(expr, exp.Column):
                col_name = str(expr.name)
                table_ref = str(expr.table) if expr.table else None
                
                if table_ref == alias:
                    # Reference to derived table column
                    full_name = f"{alias}.{col_name}"
                    
                    # Find corresponding derived table column
                    derived_col = next((c for c in derived_columns if c["name"] == col_name), None)
                    if derived_col:
                        columns.append({
                            "name": full_name,
                            "upstream": [f"derived_table_{alias}.{col_name}"],
                            "type": "DIRECT"
                        })
        
        return columns
    
    def _extract_source_to_derived_transformations(self, subquery, source_table: str, derived_table_entity: str) -> List[Dict]:
        """Extract transformations from source table to derived table."""
        transformations = []
        
        # Base transformation
        trans = {
            "type": "derived_table_transformation",
            "source_table": clean_table_name_quotes(source_table),
            "target_table": clean_table_name_quotes(derived_table_entity)
        }
        
        # Add GROUP BY if present
        group_by = subquery.find(exp.Group)
        if group_by:
            group_cols = [str(expr) for expr in group_by.expressions]
            trans["group_by_columns"] = group_cols
        
        # Note: Aggregate column details are handled at the column level in metadata
        # Table transformations focus on structural changes (GROUP BY, JOIN, etc.)
        
        transformations.append(trans)
        return transformations
    
    def _extract_derived_to_result_transformations(self, outer_query, derived_table_entity: str, alias: str) -> List[Dict]:
        """Extract transformations from derived table to query result."""
        transformations = []
        
        # Base transformation
        trans = {
            "type": "table_transformation",
            "source_table": clean_table_name_quotes(derived_table_entity),
            "target_table": "QUERY_RESULT"
        }
        
        # Add WHERE conditions if they reference derived table columns
        where_clause = outer_query.find(exp.Where)
        if where_clause:
            filter_conditions = self._extract_filter_conditions(where_clause, alias)
            if filter_conditions:
                trans["filter_conditions"] = filter_conditions
        
        transformations.append(trans)
        return transformations
    
    def _extract_filter_conditions(self, where_clause, alias: str) -> List[Dict]:
        """Extract filter conditions from WHERE clause that reference this derived table alias."""
        # Use the existing transformation parser's comprehensive filter extraction logic
        from ..parsers.transformation_parser import TransformationParser
        
        # Create a temporary parser instance to leverage existing logic
        temp_parser = TransformationParser()
        
        # Extract all filter conditions using the existing comprehensive logic
        all_conditions = temp_parser._extract_filter_conditions(where_clause.this)
        
        # Filter to only include conditions that reference this derived table alias
        relevant_conditions = []
        for condition in all_conditions:
            if condition.get("column", "").startswith(f"{alias}."):
                relevant_conditions.append(condition)
        
        return relevant_conditions
    
    def _build_three_layer_structure(self, source_table: str, derived_table_entity: str,
                                   source_columns: List[Dict], derived_columns: List[Dict], 
                                   result_columns: List[Dict], source_to_derived_trans: List[Dict],
                                   derived_to_result_trans: List[Dict]) -> Dict[str, Any]:
        """Build the complete 3-layer lineage structure."""
        
        # Update upstream references in derived columns
        for derived_col in derived_columns:
            derived_col["upstream"] = [ref.replace("SOURCE", source_table) for ref in derived_col["upstream"]]
        
        # Build the nested structure: source → derived → result
        return {
            clean_table_name_quotes(source_table): {
                "entity": clean_table_name_quotes(source_table),
                "entity_type": "table",
                "depth": 0,
                "dependencies": [{
                    "entity": derived_table_entity,
                    "entity_type": "derived_table",
                    "depth": 1,
                    "metadata": {
                        "table_columns": derived_columns
                    },
                    "transformations": source_to_derived_trans,
                    "dependencies": [{
                        "entity": "QUERY_RESULT",
                        "entity_type": "table", 
                        "depth": 2,
                        "metadata": {
                            "table_columns": result_columns
                        },
                        "transformations": derived_to_result_trans
                    }]
                }],
                "metadata": {
                    "table_type": "TABLE",
                    "table_columns": source_columns
                }
            }
        }
    
    def _is_aggregate_expression(self, expression) -> bool:
        """Check if an expression contains aggregate functions."""
        aggregate_functions = ["SUM", "COUNT", "AVG", "MIN", "MAX", "GROUP_CONCAT"]
        expr_str = str(expression).upper()
        return any(func in expr_str for func in aggregate_functions)
    
    def _extract_function_type(self, expression: str) -> str:
        """Extract the function type from an aggregate expression."""
        expr_upper = expression.upper()
        for func in ["SUM", "COUNT", "AVG", "MIN", "MAX", "GROUP_CONCAT"]:
            if func in expr_upper:
                return func
        return "UNKNOWN"