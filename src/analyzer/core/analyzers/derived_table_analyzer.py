"""Derived Table Analyzer for FROM clause subqueries.

This module provides specialized analysis for derived tables (subqueries in FROM clauses)
to create comprehensive 3-layer lineage: source_table → derived_table → query_result
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import sqlglot
from sqlglot import exp
from ...utils.logging_config import get_logger
from ...utils.sql_parsing_utils import clean_table_name_quotes, normalize_entity_name


class DerivedTableAnalyzer:
    """Specialized analyzer for FROM clause derived tables (subqueries)."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize the derived table analyzer."""
        self.dialect = dialect
        self.logger = get_logger('analyzers.derived_table')
    
    def analyze_derived_tables(self, sql: str) -> Dict[str, Any]:
        """
        Analyze derived tables in FROM clauses and create 3-layer lineage structure.
        Enhanced with JOIN detection debugging.
        
        Args:
            sql: SQL query string to analyze
            
        Returns:
            Dict containing derived table analysis with 3-layer flow
        """
        self.logger.info(f"Analyzing derived tables in SQL (length: {len(sql)})")
        self.logger.debug(f"Derived table SQL: {sql[:200]}..." if len(sql) > 200 else f"Derived table SQL: {sql}")
        
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # ENHANCED DEBUG: Check if this is a JOIN query
            is_create_view = isinstance(parsed, exp.Create) and parsed.kind == "VIEW"
            if is_create_view:
                self.logger.info(f"DERIVED TABLE ANALYZER - Processing CREATE VIEW statement")
                # Check for JOINs in the CREATE VIEW
                join_count = 0
                if hasattr(parsed, 'expression') and parsed.expression:
                    for node in parsed.expression.walk():
                        if isinstance(node, (exp.Join,)):
                            join_count += 1
                            self.logger.info(f"DERIVED TABLE ANALYZER - Found JOIN in CREATE VIEW: {type(node).__name__}")
                self.logger.info(f"DERIVED TABLE ANALYZER - Total JOINs detected in CREATE VIEW: {join_count}")
            else:
                self.logger.info(f"DERIVED TABLE ANALYZER - Processing regular SELECT statement")
            
            # Detect if this is a CREATE VIEW or CREATE TABLE statement
            target_table_name = self._extract_target_table_name(parsed, sql)
            
            # Find derived tables in FROM clauses
            derived_tables = self._extract_from_subqueries(parsed)
            
            if not derived_tables:
                return {}
            
            # Process each derived table
            analysis_result = {}
            for derived_table in derived_tables:
                table_analysis = self._process_derived_table(derived_table, parsed, target_table_name)
                if table_analysis:
                    analysis_result.update(table_analysis)
            
            # Post-process CREATE VIEW/TABLE entities if any were detected
            if target_table_name:
                self._create_table_entity(analysis_result, target_table_name, parsed)
            
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
    
    def _extract_target_table_name(self, parsed_sql, sql: str) -> Optional[str]:
        """Extract the target table name from CREATE VIEW or CREATE TABLE statements."""
        try:
            # Check if this is a CREATE VIEW or CREATE TABLE statement
            if isinstance(parsed_sql, exp.Create) and parsed_sql.kind in ["VIEW", "TABLE"]:
                if parsed_sql.this and isinstance(parsed_sql.this, exp.Table):
                    table = parsed_sql.this
                    
                    # Extract fully qualified table/view name using same logic as source tables
                    if table.catalog and table.db:
                        # Handle three-part naming (e.g., "catalog"."schema"."table")
                        from ...utils.sql_parsing_utils import normalize_quoted_identifier
                        catalog = normalize_quoted_identifier(str(table.catalog))
                        schema = normalize_quoted_identifier(str(table.db))  
                        table_name_part = normalize_quoted_identifier(str(table.name))
                        return f"{catalog}.{schema}.{table_name_part}"
                    elif table.db:
                        # Handle database.table naming (e.g., "promethium.table_name")
                        from ...utils.sql_parsing_utils import normalize_quoted_identifier
                        db = normalize_quoted_identifier(str(table.db))
                        table_name_part = normalize_quoted_identifier(str(table.name))
                        return f"{db}.{table_name_part}"
                    else:
                        # Handle simple table naming (e.g., "table_name")
                        from ...utils.sql_parsing_utils import normalize_quoted_identifier
                        return normalize_quoted_identifier(str(table.name))
        except Exception as e:
            self.logger.debug(f"Could not extract view name: {e}")
        
        return None
    
    def _process_derived_table(self, derived_table: Dict, outer_query, target_table_name: Optional[str] = None) -> Dict[str, Any]:
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
        layer_analysis = self._analyze_three_layers(source_table, derived_table_entity, subquery, outer_query, alias, target_table_name)
        
        return layer_analysis
    
    def _extract_source_table(self, subquery) -> Optional[str]:
        """Extract the fully qualified source table name from the subquery."""
        from_clause = subquery.find(exp.From)
        if from_clause and isinstance(from_clause.this, exp.Table):
            table = from_clause.this
            
            # Extract fully qualified table name (same logic as in sql_parsing_utils)
            if table.catalog and table.db:
                # Handle three-part naming (e.g., "catalog"."schema"."table")
                from ...utils.sql_parsing_utils import normalize_quoted_identifier
                catalog = normalize_quoted_identifier(str(table.catalog))
                schema = normalize_quoted_identifier(str(table.db))  
                table_name_part = normalize_quoted_identifier(str(table.name))
                return f'"{catalog}"."{schema}"."{table_name_part}"'
            elif table.db:
                # Handle database.table naming (e.g., "ecommerce.users")
                from ...utils.sql_parsing_utils import normalize_quoted_identifier
                db = normalize_quoted_identifier(str(table.db))
                table_name_part = normalize_quoted_identifier(str(table.name))
                return f"{db}.{table_name_part}"
            else:
                # Handle simple table naming (e.g., "users")
                from ...utils.sql_parsing_utils import normalize_quoted_identifier
                return normalize_quoted_identifier(str(table.name))
        return None
    
    def _analyze_three_layers(self, source_table: str, derived_table_entity: str, 
                            subquery, outer_query, alias: str, target_table_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the complete 3-layer flow: source → derived → result."""
        
        # Layer 1: Source table columns - empty, populated by main chain builder
        source_columns = []
        
        # Layer 2: Derived table columns (subquery SELECT expressions)
        derived_columns = self._extract_derived_table_columns(subquery)
        
        # Layer 3: Query result columns (outer SELECT expressions)
        result_columns = self._extract_query_result_columns(outer_query, alias, derived_columns, target_table_name)
        
        # Layer transformations
        source_to_derived_trans = self._extract_source_to_derived_transformations(subquery, source_table, derived_table_entity)
        derived_to_result_trans = self._extract_derived_to_result_transformations(outer_query, derived_table_entity, alias, target_table_name)
        
        # Build the 3-layer structure
        return self._build_three_layer_structure(
            source_table, derived_table_entity, source_columns, derived_columns, result_columns,
            source_to_derived_trans, derived_to_result_trans, target_table_name
        )
    
    
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
        
        # Check if this is a SELECT * query
        has_star = any(isinstance(expr, exp.Star) for expr in subquery.expressions)
        
        if has_star:
            # For SELECT *, we need to indicate that all source columns pass through
            # We'll create a special marker that the main chain builder can populate
            # with actual column names from the source table
            return [{"name": "*", "upstream": [], "type": "PASSTHROUGH"}]
        
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
    
    def _extract_create_table_columns(self, outer_query, target_table_name: str) -> List[Dict]:
        """Extract all columns for a CREATE VIEW or CREATE TABLE statement."""
        columns = []
        
        # Extract all columns from the CREATE VIEW/TABLE SELECT statement
        for expr in outer_query.expressions:
            if isinstance(expr, exp.Column):
                col_name = str(expr.name)
                table_ref = str(expr.table) if expr.table else None
                
                if table_ref:
                    # Column with table reference (e.g., t1.country → country)
                    column_name = col_name  # Use just the column name for the target table
                    
                    # Map to the appropriate derived table  
                    derived_table_ref = f"derived_table_{table_ref}"
                    columns.append({
                        "name": column_name,
                        "upstream": [f"{derived_table_ref}.{col_name}"],
                        "type": "DIRECT"
                    })
                    self.logger.debug(f"CREATE TABLE/VIEW: Added column {column_name} from {derived_table_ref}.{col_name}")
                else:
                    # Column without table reference (shouldn't happen for CREATE with subqueries)
                    columns.append({
                        "name": col_name,
                        "upstream": [f"derived_table_unknown.{col_name}"],
                        "type": "DIRECT"
                    })
        
        self.logger.debug(f"_extract_create_table_columns: Extracted {len(columns)} columns for table {target_table_name}")
        return columns

    def _extract_query_result_columns(self, outer_query, alias: str, derived_columns: List[Dict], target_table_name: Optional[str] = None) -> List[Dict]:
        """Extract columns in the final query result."""
        columns = []
        
        if target_table_name:
            # For CREATE VIEW/TABLE, we don't extract columns here - they are handled separately
            # This avoids duplication and allows proper merging
            return []
        else:
            # For regular SELECT queries, only extract columns that reference this specific alias
            for expr in outer_query.expressions:
                if isinstance(expr, exp.Column):
                    col_name = str(expr.name)
                    table_ref = str(expr.table) if expr.table else None
                    
                    if table_ref == alias:
                        # This column references our derived table
                        full_name = f"{alias}.{col_name}"
                        
                        # Create column entry
                        columns.append({
                            "name": full_name,
                            "upstream": [f"derived_table_{alias}.{col_name}"],
                            "type": "DIRECT"
                        })
        
        self.logger.debug(f"_extract_query_result_columns: Extracted {len(columns)} columns for target_table={target_table_name}, alias={alias}")
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
    
    def _extract_derived_to_result_transformations(self, outer_query, derived_table_entity: str, alias: str, target_table_name: Optional[str] = None) -> List[Dict]:
        """Extract comprehensive transformations from derived table to query result."""
        transformations = []
        
        # Base transformation - use actual table name if available, otherwise QUERY_RESULT
        target_table = target_table_name if target_table_name else "QUERY_RESULT"
        trans = {
            "type": "table_transformation",
            "source_table": clean_table_name_quotes(derived_table_entity),
            "target_table": target_table
        }
        
        # Use comprehensive transformation parsing from transformation parser
        from ..parsers.transformation_parser import TransformationParser
        parser = TransformationParser()
        
        # Extract JOIN information
        joins = parser.parse_transformation_joins(outer_query)
        if joins:
            trans["joins"] = joins
        
        # Extract comprehensive filter conditions (WHERE clause)
        filter_result = parser.parse_transformation_filters(outer_query)
        filter_conditions = filter_result.get('filters', [])
        if filter_conditions:
            trans["filter_conditions"] = filter_conditions
        
        # Extract GROUP BY and HAVING information
        aggregation_result = parser.parse_transformation_aggregations(outer_query)
        group_by_columns = aggregation_result.get('group_by_columns', [])
        if group_by_columns:
            trans["group_by_columns"] = group_by_columns
        having_conditions = aggregation_result.get('having_conditions', [])
        if having_conditions:
            trans["having_conditions"] = having_conditions
        
        # Extract ORDER BY information
        sorting_result = parser.parse_transformation_sorting(outer_query)
        order_by_columns = sorting_result.get('order_by_columns', [])
        if order_by_columns:
            trans["order_by_columns"] = order_by_columns
        
        # Extract LIMIT information
        limiting_result = parser.parse_transformation_limiting(outer_query)
        if limiting_result.get('limit') is not None:
            trans["limiting"] = limiting_result
        
        transformations.append(trans)
        return transformations
    
    
    def _build_three_layer_structure(self, source_table: str, derived_table_entity: str,
                                   source_columns: List[Dict], derived_columns: List[Dict], 
                                   result_columns: List[Dict], source_to_derived_trans: List[Dict],
                                   derived_to_result_trans: List[Dict], target_table_name: Optional[str] = None) -> Dict[str, Any]:
        """Build the complete 3-layer lineage structure using the original fully qualified source table name."""
        
        # Handle PASSTHROUGH columns for SELECT * cases
        if len(derived_columns) == 1 and derived_columns[0].get("name") == "*":
            # This is a SELECT * - we need to defer column population to main chain builder
            # For now, set as empty and let the merge process handle it
            derived_columns = []
        else:
            # Update upstream references in derived columns and normalize quotes
            for derived_col in derived_columns:
                if "upstream" in derived_col:
                    # Replace SOURCE placeholder with actual source table and normalize quotes
                    updated_refs = []
                    for ref in derived_col["upstream"]:
                        updated_ref = ref.replace("SOURCE", source_table)
                        # Use existing utility function to normalize quotes
                        normalized_ref = normalize_entity_name(updated_ref)
                        updated_refs.append(normalized_ref)
                    derived_col["upstream"] = updated_refs
        
        # CRITICAL FIX: Use the original source_table name (with quotes) as the key
        # This ensures we don't create duplicates with different naming conventions
        # The main extractor creates entities like "hive"."promethium"."table_name"
        # We should use the same key format to integrate with existing entities
        source_table_key = source_table  # Keep original format with quotes
        
        # Build the nested structure: source → derived → result
        # Normalize entity name using existing utility function
        clean_entity_name = normalize_entity_name(source_table_key) if source_table_key else source_table_key
        
        return {
            clean_entity_name: {
                "entity": clean_entity_name,
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
                        "entity": target_table_name if target_table_name else "QUERY_RESULT",
                        "entity_type": self._get_target_entity_type(target_table_name) if target_table_name else "table", 
                        "depth": 2,
                        "metadata": {
                            "table_columns": result_columns  # Will be empty for CREATE VIEW initially, populated later
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
    
    def _get_target_entity_type(self, target_table_name: str) -> str:
        """Determine if the target is a VIEW or TABLE based on the CREATE statement."""
        # For now, we'll determine based on the parsed SQL in the create method
        # This is a placeholder - the actual determination happens in _create_table_entity
        return "view"  # Default to view for backward compatibility
    
    def _create_table_entity(self, chains: Dict, target_table_name: str, parsed_sql) -> None:
        """Populate CREATE VIEW/TABLE columns in the existing 3-layer chain structure."""
        try:
            # Determine if this is CREATE VIEW or CREATE TABLE
            create_type = "UNKNOWN"
            table_type = "TABLE"
            entity_type = "table"
            
            if isinstance(parsed_sql, exp.Create):
                if parsed_sql.kind == "VIEW":
                    create_type = "VIEW"
                    table_type = "VIEW"
                    entity_type = "view"
                elif parsed_sql.kind == "TABLE":
                    create_type = "TABLE"
                    table_type = "TABLE" 
                    entity_type = "table"
            
            # Find the SELECT statement within the CREATE statement
            outer_query = None
            if isinstance(parsed_sql, exp.Create) and parsed_sql.kind in ["VIEW", "TABLE"]:
                if hasattr(parsed_sql, 'expression') and parsed_sql.expression:
                    outer_query = parsed_sql.expression
                elif hasattr(parsed_sql, 'this') and isinstance(parsed_sql.this, exp.Select):
                    outer_query = parsed_sql.this
            
            if not outer_query:
                self.logger.debug(f"Could not find SELECT statement in CREATE {create_type} {target_table_name}")
                return
                
            # Extract all columns for the CREATE statement
            table_columns = self._extract_create_table_columns(outer_query, target_table_name)
            
            if not table_columns:
                self.logger.debug(f"No columns extracted for CREATE {create_type} {target_table_name}")
                return
            
            # Find and populate the target entities in the 3-layer structure
            target_entities_found = 0
            for entity_name, chain_data in chains.items():
                if chain_data.get('entity_type') == 'table':
                    dependencies = chain_data.get('dependencies', [])
                    for dep in dependencies:
                        if dep.get('entity_type') == 'derived_table':
                            nested_dependencies = dep.get('dependencies', [])
                            for nested_dep in nested_dependencies:
                                if nested_dep.get('entity') == target_table_name:
                                    # Found a target entity - populate its columns and fix entity type
                                    nested_dep['metadata']['table_columns'] = table_columns
                                    nested_dep['metadata']['table_type'] = table_type
                                    nested_dep['entity_type'] = entity_type
                                    target_entities_found += 1
                                    self.logger.debug(f"✅ Populated {len(table_columns)} columns for {create_type} entity in chain {entity_name}")
            
            self.logger.debug(f"✅ Populated CREATE {create_type} columns in {target_entities_found} chain entities for {target_table_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Error populating CREATE {create_type} columns: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())