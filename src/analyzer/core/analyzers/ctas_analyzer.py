"""CTAS (CREATE TABLE AS SELECT) analyzer."""

from typing import Dict, Any, List, Set
from .base_analyzer import BaseAnalyzer

# Import new utility modules
from ...utils.regex_patterns import is_ctas_query
from ...utils.column_extraction_utils import extract_all_referenced_columns
from ...utils.metadata_utils import create_table_metadata
from ...utils.sql_parsing_utils import clean_table_name_quotes
from ..transformation_engine import TransformationEngine
from ...utils.logging_config import get_logger


class CTASAnalyzer(BaseAnalyzer):
    """Analyzer for CTAS statements."""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = None, table_registry = None):
        """Initialize CTAS analyzer with transformation engine."""
        super().__init__(dialect, compatibility_mode, table_registry)
        self.transformation_engine = TransformationEngine(dialect)
        self.logger = get_logger('analyzers.ctas')
    
    def analyze_ctas(self, sql: str) -> Dict[str, Any]:
        """Analyze CREATE TABLE AS SELECT statement."""
        self.logger.info(f"Analyzing CTAS statement (length: {len(sql)})")
        self.logger.debug(f"CTAS SQL: {sql[:200]}..." if len(sql) > 200 else f"CTAS SQL: {sql}")
        
        try:
            self.logger.debug("Parsing CTAS structure")
            ctas_data = self.ctas_parser.parse(sql)
            self.logger.debug("Building CTAS lineage")
            ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
            self.logger.debug("Parsing CTAS transformations")
            transformation_data = self.transformation_parser.parse(sql)
            self.logger.info("CTAS parsing completed successfully")
            
            result = {
                'ctas_structure': ctas_data,
                'ctas_lineage': ctas_lineage,
                'transformations': transformation_data,
                'target_table': ctas_data.get('target_table', {}),
                'source_analysis': ctas_lineage.get('source_analysis', {}),
                'ctas_transformations': ctas_lineage.get('transformations', [])
            }
            
            target_table = ctas_data.get('target_table', {}).get('name', 'unknown')
            source_count = len(ctas_lineage.get('source_analysis', {}).get('source_tables', []))
            self.logger.info(f"CTAS analysis completed - target: {target_table}, {source_count} source tables")
            return result
            
        except Exception as e:
            self.logger.error(f"CTAS analysis failed: {str(e)}", exc_info=True)
            raise
    
    def parse_ctas(self, sql: str) -> Dict[str, Any]:
        """Parse CTAS using modular parser.""" 
        return self.ctas_parser.parse(sql)
    
    def extract_ctas_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """Extract column transformations from CTAS queries."""
        return self.transformation_engine.extract_ctas_column_transformations(sql, source_table, target_table)
    
    def add_ctas_result_columns(self, entity_data: Dict[str, Any], entity_name: str, sql: str = None) -> None:
        """Add result columns for CTAS target tables by extracting from SELECT clause."""
        if not sql or not (sql.strip().upper().startswith('CREATE TABLE') or sql.strip().upper().startswith('CREATE VIEW')):
            return
        
        # Check if this is a target table created by CTAS
        transformations = entity_data.get('transformations', [])
        for trans in transformations:
            if trans.get('target_table') == entity_name and trans.get('source_table'):
                # This is a CTAS target table, extract SELECT clause columns
                target_columns = set()
                
                # Extract columns from GROUP BY (these will be in result)
                group_by_columns = trans.get('group_by_columns', [])
                for col in group_by_columns:
                    target_columns.add(col)
                
                # Extract actual result columns from CTAS SELECT clause parsing
                try:
                    # Get CTAS lineage to get actual output columns
                    ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
                    column_lineage = ctas_lineage.get('column_lineage', {})
                    output_columns = column_lineage.get('output_columns', [])
                    
                    # Add all result columns from the CTAS SELECT clause
                    for output_col in output_columns:
                        col_name = output_col.get('alias') or output_col.get('name')
                        if col_name:
                            target_columns.add(col_name)
                except Exception:
                    # Fallback: just use GROUP BY columns if CTAS parsing fails
                    pass
                
                # Get column transformations for this entity
                try:
                    column_transformations = self.extract_ctas_column_transformations(sql, trans.get('source_table'), entity_name)
                    
                    # Create map of column_name -> transformation
                    column_transformations_map = {}
                    for col_trans in column_transformations:
                        col_name = col_trans.get('column_name')
                        if col_name:
                            column_transformations_map[col_name] = col_trans
                    
                    # Add result columns to target table metadata with transformations
                    if target_columns:
                        metadata = entity_data.get('metadata', {})
                        table_columns = metadata.get('table_columns', [])
                        existing_columns = {col['name'] for col in table_columns}
                        
                        for col_name in target_columns:
                            if col_name not in existing_columns:
                                column_info = {
                                    "name": col_name,
                                    "upstream": [],
                                    "type": "RESULT"
                                }
                                
                                # Add transformation if available
                                if col_name in column_transformations_map:
                                    col_trans = column_transformations_map[col_name]
                                    column_info["transformation"] = {
                                        "source_expression": col_trans.get('source_expression'),
                                        "transformation_type": col_trans.get('transformation_type'),
                                        "function_type": col_trans.get('function_type')
                                    }
                                
                                table_columns.append(column_info)
                        
                        # Update the metadata
                        if 'metadata' not in entity_data:
                            entity_data['metadata'] = {}
                        entity_data['metadata']['table_columns'] = table_columns
                
                except Exception as e:
                    # If transformation extraction fails, continue without transformations
                    self.logger.warning(f"CTAS transformation extraction failed: {str(e)} - continuing analysis without transformations", exc_info=True)
                
                break  # Found the CTAS transformation, no need to continue
    
    def handle_ctas_lineage_chain_logic(self, chains: Dict, sql: str, chain_type: str, table_lineage_data: Dict) -> Dict:
        """Handle CTAS-specific logic in lineage chain building."""
        if not sql or not (sql.strip().upper().startswith('CREATE TABLE') or sql.strip().upper().startswith('CREATE VIEW')):
            return chains
        
        if chain_type == "downstream":
            # For downstream analysis with CTAS queries, ensure target tables appear as dependencies of source tables
            # rather than as separate top-level entities
            upstream_lineage_data = {}  # We'd need to get this from the result
            downstream_lineage_data = table_lineage_data
            
            # Identify CTAS source and target tables
            ctas_source_tables = set()
            ctas_target_tables = set()
            
            # If we have upstream data available, use it
            for target_table, source_tables in upstream_lineage_data.items():
                if target_table != 'QUERY_RESULT':
                    ctas_target_tables.add(target_table)
                    ctas_source_tables.update(source_tables)
            
            # Remove target tables from chains if they were added as top-level
            for target_table in ctas_target_tables:
                if target_table in chains:
                    del chains[target_table]
            
            # Ensure target tables appear as dependencies in source table chains
            for source_table in ctas_source_tables:
                if source_table in chains:
                    # Add target tables as dependencies of this source table
                    source_chain = chains[source_table]
                    source_dependencies = source_chain.get('dependencies', [])
                    
                    # Get target tables that depend on this source
                    for target_table in downstream_lineage_data.get(source_table, []):
                        if target_table != 'QUERY_RESULT' and target_table in ctas_target_tables:
                            # Check if this target is not already in dependencies
                            existing_dep_entities = {dep.get('entity') for dep in source_dependencies}
                            if target_table not in existing_dep_entities:
                                # Build the target table as a dependency
                                target_depth = source_chain.get('depth', 0) + 1
                                target_chain = self._build_ctas_target_chain(target_table, target_depth + 1, source_table)
                                source_dependencies.append(target_chain)
                    
                    source_chain['dependencies'] = source_dependencies
        
        return chains
    
    def _build_ctas_target_chain(self, target_table: str, depth: int, source_table: str) -> Dict[str, Any]:
        """Build a chain for CTAS target table."""
        return {
            "entity": clean_table_name_quotes(target_table),
            "entity_type": "table",
            "depth": depth,
            "dependencies": [],
            "transformations": [{
                "type": "table_transformation",
                "source_table": clean_table_name_quotes(source_table),
                "target_table": clean_table_name_quotes(target_table),
                "filter_conditions": [],
                "group_by_columns": [],
                "joins": []
            }],
            "metadata": {
                "table_columns": [],
                "is_ctas_target": True
            }
        }
    
    def should_skip_query_result_for_ctas(self, sql: str, dependent_table: str) -> bool:
        """Check if QUERY_RESULT should be skipped for CTAS queries."""
        # For CTAS queries, don't add QUERY_RESULT since the target table itself is the final result
        if (sql and (sql.strip().upper().startswith('CREATE TABLE') or sql.strip().upper().startswith('CREATE VIEW')) and 
            dependent_table == 'QUERY_RESULT'):
            return True
        return False
    
    def is_ctas_transformation_relevant_to_table(self, col_trans: dict, table_name: str, sql: str) -> bool:
        """Check if a column transformation is relevant to a specific table for CTAS queries."""
        if not col_trans or not table_name:
            return False
        
        source_expression = col_trans.get('source_expression', '')
        if not source_expression:
            return False
        
        # For COUNT(*) and similar general aggregations in CTAS queries
        if source_expression.upper().strip() == 'COUNT(*)':
            # Special handling for CTAS queries - COUNT(*) is always relevant to the source table
            if sql and (sql.strip().upper().startswith('CREATE TABLE') or sql.strip().upper().startswith('CREATE VIEW')):
                return True
        
        return False
    
    def determine_ctas_context(self, transformation: Dict[str, Any]) -> Dict[str, Any]:
        """Determine context for CTAS transformations."""
        # Single-table context includes both QUERY_RESULT and CTAS scenarios
        is_single_table = (
            transformation.get('target_table') == "QUERY_RESULT" or  # Regular SELECT
            (transformation.get('source_table') != transformation.get('target_table') and  # CTAS/CTE
             transformation.get('target_table') != "QUERY_RESULT")
        )
        
        return {
            'is_single_table_context': is_single_table,
            'tables_in_context': [transformation.get('source_table')] if transformation.get('source_table') else []
        }
    
    def extract_ctas_target_and_source_tables(self, upstream_lineage_data: Dict) -> tuple:
        """Extract CTAS source and target tables from upstream lineage data."""
        ctas_source_tables = set()
        ctas_target_tables = set()
        
        for target_table, source_tables in upstream_lineage_data.items():
            if target_table != 'QUERY_RESULT':
                ctas_target_tables.add(target_table)
                ctas_source_tables.update(source_tables)
        
        return ctas_source_tables, ctas_target_tables
    
    def is_ctas_query(self, sql: str) -> bool:
        """Check if the SQL is a CTAS query."""
        return is_ctas_query(sql)
    
    def get_ctas_lineage_metadata(self, sql: str) -> Dict[str, Any]:
        """Get comprehensive CTAS lineage metadata."""
        if not self.is_ctas_query(sql):
            return {}
        
        try:
            ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
            return {
                'is_ctas': True,
                'target_table': ctas_lineage.get('target_table', {}),
                'source_analysis': ctas_lineage.get('source_analysis', {}),
                'column_lineage': ctas_lineage.get('column_lineage', {}),
                'transformations': ctas_lineage.get('transformations', [])
            }
        except Exception:
            return {'is_ctas': True, 'error': 'Failed to parse CTAS lineage'}
    
    def build_ctas_comprehensive_chain(self, entity_name: str, entity_type: str, current_depth: int, 
                                     visited_in_path: set, parent_entity: str, sql: str, 
                                     table_lineage_data: Dict) -> Dict[str, Any]:
        """Build comprehensive chain with CTAS-specific handling."""
        # Standard chain building
        chain = {
            "entity": clean_table_name_quotes(entity_name),
            "entity_type": entity_type,
            "depth": current_depth - 1,
            "dependencies": [],
            "metadata": {"table_columns": []}
        }
        
        # Add CTAS-specific metadata if this is a CTAS query
        if self.is_ctas_query(sql):
            ctas_metadata = self.get_ctas_lineage_metadata(sql)
            chain['metadata'].update(ctas_metadata)
            
            # Handle CTAS result columns
            if entity_type == "table" and current_depth > 1:  # Target table
                self.add_ctas_result_columns(chain, entity_name, sql)
        
        # Process table dependencies with CTAS logic
        if entity_type == "table" and entity_name in table_lineage_data:
            for dependent_table in table_lineage_data[entity_name]:
                # Skip QUERY_RESULT for CTAS queries
                if self.should_skip_query_result_for_ctas(sql, dependent_table):
                    continue
                
                # Build dependency chain
                if (current_depth <= 10 and dependent_table not in visited_in_path):  # Prevent infinite loops
                    visited_in_path_new = visited_in_path | {entity_name}
                    dep_chain = self.build_ctas_comprehensive_chain(
                        dependent_table, "table", current_depth + 1, 
                        visited_in_path_new, entity_name, sql, table_lineage_data
                    )
                    chain["dependencies"].append(dep_chain)
        
        return chain


def is_ctas_target_table(sql: str, table_name: str) -> bool:
    """Check if this table is a CTAS target table."""
    if not is_ctas_query(sql):
        return False
    
    if not ('CREATE TABLE' in sql.upper() or 'CREATE VIEW' in sql.upper()):
        return False
    
    # Normalize by removing quotes from both SQL and table name for comparison
    sql_normalized = sql.replace('"', '').replace("'", "").lower()
    table_normalized = table_name.replace('"', '').replace("'", "").lower()
    
    return table_normalized in sql_normalized


def build_ctas_target_columns(sql: str, select_columns: List[Dict]) -> List[Dict]:
    """Build target table columns for CTAS queries with aggregate handling."""
    table_columns = []
    
    for sel_col in select_columns:
        raw_expression = sel_col.get('raw_expression', '')
        column_name = sel_col.get('column_name', raw_expression)
        
        # Extract clean name (prefer alias over raw column name)
        from ...utils.sql_parsing_utils import extract_clean_column_name, normalize_entity_name
        clean_name = extract_clean_column_name(raw_expression, column_name)
        
        # Normalize column name to remove quotes
        normalized_name = normalize_entity_name(clean_name)
        
        # Check if this is an aggregate function
        from ...utils.aggregate_utils import is_aggregate_function
        from ...utils.sql_parsing_utils import extract_function_type
        if is_aggregate_function(raw_expression):
            # Aggregate column with transformation details
            from ...utils.aggregate_utils import extract_alias_from_expression
            func_type = extract_function_type(raw_expression)
            alias = extract_alias_from_expression(raw_expression)
            
            # Normalize source expression to remove quotes
            normalized_source_expr = raw_expression.replace(f" as {alias}", "").replace(f" AS {alias}", "") if alias else raw_expression
            normalized_source_expr = normalize_entity_name(normalized_source_expr)
            
            column_info = {
                "name": alias or normalized_name,
                "upstream": [],
                "type": "RESULT",
                "transformation": {
                    "source_expression": normalized_source_expr,
                    "transformation_type": "AGGREGATE",
                    "function_type": func_type
                }
            }
        else:
            # Regular column (pass-through)  
            column_info = {
                "name": normalized_name,
                "upstream": [],
                "type": "SOURCE"
            }
        
        table_columns.append(column_info)
    
    return table_columns


def add_group_by_to_ctas_transformations(dep: Dict, sql: str) -> None:
    """Add GROUP BY information to CTAS transformations."""
    import re
    
    # Extract GROUP BY columns from SQL
    group_by_match = re.search(r'GROUP\s+BY\s+([^)]+?)(?:\s+(?:HAVING|ORDER|LIMIT)|$)', sql, re.IGNORECASE)
    if group_by_match:
        group_by_clause = group_by_match.group(1).strip()
        group_by_columns = [col.strip() for col in group_by_clause.split(',')]
        
        # Normalize group by columns to remove quotes
        from ...utils.sql_parsing_utils import normalize_entity_name
        normalized_group_by_columns = [normalize_entity_name(col) for col in group_by_columns]
        
        # Add to transformations
        for transformation in dep.get('transformations', []):
            transformation['group_by_columns'] = normalized_group_by_columns