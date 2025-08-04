"""Lineage chain builder analyzer."""

from typing import Dict, Any, List, Optional, Set
import json
import sqlglot
from sqlglot import expressions as exp
from .base_analyzer import BaseAnalyzer

# Import new utility modules
from ...utils.sql_parsing_utils import (
    build_alias_to_table_mapping, extract_function_type, is_column_from_table,
    clean_source_expression, extract_table_references_from_sql,
    get_union_columns_for_table, collect_union_selects_helper,
    infer_query_result_columns_simple, extract_clean_column_name,
    infer_query_result_columns_simple_fallback, extract_table_columns_from_sql,
    extract_join_columns_from_sql, extract_all_referenced_columns,
    extract_qualified_filter_columns
)
from ...utils.column_extraction_utils import (
    extract_aggregate_columns, choose_best_column, process_select_expression
)
from ...utils.aggregate_utils import (
    is_aggregate_function, query_has_aggregates, extract_alias_from_expression,
    is_aggregate_function_for_table, column_expression_belongs_to_table,
    extract_aggregate_result_columns, extract_aggregate_source_columns,
    is_main_aggregating_table, extract_upstream_from_aggregate
)
from ...utils.chain_utils import (
    calculate_max_depth, detect_transformations_in_chains,
    detect_metadata_in_chains, integrate_column_transformations
)
from ...utils.metadata_utils import (
    create_source_column_metadata, create_result_column_metadata,
    merge_metadata_entries, create_table_metadata
)
from ...utils.sqlglot_helpers import (
    parse_sql_safely, extract_table_references, get_select_expressions,
    get_where_conditions, validate_sql_syntax
)
from ...utils.regex_patterns import (
    extract_where_clause, extract_join_conditions, is_ctas_query,
    extract_filter_conditions, SQLPatterns
)
from .ctas_analyzer import is_ctas_target_table, build_ctas_target_columns, add_group_by_to_ctas_transformations
from ..transformation_engine import TransformationEngine
from ..chain_builder_engine import ChainBuilderEngine


class LineageChainBuilder(BaseAnalyzer):
    """Analyzer for building lineage chains."""
    
    def __init__(self, dialect: str = "trino", main_analyzer=None, table_registry = None):
        """Initialize lineage chain builder with optional reference to main analyzer."""
        # Get compatibility mode from main analyzer if available
        compatibility_mode = getattr(main_analyzer, 'compatibility_mode', None) if main_analyzer else None
        registry = table_registry or getattr(main_analyzer, 'table_registry', None)
        
        super().__init__(dialect, compatibility_mode, registry)
        self.main_analyzer = main_analyzer
        self.transformation_engine = TransformationEngine(dialect)
        self.chain_builder_engine = ChainBuilderEngine(dialect)
    
    def get_table_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the table lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the table lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        # Parse SQL and extract lineage using LineageExtractor
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            table_lineage = self.extractor.extract_table_lineage(parsed)
            column_lineage = self.extractor.extract_column_lineage(parsed)
            
            # Create a mock result object with the required structure
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = table_lineage.upstream
            result.table_lineage.downstream = table_lineage.downstream
            result.table_lineage.transformations = {}
            result.column_lineage = SimpleNamespace()
            result.column_lineage.upstream = column_lineage.upstream
            result.column_lineage.downstream = column_lineage.downstream
            result.column_lineage.transformations = {}
            result.dialect = self.dialect
            result.errors = []
            result.warnings = []
            
            # No external metadata - analyze tables purely from SQL context
            result.metadata = {}
        except Exception as e:
            # Create empty result on error
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = {}
            result.table_lineage.downstream = {}
            result.table_lineage.transformations = {}
            result.column_lineage = SimpleNamespace() 
            result.column_lineage.upstream = {}
            result.column_lineage.downstream = {}
            result.column_lineage.transformations = {}
            result.dialect = self.dialect
            result.errors = [f"Parsing failed: {str(e)}"]
            result.warnings = []
            result.metadata = {}
        
        # Get the appropriate lineage direction
        lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        
        # Build the chain starting from all tables at the current level
        chain = {}
        
        def build_chain(table_name: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth or if we have a circular dependency
            if current_depth > depth or table_name in visited_in_path:
                return {"table": table_name, "depth": current_depth - 1, "dependencies": []}
            
            # Add current table to the path to prevent cycles
            visited_in_path = visited_in_path | {table_name}
            
            dependencies = []
            if table_name in lineage_data:
                for dependent_table in lineage_data[table_name]:
                    dep_chain = build_chain(dependent_table, current_depth + 1, visited_in_path)
                    dependencies.append(dep_chain)
            
            return {
                "table": table_name,
                "depth": current_depth - 1,
                "dependencies": dependencies
            }
        
        # Start chain building from all root tables
        for table_name in lineage_data.keys():
            chain[table_name] = build_chain(table_name, 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": result.errors,
            "warnings": result.warnings
        }

    def get_table_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """
        Get the JSON representation of table lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the lineage chain
        """
        chain_data = self.get_table_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)

    def get_column_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the column lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the column lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        # Parse SQL and extract lineage using LineageExtractor
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            table_lineage = self.extractor.extract_table_lineage(parsed)
            column_lineage = self.extractor.extract_column_lineage(parsed)
            
            # Create a mock result object with the required structure
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = table_lineage.upstream
            result.table_lineage.downstream = table_lineage.downstream
            result.table_lineage.transformations = {}
            result.column_lineage = SimpleNamespace()
            result.column_lineage.upstream = column_lineage.upstream
            result.column_lineage.downstream = column_lineage.downstream
            result.column_lineage.transformations = {}
            result.dialect = self.dialect
            result.errors = []
            result.warnings = []
            
            # No external metadata - analyze tables purely from SQL context
            result.metadata = {}
        except Exception as e:
            # Create empty result on error
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = {}
            result.table_lineage.downstream = {}
            result.table_lineage.transformations = {}
            result.column_lineage = SimpleNamespace() 
            result.column_lineage.upstream = {}
            result.column_lineage.downstream = {}
            result.column_lineage.transformations = {}
            result.dialect = self.dialect
            result.errors = [f"Parsing failed: {str(e)}"]
            result.warnings = []
            result.metadata = {}
        
        # Get the appropriate lineage direction
        lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Build the chain starting from all columns at the current level
        chain = {}
        
        def build_column_chain(column_ref: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth or if we have a circular dependency
            if current_depth > depth or column_ref in visited_in_path:
                return {"column": column_ref, "depth": current_depth - 1, "dependencies": []}
            
            # Add current column to the path to prevent cycles
            visited_in_path = visited_in_path | {column_ref}
            
            dependencies = []
            if column_ref in lineage_data:
                for dependent_column in lineage_data[column_ref]:
                    dep_chain = build_column_chain(dependent_column, current_depth + 1, visited_in_path)
                    dependencies.append(dep_chain)
            
            return {
                "column": column_ref,
                "depth": current_depth - 1,
                "dependencies": dependencies
            }
        
        # Start chain building from all root columns
        for column_ref in lineage_data.keys():
            chain[column_ref] = build_column_chain(column_ref, 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": result.errors,
            "warnings": result.warnings
        }

    def get_column_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """
        Get the JSON representation of column lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the column lineage chain
        """
        chain_data = self.get_column_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)

    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive lineage chain combining table and column lineage with transformations.
        Supports CTAS, CTE, and other SQL block types with detailed metadata and transformations.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing comprehensive lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 0:
            raise ValueError("depth must be 0 or greater (0 means unlimited depth)")
        
        # Check if this is a CTE query and use CTE-specific processing
        if "WITH" in sql.upper():
            return self._build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        
        # Get full analysis result to include transformations and metadata
        try:
            result = self.main_analyzer.analyze(sql, **kwargs)
        except Exception as e:
            # Create empty result on error
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = {}
            result.table_lineage.downstream = {}
            result.table_lineage.transformations = {}
            result.column_lineage = SimpleNamespace() 
            result.column_lineage.upstream = {}
            result.column_lineage.downstream = {}
            result.column_lineage.transformations = {}
            result.dialect = self.dialect
            result.errors = [f"Parsing failed: {str(e)}"]
            result.warnings = []
            result.metadata = {}
        
        # Get table lineage data for the appropriate direction
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        column_transformations_data = result.column_lineage.transformations if hasattr(result.column_lineage, 'transformations') else {}
        
        # Build chains dictionary
        chains = {}
        
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, 
                                    visited_in_path: set, parent_entity: str = None) -> Dict[str, Any]:
            """Build comprehensive chain with metadata and transformations."""
            # Standard chain building
            chain = {
                "entity": entity_name,
                "entity_type": entity_type,
                "depth": current_depth - 1,
                "dependencies": [],
                "metadata": {"table_columns": []}
            }
            
            if entity_name in result.metadata and result.metadata[entity_name] is not None:
                table_meta = result.metadata[entity_name]
                metadata = {
                    "table_type": table_meta.table_type.value
                }
                
                # Only include non-null values to keep output clean
                if table_meta.schema:
                    metadata["schema"] = table_meta.schema
                if table_meta.description:
                    metadata["description"] = table_meta.description
            else:
                # No external metadata - use basic defaults
                metadata = {
                    "table_type": "TABLE",
                    "schema": "default", 
                    "description": "Table information"
                }
                
                # Update existing metadata with table metadata
                chain["metadata"].update(metadata)
            
            # Process table dependencies
            if entity_type == "table" and entity_name in table_lineage_data:
                for dependent_table in table_lineage_data[entity_name]:
                    # Skip if would cause circular dependency or exceed depth
                    if (depth > 0 and current_depth > depth) or dependent_table in visited_in_path:
                        continue
                    
                    # Build dependency chain
                    visited_in_path_new = visited_in_path | {entity_name}
                    dep_chain = build_comprehensive_chain(
                        dependent_table, "table", current_depth + 1, 
                        visited_in_path_new, entity_name
                    )
                    
                    # Add transformations to dependency - extracted from analyzer-bkup.py lines 1442-1447
                    if hasattr(result, 'table_lineage') and hasattr(result.table_lineage, 'transformations'):
                        transformations = []
                        for transformation_list in result.table_lineage.transformations.values():
                            for trans in transformation_list:
                                # Create transformation data structure matching original format
                                trans_data = {
                                    "type": "table_transformation",
                                    "source_table": trans.source_table,
                                    "target_table": trans.target_table
                                }
                                
                                # Add join information if present
                                if hasattr(trans, 'join_conditions') and trans.join_conditions:
                                    join_entry = {
                                        "join_type": trans.join_type.value if hasattr(trans, 'join_type') and trans.join_type else "INNER JOIN",
                                        "right_table": None,
                                        "conditions": [
                                            {
                                                "left_column": jc.left_column,
                                                "operator": jc.operator.value if hasattr(jc.operator, 'value') else str(jc.operator),
                                                "right_column": jc.right_column
                                            }
                                            for jc in trans.join_conditions
                                        ]
                                    }
                                    
                                    # Extract right table from first condition
                                    if trans.join_conditions:
                                        first_condition = trans.join_conditions[0]
                                        if hasattr(first_condition, 'right_column') and '.' in first_condition.right_column:
                                            right_table = first_condition.right_column.split('.')[0]
                                            join_entry["right_table"] = right_table
                                    
                                    trans_data["joins"] = [join_entry]
                                
                                # Determine context for column filtering - used for all transformation types
                                # Single-table context includes both QUERY_RESULT and CTAS scenarios
                                is_single_table = (
                                    trans.target_table == "QUERY_RESULT" or  # Regular SELECT
                                    (trans.source_table != trans.target_table and  # CTAS/CTE
                                     trans.target_table != "QUERY_RESULT")
                                )
                                
                                context_info = {
                                    'single_table': is_single_table,
                                    'tables_in_context': [trans.source_table] if is_single_table else [],
                                    'sql': sql
                                }
                                
                                # Filter conditions to only include those relevant to the current entity
                                if hasattr(trans, 'filter_conditions') and trans.filter_conditions:
                                    relevant_filters = []
                                    
                                    for fc in trans.filter_conditions:
                                        # Include filter conditions that reference columns from this entity (including subquery context)
                                        if is_column_from_table(fc.column, entity_name, sql):
                                            relevant_filters.append({
                                                "column": fc.column,
                                                "operator": fc.operator.value if hasattr(fc.operator, 'value') else str(fc.operator),
                                                "value": fc.value
                                            })
                                    if relevant_filters:
                                        trans_data["filter_conditions"] = relevant_filters
                                
                                # Group by columns - only include those from this entity
                                if hasattr(trans, 'group_by_columns') and trans.group_by_columns:
                                    relevant_group_by = []
                                    for col in trans.group_by_columns:
                                        if is_column_from_table(col, entity_name, sql):
                                            relevant_group_by.append(col)
                                    if relevant_group_by:
                                        trans_data["group_by_columns"] = relevant_group_by
                                
                                # Having conditions - only include those referencing columns from this entity
                                if hasattr(trans, 'having_conditions') and trans.having_conditions:
                                    relevant_having = []
                                    for hc in trans.having_conditions:
                                        # Having conditions often involve aggregations like COUNT(*) or AVG(u.salary)
                                        # Check if they reference this entity or if they are general aggregations for this table
                                        is_relevant = (is_column_from_table(hc.column, entity_name, sql) or 
                                                     is_aggregate_function_for_table(hc.column, entity_name, sql))
                                        if is_relevant:
                                            relevant_having.append({
                                                "column": hc.column,
                                                "operator": hc.operator.value if hasattr(hc.operator, 'value') else str(hc.operator),
                                                "value": hc.value
                                            })
                                    if relevant_having:
                                        trans_data["having_conditions"] = relevant_having
                                
                                # Order by columns - only include those from this entity
                                if hasattr(trans, 'order_by_columns') and trans.order_by_columns:
                                    relevant_order_by = []
                                    for col in trans.order_by_columns:
                                        # Extract just the column name part (before ASC/DESC)
                                        col_name = col.split()[0] if ' ' in col else col
                                        if is_column_from_table(col_name, entity_name, sql):
                                            relevant_order_by.append(col)
                                    if relevant_order_by:
                                        trans_data["order_by_columns"] = relevant_order_by
                                
                                # Add UNION information if present
                                if hasattr(trans, 'union_type') and trans.union_type:
                                    union_entry = {
                                        "union_type": trans.union_type,
                                        "union_source": trans.source_table  # Only the specific source table
                                    }
                                    trans_data["unions"] = [union_entry]
                                
                                transformations.append(trans_data)
                        
                        # Filter transformations to only include those relevant to this entity
                        relevant_transformations = []
                        for trans in transformations:
                            # Standard case: source_table matches entity, target_table matches dependent
                            standard_match = (trans.get("source_table") == entity_name and 
                                            trans.get("target_table") == dependent_table)
                            
                            # UNION case: source_table matches dependent, target_table matches entity
                            # This handles UNION operations where multiple sources combine into one target
                            union_match = (trans.get("source_table") == dependent_table and 
                                         trans.get("target_table") == entity_name and
                                         trans.get("union_type") is not None)
                            
                            if standard_match or union_match:
                                relevant_transformations.append(trans)
                        
                        if relevant_transformations:
                            dep_chain["transformations"] = relevant_transformations
                    
                    chain["dependencies"].append(dep_chain)
            
            return chain
        
        # Start chain building from all root entities
        if target_entity:
            # Build chain for specific target entity
            if target_entity in table_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "table", 1, set())
        else:
            # Build chains for all root entities
            for entity_name in table_lineage_data.keys():
                chains[entity_name] = build_comprehensive_chain(entity_name, "table", 1, set())
        
        # Process derived tables and merge with existing chains
        derived_table_chains = self.chain_builder_engine.process_derived_tables(sql)
        if derived_table_chains:
            chains = self.chain_builder_engine.merge_derived_table_chains(chains, derived_table_chains)
        
        # Post-process chains to add missing source columns from filter conditions
        self.chain_builder_engine.add_missing_source_columns(chains, sql, column_lineage_data, column_transformations_data)
        
        # Post-process to integrate column transformations into column metadata
        self.transformation_engine.integrate_column_transformations(chains, sql, self.main_analyzer)
        
        # Calculate actual max depth
        actual_max_depth = calculate_max_depth(chains)
        
        # Update summary with actual transformation and metadata detection
        has_transformations = detect_transformations_in_chains(chains)
        has_metadata = detect_metadata_in_chains(chains)
        
        # Calculate actually used columns from transformations and lineage - extracted from analyzer-bkup.py lines 1583-1651
        # We need to normalize column references to avoid counting duplicates
        # (e.g., 'name' and 'QUERY_RESULT.name' should be treated as the same logical column)
        used_columns = set()
        
        def normalize_column_name(column_ref: str) -> str:
            """Normalize column reference to just the column name, ignoring table prefixes for counting."""
            if column_ref and '.' in column_ref:
                # Skip QUERY_RESULT columns as they are output columns, not source columns
                if column_ref.startswith('QUERY_RESULT.'):
                    return None
                return column_ref.split('.')[-1]  # Get just the column name
            return column_ref
        
        # Add columns from upstream lineage data (these are the actual source columns)
        for column_ref, upstream_columns in column_lineage_data.items():
            # Add upstream columns (source columns)
            for upstream_col in upstream_columns:
                normalized = normalize_column_name(upstream_col)
                if normalized:
                    used_columns.add(normalized)
        
        # Add columns from column transformations (focus on source columns)
        for transformation_list in result.column_lineage.transformations.values():
            for transformation in transformation_list:
                if transformation.source_column:
                    normalized = normalize_column_name(transformation.source_column)
                    if normalized:
                        used_columns.add(normalized)
        
        # Add columns from table transformations (join conditions, filters, etc.)
        for transformation_list in result.table_lineage.transformations.values():
            for transformation in transformation_list:
                # Join conditions
                for join_condition in transformation.join_conditions:
                    if join_condition.left_column:
                        normalized = normalize_column_name(join_condition.left_column)
                        if normalized:
                            used_columns.add(normalized)
                    if join_condition.right_column:
                        normalized = normalize_column_name(join_condition.right_column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Filter conditions
                for filter_condition in transformation.filter_conditions:
                    if filter_condition.column:
                        normalized = normalize_column_name(filter_condition.column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Group by columns
                for group_col in transformation.group_by_columns:
                    normalized = normalize_column_name(group_col)
                    if normalized:
                        used_columns.add(normalized)
                
                # Having conditions
                for having_condition in transformation.having_conditions:
                    if having_condition.column:
                        normalized = normalize_column_name(having_condition.column)
                        if normalized:
                            used_columns.add(normalized)
                
                # Order by columns
                for order_col in transformation.order_by_columns:
                    normalized = normalize_column_name(order_col)
                    if normalized:
                        used_columns.add(normalized)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth if depth > 0 else "unlimited",
            "actual_max_depth": actual_max_depth,
            "target_entity": target_entity,
            "chains": chains,
            "summary": {
                "total_tables": len(set(table_lineage_data.keys()) | set().union(*table_lineage_data.values()) if table_lineage_data else set()),
                "total_columns": len(used_columns),
                "has_transformations": has_transformations,
                "has_metadata": has_metadata,
                "chain_count": len(chains)
            },
            "errors": result.errors,
            "warnings": result.warnings
        }

    def get_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """
        Get the JSON representation of comprehensive lineage chain for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 0 for unlimited depth)
            target_entity: Specific table or column to focus on (optional)
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the comprehensive lineage chain
        """
        chain_data = self.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2)



    def _build_cte_lineage_chain(self, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries with proper single-flow chains."""
        # Import and create CTE analyzer
        from .cte_analyzer import CTEAnalyzer
        cte_analyzer = CTEAnalyzer(self.dialect, main_analyzer=self.main_analyzer)
        return cte_analyzer.build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
