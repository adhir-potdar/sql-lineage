"""Lineage chain builder analyzer."""

from typing import Dict, Any, List, Optional, Set
import json
import sqlglot
from sqlglot import expressions as exp
from .base_analyzer import BaseAnalyzer

# Import new utility modules
from ...utils.sql_parsing_utils import (
    build_alias_to_table_mapping, extract_function_type, is_column_from_table,
    clean_source_expression, extract_table_references_from_sql
)
from ...utils.column_extraction_utils import (
    extract_all_referenced_columns, extract_aggregate_columns, 
    extract_qualified_filter_columns
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
from ..transformation_engine import TransformationEngine
from ..chain_builder_engine import ChainBuilderEngine


class LineageChainBuilder(BaseAnalyzer):
    """Analyzer for building lineage chains."""
    
    def __init__(self, dialect: str = "trino", main_analyzer=None):
        """Initialize lineage chain builder with optional reference to main analyzer."""
        super().__init__(dialect)
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
            
            # Get metadata from the main analyzer's metadata registry
            result.metadata = {}
            if hasattr(self.main_analyzer, 'metadata_registry') and self.main_analyzer.metadata_registry:
                # Get all unique tables from both upstream and downstream
                all_tables = set()
                all_tables.update(table_lineage.upstream.keys())
                all_tables.update(table_lineage.downstream.keys())  
                for tables in table_lineage.upstream.values():
                    all_tables.update(tables)
                for tables in table_lineage.downstream.values():
                    all_tables.update(tables)
                
                # Get metadata for each table
                for table_name in all_tables:
                    table_metadata = self.main_analyzer.metadata_registry.get_table_metadata(table_name)
                    if table_metadata:
                        result.metadata[table_name] = table_metadata
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
            
            # Get metadata from the main analyzer's metadata registry
            result.metadata = {}
            if hasattr(self.main_analyzer, 'metadata_registry') and self.main_analyzer.metadata_registry:
                # Get all unique tables from both upstream and downstream
                all_tables = set()
                all_tables.update(table_lineage.upstream.keys())
                all_tables.update(table_lineage.downstream.keys())  
                for tables in table_lineage.upstream.values():
                    all_tables.update(tables)
                for tables in table_lineage.downstream.values():
                    all_tables.update(tables)
                
                # Get metadata for each table
                for table_name in all_tables:
                    table_metadata = self.main_analyzer.metadata_registry.get_table_metadata(table_name)
                    if table_metadata:
                        result.metadata[table_name] = table_metadata
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
            
            # Add table metadata if available - extracted from analyzer-bkup.py lines 1299-1310
            if entity_name in result.metadata:
                table_meta = result.metadata[entity_name]
                metadata = {
                    "table_type": table_meta.table_type.value
                }
                
                # Only include non-null values to keep output clean
                if table_meta.schema:
                    metadata["schema"] = table_meta.schema
                if table_meta.description:
                    metadata["description"] = table_meta.description
                
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
                                    'is_single_table_context': is_single_table,
                                    'tables_in_context': [trans.source_table] if is_single_table else [],
                                    'sql': sql
                                }
                                
                                # Filter conditions to only include those relevant to the current entity
                                if hasattr(trans, 'filter_conditions') and trans.filter_conditions:
                                    relevant_filters = []
                                    
                                    for fc in trans.filter_conditions:
                                        # Only include filter conditions that reference columns from this entity
                                        if self._is_column_from_table(fc.column, entity_name, context_info):
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
                                        if self._is_column_from_table(col, entity_name, context_info):
                                            relevant_group_by.append(col)
                                    if relevant_group_by:
                                        trans_data["group_by_columns"] = relevant_group_by
                                
                                # Having conditions - only include those referencing columns from this entity
                                if hasattr(trans, 'having_conditions') and trans.having_conditions:
                                    relevant_having = []
                                    for hc in trans.having_conditions:
                                        # Having conditions often involve aggregations like COUNT(*) or AVG(u.salary)
                                        # Check if they reference this entity or if they are general aggregations for this table
                                        is_relevant = (self._is_column_from_table(hc.column, entity_name, context_info) or 
                                                     self._is_aggregate_function_for_table(hc.column, entity_name, sql))
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
                                        if self._is_column_from_table(col_name, entity_name, context_info):
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
        
        # Post-process chains to add missing source columns from filter conditions
        self._add_missing_source_columns(chains, sql, column_lineage_data, column_transformations_data)
        
        # Post-process to integrate column transformations into column metadata
        self._integrate_column_transformations(chains, sql)
        
        # Calculate actual max depth
        actual_max_depth = self._calculate_max_depth(chains)
        
        # Update summary with actual transformation and metadata detection
        has_transformations = self._detect_transformations_in_chains(chains)
        has_metadata = self._detect_metadata_in_chains(chains)
        
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

    def _calculate_max_depth(self, chains: Dict[str, Any]) -> int:
        """Calculate the maximum depth in the chains."""
        max_depth = 0
        
        def get_chain_depth(chain_entity: Dict[str, Any]) -> int:
            current_depth = chain_entity.get('depth', 0)
            dependencies = chain_entity.get('dependencies', [])
            
            if not dependencies:
                return current_depth
            
            max_dep_depth = max(get_chain_depth(dep) for dep in dependencies)
            return max(current_depth, max_dep_depth)
        
        for chain in chains.values():
            chain_depth = get_chain_depth(chain)
            max_depth = max(max_depth, chain_depth)
        
        return max_depth
    
    def _add_missing_source_columns(self, chains: Dict, sql: str = None, column_lineage_data: Dict = None, column_transformations_data: Dict = None) -> None:
        """Add missing source columns and handle QUERY_RESULT dependencies - extracted from analyzer-bkup.py."""
        if not sql:
            return
            
        try:
            # Parse SQL to get select columns and alias mapping for QUERY_RESULT columns
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[table.alias] = table.name
            
            # Get select columns from parsing
            select_columns = []
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                for expr in select_stmt.expressions:
                    if isinstance(expr, sqlglot.exp.Column):
                        raw_expr = str(expr)
                        table_part = str(expr.table) if expr.table else None
                        column_name = str(expr.name) if expr.name else raw_expr
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': column_name,
                            'source_table': table_part
                        })
                    else:
                        # Handle other expressions
                        raw_expr = str(expr)
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': raw_expr,
                            'source_table': None
                        })
        except:
            select_columns = []
            alias_to_table = {}
        
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Handle QUERY_RESULT and CTAS dependencies specially
            for dep in entity_data.get('dependencies', []):
                dep_entity = dep.get('entity')
                if dep_entity == 'QUERY_RESULT' or self._is_ctas_target_table(sql, dep_entity):
                    # Get existing columns to avoid duplication
                    existing_metadata = dep.get('metadata', {})
                    existing_columns = existing_metadata.get('table_columns', [])
                    existing_column_names = {col.get('name') for col in existing_columns}
                    
                    # Add qualified column names to QUERY_RESULT metadata
                    table_columns = list(existing_columns)  # Start with existing columns
                    # Check if this is a CTAS query
                    is_ctas = self._is_ctas_target_table(sql, dep_entity)
                    
                    if is_ctas:
                        # CTAS query: Add target table columns with special handling for aggregates
                        table_columns = self._build_ctas_target_columns(sql, select_columns)
                        
                        # Add GROUP BY information to transformations
                        self._add_group_by_to_ctas_transformations(dep, sql)
                    else:
                        # Check if this is a JOIN or UNION query first
                        is_join_query = any(sel_col.get('source_table') is not None for sel_col in select_columns)
                        is_union_query = 'UNION' in sql.upper()
                        
                        # First try to use column lineage data if available (but skip for JOIN/UNION queries to avoid duplicates)
                        if not (is_join_query or is_union_query) and (column_lineage_data and column_transformations_data) and (column_lineage_data or column_transformations_data):
                            # Collect all QUERY_RESULT columns from both upstream and transformations
                            query_result_columns = set()
                            
                            # Add columns from upstream dependencies
                            for target_column in column_lineage_data.keys():
                                if target_column.startswith('QUERY_RESULT.'):
                                    column_name = target_column.replace('QUERY_RESULT.', '')
                                    query_result_columns.add(column_name)
                            
                            # Add columns from transformations (includes literal columns)
                            for target_column in column_transformations_data.keys():
                                if target_column.startswith('QUERY_RESULT.'):
                                    column_name = target_column.replace('QUERY_RESULT.', '')
                                    query_result_columns.add(column_name)
                            
                            # Build QUERY_RESULT columns
                            for column_name in sorted(query_result_columns):
                                # Build upstream list from source columns or mark as computed
                                target_col_key = f"QUERY_RESULT.{column_name}"
                                source_columns = column_lineage_data.get(target_col_key, set())
                                
                                if source_columns:
                                    # Has upstream dependencies
                                    upstream_list = [f"QUERY_RESULT.{column_name}"]
                                    column_type = "DIRECT"
                                else:
                                    # No upstream dependencies (likely computed/literal)
                                    upstream_list = [f"QUERY_RESULT.{column_name}"]
                                    column_type = "SOURCE"
                                
                                # Only add if not already present
                                if column_name not in existing_column_names:
                                    column_info = {
                                        "name": column_name,
                                        "upstream": upstream_list,
                                        "type": column_type
                                    }
                                    table_columns.append(column_info)
                        
                        # Run aggregate-aware processing if needed, or for JOIN/UNION queries
                        needs_aggregate_processing = self._query_has_aggregates(sql)
                        
                        if needs_aggregate_processing or is_join_query or is_union_query:
                            # Use same logic as original analyzer-bkup.py
                            inferred_columns = self._infer_query_result_columns_simple(sql, select_columns)
                            has_table_prefixes = any('.' in col.get('name', '') for col in inferred_columns)
                            
                            if has_table_prefixes or is_join_query:
                                # JOIN query: Use qualified names from select columns
                                for sel_col in select_columns:
                                    source_table = sel_col.get('source_table')
                                    raw_expression = sel_col.get('raw_expression')
                                    column_name = sel_col.get('column_name')
                                    
                                    if (source_table in alias_to_table and 
                                        alias_to_table[source_table] == entity_name):
                                        # Regular table-prefixed column
                                        upstream_col = f"{entity_name}.{column_name}"
                                        
                                        column_info = {
                                            "name": raw_expression,
                                            "upstream": [upstream_col],
                                            "type": "DIRECT"
                                        }
                                        table_columns.append(column_info)
                                    elif source_table is None and self._is_aggregate_function(raw_expression):
                                        # Aggregate function column - only add if relevant to this entity
                                        if self._is_aggregate_function_for_table(raw_expression, entity_name, sql):
                                            alias = self._extract_alias_from_expression(raw_expression)
                                            func_type = self._extract_function_type(raw_expression)
                                            
                                            column_info = {
                                                "name": alias or column_name,
                                                "upstream": [f"{entity_name}.{alias or column_name}"],
                                                "type": "DIRECT",
                                                "transformation": {
                                                    "source_expression": raw_expression.replace(f" as {alias}", "").replace(f" AS {alias}", "") if alias else raw_expression,
                                                    "transformation_type": "AGGREGATE",
                                                    "function_type": func_type
                                                }
                                            }
                                            table_columns.append(column_info)
                            elif is_union_query:
                                # UNION query: Only add columns that belong to this specific table
                                union_columns = self._get_union_columns_for_table(sql, entity_name)
                                for column_name in union_columns:
                                    column_info = {
                                        "name": column_name,
                                        "upstream": [f"{entity_name}.{column_name}"],
                                        "type": "DIRECT"
                                    }
                                    table_columns.append(column_info)
                        else:
                            # Simple query: Add columns from select statement to QUERY_RESULT
                            for sel_col in select_columns:
                                if sel_col.get('source_table') is None:  # Unprefixed columns
                                    raw_expression = sel_col.get('raw_expression')
                                    column_name = sel_col.get('column_name')
                                    
                                    # Extract clean name (prefer alias over raw column name)
                                    clean_name = self._extract_clean_column_name(raw_expression, column_name)
                                    
                                    column_info = {
                                        "name": clean_name,
                                        "upstream": [f"QUERY_RESULT.{clean_name}"],
                                        "type": "DIRECT"
                                    }
                                    table_columns.append(column_info)
                    
                    if table_columns:
                        # Deduplicate columns before assigning (by name only)
                        # Prefer columns with more detailed information (transformations, proper names)
                        deduplicated_columns = []
                        seen_column_names = set()
                        column_map = {}
                        
                        # First pass: group columns by name
                        for col in table_columns:
                            col_name = col.get('name')
                            if col_name not in column_map:
                                column_map[col_name] = []
                            column_map[col_name].append(col)
                        
                        # Second pass: choose the best column for each name
                        for col_name, columns in column_map.items():
                            if len(columns) == 1:
                                deduplicated_columns.append(columns[0])
                            else:
                                # Multiple columns with same name - choose the best one
                                best_col = self._choose_best_column(columns)
                                deduplicated_columns.append(best_col)
                        
                        if 'metadata' not in dep:
                            dep['metadata'] = {}
                        dep['metadata']['table_columns'] = deduplicated_columns
                
            # Handle source table missing columns
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies
            if entity_data.get('depth') == 0:
                source_columns = set()
                
                # Look through dependencies to find transformations that reference this table
                for dep in entity_data.get('dependencies', []):
                    for trans in dep.get('transformations', []):
                        if trans.get('source_table') == entity_name:
                            # Extract filter condition columns
                            for condition in trans.get('filter_conditions', []):
                                col = condition.get('column', '')
                                if col:
                                    # Clean column name by removing table prefix if present
                                    clean_col = col.split('.')[-1] if '.' in col else col
                                    source_columns.add(clean_col)
                            
                            # Extract group by columns
                            for group_col in trans.get('group_by_columns', []):
                                if group_col:
                                    # Clean column name by removing table prefix if present
                                    clean_col = group_col.split('.')[-1] if '.' in group_col else group_col
                                    source_columns.add(clean_col)
                            
                            # Extract join condition columns
                            for join in trans.get('joins', []):
                                for condition in join.get('conditions', []):
                                    for col_key in ['left_column', 'right_column']:
                                        col_ref = condition.get(col_key, '')
                                        if col_ref and entity_name in col_ref:
                                            clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                                            source_columns.add(clean_col)
                
                # Add the columns found in transformations, merging with existing
                if source_columns:
                    # Get existing column names to avoid duplicates
                    existing_columns = {col['name'] for col in table_columns}
                    
                    # Add new columns from transformations
                    for column_name in source_columns:
                        if column_name not in existing_columns:
                            column_info = {
                                "name": column_name,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            table_columns.append(column_info)
                    
                    # Update the metadata
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    entity_data['metadata']['table_columns'] = table_columns

    def _integrate_column_transformations(self, chains: Dict, sql: str = None) -> None:
        """Integrate column transformations into column metadata throughout the chain."""
        self.transformation_engine.integrate_column_transformations(chains, sql, self.main_analyzer)

    def _add_columns_for_entity(self, entity_data: Dict, transformations: List, sql: str, column_lineage_data: Dict = None, column_transformations_data: Dict = None):
        """Add column information to an entity based on transformations - extracted from analyzer-bkup.py."""
        entity_name = entity_data.get('entity')
        entity_type = entity_data.get('entity_type', 'table')
        if not entity_name:
            return
        
            
        metadata = entity_data.get('metadata', {})
        table_columns = metadata.get('table_columns', [])
        
        
        # Logic extracted from analyzer-bkup.py lines 1353-1396
        if not table_columns and entity_type == "table" and entity_name != "QUERY_RESULT":
            source_columns = set()
            
            # Get selected columns from column lineage (SELECT clause columns)
            if column_lineage_data:
                for column_ref in column_lineage_data.keys():
                    # Column refs without table prefix are from the source table
                    if '.' not in column_ref:
                        source_columns.add(column_ref)
            
            # Add all found columns to table metadata
            for column_name in source_columns:
                column_info = {
                    "name": column_name,
                    "upstream": [],
                    "type": "SOURCE" 
                }
                table_columns.append(column_info)
                
            # Add table metadata for source tables
            table_metadata_dict = {
                "table_type": "TABLE",
                "schema": "default",
                "table_columns": table_columns
            }
            
            # Get description from metadata registry if available
            if self.main_analyzer and hasattr(self.main_analyzer, 'metadata_registry') and self.main_analyzer.metadata_registry:
                table_metadata = self.main_analyzer.metadata_registry.get_table_metadata(entity_name)
                if table_metadata and 'description' in table_metadata:
                    table_metadata_dict["description"] = table_metadata['description']
                else:
                    table_metadata_dict["description"] = "Table information"
            else:
                table_metadata_dict["description"] = "Table information"
            
            metadata.update(table_metadata_dict)
        
        # Special handling for QUERY_RESULT - infer result columns from SQL parsing (lines 1372-1392)
        elif entity_name == "QUERY_RESULT":
            # For QUERY_RESULT, we should infer columns from the SELECT statement
            all_query_result_columns = self._infer_query_result_columns(sql, column_lineage_data or {})
            table_columns = all_query_result_columns
            
            # Add transformations for table-level transformations
            if sql and 'WHERE' in sql.upper() and self.main_analyzer:
                transformations_list = self.main_analyzer.transformation_analyzer.extract_filter_transformations(sql)
                if transformations_list:
                    entity_data["transformations"] = transformations_list
            
            metadata.update({
                "table_columns": table_columns
            })
        
        # Only add table_columns if not empty (line 1395-1396)
        if table_columns:
            metadata["table_columns"] = table_columns
        
        # Ensure all source tables have basic metadata fields - maintain expected field order
        if entity_data.get('depth') == 0 and entity_type == 'table':
            # Create ordered metadata structure matching expected output
            ordered_metadata = {}
            ordered_metadata["table_type"] = metadata.get("table_type", "TABLE")
            ordered_metadata["schema"] = metadata.get("schema", "default")
            
            # Get description from metadata registry if available
            if self.main_analyzer and hasattr(self.main_analyzer, 'metadata_registry') and self.main_analyzer.metadata_registry:
                table_metadata = self.main_analyzer.metadata_registry.get_table_metadata(entity_name)
                if table_metadata and 'description' in table_metadata:
                    ordered_metadata["description"] = table_metadata['description']
                else:
                    ordered_metadata["description"] = metadata.get("description", "Table information")
            else:
                ordered_metadata["description"] = metadata.get("description", "Table information")
            
            # Add table_columns last to maintain proper ordering
            if table_columns:
                ordered_metadata["table_columns"] = table_columns
            
            metadata = ordered_metadata
        
        # Update the entity metadata  
        if 'metadata' not in entity_data:
            entity_data['metadata'] = {}
        entity_data['metadata'].update(metadata)
        
        # Process dependencies recursively
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            self._add_columns_for_entity(dep, transformations, sql, column_lineage_data, column_transformations_data)

    def _detect_transformations_in_chains(self, chains: Dict) -> bool:
        """Detect if there are any transformations in the chains."""
        def has_transformations_recursive(entity_data):
            # Check current entity
            if entity_data.get('transformations'):
                return True
            
            # Check dependencies recursively
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                if has_transformations_recursive(dep):
                    return True
            
            return False
        
        # Check all top-level chains
        for entity_data in chains.values():
            if has_transformations_recursive(entity_data):
                return True
        
        return False

    def _detect_metadata_in_chains(self, chains: Dict) -> bool:
        """Detect if there are any meaningful metadata in the chains."""
        def has_metadata_recursive(entity_data):
            # Check current entity metadata
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            if table_columns:
                return True
            
            # Check dependencies recursively
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                if has_metadata_recursive(dep):
                    return True
            
            return False
        
        # Check all top-level chains
        for entity_data in chains.values():
            if has_metadata_recursive(entity_data):
                return True
        
        return False

    def _infer_query_result_columns(self, sql: str, column_lineage_data: Dict) -> List[Dict]:
        """
        Infer QUERY_RESULT columns from SQL query with proper aggregate function handling.
        """
        import sqlglot
        
        result_columns = []
        
        try:
            # Parse SQL with SQLGlot for proper expression handling
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            
            if select_stmt and select_stmt.expressions:
                for expr in select_stmt.expressions:
                    column_info = self._process_select_expression(expr, sql)
                    if column_info:
                        result_columns.append(column_info)
            
        except Exception:
            # Fallback to simple regex parsing if SQLGlot fails
            result_columns = self._infer_query_result_columns_simple_fallback(sql)
        
        return result_columns
    
    def _process_select_expression(self, expr, sql: str) -> Dict:
        """Process a SELECT expression to create proper column info with transformation details."""
        import sqlglot
        
        # Get the raw expression string
        raw_expr = str(expr)
        
        # Extract alias if present
        alias = None
        if hasattr(expr, 'alias') and expr.alias:
            alias = str(expr.alias)
        
        # Determine column name (use alias if available, otherwise expression)
        if alias:
            column_name = alias
        elif isinstance(expr, sqlglot.exp.Column):
            column_name = str(expr.name) if expr.name else raw_expr
        else:
            column_name = raw_expr
        
        # Check if this is an aggregate function
        if self._is_aggregate_function(raw_expr):
            # Handle aggregate function with transformation details
            func_type = self._extract_function_type(raw_expr)
            source_expr = raw_expr.replace(f" AS {alias}", "").replace(f" as {alias}", "") if alias else raw_expr
            
            # Extract source columns from the aggregate function
            upstream_columns = self._extract_upstream_from_aggregate(raw_expr, sql)
            
            return {
                "name": column_name,
                "upstream": upstream_columns,
                "type": "DIRECT",
                "transformation": {
                    "source_expression": source_expr,
                    "transformation_type": "AGGREGATE",
                    "function_type": func_type
                }
            }
        else:
            # Handle regular column or expression
            if isinstance(expr, sqlglot.exp.Column):
                # Simple column reference
                table_part = str(expr.table) if expr.table else None
                if table_part:
                    upstream = [f"{table_part}.{column_name}"]
                else:
                    upstream = [column_name]
                return {
                    "name": column_name,
                    "upstream": upstream,
                    "type": "DIRECT"
                }
            else:
                # Other expression
                return {
                    "name": column_name,
                    "upstream": [f"QUERY_RESULT.{column_name}"],
                    "type": "DIRECT"
                }
    
    def _extract_upstream_from_aggregate(self, aggregate_expr: str, sql: str) -> List[str]:
        """Extract upstream column references from aggregate function."""
        import re
        
        # Extract column references from inside aggregate functions
        upstream = []
        
        # Pattern to match columns inside aggregate functions: FUNC(table.column) or FUNC(column)
        func_pattern = r'\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([^)]+)\s*\)'
        matches = re.findall(func_pattern, aggregate_expr, re.IGNORECASE)
        
        for match in matches:
            col_ref = match.strip()
            if col_ref == '*':
                # COUNT(*) - doesn't reference specific columns
                continue
            elif '.' in col_ref:
                # Qualified column reference (table.column)
                upstream.append(col_ref)
            else:
                # Unqualified column - try to infer table from SQL context
                # For now, just use the column name
                upstream.append(col_ref)
        
        return upstream if upstream else [f"QUERY_RESULT.{aggregate_expr}"]
    
    def _infer_query_result_columns_simple_fallback(self, sql: str) -> List[Dict]:
        """Simple fallback method using regex parsing."""
        import re
        
        result_columns = []
        
        # Try to extract SELECT columns from SQL
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            
            # Handle simple cases like "SELECT name, email FROM table_name"
            # Split by comma and clean up
            columns = [col.strip() for col in select_clause.split(',')]
            
            for col in columns:
                original_col = col.strip()  # Keep original column reference
                
                # Extract alias name from "expression AS alias" -> "alias"
                col_name = original_col
                if ' AS ' in col.upper():
                    # Split on AS and take the alias part (after AS)
                    parts = col.split(' AS ', 1) if ' AS ' in col else col.split(' as ', 1)
                    if len(parts) > 1:
                        col_name = parts[1].strip()
                    else:
                        col_name = parts[0].strip()
                elif ' as ' in col:
                    # Handle lowercase 'as'
                    parts = col.split(' as ', 1)
                    if len(parts) > 1:
                        col_name = parts[1].strip()
                    else:
                        col_name = parts[0].strip()
                
                # Clean column name (remove table prefixes and quotes)
                clean_col_name = col_name.split('.')[-1].strip().strip('"').strip("'")
                
                # Create column info
                column_info = {
                    "name": clean_col_name,
                    "upstream": [f"QUERY_RESULT.{clean_col_name}"],
                    "type": "DIRECT"
                }
                result_columns.append(column_info)
        
        return result_columns


    def _extract_table_columns_from_sql(self, sql: str, table_name: str) -> set:
        """Extract columns that are referenced from a specific table in the SQL."""
        import re
        
        columns = set()
        
        # Look for patterns like "table.column" or "alias.column" in SELECT and WHERE clauses
        # For simple query "SELECT name, email FROM table_name WHERE age > 25"
        # We want to extract: name, email, age for the specified table
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            select_columns = [col.strip() for col in select_clause.split(',')]
            
            for col in select_columns:
                # Clean column name (remove functions, aliases, table prefixes)
                clean_col = col.split('.')[-1].strip()  # Remove table prefix
                clean_col = re.sub(r'\s+as\s+\w+', '', clean_col, flags=re.IGNORECASE)  # Remove " AS alias"
                clean_col = clean_col.strip().strip('"').strip("'")  # Remove quotes
                if clean_col and clean_col != '*':
                    columns.add(clean_col)
        
        # Extract WHERE clause columns
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Find column references in conditions
            column_patterns = [
                r'\b(\w+)\s*[><=!]+',  # column > value
                r'\b(\w+)\s+IN\s*\(',  # column IN (...)
                r'\b(\w+)\s+LIKE\s',   # column LIKE ...
            ]
            
            for pattern in column_patterns:
                matches = re.findall(pattern, where_clause, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        column = match[0]
                    else:
                        column = match
                    # Only add if it's a simple column name (not a function or keyword)
                    if column and column.lower() not in ['and', 'or', 'not', 'is', 'null', 'true', 'false']:
                        columns.add(column)
        
        return columns

    def _extract_join_columns_from_sql(self, sql: str, table_name: str) -> set:
        """Extract JOIN columns for a specific table from SQL query."""
        from ...utils.column_extraction_utils import extract_columns_from_joins
        return extract_columns_from_joins(sql, table_name)

    def _is_ctas_target_table(self, sql: str, table_name: str) -> bool:
        """Check if this table is a CTAS target table."""
        if not is_ctas_query(sql):
            return False
        # Simple check if table name appears after CREATE TABLE
        return table_name.lower() in sql.lower() and 'CREATE TABLE' in sql.upper()

    def _build_ctas_target_columns(self, sql: str, select_columns: List[Dict]) -> List[Dict]:
        """Build target table columns for CTAS queries with aggregate handling."""
        table_columns = []
        
        for sel_col in select_columns:
            raw_expression = sel_col.get('raw_expression', '')
            column_name = sel_col.get('column_name', raw_expression)
            
            # Check if this is an aggregate function
            if self._is_aggregate_function(raw_expression):
                # Handle aggregate function with transformation details
                alias = self._extract_alias_from_expression(raw_expression)
                func_type = self._extract_function_type(raw_expression)
                
                column_info = {
                    "name": alias or column_name,
                    "upstream": [],
                    "type": "RESULT",
                    "transformation": {
                        "source_expression": raw_expression.replace(f" as {alias}", "").replace(f" AS {alias}", "") if alias else raw_expression,
                        "transformation_type": "AGGREGATE",
                        "function_type": func_type
                    }
                }
            else:
                # Regular column (pass-through)
                column_info = {
                    "name": column_name,
                    "upstream": [],
                    "type": "SOURCE"
                }
            
            table_columns.append(column_info)
        
        return table_columns

    def _extract_all_referenced_columns(self, sql: str, table_name: str) -> set:
        """Extract all columns referenced in SQL for a specific table."""
        return extract_all_referenced_columns(sql, table_name, self.dialect)

    def _extract_aggregate_result_columns(self, sql: str, table_name: str) -> List[Dict]:
        """Extract aggregate RESULT columns that belong to a specific table."""
        result_columns = []
        
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Only extract aggregate columns for tables that are being grouped by
            # Determine if this table is the main aggregating table by checking GROUP BY columns
            if not self._is_main_aggregating_table(sql, table_name):
                return result_columns
                
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                for expr in select_stmt.expressions:
                    if not isinstance(expr, sqlglot.exp.Column):
                        # Check if this is an aggregate function
                        raw_expr = str(expr)
                        
                        # Extract alias if present
                        alias = None
                        if hasattr(expr, 'alias') and expr.alias:
                            alias = str(expr.alias)
                        
                        # Check for aggregate functions
                        if any(agg_func in raw_expr.upper() for agg_func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']):
                            column_name = alias if alias else raw_expr
                            
                            # Clean source expression to remove AS alias part
                            clean_source_expr = raw_expr
                            if ' AS ' in raw_expr.upper():
                                clean_source_expr = raw_expr.split(' AS ')[0].strip()
                            elif ' as ' in raw_expr:
                                clean_source_expr = raw_expr.split(' as ')[0].strip()
                            
                            # Determine function type
                            function_type = None
                            if 'COUNT(' in raw_expr.upper():
                                function_type = "COUNT"
                            elif 'SUM(' in raw_expr.upper():
                                function_type = "SUM"
                            elif 'AVG(' in raw_expr.upper():
                                function_type = "AVG"
                            elif 'MAX(' in raw_expr.upper():
                                function_type = "MAX"
                            elif 'MIN(' in raw_expr.upper():
                                function_type = "MIN"
                            
                            if function_type:
                                result_col = {
                                    "name": column_name,
                                    "upstream": [],
                                    "type": "RESULT",
                                    "transformation": {
                                        "source_expression": clean_source_expr,
                                        "transformation_type": "AGGREGATE",
                                        "function_type": function_type
                                    }
                                }
                                result_columns.append(result_col)
                                
        except Exception:
            # If parsing fails, return empty list
            pass
            
        return result_columns

    def _is_main_aggregating_table(self, sql: str, table_name: str) -> bool:
        """Determine if a table is the main table being aggregated (appears in GROUP BY)."""
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[table.alias] = table.name
                    
            # Find table aliases for this table_name
            table_aliases = []
            for alias, actual_table in alias_to_table.items():
                if actual_table == table_name:
                    table_aliases.append(alias)
            
            # If no alias found, use the table name itself
            if not table_aliases:
                table_aliases = [table_name]
            
            # Check GROUP BY clause
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                group_by = select_stmt.find(sqlglot.exp.Group)
                if group_by:
                    for expr in group_by.expressions:
                        if isinstance(expr, sqlglot.exp.Column):
                            table_part = str(expr.table) if expr.table else None
                            
                            # If this column belongs to our table, it's the main aggregating table
                            if table_part in table_aliases or (not table_part and len(table_aliases) == 1):
                                return True
                            elif not table_part:  # No table prefix, could be from our table
                                return True
            
            return False
            
        except Exception:
            # If parsing fails, default to False (don't add aggregate columns)
            return False

    def _extract_aggregate_source_columns(self, sql: str, table_name: str) -> List[Dict]:
        """Extract columns that are used in aggregate functions and should have upstream relationships."""
        result_columns = []
        
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[table.alias] = table.name
                    
            # Find table aliases for this table_name
            table_aliases = []
            for alias, actual_table in alias_to_table.items():
                if actual_table == table_name:
                    table_aliases.append(alias)
            
            if not table_aliases:
                table_aliases = [table_name]
            
            # Track which columns are used in which aggregate functions
            column_to_aggregates = {}
            
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                for expr in select_stmt.expressions:
                    if not isinstance(expr, sqlglot.exp.Column):
                        raw_expr = str(expr)
                        
                        # Extract alias if present
                        alias = None
                        if hasattr(expr, 'alias') and expr.alias:
                            alias = str(expr.alias)
                        
                        # Find aggregate functions and extract the column they operate on
                        for agg_func in ['SUM', 'AVG', 'MAX', 'MIN']:
                            if f'{agg_func}(' in raw_expr.upper():
                                # Extract the column inside the aggregate function
                                import re
                                pattern = f'{agg_func}\\s*\\(\\s*([^)]+)\\s*\\)'
                                match = re.search(pattern, raw_expr, re.IGNORECASE)
                                if match:
                                    inner_expr = match.group(1).strip()
                                    # Check if this is a column reference with table prefix
                                    if '.' in inner_expr:
                                        table_part, col_part = inner_expr.split('.', 1)
                                        if table_part in table_aliases:
                                            # This column is used in an aggregate function
                                            if col_part not in column_to_aggregates:
                                                column_to_aggregates[col_part] = []
                                            
                                            result_col_name = alias if alias else f"{agg_func.lower()}_{col_part}"
                                            column_to_aggregates[col_part].append(f"QUERY_RESULT.{result_col_name}")
            
            # Create column info for each source column that has aggregate relationships
            for col_name, upstream_list in column_to_aggregates.items():
                if upstream_list:
                    column_info = {
                        "name": col_name,
                        "upstream": upstream_list,
                        "type": "DIRECT"
                    }
                    result_columns.append(column_info)
                    
        except Exception:
            # If parsing fails, return empty list
            pass
            
        return result_columns

    def _extract_qualified_filter_columns(self, sql: str, table_name: str) -> List[Dict]:
        """Extract qualified column names used in filter conditions."""
        result_columns = []
        
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[table.alias] = table.name
                    
            # Find table aliases for this table_name
            table_aliases = []
            for alias, actual_table in alias_to_table.items():
                if actual_table == table_name:
                    table_aliases.append(alias)
            
            if not table_aliases:
                table_aliases = [table_name]
            
            qualified_columns = set()
            
            # Extract from WHERE clause
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                where_clause = select_stmt.find(sqlglot.exp.Where)
                if where_clause:
                    for column in where_clause.find_all(sqlglot.exp.Column):
                        table_part = str(column.table) if column.table else None
                        column_name = str(column.name) if column.name else None
                        
                        if table_part in table_aliases and column_name:
                            qualified_name = f"{table_name}.{column_name}"
                            qualified_columns.add(qualified_name)
            
            # Create column info for each qualified column
            for qualified_col in qualified_columns:
                column_info = {
                    "name": qualified_col,
                    "upstream": [],
                    "type": "SOURCE"
                }
                result_columns.append(column_info)
                
        except Exception:
            # If parsing fails, return empty list
            pass
            
        return result_columns

    def _add_group_by_to_ctas_transformations(self, dep: Dict, sql: str) -> None:
        """Add GROUP BY information to CTAS transformations."""
        import re
        
        # Extract GROUP BY columns from SQL
        group_by_match = re.search(r'GROUP\s+BY\s+([^)]+?)(?:\s+(?:HAVING|ORDER|LIMIT)|$)', sql, re.IGNORECASE)
        if group_by_match:
            group_by_clause = group_by_match.group(1).strip()
            group_by_columns = [col.strip() for col in group_by_clause.split(',')]
            
            # Add to transformations
            for transformation in dep.get('transformations', []):
                transformation['group_by_columns'] = group_by_columns

    def _is_aggregate_function(self, expression: str) -> bool:
        """Check if expression contains an aggregate function."""
        import re
        aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP_CONCAT']
        pattern = r'\b(' + '|'.join(aggregate_functions) + r')\s*\('
        return bool(re.search(pattern, expression, re.IGNORECASE))

    def _extract_alias_from_expression(self, expression: str) -> str:
        """Extract alias from expression like 'COUNT(*) as login_count'."""
        from ...utils.regex_patterns import extract_alias_from_expression
        return extract_alias_from_expression(expression)

    def _extract_function_type(self, expression: str) -> str:
        """Extract function type from aggregate expression."""
        return extract_function_type(expression)
    
    def _get_union_columns_for_table(self, sql: str, table_name: str) -> List[str]:
        """Get the columns that a specific table contributes to a UNION query."""
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Find the UNION expression
            union_expr = parsed.find(sqlglot.exp.Union)
            if not union_expr:
                return []
            
            # Find the SELECT statement that references this table
            select_stmts = []
            self._collect_union_selects_helper(union_expr, select_stmts)
            
            for select_stmt in select_stmts:
                # Check if this SELECT uses the table
                tables = list(select_stmt.find_all(sqlglot.exp.Table))
                for table in tables:
                    if table.name == table_name:
                        # Extract column names from this SELECT
                        columns = []
                        for expr in select_stmt.expressions:
                            if hasattr(expr, 'alias') and expr.alias:
                                # Use the full expression with alias for clarity
                                columns.append(f"{str(expr.this)} as {str(expr.alias)}")
                            elif isinstance(expr, sqlglot.exp.Column):
                                # Use just the column name
                                columns.append(str(expr.name) if expr.name else str(expr))
                            else:
                                # Handle literals and expressions
                                expr_str = str(expr)
                                columns.append(expr_str)
                        return columns
            
            return []
        except Exception:
            # Fallback: return standard UNION column names
            return ['type', 'id', 'identifier']
    
    def _collect_union_selects_helper(self, union_expr, select_stmts: List) -> None:
        """Helper to collect all SELECT statements from a UNION."""
        try:
            if hasattr(union_expr, 'left') and union_expr.left:
                if isinstance(union_expr.left, sqlglot.exp.Union):
                    self._collect_union_selects_helper(union_expr.left, select_stmts)
                elif isinstance(union_expr.left, sqlglot.exp.Select):
                    select_stmts.append(union_expr.left)
            
            if hasattr(union_expr, 'right') and union_expr.right:
                if isinstance(union_expr.right, sqlglot.exp.Union):
                    self._collect_union_selects_helper(union_expr.right, select_stmts)
                elif isinstance(union_expr.right, sqlglot.exp.Select):
                    select_stmts.append(union_expr.right)
        except Exception:
            pass

    def _infer_query_result_columns_simple(self, sql: str, select_columns: List[Dict]) -> List[Dict]:
        """
        Simple helper to infer query result columns similar to original analyzer-bkup.py approach.
        Returns columns with their names to check for table prefixes.
        """
        result_columns = []
        
        for sel_col in select_columns:
            raw_expression = sel_col.get('raw_expression', '')
            column_name = sel_col.get('column_name', raw_expression)
            
            # Create column info with the raw expression as name (this will have dots for qualified names)
            column_info = {
                "name": raw_expression,  # This preserves table prefixes like "u.name"
                "upstream": [],
                "type": "DIRECT"
            }
            result_columns.append(column_info)
        
        return result_columns
    
    def _choose_best_column(self, columns: List[Dict]) -> Dict:
        """Choose the best column from duplicates, preferring ones with transformation details."""
        if len(columns) == 1:
            return columns[0]
        
        # Scoring criteria (higher score = better column)
        def score_column(col):
            score = 0
            
            # Prefer columns that don't have SQL expressions as names (clean aliases)
            name = col.get('name', '')
            has_sql_expression = any(keyword in name.upper() for keyword in ['AS ', 'COUNT', 'AVG', 'SUM', 'MAX', 'MIN', '(', ')'])
            if not has_sql_expression:
                score += 20  # Higher score for clean names
            else:
                # But if it has transformation details, that's even better
                if 'transformation' in col:
                    score += 25  # Override the penalty if it has proper transformation info
            
            # Prefer columns with transformation details
            if 'transformation' in col:
                score += 20
                transformation = col['transformation']
                if transformation.get('transformation_type') == 'AGGREGATE':
                    score += 15
                if transformation.get('function_type'):
                    score += 5
            
            # Prefer columns with proper upstream references (not self-referential)
            upstream = col.get('upstream', [])
            if upstream and not any('QUERY_RESULT.' in ref for ref in upstream):
                score += 5
                
            # Prefer DIRECT type for computed columns, SOURCE for base columns
            col_type = col.get('type')
            if col_type == 'DIRECT' and 'transformation' in col:
                score += 8
            elif col_type == 'SOURCE' and 'transformation' not in col:
                score += 3
                
            return score
        
        # Choose column with highest score
        best_column = max(columns, key=score_column)
        return best_column

    def _extract_clean_column_name(self, raw_expression: str, fallback_name: str) -> str:
        """Extract clean column name from expression, preferring alias over raw expression."""
        if not raw_expression:
            return fallback_name or 'unknown'
        
        # Check for AS alias pattern
        if ' AS ' in raw_expression.upper():
            parts = raw_expression.split(' AS ', 1) if ' AS ' in raw_expression else raw_expression.split(' as ', 1)
            if len(parts) > 1:
                return parts[1].strip()
        elif ' as ' in raw_expression:
            parts = raw_expression.split(' as ', 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        # If no alias, try to extract a meaningful name from the expression
        # For simple column references, use the column name
        if '(' not in raw_expression and '.' in raw_expression:
            # Qualified column reference like "u.salary" -> "salary"
            return raw_expression.split('.')[-1].strip()
        elif '(' not in raw_expression:
            # Simple column reference
            return raw_expression.strip()
        
        # For complex expressions without aliases, use fallback
        return fallback_name or raw_expression.strip()

    def _query_has_aggregates(self, sql: str) -> bool:
        """Check if the SQL query contains aggregate functions."""
        if not sql:
            return False
        
        sql_upper = sql.upper()
        
        # Check for aggregate functions
        aggregate_functions = ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'GROUP_CONCAT(']
        has_aggregates = any(func in sql_upper for func in aggregate_functions)
        
        # Check for GROUP BY clause (strong indicator of aggregation)
        has_group_by = 'GROUP BY' in sql_upper
        
        return has_aggregates or has_group_by

    def _is_column_from_table(self, column_name: str, table_name: str, context_info: dict = None) -> bool:
        """Check if a column belongs to a specific table using proper SQL parsing."""
        sql = context_info.get('sql') if context_info else None
        return is_column_from_table(column_name, table_name, sql, self.dialect)

    def _build_alias_to_table_mapping(self, sql: str) -> dict:
        """Build a mapping from table aliases to actual table names by parsing SQL."""
        return build_alias_to_table_mapping(sql, self.dialect)

    def _is_aggregate_function_for_table(self, column_expr: str, table_name: str, sql: str = None) -> bool:
        """Check if an aggregate function expression is relevant to a specific table."""
        if not column_expr or not table_name:
            return False
        
        # Handle aggregate functions like COUNT(*), AVG(u.salary), SUM(users.amount)
        column_expr_lower = column_expr.lower()
        
        # Check if the expression contains explicit table references first
        if table_name.lower() in column_expr_lower:
            return True
        
        # Check for table aliases dynamically by analyzing the SQL if SQL is provided
        if sql and self._column_expression_belongs_to_table(column_expr, table_name, sql):
            return True
        
        # COUNT(*) is only relevant to the main grouped table
        # Only assign COUNT(*) to the table that appears in GROUP BY
        if sql and 'count(*)' in column_expr_lower and self._is_main_aggregating_table(sql, table_name):
            return True
        
        return False

    def _column_expression_belongs_to_table(self, column_expr: str, table_name: str, sql: str) -> bool:
        """Check if a column expression belongs to a specific table by analyzing aliases in SQL."""
        try:
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping
            alias_to_table = {}
            tables = list(parsed.find_all(sqlglot.exp.Table))
            for table in tables:
                if table.alias:
                    alias_to_table[table.alias] = table.name
                    
            # Find aliases for this table_name
            table_aliases = []
            for alias, actual_table in alias_to_table.items():
                if actual_table == table_name:
                    table_aliases.append(alias)
            
            # Check if any of the table aliases appear in the column expression
            column_expr_lower = column_expr.lower()
            for alias in table_aliases:
                if f'{alias}.' in column_expr_lower:
                    return True
                    
            return False
            
        except Exception:
            # If parsing fails, fallback to simple heuristics
            # Check if table name starts with same letter as expression prefix
            if '.' in column_expr:
                prefix = column_expr.split('.')[0].lower()
                return table_name.lower().startswith(prefix)
            return False
