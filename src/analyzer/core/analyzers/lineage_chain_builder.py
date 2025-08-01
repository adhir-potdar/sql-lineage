"""Lineage chain builder analyzer."""

from typing import Dict, Any, List, Optional, Set
import json
import re
import sqlglot
from sqlglot import expressions as exp
from .base_analyzer import BaseAnalyzer


class LineageChainBuilder(BaseAnalyzer):
    """Analyzer for building lineage chains."""
    
    def __init__(self, dialect: str = "trino", main_analyzer=None):
        """Initialize lineage chain builder with optional reference to main analyzer."""
        super().__init__(dialect)
        self.main_analyzer = main_analyzer
    
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
                                        "join_type": trans.join_type.value if hasattr(trans, 'join_type') and trans.join_type else "INNER",
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
                                
                                # Add filter conditions if present
                                if hasattr(trans, 'filter_conditions') and trans.filter_conditions:
                                    filter_conditions = []
                                    for fc in trans.filter_conditions:
                                        filter_conditions.append({
                                            "column": fc.column,
                                            "operator": fc.operator.value if hasattr(fc.operator, 'value') else str(fc.operator),
                                            "value": fc.value
                                        })
                                    trans_data["filter_conditions"] = filter_conditions
                                
                                transformations.append(trans_data)
                        
                        # Filter transformations to only include those relevant to this entity
                        relevant_transformations = []
                        for trans in transformations:
                            if (trans.get("source_table") == entity_name and 
                                trans.get("target_table") == dependent_table):
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
        self._add_missing_source_columns(chains, sql)
        
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
    
    def _add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
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
        except:
            select_columns = []
            alias_to_table = {}
        
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Handle QUERY_RESULT dependencies specially
            for dep in entity_data.get('dependencies', []):
                if dep.get('entity') == 'QUERY_RESULT':
                    # Add qualified column names to QUERY_RESULT metadata
                    table_columns = []
                    # Use same logic as original analyzer-bkup.py
                    inferred_columns = self._infer_query_result_columns_simple(sql, select_columns)
                    has_table_prefixes = any('.' in col.get('name', '') for col in inferred_columns)
                    
                    if has_table_prefixes:
                        # JOIN query: Use qualified names from select columns
                        for sel_col in select_columns:
                            source_table = sel_col.get('source_table')
                            if (source_table in alias_to_table and 
                                alias_to_table[source_table] == entity_name):
                                raw_expression = sel_col.get('raw_expression')
                                column_name = sel_col.get('column_name')
                                upstream_col = f"{entity_name}.{column_name}"
                                
                                column_info = {
                                    "name": raw_expression,
                                    "upstream": [upstream_col],
                                    "type": "DIRECT"
                                }
                                table_columns.append(column_info)
                    else:
                        # Simple query: Add columns from select statement to QUERY_RESULT
                        for sel_col in select_columns:
                            if sel_col.get('source_table') is None:  # Unprefixed columns
                                raw_expression = sel_col.get('raw_expression')
                                column_name = sel_col.get('column_name')
                                
                                column_info = {
                                    "name": raw_expression,
                                    "upstream": [f"QUERY_RESULT.{column_name}"],
                                    "type": "DIRECT"
                                }
                                table_columns.append(column_info)
                    
                    if table_columns:
                        if 'metadata' not in dep:
                            dep['metadata'] = {}
                        dep['metadata']['table_columns'] = table_columns
                
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
        """Integrate column transformations into column metadata throughout the chain - extracted from analyzer-bkup.py."""
        if not sql or not self.main_analyzer:
            return
        
        try:
            # Get the analysis result to access column lineage
            result = self.main_analyzer.analyze(sql)
            column_lineage_data = result.column_lineage.downstream  # For downstream chains
            
            # Helper functions extracted from analyzer-bkup.py
            def extract_table_from_column(column_ref: str) -> str:
                if '.' in column_ref:
                    parts = column_ref.split('.')
                    return '.'.join(parts[:-1])  # Everything except the last part (column name)
                return "unknown_table"
            
            def extract_column_from_ref(column_ref: str) -> str:
                if '.' in column_ref:
                    return column_ref.split('.')[-1]
                return column_ref
            
            # Parse SQL to get select columns and alias mapping
            import sqlglot
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping from sqlglot parsing
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
            
            # Process each entity in chains
            for entity_name, entity_data in chains.items():
                if entity_data.get('entity_type') != 'table':
                    continue
                    
                # Update table columns with proper qualified names and upstream relationships
                metadata = entity_data.get('metadata', {})
                table_columns = []
                columns_added = set()
                
                # Infer query result columns to check if they have table prefixes (like original analyzer-bkup.py)
                inferred_columns = self._infer_query_result_columns_simple(sql, select_columns)
                has_table_prefixes = any('.' in col.get('name', '') for col in inferred_columns)
                
                if has_table_prefixes:
                    # JOIN query logic: Add columns from select statement that belong to this entity
                    for sel_col in select_columns:
                        source_table = sel_col.get('source_table')
                        if (source_table in alias_to_table and 
                            alias_to_table[source_table] == entity_name):
                            raw_expression = sel_col.get('raw_expression')
                            column_name = sel_col.get('column_name')
                            
                            if raw_expression and raw_expression not in columns_added:
                                column_info = {
                                    "name": column_name,  # Use unqualified name for source table
                                    "upstream": [f"QUERY_RESULT.{column_name}"],  # Point to QUERY_RESULT
                                    "type": "DIRECT"
                                }
                                table_columns.append(column_info)
                                columns_added.add(raw_expression)
                    
                    # Add JOIN columns with SOURCE type (extract from JOIN conditions)
                    join_columns = self._extract_join_columns_from_sql(sql, entity_name)
                    for join_col in join_columns:
                        if join_col and join_col not in columns_added:
                            column_info = {
                                "name": join_col,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            table_columns.append(column_info)
                            columns_added.add(join_col)
                else:
                    # Simple query logic: Add all referenced columns as SOURCE columns with empty upstream
                    # This includes SELECT columns and WHERE clause columns
                    referenced_columns = set()
                    
                    # Add SELECT columns
                    for sel_col in select_columns:
                        if sel_col.get('source_table') is None:  # Unprefixed columns
                            column_name = sel_col.get('column_name')
                            if column_name:
                                referenced_columns.add(column_name)
                    
                    # Add columns from WHERE clause (extract from SQL)
                    where_columns = self._extract_table_columns_from_sql(sql, entity_name)
                    referenced_columns.update(where_columns)
                    
                    # Add all referenced columns as SOURCE type
                    for column_name in referenced_columns:
                        if column_name and column_name not in columns_added:
                            column_info = {
                                "name": column_name,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            table_columns.append(column_info)
                            columns_added.add(column_name)
                
                # Update metadata
                if table_columns:
                    metadata['table_columns'] = table_columns
                    entity_data['metadata'] = metadata
                
        except Exception:
            # If column integration fails, continue without column updates
            pass

    def _add_columns_for_entity(self, entity_data: Dict, transformations: List, sql: str):
        """Add column information to an entity based on transformations - extracted from analyzer-bkup.py."""
        entity_name = entity_data.get('entity')
        entity_type = entity_data.get('entity_type', 'table')
        if not entity_name:
            return
            
        metadata = entity_data.get('metadata', {})
        table_columns = metadata.get('table_columns', [])
        
        # Extract column lineage data from main analyzer for context
        try:
            if self.main_analyzer:
                parsed = sqlglot.parse_one(sql, dialect=self.dialect)
                column_lineage = self.extractor.extract_column_lineage(parsed)
                column_lineage_data = column_lineage.upstream
            else:
                column_lineage_data = {}
        except Exception:
            column_lineage_data = {}
        
        # Logic extracted from analyzer-bkup.py lines 1353-1396
        if not table_columns and entity_type == "table" and entity_name != "QUERY_RESULT":
            source_columns = set()
            
            # Get selected columns from column lineage (SELECT clause columns)
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
            metadata.update({
                "table_type": "TABLE",
                "schema": "default", 
                "description": "User profile information",
                "table_columns": table_columns
            })
        
        # Special handling for QUERY_RESULT - infer result columns from SQL parsing (lines 1372-1392)
        elif entity_name == "QUERY_RESULT" and not table_columns:
            # For QUERY_RESULT, we should infer columns from the SELECT statement
            all_query_result_columns = self._infer_query_result_columns(sql, column_lineage_data)
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
        
        # Update the entity metadata  
        if 'metadata' not in entity_data:
            entity_data['metadata'] = {}
        entity_data['metadata'].update(metadata)
        
        # Process dependencies recursively
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            self._add_columns_for_entity(dep, transformations, sql)

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
        Infer QUERY_RESULT columns from SQL query - extracted from analyzer-bkup.py lines 1690-1740.
        """
        import re
        
        result_columns = []
        
        # Try to extract SELECT columns from SQL
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1).strip()
            
            # Handle simple cases like "SELECT name, email FROM users"
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
        # For simple query "SELECT name, email FROM users WHERE age > 25"
        # We want to extract: name, email, age for the "users" table
        
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
        import re
        
        join_columns = set()
        
        # Extract the ON clause
        on_match = re.search(r'ON\s+(.+?)(?:\s*$|\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT)', sql, re.IGNORECASE)
        if on_match:
            on_clause = on_match.group(1)
            
            # Extract column references from ON clause
            # Look for patterns like "u.id = o.user_id"
            column_refs = re.findall(r'(\w+)\.(\w+)', on_clause)
            
            for table_alias, column_name in column_refs:
                # Check if this column belongs to our table
                if table_alias == table_name.lower()[:1]:  # Simple heuristic: u -> users, o -> orders
                    join_columns.add(column_name)
                elif table_name.lower().startswith(table_alias):
                    join_columns.add(column_name)
        
        return join_columns

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