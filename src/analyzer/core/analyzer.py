"""Main SQL lineage analyzer."""

from typing import Optional, Dict, Any
import sqlglot
from sqlglot import Expression

from .models import LineageResult, TableMetadata
from .extractor import LineageExtractor
from ..metadata.registry import MetadataRegistry
from ..utils.validation import validate_sql_input


class SQLLineageAnalyzer:
    """Main SQL lineage analyzer class."""
    
    def __init__(self, dialect: str = "trino"):
        """
        Initialize the SQL lineage analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
        """
        self.dialect = dialect
        self.metadata_registry = MetadataRegistry()
        self.extractor = LineageExtractor()
    
    def analyze(self, sql: str, **kwargs) -> LineageResult:
        """
        Analyze SQL query for lineage information.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options (for future extensibility)
            
        Returns:
            LineageResult containing table and column lineage information
        """
        # Validate input
        validation_error = validate_sql_input(sql)
        if validation_error:
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[validation_error]
            )
        
        try:
            # Parse SQL
            expression = self._parse_sql(sql)
            
            # Extract lineage
            table_lineage = self.extractor.extract_table_lineage(expression)
            column_lineage = self.extractor.extract_column_lineage(expression)
            
            # Get metadata for involved tables
            metadata = self._collect_metadata(table_lineage)
            
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=table_lineage,
                column_lineage=column_lineage,
                metadata=metadata
            )
            
        except Exception as e:
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"Failed to analyze SQL: {str(e)}"]
            )
    
    def analyze_file(self, file_path: str, **kwargs) -> LineageResult:
        """
        Analyze SQL from a file.
        
        Args:
            file_path: Path to SQL file
            **kwargs: Additional options
            
        Returns:
            LineageResult containing analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql = f.read()
            return self.analyze(sql, **kwargs)
        except Exception as e:
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"Failed to read file {file_path}: {str(e)}"]
            )
    
    def analyze_multiple(self, queries: list[str], **kwargs) -> list[LineageResult]:
        """
        Analyze multiple SQL queries.
        
        Args:
            queries: List of SQL query strings
            **kwargs: Additional options
            
        Returns:
            List of LineageResult objects
        """
        return [self.analyze(query, **kwargs) for query in queries]
    
    def _parse_sql(self, sql: str) -> Expression:
        """Parse SQL string into SQLGlot expression."""
        try:
            return sqlglot.parse_one(sql, dialect=self.dialect)
        except Exception as e:
            raise ValueError(f"Failed to parse SQL query: {str(e)}")
    
    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """Collect metadata for tables involved in lineage."""
        metadata = {}
        
        # Collect all table names from upstream and downstream
        all_tables = set()
        for target, sources in table_lineage.upstream.items():
            all_tables.add(target)
            all_tables.update(sources)
        
        # Get metadata for each table
        for table_name in all_tables:
            table_metadata = self.metadata_registry.get_table_metadata(table_name)
            if table_metadata:
                metadata[table_name] = table_metadata
        
        return metadata
    
    def set_metadata_registry(self, metadata_registry: MetadataRegistry) -> None:
        """
        Set the metadata registry for the analyzer.
        
        Args:
            metadata_registry: MetadataRegistry instance to use
        """
        self.metadata_registry = metadata_registry
        self.extractor = LineageExtractor()
    
    def add_metadata_provider(self, provider) -> None:
        """Add a metadata provider to the registry."""
        self.metadata_registry.add_provider(provider)
    
    def set_dialect(self, dialect: str) -> None:
        """Set the SQL dialect for parsing."""
        self.dialect = dialect
    
    def get_lineage_result(self, sql: str, **kwargs) -> LineageResult:
        """
        Get the LineageResult object for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            LineageResult object containing table and column lineage information
        """
        return self.analyze(sql, **kwargs)
    
    def get_lineage_json(self, sql: str, **kwargs) -> str:
        """
        Get the JSON representation of lineage analysis for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the lineage analysis
        """
        import json
        result = self.analyze(sql, **kwargs)
        return json.dumps(result.to_dict(), indent=2)
    
    def get_table_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the table lineage chain for a SQL query with specified direction and depth.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 1)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        if depth < 1:
            raise ValueError("depth must be at least 1")
        
        result = self.analyze(sql, **kwargs)
        
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
        import json
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
        
        result = self.analyze(sql, **kwargs)
        
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
        import json
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
        
        result = self.analyze(sql, **kwargs)
        
        # Get both table and column lineage data
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Helper function to get table name from column reference
        def extract_table_from_column(column_ref: str) -> str:
            if '.' in column_ref:
                parts = column_ref.split('.')
                return '.'.join(parts[:-1])  # Everything except the last part (column name)
            return "unknown_table"
        
        # Helper function to get column name from column reference
        def extract_column_from_ref(column_ref: str) -> str:
            if '.' in column_ref:
                return column_ref.split('.')[-1]
            return column_ref
        
        # Build comprehensive lineage chain
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, visited_in_path: set = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we have a circular dependency
            if entity_name in visited_in_path:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "transformations": [],
                    "metadata": {}
                }
            
            # Stop if we've exceeded max depth (only when depth > 0, meaning limited depth)
            if depth > 0 and current_depth > depth:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "transformations": [],
                    "metadata": {}
                }
            
            # Add current entity to the path to prevent cycles
            visited_in_path = visited_in_path | {entity_name}
            
            dependencies = []
            transformations = []
            metadata = {}
            
            if entity_type == "table":
                # Handle table-level lineage
                if entity_name in table_lineage_data:
                    for dependent_table in table_lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dependent_table, "table", current_depth + 1, visited_in_path)
                        dependencies.append(dep_chain)
                
                # Get table transformations
                if entity_name in result.table_lineage.transformations:
                    for transformation in result.table_lineage.transformations[entity_name]:
                        transformations.append({
                            "type": "table_transformation",
                            "source_table": transformation.source_table,
                            "target_table": transformation.target_table,
                            "join_type": transformation.join_type.value if transformation.join_type else None,
                            "join_conditions": [
                                {
                                    "left_column": jc.left_column,
                                    "operator": jc.operator.value,
                                    "right_column": jc.right_column
                                }
                                for jc in transformation.join_conditions
                            ],
                            "filter_conditions": [
                                {
                                    "column": fc.column,
                                    "operator": fc.operator.value,
                                    "value": fc.value
                                }
                                for fc in transformation.filter_conditions
                            ],
                            "group_by_columns": transformation.group_by_columns,
                            "having_conditions": [
                                {
                                    "column": hc.column,
                                    "operator": hc.operator.value,
                                    "value": hc.value
                                }
                                for hc in transformation.having_conditions
                            ],
                            "order_by_columns": transformation.order_by_columns
                        })
                
                # Get table metadata
                if entity_name in result.metadata:
                    table_meta = result.metadata[entity_name]
                    metadata = {
                        "catalog": table_meta.catalog,
                        "schema": table_meta.schema,
                        "table": table_meta.table,
                        "table_type": table_meta.table_type.value,
                        "description": table_meta.description,
                        "owner": table_meta.owner,
                        "created_date": table_meta.created_date.isoformat() if table_meta.created_date else None,
                        "row_count": table_meta.row_count,
                        "storage_format": table_meta.storage_format,
                        "columns": [
                            {
                                "name": col.name,
                                "data_type": col.data_type,
                                "nullable": col.nullable,
                                "primary_key": col.primary_key,
                                "foreign_key": col.foreign_key,
                                "description": col.description,
                                "default_value": col.default_value
                            }
                            for col in table_meta.columns
                        ]
                    }
                
                # Add column-level information for this table
                table_columns = []
                for column_ref in column_lineage_data.keys():
                    if extract_table_from_column(column_ref) == entity_name:
                        column_name = extract_column_from_ref(column_ref)
                        column_info = {
                            "column_name": column_name,
                            "column_ref": column_ref,
                            "upstream_columns": list(column_lineage_data.get(column_ref, set())),
                            "transformations": []
                        }
                        
                        # Get column transformations
                        if column_ref in result.column_lineage.transformations:
                            for col_transformation in result.column_lineage.transformations[column_ref]:
                                col_trans_data = {
                                    "type": "column_transformation",
                                    "source_column": col_transformation.source_column,
                                    "target_column": col_transformation.target_column,
                                    "expression": col_transformation.expression
                                }
                                
                                if col_transformation.aggregate_function:
                                    col_trans_data["aggregate_function"] = {
                                        "function_type": col_transformation.aggregate_function.function_type.value,
                                        "column": col_transformation.aggregate_function.column,
                                        "distinct": col_transformation.aggregate_function.distinct
                                    }
                                
                                if col_transformation.window_function:
                                    col_trans_data["window_function"] = {
                                        "function_name": col_transformation.window_function.function_name,
                                        "partition_by": col_transformation.window_function.partition_by,
                                        "order_by": col_transformation.window_function.order_by
                                    }
                                
                                if col_transformation.case_expression:
                                    col_trans_data["case_expression"] = {
                                        "when_conditions": [
                                            {
                                                "column": wc.column,
                                                "operator": wc.operator.value,
                                                "value": wc.value
                                            }
                                            for wc in col_transformation.case_expression.when_conditions
                                        ],
                                        "then_values": col_transformation.case_expression.then_values,
                                        "else_value": col_transformation.case_expression.else_value
                                    }
                                
                                column_info["transformations"].append(col_trans_data)
                        
                        table_columns.append(column_info)
                
                metadata["table_columns"] = table_columns
            
            elif entity_type == "column":
                # Handle column-level lineage
                if entity_name in column_lineage_data:
                    for dependent_column in column_lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dependent_column, "column", current_depth + 1, visited_in_path)
                        dependencies.append(dep_chain)
                
                # Get column transformations
                if entity_name in result.column_lineage.transformations:
                    for col_transformation in result.column_lineage.transformations[entity_name]:
                        trans_data = {
                            "type": "column_transformation",
                            "source_column": col_transformation.source_column,
                            "target_column": col_transformation.target_column,
                            "expression": col_transformation.expression
                        }
                        
                        if col_transformation.aggregate_function:
                            trans_data["aggregate_function"] = {
                                "function_type": col_transformation.aggregate_function.function_type.value,
                                "column": col_transformation.aggregate_function.column,
                                "distinct": col_transformation.aggregate_function.distinct
                            }
                        
                        if col_transformation.window_function:
                            trans_data["window_function"] = {
                                "function_name": col_transformation.window_function.function_name,
                                "partition_by": col_transformation.window_function.partition_by,
                                "order_by": col_transformation.window_function.order_by
                            }
                        
                        if col_transformation.case_expression:
                            trans_data["case_expression"] = {
                                "when_conditions": [
                                    {
                                        "column": wc.column,
                                        "operator": wc.operator.value,
                                        "value": wc.value
                                    }
                                    for wc in col_transformation.case_expression.when_conditions
                                ],
                                "then_values": col_transformation.case_expression.then_values,
                                "else_value": col_transformation.case_expression.else_value
                            }
                        
                        transformations.append(trans_data)
                
                # Get column metadata from parent table
                parent_table = extract_table_from_column(entity_name)
                if parent_table in result.metadata:
                    table_meta = result.metadata[parent_table]
                    column_name = extract_column_from_ref(entity_name)
                    for col in table_meta.columns:
                        if col.name == column_name:
                            metadata = {
                                "parent_table": parent_table,
                                "column_name": col.name,
                                "data_type": col.data_type,
                                "nullable": col.nullable,
                                "primary_key": col.primary_key,
                                "foreign_key": col.foreign_key,
                                "description": col.description,
                                "default_value": col.default_value
                            }
                            break
            
            return {
                "entity": entity_name,
                "entity_type": entity_type,
                "depth": current_depth - 1,
                "dependencies": dependencies,
                "transformations": transformations,
                "metadata": metadata
            }
        
        # Build chains starting from the target entity or all entities
        chains = {}
        
        if target_entity:
            # Focus on specific entity
            if target_entity in table_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "table", 1)
            elif target_entity in column_lineage_data:
                chains[target_entity] = build_comprehensive_chain(target_entity, "column", 1)
            else:
                # Try to find it as a partial match
                found = False
                for table_name in table_lineage_data.keys():
                    if target_entity in table_name:
                        chains[table_name] = build_comprehensive_chain(table_name, "table", 1)
                        found = True
                        break
                
                if not found:
                    for column_ref in column_lineage_data.keys():
                        if target_entity in column_ref:
                            chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1)
                            break
        else:
            # Build chains for all tables
            for table_name in table_lineage_data.keys():
                chains[table_name] = build_comprehensive_chain(table_name, "table", 1)
            
            # Build chains for all columns (only if no table chains exist to avoid redundancy)
            if not chains:
                for column_ref in column_lineage_data.keys():
                    chains[column_ref] = build_comprehensive_chain(column_ref, "column", 1)
        
        # Calculate actual max depth achieved
        actual_max_depth = 0
        for chain_data in chains.values():
            if "depth" in chain_data:
                actual_max_depth = max(actual_max_depth, chain_data["depth"])
            # Also check nested dependencies for max depth
            def get_max_depth_from_chain(chain_obj, current_max=0):
                max_depth = current_max
                if isinstance(chain_obj, dict):
                    if "depth" in chain_obj:
                        max_depth = max(max_depth, chain_obj["depth"])
                    if "dependencies" in chain_obj:
                        for dep in chain_obj["dependencies"]:
                            max_depth = max(max_depth, get_max_depth_from_chain(dep, max_depth))
                return max_depth
            
            actual_max_depth = max(actual_max_depth, get_max_depth_from_chain(chain_data))
        
        # Calculate actually used columns from transformations and lineage
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
                "has_transformations": bool(result.table_lineage.transformations or result.column_lineage.transformations),
                "has_metadata": bool(result.metadata),
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
        import json
        chain_data = self.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2)