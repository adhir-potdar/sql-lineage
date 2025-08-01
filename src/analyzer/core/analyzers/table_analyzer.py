"""Table lineage analyzer."""

from typing import Dict, Any, List, Optional
import json
from .base_analyzer import BaseAnalyzer


class TableAnalyzer(BaseAnalyzer):
    """Analyzer for table-level lineage chains."""
    
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
        Get the table lineage chain as JSON string.
        
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
    
    def _add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
        """Add missing source columns to table metadata by extracting from transformations."""
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies OR add to existing
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
                                    source_columns.add(col)
                            
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
            
            # For target tables (depth > 0), extract result columns from CTAS SELECT clause
            elif entity_data.get('depth', 0) > 0:
                # Check if this is a target table created by CTAS
                transformations = entity_data.get('transformations', [])
                for trans in transformations:
                    if trans.get('target_table') == entity_name and trans.get('source_table'):
                        # This is a CTAS target table, extract SELECT clause columns
                        target_columns = set()
                        
                        # Extract columns from GROUP BY (these will be in result)
                        group_by_columns = trans.get('group_by_columns', [])
                        for col in group_by_columns:
                            clean_col = col.split('.')[-1] if '.' in col else col
                            target_columns.add(clean_col)
                        
                        # Extract actual result columns from CTAS SELECT clause parsing
                        if sql and sql.strip().upper().startswith('CREATE TABLE'):
                            try:
                                # Get CTAS lineage to get actual output columns
                                ctas_lineage = self.ctas_parser.get_ctas_lineage(sql)
                                column_lineage = ctas_lineage.get('column_lineage', {})
                                output_columns = column_lineage.get('output_columns', [])
                                
                                # Add all result columns from the CTAS SELECT clause
                                for output_col in output_columns:
                                    col_name = output_col.get('alias') or output_col.get('name')
                                    if col_name:
                                        clean_col = col_name.split('.')[-1] if '.' in col_name else col_name
                                        target_columns.add(clean_col)
                            except Exception:
                                # Fallback: just use GROUP BY columns if CTAS parsing fails
                                pass
                        
                        # Get column transformations for this entity
                        column_transformations_map = {}
                        if sql:
                            try:
                                column_transformations = self._extract_column_transformations(
                                    sql, trans.get('source_table'), entity_name
                                )
                                # Create map of column_name -> transformation
                                for col_trans in column_transformations:
                                    col_name = col_trans.get('column_name')
                                    if col_name:
                                        column_transformations_map[col_name] = col_trans
                            except Exception:
                                pass
                        
                        # Add result columns to target table metadata with transformations
                        if target_columns:
                            existing_columns = {col['name'] for col in table_columns}
                            
                            for column_name in target_columns:
                                if column_name not in existing_columns:
                                    column_info = {
                                        "name": column_name,
                                        "upstream": [],
                                        "type": "RESULT" if column_name not in group_by_columns else "SOURCE"
                                    }
                                    
                                    # Add transformation info if available
                                    if column_name in column_transformations_map:
                                        col_trans = column_transformations_map[column_name]
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
    
    def _extract_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """Extract column transformations using transformation analyzer."""
        from .transformation_analyzer import TransformationAnalyzer
        transformer = TransformationAnalyzer(self.dialect)
        transformer.set_metadata_registry(self.metadata_registry)
        return transformer.extract_column_transformations(sql, source_table, target_table)