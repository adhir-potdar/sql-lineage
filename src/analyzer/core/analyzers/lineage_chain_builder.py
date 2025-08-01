"""Lineage chain builder analyzer."""

from typing import Dict, Any, List, Optional, Set
import json
from .base_analyzer import BaseAnalyzer


class LineageChainBuilder(BaseAnalyzer):
    """Analyzer for building lineage chains."""
    
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
        
        result = self.extractor.analyze(sql, **kwargs)
        
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
        
        result = self.extractor.analyze(sql, **kwargs)
        
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
    
    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get the comprehensive lineage chain combining table and column lineage.
        
        Args:
            sql: SQL query string to analyze
            chain_type: Direction of chain - "upstream" or "downstream"
            depth: Maximum depth of the chain (default: 0 for unlimited)
            target_entity: Specific entity to start chain from
            **kwargs: Additional options
            
        Returns:
            Dictionary containing the comprehensive lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        # For comprehensive lineage, we need to handle CTE queries specially
        sql_upper = sql.strip().upper()
        if 'WITH' in sql_upper:
            return self._build_cte_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        
        # For non-CTE queries, use the standard lineage building
        result = self.extractor.analyze(sql, **kwargs)
        
        # Get both table and column lineage
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        column_lineage_data = result.column_lineage.upstream if chain_type == "upstream" else result.column_lineage.downstream
        
        # Build comprehensive chain with metadata
        chains = {}
        
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, visited_in_path: set = None, parent_entity: str = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            # Stop if we've exceeded max depth (0 means unlimited) or circular dependency
            if (depth > 0 and current_depth > depth) or entity_name in visited_in_path:
                return {
                    "entity": entity_name,
                    "entity_type": entity_type,
                    "depth": current_depth - 1,
                    "dependencies": [],
                    "metadata": {}
                }
            
            # Add current entity to path
            visited_in_path = visited_in_path | {entity_name}
            
            dependencies = []
            metadata = {"table_columns": []}
            
            # Process table dependencies
            if entity_type == "TABLE" and entity_name in table_lineage_data:
                for dependent_table in table_lineage_data[entity_name]:
                    dep_chain = build_comprehensive_chain(dependent_table, "TABLE", current_depth + 1, visited_in_path, entity_name)
                    dependencies.append(dep_chain)
            
            # Process column dependencies
            if entity_type == "COLUMN" and entity_name in column_lineage_data:
                for dependent_column in column_lineage_data[entity_name]:
                    dep_chain = build_comprehensive_chain(dependent_column, "COLUMN", current_depth + 1, visited_in_path, entity_name)
                    dependencies.append(dep_chain)
            
            return {
                "entity": entity_name,
                "entity_type": entity_type,
                "depth": current_depth - 1,
                "dependencies": dependencies,
                "metadata": metadata
            }
        
        # Build chains for all entities
        if target_entity:
            # Build chain for specific entity
            entity_type = "TABLE" if target_entity in table_lineage_data else "COLUMN"
            chains[target_entity] = build_comprehensive_chain(target_entity, entity_type, 1)
        else:
            # Build chains for all tables
            for table_name in table_lineage_data.keys():
                chains[table_name] = build_comprehensive_chain(table_name, "TABLE", 1)
        
        return {
            "sql": sql,
            "dialect": result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "target_entity": target_entity,
            "chains": chains,
            "errors": result.errors,
            "warnings": result.warnings
        }
    
    def get_table_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get the JSON representation of table lineage chain for a SQL query."""
        chain_data = self.get_table_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def get_column_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get the JSON representation of column lineage chain for a SQL query."""
        chain_data = self.get_column_lineage_chain(sql, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def get_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """Get the JSON representation of comprehensive lineage chain for a SQL query."""
        chain_data = self.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2)
    
    def _build_cte_lineage_chain(self, sql: str, chain_type: str, depth: int, target_entity: Optional[str], **kwargs) -> Dict[str, Any]:
        """Build lineage chain for CTE queries."""
        cte_lineage = self.cte_parser.get_cte_lineage_chain(sql)
        
        # Basic CTE chain structure
        chains = {}
        
        # Get CTE data
        ctes = cte_lineage.get('ctes', {})
        execution_order = cte_lineage.get('execution_order', [])
        
        # Build chains for CTEs
        for cte_name in execution_order:
            if cte_name in ctes:
                cte_data = ctes[cte_name]
                chains[cte_name] = {
                    "entity": cte_name,
                    "entity_type": "CTE",
                    "depth": 0,
                    "dependencies": [],
                    "metadata": {
                        "columns": cte_data.get('columns', []),
                        "transformations": cte_data.get('transformations', [])
                    }
                }
        
        return {
            "sql": sql,
            "dialect": self.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "target_entity": target_entity,
            "chains": chains,
            "errors": [],
            "warnings": []
        }
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
        """Add missing source columns to table metadata by extracting from transformations."""
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
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
                                    source_columns.add(col)
                            
                            # Extract join condition columns
                            for join in trans.get('joins', []):
                                for condition in join.get('conditions', []):
                                    for col_key in ['left_column', 'right_column']:
                                        col_ref = condition.get(col_key, '')
                                        if col_ref and entity_name in col_ref:
                                            clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                                            source_columns.add(clean_col)
                
                # Add the columns found in transformations
                if source_columns:
                    existing_columns = {col['name'] for col in table_columns}
                    for col_name in source_columns:
                        if col_name not in existing_columns:
                            table_columns.append({
                                "name": col_name,
                                "upstream": [],
                                "type": "SOURCE"
                            })
                    
                    # Update metadata
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    entity_data['metadata']['table_columns'] = table_columns