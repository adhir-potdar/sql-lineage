"""Lineage chain builder for table and column dependencies."""

from typing import Dict, Any, Optional, List, Set
import json
from .base_analyzer import BaseAnalyzer


class ChainBuilder(BaseAnalyzer):
    """Builder for creating table and column lineage chains."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize chain builder."""
        super().__init__(dialect)
    
    def get_table_lineage_chain(self, analyzer_result, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the table lineage chain with specified direction and depth.
        
        Args:
            analyzer_result: Result from main analyzer analyze() method
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
        
        # Get the appropriate lineage direction
        lineage_data = analyzer_result.table_lineage.upstream if chain_type == "upstream" else analyzer_result.table_lineage.downstream
        
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
            "sql": analyzer_result.sql,
            "dialect": analyzer_result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": analyzer_result.errors,
            "warnings": analyzer_result.warnings
        }
    
    def get_table_lineage_chain_json(self, analyzer_result, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get table lineage chain as JSON string."""
        chain_data = self.get_table_lineage_chain(analyzer_result, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2, default=str)
    
    def get_column_lineage_chain(self, analyzer_result, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """
        Get the column lineage chain for a SQL query with specified direction and depth.
        
        Args:
            analyzer_result: Result from main analyzer analyze() method
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
        
        # Get the appropriate lineage direction
        column_lineage = analyzer_result.column_lineage
        lineage_data = column_lineage.upstream if chain_type == "upstream" else column_lineage.downstream
        
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
            "sql": analyzer_result.sql,
            "dialect": analyzer_result.dialect,
            "chain_type": chain_type,
            "max_depth": depth,
            "chains": chain,
            "errors": analyzer_result.errors,
            "warnings": analyzer_result.warnings
        }
    
    def get_column_lineage_chain_json(self, analyzer_result, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get column lineage chain as JSON string."""
        chain_data = self.get_column_lineage_chain(analyzer_result, chain_type, depth, **kwargs)
        return json.dumps(chain_data, indent=2, default=str)
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None) -> None:
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
                        source_tables = trans.get('source_tables', [])
                        for source_table in source_tables:
                            if source_table.get('table_name') == entity_name:
                                columns_used = source_table.get('columns_used', [])
                                source_columns.update(columns_used)
                
                # Add missing columns to table_columns if they don't exist
                existing_column_names = {col.get('name') for col in table_columns}
                for col_name in source_columns:
                    if col_name and col_name not in existing_column_names:
                        table_columns.append({
                            'name': col_name,
                            'data_type': 'UNKNOWN',
                            'source': 'INFERRED_FROM_USAGE',
                            'is_nullable': True
                        })
                        existing_column_names.add(col_name)
                
                # Update metadata
                metadata['table_columns'] = table_columns
                entity_data['metadata'] = metadata
    
    def get_lineage_chain(self, analyzer_result, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get comprehensive lineage chain (both table and column) for a SQL query.
        
        Args:
            analyzer_result: Result from main analyzer analyze() method
            chain_type: Direction of chain - "upstream" or "downstream"  
            depth: Maximum depth of the chain (0 = all levels)
            target_entity: Specific table/column to trace (None = all entities)
            **kwargs: Additional options
            
        Returns:
            Dictionary containing comprehensive lineage chain information
        """
        if chain_type not in ["upstream", "downstream"]:
            raise ValueError("chain_type must be 'upstream' or 'downstream'")
        
        sql = analyzer_result.query
        
        # Build comprehensive chain combining table and column lineage
        chain_data = {
            'sql': sql,
            'dialect': analyzer_result.dialect,
            'chain_type': chain_type,
            'max_depth': depth if depth > 0 else "unlimited",
            'target_entity': target_entity,
            'entities': {},
            'summary': {
                'total_tables': 0,
                'total_columns': 0,
                'max_depth_reached': 0
            }
        }
        
        def build_comprehensive_chain(entity_name: str, entity_type: str, current_depth: int, visited_in_path: Set = None, parent_entity: str = None) -> Dict[str, Any]:
            if visited_in_path is None:
                visited_in_path = set()
            
            entity_key = f"{entity_type}:{entity_name}"
            
            # Stop if we've exceeded max depth or have circular dependency
            if (depth > 0 and current_depth > depth) or entity_key in visited_in_path:
                return {
                    'entity_name': entity_name,
                    'entity_type': entity_type,
                    'depth': current_depth - 1,
                    'circular_reference': entity_key in visited_in_path,
                    'dependencies': []
                }
            
            # Add to visited path
            visited_in_path = visited_in_path | {entity_key}
            
            entity_info = {
                'entity_name': entity_name,
                'entity_type': entity_type,
                'depth': current_depth - 1,
                'parent_entity': parent_entity,
                'dependencies': [],
                'metadata': {}
            }
            
            # Get appropriate lineage data based on entity type
            if entity_type == 'table':
                lineage_data = analyzer_result.table_lineage.upstream if chain_type == "upstream" else analyzer_result.table_lineage.downstream
                if entity_name in lineage_data:
                    for dep_table in lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dep_table, 'table', current_depth + 1, visited_in_path, entity_name)
                        entity_info['dependencies'].append(dep_chain)
            elif entity_type == 'column':
                lineage_data = analyzer_result.column_lineage.upstream if chain_type == "upstream" else analyzer_result.column_lineage.downstream
                if entity_name in lineage_data:
                    for dep_column in lineage_data[entity_name]:
                        dep_chain = build_comprehensive_chain(dep_column, 'column', current_depth + 1, visited_in_path, entity_name)
                        entity_info['dependencies'].append(dep_chain)
            
            # Update summary stats
            if entity_type == 'table':
                chain_data['summary']['total_tables'] += 1
            elif entity_type == 'column':
                chain_data['summary']['total_columns'] += 1
            
            chain_data['summary']['max_depth_reached'] = max(
                chain_data['summary']['max_depth_reached'], 
                current_depth - 1
            )
            
            return entity_info
        
        # Build chains for all entities or target entity
        if target_entity:
            # Determine entity type and build chain for specific entity
            if '.' in target_entity:
                entity_info = build_comprehensive_chain(target_entity, 'column', 1)
            else:
                entity_info = build_comprehensive_chain(target_entity, 'table', 1)
            chain_data['entities'][target_entity] = entity_info
        else:
            # Build chains for all tables
            table_lineage = analyzer_result.table_lineage.upstream if chain_type == "upstream" else analyzer_result.table_lineage.downstream
            for table_name in table_lineage.keys():
                entity_info = build_comprehensive_chain(table_name, 'table', 1)
                chain_data['entities'][table_name] = entity_info
            
            # Build chains for all columns
            column_lineage = analyzer_result.column_lineage.upstream if chain_type == "upstream" else analyzer_result.column_lineage.downstream
            for column_name in column_lineage.keys():
                entity_info = build_comprehensive_chain(column_name, 'column', 1)
                chain_data['entities'][column_name] = entity_info
        
        return chain_data
    
    def get_lineage_chain_json(self, analyzer_result, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """Get comprehensive lineage chain as JSON string."""
        chain_data = self.get_lineage_chain(analyzer_result, chain_type, depth, target_entity, **kwargs)
        return json.dumps(chain_data, indent=2, default=str)