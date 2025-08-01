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
            result.column_lineage = SimpleNamespace()
            result.column_lineage.upstream = column_lineage.upstream
            result.column_lineage.downstream = column_lineage.downstream
            result.dialect = self.dialect
            result.errors = []
            result.warnings = []
            result.metadata = {}
        except Exception as e:
            # Create empty result on error
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = {}
            result.table_lineage.downstream = {}
            result.column_lineage = SimpleNamespace() 
            result.column_lineage.upstream = {}
            result.column_lineage.downstream = {}
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
            result.column_lineage = SimpleNamespace()
            result.column_lineage.upstream = column_lineage.upstream
            result.column_lineage.downstream = column_lineage.downstream
            result.dialect = self.dialect
            result.errors = []
            result.warnings = []
            result.metadata = {}
        except Exception as e:
            # Create empty result on error
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = {}
            result.table_lineage.downstream = {}
            result.column_lineage = SimpleNamespace() 
            result.column_lineage.upstream = {}
            result.column_lineage.downstream = {}
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
            result.column_lineage = SimpleNamespace()
            result.column_lineage.upstream = column_lineage.upstream
            result.column_lineage.downstream = column_lineage.downstream
            result.dialect = self.dialect
            result.errors = []
            result.warnings = []
            result.metadata = {}
        except Exception as e:
            # Create empty result on error
            from types import SimpleNamespace
            result = SimpleNamespace()
            result.table_lineage = SimpleNamespace()
            result.table_lineage.upstream = {}
            result.table_lineage.downstream = {}
            result.column_lineage = SimpleNamespace() 
            result.column_lineage.upstream = {}
            result.column_lineage.downstream = {}
            result.dialect = self.dialect
            result.errors = [f"Parsing failed: {str(e)}"]
            result.warnings = []
            result.metadata = {}
        
        # Get table lineage data for the appropriate direction
        table_lineage_data = result.table_lineage.upstream if chain_type == "upstream" else result.table_lineage.downstream
        
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
        
        # Calculate actual max depth
        actual_max_depth = self._calculate_max_depth(chains)
        
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
                "total_columns": len(set(result.column_lineage.upstream.keys()) | set().union(*result.column_lineage.upstream.values()) if result.column_lineage.upstream else set()),
                "has_transformations": False,  # TODO: Add transformation detection
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