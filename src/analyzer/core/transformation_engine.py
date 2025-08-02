"""Transformation processing engine for SQL lineage analysis."""

import re
from typing import Dict, List, Any, Optional, Set
import sqlglot
from ..utils.sql_parsing_utils import build_alias_to_table_mapping, extract_function_type, clean_source_expression
from ..utils.column_extraction_utils import extract_all_referenced_columns, extract_aggregate_columns
from ..utils.regex_patterns import is_aggregate_function
from ..utils.metadata_utils import (
    create_source_column_metadata, create_result_column_metadata, 
    merge_metadata_entries, add_missing_source_columns
)


class TransformationEngine:
    """Engine for processing SQL transformations and integrating column metadata."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize the transformation engine."""
        self.dialect = dialect
    
    def integrate_column_transformations(self, chains: Dict, sql: str = None, main_analyzer=None) -> None:
        """
        Integrate column transformations into column metadata throughout the chain.
        Extracted and consolidated from lineage_chain_builder.py.
        """
        if not sql or not main_analyzer:
            return
        
        try:
            # Get the analysis result to access column lineage
            result = main_analyzer.analyze(sql)
            column_lineage_data = result.column_lineage.downstream  # For downstream chains
            
            # Parse SQL to get select columns and alias mapping
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping from sqlglot parsing
            alias_to_table = build_alias_to_table_mapping(sql, self.dialect)
            
            # Get select columns from parsing
            select_columns = self._extract_select_columns(parsed)
            
            # Process each entity in chains
            for entity_name, entity_data in chains.items():
                if entity_data.get('entity_type') != 'table' or entity_data.get('depth', 0) != 0:
                    continue  # Only process top-level table entities
                    
                self._process_entity_transformations(entity_data, entity_name, sql, main_analyzer)
                
        except Exception:
            # If column integration fails, continue without column updates
            pass
    
    def _extract_select_columns(self, parsed_sql) -> List[Dict]:
        """Extract SELECT columns from parsed SQL."""
        select_columns = []
        select_stmt = parsed_sql if isinstance(parsed_sql, sqlglot.exp.Select) else parsed_sql.find(sqlglot.exp.Select)
        
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
        
        return select_columns
    
    def _process_entity_transformations(self, entity_data: Dict, entity_name: str, sql: str, main_analyzer) -> None:
        """Process transformations for a specific entity."""
        # For source tables (depth 0), we only want to add actual source columns referenced in the SQL
        # Don't add QUERY_RESULT columns, aggregate result columns, or upstream relationships
        metadata = entity_data.get('metadata', {})
        table_columns = []
        columns_added = set()
        
        # Only extract source columns that are actually referenced in the SQL
        referenced_columns = extract_all_referenced_columns(sql, entity_name, self.dialect)
        
        # Add all referenced columns as SOURCE type with empty upstream arrays
        for column_name in referenced_columns:
            if column_name and column_name not in columns_added:
                column_info = create_source_column_metadata(column_name)
                table_columns.append(column_info)
                columns_added.add(column_name)
        
        # Update metadata with proper field ordering
        if table_columns:
            if entity_data.get('depth') == 0:
                # Maintain proper metadata field ordering for source tables
                ordered_metadata = self._create_ordered_metadata(metadata, entity_name, main_analyzer)
                ordered_metadata["table_columns"] = table_columns
                metadata = ordered_metadata
            else:
                metadata['table_columns'] = table_columns
        else:
            # Ensure basic metadata fields for source tables even without columns
            if entity_data.get('depth') == 0:
                metadata = self._ensure_basic_metadata(metadata, entity_name, main_analyzer)
        
        entity_data['metadata'] = metadata
        
        # Ensure the updated table_columns are properly set in the entity metadata
        if table_columns:
            entity_data['metadata']['table_columns'] = table_columns
    
    def _create_ordered_metadata(self, metadata: Dict, entity_name: str, main_analyzer) -> Dict:
        """Create ordered metadata for source tables."""
        ordered_metadata = {}
        ordered_metadata["table_type"] = metadata.get("table_type", "TABLE")
        ordered_metadata["schema"] = metadata.get("schema", "default")
        
        # Get description from metadata registry if available
        if (main_analyzer and hasattr(main_analyzer, 'metadata_registry') 
            and main_analyzer.metadata_registry):
            table_metadata = main_analyzer.metadata_registry.get_table_metadata(entity_name)
            if table_metadata and 'description' in table_metadata:
                ordered_metadata["description"] = table_metadata['description']
            else:
                ordered_metadata["description"] = metadata.get("description", "Table information")
        else:
            ordered_metadata["description"] = metadata.get("description", "Table information")
        
        return ordered_metadata
    
    def _ensure_basic_metadata(self, metadata: Dict, entity_name: str, main_analyzer) -> Dict:
        """Ensure basic metadata fields for source tables."""
        metadata.setdefault("table_type", "TABLE")
        metadata.setdefault("schema", "default")
        
        # Get description from metadata registry if available
        if (main_analyzer and hasattr(main_analyzer, 'metadata_registry') 
            and main_analyzer.metadata_registry):
            table_metadata = main_analyzer.metadata_registry.get_table_metadata(entity_name)
            if table_metadata and 'description' in table_metadata:
                metadata.setdefault("description", table_metadata['description'])
            else:
                metadata.setdefault("description", "Table information")
        else:
            metadata.setdefault("description", "Table information")
        
        return metadata
    
    def process_aggregate_transformations(self, sql: str, table_name: str) -> List[Dict]:
        """
        Process aggregate transformations (SUM, COUNT, AVG, etc.) for a specific table.
        Consolidated from lineage_chain_builder.py.
        """
        return extract_aggregate_columns(sql, table_name, self.dialect)
    
    def handle_window_functions(self, sql: str, table_name: str) -> List[Dict]:
        """Handle window functions like ROW_NUMBER, RANK, etc."""
        window_columns = []
        
        try:
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Build alias to table mapping
            alias_to_table = build_alias_to_table_mapping(sql, self.dialect)
            
            # Find table aliases for this table_name
            table_aliases = []
            for alias, actual_table in alias_to_table.items():
                if actual_table == table_name:
                    table_aliases.append(alias)
            
            # If no alias found, use the table name itself
            if not table_aliases:
                table_aliases = [table_name]
            
            select_stmt = parsed if isinstance(parsed, sqlglot.exp.Select) else parsed.find(sqlglot.exp.Select)
            if select_stmt:
                for expr in select_stmt.expressions:
                    # Look for window functions
                    raw_expr = str(expr)
                    if any(func in raw_expr.upper() for func in ['ROW_NUMBER()', 'RANK()', 'DENSE_RANK()', 'LEAD(', 'LAG(']):
                        # Extract alias if present
                        alias = None
                        if hasattr(expr, 'alias') and expr.alias:
                            alias = str(expr.alias)
                        
                        column_name = alias if alias else raw_expr
                        function_type = extract_function_type(raw_expr)
                        
                        window_col = create_result_column_metadata(
                            column_name=column_name,
                            source_expression=raw_expr,
                            transformation_type="WINDOW",
                            function_type=function_type
                        )
                        window_columns.append(window_col)
                        
        except Exception:
            # If parsing fails, return empty list
            pass
            
        return window_columns
    
    def extract_filter_transformations(self, sql: str) -> List[Dict]:
        """
        Extract filter transformations from WHERE clause.
        Consolidated from transformation_analyzer.py.
        """
        transformations = []
        
        # Simple extraction of WHERE conditions
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Simple pattern matching for common conditions
            # Pattern: column operator value
            condition_patterns = [
                r'(\w+)\s*(>|<|>=|<=|=|!=)\s*([^\s]+)',
                r'(\w+\.\w+)\s*(>|<|>=|<=|=|!=)\s*([^\s]+)'
            ]
            
            filter_conditions = []
            for pattern in condition_patterns:
                matches = re.findall(pattern, where_clause, re.IGNORECASE)
                for match in matches:
                    column, operator, value = match
                    # Clean up the value (remove quotes)
                    clean_value = value.strip().strip("'").strip('"')
                    
                    filter_conditions.append({
                        "column": column.split('.')[-1],  # Remove table prefix
                        "operator": operator,
                        "value": clean_value
                    })
            
            if filter_conditions:
                # Try to infer source and target tables from SQL
                from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                source_table = from_match.group(1) if from_match else "unknown"
                
                transformation = {
                    "type": "table_transformation",
                    "source_table": source_table,
                    "target_table": "QUERY_RESULT",
                    "filter_conditions": filter_conditions
                }
                transformations.append(transformation)
        
        return transformations
    
    def extract_ctas_column_transformations(self, sql: str, source_table: str, target_table: str) -> List[Dict]:
        """
        Extract column transformations from CTAS queries.
        Consolidated from ctas_analyzer.py.
        """
        column_transformations = []
        
        try:
            # Parse CTAS query
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            
            # Find the SELECT part of the CTAS
            select_stmt = None
            if hasattr(parsed, 'expression') and isinstance(parsed.expression, sqlglot.exp.Select):
                select_stmt = parsed.expression
            else:
                select_stmt = parsed.find(sqlglot.exp.Select)
            
            if select_stmt:
                for expr in select_stmt.expressions:
                    if isinstance(expr, sqlglot.exp.Column):
                        # Direct column reference
                        column_name = str(expr.name)
                        transformation = {
                            "source_column": f"{source_table}.{column_name}",
                            "target_column": f"{target_table}.{column_name}",
                            "transformation_type": "DIRECT"
                        }
                        column_transformations.append(transformation)
                    else:
                        # Expression or function
                        raw_expr = str(expr)
                        alias = str(expr.alias) if hasattr(expr, 'alias') and expr.alias else raw_expr
                        
                        function_type = extract_function_type(raw_expr)
                        
                        # Determine transformation type based on function type
                        if is_aggregate_function(raw_expr):
                            transformation_type = "AGGREGATE"
                        elif function_type in ["CASE", "IF", "COALESCE", "NULLIF"]:
                            transformation_type = "CASE"
                        elif function_type in ["ROW_NUMBER", "RANK", "DENSE_RANK", "LEAD", "LAG", "FIRST_VALUE", "LAST_VALUE"]:
                            transformation_type = "WINDOW_FUNCTION"
                        else:
                            transformation_type = "COMPUTED"
                        
                        transformation = {
                            "source_expression": raw_expr,
                            "target_column": f"{target_table}.{alias}",
                            "transformation_type": transformation_type,
                            "function_type": function_type
                        }
                        column_transformations.append(transformation)
                        
        except Exception:
            # If parsing fails, return empty list
            pass
            
        return column_transformations
    
    def build_transformation_metadata(self, transformations: List[Dict], entity_name: str, sql: str) -> Dict:
        """Build comprehensive transformation metadata for an entity."""
        metadata = {
            "transformations": transformations,
            "source_sql": sql,
            "entity": entity_name
        }
        
        # Categorize transformations
        filter_transformations = [t for t in transformations if t.get("type") == "FILTER"]
        join_transformations = [t for t in transformations if t.get("type") == "JOIN"]
        aggregate_transformations = [t for t in transformations if t.get("type") == "AGGREGATE"]
        
        if filter_transformations:
            metadata["has_filters"] = True
            metadata["filter_count"] = len(filter_transformations)
        
        if join_transformations:
            metadata["has_joins"] = True
            metadata["join_count"] = len(join_transformations)
        
        if aggregate_transformations:
            metadata["has_aggregations"] = True
            metadata["aggregate_count"] = len(aggregate_transformations)
        
        return metadata