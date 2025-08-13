"""Metadata utility functions."""

from typing import Dict, List, Any, Optional, Set
from .sql_parsing_utils import extract_alias_from_expression, extract_function_type, clean_source_expression
from .regex_patterns import is_aggregate_function


def build_column_metadata(column_name: str, column_type: str = "VARCHAR", 
                         upstream: List[str] = None, transformation: Dict = None) -> Dict[str, Any]:
    """Build standardized column metadata structure."""
    if upstream is None:
        upstream = []
        
    metadata = {
        "name": column_name,
        "upstream": upstream,
        "type": column_type
    }
    
    if transformation:
        metadata["transformation"] = transformation
        
    return metadata


def create_metadata_entry(name: str, entity_type: str = "table", 
                         table_columns: List[Dict] = None, **kwargs) -> Dict[str, Any]:
    """Create a standard metadata entry."""
    if table_columns is None:
        table_columns = []
        
    metadata = {
        "table_columns": table_columns,
        "is_cte": kwargs.get("is_cte", False)
    }
    
    # Add additional metadata fields if provided
    if "table_type" in kwargs:
        metadata["table_type"] = kwargs["table_type"]
    if "schema" in kwargs:
        metadata["schema"] = kwargs["schema"]
    if "description" in kwargs:
        metadata["description"] = kwargs["description"]
        
    return metadata


def validate_metadata_structure(metadata: Dict[str, Any]) -> bool:
    """Validate that metadata follows the expected structure."""
    required_fields = ["table_columns"]
    
    for field in required_fields:
        if field not in metadata:
            return False
            
    # Validate table_columns structure
    if not isinstance(metadata["table_columns"], list):
        return False
        
    for column in metadata["table_columns"]:
        if not isinstance(column, dict):
            return False
        if "name" not in column:
            return False
        if "upstream" not in column:
            return False
        if "type" not in column:
            return False
            
    return True


def merge_metadata_entries(base_metadata: Dict[str, Any], 
                          additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two metadata entries, combining their columns."""
    merged = base_metadata.copy()
    
    # Merge table_columns, avoiding duplicates
    base_columns = {col["name"]: col for col in base_metadata.get("table_columns", [])}
    additional_columns = additional_metadata.get("table_columns", [])
    
    for col in additional_columns:
        col_name = col["name"]
        if col_name not in base_columns:
            base_columns[col_name] = col
        else:
            # Merge upstream lists
            existing_upstream = set(base_columns[col_name].get("upstream", []))
            new_upstream = set(col.get("upstream", []))
            base_columns[col_name]["upstream"] = list(existing_upstream.union(new_upstream))
            
            # Prefer transformation info from additional metadata if present
            if "transformation" in col:
                base_columns[col_name]["transformation"] = col["transformation"]
    
    merged["table_columns"] = list(base_columns.values())
    
    # Merge other fields from additional metadata
    for key, value in additional_metadata.items():
        if key != "table_columns" and key not in merged:
            merged[key] = value
            
    return merged


def create_source_column_metadata(column_name: str, upstream: List[str] = None) -> Dict[str, Any]:
    """Create metadata for a SOURCE type column."""
    return build_column_metadata(
        column_name=column_name,
        column_type="SOURCE",
        upstream=upstream or []
    )


def create_result_column_metadata(column_name: str, source_expression: str,
                                transformation_type: str = None, 
                                function_type: str = None) -> Dict[str, Any]:
    """Create metadata for a RESULT type column with transformation info."""
    if function_type is None:
        function_type = extract_function_type(source_expression)
    
    # Auto-detect transformation type if not provided
    if transformation_type is None:
        if is_aggregate_function(source_expression):
            transformation_type = "AGGREGATE"
        elif function_type in ["CASE", "IF", "COALESCE", "NULLIF"]:
            transformation_type = "CASE"
        elif function_type in ["ROW_NUMBER", "RANK", "DENSE_RANK", "LEAD", "LAG", "FIRST_VALUE", "LAST_VALUE"]:
            transformation_type = "WINDOW_FUNCTION"
        else:
            transformation_type = "COMPUTED"
        
    transformation = {
        "column_name": column_name,
        "source_expression": clean_source_expression(source_expression),
        "transformation_type": transformation_type,
        "function_type": function_type,
        "full_expression": clean_source_expression(source_expression)
    }
    
    return build_column_metadata(
        column_name=column_name,
        column_type="VARCHAR",  # Default type
        upstream=[],
        transformation=transformation
    )


def create_direct_column_metadata(column_name: str, upstream: List[str] = None) -> Dict[str, Any]:
    """Create metadata for a DIRECT type column."""
    return build_column_metadata(
        column_name=column_name,
        column_type="DIRECT",
        upstream=upstream or []
    )


def extract_metadata_from_select_columns(select_columns: List[Dict]) -> List[Dict[str, Any]]:
    """Extract metadata from SELECT column information."""
    result_columns = []
    
    for sel_col in select_columns:
        raw_expression = sel_col.get('raw_expression', '')
        column_name = sel_col.get('column_name', raw_expression)
        
        # Create column info with the raw expression as name (preserves table prefixes)
        column_info = {
            "name": raw_expression,  # This preserves table prefixes like "u.name"
            "upstream": [],
            "type": "DIRECT"
        }
        result_columns.append(column_info)
    
    return result_columns


def deduplicate_columns(columns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate columns from a list, keeping the most complete version."""
    seen_columns = {}
    
    for col in columns:
        col_name = col["name"]
        if col_name not in seen_columns:
            seen_columns[col_name] = col
        else:
            # Keep the column with more complete information
            existing = seen_columns[col_name]
            
            # Prefer columns with transformations
            if "transformation" in col and "transformation" not in existing:
                seen_columns[col_name] = col
            # Merge upstream information
            elif "upstream" in col and "upstream" in existing:
                existing_upstream = set(existing.get("upstream", []))
                new_upstream = set(col.get("upstream", []))
                seen_columns[col_name]["upstream"] = list(existing_upstream.union(new_upstream))
    
    return list(seen_columns.values())


def filter_columns_by_type(columns: List[Dict[str, Any]], column_types: Set[str]) -> List[Dict[str, Any]]:
    """Filter columns by their type."""
    return [col for col in columns if col.get("type") in column_types]


def add_missing_source_columns(existing_columns: List[Dict[str, Any]], 
                              referenced_columns: Set[str]) -> List[Dict[str, Any]]:
    """Add missing SOURCE columns that are referenced but not in existing columns."""
    existing_names = {col["name"] for col in existing_columns}
    all_columns = existing_columns.copy()
    
    for column_name in referenced_columns:
        if column_name and column_name not in existing_names:
            source_col = create_source_column_metadata(column_name)
            all_columns.append(source_col)
            existing_names.add(column_name)
    
    return all_columns


def create_cte_metadata(cte_columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create metadata specifically for CTE entities."""
    return create_metadata_entry(
        name="cte",
        entity_type="cte", 
        table_columns=cte_columns,
        is_cte=True
    )


def create_table_metadata(table_columns: List[Dict[str, Any]], 
                         table_type: str = "TABLE",
                         schema: str = "default",
                         description: str = None) -> Dict[str, Any]:
    """Create metadata for table entities."""
    return create_metadata_entry(
        name="table",
        entity_type="table",
        table_columns=table_columns,
        table_type=table_type,
        schema=schema,
        description=description,
        is_cte=False
    )