"""SQL Lineage Visualizer using Graphviz for creating lineage chain diagrams."""

import json
from typing import Dict, Any, Optional, List, Tuple
from graphviz import Digraph
import os

# Constants for entity names
QUERY_RESULT_ENTITY = 'QUERY_RESULT'
FINAL_RESULT_ENTITY = 'FINAL_RESULT'

# Constants for join types
DEFAULT_JOIN_TYPE = 'INNER JOIN'
JOIN_TYPES = {
    'INNER': 'INNER JOIN',
    'LEFT': 'LEFT JOIN', 
    'RIGHT': 'RIGHT JOIN',
    'FULL': 'FULL JOIN',
    'CROSS': 'CROSS JOIN'
}

# Constants for SQL functions
SQL_FUNCTIONS = {
    'STRING_FUNCTIONS': ['UPPER', 'LOWER', 'CONCAT'],
    'AGGREGATE_FUNCTIONS': ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN'],
    'CASE_FUNCTION': 'CASE',
    'MATH_OPERATORS': ['*', '+', '-', '/'],
    'CONCAT_OPERATORS': ['||']
}

# Constants for data types and column types
DEFAULT_DATA_TYPE = 'VARCHAR'
COLUMN_TYPES = {
    'DIRECT': 'DIRECT',
    'COMPUTED': 'COMPUTED',
    'SELECT': 'SELECT'
}

# Constants for operators
SQL_OPERATORS = {
    'IN': 'IN',
    'EXISTS': 'EXISTS', 
    'NOT_EXISTS': 'NOT EXISTS'
}

# Constants for transformation types
TRANSFORMATION_TYPES = {
    'TABLE_TRANSFORMATION': 'TABLE_TRANSFORMATION',
    'TRANSFORM': 'TRANSFORM',
    'FILTER': 'FILTER',
    'GROUP_BY': 'GROUP_BY',
    'CASE': 'CASE',
    'COMPUTED': 'COMPUTED',
    'AGGREGATE': 'AGGREGATE',
    'UNION': 'UNION',
    'UNION_ALL': 'UNION_ALL'
}


# Constants for unknown/default values
UNKNOWN_VALUE = 'unknown'
UNKNOWN_FUNCTION = 'UNKNOWN'

# Constants for layout directions
LAYOUT_DIRECTIONS = {
    'HORIZONTAL_UPSTREAM': 'RL',
    'HORIZONTAL_DOWNSTREAM': 'LR', 
    'VERTICAL_UPSTREAM': 'BT',
    'VERTICAL_DOWNSTREAM': 'TB'
}


class SQLLineageVisualizer:
    """
    Creates visual representations of SQL lineage chains using Graphviz.
    Supports both table and column lineage with direction-specific flow.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.default_config = {
            'node_style': {
                'table': {
                    'shape': 'box',
                    'style': 'filled,rounded',
                    'fillcolor': '#E3F2FD',
                    'color': '#1976D2',
                    'fontname': 'Arial',
                    'fontsize': '12',
                    'fontcolor': '#1976D2'
                },
                'column': {
                    'shape': 'ellipse',
                    'style': 'filled',
                    'fillcolor': '#F3E5F5',
                    'color': '#7B1FA2',
                    'fontname': 'Arial',
                    'fontsize': '10',
                    'fontcolor': '#7B1FA2'
                },
                'cte': {
                    'shape': 'diamond',
                    'style': 'filled',
                    'fillcolor': '#FFF3E0',
                    'color': '#F57C00',
                    'fontname': 'Arial',
                    'fontsize': '11',
                    'fontcolor': '#F57C00'
                },
                'query_result': {
                    'shape': 'doublecircle',
                    'style': 'filled,bold',
                    'fillcolor': '#E8F5E8',
                    'color': '#2E7D32',
                    'fontname': 'Arial Bold',
                    'fontsize': '12',
                    'fontcolor': '#1B5E20'
                }
            },
            'edge_style': {
                'table': {
                    'color': '#1976D2',
                    'penwidth': '2',
                    'arrowhead': 'vee'
                },
                'column': {
                    'color': '#7B1FA2',
                    'penwidth': '1',
                    'arrowhead': 'normal',
                    'style': 'dashed'
                }
            },
            'graph_attributes': {
                'size': '16,10',
                'dpi': '300',
                'bgcolor': 'white',
                'pad': '0.5',
                'nodesep': '0.5',
                'ranksep': '1.5',
                'compound': 'true'  # Allow edges between clusters
            }
        }
    
    def _format_sql_for_display(self, sql_query: str) -> str:
        """
        Format SQL query for display in diagram label.
        
        Args:
            sql_query: Raw SQL query string
            
        Returns:
            Formatted SQL query string suitable for display
        """
        if not sql_query:
            return ""
        
        # Clean the query
        lines = sql_query.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive whitespace but preserve indentation
            stripped_line = line.rstrip()
            if stripped_line:  # Skip empty lines
                cleaned_lines.append(stripped_line)
        
        # Join lines for display - show full query without truncation
        cleaned_query = '\n'.join(cleaned_lines)
        
        # Escape special characters for Graphviz
        cleaned_query = cleaned_query.replace('"', '\\"')
        cleaned_query = cleaned_query.replace('{', '\\{')
        cleaned_query = cleaned_query.replace('}', '\\}')
        cleaned_query = cleaned_query.replace('<', '\\<')
        cleaned_query = cleaned_query.replace('>', '\\>')
        
        return cleaned_query
    
    def create_lineage_diagram(self, 
                             table_chain_json: str,
                             column_chain_json: Optional[str] = None,
                             output_path: str = "lineage_diagram",
                             output_format: str = "png",
                             config: Optional[Dict] = None,
                             show_columns: bool = True,
                             layout: str = "horizontal",
                             sql_query: Optional[str] = None) -> str:
        """
        Create a lineage diagram from JSON chain data.
        
        Args:
            table_chain_json: JSON string of table lineage chain
            column_chain_json: Optional JSON string of column lineage chain
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
            config: Custom configuration dictionary
            show_columns: Whether to include column lineage in diagram
            layout: Layout direction ('horizontal' or 'vertical', default: 'horizontal')
            sql_query: Optional SQL query text to display at the top of the diagram
            
        Returns:
            Path to generated diagram file
        """
        # Parse JSON data
        table_chain = json.loads(table_chain_json)
        column_chain = json.loads(column_chain_json) if column_chain_json and show_columns else None
        
        # Determine flow direction based on chain type and layout
        chain_type = table_chain.get('chain_type', 'upstream')
        if layout == "horizontal":
            # Horizontal layout: upstream (RL), downstream (LR)
            rankdir = LAYOUT_DIRECTIONS['HORIZONTAL_UPSTREAM'] if chain_type == 'upstream' else LAYOUT_DIRECTIONS['HORIZONTAL_DOWNSTREAM']
        else:
            # Vertical layout: upstream (BT), downstream (TB)
            rankdir = LAYOUT_DIRECTIONS['VERTICAL_UPSTREAM'] if chain_type == 'upstream' else LAYOUT_DIRECTIONS['VERTICAL_DOWNSTREAM']
        
        # Create Graphviz digraph
        dot = self._create_digraph(table_chain, rankdir, config, sql_query)
        
        # Determine if we have column lineage to integrate
        has_column_lineage = column_chain and show_columns
        
        # Add table nodes and edges
        self._add_table_lineage(dot, table_chain, chain_type, config, has_column_lineage)
        
        # Add integrated column lineage if provided
        if has_column_lineage:
            column_chain_type = column_chain.get('chain_type', chain_type)
            self._add_column_lineage(dot, column_chain, column_chain_type, config, table_chain)
        
        # Render the diagram
        output_file = dot.render(output_path, format=output_format, cleanup=True)
        return output_file
    
    def create_table_only_diagram(self,
                                table_chain_json: str,
                                output_path: str = "table_lineage",
                                output_format: str = "png",
                                config: Optional[Dict] = None,
                                layout: str = "horizontal",
                                sql_query: Optional[str] = None) -> str:
        """
        Create a table-only lineage diagram.
        
        Args:
            table_chain_json: JSON string of table lineage chain
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
            config: Custom configuration dictionary
            layout: Layout direction ('horizontal' or 'vertical', default: 'horizontal')
            sql_query: Optional SQL query text to display at the top of the diagram
            
        Returns:
            Path to generated diagram file
        """
        return self.create_lineage_diagram(
            table_chain_json=table_chain_json,
            column_chain_json=None,
            output_path=output_path,
            output_format=output_format,
            config=config,
            show_columns=False,
            layout=layout,
            sql_query=sql_query
        )
    
    def create_column_focused_diagram(self,
                                    table_chain_json: str,
                                    column_chain_json: str,
                                    output_path: str = "column_lineage",
                                    output_format: str = "png",
                                    config: Optional[Dict] = None,
                                    layout: str = "horizontal",
                                    sql_query: Optional[str] = None) -> str:
        """
        Create a column-focused lineage diagram with tables as containers.
        
        Args:
            table_chain_json: JSON string of table lineage chain
            column_chain_json: JSON string of column lineage chain
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
            config: Custom configuration dictionary
            layout: Layout direction ('horizontal' or 'vertical', default: 'horizontal')
            sql_query: Optional SQL query text to display at the top of the diagram
            
        Returns:
            Path to generated diagram file
        """
        return self.create_lineage_diagram(
            table_chain_json=table_chain_json,
            column_chain_json=column_chain_json,
            output_path=output_path,
            output_format=output_format,
            config=config,
            show_columns=True,
            layout=layout,
            sql_query=sql_query
        )
    
    def create_lineage_chain_diagram(self, 
                                   lineage_chain_json: str,
                                   output_path: str = "lineage_chain_diagram",
                                   output_format: str = "png",
                                   config: Optional[Dict] = None,
                                   layout: str = "horizontal") -> str:
        """
        Create a lineage diagram from comprehensive lineage chain JSON.
        
        Args:
            lineage_chain_json: JSON string from get_lineage_chain_json function
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
            config: Custom configuration dictionary
            layout: Layout direction ('horizontal' or 'vertical', default: 'horizontal')
            
        Returns:
            Path to generated diagram file
        """
        # Parse JSON data
        chain_data = json.loads(lineage_chain_json)
        
        # Extract SQL query for title
        sql_query = chain_data.get('sql', '')
        chain_type = chain_data.get('chain_type', 'downstream')
        chains = chain_data.get('chains', {})
        
        # Determine flow direction based on chain type and layout
        if layout == "horizontal":
            # Horizontal layout: upstream (RL), downstream (LR)
            rankdir = LAYOUT_DIRECTIONS['HORIZONTAL_UPSTREAM'] if chain_type == 'upstream' else LAYOUT_DIRECTIONS['HORIZONTAL_DOWNSTREAM']
        else:
            # Vertical layout: upstream (BT), downstream (TB)
            rankdir = LAYOUT_DIRECTIONS['VERTICAL_UPSTREAM'] if chain_type == 'upstream' else LAYOUT_DIRECTIONS['VERTICAL_DOWNSTREAM']
        
        # Create Graphviz digraph
        dot = self._create_lineage_chain_digraph(chain_data, rankdir, config, sql_query)
        
        # Add nodes and edges from lineage chain
        self._add_lineage_chain_elements(dot, chain_data, config)
        
        # Render the diagram
        output_file = dot.render(output_path, format=output_format, cleanup=True)
        return output_file
    
    def _create_lineage_chain_digraph(self, chain_data: Dict, rankdir: str, config: Optional[Dict], sql_query: Optional[str] = None) -> Digraph:
        """Create and configure the Graphviz digraph for lineage chain."""
        # Merge configuration
        graph_config = self.default_config.copy()
        if config:
            self._merge_config(graph_config, config)
        
        # Create digraph
        dot = Digraph(comment=f"SQL Lineage Chain - {chain_data.get('chain_type', UNKNOWN_VALUE).title()}")
        
        # Set graph attributes
        dot.attr(rankdir=rankdir)
        
        # Apply graph attributes
        graph_attrs = graph_config['graph_attributes']
        dot.graph_attr.update(graph_attrs)
        
        # Add title with SQL query
        title = f"SQL Lineage Chain ({chain_data.get('chain_type', UNKNOWN_VALUE).title()})"
        if chain_data.get('actual_max_depth'):
            title += f" - Depth: {chain_data['actual_max_depth']}"
        
        # Include SQL query if provided
        if sql_query:
            # Clean and format the SQL query
            cleaned_query = self._format_sql_for_display(sql_query)
            full_label = f"SQL Query:\n{cleaned_query}\n\n{title}"
        else:
            full_label = title
        
        dot.graph_attr.update({
            'label': full_label,
            'labelloc': 't',
            'fontsize': '12',
            'fontname': 'Arial'
        })
        
        return dot
    

    def _add_lineage_chain_elements(self, dot: Digraph, chain_data: Dict, config: Optional[Dict] = None) -> None:
        """Add nodes and edges from lineage chain data."""
        chains = chain_data.get('chains', {})
        node_config = self.default_config['node_style']
        edge_config = self.default_config['edge_style']['table']
        
        if config and 'node_style' in config:
            node_config.update(config['node_style'])
        if config and 'edge_style' in config and 'table' in config['edge_style']:
            edge_config.update(config['edge_style']['table'])
        
        # Check if this is a CTAS query to handle entity display differently
        sql = chain_data.get('sql', '')
        is_ctas = sql.strip().upper().startswith('CREATE TABLE')
        
        # First, deduplicate QUERY_RESULT nodes and merge their data
        deduplicated_chains = self._deduplicate_query_result_nodes(chains)
        
        # Recursively process all entities in the deduplicated chains
        def add_entities_recursive(entity_name: str, entity_data: Dict, processed_entities: set = None):
            """Recursively add entities and their dependencies, avoiding duplicates."""
            if processed_entities is None:
                processed_entities = set()
            
            # Skip if already processed (avoid duplicate nodes)
            if entity_name in processed_entities:
                return
                
            if entity_name != '_sql':  # Skip temp SQL context
                processed_entities.add(entity_name)
                # Add the current entity (pass SQL context for column analysis)
                entity_data_with_sql = {**entity_data, '_sql_context': sql}
                self._add_entity_with_columns_improved(dot, entity_name, entity_data_with_sql, node_config, edge_config)
                
                # Recursively add dependencies
                dependencies = entity_data.get('dependencies', [])
                for dep in dependencies:
                    dep_entity = dep.get('entity')
                    if dep_entity:
                        add_entities_recursive(dep_entity, dep, processed_entities)
        
        # Process each top-level entity in the deduplicated chains
        processed_entities = set()
        for entity_name, entity_data in deduplicated_chains.items():
            add_entities_recursive(entity_name, entity_data, processed_entities)
        
        # For CTAS queries, ensure target entities are included for transformations
        if is_ctas:
            self._add_missing_ctas_target_entities(dot, chains, node_config, edge_config, chain_data)
        
        # Add transformation boxes and edges using deduplicated chains
        self._add_transformation_boxes(dot, deduplicated_chains, node_config, edge_config, chain_data)
        
        # Add connections between entities using deduplicated chains (with subquery handling)
        self._add_entity_connections(dot, deduplicated_chains, edge_config)
        
        # Add result box at the end of the flow (only if QUERY_RESULT not already processed)
        # Check if QUERY_RESULT exists in the deduplicated chains
        has_query_result = any(
            self._has_query_result_in_chain(entity_data) 
            for entity_data in deduplicated_chains.values()
        )
        
        if not has_query_result:
            self._add_final_result_box(dot, chain_data, node_config)
    
    def _add_entity_with_columns_improved(self, dot: Digraph, entity_name: str, entity_data: Dict, node_config: Dict, edge_config: Dict) -> None:
        """Add an entity node with only the columns used in query."""
        entity_type = entity_data.get('entity_type', 'table')
        metadata = entity_data.get('metadata', {})
        
        # Build the node label based on entity type
        if entity_type == 'cte':
            # CTE entity styling
            label_parts = [f"**{entity_name}**"]
            label_parts.append("(CTE)")
        elif entity_type == 'query_result':
            # Query result entity styling
            label_parts = [f"**{entity_name}**"]
            label_parts.append("(FINAL RESULT)")
        else:  # entity_type == 'table' or default
            # Regular table entity styling
            label_parts = [f"**{entity_name}**"]
        
        # Common column processing for all entity types
        table_columns = metadata.get('table_columns', [])
        used_columns = []
        
        # 1. Extract from table_columns metadata (columns listed as used in output)
        if table_columns:
            for col_info in table_columns:
                col_name = col_info.get('name', col_info.get('column_name', UNKNOWN_VALUE))
                col_type = col_info.get('type', COLUMN_TYPES['DIRECT'])
                
                # For CTE entities, show column type information
                if entity_type == 'cte':
                    if col_type == COLUMN_TYPES['COMPUTED']:
                        used_columns.append(f"{col_name} (computed)")
                    else:
                        used_columns.append(col_name)
                else:
                    used_columns.append(col_name)
        
        # 2. Extract from transformations to capture JOIN columns, filter columns, etc.
        if entity_type != 'query_result':  # Skip for query result to avoid duplication
            transformation_columns = self._extract_columns_from_transformations(entity_name, entity_data)
            used_columns.extend(transformation_columns)
            
            # 3. Extract from dependencies/transformations to capture additional usage patterns
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                # Extract from dependency transformations
                dep_transformations = dep.get('transformations', [])
                for trans in dep_transformations:
                    source_table = trans.get('source_table', '')
                    
                    # If this entity is the source, extract all columns used in this transformation
                    if source_table == entity_name:
                        # Extract from JOIN conditions
                        joins = trans.get('joins', [])
                        for join in joins:
                            for condition in join.get('conditions', []):
                                if isinstance(condition, dict):
                                    left_col = condition.get('left_column', '')
                                    right_col = condition.get('right_column', '')
                                
                                # Extract column names, handling table prefixes generically
                                # Check if column reference belongs to this entity (table.column or alias.column)
                                if left_col:
                                    col_parts = left_col.split('.')
                                    if len(col_parts) == 2:
                                        table_ref, col_name = col_parts
                                        # Match by exact entity name or check if it's a table alias pattern
                                        if (table_ref == entity_name or 
                                            table_ref.lower() == entity_name.lower() or
                                            self._is_table_alias_match(table_ref, entity_name)):
                                            if col_name not in used_columns:
                                                used_columns.append(col_name)
                                    elif len(col_parts) == 1 and left_col not in used_columns:
                                        # Column without prefix - might belong to this table
                                        used_columns.append(left_col)
                                
                                if right_col:
                                    col_parts = right_col.split('.')
                                    if len(col_parts) == 2:
                                        table_ref, col_name = col_parts
                                        # Match by exact entity name or check if it's a table alias pattern
                                        if (table_ref == entity_name or 
                                            table_ref.lower() == entity_name.lower() or
                                            self._is_table_alias_match(table_ref, entity_name)):
                                            if col_name not in used_columns:
                                                used_columns.append(col_name)
                                    elif len(col_parts) == 1 and right_col not in used_columns:
                                        # Column without prefix - might belong to this table
                                        used_columns.append(right_col)
                        
                        # Extract from filter conditions
                        filter_conditions = trans.get('filter_conditions', [])
                        for condition in filter_conditions:
                            if isinstance(condition, dict):
                                column = condition.get('column', '')
                                if column and column not in used_columns:
                                    clean_col = column.split('.')[-1] if '.' in column else column
                                    used_columns.append(clean_col)
                        
                        # Extract from group by and order by
                        for col_list_key in ['group_by_columns', 'order_by_columns']:
                            col_list = trans.get(col_list_key, [])
                            for col in col_list:
                                clean_col = str(col).split()[0] if col else ''  # Remove DESC/ASC
                                clean_col = clean_col.split('.')[-1] if '.' in clean_col else clean_col
                                if clean_col and clean_col not in used_columns:
                                    used_columns.append(clean_col)
            
            
            # Remove duplicates and sort, filter out QUERY_RESULT columns
            filtered_columns = []
            for col in used_columns:
                col_str = str(col)
                # Filter out QUERY_RESULT columns and other invalid column references
                if not (col_str.startswith('QUERY_RESULT.') or 
                       col_str == QUERY_RESULT_ENTITY or 
                       col_str.startswith(UNKNOWN_VALUE) or
                       col_str == UNKNOWN_VALUE):
                    filtered_columns.append(col)
            
            used_columns = sorted(list(set(filtered_columns)))
            
            # Add used columns with their types from full metadata
            all_columns = metadata.get('columns', [])
            if used_columns:
                label_parts.append("─" * 20)
                label_parts.append("Used Columns:")
                
                # Show all used columns with type information
                for col_name in used_columns:
                    # Find column details from full metadata
                    col_details = next((col for col in all_columns if col.get('name') == col_name), None) if all_columns else None
                    
                    if col_details:
                        col_type = col_details.get('data_type', DEFAULT_DATA_TYPE)
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        safe_col_type = str(col_type).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"{safe_col_name}: {safe_col_type}")
                    else:
                        # No metadata available - show without type
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"{safe_col_name}")
                    
            elif all_columns:
                # For CTEs, try to extract output columns first
                cte_output_columns = []
                if self._is_cte(entity_name):
                    cte_output_columns = self._extract_cte_output_columns(entity_name, entity_data)
                
                # For regular tables, try to infer used columns from transformations
                inferred_columns = []
                if not cte_output_columns:
                    inferred_columns = self._infer_used_columns_from_dependencies(entity_data, all_columns)
                    # Also try to extract columns from direct transformations targeting this entity
                    direct_columns = self._extract_columns_from_transformations(entity_name, entity_data)
                    if direct_columns:
                        inferred_columns = direct_columns
                
                # Determine which columns to show and the appropriate label
                columns_to_show = cte_output_columns or inferred_columns
                if cte_output_columns:
                    label_parts.append("─" * 20)
                    label_parts.append("Output Columns:")
                elif inferred_columns:
                    label_parts.append("─" * 20)
                    label_parts.append("Used Columns:")
                
                if columns_to_show:
                    for col_name in columns_to_show:
                        col_details = next((col for col in all_columns if col.get('name') == col_name), None)
                        if col_details:
                            col_type = col_details.get('data_type', 'unknown')
                            safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                            
                            if col_type != 'unknown':
                                safe_col_type = str(col_type).replace('<', '&lt;').replace('>', '&gt;')
                                label_parts.append(f"{safe_col_name}: {safe_col_type}")
                            else:
                                label_parts.append(f"{safe_col_name}")
                        else:
                            # For CTE output columns, show all columns
                            safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                            label_parts.append(f"{safe_col_name}")
                # DO NOT show columns from metadata registry - only show columns discovered from SQL query
                # The metadata registry should only provide additional details, not discover columns
            
        # Join all parts to create the final label
        full_label = "\\n".join(label_parts)
        
        # Get appropriate node style based on entity type
        node_style = self._get_node_style_by_type(entity_name, entity_type, node_config)
        
        # Add the node
        dot.node(entity_name, full_label, **node_style)
    
    def _create_digraph(self, table_chain: Dict, rankdir: str, config: Optional[Dict], sql_query: Optional[str] = None) -> Digraph:
        """Create and configure the Graphviz digraph."""
        # Merge configuration
        graph_config = self.default_config.copy()
        if config:
            self._merge_config(graph_config, config)
        
        # Create digraph
        dot = Digraph(comment=f"SQL Lineage - {table_chain.get('chain_type', 'unknown').title()}")
        
        # Set graph attributes
        dot.attr(rankdir=rankdir)
        
        # Apply graph attributes using the graph method
        graph_attrs = graph_config['graph_attributes']
        dot.graph_attr.update(graph_attrs)
        
        # Add title with optional SQL query
        title = f"SQL Lineage Chain ({table_chain.get('chain_type', 'unknown').title()})"
        if table_chain.get('max_depth'):
            title += f" - Depth: {table_chain['max_depth']}"
        
        # Include SQL query if provided
        if sql_query:
            # Clean and format the SQL query
            cleaned_query = self._format_sql_for_display(sql_query)
            full_label = f"SQL Query:\n{cleaned_query}\n\n{title}"
        else:
            full_label = title
        
        dot.graph_attr.update({
            'label': full_label,
            'labelloc': 't',
            'fontsize': '12',  # Reduced to accommodate longer text
            'fontname': 'Arial'
        })
        
        return dot
    
    def _add_table_lineage(self, dot: Digraph, table_chain: Dict, chain_type: str, config: Optional[Dict], has_column_lineage: bool = False) -> None:
        """Add table nodes and edges to the diagram."""
        chains = table_chain.get('chains', {})
        node_config = self.default_config['node_style']
        edge_config = self.default_config['edge_style']['table']
        
        if config and 'node_style' in config:
            node_config.update(config['node_style'])
        if config and 'edge_style' in config and 'table' in config['edge_style']:
            edge_config.update(config['edge_style']['table'])
        
        # Add table nodes (only if not using integrated column lineage)
        if not has_column_lineage:
            for table_name, table_info in chains.items():
                node_style = self._get_node_style(table_name, node_config)
                display_name = self._get_display_name(table_name, table_chain)
                dot.node(table_name, display_name, **node_style)
        
        # Add table-to-table edges
        for table_name, table_info in chains.items():
            dependencies = table_info.get('dependencies', [])
            for dep in dependencies:
                dep_table = dep.get('table')
                if dep_table and has_column_lineage:
                    # In integrated mode, we rely on column edges to show relationships
                    # The subgraph layout will handle table positioning
                    pass
                elif dep_table and not has_column_lineage:
                    # Regular table-to-table edges for table-only mode
                    if chain_type == 'upstream':
                        dot.edge(dep_table, table_name, **edge_config)
                    else:
                        dot.edge(table_name, dep_table, **edge_config)
    
    def _add_column_lineage(self, dot: Digraph, column_chain: Dict, chain_type: str, config: Optional[Dict], table_chain: Optional[Dict] = None) -> None:
        """Add integrated column lineage within table structure."""
        chains = column_chain.get('chains', {})
        node_config = self.default_config['node_style']['column']
        edge_config = self.default_config['edge_style']['column']
        
        if config and 'node_style' in config and 'column' in config['node_style']:
            node_config.update(config['node_style']['column'])
        if config and 'edge_style' in config and 'column' in config['edge_style']:
            edge_config.update(config['edge_style']['column'])
        
        # Group columns by their parent tables
        table_columns = self._group_columns_by_table(chains, chain_type, table_chain)
        
        # Create subgraphs for each table containing its columns
        for table_name, columns in table_columns.items():
            if columns:  # Only create subgraph if there are columns
                # Create a cluster (subgraph) for each table
                cluster_name = f"cluster_{table_name}"
                with dot.subgraph(name=cluster_name) as table_subgraph:
                    # Configure subgraph for vertical column layout within horizontal table flow
                    display_name = self._get_display_name(table_name, table_chain)
                    table_subgraph.attr(
                        style='filled,rounded',
                        fillcolor='#F8F9FA',
                        color='#6C757D',
                        label=display_name,
                        fontname='Arial Bold',
                        fontsize='14',
                        labelloc='t',
                        penwidth='2',
                        margin='10'
                    )
                    
                    # Add column nodes within the table subgraph in vertical arrangement
                    for i, column_name in enumerate(sorted(columns)):
                        display_name = self._clean_column_name(column_name)
                        table_subgraph.node(f"col_{column_name}", display_name, **node_config)
                        
                        # Add invisible edges to force vertical stacking within the table
                        if i > 0:
                            prev_column = sorted(columns)[i-1]
                            table_subgraph.edge(f"col_{prev_column}", f"col_{column_name}", 
                                              style='invis', weight='10')
        
        # Add column-to-column edges (these will connect across table boundaries)
        for column_name, column_info in chains.items():
            dependencies = column_info.get('dependencies', [])
            for dep in dependencies:
                dep_column = dep.get('column')
                if dep_column:
                    if chain_type == 'upstream':
                        # Upstream: dependency → current (data flows from source to result)
                        dot.edge(f"col_{dep_column}", f"col_{column_name}", **edge_config)
                    else:  # downstream
                        # Downstream: current → dependency (impact flows from source to affected)
                        dot.edge(f"col_{column_name}", f"col_{dep_column}", **edge_config)
    
    def _group_columns_by_table(self, column_chains: Dict, chain_type: str = "upstream", table_chain: Optional[Dict] = None) -> Dict[str, List[str]]:
        """Group columns by their parent table names."""
        table_columns = {}
        
        for column_name in column_chains.keys():
            # Extract table name from column reference
            table_name = self._extract_table_from_column(column_name, column_chains, chain_type, table_chain)
            
            if table_name not in table_columns:
                table_columns[table_name] = []
            table_columns[table_name].append(column_name)
        
        return table_columns
    
    def _extract_table_from_column(self, column_name: str, column_chains: Dict = None, chain_type: str = "upstream", table_chain: Optional[Dict] = None) -> str:
        """Extract table name from column reference."""
        # Handle different column reference formats:
        # "table.column", "schema.table.column", "QUERY_RESULT.column"
        if '.' in column_name:
            parts = column_name.split('.')
            if len(parts) >= 2:
                # For "QUERY_RESULT.column", return "QUERY_RESULT"
                if parts[0] == QUERY_RESULT_ENTITY:
                    return QUERY_RESULT_ENTITY
                # For "table.column", return "table"  
                elif len(parts) == 2:
                    return parts[0]
                # For "schema.table.column" or "catalog.schema.table.column", return the table part
                else:
                    # Usually the second-to-last part is the table name
                    return parts[-2]
        
        # If no dot, try to infer table from column lineage context
        if column_chains and column_name in column_chains:
            column_info = column_chains[column_name]
            dependencies = column_info.get('dependencies', [])
            
            # For downstream lineage, columns without prefixes are typically from base tables
            # Try to infer actual table name from dependencies and table chain
            if chain_type == "downstream" and dependencies:
                for dep in dependencies:
                    dep_column = dep.get('column', '')
                    if '.' in dep_column:
                        # If dependency has table prefix, trace back to find source table
                        dep_parts = dep_column.split('.')
                        if len(dep_parts) >= 2:
                            dep_table = dep_parts[0]
                            
                            # If dependency is from QUERY_RESULT, trace back to find actual source table
                            if dep_table == QUERY_RESULT_ENTITY and table_chain:
                                # Look for table lineage to find source tables
                                table_chains = table_chain.get('chains', {})
                                if QUERY_RESULT_ENTITY in table_chains:
                                    query_deps = table_chains[QUERY_RESULT_ENTITY].get('dependencies', [])
                                    if query_deps:
                                        # Use the first source table as the most likely source
                                        source_table = query_deps[0].get('table', 'source_table')
                                        return source_table
                            elif dep_table != QUERY_RESULT_ENTITY:
                                return dep_table
        
        # If no dot, it might be a computed column or aggregate
        # Try to infer from common patterns or return a meaningful default
        if any(keyword in column_name.lower() for keyword in ['count', 'sum', 'avg', 'max', 'min']):
            return QUERY_RESULT_ENTITY  # Group computed columns under result
        
        # For columns without table prefixes (typically in downstream lineage),
        # use a more descriptive name than "Unknown"
        if chain_type == "downstream":
            return "base_tables"
        else:
            return "source_columns"
    
    def _get_display_name(self, table_name: str, table_chain: Dict = None) -> str:
        """Get appropriate display name for table."""
        return table_name
    
    def _get_node_style(self, table_name: str, node_config: Dict) -> Dict:
        """Get appropriate node style based on table type."""
        if table_name == QUERY_RESULT_ENTITY:
            return node_config.get('query_result', node_config['table'])
        elif self._is_cte(table_name):
            return node_config.get('cte', node_config['table'])
        else:
            return node_config['table']
    
    def _get_node_style_by_type(self, entity_name: str, entity_type: str, node_config: Dict) -> Dict:
        """Get appropriate node style based on entity type from enhanced analysis."""
        if entity_type == 'cte':
            # CTE entities get special styling
            cte_style = node_config.get('cte', node_config['table']).copy()
            cte_style.update({
                'fillcolor': '#E8F4FD',  # Light blue for CTEs
                'color': '#1976D2',      # Blue border
                'fontcolor': '#0D47A1'   # Dark blue text
            })
            return cte_style
        elif entity_type == 'query_result':
            # Query result entities get final result styling
            result_style = node_config.get('query_result', node_config['table']).copy()
            result_style.update({
                'fillcolor': '#E8F5E8',  # Light green for results
                'color': '#2E7D32',      # Green border
                'fontcolor': '#1B5E20',  # Dark green text
                'style': 'filled,bold'   # Bold styling for emphasis
            })
            return result_style
        else:  # entity_type == 'table' or default
            # Regular table entities
            return node_config['table']
    
    def _add_transformation_boxes(self, dot: Digraph, chains: Dict, node_config: Dict, edge_config: Dict, chain_data: Optional[Dict] = None) -> None:
        """Add transformation boxes between entities showing operations like JOIN, FILTER, etc."""
        transformation_style = {
            'shape': 'box',
            'style': 'filled,rounded',
            'fillcolor': '#FFF8DC',
            'color': '#D2691E',
            'fontname': 'Arial',
            'fontsize': '10',
            'fontcolor': '#8B4513'
        }
        
        # Special style for window functions
        window_function_style = {
            'shape': 'box',
            'style': 'filled,rounded',
            'fillcolor': '#E6F3FF',
            'color': '#0066CC',
            'fontname': 'Arial',
            'fontsize': '10',
            'fontcolor': '#003399'
        }
        
        transformation_counter = 0
        
        # First, add window function transformations if detected
        sql = chain_data.get('sql', '') if chain_data else ''
        
        # Add window function boxes if we can detect them
        window_functions = self._extract_window_functions_from_context(chains)
        for window_func in window_functions:
            transformation_counter += 1
            trans_id = f"window_{transformation_counter}"
            
            label_parts = [f"**WINDOW FUNCTION**"]
            label_parts.append(f"Function: {window_func['function']}")
            if window_func.get('partition_by'):
                label_parts.append(f"Partition By: {window_func['partition_by']}")
            if window_func.get('order_by'):
                label_parts.append(f"Order By: {window_func['order_by']}")
            if window_func.get('alias'):
                label_parts.append(f"As: {window_func['alias']}")
            
            trans_label = "\\n".join(label_parts)
            dot.node(trans_id, trans_label, **window_function_style)
            
            # Connect to source and target
            source_table = window_func.get('source_table')
            if source_table and source_table in chains:
                dot.edge(source_table, trans_id, color='#0066CC', style='dashed')
                
                # Connect to result
                result_target = window_func.get('target', QUERY_RESULT_ENTITY)
                if result_target == QUERY_RESULT_ENTITY:
                    # We'll connect to result box later
                    pass
                else:
                    dot.edge(trans_id, result_target, color='#0066CC', style='dashed')
        
        # Collect all transformations from all entities recursively and deduplicate
        all_transformations_with_context = []
        
        def collect_transformations_recursive(entity_name: str, entity_data: Dict):
            """Recursively collect transformations from entity and all its dependencies."""
            # Get direct transformations
            transformations = entity_data.get('transformations', [])
            for trans in transformations:
                all_transformations_with_context.append({
                    'transformation': trans,
                    'entity': entity_name,
                    'type': 'direct'
                })
            
            # Recursively get dependency transformations 
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_entity = dep.get('entity')
                if dep_entity:
                    # Add transformations from this dependency
                    dep_transformations = dep.get('transformations', [])
                    for trans in dep_transformations:
                        all_transformations_with_context.append({
                            'transformation': trans,
                            'entity': dep_entity,
                            'type': 'dependency'
                        })
                    
                    # Recursively process nested dependencies
                    collect_transformations_recursive(dep_entity, dep)
        
        # Process all top-level chains recursively
        for entity_name, entity_data in chains.items():
            collect_transformations_recursive(entity_name, entity_data)
        
        # Deduplicate transformations based on their signature
        # First extract just the transformations from the context
        raw_transformations = [item['transformation'] for item in all_transformations_with_context]
        
        # Use the simple deduplication that merges transformations to same target
        merged_transformations = self._deduplicate_transformations_simple(raw_transformations)
        
        # Convert back to the expected format with context
        unique_transformations = []
        for trans in merged_transformations:
            # Find the original context for this transformation (use first match)
            original_context = None
            for item in all_transformations_with_context:
                if item['transformation'].get('target_table') == trans.get('target_table'):
                    original_context = item
                    break
            
            if original_context:
                unique_transformations.append({
                    'transformation': trans,
                    'entity': original_context['entity'],
                    'type': original_context['type'],
                    'source_tables': trans.get('source_tables', [trans.get('source_table')])
                })
            else:
                # Fallback if no context found
                unique_transformations.append({
                    'transformation': trans,
                    'entity': QUERY_RESULT_ENTITY,
                    'type': 'direct',
                    'source_tables': trans.get('source_tables', [trans.get('source_table')])
                })
        
        # Add missing CTE connections based on dependency structure
        # This handles cases where CTEs depend on other CTEs but transformations are missing
        missing_cte_transformations = self._create_missing_cte_transformations(chains)
        for missing_trans in missing_cte_transformations:
            unique_transformations.append(missing_trans)
        
        for transformation_info in unique_transformations:
            transformation = transformation_info['transformation']
            
            # Don't skip transformation boxes - all transformations should be nodes in the flow
            # The flow integration will handle connecting them properly
            
            transformation_counter += 1
            trans_id = f"trans_{transformation_counter}"
            
            # Build transformation label - use new joins format only
            joins = transformation.get('joins', [])
            
            # Determine the transformation type - consider all available transformation information
            group_by = transformation.get('group_by_columns', [])
            filter_conditions = transformation.get('filter_conditions', [])
            target_table = transformation.get('target_table', '')
            source_tables = transformation_info.get('source_tables', [transformation.get('source_table')])
            
            # Check for column-level transformations in the target entity
            column_transformations = self._extract_column_transformations(target_table, chains)
            
            if group_by:
                # If there's GROUP BY, this is primarily an aggregation
                trans_type = "GROUP BY AGGREGATION"
            elif joins:
                # This is a JOIN transformation (without GROUP BY)
                if len(joins) > 1:
                    # Multiple JOINs
                    trans_type = "MULTI-JOIN"
                elif len(joins) == 1:
                    # Single JOIN
                    trans_type = f"{joins[0].get('join_type', DEFAULT_JOIN_TYPE)}"
                else:
                    trans_type = DEFAULT_JOIN_TYPE  # Default JOIN type
            elif column_transformations.get('has_case'):
                # If there are CASE statements in columns
                trans_type = "CASE TRANSFORMATION"
            elif column_transformations.get('has_aggregation'):
                # If there are aggregate functions like MAX, MIN, COUNT, SUM, AVG
                trans_type = "AGGREGATE TRANSFORMATION"
            elif column_transformations.get('has_computed'):
                # If there are computed/aggregate columns
                trans_type = "COMPUTED TRANSFORMATION"
            elif transformation.get('unions'):
                # This is a UNION transformation
                unions = transformation.get('unions', [])
                if unions:
                    union_type = unions[0].get('union_type', 'UNION')
                    trans_type = union_type
                else:
                    trans_type = 'UNION'
            elif filter_conditions:
                # This is primarily a filter transformation
                trans_type = TRANSFORMATION_TYPES['TABLE_TRANSFORMATION']
            else:
                # Check if this looks like a multi-source final query (likely JOINs without explicit conditions)
                if target_table == QUERY_RESULT_ENTITY and len(source_tables) > 1:
                    trans_type = "MULTI-TABLE QUERY"
                elif target_table == QUERY_RESULT_ENTITY:
                    trans_type = "FINAL SELECT"
                else:
                    trans_type = transformation.get('type', TRANSFORMATION_TYPES['TRANSFORM'])
            
            label_parts = [f"**{trans_type.upper()}**"]
            
            # Add join information using new joins structure only
            if joins:
                for join in joins[:2]:  # Limit to first 2 JOINs to keep label manageable
                    join_type_name = join.get('join_type', DEFAULT_JOIN_TYPE)
                    right_table = join.get('right_table', '')
                    conditions = join.get('conditions', [])
                    
                    if right_table:
                        label_parts.append(f"**{join_type_name}** {right_table}")
                    
                    # Add conditions for this specific JOIN
                    valid_conditions = []
                    for condition in conditions[:2]:  # Limit to first 2 conditions per JOIN
                        if isinstance(condition, dict):
                            left_col = condition.get('left_column', '')
                            operator = condition.get('operator', '=')
                            right_col = condition.get('right_column', '')
                            
                            if left_col and right_col and left_col != 'unknown' and right_col != 'unknown':
                                valid_conditions.append(f"  {left_col} {operator} {right_col}")
                    
                    if valid_conditions:
                        label_parts.extend(valid_conditions)
            
            # Add filter conditions (only if we have valid conditions)
            filter_conditions = transformation.get('filter_conditions', [])
            if filter_conditions:
                valid_filters = []
                for condition in filter_conditions[:2]:  # Limit to 2 filters
                    if isinstance(condition, dict):
                        # Handle nested filter structure: {"type": "FILTER", "conditions": [...]}
                        if condition.get('type') == TRANSFORMATION_TYPES['FILTER'] and 'conditions' in condition:
                            nested_conditions = condition.get('conditions', [])
                            for nested_condition in nested_conditions[:2]:  # Limit nested conditions
                                if isinstance(nested_condition, dict):
                                    column = nested_condition.get('column', '')
                                    operator = nested_condition.get('operator', '=')
                                    value = nested_condition.get('value', '')
                                    
                                    # Only add filter if we have valid column and value
                                    if column and column != 'unknown' and value != '':
                                        # Clean and format the condition
                                        safe_column = str(column).replace('<', '&lt;').replace('>', '&gt;')
                                        safe_value = str(value).replace('<', '&lt;').replace('>', '&gt;').replace("'", "")
                                        valid_filters.append(f"  {safe_column} {operator} {safe_value}")
                        # Handle GROUP BY structure: {"type": "GROUP_BY", "columns": [...]}
                        elif condition.get('type') == TRANSFORMATION_TYPES['GROUP_BY'] and 'columns' in condition:
                            group_columns = condition.get('columns', [])
                            if group_columns:
                                # This should be handled by the main GROUP BY logic, but if it's in filter_conditions
                                # we can add it as a note
                                valid_filters.append(f"  GROUP BY: {', '.join(group_columns[:3])}")
                        # Handle inferred transformation details
                        elif condition.get('type') in [COLUMN_TYPES['COMPUTED'], COLUMN_TYPES['SELECT']] and 'description' in condition:
                            description = condition.get('description', '')
                            if description:
                                valid_filters.append(f"  {description}")
                        else:
                            # Handle direct filter structure: {"column": "...", "operator": "...", "value": "..."}
                            column = condition.get('column', '')
                            operator = condition.get('operator', '=')
                            value = condition.get('value', '')
                            
                            # Only add filter if we have valid column and value
                            if column and column != 'unknown' and value != '':
                                # Clean and format the condition
                                safe_column = str(column).replace('<', '&lt;').replace('>', '&gt;')
                                safe_value = str(value).replace('<', '&lt;').replace('>', '&gt;').replace("'", "")
                                valid_filters.append(f"  {safe_column} {operator} {safe_value}")
                    else:
                        # Handle string conditions
                        condition_str = str(condition).strip()
                        if condition_str and condition_str != 'unknown' and condition_str != 'unknown =':
                            safe_condition = condition_str.replace('<', '&lt;').replace('>', '&gt;')
                            valid_filters.append(f"  {safe_condition}")
                
                if valid_filters:
                    label_parts.append("Filters:")
                    label_parts.extend(valid_filters)
            
            # Add column transformation details if available
            if column_transformations.get('details'):
                for detail in column_transformations['details']:  # Show all transformation details
                    label_parts.append(f"  {detail}")
            
            # Add group by information (we already determined trans_type above)
            if group_by:
                label_parts.append(f"Group By: {', '.join(group_by[:3])}")
            
            # Add having conditions (post-aggregation filters)
            having_conditions = transformation.get('having_conditions', [])
            if having_conditions:
                valid_having = []
                for condition in having_conditions[:2]:  # Limit to 2 having conditions
                    if isinstance(condition, dict):
                        column = condition.get('column', '')
                        operator = condition.get('operator', '=')
                        value = condition.get('value', '')
                        
                        # Only add having condition if we have valid column and value
                        if column and column != 'unknown' and value != '':
                            # Clean and format the condition
                            safe_column = str(column).replace('<', '&lt;').replace('>', '&gt;')
                            safe_value = str(value).replace('<', '&lt;').replace('>', '&gt;')
                            valid_having.append(f"  {safe_column} {operator} {safe_value}")
                
                if valid_having:
                    label_parts.append("Having:")
                    label_parts.extend(valid_having)
            
            # Add order by information
            order_by = transformation.get('order_by_columns', [])
            if order_by:
                label_parts.append(f"Order By: {', '.join(order_by[:2])}")
            
            # Connect multiple source tables to transformation (NEW LOGIC)
            source_tables = transformation_info.get('source_tables', [transformation.get('source_table')])
            target_table = transformation.get('target_table')
            
            # Connect source entities to transformation
            # For transformation from source to target, we want: source → transformation → target
            
            # Check if this transformation should be integrated into entity flow
            # Use the comprehensive entity connection detection for ALL sources
            flow_integration_sources = []
            separate_connection_sources = []
            
            for source_table in source_tables:
                if source_table and target_table:
                    # Use the comprehensive connection detection method
                    has_connection = self._has_entity_connection_in_chains(source_table, target_table, chains)
                    
                    if has_connection:
                        flow_integration_sources.append(source_table)
                    else:
                        separate_connection_sources.append(source_table)
            
            # Always create transformation node for flow integration
            trans_label = "\\n".join(label_parts)
            dot.node(trans_id, trans_label, **transformation_style)
            
            # Always integrate ALL transformations into flow - no more disconnected transformations
            if not hasattr(dot, '_flow_transformations'):
                dot._flow_transformations = []
            
            # Use flow_integration_sources if detected, otherwise use all sources
            sources_to_use = flow_integration_sources if flow_integration_sources else source_tables
            
            if sources_to_use and target_table:
                # For multi-source transformations (JOINs), we need individual table names, not merged strings
                if isinstance(sources_to_use, list) and len(sources_to_use) == 1 and '+' in str(sources_to_use[0]):
                    # This is a merged string like "orders + users", split it back to individual tables
                    merged_string = sources_to_use[0]
                    individual_sources = [s.strip() for s in merged_string.split('+')]
                else:
                    # Use individual source tables or sources_to_use as is
                    individual_sources = source_tables if source_tables else sources_to_use
                
                dot._flow_transformations.append({
                    'trans_id': trans_id,
                    'sources': individual_sources,  # Use actual individual table names
                    'target': target_table,
                    'all_sources': source_tables
                })
            # Flow transformations will be connected by the entity flow logic (entity → trans → target)
    
    def _extract_window_functions_from_context(self, chains: Dict) -> List[Dict]:
        """Extract actual window functions from transformation context or dependencies."""
        window_functions = []
        
        # Only look for ACTUAL window functions in the analyzer data
        # Don't assume ORDER BY means window functions - that's incorrect!
        for entity_name, entity_data in chains.items():
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_transformations = dep.get('transformations', [])
                for trans in dep_transformations:
                    # Only process if this is explicitly marked as a window function transformation
                    if trans.get('type') == 'window_function' and trans.get('window_functions'):
                        # Extract actual window function data from analyzer
                        for window_func in trans.get('window_functions', []):
                            window_functions.append({
                                'function': window_func.get('function', UNKNOWN_FUNCTION),
                                'partition_by': ', '.join(window_func.get('partition_by', [])),
                                'order_by': ', '.join(window_func.get('order_by', [])),
                                'source_table': entity_name,
                                'target': dep.get('entity', QUERY_RESULT_ENTITY),
                                'alias': window_func.get('alias', '')
                            })
        
        return window_functions
    
    def _deduplicate_query_result_nodes(self, chains: Dict) -> Dict:
        """Deduplicate QUERY_RESULT nodes and merge their transformations and metadata."""
        # Find all QUERY_RESULT nodes and their data
        query_result_nodes = []
        merged_transformations = []
        merged_columns = []
        
        def find_query_results_recursive(entity_name: str, entity_data: Dict, path: List[str] = None):
            """Find all QUERY_RESULT nodes recursively."""
            if path is None:
                path = []
            
            if entity_name == QUERY_RESULT_ENTITY:
                query_result_nodes.append({
                    'entity_data': entity_data,
                    'path': path.copy(),
                    'parent_entities': [p for p in path if p != QUERY_RESULT_ENTITY]
                })
                
                # Collect transformations
                transformations = entity_data.get('transformations', [])
                merged_transformations.extend(transformations)
                
                # Collect columns
                metadata = entity_data.get('metadata', {})
                table_columns = metadata.get('table_columns', [])
                merged_columns.extend(table_columns)
            
            # Continue searching in dependencies
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_entity = dep.get('entity')
                if dep_entity:
                    find_query_results_recursive(dep_entity, dep, path + [entity_name])
        
        # Find all QUERY_RESULT nodes
        for entity_name, entity_data in chains.items():
            find_query_results_recursive(entity_name, entity_data)
        
        # If we found multiple QUERY_RESULT nodes, merge them
        if len(query_result_nodes) > 1:
            # Create merged QUERY_RESULT
            merged_query_result = {
                'entity': QUERY_RESULT_ENTITY,
                'entity_type': 'table',
                'depth': max(qr['entity_data'].get('depth', 0) for qr in query_result_nodes),
                'dependencies': [],
                'metadata': {
                    'table_columns': self._deduplicate_columns(merged_columns),
                    'is_cte': False
                },
                'transformations': self._deduplicate_transformations_simple(merged_transformations)
            }
            
            # Create new chains structure with merged QUERY_RESULT
            new_chains = {}
            
            # Process each original chain and replace QUERY_RESULT dependencies
            for entity_name, entity_data in chains.items():
                new_entity_data = self._replace_query_result_dependencies(entity_data, merged_query_result)
                new_chains[entity_name] = new_entity_data
            
            return new_chains
        else:
            # No duplication, return original chains
            return chains
    
    def _deduplicate_columns(self, columns: List[Dict]) -> List[Dict]:
        """Deduplicate columns by name, keeping the most complete data."""
        seen_columns = {}
        
        for col in columns:
            col_name = col.get('name', '')
            if col_name:
                if col_name not in seen_columns:
                    seen_columns[col_name] = col
                else:
                    # Merge column data - prefer more complete entries
                    existing = seen_columns[col_name]
                    if len(col.get('upstream', [])) > len(existing.get('upstream', [])):
                        seen_columns[col_name] = col
                    elif col.get('transformation') and not existing.get('transformation'):
                        seen_columns[col_name] = col
        
        return list(seen_columns.values())
    
    def _deduplicate_transformations_simple(self, transformations: List[Dict]) -> List[Dict]:
        """Enhanced transformation deduplication with intelligent merging."""
        if not transformations:
            return []
        
        # For JOIN queries with multiple source tables, combine them into a single transformation
        target_groups = {}
        
        # Group transformations by target table
        for trans in transformations:
            target = trans.get('target_table', QUERY_RESULT_ENTITY)
            if target not in target_groups:
                target_groups[target] = []
            target_groups[target].append(trans)
        
        merged_transformations = []
        
        for target, trans_list in target_groups.items():
            if len(trans_list) == 1:
                # Single transformation, use as-is
                merged_transformations.append(trans_list[0])
            else:
                # Multiple transformations to same target - merge them
                merged_trans = self._merge_transformations_to_target(trans_list, target)
                merged_transformations.append(merged_trans)
        
        return merged_transformations
    
    def _merge_transformations_to_target(self, transformations: List[Dict], target: str) -> Dict:
        """Merge multiple transformations that go to the same target."""
        # Start with the most complete transformation as base
        base_trans = max(transformations, key=lambda t: (
            len(t.get('filter_conditions', [])) +
            len(t.get('group_by_columns', [])) +
            len(t.get('having_conditions', [])) +
            sum(len(j.get('conditions', [])) for j in t.get('joins', []))
        ))
        
        merged = base_trans.copy()
        
        # Collect all source tables
        source_tables = []
        for trans in transformations:
            source = trans.get('source_table', '')
            if source and source not in source_tables:
                source_tables.append(source)
        
        # Update source table to show combined sources
        merged['source_tables'] = source_tables  # Store list for connection logic
        if len(source_tables) > 1:
            merged['source_table'] = ' + '.join(sorted(source_tables))
            merged['multi_source'] = True
        elif len(source_tables) == 1:
            merged['source_table'] = source_tables[0]
        
        # Merge filter conditions from all transformations
        all_filters = []
        for trans in transformations:
            filters = trans.get('filter_conditions', [])
            for filt in filters:
                if filt not in all_filters:
                    all_filters.append(filt)
        merged['filter_conditions'] = all_filters
        
        # Merge having conditions
        all_having = []
        for trans in transformations:
            having = trans.get('having_conditions', [])
            for hav in having:
                if hav not in all_having:
                    all_having.append(hav)
        merged['having_conditions'] = all_having
        
        # Merge joins
        all_joins = []
        for trans in transformations:
            joins = trans.get('joins', [])
            for join in joins:
                if join not in all_joins:
                    all_joins.append(join)
        merged['joins'] = all_joins
        
        # Keep the most complete group_by and order_by
        for trans in transformations:
            if len(trans.get('group_by_columns', [])) > len(merged.get('group_by_columns', [])):
                merged['group_by_columns'] = trans.get('group_by_columns', [])
            if len(trans.get('order_by_columns', [])) > len(merged.get('order_by_columns', [])):
                merged['order_by_columns'] = trans.get('order_by_columns', [])
        
        return merged
    
    def _has_query_result_in_chain(self, entity_data: Dict) -> bool:
        """Check if QUERY_RESULT exists in a chain recursively."""
        if entity_data.get('entity') == QUERY_RESULT_ENTITY:
            return True
        
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            if self._has_query_result_in_chain(dep):
                return True
        
        return False
    
    def _replace_query_result_dependencies(self, entity_data: Dict, merged_query_result: Dict) -> Dict:
        """Replace QUERY_RESULT dependencies with the merged one."""
        new_entity_data = entity_data.copy()
        new_dependencies = []
        
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            dep_entity = dep.get('entity')
            if dep_entity == QUERY_RESULT_ENTITY:
                # Replace with merged QUERY_RESULT (but only add once per entity)
                if not any(d.get('entity') == QUERY_RESULT_ENTITY for d in new_dependencies):
                    new_dependencies.append(merged_query_result)
            else:
                # Recursively process non-QUERY_RESULT dependencies
                new_dep = self._replace_query_result_dependencies(dep, merged_query_result)
                new_dependencies.append(new_dep)
        
        new_entity_data['dependencies'] = new_dependencies
        return new_entity_data
    
    def _add_entity_connections(self, dot: Digraph, chains: Dict, edge_config: Dict) -> None:
        """Add connections between entities with special handling for subqueries and transformations."""
        subquery_edge_config = edge_config.copy()
        subquery_edge_config.update({
            'style': 'dashed',
            'color': '#FFA500',
            'penwidth': '1',
            'arrowhead': 'diamond'
        })
        
        # Get list of connections that should have transformations inserted
        flow_transformations = getattr(dot, '_flow_transformations', [])
        transformation_connections = set()
        for ft in flow_transformations:
            # Handle both old format (single source) and new format (multiple sources)
            if 'sources' in ft:
                # New multi-source format - add all source connections
                for source in ft['sources']:
                    transformation_connections.add((source, ft['target']))
            else:
                # Old single-source format (backward compatibility)
                transformation_connections.add((ft.get('source', ''), ft['target']))
        
        
        # Recursively traverse all entity dependencies
        def add_connections_recursive(entity_name: str, entity_data: Dict):
            """Recursively add connections for entity and all its dependencies."""
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_entity = dep.get('entity')
                if dep_entity:
                    # Check if there's a transformation node that should be inserted in this flow
                    transformation_in_flow = None
                    
                    # Look for transformation boxes that represent this entity connection
                    if hasattr(dot, '_flow_transformations'):
                        for ft in dot._flow_transformations:
                            # Handle both old format (single source) and new format (multiple sources)
                            if 'sources' in ft:
                                # New multi-source format
                                # When processing dependencies, we have entity_name (target) connecting to dep_entity (source)
                                # For multi-source transformations, if the dependency is one of the transformation sources
                                # and the current entity is the transformation target, route through transformation
                                if dep_entity in ft['sources'] and ft['target'] == entity_name:
                                    transformation_in_flow = ft['trans_id']
                                    break
                            else:
                                # Old single-source format (backward compatibility)
                                if ft.get('source') == dep_entity and ft['target'] == entity_name:
                                    transformation_in_flow = ft['trans_id']
                                    break
                    
                    # Skip all entity-level connections - transformations handle them directly
                    # This prevents duplicate arrows
                    pass
                    
                    # Recursively process dependencies of this dependency
                    add_connections_recursive(dep_entity, dep)
        
        # Process all top-level chains and their nested dependencies
        for entity_name, entity_data in chains.items():
            add_connections_recursive(entity_name, entity_data)
        
        # Also process transformation targets that are not in chains (like QUERY_RESULT)
        if hasattr(dot, '_flow_transformations'):
            processed_targets = set()
            for ft in dot._flow_transformations:
                target = ft['target']
                if target and target not in chains and target not in processed_targets:
                    processed_targets.add(target)
                    # Create a minimal entity data structure for the target
                    target_dependencies = []
                    # Find dependencies that point to this target
                    for entity_name, entity_data in chains.items():
                        deps = entity_data.get('dependencies', [])
                        for dep in deps:
                            if dep.get('entity') == target:
                                target_dependencies.append({'entity': entity_name})
                    
                    if target_dependencies:
                        target_entity_data = {'dependencies': target_dependencies}
                        add_connections_recursive(target, target_entity_data)
        
        # DIRECT FIX: Ensure all transformations are connected regardless of entity routing
        if hasattr(dot, '_flow_transformations'):
            for ft in dot._flow_transformations:
                trans_id = ft['trans_id']
                target = ft['target'] 
                sources = ft.get('sources', [])
                
                # Connect each source to transformation
                for source in sources:
                    # Connect all sources - they should exist as nodes by this point
                    dot.edge(source, trans_id, **edge_config)
                
                # Connect transformation to target (always)
                if target:
                    dot.edge(trans_id, target, **edge_config)
    
    def _merge_transformations_for_edge(self, transformations: List[Dict]) -> Dict:
        """Merge multiple transformations for display on an edge."""
        if not transformations:
            return {}
        
        if len(transformations) == 1:
            return transformations[0]
        
        # Use the same merging logic as _merge_transformations_to_target
        base_trans = max(transformations, key=lambda t: (
            len(t.get('filter_conditions', [])) +
            len(t.get('group_by_columns', [])) +
            len(t.get('having_conditions', [])) +
            sum(len(j.get('conditions', [])) for j in t.get('joins', []))
        ))
        
        merged = base_trans.copy()
        
        # Merge conditions from all transformations
        all_filters = []
        all_having = []
        all_joins = []
        all_groups = set()
        all_order_by = set()
        
        for trans in transformations:
            # Merge filters
            filters = trans.get('filter_conditions', [])
            for filt in filters:
                if filt not in all_filters:
                    all_filters.append(filt)
            
            # Merge having conditions
            having = trans.get('having_conditions', [])
            for hav in having:
                if hav not in all_having:
                    all_having.append(hav)
            
            # Merge join conditions  
            joins = trans.get('join_conditions', [])
            for join in joins:
                if join not in all_joins:
                    all_joins.append(join)
            
            # Merge group by (use set to avoid duplicates)
            groups = trans.get('group_by_columns', [])
            all_groups.update(groups)
            
            # Merge order by
            order_by = trans.get('order_by_columns', [])
            all_order_by.update(order_by)
        
        merged['filter_conditions'] = all_filters
        merged['having_conditions'] = all_having
        merged['join_conditions'] = all_joins
        merged['group_by_columns'] = list(all_groups)
        merged['order_by_columns'] = list(all_order_by)
        
        return merged
    
    def _create_transformation_edge_label(self, transformation: Dict) -> str:
        """Create a concise label for transformation information on edges."""
        if not transformation:
            return ""
        
        label_parts = []
        
        # Join type is now handled within the joins array, so this is no longer needed
        
        # Add key conditions (limit to keep label concise)
        joins = transformation.get('joins', [])
        if joins:
            # Show first join and its first condition
            first_join = joins[0]
            conditions = first_join.get('conditions', [])
            if conditions:
                condition = conditions[0]
                if isinstance(condition, dict):
                    left_col = condition.get('left_column', '')
                    right_col = condition.get('right_column', '')
                    if left_col and right_col:
                        # Simplify column names for edge labels
                        left_simple = left_col.split('.')[-1] if '.' in left_col else left_col
                        right_simple = right_col.split('.')[-1] if '.' in right_col else right_col
                        label_parts.append(f"ON {left_simple}={right_simple}")
        
        # Add filter info if present
        filters = transformation.get('filter_conditions', [])
        if filters:
            label_parts.append(f"WHERE ({len(filters)} filters)")
        
        # Add group by info if present
        groups = transformation.get('group_by_columns', [])
        if groups:
            label_parts.append(f"GROUP BY ({len(groups)} cols)")
        
        # Add having info if present
        having = transformation.get('having_conditions', [])
        if having:
            label_parts.append(f"HAVING ({len(having)} conditions)")
        
        # Add UNION info if present
        unions = transformation.get('unions', [])
        if unions:
            union_info = unions[0]  # Take the first union info
            union_type = union_info.get('union_type', 'UNION')
            union_source = union_info.get('union_source', '')
            if union_source:
                label_parts.append(f"{union_type} (from {union_source})")
            else:
                label_parts.append(union_type)
        
        return "\\n".join(label_parts)
    
    def _has_entity_connection_in_chains(self, source_entity: str, target_entity: str, chains: Dict) -> bool:
        """Check if there's an entity connection from source to target anywhere in the nested chains structure."""
        if not source_entity or not target_entity:
            return False
        
        # Search through all entities in the nested structure
        def search_for_connection(entity_name: str, entity_data: Dict) -> bool:
            # Check if this entity matches the source we're looking for
            if entity_name == source_entity:
                # Check if any of its dependencies match the target
                dependencies = entity_data.get('dependencies', [])
                for dep in dependencies:
                    if dep.get('entity') == target_entity:
                        return True
            
            # Recursively search through dependencies
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_entity = dep.get('entity')
                if dep_entity and search_for_connection(dep_entity, dep):
                    return True
            
            return False
        
        # Search through all top-level chains
        for entity_name, entity_data in chains.items():
            if search_for_connection(entity_name, entity_data):
                return True
        
        return False
    
    def _is_subquery_connection(self, source_entity: str, target_entity: str, source_data: Dict, dependency: Dict) -> bool:
        """Determine if this connection represents a subquery relationship."""
        # Check for subquery indicators in transformations
        transformations = dependency.get('transformations', [])
        
        for trans in transformations:
            # Look for subquery patterns in filter conditions
            filter_conditions = trans.get('filter_conditions', [])
            for condition in filter_conditions:
                if isinstance(condition, dict):
                    operator = condition.get('operator', '').upper()
                    # IN, EXISTS, NOT EXISTS are typical subquery operators
                    if operator in [SQL_OPERATORS['IN'], SQL_OPERATORS['EXISTS'], SQL_OPERATORS['NOT_EXISTS']]:
                        return True
                    # Check for correlated subquery patterns
                    column = condition.get('column', '')
                    if '(' in str(condition.get('value', '')) or 'SELECT' in str(condition.get('value', '')).upper():
                        return True
        
        # Check if source and target are different table entities (indicating subquery)
        # Generic check - if both are table entities (not CTEs) and different, might be subquery
        if (not self._is_cte(source_entity) and not self._is_cte(target_entity) and 
            source_entity != target_entity and 
            source_entity not in [QUERY_RESULT_ENTITY] and target_entity not in [QUERY_RESULT_ENTITY]):
            return True
        
        return False
    
    def _add_final_result_box(self, dot: Digraph, chain_data: Dict, node_config: Dict) -> None:
        """Add the final result box at the end of the lineage flow."""
        chains = chain_data.get('chains', {})
        sql = chain_data.get('sql', '')
        
        # Check if this is a CTAS query
        is_ctas = sql.strip().upper().startswith('CREATE TABLE')
        
        # Look for QUERY_RESULT entity or find the final output entity
        result_entity_name = None
        result_entity_data = None
        
        if QUERY_RESULT_ENTITY in chains:
            result_entity_name = QUERY_RESULT_ENTITY
            result_entity_data = chains[QUERY_RESULT_ENTITY]
        elif is_ctas:
            # For CTAS queries with transformations, don't create a separate result box
            # The flow should end at the created table entity which already has its own node
            # Check if there are transformations that would create the proper flow
            has_transformations = any(
                entity_data.get('transformations', []) or 
                any(dep.get('transformations', []) for dep in entity_data.get('dependencies', []))
                for entity_data in chains.values()
            )
            
            if has_transformations:
                # Skip creating final result box - transformations will handle the flow
                return
            
            # For CTAS without transformations, find the created table (entity with highest depth)
            max_depth = -1
            for entity_name, entity_data in chains.items():
                depth = entity_data.get('depth', 0)
                if depth > max_depth:
                    max_depth = depth
                    result_entity_name = entity_name
                    result_entity_data = entity_data
        else:
            # Find entity with highest depth or specified target
            target_entity = chain_data.get('target_entity')
            if target_entity and target_entity in chains:
                result_entity_name = target_entity
                result_entity_data = chains[target_entity]
            else:
                # Find entity with maximum depth (furthest in the chain)
                max_depth = -1
                for entity_name, entity_data in chains.items():
                    depth = entity_data.get('depth', 0)
                    if depth > max_depth:
                        max_depth = depth
                        result_entity_name = entity_name
                        result_entity_data = entity_data
        
        if not result_entity_data or not result_entity_name:
            return
        
        # For CTAS queries, modify the existing table node instead of creating separate result box
        if is_ctas and result_entity_name != QUERY_RESULT_ENTITY:
            self._modify_ctas_result_node(dot, result_entity_name, result_entity_data, chain_data, node_config)
            return
        
        # Build separate result box for non-CTAS queries
        result_style = {
            'shape': 'box',
            'style': 'filled,bold',
            'fillcolor': '#E8F5E8',
            'color': '#2E7D32',
            'fontname': 'Arial Bold',
            'fontsize': '12',
            'fontcolor': '#1B5E20',
            'rank': 'sink'  # Force to be at the end
        }
        
        label_parts = ["**QUERY RESULT**"]
        label_parts.append("═" * 25)
        
        # Get result columns from SQL query or metadata
        sql = chain_data.get('sql', '')
        if sql:
            # Extract SELECT columns from SQL for display
            select_columns = self._extract_select_columns_from_sql(sql)
            if select_columns:
                label_parts.append("Output Columns:")
                for col in select_columns[:8]:  # Limit to 8 columns for readability
                    safe_col = str(col).replace('<', '&lt;').replace('>', '&gt;')
                    label_parts.append(f"• {safe_col}")
                if len(select_columns) > 8:
                    label_parts.append(f"... and {len(select_columns) - 8} more")
        
        # Fallback: try to get columns from metadata
        if len(label_parts) <= 2:  # Only has header
            metadata = result_entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            if table_columns:
                label_parts.append("Output Columns:")
                for col_info in table_columns[:6]:  # Limit for readability
                    col_name = col_info.get('name', col_info.get('column_name', UNKNOWN_VALUE))  # Support both old and new format
                    
                    # Check for old format transformations first, then new format type
                    transformations = col_info.get('transformations', [])
                    col_type = col_info.get('type', COLUMN_TYPES['DIRECT'])
                    
                    if transformations:
                        # Old format - use transformation expression
                        expr = transformations[0].get('expression', col_name)
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        safe_expr = str(expr).replace('<', '&lt;').replace('>', '&gt;')
                        if expr != col_name and len(expr) < 30:
                            label_parts.append(f"• {safe_col_name}: {safe_expr}")
                        else:
                            label_parts.append(f"• {safe_col_name}")
                    elif col_type and col_type != COLUMN_TYPES['DIRECT']:
                        # New format - show column with type
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"• {safe_col_name} ({col_type})")
                    else:
                        # Simple column name
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"• {safe_col_name}")
        
        result_label = "\\n".join(label_parts)
        result_id = FINAL_RESULT_ENTITY
        
        # Add result box with rank constraint to position it at the end
        dot.node(result_id, result_label, **result_style)
        
        # Connect all entities that don't have outgoing dependencies to the result box
        entities_connected_to_result = set()
        
        # First pass: identify entities that should connect to result
        for entity_name, entity_data in chains.items():
            if entity_name == QUERY_RESULT_ENTITY:
                continue
                
            # Check if this entity has dependencies that lead to QUERY_RESULT or has no dependencies
            dependencies = entity_data.get('dependencies', [])
            has_query_result_dep = any(dep.get('entity') == QUERY_RESULT_ENTITY for dep in dependencies)
            has_no_deps = len(dependencies) == 0
            
            if has_query_result_dep or has_no_deps:
                entities_connected_to_result.add(entity_name)
        
        # If no entities found, connect the one with highest depth
        if not entities_connected_to_result and result_entity_name != QUERY_RESULT_ENTITY:
            entities_connected_to_result.add(result_entity_name)
        
        # Connect transformations to result box if any were stored
        if hasattr(dot, '_result_transformations'):
            for trans_id in dot._result_transformations:
                dot.edge(trans_id, result_id, color='#2E7D32', penwidth='2', arrowhead='vee')
        
        # Connect identified entities to result box (only if no transformations are connecting)  
        if not hasattr(dot, '_result_transformations') or not dot._result_transformations:
            for entity_name in entities_connected_to_result:
                dot.edge(entity_name, result_id, color='#2E7D32', penwidth='3', arrowhead='vee')
        
        # Add rank constraint to ensure result box appears at the end
        dot.body.append('{rank=sink; FINAL_RESULT}')
    
    def _add_missing_ctas_target_entities(self, dot: Digraph, chains: Dict, node_config: Dict, edge_config: Dict, chain_data: Dict) -> None:
        """Add missing target entities for CTAS transformations."""
        # Collect all transformation targets that are missing from chains
        missing_targets = set()
        
        for entity_name, entity_data in chains.items():
            # Check direct transformations
            transformations = entity_data.get('transformations', [])
            for trans in transformations:
                target_table = trans.get('target_table')
                if target_table and target_table not in chains:
                    missing_targets.add(target_table)
            
            # Check dependency transformations
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_transformations = dep.get('transformations', [])
                for trans in dep_transformations:
                    target_table = trans.get('target_table')
                    if target_table and target_table not in chains:
                        missing_targets.add(target_table)
        
        # Add missing target entities as nodes
        for target_name in missing_targets:
            # Create a basic entity node for the missing target
            target_style = {
                'shape': 'box',
                'style': 'filled,bold',
                'fillcolor': '#E8F5E8',
                'color': '#2E7D32',
                'fontname': 'Arial Bold',
                'fontsize': '12',
                'fontcolor': '#1B5E20'
            }
            
            # Build label for the created table
            label_parts = [f"**{target_name}**"]
            label_parts.append("(CREATED TABLE)")
            
            # Try to extract columns from SQL if available  
            sql = chain_data.get('sql', '')
            if sql:
                select_columns = self._extract_select_columns_from_sql(sql)
                if select_columns:
                    label_parts.append("─" * 20)
                    label_parts.append("Columns:")
                    for col in select_columns[:6]:  # Limit to 6 columns
                        safe_col = str(col).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"• {safe_col}")
                    if len(select_columns) > 6:
                        label_parts.append(f"... and {len(select_columns) - 6} more")
            
            target_label = "\\n".join(label_parts)
            dot.node(target_name, target_label, **target_style)
            
            # Add the target to chains so transformation logic can find it
            chains[target_name] = {
                'entity': target_name,
                'entity_type': 'table',
                'depth': max([data.get('depth', 0) for data in chains.values()]) + 1,
                'dependencies': [],
                'transformations': [],
                'metadata': {}
            }
    
    def _modify_ctas_result_node(self, dot: Digraph, table_name: str, table_data: Dict, chain_data: Dict, node_config: Dict) -> None:
        """Modify the CTAS table node to show both table name and query result."""
        sql = chain_data.get('sql', '')
        
        # Build combined label showing both the table name and that it's the query result
        combined_style = {
            'shape': 'box',
            'style': 'filled,bold',
            'fillcolor': '#E8F5E8',
            'color': '#2E7D32',
            'fontname': 'Arial Bold',
            'fontsize': '12',
            'fontcolor': '#1B5E20',
            'rank': 'sink'
        }
        
        label_parts = [f"**{table_name}**"]
        label_parts.append("(QUERY RESULT)")
        label_parts.append("═" * 25)
        
        # Get output columns from SQL query
        select_columns = self._extract_select_columns_from_sql(sql)
        if select_columns:
            label_parts.append("Output Columns:")
            for col in select_columns[:8]:  # Limit to 8 columns
                safe_col = str(col).replace('<', '&lt;').replace('>', '&gt;')
                label_parts.append(f"• {safe_col}")
            if len(select_columns) > 8:
                label_parts.append(f"... and {len(select_columns) - 8} more")
        
        # Fallback: try to get columns from metadata
        if len(label_parts) <= 3:  # Only has header
            metadata = table_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            if table_columns:
                label_parts.append("Output Columns:")
                for col_info in table_columns[:6]:
                    col_name = col_info.get('name', col_info.get('column_name', UNKNOWN_VALUE))  # Support both old and new format
                    safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                    label_parts.append(f"• {safe_col_name}")
        
        combined_label = "\\n".join(label_parts)
        
        # Remove the existing table node and add the combined one
        # We need to find and update the existing node
        new_node_id = f"{table_name}_RESULT"
        dot.node(new_node_id, combined_label, **combined_style)
        
        # Update edges to point to the new combined node
        chains = chain_data.get('chains', {})
        for entity_name, entity_data in chains.items():
            if entity_name != table_name:  # Don't self-reference
                dependencies = entity_data.get('dependencies', [])
                for dep in dependencies:
                    if dep.get('entity') == table_name:
                        # Add edge to the combined result node
                        dot.edge(entity_name, new_node_id, color='#2E7D32', penwidth='3', arrowhead='vee')
        
        # Add rank constraint
        dot.body.append(f'{{rank=sink; {new_node_id}}}')
    
    def _extract_select_columns_from_sql(self, sql: str) -> List[str]:
        """Extract column names from SELECT clause for display in result box."""
        try:
            import re
            
            # Simple regex to extract SELECT columns (this is a basic implementation)
            # Remove comments and normalize whitespace
            sql_clean = re.sub(r'--.*?\n', ' ', sql)
            sql_clean = re.sub(r'/\*.*?\*/', ' ', sql_clean, flags=re.DOTALL)
            sql_clean = re.sub(r'\s+', ' ', sql_clean).strip()
            
            # Find SELECT clause
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_clean, re.IGNORECASE | re.DOTALL)
            if not select_match:
                return []
            
            select_part = select_match.group(1)
            
            # Split by comma but be careful with functions and subqueries
            columns = []
            paren_level = 0
            current_col = []
            
            for char in select_part:
                if char == '(':
                    paren_level += 1
                elif char == ')':
                    paren_level -= 1
                elif char == ',' and paren_level == 0:
                    if current_col:
                        col_text = ''.join(current_col).strip()
                        if col_text:
                            # Extract alias or column name
                            alias_match = re.search(r'\s+(?:AS\s+)?([\w_]+)\s*$', col_text, re.IGNORECASE)
                            if alias_match:
                                columns.append(alias_match.group(1))
                            else:
                                # Extract just the column name part
                                clean_col = re.sub(r'^.*\.', '', col_text.split()[0])  # Remove table prefix
                                columns.append(clean_col)
                    current_col = []
                    continue
                
                current_col.append(char)
            
            # Handle last column
            if current_col:
                col_text = ''.join(current_col).strip()
                if col_text:
                    alias_match = re.search(r'\s+(?:AS\s+)?([\w_]+)\s*$', col_text, re.IGNORECASE)
                    if alias_match:
                        columns.append(alias_match.group(1))
                    else:
                        clean_col = re.sub(r'^.*\.', '', col_text.split()[0])
                        columns.append(clean_col)
            
            return columns[:10]  # Limit to 10 columns
            
        except Exception:
            return []
    
    def _infer_used_columns_from_dependencies(self, entity_data: Dict, all_columns: List[Dict]) -> List[str]:
        """Infer likely used columns from transformation dependencies."""
        used_columns = []
        
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            dep_transformations = dep.get('transformations', [])
            for trans in dep_transformations:
                # Check filter conditions for column usage
                filter_conditions = trans.get('filter_conditions', [])
                for condition in filter_conditions:
                    if isinstance(condition, dict):
                        column = condition.get('column', '')
                        if column and column not in used_columns:
                            used_columns.append(column)
                
                # Check order by columns
                order_by_columns = trans.get('order_by_columns', [])
                for order_col in order_by_columns:
                    # Extract column name from "column NULLS LAST" format
                    clean_col = str(order_col).split()[0] if order_col else ''
                    if clean_col and clean_col not in used_columns:
                        used_columns.append(clean_col)
                
                # Check group by columns
                group_by_columns = trans.get('group_by_columns', [])
                for group_col in group_by_columns:
                    if group_col and group_col not in used_columns:
                        used_columns.append(group_col)
        
        # For window function queries, DO NOT add columns from metadata registry
        # Only show columns that are actually used in the SQL query transformations
        
        return used_columns
    
    def _extract_cte_output_columns(self, cte_name: str, entity_data: Dict) -> List[str]:
        """Extract output columns for CTE from its definition and usage."""
        output_columns = []
        
        # First, look at transformations where this CTE is the target (being created)
        transformations = entity_data.get('transformations', [])
        for trans in transformations:
            source_table = trans.get('source_table', '')
            target_table = trans.get('target_table', '')
            
            # If this CTE is the target, it's being created by this transformation
            if target_table == cte_name:
                # Extract columns that are being filtered or used in the source transformation
                source_columns = self._extract_columns_from_source_table(source_table, trans)
                output_columns.extend(source_columns)
        
        # Also check dependencies to see what columns are used downstream from this CTE
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            dep_transformations = dep.get('transformations', [])
            for trans in dep_transformations:
                source_table = trans.get('source_table', '')
                
                # If this CTE is used as a source in downstream transformations
                if source_table == cte_name:
                    # Check filter conditions to see what columns are expected from this CTE
                    filter_conditions = trans.get('filter_conditions', [])
                    for condition in filter_conditions:
                        if isinstance(condition, dict):
                            column = condition.get('column', '')
                            if column and column not in output_columns:
                                output_columns.append(column)
        
        # If no output columns found, don't assume any hardcoded column names
        # The visualizer should work with whatever columns are actually present in the metadata
        # or detected from transformations, without making assumptions about field names
        
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in output_columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)
        
        return unique_columns
    
    def _extract_columns_from_source_table(self, source_table: str, transformation: Dict) -> List[str]:
        """Extract columns that are likely selected from the source table based on transformation details."""
        columns = []
        
        # Extract from filter conditions (these columns are being used)
        filter_conditions = transformation.get('filter_conditions', [])
        for condition in filter_conditions:
            if isinstance(condition, dict):
                column = condition.get('column', '')
                if column:
                    columns.append(column)
        
        # Extract from group by columns (these are selected)
        group_by_columns = transformation.get('group_by_columns', [])
        columns.extend(group_by_columns)
        
        # Extract from order by columns (these are selected)
        order_by_columns = transformation.get('order_by_columns', [])
        for col in order_by_columns:
            clean_col = str(col).split()[0] if col else ''
            if clean_col:
                columns.append(clean_col)
        
        # For common patterns based on source table name - make it generic
        source_lower = source_table.lower()
        
        # Look for filter conditions to identify additional columns being used
        for condition in filter_conditions:
            if isinstance(condition, dict):
                column = condition.get('column', '')
                if column and column not in columns:
                    columns.append(column)
        
        # No hardcoded column assumptions - let the actual transformation data drive what columns are shown
        
        # Filter out QUERY_RESULT columns and invalid references
        filtered_columns = []
        for col in columns:
            col_str = str(col)
            if not (col_str.startswith('QUERY_RESULT.') or 
                   col_str == QUERY_RESULT_ENTITY or 
                   col_str.startswith('unknown') or
                   col_str == 'unknown'):
                filtered_columns.append(col)
        
        return filtered_columns
    
    def _deduplicate_transformations(self, transformations_with_context: List[Dict]) -> List[Dict]:
        """Deduplicate transformations based on their signature, merging multiple source tables for identical transformations."""
        unique_transformations = {}
        
        for trans_info in transformations_with_context:
            transformation = trans_info['transformation']
            
            # Create a signature based on transformation characteristics (excluding source_table for true deduplication)
            target_table = transformation.get('target_table', '')
            trans_type = transformation.get('type', TRANSFORMATION_TYPES['TRANSFORM'])
            # Extract join_type from joins array for signature
            joins = transformation.get('joins', [])
            join_type = joins[0].get('join_type', '') if joins else ''
            
            # Sort conditions for consistent comparison
            filter_conditions = transformation.get('filter_conditions', [])
            filter_signature = tuple(sorted([
                f"{c.get('column', 'unknown')}_{c.get('operator', '=')}_{c.get('value', '')}"
                for c in filter_conditions if isinstance(c, dict)
            ]))
            
            joins = transformation.get('joins', [])
            join_signature = tuple(sorted([
                f"{c.get('left_column', 'unknown')}_{c.get('operator', '=')}_{c.get('right_column', 'unknown')}"
                for join in joins for c in join.get('conditions', []) if isinstance(c, dict)
            ]))
            
            group_by = tuple(sorted(transformation.get('group_by_columns', [])))
            order_by = tuple(sorted(transformation.get('order_by_columns', [])))
            
            # Create unique signature WITHOUT source_table to allow merging from multiple sources
            signature = (target_table, trans_type, join_type, filter_signature, join_signature, group_by, order_by)
            
            if signature not in unique_transformations:
                # First occurrence - store with source tables list
                unique_transformations[signature] = {
                    'transformation': transformation,
                    'entity': trans_info['entity'],
                    'type': trans_info['type'],
                    'source_tables': [transformation.get('source_table', '')]  # Track all source tables
                }
            else:
                # Duplicate found - add source table to the list if not already present
                source_table = transformation.get('source_table', '')
                if source_table and source_table not in unique_transformations[signature]['source_tables']:
                    unique_transformations[signature]['source_tables'].append(source_table)
        
        return list(unique_transformations.values())
    
    def _extract_columns_from_transformations(self, entity_name: str, entity_data: Dict) -> List[str]:
        """Extract specific columns used in transformations involving this entity."""
        extracted_columns = []
        
        # Look at transformations that involve this entity as source
        transformations = entity_data.get('transformations', [])
        for trans in transformations:
            source_table = trans.get('source_table', '')
            
            # If this entity is the source, extract columns from various transformation types
            if source_table == entity_name:
                # Extract from filter conditions
                filter_conditions = trans.get('filter_conditions', [])
                for condition in filter_conditions:
                    if isinstance(condition, dict):
                        column = condition.get('column', '')
                        if column and column not in extracted_columns:
                            extracted_columns.append(column)
                
                # Extract from join conditions - be more specific about which column belongs to which table
                joins = trans.get('joins', [])
                for join in joins:
                    for condition in join.get('conditions', []):
                        if isinstance(condition, dict):
                            left_col = condition.get('left_column', '')
                            right_col = condition.get('right_column', '')
                        
                        # For JOIN conditions like u.id = o.user_id, extract the relevant column for this table
                        if left_col:
                            # Remove table alias (u.id -> id)
                            clean_left_col = left_col.split('.')[-1] if '.' in left_col else left_col
                            if clean_left_col and clean_left_col not in extracted_columns:
                                extracted_columns.append(clean_left_col)
                        
                        if right_col:
                            # Remove table alias (o.user_id -> user_id)
                            clean_right_col = right_col.split('.')[-1] if '.' in right_col else right_col
                            if clean_right_col and clean_right_col not in extracted_columns:
                                extracted_columns.append(clean_right_col)
                
                # Extract from group by columns
                group_by_columns = trans.get('group_by_columns', [])
                for col in group_by_columns:
                    if col and col not in extracted_columns:
                        # Remove table alias if present
                        clean_col = col.split('.')[-1] if '.' in col else col
                        extracted_columns.append(clean_col)
                
                # Extract from order by columns
                order_by_columns = trans.get('order_by_columns', [])
                for col in order_by_columns:
                    # Clean column name (remove DESC, ASC, etc.)
                    clean_col = str(col).split()[0] if col else ''
                    if clean_col:
                        # Remove table alias if present
                        final_col = clean_col.split('.')[-1] if '.' in clean_col else clean_col
                        if final_col and final_col not in extracted_columns:
                            extracted_columns.append(final_col)
        
        # Extract columns that are selected in the final query by looking at column mappings
        # Check if this entity has columns that map to the final result
        metadata = entity_data.get('metadata', {})
        table_columns = metadata.get('table_columns', [])
        
        for col_info in table_columns:
            # If this column from this table is used in the result
            col_name = col_info.get('name', col_info.get('column_name', ''))  # Support both old and new format
            upstream_cols = col_info.get('upstream', col_info.get('upstream_columns', []))  # Support both old and new format
            
            # If upstream_columns points to columns from this entity, those are the used columns
            for upstream_col in upstream_cols:
                # Clean the upstream column name (remove table prefix if present)
                clean_upstream = upstream_col.split('.')[-1] if '.' in upstream_col else upstream_col
                if clean_upstream and clean_upstream not in extracted_columns:
                    extracted_columns.append(clean_upstream)
        
        # Also look at dependencies for column usage
        dependencies = entity_data.get('dependencies', [])
        for dep in dependencies:
            dep_transformations = dep.get('transformations', [])
            for trans in dep_transformations:
                source_table = trans.get('source_table', '')
                if source_table == entity_name:
                    # Extract similar column information from dependencies
                    filter_conditions = trans.get('filter_conditions', [])
                    for condition in filter_conditions:
                        if isinstance(condition, dict):
                            column = condition.get('column', '')
                            if column and column not in extracted_columns:
                                # Clean column name
                                clean_column = column.split('.')[-1] if '.' in column else column
                                extracted_columns.append(clean_column)
        
        # Generic pattern detection for any entity name
        # Extract columns from JOIN conditions involving this entity
        for trans in transformations + [t for dep in dependencies for t in dep.get('transformations', [])]:
            joins = trans.get('joins', [])
            for join in joins:
                for condition in join.get('conditions', []):
                    if isinstance(condition, dict):
                        left_col = condition.get('left_column', '')
                        right_col = condition.get('right_column', '')
                    
                    # Check if this entity is referenced in JOIN conditions
                    for col_ref in [left_col, right_col]:
                        if col_ref and '.' in col_ref:
                            table_ref, col_name = col_ref.split('.', 1)
                            if (table_ref == entity_name or 
                                table_ref.lower() == entity_name.lower() or
                                self._is_table_alias_match(table_ref, entity_name)):
                                if col_name not in extracted_columns:
                                    extracted_columns.append(col_name)
        
        # Extract columns from filter conditions for this entity
        for trans in transformations + [t for dep in dependencies for t in dep.get('transformations', [])]:
            filter_conditions = trans.get('filter_conditions', [])
            for condition in filter_conditions:
                if isinstance(condition, dict):
                    column = condition.get('column', '')
                    if column and column not in extracted_columns:
                        # If column has table prefix, check if it belongs to this entity
                        if '.' in column:
                            table_ref, col_name = column.split('.', 1)
                            if (table_ref == entity_name or 
                                table_ref.lower() == entity_name.lower() or
                                self._is_table_alias_match(table_ref, entity_name)):
                                extracted_columns.append(col_name)
                        else:
                            # Column without prefix - might belong to this entity if it's the source
                            source_table = trans.get('source_table', '')
                            if source_table == entity_name:
                                extracted_columns.append(column)
        
        # Filter out QUERY_RESULT columns and invalid references
        filtered_columns = []
        for col in extracted_columns:
            col_str = str(col)
            if not (col_str.startswith('QUERY_RESULT.') or 
                   col_str == QUERY_RESULT_ENTITY or 
                   col_str.startswith('unknown') or
                   col_str == 'unknown'):
                filtered_columns.append(col)
        
        return filtered_columns
    
    def _is_table_alias_match(self, alias: str, table_name: str) -> bool:
        """Check if an alias matches a table name using common alias patterns."""
        if not alias or not table_name:
            return False
        
        alias_lower = alias.lower()
        table_lower = table_name.lower()
        
        # Common alias patterns:
        # 1. First letter (table_name -> t)
        if len(alias_lower) == 1 and table_lower.startswith(alias_lower):
            return True
        
        # 2. First few characters (table_name -> tabl)
        if len(alias_lower) <= 4 and table_lower.startswith(alias_lower):
            return True
        
        # 3. Abbreviations (customer_orders -> co, user_profiles -> up)
        if '_' in table_lower:
            parts = table_lower.split('_')
            abbrev = ''.join(part[0] for part in parts if part)
            if alias_lower == abbrev:
                return True
        
        return False
    
    def _is_cte(self, table_name: str) -> bool:
        """Check if table name represents a CTE."""
        # Simple heuristic - CTEs typically don't have schema prefixes
        # and often contain underscores or are descriptive names
        return ('.' not in table_name and 
                ('_' in table_name or 
                 table_name.lower() not in ['table', 'view', 'temp']))  # Avoid common reserved words
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name for display."""
        # Remove table prefixes for cleaner display
        if '.' in column_name:
            parts = column_name.split('.')
            return parts[-1]  # Return just the column name
        return column_name
    
    def _create_missing_cte_transformations(self, chains: Dict) -> List[Dict]:
        """Create missing transformations for CTE dependencies that lack explicit transformations."""
        missing_transformations = []
        existing_targets = set()
        
        # First, collect all existing transformation targets
        for entity_name, entity_data in chains.items():
            def collect_targets_recursive(data):
                transformations = data.get('transformations', [])
                for trans in transformations:
                    target = trans.get('target_table')
                    if target:
                        existing_targets.add(target)
                
                dependencies = data.get('dependencies', [])
                for dep in dependencies:
                    collect_targets_recursive(dep)
            
            collect_targets_recursive(entity_data)
        
        # Find CTEs that are dependency targets but don't have transformations
        def find_missing_connections(parent_name, parent_data):
            dependencies = parent_data.get('dependencies', [])
            for dep in dependencies:
                dep_name = dep.get('entity', '')
                dep_type = dep.get('entity_type', '')
                
                # If this is a CTE dependency but no transformation creates it
                if dep_type == 'cte' and dep_name not in existing_targets:
                    # Try to infer transformation details from dependency metadata
                    transformation_details = self._infer_cte_transformation_details(parent_name, dep_name, dep)
                    
                    # Create an implicit transformation with inferred details
                    missing_transformations.append({
                        'transformation': {
                            'type': 'table_transformation',
                            'source_table': parent_name,
                            'target_table': dep_name,
                            'filter_conditions': transformation_details.get('filter_conditions', []),
                            'group_by_columns': transformation_details.get('group_by_columns', []),
                            'joins': transformation_details.get('joins', [])
                        },
                        'entity': dep_name,
                        'type': 'cte',
                        'source_tables': [parent_name]
                    })
                    existing_targets.add(dep_name)  # Prevent duplicates
                
                # Recursively check deeper dependencies
                find_missing_connections(dep_name, dep)
        
        # Check all chains for missing CTE connections
        for entity_name, entity_data in chains.items():
            find_missing_connections(entity_name, entity_data)
        
        return missing_transformations

    def _infer_cte_transformation_details(self, source_name: str, target_name: str, target_data: Dict) -> Dict:
        """Infer transformation details for missing CTE connections based on metadata."""
        details = {
            'filter_conditions': [],
            'group_by_columns': [],
            'joins': []
        }
        
        # Look at the target CTE's metadata to infer what transformation might have occurred
        metadata = target_data.get('metadata', {})
        table_columns = metadata.get('table_columns', [])
        
        # Check for computed columns which suggest transformations
        computed_columns = []
        for col in table_columns:
            transformation = col.get('transformation', {})
            if transformation:
                transformation_type = transformation.get('transformation_type', '')
                function_type = transformation.get('function_type', '')
                source_expression = transformation.get('source_expression', '')
                
                if transformation_type == TRANSFORMATION_TYPES['CASE']:
                    # This suggests a CASE statement transformation
                    details['filter_conditions'].append({
                        'type': COLUMN_TYPES['COMPUTED'], 
                        'description': f"CASE statement for {col.get('name', 'column')}"
                    })
                elif transformation_type in [TRANSFORMATION_TYPES['COMPUTED'], TRANSFORMATION_TYPES['AGGREGATE']]:
                    # Avoid double function names like COUNT(COUNT(*))
                    if function_type and source_expression and f"{function_type}(" in source_expression.upper():
                        computed_columns.append(source_expression)
                    else:
                        computed_columns.append(f"{function_type}({source_expression})")
        
        # If we found computed columns, add them as a note
        if computed_columns:
            details['filter_conditions'].append({
                'type': COLUMN_TYPES['COMPUTED'],
                'description': f"Computed: {', '.join(computed_columns[:2])}"
            })
        
        # If no specific details found, add a generic note
        if not details['filter_conditions']:
            details['filter_conditions'].append({
                'type': COLUMN_TYPES['SELECT'],
                'description': f"SELECT from {source_name}"
            })
        
        return details

    def _extract_column_transformations(self, target_table: str, chains: Dict) -> Dict:
        """Extract column-level transformation information from target entity metadata."""
        result = {
            'has_case': False,
            'has_computed': False,
            'has_aggregation': False,
            'details': []
        }
        
        # Find the target entity in chains
        target_entity_data = None
        
        def find_entity_recursive(entity_name, entity_data):
            matches = []
            
            if entity_name == target_table:
                matches.append(entity_data)
            
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_name = dep.get('entity', '')
                if dep_name == target_table:
                    matches.append(dep)
                
                # Recursively search deeper
                found = find_entity_recursive(dep_name, dep)
                if found:
                    if isinstance(found, list):
                        matches.extend(found)
                    else:
                        matches.append(found)
            
            return matches if matches else None
        
        # Search for all instances of the target entity and find the best one
        all_target_entities = []
        for entity_name, entity_data in chains.items():
            if entity_name == target_table:
                all_target_entities.append(entity_data)
            else:
                found_entities = find_entity_recursive(entity_name, entity_data)
                if found_entities:
                    all_target_entities.extend(found_entities)
        
        # Select the target entity with the most complete metadata (non-empty table_columns)
        target_entity_data = None
        for entity in all_target_entities:
            table_columns = entity.get('metadata', {}).get('table_columns', [])
            if table_columns:  # Prioritize entities with actual column data
                target_entity_data = entity
                break
        
        # If no entity with column data found, use the first one as fallback
        if not target_entity_data and all_target_entities:
            target_entity_data = all_target_entities[0]
        
        if not target_entity_data:
            return result
        
        # Extract column transformations from metadata
        metadata = target_entity_data.get('metadata', {})
        table_columns = metadata.get('table_columns', [])
        
        for col in table_columns:
            transformation = col.get('transformation', {})
            if transformation:
                transformation_type = transformation.get('transformation_type', '')
                function_type = transformation.get('function_type', '')
                source_expression = transformation.get('source_expression', '')
                column_name = col.get('name', 'column')
                
                if transformation_type == TRANSFORMATION_TYPES['CASE']:
                    result['has_case'] = True
                    # Simplify CASE statement display
                    if 'WHEN' in source_expression and 'THEN' in source_expression:
                        result['details'].append(f"CASE ==> {column_name}")
                    else:
                        result['details'].append(f"{source_expression[:50]}... ==> {column_name}")
                
                elif transformation_type in [TRANSFORMATION_TYPES['COMPUTED'], TRANSFORMATION_TYPES['AGGREGATE']]:
                    if function_type in SQL_FUNCTIONS['AGGREGATE_FUNCTIONS']:
                        result['has_aggregation'] = True
                        # Avoid double function names like COUNT(COUNT(*))
                        if function_type and source_expression and f"{function_type}(" in source_expression.upper():
                            result['details'].append(f"{source_expression} ==> {column_name}")
                        else:
                            result['details'].append(f"{function_type}({source_expression}) ==> {column_name}")
                    else:
                        result['has_computed'] = True
                        result['details'].append(f"Computed ==> {column_name}")
        
        return result

    def _merge_config(self, base_config: Dict, user_config: Dict) -> None:
        """Recursively merge user configuration with default configuration."""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return ['png', 'svg', 'pdf', 'jpg', 'jpeg', 'dot', 'ps']
    
    def validate_json_input(self, json_string: str) -> bool:
        """Validate that the input is valid JSON with required structure."""
        try:
            data = json.loads(json_string)
            required_keys = ['chain_type', 'chains']
            return all(key in data for key in required_keys)
        except (json.JSONDecodeError, TypeError):
            return False


# Example usage and configuration
DEFAULT_VISUALIZATION_CONFIG = {
    'node_style': {
        'table': {
            'fillcolor': '#E3F2FD',
            'color': '#1976D2'
        },
        'column': {
            'fillcolor': '#F3E5F5', 
            'color': '#7B1FA2'
        }
    },
    'graph_attributes': {
        'size': '16,12',
        'dpi': '300'
    }
}


def create_lineage_visualization(table_chain_json: str,
                               column_chain_json: Optional[str] = None,
                               output_path: str = "sql_lineage",
                               output_format: str = "png",
                               show_columns: bool = True,
                               layout: str = "horizontal",
                               sql_query: Optional[str] = None) -> str:
    """
    Convenience function to create lineage visualization.
    
    Args:
        table_chain_json: JSON string of table lineage chain
        column_chain_json: Optional JSON string of column lineage chain
        output_path: Output file path (without extension)
        output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
        show_columns: Whether to include column lineage
        layout: Layout direction ('vertical' or 'horizontal')
        sql_query: Optional SQL query text to display at the top
        
    Returns:
        Path to generated visualization file
    """
    visualizer = SQLLineageVisualizer()
    return visualizer.create_lineage_diagram(
        table_chain_json=table_chain_json,
        column_chain_json=column_chain_json,
        output_path=output_path,
        output_format=output_format,
        show_columns=show_columns,
        layout=layout,
        sql_query=sql_query
    )


def create_lineage_chain_visualization(lineage_chain_json: str,
                                     output_path: str = "lineage_chain",
                                     output_format: str = "png",
                                     layout: str = "horizontal") -> str:
    """
    Convenience function to create lineage chain visualization from comprehensive lineage chain JSON.
    
    Args:
        lineage_chain_json: JSON string from get_lineage_chain_json function
        output_path: Output file path (without extension)
        output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
        layout: Layout direction ('horizontal' or 'vertical')
        
    Returns:
        Path to generated visualization file
    """
    visualizer = SQLLineageVisualizer()
    return visualizer.create_lineage_chain_diagram(
        lineage_chain_json=lineage_chain_json,
        output_path=output_path,
        output_format=output_format,
        layout=layout
    )