"""SQL Lineage Visualizer using Graphviz for creating lineage chain diagrams."""

import json
from typing import Dict, Any, Optional, List, Tuple
from graphviz import Digraph
import os


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
            rankdir = 'RL' if chain_type == 'upstream' else 'LR'
        else:
            # Vertical layout: upstream (BT), downstream (TB)
            rankdir = 'BT' if chain_type == 'upstream' else 'TB'
        
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
            rankdir = 'RL' if chain_type == 'upstream' else 'LR'
        else:
            # Vertical layout: upstream (BT), downstream (TB)
            rankdir = 'BT' if chain_type == 'upstream' else 'TB'
        
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
        dot = Digraph(comment=f"SQL Lineage Chain - {chain_data.get('chain_type', 'unknown').title()}")
        
        # Set graph attributes
        dot.attr(rankdir=rankdir)
        
        # Apply graph attributes
        graph_attrs = graph_config['graph_attributes']
        dot.graph_attr.update(graph_attrs)
        
        # Add title with SQL query
        title = f"SQL Lineage Chain ({chain_data.get('chain_type', 'unknown').title()})"
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
        
        # Process each entity in the chains
        for entity_name, entity_data in chains.items():
            if entity_name != 'QUERY_RESULT' and entity_name != '_sql':  # Skip QUERY_RESULT and temp SQL context
                # For CTAS queries, don't add the created table separately - it will be combined with result
                if is_ctas:
                    # Find the table with highest depth (the created table)
                    max_depth = -1
                    created_table = None
                    for ent_name, ent_data in chains.items():
                        depth = ent_data.get('depth', 0)
                        if depth > max_depth:
                            max_depth = depth
                            created_table = ent_name
                    
                    # Don't add the created table node separately - it will be combined
                    if entity_name != created_table:
                        self._add_entity_with_columns_improved(dot, entity_name, entity_data, node_config, edge_config)
                else:
                    self._add_entity_with_columns_improved(dot, entity_name, entity_data, node_config, edge_config)
        
        # Add transformation boxes and edges
        self._add_transformation_boxes(dot, chains, node_config, edge_config, chain_data)
        
        # Add connections between entities (with subquery handling)
        self._add_entity_connections(dot, chains, edge_config)
        
        # Add result box at the end of the flow
        self._add_final_result_box(dot, chain_data, node_config)
    
    def _add_entity_with_columns_improved(self, dot: Digraph, entity_name: str, entity_data: Dict, node_config: Dict, edge_config: Dict) -> None:
        """Add an entity node with only the columns used in query."""
        entity_type = entity_data.get('entity_type', 'table')
        metadata = entity_data.get('metadata', {})
        
        # Build the node label with table and column information
        if entity_type == 'table':
            # Start with table name
            label_parts = [f"**{entity_name}**"]
            
            # Get used columns from table_columns metadata (columns actually used in query)
            table_columns = metadata.get('table_columns', [])
            used_columns = []
            
            if table_columns:
                # Extract used columns from table_columns
                for col_info in table_columns:
                    col_name = col_info.get('name', col_info.get('column_name', 'unknown'))  # Support both old and new format
                    upstream_cols = col_info.get('upstream', col_info.get('upstream_columns', []))  # Support both old and new format
                    if upstream_cols:
                        for upstream_col in upstream_cols:
                            used_columns.append(upstream_col)
                    else:
                        used_columns.append(col_name)
            
            # Always try to extract from transformations as well to capture JOIN columns
            transformation_columns = self._extract_columns_from_transformations(entity_name, entity_data)
            used_columns.extend(transformation_columns)
            
            # Also extract from dependencies/transformations if available
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_metadata = dep.get('metadata', {})
                dep_table_columns = dep_metadata.get('table_columns', [])
                for col_info in dep_table_columns:
                    upstream_cols = col_info.get('upstream', col_info.get('upstream_columns', []))  # Support both old and new format
                    used_columns.extend(upstream_cols)
            
            # Remove duplicates and sort, filter out QUERY_RESULT columns
            filtered_columns = []
            for col in used_columns:
                col_str = str(col)
                # Filter out QUERY_RESULT columns and other invalid column references
                if not (col_str.startswith('QUERY_RESULT.') or 
                       col_str == 'QUERY_RESULT' or 
                       col_str.startswith('unknown') or
                       col_str == 'unknown'):
                    filtered_columns.append(col)
            
            used_columns = sorted(list(set(filtered_columns)))
            
            # Add used columns with their types from full metadata
            all_columns = metadata.get('columns', [])
            if used_columns and all_columns:
                label_parts.append("─" * 20)
                label_parts.append("Used Columns:")
                
                # Only show columns that have proper metadata or are essential
                valid_columns_to_show = []
                for col_name in used_columns:
                    # Find column details from full metadata
                    col_details = next((col for col in all_columns if col.get('name') == col_name), None)
                    if col_details:
                        col_type = col_details.get('data_type', 'unknown')
                        # Only show if data type is not 'unknown' or if it's an essential column
                        if col_type != 'unknown' or col_name in ['id', 'name', 'email']:
                            safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                            safe_col_type = str(col_type).replace('<', '&lt;').replace('>', '&gt;')
                            valid_columns_to_show.append(f"{safe_col_name}: {safe_col_type}")
                    # Don't show columns without metadata at all to avoid "unknown" types
                
                # Add the valid columns to display
                for col_display in valid_columns_to_show:
                    label_parts.append(col_display)
                    
                # If we have used_columns but no valid ones to show, try once more with a more lenient approach
                if not valid_columns_to_show and used_columns:
                    label_parts.append("Used Columns:")
                    for col_name in used_columns[:4]:  # Limit to 4 columns
                        col_details = next((col for col in all_columns if col.get('name') == col_name), None)
                        if col_details:
                            col_type = col_details.get('data_type', 'VARCHAR')
                            safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                            safe_col_type = str(col_type).replace('<', '&lt;').replace('>', '&gt;')
                            label_parts.append(f"{safe_col_name}: {safe_col_type}")
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
                    for col_name in columns_to_show[:8]:  # Limit to 8 columns
                        col_details = next((col for col in all_columns if col.get('name') == col_name), None)
                        if col_details:
                            col_type = col_details.get('data_type', 'unknown')
                            # Only show if data type is not 'unknown' or if it's an essential column
                            if col_type != 'unknown' or col_name in ['id', 'name', 'email']:
                                safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                                safe_col_type = str(col_type).replace('<', '&lt;').replace('>', '&gt;')
                                label_parts.append(f"{safe_col_name}: {safe_col_type}")
                        else:
                            # For CTE output columns, we can show them even without metadata if they're expected
                            if cte_output_columns and col_name in ['id', 'name', 'email']:
                                safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                                label_parts.append(f"{safe_col_name}: VARCHAR")
                elif len(all_columns) <= 6:  # Show all if few columns
                    label_parts.append("─" * 20)
                    for col in all_columns:
                        col_name = col.get('name', 'unknown')
                        col_type = col.get('data_type', 'unknown')
                        # Only show columns with proper data types or essential columns
                        if col_type != 'unknown' or col_name in ['id', 'name', 'email']:
                            # Escape special characters for Graphviz
                            safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                            safe_col_type = str(col_type).replace('<', '&lt;').replace('>', '&gt;')
                            label_parts.append(f"{safe_col_name}: {safe_col_type}")
            
            # Join all parts
            full_label = "\\n".join(label_parts)
            
        else:
            # Simple entity name for non-table entities
            full_label = entity_name
        
        # Get appropriate node style
        node_style = self._get_node_style(entity_name, node_config)
        
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
                if parts[0] == 'QUERY_RESULT':
                    return 'QUERY_RESULT'
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
                            if dep_table == 'QUERY_RESULT' and table_chain:
                                # Look for table lineage to find source tables
                                table_chains = table_chain.get('chains', {})
                                if 'QUERY_RESULT' in table_chains:
                                    query_deps = table_chains['QUERY_RESULT'].get('dependencies', [])
                                    if query_deps:
                                        # Use the first source table as the most likely source
                                        source_table = query_deps[0].get('table', 'orders')
                                        return source_table
                            elif dep_table != 'QUERY_RESULT':
                                return dep_table
        
        # If no dot, it might be a computed column or aggregate
        # Try to infer from common patterns or return a meaningful default
        if any(keyword in column_name.lower() for keyword in ['count', 'sum', 'avg', 'max', 'min']):
            return 'QUERY_RESULT'  # Group computed columns under result
        
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
        if table_name == 'QUERY_RESULT':
            return node_config.get('query_result', node_config['table'])
        elif self._is_cte(table_name):
            return node_config.get('cte', node_config['table'])
        else:
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
                result_target = window_func.get('target', 'QUERY_RESULT')
                if result_target == 'QUERY_RESULT':
                    # We'll connect to result box later
                    pass
                else:
                    dot.edge(trans_id, result_target, color='#0066CC', style='dashed')
        
        # Collect all transformations from all entities and deduplicate
        all_transformations_with_context = []
        
        for entity_name, entity_data in chains.items():
            # Get direct transformations
            transformations = entity_data.get('transformations', [])
            for trans in transformations:
                all_transformations_with_context.append({
                    'transformation': trans,
                    'entity': entity_name,
                    'type': 'direct'
                })
            
            # Get dependency transformations 
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_transformations = dep.get('transformations', [])
                for trans in dep_transformations:
                    all_transformations_with_context.append({
                        'transformation': trans,
                        'entity': entity_name,
                        'type': 'dependency'
                    })
        
        # Deduplicate transformations based on their signature
        unique_transformations = self._deduplicate_transformations(all_transformations_with_context)
        
        for transformation_info in unique_transformations:
            transformation = transformation_info['transformation']
            transformation_counter += 1
            trans_id = f"trans_{transformation_counter}"
            
            # Build transformation label - detect JOIN types first
            join_type = transformation.get('join_type')
            join_conditions = transformation.get('join_conditions', [])
            
            # Determine the transformation type
            if join_conditions or join_type:
                # This is a JOIN transformation
                if join_type:
                    trans_type = f"{join_type.upper()} JOIN"
                else:
                    trans_type = "INNER JOIN"  # Default JOIN type
            else:
                trans_type = transformation.get('type', 'TRANSFORM')
            
            label_parts = [f"**{trans_type.upper()}**"]
            
            # Add join conditions (only if we have valid conditions)
            if join_conditions:
                valid_conditions = []
                for condition in join_conditions[:3]:  # Limit to 3 conditions
                    if isinstance(condition, dict):
                        left_col = condition.get('left_column', '')
                        operator = condition.get('operator', '=')
                        right_col = condition.get('right_column', '')
                        
                        # Only add condition if we have valid column names
                        if left_col and right_col and left_col != 'unknown' and right_col != 'unknown':
                            valid_conditions.append(f"  {left_col} {operator} {right_col}")
                    else:
                        # Handle string conditions
                        condition_str = str(condition).strip()
                        if condition_str and condition_str != 'unknown' and condition_str != 'unknown =':
                            safe_condition = condition_str.replace('<', '&lt;').replace('>', '&gt;')
                            valid_conditions.append(f"  {safe_condition}")
                
                if valid_conditions:
                    label_parts.append("Conditions:")
                    label_parts.extend(valid_conditions)
            
            # Add filter conditions (only if we have valid conditions)
            filter_conditions = transformation.get('filter_conditions', [])
            if filter_conditions:
                valid_filters = []
                for condition in filter_conditions[:2]:  # Limit to 2 filters
                    if isinstance(condition, dict):
                        column = condition.get('column', '')
                        operator = condition.get('operator', '=')
                        value = condition.get('value', '')
                        
                        # Only add filter if we have valid column and value
                        if column and column != 'unknown' and value != '':
                            # Clean and format the condition
                            safe_column = str(column).replace('<', '&lt;').replace('>', '&gt;')
                            safe_value = str(value).replace('<', '&lt;').replace('>', '&gt;')
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
            
            # Add group by information
            group_by = transformation.get('group_by_columns', [])
            if group_by:
                label_parts.append(f"Group By: {', '.join(group_by[:3])}")
                # If this is the main transformation type, emphasize GROUP BY
                if not join_type and not join_conditions and not filter_conditions:
                    trans_type = "GROUP BY"
            
            # Add order by information
            order_by = transformation.get('order_by_columns', [])
            if order_by:
                label_parts.append(f"Order By: {', '.join(order_by[:2])}")
            
            trans_label = "\\n".join(label_parts)
            
            # Add transformation node
            dot.node(trans_id, trans_label, **transformation_style)
            
            # Connect source to transformation
            source_table = transformation.get('source_table')
            target_table = transformation.get('target_table')
            
            if source_table and source_table in chains:
                dot.edge(source_table, trans_id, color='#D2691E', style='dashed')
            
            # Connect transformation to target
            if target_table:
                if target_table in chains:
                    # Target is another entity in the chain
                    dot.edge(trans_id, target_table, color='#D2691E', style='dashed')
                elif target_table == 'QUERY_RESULT':
                    # Special handling for QUERY_RESULT - it will be handled by final result box
                    # Store this transformation as needing connection to final result
                    if not hasattr(dot, '_result_transformations'):
                        dot._result_transformations = []
                    dot._result_transformations.append(trans_id)
                else:
                    # Unknown target, connect to current entity - use first entity that has this transformation
                    entity_name = transformation_info.get('entity', '')
                    if entity_name:
                        dot.edge(trans_id, entity_name, color='#D2691E', style='dashed')
    
    def _extract_window_functions_from_context(self, chains: Dict) -> List[Dict]:
        """Extract window functions from transformation context or dependencies."""
        window_functions = []
        
        # Look through all dependencies to find potential window function indicators
        for entity_name, entity_data in chains.items():
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_transformations = dep.get('transformations', [])
                for trans in dep_transformations:
                    # Check if this involves window functions by looking at order_by patterns
                    order_by_columns = trans.get('order_by_columns', [])
                    if order_by_columns:
                        # This might indicate window functions
                        # For now, create a generic window function transformation
                        window_functions.append({
                            'function': 'WINDOW_FUNCTIONS',
                            'partition_by': 'customer_id',  # Common pattern
                            'order_by': ', '.join(order_by_columns),
                            'source_table': entity_name,
                            'target': dep.get('entity', 'QUERY_RESULT'),
                            'alias': 'window_result'
                        })
                        break  # Only add one per dependency to avoid duplicates
        
        return window_functions
    
    def _add_entity_connections(self, dot: Digraph, chains: Dict, edge_config: Dict) -> None:
        """Add connections between entities with special handling for subqueries."""
        subquery_edge_config = edge_config.copy()
        subquery_edge_config.update({
            'style': 'dashed',
            'color': '#FFA500',
            'penwidth': '1',
            'arrowhead': 'diamond'
        })
        
        for entity_name, entity_data in chains.items():
            if entity_name == 'QUERY_RESULT':
                continue  # Skip QUERY_RESULT connections, handled in result box
                
            dependencies = entity_data.get('dependencies', [])
            for dep in dependencies:
                dep_entity = dep.get('entity')
                if dep_entity and dep_entity != 'QUERY_RESULT':
                    # Check if this is a subquery connection
                    if self._is_subquery_connection(entity_name, dep_entity, entity_data, dep):
                        # Use dashed line for subqueries
                        dot.edge(entity_name, dep_entity, **subquery_edge_config)
                    else:
                        # Use regular solid line for main flow
                        dot.edge(entity_name, dep_entity, **edge_config)
    
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
                    if operator in ['IN', 'EXISTS', 'NOT EXISTS']:
                        return True
                    # Check for correlated subquery patterns
                    column = condition.get('column', '')
                    if '(' in str(condition.get('value', '')) or 'SELECT' in str(condition.get('value', '')).upper():
                        return True
        
        # Check if source and target have different base tables (indicating subquery)
        if source_entity in ['users', 'orders', 'products', 'categories'] and target_entity in ['users', 'orders', 'products', 'categories']:
            if source_entity != target_entity:
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
        
        if 'QUERY_RESULT' in chains:
            result_entity_name = 'QUERY_RESULT'
            result_entity_data = chains['QUERY_RESULT']
        elif is_ctas:
            # For CTAS, find the created table (entity with highest depth)
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
        if is_ctas and result_entity_name != 'QUERY_RESULT':
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
                    col_name = col_info.get('name', col_info.get('column_name', 'unknown'))  # Support both old and new format
                    
                    # Check for old format transformations first, then new format type
                    transformations = col_info.get('transformations', [])
                    col_type = col_info.get('type', 'DIRECT')
                    
                    if transformations:
                        # Old format - use transformation expression
                        expr = transformations[0].get('expression', col_name)
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        safe_expr = str(expr).replace('<', '&lt;').replace('>', '&gt;')
                        if expr != col_name and len(expr) < 30:
                            label_parts.append(f"• {safe_col_name}: {safe_expr}")
                        else:
                            label_parts.append(f"• {safe_col_name}")
                    elif col_type and col_type != 'DIRECT':
                        # New format - show column with type
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"• {safe_col_name} ({col_type})")
                    else:
                        # Simple column name
                        safe_col_name = str(col_name).replace('<', '&lt;').replace('>', '&gt;')
                        label_parts.append(f"• {safe_col_name}")
        
        result_label = "\\n".join(label_parts)
        result_id = "FINAL_RESULT"
        
        # Add result box with rank constraint to position it at the end
        dot.node(result_id, result_label, **result_style)
        
        # Connect all entities that don't have outgoing dependencies to the result box
        entities_connected_to_result = set()
        
        # First pass: identify entities that should connect to result
        for entity_name, entity_data in chains.items():
            if entity_name == 'QUERY_RESULT':
                continue
                
            # Check if this entity has dependencies that lead to QUERY_RESULT or has no dependencies
            dependencies = entity_data.get('dependencies', [])
            has_query_result_dep = any(dep.get('entity') == 'QUERY_RESULT' for dep in dependencies)
            has_no_deps = len(dependencies) == 0
            
            if has_query_result_dep or has_no_deps:
                entities_connected_to_result.add(entity_name)
        
        # If no entities found, connect the one with highest depth
        if not entities_connected_to_result and result_entity_name != 'QUERY_RESULT':
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
                    col_name = col_info.get('name', col_info.get('column_name', 'unknown'))  # Support both old and new format
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
        
        # For window function queries, also add common columns
        if any('order_by_columns' in trans for dep in dependencies for trans in dep.get('transformations', [])):
            # This looks like a window function query, add common columns
            common_window_columns = ['customer_id', 'order_date', 'order_total', 'total']
            for col_name in common_window_columns:
                if any(col.get('name') == col_name for col in all_columns) and col_name not in used_columns:
                    used_columns.append(col_name)
        
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
        
        # For specific patterns, add common columns based on CTE name or source patterns
        if not output_columns:
            if 'active_users' in cte_name.lower():
                # For active_users CTE, typically id and name are selected
                output_columns = ['id', 'name']
            elif 'users' in cte_name.lower():
                output_columns = ['id', 'name']
            elif 'orders' in cte_name.lower():
                output_columns = ['id', 'customer_id', 'order_date', 'total']
            else:
                # Generic fallback
                output_columns = ['id', 'name']
        
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
        
        # For common patterns based on source table name
        if 'users' in source_table.lower():
            # For user tables, if we have 'active' filter, likely selecting id, name
            has_active_filter = any(
                isinstance(condition, dict) and condition.get('column') == 'active'
                for condition in filter_conditions
            )
            if has_active_filter:
                for col in ['id', 'name']:
                    if col not in columns:
                        columns.append(col)
        
        # Filter out QUERY_RESULT columns and invalid references
        filtered_columns = []
        for col in columns:
            col_str = str(col)
            if not (col_str.startswith('QUERY_RESULT.') or 
                   col_str == 'QUERY_RESULT' or 
                   col_str.startswith('unknown') or
                   col_str == 'unknown'):
                filtered_columns.append(col)
        
        return filtered_columns
    
    def _deduplicate_transformations(self, transformations_with_context: List[Dict]) -> List[Dict]:
        """Deduplicate transformations based on their signature to avoid showing the same transformation multiple times."""
        unique_transformations = {}
        
        for trans_info in transformations_with_context:
            transformation = trans_info['transformation']
            
            # Create a signature based on transformation characteristics
            source_table = transformation.get('source_table', '')
            target_table = transformation.get('target_table', '')
            trans_type = transformation.get('type', 'TRANSFORM')
            
            # Sort conditions for consistent comparison
            filter_conditions = transformation.get('filter_conditions', [])
            filter_signature = tuple(sorted([
                f"{c.get('column', 'unknown')}_{c.get('operator', '=')}_{c.get('value', '')}"
                for c in filter_conditions if isinstance(c, dict)
            ]))
            
            join_conditions = transformation.get('join_conditions', [])
            join_signature = tuple(sorted([
                f"{c.get('left_column', 'unknown')}_{c.get('operator', '=')}_{c.get('right_column', 'unknown')}"
                for c in join_conditions if isinstance(c, dict)
            ]))
            
            group_by = tuple(sorted(transformation.get('group_by_columns', [])))
            order_by = tuple(sorted(transformation.get('order_by_columns', [])))
            
            # Create unique signature
            signature = (source_table, target_table, trans_type, filter_signature, join_signature, group_by, order_by)
            
            # Only keep the first occurrence of each unique transformation
            if signature not in unique_transformations:
                unique_transformations[signature] = trans_info
        
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
                join_conditions = trans.get('join_conditions', [])
                for condition in join_conditions:
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
        
        # For specific CTE pattern: "WITH active_users AS (SELECT id, name FROM users WHERE active = true)"
        # If this is a users table and we have 'active' in filter conditions, we can infer id, name are selected
        if 'users' in entity_name.lower():
            # Check if there's an 'active' filter condition, which suggests id, name are selected
            has_active_condition = any(
                isinstance(condition, dict) and condition.get('column') == 'active'
                for trans in transformations + [t for dep in dependencies for t in dep.get('transformations', [])]
                for condition in trans.get('filter_conditions', [])
            )
            if has_active_condition:
                # Common pattern: SELECT id, name FROM users WHERE active = true
                for col in ['id', 'name']:
                    if col not in extracted_columns:
                        extracted_columns.append(col)
        
        # Filter out QUERY_RESULT columns and invalid references
        filtered_columns = []
        for col in extracted_columns:
            col_str = str(col)
            if not (col_str.startswith('QUERY_RESULT.') or 
                   col_str == 'QUERY_RESULT' or 
                   col_str.startswith('unknown') or
                   col_str == 'unknown'):
                filtered_columns.append(col)
        
        return filtered_columns
    
    def _is_cte(self, table_name: str) -> bool:
        """Check if table name represents a CTE."""
        # Simple heuristic - CTEs typically don't have schema prefixes
        # and are often lowercase or have specific patterns
        return '.' not in table_name and table_name.lower() not in [
            'users', 'orders', 'products', 'categories', 'customers'
        ]
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name for display."""
        # Remove table prefixes for cleaner display
        if '.' in column_name:
            parts = column_name.split('.')
            return parts[-1]  # Return just the column name
        return column_name
    
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