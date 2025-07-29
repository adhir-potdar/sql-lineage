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
                    'shape': 'box',
                    'style': 'filled,bold',
                    'fillcolor': '#E8F5E8',
                    'color': '#388E3C',
                    'fontname': 'Arial Bold',
                    'fontsize': '14',
                    'fontcolor': '#388E3C'
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
    
    def create_lineage_diagram(self, 
                             table_chain_json: str,
                             column_chain_json: Optional[str] = None,
                             output_path: str = "lineage_diagram",
                             output_format: str = "png",
                             config: Optional[Dict] = None,
                             show_columns: bool = True,
                             layout: str = "horizontal") -> str:
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
        dot = self._create_digraph(table_chain, rankdir, config)
        
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
                                layout: str = "horizontal") -> str:
        """
        Create a table-only lineage diagram.
        
        Args:
            table_chain_json: JSON string of table lineage chain
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
            config: Custom configuration dictionary
            layout: Layout direction ('horizontal' or 'vertical', default: 'horizontal')
            
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
            layout=layout
        )
    
    def create_column_focused_diagram(self,
                                    table_chain_json: str,
                                    column_chain_json: str,
                                    output_path: str = "column_lineage",
                                    output_format: str = "png",
                                    config: Optional[Dict] = None,
                                    layout: str = "horizontal") -> str:
        """
        Create a column-focused lineage diagram with tables as containers.
        
        Args:
            table_chain_json: JSON string of table lineage chain
            column_chain_json: JSON string of column lineage chain
            output_path: Output file path (without extension)
            output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
            config: Custom configuration dictionary
            layout: Layout direction ('horizontal' or 'vertical', default: 'horizontal')
            
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
            layout=layout
        )
    
    def _create_digraph(self, table_chain: Dict, rankdir: str, config: Optional[Dict]) -> Digraph:
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
        
        # Add title
        title = f"SQL Lineage Chain ({table_chain.get('chain_type', 'unknown').title()})"
        if table_chain.get('max_depth'):
            title += f" - Depth: {table_chain['max_depth']}"
        
        dot.graph_attr.update({
            'label': title,
            'labelloc': 't',
            'fontsize': '16',
            'fontname': 'Arial Bold'
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
                dot.node(table_name, table_name, **node_style)
        
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
                    table_subgraph.attr(
                        style='filled,rounded',
                        fillcolor='#F8F9FA',
                        color='#6C757D',
                        label=table_name,
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
    
    def _get_node_style(self, table_name: str, node_config: Dict) -> Dict:
        """Get appropriate node style based on table type."""
        if table_name == 'QUERY_RESULT':
            return node_config.get('query_result', node_config['table'])
        elif self._is_cte(table_name):
            return node_config.get('cte', node_config['table'])
        else:
            return node_config['table']
    
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
                               layout: str = "horizontal") -> str:
    """
    Convenience function to create lineage visualization.
    
    Args:
        table_chain_json: JSON string of table lineage chain
        column_chain_json: Optional JSON string of column lineage chain
        output_path: Output file path (without extension)
        output_format: Output format ('png', 'svg', 'pdf', 'jpg', 'jpeg')
        show_columns: Whether to include column lineage
        layout: Layout direction ('vertical' or 'horizontal')
        
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
        layout=layout
    )