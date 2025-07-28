"""Rich console output formatter."""

from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from ..core.models import LineageResult


class ConsoleFormatter:
    """Formats lineage results for rich console output."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize console formatter.
        
        Args:
            console: Rich console instance (creates new one if None)
        """
        self.console = console or Console()
    
    def format(self, result: LineageResult) -> None:
        """
        Format and print lineage result to console.
        
        Args:
            result: LineageResult to format
        """
        self.console.print()
        self._print_header(result)
        
        if result.has_errors():
            self._print_errors(result)
            return
        
        if result.has_warnings():
            self._print_warnings(result)
        
        self._print_table_lineage(result)
        self._print_column_lineage(result)
        
        if result.metadata:
            self._print_metadata(result)
    
    def _print_header(self, result: LineageResult) -> None:
        """Print analysis header."""
        title = Text("SQL Lineage Analysis Results", style="bold blue")
        panel = Panel.fit(
            f"[bold]Query:[/bold] {result.sql[:100]}{'...' if len(result.sql) > 100 else ''}\n"
            f"[bold]Dialect:[/bold] {result.dialect}",
            title=title,
            border_style="blue"
        )
        self.console.print(panel)
    
    def _print_errors(self, result: LineageResult) -> None:
        """Print errors."""
        error_text = "\n".join(f"• {error}" for error in result.errors)
        panel = Panel(
            error_text,
            title="[red]Errors[/red]",
            border_style="red"
        )
        self.console.print(panel)
    
    def _print_warnings(self, result: LineageResult) -> None:
        """Print warnings."""
        warning_text = "\n".join(f"• {warning}" for warning in result.warnings)
        panel = Panel(
            warning_text,
            title="[yellow]Warnings[/yellow]",
            border_style="yellow"
        )
        self.console.print(panel)
    
    def _print_table_lineage(self, result: LineageResult) -> None:
        """Print table lineage information."""
        self.console.print("\n[bold blue]📊 TABLE LINEAGE[/bold blue]")
        
        if not result.table_lineage.upstream and not result.table_lineage.downstream:
            self.console.print("  [dim]No table lineage found[/dim]")
            return
        
        # Upstream lineage
        if result.table_lineage.upstream:
            self.console.print("\n[bold]Upstream Dependencies (Target ← Sources):[/bold]")
            tree = Tree("🎯 Targets")
            
            for target in sorted(result.table_lineage.upstream.keys()):
                sources = result.table_lineage.upstream[target]
                target_node = tree.add(f"[green]{target}[/green]")
                
                if sources:
                    for source in sorted(sources):
                        target_node.add(f"[blue]← {source}[/blue]")
                else:
                    target_node.add("[dim]← No dependencies[/dim]")
            
            self.console.print(tree)
        
        # Downstream lineage
        if result.table_lineage.downstream:
            self.console.print("\n[bold]Downstream Dependencies (Source → Targets):[/bold]")
            tree = Tree("📦 Sources")
            
            for source in sorted(result.table_lineage.downstream.keys()):
                targets = result.table_lineage.downstream[source]
                source_node = tree.add(f"[blue]{source}[/blue]")
                
                if targets:
                    for target in sorted(targets):
                        source_node.add(f"[green]→ {target}[/green]")
                else:
                    source_node.add("[dim]→ No dependents[/dim]")
            
            self.console.print(tree)
    
    def _print_column_lineage(self, result: LineageResult) -> None:
        """Print column lineage information."""
        self.console.print("\n[bold blue]🔍 COLUMN LINEAGE[/bold blue]")
        
        if not result.column_lineage.upstream and not result.column_lineage.downstream:
            self.console.print("  [dim]No column lineage found[/dim]")
            return
        
        # Group by table for better readability
        if result.column_lineage.upstream:
            self.console.print("\n[bold]Column Dependencies:[/bold]")
            
            # Group columns by table
            table_columns = {}
            for target_col in result.column_lineage.upstream.keys():
                if '.' in target_col:
                    table = target_col.split('.', 1)[0]
                    col = target_col.split('.', 1)[1]
                else:
                    table = "unknown"
                    col = target_col
                
                if table not in table_columns:
                    table_columns[table] = []
                table_columns[table].append((target_col, col))
            
            tree = Tree("🎯 Target Tables")
            for table in sorted(table_columns.keys()):
                table_node = tree.add(f"[green]{table}[/green]")
                
                for target_col, col_name in sorted(table_columns[table]):
                    sources = result.column_lineage.upstream[target_col]
                    col_node = table_node.add(f"[yellow]{col_name}[/yellow]")
                    
                    if sources:
                        for source in sorted(sources):
                            col_node.add(f"[blue]← {source}[/blue]")
                    else:
                        col_node.add("[dim]← No dependencies[/dim]")
            
            self.console.print(tree)
    
    def _print_metadata(self, result: LineageResult) -> None:
        """Print table metadata information."""
        self.console.print("\n[bold blue]📋 TABLE METADATA[/bold blue]")
        
        for table_name, metadata in sorted(result.metadata.items()):
            table = Table(title=f"[green]{table_name}[/green]", show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Description", metadata.description or "N/A")
            table.add_row("Owner", metadata.owner or "N/A")
            table.add_row("Storage Format", metadata.storage_format or "N/A")
            table.add_row("Row Count", f"{metadata.row_count:,}" if metadata.row_count else "N/A")
            table.add_row("Columns", str(len(metadata.columns)))
            
            self.console.print(table)
            
            # Show column details
            if metadata.columns:
                col_table = Table(title="Columns", show_header=True)
                col_table.add_column("Name", style="yellow")
                col_table.add_column("Type", style="blue")
                col_table.add_column("Nullable", style="green")
                col_table.add_column("Key", style="red")
                col_table.add_column("Description", style="white")
                
                for col in metadata.columns[:10]:  # Show first 10 columns
                    key_info = "PK" if col.primary_key else ("FK" if col.foreign_key else "")
                    col_table.add_row(
                        col.name,
                        col.data_type,
                        "Yes" if col.nullable else "No",
                        key_info,
                        col.description or ""
                    )
                
                if len(metadata.columns) > 10:
                    col_table.add_row(
                        "[dim]...[/dim]",
                        f"[dim]({len(metadata.columns) - 10} more columns)[/dim]",
                        "", "", ""
                    )
                
                self.console.print(col_table)
            
            self.console.print()
    
    def format_compact(self, result: LineageResult) -> None:
        """Format lineage result in compact form."""
        if result.has_errors():
            self.console.print(f"[red]Error:[/red] {'; '.join(result.errors)}")
            return
        
        # Compact table lineage
        if result.table_lineage.upstream:
            self.console.print("[bold]Table Lineage:[/bold]")
            for target, sources in result.table_lineage.upstream.items():
                sources_str = ", ".join(sorted(sources)) if sources else "None"
                self.console.print(f"  {target} ← {sources_str}")
        
        # Compact column lineage summary
        if result.column_lineage.upstream:
            col_count = len(result.column_lineage.upstream)
            self.console.print(f"[bold]Column Dependencies:[/bold] {col_count} columns")