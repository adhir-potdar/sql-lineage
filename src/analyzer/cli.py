"""Command-line interface for SQL lineage analyzer."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from .core.analyzer import SQLLineageAnalyzer
from .formatters.json_formatter import JSONFormatter
from .formatters.console_formatter import ConsoleFormatter
from .utils.validation import validate_dialect, validate_file_path
from .utils.logging_config import get_logger


@click.group()
@click.version_option(version="1.0.0", prog_name="sql-lineage")
def cli():
    """SQL Lineage Analyzer - Extract upstream and downstream dependencies from SQL queries."""
    logger = get_logger('cli')
    logger.info("SQL Lineage Analyzer CLI started")
    pass


@cli.command()
@click.option(
    '--sql', '-s',
    help='SQL query string to analyze'
)
@click.option(
    '--file', '-f', 'file_path',
    type=click.Path(exists=True, readable=True),
    help='Path to SQL file to analyze'
)
@click.option(
    '--dialect', '-d',
    default='trino',
    help='SQL dialect (default: trino)'
)
@click.option(
    '--output-format', '-o',
    type=click.Choice(['console', 'json', 'compact']),
    default='console',
    help='Output format (default: console)'
)
@click.option(
    '--output-file', '-F',
    type=click.Path(),
    help='Output file path (only for json format)'
)
@click.option(
    '--table-only',
    is_flag=True,
    help='Show only table lineage (not column lineage)'
)
@click.option(
    '--column-only',
    is_flag=True,
    help='Show only column lineage (not table lineage)'
)
def analyze(
    sql: Optional[str],
    file_path: Optional[str],
    dialect: str,
    output_format: str,
    output_file: Optional[str],
    table_only: bool,
    column_only: bool
):
    """Analyze SQL query for lineage information."""
    logger = get_logger('cli.analyze')
    console = Console()
    
    logger.info(f"Starting SQL analysis - dialect: {dialect}, format: {output_format}, table_only: {table_only}, column_only: {column_only}")
    
    # Validate inputs
    if not sql and not file_path:
        logger.error("No input provided: must specify either --sql or --file")
        console.print("[red]Error:[/red] Must provide either --sql or --file")
        sys.exit(1)
    
    if sql and file_path:
        logger.error("Invalid input: both --sql and --file specified")
        console.print("[red]Error:[/red] Cannot specify both --sql and --file")
        sys.exit(1)
    
    # Validate dialect
    dialect_error = validate_dialect(dialect)
    if dialect_error:
        logger.error(f"Invalid dialect: {dialect_error}")
        console.print(f"[red]Error:[/red] {dialect_error}")
        sys.exit(1)
    
    logger.debug(f"Using dialect: {dialect}")
    
    # Validate file path if provided
    if file_path:
        file_error = validate_file_path(file_path)
        if file_error:
            logger.error(f"File validation failed: {file_error}")
            console.print(f"[red]Error:[/red] {file_error}")
            sys.exit(1)
        logger.info(f"Analyzing file: {file_path}")
    else:
        logger.debug(f"Analyzing SQL string (length: {len(sql) if sql else 0})")
    
    # Initialize analyzer
    logger.debug("Initializing SQL lineage analyzer")
    analyzer = SQLLineageAnalyzer(dialect=dialect)
    logger.info(f"Analyzer initialized successfully with dialect: {dialect}")
    
    # Analyze SQL
    try:
        logger.info("Starting SQL analysis")
        if file_path:
            result = analyzer.analyze_file(file_path)
            logger.info(f"File analysis completed for: {file_path}")
        else:
            result = analyzer.analyze(sql)
            logger.info("SQL string analysis completed")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        console.print(f"[red]Error:[/red] Failed to analyze SQL: {e}")
        sys.exit(1)
    
    # Format and output results
    if output_format == 'json':
        formatter = JSONFormatter()
        
        if table_only:
            output = formatter.format_table_lineage_only(result)
        elif column_only:
            output = formatter.format_column_lineage_only(result)
        else:
            output = formatter.format(result)
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(output)
                logger.info(f"Results written to file: {output_file}")
                console.print(f"[green]Results written to:[/green] {output_file}")
            except Exception as e:
                logger.error(f"Failed to write output file {output_file}: {str(e)}")
                console.print(f"[red]Error:[/red] Failed to write output file: {e}")
        else:
            console.print(output)
            logger.debug("Results displayed to console")
    
    elif output_format == 'compact':
        formatter = ConsoleFormatter(console)
        formatter.format_compact(result)
    
    else:  # console format
        formatter = ConsoleFormatter(console)
        
        if table_only:
            formatter._print_header(result)
            if result.has_errors():
                formatter._print_errors(result)
            else:
                formatter._print_table_lineage(result)
        elif column_only:
            formatter._print_header(result)
            if result.has_errors():
                formatter._print_errors(result)
            else:
                formatter._print_column_lineage(result)
        else:
            formatter.format(result)
    
    # Exit with error code if analysis had errors
    if result.has_errors():
        logger.warning(f"Analysis completed with errors: {result.errors}")
        sys.exit(1)
    else:
        logger.info("Analysis completed successfully")


@cli.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    '--pattern', '-p',
    default='*.sql',
    help='File pattern to match (default: *.sql)'
)
@click.option(
    '--dialect', '-d',
    default='trino',
    help='SQL dialect (default: trino)'
)
@click.option(
    '--output-format', '-o',
    type=click.Choice(['console', 'json']),
    default='console',
    help='Output format (default: console)'
)
@click.option(
    '--output-dir', '-O',
    type=click.Path(),
    help='Output directory for results (creates individual files per SQL file)'
)
def batch(
    directory: str,
    pattern: str,
    dialect: str,
    output_format: str,
    output_dir: Optional[str]
):
    """Analyze multiple SQL files in a directory."""
    logger = get_logger('cli.batch')
    console = Console()
    
    logger.info(f"Starting batch analysis - directory: {directory}, pattern: {pattern}, dialect: {dialect}")
    
    # Find SQL files
    dir_path = Path(directory)
    sql_files = list(dir_path.glob(pattern))
    
    if not sql_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {directory}")
        console.print(f"[yellow]Warning:[/yellow] No files matching pattern '{pattern}' found in {directory}")
        return
    
    logger.info(f"Found {len(sql_files)} files matching pattern '{pattern}'")
    
    console.print(f"[blue]Found {len(sql_files)} SQL files to analyze[/blue]")
    
    # Initialize analyzer
    logger.debug("Initializing SQL lineage analyzer for batch processing")
    analyzer = SQLLineageAnalyzer(dialect=dialect)
    logger.info(f"Analyzer initialized for batch processing with dialect: {dialect}")
    
    # Prepare output directory if needed
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory prepared: {output_dir}")
    else:
        logger.debug("No output directory specified, results will go to console")
    
    # Analyze each file
    for sql_file in sql_files:
        console.print(f"\n[bold]Analyzing:[/bold] {sql_file.name}")
        
        try:
            logger.debug(f"Analyzing file: {sql_file}")
            result = analyzer.analyze_file(str(sql_file))
            logger.info(f"Successfully analyzed: {sql_file.name}")
            
            if output_format == 'json':
                formatter = JSONFormatter()
                output = formatter.format(result)
                
                if output_dir:
                    output_file = output_path / f"{sql_file.stem}_lineage.json"
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(output)
                        logger.info(f"Results written to: {output_file}")
                        console.print(f"  [green]Results written to:[/green] {output_file}")
                    except Exception as e:
                        logger.error(f"Failed to write output file {output_file}: {str(e)}")
                        console.print(f"  [red]Error writing file:[/red] {e}")
                else:
                    console.print(output)
                    logger.debug(f"Results for {sql_file.name} displayed to console")
            
            else:  # console format
                formatter = ConsoleFormatter(console)
                formatter.format_compact(result)
                logger.debug(f"Results for {sql_file.name} formatted to console")
        
        except Exception as e:
            logger.error(f"Failed to analyze {sql_file}: {str(e)}", exc_info=True)
            console.print(f"  [red]Error:[/red] {e}")


def main():
    """Main entry point."""
    logger = get_logger('main')
    logger.info("SQL Lineage Analyzer starting")
    try:
        cli()
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("SQL Lineage Analyzer session ended")


if __name__ == '__main__':
    main()