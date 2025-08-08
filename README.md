# SQL Lineage Analyzer

A production-quality SQL lineage analysis tool that extracts table and column dependencies from SQL queries using SQLGlot. Supports multiple SQL dialects and provides rich metadata integration capabilities.

## Features

✅ **Multi-Dialect Support**: Trino, PostgreSQL, MySQL, SQLite, and more  
✅ **Table Lineage**: Track upstream and downstream table dependencies  
✅ **Column Lineage**: Detailed column-to-column mapping  
✅ **CTE Support**: Complex Common Table Expression analysis  
✅ **External Metadata**: Integrate with your data catalog  
✅ **Rich Output**: JSON, console, and custom formatters  
✅ **Extensible Architecture**: Plugin-based metadata providers  
✅ **Comprehensive Logging**: Built-in logging for all components with configurable levels  
✨ **Optimized JSON Output**: 81% size reduction with comprehensive analysis  
✨ **Comprehensive Lineage**: Combined table+column analysis in single call  
✨ **Advanced Visualization**: Professional diagrams with Graphviz integration  

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/adhir-potdar/sql-lineage.git
cd sql-lineage

# Install system dependencies first (for visualization)
# macOS
brew install graphviz

# Linux (Ubuntu/Debian)
sudo apt-get install graphviz

# Install Python dependencies
./install.sh

# Or manually with virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install graphviz  # For visualization features
```

### Basic Usage

```python
from analyzer import SQLLineageAnalyzer
from analyzer.metadata import SampleMetadataRegistry

# Create analyzer with sample data for testing
analyzer = SQLLineageAnalyzer(dialect="trino")
analyzer.set_metadata_registry(SampleMetadataRegistry())

# Analyze SQL query (logging is automatic)
sql = \"\"\"
SELECT u.name, COUNT(o.id) as order_count
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.name
\"\"\"

result = analyzer.analyze(sql)
# Logs: "Starting analysis for SQL (length: 73)" (DEBUG level shows detailed info)
# Logs: "Analysis completed successfully"

# Check results
print(f"Tables: {list(result.table_lineage.upstream.keys())}")
print(f"Dependencies: {dict(result.table_lineage.upstream)}")
print(f"Column mappings: {len(result.column_lineage.upstream)}")

# ✨ Get optimized comprehensive lineage (recommended)
comprehensive_json = analyzer.get_lineage_chain_json(sql, "upstream")
print(f"Comprehensive JSON: {len(comprehensive_json):,} characters")
# 81% smaller than basic JSON while providing more complete analysis!

# Optional: Override defaults (DEBUG level and /tmp/sql_lineage.log are automatic)
import os
os.environ['SQL_LINEAGE_LOG_FILE'] = '/my/custom/path/analysis.log'
os.environ['SQL_LINEAGE_LOG_LEVEL'] = 'INFO'  # Reduce verbosity if needed
```

### Quick Test

```bash
# Run comprehensive tests
./test_simple.py

# Quick functionality check
./test_quick.py

# Sample queries test
./test_samples.py

# ✨ Test optimized lineage chain functionality
./test_lineage_chain.py
```

## Architecture

### Core Components

- **SQLLineageAnalyzer**: Main entry point for lineage analysis
- **LineageExtractor**: Parses SQL expressions and extracts relationships
- **MetadataRegistry**: Base class for metadata management
- **SampleMetadataRegistry**: Rich sample data for testing and demos

### Supported SQL Features

- **SELECT queries** with JOINs, subqueries, window functions
- **Common Table Expressions (CTEs)** with complex dependencies
- **CREATE TABLE AS SELECT** statements
- **UNION/UNION ALL** queries
- **Cross-schema and cross-catalog** references

## External Metadata Integration

### Production Usage (Clean Start)

```python
# Empty registry for production
analyzer = SQLLineageAnalyzer(dialect="trino")
analyzer.add_metadata_provider(YourMetadataProvider("connection"))

result = analyzer.analyze("SELECT * FROM production_table")
```

### With Sample Data (Testing/Demos)

```python
# Rich sample data for demonstrations
analyzer = SQLLineageAnalyzer(dialect="trino")
analyzer.set_metadata_registry(SampleMetadataRegistry())

result = analyzer.analyze("SELECT * FROM users")
# Will have rich metadata: descriptions, owners, row counts, etc.
```

## API Reference

### SQLLineageAnalyzer

```python
class SQLLineageAnalyzer:
    def __init__(self, dialect: str = "trino")
    def analyze(self, sql: str) -> LineageResult
    def analyze_file(self, file_path: str) -> LineageResult
    def analyze_multiple(self, queries: List[str]) -> List[LineageResult]
    def set_metadata_registry(self, registry: MetadataRegistry) -> None
    def add_metadata_provider(self, provider: MetadataProvider) -> None
    
    # JSON output methods
    def get_lineage_json(self, sql: str) -> str
    
    # Lineage chain methods
    def get_table_lineage_chain(self, sql: str, chain_type: str, depth: int) -> Dict
    def get_table_lineage_chain_json(self, sql: str, chain_type: str, depth: int) -> str
    def get_column_lineage_chain(self, sql: str, chain_type: str, depth: int) -> Dict
    def get_column_lineage_chain_json(self, sql: str, chain_type: str, depth: int) -> str
    
    # ✨ Comprehensive lineage methods (recommended)
    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, 
                         target_entity: str = None) -> Dict
    def get_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 0,
                              target_entity: str = None) -> str
```

### SQLLineageVisualizer

```python
class SQLLineageVisualizer:
    def __init__(self)
    
    # ✨ Comprehensive lineage visualization (recommended)
    def create_lineage_chain_diagram(self, lineage_chain_json: str, output_path: str,
                                   output_format: str, layout: str) -> str
    
    # Traditional separate lineage visualizations
    def create_lineage_diagram(self, table_chain_json: str, column_chain_json: str, 
                              output_path: str, output_format: str) -> str
    def create_table_only_diagram(self, table_chain_json: str, output_path: str, 
                                 output_format: str) -> str
    def create_column_focused_diagram(self, table_chain_json: str, column_chain_json: str,
                                    output_path: str, output_format: str) -> str
    def get_supported_formats(self) -> List[str]
```

### LineageResult

```python
class LineageResult:
    sql: str                              # Original SQL query
    dialect: str                          # SQL dialect used
    table_lineage: TableLineage          # Table dependencies
    column_lineage: ColumnLineage        # Column mappings
    metadata: Dict[str, TableMetadata]   # Table metadata
    errors: List[str]                    # Analysis errors
    warnings: List[str]                  # Analysis warnings
    
    def has_errors(self) -> bool
    def has_warnings(self) -> bool
    def to_dict(self) -> Dict[str, Any]
```

### JSON Output

Get complete lineage analysis as JSON:

```python
# Get JSON representation of lineage result
json_output = analyzer.get_lineage_json(sql)

# Contains table_lineage, column_lineage, metadata, etc.
# Useful for storing results or API responses
```

### ✨ Comprehensive Lineage (Recommended)

**New optimized methods that combine table and column lineage in a single, efficient output:**

```python
# 🚀 Comprehensive lineage with unlimited depth (recommended)
comprehensive_chain = analyzer.get_lineage_chain(sql, "upstream")  # depth=0 = unlimited
comprehensive_json = analyzer.get_lineage_chain_json(sql, "upstream")

# Target specific entity analysis
targeted_chain = analyzer.get_lineage_chain(sql, "downstream", depth=0, target_entity="users")

# Limited depth analysis
limited_chain = analyzer.get_lineage_chain(sql, "upstream", depth=3)
```

**✨ Key Benefits:**
- **81% smaller JSON** output compared to basic lineage (optimized metadata)
- **Combined analysis**: Tables + columns in single comprehensive format
- **Unlimited depth**: `depth=0` traverses complete dependency chains  
- **Efficient serialization**: Optimized JSON structure with minimal bloat
- **Visualization ready**: Direct input for `create_lineage_chain_diagram()`

**Parameters:**
- `chain_type`: `"upstream"` or `"downstream"`
- `depth`: `0` (unlimited), or `1-N` (limited levels)
- `target_entity`: Focus analysis on specific table/entity (optional)

## Lineage Event Mapper

Transform lineage chains into standardized events for integration with external systems:

```python
from src.analyzer.lineage_mapper import LineageEventMapper

# Create mapper and transform lineage chain to events
mapper = LineageEventMapper()
events = mapper.map_lineage_chain_to_events(
    lineage_json_string=lineage_json,
    tenant_id="tenant-123",
    association_type="DATAMAP", 
    association_id="datamap-456",
    query_id="query-789"
)

# Batch process all lineage files
python test_lineage_mapper.py  # Processes all JSON files in output/
```

**Features:**
- Converts nested lineage chains to flat event streams
- Multi-path metadata merging for complex dependencies
- Rich event metadata with source/target details and transformations
- Built-in logging and error handling

### Traditional Lineage Chains

Build separate table and column dependency chains:

```python
# Table lineage chains
upstream_chain = analyzer.get_table_lineage_chain(sql, "upstream", depth=3)
downstream_chain = analyzer.get_table_lineage_chain(sql, "downstream", depth=2)

# Column lineage chains  
column_chain = analyzer.get_column_lineage_chain(sql, "upstream", depth=2)

# JSON representation
chain_json = analyzer.get_table_lineage_chain_json(sql, "upstream", depth=4)
column_json = analyzer.get_column_lineage_chain_json(sql, "upstream", depth=3)
```

**Chain Features:**
- **Configurable Direction**: `upstream` or `downstream` 
- **Depth Control**: Limit traversal depth (1-N levels)
- **Cycle Prevention**: Handles circular dependencies
- **JSON Export**: Complete serialization for storage/analysis

### Lineage Visualization

Create visual diagrams from lineage chains using Graphviz:

```python
from analyzer.visualization import SQLLineageVisualizer

# Create visualizer
visualizer = SQLLineageVisualizer()

# ✨ Comprehensive lineage diagram (recommended)
comprehensive_json = analyzer.get_lineage_chain_json(sql, "upstream")
diagram_path = visualizer.create_lineage_chain_diagram(
    lineage_chain_json=comprehensive_json,
    output_path="comprehensive_lineage",
    output_format="png",
    layout="horizontal"  # upstream: right-to-left flow
)

# Traditional separate diagrams
table_json = analyzer.get_table_lineage_chain_json(sql, "upstream", depth=3)
column_json = analyzer.get_column_lineage_chain_json(sql, "upstream", depth=3)

# Create table-only diagram
visualizer.create_table_only_diagram(
    table_chain_json=table_json,
    output_path="table_only_lineage",
    output_format="png",
    layout="horizontal"
)

# Create integrated table + column diagram  
# (columns appear within their parent tables)
visualizer.create_lineage_diagram(
    table_chain_json=table_json,
    column_chain_json=column_json,
    output_path="integrated_lineage",
    output_format="svg",
    show_columns=True,
    layout="horizontal"  # shows tables as containers with columns inside
)
```

**Visualization Features:**
- **Direction-Specific Flow**: Upstream (right-to-left), Downstream (left-to-right) by default
- **Layout Options**: Horizontal (default) or vertical layouts
- **Integrated Column Lineage**: Columns displayed within their parent tables as containers
- **Multiple Formats**: PNG, SVG, PDF, JPG output
- **Custom Styling**: Colors, shapes, fonts configurable
- **Professional Layout**: Automatic positioning and spacing

## Testing

```bash
# Simple standalone tests
./test_simple.py

# Quick functionality tests  
./test_quick.py

# Comprehensive sample queries
./test_samples.py

# ✨ Optimized lineage chain functionality
./test_lineage_chain.py

# Visualization tests (requires system Graphviz + Python package)
./test_visualization.py

# Pytest suite
source venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Troubleshooting

### Visualization Issues

**Error: `failed to execute PosixPath('dot')`**
- **Cause**: System Graphviz not installed
- **Solution**: Install system Graphviz first (see System Requirements section)

**Error: `ModuleNotFoundError: No module named 'graphviz'`**  
- **Cause**: Python graphviz package not installed in virtual environment
- **Solution**: `pip install graphviz` in your activated virtual environment

**Error: `attr statement must target graph, node, or edge`**
- **Cause**: Outdated graphviz package version
- **Solution**: `pip install --upgrade graphviz`

### General Issues

**Import errors for core analyzer**
- Ensure you're in the project directory
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

## Logging

SQL Lineage Analyzer provides comprehensive logging support across all components including the API, analyzers, and parsers. The logging system uses hierarchical loggers and supports configurable output levels, formats, and destinations.

**Default Log File**: `/tmp/sql_lineage.log` (automatically created)  
**Default Log Level**: `DEBUG` (shows all analysis details)

### Logging Configuration

The logging system is automatically configured when you use any analyzer component with both console and file logging enabled by default. You can customize the logging behavior using environment variables:

```bash
# Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - defaults to DEBUG
export SQL_LINEAGE_LOG_LEVEL=INFO

# Override default log file location (optional, defaults to /tmp/sql_lineage.log)
export SQL_LINEAGE_LOG_FILE=/var/log/sql_lineage/analysis.log

# Custom log format (optional)
export SQL_LINEAGE_LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Python Configuration

```python
from analyzer import SQLLineageAnalyzer
from analyzer.utils.logging_config import SQLLineageLogger

# Basic usage - logging is automatically configured (console + /tmp/sql_lineage.log at DEBUG level)
analyzer = SQLLineageAnalyzer(dialect="trino")
result = analyzer.analyze("SELECT * FROM users")

# Manual logging configuration
SQLLineageLogger.configure(
    level="DEBUG",
    enable_console=True,
    log_file="/tmp/sql_analysis.log",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)

# Get logger for custom components
logger = SQLLineageLogger.get_logger("my_component")
logger.info("Custom logging message")
```

### Logging Levels

- **DEBUG**: Detailed SQL parsing, AST analysis, transformation details
- **INFO**: Analysis start/completion, table/column counts, successful operations
- **WARNING**: Non-critical issues, fallback behaviors, missing metadata
- **ERROR**: Analysis failures, parsing errors, configuration issues
- **CRITICAL**: System-level failures, initialization errors

### CLI Logging

The CLI automatically logs important operations:

```bash
# Default behavior - logs to console and /tmp/sql_lineage.log at DEBUG level
python -m analyzer.cli analyze "SELECT * FROM users"

# Reduce verbosity and use custom file location
export SQL_LINEAGE_LOG_LEVEL=INFO SQL_LINEAGE_LOG_FILE=/tmp/analysis.log
python -m analyzer.cli analyze "SELECT * FROM users"

# View log output (default location)
tail -f /tmp/sql_lineage.log

# View custom log location  
tail -f /tmp/analysis.log
```

### Log Structure

All logs follow a consistent hierarchical naming convention:

- `sql_lineage.analyzer` - Main analyzer operations
- `sql_lineage.analyzers.*` - Specific analyzer components
  - `sql_lineage.analyzers.select` - SELECT statement analysis  
  - `sql_lineage.analyzers.cte` - CTE analysis
  - `sql_lineage.analyzers.ctas` - CREATE TABLE AS analysis
- `sql_lineage.parsers.*` - SQL parsing components
  - `sql_lineage.parsers.select` - SELECT parsing
  - `sql_lineage.parsers.cte` - CTE parsing
- `sql_lineage.core.*` - Core components
  - `sql_lineage.core.extractor` - Lineage extraction
  - `sql_lineage.core.transformation_engine` - Transformation processing
  - `sql_lineage.core.chain_builder_engine` - Chain building

### Example Log Output

```
2025-08-07 11:37:40 - sql_lineage.analyzer - INFO - Initializing SQLLineageAnalyzer with dialect: trino
2025-08-07 11:37:40 - sql_lineage.analyzer - INFO - Starting analysis for SQL (length: 73)
2025-08-07 11:37:40 - sql_lineage.core.extractor - INFO - Starting table lineage extraction
2025-08-07 11:37:40 - sql_lineage.analyzers.select - INFO - Analyzing SELECT statement (length: 73)
2025-08-07 11:37:40 - sql_lineage.parsers.select - INFO - Parsing SELECT statement (length: 73)
2025-08-07 11:37:40 - sql_lineage.parsers.select - INFO - SELECT parsing completed - 2 columns, 1 tables, 1 joins
2025-08-07 11:37:40 - sql_lineage.analyzers.select - INFO - SELECT analysis completed - found 1 source tables, 2 result columns
2025-08-07 11:37:40 - sql_lineage.core.extractor - INFO - Table lineage extraction completed - upstream: 1 entries, downstream: 2 entries
2025-08-07 11:37:40 - sql_lineage.analyzer - INFO - Analysis completed successfully
```

### Production Logging

For production environments, configure logging appropriately:

```python
# Production configuration
import logging
from analyzer.utils.logging_config import SQLLineageLogger

# Configure for production (override DEBUG default)
SQLLineageLogger.configure(
    level="WARNING",  # Only log warnings and errors (instead of DEBUG default)
    enable_console=False,  # Disable console output
    log_file="/var/log/sql_lineage/analysis.log",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

# Use analyzer as normal - all logging is handled automatically
analyzer = SQLLineageAnalyzer(dialect="trino")
```

## Dependencies

- **sqlglot** >= 27.6.0 - SQL parsing and transformation  
- **rich** >= 13.0.0 - Console formatting  
- **click** >= 8.0.0 - CLI interface
- **pydantic** >= 2.0.0 - Data validation
- **networkx** >= 3.0 - Graph operations
- **graphviz** >= 0.20.0 - Lineage visualization (optional)

### System Requirements for Visualization

**Important**: Graphviz requires both system binaries and Python package:

```bash
# 1. Install system Graphviz executables (REQUIRED)
# macOS
brew install graphviz

# Ubuntu/Debian
sudo apt-get install graphviz

# CentOS/RHEL/Fedora
sudo yum install graphviz
# or: sudo dnf install graphviz

# Windows
choco install graphviz
# Or download from: https://graphviz.org/download/

# 2. Install Python package (in your virtual environment)
pip install graphviz

# 3. Verify installation
dot -V                    # Should show Graphviz version
python -c "import graphviz"  # Should not error
```

**Note**: The Python `graphviz` package is a wrapper that calls system `dot` executables. Both components are required for visualization features to work.

## License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for the data engineering community**