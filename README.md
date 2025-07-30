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

# Analyze SQL query
sql = \"\"\"
SELECT u.name, COUNT(o.id) as order_count
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.name
\"\"\"

result = analyzer.analyze(sql)

# Check results
print(f"Tables: {list(result.table_lineage.upstream.keys())}")
print(f"Dependencies: {dict(result.table_lineage.upstream)}")
print(f"Column mappings: {len(result.column_lineage.upstream)}")

# ✨ Get optimized comprehensive lineage (recommended)
comprehensive_json = analyzer.get_lineage_chain_json(sql, "upstream")
print(f"Comprehensive JSON: {len(comprehensive_json):,} characters")
# 81% smaller than basic JSON while providing more complete analysis!
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

## Dependencies

- **sqlglot** >= 20.0.0 - SQL parsing and transformation
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