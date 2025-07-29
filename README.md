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

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/adhir-potdar/sql-lineage.git
cd sql-lineage

# Install dependencies
./install.sh

# Or manually with virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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
```

### Quick Test

```bash
# Run comprehensive tests
./test_simple.py

# Quick functionality check
./test_quick.py

# Sample queries test
./test_samples.py
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

### Lineage Chains

Build hierarchical dependency chains with configurable depth:

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

## Testing

```bash
# Simple standalone tests
./test_simple.py

# Quick functionality tests  
./test_quick.py

# Comprehensive sample queries
./test_samples.py

# Pytest suite
source venv/bin/activate
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Dependencies

- **sqlglot** >= 20.0.0 - SQL parsing and transformation
- **rich** >= 13.0.0 - Console formatting  
- **click** >= 8.0.0 - CLI interface
- **pydantic** >= 2.0.0 - Data validation
- **networkx** >= 3.0 - Graph operations

## License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for the data engineering community**