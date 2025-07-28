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