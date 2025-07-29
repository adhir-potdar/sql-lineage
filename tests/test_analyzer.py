"""Tests for SQL lineage analyzer."""

import pytest
from analyzer import SQLLineageAnalyzer
from analyzer.metadata import SampleMetadataRegistry
from analyzer.core.models import LineageResult


class TestSQLLineageAnalyzer:
    """Test cases for SQLLineageAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        analyzer = SQLLineageAnalyzer(dialect="trino")
        analyzer.set_metadata_registry(SampleMetadataRegistry())
        return analyzer
    
    def test_simple_select(self, analyzer):
        """Test simple SELECT query analysis."""
        sql = "SELECT id, name FROM users WHERE age > 21"
        result = analyzer.analyze(sql)
        
        assert isinstance(result, LineageResult)
        assert result.sql == sql
        assert result.dialect == "trino"
        assert not result.has_errors()
        
        # Should have upstream dependency on users table
        assert "QUERY_RESULT" in result.table_lineage.upstream
        assert "users" in result.table_lineage.upstream["QUERY_RESULT"]
    
    def test_join_query(self, analyzer):
        """Test JOIN query analysis."""
        sql = """
        SELECT u.name, o.total
        FROM users u
        JOIN orders o ON u.id = o.user_id
        """
        result = analyzer.analyze(sql)
        
        assert not result.has_errors()
        
        # Should have upstream dependencies on both tables
        upstream = result.table_lineage.upstream["QUERY_RESULT"]
        assert "users" in upstream
        assert "orders" in upstream
        
        # Should have column lineage
        assert len(result.column_lineage.upstream) > 0
    
    def test_cte_query(self, analyzer):
        """Test CTE query analysis."""
        sql = """
        WITH user_orders AS (
            SELECT u.id, u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            GROUP BY u.id, u.name
        )
        SELECT * FROM user_orders WHERE order_count > 5
        """
        result = analyzer.analyze(sql)
        
        assert not result.has_errors()
        
        # Should have CTE in upstream
        assert "user_orders" in result.table_lineage.upstream
        assert "QUERY_RESULT" in result.table_lineage.upstream
        
        # CTE should depend on source tables
        cte_upstream = result.table_lineage.upstream["user_orders"]
        assert "users" in cte_upstream
        assert "orders" in cte_upstream
        
        # Main query should depend on CTE
        main_upstream = result.table_lineage.upstream["QUERY_RESULT"]
        assert "user_orders" in main_upstream
    
    def test_create_table_as_select(self, analyzer):
        """Test CREATE TABLE AS SELECT analysis."""
        sql = """
        CREATE TABLE user_summary AS
        SELECT id, name, age
        FROM users
        WHERE age >= 18
        """
        result = analyzer.analyze(sql)
        
        assert not result.has_errors()
        
        # Should have target table in upstream
        assert "user_summary" in result.table_lineage.upstream
        assert "users" in result.table_lineage.upstream["user_summary"]
        
        # Should have downstream dependency
        assert "users" in result.table_lineage.downstream
        assert "user_summary" in result.table_lineage.downstream["users"]
    
    def test_invalid_sql(self, analyzer):
        """Test handling of invalid SQL."""
        sql = "SELECT FROM WHERE"
        result = analyzer.analyze(sql)
        
        assert result.has_errors()
        assert len(result.errors) > 0
    
    def test_empty_sql(self, analyzer):
        """Test handling of empty SQL."""
        result = analyzer.analyze("")
        
        assert result.has_errors()
        assert "empty" in result.errors[0].lower()
    
    def test_analyze_file(self, analyzer, tmp_path):
        """Test analyzing SQL from file."""
        # Create temporary SQL file
        sql_file = tmp_path / "test.sql"
        sql_content = "SELECT id, name FROM users"
        sql_file.write_text(sql_content)
        
        result = analyzer.analyze_file(str(sql_file))
        
        assert not result.has_errors()
        assert result.sql == sql_content
        assert "users" in result.table_lineage.upstream["QUERY_RESULT"]
    
    def test_analyze_file_not_found(self, analyzer):
        """Test analyzing non-existent file."""
        result = analyzer.analyze_file("/nonexistent/file.sql")
        
        assert result.has_errors()
        assert "Failed to read file" in result.errors[0]
    
    def test_analyze_multiple(self, analyzer):
        """Test analyzing multiple queries."""
        queries = [
            "SELECT id FROM users",
            "SELECT total FROM orders",
            "SELECT name FROM products"
        ]
        
        results = analyzer.analyze_multiple(queries)
        
        assert len(results) == 3
        assert all(isinstance(r, LineageResult) for r in results)
        assert all(not r.has_errors() for r in results)
    
    def test_different_dialects(self):
        """Test different SQL dialects."""
        sql = "SELECT id, name FROM users"
        
        # Test with different dialects
        for dialect in ["trino", "mysql", "postgres"]:
            analyzer = SQLLineageAnalyzer(dialect=dialect)
            analyzer.set_metadata_registry(SampleMetadataRegistry())
            result = analyzer.analyze(sql)
            
            assert result.dialect == dialect
            assert not result.has_errors()
    
    def test_complex_query_with_metadata(self, analyzer):
        """Test complex query that uses metadata."""
        sql = """
        SELECT 
            u.name,
            COUNT(o.id) as order_count,
            SUM(o.total) as total_spent,
            p.product_name
        FROM users u
        JOIN orders o ON u.id = o.user_id
        JOIN products p ON o.product_id = p.product_id
        WHERE u.created_at >= '2023-01-01'
        GROUP BY u.name, p.product_name
        HAVING COUNT(o.id) > 5
        """
        
        result = analyzer.analyze(sql)
        
        assert not result.has_errors()
        
        # Should have metadata for the tables
        assert len(result.metadata) > 0
        
        # Check table lineage
        upstream = result.table_lineage.upstream["QUERY_RESULT"]
        assert "users" in upstream
        assert "orders" in upstream
        assert "products" in upstream
        
        # Check column lineage
        assert len(result.column_lineage.upstream) > 0
    
    def test_column_lineage_flag_functionality(self, analyzer, capsys):
        """Test that column lineage flag controls output display."""
        from test_formatter import print_lineage_analysis
        
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        result = analyzer.analyze(sql)
        
        # Test with column lineage enabled (default)
        print_lineage_analysis(result, sql, "Test with Columns", show_column_lineage=True)
        captured_with = capsys.readouterr()
        
        # Test with column lineage disabled
        print_lineage_analysis(result, sql, "Test without Columns", show_column_lineage=False)
        captured_without = capsys.readouterr()
        
        # With column lineage should contain the detailed section
        assert "🔍 COLUMN LINEAGE (Detailed):" in captured_with.out
        assert "Column Dependencies:" in captured_with.out
        
        # Without column lineage should not contain the detailed section
        assert "🔍 COLUMN LINEAGE (Detailed):" not in captured_without.out
        assert "Column Dependencies:" not in captured_without.out
        
        # Both should still contain upstream and downstream lineage
        assert "📊 UPSTREAM LINEAGE" in captured_with.out
        assert "📈 DOWNSTREAM LINEAGE" in captured_with.out
        assert "📊 UPSTREAM LINEAGE" in captured_without.out
        assert "📈 DOWNSTREAM LINEAGE" in captured_without.out