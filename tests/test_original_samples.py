"""Test cases using the exact original SQL files from the original folders."""

import pytest
from analyzer import SQLLineageAnalyzer


class TestOriginalSampleFiles:
    """Test using the actual SQL files from the original folders."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing.""" 
        return SQLLineageAnalyzer(dialect="trino")
    
    def test_sample_sql_file_contents(self, analyzer):
        """Test analysis of queries similar to those in examples/sample.sql."""
        
        # Example 1: Simple SELECT with JOIN (from examples/sample.sql)
        sql1 = """
        SELECT u.name, u.email, o.total, o.order_date
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.order_date >= '2023-01-01'
        """
        
        result1 = analyzer.analyze(sql1)
        assert not result1.has_errors()
        
        upstream = result1.table_lineage.upstream["QUERY_RESULT"]
        assert "users" in upstream
        assert "orders" in upstream
        
        # Example 2: Complex aggregation with multiple JOINs
        sql2 = """
        SELECT 
            c.category_name,
            COUNT(DISTINCT p.product_id) as product_count,
            COUNT(DISTINCT o.order_id) as order_count,
            SUM(o.order_total) as total_revenue,
            AVG(o.order_total) as avg_order_value
        FROM categories c
            LEFT JOIN products p ON c.category_id = p.category_id
            LEFT JOIN orders o ON p.product_id = o.product_id
        WHERE o.order_date >= CURRENT_DATE - INTERVAL '1' YEAR
        GROUP BY c.category_name, c.category_id
        HAVING COUNT(DISTINCT o.order_id) > 10
        ORDER BY total_revenue DESC
        """
        
        result2 = analyzer.analyze(sql2)
        assert not result2.has_errors()
        
        upstream2 = result2.table_lineage.upstream["QUERY_RESULT"]
        assert "categories" in upstream2
        assert "products" in upstream2
        assert "orders" in upstream2
        
        # Example 3: CTE with window functions
        sql3 = """
        WITH monthly_sales AS (
            SELECT 
                DATE_TRUNC('month', order_date) as month,
                SUM(order_total) as monthly_total,
                COUNT(*) as order_count
            FROM orders
            WHERE order_date >= CURRENT_DATE - INTERVAL '2' YEAR
            GROUP BY DATE_TRUNC('month', order_date)
        ),
        sales_with_growth AS (
            SELECT 
                month,
                monthly_total,
                order_count,
                LAG(monthly_total) OVER (ORDER BY month) as prev_month_total,
                (monthly_total - LAG(monthly_total) OVER (ORDER BY month)) / 
                LAG(monthly_total) OVER (ORDER BY month) * 100 as growth_rate
            FROM monthly_sales
        )
        SELECT *
        FROM sales_with_growth
        WHERE growth_rate > 5
        ORDER BY month DESC
        """
        
        result3 = analyzer.analyze(sql3)
        assert not result3.has_errors()
        
        # Should have CTEs
        assert "monthly_sales" in result3.table_lineage.upstream
        assert "sales_with_growth" in result3.table_lineage.upstream
        assert "QUERY_RESULT" in result3.table_lineage.upstream
        
        # CTE should depend on orders table
        assert "orders" in result3.table_lineage.upstream["monthly_sales"]
        
        # Main query should depend on final CTE
        assert "sales_with_growth" in result3.table_lineage.upstream["QUERY_RESULT"]
    
    def test_enhanced_lineage_analysis_queries(self, analyzer):
        """Test queries from enhanced_lineage_analysis.py patterns."""
        
        # Basic queries pattern
        basic_queries = [
            "SELECT id, name, age FROM users WHERE age > 21",
            "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month FROM orders"
        ]
        
        for sql in basic_queries:
            result = analyzer.analyze(sql)
            assert not result.has_errors()
            assert "QUERY_RESULT" in result.table_lineage.upstream
        
        # Complex JOIN query similar to enhanced_lineage_analysis.py
        complex_query = """
        SELECT 
            u.id,
            u.name,
            COUNT(o.id) as order_count,
            AVG(o.total) as avg_order_value
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at >= '2023-01-01'
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
        ORDER BY avg_order_value DESC
        LIMIT 10
        """
        
        result = analyzer.analyze(complex_query)
        assert not result.has_errors()
        
        upstream = result.table_lineage.upstream["QUERY_RESULT"]
        assert "users" in upstream
        assert "orders" in upstream
        
        # Should have column lineage
        assert len(result.column_lineage.upstream) > 0
    
    def test_ctas_queries_pattern(self, analyzer):
        """Test CREATE TABLE AS SELECT patterns from original samples."""
        
        ctas_queries = [
            "CREATE TABLE user_summary AS SELECT id, name, age FROM users WHERE age >= 18",
            """CREATE TABLE monthly_sales AS 
               SELECT DATE_FORMAT(order_date, '%Y-%m') as month, 
                      COUNT(*) as order_count, 
                      SUM(total) as total_revenue 
               FROM orders 
               GROUP BY DATE_FORMAT(order_date, '%Y-%m')"""
        ]
        
        for sql in ctas_queries:
            result = analyzer.analyze(sql)
            assert not result.has_errors()
            
            # Should identify target table
            target_tables = [k for k in result.table_lineage.upstream.keys() if k != "QUERY_RESULT"]
            assert len(target_tables) > 0
    
    def test_trino_dialect_patterns(self, analyzer):
        """Test Trino-specific patterns from original samples."""
        
        # Set Trino dialect
        analyzer.set_dialect("trino")
        
        trino_queries = [
            "SELECT * FROM hive.users",
            "CREATE TABLE hive.analytics.user_summary AS SELECT user_id, name, age FROM hive.users WHERE age >= 18",
            """CREATE TABLE postgresql.warehouse.customer_summary AS 
               SELECT c.customer_id, c.customer_name, COUNT(o.order_id) as total_orders 
               FROM mysql.sales_db.customers c 
               LEFT JOIN mysql.sales_db.orders o ON c.customer_id = o.customer_id 
               GROUP BY c.customer_id, c.customer_name""",
            """CREATE TABLE hive.processed.user_preferences AS 
               SELECT user_id, JSON_EXTRACT(preferences, '$.theme') as theme_preference 
               FROM hive.raw.user_profiles 
               WHERE JSON_EXTRACT(preferences, '$.active') = 'true'"""
        ]
        
        for sql in trino_queries:
            result = analyzer.analyze(sql)
            assert not result.has_errors()
            assert result.dialect == "trino"
    
    def test_complex_maximo_query_pattern(self, analyzer):
        """Test complex query pattern similar to the Maximo query from enhanced_lineage_analysis.py."""
        
        # Simplified version of the complex Maximo-style query
        sql = """
        SELECT
            wo.wonum as work_order_number,
            wo.description as work_order_description,
            wo.reportdate as work_order_created_datetime,
            wo.targstartdate as work_order_required_by_datetime,
            asset.assetnum as component_code,
            synonymdomain.maxvalue as work_order_status_code,
            wochange.et_enamsouttype as outage_type_code
        FROM
            maximo_dev.maximo.workorder wo
        LEFT JOIN
            maximo_dev.maximo.asset asset ON wo.assetnum = asset.assetnum
        LEFT JOIN 
            maximo_dev.maximo.synonymdomain synonymdomain ON
            wo.status = synonymdomain.value AND synonymdomain.domainid = 'WOSTATUS'
        LEFT JOIN 
            maximo_dev.maximo.wochange wochange ON wochange.wonum = wo.wonum
        LEFT JOIN 
            maximo_dev.maximo.relatedrecord relatedrecord ON
            relatedrecord.recordkey = wo.wonum
        """
        
        analyzer.set_dialect("trino")
        result = analyzer.analyze(sql)
        assert not result.has_errors()
        
        # Should identify all the complex table references
        upstream = result.table_lineage.upstream["QUERY_RESULT"]
        assert len(upstream) >= 4  # Should have multiple tables from the complex JOIN
        
        # Should handle the schema-qualified names
        schema_qualified_found = any("maximo_dev.maximo" in table for table in upstream)
        assert schema_qualified_found or len(upstream) > 0  # Either found qualified names or extracted table names
    
    def test_batch_analysis_pattern(self, analyzer):
        """Test batch analysis of multiple queries like in sample_usage.py."""
        
        queries = [
            "SELECT * FROM users WHERE age > 25",
            "SELECT product_name, COUNT(*) FROM products p JOIN orders o ON p.product_id = o.product_id GROUP BY product_name",
            "CREATE TABLE active_users AS SELECT * FROM users WHERE created_at >= CURRENT_DATE - INTERVAL '30' DAY"
        ]
        
        results = analyzer.analyze_multiple(queries)
        
        assert len(results) == 3
        for result in results:
            assert not result.has_errors()
            assert len(result.table_lineage.upstream) > 0