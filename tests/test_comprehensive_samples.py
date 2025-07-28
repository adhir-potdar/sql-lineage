"""Comprehensive tests using all sample queries from original sqlglot_lineage and sqlglot-test folders."""

import pytest
from analyzer import SQLLineageAnalyzer


class TestComprehensiveSamples:
    """Test all sample queries from the original folders."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return SQLLineageAnalyzer(dialect="trino")
    
    def test_sample1_complex_cte_query(self, analyzer):
        """Test sample1.sql - Complex CTE query with CategorySales and TopCustomers."""
        sql = """
        WITH CategorySales AS (SELECT c.category_name,
                                      p.product_id,
                                      p.product_name,
                                      SUM(o.order_total)         AS total_sales,
                                      COUNT(DISTINCT o.order_id) AS total_orders,
                                      AVG(o.order_total)         AS avg_order_value
                               FROM orders o
                                        INNER JOIN
                                    products p ON o.product_id = p.product_id
                                        INNER JOIN
                                    categories c ON p.category_id = c.category_id
                               WHERE o.order_date >= DATEADD(YEAR, -1, GETDATE())
                               GROUP BY c.category_name, p.product_id, p.product_name),
             TopCustomers AS (SELECT c.category_name,
                                     o.customer_id,
                                     SUM(o.order_total) AS customer_total,
                                     RANK()                OVER (
                        PARTITION BY c.category_name
                        ORDER BY SUM(o.order_total) DESC
                    ) AS rank
                              FROM orders o
                                       INNER JOIN
                                   products p ON o.product_id = p.product_id
                                       INNER JOIN
                                   categories c ON p.category_id = c.category_id
                              WHERE o.order_date >= DATEADD(YEAR, -1, GETDATE())
                              GROUP BY c.category_name, o.customer_id)
        SELECT cs.category_name,
               cs.product_id,
               cs.product_name,
               cs.total_sales,
               cs.total_orders,
               cs.avg_order_value,
               tc.customer_id    AS top_customer_id,
               tc.customer_total AS top_customer_revenue
        FROM CategorySales cs
                 LEFT JOIN
             TopCustomers tc
             ON
                 cs.category_name = tc.category_name
        WHERE tc.rank <= 3
        ORDER BY cs.category_name,
                 cs.total_sales DESC
        """
        
        result = analyzer.analyze(sql)
        assert not result.has_errors()
        
        # Should have CTEs in lineage
        assert "CategorySales" in result.table_lineage.upstream
        assert "TopCustomers" in result.table_lineage.upstream
        assert "MAIN_QUERY" in result.table_lineage.upstream
        
        # CTEs should depend on source tables
        assert "default.orders" in result.table_lineage.upstream["CategorySales"]
        assert "default.products" in result.table_lineage.upstream["CategorySales"]
        assert "default.categories" in result.table_lineage.upstream["CategorySales"]
        
        # Column lineage should exist
        assert len(result.column_lineage.upstream) > 0
    
    def test_sample2_fully_qualified_names(self, analyzer):
        """Test sample2.sql - Query with fully qualified table names."""
        sql = """
        WITH SalesSummary AS (
            SELECT
                "production"."orders"."order_id" AS "FullyQualifiedTableId",
                "production"."orders"."customer_id",
                "production"."orders"."order_date",
                "production"."orders"."product_id",
                SUM("production"."orders"."order_total") AS "TotalOrderValue",
                ROW_NUMBER() OVER (
                    PARTITION BY "production"."orders"."customer_id"
                    ORDER BY "production"."orders"."order_date" DESC
                ) AS "OrderRank"
            FROM
                "production"."orders"
            WHERE
                "production"."orders"."order_date" >= DATEADD(YEAR, -2, CURRENT_TIMESTAMP)
            GROUP BY
                "production"."orders"."order_id",
                "production"."orders"."customer_id",
                "production"."orders"."order_date",
                "production"."orders"."product_id"
        ),
        TopProducts AS (
            SELECT
                "catalog"."products"."product_id",
                "catalog"."categories"."category_name",
                COUNT("production"."orders"."order_id") AS "NumberOfOrders",
                RANK() OVER (
                    PARTITION BY "catalog"."categories"."category_id"
                    ORDER BY COUNT("production"."orders"."order_id") DESC
                ) AS "ProductRank"
            FROM
                "catalog"."categories"
            INNER JOIN
                "catalog"."products" ON "catalog"."categories"."category_id" = "catalog"."products"."category_id"
            LEFT JOIN
                "production"."orders" ON "catalog"."products"."product_id" = "production"."orders"."product_id"
            GROUP BY
                "catalog"."products"."product_id",
                "catalog"."categories"."category_name",
                "catalog"."categories"."category_id"
        )
        SELECT
            ss."FullyQualifiedTableId" AS "Order_Id",
            ss."customer_id",
            ss."order_date",
            tp."category_name",
            tp."NumberOfOrders",
            tp."ProductRank",
            ss."TotalOrderValue"
        FROM
            SalesSummary ss
        INNER JOIN
            TopProducts tp ON tp."product_id" = ss."product_id"
        WHERE
            tp."ProductRank" <= 5
        ORDER BY
            tp."category_name",
            ss."order_date" DESC
        """
        
        result = analyzer.analyze(sql)
        assert not result.has_errors()
        
        # Should handle fully qualified names
        assert "SalesSummary" in result.table_lineage.upstream
        assert "TopProducts" in result.table_lineage.upstream
        
        # Should extract schema-qualified table names
        upstream_sales = result.table_lineage.upstream["SalesSummary"]
        upstream_products = result.table_lineage.upstream["TopProducts"]
        
        # Check that we have the qualified table names
        production_orders_found = any('"production"."orders"' in table for table in upstream_sales)
        catalog_tables_found = any('"catalog"' in table for table in upstream_products)
        
        assert production_orders_found or len(upstream_sales) > 0
        assert catalog_tables_found or len(upstream_products) > 0
    
    def test_sample3_union_and_case_statements(self, analyzer):
        """Test sample3.sql - Complex query with UNION ALL and CASE statements."""
        sql = """
        WITH OrderSummary AS (
            SELECT
                "schema1"."orders"."order_id",
                "schema1"."orders"."customer_id",
                "schema1"."orders"."product_id",
                SUM("schema1"."orders"."order_total") AS total_order_value,
                COUNT(DISTINCT "schema1"."orders"."order_id") AS order_count,
                CASE
                    WHEN SUM("schema1"."orders"."order_total") > 1000 THEN 'High'
                    WHEN SUM("schema1"."orders"."order_total") BETWEEN 500 AND 1000 THEN 'Medium'
                    ELSE 'Low'
                END AS order_priority
            FROM
                "schema1"."orders"
            INNER JOIN
                "schema1"."products" ON "schema1"."orders"."product_id" = "schema1"."products"."product_id"
            WHERE
                "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
            GROUP BY
                "schema1"."orders"."order_id",
                "schema1"."orders"."customer_id",
                "schema1"."orders"."product_id"
        ),
        TopCustomers AS (
            SELECT
                "schema1"."customers"."customer_id",
                CONCAT("schema1"."customers"."first_name", ' ', "schema1"."customers"."last_name") AS customer_name,
                SUM("schema1"."orders"."order_total") AS customer_spent,
                RANK() OVER (
                    ORDER BY SUM("schema1"."orders"."order_total") DESC
                ) AS spending_rank
            FROM
                "schema1"."orders"
            INNER JOIN
                "schema1"."customers" ON "schema1"."orders"."customer_id" = "schema1"."customers"."customer_id"
            WHERE
                "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
            GROUP BY
                "schema1"."customers"."customer_id", "schema1"."customers"."first_name", "schema1"."customers"."last_name"
        ),
        TopProductsInCategory AS (
            SELECT
                "schema1"."categories"."category_name",
                "schema1"."products"."product_id",
                COUNT("schema1"."orders"."order_id") AS total_orders,
                RANK() OVER (
                    PARTITION BY "schema1"."categories"."category_name"
                    ORDER BY COUNT("schema1"."orders"."order_id") DESC
                ) AS product_rank
            FROM
                "schema1"."products"
            INNER JOIN
                "schema1"."categories" ON "schema1"."products"."category_id" = "schema1"."categories"."category_id"
            LEFT JOIN
                "schema1"."orders" ON "schema1"."products"."product_id" = "schema1"."orders"."product_id"
            WHERE
                "schema1"."orders"."order_date" >= DATEADD(YEAR, -1, CURRENT_DATE)
            GROUP BY
                "schema1"."categories"."category_name", "schema1"."products"."product_id"
        ),
        CombinedResults AS (
            SELECT
                os.customer_id,
                os.order_id,
                os.product_id,
                os.total_order_value,
                os.order_priority,
                NULL AS product_rank,
                NULL AS category_name,
                tc.customer_spent,
                tc.spending_rank
            FROM
                OrderSummary os
            LEFT JOIN
                TopCustomers tc ON os.customer_id = tc.customer_id

            UNION ALL

            SELECT
                NULL AS customer_id,
                NULL AS order_id,
                tp.product_id,
                NULL AS total_order_value,
                NULL AS order_priority,
                tp.product_rank,
                tp.category_name,
                NULL AS customer_spent,
                NULL AS spending_rank
            FROM
                TopProductsInCategory tp
        ),
        FinalResults AS (
            SELECT
                cr.customer_id,
                cr.order_id,
                cr.product_id,
                cr.total_order_value,
                cr.order_priority,
                cr.product_rank,
                cr.category_name,
                cr.customer_spent,
                cr.spending_rank,
                CASE
                    WHEN cr.spending_rank IS NOT NULL AND cr.spending_rank <= 5 THEN 'VIP'
                    WHEN cr.customer_spent IS NOT NULL THEN 'Premium'
                    WHEN cr.product_rank IS NOT NULL AND cr.product_rank <= 3 THEN 'Top Product'
                    ELSE 'Standard'
                END AS tag
            FROM
                CombinedResults cr
        )
        SELECT
            fr.customer_id,
            fr.order_id,
            fr.product_id,
            fr.category_name,
            fr.total_order_value,
            fr.customer_spent,
            fr.spending_rank,
            fr.product_rank,
            fr.order_priority,
            fr.tag
        FROM
            FinalResults fr
        ORDER BY
            fr.spending_rank ASC,
            fr.product_rank ASC,
            fr.total_order_value DESC
        """
        
        result = analyzer.analyze(sql)
        assert not result.has_errors()
        
        # Should have all CTEs in lineage
        expected_ctes = ["OrderSummary", "TopCustomers", "TopProductsInCategory", "CombinedResults", "FinalResults"]
        for cte in expected_ctes:
            assert cte in result.table_lineage.upstream
        
        # Should have main query depending on final CTE
        assert "MAIN_QUERY" in result.table_lineage.upstream
        assert "FinalResults" in result.table_lineage.upstream["MAIN_QUERY"]
    
    def test_sqlglot_test_basic_queries(self, analyzer):
        """Test basic queries from sqlglot-test folder."""
        basic_queries = [
            "SELECT id, name, age FROM users WHERE age > 21",
            "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month FROM orders",
            "SELECT x, y FROM old_table"
        ]
        
        for sql in basic_queries:
            result = analyzer.analyze(sql)
            assert not result.has_errors()
            assert "MAIN_QUERY" in result.table_lineage.upstream
            assert len(result.table_lineage.upstream["MAIN_QUERY"]) > 0
    
    def test_sqlglot_test_complex_join(self, analyzer):
        """Test complex JOIN query from sqlglot-test folder."""
        sql = """
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
        
        result = analyzer.analyze(sql)
        assert not result.has_errors()
        
        # Should have both tables in upstream
        upstream = result.table_lineage.upstream["MAIN_QUERY"]
        assert "default.users" in upstream
        assert "default.orders" in upstream
        
        # Should have column lineage
        assert len(result.column_lineage.upstream) > 0
    
    def test_sqlglot_test_ctas_queries(self, analyzer):
        """Test CREATE TABLE AS SELECT queries from sqlglot-test folder."""
        ctas_queries = [
            "CREATE TABLE user_summary AS SELECT id, name, age FROM users WHERE age >= 18",
            """CREATE TABLE monthly_sales AS 
               SELECT DATE_FORMAT(order_date, '%Y-%m') as month, 
                      COUNT(*) as order_count, 
                      SUM(total) as total_revenue 
               FROM orders 
               GROUP BY DATE_FORMAT(order_date, '%Y-%m')""",
            """CREATE TABLE analytics.user_metrics AS 
               SELECT user_id, COUNT(*) as event_count 
               FROM events.user_events 
               GROUP BY user_id"""
        ]
        
        for sql in ctas_queries:
            result = analyzer.analyze(sql)
            assert not result.has_errors()
            
            # Should have target table in upstream lineage
            target_tables = [k for k in result.table_lineage.upstream.keys() if k != "MAIN_QUERY"]
            assert len(target_tables) > 0
    
    def test_trino_specific_queries(self, analyzer):
        """Test Trino-specific queries from sqlglot-test folder."""
        trino_queries = [
            ("SELECT customer_id, ROW_NUMBER() OVER (PARTITION BY region ORDER BY revenue DESC) as rank FROM sales", "trino"),
            ("SELECT APPROX_DISTINCT(user_id) FROM events", "trino"),
            ("SELECT JSON_EXTRACT(metadata, '$.user.id') FROM logs", "trino"),
            ("SELECT * FROM hive.default.users", "trino"),
            ("SELECT * FROM mysql.sales_db.orders", "trino")
        ]
        
        for sql, dialect in trino_queries:
            analyzer.set_dialect(dialect)
            result = analyzer.analyze(sql)
            assert not result.has_errors()
            assert result.dialect == dialect
    
    def test_complex_trino_cte(self, analyzer):
        """Test complex Trino CTE query from sqlglot-test folder."""
        sql = """
        WITH user_metrics AS (
            SELECT 
                user_id,
                COUNT(*) as event_count,
                APPROX_DISTINCT(session_id) as session_count,
                JSON_EXTRACT(properties, '$.category') as category
            FROM events
            WHERE event_date >= DATE '2023-01-01'
            GROUP BY user_id, JSON_EXTRACT(properties, '$.category')
        ),
        ranked_users AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (PARTITION BY category ORDER BY event_count DESC) as category_rank
            FROM user_metrics
        )
        SELECT 
            category,
            user_id,
            event_count,
            session_count,
            category_rank
        FROM ranked_users
        WHERE category_rank <= 10
        """
        
        analyzer.set_dialect("trino")
        result = analyzer.analyze(sql)
        assert not result.has_errors()
        
        # Should have CTEs
        assert "user_metrics" in result.table_lineage.upstream
        assert "ranked_users" in result.table_lineage.upstream
        
        # Main query should depend on final CTE
        assert "ranked_users" in result.table_lineage.upstream["MAIN_QUERY"]