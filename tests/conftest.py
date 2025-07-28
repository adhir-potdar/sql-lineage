"""Pytest configuration and fixtures."""

import pytest
from analyzer import SQLLineageAnalyzer
from analyzer.metadata import MetadataRegistry


@pytest.fixture
def sample_sql_queries():
    """Sample SQL queries for testing."""
    return {
        "simple_select": "SELECT id, name FROM users WHERE age > 21",
        "join_query": """
            SELECT u.name, o.total
            FROM users u
            JOIN orders o ON u.id = o.user_id
        """,
        "cte_query": """
            WITH user_stats AS (
                SELECT user_id, COUNT(*) as order_count
                FROM orders
                GROUP BY user_id
            )
            SELECT u.name, us.order_count
            FROM users u
            JOIN user_stats us ON u.id = us.user_id
        """,
        "ctas_query": """
            CREATE TABLE active_users AS
            SELECT id, name, email
            FROM users
            WHERE created_at >= CURRENT_DATE - INTERVAL '30' DAY
        """,
        "complex_query": """
            WITH monthly_sales AS (
                SELECT 
                    DATE_TRUNC('month', o.order_date) as month,
                    p.category_id,
                    SUM(o.total) as monthly_total
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                WHERE o.order_date >= CURRENT_DATE - INTERVAL '1' YEAR
                GROUP BY DATE_TRUNC('month', o.order_date), p.category_id
            )
            SELECT 
                ms.month,
                c.category_name,
                ms.monthly_total,
                LAG(ms.monthly_total) OVER (
                    PARTITION BY ms.category_id 
                    ORDER BY ms.month
                ) as prev_month_total
            FROM monthly_sales ms
            JOIN categories c ON ms.category_id = c.category_id
            ORDER BY ms.month DESC, ms.monthly_total DESC
        """
    }


@pytest.fixture
def metadata_registry():
    """Create a metadata registry for testing."""
    return MetadataRegistry()


@pytest.fixture  
def analyzer():
    """Create an analyzer instance for testing."""
    return SQLLineageAnalyzer(dialect="trino")