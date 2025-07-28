#!/usr/bin/env python3
"""
Sample usage examples for SQL Lineage Analyzer.
"""

from analyzer import SQLLineageAnalyzer
from analyzer.formatters import JSONFormatter, ConsoleFormatter


def main():
    """Demonstrate various usage patterns."""
    print("SQL Lineage Analyzer - Sample Usage")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SQLLineageAnalyzer(dialect="trino")
    
    # Example 1: Simple JOIN query
    print("\n1. Simple JOIN Query:")
    sql1 = """
    SELECT u.name, COUNT(o.id) as order_count
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.created_at >= '2023-01-01'
    GROUP BY u.name
    ORDER BY order_count DESC
    """
    
    result1 = analyzer.analyze(sql1)
    ConsoleFormatter().format_compact(result1)
    
    # Example 2: Complex CTE query
    print("\n2. Complex CTE Query:")
    sql2 = """
    WITH CategorySales AS (
        SELECT c.category_name,
               p.product_id,
               p.product_name,
               SUM(o.order_total) AS total_sales,
               COUNT(DISTINCT o.order_id) AS total_orders,
               AVG(o.order_total) AS avg_order_value
        FROM orders o
        INNER JOIN products p ON o.product_id = p.product_id
        INNER JOIN categories c ON p.category_id = c.category_id
        WHERE o.order_date >= CURRENT_DATE - INTERVAL '1' YEAR
        GROUP BY c.category_name, p.product_id, p.product_name
    ),
    TopCustomers AS (
        SELECT c.category_name,
               o.customer_id,
               SUM(o.order_total) AS customer_total,
               RANK() OVER (
                   PARTITION BY c.category_name
                   ORDER BY SUM(o.order_total) DESC
               ) AS rank
        FROM orders o
        INNER JOIN products p ON o.product_id = p.product_id
        INNER JOIN categories c ON p.category_id = c.category_id
        WHERE o.order_date >= CURRENT_DATE - INTERVAL '1' YEAR
        GROUP BY c.category_name, o.customer_id
    )
    SELECT cs.category_name,
           cs.product_id,
           cs.product_name,
           cs.total_sales,
           cs.total_orders,
           cs.avg_order_value,
           tc.customer_id AS top_customer_id,
           tc.customer_total AS top_customer_revenue
    FROM CategorySales cs
    LEFT JOIN TopCustomers tc ON cs.category_name = tc.category_name
    WHERE tc.rank <= 3
    ORDER BY cs.category_name, cs.total_sales DESC
    """
    
    result2 = analyzer.analyze(sql2)
    ConsoleFormatter().format_compact(result2)
    
    # Example 3: CREATE TABLE AS SELECT
    print("\n3. CREATE TABLE AS SELECT:")
    sql3 = """
    CREATE TABLE user_summary AS
    SELECT u.id,
           u.name,
           u.age,
           COUNT(o.id) as total_orders,
           SUM(o.total) as total_spent,
           AVG(o.total) as avg_order_value
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.age >= 18
    GROUP BY u.id, u.name, u.age
    """
    
    result3 = analyzer.analyze(sql3)
    ConsoleFormatter().format_compact(result3)
    
    # Example 4: JSON output
    print("\n4. JSON Output Example:")
    json_formatter = JSONFormatter(indent=2)
    json_output = json_formatter.format_table_lineage_only(result1)
    print(json_output)
    
    # Example 5: Analyze multiple queries
    print("\n5. Batch Analysis:")
    queries = [
        "SELECT * FROM users WHERE age > 25",
        "SELECT product_name, COUNT(*) FROM products p JOIN orders o ON p.product_id = o.product_id GROUP BY product_name",
        "CREATE TABLE active_users AS SELECT * FROM users WHERE created_at >= CURRENT_DATE - INTERVAL '30' DAY"
    ]
    
    results = analyzer.analyze_multiple(queries)
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i} lineage:")
        ConsoleFormatter().format_compact(result)


if __name__ == "__main__":
    main()