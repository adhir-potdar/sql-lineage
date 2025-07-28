-- Sample SQL queries for testing lineage analysis

-- Example 1: Simple SELECT with JOIN
SELECT u.name, u.email, o.total, o.order_date
FROM users u
JOIN orders o ON u.id = o.user_id
WHERE o.order_date >= '2023-01-01';

-- Example 2: Complex aggregation with multiple JOINs
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
ORDER BY total_revenue DESC;

-- Example 3: CTE with window functions
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
ORDER BY month DESC;