"""Sample metadata registry with test data for demonstrations and testing."""

from .registry import MetadataRegistry
from ..core.models import TableMetadata, ColumnMetadata


class SampleMetadataRegistry(MetadataRegistry):
    """Registry with sample metadata for testing and demonstrations."""
    
    def __init__(self):
        """Initialize with sample metadata data."""
        super().__init__()
        # Clear any existing data and initialize with sample data
        self.tables.clear()
        self._initialize_sample_metadata()
    
    def _initialize_sample_metadata(self):
        """Initialize metadata for sample tables."""
        
        # Users table
        users_columns = [
            ColumnMetadata("id", "BIGINT", False, True, description="Unique user identifier"),
            ColumnMetadata("name", "VARCHAR(255)", False, description="User full name"),
            ColumnMetadata("age", "INTEGER", True, description="User age in years"),
            ColumnMetadata("email", "VARCHAR(255)", True, description="User email address"),
            ColumnMetadata("created_at", "TIMESTAMP", False, description="User registration timestamp"),
            ColumnMetadata("user_id", "BIGINT", False, description="Alternative user ID reference")
        ]
        
        self.tables["default.users"] = TableMetadata(
            catalog=None, schema="default", table="users",
            columns=users_columns,
            description="User profile information",
            owner="data_team",
            row_count=150000,
            storage_format="PARQUET"
        )
        
        # Orders table
        orders_columns = [
            ColumnMetadata("id", "BIGINT", False, True, description="Unique order identifier"),
            ColumnMetadata("user_id", "BIGINT", False, foreign_key="users.id", description="Reference to user"),
            ColumnMetadata("customer_id", "BIGINT", False, description="Customer identifier"),
            ColumnMetadata("total", "DECIMAL(10,2)", False, description="Order total amount"),
            ColumnMetadata("order_date", "DATE", False, description="Order placement date"),
            ColumnMetadata("created_at", "TIMESTAMP", False, description="Order creation timestamp"),
            ColumnMetadata("order_id", "BIGINT", False, description="Alternative order ID"),
            ColumnMetadata("order_total", "DECIMAL(10,2)", False, description="Order total value"),
            ColumnMetadata("product_id", "BIGINT", False, description="Product identifier")
        ]
        
        self.tables["default.orders"] = TableMetadata(
            catalog=None, schema="default", table="orders",
            columns=orders_columns,
            description="Customer order transactions",
            owner="sales_team",
            row_count=500000,
            storage_format="DELTA"
        )
        
        # Products table
        products_columns = [
            ColumnMetadata("product_id", "BIGINT", False, True, description="Unique product identifier"),
            ColumnMetadata("product_name", "VARCHAR(255)", False, description="Product name"),
            ColumnMetadata("category_id", "BIGINT", False, description="Category identifier"),
            ColumnMetadata("price", "DECIMAL(10,2)", False, description="Product price")
        ]
        
        self.tables["default.products"] = TableMetadata(
            catalog=None, schema="default", table="products",
            columns=products_columns,
            description="Product catalog",
            owner="product_team",
            row_count=25000,
            storage_format="PARQUET"
        )
        
        # Categories table
        categories_columns = [
            ColumnMetadata("category_id", "BIGINT", False, True, description="Unique category identifier"),
            ColumnMetadata("category_name", "VARCHAR(255)", False, description="Category name"),
            ColumnMetadata("description", "TEXT", True, description="Category description")
        ]
        
        self.tables["default.categories"] = TableMetadata(
            catalog=None, schema="default", table="categories",
            columns=categories_columns,
            description="Product categories",
            owner="product_team",
            row_count=100,
            storage_format="PARQUET"
        )
        
        # Events table
        events_columns = [
            ColumnMetadata("user_id", "BIGINT", False, description="User performing the event"),
            ColumnMetadata("event_type", "VARCHAR(100)", False, description="Type of event"),
            ColumnMetadata("event_date", "DATE", False, description="Event occurrence date"),
            ColumnMetadata("event_timestamp", "TIMESTAMP", False, description="Precise event time"),
            ColumnMetadata("session_id", "VARCHAR(255)", True, description="User session identifier"),
            ColumnMetadata("properties", "JSON", True, description="Event properties in JSON format"),
            ColumnMetadata("metadata", "JSON", True, description="Additional metadata"),
            ColumnMetadata("duration", "INTEGER", True, description="Event duration in milliseconds")
        ]
        
        self.tables["default.events"] = TableMetadata(
            catalog=None, schema="default", table="events",
            columns=events_columns,
            description="User interaction events",
            owner="analytics_team",
            row_count=10000000,
            storage_format="PARQUET"
        )
        
        # Sales table
        sales_columns = [
            ColumnMetadata("customer_id", "BIGINT", False, description="Customer identifier"),
            ColumnMetadata("region", "VARCHAR(100)", False, description="Sales region"),
            ColumnMetadata("revenue", "DECIMAL(12,2)", False, description="Revenue amount"),
            ColumnMetadata("sale_date", "DATE", False, description="Sale date")
        ]
        
        self.tables["default.sales"] = TableMetadata(
            catalog=None, schema="default", table="sales",
            columns=sales_columns,
            description="Sales performance data",
            owner="sales_team",
            row_count=75000,
            storage_format="PARQUET"
        )
        
        # Logs table
        logs_columns = [
            ColumnMetadata("log_id", "BIGINT", False, True, description="Log entry identifier"),
            ColumnMetadata("metadata", "JSON", True, description="Log metadata in JSON format"),
            ColumnMetadata("log_level", "VARCHAR(20)", False, description="Log severity level"),
            ColumnMetadata("timestamp", "TIMESTAMP", False, description="Log entry timestamp")
        ]
        
        self.tables["default.logs"] = TableMetadata(
            catalog=None, schema="default", table="logs",
            columns=logs_columns,
            description="System and application logs",
            owner="platform_team",
            row_count=50000000,
            storage_format="JSON"
        )
        
        # Customers table (for cross-catalog examples)
        customers_columns = [
            ColumnMetadata("customer_id", "BIGINT", False, True, description="Unique customer identifier"),
            ColumnMetadata("customer_name", "VARCHAR(255)", False, description="Customer full name"),
            ColumnMetadata("registration_date", "DATE", True, description="Customer registration date"),
            ColumnMetadata("id", "BIGINT", False, description="Alternative customer ID")
        ]
        
        self.tables["mysql.sales_db.customers"] = TableMetadata(
            catalog="mysql", schema="sales_db", table="customers",
            columns=customers_columns,
            description="Customer master data",
            owner="sales_team",
            row_count=25000,
            storage_format="MYSQL_INNODB"
        )
        
        # Hive tables metadata
        hive_users_columns = [
            ColumnMetadata("user_id", "BIGINT", False, True, description="Unique user identifier"),
            ColumnMetadata("name", "STRING", True, description="User name"),
            ColumnMetadata("age", "INT", True, description="User age"),
            ColumnMetadata("active", "BOOLEAN", True, description="User active status")
        ]
        
        self.tables["hive.default.users"] = TableMetadata(
            catalog="hive", schema="default", table="users",
            columns=hive_users_columns,
            description="Hive user data warehouse table",
            owner="data_engineering",
            row_count=200000,
            storage_format="ORC"
        )
        
        # User profiles table
        user_profiles_columns = [
            ColumnMetadata("user_id", "BIGINT", False, True, description="User identifier"),
            ColumnMetadata("preferences", "JSON", True, description="User preferences in JSON"),
            ColumnMetadata("profile_data", "MAP<STRING,STRING>", True, description="Additional profile data")
        ]
        
        self.tables["hive.raw.user_profiles"] = TableMetadata(
            catalog="hive", schema="raw", table="user_profiles",
            columns=user_profiles_columns,
            description="Raw user profile data",
            owner="data_engineering",
            row_count=180000,
            storage_format="AVRO"
        )