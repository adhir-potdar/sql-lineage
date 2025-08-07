"""Main SQL lineage analyzer - Wrapper for modular analyzers."""

from typing import Optional, Dict, Any, List
import sqlglot
from sqlglot import Expression
import json

from .models import LineageResult, TableMetadata
from .extractor import LineageExtractor
from .parsers import SelectParser, TransformationParser, CTEParser, CTASParser, InsertParser, UpdateParser
from ..utils.validation import validate_sql_input
from ..utils.sql_parsing_utils import TableNameRegistry, CompatibilityMode
from ..utils.logging_config import get_logger

# Import the new modular analyzers
from .analyzers import (
    BaseAnalyzer,
    SelectAnalyzer,
    InsertAnalyzer,
    UpdateAnalyzer,
    CTEAnalyzer,
    CTASAnalyzer,
    TransformationAnalyzer,
    LineageChainBuilder
)

class SQLLineageAnalyzer:
    """Main SQL lineage analyzer class - Wrapper for modular analyzers."""
    
    def __init__(self, dialect: str = "trino", compatibility_mode: str = CompatibilityMode.FULL):
        """
        Initialize the SQL lineage analyzer.
        
        Args:
            dialect: SQL dialect to use for parsing
            compatibility_mode: Table name normalization mode
        """
        self.logger = get_logger('analyzer')
        self.logger.info(f"Initializing SQLLineageAnalyzer with dialect: {dialect}, compatibility_mode: {compatibility_mode}")
        
        self.dialect = dialect
        self.compatibility_mode = compatibility_mode
        self.table_registry = TableNameRegistry(dialect, compatibility_mode)
        self.extractor = LineageExtractor(dialect, compatibility_mode)
        
        self.logger.debug("Core components initialized")
        
        # Initialize modular parsers as core components
        self.select_parser = SelectParser(dialect)
        self.transformation_parser = TransformationParser(dialect)
        self.cte_parser = CTEParser(dialect)
        self.ctas_parser = CTASParser(dialect)
        self.insert_parser = InsertParser(dialect)
        self.update_parser = UpdateParser(dialect)
        
        # Initialize the new modular analyzers with shared registry
        self.logger.debug("Initializing modular analyzers")
        self.base_analyzer = BaseAnalyzer(dialect, compatibility_mode, self.table_registry)
        self.select_analyzer = SelectAnalyzer(dialect, compatibility_mode, self.table_registry)
        self.insert_analyzer = InsertAnalyzer(dialect, compatibility_mode, self.table_registry)
        self.update_analyzer = UpdateAnalyzer(dialect, compatibility_mode, self.table_registry)
        self.cte_analyzer = CTEAnalyzer(dialect, main_analyzer=self, table_registry=self.table_registry)
        self.ctas_analyzer = CTASAnalyzer(dialect, compatibility_mode, self.table_registry)
        self.transformation_analyzer = TransformationAnalyzer(dialect, compatibility_mode, self.table_registry)
        self.lineage_chain_builder = LineageChainBuilder(dialect, main_analyzer=self, table_registry=self.table_registry)
        
        self.logger.info("SQLLineageAnalyzer initialization completed successfully")
    
    def analyze_comprehensive(self, sql: str) -> Dict[str, Any]:
        """
        Comprehensive analysis using modular parsers.
        
        Args:
            sql: SQL query string to analyze
            
        Returns:
            Comprehensive analysis result
        """
        self.logger.info(f"Starting comprehensive analysis for SQL (length: {len(sql)})")
        self.logger.debug(f"SQL query: {sql[:200]}..." if len(sql) > 200 else f"SQL query: {sql}")
        
        try:
            # Determine SQL type and route to appropriate parser
            sql_type = self.base_analyzer._determine_sql_type(sql)
            self.logger.info(f"Detected SQL type: {sql_type}")
            
            analysis_result = {
                'sql': sql,
                'sql_type': sql_type,
                'dialect': self.dialect,
                'success': True
            }
            
            try:
                self.logger.debug(f"Routing to {sql_type} analyzer")
                if sql_type == 'SELECT':
                    analysis_result.update(self.select_analyzer.analyze_select(sql))
                elif sql_type == 'CTE':
                    analysis_result.update(self.cte_analyzer.analyze_cte(sql))
                elif sql_type == 'CTAS':
                    analysis_result.update(self.ctas_analyzer.analyze_ctas(sql))
                elif sql_type == 'INSERT':
                    analysis_result.update(self.insert_analyzer.analyze_insert(sql))
                elif sql_type == 'UPDATE':
                    analysis_result.update(self.update_analyzer.analyze_update(sql))
                else:
                    self.logger.warning(f"Using generic analysis for SQL type: {sql_type}")
                    analysis_result.update(self._analyze_generic(sql))
                
                self.logger.info(f"Analysis completed successfully for {sql_type}")
            except Exception as analysis_error:
                self.logger.error(f"Analysis failed for {sql_type}: {str(analysis_error)}", exc_info=True)
                analysis_result['success'] = False
                analysis_result['error'] = f"Analysis error for {sql_type}: {str(analysis_error)}"
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {str(e)}", exc_info=True)
            return {
                'sql': sql,
                'error': str(e),
                'success': False
            }
    
    def _analyze_generic(self, sql: str) -> Dict[str, Any]:
        """Analyze other types of SQL statements."""
        self.logger.debug("Attempting generic analysis")
        # For now, try to extract what we can using the select parser
        try:
            select_data = self.select_parser.parse(sql)
            self.logger.info("Generic analysis completed with partial data")
            return {
                'partial_analysis': select_data,
                'note': 'Generic analysis - limited information available'
            }
        except Exception as e:
            self.logger.warning(f"Generic analysis failed: {str(e)}")
            return {
                'error': 'Unable to parse this SQL type'
            }
    
    def analyze(self, sql: str, **kwargs) -> LineageResult:
        """
        Analyze SQL query for lineage information.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options (for future extensibility)
            
        Returns:
            LineageResult containing table and column lineage information
        """
        self.logger.info(f"Starting analysis for SQL (length: {len(sql)})")
        self.logger.debug(f"SQL query: {sql[:200]}..." if len(sql) > 200 else f"SQL query: {sql}")
        
        # Validate input
        validation_error = validate_sql_input(sql)
        if validation_error:
            self.logger.error(f"SQL validation failed: {validation_error}")
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[validation_error]
            )
        
        try:
            # Parse SQL into AST
            self.logger.debug("Parsing SQL into AST")
            parsed = sqlglot.parse_one(sql, dialect=self.dialect)
            self.logger.info(f"SQL parsed successfully using dialect: {self.dialect}")
            
            # Extract lineage using the extractor
            self.logger.debug("Extracting table lineage")
            table_lineage = self.extractor.extract_table_lineage(parsed)
            self.logger.debug("Extracting column lineage")
            column_lineage = self.extractor.extract_column_lineage(parsed)
            self.logger.info("Lineage extraction completed")
            
            # Extract metadata
            self.logger.debug("Collecting metadata")
            metadata = self._collect_metadata(table_lineage)
            
            result = LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=table_lineage,
                column_lineage=column_lineage,
                metadata=metadata
            )
            self.logger.info("Analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            return LineageResult(
                sql=sql,
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"Analysis failed: {str(e)}"]
            )
    
    def _collect_metadata(self, table_lineage) -> Dict[str, TableMetadata]:
        """Collect metadata for tables involved in lineage."""
        metadata = {}
        
        # Collect all table names from upstream and downstream
        all_tables = set()
        for target, sources in table_lineage.upstream.items():
            all_tables.add(target)
            all_tables.update(sources)
        
        # No external metadata - return empty metadata for each table
        # Tables will be analyzed purely from SQL context
        for table_name in all_tables:
            metadata[table_name] = None
        
        return metadata
    
    
    def set_dialect(self, dialect: str) -> None:
        """Set the SQL dialect for parsing."""
        old_dialect = self.dialect
        self.dialect = dialect
        self.logger.info(f"Dialect changed from {old_dialect} to {dialect}")
    
    def get_lineage_result(self, sql: str, **kwargs) -> LineageResult:
        """
        Get the LineageResult object for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            LineageResult object containing table and column lineage information
        """
        self.logger.debug("get_lineage_result called")
        return self.analyze(sql, **kwargs)
    
    def get_lineage_json(self, sql: str, **kwargs) -> str:
        """
        Get the JSON representation of lineage analysis for a SQL query.
        
        Args:
            sql: SQL query string to analyze
            **kwargs: Additional options
            
        Returns:
            JSON string representation of the lineage analysis
        """
        self.logger.debug("get_lineage_json called")
        import json
        result = self.analyze(sql, **kwargs)
        json_result = json.dumps(result.to_dict(), indent=2)
        self.logger.info(f"Generated JSON result (length: {len(json_result)})")
        return json_result
    
    def analyze_file(self, file_path: str, **kwargs) -> LineageResult:
        """
        Analyze SQL from a file.
        
        Args:
            file_path: Path to SQL file to analyze
            **kwargs: Additional options
            
        Returns:
            LineageResult object containing table and column lineage information
        """
        self.logger.info(f"Analyzing SQL file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql = f.read()
            self.logger.debug(f"Read {len(sql)} characters from file: {file_path}")
            return self.analyze(sql, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {str(e)}", exc_info=True)
            return LineageResult(
                sql="",
                dialect=self.dialect,
                table_lineage=self.extractor.extract_table_lineage(sqlglot.expressions.Anonymous()),
                column_lineage=self.extractor.extract_column_lineage(sqlglot.expressions.Anonymous()),
                errors=[f"Failed to read file {file_path}: {str(e)}"]
            )
    
    # Lineage chain methods - delegate to LineageChainBuilder
    def get_table_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """Get the table lineage chain for a SQL query."""
        return self.lineage_chain_builder.get_table_lineage_chain(sql, chain_type, depth, **kwargs)
    
    def get_table_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get the JSON representation of table lineage chain for a SQL query."""
        return self.lineage_chain_builder.get_table_lineage_chain_json(sql, chain_type, depth, **kwargs)
    
    def get_column_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> Dict[str, Any]:
        """Get the column lineage chain for a SQL query."""
        return self.lineage_chain_builder.get_column_lineage_chain(sql, chain_type, depth, **kwargs)
    
    def get_column_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 1, **kwargs) -> str:
        """Get the JSON representation of column lineage chain for a SQL query."""
        return self.lineage_chain_builder.get_column_lineage_chain_json(sql, chain_type, depth, **kwargs)
    
    def get_lineage_chain(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Get the comprehensive lineage chain combining table and column lineage."""
        return self.lineage_chain_builder.get_lineage_chain(sql, chain_type, depth, target_entity, **kwargs)
    
    def get_lineage_chain_json(self, sql: str, chain_type: str = "upstream", depth: int = 0, target_entity: Optional[str] = None, **kwargs) -> str:
        """Get the JSON representation of comprehensive lineage chain for a SQL query."""
        return self.lineage_chain_builder.get_lineage_chain_json(sql, chain_type, depth, target_entity, **kwargs)
    
    # Transformation methods - delegate to TransformationAnalyzer
    def parse_transformations(self, sql: str) -> Dict[str, Any]:
        """Parse transformations using modular parser."""
        return self.transformation_analyzer.parse_transformations(sql)
    
    def parse_cte(self, sql: str) -> Dict[str, Any]:
        """Parse CTE using modular parser."""
        return self.cte_parser.parse(sql)
    
    def parse_select(self, sql: str) -> Dict[str, Any]:
        """Parse SELECT using modular parser."""
        return self.select_parser.parse(sql)