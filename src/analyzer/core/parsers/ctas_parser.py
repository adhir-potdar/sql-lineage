"""Parser for CREATE TABLE AS SELECT (CTAS) statements."""

from typing import Dict, List, Any, Optional
import sqlglot
from sqlglot import exp
from .base_parser import BaseParser
from .select_parser import SelectParser
from ...utils.logging_config import get_logger


class CTASParser(BaseParser):
    """Parser for CREATE TABLE AS SELECT statements."""
    
    def __init__(self, dialect: str = "trino"):
        super().__init__(dialect)
        self.select_parser = SelectParser(dialect)
        self.logger = get_logger('parsers.ctas')
    
    def parse(self, sql: str) -> Dict[str, Any]:
        """Parse CTAS statement structure."""
        self.logger.info(f"Parsing CTAS statement (length: {len(sql)})")
        self.logger.debug(f"CTAS SQL: {sql[:200]}..." if len(sql) > 200 else f"CTAS SQL: {sql}")
        
        try:
            ast = self.parse_sql(sql)
            
            if not isinstance(ast, exp.Create):
                self.logger.warning("No CREATE statement found in CTAS SQL")
                return {}
            
            self.logger.debug("CREATE statement found")
            
            result = {
                'target_table': self.parse_target_table(ast),
                'select_query': self.parse_select_query(ast),
                'table_properties': self.parse_table_properties(ast),
                'column_definitions': self.parse_column_definitions(ast)
            }
            
            target_table = result.get('target_table', {}).get('name', 'unknown')
            self.logger.info(f"CTAS parsing completed - target table: {target_table}")
            return result
            
        except Exception as e:
            self.logger.error(f"CTAS parsing failed: {str(e)}", exc_info=True)
            raise
    
    def parse_target_table(self, create_stmt: exp.Create) -> Dict[str, Any]:
        """Parse target table information."""
        target_info = {
            'name': None,
            'schema': None,
            'catalog': None,
            'full_name': None
        }
        
        if create_stmt.this:
            table_ref = create_stmt.this
            
            if isinstance(table_ref, exp.Schema):
                # Handle CREATE TABLE schema (columns)
                table_ref = table_ref.this
            
            target_info['name'] = self.extract_table_name(table_ref)
            target_info['full_name'] = str(table_ref)
            
            # Extract schema and catalog if present
            parts = str(table_ref).split('.')
            if len(parts) == 3:
                target_info['catalog'] = parts[0]
                target_info['schema'] = parts[1]
                target_info['name'] = parts[2]
            elif len(parts) == 2:
                target_info['schema'] = parts[0]
                target_info['name'] = parts[1]
        
        return target_info
    
    def parse_select_query(self, create_stmt: exp.Create) -> Dict[str, Any]:
        """Parse the SELECT query part of CTAS."""
        select_query_data = {}
        
        # Find the SELECT statement in the CTAS
        select_stmt = None
        
        # Look for SELECT in different places
        if hasattr(create_stmt, 'expression') and create_stmt.expression:
            if isinstance(create_stmt.expression, exp.Select):
                select_stmt = create_stmt.expression
        
        # Alternative: find SELECT in the AST
        if not select_stmt:
            for node in create_stmt.find_all(exp.Select):
                select_stmt = node
                break
        
        if select_stmt:
            select_query_data = self.select_parser.parse(str(select_stmt))
            select_query_data['sql'] = str(select_stmt)
        
        return select_query_data
    
    def parse_table_properties(self, create_stmt: exp.Create) -> Dict[str, Any]:
        """Parse table properties and options."""
        properties = {
            'temporary': False,
            'if_not_exists': False,
            'replace': False,
            'external': False,
            'storage_format': None,
            'location': None,
            'partitioned_by': [],
            'clustered_by': [],
            'custom_properties': {}
        }
        
        # Check for temporary table
        if hasattr(create_stmt, 'temporary') and create_stmt.temporary:
            properties['temporary'] = True
        
        # Check for IF NOT EXISTS
        if hasattr(create_stmt, 'exists') and create_stmt.exists:
            properties['if_not_exists'] = True
        
        # Check for REPLACE
        if hasattr(create_stmt, 'replace') and create_stmt.replace:
            properties['replace'] = True
        
        # Parse properties from the properties list
        if hasattr(create_stmt, 'properties') and create_stmt.properties:
            for prop in create_stmt.properties.expressions:
                self._parse_property(prop, properties)
        
        return properties
    
    def _parse_property(self, prop, properties: Dict[str, Any]):
        """Parse individual table property."""
        prop_str = str(prop).lower()
        
        if 'format' in prop_str:
            # Extract storage format
            if '=' in prop_str:
                properties['storage_format'] = prop_str.split('=')[1].strip().strip("'\"")
        elif 'location' in prop_str:
            # Extract location
            if '=' in prop_str:
                properties['location'] = prop_str.split('=')[1].strip().strip("'\"")
        elif 'partitioned' in prop_str:
            # Handle partitioning
            properties['partitioned_by'] = self._extract_partition_columns(prop)
        elif 'clustered' in prop_str:
            # Handle clustering
            properties['clustered_by'] = self._extract_cluster_columns(prop)
        else:
            # Custom property
            if '=' in prop_str:
                key, value = prop_str.split('=', 1)
                properties['custom_properties'][key.strip()] = value.strip().strip("'\"")
    
    def _extract_partition_columns(self, prop) -> List[str]:
        """Extract partition column names."""
        # This is a simplified extraction - real implementation would need
        # more sophisticated parsing based on the specific dialect
        prop_str = str(prop)
        # Look for column names in parentheses
        import re
        matches = re.findall(r'\((.*?)\)', prop_str)
        if matches:
            columns = [col.strip() for col in matches[0].split(',')]
            return columns
        return []
    
    def _extract_cluster_columns(self, prop) -> List[str]:
        """Extract cluster column names."""
        # Similar to partition columns
        return self._extract_partition_columns(prop)
    
    def parse_column_definitions(self, create_stmt: exp.Create) -> List[Dict[str, Any]]:
        """Parse explicit column definitions if present."""
        columns = []
        
        # Look for schema with column definitions
        if hasattr(create_stmt, 'this') and isinstance(create_stmt.this, exp.Schema):
            schema = create_stmt.this
            
            if hasattr(schema, 'expressions'):
                for col_def in schema.expressions:
                    if isinstance(col_def, exp.ColumnDef):
                        column_info = {
                            'name': col_def.this.name if col_def.this else None,
                            'data_type': str(col_def.kind) if col_def.kind else None,
                            'nullable': True,  # Default assumption
                            'primary_key': False,
                            'constraints': []
                        }
                        
                        # Parse constraints
                        if hasattr(col_def, 'constraints'):
                            for constraint in col_def.constraints:
                                constraint_info = self._parse_column_constraint(constraint)
                                if constraint_info:
                                    column_info['constraints'].append(constraint_info)
                                    
                                    # Update flags based on constraints
                                    if constraint_info['type'] == 'NOT NULL':
                                        column_info['nullable'] = False
                                    elif constraint_info['type'] == 'PRIMARY KEY':
                                        column_info['primary_key'] = True
                                        column_info['nullable'] = False
                        
                        columns.append(column_info)
        
        return columns
    
    def _parse_column_constraint(self, constraint) -> Optional[Dict[str, Any]]:
        """Parse column constraint."""
        constraint_info = {
            'type': None,
            'details': None
        }
        
        if isinstance(constraint, exp.NotNullColumnConstraint):
            constraint_info['type'] = 'NOT NULL'
        elif isinstance(constraint, exp.PrimaryKeyColumnConstraint):
            constraint_info['type'] = 'PRIMARY KEY'
        elif isinstance(constraint, exp.UniqueColumnConstraint):
            constraint_info['type'] = 'UNIQUE'
        elif isinstance(constraint, exp.CheckColumnConstraint):
            constraint_info['type'] = 'CHECK'
            constraint_info['details'] = str(constraint.this)
        else:
            constraint_info['type'] = str(type(constraint).__name__)
            constraint_info['details'] = str(constraint)
        
        return constraint_info
    
    def get_ctas_lineage(self, sql: str) -> Dict[str, Any]:
        """Get complete CTAS lineage information."""
        ctas_data = self.parse(sql)
        
        if not ctas_data:
            return {}
        
        lineage = {
            'type': 'CTAS',
            'source_analysis': self._analyze_sources(ctas_data),
            'target_analysis': self._analyze_target(ctas_data),
            'transformations': self._analyze_transformations(ctas_data),
            'column_lineage': self._analyze_column_lineage(ctas_data)
        }
        
        return lineage
    
    def _analyze_sources(self, ctas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze source tables and their usage."""
        select_data = ctas_data.get('select_query', {})
        
        source_analysis = {
            'tables': [],
            'total_tables': 0,
            'has_joins': False,
            'has_subqueries': False
        }
        
        # Analyze FROM tables
        from_tables = select_data.get('from_tables', [])
        for table in from_tables:
            source_analysis['tables'].append({
                'name': table.get('table_name'),
                'alias': table.get('alias'),
                'type': 'base_table',
                'is_subquery': table.get('is_subquery', False)
            })
            
            if table.get('is_subquery'):
                source_analysis['has_subqueries'] = True
        
        # Analyze JOIN tables
        joins = select_data.get('joins', [])
        if joins:
            source_analysis['has_joins'] = True
            for join in joins:
                source_analysis['tables'].append({
                    'name': join.get('table_name'),
                    'alias': join.get('alias'),
                    'type': 'joined_table',
                    'join_type': join.get('join_type')
                })
        
        source_analysis['total_tables'] = len(source_analysis['tables'])
        
        return source_analysis
    
    def _analyze_target(self, ctas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze target table creation."""
        target_table = ctas_data.get('target_table', {})
        table_properties = ctas_data.get('table_properties', {})
        
        return {
            'table_name': target_table.get('name'),
            'full_name': target_table.get('full_name'),
            'schema': target_table.get('schema'),
            'catalog': target_table.get('catalog'),
            'properties': table_properties
        }
    
    def _analyze_transformations(self, ctas_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze transformations applied in CTAS."""
        transformations = []
        select_data = ctas_data.get('select_query', {})
        
        # Add filter transformations
        where_conditions = select_data.get('where_conditions', [])
        if where_conditions:
            transformations.append({
                'type': 'FILTER',
                'conditions': where_conditions
            })
        
        # Add join transformations
        joins = select_data.get('joins', [])
        for join in joins:
            transformations.append({
                'type': 'JOIN',
                'join_type': join.get('join_type'),
                'table': join.get('table_name'),
                'conditions': join.get('conditions', [])
            })
        
        # Add aggregation transformations
        group_by = select_data.get('group_by', [])
        if group_by:
            transformations.append({
                'type': 'GROUP_BY',
                'columns': group_by
            })
        
        return transformations
    
    def _analyze_column_lineage(self, ctas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze column-level lineage for CTAS."""
        select_data = ctas_data.get('select_query', {})
        select_columns = select_data.get('select_columns', [])
        
        column_lineage = {
            'output_columns': [],
            'column_transformations': [],
            'source_columns': set()
        }
        
        for col in select_columns:
            output_col = {
                'name': col.get('column_name'),
                'alias': col.get('alias'),
                'source_table': col.get('source_table'),
                'expression': col.get('raw_expression'),
                'is_computed': col.get('is_aggregate') or col.get('is_window_function')
            }
            
            column_lineage['output_columns'].append(output_col)
            
            # Create column transformation for computed columns (aggregates, etc.)
            if output_col['is_computed']:
                # Use alias as the column name, and show transformation as source expression
                target_column_name = col.get('alias') or col.get('column_name')
                source_expression = self._extract_source_from_expression(col.get('raw_expression', ''), target_column_name)
                
                col_transformation = {
                    'column_name': target_column_name,  # This will be the column name in the table
                    'source_expression': source_expression,  # e.g., "SUM(amount)", "UPPER(email)"
                    'transformation_type': 'AGGREGATE' if col.get('is_aggregate') else 'FUNCTION',
                    'function_type': self._extract_function_type(col.get('raw_expression', '')),
                    'full_expression': col.get('raw_expression', '')  # Keep full expression for reference
                }
                column_lineage['column_transformations'].append(col_transformation)
            
            # Track source columns
            if col.get('source_table') and col.get('column_name'):
                column_lineage['source_columns'].add(
                    f"{col['source_table']}.{col['column_name']}"
                )
        
        column_lineage['source_columns'] = list(column_lineage['source_columns'])
        
        return column_lineage
    
    def _extract_function_type(self, expression: str) -> str:
        """Extract function type from expression generically."""
        if not expression:
            return 'UNKNOWN'
        
        # Convert to uppercase for matching
        expr_upper = expression.upper().strip()
        
        # Extract function name from expressions like "COUNT(*)", "SUM(amount)", etc.
        # Match pattern: FUNCTION_NAME(...)
        import re
        function_match = re.match(r'^([A-Z_]+)\s*\(', expr_upper)
        if function_match:
            return function_match.group(1)
        
        return 'UNKNOWN'
    
    def _extract_source_from_expression(self, expression: str, target_name: str) -> str:
        """Extract the source part from transformation expression."""
        if not expression:
            return 'UNKNOWN'
        
        # Remove the alias part if present (e.g., "SUM(amount) AS total" -> "SUM(amount)")
        expr = expression.strip()
        
        # Split by " AS " (case insensitive) and take the first part
        import re
        as_split = re.split(r'\s+AS\s+', expr, flags=re.IGNORECASE)
        if len(as_split) > 1:
            return as_split[0].strip()
        
        # If no AS clause, return the expression as is
        return expr