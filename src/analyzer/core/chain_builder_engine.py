"""Chain building engine for SQL lineage analysis."""

from typing import Dict, List, Any, Optional, Set
import sqlglot
from ..utils.sql_parsing_utils import is_column_from_table, extract_function_type, extract_clean_column_name, is_subquery_expression, is_subquery_relevant_to_table, extract_columns_referenced_by_table_in_union, clean_table_name_quotes, normalize_entity_name
from ..utils.metadata_utils import create_metadata_entry
from ..utils.regex_patterns import is_aggregate_function
from ..utils.aggregate_utils import is_aggregate_function_for_table, extract_alias_from_expression
from .analyzers.derived_table_analyzer import DerivedTableAnalyzer
from .transformation_engine import TransformationEngine
from ..utils.logging_config import get_logger

class ChainBuilderEngine:
    """Engine for building lineage chains from dependencies."""
    
    def __init__(self, dialect: str = "trino"):
        """Initialize the chain builder engine."""
        self.dialect = dialect
        self.derived_table_analyzer = DerivedTableAnalyzer(dialect)
        self.transformation_engine = TransformationEngine(dialect)
        self.logger = get_logger('core.chain_builder_engine')
        # Performance optimization: cache expensive operations
        self._column_cache = {}
        self._metadata_cache = {}
        self._alias_cache = {}  # Cache alias lookups
        # SURGICAL PERFORMANCE OPTIMIZATION: Add more granular caches
        self._sql_parse_cache = {}  # Cache SQL parsing results
        self._regex_cache = {}  # Cache compiled regex patterns
        self._expression_analysis_cache = {}  # Cache expensive expression analysis
    
    def build_chain_from_dependencies(self, entity_name: str, entity_type: str, 
                                    table_lineage_data: Dict, column_lineage_data: Dict,
                                    result, sql: str, current_depth: int = 0, 
                                    visited_in_path: Set = None, depth: int = 0,
                                    parent_entity: str = None) -> Dict[str, Any]:
        """
        Build comprehensive chain with metadata and transformations.
        Consolidated from lineage_chain_builder.py.
        """
        self.logger.debug(f"Building chain for entity: {entity_name} (type: {entity_type}, depth: {current_depth})")
        
        if visited_in_path is None:
            visited_in_path = set()
            
        # Standard chain building
        chain = {
            "entity": entity_name,
            "entity_type": entity_type,
            "depth": current_depth - 1,
            "dependencies": [],
            "metadata": {"table_columns": []}
        }
        
        # Add table metadata if available
        if hasattr(result, 'metadata') and entity_name in result.metadata:
            table_meta = result.metadata[entity_name]
            metadata = {
                "table_type": table_meta.table_type.value
            }
            
            # Only include non-null values to keep output clean
            if table_meta.schema:
                metadata["schema"] = table_meta.schema
            if table_meta.description:
                metadata["description"] = table_meta.description
            
            # Update existing metadata with table metadata
            chain["metadata"].update(metadata)
        
        # Process table dependencies
        if entity_type == "table" and entity_name in table_lineage_data:
            for dependent_table in table_lineage_data[entity_name]:
                # Skip if would cause circular dependency or exceed depth
                if (depth > 0 and current_depth > depth) or dependent_table in visited_in_path:
                    continue
                
                # Build dependency chain
                visited_in_path_new = visited_in_path | {entity_name}
                dep_chain = self.build_chain_from_dependencies(
                    dependent_table, "table", table_lineage_data, column_lineage_data,
                    result, sql, current_depth + 1, visited_in_path_new, depth, entity_name
                )
                
                # Add transformations to dependency
                dep_chain = self._add_transformations_to_chain(
                    dep_chain, entity_name, dependent_table, result, sql
                )
                
                chain["dependencies"].append(dep_chain)
        
        self.logger.debug(f"Chain building completed for {entity_name} with {len(chain['dependencies'])} dependencies")
        return chain
    
    def _add_transformations_to_chain(self, dep_chain: Dict, entity_name: str, 
                                    dependent_table: str, result, sql: str) -> Dict:
        """Add transformation information to a dependency chain."""
        
        if hasattr(result, 'table_lineage') and hasattr(result.table_lineage, 'transformations'):
            transformations = []
            for transformation_list in result.table_lineage.transformations.values():
                for trans in transformation_list:
                    # Create transformation data structure matching original format
                    trans_data = {
                        "type": "table_transformation",
                        "source_table": str(trans.source_table) if trans.source_table else trans.source_table,
                        "target_table": str(trans.target_table) if trans.target_table else trans.target_table
                    }
                    
                    # Add join information if present
                    if hasattr(trans, 'join_conditions') and trans.join_conditions:
                        join_entry = self._build_join_entry(trans)
                        trans_data["joins"] = [join_entry]
                    
                    # Add filter conditions
                    trans_data = self._add_filter_conditions(trans_data, trans, entity_name, sql)
                    
                    # Add group by columns
                    trans_data = self._add_group_by_columns(trans_data, trans, entity_name, sql)
                    
                    
                    # Add having conditions
                    trans_data = self._add_having_conditions(trans_data, trans, entity_name, sql)
                    
                    # Add order by columns
                    trans_data = self._add_order_by_columns(trans_data, trans, entity_name, sql)
                    
                    transformations.append(trans_data)
            
            # Filter transformations to only include those relevant to this entity
            relevant_transformations = []
            for trans in transformations:
                source_matches = trans.get("source_table") == entity_name
                target_matches = trans.get("target_table") == dependent_table
                if (source_matches and target_matches):
                    relevant_transformations.append(trans)
            
            if relevant_transformations:
                dep_chain["transformations"] = relevant_transformations
        else:
            pass
        
        return dep_chain
    
    def _build_join_entry(self, trans) -> Dict:
        """Build join entry from transformation data."""
        join_entry = {
            "join_type": trans.join_type.value if hasattr(trans, 'join_type') and trans.join_type else "INNER JOIN",
            "right_table": None,
            "conditions": [
                {
                    "left_column": normalize_entity_name(jc.left_column),
                    "operator": jc.operator.value if hasattr(jc.operator, 'value') else str(jc.operator),
                    "right_column": normalize_entity_name(jc.right_column)
                }
                for jc in trans.join_conditions
            ]
        }
        
        # Extract right table from first condition
        if trans.join_conditions:
            first_condition = trans.join_conditions[0]
            if hasattr(first_condition, 'right_column') and '.' in first_condition.right_column:
                # Normalize right_column before extracting table name
                normalized_right_column = normalize_entity_name(first_condition.right_column)
                parts = normalized_right_column.split('.')
                if len(parts) >= 2:
                    # For database.table.column format, take database.table
                    right_table = '.'.join(parts[:-1])
                else:
                    right_table = parts[0]
                join_entry["right_table"] = right_table
        
        return join_entry
    
    def _add_filter_conditions(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add filter conditions to transformation data."""
        
        if hasattr(trans, 'filter_conditions') and trans.filter_conditions:
            # Determine context for column filtering
            is_single_table = (
                trans.target_table == "QUERY_RESULT" or  # Regular SELECT
                (trans.source_table != trans.target_table and  # CTAS/CTE
                 trans.target_table != "QUERY_RESULT")
            )
            
            context_info = {
                'is_single_table_context': is_single_table,
                'tables_in_context': [trans.source_table] if is_single_table else [],
                'sql': sql
            }
            
            relevant_filters = []
            for fc in trans.filter_conditions:
                # Only include filter conditions that reference columns from this entity
                column_belongs = is_column_from_table(fc.column, entity_name, sql, self.dialect)
                if column_belongs:
                    # Normalize column reference to ensure consistent quoting
                    normalized_column = normalize_entity_name(fc.column)
                    relevant_filters.append({
                        "column": normalized_column,
                        "operator": fc.operator.value if hasattr(fc.operator, 'value') else str(fc.operator),
                        "value": fc.value
                    })
            
            if relevant_filters:
                trans_data["filter_conditions"] = relevant_filters
        
        return trans_data
    
    def _add_group_by_columns(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add group by columns to transformation data."""
        if hasattr(trans, 'group_by_columns') and trans.group_by_columns:
            # Determine context for column filtering
            is_single_table = (
                trans.target_table == "QUERY_RESULT" or
                (trans.source_table != trans.target_table and trans.target_table != "QUERY_RESULT")
            )
            
            context_info = {
                'is_single_table_context': is_single_table,
                'tables_in_context': [trans.source_table] if is_single_table else [],
                'sql': sql
            }
            
            relevant_group_by = []
            for col in trans.group_by_columns:
                if is_column_from_table(col, entity_name, sql, self.dialect):
                    # Normalize column reference to ensure consistent quoting
                    normalized_col = normalize_entity_name(col)
                    relevant_group_by.append(normalized_col)
            
            if relevant_group_by:
                trans_data["group_by_columns"] = relevant_group_by
        
        return trans_data
    
    def _add_having_conditions(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add having conditions to transformation data."""
        if hasattr(trans, 'having_conditions') and trans.having_conditions:
            relevant_having = []
            for hc in trans.having_conditions:
                # Having conditions often involve aggregations like COUNT(*) or AVG(u.salary)
                # Check if they reference this entity or if they are general aggregations for this table
                is_relevant = (is_column_from_table(hc.column, entity_name, sql, self.dialect) or 
                             self._is_aggregate_function_for_table(hc.column, entity_name, sql))
                if is_relevant:
                    relevant_having.append({
                        "column": hc.column,
                        "operator": hc.operator.value if hasattr(hc.operator, 'value') else str(hc.operator),
                        "value": hc.value
                    })
            
            if relevant_having:
                trans_data["having_conditions"] = relevant_having
        
        return trans_data
    
    def _add_order_by_columns(self, trans_data: Dict, trans, entity_name: str, sql: str) -> Dict:
        """Add order by columns to transformation data."""
        if hasattr(trans, 'order_by_columns') and trans.order_by_columns:
            # Determine context for column filtering
            is_single_table = (
                trans.target_table == "QUERY_RESULT" or
                (trans.source_table != trans.target_table and trans.target_table != "QUERY_RESULT")
            )
            
            context_info = {
                'is_single_table_context': is_single_table,
                'tables_in_context': [trans.source_table] if is_single_table else [],
                'sql': sql
            }
            
            relevant_order_by = []
            for col in trans.order_by_columns:
                # Extract just the column name part (before ASC/DESC)
                col_name = col.split()[0] if ' ' in col else col
                if is_column_from_table(col_name, entity_name, sql, self.dialect):
                    # Normalize column reference to ensure consistent quoting, while preserving ASC/DESC/NULLS LAST
                    parts = col.split()
                    normalized_col_name = normalize_entity_name(parts[0])
                    if len(parts) > 1:
                        # Reconstruct with ASC/DESC/NULLS LAST parts
                        normalized_col = normalized_col_name + ' ' + ' '.join(parts[1:])
                    else:
                        normalized_col = normalized_col_name
                    relevant_order_by.append(normalized_col)
            
            if relevant_order_by:
                trans_data["order_by_columns"] = relevant_order_by
        
        return trans_data
    
    def _is_aggregate_function_for_table(self, column_expr: str, table_name: str, sql: str = None) -> bool:
        """Check if an aggregate function expression is relevant to a specific table."""
        if not column_expr or not table_name:
            return False
        
        # Check if the expression contains aggregate functions
        aggregate_functions = ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(', 'GROUP_CONCAT(']
        has_aggregate = any(func in column_expr.upper() for func in aggregate_functions)
        
        if not has_aggregate:
            return False
        
        # If it's a simple COUNT(*), it could apply to any table
        if 'COUNT(*)' in column_expr.upper():
            return True
        
        # Check if the expression references columns from this table
        if sql:
            return is_column_from_table(column_expr, table_name, sql, self.dialect)
        
        return False
    
    def create_dependency_chain(self, entity_name: str, entity_type: str, depth: int,
                              metadata: Dict = None, transformations: List = None) -> Dict[str, Any]:
        """Create a basic dependency chain structure."""
        chain = {
            "entity": entity_name,
            "entity_type": entity_type,
            "depth": depth,
            "dependencies": [],
            "metadata": metadata or {"table_columns": []}
        }
        
        if transformations:
            chain["transformations"] = transformations
        
        return chain
    
    def merge_chain_branches(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple chain branches into a single comprehensive chain."""
        if not chains:
            return {}
        
        if len(chains) == 1:
            return chains[0]
        
        # Use the first chain as the base
        merged_chain = chains[0].copy()
        
        # Merge dependencies from all chains
        all_dependencies = merged_chain.get("dependencies", [])
        
        for chain in chains[1:]:
            chain_deps = chain.get("dependencies", [])
            
            # Add dependencies that aren't already present
            existing_entities = {dep["entity"] for dep in all_dependencies}
            for dep in chain_deps:
                if dep["entity"] not in existing_entities:
                    all_dependencies.append(dep)
                    existing_entities.add(dep["entity"])
        
        merged_chain["dependencies"] = all_dependencies
        
        # Merge transformations if present
        all_transformations = merged_chain.get("transformations", [])
        for chain in chains[1:]:
            chain_transformations = chain.get("transformations", [])
            all_transformations.extend(chain_transformations)
        
        if all_transformations:
            merged_chain["transformations"] = all_transformations
        
        return merged_chain
    
    def calculate_max_depth(self, chains: Dict[str, Any]) -> int:
        """Calculate the maximum depth across all chains."""
        max_depth = 0
        
        def get_chain_depth(chain_entity: Dict[str, Any]) -> int:
            """Recursively calculate maximum depth of a chain."""
            current_depth = chain_entity.get('depth', 0)
            max_dep_depth = 0
            
            for dep in chain_entity.get('dependencies', []):
                dep_depth = get_chain_depth(dep)
                max_dep_depth = max(max_dep_depth, dep_depth)
            
            return current_depth + max_dep_depth
        
        for chain in chains.values():
            chain_depth = get_chain_depth(chain)
            max_depth = max(max_depth, chain_depth)
        
        return max_depth
    
    def process_derived_tables(self, sql: str) -> Dict[str, Any]:
        """Process derived tables and return 3-layer lineage structure."""
        if not sql:
            return {}
        
        return self.derived_table_analyzer.analyze_derived_tables(sql)
    
    def merge_derived_table_chains(self, base_chains: Dict, derived_chains: Dict) -> Dict:
        """Merge derived table chains with base chains, giving priority to derived table analysis."""
        if not derived_chains:
            return base_chains
        
        # If we have derived table chains, they provide more complete analysis
        # Merge them with base chains, with derived tables taking priority
        merged = base_chains.copy()
        
        for entity_name, chain_data in derived_chains.items():
            # If this entity exists in base chains, merge metadata
            if entity_name in merged:
                # Keep the derived table structure but merge any additional metadata
                base_metadata = merged[entity_name].get('metadata', {})
                derived_metadata = chain_data.get('metadata', {})
                
                # Merge table metadata (keep derived table columns as they're more complete)
                if 'table_type' in base_metadata and 'table_type' not in derived_metadata:
                    derived_metadata['table_type'] = base_metadata['table_type']
                if 'schema' in base_metadata and 'schema' not in derived_metadata:
                    derived_metadata['schema'] = base_metadata['schema']
                if 'description' in base_metadata and 'description' not in derived_metadata:
                    derived_metadata['description'] = base_metadata['description']
                
                # Update the derived chain with merged metadata
                chain_data['metadata'].update(derived_metadata)
            
            # Use the derived table chain (it has the complete 3-layer structure)
            merged[entity_name] = chain_data
        
        return merged
    
    def _apply_cached_column_data(self, chains: Dict, cached_data: Dict) -> None:
        """Apply cached column data to chains for performance."""
        try:
            chain_metadata = cached_data.get('chain_metadata', {})
            for entity_name, entity_data in chains.items():
                if entity_name in chain_metadata:
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    # Apply cached metadata including table_columns
                    entity_data['metadata'].update(chain_metadata[entity_name])
            self.logger.debug(f"Applied cached column data to {len(chain_metadata)} entities")
        except Exception as e:
            self.logger.warning(f"Failed to apply cached column data: {e}")
            # Fallback: continue without cache
            pass
    
    def _optimize_alias_matching(self, raw_expression: str, alias_to_table: Dict, entity_name: str) -> bool:
        """Optimized alias matching using cached lookups and pre-compiled patterns."""
        # PERFORMANCE OPTIMIZATION: Use more efficient caching strategy
        cache_key = f"{hash(raw_expression)}_{hash(entity_name)}"
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]
        
        # Build reverse lookup: table -> aliases for faster matching
        table_aliases_key = f"table_to_aliases_{hash(tuple(sorted(alias_to_table.items())))}"
        if table_aliases_key not in self._alias_cache:
            table_to_aliases = {}
            for alias, table in alias_to_table.items():
                table_clean = normalize_entity_name(table)
                if table_clean not in table_to_aliases:
                    table_to_aliases[table_clean] = []
                table_to_aliases[table_clean].append(alias)
                if table != table_clean:  # Also add original table name
                    if table not in table_to_aliases:
                        table_to_aliases[table] = []
                    table_to_aliases[table].append(alias)
            self._alias_cache[table_aliases_key] = table_to_aliases
        
        table_to_aliases = self._alias_cache[table_aliases_key]
        result = False
        
        # PERFORMANCE OPTIMIZATION: Use string containment check instead of regex
        # Normalize entity_name to match the normalized keys in table_to_aliases
        entity_name_normalized = normalize_entity_name(entity_name)
        if entity_name_normalized in table_to_aliases:
            aliases = table_to_aliases[entity_name_normalized]
            # Use any() for short-circuit evaluation
            result = any(f'{alias}.' in raw_expression for alias in aliases)
        
        # Cache the result
        self._alias_cache[cache_key] = result
        return result
    
    def add_missing_source_columns(self, chains: Dict, sql: str = None, column_lineage_data: Dict = None, column_transformations_data: Dict = None) -> None:
        """Add missing source columns and handle QUERY_RESULT dependencies - moved from LineageChainBuilder."""
        import time
        start_time = time.time()
        self.logger.info(f"Adding missing source columns for {len(chains)} chain entities")
        
        # PERFORMANCE OPTIMIZATION: Create cache key for expensive operations
        cache_key = f"{hash(sql)}_{len(chains)}"
        if cache_key in self._column_cache:
            self.logger.debug("Using cached column data for performance")
            cached_data = self._column_cache[cache_key]
            self._apply_cached_column_data(chains, cached_data)
            cache_time = time.time() - start_time
            self.logger.info(f"add_missing_source_columns completed (cached) in {cache_time:.3f} seconds")
            return
        
        if not sql:
            self.logger.warning("No SQL provided for source column addition")
            return 
            
        try:
            parse_start = time.time()
            # Parse SQL to get select columns and alias mapping for QUERY_RESULT columns
            import sqlglot
            from ..utils.sql_parsing_utils import is_subquery_expression
            
            # CRITICAL FIX: Helper function to normalize table names for comparison (quoted vs unquoted)
            def normalize_table_for_comparison(table_name):
                if not table_name:
                    return ""
                # Remove quotes and normalize
                return table_name.replace('"', '').replace("'", "")
                
            # CRITICAL FIX: Helper function to normalize column names for duplicate detection
            def normalize_column_for_comparison(column_name):
                """Normalize column names for duplicate detection to prevent malformed name duplication."""
                if not column_name:
                    return ""
                # Remove quotes, normalize case and structure
                normalized = normalize_entity_name(column_name) if '.' in column_name else column_name
                # Extract just the column name part if it's fully qualified
                if '.' in normalized:
                    parts = normalized.split('.')
                    return parts[-1].lower()  # Return just the column name in lowercase
                return normalized.lower()
                
            
            # PERFORMANCE OPTIMIZATION: Cache SQL parsing results
            sql_hash = hash(sql)
            if sql_hash not in self._sql_parse_cache:
                parsed = sqlglot.parse_one(sql, dialect='trino')
                self._sql_parse_cache[sql_hash] = parsed
            else:
                parsed = self._sql_parse_cache[sql_hash]
            
            # Build alias to table mapping using the proper utility function
            from ..utils.sql_parsing_utils import build_alias_to_table_mapping
            
            # PERFORMANCE OPTIMIZATION: Cache alias to table mapping
            alias_cache_key = f"alias_{sql_hash}"
            if alias_cache_key not in self._alias_cache:
                alias_to_table = build_alias_to_table_mapping(sql, self.dialect)
                self._alias_cache[alias_cache_key] = alias_to_table
            else:
                alias_to_table = self._alias_cache[alias_cache_key]
            
            # PERFORMANCE OPTIMIZATION: Extract columns for all tables at once
            from ..utils.sql_parsing_utils import extract_table_columns_from_sql_batch
            entity_names = list(chains.keys())
            # Use processed entity names for cache key to match what's used in column extraction
            processed_entity_names = [name.replace('"', '') for name in entity_names]
            
            batch_cache_key = f"batch_{sql_hash}_{hash(tuple(sorted(processed_entity_names)))}"
            if batch_cache_key not in self._column_cache:
                all_table_columns_batch = extract_table_columns_from_sql_batch(sql, processed_entity_names, self.dialect)
                self._column_cache[batch_cache_key] = all_table_columns_batch
                # Cache for later source column extraction
                self._cached_table_columns_batch = all_table_columns_batch
            else:
                all_table_columns_batch = self._column_cache[batch_cache_key]
                self._cached_table_columns_batch = all_table_columns_batch
            
            parse_time = time.time() - parse_start
            self.logger.debug(f"SQL parsing and batch preprocessing completed in {parse_time:.3f} seconds")
            
            # Get select columns from parsing - handle all SELECT statements including UNIONs
            select_start = time.time()
            select_columns = []
            
            # PERFORMANCE OPTIMIZATION: Cache SELECT column extraction  
            select_cache_key = f"select_{sql_hash}"
            if select_cache_key not in self._expression_analysis_cache:
                # Collect all SELECT statements (handles UNION queries properly)
                select_stmts = list(parsed.find_all(sqlglot.exp.Select))
            else:
                select_columns = self._expression_analysis_cache[select_cache_key]
                select_stmts = []  # Skip processing if cached
            
            # Process each SELECT statement
            for select_stmt in select_stmts:
                # Get the FROM table for this SELECT to track source
                from_clause = select_stmt.find(sqlglot.exp.From)
                stmt_source_table = None
                if from_clause and isinstance(from_clause.this, sqlglot.exp.Table):
                    stmt_source_table = str(from_clause.this.name)
                
                for expr in select_stmt.expressions:
                    if isinstance(expr, sqlglot.exp.Column):
                        raw_expr = str(expr)
                        table_part = str(expr.table) if expr.table else stmt_source_table
                        column_name = str(expr.name) if expr.name else raw_expr
                        
                        # Resolve table aliases for proper source attribution
                        if expr.table and table_part.lower() in alias_to_table:
                            table_part = alias_to_table[table_part.lower()]
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': column_name,
                            'source_table': table_part,
                            'from_table': stmt_source_table  # Track which table this SELECT is from
                        })
                    else:
                        # Handle other expressions (including Alias expressions for subqueries)
                        raw_expr = str(expr)
                        
                        # For Alias expressions, extract the alias name
                        if isinstance(expr, sqlglot.exp.Alias) and expr.alias:
                            column_name = str(expr.alias)
                        else:
                            column_name = raw_expr
                        
                        # For subquery expressions, set source_table to the subquery's table
                        source_table_for_expr = stmt_source_table
                        
                        if is_subquery_expression(raw_expr, self.dialect):
                            # Extract the actual source table from the subquery
                            from ..utils.sql_parsing_utils import extract_tables_from_subquery
                            subquery_tables = extract_tables_from_subquery(raw_expr, self.dialect)
                            if subquery_tables:
                                source_table_for_expr = subquery_tables[0]  # Use the first table from subquery
                            
                        select_columns.append({
                            'raw_expression': raw_expr,
                            'column_name': column_name,
                            'source_table': source_table_for_expr,
                            'from_table': stmt_source_table  # Track which table this SELECT is from
                        })
            
            # Cache the processed select columns for next time
            if select_cache_key not in self._expression_analysis_cache:
                self._expression_analysis_cache[select_cache_key] = select_columns
                
            select_time = time.time() - select_start
            self.logger.debug(f"SELECT column extraction completed in {select_time:.3f} seconds")
            
        except Exception as e:
            self.logger.warning(f"SQL parsing failed: {e}")
            select_columns = []
            alias_to_table = {}
        
        
        # Import required utility functions
        from ..utils.sql_parsing_utils import extract_clean_column_name
        from ..utils.aggregate_utils import (
            is_aggregate_function, extract_alias_from_expression, 
            is_aggregate_function_for_table, query_has_aggregates
        )
        from ..utils.sql_parsing_utils import extract_function_type, infer_query_result_columns_simple
        from .analyzers.ctas_analyzer import (
            is_ctas_target_table, build_ctas_target_columns, add_group_by_to_ctas_transformations
        )
        
        # PERFORMANCE OPTIMIZATION: Pre-compute expensive operations once
        main_processing_start = time.time()
        
        # PERFORMANCE OPTIMIZATION: Pre-compute ALL expensive operations once
        precompute_start = time.time()
        
        # Pre-compute query characteristics to avoid repeated analysis
        is_union_query = 'UNION' in sql.upper()
        needs_aggregate_processing = False
        unique_source_tables = set()
        is_join_query = False
        primary_table = None
        
        # Pre-compute SQL parsing results (avoid repeated parsing in entity loop)
        cached_parsed_sql = None
        cached_primary_table_extraction = None
        
        try:
            # Extract query characteristics once
            if select_columns:
                unique_source_tables = set(sel_col.get('source_table') for sel_col in select_columns if sel_col.get('source_table'))
                is_join_query = len(unique_source_tables) > 1 or 'JOIN' in sql.upper()
            
            # Check for aggregates once
            from ..utils.aggregate_utils import query_has_aggregates
            needs_aggregate_processing = query_has_aggregates(sql)
            
            # MAJOR OPTIMIZATION: Pre-parse SQL once for primary table extraction (eliminating repeated parsing)
            if parsed:
                cached_parsed_sql = parsed  # Use already parsed SQL
                from_clause = parsed.find(sqlglot.exp.From)
                if from_clause and hasattr(from_clause.this, 'name'):
                    table = from_clause.this
                    if table.db:
                        primary_table = f"{table.db}.{table.name}"
                    elif table.catalog and table.db:
                        primary_table = f"{table.catalog}.{table.db}.{table.name}"
                    else:
                        primary_table = str(table.name)
                
                # Cache the primary table extraction result
                cached_primary_table_extraction = primary_table
        except Exception as e:
            self.logger.debug(f"Query characteristic analysis failed: {e}")
        
        # PERFORMANCE OPTIMIZATION: Pre-analyze select columns for expensive operations
        preanalyzed_columns = {}
        if select_columns:
            for i, sel_col in enumerate(select_columns):
                raw_expression = sel_col.get('raw_expression', '')
                preanalyzed_columns[i] = {
                    'is_subquery': is_subquery_expression(raw_expression, self.dialect),
                    'is_aggregate': is_aggregate_function(raw_expression),
                    'raw_expression': raw_expression,
                    'column_name': sel_col.get('column_name'),
                    'source_table': sel_col.get('source_table')
                }
        
        precompute_time = time.time() - precompute_start
        self.logger.debug(f"Pre-computation completed in {precompute_time:.3f} seconds")
        
        processed_entities = 0
        
        # PERFORMANCE OPTIMIZATION: Step 2 - Batch Processing
        # Instead of nested entity->dependency loops, batch process all QUERY_RESULT dependencies
        batch_start = time.time()
        
        # Collect all QUERY_RESULT and CTAS dependencies for batch processing
        query_result_deps = []
        ctas_deps = []
        entity_to_deps = {}  # Map entity to its dependencies for efficient lookup
        
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
                
            entity_to_deps[entity_name] = []
            for dep in entity_data.get('dependencies', []):
                dep_entity = dep.get('entity')
                if dep_entity == 'QUERY_RESULT':
                    query_result_deps.append((entity_name, dep))
                    entity_to_deps[entity_name].append(dep)
                elif is_ctas_target_table(sql, dep_entity):
                    ctas_deps.append((entity_name, dep))
                    entity_to_deps[entity_name].append(dep)
        
        self.logger.debug(f"Batch processing: {len(query_result_deps)} QUERY_RESULT deps, {len(ctas_deps)} CTAS deps")
        
        # BATCH PROCESS: Handle all QUERY_RESULT dependencies together
        if query_result_deps or ctas_deps:
            # Process all dependencies in batches
            for entity_name, dep in query_result_deps + ctas_deps:
                dep_entity = dep.get('entity')
                
                # Use existing logic but without the nested loop overhead
                if dep_entity == 'QUERY_RESULT' or is_ctas_target_table(sql, dep_entity):
                    # Get existing columns to avoid duplication
                    existing_metadata = dep.get('metadata', {})
                    existing_columns = existing_metadata.get('table_columns', [])
                    existing_column_names = {col.get('name') for col in existing_columns}
                    
                    # Add qualified column names to QUERY_RESULT metadata
                    table_columns = list(existing_columns)  # Start with existing columns
                    # Check if this is a CTAS query
                    is_ctas = is_ctas_target_table(sql, dep_entity)
                    
                    if is_ctas:
                        # CTAS query: Add target table columns with special handling for aggregates
                        table_columns = build_ctas_target_columns(sql, select_columns)
                        
                        # Add GROUP BY information to transformations
                        add_group_by_to_ctas_transformations(dep, sql)
                    else:
                        # Check if this is a JOIN or UNION query first
                        # A JOIN query should have multiple different source tables or JOIN keywords
                        unique_source_tables = set(sel_col.get('source_table') for sel_col in select_columns if sel_col.get('source_table'))
                        is_join_query = len(unique_source_tables) > 1 or 'JOIN' in sql.upper()
                        is_union_query = 'UNION' in sql.upper()
                        
                        # Run aggregate-aware processing if needed, or for JOIN/UNION queries
                        needs_aggregate_processing = query_has_aggregates(sql)
                        
                        if needs_aggregate_processing or is_join_query or is_union_query:
                            # PERFORMANCE OPTIMIZATION: Batch compute inferred columns once
                            if not hasattr(self, '_cached_inferred_columns'):
                                self._cached_inferred_columns = infer_query_result_columns_simple(sql, select_columns)
                                self._cached_has_table_prefixes = any('.' in col.get('name', '') for col in self._cached_inferred_columns)
                            
                            inferred_columns = self._cached_inferred_columns
                            has_table_prefixes = self._cached_has_table_prefixes
                            
                            # Special handling for QUERY_RESULT: Add all SELECT columns with transformation details
                            # Only add to the primary table (FROM clause) or the table that most SELECT expressions reference
                            if dep_entity == 'QUERY_RESULT':
                                # PERFORMANCE OPTIMIZATION: Pre-filter select columns by entity
                                if is_union_query:
                                    entity_cache_key = f"union_cols_{entity_name}"
                                    if entity_cache_key not in self._expression_analysis_cache:
                                        normalized_entity = normalize_table_for_comparison(entity_name)
                                        self._expression_analysis_cache[entity_cache_key] = [
                                            col for col in select_columns 
                                            if normalize_table_for_comparison(col.get('source_table', '')) == normalized_entity
                                        ]
                                    entity_select_columns = self._expression_analysis_cache[entity_cache_key]
                                else:
                                    entity_select_columns = select_columns
                                
                                # PERFORMANCE OPTIMIZATION: Use pre-computed primary table (avoid repeated SQL parsing)
                                primary_table = cached_primary_table_extraction
                                
                                # For aggregate queries, process columns for all tables but filter correctly later
                                # For JOIN/UNION queries, add to the main table that most SELECT expressions reference
                                should_add_columns = False
                                
                                if needs_aggregate_processing:
                                    # For aggregate queries, process columns for all tables
                                    should_add_columns = True
                                elif is_join_query or is_union_query:
                                    # For JOIN/UNION queries, always process columns but filter by table relevance
                                    should_add_columns = True
                                else:
                                    # For simple queries, add to primary table
                                    should_add_columns = (entity_name == primary_table)
                                
                                if should_add_columns:
                                    # PERFORMANCE OPTIMIZATION: Batch process columns instead of individual processing
                                    batch_cache_key = f"batch_columns_{entity_name}_{hash(tuple(sel_col.get('raw_expression', '') for sel_col in entity_select_columns))}"
                                    
                                    if batch_cache_key not in self._column_cache:
                                        # Batch process all columns for this entity at once
                                        batch_processed_columns = []
                                        
                                        for i, sel_col in enumerate(entity_select_columns):
                                            # Get pre-analyzed data to avoid expensive function calls
                                            preanalyzed = preanalyzed_columns.get(i, {})
                                            raw_expression = preanalyzed.get('raw_expression', sel_col.get('raw_expression'))
                                            column_name = preanalyzed.get('column_name', sel_col.get('column_name'))
                                            source_table = preanalyzed.get('source_table', sel_col.get('source_table'))
                                            is_subquery_expr = preanalyzed.get('is_subquery', False)
                                            is_aggregate_expr = preanalyzed.get('is_aggregate', False)
                                            
                                            batch_processed_columns.append({
                                                'raw_expression': raw_expression,
                                                'column_name': column_name,
                                                'source_table': source_table,
                                                'is_subquery': is_subquery_expr,
                                                'is_aggregate': is_aggregate_expr,
                                                'index': i
                                            })
                                        
                                        self._column_cache[batch_cache_key] = batch_processed_columns
                                    
                                    # Use batch processed columns
                                    for col_data in self._column_cache[batch_cache_key]:
                                        raw_expression = col_data['raw_expression']
                                        column_name = col_data['column_name']
                                        source_table = col_data['source_table']
                                        is_subquery_expr = col_data['is_subquery']
                                        is_aggregate_expr = col_data['is_aggregate']
                                        
                                        # Skip subquery columns for entities that don't match the subquery's source table
                                        if (is_subquery_expr and normalize_table_for_comparison(source_table) != normalize_table_for_comparison(entity_name)):
                                            continue
                                        
                                        # Skip individual aggregate functions that are part of subqueries
                                        if (is_aggregate_expr and not is_subquery_expr and normalize_table_for_comparison(source_table) != normalize_table_for_comparison(entity_name)):
                                            # Check if this aggregate function is part of a subquery by looking at other columns
                                            skip_aggregate = False
                                            for j, other_col in enumerate(entity_select_columns):
                                                other_preanalyzed = preanalyzed_columns.get(j, {})
                                                other_expr = other_preanalyzed.get('raw_expression', other_col.get('raw_expression', ''))
                                                other_is_subquery = other_preanalyzed.get('is_subquery', False)
                                                if (other_is_subquery and raw_expression in other_expr):
                                                    skip_aggregate = True
                                                    break
                                            if skip_aggregate:
                                                continue
                                        
                                        # For JOIN/UNION queries, only add columns that reference this specific table
                                        column_belongs_to_this_table = False
                                        
                                        # Handle different query types with proper precedence
                                        if needs_aggregate_processing:
                                            # For aggregate queries (including those with JOINs), assign columns based on their source tables
                                            
                                            # Check if this column expression references this entity's table alias (optimized)
                                            column_belongs_to_this_table = self._optimize_alias_matching(raw_expression, alias_to_table, entity_name)
                                            
                                            # PERFORMANCE OPTIMIZATION: Use pre-analyzed aggregate detection
                                            # Special handling for aggregate functions without table prefixes (like COUNT(*))
                                            # These should belong to the primary table
                                            if not column_belongs_to_this_table and is_aggregate_expr:
                                                # Check if no specific table alias is referenced (like COUNT(*))
                                                has_table_reference = any(f'{alias}.' in raw_expression for alias in alias_to_table.keys())
                                                if not has_table_reference and entity_name == primary_table:
                                                    column_belongs_to_this_table = True
                                                    
                                            # Handle direct column references (non-aggregate) (optimized)
                                            if not column_belongs_to_this_table and not is_aggregate_expr:
                                                # For columns like "u.user_id", assign to the table whose alias is referenced
                                                column_belongs_to_this_table = self._optimize_alias_matching(raw_expression, alias_to_table, entity_name)
                                        
                                        elif is_join_query or is_union_query:
                                            # For JOIN/UNION queries without aggregates
                                            # FIRST: Check if this is a subquery expression - these should only belong to their source table
                                            if is_subquery_expr:
                                                from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                                subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                                column_belongs_to_this_table = entity_name in subquery_tables
                                            else:
                                                # SECOND: Check if this column expression references this entity's table alias (optimized)
                                                column_belongs_to_this_table = self._optimize_alias_matching(raw_expression, alias_to_table, entity_name)
                                            
                                            # PERFORMANCE OPTIMIZATION: Use pre-analyzed aggregate detection
                                            # Special handling for aggregate functions without table prefixes (like COUNT(*))
                                            # These should belong to the primary table
                                            if not column_belongs_to_this_table and is_aggregate_expr:
                                                if entity_name == primary_table:
                                                    column_belongs_to_this_table = True
                                            
                                            # For UNION queries, use existing helper to get columns for this table
                                            if is_union_query and not column_belongs_to_this_table:
                                                from ..utils.sql_parsing_utils import get_union_columns_for_table
                                                union_columns = get_union_columns_for_table(sql, entity_name)
                                                # Check if this column expression matches any of the table's UNION columns
                                                for union_col in union_columns:
                                                    if (raw_expression in union_col or 
                                                        column_name in union_col or
                                                        raw_expression == union_col):
                                                        column_belongs_to_this_table = True
                                                        break
                                        
                                        else:
                                            # For simple queries, add all columns to primary table
                                            column_belongs_to_this_table = True
                                        
                                        if not column_belongs_to_this_table:
                                            continue
                                        
                                        # Extract clean name (prefer alias over raw column name)
                                        clean_name = extract_clean_column_name(raw_expression, column_name)
                                        
                                        # PERFORMANCE OPTIMIZATION: Use pre-analyzed aggregate detection
                                        # For non-aggregate columns, use the raw expression format (e.g., "u.department")
                                        # For aggregate columns, use clean name (e.g., "employee_count")
                                        # For UNION queries, use the output column names (aliases)
                                        if is_aggregate_expr:
                                            display_name = clean_name
                                        elif is_union_query:
                                            # For UNION queries, use the output column name (alias)
                                            display_name = column_name
                                        else:
                                            display_name = raw_expression if not ' AS ' in raw_expression.upper() else raw_expression.split(' AS ')[0].strip()
                                        
                                        if clean_name not in existing_column_names:
                                            # For UNION queries in QUERY_RESULT, use table-specific expressions and upstream
                                            if is_union_query and dep_entity == 'QUERY_RESULT':
                                                # Use the raw expression as the name and table-specific upstream
                                                column_name_to_use = raw_expression  # Full expression like "'product' as type"
                                                upstream_ref = f"{entity_name}.{raw_expression}"
                                            else:
                                                column_name_to_use = display_name
                                                # Normalize entity_name to ensure consistent quoting for upstream references and prevent malformed names
                                                entity_name_normalized = normalize_entity_name(entity_name)
                                                clean_name_normalized = normalize_entity_name(clean_name) if '.' in clean_name else clean_name
                                                upstream_ref = f"{entity_name_normalized}.{clean_name_normalized}"
                                                
                                            column_info = {
                                                "name": column_name_to_use,
                                                "upstream": [upstream_ref],
                                                "type": "DIRECT"
                                            }
                                            
                                            # PERFORMANCE OPTIMIZATION: Use pre-analyzed column type detection
                                            # Check if this is a subquery first (before checking for aggregates)
                                            if is_subquery_expr:
                                                from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                                subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                                if subquery_tables:
                                                    main_subquery_table = subquery_tables[0]
                                                    column_info["upstream"] = [f"{main_subquery_table}.{clean_name}"]
                                                column_info["transformation"] = {
                                                    "source_expression": raw_expression,
                                                    "transformation_type": "SUBQUERY",
                                                    "function_type": "SUBQUERY"
                                                }
                                            # Check if this is an aggregate function and add transformation details
                                            elif is_aggregate_expr:
                                                from ..utils.aggregate_utils import extract_upstream_from_aggregate
                                                
                                                function_type = extract_function_type(raw_expression)
                                                # Clean source expression: remove AS clause
                                                source_expr = raw_expression.split(' AS ')[0].strip() if ' AS ' in raw_expression.upper() else raw_expression
                                                
                                                # Extract actual upstream columns from the aggregate expression
                                                upstream_cols = extract_upstream_from_aggregate(raw_expression, sql)
                                                # Convert alias references to full table names
                                                resolved_upstream = []
                                                for upstream_col in upstream_cols:
                                                    if '.' in upstream_col:
                                                        # Extract alias and column parts
                                                        parts = upstream_col.split('.')
                                                        if len(parts) == 2:
                                                            alias, col = parts
                                                            # Resolve alias to full table name
                                                            if alias in alias_to_table:
                                                                table_clean = normalize_entity_name(alias_to_table[alias])
                                                                resolved_upstream.append(f"{table_clean}.{col}")
                                                            else:
                                                                resolved_upstream.append(upstream_col)
                                                        else:
                                                            resolved_upstream.append(upstream_col)
                                                    else:
                                                        # Unqualified column - use current entity
                                                        # Normalize entity_name to ensure consistent quoting
                                                        entity_name_normalized = normalize_entity_name(entity_name)
                                                        resolved_upstream.append(f"{entity_name_normalized}.{upstream_col}")
                                                
                                                # Update upstream with actual source columns instead of calculated column
                                                if resolved_upstream:
                                                    column_info["upstream"] = resolved_upstream
                                                    column_info["type"] = "CALCULATED"  # More accurate than DIRECT
                                                
                                                column_info["transformation"] = {
                                                    "source_expression": source_expr,
                                                    "transformation_type": "AGGREGATE", 
                                                    "function_type": function_type
                                                }
                                            
                                            table_columns.append(column_info)
                                            existing_column_names.add(clean_name)
                            
                            elif has_table_prefixes or is_join_query:
                                # JOIN query: Use qualified names from select columns
                                for sel_col in select_columns:
                                    source_table = sel_col.get('source_table')
                                    raw_expression = sel_col.get('raw_expression')
                                    column_name = sel_col.get('column_name')
                                    
                                    if (source_table in alias_to_table and 
                                        alias_to_table[source_table] == entity_name):
                                        # Regular table-prefixed column
                                        # Normalize entity_name to ensure consistent quoting and prevent malformed names
                                        entity_name_normalized = normalize_entity_name(entity_name)
                                        column_name_normalized = normalize_entity_name(column_name) if '.' in column_name else column_name
                                        upstream_col = f"{entity_name_normalized}.{column_name_normalized}"
                                        
                                        if raw_expression not in existing_column_names:
                                            column_info = {
                                                "name": raw_expression,
                                                "upstream": [upstream_col],
                                                "type": "DIRECT"
                                            }
                                            table_columns.append(column_info)
                                    elif source_table is None and is_subquery_expression(raw_expression, self.dialect):
                                        # Subquery expression - use transformation engine for proper handling
                                        subquery_columns = self.transformation_engine.handle_subquery_functions(sql, entity_name)
                                        for subquery_col in subquery_columns:
                                            col_name = subquery_col.get("name")
                                            if col_name and col_name not in existing_column_names:
                                                table_columns.append(subquery_col)
                                                existing_column_names.add(col_name)
                                    elif source_table is None and is_aggregate_expr:
                                        # Aggregate function column - only add if relevant to this entity
                                        if is_aggregate_function_for_table(raw_expression, entity_name, sql):
                                            alias = extract_alias_from_expression(raw_expression)
                                            func_type = extract_function_type(raw_expression)
                                            
                                            column_name_to_use = alias or column_name
                                            if column_name_to_use not in existing_column_names:
                                                # Normalize entity_name to ensure consistent quoting and prevent malformed names
                                                entity_name_normalized = normalize_entity_name(entity_name)
                                                column_name_to_use_normalized = normalize_entity_name(column_name_to_use) if '.' in column_name_to_use else column_name_to_use
                                                column_info = {
                                                    "name": column_name_to_use,
                                                    "upstream": [f"{entity_name_normalized}.{column_name_to_use_normalized}"],
                                                    "type": "DIRECT",
                                                    "transformation": {
                                                        "source_expression": raw_expression.replace(f" as {alias}", "").replace(f" AS {alias}", "") if alias else raw_expression,
                                                        "transformation_type": "AGGREGATE",
                                                        "function_type": func_type
                                                    }
                                                }
                                                table_columns.append(column_info)
                            elif is_union_query:
                                # UNION query: Add unified output columns with proper upstream attribution
                                from ..utils.sql_parsing_utils import get_union_columns_for_table
                                union_columns = get_union_columns_for_table(sql, entity_name)
                                for column_expr in union_columns:
                                    # Extract clean column name (remove aliases and expressions)
                                    clean_name = extract_clean_column_name(column_expr, column_expr)
                                    
                                    if clean_name not in existing_column_names:
                                        # Normalize entity_name to ensure consistent quoting and prevent malformed names
                                        entity_name_normalized = normalize_entity_name(entity_name)
                                        clean_name_normalized = normalize_entity_name(clean_name) if '.' in clean_name else clean_name
                                        column_info = {
                                            "name": clean_name,
                                            "upstream": [f"{entity_name_normalized}.{clean_name_normalized}"],
                                            "type": "DIRECT"
                                        }
                                        table_columns.append(column_info)
                        else:
                            # Simple query: Add ALL columns from SELECT statement to QUERY_RESULT
                            # For simple queries, QUERY_RESULT should contain all SELECT columns
                            # PERFORMANCE OPTIMIZATION: Use pre-analyzed column data
                            for i, sel_col in enumerate(select_columns):
                                # Get pre-analyzed data to avoid expensive function calls
                                preanalyzed = preanalyzed_columns.get(i, {})
                                raw_expression = preanalyzed.get('raw_expression', sel_col.get('raw_expression'))
                                column_name = preanalyzed.get('column_name', sel_col.get('column_name'))
                                is_subquery_expr = preanalyzed.get('is_subquery', False)
                                is_aggregate_expr = preanalyzed.get('is_aggregate', False)
                                
                                # Extract clean name (prefer alias over raw column name)
                                clean_name = extract_clean_column_name(raw_expression, column_name)
                                
                                if clean_name not in existing_column_names:
                                    # Get source table for proper upstream reference and normalize it
                                    source_table = preanalyzed.get('source_table', sel_col.get('source_table', entity_name))
                                    source_table_normalized = normalize_entity_name(source_table)
                                    clean_name_normalized = normalize_entity_name(clean_name) if '.' in clean_name else clean_name
                                    column_info = {
                                        "name": clean_name,
                                        "upstream": [f"{source_table_normalized}.{clean_name_normalized}"],
                                        "type": "DIRECT"
                                    }
                                    
                                    # PERFORMANCE OPTIMIZATION: Use pre-analyzed column type detection
                                    # Check if this is a subquery first (before checking for aggregates)
                                    if is_subquery_expr:
                                        # Use transformation engine for subquery handling
                                        from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                        subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                        if subquery_tables:
                                            main_subquery_table = subquery_tables[0]
                                            column_info["upstream"] = [f"{main_subquery_table}.{clean_name}"]
                                        column_info["transformation"] = {
                                            "source_expression": raw_expression,
                                            "transformation_type": "SUBQUERY",
                                            "function_type": "SUBQUERY"
                                        }
                                    # Check if this is an aggregate function and add transformation details
                                    elif is_aggregate_expr:
                                        function_type = extract_function_type(raw_expression)
                                        column_info["transformation"] = {
                                            "source_expression": raw_expression,
                                            "transformation_type": "AGGREGATE",
                                            "function_type": function_type
                                        }
                                    
                                    table_columns.append(column_info)
                                    existing_column_names.add(clean_name)
                    
                    # Add subquery result columns to the QUERY_RESULT of tables referenced in subqueries
                    if dep_entity == 'QUERY_RESULT' and not table_columns:
                        # Check if any SELECT columns contain subqueries that reference this entity
                        for sel_col in select_columns:
                            raw_expression = sel_col.get('raw_expression', '')
                            if is_subquery_expression(raw_expression, self.dialect):
                                # Check if this subquery references the current entity
                                from ..utils.sql_parsing_utils import extract_tables_from_subquery
                                subquery_tables = extract_tables_from_subquery(raw_expression, self.dialect)
                                if entity_name in subquery_tables:
                                    # Extract the alias/column name for this subquery result
                                    clean_name = extract_clean_column_name(raw_expression, f"subquery_{len(table_columns)}")
                                    if clean_name not in existing_column_names:
                                        entity_name_normalized = normalize_entity_name(entity_name)
                                        clean_name_normalized = normalize_entity_name(clean_name) if '.' in clean_name else clean_name
                                        column_info = {
                                            "name": clean_name,
                                            "upstream": [f"{entity_name_normalized}.{clean_name_normalized}"],
                                            "type": "DIRECT",
                                            "transformation": {
                                                "source_expression": raw_expression,
                                                "transformation_type": "SUBQUERY",
                                                "function_type": "SUBQUERY"
                                            }
                                        }
                                        table_columns.append(column_info)
                                        existing_column_names.add(clean_name)
                    
                    if table_columns:
                        if 'metadata' not in dep:
                            dep['metadata'] = {}
                        dep['metadata']['table_columns'] = table_columns
                        self.logger.debug(f"Set table_columns for {dep_entity}: {len(table_columns)} columns")
                    else:
                        self.logger.debug(f"No table_columns to set for {dep_entity} (empty list)")
                
            # Handle source table missing columns
            metadata = entity_data.get('metadata', {})
            table_columns = metadata.get('table_columns', [])
            
            # For source tables (depth 0), extract from dependencies
            if entity_data.get('depth') == 0:
                source_columns = set()
                
                # Check if this is a UNION query - use specialized column extraction
                is_union_query = sql and 'UNION' in sql.upper()
                if is_union_query:
                    # For UNION queries, get table-specific columns directly
                    union_table_columns = extract_columns_referenced_by_table_in_union(sql, entity_name, self.dialect)
                    source_columns.update(union_table_columns)
                else:
                    # Look through dependencies to find transformations that reference this table
                    for dep in entity_data.get('dependencies', []):
                        for trans in dep.get('transformations', []):
                            if trans.get('source_table') == entity_name:
                                # Extract filter condition columns - only if they actually belong to this entity
                                for condition in trans.get('filter_conditions', []):
                                    col = condition.get('column', '')
                                    if col:
                                        # Only add columns that actually belong to this entity
                                        belongs = is_column_from_table(col, entity_name, sql, self.dialect)
                                        if belongs:
                                            # Clean column name by removing table prefix if present
                                            clean_col = col.split('.')[-1] if '.' in col else col
                                            source_columns.add(clean_col)
                                
                                # Extract group by columns - only if they actually belong to this entity
                                for group_col in trans.get('group_by_columns', []):
                                    if group_col:
                                        # Only add columns that actually belong to this entity
                                        if is_column_from_table(group_col, entity_name, sql, self.dialect):
                                            # Clean column name by removing table prefix if present
                                            clean_col = group_col.split('.')[-1] if '.' in group_col else group_col
                                            source_columns.add(clean_col)
                                
                                # Extract join condition columns
                                for join in trans.get('joins', []):
                                    for condition in join.get('conditions', []):
                                        for col_key in ['left_column', 'right_column']:
                                            col_ref = condition.get(col_key, '')
                                            if col_ref:
                                                # Only add columns that actually belong to this entity
                                                if is_column_from_table(col_ref, entity_name, sql, self.dialect):
                                                    clean_col = col_ref.split('.')[-1] if '.' in col_ref else col_ref
                                                    source_columns.add(clean_col)
                
                # Always extract columns from SELECT expressions (including aggregate expressions)
                # This ensures columns used in complex expressions like SUM(a - b + c) are captured
                if sql:
                    # PERFORMANCE OPTIMIZATION: Use pre-computed batch results instead of parsing again
                    table_columns_in_select = all_table_columns_batch.get(entity_name, set())
                    source_columns.update(table_columns_in_select)
                
                # Add the columns found in transformations, merging with existing
                if source_columns:
                    # Get existing column names to avoid duplicates
                    existing_columns = {col['name'] for col in table_columns}
                    
                    # Add new columns from transformations
                    for column_name in source_columns:
                        if column_name not in existing_columns:
                            column_info = {
                                "name": column_name,
                                "upstream": [],
                                "type": "SOURCE"
                            }
                            table_columns.append(column_info)
                    
                    # Update the metadata
                    if 'metadata' not in entity_data:
                        entity_data['metadata'] = {}
                    entity_data['metadata']['table_columns'] = table_columns
                    
                    processed_entities += 1
        
        # Complete batch processing timing
        batch_time = time.time() - batch_start
        self.logger.debug(f"Batch processing completed {len(query_result_deps + ctas_deps)} dependencies in {batch_time:.3f} seconds")
        
        # CRITICAL FIX: Add missing source column extraction phase
        # This was missing from batch processing and caused data loss
        source_columns_start = time.time()
        
        self.logger.debug("Adding missing source columns to all entities")
        
        # PERFORMANCE OPTIMIZATION: Use cached batch column extraction from preprocessing
        entity_names = [name.replace('"', '') for name in chains.keys()]
        
        # Use the batch column extraction we already computed in preprocessing
        if hasattr(self, '_cached_table_columns_batch') and self._cached_table_columns_batch:
            all_referenced_columns = self._cached_table_columns_batch
        else:
            # Fall back to batch extraction if not cached
            from ..utils.sql_parsing_utils import extract_table_columns_from_sql_batch
            self.logger.debug(f"Batch extracting columns for {len(entity_names)} entities")
            all_referenced_columns = extract_table_columns_from_sql_batch(sql, entity_names, self.dialect)
            self._cached_table_columns_batch = all_referenced_columns
        
        # Add missing source columns to each entity
        for entity_name, entity_data in chains.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Ensure metadata exists
            if 'metadata' not in entity_data:
                entity_data['metadata'] = {}
            if 'table_columns' not in entity_data['metadata']:
                entity_data['metadata']['table_columns'] = []
            
            # Get existing column names
            existing_columns = entity_data['metadata']['table_columns']
            existing_column_names = {col.get('name', '') for col in existing_columns}
            
            # Add referenced columns that are missing  
            # Use processed entity name (without quotes) to match the keys in all_referenced_columns
            processed_entity_name = entity_name.replace('"', '')
            referenced_columns = all_referenced_columns.get(processed_entity_name, [])
            # Convert to set if it's a list for consistent handling
            if isinstance(referenced_columns, list):
                referenced_columns = set(referenced_columns)
            elif not isinstance(referenced_columns, set):
                referenced_columns = set()
                
            for column_name in referenced_columns:
                if column_name and column_name not in existing_column_names:
                    entity_data['metadata']['table_columns'].append({
                        'name': column_name,
                        'upstream': [],
                        'type': 'SOURCE'
                    })
        
        source_columns_time = time.time() - source_columns_start
        self.logger.debug(f"Source column extraction completed in {source_columns_time:.3f} seconds")
        
        main_processing_time = time.time() - main_processing_start
        total_entities = len(chains)
        self.logger.info(f"Batch processing completed {total_entities} total entities in {main_processing_time:.3f} seconds")
        self.logger.info(f"  - Batch processed: {processed_entities} entities with QUERY_RESULT/CTAS dependencies")
        self.logger.info(f"  - Source columns: Added to all {total_entities} entities")
        
        # Step 4: Handle subquery-specific column detection
        subquery_start = time.time()
        self._add_subquery_columns(chains, sql, select_columns, alias_to_table)
        subquery_time = time.time() - subquery_start
        self.logger.debug(f"Subquery column detection completed in {subquery_time:.3f} seconds")
        
        # PERFORMANCE OPTIMIZATION: Cache the processed data for future use
        try:
            cached_data = {
                'select_columns': select_columns,
                'alias_to_table': alias_to_table,
                'all_table_columns_batch': all_table_columns_batch,
                'chain_metadata': {entity_name: entity_data.get('metadata', {}) for entity_name, entity_data in chains.items()}
            }
            self._column_cache[cache_key] = cached_data
            self.logger.debug(f"Cached column data for key: {cache_key}")
        except Exception as cache_error:
            self.logger.warning(f"Failed to cache column data: {cache_error}")
        
        total_time = time.time() - start_time
        self.logger.info(f"add_missing_source_columns completed in {total_time:.3f} seconds")

    def _add_subquery_columns(self, chains, sql, select_columns, alias_to_table):
        """
        Detect and add columns referenced in nested subquery WHERE clauses.
        This fixes the regression where subquery filter columns were missed.
        """
        try:
            # Enhanced subquery pattern detection - handles all subquery types
            sql_upper = sql.upper()
            has_subquery = (
                # Standard subquery patterns
                'WHERE' in sql_upper and 'SELECT' in sql_upper and (
                    'IN (' in sql_upper or          # WHERE col IN (SELECT...)
                    'EXISTS (' in sql_upper or      # WHERE EXISTS (SELECT...)
                    'NOT EXISTS (' in sql_upper or # WHERE NOT EXISTS (SELECT...)
                    'ANY (' in sql_upper or         # WHERE col = ANY (SELECT...)
                    'ALL (' in sql_upper or         # WHERE col = ALL (SELECT...)
                    'SOME (' in sql_upper           # WHERE col = SOME (SELECT...)
                ) or
                # SELECT clause subqueries  
                ('SELECT' in sql_upper and '(' in sql_upper and 'SELECT' in sql_upper[sql_upper.find('('):]) or
                # FROM clause subqueries (derived tables)
                ('FROM (' in sql_upper and 'SELECT' in sql_upper) or
                # JOIN subqueries
                ('JOIN (' in sql_upper and 'SELECT' in sql_upper) or
                # WITH clause (CTE) - also contains subquery-like structures
                'WITH ' in sql_upper
            )
            
            if not has_subquery:
                self.logger.debug("No subqueries detected, skipping subquery column detection")
                return
                
            self.logger.debug("Subqueries detected, parsing WHERE clause columns")
            
            # Extract subquery WHERE clause column references
            subquery_columns = self._extract_subquery_where_columns(sql)
            
            if not subquery_columns:
                self.logger.debug("No subquery WHERE clause columns found")
                return
            
            # Add detected columns to appropriate table entities
            added_columns_count = 0
            for table_name, column_names in subquery_columns.items():
                # Find the table entity in chains
                table_entity = None
                for entity_name, entity_data in chains.items():
                    if (entity_data.get('entity_type') == 'table' and 
                        (entity_name == table_name or entity_name.endswith(f'.{table_name}'))):
                        table_entity = entity_data
                        break
                
                if table_entity:
                    existing_columns = {col.get('name') for col in 
                                      table_entity.get('metadata', {}).get('table_columns', [])}
                    
                    for col_name in column_names:
                        if col_name not in existing_columns:
                            # Add the missing column
                            table_entity.setdefault('metadata', {}).setdefault('table_columns', []).append({
                                "name": col_name,
                                "upstream": [],
                                "type": "SOURCE"
                            })
                            added_columns_count += 1
                            self.logger.debug(f"Added missing subquery column: {table_name}.{col_name}")
            
            if added_columns_count > 0:
                self.logger.info(f"Added {added_columns_count} missing subquery columns")
            else:
                self.logger.debug("No missing subquery columns to add")
                        
        except Exception as e:
            # Graceful degradation - log and continue
            self.logger.debug(f"Subquery column detection failed: {e}")
            return

    def _extract_subquery_where_columns(self, sql):
        """
        Parse SQL to find column references in subquery WHERE clauses.
        
        Example:
        WHERE u.id IN (SELECT customer_id FROM orders WHERE order_date >= '2023-01-01')
        Returns: {'orders': {'order_date'}}
        """
        import sqlglot
        from sqlglot import expressions as exp
        
        subquery_columns = {}
        
        try:
            parsed = sqlglot.parse_one(sql, dialect='trino')
            
            # Find all subqueries
            for subquery in parsed.find_all(exp.Subquery):
                # Get the SELECT statement within subquery  
                select_stmt = subquery.this
                if not select_stmt:
                    continue
                    
                # Find WHERE clause in the subquery
                where_clause = select_stmt.find(exp.Where)
                if not where_clause:
                    continue
                    
                # Extract table name from FROM clause
                from_clause = select_stmt.find(exp.From)
                if not from_clause:
                    continue
                    
                table_name = str(from_clause.this.name) if from_clause.this else None
                if not table_name:
                    continue
                    
                # Extract column references from WHERE clause
                columns = set()
                for column in where_clause.find_all(exp.Column):
                    # Only include columns that don't have a table qualifier or match the current table
                    if not column.table or str(column.table) in (table_name, table_name.split('.')[-1]):
                        columns.add(str(column.name))
                
                if columns:
                    # Handle fully qualified table names
                    clean_table_name = table_name.split('.')[-1] if '.' in table_name else table_name
                    # Try to find the full qualified name in chains first, fallback to clean name
                    final_table_name = table_name
                    subquery_columns[final_table_name] = columns
                    self.logger.debug(f"Found subquery WHERE columns for {final_table_name}: {columns}")
                    
        except Exception as e:
            self.logger.debug(f"Subquery parsing error: {e}")
            
        return subquery_columns
