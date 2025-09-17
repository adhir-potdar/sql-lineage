"""Generic condition handling utilities to avoid condition-specific logic."""

from typing import List, Dict, Any, Union, Optional, Callable
import sqlglot
from sqlglot import expressions as exp
from ..core.models import OperatorType, FilterCondition, JoinCondition


class GenericConditionHandler:
    """Generic handler for all SQL condition types without hardcoded logic."""
    
    # Comprehensive mapping of expression types to operator types
    OPERATOR_MAPPING = {
        exp.EQ: OperatorType.EQ,
        exp.NEQ: OperatorType.NEQ, 
        exp.GT: OperatorType.GT,
        exp.GTE: OperatorType.GTE,
        exp.LT: OperatorType.LT,
        exp.LTE: OperatorType.LTE,
        exp.In: OperatorType.IN,
        exp.Like: OperatorType.LIKE,
        exp.Between: OperatorType.BETWEEN,
        exp.Is: OperatorType.IS_NULL,
    }
    
    @classmethod
    def extract_all_conditions(cls, filter_node: exp.Expression, 
                              column_resolver: Optional[Callable[[str], str]] = None,
                              output_format: str = "dict") -> Union[List[Dict[str, Any]], List[FilterCondition]]:
        """Extract all conditions from a filter node generically.
        
        Args:
            filter_node: SQLGlot expression node to extract conditions from
            column_resolver: Optional function to resolve column references
            output_format: Either 'dict' for dictionary format or 'dataclass' for FilterCondition objects
            
        Returns:
            List of conditions in specified format
        """
        conditions = []
        
        # Handle complex expressions (AND, OR, NOT)
        cls._extract_complex_conditions(filter_node, conditions, column_resolver, output_format)
        
        # Get all condition types dynamically from the mapping
        for exp_type, op_type in cls.OPERATOR_MAPPING.items():
            for comp in filter_node.find_all(exp_type):
                condition = cls._extract_condition_generic(comp, exp_type, op_type, column_resolver, output_format)
                if condition:
                    conditions.append(condition)
        
        return conditions
    
    @classmethod
    def extract_join_conditions(cls, join_node: exp.Expression,
                               output_format: str = "dict") -> Union[List[Dict[str, Any]], List[JoinCondition]]:
        """Extract JOIN conditions generically - supports all operators, not just EQ."""
        conditions = []
        
        # Extract all comparison types for joins (not just EQ)
        for exp_type, op_type in cls.OPERATOR_MAPPING.items():
            for comp in join_node.find_all(exp_type):
                if hasattr(comp, 'left') and hasattr(comp, 'right'):
                    if output_format == "dataclass":
                        condition = JoinCondition(
                            left_column=str(comp.left).strip(),
                            operator=op_type,
                            right_column=str(comp.right).strip()
                        )
                    else:
                        condition = {
                            "left_column": str(comp.left).strip(),
                            "operator": op_type.value,
                            "right_column": str(comp.right).strip()
                        }
                    conditions.append(condition)
        
        return conditions
    
    @classmethod
    def _extract_complex_conditions(cls, node: exp.Expression, conditions: List, 
                                   column_resolver: Optional[Callable], output_format: str) -> None:
        """Handle complex expressions like NOT LIKE, IS NOT NULL, etc."""
        # Handle NOT expressions
        for not_expr in node.find_all(exp.Not):
            if isinstance(not_expr.this, exp.Like):
                condition = cls._create_condition(
                    column=str(not_expr.this.left).strip(),
                    operator="NOT LIKE",
                    value=str(not_expr.this.right).strip(),
                    column_resolver=column_resolver,
                    output_format=output_format
                )
                if condition:
                    conditions.append(condition)
            elif isinstance(not_expr.this, exp.In):
                condition = cls._create_condition(
                    column=str(not_expr.this.this).strip(),
                    operator="NOT IN",
                    value=[str(expr) for expr in not_expr.this.expressions],
                    column_resolver=column_resolver,
                    output_format=output_format
                )
                if condition:
                    conditions.append(condition)
    
    @classmethod
    def _extract_condition_generic(cls, comp: exp.Expression, exp_type: type, 
                                  op_type: OperatorType, column_resolver: Optional[Callable],
                                  output_format: str) -> Optional[Union[Dict[str, Any], FilterCondition]]:
        """Extract condition generically without type-specific logic."""
        try:
            # Generic column extraction
            column = cls._extract_column_generic(comp)
            if column_resolver:
                column = column_resolver(column)
            
            # Generic value extraction based on expression structure
            value = cls._extract_value_generic(comp, exp_type)
            
            # Handle special cases
            operator = cls._get_operator_string(op_type, comp)
            
            return cls._create_condition(column, operator, value, column_resolver, output_format)
            
        except Exception:
            return None
    
    @classmethod
    def _extract_column_generic(cls, comp: exp.Expression) -> str:
        """Extract column name generically."""
        if hasattr(comp, 'this'):
            return str(comp.this).strip()
        elif hasattr(comp, 'left'):
            return str(comp.left).strip()
        else:
            return str(comp).strip()
    
    @classmethod
    def _extract_value_generic(cls, comp: exp.Expression, exp_type: type) -> Union[str, List[str], None]:
        """Extract value generically based on expression structure."""
        # Handle NULL values
        if exp_type == exp.Is:
            return None
            
        # Handle multi-value expressions (IN, BETWEEN)
        if hasattr(comp, 'expressions') and comp.expressions:
            return [str(expr).strip() for expr in comp.expressions]
        elif hasattr(comp, 'low') and hasattr(comp, 'high'):
            # BETWEEN has low/high attributes (direct access)
            return [str(comp.low).strip(), str(comp.high).strip()]
        elif hasattr(comp, 'args') and isinstance(comp.args, dict) and 'low' in comp.args and 'high' in comp.args:
            # BETWEEN has low/high in args dictionary (SQLGlot structure)
            return [str(comp.args['low']).strip(), str(comp.args['high']).strip()]
        elif hasattr(comp, 'expression'):
            # Most binary operations have expression attribute
            return str(comp.expression).strip()
        elif hasattr(comp, 'right'):
            # Some have right attribute
            return str(comp.right).strip()
        else:
            # Fallback to string representation
            return str(comp).strip()
    
    @classmethod
    def _get_operator_string(cls, op_type: OperatorType, comp: exp.Expression) -> str:
        """Get operator string, handling special cases."""
        return op_type.value
    
    @classmethod
    def _create_condition(cls, column: str, operator: str, value: Any, 
                         column_resolver: Optional[Callable], 
                         output_format: str) -> Union[Dict[str, Any], FilterCondition]:
        """Create condition in specified format."""
        if output_format == "dataclass":
            # Convert string operator back to OperatorType enum
            op_enum = next((op for op in OperatorType if op.value == operator), OperatorType.EQ)
            return FilterCondition(
                column=column,
                operator=op_enum,
                value=value
            )
        else:
            return {
                "column": column,
                "operator": operator,
                "value": value
            }
    
    @classmethod
    def get_operator_symbol(cls, node: exp.Expression) -> str:
        """Get operator symbol generically without hardcoded conditions."""
        exp_type = type(node)
        if exp_type in cls.OPERATOR_MAPPING:
            return cls.OPERATOR_MAPPING[exp_type].value
        
        # Fallback to expression key or class name
        return getattr(node, 'key', node.__class__.__name__)