"""Generic condition handling utilities to avoid condition-specific logic."""

from typing import List, Dict, Any, Union, Optional
import sqlglot
from sqlglot import expressions as exp
from ..core.models import OperatorType


class GenericConditionHandler:
    """Generic handler for all SQL condition types without hardcoded logic."""
    
    # Map expression types to operator types - single source of truth
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
        exp.Is: OperatorType.IS,
        exp.IsNull: OperatorType.IS,
    }
    
    @classmethod
    def extract_all_conditions(cls, filter_node: exp.Expression, 
                              column_resolver=None) -> List[Dict[str, Any]]:
        """Extract all conditions from a filter node generically."""
        conditions = []
        
        # Get all condition types dynamically from the mapping
        for exp_type, op_type in cls.OPERATOR_MAPPING.items():
            for comp in filter_node.find_all(exp_type):
                condition = cls._extract_condition_generic(comp, exp_type, op_type, column_resolver)
                if condition:
                    conditions.append(condition)
        
        return conditions
    
    @classmethod
    def _extract_condition_generic(cls, comp: exp.Expression, exp_type: type, 
                                  op_type: OperatorType, column_resolver=None) -> Optional[Dict[str, Any]]:
        """Extract condition generically without type-specific logic."""
        try:
            # Generic column extraction
            column = str(comp.this) if hasattr(comp, 'this') else str(comp.left) if hasattr(comp, 'left') else None
            if column_resolver:
                column = column_resolver(column)
            
            # Generic value extraction based on expression structure
            value = cls._extract_value_generic(comp, exp_type)
            
            return {
                "column": column,
                "operator": op_type.value if hasattr(op_type, 'value') else str(op_type),
                "value": value
            }
        except Exception:
            return None
    
    @classmethod
    def _extract_value_generic(cls, comp: exp.Expression, exp_type: type) -> Union[str, List[str]]:
        """Extract value generically based on expression structure."""
        # Handle multi-value expressions (IN, BETWEEN)
        if hasattr(comp, 'expressions') and comp.expressions:
            return [str(expr) for expr in comp.expressions]
        elif hasattr(comp, 'low') and hasattr(comp, 'high'):
            # BETWEEN has low/high attributes
            return [str(comp.low), str(comp.high)]
        elif hasattr(comp, 'expression'):
            # Most binary operations have expression attribute
            return str(comp.expression)
        elif hasattr(comp, 'right'):
            # Some have right attribute
            return str(comp.right)
        else:
            # Fallback to string representation
            return str(comp)
    
    @classmethod
    def get_operator_symbol(cls, node: exp.Expression) -> str:
        """Get operator symbol generically without hardcoded conditions."""
        # Use the expression's built-in key or class name as fallback
        exp_type = type(node)
        if exp_type in cls.OPERATOR_MAPPING:
            op_type = cls.OPERATOR_MAPPING[exp_type]
            return op_type.value if hasattr(op_type, 'value') else str(op_type)
        
        # Fallback to expression key or class name
        return getattr(node, 'key', node.__class__.__name__)