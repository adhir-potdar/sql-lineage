"""Centralized dialect auto-detection service."""

import re
from typing import List, Optional
from ..utils.logging_config import get_logger


class DialectAutoDetector:
    """Centralized service for SQL dialect auto-detection and correction."""
    
    # Dialect-specific feature patterns for auto-detection
    DIALECT_SPECIFIC_PATTERNS = {
        r'\bTOP\s+\d+\b': ['tsql'],
        r'\bLIMIT\s+\d+\b': ['postgres', 'mysql', 'trino', 'spark'],
        r'\bDATEADD\s*\(': ['tsql'],
        r'\bDATE_ADD\s*\(': ['mysql'],
        r"'\s*\+\s*'": ['tsql'],  # String concat with +
        r'\|\|': ['postgres', 'sqlite'],  # String concat with ||
        r'\bIFNULL\s*\(': ['mysql'],
        r'\bCOALESCE\s*\(': ['postgres', 'mysql', 'trino', 'spark'],
        r'\bISNULL\s*\(': ['tsql'],
        r'\bLEN\s*\(': ['tsql'],
        r'\bLENGTH\s*\(': ['postgres', 'mysql', 'trino'],
        r'\bGETDATE\s*\(': ['tsql'],
        r'\bNOW\s*\(': ['postgres', 'mysql'],
        r'\bCURRENT_TIMESTAMP\b': ['postgres', 'mysql', 'trino', 'spark'],
        r'\bROWNUM\b': ['oracle'],
        r'\bDUAL\b': ['oracle'],
        r'\bNVL\s*\(': ['oracle'],
        r'\bREGEXP_LIKE\s*\(': ['oracle'],
        r'\bSTR_TO_DATE\s*\(': ['mysql'],
        r'\bDATE_FORMAT\s*\(': ['mysql'],
        r'\bCONCAT_WS\s*\(': ['mysql'],
        r'\bIIF\s*\(': ['tsql'],
        r'\bCHARINDEX\s*\(': ['tsql'],
    }
    
    def __init__(self):
        """Initialize the dialect auto-detector."""
        self.logger = get_logger('dialect_auto_detector')
    
    def detect_and_correct_dialect(self, sql: str, current_dialect: str) -> str:
        """
        Detect and correct SQL dialect based on content analysis.
        
        Args:
            sql: SQL query string to analyze
            current_dialect: Current dialect being used
            
        Returns:
            Corrected dialect string (same as current_dialect if no correction needed)
        """
        if not sql or not sql.strip():
            return current_dialect
            
        try:
            # First, try parsing with current dialect (simulate the error)
            import sqlglot
            sqlglot.parse_one(sql, dialect=current_dialect)
            # If successful, no correction needed
            return current_dialect
            
        except Exception as e:
            # Parsing failed - check if it's a candidate for auto-detection
            candidate_dialects = self._should_attempt_dialect_detection(e, sql, current_dialect)
            
            if not candidate_dialects:
                # No auto-detection candidates - return current dialect
                self.logger.debug(f"No auto-detection candidates for dialect '{current_dialect}'")
                return current_dialect
            
            # Try candidate dialects
            for candidate_dialect in candidate_dialects:
                try:
                    self.logger.info(f"Attempting dialect auto-correction: {current_dialect} → {candidate_dialect}")
                    sqlglot.parse_one(sql, dialect=candidate_dialect)
                    
                    # Success! Return corrected dialect
                    self.logger.info(f"Successfully auto-detected dialect: {candidate_dialect}")
                    return candidate_dialect
                    
                except Exception:
                    continue  # Try next candidate
            
            # All candidates failed - return current dialect
            self.logger.debug(f"All dialect candidates failed, keeping original: {current_dialect}")
            return current_dialect
    
    def _should_attempt_dialect_detection(self, error, sql: str, current_dialect: str) -> List[str]:
        """
        Determine if dialect auto-detection should be attempted based on error and SQL content.
        
        Args:
            error: The parsing error that occurred
            sql: SQL query string
            current_dialect: Current dialect
            
        Returns:
            List of candidate dialects to try (empty if no detection should be attempted)
        """
        error_message = str(error).lower()
        
        # Only attempt detection for parsing errors, not other types of errors
        parsing_error_indicators = [
            'parseError', 'invalid expression', 'unexpected token', 
            'syntax error', 'expected', 'malformed'
        ]
        
        if not any(indicator.lower() in error_message for indicator in parsing_error_indicators):
            return []
        
        # Analyze SQL content for dialect-specific features
        detected_features = []
        candidates = set()
        
        sql_lower = sql.lower()
        
        for pattern, dialects in self.DIALECT_SPECIFIC_PATTERNS.items():
            if re.search(pattern, sql, re.IGNORECASE):
                detected_features.append(pattern)
                candidates.update(dialects)
                self.logger.debug(f"Detected dialect-specific pattern: {pattern}")
        
        # Remove current dialect from candidates (we know it failed)
        candidates.discard(current_dialect.lower())
        
        # Priority ordering for common cases
        candidate_list = list(candidates)
        
        # Prioritize based on detected features
        if re.search(r'\btop\s+\d+', sql_lower):
            # TOP clause - prioritize SQL Server dialects
            candidate_list = [d for d in candidate_list if d == 'tsql'] + [d for d in candidate_list if d != 'tsql']
        elif re.search(r'\blimit\s+\d+', sql_lower):
            # LIMIT clause - prioritize PostgreSQL-family dialects
            preferred_order = ['postgres', 'trino', 'mysql', 'spark']
            candidate_list = sorted(candidate_list, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)
        
        if candidate_list:
            self.logger.info(f"Dialect auto-detection triggered. Detected features: {detected_features}")
            self.logger.info(f"Will try candidate dialects: {candidate_list}")
        
        return candidate_list