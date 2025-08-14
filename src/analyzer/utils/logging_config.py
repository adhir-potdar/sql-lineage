"""Logging configuration for SQL Lineage Analyzer."""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


class SQLLineageLogger:
    """Centralized logging configuration for SQL Lineage Analyzer."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance for the given name."""
        if name not in cls._loggers:
            cls._setup_logging_if_needed()
            cls._loggers[name] = logging.getLogger(f"sql_lineage.{name}")
        return cls._loggers[name]
    
    @classmethod
    def _setup_logging_if_needed(cls):
        """Set up logging configuration if not already done."""
        if not cls._configured:
            cls.setup_logging()
    
    @classmethod
    def setup_logging(
        cls,
        level: str = None,
        log_file: Optional[str] = None,
        format_string: Optional[str] = None,
        enable_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Configure logging for the SQL Lineage Analyzer.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            format_string: Custom log format string
            enable_console: Whether to log to console
            max_file_size: Maximum size of log file before rotation
            backup_count: Number of backup log files to keep
        """
        # Get log level from environment variable or parameter, default to DEBUG
        log_level = level or os.getenv('SQL_LINEAGE_LOG_LEVEL', 'DEBUG').upper()
        
        # Get log file from environment variable or parameter, default to /tmp/sql_lineage.log
        #if not log_file:
        #    log_file = os.getenv('SQL_LINEAGE_LOG_FILE', '/tmp/sql_lineage.log')
        #always want to log at the standarad out so that it can be streamed to the logging tools & modules.
        log_file =  None
        
        # Default format string
        if not format_string:
            format_string = os.getenv(
                'SQL_LINEAGE_LOG_FORMAT',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Create logging configuration
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': format_string,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {},
            'loggers': {
                'sql_lineage': {
                    'level': log_level,
                    'handlers': [],
                    'propagate': False
                }
            },
            'root': {
                'level': 'WARNING',
                'handlers': []
            }
        }
        
        # Add console handler if enabled
        if enable_console:
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
            config['loggers']['sql_lineage']['handlers'].append('console')
        
        # Add file handler if log file is specified
        if log_file:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'detailed',
                'filename': log_file,
                'maxBytes': max_file_size,
                'backupCount': backup_count,
                'encoding': 'utf8'
            }
            config['loggers']['sql_lineage']['handlers'].append('file')
        
        # Apply configuration
        logging.config.dictConfig(config)
        cls._configured = True
        
        # Log the configuration
        logger = logging.getLogger('sql_lineage.config')
        logger.info(f"Logging configured - Level: {log_level}, Console: {enable_console}, File: {log_file or 'None'}")


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger instance."""
    return SQLLineageLogger.get_logger(name)


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger instance to use
        level: Log level for the messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger, level: int = logging.INFO):
    """
    Decorator to log function performance.
    
    Args:
        logger: Logger instance to use
        level: Log level for the messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log(level, f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{func.__name__} failed after {duration:.3f}s with error: {str(e)}")
                raise
        return wrapper
    return decorator