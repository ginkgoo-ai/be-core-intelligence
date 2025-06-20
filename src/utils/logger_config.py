import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Optional


class LoggerConfig:
    """Centralized logging configuration for the entire application"""
    
    _configured = False
    
    @classmethod
    def setup(cls, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
        """Setup application-wide logging configuration
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory to store log files
            
        Returns:
            Configured root logger
        """
        if cls._configured:
            return logging.getLogger()
            
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure logging format
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        
        # Create formatters
        formatter = logging.Formatter(log_format)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers to avoid duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for general logs
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Separate error log file
        error_handler = logging.FileHandler(
            os.path.join(log_dir, f"error_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Separate debug log file
        debug_handler = logging.FileHandler(
            os.path.join(log_dir, f"debug_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        root_logger.addHandler(debug_handler)
        
        cls._configured = True
        return root_logger
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for a specific module/class
        
        Args:
            name: Name of the logger (usually __name__)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)
    
    @staticmethod
    def log_exception(logger: logging.Logger, message: str = "Exception occurred", 
                     exc_info: Optional[Exception] = None):
        """Log an exception with full traceback
        
        Args:
            logger: Logger instance to use
            message: Custom message to include
            exc_info: Exception instance (if None, current exception is used)
        """
        logger.error("=" * 60)
        logger.error(f"🚨 {message}")
        if exc_info:
            logger.error(f"Exception type: {type(exc_info).__name__}")
            logger.error(f"Exception message: {str(exc_info)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
    
    @staticmethod
    def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
        """Log function entry with parameters
        
        Args:
            logger: Logger instance
            func_name: Name of the function
            **kwargs: Function parameters to log
        """
        params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.debug(f"🔵 Entering {func_name}({params_str})")
    
    @staticmethod
    def log_function_exit(logger: logging.Logger, func_name: str, result=None):
        """Log function exit with result
        
        Args:
            logger: Logger instance
            func_name: Name of the function
            result: Function return value
        """
        if result is not None:
            logger.debug(f"🔵 Exiting {func_name}() -> {result}")
        else:
            logger.debug(f"🔵 Exiting {func_name}()")
    
    @staticmethod
    def log_database_operation(logger: logging.Logger, operation: str, table: str, 
                              record_id: str = None, success: bool = True, 
                              error: str = None):
        """Log database operations
        
        Args:
            logger: Logger instance
            operation: Type of operation (CREATE, READ, UPDATE, DELETE)
            table: Table name
            record_id: Record identifier
            success: Whether operation was successful
            error: Error message if operation failed
        """
        if success:
            id_str = f" (ID: {record_id})" if record_id else ""
            logger.info(f"💾 DB {operation} successful: {table}{id_str}")
        else:
            id_str = f" (ID: {record_id})" if record_id else ""
            logger.error(f"💾 DB {operation} failed: {table}{id_str} - {error}")
    
    @staticmethod
    def log_api_request(logger: logging.Logger, method: str, url: str, 
                       status_code: int = None, response_time: float = None):
        """Log API requests
        
        Args:
            logger: Logger instance
            method: HTTP method
            url: Request URL
            status_code: Response status code
            response_time: Response time in seconds
        """
        if status_code and response_time:
            logger.info(f"🌐 API {method} {url} -> {status_code} ({response_time:.3f}s)")
        else:
            logger.info(f"🌐 API {method} {url}")
    
    @staticmethod
    def log_business_logic(logger: logging.Logger, operation: str, details: str = None):
        """Log business logic operations
        
        Args:
            logger: Logger instance
            operation: Business operation description
            details: Additional details
        """
        detail_str = f" - {details}" if details else ""
        logger.info(f"🔄 Business Logic: {operation}{detail_str}")


# Convenience function to get a configured logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with automatic configuration"""
    if not LoggerConfig._configured:
        LoggerConfig.setup()
    return LoggerConfig.get_logger(name) 