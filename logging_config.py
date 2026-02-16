"""
Structured logging configuration for trading system.

Provides JSON-formatted logs for better observability and log parsing.
Supports both structured and traditional f-string logging styles during transition.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_exc_info: bool = True
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_exc_info = include_exc_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "message": record.getMessage(),
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat()
        
        if self.include_level:
            log_data["level"] = record.levelname
        
        if self.include_logger:
            log_data["logger"] = record.name
            log_data["module"] = record.module
            log_data["function"] = record.funcName
            log_data["line"] = record.lineno
        
        # Add thread information
        log_data["thread_id"] = record.thread
        log_data["thread_name"] = record.threadName
        
        # Add extra fields if present
        if hasattr(record, "extra_data") and record.extra_data:
            log_data.update(record.extra_data)
        
        # Add exception info
        if self.include_exc_info and record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            log_data["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
                "traceback": traceback.format_exception(*record.exc_info) if exc_tb else None,
            }
        
        return json.dumps(log_data, default=str)


class StructuredLogAdapter:
    """Adapter for adding structured context to log messages.
    
    Usage:
        logger = StructuredLogAdapter(logging.getLogger(__name__), symbol="BTCUSDT")
        logger.info("Order placed", side="Buy", qty=0.001, price=50000)
    
    This produces:
        {"message": "Order placed", "symbol": "BTCUSDT", "side": "Buy", "qty": 0.001, "price": 50000}
    """
    
    def __init__(self, logger: logging.Logger, **default_context):
        self.logger = logger
        self.default_context = default_context
    
    def _log(
        self,
        level: int,
        message: str,
        **kwargs
    ):
        """Internal log method with context merging."""
        extra = kwargs.pop("extra", {})
        context = {**self.default_context, **kwargs}
        
        if "extra_data" not in extra:
            extra["extra_data"] = {}
        extra["extra_data"].update(context)
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        kwargs["exc_info"] = True
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)
    
    def with_context(self, **context) -> "StructuredLogAdapter":
        """Create a new adapter with additional context."""
        merged_context = {**self.default_context, **context}
        return StructuredLogAdapter(self.logger, **merged_context)


def configure_logging(
    level: int = logging.INFO,
    use_json: bool = True,
    include_timestamp: bool = True,
    log_to_file: Optional[str] = None,
    file_level: int = logging.DEBUG
):
    """Configure logging for the trading system.
    
    Args:
        level: Console logging level
        use_json: Use JSON formatter (vs plain text)
        include_timestamp: Include timestamp in logs
        log_to_file: Optional file path for additional logging
        file_level: File logging level
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, file_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_json:
        formatter = JSONFormatter(include_timestamp=include_timestamp)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            if include_timestamp else
            '%(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file)
        file_handler.setLevel(file_level)
        
        if use_json:
            file_formatter = JSONFormatter(include_timestamp=True)
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)


def get_structured_logger(name: str, **context) -> StructuredLogAdapter:
    """Get a structured logger with default context.
    
    Args:
        name: Logger name
        **context: Default context fields
        
    Returns:
        StructuredLogAdapter instance
    """
    logger = logging.getLogger(name)
    return StructuredLogAdapter(logger, **context)


# Convenience function for quick setup
def setup_logging(
    debug: bool = False,
    json_format: bool = True,
    log_file: Optional[str] = None
):
    """Quick setup for logging configuration.
    
    Args:
        debug: Enable debug level logging
        json_format: Use JSON formatting
        log_file: Optional log file path
    """
    configure_logging(
        level=logging.DEBUG if debug else logging.INFO,
        use_json=json_format,
        log_to_file=log_file,
        file_level=logging.DEBUG
    )
