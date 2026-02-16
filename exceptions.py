"""
Typed exception hierarchy for trading system.

Provides structured error handling with error codes, retryable flags,
and context preservation for debugging.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class TradingError(Exception):
    """Base exception for all trading-related errors.
    
    Attributes:
        code: Error code for categorization
        message: Human-readable error message
        retryable: Whether the operation can be retried
        context: Additional context for debugging
    """
    
    def __init__(
        self,
        message: str,
        code: str = "TRADING_ERROR",
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "retCode": -1,
            "retMsg": self.message,
            "error_code": self.code,
            "retryable": self.retryable,
            "context": self.context,
        }
    
    def __str__(self) -> str:
        ctx = f" | context={self.context}" if self.context else ""
        return f"[{self.code}] {self.message} (retryable={self.retryable}){ctx}"


class NetworkError(TradingError):
    """Base class for network-related errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "NETWORK_ERROR",
        retryable: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, retryable, context)


class ConnectionError(NetworkError):
    """Failed to establish connection to server."""
    
    def __init__(
        self,
        message: str = "Failed to connect to server",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            code="CONNECTION_ERROR",
            retryable=True,
            context=context
        )


class TimeoutError(NetworkError):
    """Request timed out."""
    
    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        if timeout_seconds is not None:
            ctx["timeout_seconds"] = timeout_seconds
        super().__init__(
            message,
            code="TIMEOUT_ERROR",
            retryable=True,
            context=ctx
        )


class RateLimitError(NetworkError):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after_seconds: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        if retry_after_seconds is not None:
            ctx["retry_after_seconds"] = retry_after_seconds
        super().__init__(
            message,
            code="RATE_LIMIT_ERROR",
            retryable=True,
            context=ctx
        )


class APIError(TradingError):
    """Base class for API-related errors."""
    
    def __init__(
        self,
        message: str,
        code: str = "API_ERROR",
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, retryable, context)


class AuthenticationError(APIError):
    """API authentication failed (invalid credentials)."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            code="AUTHENTICATION_ERROR",
            retryable=False,
            context=context
        )


class InvalidRequestError(APIError):
    """Invalid request parameters."""
    
    def __init__(
        self,
        message: str = "Invalid request",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code="INVALID_REQUEST", retryable=False, context=context)


class ServerError(APIError):
    """Server-side error (5xx responses)."""
    
    def __init__(
        self,
        message: str = "Server error",
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        if status_code is not None:
            ctx["status_code"] = status_code
        super().__init__(
            message,
            code="SERVER_ERROR",
            retryable=True,
            context=ctx
        )


class ValidationError(TradingError):
    """Data validation error."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        if field is not None:
            ctx["field"] = field
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            retryable=False,
            context=ctx
        )


class ExecutionError(TradingError):
    """Signal execution error."""
    
    def __init__(
        self,
        message: str,
        signal_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        ctx = context or {}
        if signal_id is not None:
            ctx["signal_id"] = signal_id
        super().__init__(
            message,
            code="EXECUTION_ERROR",
            retryable=False,
            context=ctx
        )


class WebSocketError(NetworkError):
    """WebSocket-specific error."""
    
    def __init__(
        self,
        message: str,
        code: str = "WEBSOCKET_ERROR",
        retryable: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, retryable, context)


class WebSocketConnectionError(WebSocketError):
    """WebSocket connection establishment failed."""
    
    def __init__(
        self,
        message: str = "WebSocket connection failed",
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            code="WEBSOCKET_CONNECTION_ERROR",
            retryable=True,
            context=context
        )


class DataError(TradingError):
    """Data-related error (missing data, invalid format, etc.)."""
    
    def __init__(
        self,
        message: str,
        code: str = "DATA_ERROR",
        retryable: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, code, retryable, context)


class CacheError(TradingError):
    """Cache-related error."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message,
            code="CACHE_ERROR",
            retryable=True,
            context=context
        )


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if the error is retryable
    """
    if isinstance(error, TradingError):
        return error.retryable
    
    # Check for standard library exceptions
    retryable_exceptions = (
        ConnectionError,
        TimeoutError,
    )
    
    return isinstance(error, retryable_exceptions)
