"""
Tests for the exceptions module.

Tests the exception hierarchy, retryable logic, context preservation,
and error code handling.
"""

from __future__ import annotations

import pytest

from exceptions import (
    TradingError,
    NetworkError,
    ConnectionError,
    TimeoutError,
    RateLimitError,
    APIError,
    AuthenticationError,
    InvalidRequestError,
    ServerError,
    ValidationError,
    ExecutionError,
    WebSocketError,
    WebSocketConnectionError,
    DataError,
    CacheError,
    is_retryable_error,
)


class TestTradingError:
    """Test the base TradingError class."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        err = TradingError("Something went wrong")
        assert err.message == "Something went wrong"
        assert err.code == "TRADING_ERROR"
        assert err.retryable is False
        assert err.context == {}
    
    def test_error_with_code(self):
        """Test error with custom code."""
        err = TradingError("Custom error", code="CUSTOM_001")
        assert err.code == "CUSTOM_001"
    
    def test_retryable_error(self):
        """Test retryable error flag."""
        err = TradingError("Retryable error", retryable=True)
        assert err.retryable is True
    
    def test_error_with_context(self):
        """Test error with context."""
        ctx = {"symbol": "BTCUSDT", "price": 50000}
        err = TradingError("Context error", context=ctx)
        assert err.context == ctx
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        err = TradingError(
            "Test error",
            code="TEST_001",
            retryable=True,
            context={"key": "value"}
        )
        
        result = err.to_dict()
        assert result["retCode"] == -1
        assert result["retMsg"] == "Test error"
        assert result["error_code"] == "TEST_001"
        assert result["retryable"] is True
        assert result["context"] == {"key": "value"}
    
    def test_str_representation(self):
        """Test string representation."""
        err = TradingError("Test message", code="TEST")
        str_repr = str(err)
        assert "[TEST]" in str_repr
        assert "Test message" in str_repr
        assert "retryable=False" in str_repr
    
    def test_str_with_context(self):
        """Test string representation with context."""
        err = TradingError("Test", context={"key": "value"})
        str_repr = str(err)
        assert "context={" in str_repr


class TestNetworkErrors:
    """Test network-related errors."""
    
    def test_connection_error(self):
        """Test ConnectionError."""
        err = ConnectionError()
        assert err.code == "CONNECTION_ERROR"
        assert err.retryable is True
        assert "Failed to connect" in err.message
    
    def test_connection_error_with_context(self):
        """Test ConnectionError with context."""
        ctx = {"endpoint": "https://api.example.com"}
        err = ConnectionError("Custom message", context=ctx)
        assert err.context["endpoint"] == "https://api.example.com"
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        err = TimeoutError(timeout_seconds=30.0)
        assert err.code == "TIMEOUT_ERROR"
        assert err.retryable is True
        assert err.context["timeout_seconds"] == 30.0
    
    def test_timeout_error_without_seconds(self):
        """Test TimeoutError without seconds."""
        err = TimeoutError()
        assert "timeout_seconds" not in err.context
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        err = RateLimitError(retry_after_seconds=60)
        assert err.code == "RATE_LIMIT_ERROR"
        assert err.retryable is True
        assert err.context["retry_after_seconds"] == 60
    
    def test_rate_limit_error_default(self):
        """Test RateLimitError with default message."""
        err = RateLimitError()
        assert "Rate limit exceeded" in err.message


class TestAPIErrors:
    """Test API-related errors."""
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        err = AuthenticationError()
        assert err.code == "AUTHENTICATION_ERROR"
        assert err.retryable is False
        assert "Authentication failed" in err.message
    
    def test_invalid_request_error(self):
        """Test InvalidRequestError."""
        err = InvalidRequestError("Invalid parameter")
        assert err.code == "INVALID_REQUEST"
        assert err.retryable is False
        assert err.message == "Invalid parameter"
    
    def test_server_error(self):
        """Test ServerError."""
        err = ServerError(status_code=503)
        assert err.code == "SERVER_ERROR"
        assert err.retryable is True
        assert err.context["status_code"] == 503
    
    def test_server_error_default(self):
        """Test ServerError without status code."""
        err = ServerError()
        assert "status_code" not in err.context


class TestValidationError:
    """Test ValidationError."""
    
    def test_validation_error(self):
        """Test basic validation error."""
        err = ValidationError("Invalid field value")
        assert err.code == "VALIDATION_ERROR"
        assert err.retryable is False
    
    def test_validation_error_with_field(self):
        """Test validation error with field."""
        err = ValidationError("Invalid value", field="symbol")
        assert err.context["field"] == "symbol"


class TestExecutionError:
    """Test ExecutionError."""
    
    def test_execution_error(self):
        """Test basic execution error."""
        err = ExecutionError("Execution failed")
        assert err.code == "EXECUTION_ERROR"
        assert err.retryable is False
    
    def test_execution_error_with_signal_id(self):
        """Test execution error with signal ID."""
        err = ExecutionError("Failed", signal_id="sig_123")
        assert err.context["signal_id"] == "sig_123"


class TestWebSocketErrors:
    """Test WebSocket errors."""
    
    def test_websocket_error(self):
        """Test base WebSocketError."""
        err = WebSocketError("WebSocket failed")
        assert err.code == "WEBSOCKET_ERROR"
        assert err.retryable is True
    
    def test_websocket_connection_error(self):
        """Test WebSocketConnectionError."""
        err = WebSocketConnectionError()
        assert err.code == "WEBSOCKET_CONNECTION_ERROR"
        assert err.retryable is True


class TestDataAndCacheErrors:
    """Test DataError and CacheError."""
    
    def test_data_error(self):
        """Test DataError."""
        err = DataError("Data missing")
        assert err.code == "DATA_ERROR"
        assert err.retryable is False
    
    def test_cache_error(self):
        """Test CacheError."""
        err = CacheError("Cache unavailable")
        assert err.code == "CACHE_ERROR"
        assert err.retryable is True


class TestIsRetryableError:
    """Test the is_retryable_error function."""
    
    def test_retryable_trading_error(self):
        """Test retryable TradingError."""
        err = TradingError("Retryable", retryable=True)
        assert is_retryable_error(err) is True
    
    def test_non_retryable_trading_error(self):
        """Test non-retryable TradingError."""
        err = TradingError("Not retryable", retryable=False)
        assert is_retryable_error(err) is False
    
    def test_retryable_network_error(self):
        """Test network errors are retryable."""
        err = ConnectionError()
        assert is_retryable_error(err) is True
        
        err = TimeoutError()
        assert is_retryable_error(err) is True
    
    def test_non_retryable_authentication_error(self):
        """Test authentication error is not retryable."""
        err = AuthenticationError()
        assert is_retryable_error(err) is False
    
    def test_standard_connection_error(self):
        """Test standard ConnectionError."""
        err = ConnectionError("Standard error")
        assert is_retryable_error(err) is True
    
    def test_standard_timeout_error(self):
        """Test standard TimeoutError."""
        err = TimeoutError("Standard timeout")
        assert is_retryable_error(err) is True
    
    def test_non_retryable_standard_error(self):
        """Test standard ValueError is not retryable."""
        err = ValueError("Some error")
        assert is_retryable_error(err) is False
    
    def test_non_retryable_type_error(self):
        """Test TypeError is not retryable."""
        err = TypeError("Type error")
        assert is_retryable_error(err) is False
