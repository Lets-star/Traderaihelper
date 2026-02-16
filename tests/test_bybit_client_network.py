"""
Network error tests for ByBit client.

Tests timeout errors, connection errors, rate limiting, invalid JSON responses,
server errors (500, 502, 503, 504), and retry exhaustion.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from bybit_client import ByBitClient


class TestByBitClientNetworkErrors:
    """Test ByBit client network error handling."""
    
    @pytest.fixture
    def client(self):
        """Create a ByBit client for testing."""
        return ByBitClient(
            api_key="test_key_123456789",
            api_secret="test_secret_123456789",
            testnet=True,
            log_trades=False
        )
    
    def test_timeout_error_with_retry(self, client):
        """Test timeout error triggers retry logic."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            # First call raises timeout, second succeeds
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.text = '{"retCode": 0, "retMsg": "OK"}'
            success_response.json.return_value = {"retCode": 0, "retMsg": "OK"}
            
            mock_session.request.side_effect = [
                requests.exceptions.Timeout("Connection timed out"),
                success_response
            ]
            
            # Patch time.sleep to speed up test
            with patch('time.sleep'):
                result = client._make_request("GET", "/v5/account/wallet-balance")
            
            # Should have been called twice (initial + retry)
            assert mock_session.request.call_count == 2
            assert result["retCode"] == 0
    
    def test_timeout_error_exhausted(self, client):
        """Test timeout error after max retries exhausted."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            # All calls raise timeout
            mock_session.request.side_effect = requests.exceptions.Timeout("Connection timed out")
            
            with patch('time.sleep'):
                result = client._make_request("GET", "/v5/account/wallet-balance")
            
            # Should have been called MAX_RETRIES + 1 times
            assert mock_session.request.call_count == client.MAX_RETRIES + 1
            assert result["retCode"] == -1
            assert "timeout" in result["retMsg"].lower()
    
    def test_connection_error_with_retry(self, client):
        """Test connection error triggers retry logic."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            # First call raises connection error, second succeeds
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.text = '{"retCode": 0, "retMsg": "OK"}'
            success_response.json.return_value = {"retCode": 0, "retMsg": "OK"}
            
            mock_session.request.side_effect = [
                requests.exceptions.ConnectionError("Connection refused"),
                success_response
            ]
            
            with patch('time.sleep'):
                result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert mock_session.request.call_count == 2
            assert result["retCode"] == 0
    
    def test_rate_limit_429_with_retry(self, client):
        """Test 429 rate limit triggers retry with backoff."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            # First call returns 429, second succeeds
            rate_limit_response = MagicMock()
            rate_limit_response.status_code = 429
            rate_limit_response.text = '{"retCode": 10006, "retMsg": "Rate limit exceeded"}'
            rate_limit_response.json.return_value = {"retCode": 10006, "retMsg": "Rate limit exceeded"}
            
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.text = '{"retCode": 0, "retMsg": "OK"}'
            success_response.json.return_value = {"retCode": 0, "retMsg": "OK"}
            
            mock_session.request.side_effect = [
                rate_limit_response,
                success_response
            ]
            
            with patch('time.sleep') as mock_sleep:
                result = client._make_request("GET", "/v5/account/wallet-balance")
                
                # Should have slept
                assert mock_sleep.called
            
            assert mock_session.request.call_count == 2
            assert result["retCode"] == 0
    
    def test_rate_limit_429_exhausted(self, client):
        """Test 429 rate limit after max retries."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            # All calls return 429
            rate_limit_response = MagicMock()
            rate_limit_response.status_code = 429
            rate_limit_response.text = '{"retCode": 10006, "retMsg": "Rate limit exceeded"}'
            rate_limit_response.json.return_value = {"retCode": 10006, "retMsg": "Rate limit exceeded"}
            mock_session.request.return_value = rate_limit_response
            
            with patch('time.sleep'):
                result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
            assert "rate limit exceeded" in result["retMsg"].lower()
    
    def test_invalid_json_response(self, client):
        """Test handling of invalid JSON response."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            invalid_response = MagicMock()
            invalid_response.status_code = 200
            invalid_response.text = 'not valid json'
            invalid_response.json.side_effect = ValueError("Invalid JSON")
            mock_session.request.return_value = invalid_response
            
            result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
            assert "invalid json" in result["retMsg"].lower()
    
    def test_server_error_500(self, client):
        """Test handling of 500 server error."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            error_response = MagicMock()
            error_response.status_code = 500
            error_response.text = 'Internal Server Error'
            error_response.json.return_value = {"retCode": -1, "retMsg": "Internal Server Error"}
            mock_session.request.return_value = error_response
            
            result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
    
    def test_server_error_502(self, client):
        """Test handling of 502 Bad Gateway."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            error_response = MagicMock()
            error_response.status_code = 502
            error_response.text = 'Bad Gateway'
            error_response.json.return_value = {"retCode": -1, "retMsg": "Bad Gateway"}
            mock_session.request.return_value = error_response
            
            result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
    
    def test_server_error_503(self, client):
        """Test handling of 503 Service Unavailable."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            error_response = MagicMock()
            error_response.status_code = 503
            error_response.text = 'Service Unavailable'
            error_response.json.return_value = {"retCode": -1, "retMsg": "Service Unavailable"}
            mock_session.request.return_value = error_response
            
            result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
    
    def test_server_error_504(self, client):
        """Test handling of 504 Gateway Timeout."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            error_response = MagicMock()
            error_response.status_code = 504
            error_response.text = 'Gateway Timeout'
            error_response.json.return_value = {"retCode": -1, "retMsg": "Gateway Timeout"}
            mock_session.request.return_value = error_response
            
            result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
    
    def test_request_exception_generic(self, client):
        """Test handling of generic RequestException."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            mock_session.request.side_effect = requests.exceptions.RequestException(
                "Something went wrong"
            )
            
            with patch('time.sleep'):
                result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
            assert "request failed" in result["retMsg"].lower()
    
    def test_unexpected_exception(self, client):
        """Test handling of unexpected exception."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            mock_session.request.side_effect = ValueError("Unexpected error")
            
            result = client._make_request("GET", "/v5/account/wallet-balance")
            
            assert result["retCode"] == -1
            assert "unexpected error" in result["retMsg"].lower()
    
    def test_retry_backoff_increases(self, client):
        """Test that retry backoff increases with each attempt."""
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.request.side_effect = requests.exceptions.Timeout()
            
            sleep_times = []
            
            def capture_sleep(seconds):
                sleep_times.append(seconds)
            
            with patch('time.sleep', side_effect=capture_sleep):
                client._make_request("GET", "/v5/account/wallet-balance")
            
            # Should have increasing backoff
            assert len(sleep_times) == client.MAX_RETRIES
            assert sleep_times[0] < sleep_times[-1]


class TestByBitClientRetryStrategies:
    """Test retry strategies and configurations."""
    
    def test_custom_max_retries(self):
        """Test client with custom max retries."""
        client = ByBitClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            log_trades=False
        )
        
        # Default should be 3
        assert client.MAX_RETRIES == 3
    
    def test_rate_limit_backoff_values(self):
        """Test rate limit backoff configuration."""
        client = ByBitClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            log_trades=False
        )
        
        # Should have backoff values
        assert len(client.RATE_LIMIT_BACKOFF) > 0
        assert all(isinstance(x, int) for x in client.RATE_LIMIT_BACKOFF)


class TestByBitClientSessionRecovery:
    """Test session recovery after errors."""
    
    def test_session_reuse_after_error(self):
        """Test that session is reused after error recovery."""
        client = ByBitClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            log_trades=False
        )
        
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            
            # Error then success
            success_response = MagicMock()
            success_response.status_code = 200
            success_response.text = '{"retCode": 0}'
            success_response.json.return_value = {"retCode": 0}
            
            mock_session.request.side_effect = [
                requests.exceptions.Timeout(),
                success_response
            ]
            
            with patch('time.sleep'):
                result = client._make_request("GET", "/test")
            
            # Session is fetched for each retry attempt + 1 initial
            assert mock_get_session.call_count >= 1


class TestByBitClientLoggingDuringErrors:
    """Test logging behavior during network errors."""
    
    def test_timeout_error_logged(self, caplog):
        """Test that timeout errors are logged."""
        client = ByBitClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            log_trades=False
        )
        
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.request.side_effect = requests.exceptions.Timeout()
            
            with patch('time.sleep'):
                client._make_request("GET", "/test")
        
        # Check logs contain timeout message
        assert any("timeout" in record.message.lower() for record in caplog.records)
    
    def test_rate_limit_logged(self, caplog):
        """Test that rate limit hits are logged."""
        client = ByBitClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            log_trades=False
        )
        
        with patch.object(client, '_get_session') as mock_get_session:
            mock_session = MagicMock()
            mock_get_session.return_value = mock_session
            mock_session.request.return_value = MagicMock(
                status_code=429,
                text='Rate limit'
            )
            
            with patch('time.sleep'):
                client._make_request("GET", "/test")
        
        # Check logs contain rate limit message
        assert any("rate limit" in record.message.lower() for record in caplog.records)
