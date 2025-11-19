"""Tests for Binance robust network handling and circuit breaker."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout, Timeout

from indicator_collector.trading_system.data_sources import BinanceKlinesSource
from indicator_collector.timeframes import Timeframe


class TestBinanceCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_trips_after_max_failures(self):
        """Circuit breaker should trip after max_retries failures."""
        source = BinanceKlinesSource(
            max_retries=2,
            enable_circuit_breaker=True,
            circuit_breaker_cooldown=5.0,
        )

        # Record failures
        for _ in range(2):
            source._record_failure(
                "https://api.binance.com",
                ConnectionError("Connection refused"),
                retryable=True,
            )

        # Check circuit is tripped
        wait_time = source._is_circuit_open("https://api.binance.com")
        assert wait_time > 0

    def test_circuit_breaker_disabled(self):
        """When disabled, circuit breaker should not trip."""
        source = BinanceKlinesSource(
            max_retries=2,
            enable_circuit_breaker=False,
        )

        # Record multiple failures
        for _ in range(10):
            source._record_failure(
                "https://api.binance.com",
                ConnectionError("Connection refused"),
                retryable=True,
            )

        # Check circuit is not tripped
        wait_time = source._is_circuit_open("https://api.binance.com")
        assert wait_time == 0

    def test_circuit_breaker_resets_after_cooldown(self):
        """Circuit breaker should reset after cooldown period."""
        with patch("time.monotonic") as mock_monotonic:
            source = BinanceKlinesSource(
                max_retries=1,
                enable_circuit_breaker=True,
                circuit_breaker_cooldown=1.0,
            )

            # Set initial time
            mock_monotonic.return_value = 100.0

            # Trip circuit
            source._circuit_breaker_tripped_at["https://api.binance.com"] = 100.0

            # Check circuit is tripped
            wait_time = source._is_circuit_open("https://api.binance.com")
            assert wait_time > 0

            # Move time forward past cooldown
            mock_monotonic.return_value = 101.5

            # Check circuit is reset
            wait_time = source._is_circuit_open("https://api.binance.com")
            assert wait_time == 0


class TestBinanceHealthChecks:
    """Test health check functionality."""

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    def test_healthcheck_success(self, mock_get):
        """Test successful health check."""
        source = BinanceKlinesSource()

        # Mock ping response
        ping_response = Mock()
        ping_response.status_code = 200

        # Mock time response
        time_response = Mock()
        time_response.status_code = 200
        time_response.json.return_value = {"serverTime": 1704067200000}

        mock_get.side_effect = [ping_response, time_response]

        server_time = source._run_healthcheck("https://api.binance.com")
        assert server_time == 1704067200000

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    def test_healthcheck_ping_fails(self, mock_get):
        """Test health check when ping fails."""
        source = BinanceKlinesSource()

        # Mock ping response failure
        ping_response = Mock()
        ping_response.status_code = 503

        mock_get.return_value = ping_response

        with pytest.raises(RuntimeError, match="Ping failed"):
            source._run_healthcheck("https://api.binance.com")

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    def test_healthcheck_connection_refused(self, mock_get):
        """Test health check with connection refused error."""
        source = BinanceKlinesSource()

        # Mock connection error
        mock_get.side_effect = ConnectionError("[WinError 10061] Connection refused")

        with pytest.raises(RuntimeError, match="Connection refused"):
            source._run_healthcheck("https://api.binance.com")


class TestBinanceFallbackURLs:
    """Test fallback URL functionality."""

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    def test_fallback_to_secondary_url(self, mock_get):
        """Test falling back to secondary URL after primary fails."""
        source = BinanceKlinesSource(
            base_url="https://api.binance.com",
            fallback_urls=["https://api1.binance.com", "https://api2.binance.com"],
        )

        # Mock ping/time responses - fail primary, succeed on fallback
        def get_side_effect(url, *args, **kwargs):
            if "api.binance.com" in url:
                raise ConnectionError("Connection refused")
            else:
                if "/ping" in url:
                    response = Mock()
                    response.status_code = 200
                    return response
                else:  # /time
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {"serverTime": 1704067200000}
                    return response

        mock_get.side_effect = get_side_effect

        active_url = source._ensure_active_base_url()
        assert "api1.binance.com" in active_url or "api2.binance.com" in active_url


class TestBinanceRequestRetries:
    """Test request retry logic with exponential backoff."""

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    @patch("time.sleep")
    def test_retry_on_429_with_backoff(self, mock_sleep, mock_get):
        """Test retry with exponential backoff on 429."""
        source = BinanceKlinesSource(max_retries=3, backoff_base=0.1, backoff_jitter=0.0)

        # Fail twice with 429, then succeed
        responses = []
        for _ in range(2):
            r = Mock()
            r.status_code = 429
            r.text = "Rate Limited"
            r.ok = False
            responses.append(r)

        success_response = Mock()
        success_response.status_code = 200
        success_response.ok = True
        success_response.json.return_value = [[1704067200000, "50000", "50100", "49900", "50050", "100"]]
        responses.append(success_response)

        mock_get.side_effect = responses

        # Mock health check
        source._active_base_url = "https://api.binance.com"

        result = source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)

        assert len(result) == 1
        # Should have backed off twice
        assert mock_sleep.call_count == 2

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    @patch("time.sleep")
    def test_retry_on_5xx_error(self, mock_sleep, mock_get):
        """Test retry on 5xx server errors."""
        source = BinanceKlinesSource(max_retries=3, backoff_base=0.1, backoff_jitter=0.0)

        # Fail with 503, then succeed
        error_response = Mock()
        error_response.status_code = 503
        error_response.text = "Service Unavailable"
        error_response.ok = False

        success_response = Mock()
        success_response.status_code = 200
        success_response.ok = True
        success_response.json.return_value = [[1704067200000, "50000", "50100", "49900", "50050", "100"]]

        mock_get.side_effect = [error_response, success_response]

        # Mock health check
        source._active_base_url = "https://api.binance.com"

        result = source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)

        assert len(result) == 1
        assert mock_sleep.call_count == 1

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    @patch("time.sleep")
    def test_connection_error_retry(self, mock_sleep, mock_get):
        """Test retry on connection errors."""
        source = BinanceKlinesSource(max_retries=3, backoff_base=0.1, backoff_jitter=0.0)

        # Fail with connection error, then succeed
        success_response = Mock()
        success_response.status_code = 200
        success_response.ok = True
        success_response.json.return_value = [[1704067200000, "50000", "50100", "49900", "50050", "100"]]

        mock_get.side_effect = [
            ConnectionError("[WinError 10061] Connection refused"),
            success_response,
        ]

        # Mock health check
        source._active_base_url = "https://api.binance.com"

        result = source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)

        assert len(result) == 1
        assert mock_sleep.call_count == 1

    @patch("indicator_collector.trading_system.data_sources.binance_source.requests.Session.get")
    def test_max_retries_exhausted(self, mock_get):
        """Test failure when max retries are exhausted."""
        source = BinanceKlinesSource(max_retries=2)

        # Always fail with connection error
        mock_get.side_effect = ConnectionError("[Errno 111] Connection refused")

        # Mock health check
        source._active_base_url = "https://api.binance.com"

        with pytest.raises(RuntimeError, match="Max retries exceeded"):
            source._fetch_klines_batch("BTCUSDT", "1h", 1704067200000)


class TestBinanceErrorMessages:
    """Test error message formatting."""

    def test_format_connection_refused_error(self):
        """Test formatting of connection refused errors."""
        source = BinanceKlinesSource()

        error = ConnectionError("[WinError 10061] No connection could be made")
        message = source._format_request_error(error)

        assert "Connection refused" in message
        assert "BINANCE_API_BASE_URL" in message
        assert "proxy" in message.lower()

    def test_format_errno_111_error(self):
        """Test formatting of errno 111 (Linux connection refused)."""
        source = BinanceKlinesSource()

        error = ConnectionError("[Errno 111] Connection refused")
        message = source._format_request_error(error)

        assert "Connection refused" in message

    def test_format_generic_error(self):
        """Test formatting of generic errors."""
        source = BinanceKlinesSource()

        error = RuntimeError("Something went wrong")
        message = source._format_request_error(error)

        assert message == "Something went wrong"


class TestBinanceCaching:
    """Test data caching for graceful degradation."""

    @patch("indicator_collector.trading_system.data_sources.binance_source.BinanceKlinesSource._fetch_candles_paginated")
    def test_cache_stored_on_success(self, mock_fetch):
        """Test that successful fetches are cached."""
        source = BinanceKlinesSource()

        base_time = 1704067200000
        candles_data = [
            [base_time, "50000", "50100", "49900", "50050", "100", base_time + 3600000, "10000"],
        ]
        mock_fetch.return_value = candles_data

        # Mock health check
        with patch.object(source, "_ensure_active_base_url", return_value="https://api.binance.com"):
            start = datetime(2024, 1, 1, 0, 0, 0)
            end = datetime(2024, 1, 1, 1, 0, 0)

            df = source.load_candles("BTCUSDT", "1h", start, end)

            # Check cache
            cached = source._load_cache("BTCUSDT", "1h")
            assert cached is not None
            assert len(cached) == len(df)

    @patch("indicator_collector.trading_system.data_sources.binance_source.BinanceKlinesSource._fetch_candles_paginated")
    def test_cache_used_on_failure(self, mock_fetch):
        """Test that cache is used when fetch fails."""
        source = BinanceKlinesSource()

        base_time = 1704067200000
        candles_data = [
            [base_time, "50000", "50100", "49900", "50050", "100", base_time + 3600000, "10000"],
        ]

        # First call succeeds
        mock_fetch.return_value = candles_data

        with patch.object(source, "_ensure_active_base_url", return_value="https://api.binance.com"):
            start = datetime(2024, 1, 1, 0, 0, 0)
            end = datetime(2024, 1, 1, 1, 0, 0)

            df1 = source.load_candles("BTCUSDT", "1h", start, end)

        # Second call fails but should return cached data
        mock_fetch.side_effect = RuntimeError("Network failure")

        with patch.object(source, "_ensure_active_base_url", return_value="https://api.binance.com"):
            df2 = source.load_candles("BTCUSDT", "1h", start, end)

            # Should get cached data
            assert len(df2) == len(df1)
            assert df2.attrs.get("binance_status", {}).get("used_cache") is True
