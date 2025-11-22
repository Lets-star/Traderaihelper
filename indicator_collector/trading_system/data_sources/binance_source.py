"""Binance API integration for real historical OHLCV data."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ConnectionError,
    ConnectTimeout,
    ReadTimeout,
    RequestException,
    Timeout,
)

from ...timeframes import Timeframe
from .interfaces import HistoricalDataSource
from .timestamp_utils import (
    normalize_timestamp,
    validate_no_future_timestamps,
    validate_timestamps_monotonic,
    ensure_utc_datetime,
    datetime_to_milliseconds,
    floor_to_interval,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS: List[str] = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
]
PING_ENDPOINT = "/api/v3/ping"
SERVER_TIME_ENDPOINT = "/api/v3/time"
KLINES_ENDPOINT = "/api/v3/klines"

BINANCE_RATE_LIMIT_DELAY = 0.1  # 100ms between requests to respect rate limits
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # Exponential backoff multiplier
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 20.0
DEFAULT_BACKOFF_JITTER = 0.75
DEFAULT_CIRCUIT_BREAKER_COOLDOWN = 30.0
DEFAULT_HEALTHCHECK_TTL = 45.0
DEFAULT_USER_AGENT = "indicator-collector/1.0 (+https://indicator-collector)"
BINANCE_INTERVAL_TO_MILLISECONDS = {
    "1m": 60 * 1000,
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}
DEFAULT_FUTURE_TOLERANCE_MS = 60 * 1000


class BinanceKlinesSource(HistoricalDataSource):
    """Load historical OHLCV candles from Binance API."""

    # Mapping of internal timeframes to Binance intervals
    TIMEFRAME_TO_BINANCE_INTERVAL: Dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "3h": "1h",  # 3h will be aggregated from 1h
    }

    # Max candles per Binance API request
    MAX_CANDLES_PER_REQUEST = 1000

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit_delay: float = BINANCE_RATE_LIMIT_DELAY,
        max_retries: int = MAX_RETRIES,
        backoff_base: float = 0.5,
        backoff_jitter: float = DEFAULT_BACKOFF_JITTER,
        sleep_func: Optional[Callable[[float], None]] = None,
        base_url: Optional[str] = None,
        fallback_urls: Optional[List[str]] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        enable_circuit_breaker: bool = True,
        circuit_breaker_cooldown: float = DEFAULT_CIRCUIT_BREAKER_COOLDOWN,
        healthcheck_ttl: float = DEFAULT_HEALTHCHECK_TTL,
    ):
        """
        Initialize Binance data source.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            rate_limit_delay: Delay between requests in seconds
            max_retries: Maximum retry attempts for failed requests
            backoff_base: Base delay (seconds) used for exponential backoff between retries
            sleep_func: Optional sleep function override (useful for testing)
            base_url: Base URL for Binance API (overrides default and env var)
            fallback_urls: List of fallback URLs (used if base URL fails)
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            user_agent: User-Agent header value
            enable_circuit_breaker: Enable circuit breaker for failed endpoints
            circuit_breaker_cooldown: Cooldown period after circuit breaker trips (seconds)
            healthcheck_ttl: Time-to-live for health check cache (seconds)
        """
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if backoff_base < 0:
            raise ValueError("backoff_base must be non-negative")
        if backoff_jitter < 0:
            raise ValueError("backoff_jitter must be non-negative")

        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_jitter = backoff_jitter
        self._sleep = sleep_func or time.sleep
        
        # URL configuration
        self.base_url = base_url or os.environ.get("BINANCE_BASE_URL") or DEFAULT_BASE_URLS[0]
        self.fallback_urls = fallback_urls or DEFAULT_BASE_URLS[1:]
        
        # Timeout configuration
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        
        # Circuit breaker state
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_cooldown = circuit_breaker_cooldown
        self._circuit_breaker_tripped_at: Dict[str, float] = {}
        self._failed_url_count: Dict[str, int] = {}
        
        # Health check cache
        self.healthcheck_ttl = healthcheck_ttl
        self._last_healthcheck: Dict[str, float] = {}
        self._healthcheck_status: Dict[str, bool] = {}
        
        # Cached data for graceful degradation
        self._last_successful_data: Dict[str, pd.DataFrame] = {}
        
        # HTTP session with custom configuration
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        
        # Configure proxy from environment
        proxies = {}
        if os.environ.get("HTTP_PROXY"):
            proxies["http"] = os.environ["HTTP_PROXY"]
        if os.environ.get("HTTPS_PROXY"):
            proxies["https"] = os.environ["HTTPS_PROXY"]
        if proxies:
            self.session.proxies.update(proxies)
            logger.info(f"Using proxy configuration: {proxies}")
        
        # Retry adapter
        adapter = HTTPAdapter(max_retries=0)  # We'll handle retries manually
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # State tracking
        self._active_base_url: Optional[str] = None
        self._last_server_time_ms: Optional[int] = None
        self._server_time_checked_at: Optional[datetime] = None
        self._server_time_checked_monotonic: Optional[float] = None
        self._last_status: Dict[str, Union[str, int, float, bool, None]] = {}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _cache_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"

    def _candidate_base_urls(self) -> List[str]:
        urls: List[str] = []
        env_primary = os.environ.get("BINANCE_API_BASE_URL")
        env_fallbacks = os.environ.get("BINANCE_API_FALLBACK_URLS")

        candidates = [self.base_url]
        if env_primary:
            candidates.insert(0, env_primary)

        if self.fallback_urls:
            candidates.extend(self.fallback_urls)

        if env_fallbacks:
            candidates.extend([item.strip() for item in env_fallbacks.split(",") if item.strip()])

        candidates.extend(DEFAULT_BASE_URLS)

        for url in candidates:
            if not url:
                continue
            normalized = url.rstrip("/")
            if normalized not in urls:
                urls.append(normalized)
        return urls

    def _is_circuit_open(self, base_url: str) -> float:
        if not self.enable_circuit_breaker:
            return 0.0
        tripped_at = self._circuit_breaker_tripped_at.get(base_url)
        if tripped_at is None:
            return 0.0
        elapsed = time.monotonic() - tripped_at
        if elapsed >= self.circuit_breaker_cooldown:
            self._circuit_breaker_tripped_at.pop(base_url, None)
            self._failed_url_count[base_url] = 0
            return 0.0
        return max(self.circuit_breaker_cooldown - elapsed, 0.0)

    def _record_failure(self, base_url: str, error: Exception, *, retryable: bool) -> None:
        count = self._failed_url_count.get(base_url, 0) + 1
        self._failed_url_count[base_url] = count
        self._healthcheck_status[base_url] = False
        if retryable and self.enable_circuit_breaker and count >= self.max_retries:
            self._circuit_breaker_tripped_at[base_url] = time.monotonic()
        self._last_status = {
            "status": "failure",
            "base_url": base_url,
            "retryable": retryable,
            "consecutive_failures": count,
            "error": self._format_request_error(error),
        }

    def _record_success(self, base_url: str) -> None:
        self._failed_url_count[base_url] = 0
        self._healthcheck_status[base_url] = True
        self._circuit_breaker_tripped_at.pop(base_url, None)
        self._last_status = {
            "status": "success",
            "base_url": base_url,
        }

    def _format_request_error(self, error: Exception) -> str:
        message = str(error)
        lower = message.lower()
        if "winerror 10061" in lower or "errno 111" in lower or "connection refused" in lower:
            return (
                f"Connection refused ({message}). "
                "Binance API may be blocking direct access. Try setting BINANCE_API_BASE_URL "
                "or configuring an HTTP(S) proxy."
            )
        return message

    def _run_healthcheck(self, base_url: str) -> Optional[int]:
        now = time.monotonic()
        last_check = self._last_healthcheck.get(base_url)
        if (
            last_check is not None
            and (now - last_check) < self.healthcheck_ttl
            and self._healthcheck_status.get(base_url)
        ):
            return self._last_server_time_ms

        ping_url = f"{base_url}{PING_ENDPOINT}"
        time_url = f"{base_url}{SERVER_TIME_ENDPOINT}"
        try:
            ping_response = self.session.get(
                ping_url,
                timeout=(self.connect_timeout, min(self.read_timeout, 5.0)),
            )
            if ping_response.status_code != 200:
                self._last_healthcheck[base_url] = now
                self._healthcheck_status[base_url] = False
                raise RuntimeError(f"Ping failed with status {ping_response.status_code}")
        except RequestException as exc:
            self._last_healthcheck[base_url] = now
            self._healthcheck_status[base_url] = False
            raise RuntimeError(self._format_request_error(exc)) from exc

        try:
            time_response = self.session.get(
                time_url,
                timeout=(self.connect_timeout, min(self.read_timeout, 5.0)),
            )
            if time_response.status_code != 200:
                raise RuntimeError(f"Time endpoint failed with status {time_response.status_code}")
            payload = time_response.json()
            server_time_ms = int(payload.get("serverTime"))
        except (ValueError, TypeError, KeyError) as exc:
            self._last_healthcheck[base_url] = now
            self._healthcheck_status[base_url] = False
            raise RuntimeError(f"Invalid server time response: {exc}") from exc
        except RequestException as exc:
            self._last_healthcheck[base_url] = now
            self._healthcheck_status[base_url] = False
            raise RuntimeError(self._format_request_error(exc)) from exc

        self._last_healthcheck[base_url] = now
        self._healthcheck_status[base_url] = True
        self._last_server_time_ms = server_time_ms
        self._server_time_checked_at = datetime.now(timezone.utc)
        return server_time_ms

    def _ensure_active_base_url(self) -> str:
        candidates = self._candidate_base_urls()
        errors: List[str] = []

        # Prefer current active URL if healthy
        if self._active_base_url:
            wait_time = self._is_circuit_open(self._active_base_url)
            if wait_time == 0.0:
                try:
                    self._run_healthcheck(self._active_base_url)
                    return self._active_base_url
                except Exception as exc:  # pragma: no cover - defensive logging
                    errors.append(f"{self._active_base_url}: {exc}")

        for base_url in candidates:
            wait_time = self._is_circuit_open(base_url)
            if wait_time > 0:
                errors.append(f"{base_url}: circuit breaker active ({wait_time:.1f}s remaining)")
                continue
            try:
                self._run_healthcheck(base_url)
                self._active_base_url = base_url
                return base_url
            except Exception as exc:
                errors.append(f"{base_url}: {exc}")

        raise RuntimeError(
            "Unable to reach Binance API base URLs. "
            + " | ".join(errors)
            + ". Consider setting BINANCE_API_BASE_URL or using a proxy."
        )

    def _determine_effective_end_ms(self, user_end_ms: int, tolerance_ms: int) -> int:
        server_time_ms = self._last_server_time_ms
        if server_time_ms is None:
            server_time_ms = datetime_to_milliseconds(datetime.now(timezone.utc))
        allowed = max(server_time_ms - tolerance_ms, 0)
        return min(user_end_ms, allowed)

    def _store_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        self._last_successful_data[self._cache_key(symbol, timeframe)] = df.copy(deep=True)

    def _load_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        cached = self._last_successful_data.get(self._cache_key(symbol, timeframe))
        if cached is None:
            return None
        return cached.copy(deep=True)

    def get_server_time(self) -> Optional[int]:
        """
        Get Binance server time in milliseconds using /api/v3/time endpoint.
        Caches result for 1 second to avoid excessive API calls.
        
        Returns:
            Server time in milliseconds, or None if the request fails
        """
        now_mono = time.monotonic()
        
        # Check if we have a recent cached value (within 1 second)
        if (
            self._last_server_time_ms is not None
            and self._server_time_checked_monotonic is not None
            and (now_mono - self._server_time_checked_monotonic) < 1.0
        ):
            # Extrapolate from cached value
            elapsed_ms = int((now_mono - self._server_time_checked_monotonic) * 1000)
            return self._last_server_time_ms + elapsed_ms
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                base_url = self._active_base_url or self._ensure_active_base_url()
                self._active_base_url = base_url
                url = f"{base_url}{SERVER_TIME_ENDPOINT}"
                
                response = self.session.get(
                    url,
                    timeout=(self.connect_timeout, min(self.read_timeout, 5.0)),
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Server time endpoint returned status {response.status_code}")
                
                payload = response.json()
                server_time_ms = int(payload.get("serverTime"))
                
                # Cache the server time
                self._last_server_time_ms = server_time_ms
                self._server_time_checked_at = datetime.now(timezone.utc)
                self._server_time_checked_monotonic = time.monotonic()
                self._record_success(base_url)
                
                return server_time_ms
                
            except (ValueError, TypeError, KeyError) as exc:
                last_error = exc
                logger.warning(
                    "Invalid server time response (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if base_url:
                    self._record_failure(base_url, exc, retryable=False)
                self._active_base_url = None
                if attempt < self.max_retries:
                    backoff = self.backoff_base * (RETRY_BACKOFF ** (attempt - 1))
                    jitter = random.uniform(0, self.backoff_jitter * backoff) if backoff > 0 else 0.0
                    self._sleep(max(backoff + jitter, 0.0))
            except (ConnectionError, ConnectTimeout, ReadTimeout, Timeout) as exc:
                last_error = exc
                logger.warning(
                    "Network error fetching server time (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    self._format_request_error(exc),
                )
                if base_url:
                    self._record_failure(base_url, exc, retryable=True)
                self._active_base_url = None
                if attempt < self.max_retries:
                    backoff = self.backoff_base * (RETRY_BACKOFF ** (attempt - 1))
                    jitter = random.uniform(0, self.backoff_jitter * backoff) if backoff > 0 else 0.0
                    self._sleep(max(backoff + jitter, 0.0))
            except RequestException as exc:
                last_error = exc
                logger.warning(
                    "Request error fetching server time (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    self._format_request_error(exc),
                )
                if base_url:
                    self._record_failure(base_url, exc, retryable=True)
                self._active_base_url = None
                if attempt < self.max_retries:
                    backoff = self.backoff_base * (RETRY_BACKOFF ** (attempt - 1))
                    jitter = random.uniform(0, self.backoff_jitter * backoff) if backoff > 0 else 0.0
                    self._sleep(max(backoff + jitter, 0.0))
            except Exception as exc:
                last_error = exc
                logger.error(
                    "Unexpected error fetching server time (attempt %s/%s): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if base_url:
                    self._record_failure(base_url, exc, retryable=False)
                self._active_base_url = None
                if attempt < self.max_retries:
                    backoff = self.backoff_base * (RETRY_BACKOFF ** (attempt - 1))
                    jitter = random.uniform(0, self.backoff_jitter * backoff) if backoff > 0 else 0.0
                    self._sleep(max(backoff + jitter, 0.0))
        
        # All retries exhausted
        if last_error:
            logger.error(f"Failed to get server time after {self.max_retries} attempts: {last_error}")
        
        return None

    def load_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV candles from Binance.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe for candles
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
            All timestamps are in UTC milliseconds.

        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        # Normalize inputs
        tf = Timeframe.from_value(timeframe)
        symbol = symbol.upper().strip()

        start_utc = ensure_utc_datetime(start)
        end_utc = ensure_utc_datetime(end)
        if start_utc >= end_utc:
            raise ValueError(
                f"Start time {start} must be before end time {end} for Binance candles"
            )

        user_start_ms = datetime_to_milliseconds(start_utc)
        user_end_ms = datetime_to_milliseconds(end_utc)
        tolerance_ms = DEFAULT_FUTURE_TOLERANCE_MS

        # Determine if 3h aggregation is needed
        is_3h = tf.value == "3h"
        source_timeframe = "1h" if is_3h else tf.value
        binance_interval = self.TIMEFRAME_TO_BINANCE_INTERVAL[source_timeframe]

        # Run health check and get active base URL
        try:
            active_url = self._ensure_active_base_url()
        except Exception as health_exc:
            logger.error(f"Health check failed: {health_exc}")
            cached_df = self._load_cache(symbol, tf.value)
            if cached_df is not None:
                logger.warning(
                    f"Using cached data for {symbol} {tf.value} due to health check failure"
                )
                return cached_df
            raise ValueError(
                f"Cannot reach Binance API and no cached data available for {symbol} {tf.value}: {health_exc}"
            ) from health_exc

        effective_end_ms = self._determine_effective_end_ms(user_end_ms, tolerance_ms)
        self._last_status.update(
            {
                "status": "fetching",
                "symbol": symbol,
                "timeframe": tf.value,
                "active_base_url": active_url,
                "effective_end_ms": effective_end_ms,
            }
        )

        try:
            # Fetch raw candles
            raw_candles = self._fetch_candles_paginated(
                symbol,
                binance_interval,
                start_utc,
                end_utc,
                effective_end_ms=effective_end_ms,
            )

            if not raw_candles:
                raise ValueError(
                    f"No data available for {symbol} {timeframe} from {start} to {end}"
                )

            # Convert to DataFrame
            df = self._candles_to_dataframe(raw_candles)

            # Apply 3h aggregation if needed
            if is_3h:
                df = self._aggregate_to_3h(df)

            # Validate and normalize without future validation (handled after filtering)
            df = self._validate_and_normalize(
                df,
                tf,
                skip_future_validation=True,
            )

            df = self._apply_time_range_filters(
                df,
                tf,
                user_start_ms=user_start_ms,
                user_end_ms=user_end_ms,
                effective_end_ms=effective_end_ms,
                tolerance_ms=tolerance_ms,
            )

            if df.empty:
                raise ValueError(
                    f"No candles within requested range for {symbol} {tf.value} "
                    "after applying boundaries"
                )

            status_payload = {
                "active_base_url": active_url,
                "used_cache": False,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "requested_start_ms": user_start_ms,
                "requested_end_ms": user_end_ms,
                "effective_end_ms": effective_end_ms,
            }
            df.attrs["binance_status"] = status_payload
            self._store_cache(symbol, tf.value, df)
            return df

        except Exception as e:
            logger.error(f"Failed to load candles from Binance: {e}")
            cached_df = self._load_cache(symbol, tf.value)
            if cached_df is not None:
                logger.warning(
                    "Returning cached Binance candles after failure for %s %s: %s",
                    symbol,
                    tf.value,
                    e,
                )
                cached_df.attrs["binance_status"] = {
                    "active_base_url": self._last_status.get("base_url", active_url),
                    "used_cache": True,
                    "error": str(e),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "requested_start_ms": user_start_ms,
                    "requested_end_ms": user_end_ms,
                    "effective_end_ms": effective_end_ms,
                }
                return cached_df
            raise ValueError(f"Failed to load {symbol} {timeframe} data: {e}") from e

    def _fetch_candles_paginated(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        effective_end_ms: Optional[int] = None,
    ) -> list[list]:
        """
        Fetch candles with pagination to handle large date ranges.

        Args:
            symbol: Trading symbol
            interval: Binance interval (1m, 5m, 15m, 1h, 4h, 1d)
            start: Start datetime
            end: End datetime
            effective_end_ms: Optional effective end time in milliseconds

        Returns:
            List of candle data (each is a list from Binance API)

        Raises:
            RuntimeError: If API requests fail after retries
        """
        all_candles = []
        start_ms = datetime_to_milliseconds(ensure_utc_datetime(start))
        end_ms = datetime_to_milliseconds(ensure_utc_datetime(end))
        if effective_end_ms is not None:
            end_ms = min(end_ms, effective_end_ms)

        current_start_ms = start_ms

        while current_start_ms < end_ms:
            try:
                # Fetch batch of candles
                candles = self._fetch_klines_batch(
                    symbol,
                    interval,
                    current_start_ms,
                    end_ms=end_ms,
                )

                if not candles:
                    break  # No more data available

                all_candles.extend(candles)

                # Update start for next batch
                last_candle_time = candles[-1][0]
                current_start_ms = last_candle_time + 1  # Start after last candle

                # Respect rate limits
                if self.rate_limit_delay > 0:
                    self._sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching batch starting at {current_start_ms}: {e}")
                raise

        return all_candles

    def _fetch_klines_batch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: Optional[int] = None,
    ) -> list[list]:
        """
        Fetch a single batch of klines with retry logic.

        Args:
            symbol: Trading symbol
            interval: Binance interval
            start_ms: Start time in milliseconds

        Returns:
            List of candle data from Binance API

        Raises:
            RuntimeError: If all retries fail
        """
        last_error: Optional[Exception] = None
        retryable = True

        for attempt in range(1, self.max_retries + 1):
            base_url = self._active_base_url or self._ensure_active_base_url()
            self._active_base_url = base_url
            url = (
                f"{base_url}{KLINES_ENDPOINT}?"
                f"symbol={symbol}&"
                f"interval={interval}&"
                f"startTime={start_ms}&"
                f"limit={self.MAX_CANDLES_PER_REQUEST}"
            )
            try:
                response = self.session.get(
                    url,
                    timeout=(self.connect_timeout, self.read_timeout),
                )

                if response.status_code == 429:
                    last_error = RuntimeError(f"HTTP 429 (Rate Limited): {response.text[:200]}")
                    if attempt < self.max_retries:
                        backoff_time = self._compute_backoff_delay(attempt)
                        logger.warning(
                            "Rate limited while fetching %s %s (attempt %s/%s); backing off %.2fs",
                            symbol,
                            interval,
                            attempt,
                            self.max_retries,
                            backoff_time,
                        )
                        if backoff_time > 0:
                            self._sleep(backoff_time)
                        continue
                    retryable = True
                    break

                if response.status_code >= 500:
                    last_error = RuntimeError(
                        f"HTTP {response.status_code} (Server Error): {response.text[:200]}"
                    )
                    if attempt < self.max_retries:
                        backoff_time = self._compute_backoff_delay(attempt)
                        logger.warning(
                            "Server error (%s) while fetching %s %s (attempt %s/%s); retrying after %.2fs",
                            response.status_code,
                            symbol,
                            interval,
                            attempt,
                            self.max_retries,
                            backoff_time,
                        )
                        if backoff_time > 0:
                            self._sleep(backoff_time)
                        continue
                    retryable = True
                    break

                if not response.ok:
                    message = (
                        f"HTTP {response.status_code} while fetching klines for {symbol} "
                        f"interval={interval} startTime={start_ms}: {response.text[:200]}"
                    )
                    logger.error(message)
                    retryable = False
                    raise RuntimeError(message)

                self._record_success(self._active_base_url)
                return response.json()

            except (ConnectionError, ConnectTimeout) as exc:
                last_error = exc
                retryable = True
                self._record_failure(self._active_base_url, exc, retryable=retryable)

                if attempt < self.max_retries:
                    backoff_time = self._compute_backoff_delay(attempt)
                    logger.warning(
                        "Connection error while fetching %s %s (attempt %s/%s): %s – retrying after %.2fs",
                        symbol,
                        interval,
                        attempt,
                        self.max_retries,
                        self._format_request_error(exc),
                        backoff_time,
                    )
                    if backoff_time > 0:
                        self._sleep(backoff_time)
                    continue
                break

            except (ReadTimeout, Timeout) as exc:
                last_error = exc
                retryable = True
                self._record_failure(self._active_base_url, exc, retryable=retryable)

                if attempt < self.max_retries:
                    backoff_time = self._compute_backoff_delay(attempt)
                    logger.warning(
                        "Timeout while fetching %s %s (attempt %s/%s): %s – retrying after %.2fs",
                        symbol,
                        interval,
                        attempt,
                        self.max_retries,
                        exc,
                        backoff_time,
                    )
                    if backoff_time > 0:
                        self._sleep(backoff_time)
                    continue
                break

            except RequestException as exc:
                last_error = exc
                retryable = True
                self._record_failure(self._active_base_url, exc, retryable=retryable)

                if attempt < self.max_retries:
                    backoff_time = self._compute_backoff_delay(attempt)
                    logger.warning(
                        "Request error while fetching %s %s (attempt %s/%s): %s – retrying after %.2fs",
                        symbol,
                        interval,
                        attempt,
                        self.max_retries,
                        self._format_request_error(exc),
                        backoff_time,
                    )
                    if backoff_time > 0:
                        self._sleep(backoff_time)
                    continue
                break

            except (ValueError, json.JSONDecodeError) as exc:
                message = (
                    f"Failed to decode Binance response for {symbol} interval={interval} "
                    f"startTime={start_ms}: {exc}"
                )
                logger.error(message)
                retryable = False
                self._record_failure(self._active_base_url, exc, retryable=retryable)
                raise RuntimeError(message) from exc

        if last_error is not None:
            self._record_failure(self._active_base_url, last_error, retryable=retryable)

        max_retry_message = self._format_max_retry_message(symbol, interval, start_ms, last_error)
        logger.error(max_retry_message)
        raise RuntimeError(max_retry_message) from last_error

    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with jitter for the given attempt."""
        if attempt <= 0 or self.backoff_base == 0:
            return 0.0
        base_delay = self.backoff_base * (RETRY_BACKOFF ** (attempt - 1))
        jitter = base_delay * self.backoff_jitter * random.uniform(0, 1)
        return base_delay + jitter

    def _estimate_end_time(self, start_ms: int, interval: str) -> Optional[int]:
        """Estimate end time for request based on interval and limit."""
        interval_ms = BINANCE_INTERVAL_TO_MILLISECONDS.get(interval)
        if interval_ms is None:
            return None
        return start_ms + interval_ms * self.MAX_CANDLES_PER_REQUEST

    def _format_max_retry_message(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        last_error: Optional[Exception],
    ) -> str:
        """Create a detailed error message when retry budget is exhausted."""
        end_ms = self._estimate_end_time(start_ms, interval)
        message = (
            f"Max retries exceeded while fetching klines for {symbol} "
            f"interval={interval} startTime={start_ms}"
        )
        if end_ms is not None:
            message += f" endTime≈{end_ms}"

        if last_error is not None:
            reason = getattr(last_error, "reason", None) or getattr(last_error, "msg", None) or str(last_error)
            if reason:
                message += f": {reason}"

        return message

    def _candles_to_dataframe(self, candles: list[list]) -> pd.DataFrame:
        """
        Convert Binance candle data to DataFrame.

        Args:
            candles: List of candle data from Binance API
                Each candle is: [openTime, open, high, low, close, volume, ...]

        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
        """
        df_data = {
            "ts": [int(c[0]) for c in candles],
            "open": [float(c[1]) for c in candles],
            "high": [float(c[2]) for c in candles],
            "low": [float(c[3]) for c in candles],
            "close": [float(c[4]) for c in candles],
            "volume": [float(c[5]) for c in candles],  # Volume at index 5
        }

        df = pd.DataFrame(df_data)
        return df

    def _aggregate_to_3h(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1h candles to 3h candles.

        Args:
            df: DataFrame with 1h candles

        Returns:
            DataFrame with 3h candles (aligned to 00:00, 03:00, 06:00, etc. UTC)
        """
        if df.empty:
            return df

        # Convert timestamp to datetime for grouping
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)

        # Calculate 3h bucket start time (aligned to 00:00, 03:00, 06:00, 09:00, ...)
        # Get the hour from UTC datetime
        df["hour"] = df["datetime"].dt.hour
        # Calculate which 3h bucket this hour belongs to (0-2->0, 3-5->3, 6-8->6, ...)
        df["bucket_hour"] = (df["hour"] // 3) * 3
        # Create bucket date (date at start of 3h period)
        df["bucket_date"] = df["datetime"].dt.normalize()
        # Create bucket start time
        df["bucket_start"] = (
            df["bucket_date"] + pd.to_timedelta(df["bucket_hour"], unit="h")
        )
        df["bucket_start_ms"] = (df["bucket_start"].astype(int) // 1e6).astype(int)

        # Group by 3h bucket and aggregate
        aggregated = df.groupby("bucket_start_ms", as_index=False).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Use bucket start time as ts
        aggregated["ts"] = aggregated["bucket_start_ms"]

        # Return only the required columns
        result = aggregated[["ts", "open", "high", "low", "close", "volume"]].copy()

        return result

    def _apply_time_range_filters(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        *,
        user_start_ms: int,
        user_end_ms: int,
        effective_end_ms: int,
        tolerance_ms: int,
    ) -> pd.DataFrame:
        """Apply user range and closed-candle constraints to candle data."""
        if user_end_ms <= user_start_ms:
            raise ValueError("Normalized end time must be after start time")

        interval_ms = timeframe.to_milliseconds()
        start_boundary = floor_to_interval(user_start_ms, interval_ms)
        adjusted_end = max(user_end_ms - 1, user_start_ms)
        end_boundary = floor_to_interval(adjusted_end, interval_ms)

        if end_boundary < start_boundary:
            return df.iloc[0:0]

        filtered = df[(df["ts"] >= start_boundary) & (df["ts"] <= end_boundary)].copy()
        if filtered.empty:
            return filtered

        last_closed_reference_ms = max(effective_end_ms, 0)
        last_closed_close_ms = floor_to_interval(last_closed_reference_ms, interval_ms)

        if last_closed_close_ms < start_boundary + interval_ms:
            # No fully closed candles available within the requested window
            return filtered.iloc[0:0]

        close_times = filtered["ts"] + interval_ms
        filtered = filtered[close_times <= last_closed_close_ms].copy()
        if filtered.empty:
            return filtered

        filtered = (
            filtered.drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True)
        )

        validate_timestamps_monotonic(filtered["ts"].tolist())

        close_time_list = (filtered["ts"] + interval_ms).tolist()
        validate_no_future_timestamps(
            close_time_list,
            tolerance_ms=tolerance_ms,
            reference_ms=effective_end_ms,
        )

        return filtered

    def _validate_and_normalize(
        self,
        df: pd.DataFrame,
        timeframe: Timeframe,
        *,
        skip_future_validation: bool = False,
        tolerance_ms: int = 60 * 1000,
        future_reference_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Validate and normalize the candle DataFrame.

        Args:
            df: DataFrame with candle data
            timeframe: Expected timeframe
            skip_future_validation: When True, skip future timestamp validation.
            tolerance_ms: Future tolerance window in milliseconds.
            future_reference_ms: Optional reference timestamp for future validation.

        Returns:
            Validated and normalized DataFrame

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Empty dataframe")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Validate columns exist
        required_cols = ["ts", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected {required_cols}, got {list(df.columns)}")

        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values
        if df[required_cols].isna().any().any():
            raise ValueError("NaN values found in candle data")

        # Normalize timestamps
        try:
            df["ts"] = df["ts"].apply(normalize_timestamp)
        except Exception as e:
            raise ValueError(f"Failed to normalize timestamps: {e}") from e

        # Validate monotonicity
        try:
            validate_timestamps_monotonic(df["ts"].tolist())
        except Exception as e:
            raise ValueError(f"Timestamps not monotonic: {e}") from e

        # Validate no future timestamps
        if not skip_future_validation:
            try:
                validate_no_future_timestamps(
                    df["ts"].tolist(),
                    tolerance_ms=tolerance_ms,
                    reference_ms=future_reference_ms,
                )
            except Exception as e:
                raise ValueError(f"Future timestamps detected: {e}") from e

        # Validate no zero prices (before OHLC relationships)
        if (df[["open", "high", "low", "close"]] == 0).any().any():
            raise ValueError("Zero prices detected in OHLC data")

        # Validate positive volume
        if (df["volume"] < 0).any():
            raise ValueError("Negative volume detected")

        # Validate OHLC relationships
        if not (df["low"] <= df["open"]).all() or not (df["open"] <= df["high"]).all():
            raise ValueError("OHLC data violates low <= open <= high")

        if not (df["low"] <= df["close"]).all() or not (df["close"] <= df["high"]).all():
            raise ValueError("OHLC data violates low <= close <= high")

        # Reset index and return
        df = df.reset_index(drop=True)
        return df
