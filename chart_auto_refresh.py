"""
Chart auto-refresh worker for the Charts tab.

TIMESTAMP SEMANTICS:
-------------------
All internal timestamps are in UTC milliseconds.

- DataFrame 'ts' column: open_time (from Binance API's openTime field)
- last_closed_close_ms: close_time of the last closed bar (stored in session state)
- Relationship: close_time = open_time + tf_ms

For example, a 1h candle:
  - open_time:  1700000000000 (2023-11-14 22:00:00 UTC)
  - close_time: 1700003600000 (2023-11-14 23:00:00 UTC)

The next candle's open_time equals the previous candle's close_time.

floor_closed_bar_local(now_ms, tf_ms) returns the close_time of the last closed bar.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from requests.exceptions import RequestException

from indicator_collector.trading_system.auto_analyze_worker import get_binance_server_time_ms
from indicator_collector.trading_system.data_sources.binance_source import (
    BinanceKlinesSource,
    KLINES_ENDPOINT,
)
from update_bus import UpdateBus
from timeframe_utils import TIMEFRAME_TO_MS, map_tf_to_ms, get_boundary

logger = logging.getLogger(__name__)

_CANDLE_CACHE: Dict[tuple[str, str, int, int], pd.DataFrame] = {}
_CACHE_LOCK = threading.Lock()
_CHART_DATA_LOCK = threading.Lock()

STATE_LAST_CLOSED_KEY = "last_closed_ts_per_tf"
DEFAULT_TOLERANCE_MS = 1_500
OVERLAP_BARS = 3
STORE_STATE_KEY = "_chart_data_store"


class ChartDataStore:
    """Thread-safe storage for chart data shared between UI and worker threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._closed_df = pd.DataFrame()
        self._with_forming_df: Optional[pd.DataFrame] = None
        self._forming_raw_df: Optional[pd.DataFrame] = None
        self._closed_indicators: Dict[str, Any] = {}
        self._with_forming_indicators: Optional[Dict[str, Any]] = None
        self._last_closed_close_ms: int = 0
        self._analysis_pending: bool = False
        self._show_forming_bar: bool = False

    def reset(self) -> None:
        """Clear all stored data (used when symbol/timeframe changes)."""
        with self._lock:
            self._closed_df = pd.DataFrame()
            self._with_forming_df = None
            self._forming_raw_df = None
            self._closed_indicators = {}
            self._with_forming_indicators = None
            self._last_closed_close_ms = 0
            self._analysis_pending = True

    @staticmethod
    def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        return (
            df.drop_duplicates(subset="ts", keep="last")
            .sort_values("ts")
            .reset_index(drop=True)
        )

    def _rebuild_with_forming_locked(self) -> None:
        if self._forming_raw_df is None or self._forming_raw_df.empty:
            self._with_forming_df = None
            self._with_forming_indicators = None
            return

        frames = []
        if self._closed_df is not None and not self._closed_df.empty:
            frames.append(self._closed_df.copy(deep=True))
        frames.append(self._forming_raw_df.copy(deep=True))
        combined = pd.concat(frames, ignore_index=True)
        combined = self._dedupe_sort(combined)
        self._with_forming_df = combined
        self._with_forming_indicators = compute_chart_indicators(combined)

    def update_closed(
        self,
        df: Optional[pd.DataFrame],
        last_closed_close_ms: int,
        *,
        append: bool,
    ) -> tuple[int, int, int]:
        """Update closed-bar dataset; returns (appended, deduped, total_rows)."""
        df_copy = pd.DataFrame() if df is None else df.copy(deep=True)
        with self._lock:
            previous_len = 0 if self._closed_df is None else len(self._closed_df)

            if append and previous_len > 0:
                if not df_copy.empty:
                    combined = pd.concat(
                        [self._closed_df.copy(deep=True), df_copy],
                        ignore_index=True,
                    )
                else:
                    combined = self._closed_df.copy(deep=True)
            else:
                combined = df_copy

            if combined.empty:
                self._closed_df = pd.DataFrame()
                self._closed_indicators = {}
                self._last_closed_close_ms = int(last_closed_close_ms)
                self._analysis_pending = True
                self._rebuild_with_forming_locked()
                return 0, 0, 0

            combined = self._dedupe_sort(combined)

            if append and previous_len > 0:
                appended = max(len(combined) - previous_len, 0)
                deduped = max(len(df_copy) - appended, 0)
            else:
                appended = len(combined)
                deduped = 0

            self._closed_df = combined
            self._closed_indicators = compute_chart_indicators(combined)
            self._last_closed_close_ms = int(last_closed_close_ms)
            self._analysis_pending = True
            self._rebuild_with_forming_locked()
            return appended, deduped, len(combined)

    def set_forming_bar(self, forming_df: Optional[pd.DataFrame]) -> None:
        forming_copy = None if forming_df is None else forming_df.copy(deep=True)
        with self._lock:
            previous_exists = self._forming_raw_df is not None and not self._forming_raw_df.empty
            incoming_exists = forming_copy is not None and not forming_copy.empty

            if not incoming_exists:
                if previous_exists:
                    self._forming_raw_df = None
                    self._with_forming_df = None
                    self._with_forming_indicators = None
                    self._analysis_pending = True
                return

            self._forming_raw_df = forming_copy
            self._analysis_pending = True
            self._rebuild_with_forming_locked()

    def clear_forming_bar(self) -> None:
        self.set_forming_bar(None)

    def set_show_forming_bar(self, value: bool) -> bool:
        value = bool(value)
        with self._lock:
            if self._show_forming_bar == value:
                return False
            self._show_forming_bar = value
            self._analysis_pending = True
            return True

    def get_show_forming_bar(self) -> bool:
        with self._lock:
            return bool(self._show_forming_bar)

    def snapshot(
        self,
        *,
        include_forming: bool,
    ) -> tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
        with self._lock:
            if include_forming and self._with_forming_df is not None and not self._with_forming_df.empty:
                df_source = self._with_forming_df
                indicators = self._with_forming_indicators or {}
            else:
                if self._closed_df is None or self._closed_df.empty:
                    return None, {}, self._last_closed_close_ms
                df_source = self._closed_df
                indicators = self._closed_indicators or {}

            df_copy = df_source.copy(deep=True)
            return df_copy, copy.deepcopy(indicators), self._last_closed_close_ms

    def has_closed_data(self) -> bool:
        with self._lock:
            return self._closed_df is not None and not self._closed_df.empty

    def closed_len(self) -> int:
        with self._lock:
            return 0 if self._closed_df is None else len(self._closed_df)

    def get_last_closed_close_ms(self) -> int:
        with self._lock:
            return int(self._last_closed_close_ms)

    def has_pending_update(self) -> bool:
        with self._lock:
            return bool(self._analysis_pending)

    def consume_pending_update(self) -> bool:
        with self._lock:
            flag = bool(self._analysis_pending)
            self._analysis_pending = False
            return flag


def ensure_chart_store(session_state: Any) -> ChartDataStore:
    store = getattr(session_state, STORE_STATE_KEY, None)
    if not isinstance(store, ChartDataStore):
        store = ChartDataStore()
        setattr(session_state, STORE_STATE_KEY, store)
    return store


def _state_map_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"


def _ensure_last_closed_map(session_state: Any) -> Dict[str, int]:
    mapping = getattr(session_state, STATE_LAST_CLOSED_KEY, None)
    if not isinstance(mapping, dict):
        mapping = {}
        setattr(session_state, STATE_LAST_CLOSED_KEY, mapping)
    return mapping


def get_last_closed_from_state(session_state: Any, symbol: str, timeframe: str) -> int:
    mapping = _ensure_last_closed_map(session_state)
    try:
        return int(mapping.get(_state_map_key(symbol, timeframe), 0))
    except (TypeError, ValueError):
        return 0


def set_last_closed_in_state(session_state: Any, symbol: str, timeframe: str, value: int) -> None:
    mapping = _ensure_last_closed_map(session_state)
    mapping[_state_map_key(symbol, timeframe)] = int(value)
    setattr(session_state, "last_closed_ts", int(value))
    setattr(session_state, STATE_LAST_CLOSED_KEY, mapping)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR) indicator."""
    if df.empty or len(df) < period:
        return pd.Series(dtype=float)
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    # True Range calculation
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1, skipna=True)
    
    # RMA (Running Moving Average) for ATR
    atr_values = tr.ewm(alpha=1.0/period, adjust=False).mean()
    return atr_values


def compute_atr_channels(df: pd.DataFrame, atr_period: int = 14) -> Dict[str, pd.Series]:
    """Compute ATR channel overlays with multiple multipliers."""
    if df.empty:
        return {}
    
    atr_values = compute_atr(df, period=atr_period)
    close = df["close"].astype(float)
    
    channels: Dict[str, Dict[str, pd.Series]] = {}
    multipliers = [1, 3, 8, 21]
    
    for mult in multipliers:
        key = f"atr_trend_{mult}x"
        upper = close + (atr_values * mult)
        lower = close - (atr_values * mult)
        channels[key] = {
            "upper": upper,
            "lower": lower,
        }
    
    return channels


def detect_order_blocks(df: pd.DataFrame, lookback: int = 20) -> list[Dict[str, Any]]:
    """Detect bullish and bearish order blocks (simplified version for charts)."""
    if df.empty or len(df) < lookback:
        return []
    
    order_blocks = []
    
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    open_ = df["open"].to_numpy()
    volume = df["volume"].to_numpy()
    
    for i in range(lookback, len(df)):
        # Look for strong momentum candles with high volume
        body_size = abs(close[i] - open_[i])
        avg_body = np.mean([abs(close[j] - open_[j]) for j in range(max(0, i-10), i)])
        
        if body_size > avg_body * 1.5 and volume[i] > np.mean(volume[max(0, i-10):i]) * 1.5:
            # Bullish order block
            if close[i] > open_[i]:
                order_blocks.append({
                    "zone_type": "BullOB",
                    "top": high[i],
                    "bottom": low[i],
                    "created_index": i,
                })
            # Bearish order block
            elif close[i] < open_[i]:
                order_blocks.append({
                    "zone_type": "BearOB",
                    "top": high[i],
                    "bottom": low[i],
                    "created_index": i,
                })
    
    # Keep only the most recent order blocks
    return order_blocks[-10:] if len(order_blocks) > 10 else order_blocks


def compute_chart_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all chart indicators (ATR channels, order blocks) for overlay rendering."""
    if df.empty:
        return {"atr_channels": {}, "order_blocks": []}
    
    return {
        "atr_channels": compute_atr_channels(df),
        "order_blocks": detect_order_blocks(df),
    }


def read_chart_state(
    session_state: Any,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
    """Safely read chart DataFrame and indicators from session state."""
    with _CHART_DATA_LOCK:
        prefer_forming = getattr(session_state, "show_forming_bar", False)
        df_source = None
        if prefer_forming:
            forming_df = getattr(session_state, "chart_df_with_forming", None)
            if isinstance(forming_df, pd.DataFrame) and not forming_df.empty:
                df_source = forming_df
        if df_source is None:
            df_source = getattr(session_state, "chart_df", None)
        if df_source is not None and isinstance(df_source, pd.DataFrame):
            df_copy: Optional[pd.DataFrame] = df_source.copy(deep=True)
        else:
            df_copy = None
        indicators = copy.deepcopy(getattr(session_state, "chart_indicators", {}))
        if symbol and timeframe:
            last_closed_ts = get_last_closed_from_state(session_state, symbol, timeframe)
        else:
            last_closed_ts = getattr(session_state, "last_closed_ts", 0)
    return df_copy, indicators, last_closed_ts


def update_chart_state(
    session_state: Any,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    last_closed_close_ms: int,
    *,
    append: bool = False,
) -> None:
    """
    Safely update chart data and indicators in session state.
    
    Args:
        session_state: Streamlit session state
        symbol: Trading symbol
        timeframe: Timeframe string
        df: DataFrame with candles (ts column = open_time in UTC ms)
        last_closed_close_ms: Close time of the last closed bar in UTC ms
        append: If True, append to existing data; if False, replace
        
    Note:
        - df['ts'] contains open_time (UTC milliseconds)
        - last_closed_close_ms is the close_time of the last closed bar
        - For a bar with open_time T and timeframe tf_ms: close_time = T + tf_ms
    """
    if df is None:
        df = pd.DataFrame()
    with _CHART_DATA_LOCK:
        if append:
            existing_df = getattr(session_state, "chart_df", None)
            if isinstance(existing_df, pd.DataFrame) and not existing_df.empty:
                frames = [existing_df.copy(deep=True)]
                if not df.empty:
                    frames.append(df.copy(deep=True))
                combined = pd.concat(frames, ignore_index=True)
            else:
                combined = df.copy(deep=True)
        else:
            combined = df.copy(deep=True)
        if not combined.empty:
            combined = (
                combined.drop_duplicates(subset="ts", keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
        indicators = compute_chart_indicators(combined)
        session_state.chart_df = combined
        session_state.chart_indicators = indicators
        set_last_closed_in_state(session_state, symbol, timeframe, last_closed_close_ms)
        session_state.analysis_updated = True


def get_poll_interval(timeframe: str) -> float:
    """
    Get poll interval in seconds based on timeframe (TradingView-like).
    
    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")
        
    Returns:
        Poll interval in seconds
    """
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    if tf_ms <= 900_000:
        return 1.0
    else:
        return 5.0



def invalidate_cache(symbol: str, timeframe: str) -> None:
    """Invalidate cache for a specific symbol/timeframe combination."""
    with _CACHE_LOCK:
        keys_to_remove = [k for k in _CANDLE_CACHE.keys() if k[0] == symbol and k[1] == timeframe]
        for key in keys_to_remove:
            del _CANDLE_CACHE[key]
        logger.info(f"Invalidated {len(keys_to_remove)} cache entries for {symbol} {timeframe}")


def fetch_closed_candles(
    symbol: str,
    timeframe: str,
    num_bars: int = 200,
    data_source: Optional[BinanceKlinesSource] = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Fetch only CLOSED bars for the active timeframe.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "1h", "3h")
        num_bars: Number of bars to fetch
        data_source: Optional BinanceKlinesSource instance
        use_cache: Whether to use cache
        
    Returns:
        Tuple of (DataFrame, last_closed_ts)
    """
    if data_source is None:
        data_source = BinanceKlinesSource()
    
    # Get Binance server time
    server_time_ms = get_binance_server_time_ms(data_source)
    
    # Get timeframe in milliseconds
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    # Calculate last closed bar timestamp using tight tolerance
    tol_ms = DEFAULT_TOLERANCE_MS
    last_closed_ts = get_boundary(server_time_ms, tf_ms, tolerance_ms=tol_ms)
    
    # Calculate start time (go back num_bars plus overlap)
    bars_to_fetch = max(num_bars + OVERLAP_BARS, num_bars)
    start_ms = max(0, last_closed_ts - (tf_ms * bars_to_fetch))
    
    # Check cache
    cache_key = (symbol, timeframe, start_ms, last_closed_ts)
    if use_cache:
        with _CACHE_LOCK:
            if cache_key in _CANDLE_CACHE:
                logger.debug(f"Using cached candles for {symbol} {timeframe}")
                return _CANDLE_CACHE[cache_key].copy(), last_closed_ts
    
    # Convert to datetime
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(last_closed_ts / 1000, tz=timezone.utc)
    
    # Fetch candles using BinanceKlinesSource
    try:
        # Strip BINANCE: prefix if present
        clean_symbol = symbol
        if clean_symbol.startswith("BINANCE:"):
            clean_symbol = clean_symbol[8:]
        
        df = data_source.load_candles(
            symbol=clean_symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
        )
        
        # Store in cache
        with _CACHE_LOCK:
            _CANDLE_CACHE[cache_key] = df.copy()
        
        return df, last_closed_ts
    except Exception as e:
        logger.error(f"Failed to fetch candles for {symbol} {timeframe}: {e}")
        raise


class ChartAutoRefreshWorker:
    """WebSocket-based worker that refreshes chart data on new closed bars."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        update_bus: UpdateBus,
    ):
        """
        Initialize the chart auto-refresh worker.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            update_bus: UpdateBus instance for publishing updates
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.update_bus = update_bus
        self.ws_client: Optional[BinanceWebSocketClient] = None
        self.tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)

    def start(self) -> None:
        """Start the worker thread."""
        if self.ws_client is not None:
            return

        from websocket_client import BinanceWebSocketClient
        self.ws_client = BinanceWebSocketClient(
            symbol=self.symbol,
            timeframe=self.timeframe,
            on_closed_kline=self._on_closed_kline,
            on_forming_kline=self._on_forming_kline,
            on_error=self._on_error,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
            backfill_bars=3,
        )
        self.ws_client.start()
        logger.info(f"Chart WebSocket worker started for {self.symbol} {self.timeframe}")

    def stop(self) -> None:
        """Stop the worker thread."""
        if self.ws_client:
            self.ws_client.stop()
            self.ws_client = None
        logger.info(f"Chart WebSocket worker stopped for {self.symbol} {self.timeframe}")

    def _on_closed_kline(self, df: pd.DataFrame) -> None:
        """Callback for closed kline events (called from WebSocket thread)."""
        if df.empty:
            return

        open_time_ms = int(df["ts"].iloc[0])
        close_time_ms = open_time_ms + self.tf_ms

        self.update_bus.publish({
            "type": "chart_closed_kline",
            "df": df,
            "last_closed_close_ms": close_time_ms,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def _on_forming_kline(self, df: pd.DataFrame) -> None:
        """Callback for forming kline updates (called from WebSocket thread)."""
        self.update_bus.publish({
            "type": "chart_forming_kline",
            "df": df,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def _on_error(self, error: Exception) -> None:
        """Callback for WebSocket errors (called from WebSocket thread)."""
        self.update_bus.publish({
            "type": "chart_error",
            "error": str(error),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def _on_connect(self) -> None:
        """Callback for WebSocket connection (called from WebSocket thread)."""
        self.update_bus.publish({
            "type": "chart_connect",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def _on_disconnect(self) -> None:
        """Callback for WebSocket disconnection (called from WebSocket thread)."""
        self.update_bus.publish({
            "type": "chart_disconnect",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def is_running(self) -> bool:
        """Check if worker is running."""
        return self.ws_client is not None and self.ws_client.is_connected()
