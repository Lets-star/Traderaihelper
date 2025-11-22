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

logger = logging.getLogger(__name__)

# Mapping of timeframe to milliseconds
TIMEFRAME_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "3h": 10_800_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

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


def floor_closed_bar_local(now_ms: int, tf_ms: int, tol_ms: int = 60_000) -> int:
    """
    Calculate the close_time of the last closed bar boundary.
    
    This returns the close_time (not open_time) of the last closed candle.
    For a candle with open_time T, its close_time is T + tf_ms.
    
    Args:
        now_ms: Current time in milliseconds (UTC)
        tf_ms: Timeframe interval in milliseconds
        tol_ms: Tolerance in milliseconds (default 60s)
        
    Returns:
        close_time of the last closed bar in milliseconds (UTC)
    """
    if tf_ms <= 0:
        return now_ms
    
    effective_now = max(now_ms - tol_ms, 0)
    last_closed_close_ms = (effective_now // tf_ms) * tf_ms
    return last_closed_close_ms


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
    last_closed_ts = floor_closed_bar_local(server_time_ms, tf_ms, tol_ms=tol_ms)
    
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


def fetch_delta_candles(
    symbol: str,
    timeframe: str,
    last_closed_close_ms: int,
    data_source: Optional[BinanceKlinesSource] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Fetch new CLOSED bars since last_closed_close_ms using delta/incremental update.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "1h", "3h")
        last_closed_close_ms: Close time of the last closed bar we already have (UTC ms)
        data_source: Optional BinanceKlinesSource instance
        
    Returns:
        Tuple of (DataFrame with new bars, new_last_closed_close_ms)
        
    Note:
        - last_closed_close_ms is the close_time of the last bar we have
        - DataFrame 'ts' column contains open_time (close_time - tf_ms)
        - To get new bars, we filter by: open_time >= last_closed_close_ms
          (since the next bar's open_time equals the previous bar's close_time)
    """
    if data_source is None:
        data_source = BinanceKlinesSource()
    
    # Get Binance server time
    server_time_ms = get_binance_server_time_ms(data_source)
    
    # Get timeframe in milliseconds
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    # Calculate new last closed bar close_time using tight tolerance
    tol_ms = DEFAULT_TOLERANCE_MS
    new_last_closed_close_ms = floor_closed_bar_local(server_time_ms, tf_ms, tol_ms=tol_ms)
    
    # Check if there are new closed bars
    if new_last_closed_close_ms <= last_closed_close_ms:
        logger.debug(f"No new closed bars for {symbol} {timeframe}")
        return pd.DataFrame(), last_closed_close_ms
    
    # Fetch with overlap to ensure continuity and avoid gaps
    # Start from (OVERLAP_BARS) bars before the last closed bar
    start_open_ms = max(0, last_closed_close_ms - (tf_ms * OVERLAP_BARS))
    
    # Convert to datetime
    start_dt = datetime.fromtimestamp(start_open_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(new_last_closed_close_ms / 1000, tz=timezone.utc)
    
    # Strip BINANCE: prefix if present
    clean_symbol = symbol
    if clean_symbol.startswith("BINANCE:"):
        clean_symbol = clean_symbol[8:]
    
    try:
        df = data_source.load_candles(
            symbol=clean_symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
        )
        
        # Filter to NEW bars only: bars with open_time >= last_closed_close_ms
        # (next bar's open_time equals previous bar's close_time)
        if not df.empty:
            df = df[df["ts"] >= last_closed_close_ms].copy()
        
        return df, new_last_closed_close_ms
    except Exception as e:
        logger.error(f"Failed to fetch delta candles for {symbol} {timeframe}: {e}")
        # On failure, return empty dataframe with unchanged last_closed_close_ms
        return pd.DataFrame(), last_closed_close_ms


def fetch_forming_bar(
    symbol: str,
    timeframe: str,
    data_source: Optional[BinanceKlinesSource] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch the current forming bar (if available).
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "1h", "3h")
        data_source: Optional BinanceKlinesSource instance
        
    Returns:
        DataFrame with a single row for the forming bar, or None if not available
    """
    if data_source is None:
        data_source = BinanceKlinesSource()
    
    # Get Binance server time
    server_time_ms = get_binance_server_time_ms(data_source)
    
    # Get timeframe in milliseconds
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    # Calculate current bar start
    if tf_ms == 10_800_000:
        day_start_ms = (server_time_ms // 86_400_000) * 86_400_000
        elapsed_from_day_start = server_time_ms - day_start_ms
        current_3h_index = elapsed_from_day_start // tf_ms
        open_ms = day_start_ms + (current_3h_index * tf_ms)
    else:
        open_ms = (server_time_ms // tf_ms) * tf_ms
    
    # For 3h, we need to aggregate from 1h candles
    if timeframe == "3h":
        try:
            clean_symbol = symbol
            if clean_symbol.startswith("BINANCE:"):
                clean_symbol = clean_symbol[8:]
            
            # Fetch 1h candles for the current 3h period
            start_dt = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc)
            
            df_1h = data_source.load_candles(
                symbol=clean_symbol,
                timeframe="1h",
                start=start_dt,
                end=end_dt,
            )
            
            if df_1h.empty:
                return None
            
            # Filter to current 3h period
            df_1h = df_1h[df_1h["ts"] >= open_ms].copy()
            
            if df_1h.empty:
                return None
            
            # Aggregate to forming 3h bar
            forming_bar = pd.DataFrame([{
                "ts": open_ms,
                "open": df_1h.iloc[0]["open"],
                "high": df_1h["high"].max(),
                "low": df_1h["low"].min(),
                "close": df_1h.iloc[-1]["close"],
                "volume": df_1h["volume"].sum(),
            }])
            
            return forming_bar
            
        except Exception as e:
            logger.error(f"Failed to fetch forming 3h bar for {symbol}: {e}")
            return None
    else:
        # For other timeframes, fetch the current bar directly via klines endpoint
        base_url: Optional[str] = None
        try:
            clean_symbol = symbol
            if clean_symbol.startswith("BINANCE:"):
                clean_symbol = clean_symbol[8:]
            
            interval = data_source.TIMEFRAME_TO_BINANCE_INTERVAL.get(timeframe, timeframe)
            base_url = data_source._active_base_url or data_source._ensure_active_base_url()
            data_source._active_base_url = base_url
            params = {
                "symbol": clean_symbol,
                "interval": interval,
                "limit": 2,
            }
            response = data_source.session.get(
                f"{base_url}{KLINES_ENDPOINT}",
                params=params,
                timeout=(data_source.connect_timeout, min(data_source.read_timeout, 5.0)),
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch forming bar klines ({response.status_code}): {response.text[:200]}"
                )
            payload = response.json()
            if not payload:
                return None
            target = None
            for entry in reversed(payload):
                if int(entry[0]) == open_ms:
                    target = entry
                    break
            if target is None:
                target = payload[-1]
                if int(target[0]) != open_ms:
                    return None
            close_time_ms = int(target[6])
            if close_time_ms <= server_time_ms:
                # Kline already closed; nothing to preview
                return None
            forming_bar = pd.DataFrame([
                {
                    "ts": int(target[0]),
                    "open": float(target[1]),
                    "high": float(target[2]),
                    "low": float(target[3]),
                    "close": float(target[4]),
                    "volume": float(target[5]),
                }
            ])
            if data_source.rate_limit_delay > 0:
                data_source._sleep(data_source.rate_limit_delay)
            data_source._record_success(base_url)
            return forming_bar
        except RequestException as exc:
            if base_url:
                data_source._record_failure(base_url, exc, retryable=True)
            logger.debug("Request error while fetching forming bar: %s", data_source._format_request_error(exc))
            return None
        except Exception as e:
            if base_url:
                data_source._record_failure(base_url, e, retryable=False)
            logger.debug(f"Failed to fetch forming bar for {symbol} {timeframe}: {e}", exc_info=True)
            return None


class ChartAutoRefreshWorker:
    """Background worker that refreshes chart data on new closed bars."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        num_bars: int,
        session_state: Any,
    ):
        """
        Initialize the chart auto-refresh worker.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            num_bars: Number of bars to fetch
            session_state: Streamlit session state object
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.num_bars = num_bars
        self.session_state = session_state
        self.data_source = BinanceKlinesSource()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Get timeframe interval in milliseconds
        self.tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Chart worker thread already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.session_state.worker_running = True
        logger.info(f"Chart auto-refresh worker started for {self.symbol} {self.timeframe}")
    
    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.session_state.worker_running = False
        logger.info(f"Chart auto-refresh worker stopped for {self.symbol} {self.timeframe}")
    
    def _run_loop(self) -> None:
        """Main worker loop that checks for new closed bars with TradingView-like behavior."""
        poll_interval = get_poll_interval(self.timeframe)
        
        while not self._stop_event.is_set():
            try:
                now_ms = get_binance_server_time_ms(self.data_source)
                new_last_closed_close_ms = floor_closed_bar_local(now_ms, self.tf_ms, tol_ms=DEFAULT_TOLERANCE_MS)
                
                current_last_closed_close_ms = get_last_closed_from_state(
                    self.session_state,
                    self.symbol,
                    self.timeframe,
                )
                
                if new_last_closed_close_ms > current_last_closed_close_ms:
                    logger.info(
                        f"[{self.symbol} {self.timeframe}] New closed bar boundary: "
                        f"last_closed_close_ms={new_last_closed_close_ms}, previous={current_last_closed_close_ms}"
                    )
                    
                    try:
                        if current_last_closed_close_ms == 0:
                            logger.info(f"Initial fetch for {self.symbol} {self.timeframe}")
                            df_new, actual_last_closed_close_ms = fetch_closed_candles(
                                symbol=self.symbol,
                                timeframe=self.timeframe,
                                num_bars=self.num_bars,
                                data_source=self.data_source,
                            )
                            update_chart_state(
                                self.session_state,
                                self.symbol,
                                self.timeframe,
                                df_new,
                                actual_last_closed_close_ms,
                                append=False,
                            )
                            with _CHART_DATA_LOCK:
                                stored_df = getattr(self.session_state, "chart_df", None)
                                final_len = len(stored_df) if isinstance(stored_df, pd.DataFrame) else 0
                            deduped = max(len(df_new) - final_len, 0)
                            logger.info(
                                "[%s %s] Boundary update (initial): last_closed_close_ms=%s, fetched=%s, appended=%s, deduped=%s",
                                self.symbol,
                                self.timeframe,
                                actual_last_closed_close_ms,
                                len(df_new),
                                final_len,
                                deduped,
                            )
                        else:
                            df_new, actual_last_closed_close_ms = fetch_delta_candles(
                                symbol=self.symbol,
                                timeframe=self.timeframe,
                                last_closed_close_ms=current_last_closed_close_ms,
                                data_source=self.data_source,
                            )
                            
                            if df_new.empty and actual_last_closed_close_ms <= current_last_closed_close_ms:
                                logger.debug(f"No new closed bars yet for {self.symbol} {self.timeframe}")
                            else:
                                current_df, _, _ = read_chart_state(
                                    self.session_state,
                                    self.symbol,
                                    self.timeframe,
                                )
                                
                                append_mode = current_df is not None and not current_df.empty
                                before_len = len(current_df) if append_mode else 0
                                update_chart_state(
                                    self.session_state,
                                    self.symbol,
                                    self.timeframe,
                                    df_new,
                                    actual_last_closed_close_ms,
                                    append=append_mode,
                                )
                                with _CHART_DATA_LOCK:
                                    stored_df = getattr(self.session_state, "chart_df", None)
                                    after_len = len(stored_df) if isinstance(stored_df, pd.DataFrame) else before_len
                                appended = max(after_len - before_len, 0)
                                deduped = max(len(df_new) - appended, 0)
                                logger.info(
                                    "[%s %s] Boundary update: last_closed_close_ms=%s, fetched=%s, appended=%s, deduped=%s",
                                    self.symbol,
                                    self.timeframe,
                                    actual_last_closed_close_ms,
                                    len(df_new),
                                    appended,
                                    deduped,
                                )
                    
                    except Exception as exc:
                        logger.error(f"Failed to update chart data: {exc}", exc_info=True)
                
                # Check if forming bar is enabled
                show_forming_bar = getattr(self.session_state, "show_forming_bar", False)
                
                if show_forming_bar:
                    try:
                        forming_bar_df = fetch_forming_bar(
                            self.symbol,
                            self.timeframe,
                            self.data_source,
                        )
                        
                        if forming_bar_df is not None and not forming_bar_df.empty:
                            with _CHART_DATA_LOCK:
                                current_df = getattr(self.session_state, "chart_df", None)
                                if current_df is not None and not current_df.empty:
                                    closed_bars = current_df.copy(deep=True)
                                    combined = pd.concat([closed_bars, forming_bar_df], ignore_index=True)
                                    combined = (
                                        combined.drop_duplicates(subset="ts", keep="last")
                                        .sort_values("ts")
                                        .reset_index(drop=True)
                                    )
                                    indicators = compute_chart_indicators(combined)
                                    self.session_state.chart_df_with_forming = combined
                                    self.session_state.chart_indicators = indicators
                                    self.session_state.analysis_updated = True
                    except Exception as exc:
                        logger.debug(f"Failed to fetch forming bar: {exc}")
                else:
                    with _CHART_DATA_LOCK:
                        if getattr(self.session_state, "chart_df_with_forming", None) is not None:
                            self.session_state.chart_df_with_forming = None
                            closed_df = getattr(self.session_state, "chart_df", None)
                            if isinstance(closed_df, pd.DataFrame) and not closed_df.empty:
                                self.session_state.chart_indicators = compute_chart_indicators(closed_df)
                                self.session_state.analysis_updated = True

                # Adaptive poll interval: sleep poll_interval seconds
                sleep_seconds = poll_interval
                while not self._stop_event.is_set() and sleep_seconds > 0:
                    sleep_chunk = min(0.5, sleep_seconds)
                    time.sleep(sleep_chunk)
                    sleep_seconds -= sleep_chunk
            
            except Exception as exc:
                logger.error(f"Error in chart worker loop: {exc}", exc_info=True)
                time.sleep(5.0)
