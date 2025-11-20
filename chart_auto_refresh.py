"""Chart auto-refresh worker for the Charts tab."""

from __future__ import annotations

import copy
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from indicator_collector.trading_system.auto_analyze_worker import get_binance_server_time_ms
from indicator_collector.trading_system.data_sources.binance_source import BinanceKlinesSource

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


def read_chart_state(session_state: Any) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
    """Safely read chart DataFrame and indicators from session state."""
    with _CHART_DATA_LOCK:
        df = getattr(session_state, "chart_df", None)
        df_copy: Optional[pd.DataFrame]
        if df is not None and isinstance(df, pd.DataFrame):
            df_copy = df.copy(deep=True)
        else:
            df_copy = None
        indicators = copy.deepcopy(getattr(session_state, "chart_indicators", {}))
        last_closed_ts = getattr(session_state, "last_closed_ts", 0)
    return df_copy, indicators, last_closed_ts


def update_chart_state(
    session_state: Any,
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    last_closed_ts: int,
) -> None:
    """Safely update chart data and indicators in session state."""
    with _CHART_DATA_LOCK:
        session_state.chart_df = df
        session_state.chart_indicators = indicators
        session_state.last_closed_ts = last_closed_ts
        session_state.analysis_updated = True


def floor_closed_bar_local(now_ms: int, tf_ms: int, tol_ms: int = 60_000) -> int:
    """
    Calculate the timestamp of the last closed bar boundary.
    
    Args:
        now_ms: Current time in milliseconds
        tf_ms: Timeframe interval in milliseconds
        tol_ms: Tolerance in milliseconds (default 60s)
        
    Returns:
        Timestamp of the last closed bar boundary in milliseconds
    """
    if tf_ms <= 0:
        return now_ms
    
    # For 3h, we need to align to 00:00, 03:00, 06:00, etc.
    if tf_ms == 10_800_000:  # 3h in milliseconds
        # Align to midnight UTC
        day_start_ms = (now_ms // 86_400_000) * 86_400_000
        # Calculate which 3h boundary we're in
        elapsed_from_day_start = now_ms - day_start_ms
        current_3h_index = elapsed_from_day_start // tf_ms
        current_bar_start = day_start_ms + (current_3h_index * tf_ms)
    else:
        # Floor to the current bar start
        current_bar_start = (now_ms // tf_ms) * tf_ms
    
    # Last closed bar is the bar before the current one
    last_closed = current_bar_start - tf_ms
    
    # Ensure we're not too close to the boundary (within tolerance)
    if (now_ms - current_bar_start) < tol_ms:
        # We're too close to the current bar start, use the previous bar
        last_closed = current_bar_start - tf_ms
    
    return last_closed


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
    
    # Calculate last closed bar timestamp
    last_closed_ts = floor_closed_bar_local(server_time_ms, tf_ms, tol_ms=60_000)
    
    # Calculate start time (go back num_bars from last closed)
    start_ms = max(0, last_closed_ts - (tf_ms * num_bars))
    
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
        """Main worker loop that checks for new closed bars."""
        while not self._stop_event.is_set():
            try:
                # Get Binance server time
                now_ms = get_binance_server_time_ms(self.data_source)
                
                # Calculate last closed bar
                last_closed = floor_closed_bar_local(now_ms, self.tf_ms)
                
                # Get current last_closed_ts from session state
                current_last_closed_ts = getattr(self.session_state, "last_closed_ts", 0)
                
                # Check if we have a new closed bar
                if last_closed > current_last_closed_ts:
                    logger.info(
                        f"New closed bar detected: {last_closed} (previous: {current_last_closed_ts})"
                    )
                    
                    try:
                        # Fetch updated candles
                        df, actual_last_closed = fetch_closed_candles(
                            symbol=self.symbol,
                            timeframe=self.timeframe,
                            num_bars=self.num_bars,
                            data_source=self.data_source,
                        )
                        
                        # Compute indicators for overlays
                        indicators = compute_chart_indicators(df)
                        
                        # Update session state atomically with helper function
                        update_chart_state(
                            self.session_state,
                            df,
                            indicators,
                            actual_last_closed,
                        )
                        
                        logger.info(f"Chart data updated successfully for closed bar {actual_last_closed}")
                        last_closed = actual_last_closed
                    
                    except Exception as exc:
                        logger.error(f"Failed to update chart data: {exc}", exc_info=True)
                
                # Calculate sleep time until next boundary
                next_boundary = last_closed + self.tf_ms
                sleep_ms = max(5_000, next_boundary - now_ms)  # At least 5 seconds
                sleep_seconds = sleep_ms / 1000.0
                
                # Sleep in small intervals to allow quick stop
                while not self._stop_event.is_set() and sleep_seconds > 0:
                    sleep_chunk = min(1.0, sleep_seconds)
                    time.sleep(sleep_chunk)
                    sleep_seconds -= sleep_chunk
            
            except Exception as exc:
                logger.error(f"Error in chart worker loop: {exc}", exc_info=True)
                time.sleep(5.0)  # Wait before retrying
