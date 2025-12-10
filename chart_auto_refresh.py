"""Chart auto-refresh worker for the Charts tab."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

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

CACHE_SOURCE = "binance"
OVERLAP_BARS = 3

_CANDLE_CACHE: Dict[tuple[str, str, str, int, int], pd.DataFrame] = {}
_CACHE_LOCK = threading.Lock()


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
    
    # Ensure non-negative timestamps
    if last_closed < 0:
        last_closed = 0
    
    # If we're within the tolerance of a new bar boundary, keep previous bar
    if (now_ms - current_bar_start) < tol_ms:
        return last_closed
    
    return last_closed


def invalidate_cache(symbol: str, timeframe: str) -> None:
    """Invalidate cache for a specific symbol/timeframe combination."""
    with _CACHE_LOCK:
        keys_to_remove = [
            k for k in _CANDLE_CACHE.keys()
            if k[0] == CACHE_SOURCE and k[1] == symbol and k[2] == timeframe
        ]
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
        Tuple of (DataFrame, last_closed_close_ts)
    """
    if data_source is None:
        data_source = BinanceKlinesSource()
    
    # Get Binance server time
    server_time_ms = get_binance_server_time_ms(data_source)
    
    # Get timeframe in milliseconds
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    if tf_ms <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Calculate last closed bar start and close timestamps
    last_closed_start = floor_closed_bar_local(server_time_ms, tf_ms, tol_ms=60_000)
    last_closed_start = max(last_closed_start, 0)
    last_closed_close = last_closed_start + tf_ms
    if last_closed_close <= 0:
        raise ValueError("No closed candles available yet from Binance")
    
    effective_bars = max(int(num_bars), 1)
    fetch_span_bars = effective_bars + OVERLAP_BARS
    start_ms = max(0, last_closed_close - (tf_ms * fetch_span_bars))
    
    # Check cache using (source, symbol, timeframe, start, end)
    cache_key = (CACHE_SOURCE, symbol, timeframe, start_ms, last_closed_close)
    if use_cache:
        with _CACHE_LOCK:
            cached_df = _CANDLE_CACHE.get(cache_key)
        if cached_df is not None:
            logger.debug(f"Using cached candles for {symbol} {timeframe}")
            trimmed = cached_df.tail(effective_bars).reset_index(drop=True)
            return trimmed, last_closed_close
    
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(last_closed_close / 1000, tz=timezone.utc)
    
    try:
        clean_symbol = symbol[8:] if symbol.startswith("BINANCE:") else symbol
        df = data_source.load_candles(
            symbol=clean_symbol,
            timeframe=timeframe,
            start=start_dt,
            end=end_dt,
        )
    except Exception as exc:
        logger.error(f"Failed to fetch candles for {symbol} {timeframe}: {exc}")
        raise
    
    if df is None or df.empty:
        raise ValueError(f"No Binance candles returned for {symbol} {timeframe}")
    
    # Sort, deduplicate, and ensure only fully closed bars (ts is bar start)
    df = (
        df.drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    df = df[df["ts"] <= last_closed_start].copy()
    if df.empty:
        raise ValueError(f"No closed candles available for {symbol} {timeframe}")
    
    trimmed_df = df.tail(effective_bars).reset_index(drop=True)
    
    with _CACHE_LOCK:
        _CANDLE_CACHE[cache_key] = df.copy()
    
    logger.info(
        "Fetched %s candles for %s %s (start=%s, end=%s, last_closed=%s)",
        len(trimmed_df),
        symbol,
        timeframe,
        start_dt.strftime("%Y-%m-%d %H:%M"),
        end_dt.strftime("%Y-%m-%d %H:%M"),
        datetime.fromtimestamp(last_closed_close / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
    )
    
    return trimmed_df, last_closed_close


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
                        
                        # Update session state
                        self.session_state.chart_df = df
                        self.session_state.last_closed_ts = actual_last_closed
                        self.session_state.analysis_updated = True
                        
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
