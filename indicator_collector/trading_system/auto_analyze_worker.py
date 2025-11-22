"""Background worker for timeframe-based auto-analysis."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..timeframes import Timeframe
from .automated_signals import run_automated_signal_flow
from .data_sources.binance_source import BinanceKlinesSource
from .signal_generator import SignalConfig

logger = logging.getLogger(__name__)

# Mapping of timeframe to milliseconds
TIMEFRAME_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "3h": 10_800_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

_SERVER_TIME_FALLBACK_WARNED = False


def floor_closed_bar(now_ms: int, tf_ms: int, tol_ms: int = 60_000) -> int:
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
    
    # Floor to the current bar start
    current_bar_start = (now_ms // tf_ms) * tf_ms
    
    # Last closed bar is the bar before the current one
    last_closed = current_bar_start - tf_ms
    
    # Ensure we're not too close to the boundary (within tolerance)
    if (now_ms - current_bar_start) < tol_ms:
        # We're too close to the current bar start, use the previous bar
        last_closed = current_bar_start - tf_ms
    
    return last_closed


def get_binance_server_time_ms(source: Optional[BinanceKlinesSource] = None) -> int:
    """
    Get Binance server time in milliseconds.
    
    Args:
        source: Optional BinanceKlinesSource instance
        
    Returns:
        Server time in milliseconds
    """
    global _SERVER_TIME_FALLBACK_WARNED
    
    if source is None:
        source = BinanceKlinesSource()
    
    try:
        server_time = source.get_server_time()
        if server_time is not None:
            return server_time
    except Exception as exc:
        logger.debug("Error while fetching Binance server time", exc_info=True)
        last_error = exc
    else:
        last_error = RuntimeError("get_server_time() returned None")
    
    if not _SERVER_TIME_FALLBACK_WARNED:
        _SERVER_TIME_FALLBACK_WARNED = True
        logger.warning(
            "Falling back to system clock for Binance server time: %s", last_error
        )
    
    fallback_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return fallback_ms


def run_analysis(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    signal_config: Optional[SignalConfig] = None,
    indicator_params: Optional[Dict[str, Any]] = None,
    signal_params: Optional[Dict[str, Any]] = None,
    data_source: Optional[BinanceKlinesSource] = None,
) -> Dict[str, Any]:
    """
    Run a single analysis for the given symbol and timeframe.
    
    This function fetches only CLOSED bars using Binance server time,
    builds a normalized payload via load_full_payload, and calls generate_signals.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "1h", "3h")
        start_ms: Start timestamp in milliseconds
        end_ms: End timestamp in milliseconds (should be aligned to closed bar)
        signal_config: Optional signal configuration
        indicator_params: Optional indicator parameters
        signal_params: Optional signal parameters
        data_source: Optional BinanceKlinesSource instance
        
    Returns:
        Dictionary with 'result' (signal JSON) and 'last_closed_ts' (int ms)
    """
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    
    # Use run_automated_signal_flow which already handles closed bars correctly
    result = run_automated_signal_flow(
        symbol=symbol,
        timeframe=timeframe,
        start=start_dt,
        end=end_dt,
        data_source=data_source,
        validate_real_data=True,
        signal_config=signal_config,
        indicator_params=indicator_params,
        signal_params=signal_params,
    )
    
    # Get the last closed timestamp from the result
    candles = result.candles
    if not candles:
        raise ValueError("No candles returned from analysis")
    
    last_closed_ts = int(candles[-1]["ts"])
    
    return {
        "result": result.explicit_signal,
        "processed_payload": result.processed_payload,
        "candles": result.candles,
        "last_closed_ts": last_closed_ts,
    }


class AutoAnalyzeWorker:
    """Background worker that runs analysis on new closed bars."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        session_state: Any,
        signal_config: Optional[SignalConfig] = None,
        indicator_params: Optional[Dict[str, Any]] = None,
        signal_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the auto-analyze worker.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            start_ms: Start timestamp in milliseconds
            end_ms: End timestamp in milliseconds (user-defined end)
            session_state: Streamlit session state object
            signal_config: Optional signal configuration
            indicator_params: Optional indicator parameters
            signal_params: Optional signal parameters
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_ms = start_ms
        self.user_end_ms = end_ms
        self.session_state = session_state
        self.signal_config = signal_config
        self.indicator_params = indicator_params
        self.signal_params = signal_params
        self.data_source = BinanceKlinesSource()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Get timeframe interval in milliseconds
        self.tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Worker thread already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Auto-analyze worker started for {self.symbol} {self.timeframe}")
    
    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"Auto-analyze worker stopped for {self.symbol} {self.timeframe}")
    
    def _run_loop(self) -> None:
        """Main worker loop that checks for new closed bars."""
        while not self._stop_event.is_set():
            try:
                # Get Binance server time
                now_ms = get_binance_server_time_ms(self.data_source)
                
                # Calculate last closed bar
                last_closed = floor_closed_bar(now_ms, self.tf_ms)
                
                # Get current last_closed_ts from session state
                current_last_closed_ts = getattr(self.session_state, "last_closed_ts", None)
                
                # Check if we have a new closed bar
                if current_last_closed_ts is None or last_closed > current_last_closed_ts:
                    logger.info(
                        f"New closed bar detected: {last_closed} (previous: {current_last_closed_ts})"
                    )
                    
                    # Run analysis with end time = min(user_end_ms, last_closed)
                    effective_end_ms = min(self.user_end_ms, last_closed)
                    
                    try:
                        result = run_analysis(
                            symbol=self.symbol,
                            timeframe=self.timeframe,
                            start_ms=self.start_ms,
                            end_ms=effective_end_ms,
                            signal_config=self.signal_config,
                            indicator_params=self.indicator_params,
                            signal_params=self.signal_params,
                            data_source=self.data_source,
                        )
                        
                        # Update session state
                        self.session_state.last_closed_ts = last_closed
                        self.session_state.last_analysis_ts = now_ms
                        self.session_state.analysis_result = result
                        self.session_state.analysis_error = None
                        self.session_state.analysis_updated = True
                        
                        logger.info(f"Analysis completed successfully for closed bar {last_closed}")
                    
                    except Exception as exc:
                        logger.error(f"Analysis failed: {exc}", exc_info=True)
                        self.session_state.analysis_error = str(exc)
                        self.session_state.analysis_updated = True
                
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
                logger.error(f"Error in worker loop: {exc}", exc_info=True)
                time.sleep(5.0)  # Wait before retrying
