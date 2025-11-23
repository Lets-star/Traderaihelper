"""
Automated signals auto-refresh worker for the Automated Signals tab.

This worker automatically advances the End time to the latest closed TF boundary
and triggers immediate data refresh and signal recomputation without user interaction.

TIMESTAMP SEMANTICS:
-------------------
All internal timestamps are in UTC milliseconds.

- auto_end_time_ms: close_time of the last closed bar (stored in session state)
- Worker polls Binance server time and detects new closed bars
- When detected, immediately fetches new candles and recomputes signals

The worker:
1. Polls at adaptive cadence (1s for <=15m, 5s for >=1h)
2. Detects new closed bar boundaries
3. Triggers immediate data refresh and signal recomputation
4. Sets analysis_updated flag to notify UI
5. Uses threading.Lock for thread-safe session state updates
6. Never calls st.* APIs from the worker thread
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import threading
import time
from typing import Any, Dict, Optional

from chart_auto_refresh import TIMEFRAME_TO_MS, floor_closed_bar_local
from indicator_collector.trading_system.auto_analyze_worker import get_binance_server_time_ms
from indicator_collector.trading_system.automated_signals import run_automated_signal_flow
from indicator_collector.trading_system.data_sources.binance_source import BinanceKlinesSource
from indicator_collector.trading_system.data_sources.timestamp_utils import normalize_timestamp
from indicator_collector.trading_system.signal_generator import SignalConfig

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE_MS = 60_000  # 60 seconds tolerance to avoid fetching incomplete bars


def get_poll_interval(timeframe: str) -> float:
    """
    Get poll interval in seconds based on timeframe.
    
    Args:
        timeframe: Timeframe string (e.g., "1m", "5m", "1h")
        
    Returns:
        Poll interval in seconds (1s for <=15m, 5s for >=1h)
    """
    tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    if tf_ms <= 900_000:  # <=15m
        return 1.0
    else:  # >=1h
        return 5.0


class AutomatedSignalsWorker:
    """Background worker that auto-advances End time and refreshes signals on new closed bars."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        session_state: Any,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
    ):
        """
        Initialize the automated signals worker.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe string (e.g., "1h", "3h")
            session_state: Streamlit session state object
            signal_config_payload: Signal configuration payload
            indicator_params: Indicator parameters
            signal_params: Signal parameters
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.session_state = session_state
        self.signal_config_payload = signal_config_payload
        self.indicator_params = indicator_params
        self.signal_params = signal_params
        self.data_source = BinanceKlinesSource()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Get timeframe interval in milliseconds
        self.tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
    
    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Automated signals worker thread already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        with self._lock:
            self.session_state.automated_signals_worker_running = True
        logger.info(f"Automated signals worker started for {self.symbol} {self.timeframe}")
    
    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with self._lock:
            self.session_state.automated_signals_worker_running = False
        logger.info(f"Automated signals worker stopped for {self.symbol} {self.timeframe}")
    
    def _run_loop(self) -> None:
        """Main worker loop that checks for new closed bars and triggers signal refresh."""
        poll_interval = get_poll_interval(self.timeframe)
        
        while not self._stop_event.is_set():
            try:
                # Check if auto-advance is still enabled
                with self._lock:
                    state = getattr(self.session_state, "automated_signals_state", {})
                    end_time_user_set = state.get("end_time_user_set", False)
                    
                    if end_time_user_set:
                        # Auto-advance disabled, skip this iteration
                        logger.debug(f"Auto-advance disabled for {self.symbol} {self.timeframe}, skipping")
                        time.sleep(poll_interval)
                        continue
                
                # Get Binance server time
                server_time_ms = get_binance_server_time_ms(self.data_source)
                server_time_ms = normalize_timestamp(server_time_ms)
                
                # Calculate new last closed bar boundary (close_time)
                new_last_closed_close_ms = floor_closed_bar_local(
                    server_time_ms,
                    self.tf_ms,
                    tol_ms=DEFAULT_TOLERANCE_MS,
                )
                
                with self._lock:
                    state = getattr(self.session_state, "automated_signals_state", {})
                    current_auto_end_time_ms = state.get("auto_end_time_ms", 0)
                    
                    # Check if there's a new closed bar
                    if new_last_closed_close_ms > current_auto_end_time_ms:
                        logger.info(
                            f"[{self.symbol} {self.timeframe}] New closed bar boundary detected: "
                            f"auto_end_time_ms={new_last_closed_close_ms}, previous={current_auto_end_time_ms}"
                        )
                        
                        # Update auto_end_time_ms
                        state["auto_end_time_ms"] = new_last_closed_close_ms
                        state["fetch_needed"] = True
                        setattr(self.session_state, "automated_signals_state", state)
                        
                        # Trigger immediate data refresh and signal recomputation
                        try:
                            self._refresh_signals(new_last_closed_close_ms)
                        except Exception as exc:
                            logger.error(f"Failed to refresh signals: {exc}", exc_info=True)
                            with self._lock:
                                state["fetch_needed"] = False
                                state["error"] = f"Signal refresh failed: {exc}"
                                setattr(self.session_state, "automated_signals_state", state)
                
                # Adaptive poll interval: sleep poll_interval seconds
                sleep_seconds = poll_interval
                while not self._stop_event.is_set() and sleep_seconds > 0:
                    sleep_chunk = min(0.5, sleep_seconds)
                    time.sleep(sleep_chunk)
                    sleep_seconds -= sleep_chunk
            
            except Exception as exc:
                logger.error(f"Error in automated signals worker loop: {exc}", exc_info=True)
                time.sleep(5.0)
    
    def _refresh_signals(self, end_time_ms: int) -> None:
        """
        Fetch new candles and recompute signals.
        
        Args:
            end_time_ms: End time in milliseconds (UTC)
        """
        logger.info(f"Refreshing signals for {self.symbol} {self.timeframe} up to {end_time_ms}")
        
        # Convert end time to datetime
        end_dt = dt.datetime.fromtimestamp(end_time_ms / 1000, tz=dt.timezone.utc)
        
        # Get start time from session state
        with self._lock:
            state = getattr(self.session_state, "automated_signals_state", {})
            start_dt_str = state.get("start_datetime_iso")
            if start_dt_str:
                start_dt = dt.datetime.fromisoformat(start_dt_str)
            else:
                # Default to 200 bars back
                start_dt = end_dt - dt.timedelta(milliseconds=self.tf_ms * 200)
        
        # Build signal config
        weights = self.signal_config_payload.get("weights", {})
        signal_config = SignalConfig(
            technical_weight=weights.get("technical", 0.25),
            sentiment_weight=weights.get("sentiment", 0.15),
            multitimeframe_weight=weights.get("multitimeframe", 0.10),
            volume_weight=weights.get("volume", 0.20),
            structure_weight=weights.get("market_structure", 0.15),
            composite_weight=weights.get("composite", 0.0),
            min_factors_confirm=int(self.signal_config_payload.get("min_confirmations", 3)),
            buy_threshold=float(self.signal_config_payload.get("buy_threshold", 0.65)),
            sell_threshold=float(self.signal_config_payload.get("sell_threshold", 0.35)),
            min_confidence=float(self.signal_config_payload.get("min_confidence", 0.6)),
        )
        
        # Calculate minimum candles needed
        indicator_periods = self.indicator_params.get("rsi", {})
        atr_period = int(self.indicator_params.get("atr", {}).get("period", 14))
        macd_slow = int(self.indicator_params.get("macd", {}).get("slow", 26))
        macd_signal = int(self.indicator_params.get("macd", {}).get("signal", 9))
        rsi_period = int(indicator_periods.get("period", 14))
        min_candles = max(
            30,
            rsi_period + 2,
            atr_period + 2,
            macd_slow + macd_signal,
        )
        
        # Run automated signal flow
        result = run_automated_signal_flow(
            self.symbol,
            self.timeframe,
            start_dt,
            end_dt,
            validate_real_data=True,
            signal_config=signal_config,
            indicator_params=self.indicator_params,
            signal_params=self.signal_params,
            min_candles=min_candles,
        )
        
        # Store results in session state
        result_dict = {
            "candles": result.candles,
            "processed_payload": result.processed_payload,
            "explicit_signal": result.explicit_signal,
        }
        
        final_indicator_params = (
            result_dict.get("explicit_signal", {})
            .get("metadata", {})
            .get("indicator_params")
            or result_dict.get("processed_payload", {})
            .get("metadata", {})
            .get("indicator_params")
            or self.indicator_params
        )
        
        with self._lock:
            state = getattr(self.session_state, "automated_signals_state", {})
            state.update(
                {
                    "result": result_dict,
                    "error": None,
                    "candles": result_dict.get("candles", []),
                    "processed_payload": result_dict.get("processed_payload"),
                    "explicit_signal": result_dict.get("explicit_signal"),
                    "indicator_params": final_indicator_params,
                    "analysis_updated": True,
                    "fetch_needed": False,
                }
            )
            setattr(self.session_state, "automated_signals_state", state)
        
        # Trigger ByBit execution if configured
        try:
            executor = getattr(self.session_state, "signal_executor", None)
            if executor and executor.enabled:
                explicit_signal = result_dict.get("explicit_signal", {})
                signal_type = explicit_signal.get("signal", "HOLD")
                
                if signal_type in ["BUY", "SELL"]:
                    entries = explicit_signal.get("entries", [])
                    entry_price = float(entries[0]) if entries else 0.0
                    
                    take_profits = explicit_signal.get("take_profits", {})
                    tp_price = 0.0
                    if isinstance(take_profits, dict) and take_profits:
                        tp_price = float(list(take_profits.values())[0])
                    
                    stop_loss = float(explicit_signal.get("stop_loss", 0.0))
                    
                    payload = {
                        "signal_id": f"{self.symbol}_{end_time_ms}",
                        "symbol": self.symbol,
                        "direction": "LONG" if signal_type == "BUY" else "SHORT",
                        "entry_price": entry_price,
                        "take_profit": tp_price,
                        "stop_loss": stop_loss,
                        "leverage": 5,
                        "quantity": 0.001,
                        "generated_at": end_time_ms
                    }
                    
                    executor.execute_signal(payload)
        except Exception as exc:
             logger.error(f"Failed to execute signal: {exc}", exc_info=True)
        
        logger.info(
            f"Signal refresh complete for {self.symbol} {self.timeframe}: "
            f"{len(result.candles)} candles, signal={result.explicit_signal.get('signal', 'UNKNOWN')}"
        )
    
    def update_config(
        self,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
    ) -> None:
        """
        Update configuration without restarting the worker.
        
        Args:
            signal_config_payload: Signal configuration payload
            indicator_params: Indicator parameters
            signal_params: Signal parameters
        """
        with self._lock:
            self.signal_config_payload = copy.deepcopy(signal_config_payload)
            self.indicator_params = copy.deepcopy(indicator_params)
            self.signal_params = copy.deepcopy(signal_params)
        logger.info(f"Updated config for automated signals worker {self.symbol} {self.timeframe}")
