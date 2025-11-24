"""
WorkerManager classes for managing background worker lifecycle.

Enforces single worker per feature, handles clean start/stop with join,
and provides polling interface for main thread integration.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import pandas as pd

from update_bus import UpdateBus
from websocket_client import BinanceWebSocketClient

logger = logging.getLogger(__name__)


class ChartWorkerManager:
    """
    Manages chart auto-refresh worker with WebSocket integration.
    
    Enforces single worker per symbol/timeframe combination.
    """
    
    def __init__(self) -> None:
        """Initialize chart worker manager."""
        self._lock = threading.RLock()
        self._ws_client: Optional[BinanceWebSocketClient] = None
        self._update_bus: Optional[UpdateBus] = None
        self._current_symbol: Optional[str] = None
        self._current_timeframe: Optional[str] = None
        self._use_websocket = True
        self._fallback_worker: Optional[Any] = None  # For REST polling fallback
    
    def start_new(
        self,
        symbol: str,
        timeframe: str,
        update_bus: UpdateBus,
        use_websocket: bool = True,
        session_state: Optional[Any] = None,
        num_bars: int = 200,
    ) -> bool:
        """
        Start a new chart worker (stops existing worker if any).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            update_bus: UpdateBus instance for publishing updates
            use_websocket: Whether to use WebSocket (True) or REST polling (False)
            session_state: Streamlit session state (for fallback polling)
            num_bars: Number of bars to fetch (for initial load)
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            # Stop existing worker
            self.stop()
            
            self._current_symbol = symbol
            self._current_timeframe = timeframe
            self._update_bus = update_bus
            self._use_websocket = use_websocket
            
            if use_websocket:
                try:
                    self._ws_client = BinanceWebSocketClient(
                        symbol=symbol,
                        timeframe=timeframe,
                        on_closed_kline=self._on_closed_kline,
                        on_forming_kline=self._on_forming_kline,
                        on_error=self._on_error,
                        on_connect=self._on_connect,
                        on_disconnect=self._on_disconnect,
                        backfill_bars=3,
                    )
                    self._ws_client.start()
                    logger.info(f"Started WebSocket chart worker for {symbol} {timeframe}")
                    return True
                except Exception as exc:
                    logger.error(f"Failed to start WebSocket chart worker: {exc}", exc_info=True)
                    # Fall back to REST polling
                    if session_state is not None:
                        logger.info("Falling back to REST polling for chart updates")
                        return self._start_fallback_worker(symbol, timeframe, num_bars, session_state)
                    return False
            else:
                # Use REST polling
                if session_state is not None:
                    return self._start_fallback_worker(symbol, timeframe, num_bars, session_state)
                else:
                    logger.error("Cannot start REST polling without session_state")
                    return False
    
    def _start_fallback_worker(
        self,
        symbol: str,
        timeframe: str,
        num_bars: int,
        session_state: Any,
    ) -> bool:
        """Start REST polling fallback worker."""
        try:
            from chart_auto_refresh import ChartAutoRefreshWorker
            
            self._fallback_worker = ChartAutoRefreshWorker(
                symbol=symbol,
                timeframe=timeframe,
                num_bars=num_bars,
                session_state=session_state,
            )
            self._fallback_worker.start()
            logger.info(f"Started REST polling chart worker for {symbol} {timeframe}")
            return True
        except Exception as exc:
            logger.error(f"Failed to start REST polling chart worker: {exc}", exc_info=True)
            return False
    
    def stop(self) -> None:
        """Stop the current worker."""
        with self._lock:
            if self._ws_client is not None:
                try:
                    self._ws_client.stop()
                except Exception as exc:
                    logger.error(f"Error stopping WebSocket client: {exc}")
                self._ws_client = None
            
            if self._fallback_worker is not None:
                try:
                    self._fallback_worker.stop()
                except Exception as exc:
                    logger.error(f"Error stopping fallback worker: {exc}")
                self._fallback_worker = None
            
            self._current_symbol = None
            self._current_timeframe = None
    
    def poll_and_apply(self, session_state: Any) -> bool:
        """
        Poll for updates and apply to session state (called from main thread).
        
        Args:
            session_state: Streamlit session state
            
        Returns:
            True if updates were applied, False otherwise
        """
        if self._update_bus is None:
            return False
        
        updates = self._update_bus.drain(max_updates=100)
        
        if not updates:
            return False
        
        # Apply updates to session state
        applied = False
        
        for update in updates:
            update_type = update.get("type")
            
            if update_type == "chart_closed_kline":
                # Apply closed kline update
                self._apply_closed_kline(session_state, update)
                applied = True
            
            elif update_type == "chart_forming_kline":
                # Apply forming kline update
                self._apply_forming_kline(session_state, update)
                applied = True
            
            elif update_type == "chart_error":
                # Handle error
                logger.error(f"Chart worker error: {update.get('error')}")
            
            elif update_type == "chart_connect":
                # Connection established
                logger.info("Chart WebSocket connected")
            
            elif update_type == "chart_disconnect":
                # Connection lost
                logger.warning("Chart WebSocket disconnected")
        
        return applied
    
    def _apply_closed_kline(self, session_state: Any, update: Dict[str, Any]) -> None:
        """Apply closed kline update to session state."""
        try:
            from chart_auto_refresh import ensure_chart_store
            
            df = update.get("df")
            last_closed_close_ms = update.get("last_closed_close_ms")
            
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return
            
            store = ensure_chart_store(session_state)
            
            # Update closed bars (with append=True for delta updates)
            appended, deduped, total = store.update_closed(
                df,
                last_closed_close_ms,
                append=True,
            )
            
            logger.debug(
                f"Applied closed kline: appended={appended}, deduped={deduped}, total={total}"
            )
        
        except Exception as exc:
            logger.error(f"Error applying closed kline: {exc}", exc_info=True)
    
    def _apply_forming_kline(self, session_state: Any, update: Dict[str, Any]) -> None:
        """Apply forming kline update to session state."""
        try:
            from chart_auto_refresh import ensure_chart_store
            
            df = update.get("df")
            
            if df is None or not isinstance(df, pd.DataFrame):
                return
            
            store = ensure_chart_store(session_state)
            
            # Update forming bar
            store.set_forming_bar(df if not df.empty else None)
            
            logger.debug("Applied forming kline update")
        
        except Exception as exc:
            logger.error(f"Error applying forming kline: {exc}", exc_info=True)
    
    def _on_closed_kline(self, df: pd.DataFrame) -> None:
        """Callback for closed kline events (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        # Calculate last_closed_close_ms from df
        if df.empty:
            return
        
        open_time_ms = int(df["ts"].iloc[0])
        close_time_ms = open_time_ms + self._get_tf_ms()
        
        self._update_bus.publish({
            "type": "chart_closed_kline",
            "df": df,
            "last_closed_close_ms": close_time_ms,
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_forming_kline(self, df: pd.DataFrame) -> None:
        """Callback for forming kline updates (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "chart_forming_kline",
            "df": df,
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_error(self, error: Exception) -> None:
        """Callback for WebSocket errors (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "chart_error",
            "error": str(error),
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_connect(self) -> None:
        """Callback for WebSocket connection (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "chart_connect",
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_disconnect(self) -> None:
        """Callback for WebSocket disconnection (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "chart_disconnect",
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _get_tf_ms(self) -> int:
        """Get timeframe in milliseconds."""
        from chart_auto_refresh import TIMEFRAME_TO_MS
        
        return TIMEFRAME_TO_MS.get(self._current_timeframe, 3_600_000)
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        with self._lock:
            if self._ws_client is not None:
                return self._ws_client.is_connected()
            if self._fallback_worker is not None:
                return getattr(self._fallback_worker, "_thread", None) is not None
            return False


class SignalsWorkerManager:
    """
    Manages automated signals worker with WebSocket integration.
    
    Enforces single worker per symbol/timeframe combination.
    """
    
    def __init__(self) -> None:
        """Initialize signals worker manager."""
        self._lock = threading.RLock()
        self._ws_client: Optional[BinanceWebSocketClient] = None
        self._update_bus: Optional[UpdateBus] = None
        self._current_symbol: Optional[str] = None
        self._current_timeframe: Optional[str] = None
        self._use_websocket = True
        self._fallback_worker: Optional[Any] = None  # For REST polling fallback
        self._signal_config_payload: Optional[Dict[str, Any]] = None
        self._indicator_params: Optional[Dict[str, Any]] = None
        self._signal_params: Optional[Dict[str, Any]] = None
    
    def start_new(
        self,
        symbol: str,
        timeframe: str,
        update_bus: UpdateBus,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
        use_websocket: bool = True,
        session_state: Optional[Any] = None,
    ) -> bool:
        """
        Start a new signals worker (stops existing worker if any).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            update_bus: UpdateBus instance for publishing updates
            signal_config_payload: Signal configuration payload
            indicator_params: Indicator parameters
            signal_params: Signal parameters
            use_websocket: Whether to use WebSocket (True) or REST polling (False)
            session_state: Streamlit session state (for fallback polling)
            
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            # Stop existing worker
            self.stop()
            
            self._current_symbol = symbol
            self._current_timeframe = timeframe
            self._update_bus = update_bus
            self._use_websocket = use_websocket
            self._signal_config_payload = signal_config_payload
            self._indicator_params = indicator_params
            self._signal_params = signal_params
            
            if use_websocket:
                try:
                    self._ws_client = BinanceWebSocketClient(
                        symbol=symbol,
                        timeframe=timeframe,
                        on_closed_kline=self._on_closed_kline,
                        on_forming_kline=None,  # Signals only care about closed klines
                        on_error=self._on_error,
                        on_connect=self._on_connect,
                        on_disconnect=self._on_disconnect,
                        backfill_bars=0,  # Don't backfill for signals (we'll fetch full history separately)
                    )
                    self._ws_client.start()
                    logger.info(f"Started WebSocket signals worker for {symbol} {timeframe}")
                    return True
                except Exception as exc:
                    logger.error(f"Failed to start WebSocket signals worker: {exc}", exc_info=True)
                    # Fall back to REST polling
                    if session_state is not None:
                        logger.info("Falling back to REST polling for signals")
                        return self._start_fallback_worker(
                            symbol,
                            timeframe,
                            session_state,
                            signal_config_payload,
                            indicator_params,
                            signal_params,
                        )
                    return False
            else:
                # Use REST polling
                if session_state is not None:
                    return self._start_fallback_worker(
                        symbol,
                        timeframe,
                        session_state,
                        signal_config_payload,
                        indicator_params,
                        signal_params,
                    )
                else:
                    logger.error("Cannot start REST polling without session_state")
                    return False
    
    def _start_fallback_worker(
        self,
        symbol: str,
        timeframe: str,
        session_state: Any,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
    ) -> bool:
        """Start REST polling fallback worker."""
        try:
            from automated_signals_worker import AutomatedSignalsWorker
            
            self._fallback_worker = AutomatedSignalsWorker(
                symbol=symbol,
                timeframe=timeframe,
                session_state=session_state,
                signal_config_payload=signal_config_payload,
                indicator_params=indicator_params,
                signal_params=signal_params,
            )
            self._fallback_worker.start()
            logger.info(f"Started REST polling signals worker for {symbol} {timeframe}")
            return True
        except Exception as exc:
            logger.error(f"Failed to start REST polling signals worker: {exc}", exc_info=True)
            return False
    
    def stop(self) -> None:
        """Stop the current worker."""
        with self._lock:
            if self._ws_client is not None:
                try:
                    self._ws_client.stop()
                except Exception as exc:
                    logger.error(f"Error stopping WebSocket client: {exc}")
                self._ws_client = None
            
            if self._fallback_worker is not None:
                try:
                    self._fallback_worker.stop()
                except Exception as exc:
                    logger.error(f"Error stopping fallback worker: {exc}")
                self._fallback_worker = None
            
            self._current_symbol = None
            self._current_timeframe = None
    
    def poll_and_apply(self, session_state: Any) -> bool:
        """
        Poll for updates and apply to session state (called from main thread).
        
        Args:
            session_state: Streamlit session state
            
        Returns:
            True if updates were applied, False otherwise
        """
        if self._update_bus is None:
            return False
        
        updates = self._update_bus.drain(max_updates=100)
        
        if not updates:
            return False
        
        # Apply updates to session state
        applied = False
        
        for update in updates:
            update_type = update.get("type")
            
            if update_type == "signals_closed_kline":
                # Trigger signal refresh on closed kline
                self._trigger_signal_refresh(session_state, update)
                applied = True
            
            elif update_type == "EXECUTION_UPDATE":
                # Execution occurred, trigger UI refresh
                applied = True
            
            elif update_type == "signals_result":
                # Apply signal result
                self._apply_signal_result(session_state, update)
                applied = True
            
            elif update_type == "signals_error":
                # Handle error
                logger.error(f"Signals worker error: {update.get('error')}")
                # Store error in session state
                state = getattr(session_state, "automated_signals_state", {})
                state["error"] = update.get("error")
                session_state.automated_signals_state = state
                applied = True
            
            elif update_type == "signals_connect":
                # Connection established
                logger.info("Signals WebSocket connected")
            
            elif update_type == "signals_disconnect":
                # Connection lost
                logger.warning("Signals WebSocket disconnected")
        
        return applied
    
    def _trigger_signal_refresh(self, session_state: Any, update: Dict[str, Any]) -> None:
        """Trigger signal refresh on closed kline."""
        try:
            import datetime as dt
            from chart_auto_refresh import TIMEFRAME_TO_MS
            from indicator_collector.trading_system.automated_signals import run_automated_signal_flow
            from indicator_collector.trading_system.signal_generator import SignalConfig
            
            df = update.get("df")
            
            if df is None or df.empty:
                return
            
            # Calculate end time from closed kline
            open_time_ms = int(df["ts"].iloc[0])
            tf_ms = TIMEFRAME_TO_MS.get(self._current_timeframe, 3_600_000)
            end_time_ms = open_time_ms + tf_ms
            
            logger.info(
                f"Triggering signal refresh for {self._current_symbol} {self._current_timeframe} "
                f"at end_time_ms={end_time_ms}"
            )
            
            # Get start time from session state (or default to 200 bars back)
            state = getattr(session_state, "automated_signals_state", {})
            start_dt_str = state.get("start_datetime_iso")
            
            end_dt = dt.datetime.fromtimestamp(end_time_ms / 1000, tz=dt.timezone.utc)
            
            if start_dt_str:
                start_dt = dt.datetime.fromisoformat(start_dt_str)
            else:
                start_dt = end_dt - dt.timedelta(milliseconds=tf_ms * 200)
            
            # Build signal config
            weights = self._signal_config_payload.get("weights", {})
            signal_config = SignalConfig(
                technical_weight=weights.get("technical", 0.25),
                sentiment_weight=weights.get("sentiment", 0.15),
                multitimeframe_weight=weights.get("multitimeframe", 0.10),
                volume_weight=weights.get("volume", 0.20),
                structure_weight=weights.get("market_structure", 0.15),
                composite_weight=weights.get("composite", 0.0),
                min_factors_confirm=int(self._signal_config_payload.get("min_confirmations", 3)),
                buy_threshold=float(self._signal_config_payload.get("buy_threshold", 0.65)),
                sell_threshold=float(self._signal_config_payload.get("sell_threshold", 0.35)),
                min_confidence=float(self._signal_config_payload.get("min_confidence", 0.6)),
            )
            
            # Calculate minimum candles
            atr_period = int(self._indicator_params.get("atr", {}).get("period", 14))
            macd_slow = int(self._indicator_params.get("macd", {}).get("slow", 26))
            macd_signal = int(self._indicator_params.get("macd", {}).get("signal", 9))
            rsi_period = int(self._indicator_params.get("rsi", {}).get("period", 14))
            min_candles = max(30, rsi_period + 2, atr_period + 2, macd_slow + macd_signal)
            
            # Run signal flow in background thread to avoid blocking
            def _run_signal_flow() -> None:
                try:
                    result = run_automated_signal_flow(
                        self._current_symbol,
                        self._current_timeframe,
                        start_dt,
                        end_dt,
                        validate_real_data=True,
                        signal_config=signal_config,
                        indicator_params=self._indicator_params,
                        signal_params=self._signal_params,
                        min_candles=min_candles,
                    )
                    
                    # Publish result
                    if self._update_bus:
                        self._update_bus.publish({
                            "type": "signals_result",
                            "candles": result.candles,
                            "processed_payload": result.processed_payload,
                            "explicit_signal": result.explicit_signal,
                            "end_time_ms": end_time_ms,
                        })
                except Exception as exc:
                    logger.error(f"Signal flow failed: {exc}", exc_info=True)
                    if self._update_bus:
                        self._update_bus.publish({
                            "type": "signals_error",
                            "error": str(exc),
                        })
            
            # Run in background thread
            thread = threading.Thread(target=_run_signal_flow, daemon=True)
            thread.start()
        
        except Exception as exc:
            logger.error(f"Error triggering signal refresh: {exc}", exc_info=True)
    
    def _apply_signal_result(self, session_state: Any, update: Dict[str, Any]) -> None:
        """Apply signal result to session state."""
        try:
            result_dict = {
                "candles": update.get("candles", []),
                "processed_payload": update.get("processed_payload"),
                "explicit_signal": update.get("explicit_signal"),
            }
            
            final_indicator_params = (
                result_dict.get("explicit_signal", {})
                .get("metadata", {})
                .get("indicator_params")
                or result_dict.get("processed_payload", {})
                .get("metadata", {})
                .get("indicator_params")
                or self._indicator_params
            )
            
            state = getattr(session_state, "automated_signals_state", {})
            state.update({
                "result": result_dict,
                "error": None,
                "candles": result_dict.get("candles", []),
                "processed_payload": result_dict.get("processed_payload"),
                "explicit_signal": result_dict.get("explicit_signal"),
                "indicator_params": final_indicator_params,
                "analysis_updated": True,
                "fetch_needed": False,
                "auto_end_time_ms": update.get("end_time_ms", 0),
            })
            session_state.automated_signals_state = state
            
            logger.info(
                f"Applied signal result: {len(result_dict.get('candles', []))} candles, "
                f"signal={result_dict.get('explicit_signal', {}).get('signal', 'UNKNOWN')}"
            )
        
        except Exception as exc:
            logger.error(f"Error applying signal result: {exc}", exc_info=True)
    
    def _on_closed_kline(self, df: pd.DataFrame) -> None:
        """Callback for closed kline events (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "signals_closed_kline",
            "df": df,
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_error(self, error: Exception) -> None:
        """Callback for WebSocket errors (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "signals_error",
            "error": str(error),
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_connect(self) -> None:
        """Callback for WebSocket connection (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "signals_connect",
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def _on_disconnect(self) -> None:
        """Callback for WebSocket disconnection (called from WebSocket thread)."""
        if self._update_bus is None:
            return
        
        self._update_bus.publish({
            "type": "signals_disconnect",
            "symbol": self._current_symbol,
            "timeframe": self._current_timeframe,
        })
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        with self._lock:
            if self._ws_client is not None:
                return self._ws_client.is_connected()
            if self._fallback_worker is not None:
                return getattr(self._fallback_worker, "_thread", None) is not None
            return False
