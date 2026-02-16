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
from chart_auto_refresh import ChartAutoRefreshWorker, ensure_chart_store
from automated_signals_worker import AutomatedSignalsWorker

logger = logging.getLogger(__name__)


class ChartWorkerManager:
    """
    Manages chart auto-refresh worker.
    Enforces single worker per symbol/timeframe combination.
    """
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: Optional[ChartAutoRefreshWorker] = None
        self._update_bus: Optional[UpdateBus] = None
        self._current_symbol: Optional[str] = None
        self._current_timeframe: Optional[str] = None

    @property
    def current_symbol(self) -> Optional[str]:
        with self._lock:
            return self._current_symbol
            
    @property
    def current_timeframe(self) -> Optional[str]:
        with self._lock:
            return self._current_timeframe
    
    def start_new(
        self,
        symbol: str,
        timeframe: str,
        update_bus: UpdateBus,
        num_bars: int = 200,
        use_websocket: bool = True,
    ) -> bool:
        """Start a new chart worker (stops existing worker if any).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            update_bus: UpdateBus for worker communication
            num_bars: Number of bars to fetch
            use_websocket: Enable WebSocket updates

        Returns:
            True if worker started successfully, False otherwise
        """
        with self._lock:
            self.stop()
            
            self._current_symbol = symbol
            self._current_timeframe = timeframe
            self._update_bus = update_bus
            
            try:
                self._worker = ChartAutoRefreshWorker(
                    symbol=symbol,
                    timeframe=timeframe,
                    update_bus=update_bus,
                )
                self._worker.start()
                logger.info(f"Started chart worker for {symbol} {timeframe}")
                return True
            except Exception as exc:
                logger.error(f"Failed to start chart worker: {exc}", exc_info=True)
                return False
    
    def stop(self) -> None:
        """Stop the current worker."""
        with self._lock:
            if self._worker is not None:
                try:
                    self._worker.stop()
                except Exception as exc:
                    logger.error(f"Error stopping worker: {exc}")
                self._worker = None
            
            self._current_symbol = None
            self._current_timeframe = None
    
    def is_running(self) -> bool:
        """Check if worker is running."""
        with self._lock:
            if self._worker is not None:
                return self._worker.is_running()
            return False

    def poll_and_apply(self, session_state: Any) -> bool:
        """Poll for updates and apply to session state (called from main thread)."""
        if self._update_bus is None:
            return False
        
        updates = self._update_bus.drain(max_updates=100)
        if not updates:
            return False
        
        applied = False
        for update in updates:
            update_type = update.get("type")
            
            if update_type == "chart_closed_kline":
                self._apply_closed_kline(session_state, update)
                applied = True
            elif update_type == "chart_forming_kline":
                self._apply_forming_kline(session_state, update)
                applied = True
            elif update_type == "chart_error":
                logger.error(f"Chart worker error: {update.get('error')}")
            elif update_type == "chart_connect":
                logger.info("Chart WebSocket connected")
            elif update_type == "chart_disconnect":
                logger.warning("Chart WebSocket disconnected")
        
        return applied
    
    def _apply_closed_kline(self, session_state: Any, update: Dict[str, Any]) -> None:
        try:
            from chart_auto_refresh import ensure_chart_store
            
            df = update.get("df")
            last_closed_close_ms = update.get("last_closed_close_ms")
            
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return
            
            store = ensure_chart_store(session_state)
            store.update_closed(df, last_closed_close_ms, append=True)
        except Exception as exc:
            logger.error(f"Error applying closed kline: {exc}", exc_info=True)
    
    def _apply_forming_kline(self, session_state: Any, update: Dict[str, Any]) -> None:
        try:
            from chart_auto_refresh import ensure_chart_store
            
            df = update.get("df")
            if df is None or not isinstance(df, pd.DataFrame):
                return
            
            store = ensure_chart_store(session_state)
            store.set_forming_bar(df if not df.empty else None)
        except Exception as exc:
            logger.error(f"Error applying forming kline: {exc}", exc_info=True)


class SignalsWorkerManager:
    """Manages automated signals worker."""
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._worker: Optional[AutomatedSignalsWorker] = None
        self._update_bus: Optional[UpdateBus] = None
        self._current_symbol: Optional[str] = None
        self._current_timeframe: Optional[str] = None

    @property
    def current_symbol(self) -> Optional[str]:
        with self._lock:
            return self._current_symbol
            
    @property
    def current_timeframe(self) -> Optional[str]:
        with self._lock:
            return self._current_timeframe
    
    def start_new(
        self,
        symbol: str,
        timeframe: str,
        update_bus: UpdateBus,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
        signal_executor: Optional[Any] = None,
        use_websocket: bool = True,
    ) -> bool:
        """Start a new signals worker."""
        with self._lock:
            self.stop()

            self._current_symbol = symbol
            self._current_timeframe = timeframe
            self._update_bus = update_bus

            try:
                self._worker = AutomatedSignalsWorker(
                    symbol=symbol,
                    timeframe=timeframe,
                    update_bus=update_bus,
                    signal_config_payload=signal_config_payload,
                    indicator_params=indicator_params,
                    signal_params=signal_params,
                    signal_executor=signal_executor
                )
                self._worker.start()
                logger.info(f"Started signals worker for {symbol} {timeframe}")
                return True
            except Exception as exc:
                logger.error(f"Failed to start signals worker: {exc}", exc_info=True)
                return False
    
    def stop(self) -> None:
        with self._lock:
            if self._worker is not None:
                try:
                    self._worker.stop()
                except Exception as exc:
                    logger.error(f"Error stopping worker: {exc}")
                self._worker = None
            
            self._current_symbol = None
            self._current_timeframe = None

    def is_running(self) -> bool:
        with self._lock:
            if self._worker is not None:
                return self._worker.is_running()
            return False
            
    def poll_and_apply(self, session_state: Any) -> bool:
        """Poll for updates and apply to session state."""
        if self._update_bus is None:
            return False
        
        updates = self._update_bus.drain(max_updates=100)
        if not updates:
            return False
        
        applied = False
        for update in updates:
            update_type = update.get("type")
            
            if update_type == "signals_update":
                self._apply_signals_update(session_state, update)
                applied = True
            elif update_type == "signals_error":
                 logger.error(f"Signals worker error: {update.get('error')}")
            elif update_type == "signals_connect":
                 logger.info("Signals WebSocket connected")
            elif update_type == "signals_disconnect":
                 logger.warning("Signals WebSocket disconnected")
            elif update_type == "EXECUTION_UPDATE":
                 self._apply_execution_update(session_state, update)
                 applied = True
        
        return applied

    def _apply_signals_update(self, session_state: Any, update: Dict[str, Any]) -> None:
        result = update.get("result", {})
        auto_end_time_ms = update.get("auto_end_time_ms")
        
        state = getattr(session_state, "automated_signals_state", {})
        
        final_indicator_params = (
            result.get("explicit_signal", {})
            .get("metadata", {})
            .get("indicator_params")
            or result.get("processed_payload", {})
            .get("metadata", {})
            .get("indicator_params")
            or {}
        )

        state.update({
            "result": result,
            "error": None,
            "candles": result.get("candles", []),
            "processed_payload": result.get("processed_payload"),
            "explicit_signal": result.get("explicit_signal"),
            "indicator_params": final_indicator_params,
            "analysis_updated": True,
            "fetch_needed": False,
        })
        
        if auto_end_time_ms:
            state["auto_end_time_ms"] = auto_end_time_ms
            
        setattr(session_state, "automated_signals_state", state)

    def _apply_execution_update(self, session_state: Any, update: Dict[str, Any]) -> None:
        log = getattr(session_state, "execution_log", [])
        if not isinstance(log, list):
            log = []
        log.append(update)
        setattr(session_state, "execution_log", log)
