"""
Automated signals auto-refresh worker for the Automated Signals tab.

This worker automatically advances the End time to the latest closed TF boundary
and triggers immediate data refresh and signal recomputation without user interaction.

TIMESTAMP SEMANTICS:
-------------------
All internal timestamps are in UTC milliseconds.

- auto_end_time_ms: close_time of the last closed bar
- Worker listens to WebSocket closed bar events
- When detected, immediately appends data and recomputes signals
- Publishes updates via UpdateBus
"""

from __future__ import annotations

import copy
import datetime as dt
import logging
import threading
import time
from datetime import timezone
from typing import Any, Dict, Optional

import pandas as pd

from timeframe_utils import TIMEFRAME_TO_MS, map_tf_to_ms
from indicator_collector.trading_system.auto_analyze_worker import get_binance_server_time_ms
from indicator_collector.trading_system.automated_signals import run_automated_signal_flow
from indicator_collector.trading_system.data_sources.binance_source import BinanceKlinesSource
from indicator_collector.trading_system.signal_generator import SignalConfig
from update_bus import UpdateBus
from websocket_client import BinanceWebSocketClient
from logging_config import get_structured_logger

# Optional metrics import
try:
    from metrics import worker_starts, worker_stops, worker_errors, worker_processing_time
    from metrics.collectors import get_worker_collector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = get_structured_logger(__name__)


class AutomatedSignalsWorker:
    """WebSocket-based worker that auto-advances End time and refreshes signals on new closed bars."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        update_bus: UpdateBus,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
        signal_executor: Optional[Any] = None,
    ):
        """
        Initialize the automated signals worker.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            update_bus: UpdateBus instance
            signal_config_payload: Signal configuration payload
            indicator_params: Indicator parameters
            signal_params: Signal parameters
            signal_executor: Optional SignalExecutor instance
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.update_bus = update_bus
        self.signal_config_payload = signal_config_payload
        self.indicator_params = indicator_params
        self.signal_params = signal_params
        self.signal_executor = signal_executor
        
        self.data_source = BinanceKlinesSource()
        self.ws_client: Optional[BinanceWebSocketClient] = None
        self.df = pd.DataFrame()
        
        # Get timeframe interval in milliseconds
        self.tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)

    def start(self) -> None:
        """Start the worker (initial fetch + WebSocket)."""
        if self.ws_client is not None:
            return
        
        logger.info(
            "Starting automated signals worker",
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        # Record metrics
        if METRICS_AVAILABLE:
            worker_starts.labels(
                worker_type="automated_signals",
                symbol=self.symbol,
                timeframe=self.timeframe
            ).inc()
            
            collector = get_worker_collector()
            collector.record_start("automated_signals", self.symbol, self.timeframe)
        
        # Initial synchronous fetch to bootstrap history
        try:
            logger.info(
                "Fetching initial history",
                symbol=self.symbol,
                timeframe=self.timeframe
            )
            server_time_ms = get_binance_server_time_ms(self.data_source)
            end_dt = dt.datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc)
            # Fetch enough bars for indicators (e.g. 500)
            start_dt = end_dt - dt.timedelta(milliseconds=self.tf_ms * 500)
            
            self.df = self.data_source.load_candles(
                self.symbol,
                self.timeframe,
                start_dt,
                end_dt
            )
            
            # Run initial analysis
            if not self.df.empty:
                self._refresh_signals()
                
        except Exception as e:
            logger.error(
                "Initial fetch failed",
                symbol=self.symbol,
                error=str(e)
            )
            
            if METRICS_AVAILABLE:
                worker_errors.labels(
                    worker_type="automated_signals",
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    error_type="initial_fetch"
                ).inc()
            
            # Continue anyway, WebSocket might fill gaps eventually or we retry later? 
            # For now, we proceed to start WS.

        self.ws_client = BinanceWebSocketClient(
            symbol=self.symbol,
            interval=self.timeframe,
            on_closed_bar=self._on_closed_kline,
            on_forming_bar=None,  # Signals only care about closed klines
        )
        self.ws_client.start()
        
        logger.info(
            "Automated signals worker started",
            symbol=self.symbol,
            timeframe=self.timeframe
        )

    def stop(self) -> None:
        """Stop the worker."""
        logger.info(
            "Stopping automated signals worker",
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        if self.ws_client:
            self.ws_client.stop()
            self.ws_client = None
        
        # Record metrics
        if METRICS_AVAILABLE:
            worker_stops.labels(
                worker_type="automated_signals",
                symbol=self.symbol,
                timeframe=self.timeframe,
                reason="normal"
            ).inc()
            
            collector = get_worker_collector()
            collector.record_stop("automated_signals", self.symbol, self.timeframe, "normal")
        
        logger.info(
            "Automated signals worker stopped",
            symbol=self.symbol,
            timeframe=self.timeframe
        )

    def _on_closed_kline(self, kline: Dict) -> None:
        """Callback for closed kline events."""
        df = pd.DataFrame([kline])
        if df.empty:
            return
        
        # Append to internal DataFrame
        if self.df.empty:
            self.df = df
        else:
            self.df = pd.concat([self.df, df], ignore_index=True)
            self.df = (
                self.df.drop_duplicates(subset="ts", keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )
            
        # Trim DataFrame to keep memory usage stable (e.g. keep last 1000 bars)
        if len(self.df) > 1000:
            self.df = self.df.iloc[-1000:].reset_index(drop=True)

        # Recompute signals
        try:
            self._refresh_signals()
        except Exception as exc:
            logger.error(f"Failed to refresh signals: {exc}", exc_info=True)
            self.update_bus.publish({
                "type": "signals_error",
                "error": str(exc),
                "symbol": self.symbol,
                "timeframe": self.timeframe
            })

    def _refresh_signals(self) -> None:
        """Recompute signals using internal DataFrame."""
        if self.df.empty:
            return
        
        start_time = time.time()
        
        last_ts = int(self.df["ts"].iloc[-1])
        end_dt = dt.datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        # We pass the full range of our dataframe
        start_ts = int(self.df["ts"].iloc[0])
        start_dt = dt.datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc)

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

        # Validate we have enough data before running signal flow
        available_candles = len(self.df) if self.df is not None else 0
        if available_candles < min_candles:
            logger.warning(
                "Insufficient candles for signal generation",
                symbol=self.symbol,
                timeframe=self.timeframe,
                available=available_candles,
                required=min_candles
            )
            
            if METRICS_AVAILABLE:
                worker_errors.labels(
                    worker_type="automated_signals",
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    error_type="insufficient_data"
                ).inc()
            
            return

        try:
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
                preloaded_df=self.df,
            )
            
            processing_time = time.time() - start_time
            
            # Record metrics
            if METRICS_AVAILABLE:
                worker_processing_time.labels(
                    worker_type="automated_signals",
                    symbol=self.symbol,
                    timeframe=self.timeframe
                ).observe(processing_time)
            
            logger.debug(
                "Signal refresh completed",
                symbol=self.symbol,
                timeframe=self.timeframe,
                processing_time_ms=round(processing_time * 1000, 2)
            )

            # Prepare result dict
            result_dict = {
                "candles": result.candles,
                "processed_payload": result.processed_payload,
                "explicit_signal": result.explicit_signal,
            }
            
            # Publish update
            # Note: last_ts is the open_time of the last bar. The "end_time" of the analysis is effectively close_time = last_ts + tf_ms
            last_closed_close_ms = last_ts + self.tf_ms
            
            self.update_bus.publish({
                "type": "signals_update",
                "result": result_dict,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "auto_end_time_ms": last_closed_close_ms
            })

            # Execute if enabled
            if self.signal_executor and self.signal_executor.enabled:
                self._execute_signal(result.explicit_signal, last_closed_close_ms)
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(
                "Failed to refresh signals",
                symbol=self.symbol,
                timeframe=self.timeframe,
                error=str(e),
                processing_time_ms=round(processing_time * 1000, 2)
            )
            
            if METRICS_AVAILABLE:
                worker_errors.labels(
                    worker_type="automated_signals",
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    error_type="signal_generation"
                ).inc()
            
            raise

    def _execute_signal(self, explicit_signal: Dict[str, Any], generated_at_ms: int) -> None:
        """Execute signal via executor."""
        try:
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
                    "signal_id": f"{self.symbol}_{generated_at_ms}",
                    "symbol": self.symbol,
                    "direction": "LONG" if signal_type == "BUY" else "SHORT",
                    "entry_price": entry_price,
                    "take_profit": tp_price,
                    "stop_loss": stop_loss,
                    "leverage": 5,
                    "quantity": 0.001,
                    "generated_at": generated_at_ms
                }
                
                self.signal_executor.execute_signal(payload)
                
        except Exception as exc:
             logger.error(f"Failed to execute signal: {exc}", exc_info=True)

    def update_config(
        self,
        signal_config_payload: Dict[str, Any],
        indicator_params: Dict[str, Any],
        signal_params: Dict[str, Any],
    ) -> None:
        """Update configuration."""
        self.signal_config_payload = copy.deepcopy(signal_config_payload)
        self.indicator_params = copy.deepcopy(indicator_params)
        self.signal_params = copy.deepcopy(signal_params)
        logger.info(f"Updated config for automated signals worker {self.symbol} {self.timeframe}")
        
        # Trigger refresh with new config
        self._refresh_signals()

    def _on_error(self, error: Exception) -> None:
        self.update_bus.publish({
            "type": "signals_error",
            "error": str(error),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def _on_connect(self) -> None:
        self.update_bus.publish({
            "type": "signals_connect",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })

    def _on_disconnect(self) -> None:
        self.update_bus.publish({
            "type": "signals_disconnect",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        })
    
    def is_running(self) -> bool:
        return self.ws_client is not None
