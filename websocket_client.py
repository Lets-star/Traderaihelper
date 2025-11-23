"""
Binance WebSocket client for real-time kline streams.

Supports:
- Multiple timeframes (1m, 5m, 15m, 1h, 3h, etc.)
- Automatic reconnect with exponential backoff and jitter
- Backfill on connect/reconnect for continuity
- Thread-safe event handling
- Forming and closed bar detection
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
import websocket
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

import pandas as pd

from chart_auto_refresh import TIMEFRAME_TO_MS
from indicator_collector.trading_system.data_sources.binance_source import BinanceKlinesSource

logger = logging.getLogger(__name__)

# Binance WebSocket endpoints
DEFAULT_WS_BASE_URL = "wss://stream.binance.com:9443"
WS_STREAM_PATH = "/ws"

# Reconnect configuration
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_MAX_BACKOFF = 60.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_JITTER_FACTOR = 0.25
DEFAULT_MAX_RECONNECT_ATTEMPTS = None  # None = infinite

# Heartbeat configuration
DEFAULT_HEARTBEAT_INTERVAL = 30.0  # seconds
DEFAULT_HEARTBEAT_TIMEOUT = 10.0  # seconds

# Timeframe mapping
TIMEFRAME_TO_BINANCE_INTERVAL: Dict[str, str] = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
}


class BinanceWebSocketClient:
    """
    Binance WebSocket client for kline streams.
    
    Subscribes to /ws/<symbol>@kline_<interval> and publishes:
    - Forming bar updates (k.x == false)
    - Closed bar events (k.x == true)
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        on_closed_kline: Optional[Callable[[pd.DataFrame], None]] = None,
        on_forming_kline: Optional[Callable[[pd.DataFrame], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
        ws_base_url: str = DEFAULT_WS_BASE_URL,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
        jitter_factor: float = DEFAULT_JITTER_FACTOR,
        max_reconnect_attempts: Optional[int] = DEFAULT_MAX_RECONNECT_ATTEMPTS,
        heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
        heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT,
        backfill_bars: int = 3,
        data_source: Optional[BinanceKlinesSource] = None,
    ):
        """
        Initialize Binance WebSocket client.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe string (e.g., "1h", "3h")
            on_closed_kline: Callback for closed kline events (DataFrame with 1 row)
            on_forming_kline: Callback for forming kline updates (DataFrame with 1 row)
            on_error: Callback for errors
            on_connect: Callback for connection events
            on_disconnect: Callback for disconnection events
            ws_base_url: WebSocket base URL
            initial_backoff: Initial reconnect backoff in seconds
            max_backoff: Maximum reconnect backoff in seconds
            backoff_multiplier: Exponential backoff multiplier
            jitter_factor: Jitter factor for backoff (0-1)
            max_reconnect_attempts: Maximum reconnect attempts (None = infinite)
            heartbeat_interval: Heartbeat ping interval in seconds
            heartbeat_timeout: Heartbeat timeout in seconds
            backfill_bars: Number of bars to backfill on connect
            data_source: Optional BinanceKlinesSource instance
        """
        # Strip BINANCE: prefix if present
        if symbol.startswith("BINANCE:"):
            symbol = symbol[8:]
        
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.on_closed_kline = on_closed_kline
        self.on_forming_kline = on_forming_kline
        self.on_error = on_error
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.ws_base_url = ws_base_url
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_multiplier = backoff_multiplier
        self.jitter_factor = jitter_factor
        self.max_reconnect_attempts = max_reconnect_attempts
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.backfill_bars = backfill_bars
        self.data_source = data_source or BinanceKlinesSource()
        
        # WebSocket connection state
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._lock = threading.RLock()
        
        # Reconnect state
        self._reconnect_attempt = 0
        self._current_backoff = initial_backoff
        self._last_message_time = 0.0
        
        # Timeframe metadata
        self.tf_ms = TIMEFRAME_TO_MS.get(timeframe, 3_600_000)
        self.interval = TIMEFRAME_TO_BINANCE_INTERVAL.get(timeframe, timeframe)
        
        # Stream URL
        stream_name = f"{self.symbol.lower()}@kline_{self.interval}"
        self.ws_url = f"{self.ws_base_url}{WS_STREAM_PATH}/{stream_name}"
        
        logger.info(f"BinanceWebSocketClient initialized: {self.symbol} {self.timeframe} ({self.ws_url})")
    
    def start(self) -> None:
        """Start the WebSocket connection."""
        with self._lock:
            if self._ws_thread is not None and self._ws_thread.is_alive():
                logger.warning(f"WebSocket thread already running for {self.symbol} {self.timeframe}")
                return
            
            self._stop_event.clear()
            self._connected.clear()
            self._reconnect_attempt = 0
            self._current_backoff = self.initial_backoff
            
            self._ws_thread = threading.Thread(
                target=self._run_websocket,
                name=f"ws-{self.symbol}-{self.timeframe}",
                daemon=True,
            )
            self._ws_thread.start()
            
            logger.info(f"WebSocket client started for {self.symbol} {self.timeframe}")
    
    def stop(self) -> None:
        """Stop the WebSocket connection."""
        with self._lock:
            self._stop_event.set()
            self._connected.clear()
            
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception as exc:
                    logger.debug(f"Error closing WebSocket: {exc}")
                self._ws = None
            
            if self._ws_thread is not None:
                self._ws_thread.join(timeout=5.0)
                self._ws_thread = None
            
            if self._heartbeat_thread is not None:
                self._heartbeat_thread.join(timeout=2.0)
                self._heartbeat_thread = None
            
            logger.info(f"WebSocket client stopped for {self.symbol} {self.timeframe}")
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected.is_set()
    
    def _run_websocket(self) -> None:
        """Main WebSocket run loop with automatic reconnect."""
        while not self._stop_event.is_set():
            try:
                # Check reconnect attempts
                if (
                    self.max_reconnect_attempts is not None
                    and self._reconnect_attempt >= self.max_reconnect_attempts
                ):
                    logger.error(
                        f"Max reconnect attempts ({self.max_reconnect_attempts}) reached "
                        f"for {self.symbol} {self.timeframe}"
                    )
                    break
                
                # Apply backoff if reconnecting
                if self._reconnect_attempt > 0:
                    backoff = min(
                        self._current_backoff * (self.backoff_multiplier ** (self._reconnect_attempt - 1)),
                        self.max_backoff,
                    )
                    jitter = random.uniform(0, self.jitter_factor * backoff)
                    sleep_time = backoff + jitter
                    
                    logger.info(
                        f"Reconnect attempt {self._reconnect_attempt} for {self.symbol} {self.timeframe} "
                        f"in {sleep_time:.2f}s"
                    )
                    
                    if self._stop_event.wait(timeout=sleep_time):
                        break
                
                # Create WebSocket connection
                logger.info(f"Connecting to {self.ws_url}")
                
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                
                # Run WebSocket (blocks until connection closes)
                self._ws.run_forever(
                    ping_interval=self.heartbeat_interval,
                    ping_timeout=self.heartbeat_timeout,
                )
                
                # Connection closed
                if not self._stop_event.is_set():
                    self._reconnect_attempt += 1
                    logger.warning(
                        f"WebSocket closed for {self.symbol} {self.timeframe}, "
                        f"will reconnect (attempt {self._reconnect_attempt})"
                    )
            
            except Exception as exc:
                logger.error(f"WebSocket error for {self.symbol} {self.timeframe}: {exc}", exc_info=True)
                if not self._stop_event.is_set():
                    self._reconnect_attempt += 1
                    if self.on_error:
                        try:
                            self.on_error(exc)
                        except Exception as callback_exc:
                            logger.error(f"Error in on_error callback: {callback_exc}")
    
    def _on_open(self, ws: Any) -> None:
        """WebSocket open callback."""
        logger.info(f"WebSocket connected for {self.symbol} {self.timeframe}")
        self._connected.set()
        self._reconnect_attempt = 0
        self._last_message_time = time.monotonic()
        
        # Trigger backfill on connect
        try:
            self._backfill()
        except Exception as exc:
            logger.error(f"Backfill failed on connect: {exc}", exc_info=True)
        
        # Trigger on_connect callback
        if self.on_connect:
            try:
                self.on_connect()
            except Exception as exc:
                logger.error(f"Error in on_connect callback: {exc}")
    
    def _on_message(self, ws: Any, message: str) -> None:
        """WebSocket message callback."""
        self._last_message_time = time.monotonic()
        
        try:
            data = json.loads(message)
            
            # Validate kline data structure
            if "e" not in data or data["e"] != "kline":
                logger.debug(f"Ignoring non-kline message: {data.get('e')}")
                return
            
            if "k" not in data:
                logger.warning(f"Kline message missing 'k' field: {data}")
                return
            
            kline = data["k"]
            
            # Parse kline data
            df = self._parse_kline(kline)
            
            if df is None or df.empty:
                return
            
            # Check if kline is closed
            is_closed = kline.get("x", False)
            
            if is_closed:
                # Closed kline event
                logger.debug(
                    f"Closed kline: {self.symbol} {self.timeframe} @ "
                    f"{datetime.fromtimestamp(df['ts'].iloc[0] / 1000, tz=timezone.utc).isoformat()}"
                )
                if self.on_closed_kline:
                    try:
                        self.on_closed_kline(df)
                    except Exception as exc:
                        logger.error(f"Error in on_closed_kline callback: {exc}", exc_info=True)
            else:
                # Forming kline update
                logger.debug(
                    f"Forming kline: {self.symbol} {self.timeframe} @ "
                    f"{datetime.fromtimestamp(df['ts'].iloc[0] / 1000, tz=timezone.utc).isoformat()} "
                    f"close={df['close'].iloc[0]:.8f}"
                )
                if self.on_forming_kline:
                    try:
                        self.on_forming_kline(df)
                    except Exception as exc:
                        logger.error(f"Error in on_forming_kline callback: {exc}", exc_info=True)
        
        except json.JSONDecodeError as exc:
            logger.warning(f"Invalid JSON message: {exc}")
        except Exception as exc:
            logger.error(f"Error processing WebSocket message: {exc}", exc_info=True)
    
    def _on_error(self, ws: Any, error: Exception) -> None:
        """WebSocket error callback."""
        logger.error(f"WebSocket error for {self.symbol} {self.timeframe}: {error}")
        self._connected.clear()
        
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as exc:
                logger.error(f"Error in on_error callback: {exc}")
    
    def _on_close(self, ws: Any, close_status_code: Optional[int] = None, close_msg: Optional[str] = None) -> None:
        """WebSocket close callback."""
        logger.info(
            f"WebSocket closed for {self.symbol} {self.timeframe} "
            f"(code={close_status_code}, msg={close_msg})"
        )
        self._connected.clear()
        
        if self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception as exc:
                logger.error(f"Error in on_disconnect callback: {exc}")
    
    def _parse_kline(self, kline: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Parse Binance kline data to DataFrame.
        
        Args:
            kline: Kline dict from WebSocket message
            
        Returns:
            DataFrame with columns: ts, open, high, low, close, volume
        """
        try:
            df = pd.DataFrame([{
                "ts": int(kline["t"]),  # Kline start time (open_time)
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
            }])
            return df
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning(f"Failed to parse kline: {exc}")
            return None
    
    def _backfill(self) -> None:
        """
        Backfill recent candles on connect/reconnect for continuity.
        
        Fetches OVERLAP_BARS=3 recent closed candles via REST API to ensure
        no gaps after reconnect.
        """
        if self.backfill_bars <= 0:
            return
        
        try:
            logger.info(f"Backfilling {self.backfill_bars} bars for {self.symbol} {self.timeframe}")
            
            # Get server time
            server_time_ms = self.data_source.get_server_time()
            if server_time_ms is None:
                logger.warning("Failed to get server time for backfill, using local time")
                server_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            
            # Calculate end time (last closed bar)
            end_ms = (server_time_ms // self.tf_ms) * self.tf_ms
            start_ms = end_ms - (self.tf_ms * self.backfill_bars)
            
            # Convert to datetime
            start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
            
            # Fetch candles
            df = self.data_source.load_candles(
                symbol=self.symbol,
                timeframe=self.timeframe,
                start=start_dt,
                end=end_dt,
            )
            
            if df.empty:
                logger.warning(f"Backfill returned no data for {self.symbol} {self.timeframe}")
                return
            
            logger.info(
                f"Backfilled {len(df)} bars for {self.symbol} {self.timeframe} "
                f"({start_dt.isoformat()} to {end_dt.isoformat()})"
            )
            
            # Publish backfilled candles as closed klines
            if self.on_closed_kline:
                for _, row in df.iterrows():
                    row_df = pd.DataFrame([row.to_dict()])
                    try:
                        self.on_closed_kline(row_df)
                    except Exception as exc:
                        logger.error(f"Error publishing backfilled kline: {exc}", exc_info=True)
        
        except Exception as exc:
            logger.error(f"Backfill failed for {self.symbol} {self.timeframe}: {exc}", exc_info=True)
