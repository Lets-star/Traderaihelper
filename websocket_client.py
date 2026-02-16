import json
import logging
import threading
import time
import websocket
from typing import Any, Callable, Dict, Optional

from logging_config import get_structured_logger

# Optional metrics import
try:
    from metrics import (
        websocket_connections,
        websocket_reconnections,
        websocket_latency,
        websocket_messages,
        websocket_active_connections,
        websocket_errors,
    )
    from metrics.collectors import get_websocket_collector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = get_structured_logger(__name__)

class BinanceWebSocketClient:
    def __init__(self, symbol: str, interval: str, on_closed_bar: Optional[Callable[[Dict], None]] = None, on_forming_bar: Optional[Callable[[Dict], None]] = None):
        self.symbol = symbol.upper()
        self.interval = interval
        self.sock = None  # Initialize to None
        self.ws = None
        self.callbacks = {
            "closed": on_closed_bar,
            "forming": on_forming_bar,
        }
        self.stop_event = threading.Event()
        self.reconnect_count = 0
        self.max_reconnects = 10
        self.backoff_ms = 100
        
    def start(self):
        """Start WebSocket connection with error handling."""
        if self.stop_event.is_set():
            return
            
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Failed to start WebSocket: {e}")
            self._schedule_reconnect()
    
    def _connect(self):
        """Establish WebSocket connection."""
        url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
        connect_start = time.time()
        
        try:
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            # Run in background thread
            ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            ws_thread.start()
            
            # Calculate connection latency
            latency = time.time() - connect_start
            
            self.reconnect_count = 0  # Reset on successful connection
            
            logger.info(
                "WebSocket connected",
                symbol=self.symbol,
                interval=self.interval,
                latency_ms=round(latency * 1000, 2)
            )
            
            # Record metrics
            if METRICS_AVAILABLE:
                websocket_connections.labels(
                    symbol=self.symbol,
                    interval=self.interval,
                    status="success"
                ).inc()
                websocket_latency.labels(
                    symbol=self.symbol,
                    interval=self.interval
                ).observe(latency)
                websocket_active_connections.labels(
                    symbol=self.symbol,
                    interval=self.interval
                ).set(1)
                
                # Record to collector
                collector = get_websocket_collector()
                collector.record_connect(self.symbol, self.interval, True, latency * 1000)
                
        except Exception as e:
            latency = time.time() - connect_start
            logger.error(
                "WebSocket connection error",
                symbol=self.symbol,
                interval=self.interval,
                error=str(e),
                latency_ms=round(latency * 1000, 2)
            )
            
            if METRICS_AVAILABLE:
                websocket_connections.labels(
                    symbol=self.symbol,
                    interval=self.interval,
                    status="failed"
                ).inc()
                websocket_errors.labels(
                    symbol=self.symbol,
                    interval=self.interval,
                    error_type="connection"
                ).inc()
                
                collector = get_websocket_collector()
                collector.record_connect(self.symbol, self.interval, False, latency * 1000)
            
            self.ws = None
            self._schedule_reconnect()
    
    def _run_websocket(self):
        """Run WebSocket in a safe manner."""
        try:
            if self.ws:
                self.ws.run_forever()
        except Exception as e:
            logger.error(f"Error running WebSocket: {e}")
        finally:
            # Ensure cleanup
            if not self.stop_event.is_set():
                self._schedule_reconnect()
    
    def _on_open(self, ws):
        """Handle WebSocket open."""
        logger.info(f"WebSocket opened: {self.symbol} {self.interval}")
        try:
            self.sock = ws.sock
        except Exception as e:
            logger.error(f"Error accessing WebSocket socket: {e}")
            self.sock = None

    def _on_error(self, ws, error):
        """Handle WebSocket errors."""
        logger.warning(
            "WebSocket error",
            symbol=self.symbol,
            interval=self.interval,
            error=str(error)
        )
        
        if METRICS_AVAILABLE:
            error_type = type(error).__name__ if error else "unknown"
            websocket_errors.labels(
                symbol=self.symbol,
                interval=self.interval,
                error_type=error_type
            ).inc()
            
            collector = get_websocket_collector()
            collector.record_error(self.symbol, self.interval, error_type)
        
        self._schedule_reconnect()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(
            "WebSocket disconnected",
            symbol=self.symbol,
            interval=self.interval,
            status_code=close_status_code,
            message=close_msg
        )
        
        if METRICS_AVAILABLE:
            websocket_active_connections.labels(
                symbol=self.symbol,
                interval=self.interval
            ).set(0)
            
            collector = get_websocket_collector()
            collector.record_disconnect(self.symbol, self.interval, str(close_status_code))
        
        if not self.stop_event.is_set():
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule reconnect with exponential backoff."""
        if self.stop_event.is_set():
            return

        if self.reconnect_count >= self.max_reconnects:
            logger.error(
                "Max reconnect attempts reached",
                symbol=self.symbol,
                interval=self.interval,
                max_reconnects=self.max_reconnects
            )
            return
        
        self.reconnect_count += 1
        backoff = min(self.backoff_ms * (2 ** self.reconnect_count), 30000)  # Max 30s
        
        logger.info(
            "Scheduling WebSocket reconnect",
            symbol=self.symbol,
            interval=self.interval,
            attempt=self.reconnect_count,
            backoff_ms=backoff
        )
        
        if METRICS_AVAILABLE:
            websocket_reconnections.labels(
                symbol=self.symbol,
                interval=self.interval
            ).inc()
            
            collector = get_websocket_collector()
            collector.record_reconnect(self.symbol, self.interval, self.reconnect_count)
        
        threading.Timer(backoff / 1000, self.start).start()
    
    def _on_message(self, ws, msg):
        """Handle WebSocket messages."""
        try:
            data = json.loads(msg)
            kline = data.get('k')
            if not kline:
                return
            
            # Extract kline data
            kline_dict = {
                'ts': int(kline['t']),  # Candle open time (ms)
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v']),
            }
            
            # Determine message type
            msg_type = "closed" if kline['x'] else "forming"
            
            # Record metrics
            if METRICS_AVAILABLE:
                websocket_messages.labels(
                    symbol=self.symbol,
                    interval=self.interval,
                    type=msg_type
                ).inc()
                
                collector = get_websocket_collector()
                collector.record_message(self.symbol, self.interval, msg_type)
            
            # Check if candle is closed
            if kline['x']:  # x == true means closed
                if self.callbacks.get('closed'):
                    self.callbacks['closed'](kline_dict)
            else:  # Forming bar
                if self.callbacks.get('forming'):
                    self.callbacks['forming'](kline_dict)
        except Exception as e:
            logger.error(
                "Error processing WebSocket message",
                symbol=self.symbol,
                interval=self.interval,
                error=str(e)
            )
            
            if METRICS_AVAILABLE:
                websocket_errors.labels(
                    symbol=self.symbol,
                    interval=self.interval,
                    error_type="message_parse"
                ).inc()
    
    def stop(self):
        """Stop WebSocket connection gracefully."""
        logger.info(
            "Stopping WebSocket connection",
            symbol=self.symbol,
            interval=self.interval
        )
        
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        self.sock = None
        self.ws = None
        
        if METRICS_AVAILABLE:
            websocket_active_connections.labels(
                symbol=self.symbol,
                interval=self.interval
            ).set(0)
    
    def is_connected(self):
        """Check if connected."""
        try:
            connected = (self.ws is not None and 
                        self.ws.sock is not None and 
                        self.ws.sock.connected)
            return connected
        except Exception as e:
            logger.debug(
                "Error checking WebSocket connection",
                symbol=self.symbol,
                interval=self.interval,
                error=str(e)
            )
            return False
