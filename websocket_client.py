import json
import logging
import threading
import time
import websocket
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

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
            self.reconnect_count = 0  # Reset on successful connection
            logger.info(f"WebSocket connected: {self.symbol} {self.interval}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
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
        logger.warning(f"WebSocket error for {self.symbol} {self.interval}: {error}")
        self._schedule_reconnect()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"Chart WebSocket disconnected: {self.symbol} {self.interval}")
        if not self.stop_event.is_set():
            self._schedule_reconnect()
    
    def _schedule_reconnect(self):
        """Schedule reconnect with exponential backoff."""
        if self.stop_event.is_set():
            return

        if self.reconnect_count >= self.max_reconnects:
            logger.error(f"Max reconnect attempts reached for {self.symbol} {self.interval}")
            return
        
        self.reconnect_count += 1
        backoff = min(self.backoff_ms * (2 ** self.reconnect_count), 30000)  # Max 30s
        logger.info(f"Reconnecting {self.symbol} {self.interval} in {backoff}ms (attempt {self.reconnect_count})")
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
            
            # Check if candle is closed
            if kline['x']:  # x == true means closed
                if self.callbacks.get('closed'):
                    self.callbacks['closed'](kline_dict)
            else:  # Forming bar
                if self.callbacks.get('forming'):
                    self.callbacks['forming'](kline_dict)
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def stop(self):
        """Stop WebSocket connection gracefully."""
        self.stop_event.set()
        if self.ws:
            self.ws.close()
        self.sock = None
        self.ws = None
    
    def is_connected(self):
        """Check if connected."""
        try:
            return (self.ws is not None and 
                   self.ws.sock is not None and 
                   self.ws.sock.connected)
        except Exception as e:
            logger.debug(f"Error checking WebSocket connection: {e}")
            return False
