import logging
import asyncio
import threading
import time
import json
import os
import csv
from typing import Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from bybit_client import ByBitClient
from update_bus import UpdateBus

logger = logging.getLogger(__name__)

class SignalExecutor:
    """
    Executes automated signals on ByBit.
    """

    LOG_FILE = "trade_execution_log.csv"
    _log_lock = threading.Lock()

    def __init__(self, update_bus: Optional[UpdateBus] = None):
        self.client: Optional[ByBitClient] = None
        self.update_bus = update_bus
        self.enabled = False
        self.api_key = ""
        self.api_secret = ""
        self.testnet = True
        self.default_leverage = 5
        self.pos_size_multiplier = 1.0
        self.dry_run = False

        # Ensure log file exists with header (thread-safe initialization)
        with self._log_lock:
            if not os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "signal_id", "symbol", "direction", "qty",
                        "entry_price", "take_profit", "stop_loss", "leverage",
                        "status", "response_code", "latency_ms", "error_msg"
                    ])

    def configure(self, enabled: bool, api_key: str, api_secret: str, testnet: bool, 
                  leverage: int, pos_size_multiplier: float, dry_run: bool):
        self.enabled = enabled
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.default_leverage = leverage
        self.pos_size_multiplier = pos_size_multiplier
        self.dry_run = dry_run
        
        if enabled and not self.dry_run and api_key and api_secret:
            if self.client:
                # Re-init if config changed (simplification)
                pass # Reuse or close/recreate? 
            # We will create client on demand or update it
            
    async def _execute_async(self, signal: Dict[str, Any]):
        if not self.enabled:
            return

        signal_id = signal.get("signal_id", f"sig_{int(time.time()*1000)}")
        symbol = signal.get("symbol")
        direction = signal.get("direction") # LONG / SHORT
        entry_price = float(signal.get("entry_price", 0))
        tp = float(signal.get("take_profit", 0))
        sl = float(signal.get("stop_loss", 0))
        
        # Calculate quantity based on position sizing logic
        # For now, let's assume a fixed amount or derived from signal
        # The ticket says: "Extract: symbol, direction... position_size (contracts or qty)"
        # If not present, we need a default.
        # Let's assume the signal contains 'quantity' or we default to min size for test.
        qty = float(signal.get("quantity", 0.001)) * self.pos_size_multiplier
        
        # Validate leverage
        leverage = signal.get("leverage", self.default_leverage)
        
        logger.info(f"Processing signal {signal_id} for {symbol} {direction} x {qty}")

        start_time = time.time()
        
        status = "pending"
        response_code = 0
        error_msg = ""
        
        if self.update_bus:
            self.update_bus.publish({
                "type": "EXECUTION_UPDATE",
                "signal_id": signal_id,
                "status": "pending",
                "timestamp": start_time
            })

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {direction} {qty} {symbol} @ {entry_price}")
            status = "filled (dry_run)"
            await asyncio.sleep(0.1) # Simulate network
        else:
            try:
                # Init client
                client = ByBitClient(self.api_key, self.api_secret, self.testnet)
                
                # Set Leverage
                # Note: This might fail if already set, check retCode
                lev_res = await client.set_leverage(symbol, str(leverage))
                if lev_res.get("retCode") not in [0, 110043]: # 0 success, 110043 leverage not modified
                     logger.warning(f"Set leverage failed: {lev_res}")

                side = "Buy" if direction.upper() == "LONG" else "Sell"
                
                # Place Order
                # We use limit if we want strict entry, or market. Ticket says "create market/limit order".
                # Automated signals usually imply immediate entry -> Market.
                # But if entry_price is specified, maybe Limit?
                # Let's use Market for simplicity and guaranteed execution for now, 
                # unless signal specifies order_type.
                
                res = await client.place_order(
                    symbol=symbol,
                    side=side,
                    qty=str(qty),
                    order_type="Market", # Default to Market for immediate entry
                    take_profit=str(tp),
                    stop_loss=str(sl),
                    client_order_id=f"{signal_id}"
                )
                
                response_code = res.get("retCode")
                if response_code == 0:
                    status = "filled"
                else:
                    status = "error"
                    error_msg = res.get("retMsg")
                
                await client.close()
                
            except Exception as e:
                logger.error(f"Execution error: {e}", exc_info=True)
                status = "error"
                error_msg = str(e)
                response_code = -1

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Log to file (thread-safe)
        with self._log_lock:
            with open(self.LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(), signal_id, symbol, direction, qty,
                    entry_price, tp, sl, leverage,
                    status, response_code, f"{latency_ms:.2f}", error_msg
                ])
            
        if self.update_bus:
            self.update_bus.publish({
                "type": "EXECUTION_UPDATE",
                "signal_id": signal_id,
                "status": status,
                "latency_ms": latency_ms,
                "error": error_msg,
                "timestamp": end_time
            })

    def _run_async_in_thread(self, signal: Dict[str, Any]):
        """Run async execution in a dedicated thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._execute_async(signal))
        except Exception as e:
            logger.error(f"Failed to run async execution in thread: {e}")
        finally:
            loop.close()

    def execute_signal(self, signal: Dict[str, Any]):
        """
        Synchronous wrapper to run async execution in a background thread.
        This avoids conflicts with Streamlit's event loop.
        """
        try:
            # Use threading to avoid conflicts with Streamlit's event loop
            thread = threading.Thread(target=self._run_async_in_thread, args=(signal,))
            thread.daemon = True
            thread.start()
            # Don't wait for completion - let it run in background
            # The update_bus will publish results when done
        except Exception as e:
            logger.error(f"Failed to start execution thread: {e}")

    def execute_signal_sync(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous execution that waits for completion.
        Use this when you need the result immediately.
        Returns the execution result.
        """
        result = {"status": "pending", "error": None}
        
        def run_and_capture():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._execute_async(signal))
                result["status"] = "completed"
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_and_capture)
        thread.daemon = True
        thread.start()
        thread.join(timeout=30.0)  # Wait up to 30 seconds
        
        if thread.is_alive():
            result["status"] = "timeout"
            result["error"] = "Execution timed out after 30 seconds"
        
        return result

    async def _get_position_async(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not configured"}
        
        client = ByBitClient(self.api_key, self.api_secret, self.testnet)
        try:
            result = await client.get_position(symbol)
            await client.close()
            return result
        except Exception as e:
            await client.close()
            logger.error(f"Error getting position: {e}")
            return {"error": str(e)}

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Synchronous wrapper to get position."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_position_async(symbol))
        except Exception as e:
            logger.error(f"Failed to get position: {e}")
            return {"error": str(e)}
        finally:
            loop.close()

    def is_position_open(self, symbol: str) -> bool:
        """Check if there's an open position for the symbol."""
        result = self.get_position(symbol)
        if result.get("retCode") != 0:
            return False
        
        positions = result.get("result", {}).get("list", [])
        for pos in positions:
            size = float(pos.get("size", 0))
            if size > 0:
                return True
        return False