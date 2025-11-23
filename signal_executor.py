import logging
import asyncio
import time
import json
import os
import csv
from typing import Dict, Any, Optional
from datetime import datetime
from bybit_client import ByBitClient
from update_bus import UpdateBus

logger = logging.getLogger(__name__)

class SignalExecutor:
    """
    Executes automated signals on ByBit.
    """
    
    LOG_FILE = "trade_execution_log.csv"

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
        
        # Ensure log file exists with header
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
        
        # Log to file
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

    def execute_signal(self, signal: Dict[str, Any]):
        """
        Synchronous wrapper to run async execution.
        """
        try:
            asyncio.run(self._execute_async(signal))
        except Exception as e:
            logger.error(f"Failed to run async execution: {e}")

