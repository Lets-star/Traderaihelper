"""
Signal executor with thread-safe logging, validation, and st.secrets support.

FEATURES:
- Thread-safe CSV logging with threading.Lock
- Validation for processed signals
- st.secrets support for API keys with fallback to environment variables
- Threaded execution with ThreadPoolExecutor (replaces asyncio.run() approach)
- Comprehensive error handling (no bare excepts)
- Enhanced logging and monitoring
- Context manager support for ByBitClient
- UpdateBus integration for real-time updates
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests

from bybit_client import ByBitClient
from update_bus import UpdateBus

logger = logging.getLogger(__name__)


class SignalExecutor:
    """
    Thread-safe signal executor for ByBit with enhanced validation and logging.

    This class provides:
    - Thread-safe signal execution using ThreadPoolExecutor
    - Comprehensive signal validation
    - CSV logging with file locking
    - st.secrets integration with environment variable fallbacks
    - UpdateBus integration for real-time execution updates
    - Dry run mode for testing
    """

    LOG_FILE = "trade_execution_log.csv"
    MAX_WORKER_THREADS = 3

    def __init__(self, update_bus: Optional[UpdateBus] = None) -> None:
        """
        Initialize signal executor.

        Args:
            update_bus: Optional UpdateBus for publishing execution updates
        """
        self.client: Optional[ByBitClient] = None
        self.update_bus = update_bus
        self.enabled = False
        self.api_key = ""
        self.api_secret = ""
        self.testnet = True
        self.default_leverage = 5
        self.pos_size_multiplier = 1.0
        self.dry_run = False

        # Thread safety
        self._lock = threading.RLock()
        self._csv_lock = threading.Lock()

        # Thread pool for concurrent execution
        self._executor = ThreadPoolExecutor(max_workers=self.MAX_WORKER_THREADS)

        # Statistics tracking
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._validation_errors = 0

        # Ensure log file exists with header
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure CSV log file exists with proper headers."""
        with self._csv_lock:
            if not os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "signal_id", "symbol", "direction", "qty",
                        "entry_price", "take_profit", "stop_loss", "leverage",
                        "status", "response_code", "latency_ms", "error_msg",
                        "validation_errors", "thread_id"
                    ])

    def _get_api_credentials(self) -> tuple[str, str]:
        """
        Securely get API credentials from st.secrets with fallback.

        Returns:
            Tuple of (api_key, api_secret)
        """
        try:
            # Try to get from st.secrets first
            import streamlit as st

            if hasattr(st, 'secrets') and st.secrets:
                api_key = st.secrets.get("BYBIT_API_KEY")
                api_secret = st.secrets.get("BYBIT_API_SECRET")

                if api_key and api_secret:
                    logger.info("Using ByBit credentials from st.secrets")
                    return api_key, api_secret

                # Try bybit section
                api_key = st.secrets.get("bybit", {}).get("api_key")
                api_secret = st.secrets.get("bybit", {}).get("api_secret")

                if api_key and api_secret:
                    logger.info("Using ByBit credentials from st.secrets[bybit]")
                    return api_key, api_secret

        except ImportError:
            logger.debug("Streamlit not available, using environment variables")
        except AttributeError:
            logger.debug("st.secrets not available, using environment variables")
        except Exception as e:
            logger.warning(f"Error accessing st.secrets: {e}")

        # Fallback to environment variables
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if api_key and api_secret:
            logger.info("Using ByBit credentials from environment variables")
            return api_key, api_secret

        raise ValueError("API credentials not found in st.secrets or environment variables")

    def _log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Thread-safe CSV logging for trades."""
        with self._csv_lock:
            try:
                with open(self.LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.utcnow().isoformat(),
                        trade_data.get("signal_id", ""),
                        trade_data.get("symbol", ""),
                        trade_data.get("direction", ""),
                        trade_data.get("qty", ""),
                        trade_data.get("entry_price", ""),
                        trade_data.get("take_profit", ""),
                        trade_data.get("stop_loss", ""),
                        trade_data.get("leverage", ""),
                        trade_data.get("status", ""),
                        trade_data.get("response_code", ""),
                        trade_data.get("latency_ms", ""),
                        trade_data.get("error_msg", ""),
                        trade_data.get("validation_errors", ""),
                        trade_data.get("thread_id", "")
                    ])
            except IOError as e:
                logger.error(f"Failed to write to trade log: {e}")
            except Exception as e:
                logger.error(f"Unexpected error logging trade: {e}")

    def _validate_signal(self, signal: Dict[str, Any]) -> List[str]:
        """
        Validate processed signal structure and content.

        Args:
            signal: Signal dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not signal or not isinstance(signal, dict):
            errors.append("Signal must be a non-empty dictionary")
            return errors

        # Required fields
        required_fields = ["signal_id", "symbol", "direction", "entry_price"]
        for field in required_fields:
            if field not in signal:
                errors.append(f"Missing required field: {field}")

        # Validate signal type
        signal_type = signal.get("signal")
        if signal_type not in [None, "BUY", "SELL", "HOLD"]:
            errors.append(f"Invalid signal type: {signal_type}. Must be BUY, SELL, or HOLD")

        # Validate symbol format
        symbol = signal.get("symbol")
        if symbol:
            if not isinstance(symbol, str):
                errors.append(f"Symbol must be a string: {type(symbol)}")
            elif len(symbol) < 3:
                errors.append(f"Symbol must be at least 3 characters: {symbol}")
            else:
                # Convert to uppercase
                signal["symbol"] = symbol.upper()

        # Validate direction
        direction = signal.get("direction")
        if direction:
            if not isinstance(direction, str):
                errors.append(f"Direction must be a string: {type(direction)}")
            elif direction.upper() not in ["LONG", "SHORT"]:
                errors.append(f"Invalid direction: {direction}. Must be LONG or SHORT")

        # Validate numeric fields
        numeric_fields = ["entry_price", "take_profit", "stop_loss", "leverage", "quantity"]
        for field in numeric_fields:
            value = signal.get(field)
            if value is not None:
                try:
                    float_value = float(value)
                    if field in ["leverage"] and (float_value <= 0 or float_value > 125):
                        errors.append(f"{field} must be between 0 and 125: {float_value}")
                    elif field in ["entry_price", "take_profit", "stop_loss"] and float_value <= 0:
                        errors.append(f"{field} must be positive: {float_value}")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be numeric: {value}")

        # Validate entries if present
        entries = signal.get("entries")
        if entries is not None:
            if not isinstance(entries, list):
                errors.append("Entries must be a list")
            elif len(entries) == 0:
                errors.append("Entries list cannot be empty")
            else:
                for i, entry in enumerate(entries):
                    try:
                        float(entry)
                        if float(entry) <= 0:
                            errors.append(f"Entry at index {i} must be positive: {entry}")
                    except (ValueError, TypeError):
                        errors.append(f"Entry at index {i} must be numeric: {entry}")

        # Validate take profits structure
        take_profits = signal.get("take_profits")
        if take_profits is not None:
            if not isinstance(take_profits, dict):
                errors.append("Take profits must be a dictionary")
            else:
                for key, value in take_profits.items():
                    try:
                        tp_value = float(value)
                        if tp_value <= 0:
                            errors.append(f"Take profit value for {key} must be positive: {value}")
                    except (ValueError, TypeError):
                        errors.append(f"Take profit value for {key} must be numeric: {value}")

        # Validate signal_id format
        signal_id = signal.get("signal_id")
        if signal_id and not isinstance(signal_id, str):
            errors.append(f"Signal ID must be a string: {type(signal_id)}")

        return errors

    def configure(
        self,
        enabled: bool,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        leverage: int = 5,
        pos_size_multiplier: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        """
        Configure executor with optional st.secrets integration.

        Args:
            enabled: Enable signal execution
            api_key: API key (will be ignored if using st.secrets and both are empty)
            api_secret: API secret (will be ignored if using st.secrets and both are empty)
            testnet: Use testnet or mainnet
            leverage: Default leverage
            pos_size_multiplier: Position size multiplier
            dry_run: Dry run mode (no actual trades)
        """
        self.enabled = enabled
        self.testnet = testnet
        self.default_leverage = leverage
        self.pos_size_multiplier = pos_size_multiplier
        self.dry_run = dry_run

        if enabled and not dry_run:
            try:
                # Try to get credentials from st.secrets first if direct credentials not provided
                if not api_key or not api_secret:
                    self.api_key, self.api_secret = self._get_api_credentials()
                else:
                    self.api_key = api_key
                    self.api_secret = api_secret

                logger.info(f"ByBit client configured for {'testnet' if testnet else 'mainnet'}")

            except ValueError as e:
                logger.error(f"Failed to configure API credentials: {e}")
                self.enabled = False  # Disable if credentials invalid
            except Exception as e:
                logger.error(f"Unexpected error configuring executor: {e}", exc_info=True)
                self.enabled = False

    def _execute_signal_sync(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute signal synchronously with thread safety and validation.

        Args:
            signal: Validated signal dictionary

        Returns:
            Execution result dictionary
        """
        thread_id = threading.current_thread().ident

        # Validate signal before execution
        validation_errors = self._validate_signal(signal)
        if validation_errors:
            with self._lock:
                self._total_executions += 1
                self._validation_errors += 1

            error_msg = f"Signal validation failed: {', '.join(validation_errors)}"
            logger.error(f"Signal validation failed: {validation_errors}")

            # Log validation failure
            self._log_trade({
                "signal_id": signal.get("signal_id", ""),
                "symbol": signal.get("symbol", ""),
                "direction": signal.get("direction", ""),
                "qty": signal.get("quantity", ""),
                "entry_price": signal.get("entry_price", ""),
                "status": "validation_error",
                "response_code": -1,
                "error_msg": error_msg,
                "validation_errors": "; ".join(validation_errors),
                "thread_id": str(thread_id)
            })

            return {
                "status": "validation_error",
                "error": error_msg,
                "validation_errors": validation_errors
            }

        signal_id = signal.get("signal_id", f"sig_{int(time.time()*1000)}")
        symbol = signal.get("symbol", "")
        direction = signal.get("direction", "")
        entry_price = float(signal.get("entry_price", 0))
        tp = float(signal.get("take_profit", 0))
        sl = float(signal.get("stop_loss", 0))

        # Calculate quantity
        qty = float(signal.get("quantity", 0.001)) * self.pos_size_multiplier

        # Get leverage from signal or default
        leverage = signal.get("leverage", self.default_leverage)

        logger.info(f"Processing signal {signal_id} for {symbol} {direction} x {qty} "
                   f"(thread: {thread_id})")

        start_time = time.time()

        status = "pending"
        response_code = 0
        error_msg = ""

        # Publish initial update
        if self.update_bus:
            try:
                self.update_bus.publish({
                    "type": "EXECUTION_UPDATE",
                    "signal_id": signal_id,
                    "status": "pending",
                    "timestamp": start_time,
                    "thread_id": thread_id
                })
            except Exception as e:
                logger.warning(f"Failed to publish initial execution update: {e}")

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {direction} {qty} {symbol} @ {entry_price}")
            status = "filled (dry_run)"
            time.sleep(0.1)  # Simulate network delay
        else:
            try:
                # Use context manager for client
                with ByBitClient(self.api_key, self.api_secret, self.testnet) as client:
                    if not client.validate_credentials():
                        raise ValueError("Invalid API credentials")

                    # Set leverage
                    lev_res = client.set_leverage(symbol, str(leverage))
                    lev_code = lev_res.get("retCode", -1)
                    if lev_code not in [0, 110043]:  # 0 success, 110043 leverage not modified
                        logger.warning(f"Set leverage failed: {lev_res}")

                    # Prepare order side
                    side = "Buy" if direction.upper() == "LONG" else "Sell"

                    # Place order
                    res = client.place_order(
                        symbol=symbol,
                        side=side,
                        qty=str(qty),
                        order_type="Market",
                        take_profit=str(tp) if tp > 0 else None,
                        stop_loss=str(sl) if sl > 0 else None,
                        client_order_id=f"{signal_id}"
                    )

                    response_code = res.get("retCode", -1)
                    if response_code == 0:
                        status = "filled"
                        logger.info(f"Order filled successfully: {signal_id}")
                    else:
                        status = "error"
                        error_msg = res.get("retMsg", "Unknown error")
                        logger.error(f"Order failed: {error_msg} (code: {response_code})")

            except ValueError as e:
                logger.error(f"Validation error during execution: {e}")
                status = "error"
                error_msg = str(e)
                response_code = -1

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during execution: {e}")
                status = "error"
                error_msg = f"Network error: {str(e)}"
                response_code = -1

            except Exception as e:
                logger.error(f"Execution error: {e}", exc_info=True)
                status = "error"
                error_msg = str(e)
                response_code = -1

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Update statistics
        with self._lock:
            self._total_executions += 1
            if response_code == 0:
                self._successful_executions += 1
            else:
                self._failed_executions += 1

        # Log execution
        self._log_trade({
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "qty": str(qty),
            "entry_price": str(entry_price),
            "take_profit": str(tp),
            "stop_loss": str(sl),
            "leverage": str(leverage),
            "status": status,
            "response_code": str(response_code),
            "latency_ms": f"{latency_ms:.2f}",
            "error_msg": error_msg,
            "validation_errors": "",
            "thread_id": str(thread_id)
        })

        # Publish final update
        if self.update_bus:
            try:
                self.update_bus.publish({
                    "type": "EXECUTION_UPDATE",
                    "signal_id": signal_id,
                    "status": status,
                    "latency_ms": latency_ms,
                    "error": error_msg,
                    "timestamp": end_time,
                    "thread_id": thread_id,
                    "response_code": response_code
                })
            except Exception as e:
                logger.warning(f"Failed to publish final execution update: {e}")

        return {
            "status": status,
            "error": error_msg,
            "response_code": response_code,
            "latency_ms": latency_ms,
            "validation_errors": None
        }

    def execute_signal(self, signal: Dict[str, Any]) -> None:
        """
        Execute signal asynchronously using thread pool for better concurrency.

        Args:
            signal: Signal dictionary to execute
        """
        if not self.enabled:
            logger.info("Signal execution disabled")
            return

        try:
            # Submit to thread pool for execution
            future = self._executor.submit(self._execute_signal_sync, signal)

            # Add callback for error handling
            def handle_execution(fut):
                try:
                    result = fut.result(timeout=60)  # 60 second timeout
                    if result.get("status") == "error":
                        logger.error(f"Signal execution failed: {result}")
                except TimeoutError:
                    logger.error(f"Signal execution timed out")
                except Exception as e:
                    logger.error(f"Signal execution exception: {e}")

            future.add_done_callback(handle_execution)

        except Exception as e:
            logger.error(f"Failed to start execution thread: {e}")

    def execute_signal_sync(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous execution that waits for completion.

        Args:
            signal: Signal dictionary to execute

        Returns:
            Execution result
        """
        if not self.enabled:
            return {
                "status": "disabled",
                "error": "Signal execution is disabled"
            }

        return self._execute_signal_sync(signal)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position with validation."""
        if not self.api_key or not self.api_secret:
            return {"error": "API credentials not configured"}

        try:
            with ByBitClient(self.api_key, self.api_secret, self.testnet) as client:
                return client.get_position(symbol)
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return {"error": str(e)}

    def is_position_open(self, symbol: str) -> bool:
        """Check if position is open for symbol."""
        result = self.get_position(symbol)
        if result.get("retCode") != 0:
            return False

        positions = result.get("result", {}).get("list", [])
        for pos in positions:
            size = float(pos.get("size", 0))
            if size > 0:
                return True
        return False

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
        logger.info("Signal executor cleaned up")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        with self._lock:
            total = self._total_executions
            success_rate = (
                (self._successful_executions / total * 100)
                if total > 0
                else 0.0
            )

            return {
                "total_executions": total,
                "successful_executions": self._successful_executions,
                "failed_executions": self._failed_executions,
                "validation_errors": self._validation_errors,
                "success_rate_percent": round(success_rate, 2),
                "enabled": self.enabled,
                "dry_run": self.dry_run,
            }

    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        with self._lock:
            self._total_executions = 0
            self._successful_executions = 0
            self._failed_executions = 0
            self._validation_errors = 0

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in destructor
