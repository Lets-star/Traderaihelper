"""
ByBit API Client with enhanced error handling, validation, and thread safety.

FEATURES:
- Thread-safe synchronous API client (replaces async/await approach)
- Context manager support (__enter__/__exit__)
- Comprehensive parameter validation for all methods
- Rate limit handling for 429 errors with exponential backoff
- Connection pooling for HTTP requests
- Thread-safe CSV logging for all API operations
- Improved logging and error handling
- Type-safe Enums for OrderSide and OrderType
"""

import time
import hmac
import hashlib
import json
import logging
import threading
import csv
import os
import queue
from typing import Any, Dict, Optional, List, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

from logging_config import get_structured_logger
from trader_types.enums import OrderSide, OrderType, OrderStatus

# Optional metrics import
try:
    from metrics import api_requests, api_latency, api_rate_limits, api_errors, api_active_requests
    from metrics.collectors import get_api_collector
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = get_structured_logger(__name__)


class ByBitClient:
    """
    Thread-safe synchronous client for ByBit V5 API with context manager support.
    Supports Linear Futures (USDT Perpetuals).
    """

    TESTNET_API_URL = "https://api-testnet.bybit.com"
    MAINNET_API_URL = "https://api.bybit.com"

    # Rate limiting settings
    MAX_RETRIES = 3
    RATE_LIMIT_BACKOFF = [1, 2, 4, 8]  # seconds for different retry attempts
    BASE_TIMEOUT = 30

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True,
                 connection_pool_size: int = 10, log_trades: bool = True):
        """
        Initialize ByBit client with connection pooling.

        Args:
            api_key: ByBit API key
            api_secret: ByBit API secret
            testnet: Use testnet (True) or mainnet (False)
            connection_pool_size: Size of HTTP connection pool
            log_trades: Enable CSV logging of API calls
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_API_URL if testnet else self.MAINNET_API_URL
        self.recv_window = "5000"

        # Connection pooling configuration
        self.connection_pool_size = connection_pool_size
        self.session: Optional[requests.Session] = None

        # Thread safety
        self._lock = threading.RLock()
        self._csv_lock = threading.Lock()

        # CSV logging
        self.log_trades = log_trades
        self.trade_log_file = "bybit_api_log.csv"
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        """Ensure CSV log file exists with proper headers."""
        if not self.log_trades:
            return

        with self._csv_lock:
            if not os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "method", "endpoint", "symbol", "side", "qty",
                        "order_type", "price", "status", "ret_code", "ret_msg",
                        "latency_ms", "attempt", "error_details"
                    ])

    def _log_trade(self, method: str, endpoint: str, symbol: str, side: str, qty: str,
                   order_type: str, price: Optional[str], status: str,
                   ret_code: int, ret_msg: str, latency_ms: float,
                   attempt: int, error_details: Optional[str] = None) -> None:
        """Thread-safe CSV logging for API calls."""
        if not self.log_trades:
            return

        with self._csv_lock:
            try:
                with open(self.trade_log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.utcnow().isoformat(), method, endpoint, symbol, side, qty,
                        order_type, price or "", status, ret_code, ret_msg,
                        f"{latency_ms:.2f}", attempt, error_details or ""
                    ])
            except Exception as e:
                logger.error(f"Failed to log API call: {e}")

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session with connection pooling and retry strategy."""
        with self._lock:
            if self.session is None:
                self.session = requests.Session()

                # Configure retry strategy
                retry_strategy = Retry(
                    total=self.MAX_RETRIES,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
                )

                # Configure adapter with connection pooling
                adapter = HTTPAdapter(
                    pool_connections=self.connection_pool_size,
                    pool_maxsize=self.connection_pool_size,
                    max_retries=retry_strategy
                )

                self.session.mount("http://", adapter)
                self.session.mount("https://", adapter)

                # Set default headers
                self.session.headers.update({
                    'Content-Type': 'application/json',
                    'User-Agent': 'ByBit-Client/2.0'
                })

            return self.session

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close the HTTP session."""
        with self._lock:
            if self.session is not None:
                self.session.close()
                self.session = None

    def _generate_signature(self, timestamp: str, payload: str) -> str:
        """Generate HMAC-SHA256 signature."""
        param_str = timestamp + self.api_key + self.recv_window + payload
        return hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _validate_symbol(self, symbol: str) -> None:
        """Validate trading symbol format."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        if len(symbol) < 3:
            raise ValueError("Symbol must be at least 3 characters long")
        if not symbol.isupper():
            symbol = symbol.upper()

    def _validate_side(self, side: Union[str, OrderSide]) -> str:
        """Validate and normalize order side using OrderSide Enum."""
        if isinstance(side, OrderSide):
            return side.value
        if not side or not isinstance(side, str):
            raise ValueError("Side must be a non-empty string")
        side_normalized = side.capitalize()
        try:
            # Validate using Enum
            OrderSide(side_normalized)
            return side_normalized
        except ValueError:
            raise ValueError(f"Side must be one of: {[s.value for s in OrderSide]}")

    def _validate_order_type(self, order_type: Union[str, OrderType]) -> str:
        """Validate order type using OrderType Enum."""
        if isinstance(order_type, OrderType):
            return order_type.value
        if order_type not in [t.value for t in OrderType]:
            raise ValueError(f"Order type must be one of: {[t.value for t in OrderType]}")
        return order_type

    def _validate_quantity(self, qty: Union[str, float, int]) -> str:
        """Validate order quantity and return as string."""
        try:
            qty_float = float(qty)
            if qty_float <= 0:
                raise ValueError("Quantity must be positive")
            if qty_float > 1000000:  # Reasonable upper limit
                raise ValueError("Quantity seems unreasonably large")
            return str(qty_float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid quantity: {e}")

    def _validate_price(self, price: Union[str, float, int]) -> Optional[str]:
        """Validate price if provided and return as string."""
        if price is None:
            return None
        try:
            price_float = float(price)
            if price_float <= 0:
                raise ValueError("Price must be positive")
            if price_float > 1000000:  # Reasonable upper limit
                raise ValueError("Price seems unreasonably large")
            return str(price_float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid price: {e}")

    def _validate_leverage(self, leverage: Union[str, float, int]) -> str:
        """Validate leverage and return as string."""
        try:
            lev_float = float(leverage)
            if lev_float <= 0 or lev_float > 125:
                raise ValueError("Leverage must be between 0 and 125")
            return str(lev_float)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid leverage: {e}")

    def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None,
                     retry_count: int = 0) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and retry logic.

        Args:
            method: HTTP method (GET or POST)
            endpoint: API endpoint
            params: Request parameters
            retry_count: Current retry attempt

        Returns:
            API response as dictionary
        """
        session = self._get_session()
        url = f"{self.base_url}{endpoint}"

        # Prepare payload and headers
        timestamp = str(int(time.time() * 1000))

        if method == "GET":
            # Handle query parameters for signing
            query_string = ""
            if params:
                sorted_keys = sorted(params.keys())
                query_parts = []
                for key in sorted_keys:
                    value = str(params[key])
                    query_parts.append(f"{key}={value}")
                query_string = "&".join(query_parts)

            signature = self._generate_signature(timestamp, query_string)
            full_url = f"{url}?{query_string}" if query_string else url
            data = None

            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": self.recv_window
            }
        else:
            # POST request
            payload = json.dumps(params) if params else "{}"
            signature = self._generate_signature(timestamp, payload)
            full_url = url
            data = payload

            headers = {
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-SIGN-TYPE": "2",
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": self.recv_window
            }

        start_time = time.time()
        
        # Track active requests
        if METRICS_AVAILABLE:
            api_active_requests.labels(endpoint=endpoint, method=method).inc()

        try:
            response = session.request(
                method,
                full_url,
                headers=headers,
                data=data,
                timeout=self.BASE_TIMEOUT
            )

            latency_ms = (time.time() - start_time) * 1000
            latency_sec = latency_ms / 1000

            logger.debug(
                "ByBit API call",
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                latency_ms=round(latency_ms, 2),
                attempt=retry_count + 1
            )

            # Handle HTTP errors
            if response.status_code != 200:
                # Handle rate limiting specifically
                if response.status_code == 429:
                    logger.warning(
                        "Rate limit hit",
                        method=method,
                        endpoint=endpoint,
                        attempt=retry_count + 1
                    )
                    
                    if METRICS_AVAILABLE:
                        api_rate_limits.labels(endpoint=endpoint, method=method).inc()
                        api_requests.labels(endpoint=endpoint, method=method, status="rate_limited").inc()
                        api_latency.labels(endpoint=endpoint, method=method, status="rate_limited").observe(latency_sec)

                    if retry_count < self.MAX_RETRIES:
                        wait_time = self.RATE_LIMIT_BACKOFF[min(retry_count, len(self.RATE_LIMIT_BACKOFF)-1)]
                        logger.info("Retrying after rate limit", wait_time=wait_time)
                        time.sleep(wait_time)
                        return self._make_request(method, endpoint, params, retry_count + 1)
                    else:
                        return {
                            "retCode": -1,
                            "retMsg": f"Rate limit exceeded after {self.MAX_RETRIES} retries"
                        }

                # Other client errors - don't retry
                logger.error(
                    "HTTP error",
                    status_code=response.status_code,
                    response_text=response.text[:200]
                )
                
                if METRICS_AVAILABLE:
                    api_requests.labels(endpoint=endpoint, method=method, status="error").inc()
                    api_latency.labels(endpoint=endpoint, method=method, status="error").observe(latency_sec)
                    api_errors.labels(endpoint=endpoint, method=method, error_code=str(response.status_code)).inc()
                
                try:
                    return response.json()
                except ValueError:
                    return {
                        "retCode": -1,
                        "retMsg": f"HTTP {response.status_code}: {response.text}"
                    }

            # Parse successful response
            try:
                resp_json = response.json()
                
                # Record metrics for successful request
                if METRICS_AVAILABLE:
                    api_requests.labels(endpoint=endpoint, method=method, status="success").inc()
                    api_latency.labels(endpoint=endpoint, method=method, status="success").observe(latency_sec)
                    
                    # Also record to collector for detailed stats
                    collector = get_api_collector()
                    collector.record_request(endpoint, method, "success", latency_ms)
                
                return resp_json
            except ValueError as e:
                logger.error("Invalid JSON response", response_text=response.text[:200])
                
                if METRICS_AVAILABLE:
                    api_requests.labels(endpoint=endpoint, method=method, status="error").inc()
                    api_latency.labels(endpoint=endpoint, method=method, status="error").observe(latency_sec)
                    api_errors.labels(endpoint=endpoint, method=method, error_code="JSON_PARSE").inc()
                
                return {
                    "retCode": -1,
                    "retMsg": f"Invalid JSON: {response.text}"
                }

        except requests.exceptions.Timeout as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Request timeout", attempt=retry_count + 1, error=str(e))
            
            if METRICS_AVAILABLE:
                api_requests.labels(endpoint=endpoint, method=method, status="timeout").inc()
                api_latency.labels(endpoint=endpoint, method=method, status="timeout").observe(latency_ms / 1000)
                api_errors.labels(endpoint=endpoint, method=method, error_code="TIMEOUT").inc()

            # Retry on timeout
            if retry_count < self.MAX_RETRIES:
                wait_time = self.RATE_LIMIT_BACKOFF[min(retry_count, len(self.RATE_LIMIT_BACKOFF)-1)]
                logger.info("Retrying after timeout", wait_time=wait_time)
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, retry_count + 1)
            else:
                return {
                    "retCode": -1,
                    "retMsg": f"Request timeout after {self.MAX_RETRIES} retries"
                }

        except requests.exceptions.ConnectionError as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Connection error", attempt=retry_count + 1, error=str(e))
            
            if METRICS_AVAILABLE:
                api_requests.labels(endpoint=endpoint, method=method, status="connection_error").inc()
                api_latency.labels(endpoint=endpoint, method=method, status="connection_error").observe(latency_ms / 1000)
                api_errors.labels(endpoint=endpoint, method=method, error_code="CONNECTION").inc()

            # Retry on connection errors
            if retry_count < self.MAX_RETRIES:
                wait_time = self.RATE_LIMIT_BACKOFF[min(retry_count, len(self.RATE_LIMIT_BACKOFF)-1)]
                logger.info("Retrying after connection error", wait_time=wait_time)
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, retry_count + 1)
            else:
                return {
                    "retCode": -1,
                    "retMsg": f"Connection failed after {self.MAX_RETRIES} retries: {str(e)}"
                }

        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Request error", attempt=retry_count + 1, error=str(e))
            
            if METRICS_AVAILABLE:
                api_requests.labels(endpoint=endpoint, method=method, status="error").inc()
                api_latency.labels(endpoint=endpoint, method=method, status="error").observe(latency_ms / 1000)
                api_errors.labels(endpoint=endpoint, method=method, error_code="REQUEST").inc()

            # Retry on other request errors
            if retry_count < self.MAX_RETRIES:
                wait_time = self.RATE_LIMIT_BACKOFF[min(retry_count, len(self.RATE_LIMIT_BACKOFF)-1)]
                logger.info("Retrying after request error", wait_time=wait_time)
                time.sleep(wait_time)
                return self._make_request(method, endpoint, params, retry_count + 1)
            else:
                return {
                    "retCode": -1,
                    "retMsg": f"Request failed after {self.MAX_RETRIES} retries: {str(e)}"
                }

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error("Unexpected error", attempt=retry_count + 1, error=str(e))
            
            if METRICS_AVAILABLE:
                api_requests.labels(endpoint=endpoint, method=method, status="error").inc()
                api_latency.labels(endpoint=endpoint, method=method, status="error").observe(latency_ms / 1000)
                api_errors.labels(endpoint=endpoint, method=method, error_code="UNEXPECTED").inc()
            
            return {
                "retCode": -1,
                "retMsg": f"Unexpected error: {str(e)}"
            }
        
        finally:
            # Decrement active requests
            if METRICS_AVAILABLE:
                api_active_requests.labels(endpoint=endpoint, method=method).dec()

    def set_leverage(self, symbol: str, leverage: Union[str, int, float]) -> Dict[str, Any]:
        """
        Set leverage for a symbol with validation.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            leverage: Leverage value (e.g., '5', 5, 5.0)

        Returns:
            API response
        """
        # Validate inputs
        self._validate_symbol(symbol)
        leverage_str = self._validate_leverage(leverage)

        params = {
            "category": "linear",
            "symbol": symbol.upper(),
            "buyLeverage": leverage_str,
            "sellLeverage": leverage_str
        }

        result = self._make_request("POST", "/v5/position/set-leverage", params)

        # Log the API call
        self._log_trade(
            method="POST", endpoint="/v5/position/set-leverage", symbol=symbol.upper(),
            side="", qty="", order_type="", price=None,
            status="success" if result.get("retCode") == 0 else "error",
            ret_code=result.get("retCode", -1), ret_msg=result.get("retMsg", ""),
            latency_ms=0, attempt=1
        )

        return result

    def place_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        qty: Union[str, float, int],
        order_type: Union[str, OrderType] = OrderType.MARKET,
        price: Optional[Union[str, float, int]] = None,
        take_profit: Optional[Union[str, float, int]] = None,
        stop_loss: Optional[Union[str, float, int]] = None,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
        close_on_trigger: bool = False
    ) -> Dict[str, Any]:
        """
        Place an order with comprehensive validation.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Order side ('Buy' or 'Sell') or OrderSide enum
            qty: Order quantity
            order_type: Order type ('Market', 'Limit', etc.) or OrderType enum
            price: Order price (required for Limit orders)
            take_profit: Take profit price
            stop_loss: Stop loss price
            client_order_id: Custom client order ID
            reduce_only: Reduce position only flag
            close_on_trigger: Close position on trigger

        Returns:
            API response
        """
        # Comprehensive validation
        self._validate_symbol(symbol)
        side_normalized = self._validate_side(side)
        self._validate_order_type(order_type)
        qty_str = self._validate_quantity(qty)

        if price is not None:
            price_str = self._validate_price(price)
        else:
            price_str = None

        if take_profit is not None:
            tp_str = self._validate_price(take_profit)
        else:
            tp_str = None

        if stop_loss is not None:
            sl_str = self._validate_price(stop_loss)
        else:
            sl_str = None

        # Limit order requires price
        if order_type == "Limit" and price_str is None:
            raise ValueError("Price is required for Limit orders")

        # Validate client order ID if provided
        if client_order_id is not None:
            if not isinstance(client_order_id, str) or len(client_order_id) > 36:
                raise ValueError("Client order ID must be string with max 36 characters")

        # Build parameters
        params = {
            "category": "linear",
            "symbol": symbol.upper(),
            "side": side_normalized,
            "orderType": order_type,
            "qty": qty_str,
        }

        if price_str is not None:
            params["price"] = price_str
        if tp_str is not None:
            params["takeProfit"] = tp_str
        if sl_str is not None:
            params["stopLoss"] = sl_str
        if client_order_id is not None:
            params["orderLinkId"] = client_order_id
        if reduce_only:
            params["reduceOnly"] = True
        if close_on_trigger:
            params["closeOnTrigger"] = True

        start_time = time.time()
        result = self._make_request("POST", "/v5/order/create", params)
        latency_ms = (time.time() - start_time) * 1000

        # Log the order attempt
        self._log_trade(
            method="POST", endpoint="/v5/order/create", symbol=symbol.upper(),
            side=side_normalized, qty=qty_str, order_type=order_type,
            price=price_str, status="success" if result.get("retCode") == 0 else "error",
            ret_code=result.get("retCode", -1), ret_msg=result.get("retMsg", ""),
            latency_ms=latency_ms, attempt=1
        )

        return result

    def cancel_order(self, symbol: str, order_id: Optional[str] = None,
                    client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Cancel an order with validation."""
        self._validate_symbol(symbol)

        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")

        params = {
            "category": "linear",
            "symbol": symbol.upper()
        }
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["orderLinkId"] = client_order_id

        return self._make_request("POST", "/v5/order/cancel", params)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol."""
        self._validate_symbol(symbol)

        params = {
            "category": "linear",
            "symbol": symbol.upper()
        }
        return self._make_request("POST", "/v5/order/cancel-all", params)

    def get_order_status(self, symbol: str, order_id: Optional[str] = None,
                        client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Get order status with validation."""
        self._validate_symbol(symbol)

        if not order_id and not client_order_id:
            raise ValueError("Either order_id or client_order_id must be provided")

        params = {
            "category": "linear",
            "symbol": symbol.upper()
        }
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["orderLinkId"] = client_order_id

        return self._make_request("GET", "/v5/order/realtime", params)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get position for symbol with validation."""
        self._validate_symbol(symbol)

        params = {
            "category": "linear",
            "symbol": symbol.upper()
        }
        return self._make_request("GET", "/v5/position/list", params)

    def get_all_positions(self) -> Dict[str, Any]:
        """Get all positions."""
        params = {"category": "linear"}
        return self._make_request("GET", "/v5/position/list", params)

    def get_leverage(self, symbol: str) -> Dict[str, Any]:
        """Get leverage info for symbol."""
        return self.get_position(symbol)

    def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """Get wallet balance."""
        valid_account_types = ["UNIFIED", "CONTRACT", "SPOT"]
        if account_type not in valid_account_types:
            raise ValueError(f"Account type must be one of: {valid_account_types}")

        params = {"accountType": account_type}
        return self._make_request("GET", "/v5/account/wallet-balance", params)

    def get_tickers(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get market tickers with optional symbol filter."""
        params = {"category": "linear"}
        if symbol:
            self._validate_symbol(symbol)
            params["symbol"] = symbol.upper()
        return self._make_request("GET", "/v5/market/tickers", params)

    def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get open orders with optional symbol filter."""
        params = {"category": "linear"}
        if symbol:
            self._validate_symbol(symbol)
            params["symbol"] = symbol.upper()
        return self._make_request("GET", "/v5/order/realtime", params)

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Get order history with optional symbol filter."""
        params = {
            "category": "linear",
            "limit": min(limit, 200)  # API limit is 200
        }
        if symbol:
            self._validate_symbol(symbol)
            params["symbol"] = symbol.upper()
        return self._make_request("GET", "/v5/order/history", params)

    def validate_credentials(self) -> bool:
        """
        Validate that API credentials are properly configured and work.

        This method performs a test API call to verify credentials are valid.

        Returns:
            True if credentials are valid and work, False otherwise
        """
        # Basic format validation
        if not self.api_key or not self.api_secret:
            return False
        if len(self.api_key) <= 10 or len(self.api_secret) <= 10:
            return False

        # Test credentials with actual API call
        try:
            result = self.get_wallet_balance(account_type="UNIFIED")

            # Check if API call succeeded
            ret_code = result.get("retCode")
            if ret_code == 0:
                logger.info("ByBit API credentials validated successfully")
                return True
            else:
                ret_msg = result.get("retMsg", "Unknown error")
                logger.warning(f"ByBit API credential validation failed: {ret_msg} (retCode: {ret_code})")
                return False

        except Exception as e:
            logger.warning(f"ByBit API credential validation error: {e}")
            return False
