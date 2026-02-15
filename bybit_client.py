import time
import hmac
import hashlib
import json
import logging
import asyncio
from typing import Any, Dict, Optional, List
import aiohttp
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class ByBitClient:
    """
    Async client for ByBit V5 API (Testnet).
    Supports Linear Futures (USDT Perpetuals).
    """

    TESTNET_API_URL = "https://api-testnet.bybit.com"
    MAINNET_API_URL = "https://api.bybit.com"  # Not used for now, but good to have

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_API_URL if testnet else self.MAINNET_API_URL
        self.recv_window = "5000"
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def _generate_signature(self, timestamp: str, payload: str) -> str:
        """
        Generate HMAC-SHA256 signature.
        param_str is timestamp + api_key + recv_window + payload
        """
        param_str = timestamp + self.api_key + self.recv_window + payload
        return hmac.new(
            self.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Internal request wrapper with retry logic.
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        # Prepare payload
        payload = ""
        if method == "POST":
            payload = json.dumps(params) if params else "{}"
        elif method == "GET" and params:
            # For GET, params are query string, but signature uses query string format without ?
            # However, ByBit V5 GET requests: 
            # "Sort the parameters by key in ascending order... append to timestamp+key+recvWindow"
            # Actually, standard is: timestamp + key + recvWindow + queryString
            # Query string is key=value&key2=value2
            pass

        # Prepare headers
        timestamp = str(int(time.time() * 1000))
        
        if method == "GET":
             # Sort and encode params
            query_string = ""
            if params:
                # ByBit requires params sorted by key
                # But aiohttp params handling might differ. 
                # Let's manually construct query string for signing to be safe and consistent.
                sorted_keys = sorted(params.keys())
                query_parts = []
                for key in sorted_keys:
                    query_parts.append(f"{key}={params[key]}")
                query_string = "&".join(query_parts)
            
            signature = self._generate_signature(timestamp, query_string)
            full_url = f"{url}?{query_string}" if query_string else url
            data = None
        else:
            # POST
            signature = self._generate_signature(timestamp, payload)
            full_url = url
            data = payload

        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "Content-Type": "application/json"
        }

        retries = 3
        last_exception = None

        for attempt in range(retries):
            try:
                start_time = time.time()
                async with session.request(method, full_url, headers=headers, data=data) as response:
                    latency = (time.time() - start_time) * 1000
                    text = await response.text()
                    
                    logger.info(f"ByBit API call: {method} {endpoint} | Status: {response.status} | Latency: {latency:.2f}ms")
                    logger.debug(f"Request: {data if method=='POST' else full_url} | Response: {text}")

                    if response.status != 200:
                        logger.error(f"ByBit API error (HTTP {response.status}): {text}")
                        # Don't retry client errors (4xx) unless it's rate limit
                        if 400 <= response.status < 500 and response.status != 429:
                            try:
                                return json.loads(text)
                            except:
                                return {"retCode": -1, "retMsg": f"HTTP {response.status}: {text}"}
                    
                    try:
                        resp_json = json.loads(text)
                    except json.JSONDecodeError:
                         return {"retCode": -1, "retMsg": f"Invalid JSON: {text}"}
                    
                    if resp_json.get("retCode") != 0:
                        # Business logic error
                        logger.warning(f"ByBit business error: {resp_json}")
                        # Not necessarily need to retry unless it's system error
                    
                    return resp_json

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                logger.warning(f"ByBit connection error (attempt {attempt+1}/{retries}): {e}")
                await asyncio.sleep(0.5 * (2 ** attempt)) # Exponential backoff

        return {"retCode": -1, "retMsg": f"Request failed after {retries} retries: {last_exception}"}

    async def set_leverage(self, symbol: str, leverage: str) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        """
        return await self._request("POST", "/v5/position/set-leverage", {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        })

    async def place_order(
        self, 
        symbol: str, 
        side: str, 
        qty: str, 
        order_type: str = "Market",
        price: Optional[str] = None,
        take_profit: Optional[str] = None, 
        stop_loss: Optional[str] = None, 
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place an order.
        side: "Buy" or "Sell"
        """
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": order_type,
            "qty": str(qty),
        }
        if price:
            params["price"] = str(price)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if client_order_id:
            params["orderLinkId"] = client_order_id

        return await self._request("POST", "/v5/order/create", params)

    async def cancel_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        params = {
            "category": "linear",
            "symbol": symbol
        }
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["orderLinkId"] = client_order_id
            
        return await self._request("POST", "/v5/order/cancel", params)

    async def get_order_status(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Dict[str, Any]:
        params = {
            "category": "linear",
            "symbol": symbol
        }
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["orderLinkId"] = client_order_id

        return await self._request("GET", "/v5/order/realtime", params)

    async def get_position(self, symbol: str) -> Dict[str, Any]:
        params = {
            "category": "linear",
            "symbol": symbol
        }
        return await self._request("GET", "/v5/position/list", params)

    async def get_leverage(self, symbol: str) -> Dict[str, Any]:
        # Leverage info is in get_position response
        return await self.get_position(symbol)

    async def get_wallet_balance(self, account_type: str = "UNIFIED") -> Dict[str, Any]:
        """
        Get wallet balance.
        account_type: UNIFIED, CONTRACT, SPOT
        """
        params = {"accountType": account_type}
        return await self._request("GET", "/v5/account/wallet-balance", params)

    async def get_tickers(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get latest market tickers."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/market/tickers", params)

    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get open orders."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/order/realtime", params)

    def validate_credentials(self) -> bool:
        """Validate that API credentials are configured."""
        return bool(self.api_key and self.api_secret and len(self.api_key) > 10 and len(self.api_secret) > 10)
