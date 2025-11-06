"""CME Gap detection for CME Bitcoin and Ethereum futures."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .math_utils import Candle

YAHOO_CHART_URLS = (
    "https://query1.finance.yahoo.com/v8/finance/chart",
    "https://query2.finance.yahoo.com/v8/finance/chart",
)
DEFAULT_INTERVAL = "1h"
DEFAULT_RANGE = "3mo"
_INTERVAL_SECONDS = {
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
}

_CME_SYMBOL_MAP = {
    "BTC": "BTC=F",
    "ETH": "ETH=F",
}

_KNOWN_QUOTES = ("USDT", "USD", "USDC", "EUR", "PERP", "SPOT")

_CACHE: Dict[str, Tuple[float, List[Candle], str]] = {}
_CACHE_TIMEOUT = 300
_MAX_RETRY_ATTEMPTS = 4
_INITIAL_RETRY_DELAY = 1.5
_MAX_RETRY_DELAY = 10.0


def _extract_base_symbol(symbol: str) -> str:
    value = symbol.upper()
    if ":" in value:
        value = value.split(":", 1)[1]
    for quote in _KNOWN_QUOTES:
        if value.endswith(quote) and len(value) > len(quote):
            return value[: -len(quote)]
    return value


def _map_to_cme_ticker(symbol: str) -> Optional[str]:
    base = _extract_base_symbol(symbol)
    return _CME_SYMBOL_MAP.get(base)


def fetch_cme_candles(
    symbol: str,
    *,
    interval: str = DEFAULT_INTERVAL,
    range_period: str = DEFAULT_RANGE,
) -> Tuple[List[Candle], str]:
    ticker = _map_to_cme_ticker(symbol)
    if not ticker:
        raise RuntimeError(
            f"CME futures data is not available for symbol '{symbol}'. Supported assets: {sorted(_CME_SYMBOL_MAP)}"
        )

    cache_key = f"{ticker}_{interval}_{range_period}"
    current_time = time.time()
    
    if cache_key in _CACHE:
        cache_time, cached_candles, cached_ticker = _CACHE[cache_key]
        if current_time - cache_time < _CACHE_TIMEOUT:
            return cached_candles, cached_ticker

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://finance.yahoo.com/',
        'Origin': 'https://finance.yahoo.com',
    }
    
    raw_data: Optional[bytes] = None
    last_http_error: Optional[HTTPError] = None
    last_url_error: Optional[URLError] = None

    for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
        base_url = YAHOO_CHART_URLS[(attempt - 1) % len(YAHOO_CHART_URLS)]
        url = f"{base_url}/{ticker}?interval={interval}&range={range_period}"
        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=15) as response:
                raw_data = response.read()
            break
        except HTTPError as exc:
            last_http_error = exc
            if exc.code == 429 and cache_key in _CACHE:
                _, cached_candles, cached_ticker = _CACHE[cache_key]
                return cached_candles, cached_ticker
            if exc.code == 429 and attempt < _MAX_RETRY_ATTEMPTS:
                delay = min(_MAX_RETRY_DELAY, _INITIAL_RETRY_DELAY * (2 ** (attempt - 1)))
                time.sleep(delay)
                continue
            if cache_key in _CACHE:
                _, cached_candles, cached_ticker = _CACHE[cache_key]
                return cached_candles, cached_ticker
            break
        except URLError as exc:
            last_url_error = exc
            if cache_key in _CACHE:
                _, cached_candles, cached_ticker = _CACHE[cache_key]
                return cached_candles, cached_ticker
            if attempt < _MAX_RETRY_ATTEMPTS:
                delay = min(_MAX_RETRY_DELAY, _INITIAL_RETRY_DELAY * (2 ** (attempt - 1)))
                time.sleep(delay)
                continue
            break

    if raw_data is None:
        if cache_key in _CACHE:
            _, cached_candles, cached_ticker = _CACHE[cache_key]
            return cached_candles, cached_ticker
        if last_http_error is not None:
            raise RuntimeError(
                f"HTTP error while fetching CME futures data: {last_http_error.code} {last_http_error.reason}"
            ) from last_http_error
        if last_url_error is not None:
            raise RuntimeError(
                f"Network error while fetching CME futures data: {last_url_error.reason}"
            ) from last_url_error
        raise RuntimeError("Failed to fetch CME futures data: unknown error")

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to decode CME futures response as JSON") from exc

    chart = payload.get("chart", {})
    result = chart.get("result")
    if not result:
        error_info = chart.get("error")
        raise RuntimeError(f"CME futures response missing result data: {error_info}")

    serie = result[0]
    timestamps = serie.get("timestamp") or []
    indicators = serie.get("indicators") or {}
    quotes = indicators.get("quote") or []
    if not timestamps or not quotes:
        raise RuntimeError("CME futures response missing price series")

    quote = quotes[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    interval_seconds = _INTERVAL_SECONDS.get(interval, 3600)
    candles: List[Candle] = []
    for idx, ts in enumerate(timestamps):
        if idx >= len(opens) or idx >= len(highs) or idx >= len(lows) or idx >= len(closes):
            continue
        o = opens[idx]
        h = highs[idx]
        l = lows[idx]
        c = closes[idx]
        if None in (o, h, l, c):
            continue
        try:
            open_price = float(o)
            high_price = float(h)
            low_price = float(l)
            close_price = float(c)
        except (TypeError, ValueError):
            continue
        volume_value = 0.0
        if idx < len(volumes) and volumes[idx] is not None:
            try:
                volume_value = float(volumes[idx])
            except (TypeError, ValueError):
                volume_value = 0.0

        open_time = int(ts) * 1000
        close_time = open_time + interval_seconds * 1000
        candles.append(
            Candle(
                open_time=open_time,
                close_time=close_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume_value,
            )
        )

    if not candles:
        raise RuntimeError("No valid CME futures candles retrieved.")

    candles.sort(key=lambda c: c.open_time)
    _CACHE[cache_key] = (time.time(), candles, ticker)
    return candles, ticker


def detect_cme_gaps(candles: List[Candle]) -> List[Dict[str, object]]:
    """Detect CME gaps in historical CME futures price data."""
    gaps: List[Dict[str, object]] = []

    for i in range(1, len(candles)):
        prev_candle = candles[i - 1]
        curr_candle = candles[i]

        prev_time = datetime.fromtimestamp(prev_candle.close_time / 1000, tz=timezone.utc)
        curr_time = datetime.fromtimestamp(curr_candle.open_time / 1000, tz=timezone.utc)

        time_diff_hours = (curr_candle.open_time - prev_candle.close_time) / (1000 * 3600)

        if time_diff_hours > 24:
            gap_up = curr_candle.open > prev_candle.close
            gap_down = curr_candle.open < prev_candle.close

            if gap_up:
                gap_top = curr_candle.open
                gap_bottom = prev_candle.close
                gap_size = gap_top - gap_bottom
                gap_size_pct = (gap_size / prev_candle.close) * 100 if prev_candle.close else 0

                if gap_size_pct > 0.1:
                    is_filled = False
                    filled_at_index: Optional[int] = None
                    for j in range(i + 1, len(candles)):
                        if candles[j].low <= gap_bottom:
                            is_filled = True
                            filled_at_index = j
                            break

                    gaps.append(
                        {
                            "type": "gap_up",
                            "created_index": i,
                            "created_timestamp": curr_candle.open_time,
                            "created_time_iso": curr_time.isoformat(),
                            "gap_top": gap_top,
                            "gap_bottom": gap_bottom,
                            "gap_size": gap_size,
                            "gap_size_pct": gap_size_pct,
                            "is_filled": is_filled,
                            "filled_at_index": filled_at_index,
                            "filled_at_timestamp": candles[filled_at_index].close_time if filled_at_index is not None else None,
                        }
                    )

            elif gap_down:
                gap_top = prev_candle.close
                gap_bottom = curr_candle.open
                gap_size = gap_top - gap_bottom
                gap_size_pct = (gap_size / prev_candle.close) * 100 if prev_candle.close else 0

                if gap_size_pct > 0.1:
                    is_filled = False
                    filled_at_index = None
                    for j in range(i + 1, len(candles)):
                        if candles[j].high >= gap_top:
                            is_filled = True
                            filled_at_index = j
                            break

                    gaps.append(
                        {
                            "type": "gap_down",
                            "created_index": i,
                            "created_timestamp": curr_candle.open_time,
                            "created_time_iso": curr_time.isoformat(),
                            "gap_top": gap_top,
                            "gap_bottom": gap_bottom,
                            "gap_size": gap_size,
                            "gap_size_pct": gap_size_pct,
                            "is_filled": is_filled,
                            "filled_at_index": filled_at_index,
                            "filled_at_timestamp": candles[filled_at_index].close_time if filled_at_index is not None else None,
                        }
                    )

    return gaps


def get_nearest_cme_gaps(symbol: str, current_price: float, max_gaps: int = 5) -> Dict[str, object]:
    """Get the nearest unfilled CME gaps relative to the current spot price."""
    try:
        candles, ticker = fetch_cme_candles(symbol)
    except RuntimeError as exc:
        return {
            "symbol": symbol,
            "cme_symbol": None,
            "data_source": "cme",
            "error": str(exc),
            "total_unfilled_gaps": 0,
            "total_gaps_above": 0,
            "total_gaps_below": 0,
            "nearest_gaps_above": [],
            "nearest_gaps_below": [],
        }

    all_gaps = detect_cme_gaps(candles)
    unfilled_gaps = [gap for gap in all_gaps if not gap["is_filled"]]

    gaps_above: List[Dict[str, object]] = []
    gaps_below: List[Dict[str, object]] = []

    for gap in unfilled_gaps:
        gap_bottom = gap["gap_bottom"]
        gap_top = gap["gap_top"]

        if current_price <= 0:
            distance_pct = None
            distance = None
        else:
            distance_pct = ((gap_bottom - current_price) / current_price) * 100
            distance = gap_bottom - current_price

        if gap_bottom > current_price:
            gaps_above.append(
                {
                    **gap,
                    "distance_to_price": distance,
                    "distance_pct": distance_pct,
                }
            )
        elif gap_top < current_price:
            distance_pct = ((current_price - gap_top) / current_price) * 100 if current_price else None
            gaps_below.append(
                {
                    **gap,
                    "distance_to_price": current_price - gap_top if current_price else None,
                    "distance_pct": distance_pct,
                }
            )
        else:
            gaps_below.append(
                {
                    **gap,
                    "distance_to_price": 0.0,
                    "distance_pct": 0.0,
                    "currently_inside": True,
                }
            )

    gaps_above.sort(key=lambda x: x["distance_to_price"] if x["distance_to_price"] is not None else float("inf"))
    gaps_below.sort(key=lambda x: x["distance_to_price"] if x["distance_to_price"] is not None else float("inf"))

    latest_candle = candles[-1]
    latest_time_iso = datetime.fromtimestamp(latest_candle.close_time / 1000, tz=timezone.utc).isoformat()

    return {
        "symbol": symbol,
        "cme_symbol": ticker,
        "data_source": "cme_yahoo_finance",
        "total_unfilled_gaps": len(unfilled_gaps),
        "total_gaps_above": len(gaps_above),
        "total_gaps_below": len(gaps_below),
        "nearest_gaps_above": gaps_above[:max_gaps],
        "nearest_gaps_below": gaps_below[:max_gaps],
        "latest_cme_close": latest_candle.close,
        "latest_cme_timestamp": latest_candle.close_time,
        "latest_cme_time_iso": latest_time_iso,
    }
