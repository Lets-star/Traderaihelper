from __future__ import annotations

import json
import math
import random
import time
from typing import Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .math_utils import Candle

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"

_TIMEFRAME_ALIASES: Dict[str, str] = {
    "1": "1m",
    "1m": "1m",
    "3": "3m",
    "3m": "3m",
    "5": "5m",
    "5m": "5m",
    "15": "15m",
    "15m": "15m",
    "30": "30m",
    "30m": "30m",
    "60": "1h",
    "1h": "1h",
    "120": "2h",
    "2h": "2h",
    "240": "4h",
    "4h": "4h",
    "360": "6h",
    "6h": "6h",
    "720": "12h",
    "12h": "12h",
    "1d": "1d",
    "1day": "1d",
    "d": "1d",
    "1w": "1w",
    "w": "1w",
}

_TIMEFRAME_TO_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}


def parse_symbol(symbol: str) -> str:
    if ":" in symbol:
        _, sym = symbol.split(":", 1)
        return sym.upper()
    return symbol.upper()


def timeframe_to_binance_interval(timeframe: str) -> str:
    key = timeframe.strip().lower()
    if key not in _TIMEFRAME_ALIASES:
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
    return _TIMEFRAME_ALIASES[key]


def timeframe_to_minutes(timeframe: str) -> int:
    interval = timeframe_to_binance_interval(timeframe)
    return _TIMEFRAME_TO_MINUTES[interval]


def interval_to_milliseconds(interval: str) -> int:
    minutes = timeframe_to_minutes(interval)
    return minutes * 60 * 1000


def fetch_klines(symbol: str, timeframe: str, limit: int = 500) -> List[Candle]:
    interval = timeframe_to_binance_interval(timeframe)
    parsed_symbol = parse_symbol(symbol)
    url = f"{BINANCE_BASE_URL}?symbol={parsed_symbol}&interval={interval}&limit={limit}"

    try:
        with urlopen(url) as response:
            raw_data = response.read()
    except HTTPError as exc:  # pragma: no cover - network handling
        raise RuntimeError(f"HTTP error while fetching klines: {exc.code} {exc.reason}") from exc
    except URLError as exc:  # pragma: no cover - network handling
        raise RuntimeError(f"Network error while fetching klines: {exc.reason}") from exc

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to decode Binance response as JSON") from exc

    candles: List[Candle] = []
    interval_ms = interval_to_milliseconds(interval)
    for entry in data:
        open_time = int(entry[0])
        close_time = open_time + interval_ms
        candle = Candle(
            open_time=open_time,
            close_time=close_time,
            open=float(entry[1]),
            high=float(entry[2]),
            low=float(entry[3]),
            close=float(entry[4]),
            volume=float(entry[5]),
        )
        candles.append(candle)
    return candles


def generate_synthetic_candles(symbol: str, timeframe: str, limit: int) -> List[Candle]:
    """Create deterministic synthetic OHLCV candles for offline scenarios."""
    interval = timeframe_to_binance_interval(timeframe)
    interval_ms = interval_to_milliseconds(interval)
    limit = max(1, limit)
    end_time = int(time.time() * 1000)
    start_time = end_time - interval_ms * limit
    seed = (hash(symbol) ^ hash(interval)) & 0xFFFFFFFF
    rng = random.Random(seed)

    base_price = max(5.0, 100 + (abs(hash(symbol)) % 500) / 10)
    price = base_price
    candles: List[Candle] = []

    for i in range(limit):
        open_time = start_time + i * interval_ms
        close_time = open_time + interval_ms
        drift = math.sin(i / 12.0) * 1.5
        change = drift + rng.uniform(-1.2, 1.2)
        open_price = max(0.1, price)
        close_price = max(0.1, open_price + change)
        high = max(open_price, close_price) + rng.uniform(0.05, 0.9)
        low = max(0.01, min(open_price, close_price) - rng.uniform(0.05, 0.9))
        volume = 100 + abs(change) * 40 + rng.uniform(0, 30)
        candles.append(
            Candle(
                open_time=open_time,
                close_time=close_time,
                open=open_price,
                high=high,
                low=low,
                close=close_price,
                volume=volume,
            )
        )
        price = close_price + rng.uniform(-0.3, 0.3)

    return candles


def ensure_ascending(candles: Sequence[Candle]) -> List[Candle]:
    ordered = sorted(candles, key=lambda c: c.open_time)
    return ordered


def latest_common_timestamp(series: Iterable[Candle]) -> int:
    latest = 0
    for candle in series:
        if candle.close_time > latest:
            latest = candle.close_time
    return latest

