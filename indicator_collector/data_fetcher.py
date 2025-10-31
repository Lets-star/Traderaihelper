from __future__ import annotations

import json
import math
import random
import time
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from .math_utils import Candle

BINANCE_BASE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"

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


def _section_stats(levels_list: Sequence[Tuple[float, float]], levels: int) -> Dict[str, object]:
    selected = list(levels_list[:levels])
    total_volume = sum(quantity for _, quantity in selected)
    weighted_price = sum(price * quantity for price, quantity in selected)
    average_price = weighted_price / total_volume if total_volume else None
    return {
        "levels": min(levels, len(levels_list)),
        "total_volume": total_volume,
        "weighted_price": average_price,
    }


def _aggregate_order_book_depth(bids: Sequence[Tuple[float, float]], asks: Sequence[Tuple[float, float]]) -> Dict[str, object]:
    bids_sorted = sorted(bids, key=lambda item: item[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda item: item[0])

    best_bid = bids_sorted[0][0] if bids_sorted else None
    best_ask = asks_sorted[0][0] if asks_sorted else None

    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    mid_price = (best_bid + best_ask) / 2 if spread is not None else None

    total_bid_volume = sum(quantity for _, quantity in bids_sorted)
    total_ask_volume = sum(quantity for _, quantity in asks_sorted)

    sections = {"bids": {}, "asks": {}}
    for section_size in (5, 10, 20):
        key = f"top_{section_size}"
        sections["bids"][key] = _section_stats(bids_sorted, section_size)
        sections["asks"][key] = _section_stats(asks_sorted, section_size)

    price_levels = {
        "1%": {"bid_volume": 0.0, "ask_volume": 0.0},
        "2%": {"bid_volume": 0.0, "ask_volume": 0.0},
        "5%": {"bid_volume": 0.0, "ask_volume": 0.0},
    }
    if mid_price is not None and mid_price > 0:
        for price, volume in bids_sorted:
            distance_pct = abs((mid_price - price) / mid_price) * 100
            if distance_pct <= 1:
                price_levels["1%"]["bid_volume"] += volume
            if distance_pct <= 2:
                price_levels["2%"]["bid_volume"] += volume
            if distance_pct <= 5:
                price_levels["5%"]["bid_volume"] += volume
        for price, volume in asks_sorted:
            distance_pct = abs((price - mid_price) / mid_price) * 100
            if distance_pct <= 1:
                price_levels["1%"]["ask_volume"] += volume
            if distance_pct <= 2:
                price_levels["2%"]["ask_volume"] += volume
            if distance_pct <= 5:
                price_levels["5%"]["ask_volume"] += volume
    else:
        price_levels = {}

    top10_bid = sections["bids"]["top_10"]["total_volume"]
    top10_ask = sections["asks"]["top_10"]["total_volume"]
    bid_ask_ratio_top10 = (top10_bid / top10_ask) if top10_ask else None
    imbalance_top10 = top10_bid - top10_ask if (top10_bid or top10_ask) else None

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid_price": mid_price,
        "total_bid_volume": total_bid_volume,
        "total_ask_volume": total_ask_volume,
        "sections": sections,
        "price_levels": price_levels,
        "bid_ask_ratio_top10": bid_ask_ratio_top10,
        "volume_imbalance_top10": imbalance_top10,
        "raw_levels": {
            "bids": bids_sorted[:20],
            "asks": asks_sorted[:20],
        },
        "total_levels": {
            "bids": len(bids_sorted),
            "asks": len(asks_sorted),
        },
    }


def fetch_order_book(symbol: str, limit: int = 100) -> Dict[str, object]:
    parsed_symbol = parse_symbol(symbol)
    constrained_limit = max(5, min(limit, 500))
    url = f"{BINANCE_DEPTH_URL}?symbol={parsed_symbol}&limit={constrained_limit}"

    try:
        with urlopen(url) as response:
            raw_data = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error while fetching order book: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while fetching order book: {exc.reason}") from exc

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Failed to decode Binance depth response") from exc

    bids = [(float(price), float(volume)) for price, volume in data.get("bids", [])]
    asks = [(float(price), float(volume)) for price, volume in data.get("asks", [])]

    aggregates = _aggregate_order_book_depth(bids, asks)
    aggregates.update(
        {
            "symbol": parsed_symbol,
            "limit": constrained_limit,
            "last_update_id": data.get("lastUpdateId"),
            "snapshot_time": int(time.time() * 1000),
            "source": "binance",
        }
    )
    return aggregates


def generate_synthetic_order_book(symbol: str, reference_price: float, limit: int = 100) -> Dict[str, object]:
    constrained_limit = max(5, min(limit, 500))
    base_price = reference_price if reference_price and reference_price > 0 else 100.0
    seed = (hash(symbol) ^ int(base_price * 100)) & 0xFFFFFFFF
    rng = random.Random(seed)
    price_step = max(base_price * 0.0008, 0.05)

    bids: List[Tuple[float, float]] = []
    asks: List[Tuple[float, float]] = []
    for i in range(constrained_limit):
        offset = price_step * (i + 1)
        bid_price = max(0.01, base_price - offset)
        ask_price = base_price + offset
        bid_qty = rng.uniform(0.8, 3.2) * (1 + 0.05 * i)
        ask_qty = rng.uniform(0.8, 3.2) * (1 + 0.05 * i)
        bids.append((round(bid_price, 4), round(bid_qty, 3)))
        asks.append((round(ask_price, 4), round(ask_qty, 3)))

    aggregates = _aggregate_order_book_depth(bids, asks)
    aggregates.update(
        {
            "symbol": parse_symbol(symbol),
            "limit": constrained_limit,
            "last_update_id": rng.randint(1, 1_000_000_000),
            "snapshot_time": int(time.time() * 1000),
            "source": "synthetic",
        }
    )
    return aggregates

