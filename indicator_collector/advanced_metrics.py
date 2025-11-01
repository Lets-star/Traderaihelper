"""Advanced analytics helpers for extended dashboard tabs."""

from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .math_utils import Candle, atr

try:  # Avoid circular imports at runtime
    from .indicator_metrics import SimulationSummary
except Exception:  # pragma: no cover - type checking only
    SimulationSummary = None  # type: ignore


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _pivot_seed(candles: Sequence[Candle]) -> int:
    if not candles:
        return 0
    reference = candles[-1].close if candles[-1].close != 0 else candles[-1].high
    return int(abs(reference) * 1_000_000) % (2**32 - 1)


def _format_timestamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def calculate_volume_analysis(candles: Sequence[Candle]) -> Dict[str, object]:
    if not candles:
        return {
            "vpvr": {"levels": [], "value_area": {}, "poc": None, "total_volume": 0},
            "cvd": {"latest": 0, "change": 0, "series": []},
            "delta": {"latest": 0, "average": 0, "series": []},
        }

    high_price = max(c.high for c in candles)
    low_price = min(c.low for c in candles)
    price_range = max(high_price - low_price, max(1e-6, high_price * 0.0001))

    bin_count = _clamp(len(candles) // 4, 12, 48)
    bin_size = price_range / bin_count
    volumes = [0.0 for _ in range(int(bin_count))]

    for candle in candles:
        typical_price = (candle.high + candle.low + candle.close + candle.open) / 4
        idx = int((typical_price - low_price) / bin_size)
        idx = max(0, min(len(volumes) - 1, idx))
        volumes[idx] += candle.volume

    total_volume = sum(volumes) or 1.0
    level_data = []
    for idx, volume in enumerate(volumes):
        price_mid = low_price + (idx + 0.5) * bin_size
        level_data.append(
            {
                "price": round(price_mid, 4),
                "volume": volume,
                "percentage": round(volume / total_volume * 100, 2),
            }
        )

    sorted_levels = sorted(level_data, key=lambda item: item["volume"], reverse=True)
    poc_level = sorted_levels[0]["price"] if sorted_levels else None

    value_area_volume = total_volume * 0.7
    cumulative = 0.0
    area_prices: List[float] = []
    for level in sorted_levels:
        cumulative += level["volume"]
        area_prices.append(level["price"])
        if cumulative >= value_area_volume:
            break

    value_area = {
        "high": round(max(area_prices), 4) if area_prices else poc_level,
        "low": round(min(area_prices), 4) if area_prices else poc_level,
    }

    rng = random.Random(_pivot_seed(candles))
    cvd_series = []
    delta_series = []
    cumulative = 0.0

    for candle in candles:
        body = candle.close - candle.open
        direction = 1 if body > 0 else -1 if body < 0 else 0
        body_strength = abs(body) / max((candle.high - candle.low) or 1e-6, 1e-6)
        aggression = 0.5 + direction * 0.2 * body_strength + rng.uniform(-0.05, 0.05)
        aggression = _clamp(aggression, 0.1, 0.9)

        buy_volume = candle.volume * aggression
        sell_volume = candle.volume - buy_volume
        delta = buy_volume - sell_volume
        cumulative += delta

        cvd_series.append(
            {
                "timestamp": candle.close_time,
                "time_iso": _format_timestamp(candle.close_time),
                "value": cumulative,
                "delta": delta,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
            }
        )

        market_pct = 0.35 + body_strength * 0.4 + rng.uniform(-0.05, 0.05)
        market_pct = _clamp(market_pct, 0.15, 0.95)
        market_orders = candle.volume * market_pct
        limit_orders = candle.volume - market_orders
        imbalance = market_orders - limit_orders
        imbalance_ratio = (market_orders / limit_orders) if limit_orders else None

        delta_series.append(
            {
                "timestamp": candle.close_time,
                "time_iso": _format_timestamp(candle.close_time),
                "delta": imbalance,
                "market_orders": market_orders,
                "limit_orders": limit_orders,
                "imbalance_ratio": imbalance_ratio,
            }
        )

    latest_delta = delta_series[-1]["delta"] if delta_series else 0
    average_delta = statistics.fmean(item["delta"] for item in delta_series[-20:]) if delta_series else 0

    cvd_change = 0.0
    if len(cvd_series) >= 2:
        cvd_change = cvd_series[-1]["value"] - cvd_series[-2]["value"]

    return {
        "vpvr": {
            "levels": sorted_levels[:15],
            "poc": round(poc_level, 4) if poc_level else None,
            "total_volume": total_volume,
            "value_area": value_area,
        },
        "cvd": {
            "latest": cvd_series[-1]["value"] if cvd_series else 0,
            "change": cvd_change,
            "series": cvd_series[-30:],
        },
        "delta": {
            "latest": latest_delta,
            "average": average_delta,
            "series": delta_series[-30:],
        },
    }


def calculate_market_structure(candles: Sequence[Candle]) -> Dict[str, object]:
    if len(candles) < 7:
        return {
            "trend": "neutral",
            "swing_points": {"hh": [], "hl": [], "lh": [], "ll": []},
            "key_levels": {"support": [], "resistance": []},
            "liquidity_zones": [],
        }

    lookback = max(2, min(5, len(candles) // 15))
    swing_highs: List[Dict[str, object]] = []
    swing_lows: List[Dict[str, object]] = []

    for i in range(lookback, len(candles) - lookback):
        high = candles[i].high
        low = candles[i].low
        if all(high >= candles[j].high for j in range(i - lookback, i + lookback + 1)):
            swing_highs.append(
                {
                    "timestamp": candles[i].close_time,
                    "time_iso": _format_timestamp(candles[i].close_time),
                    "price": high,
                    "type": "swing_high",
                }
            )
        if all(low <= candles[j].low for j in range(i - lookback, i + lookback + 1)):
            swing_lows.append(
                {
                    "timestamp": candles[i].close_time,
                    "time_iso": _format_timestamp(candles[i].close_time),
                    "price": low,
                    "type": "swing_low",
                }
            )

    def _label_points(points: List[Dict[str, object]], label_up: str, label_down: str) -> List[Dict[str, object]]:
        labeled: List[Dict[str, object]] = []
        for prev, curr in zip(points, points[1:]):
            if curr["price"] > prev["price"]:
                labeled.append({**curr, "structure": label_up})
            elif curr["price"] < prev["price"]:
                labeled.append({**curr, "structure": label_down})
        return labeled[-5:]

    hh = _label_points(swing_highs, "HH", "LH")
    hl = _label_points(swing_lows, "HL", "LL")

    trend = "neutral"
    if any(p.get("structure") == "HH" for p in hh) and any(p.get("structure") == "HL" for p in hl):
        trend = "bullish"
    elif any(p.get("structure") == "LH" for p in hh) and any(p.get("structure") == "LL" for p in hl):
        trend = "bearish"

    support_levels = sorted((point["price"] for point in swing_lows[-5:]), reverse=True)
    resistance_levels = sorted((point["price"] for point in swing_highs[-5:]))

    support = [
        {
            "price": round(price, 4),
            "strength": round((idx + 1) / len(support_levels), 2),
        }
        for idx, price in enumerate(support_levels[:3])
    ]
    resistance = [
        {
            "price": round(price, 4),
            "strength": round((idx + 1) / len(resistance_levels), 2),
        }
        for idx, price in enumerate(resistance_levels[:3])
    ]

    return {
        "trend": trend,
        "swing_points": {
            "hh": hh,
            "hl": [p for p in hl if p.get("structure") == "HL"],
            "lh": [p for p in hh if p.get("structure") == "LH"],
            "ll": [p for p in hl if p.get("structure") == "LL"],
        },
        "key_levels": {"support": support, "resistance": resistance},
    }


def detect_liquidity_zones(volume_analysis: Dict[str, object], last_close: float) -> List[Dict[str, object]]:
    vpvr = volume_analysis.get("vpvr", {})
    levels = vpvr.get("levels", [])
    total_volume = vpvr.get("total_volume", 0) or 1.0

    zones: List[Dict[str, object]] = []
    threshold = total_volume / max(len(levels), 1) * 1.5

    for level in levels:
        if level["volume"] >= threshold:
            zone_type = "resistance" if level["price"] > last_close else "support"
            zones.append(
                {
                    "type": zone_type,
                    "price": level["price"],
                    "volume_ratio": round(level["volume"] / total_volume, 4),
                }
            )
    return zones[:10]


def calculate_fundamental_metrics(candles: Sequence[Candle]) -> Dict[str, object]:
    if len(candles) < 2:
        return {
            "funding_rate": {},
            "open_interest": {},
            "long_short_ratio": {},
            "block_trades": [],
        }

    rng = random.Random(_pivot_seed(candles) + 17)
    latest = candles[-1]
    prev = candles[-2]
    momentum = (latest.close - prev.close) / prev.close if prev.close else 0

    funding_rate = 0.0001 + momentum * 0.02 + rng.uniform(-0.0003, 0.0003)
    funding_rate = _clamp(funding_rate, -0.004, 0.004)
    predicted = funding_rate + rng.uniform(-0.0001, 0.0001)
    annualized = funding_rate * 3 * 365

    total_volume = sum(c.volume for c in candles[-50:])
    base_oi = 500_000_000 + total_volume * rng.uniform(40, 120)
    oi_change_pct = momentum * 80 + rng.uniform(-5, 5)

    long_bias = 0.5 + momentum * 1.5 + rng.uniform(-0.1, 0.1)
    long_bias = _clamp(long_bias, 0.2, 0.8)
    short_bias = 1 - long_bias

    volumes = [c.volume for c in candles[-50:]]
    mean_volume = statistics.fmean(volumes)
    stdev_volume = statistics.pstdev(volumes) if len(volumes) > 1 else 0
    block_level = mean_volume + stdev_volume * 2

    block_trades = [
        {
            "timestamp": candle.close_time,
            "time_iso": _format_timestamp(candle.close_time),
            "price": candle.close,
            "volume": candle.volume,
            "side": "buy" if candle.close >= candle.open else "sell",
        }
        for candle in candles[-100:]
        if candle.volume >= block_level
    ][:8]

    return {
        "funding_rate": {
            "current": round(funding_rate, 6),
            "predicted": round(predicted, 6),
            "annualized": round(annualized, 2),
        },
        "open_interest": {
            "current": round(base_oi, 0),
            "change_pct": round(oi_change_pct, 2),
        },
        "long_short_ratio": {
            "long": round(long_bias, 3),
            "short": round(short_bias, 3),
            "ratio": round(long_bias / short_bias if short_bias else float("inf"), 2),
        },
        "block_trades": block_trades,
    }


def calculate_breadth_metrics(candles: Sequence[Candle]) -> Dict[str, object]:
    if len(candles) < 2:
        return {
            "btc_dominance": 0,
            "sp500_correlation": 0,
            "nasdaq_correlation": 0,
            "fear_greed_index": 50,
            "regime": "neutral",
        }

    rng = random.Random(_pivot_seed(candles) + 73)
    returns = [
        (candles[i].close - candles[i - 1].close) / candles[i - 1].close
        for i in range(1, len(candles))
        if candles[i - 1].close
    ]
    avg_return = statistics.fmean(returns[-30:]) if returns else 0

    btc_dominance = 42 + avg_return * 120 + rng.uniform(-3, 3)
    btc_dominance = _clamp(btc_dominance, 30, 70)

    sp_corr = 0.4 + avg_return * 5 + rng.uniform(-0.2, 0.2)
    nas_corr = 0.6 + avg_return * 4 + rng.uniform(-0.15, 0.15)
    sp_corr = _clamp(sp_corr, -1, 1)
    nas_corr = _clamp(nas_corr, -1, 1)

    fear_greed = 50 + avg_return * 450 + rng.uniform(-10, 10)
    fear_greed = _clamp(fear_greed, 0, 100)

    if fear_greed >= 70:
        regime = "Extreme Greed"
    elif fear_greed >= 55:
        regime = "Greed"
    elif fear_greed <= 30:
        regime = "Extreme Fear"
    elif fear_greed <= 45:
        regime = "Fear"
    else:
        regime = "Neutral"

    return {
        "btc_dominance": round(btc_dominance, 2),
        "sp500_correlation": round(sp_corr, 2),
        "nasdaq_correlation": round(nas_corr, 2),
        "fear_greed_index": round(fear_greed, 1),
        "regime": regime,
    }


def calculate_patterns_and_waves(
    candles: Sequence[Candle],
    market_structure: Dict[str, object],
    orderbook_data: Optional[Dict[str, object]],
) -> Dict[str, object]:
    if not candles:
        return {"elliott": {}, "orderbook_clusters": [], "liquidity_anomalies": []}

    swings = market_structure.get("swing_points", {}) if market_structure else {}
    wave_points = []
    for key in ("hh", "hl", "lh", "ll"):
        entries = swings.get(key, [])
        wave_points.extend(entries)
    wave_points.sort(key=lambda item: item.get("timestamp", 0))

    wave_count = min(5, len(wave_points))
    trend = market_structure.get("trend") if market_structure else "neutral"
    if trend == "bullish":
        wave_label = f"Impulse Wave {wave_count}" if wave_count else "Impulse"
        structure_type = "impulse"
    elif trend == "bearish":
        wave_label = f"Corrective Wave {wave_count}" if wave_count else "Corrective"
        structure_type = "corrective"
    else:
        wave_label = "Sideways"
        structure_type = "indecision"

    clusters: List[Dict[str, object]] = []
    if orderbook_data:
        raw = orderbook_data.get("raw_levels", {})
        bids = raw.get("bids", [])
        asks = raw.get("asks", [])
        if bids:
            avg_bid = statistics.fmean(volume for _, volume in bids)
            for price, volume in bids[:10]:
                if volume >= avg_bid * 1.5:
                    clusters.append(
                        {
                            "side": "bid",
                            "price": price,
                            "volume": volume,
                            "strength": round(volume / avg_bid, 2) if avg_bid else 0,
                        }
                    )
        if asks:
            avg_ask = statistics.fmean(volume for _, volume in asks)
            for price, volume in asks[:10]:
                if volume >= avg_ask * 1.5:
                    clusters.append(
                        {
                            "side": "ask",
                            "price": price,
                            "volume": volume,
                            "strength": round(volume / avg_ask, 2) if avg_ask else 0,
                        }
                    )
        clusters.sort(key=lambda item: item["strength"], reverse=True)
        clusters = clusters[:10]

    anomalies: List[Dict[str, object]] = []
    recent = candles[-20:]
    if recent:
        avg_volume = statistics.fmean(c.volume for c in recent)
        for candle in recent:
            if avg_volume and candle.volume > avg_volume * 4:
                anomalies.append(
                    {
                        "timestamp": candle.close_time,
                        "time_iso": _format_timestamp(candle.close_time),
                        "price": candle.close,
                        "type": "volume_spike",
                        "severity": round(candle.volume / avg_volume, 2),
                        "description": "Volume spike indicates potential liquidity sweep",
                    }
                )
            wick = abs(candle.high - candle.low)
            body = abs(candle.close - candle.open)
            if wick > 0 and body / wick < 0.15:
                anomalies.append(
                    {
                        "timestamp": candle.close_time,
                        "time_iso": _format_timestamp(candle.close_time),
                        "price": (candle.high + candle.low) / 2,
                        "type": "liquidity_wick",
                        "severity": round(wick / (body or 1), 2),
                        "description": "Long wick suggests stop-run activity",
                    }
                )
        if orderbook_data:
            imbalance = orderbook_data.get("volume_imbalance_top10")
            if imbalance and avg_volume and abs(imbalance) > avg_volume * 2:
                anomalies.append(
                    {
                        "timestamp": candles[-1].close_time,
                        "time_iso": _format_timestamp(candles[-1].close_time),
                        "price": candles[-1].close,
                        "type": "orderbook_imbalance",
                        "severity": round(abs(imbalance) / avg_volume, 2),
                        "description": "Orderbook imbalance exceeds recent volume",
                    }
                )

    return {
        "elliott": {
            "wave_count": wave_count,
            "label": wave_label,
            "structure": structure_type,
            "pivot_points": wave_points[-8:],
        },
        "orderbook_clusters": clusters,
        "liquidity_anomalies": anomalies[:10],
    }


def calculate_trade_signal_plan(
    summary: SimulationSummary,
    candles: Sequence[Candle],
    leverage: float = 10.0,
    risk_pct: float = 1.0,
    account_balance: float = 10_000.0,
    commission_rate: float = 0.0006,
) -> Dict[str, object]:
    if not candles:
        return {}

    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]
    atr_values = atr(highs, lows, closes, 14)
    latest_atr = next((value for value in reversed(atr_values) if not math.isnan(value)), None)
    if latest_atr is None:
        latest_atr = closes[-1] * 0.01

    if summary and summary.signals:
        last_signal = summary.signals[-1]
        signal_type = "BUY" if last_signal.signal_type == "bullish" else "SELL"
        entry_price = last_signal.price
        timestamp = last_signal.timestamp
    else:
        signal_type = "NEUTRAL"
        entry_price = closes[-1]
        timestamp = candles[-1].close_time

    if signal_type == "BUY":
        stop_loss = entry_price - latest_atr * 2
        take_profit = [entry_price + latest_atr * mult for mult in (3, 5, 8)]
    elif signal_type == "SELL":
        stop_loss = entry_price + latest_atr * 2
        take_profit = [entry_price - latest_atr * mult for mult in (3, 5, 8)]
    else:
        stop_loss = entry_price - latest_atr
        take_profit = [entry_price + latest_atr, entry_price + latest_atr * 2]

    risk_amount = account_balance * (risk_pct / 100)
    risk_per_unit = abs(entry_price - stop_loss)
    position_size = (risk_amount / risk_per_unit) * leverage if risk_per_unit else 0
    position_notional = position_size * entry_price / leverage
    commission = position_notional * commission_rate * 2

    targets = []
    for tp in take_profit:
        gross = abs(tp - entry_price) * position_size
        targets.append(
            {
                "price": round(tp, 4),
                "gross_pnl": gross,
                "net_pnl": gross - commission,
            }
        )

    max_loss = risk_per_unit * position_size + commission
    reward_risk = (targets[0]["gross_pnl"] / max_loss) if max_loss else 0

    return {
        "signal": {
            "type": signal_type,
            "timestamp": timestamp,
            "time_iso": _format_timestamp(timestamp),
            "entry_price": round(entry_price, 4),
        },
        "risk": {
            "atr": latest_atr,
            "stop_loss": round(stop_loss, 4),
            "risk_amount": round(risk_amount, 2),
            "max_loss": round(max_loss, 2),
        },
        "position": {
            "leverage": leverage,
            "position_size": round(position_size, 4),
            "notional": round(position_notional, 2),
            "commission_estimate": round(commission, 2),
            "reward_risk": round(reward_risk, 2),
        },
        "targets": targets,
    }


def compute_advanced_metrics(
    summary: SimulationSummary,
    candles: Sequence[Candle],
) -> Dict[str, object]:
    candles = list(candles)
    volume_analysis = calculate_volume_analysis(candles)
    market_structure = calculate_market_structure(candles)
    liquidity_zones = detect_liquidity_zones(volume_analysis, candles[-1].close if candles else 0)
    fundamentals = calculate_fundamental_metrics(candles)
    breadth = calculate_breadth_metrics(candles)
    patterns = calculate_patterns_and_waves(candles, market_structure, summary.orderbook_data if summary else None)
    trade_plan = calculate_trade_signal_plan(summary, candles)

    market_structure["liquidity_zones"] = liquidity_zones

    return {
        "volume_analysis": volume_analysis,
        "market_structure": market_structure,
        "fundamentals": fundamentals,
        "breadth": breadth,
        "patterns": patterns,
        "trade_plan": trade_plan,
    }
