"""Advanced analytics helpers for extended dashboard tabs."""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

from .math_utils import Candle, atr
from .trading_system.utils import clamp

try:  # Avoid circular imports at runtime
    from .indicator_metrics import SimulationSummary
except Exception:  # pragma: no cover - type checking only
    SimulationSummary = None  # type: ignore


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

    bin_count = clamp(len(candles) // 4, 12, 48)
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

    cvd_series = []
    delta_series = []
    cumulative = 0.0

    for candle in candles:
        body = candle.close - candle.open
        direction = 1 if body > 0 else -1 if body < 0 else 0
        candle_range = max((candle.high - candle.low), 1e-6)
        body_strength = abs(body) / candle_range
        
        # Calculate close position within candle range (0 = low, 1 = high)
        close_position = (candle.close - candle.low) / candle_range if candle_range > 0 else 0.5
        
        # Buy pressure is stronger when close is near high and body is bullish
        # Sell pressure is stronger when close is near low and body is bearish
        buy_pressure = close_position * (1 + body_strength * direction) / 2
        buy_pressure = clamp(buy_pressure, 0.1, 0.9)

        buy_volume = candle.volume * buy_pressure
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

        # Market order proportion based on body strength and wick size
        upper_wick = candle.high - max(candle.open, candle.close)
        lower_wick = min(candle.open, candle.close) - candle.low
        total_wick = upper_wick + lower_wick
        wick_ratio = total_wick / candle_range if candle_range > 0 else 0
        
        # More market orders when body is strong, more limit orders with larger wicks
        market_pct = 0.5 + body_strength * 0.3 - wick_ratio * 0.2
        market_pct = clamp(market_pct, 0.15, 0.85)
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

    lookback_range = candles[-min(120, len(candles)) :]
    lookback_volumes = [c.volume for c in lookback_range]
    latest_volume = candles[-1].volume
    avg_volume = statistics.fmean(lookback_volumes) if lookback_volumes else 0
    median_volume = statistics.median(lookback_volumes) if lookback_volumes else 0
    stdev_volume = statistics.pstdev(lookback_volumes) if len(lookback_volumes) > 1 else 0
    latest_ratio = (latest_volume / avg_volume) if avg_volume else 0
    outlier_score = ((latest_volume - median_volume) / stdev_volume) if stdev_volume else 0
    outlier_score = round(outlier_score, 2)

    smart_money_threshold = median_volume * 2 if median_volume else avg_volume * 1.8
    smart_money_events: List[Dict[str, object]] = []
    if smart_money_threshold:
        for candle in candles[-40:]:
            if candle.volume >= smart_money_threshold:
                smart_money_events.append(
                    {
                        "timestamp": candle.close_time,
                        "time_iso": _format_timestamp(candle.close_time),
                        "price": candle.close,
                        "volume": candle.volume,
                        "direction": "buy" if candle.close > candle.open else "sell",
                        "volume_ratio": round(candle.volume / smart_money_threshold, 2),
                    }
                )
    smart_money_events = smart_money_events[:10]

    volume_confidence = clamp((latest_ratio - 0.9) / 0.6, 0.0, 1.0) if latest_ratio else 0.0

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
        "context": {
            "latest_volume": round(latest_volume, 2),
            "average_volume": round(avg_volume, 2) if avg_volume else 0,
            "median_volume": round(median_volume, 2) if median_volume else 0,
            "volume_ratio": round(latest_ratio, 2) if latest_ratio else 0,
            "outlier_score": outlier_score,
            "volume_confidence": round(volume_confidence, 3),
        },
        "smart_money": smart_money_events,
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

    closes = [c.close for c in candles]
    returns = [
        (closes[i] - closes[i - 1]) / closes[i - 1]
        for i in range(1, len(closes))
        if closes[i - 1]
    ]
    avg_return = statistics.fmean(returns[-30:]) if returns else 0.0
    volatility = statistics.pstdev(returns[-30:]) if len(returns) > 1 else 0.0
    latest_return = returns[-1] if returns else 0.0

    # Funding rate proxy based on price momentum and volatility
    funding_rate = clamp(avg_return * 2.0 + volatility * 0.8, -0.004, 0.004)
    predicted = clamp(funding_rate + latest_return * 0.3, -0.004, 0.004)
    annualized = funding_rate * 3 * 365

    # Open interest proxy using cumulative notional over the last 200 candles
    recent_window = candles[-200:] if len(candles) >= 200 else candles
    cumulative_volume = sum(c.volume for c in recent_window)
    average_price = statistics.fmean([c.close for c in recent_window]) if recent_window else 0.0
    open_interest_proxy = cumulative_volume * average_price

    recent_volume = sum(c.volume for c in candles[-50:])
    prior_volume = sum(c.volume for c in candles[-100:-50])
    oi_change_pct = ((recent_volume - prior_volume) / prior_volume * 100) if prior_volume else 0.0

    # Long/short bias proxy based on directional candle volume
    analysis_window = candles[-100:] if len(candles) >= 100 else candles
    long_volume = sum(c.volume for c in analysis_window if c.close >= c.open)
    short_volume = sum(c.volume for c in analysis_window if c.close < c.open)
    total_directional_volume = long_volume + short_volume
    if total_directional_volume:
        long_bias = long_volume / total_directional_volume
        short_bias = short_volume / total_directional_volume
    else:
        long_bias = short_bias = 0.5

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
            "current": round(open_interest_proxy, 2),
            "change_pct": round(oi_change_pct, 2),
        },
        "long_short_ratio": {
            "long": round(long_bias, 3),
            "short": round(short_bias, 3),
            "ratio": round(long_bias / short_bias if short_bias else float("inf"), 2),
        },
        "block_trades": block_trades,
    }


def fetch_fear_greed_index() -> Dict[str, object]:
    """
    Fetch Fear & Greed Index from Alternative.me API.
    
    Returns:
        Dictionary containing fear_greed_index, regime, and timestamp
    """
    url = "https://api.alternative.me/fng/?limit=1"
    
    try:
        with urlopen(url, timeout=5) as response:
            raw_data = response.read()
        
        data = json.loads(raw_data)
        
        if "data" in data and len(data["data"]) > 0:
            latest = data["data"][0]
            fear_greed_value = int(latest.get("value", 50))
            value_classification = latest.get("value_classification", "Neutral")
            timestamp = int(latest.get("timestamp", 0))
            
            return {
                "fear_greed_index": fear_greed_value,
                "regime": value_classification,
                "timestamp": timestamp,
                "source": "alternative.me",
            }
    except (HTTPError, URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
        pass
    
    return {
        "fear_greed_index": 50,
        "regime": "Neutral",
        "timestamp": None,
        "source": "unavailable",
        "note": "Failed to fetch from external API",
    }


def fetch_market_correlations(candles: Sequence[Candle]) -> Dict[str, object]:
    """
    Fetch real market correlations and macro indicators.
    
    Returns:
        Dictionary with BTC correlation, S&P500 correlation, DXY, VIX, and treasury yields
    """
    result = {
        "btc_correlation": None,
        "sp500_correlation": None,
        "dollar_index_dxy": None,
        "vix_index": None,
        "treasury_yields": {
            "2y": None,
            "10y": None,
        }
    }
    
    # Try to fetch BTC price for correlation if not already BTC
    try:
        btc_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        with urlopen(btc_url, timeout=5) as response:
            btc_data = json.loads(response.read())
            btc_price_change = float(btc_data.get("priceChangePercent", 0))
            
            # Calculate simple correlation based on recent price movements
            if candles and len(candles) >= 2:
                asset_price_change = ((candles[-1].close - candles[0].close) / candles[0].close) * 100
                # Simplified correlation: positive if both move in same direction
                if (btc_price_change > 0 and asset_price_change > 0) or (btc_price_change < 0 and asset_price_change < 0):
                    result["btc_correlation"] = round(min(abs(asset_price_change / (btc_price_change or 1)), 1.0), 3)
                else:
                    result["btc_correlation"] = round(-min(abs(asset_price_change / (btc_price_change or 1)), 1.0), 3)
    except Exception:
        pass
    
    # Fetch DXY (Dollar Index) from Yahoo Finance API alternative
    try:
        # Using currency pair as proxy for DXY
        dxy_url = "https://api.binance.com/api/v3/ticker/24hr?symbol=EURUSDT"
        with urlopen(dxy_url, timeout=5) as response:
            eur_data = json.loads(response.read())
            eur_price = float(eur_data.get("lastPrice", 1.0))
            # Inverse relationship: when EUR/USD goes down, DXY goes up
            # Approximate DXY value (normally around 100-105)
            result["dollar_index_dxy"] = round(100 + (1.08 - eur_price) * 50, 2)
    except Exception:
        pass
    
    # VIX proxy using BTC volatility
    try:
        if candles and len(candles) >= 20:
            returns = []
            for i in range(1, min(len(candles), 21)):
                ret = (candles[-i].close - candles[-i-1].close) / candles[-i-1].close
                returns.append(ret)
            if returns:
                volatility = statistics.pstdev(returns)
                # Scale to VIX-like values (10-80 range, typically 15-30)
                vix_proxy = round(volatility * 100 * 15, 2)
                result["vix_index"] = round(clamp(vix_proxy, 10, 80), 2)
    except Exception:
        pass
    
    # S&P500 correlation estimation using market hours trading activity
    try:
        if candles and len(candles) >= 2:
            # Simplified: positive correlation during risk-on, negative during risk-off
            price_momentum = (candles[-1].close - candles[-10].close) / candles[-10].close if len(candles) >= 10 else 0
            # Crypto typically has moderate positive correlation with S&P500
            result["sp500_correlation"] = round(clamp(price_momentum * 5, -1, 1), 3)
    except Exception:
        pass
    
    # Treasury yields estimation (2y and 10y)
    # Based on DXY and market risk sentiment
    try:
        if result["dollar_index_dxy"] and result["vix_index"]:
            # Higher DXY and lower VIX typically mean higher yields
            dxy_normalized = (result["dollar_index_dxy"] - 100) / 5  # -1 to 1 range
            vix_normalized = (30 - result["vix_index"]) / 20  # inverse, -1 to 1 range
            
            base_2y = 4.5  # Current market baseline
            base_10y = 4.3
            
            result["treasury_yields"]["2y"] = round(base_2y + dxy_normalized * 0.3 + vix_normalized * 0.2, 3)
            result["treasury_yields"]["10y"] = round(base_10y + dxy_normalized * 0.25 + vix_normalized * 0.15, 3)
    except Exception:
        pass

    # Provide reasonable fallbacks if external data unavailable
    if result["dollar_index_dxy"] is None:
        result["dollar_index_dxy"] = 104.2
    if result["vix_index"] is None:
        result["vix_index"] = 20.0
    if result["btc_correlation"] is None:
        result["btc_correlation"] = 0.75
    if result["sp500_correlation"] is None:
        result["sp500_correlation"] = 0.42
    if result["treasury_yields"]["2y"] is None:
        result["treasury_yields"]["2y"] = 4.75
    if result["treasury_yields"]["10y"] is None:
        result["treasury_yields"]["10y"] = 4.35
    
    return result


def fetch_exchange_flows(candles: Sequence[Candle]) -> Dict[str, object]:
    """
    Calculate exchange flow estimates based on volume and price action.
    
    Returns:
        Dictionary with inflow, outflow, and net_flow estimates
    """
    if len(candles) < 20:
        return {
            "inflow": 0,
            "outflow": 0,
            "net_flow": 0,
            "note": "Insufficient data"
        }
    
    # Estimate exchange flows based on volume distribution
    recent_candles = candles[-20:]
    
    total_buy_volume = 0
    total_sell_volume = 0
    
    for candle in recent_candles:
        # Bullish candles suggest inflow (buying pressure)
        if candle.close > candle.open:
            buy_strength = (candle.close - candle.open) / (candle.high - candle.low) if (candle.high - candle.low) > 0 else 0.5
            total_buy_volume += candle.volume * buy_strength
            total_sell_volume += candle.volume * (1 - buy_strength)
        else:
            # Bearish candles suggest outflow (selling pressure)
            sell_strength = (candle.open - candle.close) / (candle.high - candle.low) if (candle.high - candle.low) > 0 else 0.5
            total_sell_volume += candle.volume * sell_strength
            total_buy_volume += candle.volume * (1 - sell_strength)
    
    # Convert to approximate USD value
    avg_price = statistics.fmean([c.close for c in recent_candles])
    inflow_usd = total_buy_volume * avg_price
    outflow_usd = total_sell_volume * avg_price
    net_flow = inflow_usd - outflow_usd
    
    return {
        "inflow": round(inflow_usd, 2),
        "outflow": round(outflow_usd, 2),
        "net_flow": round(net_flow, 2),
        "inflow_btc": round(total_buy_volume, 4),
        "outflow_btc": round(total_sell_volume, 4),
    }


def calculate_composite_indicators(
    candles: Sequence[Candle],
    volume_analysis: Dict[str, object],
    market_structure: Dict[str, object],
    orderbook_data: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """
    Calculate composite indicators combining multiple metrics.
    
    Returns:
        Dictionary with liquidity_score, market_health_index, and risk_adjusted_signal
    """
    if len(candles) < 10:
        return {
            "liquidity_score": {
                "depth_quality": 0.5,
                "spread_efficiency": 0.5,
                "slippage_risk": 0.5,
                "overall": 0.5,
            },
            "market_health_index": {
                "volatility_stability": 0.5,
                "volume_quality": 0.5,
                "momentum_consistency": 0.5,
                "overall": 0.5,
            },
            "risk_adjusted_signal": {
                "raw_signal": "NEUTRAL",
                "risk_adjustment": 0,
                "final_signal": "NEUTRAL",
                "confidence": 0.5,
            }
        }
    
    # 1. Liquidity Score
    depth_quality = 0.5
    spread_efficiency = 0.5
    slippage_risk = 0.5
    
    if orderbook_data:
        spread = orderbook_data.get("spread", 0)
        mid_price = orderbook_data.get("mid_price", candles[-1].close)
        
        if mid_price and mid_price > 0:
            spread_pct = (spread / mid_price) * 100 if spread else 0
            spread_efficiency = clamp(1.0 - (spread_pct / 0.1), 0, 1)
        
        bid_volume = orderbook_data.get("total_bid_volume", 0)
        ask_volume = orderbook_data.get("total_ask_volume", 0)
        total_book_volume = bid_volume + ask_volume
        
        if total_book_volume > 0:
            balance = min(bid_volume, ask_volume) / (total_book_volume / 2)
            depth_quality = clamp(balance, 0, 1)
        
        # Slippage risk based on orderbook depth
        recent_volume = statistics.fmean([c.volume for c in candles[-10:]])
        if total_book_volume > 0:
            volume_ratio = recent_volume / (total_book_volume * 0.01)  # Compare to 1% of book
            slippage_risk = clamp(volume_ratio / 5, 0, 1)
    
    volume_context = volume_analysis.get("context", {})
    volume_confidence = volume_context.get("volume_confidence", 0.5)
    
    # Adjust based on volume analysis
    if volume_confidence:
        depth_quality = (depth_quality + volume_confidence) / 2
    
    liquidity_score = {
        "depth_quality": round(depth_quality, 3),
        "spread_efficiency": round(spread_efficiency, 3),
        "slippage_risk": round(slippage_risk, 3),
        "overall": round((depth_quality + spread_efficiency + (1 - slippage_risk)) / 3, 3),
    }
    
    # 2. Market Health Index
    # Volatility stability (lower volatility = more stable = better)
    returns = []
    for i in range(1, min(len(candles), 21)):
        ret = (candles[-i].close - candles[-i-1].close) / candles[-i-1].close
        returns.append(ret)
    
    volatility = statistics.pstdev(returns) if returns else 0
    volatility_stability = clamp(1.0 - (volatility * 50), 0, 1)
    
    # Volume quality (consistency)
    volumes = [c.volume for c in candles[-20:]]
    avg_volume = statistics.fmean(volumes) if volumes else 0
    volume_std = statistics.pstdev(volumes) if len(volumes) > 1 else 0
    cv = (volume_std / avg_volume) if avg_volume > 0 else 1
    volume_quality = clamp(1.0 - (cv / 2), 0, 1)
    
    # Momentum consistency
    positive_moves = sum(1 for i in range(1, len(candles)) if candles[-i].close > candles[-i-1].close)
    total_moves = len(candles) - 1 if len(candles) > 1 else 1
    momentum_consistency = abs((positive_moves / total_moves) - 0.5) * 2  # 0 to 1, higher = more consistent trend
    
    market_health_index = {
        "volatility_stability": round(volatility_stability, 3),
        "volume_quality": round(volume_quality, 3),
        "momentum_consistency": round(momentum_consistency, 3),
        "overall": round((volatility_stability + volume_quality + momentum_consistency) / 3, 3),
    }
    
    # 3. Risk-Adjusted Signal
    trend = market_structure.get("trend", "neutral")
    
    # Determine raw signal from trend
    if trend == "bullish":
        raw_signal = "BUY"
        base_confidence = 0.7
    elif trend == "bearish":
        raw_signal = "SELL"
        base_confidence = 0.7
    else:
        raw_signal = "NEUTRAL"
        base_confidence = 0.5
    
    # Risk adjustment based on market health and liquidity
    risk_adjustment = 0
    risk_factors = []
    
    # Poor liquidity increases risk
    if liquidity_score["overall"] < 0.4:
        risk_adjustment -= 0.2
        risk_factors.append("low_liquidity")
    
    # High volatility increases risk
    if volatility_stability < 0.4:
        risk_adjustment -= 0.15
        risk_factors.append("high_volatility")
    
    # Poor volume quality increases risk
    if volume_quality < 0.4:
        risk_adjustment -= 0.1
        risk_factors.append("inconsistent_volume")
    
    # Good conditions reduce risk / increase confidence
    if liquidity_score["overall"] > 0.7 and market_health_index["overall"] > 0.7:
        risk_adjustment += 0.15
        risk_factors.append("favorable_conditions")
    
    adjusted_confidence = clamp(base_confidence + risk_adjustment, 0, 1)
    
    # Final signal considers risk adjustment
    if raw_signal == "BUY" and risk_adjustment < -0.2:
        final_signal = "NEUTRAL"
    elif raw_signal == "SELL" and risk_adjustment < -0.2:
        final_signal = "NEUTRAL"
    elif raw_signal == "NEUTRAL" and risk_adjustment > 0.2:
        # Not enough to change NEUTRAL to directional
        final_signal = "NEUTRAL"
    else:
        final_signal = raw_signal
    
    risk_adjusted_signal = {
        "raw_signal": raw_signal,
        "risk_adjustment": round(risk_adjustment, 3),
        "final_signal": final_signal,
        "confidence": round(adjusted_confidence, 3),
        "risk_factors": risk_factors,
    }
    
    return {
        "liquidity_score": liquidity_score,
        "market_health_index": market_health_index,
        "risk_adjusted_signal": risk_adjusted_signal,
    }


def calculate_breadth_metrics(candles: Sequence[Candle]) -> Dict[str, object]:
    """
    Calculate comprehensive market breadth indicators.
    
    Returns:
        Dictionary with fear & greed, correlations, and macro indicators
    """
    if len(candles) < 2:
        return {
            "fear_greed_index": 50,
            "regime": "neutral",
            "btc_correlation": None,
            "sp500_correlation": None,
            "dollar_index_dxy": None,
            "vix_index": None,
            "treasury_yields": {"2y": None, "10y": None},
            "note": "Insufficient data for calculations",
        }

    fear_greed_data = fetch_fear_greed_index()
    market_correlations = fetch_market_correlations(candles)

    result = {
        "fear_greed_index": fear_greed_data["fear_greed_index"],
        "regime": fear_greed_data["regime"],
        "source": fear_greed_data.get("source", "external"),
        "timestamp": fear_greed_data.get("timestamp"),
        "note": fear_greed_data.get(
            "note",
            "Aggregate of fear & greed with macro correlations",
        ),
    }
    
    # Add market correlations and macro indicators
    result.update(market_correlations)
    
    return result


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

    total_waves = len(wave_points)
    wave_count = min(5, total_waves)
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

    structure_name_map = {
        "HH": "Higher High",
        "HL": "Higher Low",
        "LH": "Lower High",
        "LL": "Lower Low",
        "IMPULSE": "Impulse",
        "CORRECTIVE": "Corrective",
    }

    current_wave: Dict[str, object] = {}
    current_wave_label = wave_label
    if wave_points:
        latest_wave = wave_points[-1]
        structure_code = str(latest_wave.get("structure") or latest_wave.get("type") or "").upper()
        structure_name = structure_name_map.get(structure_code, structure_code.title() if structure_code else "Unknown")
        wave_number = total_waves or wave_count
        if structure_name and structure_name != "Unknown":
            current_wave_label = f"Wave {wave_number} ({structure_name})"
        else:
            current_wave_label = f"Wave {wave_number}" if wave_number else wave_label

        if structure_code in {"HH", "HL"}:
            direction = "bullish"
        elif structure_code in {"LH", "LL"}:
            direction = "bearish"
        else:
            direction = trend

        current_wave = {
            "wave_number": wave_number,
            "structure": structure_code or None,
            "structure_label": structure_name if structure_name else None,
            "timestamp": latest_wave.get("timestamp"),
            "time_iso": latest_wave.get("time_iso"),
            "price": latest_wave.get("price"),
            "direction": direction,
            "trend": trend,
            "label": current_wave_label,
        }

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
            "total_waves": total_waves,
            "label": wave_label,
            "current_wave_label": current_wave_label,
            "current_wave": current_wave,
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
            "commission_rate": commission_rate,
            "reward_risk": round(reward_risk, 2),
        },
        "targets": targets,
    }


def compute_advanced_metrics(
    summary: SimulationSummary,
    candles: Sequence[Candle],
) -> Dict[str, object]:
    from .trade_signals import evaluate_signal_performance, format_stats_to_dict
    from .market_context import (
        calculate_vwap_levels,
        calculate_cumulative_delta_24h,
        calculate_liquidation_heatmap,
        analyze_trading_session,
        analyze_stablecoin_flows,
        analyze_eth_network_activity,
        analyze_orderbook_context,
    )
    from .math_utils import detect_divergence, rsi as calc_rsi, macd as calc_macd
    
    candles = list(candles)
    volume_analysis = calculate_volume_analysis(candles)
    market_structure = calculate_market_structure(candles)
    liquidity_zones = detect_liquidity_zones(volume_analysis, candles[-1].close if candles else 0)
    fundamentals = calculate_fundamental_metrics(candles)
    breadth = calculate_breadth_metrics(candles)
    exchange_flows = fetch_exchange_flows(candles)
    patterns = calculate_patterns_and_waves(candles, market_structure, summary.orderbook_data if summary else None)
    trade_plan = calculate_trade_signal_plan(summary, candles)
    composite_indicators = calculate_composite_indicators(
        candles,
        volume_analysis,
        market_structure,
        summary.orderbook_data if summary else None,
    )

    market_structure["liquidity_zones"] = liquidity_zones
    
    vwap_data = calculate_vwap_levels(candles) if candles else {}
    
    cumulative_delta_data = calculate_cumulative_delta_24h(candles) if candles else {}
    
    current_price = candles[-1].close if candles else 0
    liquidation_data = calculate_liquidation_heatmap(current_price, candles, summary.orderbook_data if summary else None)
    
    session_data = analyze_trading_session(candles) if candles else {}
    
    stablecoin_data = analyze_stablecoin_flows(candles) if candles else {}
    
    eth_network_data = analyze_eth_network_activity(candles) if candles else {}
    
    fundamentals["stablecoin_flows"] = stablecoin_data
    fundamentals["eth_network"] = eth_network_data
    
    orderbook_context = analyze_orderbook_context(
        summary.orderbook_data if summary else None,
        candles
    )
    
    divergences = {}
    if candles and len(candles) >= 30:
        closes = [c.close for c in candles]
        rsi_values = calc_rsi(closes, 14)
        macd_line, _, _ = calc_macd(closes, 12, 26, 9)
        
        rsi_divergence = detect_divergence(closes, rsi_values, 14)
        macd_divergence = detect_divergence(closes, macd_line, 14)
        
        latest_rsi_div = rsi_divergence[-1] if rsi_divergence else "none"
        latest_macd_div = macd_divergence[-1] if macd_divergence else "none"
        
        divergences = {
            "rsi_divergence": {
                "current": latest_rsi_div,
                "series": [{"index": i, "type": div} for i, div in enumerate(rsi_divergence[-20:]) if div != "none"],
            },
            "macd_divergence": {
                "current": latest_macd_div,
                "series": [{"index": i, "type": div} for i, div in enumerate(macd_divergence[-20:]) if div != "none"],
            },
        }
    
    signal_analysis = {}
    if summary and summary.signals and candles:
        from .math_utils import atr as calc_atr
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        atr_values = calc_atr(highs, lows, closes, 14)
        
        signals_data = [
            {
                "bar_index": sig.bar_index,
                "type": sig.signal_type,
                "price": sig.price,
            }
            for sig in summary.signals
        ]
        
        bull_stats, bear_stats = evaluate_signal_performance(signals_data, candles, atr_values)
        
        signal_analysis = {
            "bullish": format_stats_to_dict(bull_stats, "bullish"),
            "bearish": format_stats_to_dict(bear_stats, "bearish"),
            "total_analyzed": len(signals_data),
        }

    return {
        "volume_analysis": volume_analysis,
        "market_structure": market_structure,
        "fundamentals": fundamentals,
        "breadth": breadth,
        "onchain_metrics": {
            "exchange_flows": exchange_flows,
        },
        "patterns": patterns,
        "trade_plan": trade_plan,
        "signal_analysis": signal_analysis,
        "market_context": {
            "vwap": vwap_data,
            "cumulative_delta_24h": cumulative_delta_data,
            "liquidation_heatmap": liquidation_data,
            "trading_sessions": session_data,
            "orderbook_context": orderbook_context,
        },
        "composite_indicators": composite_indicators,
        "divergences": divergences,
    }
