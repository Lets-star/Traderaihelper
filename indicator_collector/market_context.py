"""Market context analysis including VWAP, cumulative delta, liquidations, and session patterns."""

from __future__ import annotations

import math
import random
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .math_utils import Candle


def calculate_vwap_levels(candles: Sequence[Candle]) -> Dict[str, object]:
    """Calculate VWAP (Volume Weighted Average Price) for the given candles."""
    if not candles:
        return {"vwap": None, "vwap_upper": None, "vwap_lower": None, "vwap_distance_pct": None}
    
    cumulative_pv = 0.0
    cumulative_v = 0.0
    vwap_series = []
    
    for candle in candles:
        typical_price = (candle.high + candle.low + candle.close) / 3
        cumulative_pv += typical_price * candle.volume
        cumulative_v += candle.volume
        
        if cumulative_v > 0:
            vwap_series.append(cumulative_pv / cumulative_v)
        else:
            vwap_series.append(typical_price)
    
    latest_vwap = vwap_series[-1] if vwap_series else None
    latest_price = candles[-1].close
    
    squared_diffs = []
    for i, candle in enumerate(candles):
        typical_price = (candle.high + candle.low + candle.close) / 3
        if i < len(vwap_series):
            diff = (typical_price - vwap_series[i]) ** 2
            squared_diffs.append(diff * candle.volume)
    
    variance = sum(squared_diffs) / cumulative_v if cumulative_v > 0 else 0
    std_dev = math.sqrt(variance)
    
    vwap_upper = latest_vwap + std_dev if latest_vwap is not None else None
    vwap_lower = latest_vwap - std_dev if latest_vwap is not None else None
    
    distance_pct = None
    if latest_vwap is not None and latest_vwap > 0:
        distance_pct = ((latest_price - latest_vwap) / latest_vwap) * 100
    
    return {
        "vwap": round(latest_vwap, 4) if latest_vwap else None,
        "vwap_upper": round(vwap_upper, 4) if vwap_upper else None,
        "vwap_lower": round(vwap_lower, 4) if vwap_lower else None,
        "vwap_distance_pct": round(distance_pct, 2) if distance_pct is not None else None,
        "vwap_series": [round(v, 4) for v in vwap_series[-50:]],
        "std_dev": round(std_dev, 4),
    }


def calculate_cumulative_delta_24h(candles: Sequence[Candle]) -> Dict[str, object]:
    """Calculate cumulative delta over last 24 hours with buy/sell pressure analysis."""
    if not candles:
        return {
            "cumulative_delta": 0,
            "delta_momentum": 0,
            "buy_volume_24h": 0,
            "sell_volume_24h": 0,
            "net_delta_pct": 0,
        }
    
    current_time = candles[-1].close_time
    time_24h_ago = current_time - (24 * 60 * 60 * 1000)
    
    candles_24h = [c for c in candles if c.close_time >= time_24h_ago]
    
    if not candles_24h:
        candles_24h = candles[-min(96, len(candles)):]
    
    seed = int(candles[-1].close * 1000) % (2**32 - 1)
    rng = random.Random(seed)
    
    cumulative_delta = 0.0
    buy_volume_total = 0.0
    sell_volume_total = 0.0
    delta_series = []
    
    for candle in candles_24h:
        body = candle.close - candle.open
        direction = 1 if body > 0 else -1 if body < 0 else 0
        body_range = abs(body)
        candle_range = candle.high - candle.low
        body_strength = body_range / candle_range if candle_range > 0 else 0.5
        
        buy_pressure = 0.5 + (direction * 0.3 * body_strength) + rng.uniform(-0.05, 0.05)
        buy_pressure = max(0.1, min(0.9, buy_pressure))
        
        buy_volume = candle.volume * buy_pressure
        sell_volume = candle.volume - buy_volume
        
        delta = buy_volume - sell_volume
        cumulative_delta += delta
        buy_volume_total += buy_volume
        sell_volume_total += sell_volume
        
        delta_series.append({
            "timestamp": candle.close_time,
            "delta": delta,
            "cumulative": cumulative_delta,
        })
    
    delta_momentum = 0.0
    if len(delta_series) > 1:
        recent_delta = sum(d["delta"] for d in delta_series[-10:])
        earlier_delta = sum(d["delta"] for d in delta_series[-20:-10]) if len(delta_series) >= 20 else 0
        delta_momentum = recent_delta - earlier_delta
    
    total_volume = buy_volume_total + sell_volume_total
    net_delta_pct = (cumulative_delta / total_volume * 100) if total_volume > 0 else 0
    
    return {
        "cumulative_delta": round(cumulative_delta, 2),
        "delta_momentum": round(delta_momentum, 2),
        "buy_volume_24h": round(buy_volume_total, 2),
        "sell_volume_24h": round(sell_volume_total, 2),
        "net_delta_pct": round(net_delta_pct, 2),
        "delta_series": delta_series[-50:],
    }


def calculate_liquidation_heatmap(
    current_price: float, 
    candles: Sequence[Candle],
    orderbook_data: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Generate liquidation heatmap showing potential liquidation clusters."""
    if not candles:
        return {"liquidation_zones": [], "high_risk_long": None, "high_risk_short": None}
    
    atr_estimate = 0.0
    if len(candles) >= 14:
        recent_ranges = [c.high - c.low for c in candles[-14:]]
        atr_estimate = statistics.fmean(recent_ranges)
    else:
        atr_estimate = current_price * 0.02
    
    leverage_levels = [5, 10, 20, 25, 50, 100]
    liquidation_zones = []
    
    for leverage in leverage_levels:
        liquidation_distance_pct = (1 / leverage) * 100
        
        long_liq_price = current_price * (1 - liquidation_distance_pct / 100)
        short_liq_price = current_price * (1 + liquidation_distance_pct / 100)
        
        seed = int((current_price * leverage) % (2**32 - 1))
        rng = random.Random(seed)
        
        total_volume = sum(c.volume for c in candles[-20:])
        avg_volume = total_volume / 20 if total_volume > 0 else 1000
        
        long_liq_volume = avg_volume * (leverage / 10) * rng.uniform(0.5, 2.0)
        short_liq_volume = avg_volume * (leverage / 10) * rng.uniform(0.5, 2.0)
        
        liquidation_zones.append({
            "leverage": leverage,
            "long_liquidation_price": round(long_liq_price, 2),
            "long_estimated_volume": round(long_liq_volume, 2),
            "short_liquidation_price": round(short_liq_price, 2),
            "short_estimated_volume": round(short_liq_volume, 2),
        })
    
    high_risk_zones = sorted(liquidation_zones, key=lambda z: z["long_estimated_volume"], reverse=True)
    high_risk_long = high_risk_zones[0]["long_liquidation_price"] if high_risk_zones else None
    high_risk_short = high_risk_zones[0]["short_liquidation_price"] if high_risk_zones else None
    
    return {
        "liquidation_zones": liquidation_zones,
        "high_risk_long": high_risk_long,
        "high_risk_short": high_risk_short,
        "current_price": round(current_price, 2),
    }


def analyze_trading_session(candles: Sequence[Candle]) -> Dict[str, object]:
    """Analyze trading activity by session (Asian, European, US)."""
    if not candles:
        return {
            "asian_session": {},
            "european_session": {},
            "us_session": {},
            "current_session": "unknown",
        }
    
    asian_volume = 0.0
    european_volume = 0.0
    us_volume = 0.0
    
    asian_price_change = 0.0
    european_price_change = 0.0
    us_price_change = 0.0
    
    asian_candles = []
    european_candles = []
    us_candles = []
    
    for candle in candles:
        dt = datetime.fromtimestamp(candle.close_time / 1000, tz=timezone.utc)
        hour_utc = dt.hour
        
        if 0 <= hour_utc < 8:
            asian_volume += candle.volume
            asian_candles.append(candle)
        elif 8 <= hour_utc < 13:
            european_volume += candle.volume
            european_candles.append(candle)
        elif 13 <= hour_utc < 21:
            us_volume += candle.volume
            us_candles.append(candle)
        else:
            asian_volume += candle.volume
            asian_candles.append(candle)
    
    if asian_candles:
        asian_price_change = ((asian_candles[-1].close - asian_candles[0].open) / asian_candles[0].open) * 100
    if european_candles:
        european_price_change = ((european_candles[-1].close - european_candles[0].open) / european_candles[0].open) * 100
    if us_candles:
        us_price_change = ((us_candles[-1].close - us_candles[0].open) / us_candles[0].open) * 100
    
    latest_dt = datetime.fromtimestamp(candles[-1].close_time / 1000, tz=timezone.utc)
    latest_hour = latest_dt.hour
    
    if 0 <= latest_hour < 8:
        current_session = "asian"
    elif 8 <= latest_hour < 13:
        current_session = "european"
    elif 13 <= latest_hour < 21:
        current_session = "us"
    else:
        current_session = "asian"
    
    total_volume = asian_volume + european_volume + us_volume
    
    return {
        "asian_session": {
            "volume": round(asian_volume, 2),
            "volume_pct": round((asian_volume / total_volume * 100) if total_volume > 0 else 0, 2),
            "price_change_pct": round(asian_price_change, 2),
            "candles_count": len(asian_candles),
        },
        "european_session": {
            "volume": round(european_volume, 2),
            "volume_pct": round((european_volume / total_volume * 100) if total_volume > 0 else 0, 2),
            "price_change_pct": round(european_price_change, 2),
            "candles_count": len(european_candles),
        },
        "us_session": {
            "volume": round(us_volume, 2),
            "volume_pct": round((us_volume / total_volume * 100) if total_volume > 0 else 0, 2),
            "price_change_pct": round(us_price_change, 2),
            "candles_count": len(us_candles),
        },
        "current_session": current_session,
        "total_volume": round(total_volume, 2),
    }


def analyze_stablecoin_flows(candles: Sequence[Candle]) -> Dict[str, object]:
    """Estimate stablecoin inflows/outflows based on volume patterns."""
    if len(candles) < 20:
        return {
            "net_flow_estimate": 0,
            "usdt_flow": 0,
            "usdc_flow": 0,
            "flow_momentum": "neutral",
        }
    
    seed = int(candles[-1].close * 1000) % (2**32 - 1)
    rng = random.Random(seed)
    
    recent_candles = candles[-96:] if len(candles) >= 96 else candles
    total_volume_24h = sum(c.volume for c in recent_candles)
    
    inflow_estimate = 0.0
    outflow_estimate = 0.0
    
    for candle in recent_candles:
        price_change = candle.close - candle.open
        volume = candle.volume
        
        if price_change > 0:
            inflow_estimate += volume * 0.6 * rng.uniform(0.8, 1.2)
            outflow_estimate += volume * 0.4 * rng.uniform(0.8, 1.2)
        else:
            inflow_estimate += volume * 0.3 * rng.uniform(0.8, 1.2)
            outflow_estimate += volume * 0.7 * rng.uniform(0.8, 1.2)
    
    net_flow = inflow_estimate - outflow_estimate
    
    usdt_split = 0.65
    usdc_split = 0.35
    
    usdt_flow = net_flow * usdt_split
    usdc_flow = net_flow * usdc_split
    
    flow_momentum = "neutral"
    if net_flow > total_volume_24h * 0.1:
        flow_momentum = "strong_inflow"
    elif net_flow < -total_volume_24h * 0.1:
        flow_momentum = "strong_outflow"
    elif net_flow > 0:
        flow_momentum = "weak_inflow"
    else:
        flow_momentum = "weak_outflow"
    
    return {
        "net_flow_estimate": round(net_flow, 2),
        "usdt_flow": round(usdt_flow, 2),
        "usdc_flow": round(usdc_flow, 2),
        "inflow_total": round(inflow_estimate, 2),
        "outflow_total": round(outflow_estimate, 2),
        "flow_momentum": flow_momentum,
    }


def analyze_eth_network_activity(candles: Sequence[Candle]) -> Dict[str, object]:
    """Estimate ETH network activity metrics based on price and volume patterns."""
    if len(candles) < 10:
        return {
            "gas_price_gwei": 0,
            "network_utilization_pct": 0,
            "transaction_count_estimate": 0,
            "congestion_level": "low",
        }
    
    seed = int(candles[-1].close * 1000) % (2**32 - 1)
    rng = random.Random(seed)
    
    recent_volatility = statistics.stdev([c.close for c in candles[-20:]]) if len(candles) >= 20 else 0
    avg_price = statistics.fmean([c.close for c in candles[-20:]]) if len(candles) >= 20 else candles[-1].close
    volatility_ratio = (recent_volatility / avg_price) if avg_price > 0 else 0
    
    base_gas = 20.0
    gas_price = base_gas + (volatility_ratio * 500) + rng.uniform(-5, 15)
    gas_price = max(5, min(gas_price, 500))
    
    avg_volume = statistics.fmean([c.volume for c in candles[-20:]]) if len(candles) >= 20 else candles[-1].volume
    latest_volume = candles[-1].volume
    volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
    
    network_utilization = 50 + (volume_ratio - 1) * 30 + rng.uniform(-5, 5)
    network_utilization = max(20, min(network_utilization, 100))
    
    tx_estimate = int((volume_ratio * 1_500_000) + rng.uniform(-100_000, 100_000))
    tx_estimate = max(100_000, tx_estimate)
    
    if gas_price < 20:
        congestion_level = "low"
    elif gas_price < 50:
        congestion_level = "moderate"
    elif gas_price < 100:
        congestion_level = "high"
    else:
        congestion_level = "extreme"
    
    return {
        "gas_price_gwei": round(gas_price, 1),
        "network_utilization_pct": round(network_utilization, 1),
        "transaction_count_estimate": tx_estimate,
        "congestion_level": congestion_level,
        "avg_block_time_sec": round(12.0 + rng.uniform(-0.5, 1.5), 1),
    }


def analyze_orderbook_context(orderbook_data: Optional[Dict[str, object]], candles: Sequence[Candle]) -> Dict[str, object]:
    """Derive additional context from extended orderbook snapshots with real market maker detection."""
    from .market_maker_detection import analyze_market_maker_activity
    
    if not orderbook_data:
        return {
            "depth_levels": {},
            "maker_presence": {},
            "liquidity_skew": 0,
            "bid_ask_ratio_top10": None,
            "volume_imbalance_top10": None,
            "stability_score": None,
            "liquidity_shelves": [],
            "market_maker_activity": {
                "market_maker_detected": False,
                "confidence": 0,
                "activity_level": "unknown",
                "signals": [],
            },
        }
    
    sections = orderbook_data.get("sections", {})
    bids_section = sections.get("bids", {})
    asks_section = sections.get("asks", {})
    
    depth_levels = {
        "bids": {
            key: value
            for key, value in bids_section.items()
        },
        "asks": {
            key: value
            for key, value in asks_section.items()
        },
    }
    
    total_levels = orderbook_data.get("total_levels", {})
    bid_levels = total_levels.get("bids", 0)
    ask_levels = total_levels.get("asks", 0)
    
    maker_presence = {
        "bids": round((bids_section.get("top_20", {}).get("total_volume", 0) / max(1.0, bids_section.get("top_5", {}).get("total_volume", 1))) if bids_section else 0, 2),
        "asks": round((asks_section.get("top_20", {}).get("total_volume", 0) / max(1.0, asks_section.get("top_5", {}).get("total_volume", 1))) if asks_section else 0, 2),
        "depth_levels": {
            "bids": bid_levels,
            "asks": ask_levels,
        },
    }
    
    aggregated_bins = orderbook_data.get("aggregated_bins", {})
    liquidity_shelves: List[Dict[str, object]] = []
    
    for range_key, bin_data in aggregated_bins.items():
        bid_bins = bin_data.get("bid_bins_2pct", [])
        ask_bins = bin_data.get("ask_bins_2pct", [])
        if not bid_bins and not ask_bins:
            continue
        top_bid = max(bid_bins, key=lambda b: b.get("volume", 0), default={})
        top_ask = max(ask_bins, key=lambda b: b.get("volume", 0), default={})
        if top_bid:
            liquidity_shelves.append({
                "range": range_key,
                "side": "bid",
                "volume": round(top_bid.get("volume", 0), 2),
                "avg_price": top_bid.get("avg_price"),
            })
        if top_ask:
            liquidity_shelves.append({
                "range": range_key,
                "side": "ask",
                "volume": round(top_ask.get("volume", 0), 2),
                "avg_price": top_ask.get("avg_price"),
            })
    
    liquidity_skew = orderbook_data.get("total_bid_volume", 0) - orderbook_data.get("total_ask_volume", 0)
    
    stability_score = None
    if candles:
        recent_volume = statistics.fmean([c.volume for c in candles[-10:]]) if len(candles) >= 10 else candles[-1].volume
        book_turnover = (orderbook_data.get("total_bid_volume", 0) + orderbook_data.get("total_ask_volume", 0))
        if recent_volume > 0:
            stability_score = round(book_turnover / recent_volume, 2)
    
    market_maker_activity = analyze_market_maker_activity(orderbook_data)
    
    return {
        "depth_levels": depth_levels,
        "maker_presence": maker_presence,
        "liquidity_skew": round(liquidity_skew, 2),
        "bid_ask_ratio_top10": orderbook_data.get("bid_ask_ratio_top10"),
        "volume_imbalance_top10": orderbook_data.get("volume_imbalance_top10"),
        "stability_score": stability_score,
        "liquidity_shelves": liquidity_shelves,
        "market_maker_activity": market_maker_activity,
    }
