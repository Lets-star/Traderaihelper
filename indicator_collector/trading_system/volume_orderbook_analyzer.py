"""Volume and orderbook analyzer combining volume analysis with market maker detection."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from .utils import clamp


def calculate_order_imbalance(
    orderbook_data: Dict[str, object],
) -> Dict[str, object]:
    """
    Calculate order imbalance metrics from orderbook data.
    
    Analyzes bid/ask volume ratios at different depth levels to quantify
    market side pressure and potential directional bias.
    """
    if not orderbook_data:
        return {
            "imbalance_score": 0.5,
            "bid_ask_ratio": 1.0,
            "imbalance_direction": "neutral",
            "depth_imbalance": 0.0,
            "confidence": 0.0,
        }
    
    total_bid_vol = float(orderbook_data.get("total_bid_volume", 0))
    total_ask_vol = float(orderbook_data.get("total_ask_volume", 0))
    
    sections = orderbook_data.get("sections", {})
    bid_sections = sections.get("bids", {})
    ask_sections = sections.get("asks", {})
    
    imbalance_direction = "neutral"
    imbalance_score = 0.5
    confidence = 0.0
    
    if total_bid_vol > 0 and total_ask_vol > 0:
        bid_ask_ratio = total_bid_vol / total_ask_vol
        
        if bid_ask_ratio > 1.3:
            imbalance_direction = "bullish"
            imbalance_score = clamp(0.5 + (bid_ask_ratio - 1.0) * 0.2, 0.5, 1.0)
            confidence = min((bid_ask_ratio - 1.0) * 100, 100.0)
        elif bid_ask_ratio < 0.7:
            imbalance_direction = "bearish"
            imbalance_score = clamp(0.5 - (1.0 - bid_ask_ratio) * 0.2, 0.0, 0.5)
            confidence = min((1.0 - bid_ask_ratio) * 100, 100.0)
        else:
            bid_ask_ratio = clamp(bid_ask_ratio, 0.8, 1.2)
            imbalance_score = 0.5
            confidence = max(0.0, 100.0 - abs(bid_ask_ratio - 1.0) * 200)
    else:
        bid_ask_ratio = 1.0
    
    top_5_bid_vol = float(bid_sections.get("top_5", {}).get("total_volume", 0))
    top_5_ask_vol = float(ask_sections.get("top_5", {}).get("total_volume", 0))
    
    depth_imbalance = 0.0
    if top_5_bid_vol + top_5_ask_vol > 0:
        depth_imbalance = (top_5_bid_vol - top_5_ask_vol) / (top_5_bid_vol + top_5_ask_vol)
    
    return {
        "imbalance_score": round(imbalance_score, 3),
        "bid_ask_ratio": round(bid_ask_ratio, 3),
        "imbalance_direction": imbalance_direction,
        "depth_imbalance": round(depth_imbalance, 3),
        "confidence": round(confidence, 2),
        "total_bid_volume": round(total_bid_vol, 2),
        "total_ask_volume": round(total_ask_vol, 2),
    }


def detect_liquidity_zones(
    orderbook_data: Dict[str, object],
    volume_analysis: Dict[str, object],
    last_close_price: float = 0.0,
) -> List[Dict[str, object]]:
    """
    Detect liquidity zones combining orderbook depth with volume profile.
    
    Liquidity zones are price levels with unusually high volume concentration
    where smart money might accumulate or distribute.
    """
    if not orderbook_data or not volume_analysis:
        return []
    
    zones: List[Dict[str, object]] = []
    
    vpvr = volume_analysis.get("vpvr", {})
    vpvr_levels = vpvr.get("levels", [])
    total_volume = vpvr.get("total_volume", 0) or 1.0
    poc_price = vpvr.get("poc")
    
    if not vpvr_levels:
        return zones
    
    avg_volume = total_volume / max(len(vpvr_levels), 1)
    volume_threshold = avg_volume * 0.8
    
    for level in vpvr_levels[:10]:
        if level["volume"] >= volume_threshold:
            price = level["price"]
            zone_type = "resistance" if price > last_close_price else "support"
            
            zones.append({
                "type": zone_type,
                "price": price,
                "volume_ratio": round(level["volume"] / total_volume, 4),
                "significance": round(level["volume"] / total_volume * 100, 2),
            })
    
    if poc_price and zones:
        poc_zone = next(
            (z for z in zones if abs(z["price"] - poc_price) < abs(zones[0]["price"] - poc_price) * 0.05),
            None
        )
        if poc_zone:
            poc_zone["is_poc"] = True
    
    return zones[:10]


def analyze_smart_money_activity(
    volume_analysis: Dict[str, object],
) -> Dict[str, object]:
    """
    Analyze smart money events from volume analysis data.
    
    Smart money events are large volume spikes that indicate
    institutional or informed trader activity.
    """
    if not volume_analysis:
        return {
            "smart_money_events": [],
            "activity_score": 0.0,
            "direction_bias": "neutral",
            "confidence": 0.0,
        }
    
    smart_money = volume_analysis.get("smart_money", [])
    
    if not smart_money:
        return {
            "smart_money_events": [],
            "activity_score": 0.0,
            "direction_bias": "neutral",
            "confidence": 0.0,
        }
    
    buy_events = [e for e in smart_money if e.get("direction") == "buy"]
    sell_events = [e for e in smart_money if e.get("direction") == "sell"]
    
    total_buy_volume = sum(e.get("volume", 0) for e in buy_events)
    total_sell_volume = sum(e.get("volume", 0) for e in sell_events)
    total_smart_volume = total_buy_volume + total_sell_volume
    
    activity_score = 0.0
    if total_smart_volume > 0:
        activity_score = clamp(total_smart_volume / 1_000_000, 0.0, 1.0)
    
    direction_bias = "neutral"
    if len(buy_events) > len(sell_events) * 1.5:
        direction_bias = "bullish"
    elif len(sell_events) > len(buy_events) * 1.5:
        direction_bias = "bearish"
    
    confidence = min(len(smart_money) * 10, 100.0)
    
    return {
        "smart_money_events": smart_money[:5],
        "activity_score": round(activity_score, 3),
        "direction_bias": direction_bias,
        "buy_events": len(buy_events),
        "sell_events": len(sell_events),
        "total_smart_volume": round(total_smart_volume, 2),
        "confidence": round(confidence, 2),
    }


def calculate_mm_confidence_weighted(
    mm_analysis: Dict[str, object],
    weight: float = 0.2,
) -> Dict[str, object]:
    """
    Extract and weight market maker confidence from MM analysis.
    
    Market maker activity is weighted at 20% to the final confidence score,
    indicating its importance relative to other factors.
    """
    if not mm_analysis:
        return {
            "mm_detected": False,
            "raw_confidence": 0,
            "weighted_confidence": 0.0,
            "activity_level": "unknown",
            "signals": [],
        }
    
    mm_detected = mm_analysis.get("market_maker_detected", False)
    raw_confidence = int(mm_analysis.get("confidence", 0))
    activity_level = mm_analysis.get("activity_level", "unknown")
    signals = mm_analysis.get("signals", [])
    
    weighted_confidence = (raw_confidence / 100.0) * weight
    
    return {
        "mm_detected": mm_detected,
        "raw_confidence": raw_confidence,
        "weighted_confidence": round(weighted_confidence, 3),
        "activity_level": activity_level,
        "signals": signals,
        "weight": weight,
    }


def analyze_volume_orderbook(
    orderbook_data: Dict[str, object],
    volume_analysis: Dict[str, object],
    mm_analysis: Dict[str, object],
    last_close_price: float = 0.0,
    mm_weight: float = 0.2,
) -> Dict[str, object]:
    """
    Comprehensive orderbook and volume analysis combining multiple signals.
    
    Integrates volume analysis, orderbook depth, and market maker detection
    to produce a complete picture of orderbook health and trading opportunity.
    """
    imbalance = calculate_order_imbalance(orderbook_data)
    liquidity_zones = detect_liquidity_zones(
        orderbook_data, volume_analysis, last_close_price
    )
    smart_money = analyze_smart_money_activity(volume_analysis)
    mm_confidence = calculate_mm_confidence_weighted(mm_analysis, mm_weight)
    
    order_imbalance_score = imbalance.get("imbalance_score", 0.5)
    liquidity_score = 0.5
    if liquidity_zones:
        zone_volumes = [z.get("volume_ratio", 0) for z in liquidity_zones]
        liquidity_score = min(statistics.fmean(zone_volumes) * 2, 1.0) if zone_volumes else 0.5
    
    smart_money_score = smart_money.get("activity_score", 0.0)
    
    mm_score = mm_confidence.get("weighted_confidence", 0.0)
    
    final_factors = [
        order_imbalance_score * 0.35,
        liquidity_score * 0.25,
        smart_money_score * 0.20,
        mm_score * 0.20,
    ]
    
    final_score = sum(final_factors)
    final_score = clamp(final_score, 0.0, 1.0)
    
    direction = "neutral"
    if imbalance.get("imbalance_direction") == "bullish":
        direction = "bullish"
    elif imbalance.get("imbalance_direction") == "bearish":
        direction = "bearish"
    
    if smart_money.get("direction_bias") == "bullish" and direction != "bearish":
        direction = "bullish"
    elif smart_money.get("direction_bias") == "bearish" and direction != "bullish":
        direction = "bearish"
    
    rationale_parts: List[str] = []
    
    if imbalance.get("confidence", 0) > 30:
        bias = "bullish" if imbalance.get("imbalance_direction") == "bullish" else "bearish"
        rationale_parts.append(
            f"Order imbalance suggests {bias} pressure (ratio: {imbalance.get('bid_ask_ratio')})"
        )
    
    if liquidity_zones:
        rationale_parts.append(f"Identified {len(liquidity_zones)} significant liquidity zones")
    
    if smart_money.get("confidence", 0) > 20:
        rationale_parts.append(
            f"Smart money activity detected: {smart_money.get('buy_events')} buy vs "
            f"{smart_money.get('sell_events')} sell events"
        )
    
    if mm_analysis and mm_analysis.get("market_maker_detected"):
        rationale_parts.append(
            f"Market maker activity detected ({mm_confidence.get('activity_level')} level)"
        )
    
    rationale = " | ".join(rationale_parts) if rationale_parts else "Neutral orderbook conditions"
    
    return {
        "final_score": round(final_score, 3),
        "direction": direction,
        "confidence": round(final_score * 100, 2),
        "rationale": rationale,
        "components": {
            "order_imbalance": imbalance,
            "liquidity_zones": liquidity_zones,
            "smart_money": smart_money,
            "market_maker": mm_confidence,
        },
        "factor_weights": {
            "order_imbalance": 0.35,
            "liquidity": 0.25,
            "smart_money": 0.20,
            "market_maker": 0.20,
        },
        "factor_scores": {
            "order_imbalance": round(order_imbalance_score, 3),
            "liquidity": round(liquidity_score, 3),
            "smart_money": round(smart_money_score, 3),
            "market_maker": round(mm_score, 3),
        },
        "metadata": {
            "timestamp": orderbook_data.get("snapshot_time"),
            "symbol": orderbook_data.get("symbol"),
            "source": orderbook_data.get("source", "unknown"),
        },
    }
