"""Real market maker detection algorithms based on orderbook analysis."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence, Tuple


def detect_order_walls(
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    mid_price: Optional[float] = None,
) -> Dict[str, object]:
    """
    Detect large order walls that may indicate market maker presence.
    Walls are identified as orders significantly larger than average.
    Orders within 1% of mid price are ignored to filter out noise.
    """
    if not bids or not asks:
        return {"bid_walls": [], "ask_walls": [], "wall_pressure": "neutral"}
    
    bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda x: x[0])
    
    if mid_price and mid_price > 0:
        threshold_1pct = mid_price * 0.01
        bids_filtered = [(p, v) for p, v in bids_sorted if abs(mid_price - p) > threshold_1pct]
        asks_filtered = [(p, v) for p, v in asks_sorted if abs(p - mid_price) > threshold_1pct]
    else:
        bids_filtered = bids_sorted
        asks_filtered = asks_sorted
    
    top_20_bids = bids_filtered[:20]
    top_20_asks = asks_filtered[:20]
    
    bid_volumes = [vol for _, vol in top_20_bids]
    ask_volumes = [vol for _, vol in top_20_asks]
    
    avg_bid_vol = statistics.fmean(bid_volumes) if bid_volumes else 0
    avg_ask_vol = statistics.fmean(ask_volumes) if ask_volumes else 0
    
    wall_threshold_bid = avg_bid_vol * 3.5
    wall_threshold_ask = avg_ask_vol * 3.5
    
    bid_walls = []
    for price, volume in top_20_bids:
        if volume >= wall_threshold_bid:
            distance_pct = None
            if mid_price and mid_price > 0:
                distance_pct = ((mid_price - price) / mid_price) * 100
            bid_walls.append({
                "price": round(price, 8),
                "volume": round(volume, 4),
                "volume_ratio": round(volume / avg_bid_vol, 2) if avg_bid_vol > 0 else 0,
                "distance_from_mid_pct": round(distance_pct, 3) if distance_pct is not None else None,
            })
    
    ask_walls = []
    for price, volume in top_20_asks:
        if volume >= wall_threshold_ask:
            distance_pct = None
            if mid_price and mid_price > 0:
                distance_pct = ((price - mid_price) / mid_price) * 100
            ask_walls.append({
                "price": round(price, 8),
                "volume": round(volume, 4),
                "volume_ratio": round(volume / avg_ask_vol, 2) if avg_ask_vol > 0 else 0,
                "distance_from_mid_pct": round(distance_pct, 3) if distance_pct is not None else None,
            })
    
    total_bid_wall_vol = sum(w["volume"] for w in bid_walls)
    total_ask_wall_vol = sum(w["volume"] for w in ask_walls)
    
    wall_pressure = "neutral"
    if total_bid_wall_vol > total_ask_wall_vol * 1.5:
        wall_pressure = "bullish"
    elif total_ask_wall_vol > total_bid_wall_vol * 1.5:
        wall_pressure = "bearish"
    
    return {
        "bid_walls": bid_walls[:10],
        "ask_walls": ask_walls[:10],
        "wall_pressure": wall_pressure,
        "total_bid_wall_volume": round(total_bid_wall_vol, 4),
        "total_ask_wall_volume": round(total_ask_wall_vol, 4),
        "wall_count": {"bids": len(bid_walls), "asks": len(ask_walls)},
    }


def detect_layered_orders(
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    mid_price: Optional[float] = None,
) -> Dict[str, object]:
    """
    Detect layered orders - multiple orders at consecutive price levels.
    This is a common market maker strategy to provide liquidity.
    """
    if not bids or not asks:
        return {"bid_layers": [], "ask_layers": [], "layering_score": 0}
    
    bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda x: x[0])
    
    def find_layers(orders: Sequence[Tuple[float, float]], is_bid: bool) -> List[Dict[str, object]]:
        layers = []
        if len(orders) < 3:
            return layers
        
        volumes = [vol for _, vol in orders[:30]]
        if not volumes:
            return layers
        
        avg_vol = statistics.fmean(volumes)
        median_vol = statistics.median(volumes)
        threshold = max(avg_vol * 0.7, median_vol * 0.8)
        
        i = 0
        while i < len(orders) - 2:
            consecutive = []
            j = i
            
            while j < len(orders) and orders[j][1] >= threshold:
                consecutive.append(orders[j])
                j += 1
                
                if j < len(orders):
                    price_diff_pct = abs(orders[j-1][0] - orders[j][0]) / orders[j-1][0] * 100
                    if price_diff_pct > 0.5:
                        break
            
            if len(consecutive) >= 3:
                layer_volume = sum(vol for _, vol in consecutive)
                avg_price = sum(price * vol for price, vol in consecutive) / layer_volume if layer_volume > 0 else 0
                
                distance_pct = None
                if mid_price and mid_price > 0:
                    if is_bid:
                        distance_pct = ((mid_price - avg_price) / mid_price) * 100
                    else:
                        distance_pct = ((avg_price - mid_price) / mid_price) * 100
                
                layers.append({
                    "start_price": round(consecutive[0][0], 8),
                    "end_price": round(consecutive[-1][0], 8),
                    "levels": len(consecutive),
                    "total_volume": round(layer_volume, 4),
                    "avg_price": round(avg_price, 8),
                    "distance_from_mid_pct": round(distance_pct, 3) if distance_pct is not None else None,
                })
            
            i = j if j > i else i + 1
        
        return layers
    
    bid_layers = find_layers(bids_sorted, True)
    ask_layers = find_layers(asks_sorted, False)
    
    layering_score = min((len(bid_layers) + len(ask_layers)) * 10, 100)
    
    return {
        "bid_layers": bid_layers[:5],
        "ask_layers": ask_layers[:5],
        "layering_score": layering_score,
        "total_bid_layers": len(bid_layers),
        "total_ask_layers": len(ask_layers),
    }


def detect_quote_stuffing(
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    spread_pct: Optional[float] = None,
) -> Dict[str, object]:
    """
    Detect potential quote stuffing - too many similar-sized orders at very close price levels.
    This can indicate manipulation or aggressive market making.
    """
    if not bids or not asks:
        return {"stuffing_detected": False, "concentration_score": 0}
    
    bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda x: x[0])
    
    def analyze_concentration(orders: Sequence[Tuple[float, float]], top_n: int = 10) -> Dict[str, object]:
        if len(orders) < top_n:
            return {"density": 0, "similar_sizes": 0}
        
        top_orders = orders[:top_n]
        volumes = [vol for _, vol in top_orders]
        
        if not volumes:
            return {"density": 0, "similar_sizes": 0}
        
        avg_vol = statistics.fmean(volumes)
        std_vol = statistics.pstdev(volumes) if len(volumes) > 1 else 0
        
        cv = (std_vol / avg_vol) if avg_vol > 0 else 0
        
        similar_count = sum(1 for v in volumes if abs(v - avg_vol) < avg_vol * 0.15)
        
        prices = [price for price, _ in top_orders]
        price_range = max(prices) - min(prices)
        avg_price = statistics.fmean(prices)
        price_range_pct = (price_range / avg_price * 100) if avg_price > 0 else 0
        
        density = (similar_count / top_n * 100) if similar_count >= 3 and price_range_pct < 0.3 else 0
        
        return {
            "density": round(density, 2),
            "similar_sizes": similar_count,
            "coefficient_variation": round(cv, 3),
            "price_range_pct": round(price_range_pct, 4),
        }
    
    bid_concentration = analyze_concentration(bids_sorted, 10)
    ask_concentration = analyze_concentration(asks_sorted, 10)
    
    concentration_score = (bid_concentration["density"] + ask_concentration["density"]) / 2
    
    stuffing_detected = concentration_score > 60 and (
        bid_concentration["similar_sizes"] >= 5 or ask_concentration["similar_sizes"] >= 5
    )
    
    return {
        "stuffing_detected": stuffing_detected,
        "concentration_score": round(concentration_score, 2),
        "bid_concentration": bid_concentration,
        "ask_concentration": ask_concentration,
    }


def detect_spread_manipulation(
    best_bid: Optional[float],
    best_ask: Optional[float],
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
) -> Dict[str, object]:
    """
    Detect potential spread manipulation by analyzing order distribution near the spread.
    """
    if not best_bid or not best_ask or not bids or not asks:
        return {"manipulation_risk": "low", "spread_quality": "unknown"}
    
    spread = best_ask - best_bid
    spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
    mid_price = (best_bid + best_ask) / 2
    
    bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda x: x[0])
    
    near_bid_levels = [p for p, v in bids_sorted if p >= best_bid * 0.999]
    near_ask_levels = [p for p, v in asks_sorted if p <= best_ask * 1.001]
    
    near_bid_volumes = [v for p, v in bids_sorted if p >= best_bid * 0.999]
    near_ask_volumes = [v for p, v in asks_sorted if p <= best_ask * 1.001]
    
    best_bid_volume = bids_sorted[0][1] if bids_sorted else 0
    best_ask_volume = asks_sorted[0][1] if asks_sorted else 0
    
    second_bid_volume = bids_sorted[1][1] if len(bids_sorted) > 1 else 0
    second_ask_volume = asks_sorted[1][1] if len(asks_sorted) > 1 else 0
    
    top5_bid_vol = sum(v for _, v in bids_sorted[:5])
    top5_ask_vol = sum(v for _, v in asks_sorted[:5])
    
    top_level_dominance_bid = (best_bid_volume / top5_bid_vol) if top5_bid_vol > 0 else 0
    top_level_dominance_ask = (best_ask_volume / top5_ask_vol) if top5_ask_vol > 0 else 0
    
    manipulation_indicators = []
    manipulation_score = 0
    
    if spread_pct > 0.1:
        manipulation_indicators.append("wide_spread")
        manipulation_score += 20
    
    if top_level_dominance_bid > 0.6 or top_level_dominance_ask > 0.6:
        manipulation_indicators.append("top_level_dominance")
        manipulation_score += 25
    
    if best_bid_volume > second_bid_volume * 3 or best_ask_volume > second_ask_volume * 3:
        manipulation_indicators.append("volume_spike_at_best")
        manipulation_score += 20
    
    if len(near_bid_levels) <= 2 or len(near_ask_levels) <= 2:
        manipulation_indicators.append("thin_near_spread")
        manipulation_score += 15
    
    depth_imbalance = abs(top5_bid_vol - top5_ask_vol) / max(top5_bid_vol, top5_ask_vol, 1)
    if depth_imbalance > 0.5:
        manipulation_indicators.append("depth_imbalance")
        manipulation_score += 20
    
    manipulation_risk = "low"
    if manipulation_score >= 60:
        manipulation_risk = "high"
    elif manipulation_score >= 35:
        manipulation_risk = "medium"
    
    spread_quality = "good"
    if spread_pct > 0.15:
        spread_quality = "poor"
    elif spread_pct > 0.05:
        spread_quality = "fair"
    
    return {
        "manipulation_risk": manipulation_risk,
        "manipulation_score": manipulation_score,
        "manipulation_indicators": manipulation_indicators,
        "spread_pct": round(spread_pct, 4),
        "spread_quality": spread_quality,
        "best_bid_volume": round(best_bid_volume, 4),
        "best_ask_volume": round(best_ask_volume, 4),
        "top_level_dominance": {
            "bid": round(top_level_dominance_bid, 3),
            "ask": round(top_level_dominance_ask, 3),
        },
        "depth_imbalance": round(depth_imbalance, 3),
    }


def analyze_market_maker_activity(orderbook_data: Dict[str, object]) -> Dict[str, object]:
    """
    Comprehensive market maker detection combining multiple signals.
    """
    if not orderbook_data or orderbook_data.get("source") == "synthetic":
        return {
            "market_maker_detected": False,
            "confidence": 0,
            "activity_level": "unknown",
            "signals": [],
            "warning": "Using synthetic orderbook data - real market maker detection unavailable",
        }
    
    raw_levels = orderbook_data.get("raw_levels", {})
    bids = raw_levels.get("bids", [])
    asks = raw_levels.get("asks", [])
    
    best_bid = orderbook_data.get("best_bid")
    best_ask = orderbook_data.get("best_ask")
    mid_price = orderbook_data.get("mid_price")
    spread = orderbook_data.get("spread")
    
    if not bids or not asks:
        return {
            "market_maker_detected": False,
            "confidence": 0,
            "activity_level": "unknown",
            "signals": [],
            "error": "No orderbook levels available",
        }
    
    walls = detect_order_walls(bids, asks, mid_price)
    layers = detect_layered_orders(bids, asks, mid_price)
    stuffing = detect_quote_stuffing(bids, asks, (spread / best_bid * 100) if best_bid and spread else None)
    manipulation = detect_spread_manipulation(best_bid, best_ask, bids, asks)
    
    confidence = 0
    signals = []
    
    if walls["wall_count"]["bids"] + walls["wall_count"]["asks"] >= 3:
        signals.append("multiple_order_walls")
        confidence += 25
    
    if layers["layering_score"] > 30:
        signals.append("layered_orders")
        confidence += 20
    
    if layers["total_bid_layers"] >= 2 and layers["total_ask_layers"] >= 2:
        signals.append("balanced_layering")
        confidence += 15
    
    if stuffing["concentration_score"] > 50:
        signals.append("high_order_concentration")
        confidence += 15
    
    if manipulation["manipulation_score"] < 30 and manipulation["spread_quality"] in ["good", "fair"]:
        signals.append("healthy_spread")
        confidence += 10
    
    if manipulation["top_level_dominance"]["bid"] > 0.4 and manipulation["top_level_dominance"]["ask"] > 0.4:
        signals.append("strong_top_of_book")
        confidence += 15
    
    total_bid_vol = orderbook_data.get("total_bid_volume", 0)
    total_ask_vol = orderbook_data.get("total_ask_volume", 0)
    if total_bid_vol > 0 and total_ask_vol > 0:
        volume_balance = min(total_bid_vol, total_ask_vol) / max(total_bid_vol, total_ask_vol)
        if volume_balance > 0.7:
            signals.append("balanced_depth")
            confidence += 10
    
    market_maker_detected = confidence >= 40
    
    activity_level = "low"
    if confidence >= 70:
        activity_level = "high"
    elif confidence >= 50:
        activity_level = "medium"
    
    return {
        "market_maker_detected": market_maker_detected,
        "confidence": min(confidence, 100),
        "activity_level": activity_level,
        "signals": signals,
        "details": {
            "order_walls": walls,
            "layered_orders": layers,
            "quote_stuffing": stuffing,
            "spread_analysis": manipulation,
        },
        "interpretation": _get_interpretation(market_maker_detected, confidence, signals),
    }


def _get_interpretation(detected: bool, confidence: int, signals: List[str]) -> str:
    """Generate human-readable interpretation of market maker activity."""
    if not detected:
        return "Low market maker activity detected. Orderbook may have lower liquidity or more organic trading."
    
    interpretation_parts = [f"Market maker activity detected with {confidence}% confidence."]
    
    if "multiple_order_walls" in signals:
        interpretation_parts.append("Large order walls present, indicating professional liquidity provision.")
    
    if "layered_orders" in signals or "balanced_layering" in signals:
        interpretation_parts.append("Layered orders detected across multiple price levels.")
    
    if "high_order_concentration" in signals:
        interpretation_parts.append("High concentration of similar-sized orders may indicate algorithmic market making.")
    
    if "healthy_spread" in signals:
        interpretation_parts.append("Tight spreads suggest active market making.")
    
    if confidence >= 70:
        interpretation_parts.append("Strong evidence of professional market maker presence.")
    
    return " ".join(interpretation_parts)
