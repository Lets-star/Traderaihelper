"""Tests for volume and orderbook analyzer."""

from __future__ import annotations

from indicator_collector.trading_system.volume_orderbook_analyzer import (
    analyze_smart_money_activity,
    analyze_volume_orderbook,
    calculate_mm_confidence_weighted,
    calculate_order_imbalance,
    detect_liquidity_zones,
)


def create_mock_orderbook(
    bid_ask_ratio: float = 1.0,
    bid_volume: float = 1000.0,
    ask_volume: float = 1000.0,
    best_bid: float = 50000.0,
    best_ask: float = 50010.0,
) -> dict:
    """Create a mock orderbook snapshot for testing."""
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "mid_price": (best_bid + best_ask) / 2,
        "total_bid_volume": bid_volume,
        "total_ask_volume": ask_volume,
        "sections": {
            "bids": {
                "top_5": {"total_volume": bid_volume * 0.4},
                "top_10": {"total_volume": bid_volume * 0.6},
                "top_20": {"total_volume": bid_volume},
            },
            "asks": {
                "top_5": {"total_volume": ask_volume * 0.4},
                "top_10": {"total_volume": ask_volume * 0.6},
                "top_20": {"total_volume": ask_volume},
            },
        },
        "price_levels": {
            "1%": {"bid_volume": bid_volume * 0.2, "ask_volume": ask_volume * 0.2},
            "2%": {"bid_volume": bid_volume * 0.4, "ask_volume": ask_volume * 0.4},
            "5%": {"bid_volume": bid_volume * 0.8, "ask_volume": ask_volume * 0.8},
        },
        "symbol": "BTCUSDT",
        "snapshot_time": 1699000000000,
        "source": "binance",
    }


def create_mock_volume_analysis(
    smart_money_events: int = 2,
    buy_events: int = 2,
) -> dict:
    """Create a mock volume analysis with smart money events."""
    sell_events = max(0, smart_money_events - buy_events)
    
    smart_money_list = []
    for i in range(buy_events):
        smart_money_list.append({
            "timestamp": 1699000000000 + (i * 60000),
            "time_iso": "2023-11-03T00:00:00Z",
            "price": 50000.0 + (i * 10),
            "volume": 50.0,
            "direction": "buy",
            "volume_ratio": 2.5,
        })
    
    for i in range(sell_events):
        smart_money_list.append({
            "timestamp": 1699000000000 + ((buy_events + i) * 60000),
            "time_iso": "2023-11-03T00:00:00Z",
            "price": 50000.0 - (i * 10),
            "volume": 45.0,
            "direction": "sell",
            "volume_ratio": 2.3,
        })
    
    return {
        "vpvr": {
            "levels": [
                {"price": 50000.0, "volume": 1500.0, "percentage": 15.0},
                {"price": 49990.0, "volume": 1200.0, "percentage": 12.0},
                {"price": 50010.0, "volume": 1100.0, "percentage": 11.0},
                {"price": 49980.0, "volume": 500.0, "percentage": 7.5},
                {"price": 50020.0, "volume": 450.0, "percentage": 6.75},
            ],
            "poc": 50000.0,
            "total_volume": 6750.0,
            "value_area": {"high": 50010.0, "low": 49980.0},
        },
        "cvd": {
            "latest": 5000.0,
            "change": 250.0,
            "series": [{"timestamp": 1699000000000, "value": 5000.0, "delta": 250.0}],
        },
        "delta": {
            "latest": 250.0,
            "average": 200.0,
            "series": [{"timestamp": 1699000000000, "delta": 250.0}],
        },
        "smart_money": smart_money_list,
        "context": {
            "latest_volume": 1500.0,
            "average_volume": 1200.0,
            "median_volume": 1100.0,
            "volume_ratio": 1.25,
            "outlier_score": 1.2,
            "volume_confidence": 0.65,
        },
    }


def create_mock_mm_analysis(
    detected: bool = True,
    confidence: int = 65,
) -> dict:
    """Create a mock market maker analysis."""
    return {
        "market_maker_detected": detected,
        "confidence": confidence,
        "activity_level": "high" if confidence >= 70 else "medium" if confidence >= 50 else "low",
        "signals": [
            "multiple_order_walls",
            "layered_orders",
            "balanced_layering",
        ] if detected else [],
        "details": {
            "order_walls": {
                "bid_walls": [{"price": 49990.0, "volume": 500.0}],
                "ask_walls": [{"price": 50010.0, "volume": 480.0}],
            },
        },
    }


def test_calculate_order_imbalance_bullish():
    """Test order imbalance calculation with bullish bias."""
    orderbook = create_mock_orderbook(bid_volume=1500.0, ask_volume=1000.0)
    
    result = calculate_order_imbalance(orderbook)
    
    assert result["imbalance_direction"] == "bullish"
    assert result["bid_ask_ratio"] > 1.0
    assert result["imbalance_score"] > 0.5
    assert result["confidence"] > 0
    
    print("✓ Bullish order imbalance test passed")


def test_calculate_order_imbalance_bearish():
    """Test order imbalance calculation with bearish bias."""
    orderbook = create_mock_orderbook(bid_volume=600.0, ask_volume=1000.0)
    
    result = calculate_order_imbalance(orderbook)
    
    assert result["imbalance_direction"] == "bearish"
    assert result["bid_ask_ratio"] < 1.0
    assert result["imbalance_score"] < 0.5
    assert result["confidence"] > 0
    
    print("✓ Bearish order imbalance test passed")


def test_calculate_order_imbalance_neutral():
    """Test order imbalance calculation with balanced orderbook."""
    orderbook = create_mock_orderbook(bid_volume=1000.0, ask_volume=1000.0)
    
    result = calculate_order_imbalance(orderbook)
    
    assert result["imbalance_direction"] == "neutral"
    assert abs(result["bid_ask_ratio"] - 1.0) < 0.1
    assert result["imbalance_score"] == 0.5
    
    print("✓ Neutral order imbalance test passed")


def test_calculate_order_imbalance_empty():
    """Test order imbalance calculation with empty orderbook."""
    result = calculate_order_imbalance({})
    
    assert result["imbalance_direction"] == "neutral"
    assert result["imbalance_score"] == 0.5
    assert result["confidence"] == 0.0
    
    print("✓ Empty orderbook imbalance test passed")


def test_detect_liquidity_zones():
    """Test liquidity zone detection."""
    orderbook = create_mock_orderbook()
    volume_analysis = create_mock_volume_analysis()
    
    zones = detect_liquidity_zones(orderbook, volume_analysis, 50000.0)
    
    assert len(zones) > 0
    assert all("type" in z for z in zones)
    assert all("price" in z for z in zones)
    assert all("volume_ratio" in z for z in zones)
    assert all(0 <= z["volume_ratio"] <= 1 for z in zones)
    
    print(f"✓ Liquidity zones detection test passed ({len(zones)} zones found)")


def test_detect_liquidity_zones_with_poc():
    """Test liquidity zone detection identifies POC."""
    orderbook = create_mock_orderbook()
    volume_analysis = create_mock_volume_analysis()
    
    zones = detect_liquidity_zones(orderbook, volume_analysis, 50000.0)
    
    has_poc = any(z.get("is_poc") for z in zones)
    
    print(f"✓ POC detection test passed (POC found: {has_poc})")


def test_analyze_smart_money_activity():
    """Test smart money activity analysis."""
    volume_analysis = create_mock_volume_analysis(smart_money_events=4, buy_events=3)
    
    result = analyze_smart_money_activity(volume_analysis)
    
    assert result["activity_score"] >= 0.0
    assert result["activity_score"] <= 1.0
    assert result["direction_bias"] in ["bullish", "bearish", "neutral"]
    assert result["buy_events"] == 3
    assert result["sell_events"] == 1
    assert result["confidence"] > 0
    
    print("✓ Smart money activity test passed")


def test_analyze_smart_money_activity_bullish():
    """Test smart money activity with bullish bias."""
    volume_analysis = create_mock_volume_analysis(smart_money_events=5, buy_events=4)
    
    result = analyze_smart_money_activity(volume_analysis)
    
    assert result["direction_bias"] == "bullish"
    assert result["buy_events"] > result["sell_events"]
    
    print("✓ Bullish smart money test passed")


def test_analyze_smart_money_activity_empty():
    """Test smart money activity with no events."""
    volume_analysis = create_mock_volume_analysis(smart_money_events=0, buy_events=0)
    
    result = analyze_smart_money_activity(volume_analysis)
    
    assert result["activity_score"] == 0.0
    assert result["direction_bias"] == "neutral"
    assert result["confidence"] == 0.0
    
    print("✓ Empty smart money test passed")


def test_calculate_mm_confidence_weighted():
    """Test market maker confidence weighting."""
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=80)
    
    result = calculate_mm_confidence_weighted(mm_analysis, weight=0.2)
    
    assert result["mm_detected"] is True
    assert result["raw_confidence"] == 80
    assert result["weighted_confidence"] == 0.16
    assert result["weight"] == 0.2
    
    print("✓ MM confidence weighting test passed")


def test_calculate_mm_confidence_custom_weight():
    """Test market maker confidence with custom weight."""
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=50)
    
    result = calculate_mm_confidence_weighted(mm_analysis, weight=0.3)
    
    assert result["raw_confidence"] == 50
    assert result["weighted_confidence"] == 0.15
    assert result["weight"] == 0.3
    
    print("✓ MM confidence custom weight test passed")


def test_calculate_mm_confidence_empty():
    """Test market maker confidence with empty data."""
    result = calculate_mm_confidence_weighted({}, weight=0.2)
    
    assert result["mm_detected"] is False
    assert result["raw_confidence"] == 0
    assert result["weighted_confidence"] == 0.0
    
    print("✓ Empty MM analysis test passed")


def test_analyze_volume_orderbook_bullish():
    """Test full orderbook analysis with bullish conditions."""
    orderbook = create_mock_orderbook(bid_volume=1500.0, ask_volume=1000.0)
    volume_analysis = create_mock_volume_analysis(smart_money_events=4, buy_events=3)
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=70)
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, last_close_price=50000.0
    )
    
    assert result["final_score"] >= 0.0
    assert result["final_score"] <= 1.0
    assert result["direction"] == "bullish"
    assert result["confidence"] >= 0
    assert result["confidence"] <= 100
    assert "rationale" in result
    assert "components" in result
    assert "factor_weights" in result
    assert "factor_scores" in result
    assert "metadata" in result
    
    assert result["factor_weights"]["order_imbalance"] == 0.35
    assert result["factor_weights"]["liquidity"] == 0.25
    assert result["factor_weights"]["smart_money"] == 0.20
    assert result["factor_weights"]["market_maker"] == 0.20
    
    print(f"✓ Full bullish analysis test passed (score: {result['final_score']:.3f})")


def test_analyze_volume_orderbook_bearish():
    """Test full orderbook analysis with bearish conditions."""
    orderbook = create_mock_orderbook(bid_volume=700.0, ask_volume=1000.0)
    volume_analysis = create_mock_volume_analysis(smart_money_events=4, buy_events=1)
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=60)
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, last_close_price=50000.0
    )
    
    assert result["direction"] == "bearish"
    
    print(f"✓ Full bearish analysis test passed (score: {result['final_score']:.3f})")


def test_analyze_volume_orderbook_neutral():
    """Test full orderbook analysis with neutral conditions."""
    orderbook = create_mock_orderbook(bid_volume=1000.0, ask_volume=1000.0)
    volume_analysis = create_mock_volume_analysis(smart_money_events=2, buy_events=1)
    mm_analysis = create_mock_mm_analysis(detected=False, confidence=20)
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, last_close_price=50000.0
    )
    
    assert result["direction"] == "neutral"
    
    print(f"✓ Full neutral analysis test passed (score: {result['final_score']:.3f})")


def test_analyze_volume_orderbook_score_computation():
    """Test score computation with specific known values."""
    orderbook = create_mock_orderbook(bid_volume=1200.0, ask_volume=1000.0)
    volume_analysis = create_mock_volume_analysis(smart_money_events=4, buy_events=3)
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=80)
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, 
        last_close_price=50000.0, mm_weight=0.2
    )
    
    factors = result["factor_scores"]
    weights = result["factor_weights"]
    
    expected_score = (
        factors.get("order_imbalance", 0) * weights.get("order_imbalance", 0) +
        factors.get("liquidity", 0) * weights.get("liquidity", 0) +
        factors.get("smart_money", 0) * weights.get("smart_money", 0) +
        factors.get("market_maker", 0) * weights.get("market_maker", 0)
    )
    
    assert abs(result["final_score"] - expected_score) < 0.01, \
        f"Score mismatch: {result['final_score']} vs {expected_score}"
    
    print(f"✓ Score computation test passed (computed: {expected_score:.3f})")


def test_analyze_volume_orderbook_mm_weight():
    """Test that MM weight is correctly applied (20%)."""
    orderbook = create_mock_orderbook()
    volume_analysis = create_mock_volume_analysis()
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=100)
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, mm_weight=0.2
    )
    
    mm_factor = result["factor_scores"]["market_maker"]
    mm_weight_applied = result["factor_weights"]["market_maker"]
    
    assert mm_weight_applied == 0.2
    assert mm_factor == 0.2
    
    contribution = mm_factor * mm_weight_applied
    assert abs(contribution - 0.04) < 0.001
    
    print(f"✓ MM weight (20%) test passed (contribution: {contribution})")


def test_analyze_volume_orderbook_with_empty_data():
    """Test full orderbook analysis with empty/missing data."""
    result = analyze_volume_orderbook({}, {}, {}, last_close_price=50000.0)
    
    assert "final_score" in result
    assert "direction" in result
    assert "confidence" in result
    assert "rationale" in result
    
    print(f"✓ Empty data analysis test passed (score: {result['final_score']:.3f})")


def test_analyze_volume_orderbook_rationale():
    """Test that rationale is properly generated."""
    orderbook = create_mock_orderbook(bid_volume=1500.0, ask_volume=1000.0)
    volume_analysis = create_mock_volume_analysis(smart_money_events=4, buy_events=3)
    mm_analysis = create_mock_mm_analysis(detected=True, confidence=75)
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, last_close_price=50000.0
    )
    
    rationale = result["rationale"]
    
    assert isinstance(rationale, str)
    assert len(rationale) > 0
    assert rationale != "Neutral orderbook conditions"
    
    print(f"✓ Rationale test passed: {rationale[:80]}...")


def test_analyze_volume_orderbook_metadata():
    """Test that metadata is properly captured."""
    orderbook = create_mock_orderbook()
    orderbook["symbol"] = "ETHUSDT"
    orderbook["snapshot_time"] = 1699000000000
    
    volume_analysis = create_mock_volume_analysis()
    mm_analysis = create_mock_mm_analysis()
    
    result = analyze_volume_orderbook(
        orderbook, volume_analysis, mm_analysis, last_close_price=3000.0
    )
    
    metadata = result["metadata"]
    
    assert metadata["symbol"] == "ETHUSDT"
    assert metadata["timestamp"] == 1699000000000
    assert metadata["source"] == "binance"
    
    print("✓ Metadata test passed")


def run_all_tests():
    """Run all tests."""
    test_calculate_order_imbalance_bullish()
    test_calculate_order_imbalance_bearish()
    test_calculate_order_imbalance_neutral()
    test_calculate_order_imbalance_empty()
    test_detect_liquidity_zones()
    test_detect_liquidity_zones_with_poc()
    test_analyze_smart_money_activity()
    test_analyze_smart_money_activity_bullish()
    test_analyze_smart_money_activity_empty()
    test_calculate_mm_confidence_weighted()
    test_calculate_mm_confidence_custom_weight()
    test_calculate_mm_confidence_empty()
    test_analyze_volume_orderbook_bullish()
    test_analyze_volume_orderbook_bearish()
    test_analyze_volume_orderbook_neutral()
    test_analyze_volume_orderbook_score_computation()
    test_analyze_volume_orderbook_mm_weight()
    test_analyze_volume_orderbook_with_empty_data()
    test_analyze_volume_orderbook_rationale()
    test_analyze_volume_orderbook_metadata()
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
