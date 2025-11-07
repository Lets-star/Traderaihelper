"""Tests for technical analysis module."""

from __future__ import annotations

from indicator_collector.trading_system.technical_analysis import (
    analyze_atr,
    analyze_bollinger_bands,
    analyze_macd,
    analyze_rsi,
    analyze_technical_factors,
    detect_divergences,
)


def create_mock_candles(
    count: int = 50,
    start_price: float = 50000.0,
    trend: str = "neutral",
    volatility: float = 1.0,
) -> list[dict]:
    """
    Create mock candles for testing.
    
    Args:
        count: Number of candles to generate
        start_price: Starting price
        trend: "up", "down", or "neutral"
        volatility: Volatility multiplier
    """
    candles = []
    price = start_price
    
    for i in range(count):
        if trend == "up":
            price += 10 * volatility
        elif trend == "down":
            price -= 10 * volatility
        else:
            import math
            # Neutral trend with oscillation
            price += 5 * math.sin(i * 0.2) * volatility
        
        # Ensure price stays positive
        price = max(price, 100)
        
        close = price
        open_price = price - 5 * volatility
        high = price + 8 * volatility
        low = price - 8 * volatility
        volume = 100.0
        
        candles.append({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": 1699000000000 + (i * 60000),
        })
    
    return candles


# ============= MACD Tests =============

def test_macd_bullish_signal():
    """Test MACD with bullish crossover."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_macd(candles)
    
    assert isinstance(result, dict)
    assert "macd_score" in result
    assert "macd_direction" in result
    assert 0.0 <= result["macd_score"] <= 1.0
    assert result["macd_direction"] in ("bullish", "bearish", "neutral")


def test_macd_bearish_signal():
    """Test MACD with bearish crossover."""
    candles = create_mock_candles(count=50, trend="down", volatility=1.0)
    result = analyze_macd(candles)
    
    assert isinstance(result, dict)
    assert "macd_score" in result
    assert "macd_direction" in result
    assert 0.0 <= result["macd_score"] <= 1.0


def test_macd_insufficient_data():
    """Test MACD with insufficient data."""
    candles = create_mock_candles(count=5)
    result = analyze_macd(candles)
    
    assert result["macd_score"] == 0.5
    assert result["macd_direction"] == "neutral"
    assert result["confidence"] == 0.0


def test_macd_empty_data():
    """Test MACD with empty data."""
    result = analyze_macd([])
    
    assert result["macd_score"] == 0.5
    assert result["macd_direction"] == "neutral"
    assert result["confidence"] == 0.0


# ============= RSI Tests =============

def test_rsi_overbought():
    """Test RSI overbought condition."""
    # Create strongly uptrending candles to push RSI high
    candles = create_mock_candles(count=30, trend="up", volatility=2.0)
    result = analyze_rsi(candles)
    
    assert isinstance(result, dict)
    assert "rsi_score" in result
    assert "rsi_value" in result
    assert "rsi_state" in result
    assert 0.0 <= result["rsi_score"] <= 1.0


def test_rsi_oversold():
    """Test RSI oversold condition."""
    # Create strongly downtrending candles to push RSI low
    candles = create_mock_candles(count=30, trend="down", volatility=2.0)
    result = analyze_rsi(candles)
    
    assert isinstance(result, dict)
    assert "rsi_score" in result
    assert "rsi_state" in result
    assert 0.0 <= result["rsi_score"] <= 1.0


def test_rsi_neutral():
    """Test RSI in neutral zone."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=0.5)
    result = analyze_rsi(candles)
    
    assert isinstance(result, dict)
    assert "rsi_score" in result
    assert result["rsi_direction"] in ("bullish", "bearish", "neutral")
    assert 0.0 <= result["rsi_score"] <= 1.0


def test_rsi_insufficient_data():
    """Test RSI with insufficient data."""
    candles = create_mock_candles(count=5)
    result = analyze_rsi(candles)
    
    assert result["rsi_score"] == 0.5
    assert result["confidence"] == 0.0


# ============= ATR Tests =============

def test_atr_high_volatility():
    """Test ATR with high volatility."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=3.0)
    result = analyze_atr(candles)
    
    assert isinstance(result, dict)
    assert "atr_score" in result
    assert "atr_volatility" in result
    assert "atr_channels" in result
    assert isinstance(result["atr_channels"], dict)
    assert "upper" in result["atr_channels"]
    assert "lower" in result["atr_channels"]
    assert 0.0 <= result["atr_score"] <= 1.0


def test_atr_low_volatility():
    """Test ATR with low volatility."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=0.1)
    result = analyze_atr(candles)
    
    assert isinstance(result, dict)
    assert "atr_score" in result
    assert "atr_volatility" in result
    assert 0.0 <= result["atr_score"] <= 1.0


def test_atr_channels_calculation():
    """Test ATR channels are calculated correctly."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=1.0)
    result = analyze_atr(candles)
    
    channels = result["atr_channels"]
    assert channels["upper"] > channels["lower"]
    assert channels["width"] > 0


def test_atr_insufficient_data():
    """Test ATR with insufficient data."""
    candles = create_mock_candles(count=5)
    result = analyze_atr(candles)
    
    assert result["atr_score"] == 0.5
    assert result["confidence"] == 0.0


# ============= Bollinger Bands Tests =============

def test_bollinger_bands_squeeze():
    """Test Bollinger Bands squeeze detection."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=0.2)
    result = analyze_bollinger_bands(candles)
    
    assert isinstance(result, dict)
    assert "bollinger_score" in result
    assert "bollinger_state" in result
    assert "band_squeeze" in result
    assert 0.0 <= result["band_squeeze"] <= 1.0


def test_bollinger_bands_breakout():
    """Test Bollinger Bands breakout detection."""
    candles = create_mock_candles(count=30, trend="up", volatility=2.0)
    result = analyze_bollinger_bands(candles)
    
    assert isinstance(result, dict)
    assert "price_position" in result
    assert 0.0 <= result["price_position"] <= 1.0


def test_bollinger_bands_price_near_upper():
    """Test when price is near upper band."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.5)
    result = analyze_bollinger_bands(candles)
    
    assert "price_position" in result
    # After uptrend, price should be higher


def test_bollinger_bands_insufficient_data():
    """Test Bollinger Bands with insufficient data."""
    candles = create_mock_candles(count=5)
    result = analyze_bollinger_bands(candles)
    
    assert result["bollinger_score"] == 0.5
    assert result["confidence"] == 0.0


# ============= Divergence Tests =============

def test_divergence_detection():
    """Test divergence detection functionality."""
    candles = create_mock_candles(count=40, trend="neutral", volatility=1.0)
    result = detect_divergences(candles)
    
    assert isinstance(result, dict)
    assert "divergence_score" in result
    assert "divergence_type" in result
    assert result["divergence_type"] in (
        "none", "bullish", "bearish", "hidden_bullish", "hidden_bearish"
    )
    assert 0.0 <= result["divergence_score"] <= 1.0


def test_divergence_insufficient_data():
    """Test divergence detection with insufficient data."""
    candles = create_mock_candles(count=5)
    result = detect_divergences(candles)
    
    assert result["divergence_score"] == 0.5
    assert result["divergence_type"] == "none"
    assert result["confidence"] == 0.0


def test_divergence_structure():
    """Test divergence analysis returns proper structure."""
    candles = create_mock_candles(count=40, trend="up", volatility=1.0)
    result = detect_divergences(candles)
    
    assert "confidence" in result
    assert "rationale" in result
    assert "rsi_divergence" in result


# ============= Comprehensive Technical Analysis Tests =============

def test_technical_analysis_bullish():
    """Test comprehensive technical analysis with bullish setup."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.2)
    result = analyze_technical_factors(candles)
    
    assert isinstance(result, dict)
    assert "final_score" in result
    assert "direction" in result
    assert "confidence" in result
    assert "rationale" in result
    assert "components" in result
    assert "factor_weights" in result
    assert "factor_scores" in result
    
    assert 0.0 <= result["final_score"] <= 1.0
    assert result["direction"] in ("bullish", "bearish", "neutral")
    assert 0.0 <= result["confidence"] <= 100.0


def test_technical_analysis_bearish():
    """Test comprehensive technical analysis with bearish setup."""
    candles = create_mock_candles(count=50, trend="down", volatility=1.2)
    result = analyze_technical_factors(candles)
    
    assert isinstance(result, dict)
    assert "final_score" in result
    assert "direction" in result
    assert 0.0 <= result["final_score"] <= 1.0


def test_technical_analysis_neutral():
    """Test comprehensive technical analysis with neutral setup."""
    candles = create_mock_candles(count=50, trend="neutral", volatility=0.5)
    result = analyze_technical_factors(candles)
    
    assert isinstance(result, dict)
    assert "final_score" in result
    assert "direction" in result


def test_technical_analysis_component_breakdown():
    """Test all components are included in analysis."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_technical_factors(candles)
    
    components = result["components"]
    assert "macd" in components
    assert "rsi" in components
    assert "atr" in components
    assert "bollinger" in components
    assert "divergence" in components


def test_technical_analysis_weights():
    """Test weights sum to 1.0."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_technical_factors(candles)
    
    weights = result["factor_weights"]
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.01


def test_technical_analysis_factor_scores():
    """Test factor scores are in valid range."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_technical_factors(candles)
    
    scores = result["factor_scores"]
    for factor, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Score out of range for {factor}: {score}"


def test_technical_analysis_insufficient_data():
    """Test technical analysis with insufficient data."""
    candles = create_mock_candles(count=5)
    result = analyze_technical_factors(candles)
    
    assert result["final_score"] == 0.5
    assert result["direction"] == "neutral"
    assert result["confidence"] == 0.0


def test_technical_analysis_empty_data():
    """Test technical analysis with empty data."""
    result = analyze_technical_factors([])
    
    assert result["final_score"] == 0.5
    assert result["direction"] == "neutral"
    assert result["confidence"] == 0.0


# ============= Edge Case Tests =============

def test_technical_analysis_high_volatility():
    """Test technical analysis with extreme volatility."""
    candles = create_mock_candles(count=50, trend="neutral", volatility=5.0)
    result = analyze_technical_factors(candles)
    
    assert isinstance(result, dict)
    assert 0.0 <= result["final_score"] <= 1.0


def test_technical_analysis_stable_price():
    """Test technical analysis with very stable price."""
    candles = create_mock_candles(count=50, trend="neutral", volatility=0.05)
    result = analyze_technical_factors(candles)
    
    assert isinstance(result, dict)
    assert 0.0 <= result["final_score"] <= 1.0


def test_macd_all_fields_present():
    """Test MACD analysis returns all expected fields."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_macd(candles)
    
    required_fields = [
        "macd_score", "macd_direction", "confidence", "rationale",
        "macd_value", "signal_value", "histogram", "momentum"
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


def test_rsi_all_fields_present():
    """Test RSI analysis returns all expected fields."""
    candles = create_mock_candles(count=30, trend="up", volatility=1.0)
    result = analyze_rsi(candles)
    
    required_fields = [
        "rsi_score", "rsi_direction", "confidence", "rationale",
        "rsi_value", "rsi_state"
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


def test_atr_all_fields_present():
    """Test ATR analysis returns all expected fields."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=1.0)
    result = analyze_atr(candles)
    
    required_fields = [
        "atr_score", "atr_volatility", "confidence", "rationale",
        "atr_value", "atr_channels"
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


def test_bollinger_all_fields_present():
    """Test Bollinger Bands analysis returns all expected fields."""
    candles = create_mock_candles(count=30, trend="neutral", volatility=1.0)
    result = analyze_bollinger_bands(candles)
    
    required_fields = [
        "bollinger_score", "bollinger_state", "confidence", "rationale",
        "price_position", "band_squeeze"
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


def test_divergence_all_fields_present():
    """Test divergence analysis returns all expected fields."""
    candles = create_mock_candles(count=40, trend="up", volatility=1.0)
    result = detect_divergences(candles)
    
    required_fields = [
        "divergence_score", "divergence_type", "confidence", "rationale",
    ]
    for field in required_fields:
        assert field in result, f"Missing field: {field}"


def test_technical_analysis_metadata():
    """Test technical analysis includes metadata."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_technical_factors(candles)
    
    metadata = result.get("metadata", {})
    assert "total_candles" in metadata
    assert metadata["total_candles"] == 50


def test_multiple_consecutive_analyses():
    """Test multiple consecutive analyses produce consistent output."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    
    result1 = analyze_technical_factors(candles)
    result2 = analyze_technical_factors(candles)
    
    # Should produce identical results for same input
    assert result1["final_score"] == result2["final_score"]
    assert result1["direction"] == result2["direction"]


def test_analysis_consistency_across_indicators():
    """Test that multiple indicator analyses are consistent."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.2)
    
    macd = analyze_macd(candles)
    rsi = analyze_rsi(candles)
    
    # Both should be in valid range
    assert 0.0 <= macd["macd_score"] <= 1.0
    assert 0.0 <= rsi["rsi_score"] <= 1.0


def test_scoring_boundaries():
    """Test that all scores respect boundaries."""
    candles = create_mock_candles(count=50, trend="up", volatility=2.0)
    result = analyze_technical_factors(candles)
    
    assert 0.0 <= result["final_score"] <= 1.0
    
    for factor_score in result["factor_scores"].values():
        assert 0.0 <= factor_score <= 1.0


# ============= Integration Tests =============

def test_technical_analysis_on_various_trends():
    """Test technical analysis on various trend types."""
    trends = ["up", "down", "neutral"]
    
    for trend in trends:
        candles = create_mock_candles(count=50, trend=trend, volatility=1.0)
        result = analyze_technical_factors(candles)
        
        assert isinstance(result, dict)
        assert 0.0 <= result["final_score"] <= 1.0
        assert result["direction"] in ("bullish", "bearish", "neutral")


def test_technical_analysis_rationale_not_empty():
    """Test that rationale is always provided."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    result = analyze_technical_factors(candles)
    
    assert result["rationale"]
    assert len(result["rationale"]) > 0


def test_all_analyses_have_confidence():
    """Test all analyses include confidence metric."""
    candles = create_mock_candles(count=50, trend="up", volatility=1.0)
    
    analyses = [
        analyze_macd(candles),
        analyze_rsi(candles),
        analyze_atr(candles),
        analyze_bollinger_bands(candles),
        detect_divergences(candles),
    ]
    
    for analysis in analyses:
        assert "confidence" in analysis
        assert isinstance(analysis["confidence"], (int, float))


if __name__ == "__main__":
    # Run all tests
    import inspect
    import sys
    
    current_module = sys.modules[__name__]
    test_functions = [
        (name, obj) for name, obj in inspect.getmembers(current_module)
        if inspect.isfunction(obj) and name.startswith("test_")
    ]
    
    print(f"Running {len(test_functions)} tests...")
    passed = 0
    failed = 0
    
    for test_name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
