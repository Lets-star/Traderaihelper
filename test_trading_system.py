"""Unit tests for trading system core interfaces and JSON serialization."""

from __future__ import annotations

import json
from typing import Any, Dict

from indicator_collector.math_utils import Candle
from indicator_collector.trading_system import (
    AnalyzerContext,
    FactorScore,
    OptimizationStats,
    PositionPlan,
    SignalExplanation,
    TradingSignalPayload,
    deserialize_signal_payload,
    parse_collector_payload,
    serialize_signal_payload,
)
from indicator_collector.trading_system.market_structure_analyzer import (
    analyze_market_structure,
    calculate_structure_score,
)


def test_factor_score_roundtrip():
    """Test FactorScore serialization and deserialization."""
    original = FactorScore(
        factor_name="trend_strength",
        score=75.5,
        weight=2.0,
        description="Strong uptrend detected",
        emoji="🟢",
        metadata={"category": "trend"},
    )
    
    serialized = original.to_dict()
    deserialized = FactorScore.from_dict(serialized)
    
    assert deserialized.factor_name == original.factor_name
    assert deserialized.score == original.score
    assert deserialized.weight == original.weight
    assert deserialized.description == original.description
    assert deserialized.emoji == original.emoji
    assert deserialized.metadata["category"] == "trend"
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = FactorScore.from_dict(json_dict)
    
    assert from_json.factor_name == original.factor_name
    assert from_json.score == original.score
    print("✓ FactorScore round-trip test passed")


def test_position_plan_roundtrip():
    """Test PositionPlan serialization and deserialization."""
    original = PositionPlan(
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit_levels=[51000.0, 52000.0, 53000.0],
        position_size_usd=1000.0,
        risk_reward_ratio=3.0,
        max_risk_pct=2.0,
        leverage=10.0,
        direction="long",
        notes="Test position",
    )
    
    serialized = original.to_dict()
    deserialized = PositionPlan.from_dict(serialized)
    
    assert deserialized.entry_price == original.entry_price
    assert deserialized.stop_loss == original.stop_loss
    assert deserialized.take_profit_levels == original.take_profit_levels
    assert deserialized.position_size_usd == original.position_size_usd
    assert deserialized.risk_reward_ratio == original.risk_reward_ratio
    assert deserialized.max_risk_pct == original.max_risk_pct
    assert deserialized.leverage == original.leverage
    assert deserialized.direction == "long"
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = PositionPlan.from_dict(json_dict)
    
    assert from_json.entry_price == original.entry_price
    assert len(from_json.take_profit_levels) == 3
    print("✓ PositionPlan round-trip test passed")


def test_signal_explanation_roundtrip():
    """Test SignalExplanation serialization and deserialization."""
    original = SignalExplanation(
        primary_reason="Bullish FVG with strong volume confirmation",
        supporting_factors=[
            "RSI oversold recovery",
            "MACD bullish crossover",
            "Multi-timeframe alignment"
        ],
        risk_factors=[
            "High volatility environment",
            "Approaching major resistance"
        ],
        market_context="Uptrend continuation with healthy pullback",
        notes="Test explanation",
    )
    
    serialized = original.to_dict()
    deserialized = SignalExplanation.from_dict(serialized)
    
    assert deserialized.primary_reason == original.primary_reason
    assert deserialized.supporting_factors == original.supporting_factors
    assert deserialized.risk_factors == original.risk_factors
    assert deserialized.market_context == original.market_context
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = SignalExplanation.from_dict(json_dict)
    
    assert from_json.primary_reason == original.primary_reason
    assert len(from_json.supporting_factors) == 3
    print("✓ SignalExplanation round-trip test passed")


def test_optimization_stats_roundtrip():
    """Test OptimizationStats serialization and deserialization."""
    original = OptimizationStats(
        backtest_win_rate=65.5,
        avg_profit_pct=2.5,
        avg_loss_pct=-1.2,
        sharpe_ratio=1.8,
        total_signals=100,
        profitable_signals=65,
        losing_signals=35,
    )
    
    serialized = original.to_dict()
    deserialized = OptimizationStats.from_dict(serialized)
    
    assert deserialized.backtest_win_rate == original.backtest_win_rate
    assert deserialized.avg_profit_pct == original.avg_profit_pct
    assert deserialized.avg_loss_pct == original.avg_loss_pct
    assert deserialized.sharpe_ratio == original.sharpe_ratio
    assert deserialized.total_signals == original.total_signals
    assert deserialized.profitable_signals == original.profitable_signals
    assert deserialized.losing_signals == original.losing_signals
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = OptimizationStats.from_dict(json_dict)
    
    assert from_json.total_signals == original.total_signals
    assert from_json.profitable_signals == original.profitable_signals
    print("✓ OptimizationStats round-trip test passed")


def test_trading_signal_payload_roundtrip():
    """Test TradingSignalPayload serialization and deserialization."""
    original = TradingSignalPayload(
        signal_type="BUY",
        confidence=0.85,
        timestamp=1699000000000,
        symbol="BTCUSDT",
        timeframe="1h",
        factors=[
            FactorScore("trend_strength", 80.0, 2.0, "Strong uptrend"),
            FactorScore("volume_confidence", 0.75, 1.5, "Above average volume"),
        ],
        position_plan=PositionPlan(
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_levels=[51000.0, 52000.0],
            position_size_usd=1000.0,
            risk_reward_ratio=2.0,
            max_risk_pct=2.0,
        ),
        explanation=SignalExplanation(
            primary_reason="Bullish setup",
            supporting_factors=["Factor A", "Factor B"],
            risk_factors=["Risk A"],
            market_context="Uptrend"
        ),
        optimization_stats=OptimizationStats(
            backtest_win_rate=70.0,
            avg_profit_pct=2.0,
            avg_loss_pct=-1.0,
            sharpe_ratio=2.0,
            total_signals=50,
            profitable_signals=35,
            losing_signals=15,
        ),
    )
    
    serialized = original.to_dict()
    deserialized = TradingSignalPayload.from_dict(serialized)
    
    assert deserialized.signal_type == original.signal_type
    assert deserialized.confidence == original.confidence
    assert deserialized.timestamp == original.timestamp
    assert deserialized.symbol == original.symbol
    assert deserialized.timeframe == original.timeframe
    assert len(deserialized.factors) == len(original.factors)
    assert deserialized.factors[0].factor_name == original.factors[0].factor_name
    assert deserialized.position_plan.entry_price == original.position_plan.entry_price
    assert deserialized.explanation.primary_reason == original.explanation.primary_reason
    assert deserialized.optimization_stats is not None
    assert deserialized.optimization_stats.total_signals == original.optimization_stats.total_signals
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = TradingSignalPayload.from_dict(json_dict)
    
    assert from_json.signal_type == "BUY"
    assert from_json.confidence == 0.85
    assert len(from_json.factors) == 2
    print("✓ TradingSignalPayload round-trip test passed")


def test_trading_signal_payload_without_optimization():
    """Test TradingSignalPayload without optimization stats."""
    original = TradingSignalPayload(
        signal_type="NEUTRAL",
        confidence=0.50,
        timestamp=1699000000000,
        symbol="ETHUSDT",
        timeframe="15m",
        factors=[],
        position_plan=PositionPlan(
            entry_price=3000.0,
            stop_loss=2950.0,
            take_profit_levels=[3050.0],
            position_size_usd=500.0,
            risk_reward_ratio=1.0,
            max_risk_pct=1.5,
        ),
        explanation=SignalExplanation(
            primary_reason="No clear setup",
            supporting_factors=[],
            risk_factors=["High volatility"],
            market_context="Ranging"
        ),
        optimization_stats=None,
    )
    
    serialized = original.to_dict()
    assert serialized["optimization_stats"] is None
    
    deserialized = TradingSignalPayload.from_dict(serialized)
    assert deserialized.optimization_stats is None
    assert deserialized.signal_type == "NEUTRAL"
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = TradingSignalPayload.from_dict(json_dict)
    
    assert from_json.optimization_stats is None
    print("✓ TradingSignalPayload without optimization test passed")


def test_analyzer_context_roundtrip():
    """Test AnalyzerContext serialization and deserialization."""
    original = AnalyzerContext(
        symbol="BTCUSDT",
        timeframe="1h",
        timestamp=1699000000000,
        current_price=50000.0,
        ohlcv={
            "open": 49900.0,
            "high": 50100.0,
            "low": 49800.0,
            "close": 50000.0,
            "volume": 1000000.0,
        },
        indicators={
            "rsi": 55.0,
            "macd": 50.0,
            "trend_strength": 70.0,
        },
        volume_analysis={
            "cvd": {"latest": 5000, "change": 100},
        },
        market_structure={
            "trend": "bullish",
        },
        multi_timeframe={
            "5m": {"direction": "bullish"},
            "15m": {"direction": "bullish"},
        },
        zones=[
            {"type": "BullFVG", "top": 50500.0, "bottom": 50300.0},
        ],
        historical_signals=[
            {"bar_index": 10, "type": "bullish", "price": 49500.0},
        ],
        advanced_metrics={
            "breadth": {"fear_greed": 55},
        },
    )
    
    serialized = original.to_dict()
    deserialized = AnalyzerContext.from_dict(serialized)
    
    assert deserialized.symbol == original.symbol
    assert deserialized.timeframe == original.timeframe
    assert deserialized.timestamp == original.timestamp
    assert deserialized.current_price == original.current_price
    assert deserialized.ohlcv["close"] == original.ohlcv["close"]
    assert deserialized.indicators["rsi"] == original.indicators["rsi"]
    assert len(deserialized.zones) == len(original.zones)
    assert len(deserialized.historical_signals) == len(original.historical_signals)
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    from_json = AnalyzerContext.from_dict(json_dict)
    
    assert from_json.symbol == "BTCUSDT"
    assert from_json.current_price == 50000.0
    print("✓ AnalyzerContext round-trip test passed")


def test_parse_collector_payload():
    """Test parsing collector output into AnalyzerContext."""
    collector_payload = {
        "metadata": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "period": 100,
            "token": "test_token",
            "generated_at": "2023-11-01T00:00:00Z",
        },
        "latest": {
            "timestamp": 1699000000000,
            "close": 50000.0,
            "open": 49900.0,
            "high": 50100.0,
            "low": 49800.0,
            "volume": 1000000.0,
            "rsi": 55.0,
            "macd": 50.0,
            "macd_signal": 48.0,
            "macd_histogram": 2.0,
            "trend_strength": 70.0,
            "atr": 500.0,
            "vwap": 49950.0,
            "pattern_score": 65.0,
            "market_sentiment": 60.0,
            "structure_state": "bullish",
            "volume_confirmed": True,
            "volume_ratio": 1.5,
            "volume_confidence": 0.8,
            "confluence_score": 7.5,
            "confluence_bias": "bullish",
        },
        "advanced": {
            "volume_analysis": {
                "cvd": {"latest": 5000, "change": 100},
            },
            "market_structure": {
                "trend": "bullish",
            },
            "breadth": {
                "fear_greed": 55,
            },
        },
        "multi_timeframe": {
            "trend_strength": {
                "5m": 60.0,
                "15m": 65.0,
            },
            "direction": {
                "5m": "bullish",
                "15m": "bullish",
            },
        },
        "zones": [
            {"type": "BullFVG", "top": 50500.0, "bottom": 50300.0, "breaker": False},
        ],
        "signals": [
            {"bar_index": 10, "type": "bullish", "price": 49500.0},
            {"bar_index": 25, "type": "bearish", "price": 50500.0},
        ],
    }
    
    context = parse_collector_payload(collector_payload)
    
    assert context.symbol == "BTCUSDT"
    assert context.timeframe == "1h"
    assert context.timestamp == 1699000000000
    assert context.current_price == 50000.0
    assert context.ohlcv["open"] == 49900.0
    assert context.ohlcv["high"] == 50100.0
    assert context.ohlcv["low"] == 49800.0
    assert context.ohlcv["close"] == 50000.0
    assert context.ohlcv["volume"] == 1000000.0
    assert context.indicators["rsi"] == 55.0
    assert context.indicators["macd"] == 50.0
    assert context.indicators["trend_strength"] == 70.0
    assert context.indicators["structure_state"] == "bullish"
    assert context.volume_analysis["cvd"]["latest"] == 5000
    assert context.market_structure["trend"] == "bullish"
    assert len(context.zones) == 1
    assert len(context.historical_signals) == 2
    assert context.advanced_metrics["breadth"]["fear_greed"] == 55
    
    print("✓ parse_collector_payload test passed")


def test_serialize_and_deserialize_signal_payload():
    """Test serialize/deserialize helper functions."""
    signal = TradingSignalPayload(
        signal_type="BUY",
        confidence=0.85,
        timestamp=1699000000000,
        symbol="BTCUSDT",
        timeframe="1h",
        factors=[
            FactorScore("trend", 80.0, 2.0, "Strong"),
        ],
        position_plan=PositionPlan(
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_levels=[51000.0],
            position_size_usd=1000.0,
            risk_reward_ratio=1.0,
            max_risk_pct=2.0,
        ),
        explanation=SignalExplanation(
            primary_reason="Test",
            supporting_factors=["A"],
            risk_factors=["B"],
            market_context="Test"
        ),
    )
    
    serialized = serialize_signal_payload(signal)
    
    assert serialized["signal_type"] == "BUY"
    assert serialized["confidence"] == 0.85
    assert len(serialized["factors"]) == 1
    assert serialized["position_plan"]["entry_price"] == 50000.0
    
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    
    assert json_dict["signal_type"] == "BUY"
    assert json_dict["symbol"] == "BTCUSDT"
    
    deserialized = deserialize_signal_payload(json_dict)
    assert deserialized.signal_type == "BUY"
    assert deserialized.confidence == 0.85
    assert deserialized.symbol == "BTCUSDT"
    
    print("✓ serialize/deserialize signal payload test passed")


def test_empty_collector_payload():
    """Test parsing collector payload with missing/empty fields."""
    minimal_payload: Dict[str, Any] = {
        "metadata": {},
        "latest": {},
        "advanced": {},
    }
    
    context = parse_collector_payload(minimal_payload)
    
    assert context.symbol == ""
    assert context.timeframe == ""
    assert context.timestamp == 0
    assert context.current_price == 0.0
    assert context.ohlcv["open"] == 0.0
    assert context.ohlcv["close"] == 0.0
    assert context.volume_analysis == {}
    assert context.zones == []
    assert context.historical_signals == []
    
    print("✓ Empty collector payload test passed")


def test_full_integration_roundtrip():
    """Test full integration: collector -> context -> signal -> JSON -> back."""
    collector_payload = {
        "metadata": {
            "symbol": "ETHUSDT",
            "timeframe": "15m",
        },
        "latest": {
            "timestamp": 1699000000000,
            "close": 3000.0,
            "open": 2990.0,
            "high": 3010.0,
            "low": 2980.0,
            "volume": 500000.0,
            "rsi": 60.0,
            "trend_strength": 75.0,
        },
        "advanced": {
            "volume_analysis": {},
            "market_structure": {},
        },
        "multi_timeframe": {},
        "zones": [],
        "signals": [],
    }
    
    context = parse_collector_payload(collector_payload)
    
    signal = TradingSignalPayload(
        signal_type="BUY",
        confidence=0.80,
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=[
            FactorScore("rsi", context.indicators.get("rsi", 0), 1.0, "RSI favorable"),
        ],
        position_plan=PositionPlan(
            entry_price=context.current_price,
            stop_loss=context.current_price * 0.98,
            take_profit_levels=[context.current_price * 1.02, context.current_price * 1.04],
            position_size_usd=1000.0,
            risk_reward_ratio=2.0,
            max_risk_pct=2.0,
        ),
        explanation=SignalExplanation(
            primary_reason="Integration test",
            supporting_factors=["Test A"],
            risk_factors=["Test B"],
            market_context="Test context"
        ),
    )
    
    serialized = serialize_signal_payload(signal)
    json_str = json.dumps(serialized)
    json_dict = json.loads(json_str)
    
    reconstructed = deserialize_signal_payload(json_dict)
    
    assert reconstructed.symbol == "ETHUSDT"
    assert reconstructed.timeframe == "15m"
    assert reconstructed.signal_type == "BUY"
    assert reconstructed.confidence == 0.80
    assert reconstructed.position_plan.entry_price == 3000.0
    assert len(reconstructed.factors) == 1
    assert reconstructed.factors[0].factor_name == "rsi"
    
    print("✓ Full integration round-trip test passed")


def test_minimal_objects():
    """Test minimal/default object creation."""
    factor = FactorScore(
        factor_name="test",
        score=50.0,
    )
    assert factor.weight == 1.0
    assert factor.description is None
    
    plan = PositionPlan(entry_price=1000.0)
    assert plan.stop_loss is None
    assert plan.take_profit_levels == []
    
    explanation = SignalExplanation(primary_reason="test")
    assert explanation.supporting_factors == []
    assert explanation.risk_factors == []
    
    stats = OptimizationStats()
    assert stats.total_signals == 0
    assert stats.backtest_win_rate is None
    
    signal = TradingSignalPayload(
        signal_type="NEUTRAL",
        confidence=0.5,
        timestamp=0,
        symbol="TEST",
        timeframe="1m",
    )
    assert signal.factors == []
    assert signal.position_plan is None
    
    print("✓ Minimal objects test passed")


def _create_synthetic_candles_bullish() -> list[Candle]:
    """Create synthetic candles with bullish structure (HH/HL pattern)."""
    base_time = 1699000000000
    base_price = 50000.0
    
    candles = []
    # Create a clear uptrend with swing highs and lows
    # Pattern: rise to peak, pullback, rise to higher peak, pullback to higher low, repeat
    swing_pattern = [
        # First swing: rise
        (0, 100, 150), (100, 200, 250), (200, 300, 350), (300, 400, 450), (400, 500, 550),
        # Pullback
        (500, 400, 450), (400, 350, 400), (350, 300, 350),
        # Second swing: rise to higher high
        (300, 400, 450), (400, 500, 550), (500, 600, 650), (600, 700, 750), (700, 800, 850),
        # Pullback to higher low
        (800, 700, 750), (700, 650, 700), (650, 600, 650),
        # Third swing: rise to even higher high
        (600, 700, 750), (700, 800, 850), (800, 900, 950), (900, 1000, 1050), (1000, 1100, 1150),
        # Pullback to higher low
        (1100, 1000, 1050), (1000, 950, 1000), (950, 900, 950),
        # Fourth swing
        (900, 1000, 1050), (1000, 1100, 1150), (1100, 1200, 1250), (1200, 1300, 1350),
        # Final pullback
        (1300, 1200, 1250), (1200, 1150, 1200),
    ]
    
    for i, (low_offset, close_offset, high_offset) in enumerate(swing_pattern):
        time_offset = i * 3600000
        
        open_price = base_price + close_offset - 25
        close_price = base_price + close_offset
        high_price = base_price + high_offset
        low_price = base_price + low_offset
        
        candles.append(Candle(
            open_time=base_time + time_offset,
            close_time=base_time + time_offset + 3600000,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000000.0 + (i % 5) * 200000,
        ))
    
    return candles


def _create_synthetic_candles_bearish() -> list[Candle]:
    """Create synthetic candles with bearish structure (LH/LL pattern)."""
    base_time = 1699000000000
    base_price = 50000.0
    
    candles = []
    # Create a clear downtrend with swing highs and lows
    # Pattern: fall to low, bounce, fall to lower low, bounce to lower high, repeat
    swing_pattern = [
        # First swing: fall
        (1300, 1200, 1350), (1200, 1100, 1250), (1100, 1000, 1150), (1000, 900, 1050), (900, 800, 950),
        # Bounce
        (800, 900, 950), (900, 950, 1000), (950, 1000, 1050),
        # Second swing: fall to lower low
        (1000, 900, 950), (900, 800, 850), (800, 700, 750), (700, 600, 650), (600, 500, 550),
        # Bounce to lower high
        (500, 600, 650), (600, 650, 700), (650, 700, 750),
        # Third swing: fall to even lower low
        (700, 600, 650), (600, 500, 550), (500, 400, 450), (400, 300, 350), (300, 200, 250),
        # Bounce to lower high
        (200, 300, 350), (300, 350, 400), (350, 400, 450),
        # Fourth swing
        (400, 300, 350), (300, 200, 250), (200, 100, 150), (100, 50, 100),
        # Final bounce
        (50, 100, 150), (100, 150, 200),
    ]
    
    for i, (low_offset, close_offset, high_offset) in enumerate(swing_pattern):
        time_offset = i * 3600000
        
        open_price = base_price + close_offset + 25
        close_price = base_price + close_offset
        high_price = base_price + high_offset
        low_price = base_price + low_offset
        
        candles.append(Candle(
            open_time=base_time + time_offset,
            close_time=base_time + time_offset + 3600000,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000000.0 + (i % 5) * 200000,
        ))
    
    return candles


def _create_synthetic_candles_sideways() -> list[Candle]:
    """Create synthetic candles with sideways/ranging structure."""
    base_time = 1699000000000
    base_price = 50000.0
    
    candles = []
    # Create ranging market - oscillate between support and resistance with no clear trend
    # Highs and lows stay roughly at same levels
    swing_pattern = [
        # Swing up to resistance
        (600, 650, 700), (650, 700, 750), (700, 750, 800), (750, 800, 850), (800, 850, 900),
        # Pull back to support
        (850, 800, 850), (800, 750, 800), (750, 700, 750), (700, 650, 700), (650, 600, 650),
        # Swing up again to similar resistance
        (600, 650, 700), (650, 700, 750), (700, 750, 800), (750, 800, 850), (800, 850, 900),
        # Pull back to similar support
        (850, 800, 850), (800, 750, 800), (750, 700, 750), (700, 650, 700), (650, 600, 650),
        # Swing up again
        (600, 650, 700), (650, 700, 750), (700, 750, 800), (750, 800, 850), (800, 850, 900),
        # Pull back again
        (850, 800, 850), (800, 750, 800), (750, 700, 750), (700, 650, 700), (650, 600, 650),
        # Final swing up
        (600, 650, 700), (650, 700, 750), (700, 750, 800), (750, 800, 850),
    ]
    
    for i, (low_offset, close_offset, high_offset) in enumerate(swing_pattern):
        time_offset = i * 3600000
        
        open_price = base_price + (close_offset + low_offset) / 2
        close_price = base_price + close_offset
        high_price = base_price + high_offset
        low_price = base_price + low_offset
        
        candles.append(Candle(
            open_time=base_time + time_offset,
            close_time=base_time + time_offset + 3600000,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=1000000.0 + (i % 3) * 100000,
        ))
    
    return candles


def test_market_structure_analyzer_bullish():
    """Test market structure analyzer with bullish synthetic candles."""
    candles = _create_synthetic_candles_bullish()
    
    factor_score = calculate_structure_score(candles)
    
    assert factor_score.factor_name == 'market_structure'
    assert factor_score.weight == 0.3  # 30% weight
    assert factor_score.score > 0.65, f"Expected bullish score > 0.65, got {factor_score.score}"
    assert factor_score.emoji == '🟢', "Expected green emoji for bullish"
    assert 'bullish' in factor_score.description.lower()
    
    # Check metadata
    metadata = factor_score.metadata
    assert metadata['trend'] == 'bullish'
    assert metadata['swing_points']['hh_count'] > 0, "Expected Higher Highs in bullish trend"
    assert metadata['swing_points']['hl_count'] > 0, "Expected Higher Lows in bullish trend"
    
    # Check highlights
    highlights = metadata.get('highlights', [])
    assert len(highlights) > 0, "Expected highlights to be generated"
    assert any('bullish' in h.lower() for h in highlights), "Expected bullish mention in highlights"
    
    print(f"✓ Bullish structure test passed - Score: {factor_score.score:.3f}, Trend: {metadata['trend']}")


def test_market_structure_analyzer_bearish():
    """Test market structure analyzer with bearish synthetic candles."""
    candles = _create_synthetic_candles_bearish()
    
    factor_score = calculate_structure_score(candles)
    
    assert factor_score.factor_name == 'market_structure'
    assert factor_score.weight == 0.3  # 30% weight
    assert factor_score.score < 0.35, f"Expected bearish score < 0.35, got {factor_score.score}"
    assert factor_score.emoji == '🔴', "Expected red emoji for bearish"
    assert 'bearish' in factor_score.description.lower()
    
    # Check metadata
    metadata = factor_score.metadata
    assert metadata['trend'] == 'bearish'
    assert metadata['swing_points']['lh_count'] > 0, "Expected Lower Highs in bearish trend"
    assert metadata['swing_points']['ll_count'] > 0, "Expected Lower Lows in bearish trend"
    
    # Check highlights
    highlights = metadata.get('highlights', [])
    assert len(highlights) > 0, "Expected highlights to be generated"
    assert any('bearish' in h.lower() for h in highlights), "Expected bearish mention in highlights"
    
    print(f"✓ Bearish structure test passed - Score: {factor_score.score:.3f}, Trend: {metadata['trend']}")


def test_market_structure_analyzer_neutral():
    """Test market structure analyzer with sideways/neutral synthetic candles."""
    candles = _create_synthetic_candles_sideways()
    
    factor_score = calculate_structure_score(candles)
    
    assert factor_score.factor_name == 'market_structure'
    assert factor_score.weight == 0.3  # 30% weight
    assert 0.35 <= factor_score.score <= 0.65, f"Expected neutral score between 0.35-0.65, got {factor_score.score}"
    assert factor_score.emoji == '⚪', "Expected white emoji for neutral"
    
    # Check metadata
    metadata = factor_score.metadata
    assert metadata['trend'] == 'neutral'
    
    # Check highlights
    highlights = metadata.get('highlights', [])
    assert len(highlights) > 0, "Expected highlights to be generated"
    assert any('sideways' in h.lower() or 'ranging' in h.lower() or 'neutral' in h.lower() for h in highlights), \
        "Expected sideways/ranging/neutral mention in highlights"
    
    print(f"✓ Neutral structure test passed - Score: {factor_score.score:.3f}, Trend: {metadata['trend']}")


def test_market_structure_with_context():
    """Test market structure analyzer with AnalyzerContext."""
    candles = _create_synthetic_candles_bullish()
    
    # Create an AnalyzerContext with candles in extras
    context = AnalyzerContext(
        symbol='BTCUSDT',
        timeframe='1h',
        timestamp=1699000000000,
        current_price=55000.0,
        ohlcv={
            'open': 54900.0,
            'high': 55100.0,
            'low': 54800.0,
            'close': 55000.0,
            'volume': 1000000.0,
        },
        indicators={},
        extras={'candles': candles},
    )
    
    factor_score = analyze_market_structure(context)
    
    assert factor_score.factor_name == 'market_structure'
    assert factor_score.weight == 0.3
    assert factor_score.score > 0.6, "Expected bullish score from context"
    assert 'bullish' in factor_score.description.lower()
    
    # Check that metadata contains structure data
    metadata = factor_score.metadata
    assert 'trend' in metadata
    assert 'swing_points' in metadata
    assert 'highlights' in metadata
    
    print(f"✓ Context-based structure test passed - Score: {factor_score.score:.3f}")


def test_market_structure_insufficient_data():
    """Test market structure analyzer with insufficient data."""
    # Test with empty candles
    empty_candles: list[Candle] = []
    factor_score = calculate_structure_score(empty_candles)
    
    assert factor_score.factor_name == 'market_structure'
    assert factor_score.weight == 0.3
    assert factor_score.score == 0.5, "Expected neutral score for insufficient data"
    assert 'Insufficient' in factor_score.description or 'data' in factor_score.description.lower()
    
    # Test with very few candles
    few_candles = _create_synthetic_candles_bullish()[:5]
    factor_score2 = calculate_structure_score(few_candles)
    
    assert factor_score2.factor_name == 'market_structure'
    assert factor_score2.weight == 0.3
    # With few candles, structure detection may be limited but should still work
    assert 0.0 <= factor_score2.score <= 1.0
    
    print(f"✓ Insufficient data test passed")


if __name__ == "__main__":
    print("Running trading system core tests...")
    print()
    
    test_factor_score_roundtrip()
    test_position_plan_roundtrip()
    test_signal_explanation_roundtrip()
    test_optimization_stats_roundtrip()
    test_trading_signal_payload_roundtrip()
    test_trading_signal_payload_without_optimization()
    test_analyzer_context_roundtrip()
    test_parse_collector_payload()
    test_serialize_and_deserialize_signal_payload()
    test_empty_collector_payload()
    test_full_integration_roundtrip()
    test_minimal_objects()
    test_market_structure_analyzer_bullish()
    test_market_structure_analyzer_bearish()
    test_market_structure_analyzer_neutral()
    test_market_structure_with_context()
    test_market_structure_insufficient_data()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
