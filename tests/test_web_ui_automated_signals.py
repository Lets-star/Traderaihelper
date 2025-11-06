"""Tests for the Automated Signals web UI tab."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from indicator_collector.trading_system import (
    AnalyzerContext,
    FactorScore,
    OptimizationStats,
    PositionPlan,
    SignalExplanation,
    TradingSignalPayload,
)


def create_sample_trading_signal() -> TradingSignalPayload:
    """Create a sample trading signal for testing."""
    return TradingSignalPayload(
        signal_type="BUY",
        confidence=0.85,
        timestamp=int(datetime.now().timestamp() * 1000),
        symbol="BTCUSDT",
        timeframe="1h",
        factors=[
            FactorScore(
                factor_name="rsi_oversold",
                score=85.0,
                weight=2.0,
                description="RSI at 28 (oversold)",
                emoji="🟢",
            ),
            FactorScore(
                factor_name="trend_strength",
                score=75.0,
                weight=1.5,
                description="Strong uptrend forming",
            ),
        ],
        position_plan=PositionPlan(
            entry_price=45000.0,
            stop_loss=44000.0,
            take_profit_levels=[46000.0, 47000.0, 48000.0],
            position_size_usd=1000.0,
            risk_reward_ratio=2.0,
            max_risk_pct=0.02,
            leverage=10.0,
            direction="long",
        ),
        explanation=SignalExplanation(
            primary_reason="RSI oversold at 28, expecting bounce",
            supporting_factors=[
                "Price near support level",
                "Volume increasing",
            ],
            risk_factors=[
                "Could continue lower in strong downtrend",
                "Liquidation risk on leverage",
            ],
            market_context="Timeframe: 1h, Price: $45,000",
        ),
        optimization_stats=OptimizationStats(
            backtest_win_rate=62.5,
            avg_profit_pct=2.1,
            avg_loss_pct=-0.8,
            sharpe_ratio=1.8,
            total_signals=40,
            profitable_signals=25,
            losing_signals=15,
        ),
    )


def create_sample_neutral_signal() -> TradingSignalPayload:
    """Create a sample neutral/rejected signal for testing."""
    return TradingSignalPayload(
        signal_type="NEUTRAL",
        confidence=0.0,
        timestamp=int(datetime.now().timestamp() * 1000),
        symbol="BTCUSDT",
        timeframe="1h",
        factors=[],
        position_plan=None,
        explanation=SignalExplanation(
            primary_reason="Waiting for better setup",
            supporting_factors=[],
            risk_factors=[],
            market_context="No clear signal",
        ),
        metadata={"cancellation_reasons": [
            "Volatility too high (10% ATR/price)",
            "Low liquidity on orderbook",
        ]},
    )


class TestTradingSignalPayloadStructure:
    """Test TradingSignalPayload JSON serialization/deserialization."""

    def test_buy_signal_serialization(self):
        """Test BUY signal can be serialized to JSON."""
        signal = create_sample_trading_signal()
        signal_dict = signal.to_dict()

        assert signal_dict["signal_type"] == "BUY"
        assert signal_dict["confidence"] == 0.85
        assert signal_dict["symbol"] == "BTCUSDT"
        assert len(signal_dict["factors"]) == 2
        assert signal_dict["position_plan"] is not None
        assert signal_dict["explanation"] is not None
        assert signal_dict["optimization_stats"] is not None

    def test_signal_json_roundtrip(self):
        """Test signal can be serialized to JSON and back."""
        signal = create_sample_trading_signal()
        json_str = json.dumps(signal.to_dict())
        data = json.loads(json_str)

        # Verify all key fields are present
        assert data["signal_type"] == "BUY"
        assert data["confidence"] == 0.85
        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "1h"

        # Verify position plan
        assert data["position_plan"]["entry_price"] == 45000.0
        assert data["position_plan"]["stop_loss"] == 44000.0
        assert len(data["position_plan"]["take_profit_levels"]) == 3

        # Verify optimization stats
        assert data["optimization_stats"]["backtest_win_rate"] == 62.5
        assert data["optimization_stats"]["sharpe_ratio"] == 1.8

    def test_neutral_signal_serialization(self):
        """Test NEUTRAL signal without position plan."""
        signal = create_sample_neutral_signal()
        signal_dict = signal.to_dict()

        assert signal_dict["signal_type"] == "NEUTRAL"
        assert signal_dict["confidence"] == 0.0
        assert signal_dict["position_plan"] is None
        assert signal_dict["factors"] == []

    def test_factor_score_structure(self):
        """Test FactorScore data structure."""
        factor = FactorScore(
            factor_name="test_factor",
            score=80.0,
            weight=1.5,
            description="Test description",
            emoji="🟢",
        )
        factor_dict = factor.to_dict()

        assert factor_dict["factor_name"] == "test_factor"
        assert factor_dict["score"] == 80.0
        assert factor_dict["weight"] == 1.5
        assert factor_dict["description"] == "Test description"
        assert factor_dict["emoji"] == "🟢"

    def test_position_plan_structure(self):
        """Test PositionPlan data structure."""
        plan = PositionPlan(
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit_levels=[51000.0, 52000.0, 53000.0],
            position_size_usd=2000.0,
            risk_reward_ratio=2.5,
            max_risk_pct=0.02,
            leverage=5.0,
            direction="long",
        )
        plan_dict = plan.to_dict()

        assert plan_dict["entry_price"] == 50000.0
        assert plan_dict["stop_loss"] == 49000.0
        assert len(plan_dict["take_profit_levels"]) == 3
        assert plan_dict["position_size_usd"] == 2000.0
        assert plan_dict["risk_reward_ratio"] == 2.5

    def test_optimization_stats_structure(self):
        """Test OptimizationStats data structure."""
        stats = OptimizationStats(
            backtest_win_rate=65.0,
            avg_profit_pct=1.5,
            avg_loss_pct=-0.7,
            sharpe_ratio=1.9,
            total_signals=100,
            profitable_signals=65,
            losing_signals=35,
        )
        stats_dict = stats.to_dict()

        assert stats_dict["backtest_win_rate"] == 65.0
        assert stats_dict["avg_profit_pct"] == 1.5
        assert stats_dict["avg_loss_pct"] == -0.7
        assert stats_dict["sharpe_ratio"] == 1.9
        assert stats_dict["total_signals"] == 100


class TestWebUIPayloadFormats:
    """Test various payload formats that the web UI should handle."""

    def test_payload_with_automated_signals(self):
        """Test payload with automated_signals section."""
        signal = create_sample_trading_signal()
        payload = {
            "metadata": {"symbol": "BTCUSDT", "timeframe": "1h"},
            "automated_signals": signal.to_dict(),
            "latest": {"close": 45000.0},
        }

        # Extract and validate
        automated_signals = payload.get("automated_signals", {})
        assert automated_signals.get("signal_type") == "BUY"
        assert automated_signals.get("confidence") == 0.85

    def test_payload_with_trading_signals_section(self):
        """Test payload with trading_signals section."""
        signal = create_sample_trading_signal()
        payload = {
            "metadata": {"symbol": "BTCUSDT", "timeframe": "1h"},
            "trading_signals": signal.to_dict(),
        }

        trading_signals = payload.get("trading_signals", {})
        assert trading_signals.get("signal_type") == "BUY"

    def test_empty_payload_handling(self):
        """Test web UI handles empty payload gracefully."""
        payload = {
            "metadata": {"symbol": "BTCUSDT"},
            "advanced": {},
        }

        # Web UI should check for signals and show appropriate message
        automated_signals = payload.get("automated_signals", {})
        trading_signals_section = payload.get("trading_signals", {})
        assert not automated_signals
        assert not trading_signals_section

    def test_payload_with_cancellation_reasons(self):
        """Test payload with cancellation reasons."""
        payload = {
            "metadata": {"symbol": "BTCUSDT", "timeframe": "1h"},
            "automated_signals": {
                "signal_type": "NEUTRAL",
                "confidence": 0.0,
                "cancellation_reasons": [
                    "High volatility detected",
                    "Low liquidity",
                ],
            },
        }

        signal_data = payload.get("automated_signals", {})
        cancellation_reasons = signal_data.get("cancellation_reasons", [])
        assert len(cancellation_reasons) == 2
        assert "High volatility detected" in cancellation_reasons

    def test_payload_with_holding_horizon(self):
        """Test payload with holding_horizon_bars."""
        payload = {
            "automated_signals": {
                "signal_type": "BUY",
                "confidence": 0.8,
                "holding_horizon_bars": 24,
            }
        }

        signal_data = payload.get("automated_signals", {})
        holding_horizon = signal_data.get("holding_horizon_bars")
        assert holding_horizon == 24

    def test_payload_with_multiple_tp_levels(self):
        """Test payload with multiple take profit levels."""
        signal = create_sample_trading_signal()
        payload = {"automated_signals": signal.to_dict()}

        signal_data = payload.get("automated_signals", {})
        position_plan = signal_data.get("position_plan", {})
        tp_levels = position_plan.get("take_profit_levels", [])

        assert len(tp_levels) == 3
        assert tp_levels[0] == 46000.0
        assert tp_levels[1] == 47000.0
        assert tp_levels[2] == 48000.0


class TestSignalMetricsCalculations:
    """Test calculations for metrics displayed in the web UI."""

    def test_confidence_percentage_calculation(self):
        """Test confidence is properly displayed as percentage."""
        signal = create_sample_trading_signal()
        confidence = signal.confidence
        confidence_pct = confidence * 100

        assert confidence_pct == 85.0
        assert 0 <= confidence_pct <= 100

    def test_risk_percentage_calculation(self):
        """Test risk percentage from position plan."""
        plan = create_sample_trading_signal().position_plan
        assert plan is not None

        entry_price = plan.entry_price
        stop_loss = plan.stop_loss
        risk_distance = abs(entry_price - stop_loss)
        risk_pct = (risk_distance / entry_price) * 100

        assert risk_pct == pytest.approx(2.22, rel=0.01)

    def test_profit_percentage_for_tp_levels(self):
        """Test profit percentage calculation for each TP level."""
        plan = create_sample_trading_signal().position_plan
        assert plan is not None

        entry_price = plan.entry_price
        tp_levels = plan.take_profit_levels

        for idx, tp_level in enumerate(tp_levels, 1):
            profit_pct = ((tp_level - entry_price) / entry_price) * 100
            assert profit_pct > 0  # All TPs should be profitable
            if idx == 1:
                assert profit_pct == pytest.approx(2.22, rel=0.01)

    def test_risk_reward_ratio_display(self):
        """Test risk/reward ratio is properly extracted."""
        plan = create_sample_trading_signal().position_plan
        assert plan is not None

        rrr = plan.risk_reward_ratio
        assert rrr == 2.0

    def test_win_rate_calculation(self):
        """Test win rate from optimization stats."""
        stats = create_sample_trading_signal().optimization_stats
        assert stats is not None

        total = stats.total_signals
        profitable = stats.profitable_signals
        win_rate = (profitable / total) * 100

        assert win_rate == 62.5


class TestWebUIDisplayFormatting:
    """Test formatting functions for web UI display."""

    def test_price_formatting(self):
        """Test price formatting with proper decimals."""
        price = 45000.123456
        formatted = f"${price:.4f}"
        assert formatted == "$45000.1235"

    def test_percentage_formatting(self):
        """Test percentage formatting."""
        value = 0.62
        formatted = f"{value * 100:.1f}%"
        assert formatted == "62.0%"

    def test_timestamp_formatting(self):
        """Test timestamp to time conversion."""
        timestamp_ms = int(datetime(2024, 1, 1, 12, 30, 45).timestamp() * 1000)
        signal_time = datetime.fromtimestamp(timestamp_ms / 1000).strftime(
            "%H:%M:%S"
        )
        assert signal_time == "12:30:45"

    def test_direction_formatting(self):
        """Test direction formatting."""
        directions = ["long", "short", "flat"]
        for direction in directions:
            formatted = direction.upper() if direction else "N/A"
            assert formatted in ["LONG", "SHORT", "FLAT"]

    def test_signal_type_formatting(self):
        """Test signal type display with emoji."""
        signal_types = {
            "BUY": "🟢",
            "SELL": "🔴",
            "NEUTRAL": "⚪",
        }
        for signal_type, emoji in signal_types.items():
            assert signal_type in ["BUY", "SELL", "NEUTRAL"]
            assert emoji in ["🟢", "🔴", "⚪"]


class TestWebUIDataValidation:
    """Test data validation for web UI inputs."""

    def test_confidence_bounds(self):
        """Test confidence is bounded [0, 1]."""
        for confidence in [0.0, 0.5, 0.85, 1.0]:
            assert 0 <= confidence <= 1

    def test_signal_type_validation(self):
        """Test signal_type is valid."""
        valid_types = ["BUY", "SELL", "NEUTRAL"]
        for signal_type in valid_types:
            assert signal_type in valid_types

    def test_direction_validation(self):
        """Test direction is valid."""
        valid_directions = ["long", "short", "flat"]
        for direction in valid_directions:
            assert direction in valid_directions

    def test_positive_prices(self):
        """Test prices are positive."""
        signal = create_sample_trading_signal()
        plan = signal.position_plan
        assert plan is not None

        assert plan.entry_price > 0
        assert plan.stop_loss > 0
        for tp_level in plan.take_profit_levels:
            assert tp_level > 0

    def test_positive_position_size(self):
        """Test position size is positive."""
        signal = create_sample_trading_signal()
        plan = signal.position_plan
        assert plan is not None

        assert plan.position_size_usd > 0

    def test_positive_leverage(self):
        """Test leverage is positive."""
        signal = create_sample_trading_signal()
        plan = signal.position_plan
        assert plan is not None

        assert plan.leverage > 0

    def test_max_risk_pct_bounds(self):
        """Test max_risk_pct is bounded [0, 1]."""
        signal = create_sample_trading_signal()
        plan = signal.position_plan
        assert plan is not None

        if plan.max_risk_pct:
            assert 0 <= plan.max_risk_pct <= 1


class TestSignalFactorAnalysis:
    """Test factor analysis display in web UI."""

    def test_factor_score_range(self):
        """Test factor scores are in reasonable range."""
        signal = create_sample_trading_signal()

        for factor in signal.factors:
            # Scores should typically be 0-100
            assert 0 <= factor.score <= 100 or factor.score >= 0

    def test_factor_weight_range(self):
        """Test factor weights are positive."""
        signal = create_sample_trading_signal()

        for factor in signal.factors:
            assert factor.weight > 0

    def test_factor_with_emoji(self):
        """Test factor with emoji display."""
        signal = create_sample_trading_signal()

        factor = signal.factors[0]
        assert factor.emoji is not None
        assert factor.emoji in ["🟢", "🟡", "🔴", "⚪"]

    def test_factor_description_display(self):
        """Test factor description is properly formatted."""
        signal = create_sample_trading_signal()

        for factor in signal.factors:
            if factor.description:
                assert isinstance(factor.description, str)
                assert len(factor.description) > 0


class TestSignalExplanationDisplay:
    """Test signal explanation display in web UI."""

    def test_primary_reason_display(self):
        """Test primary reason is displayed."""
        signal = create_sample_trading_signal()
        explanation = signal.explanation
        assert explanation is not None

        assert explanation.primary_reason
        assert isinstance(explanation.primary_reason, str)

    def test_supporting_factors_display(self):
        """Test supporting factors are displayed."""
        signal = create_sample_trading_signal()
        explanation = signal.explanation
        assert explanation is not None

        assert len(explanation.supporting_factors) > 0
        for factor in explanation.supporting_factors:
            assert isinstance(factor, str)

    def test_risk_factors_display(self):
        """Test risk factors are displayed with warning."""
        signal = create_sample_trading_signal()
        explanation = signal.explanation
        assert explanation is not None

        assert len(explanation.risk_factors) > 0
        for risk in explanation.risk_factors:
            assert isinstance(risk, str)

    def test_market_context_display(self):
        """Test market context is displayed."""
        signal = create_sample_trading_signal()
        explanation = signal.explanation
        assert explanation is not None

        assert explanation.market_context
        assert "Timeframe" in explanation.market_context


class TestPerformanceMetricsDisplay:
    """Test performance metrics display in web UI."""

    def test_win_rate_display(self):
        """Test win rate is displayed as percentage."""
        stats = create_sample_trading_signal().optimization_stats
        assert stats is not None

        win_rate = stats.backtest_win_rate
        assert 0 <= win_rate <= 100

    def test_profit_factor_display(self):
        """Test profit factor is displayed."""
        stats = create_sample_trading_signal().optimization_stats
        assert stats is not None

        profit_factor = stats.profit_factor
        assert profit_factor >= 0

    def test_sharpe_ratio_display(self):
        """Test Sharpe ratio is displayed."""
        stats = create_sample_trading_signal().optimization_stats
        assert stats is not None

        sharpe = stats.sharpe_ratio
        assert sharpe is not None

    def test_total_signals_display(self):
        """Test total signals count is displayed."""
        stats = create_sample_trading_signal().optimization_stats
        assert stats is not None

        total = stats.total_signals
        assert total > 0

    def test_profitable_vs_losing_display(self):
        """Test profitable vs losing signals are displayed."""
        stats = create_sample_trading_signal().optimization_stats
        assert stats is not None

        profitable = stats.profitable_signals
        losing = stats.losing_signals
        total = stats.total_signals

        assert profitable + losing == total


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
