import math

import pytest

from indicator_collector.trading_system.generate_signals import generate_signals


class TestGenerateSignalsOutput:
    def test_generate_signals_actionable_buy(self):
        payload = {
            "signal_type": "BUY",
            "confidence": 0.72,
            "timestamp": 1700000000000,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "factors": [
                {
                    "factor_name": "technical_analysis",
                    "score": 0.74,
                    "weight": 0.25,
                    "metadata": {"direction": "bullish"},
                },
                {
                    "factor_name": "sentiment",
                    "score": 0.68,
                    "weight": 0.15,
                    "metadata": {"direction": "bullish"},
                },
                {
                    "factor_name": "multitimeframe_alignment",
                    "score": 0.65,
                    "weight": 0.10,
                    "metadata": {"direction": "bullish"},
                },
                {
                    "factor_name": "volume_analysis",
                    "score": 0.62,
                    "weight": 0.20,
                    "metadata": {"direction": "bullish"},
                },
                {
                    "factor_name": "market_structure",
                    "score": 0.70,
                    "weight": 0.15,
                    "metadata": {"direction": "bullish"},
                },
            ],
            "position_plan": {
                "entry_price": 25_000.0,
                "stop_loss": 24_500.0,
                "take_profit_levels": [25_250.0, 25_500.0, 26_000.0],
                "position_size_usd": 1_500.0,
                "direction": "long",
                "leverage": 5.0,
                "metadata": {
                    "atr": 150.0,
                    "holding_horizon_bars": 18,
                    "sizing_factors": {"risk_amount_usd": 200.0},
                    "tp_sl_multipliers": {"tp1": 1.0, "tp2": 1.8, "tp3": 3.0},
                },
            },
            "explanation": {
                "primary_reason": "Bullish breakout across confluence zone.",
                "supporting_factors": ["MACD momentum aligned", "Increasing spot demand"],
                "risk_factors": ["Nearby daily resistance"],
                "market_context": "Multi-timeframe trend confirmed",
            },
            "metadata": {
                "config_weights": {
                    "technical": 0.25,
                    "sentiment": 0.15,
                    "multitimeframe": 0.10,
                    "volume": 0.20,
                    "market_structure": 0.15,
                    "composite": 0.15,
                },
                "cancellation_triggers": ["Liquidity deterioration"],
                "timeframe_used": "1h",
            },
        }

        explicit = generate_signals(payload)

        assert explicit["signal"] == "BUY"
        assert explicit["entries"] == [pytest.approx(25_000.0)]
        assert explicit["stop_loss"] == pytest.approx(24_500.0)
        assert set(explicit["take_profits"].keys()) == {"tp1", "tp2", "tp3"}
        assert explicit["take_profits"]["tp1"] > explicit["entries"][0]
        assert explicit["take_profits"]["tp3"] > explicit["take_profits"]["tp2"]
        assert explicit["position_size_pct"] == pytest.approx(15.0)
        assert explicit["holding_period"] == "medium"
        assert explicit["holding_horizon_bars"] == 18
        assert math.isclose(sum(explicit["weights"].values()), 1.0, rel_tol=1e-6)
        assert explicit["metadata"]["category_confirmations"] >= 3
        assert "Bullish breakout" in " ".join(explicit["rationale"])

    def test_generate_signals_hold_due_to_confirmations(self):
        payload = {
            "signal_type": "BUY",
            "confidence": 0.55,
            "timestamp": 1700000000000,
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "factors": [
                {
                    "factor_name": "technical_analysis",
                    "score": 0.52,
                    "weight": 0.25,
                    "metadata": {"direction": "neutral"},
                },
                {
                    "factor_name": "sentiment",
                    "score": 0.40,
                    "weight": 0.15,
                    "metadata": {"direction": "bearish"},
                },
                {
                    "factor_name": "volume_analysis",
                    "score": 0.48,
                    "weight": 0.20,
                    "metadata": {"direction": "neutral"},
                },
            ],
            "position_plan": {
                "entry_price": 25_000.0,
                "stop_loss": 24_700.0,
                "take_profit_levels": [25_300.0, 25_600.0, 25_900.0],
                "position_size_usd": 1_000.0,
                "metadata": {
                    "atr": 120.0,
                    "holding_horizon_bars": 12,
                    "sizing_factors": {"risk_amount_usd": 180.0},
                },
            },
            "explanation": {
                "primary_reason": "Neutral market structure",
            },
            "metadata": {
                "config_weights": {
                    "technical": 0.25,
                    "sentiment": 0.15,
                    "multitimeframe": 0.10,
                    "volume": 0.20,
                    "market_structure": 0.15,
                    "composite": 0.15,
                },
                "timeframe_used": "1h",
            },
        }

        explicit = generate_signals(payload)

        assert explicit["signal"] == "HOLD"
        assert explicit["entries"] == []
        assert explicit["stop_loss"] is None
        assert explicit["take_profits"] == {}
        assert explicit["position_size_pct"] is None
        assert "insufficient" in " ".join(explicit["rationale"]).lower()
        assert explicit["metadata"]["category_confirmations"] == 0

    def test_generate_signals_hold_without_position_plan(self):
        payload = {
            "signal_type": "SELL",
            "confidence": 0.65,
            "timestamp": 1700005000000,
            "symbol": "ETHUSDT",
            "timeframe": "4h",
            "factors": [
                {
                    "factor_name": "technical_analysis",
                    "score": 0.30,
                    "weight": 0.25,
                    "metadata": {"direction": "bearish"},
                },
                {
                    "factor_name": "sentiment",
                    "score": 0.32,
                    "weight": 0.15,
                    "metadata": {"direction": "bearish"},
                },
                {
                    "factor_name": "market_structure",
                    "score": 0.35,
                    "weight": 0.15,
                    "metadata": {"direction": "bearish"},
                },
            ],
            "explanation": {
                "primary_reason": "Bearish momentum but no execution plan",
            },
            "metadata": {
                "config_weights": {
                    "technical": 0.30,
                    "sentiment": 0.20,
                    "multitimeframe": 0.10,
                    "volume": 0.15,
                    "market_structure": 0.15,
                    "composite": 0.10,
                },
                "timeframe_used": "4h",
            },
        }

        explicit = generate_signals(payload)

        assert explicit["signal"] == "HOLD"
        assert explicit["entries"] == []
        assert explicit["stop_loss"] is None
        assert explicit["take_profits"] == {}
        assert explicit["position_size_pct"] is None
        # Ensure rationale mentions missing plan
        rationale_text = " ".join(explicit.get("rationale", []))
        assert "plan" in rationale_text.lower()
