"""Tests for multi-timeframe analyzer module."""

import unittest
from indicator_collector.trading_system.multitimeframe_analyzer import (
    analyze_multitimeframe_factors,
    create_multitimeframe_factor_score,
    _calculate_trend_strength_from_candles,
    _analyze_timeframe_alignment,
    _analyze_trend_agreement,
    _analyze_trend_force,
    _generate_multitf_rationale,
    _calculate_multitf_confidence,
    _normalize_to_01,
    _get_direction_emoji,
)
from indicator_collector.trading_system.utils import clamp, safe_div
from indicator_collector.math_utils import Candle


class TestMultitimeframeAnalyzer(unittest.TestCase):
    """Test cases for multi-timeframe analyzer functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample candle data for main timeframe
        self.sample_candles = [
            Candle(
                open_time=1609459200000 + i * 60000,
                close_time=1609459260000 + i * 60000,
                open=50000.0 + i * 10,
                high=50010.0 + i * 10,
                low=49990.0 + i * 10,
                close=50005.0 + i * 10,
                volume=100.0 + i * 5,
            )
            for i in range(100)
        ]
        
        # Create sample candle data for 15m timeframe (bullish trend)
        self.bullish_candles_15m = [
            Candle(
                open_time=1609459200000 + i * 900000,
                close_time=1609459260000 + i * 900000,
                open=50000.0 + i * 50,
                high=50020.0 + i * 50,
                low=49990.0 + i * 50,
                close=50015.0 + i * 50,
                volume=500.0 + i * 20,
            )
            for i in range(30)
        ]
        
        # Create sample candle data for 1h timeframe (bearish trend)
        self.bearish_candles_1h = [
            Candle(
                open_time=1609459200000 + i * 3600000,
                close_time=1609459260000 + i * 3600000,
                open=50000.0 - i * 100,
                high=50010.0 - i * 100,
                low=49980.0 - i * 100,
                close=49985.0 - i * 100,
                volume=2000.0 + i * 50,
            )
            for i in range(25)
        ]
        
        # Create sample candle data for 4h timeframe (neutral trend)
        self.neutral_candles_4h = [
            Candle(
                open_time=1609459200000 + i * 14400000,
                close_time=1609459260000 + i * 14400000,
                open=50000.0 + (i % 3 - 1) * 20,
                high=50010.0 + (i % 3 - 1) * 20,
                low=49990.0 + (i % 3 - 1) * 20,
                close=50000.0 + (i % 3 - 1) * 15,
                volume=3000.0 + i * 100,
            )
            for i in range(20)
        ]

    def test_clamp(self):
        """Test clamping function."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(15.0, 0.0, 10.0) == 10.0
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0

    def test_safe_div(self):
        """Test safe division helper."""
        assert safe_div(10, 2) == 5.0
        assert safe_div(10, 0, default=1.0) == 1.0
        assert safe_div("20", "4") == 5.0
        assert safe_div("invalid", 4, default=-1.0) == -1.0

    def test_normalize_to_01(self):
        """Test normalization to 0-1 range."""
        assert _normalize_to_01(50.0, 0.0, 100.0) == 0.5
        assert _normalize_to_01(0.0, 0.0, 100.0) == 0.0
        assert _normalize_to_01(100.0, 0.0, 100.0) == 1.0
        assert _normalize_to_01(-50.0, 0.0, 100.0) == 0.0
        assert _normalize_to_01(150.0, 0.0, 100.0) == 1.0

    def test_get_direction_emoji(self):
        """Test direction emoji mapping."""
        assert _get_direction_emoji("bullish") == "🟢"
        assert _get_direction_emoji("bearish") == "🔴"
        assert _get_direction_emoji("neutral") == "⚪"
        assert _get_direction_emoji("unknown") == "⚪"

    def test_calculate_trend_strength_from_candles_bullish(self):
        """Test trend strength calculation for bullish candles."""
        strength = _calculate_trend_strength_from_candles(self.bullish_candles_15m)
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should show bullish bias

    def test_calculate_trend_strength_from_candles_bearish(self):
        """Test trend strength calculation for bearish candles."""
        strength = _calculate_trend_strength_from_candles(self.bearish_candles_1h)
        assert 0.0 <= strength <= 1.0
        assert strength < 0.4  # Should be strong bearish

    def test_calculate_trend_strength_from_candles_neutral(self):
        """Test trend strength calculation for neutral candles."""
        strength = _calculate_trend_strength_from_candles(self.neutral_candles_4h)
        assert 0.0 <= strength <= 1.0
        assert 0.3 <= strength <= 0.7  # Should be neutral

    def test_calculate_trend_strength_insufficient_data(self):
        """Test trend strength with insufficient data."""
        short_candles = self.sample_candles[:5]
        strength = _calculate_trend_strength_from_candles(short_candles, lookback=14)
        assert strength == 0.5  # Default when insufficient data

    def test_analyze_timeframe_alignment_all_bullish(self):
        """Test timeframe alignment when all bullish."""
        directions = {
            "5m": "bullish",
            "15m": "bullish",
            "1h": "bullish",
        }
        result = _analyze_timeframe_alignment(directions)
        
        assert 0.75 <= result["alignment_score"] <= 1.0  # Bullish biased
        assert result["alignment_type"] == "all_bullish"
        assert result["bullish_count"] == 3
        assert result["bearish_count"] == 0
        assert result["aligned_timeframes"] == 3
        assert result["conflict_timeframes"] == 0

    def test_analyze_timeframe_alignment_all_bearish(self):
        """Test timeframe alignment when all bearish."""
        directions = {
            "5m": "bearish",
            "15m": "bearish",
            "1h": "bearish",
        }
        result = _analyze_timeframe_alignment(directions)
        
        assert 0.0 <= result["alignment_score"] <= 0.25  # Bearish biased
        assert result["alignment_type"] == "all_bearish"
        assert result["bullish_count"] == 0
        assert result["bearish_count"] == 3
        assert result["aligned_timeframes"] == 3
        assert result["conflict_timeframes"] == 0

    def test_analyze_timeframe_alignment_conflict(self):
        """Test timeframe alignment with conflicting signals."""
        directions = {
            "5m": "bullish",
            "15m": "bullish",
            "1h": "bearish",
        }
        result = _analyze_timeframe_alignment(directions)
        
        assert result["alignment_type"] == "conflict"
        assert result["bullish_count"] == 2
        assert result["bearish_count"] == 1
        assert result["conflict_timeframes"] == 1
        assert 0.0 <= result["alignment_score"] <= 1.0

    def test_analyze_timeframe_alignment_mixed_neutral(self):
        """Test timeframe alignment with neutral signals."""
        directions = {
            "5m": "bullish",
            "15m": "neutral",
            "1h": "bearish",
        }
        result = _analyze_timeframe_alignment(directions)
        
        assert result["neutral_count"] == 1
        assert 0.0 <= result["alignment_score"] <= 1.0

    def test_analyze_trend_agreement_strong(self):
        """Test trend agreement with strong consensus."""
        strengths = {
            "5m": 0.75,
            "15m": 0.78,
            "1h": 0.72,
        }
        result = _analyze_trend_agreement(strengths)
        
        assert result["agreement_type"] == "strong"
        assert result["agreement_score"] > 0.7  # Biased bullish + high variance alignment
        assert result["std_dev"] < 0.1

    def test_analyze_trend_agreement_moderate(self):
        """Test trend agreement with moderate variance."""
        strengths = {
            "5m": 0.8,
            "15m": 0.6,
            "1h": 0.7,
        }
        result = _analyze_trend_agreement(strengths)
        
        assert result["agreement_type"] == "moderate"
        # Agreement score includes direction bias
        assert 0.5 <= result["agreement_score"] <= 1.0
        assert 0.1 <= result["std_dev"] < 0.2

    def test_analyze_trend_agreement_weak(self):
        """Test trend agreement with weak consensus."""
        strengths = {
            "5m": 0.9,
            "15m": 0.2,
            "1h": 0.5,
        }
        result = _analyze_trend_agreement(strengths)
        
        assert result["agreement_type"] == "weak"
        assert result["std_dev"] >= 0.2

    def test_analyze_trend_force_strong(self):
        """Test trend force with strong signals."""
        strengths = {
            "5m": 0.8,
            "15m": 0.75,
            "1h": 0.78,
        }
        directions = {
            "5m": "bullish",
            "15m": "bullish",
            "1h": "bullish",
        }
        result = _analyze_trend_force(strengths, directions)
        
        assert result["force_type"] == "strong"
        assert result["trend_force_score"] > 0.7
        assert result["strong_count"] == 3

    def test_analyze_trend_force_moderate(self):
        """Test trend force with moderate signals."""
        strengths = {
            "5m": 0.6,
            "15m": 0.55,
            "1h": 0.58,
        }
        directions = {
            "5m": "bullish",
            "15m": "bullish",
            "1h": "bullish",
        }
        result = _analyze_trend_force(strengths, directions)
        
        assert result["force_type"] == "moderate"
        assert 0.5 <= result["trend_force_score"] < 0.7

    def test_analyze_trend_force_weak(self):
        """Test trend force with weak signals."""
        strengths = {
            "5m": 0.35,
            "15m": 0.32,
            "1h": 0.38,
        }
        directions = {
            "5m": "bearish",
            "15m": "bearish",
            "1h": "bearish",
        }
        result = _analyze_trend_force(strengths, directions)
        
        assert result["force_type"] == "weak"
        assert result["trend_force_score"] < 0.5

    def test_analyze_multitimeframe_factors_no_data(self):
        """Test multi-timeframe analysis with no data."""
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            {},
        )
        
        assert result["final_score"] == 0.5
        assert result["direction"] == "neutral"
        assert result["confidence"] == 0
        assert "multi-timeframe" in result["rationale"].lower()
        metadata = result["metadata"]
        assert metadata["timeframe_count"] == 0
        assert metadata.get("missing_timeframes", []) == []

    def test_analyze_multitimeframe_factors_all_aligned_bullish(self):
        """Test multi-timeframe analysis with all timeframes aligned bullish."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.bullish_candles_15m,
            "1h": self.bullish_candles_15m,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        assert "final_score" in result
        assert "direction" in result
        assert "confidence" in result
        assert "rationale" in result
        assert "per_timeframe_flags" in result
        assert "components" in result
        
        # All same-direction signals (bullish biased)
        assert result["direction"] in ["bullish", "neutral"]
        assert result["final_score"] > 0.5  # Should be above neutral
        assert result["confidence"] >= 40

    def test_analyze_multitimeframe_factors_all_aligned_bearish(self):
        """Test multi-timeframe analysis with all timeframes aligned bearish."""
        multi_tf_candles = {
            "5m": self.bearish_candles_1h,
            "15m": self.bearish_candles_1h,
            "1h": self.bearish_candles_1h,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        # All bearish signals
        assert result["direction"] == "bearish"
        assert result["final_score"] < 0.4
        assert result["confidence"] >= 30  # May vary based on strength

    def test_analyze_multitimeframe_factors_conflicting(self):
        """Test multi-timeframe analysis with conflicting signals."""
        # Create strong bullish candles
        strong_bullish = [
            Candle(
                open_time=1609459200000 + i * 900000,
                close_time=1609459260000 + i * 900000,
                open=50000.0 + i * 100,  # Stronger trend
                high=50030.0 + i * 100,
                low=49990.0 + i * 100,
                close=50020.0 + i * 100,
                volume=500.0 + i * 20,
            )
            for i in range(30)
        ]
        
        # Create strong bearish candles
        strong_bearish = [
            Candle(
                open_time=1609459200000 + i * 3600000,
                close_time=1609459260000 + i * 3600000,
                open=50000.0 - i * 100,  # Stronger trend
                high=50010.0 - i * 100,
                low=49960.0 - i * 100,
                close=49980.0 - i * 100,
                volume=2000.0 + i * 50,
            )
            for i in range(25)
        ]
        
        multi_tf_candles = {
            "5m": strong_bullish,
            "15m": strong_bullish,
            "1h": strong_bearish,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        # Should reflect conflict or significant bearish bias
        alignment_type = result.get("components", {}).get("alignment", {}).get("alignment_type", "")
        assert alignment_type in ["conflict", "all_bearish", "all_bullish"]
        # Confidence may vary, just check it's a valid integer
        assert 0 <= result["confidence"] <= 100

    def test_analyze_multitimeframe_factors_per_timeframe_flags(self):
        """Test per-timeframe flags in multi-timeframe analysis."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.neutral_candles_4h,
            "1h": self.bearish_candles_1h,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        flags = result["per_timeframe_flags"]
        
        # Check all timeframes are present
        assert "5m" in flags
        assert "15m" in flags
        assert "1h" in flags
        assert flags["5m"].get("available") is True
        
        # Check structure of flags
        for tf_name, flag_data in flags.items():
            assert "strength" in flag_data
            assert "direction" in flag_data
            assert "emoji" in flag_data
            assert "candle_count" in flag_data
            assert 0.0 <= flag_data["strength"] <= 1.0

    def test_analyze_multitimeframe_factors_components_structure(self):
        """Test structure of returned components."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.neutral_candles_4h,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        components = result["components"]
        
        # Check alignment component
        assert "alignment" in components
        assert "alignment_score" in components["alignment"]
        assert "alignment_type" in components["alignment"]
        assert "bullish_count" in components["alignment"]
        assert "bearish_count" in components["alignment"]
        
        # Check agreement component
        assert "agreement" in components
        assert "agreement_score" in components["agreement"]
        assert "agreement_type" in components["agreement"]
        assert "consensus_strength" in components["agreement"]
        
        # Check trend_force component
        assert "trend_force" in components
        assert "trend_force_score" in components["trend_force"]
        assert "force_type" in components["trend_force"]

    def test_analyze_multitimeframe_factors_metadata(self):
        """Test metadata in multi-timeframe analysis."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.neutral_candles_4h,
            "1h": self.bearish_candles_1h,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        metadata = result["metadata"]
        
        assert metadata["timeframe_count"] == 3
        assert "timeframes" in metadata
        assert set(metadata["timeframes"]) == {"5m", "15m", "1h"}

    def test_analyze_multitimeframe_factors_score_range(self):
        """Test that final score is always in valid range."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.neutral_candles_4h,
            "1h": self.bearish_candles_1h,
        }
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        assert 0.0 <= result["final_score"] <= 1.0

    def test_create_multitimeframe_factor_score(self):
        """Test creation of FactorScore from multi-timeframe analysis."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.neutral_candles_4h,
        }
        analysis = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        factor_score = create_multitimeframe_factor_score(analysis)
        
        # Check FactorScore properties
        assert factor_score.factor_name == "multitimeframe_alignment"
        assert factor_score.weight == 0.10
        assert 0.0 <= factor_score.score <= 1.0
        assert factor_score.description is not None
        assert factor_score.emoji in ["🟢", "🔴", "⚪"]
        
        # Check metadata
        assert "direction" in factor_score.metadata
        assert "confidence" in factor_score.metadata
        assert "per_timeframe_flags" in factor_score.metadata
        assert "components" in factor_score.metadata

    def test_generate_multitf_rationale(self):
        """Test rationale generation."""
        alignment = {
            "alignment_type": "all_bullish",
            "bullish_count": 3,
            "bearish_count": 0,
        }
        agreement = {
            "agreement_type": "strong",
        }
        trend_force = {
            "force_type": "strong",
            "avg_force": 0.78,
        }
        
        rationale = _generate_multitf_rationale(alignment, agreement, trend_force, "bullish")
        
        assert isinstance(rationale, str)
        assert len(rationale) > 0
        assert "bullish" in rationale.lower()
        assert "strong" in rationale.lower()

    def test_calculate_multitf_confidence(self):
        """Test confidence calculation."""
        alignment = {"alignment_score": 1.0}
        agreement = {"agreement_score": 0.9}
        
        confidence = _calculate_multitf_confidence(alignment, agreement, num_timeframes=3)
        
        assert 0 <= confidence <= 100
        assert confidence > 80  # High alignment and agreement

    def test_calculate_multitf_confidence_low(self):
        """Test confidence calculation with low scores."""
        alignment = {"alignment_score": 0.3}
        agreement = {"agreement_score": 0.3}
        
        confidence = _calculate_multitf_confidence(alignment, agreement, num_timeframes=1)
        
        assert 0 <= confidence <= 100
        assert confidence < 40  # Low alignment and agreement

    def test_multitimeframe_provided_strengths(self):
        """Test that provided multi-timeframe strengths are used."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
            "15m": self.neutral_candles_4h,
        }
        multi_tf_strengths = {
            "5m": 0.85,
            "15m": 0.50,
        }
        
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
            multi_tf_strengths,
        )
        
        # Check that provided strengths are reflected in flags
        flags = result["per_timeframe_flags"]
        assert flags["5m"]["strength"] == 0.85
        assert flags["15m"]["strength"] == 0.50
        assert flags["5m"]["direction"] == "bullish"
        assert flags["15m"]["direction"] == "neutral"

    def test_multitimeframe_empty_timeframes(self):
        """Test handling of empty timeframe candles."""
        multi_tf_candles = {
            "5m": [],
            "15m": self.neutral_candles_4h,
        }
        
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        # Should handle gracefully and only analyze valid timeframes
        assert result["metadata"]["timeframe_count"] == 1
        assert "15m" in result["per_timeframe_flags"]
        assert "5m" not in result["per_timeframe_flags"]

    def test_multitimeframe_single_timeframe(self):
        """Test multi-timeframe analysis with single timeframe."""
        multi_tf_candles = {
            "5m": self.bullish_candles_15m,
        }
        
        result = analyze_multitimeframe_factors(
            self.sample_candles,
            multi_tf_candles,
        )
        
        assert result["metadata"]["timeframe_count"] == 1
        assert result["direction"] in ["bullish", "neutral"]  # Single uptrend signal
        assert "5m" in result["per_timeframe_flags"]


if __name__ == "__main__":
    unittest.main()
