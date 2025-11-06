"""Tests for sentiment analyzer module."""

import unittest
from unittest.mock import patch, MagicMock

from indicator_collector.trading_system.sentiment_analyzer import (
    analyze_sentiment_factors,
    create_sentiment_factor_score,
    _normalize_fear_greed_score,
    _analyze_fundamental_sentiment,
    _calculate_sentiment_direction,
    _get_sentiment_emoji,
    _calculate_confidence,
    _generate_rationale,
)
from indicator_collector.math_utils import Candle


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for sentiment analyzer functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample candle data
        self.sample_candles = [
            Candle(
                open_time=1609459200000 + i * 60000,  # 1-minute intervals
                close_time=1609459260000 + i * 60000,
                open=50000.0 + i * 10,
                high=50010.0 + i * 10,
                low=49990.0 + i * 10,
                close=50005.0 + i * 10,
                volume=100.0 + i * 5,
            )
            for i in range(100)
        ]

    def test_normalize_fear_greed_score(self):
        """Test fear & greed score normalization."""
        # Test extreme values
        assert _normalize_fear_greed_score(0) == 0.0  # Extreme Fear
        assert _normalize_fear_greed_score(100) == 1.0  # Extreme Greed
        
        # Test middle value
        assert _normalize_fear_greed_score(50) == 0.5  # Neutral
        
        # Test typical values
        assert _normalize_fear_greed_score(25) == 0.25  # Fear
        assert _normalize_fear_greed_score(75) == 0.75  # Greed

    @patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index')
    @patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics')
    def test_analyze_sentiment_factors_bullish(self, mock_fundamentals, mock_fear_greed):
        """Test sentiment analysis with bullish signals."""
        # Mock fear & greed data - Extreme Greed (bullish)
        mock_fear_greed.return_value = {
            "fear_greed_index": 85,
            "regime": "Extreme Greed",
            "timestamp": 1640995200,
            "source": "alternative.me",
        }
        
        # Mock fundamental data - bullish derivatives
        mock_fundamentals.return_value = {
            "funding_rate": {
                "current": 0.002,  # Positive funding (bullish)
                "predicted": 0.0021,
                "annualized": 219.0,
            },
            "open_interest": {
                "current": 1000000000.0,
                "change_pct": 15.0,  # Growing OI (bullish)
            },
            "long_short_ratio": {
                "long": 0.68,  # Bullish positioning
                "short": 0.32,
                "ratio": 2.125,
            },
            "block_trades": [],
        }
        
        result = analyze_sentiment_factors(self.sample_candles)
        
        # Verify structure
        assert "final_score" in result
        assert "direction" in result
        assert "confidence" in result
        assert "rationale" in result
        assert "components" in result
        assert "factor_weights" in result
        assert "factor_scores" in result
        assert "metadata" in result
        
        # Verify bullish result
        assert result["final_score"] > 0.65
        assert result["direction"] == "bullish"
        assert result["emoji"] == "🟢"
        
        # Verify components
        assert "macro_sentiment" in result["components"]
        assert "derivatives_sentiment" in result["components"]
        
        # Verify weights
        assert result["factor_weights"]["macro_sentiment"] == 0.6
        assert result["factor_weights"]["derivatives_sentiment"] == 0.4
        
        # Verify scores are in valid range
        assert 0.0 <= result["final_score"] <= 1.0
        assert 0.0 <= result["factor_scores"]["macro_sentiment"] <= 1.0
        assert 0.0 <= result["factor_scores"]["derivatives_sentiment"] <= 1.0

    @patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index')
    @patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics')
    def test_analyze_sentiment_factors_bearish(self, mock_fundamentals, mock_fear_greed):
        """Test sentiment analysis with bearish signals."""
        # Mock fear & greed data - Extreme Fear (bearish)
        mock_fear_greed.return_value = {
            "fear_greed_index": 15,
            "regime": "Extreme Fear",
            "timestamp": 1640995200,
            "source": "alternative.me",
        }
        
        # Mock fundamental data - bearish derivatives
        mock_fundamentals.return_value = {
            "funding_rate": {
                "current": -0.002,  # Negative funding (bearish)
                "predicted": -0.0019,
                "annualized": -219.0,
            },
            "open_interest": {
                "current": 1000000000.0,
                "change_pct": -20.0,  # Declining OI (bearish)
            },
            "long_short_ratio": {
                "long": 0.25,  # Bearish positioning
                "short": 0.75,
                "ratio": 0.33,
            },
            "block_trades": [],
        }
        
        result = analyze_sentiment_factors(self.sample_candles)
        
        # Verify bearish result
        assert result["final_score"] < 0.35
        assert result["direction"] == "bearish"
        assert result["emoji"] == "🔴"

    @patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index')
    @patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics')
    def test_analyze_sentiment_factors_neutral(self, mock_fundamentals, mock_fear_greed):
        """Test sentiment analysis with neutral signals."""
        # Mock fear & greed data - Neutral
        mock_fear_greed.return_value = {
            "fear_greed_index": 50,
            "regime": "Neutral",
            "timestamp": 1640995200,
            "source": "alternative.me",
        }
        
        # Mock fundamental data - neutral derivatives
        mock_fundamentals.return_value = {
            "funding_rate": {
                "current": 0.0001,  # Near-zero funding
                "predicted": 0.0001,
                "annualized": 10.95,
            },
            "open_interest": {
                "current": 1000000000.0,
                "change_pct": 2.0,  # Stable OI
            },
            "long_short_ratio": {
                "long": 0.52,  # Balanced positioning
                "short": 0.48,
                "ratio": 1.08,
            },
            "block_trades": [],
        }
        
        result = analyze_sentiment_factors(self.sample_candles)
        
        # Verify neutral result
        assert 0.35 <= result["final_score"] <= 0.65
        assert result["direction"] == "neutral"
        assert result["emoji"] == "⚪"

    @patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index')
    @patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics')
    def test_analyze_sentiment_factors_api_failure(self, mock_fundamentals, mock_fear_greed):
        """Test sentiment analysis with API failure fallbacks."""
        # Mock API failure for fear & greed
        mock_fear_greed.return_value = {
            "fear_greed_index": 50,
            "regime": "Neutral",
            "timestamp": None,
            "source": "unavailable",
            "note": "Failed to fetch from external API",
        }
        
        # Mock fundamental data still works
        mock_fundamentals.return_value = {
            "funding_rate": {"current": 0.001, "predicted": 0.001, "annualized": 109.5},
            "open_interest": {"current": 1000000000.0, "change_pct": 5.0},
            "long_short_ratio": {"long": 0.6, "short": 0.4, "ratio": 1.5},
            "block_trades": [],
        }
        
        result = analyze_sentiment_factors(self.sample_candles)
        
        # Should still work with fallback data
        assert result["final_score"] >= 0.0
        assert result["confidence"] < 100  # Lower confidence due to API failure
        assert result["metadata"]["data_sources"]["fear_greed"] == "unavailable"

    def test_analyze_sentiment_factors_empty_data(self):
        """Test sentiment analysis with insufficient data."""
        result = analyze_sentiment_factors([])
        
        # Should handle empty data gracefully
        assert result["final_score"] == 0.5
        assert result["direction"] == "neutral"
        assert result["confidence"] == 0
        assert "Insufficient data" in result["rationale"]
        assert "error" in result["metadata"]

    def test_analyze_fundamental_sentiment(self):
        """Test fundamental sentiment analysis."""
        # Test bullish fundamentals
        bullish_fundamentals = {
            "funding_rate": {"current": 0.003},  # Positive funding
            "open_interest": {"change_pct": 20.0},  # Growing OI
            "long_short_ratio": {"long": 0.7},  # Bullish positioning
        }
        
        score = _analyze_fundamental_sentiment(bullish_fundamentals)
        assert score > 0.6  # Should be bullish
        
        # Test bearish fundamentals
        bearish_fundamentals = {
            "funding_rate": {"current": -0.003},  # Negative funding
            "open_interest": {"change_pct": -15.0},  # Declining OI
            "long_short_ratio": {"long": 0.3},  # Bearish positioning
        }
        
        score = _analyze_fundamental_sentiment(bearish_fundamentals)
        assert score < 0.4  # Should be bearish
        
        # Test empty fundamentals
        score = _analyze_fundamental_sentiment({})
        assert 0.0 <= score <= 1.0  # Should handle gracefully

    def test_calculate_sentiment_direction(self):
        """Test sentiment direction calculation."""
        assert _calculate_sentiment_direction(0.8) == "bullish"
        assert _calculate_sentiment_direction(0.7) == "bullish"
        assert _calculate_sentiment_direction(0.65) == "bullish"
        
        assert _calculate_sentiment_direction(0.5) == "neutral"
        assert _calculate_sentiment_direction(0.4) == "neutral"
        
        assert _calculate_sentiment_direction(0.2) == "bearish"
        assert _calculate_sentiment_direction(0.3) == "bearish"
        assert _calculate_sentiment_direction(0.35) == "bearish"

    def test_get_sentiment_emoji(self):
        """Test sentiment emoji mapping."""
        assert _get_sentiment_emoji("bullish") == "🟢"
        assert _get_sentiment_emoji("bearish") == "🔴"
        assert _get_sentiment_emoji("neutral") == "⚪"
        assert _get_sentiment_emoji("unknown") == "⚪"  # Default fallback

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Test high confidence
        confidence = _calculate_confidence(1.0, 0.9, 0.6, 0.4)
        assert confidence == 96  # (1.0*0.6 + 0.9*0.4) / 1.0 * 100 = 96
        
        # Test low confidence
        confidence = _calculate_confidence(0.3, 0.2, 0.6, 0.4)
        assert confidence == 26  # (0.3*0.6 + 0.2*0.4) / 1.0 * 100 = 26
        
        # Test zero confidence
        confidence = _calculate_confidence(0.0, 0.0, 0.6, 0.4)
        assert confidence == 0

    def test_generate_rationale(self):
        """Test rationale generation."""
        fear_greed_data = {
            "fear_greed_index": 75,
            "regime": "Greed",
        }
        
        fundamentals = {
            "funding_rate": {"current": 0.002},
            "open_interest": {"change_pct": 10.0},
            "long_short_ratio": {"long": 0.65},
        }
        
        rationale = _generate_rationale(
            fear_greed_data, fundamentals, 0.75, 0.7, "bullish"
        )
        
        assert "Macro sentiment" in rationale
        assert "Greed" in rationale
        assert "Derivatives" in rationale
        assert "positive funding" in rationale
        assert "growing OI" in rationale
        assert "bullish long/short" in rationale
        assert "bullish sentiment" in rationale

    @patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index')
    @patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics')
    def test_create_sentiment_factor_score(self, mock_fundamentals, mock_fear_greed):
        """Test FactorScore creation."""
        # Mock data
        mock_fear_greed.return_value = {
            "fear_greed_index": 70,
            "regime": "Greed",
            "source": "alternative.me",
        }
        
        mock_fundamentals.return_value = {
            "funding_rate": {"current": 0.0015, "predicted": 0.0016, "annualized": 164.25},
            "open_interest": {"current": 1000000000.0, "change_pct": 8.0},
            "long_short_ratio": {"long": 0.6, "short": 0.4, "ratio": 1.5},
            "block_trades": [],
        }
        
        factor_score = create_sentiment_factor_score(self.sample_candles)
        
        # Verify FactorScore structure
        assert factor_score.factor_name == "sentiment"
        assert factor_score.weight == 0.15  # 15% weight as specified
        assert 0.0 <= factor_score.score <= 1.0
        assert factor_score.emoji in ["🟢", "🔴", "⚪"]
        assert factor_score.description is not None
        assert "direction" in factor_score.metadata
        assert "confidence" in factor_score.metadata
        assert "components" in factor_score.metadata

    @patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index')
    @patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics')
    def test_sentiment_score_boundaries(self, mock_fundamentals, mock_fear_greed):
        """Test that sentiment scores stay within valid boundaries."""
        # Test maximum score
        mock_fear_greed.return_value = {"fear_greed_index": 100, "regime": "Extreme Greed", "source": "alternative.me"}
        mock_fundamentals.return_value = {
            "funding_rate": {"current": 0.004, "predicted": 0.004, "annualized": 438.0},
            "open_interest": {"change_pct": 50.0},
            "long_short_ratio": {"long": 0.9, "short": 0.1, "ratio": 9.0},
            "block_trades": [],
        }
        
        result = analyze_sentiment_factors(self.sample_candles)
        assert result["final_score"] <= 1.0
        
        # Test minimum score
        mock_fear_greed.return_value = {"fear_greed_index": 0, "regime": "Extreme Fear", "source": "alternative.me"}
        mock_fundamentals.return_value = {
            "funding_rate": {"current": -0.004, "predicted": -0.004, "annualized": -438.0},
            "open_interest": {"change_pct": -50.0},
            "long_short_ratio": {"long": 0.1, "short": 0.9, "ratio": 0.11},
            "block_trades": [],
        }
        
        result = analyze_sentiment_factors(self.sample_candles)
        assert result["final_score"] >= 0.0

    def test_sentiment_component_breakdown(self):
        """Test detailed component breakdown."""
        with patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index') as mock_fear_greed, \
             patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics') as mock_fundamentals:
            
            mock_fear_greed.return_value = {
                "fear_greed_index": 60,
                "regime": "Greed",
                "timestamp": 1640995200,
                "source": "alternative.me",
            }
            
            mock_fundamentals.return_value = {
                "funding_rate": {"current": 0.001, "predicted": 0.0011, "annualized": 109.5},
                "open_interest": {"current": 1000000000.0, "change_pct": 5.0},
                "long_short_ratio": {"long": 0.55, "short": 0.45, "ratio": 1.22},
                "block_trades": [],
            }
            
            result = analyze_sentiment_factors(self.sample_candles)
            
            # Verify macro sentiment component
            macro_component = result["components"]["macro_sentiment"]
            assert "score" in macro_component
            assert "weight" in macro_component
            assert "data" in macro_component
            assert "description" in macro_component
            assert macro_component["data"]["fear_greed_index"] == 60
            assert macro_component["data"]["regime"] == "Greed"
            
            # Verify derivatives sentiment component
            derivatives_component = result["components"]["derivatives_sentiment"]
            assert "score" in derivatives_component
            assert "weight" in derivatives_component
            assert "data" in derivatives_component
            assert "description" in derivatives_component
            assert "funding_rate" in derivatives_component["data"]
            assert "open_interest_change_pct" in derivatives_component["data"]
            assert "long_ratio" in derivatives_component["data"]

    def test_sentiment_metadata_completeness(self):
        """Test metadata completeness and accuracy."""
        with patch('indicator_collector.trading_system.sentiment_analyzer.fetch_fear_greed_index') as mock_fear_greed, \
             patch('indicator_collector.trading_system.sentiment_analyzer.calculate_fundamental_metrics') as mock_fundamentals:
            
            mock_fear_greed.return_value = {"fear_greed_index": 50, "regime": "Neutral", "source": "alternative.me"}
            mock_fundamentals.return_value = {"funding_rate": {"current": 0.0}, "open_interest": {"change_pct": 0.0}, "long_short_ratio": {"long": 0.5, "short": 0.5, "ratio": 1.0}, "block_trades": []}
            
            result = analyze_sentiment_factors(self.sample_candles)
            metadata = result["metadata"]
            
            # Verify required metadata fields
            assert "analysis_timestamp" in metadata
            assert "candle_count" in metadata
            assert "data_sources" in metadata
            assert "confidence_breakdown" in metadata
            assert "raw_fear_greed_data" in metadata
            assert "raw_fundamentals" in metadata
            
            # Verify values
            assert metadata["candle_count"] == 100
            assert metadata["data_sources"]["fear_greed"] == "alternative.me"
            assert metadata["data_sources"]["fundamentals"] == "calculated"
            assert "fear_greed_confidence" in metadata["confidence_breakdown"]
            assert "fundamentals_confidence" in metadata["confidence_breakdown"]