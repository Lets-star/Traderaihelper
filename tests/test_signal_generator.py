"""Tests for the trading signal generator."""

import unittest
from typing import Dict
from unittest.mock import Mock, patch

from indicator_collector.trading_system.signal_generator import (
    SignalConfig,
    SignalFactors,
    generate_trading_signal,
    _create_volume_factor,
    _create_structure_factor,
    _create_composite_factor,
    _check_cancellation_triggers,
    _calculate_confidence,
    _generate_explanation,
)
from indicator_collector.trading_system.interfaces import (
    AnalyzerContext,
    FactorScore,
    TradingSignalPayload,
)
from indicator_collector.trading_system.backtester import ParameterSet


class TestSignalConfig(unittest.TestCase):
    """Test the SignalConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SignalConfig()
        self.assertEqual(config.technical_weight, 0.25)
        self.assertEqual(config.sentiment_weight, 0.15)
        self.assertEqual(config.multitimeframe_weight, 0.10)
        self.assertEqual(config.volume_weight, 0.20)
        self.assertEqual(config.structure_weight, 0.15)
        self.assertEqual(config.composite_weight, 0.15)
        self.assertEqual(config.min_factors_confirm, 3)
        self.assertEqual(config.buy_threshold, 0.65)
        self.assertEqual(config.sell_threshold, 0.35)
        self.assertEqual(config.min_confidence, 0.6)
    
    def test_weight_validation(self):
        """Test that weights must sum to 1.0."""
        with self.assertRaises(ValueError):
            SignalConfig(technical_weight=0.5, sentiment_weight=0.3)  # Missing other weights
    
    def test_vix_adaptivity_normal(self):
        """Test VIX adaptivity with normal VIX."""
        config = SignalConfig()
        buy, sell, conf = config.get_adapted_thresholds(20.0)
        self.assertEqual(buy, config.buy_threshold)
        self.assertEqual(sell, config.sell_threshold)
        self.assertEqual(conf, config.min_confidence)
    
    def test_vix_adaptivity_high(self):
        """Test VIX adaptivity with high VIX (> 30)."""
        config = SignalConfig()
        buy, sell, conf = config.get_adapted_thresholds(35.0)
        # Should be clamped to maximum values
        self.assertEqual(buy, 0.9)  # Upper clamp
        self.assertGreaterEqual(sell, config.sell_threshold)  # Should be higher
        self.assertGreaterEqual(conf, config.min_confidence)  # Should be higher
    
    def test_vix_adaptivity_low(self):
        """Test VIX adaptivity with low VIX (< 15)."""
        config = SignalConfig()
        buy, sell, conf = config.get_adapted_thresholds(10.0)
        self.assertEqual(buy, config.buy_threshold * config.vix_loosen_factor)
        self.assertEqual(sell, config.sell_threshold * config.vix_loosen_factor)
        self.assertEqual(conf, config.min_confidence * config.vix_loosen_factor)
    
    def test_threshold_clamping(self):
        """Test that thresholds are properly clamped."""
        config = SignalConfig()
        # Test extreme tightening
        buy, sell, conf = config.get_adapted_thresholds(50.0)  # Very high VIX
        self.assertLessEqual(buy, 0.9)  # Upper clamp
        self.assertGreaterEqual(sell, 0.1)  # Lower clamp
        self.assertLessEqual(conf, 0.95)  # Upper clamp


class TestSignalFactors(unittest.TestCase):
    """Test the SignalFactors class."""
    
    def test_empty_factors(self):
        """Test empty factors container."""
        factors = SignalFactors()
        self.assertEqual(factors.count_available_factors(), 0)
        self.assertEqual(factors.get_available_factors(), [])
        self.assertEqual(factors.get_bullish_factors(), [])
        self.assertEqual(factors.get_bearish_factors(), [])
    
    def test_mixed_factors(self):
        """Test factors with mixed directions."""
        factors = SignalFactors(
            technical=FactorScore("tech", 0.7, metadata={"direction": "bullish"}),
            sentiment=FactorScore("sent", 0.3, metadata={"direction": "bearish"}),
            volume=FactorScore("vol", 0.5, metadata={"direction": "neutral"}),
        )
        
        self.assertEqual(factors.count_available_factors(), 3)
        self.assertEqual(len(factors.get_available_factors()), 3)
        self.assertEqual(len(factors.get_bullish_factors()), 1)
        self.assertEqual(len(factors.get_bearish_factors()), 1)
        
        # Test None factor filtering
        self.assertEqual(factors.get_available_factors()[0].factor_name, "tech")


class TestFactorCreation(unittest.TestCase):
    """Test factor creation functions."""
    
    def test_create_volume_factor_basic(self):
        """Test basic volume factor creation."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            volume_analysis={"volume_ratio": 1.5, "volume_confidence": 0.7},
        )
        
        factor = _create_volume_factor(context)
        self.assertIsNotNone(factor)
        self.assertEqual(factor.factor_name, "volume_analysis")
        self.assertGreaterEqual(factor.score, 0.0)
        self.assertLessEqual(factor.score, 1.0)
        self.assertEqual(factor.weight, 0.20)
        self.assertIn(factor.metadata["direction"], ["bullish", "bearish", "neutral"])
    
    def test_create_volume_factor_with_smart_money(self):
        """Test volume factor with smart money data."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            volume_analysis={"volume_ratio": 2.5, "volume_confidence": 0.8},
            advanced_metrics={"smart_money_activity": {"score": 0.8}},
        )
        
        factor = _create_volume_factor(context)
        self.assertIsNotNone(factor)
        self.assertGreater(factor.score, 0.6)  # Should be bullish with high volume and smart money
        self.assertEqual(factor.metadata["direction"], "bullish")
    
    def test_create_structure_factor_basic(self):
        """Test basic structure factor creation."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            market_structure={"structure_state": "bullish", "structure_score": 0.7},
        )
        
        factor = _create_structure_factor(context)
        self.assertIsNotNone(factor)
        self.assertEqual(factor.factor_name, "market_structure")
        self.assertGreaterEqual(factor.score, 0.0)
        self.assertLessEqual(factor.score, 1.0)
        self.assertEqual(factor.weight, 0.15)
        self.assertEqual(factor.metadata["direction"], "bullish")
    
    def test_create_composite_factor_basic(self):
        """Test basic composite factor creation."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            advanced_metrics={
                "composite_indicators": {"overall_score": 0.8},
                "market_context": {"score": 0.7},
            },
        )
        
        factor = _create_composite_factor(context)
        self.assertIsNotNone(factor)
        self.assertEqual(factor.factor_name, "composite_analysis")
        self.assertGreaterEqual(factor.score, 0.0)
        self.assertLessEqual(factor.score, 1.0)
        self.assertEqual(factor.weight, 0.15)
        self.assertEqual(factor.metadata["direction"], "bullish")


class TestParameterReactivity(unittest.TestCase):
    """Ensure analyzer parameters influence factor outputs."""

    def _base_weights(self) -> Dict[str, float]:
        return {
            "technical": 0.25,
            "volume": 0.25,
            "sentiment": 0.20,
            "market_structure": 0.20,
            "multitimeframe": 0.10,
            "composite": 0.0,
        }

    def test_volume_factor_respects_cvd_multiplier(self):
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1710000000,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={"atr": 15.0},
            volume_analysis={
                "context": {"volume_ratio": 1.2, "volume_confidence": 0.65, "atr": 15.0},
                "cvd": {"change": 45.0},
                "delta": {"latest": 180.0, "average": 90.0},
                "vpvr": {
                    "total_volume": 1000.0,
                    "poc": 50050.0,
                    "levels": [
                        {"price": 50000.0, "volume": 120.0},
                        {"price": 50050.0, "volume": 180.0},
                    ],
                },
                "smart_money": [],
            },
        )

        low_multiplier = ParameterSet(
            weights=self._base_weights(),
            indicator_params={"volume": {"cvd_atr_multiplier": 0.5}},
            timeframe="1h",
        )
        high_multiplier = ParameterSet(
            weights=self._base_weights(),
            indicator_params={"volume": {"cvd_atr_multiplier": 2.0}},
            timeframe="1h",
        )

        bullish_factor = _create_volume_factor(context, low_multiplier)
        muted_factor = _create_volume_factor(context, high_multiplier)

        self.assertGreater(bullish_factor.score, muted_factor.score)

    def test_structure_factor_respects_swing_window(self):
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1710001000,
            current_price=48000.0,
            ohlcv={"open": 47000, "high": 49000, "low": 46500, "close": 48000, "volume": 1500},
            indicators={},
            market_structure={
                "structure_state": "bullish",
                "structure_score": 0.58,
                "liquidity_score": 0.55,
                "sequence_length": 0,
                "liquidity_sweep_atr": 1.2,
            },
            advanced_metrics={"market_breadth": {"score": 0.62}},
        )

        tight_structure = ParameterSet(
            weights=self._base_weights(),
            indicator_params={"structure": {"swing_window": 3, "min_sequence": 3}},
            timeframe="1h",
        )
        wide_structure = ParameterSet(
            weights=self._base_weights(),
            indicator_params={"structure": {"swing_window": 12, "min_sequence": 3}},
            timeframe="1h",
        )

        tight_factor = _create_structure_factor(context, tight_structure)
        wide_factor = _create_structure_factor(context, wide_structure)

        self.assertGreater(wide_factor.score, tight_factor.score)

    def test_composite_factor_respects_category_weights(self):
        factors = SignalFactors(
            technical=FactorScore(
                factor_name="technical_analysis",
                score=0.55,
                weight=0.25,
                metadata={"direction": "bullish"},
            ),
            volume=FactorScore(
                factor_name="volume_analysis",
                score=0.78,
                weight=0.25,
                metadata={"direction": "bullish"},
            ),
            structure=FactorScore(
                factor_name="market_structure",
                score=0.42,
                weight=0.25,
                metadata={"direction": "bearish"},
            ),
        )

        volume_heavy = ParameterSet(
            weights={
                "technical": 0.15,
                "volume": 0.55,
                "sentiment": 0.10,
                "market_structure": 0.15,
                "multitimeframe": 0.05,
                "composite": 0.0,
            },
            indicator_params={},
            timeframe="1h",
        )
        technical_heavy = ParameterSet(
            weights={
                "technical": 0.55,
                "volume": 0.20,
                "sentiment": 0.10,
                "market_structure": 0.10,
                "multitimeframe": 0.05,
                "composite": 0.0,
            },
            indicator_params={},
            timeframe="1h",
        )

        composite_params_volume = volume_heavy.get_indicator_group("composite")
        composite_params_tech = technical_heavy.get_indicator_group("composite")

        volume_composite = _create_composite_factor(factors, volume_heavy, composite_params_volume)
        tech_composite = _create_composite_factor(factors, technical_heavy, composite_params_tech)

        self.assertGreater(volume_composite.score, tech_composite.score)


class TestCancellationTriggers(unittest.TestCase):
    """Test cancellation trigger detection."""
    
    def test_no_triggers(self):
        """Test no cancellation triggers."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
        )
        factors = SignalFactors()
        
        triggers = _check_cancellation_triggers(context, factors)
        self.assertEqual(triggers, [])
    
    def test_high_risk_trigger(self):
        """Test high risk cancellation trigger."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            advanced_metrics={"risk_metrics": {"risk_score": 0.9}},
        )
        factors = SignalFactors()
        
        triggers = _check_cancellation_triggers(context, factors)
        self.assertIn("High risk score detected", triggers)
    
    def test_low_liquidity_trigger(self):
        """Test low liquidity cancellation trigger."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            volume_analysis={"liquidity_score": 0.1},
        )
        factors = SignalFactors()
        
        triggers = _check_cancellation_triggers(context, factors)
        self.assertIn("Low liquidity detected", triggers)
    
    def test_extreme_volatility_trigger(self):
        """Test extreme volatility cancellation trigger."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={"atr": 3000},  # 6% of price
        )
        factors = SignalFactors()
        
        triggers = _check_cancellation_triggers(context, factors)
        self.assertIn("Extreme volatility:", triggers[0])
    
    def test_conflicting_signals_trigger(self):
        """Test conflicting signals cancellation trigger."""
        factors = SignalFactors(
            technical=FactorScore("tech", 0.8, metadata={"direction": "bullish"}),
            sentiment=FactorScore("sent", 0.2, metadata={"direction": "bearish"}),
            volume=FactorScore("vol", 0.7, metadata={"direction": "bullish"}),
            structure=FactorScore("struct", 0.3, metadata={"direction": "bearish"}),
        )
        
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
        )
        
        triggers = _check_cancellation_triggers(context, factors)
        self.assertIn("Strong conflicting signals detected", triggers)


class TestConfidenceCalculation(unittest.TestCase):
    """Test confidence calculation logic."""
    
    def test_bullish_confidence(self):
        """Test confidence calculation for bullish signals."""
        factors = SignalFactors(
            technical=FactorScore("tech", 0.8, metadata={"direction": "bullish"}),
            sentiment=FactorScore("sent", 0.7, metadata={"direction": "bullish"}),
            volume=FactorScore("vol", 0.6, metadata={"direction": "bullish"}),
        )
        
        confidence_int, confidence_float = _calculate_confidence(
            final_score=0.75,
            factors=factors,
            buy_threshold=0.65,
            sell_threshold=0.35,
            cancellation_triggers=[]
        )
        
        self.assertGreaterEqual(confidence_int, 1)
        self.assertLessEqual(confidence_int, 10)
        self.assertGreaterEqual(confidence_float, 0.1)
        self.assertLessEqual(confidence_float, 1.0)
        # Check that confidence_int is roughly 10x confidence_float, allowing for adjustments
        expected_float = confidence_int / 10.0
        self.assertAlmostEqual(confidence_float, expected_float, delta=0.1)
    
    def test_bearish_confidence(self):
        """Test confidence calculation for bearish signals."""
        factors = SignalFactors(
            technical=FactorScore("tech", 0.2, metadata={"direction": "bearish"}),
            sentiment=FactorScore("sent", 0.3, metadata={"direction": "bearish"}),
        )
        
        confidence_int, confidence_float = _calculate_confidence(
            final_score=0.25,
            factors=factors,
            buy_threshold=0.65,
            sell_threshold=0.35,
            cancellation_triggers=[]
        )
        
        self.assertGreaterEqual(confidence_int, 1)
        self.assertLessEqual(confidence_int, 10)
        self.assertGreaterEqual(confidence_float, 0.1)
        self.assertLessEqual(confidence_float, 1.0)
        # Check that confidence_int is roughly 10x confidence_float, allowing for adjustments
        expected_float = confidence_int / 10.0
        self.assertAlmostEqual(confidence_float, expected_float, delta=0.1)
    
    def test_cancellation_trigger_penalty(self):
        """Test confidence penalty with cancellation triggers."""
        factors = SignalFactors(
            technical=FactorScore("tech", 0.8, metadata={"direction": "bullish"}),
        )
        
        confidence_int, confidence_float = _calculate_confidence(
            final_score=0.75,
            factors=factors,
            buy_threshold=0.65,
            sell_threshold=0.35,
            cancellation_triggers=["High risk", "Low liquidity"]
        )
        
        # Should be lower due to triggers
        self.assertLess(confidence_int, 8)  # Significantly reduced


class TestExplanationGeneration(unittest.TestCase):
    """Test explanation generation."""
    
    def test_buy_explanation(self):
        """Test explanation generation for BUY signal."""
        factors = SignalFactors(
            technical=FactorScore("tech", 0.8, metadata={"direction": "bullish"}),
            sentiment=FactorScore("sent", 0.7, metadata={"direction": "bullish"}),
        )
        
        explanation = _generate_explanation(
            signal_type="BUY",
            final_score=0.75,
            factors=factors,
            confidence=8,
            cancellation_triggers=[]
        )
        
        self.assertIn("Bullish signal", explanation.primary_reason)
        self.assertGreater(len(explanation.supporting_factors), 0)
        self.assertIn("tech", explanation.supporting_factors[0])
    
    def test_cancellation_explanation(self):
        """Test explanation generation with cancellation triggers."""
        factors = SignalFactors()
        
        explanation = _generate_explanation(
            signal_type="HOLD",
            final_score=0.5,
            factors=factors,
            confidence=3,
            cancellation_triggers=["High risk detected"]
        )
        
        self.assertIn("HOLD due to cancellation triggers", explanation.primary_reason)
        self.assertIn("High risk detected", explanation.risk_factors)


class TestSignalGeneration(unittest.TestCase):
    """Test the main signal generation function."""
    
    @patch('indicator_collector.trading_system.signal_generator.analyze_technical_factors')
    @patch('indicator_collector.trading_system.signal_generator.analyze_sentiment_factors')
    @patch('indicator_collector.trading_system.signal_generator.analyze_multitimeframe_factors')
    def test_buy_signal_generation(self, mock_mt, mock_sentiment, mock_technical):
        """Test BUY signal generation."""
        # Mock analyzer responses
        mock_technical.return_value = {
            "final_score": 0.78,
            "direction": "bullish",
            "confidence": 82.0,
            "rationale": "MACD and RSI indicate strong bullish momentum",
            "factor_scores": {
                "macd": 0.8,
                "rsi": 0.72,
                "atr": 0.6,
                "bollinger": 0.68,
                "divergence": 0.7,
            },
            "factor_weights": {
                "macd": 0.25,
                "rsi": 0.25,
                "atr": 0.15,
                "bollinger": 0.20,
                "divergence": 0.15,
            },
            "metadata": {
                "total_candles": 120,
            },
        }

        mock_sentiment.return_value = TradingSignalPayload(
            signal_type="BUY",
            confidence=0.7,
            timestamp=1234567890,
            symbol="BTC/USDT",
            timeframe="1h",
            factors=[
                FactorScore("sentiment", 0.75, weight=0.15, metadata={"direction": "bullish"}),
            ]
        )

        mock_mt.return_value = TradingSignalPayload(
            signal_type="BUY",
            confidence=0.6,
            timestamp=1234567890,
            symbol="BTC/USDT",
            timeframe="1h",
            factors=[
                FactorScore("multitimeframe", 0.65, weight=0.10, metadata={"direction": "bullish"}),
            ]
        )
        
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            volume_analysis={"volume_ratio": 2.0, "volume_confidence": 0.8},
            market_structure={"structure_state": "bullish", "structure_score": 0.7},
            advanced_metrics={
                "composite_indicators": {"overall_score": 0.8},
                "market_context": {"score": 0.7},
            },
        )
        
        signal = generate_trading_signal(context)
        
        self.assertEqual(signal.signal_type, "BUY")
        self.assertGreaterEqual(signal.confidence, 0.6)  # Above minimum confidence
        self.assertGreaterEqual(len(signal.factors), 3)  # Minimum factor confirmation
        self.assertIsNotNone(signal.explanation)
        self.assertGreaterEqual(signal.metadata["final_score"], 0.65)  # Buy threshold
    
    @patch('indicator_collector.trading_system.signal_generator.analyze_technical_factors')
    @patch('indicator_collector.trading_system.signal_generator.analyze_sentiment_factors')
    @patch('indicator_collector.trading_system.signal_generator.analyze_multitimeframe_factors')
    def test_sell_signal_generation(self, mock_mt, mock_sentiment, mock_technical):
        """Test SELL signal generation."""
        # Mock analyzer responses
        mock_technical.return_value = {
            "final_score": 0.28,
            "direction": "bearish",
            "confidence": 78.0,
            "rationale": "Momentum indicators show strong bearish pressure",
            "factor_scores": {
                "macd": 0.2,
                "rsi": 0.3,
                "atr": 0.55,
                "bollinger": 0.35,
                "divergence": 0.25,
            },
            "factor_weights": {
                "macd": 0.25,
                "rsi": 0.25,
                "atr": 0.15,
                "bollinger": 0.20,
                "divergence": 0.15,
            },
            "metadata": {
                "total_candles": 120,
            },
        }

        mock_sentiment.return_value = TradingSignalPayload(
            signal_type="SELL",
            confidence=0.7,
            timestamp=1234567890,
            symbol="BTC/USDT",
            timeframe="1h",
            factors=[
                FactorScore("sentiment", 0.25, weight=0.15, metadata={"direction": "bearish"}),
            ]
        )

        mock_mt.return_value = TradingSignalPayload(
            signal_type="SELL",
            confidence=0.6,
            timestamp=1234567890,
            symbol="BTC/USDT",
            timeframe="1h",
            factors=[
                FactorScore("multitimeframe", 0.35, weight=0.10, metadata={"direction": "bearish"}),
            ]
        )
        
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            volume_analysis={"volume_ratio": 0.3, "volume_confidence": 0.2},
            market_structure={"structure_state": "bearish", "structure_score": 0.3},
            advanced_metrics={
                "composite_indicators": {"overall_score": 0.2},
                "market_context": {"score": 0.3},
            },
        )
        
        signal = generate_trading_signal(context)
        
        self.assertEqual(signal.signal_type, "SELL")
        self.assertGreaterEqual(signal.confidence, 0.6)
        self.assertGreaterEqual(len(signal.factors), 3)
        self.assertLessEqual(signal.metadata["final_score"], 0.35)  # Sell threshold
    
    @patch('indicator_collector.trading_system.signal_generator.analyze_technical_factors')
    def test_hold_signal_insufficient_factors(self, mock_technical):
        """Test HOLD signal due to insufficient factors."""
        # Mock only one factor
        mock_technical.return_value = {
            "final_score": 0.55,
            "direction": "bullish",
            "confidence": 60.0,
            "rationale": "Limited technical confirmation available",
            "factor_scores": {
                "macd": 0.8,
            },
            "factor_weights": {
                "macd": 1.0,
            },
            "metadata": {
                "total_candles": 60,
            },
        }
        
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
        )
        
        signal = generate_trading_signal(context)
        
        self.assertEqual(signal.signal_type, "HOLD")
        self.assertIn("insufficient confirmation", signal.explanation.primary_reason)
    
    @patch('indicator_collector.trading_system.signal_generator.analyze_technical_factors')
    def test_hold_signal_cancellation_trigger(self, mock_technical):
        """Test HOLD signal due to cancellation trigger."""
        mock_technical.return_value = {
            "final_score": 0.72,
            "direction": "bullish",
            "confidence": 78.0,
            "rationale": "Bullish technical structure but risk metrics elevated",
            "factor_scores": {
                "macd": 0.82,
                "rsi": 0.74,
                "atr": 0.65,
                "bollinger": 0.70,
                "divergence": 0.6,
            },
            "factor_weights": {
                "macd": 0.25,
                "rsi": 0.25,
                "atr": 0.15,
                "bollinger": 0.20,
                "divergence": 0.15,
            },
            "metadata": {
                "total_candles": 80,
            },
        }
        
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={"atr": 3000},  # Extreme volatility
            advanced_metrics={"risk_metrics": {"risk_score": 0.9}},  # High risk
        )
        
        signal = generate_trading_signal(context)
        
        self.assertEqual(signal.signal_type, "HOLD")
        self.assertGreater(len(signal.metadata["cancellation_triggers"]), 0)
        self.assertIn("cancellation triggers", signal.explanation.primary_reason)
    
    def test_vix_adaptivity_high_vix(self):
        """Test VIX adaptivity with high VIX values."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            extras={"market_context": {"vix": 35.0}},  # High VIX
        )
        
        config = SignalConfig()
        signal = generate_trading_signal(context, config)
        
        # Should have tightened thresholds
        self.assertGreater(signal.metadata["buy_threshold"], config.buy_threshold)
        self.assertGreater(signal.metadata["sell_threshold"], config.sell_threshold)
        self.assertGreater(signal.metadata["min_confidence"], config.min_confidence)
    
    def test_vix_adaptivity_low_vix(self):
        """Test VIX adaptivity with low VIX values."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            extras={"market_context": {"vix": 10.0}},  # Low VIX
        )
        
        config = SignalConfig()
        signal = generate_trading_signal(context, config)
        
        # Should have loosened thresholds
        self.assertLess(signal.metadata["buy_threshold"], config.buy_threshold)
        self.assertLess(signal.metadata["sell_threshold"], config.sell_threshold)
        self.assertLess(signal.metadata["min_confidence"], config.min_confidence)
    
    def test_custom_config(self):
        """Test signal generation with custom configuration."""
        custom_config = SignalConfig(
            min_factors_confirm=2,
            buy_threshold=0.7,
            sell_threshold=0.3,
            min_confidence=0.8,
        )
        
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
            volume_analysis={"volume_ratio": 2.0, "volume_confidence": 0.8},
            market_structure={"structure_state": "bullish", "structure_score": 0.8},
        )
        
        signal = generate_trading_signal(context, custom_config)
        
        # Should use custom thresholds
        self.assertEqual(signal.metadata["buy_threshold"], 0.7)
        self.assertEqual(signal.metadata["sell_threshold"], 0.3)
        self.assertEqual(signal.metadata["min_confidence"], 0.8)
    
    def test_no_analyzers_available(self):
        """Test signal generation when no analyzers are available."""
        context = AnalyzerContext(
            symbol="BTC/USDT",
            timeframe="1h",
            timestamp=1234567890,
            current_price=50000.0,
            ohlcv={"open": 49000, "high": 51000, "low": 48000, "close": 50000, "volume": 1000},
            indicators={},
        )
        
        signal = generate_trading_signal(context)
        
        # Should default to HOLD with neutral score
        self.assertEqual(signal.signal_type, "HOLD")
        self.assertEqual(signal.metadata["final_score"], 0.5)
        # Should have volume, structure, and composite factors even without analyzers
        self.assertGreaterEqual(signal.metadata["available_factors"], 3)


if __name__ == "__main__":
    unittest.main()