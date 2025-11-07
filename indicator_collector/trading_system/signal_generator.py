"""Trading signal generator that combines analyzer outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import math


from .interfaces import (
    AnalyzerContext,
    FactorScore,
    JsonDict,
    SignalExplanation,
    TradingSignalPayload,
)
from .technical_analysis import analyze_technical_factors
from .sentiment_analyzer import analyze_sentiment_factors
from .multitimeframe_analyzer import analyze_multitimeframe_factors


@dataclass
class SignalConfig:
    """Configuration for signal generation weights and thresholds."""
    
    # Factor weights (must sum to 1.0)
    technical_weight: float = 0.25
    sentiment_weight: float = 0.15
    multitimeframe_weight: float = 0.10
    volume_weight: float = 0.20
    structure_weight: float = 0.15
    composite_weight: float = 0.15
    
    # Thresholds
    min_factors_confirm: int = 3
    buy_threshold: float = 0.65
    sell_threshold: float = 0.35
    min_confidence: float = 0.6
    
    # VIX adaptivity
    vix_tighten_threshold: float = 30.0
    vix_loosen_threshold: float = 15.0
    vix_tighten_factor: float = 1.5  # Multiply thresholds by this
    vix_loosen_factor: float = 0.8  # Multiply thresholds by this
    
    # Cancellation triggers
    max_risk_score: float = 0.8
    min_liquidity_score: float = 0.2
    max_volatility_ratio: float = 5.0
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        total_weight = (
            self.technical_weight + self.sentiment_weight + self.multitimeframe_weight +
            self.volume_weight + self.structure_weight + self.composite_weight
        )
        if not math.isclose(total_weight, 1.0, rel_tol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def get_adapted_thresholds(self, vix_value: Optional[float] = None) -> Tuple[float, float, float]:
        """Get thresholds adapted to VIX levels."""
        buy_threshold = self.buy_threshold
        sell_threshold = self.sell_threshold
        min_confidence = self.min_confidence
        
        if vix_value is not None:
            if vix_value > self.vix_tighten_threshold:
                # Tighten filters in high volatility
                buy_threshold *= self.vix_tighten_factor
                sell_threshold *= self.vix_tighten_factor
                min_confidence *= self.vix_tighten_factor
            elif vix_value < self.vix_loosen_threshold:
                # Loosen filters in low volatility
                buy_threshold *= self.vix_loosen_factor
                sell_threshold *= self.vix_loosen_factor
                min_confidence *= self.vix_loosen_factor
        
        # Clamp to valid ranges
        buy_threshold = min(max(buy_threshold, 0.5), 0.9)
        sell_threshold = max(min(sell_threshold, 0.5), 0.1)
        min_confidence = min(max(min_confidence, 0.3), 0.95)
        
        return buy_threshold, sell_threshold, min_confidence
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "technical_weight": self.technical_weight,
            "sentiment_weight": self.sentiment_weight,
            "multitimeframe_weight": self.multitimeframe_weight,
            "volume_weight": self.volume_weight,
            "structure_weight": self.structure_weight,
            "composite_weight": self.composite_weight,
            "min_factors_confirm": self.min_factors_confirm,
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "min_confidence": self.min_confidence,
            "vix_tighten_threshold": self.vix_tighten_threshold,
            "vix_loosen_threshold": self.vix_loosen_threshold,
            "vix_tighten_factor": self.vix_tighten_factor,
            "vix_loosen_factor": self.vix_loosen_factor,
            "max_risk_score": self.max_risk_score,
            "min_liquidity_score": self.min_liquidity_score,
            "max_volatility_ratio": self.max_volatility_ratio,
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "SignalConfig":
        """Create from dictionary."""
        return cls(
            technical_weight=float(data.get("technical_weight", 0.25)),
            sentiment_weight=float(data.get("sentiment_weight", 0.15)),
            multitimeframe_weight=float(data.get("multitimeframe_weight", 0.10)),
            volume_weight=float(data.get("volume_weight", 0.20)),
            structure_weight=float(data.get("structure_weight", 0.15)),
            composite_weight=float(data.get("composite_weight", 0.15)),
            min_factors_confirm=int(data.get("min_factors_confirm", 3)),
            buy_threshold=float(data.get("buy_threshold", 0.65)),
            sell_threshold=float(data.get("sell_threshold", 0.35)),
            min_confidence=float(data.get("min_confidence", 0.6)),
            vix_tighten_threshold=float(data.get("vix_tighten_threshold", 30.0)),
            vix_loosen_threshold=float(data.get("vix_loosen_threshold", 15.0)),
            vix_tighten_factor=float(data.get("vix_tighten_factor", 1.5)),
            vix_loosen_factor=float(data.get("vix_loosen_factor", 0.8)),
            max_risk_score=float(data.get("max_risk_score", 0.8)),
            min_liquidity_score=float(data.get("min_liquidity_score", 0.2)),
            max_volatility_ratio=float(data.get("max_volatility_ratio", 5.0)),
        )


@dataclass
class SignalFactors:
    """Container for all factor scores used in signal generation."""
    
    technical: Optional[FactorScore] = None
    sentiment: Optional[FactorScore] = None
    multitimeframe: Optional[FactorScore] = None
    volume: Optional[FactorScore] = None
    structure: Optional[FactorScore] = None
    composite: Optional[FactorScore] = None
    
    def get_available_factors(self) -> List[FactorScore]:
        """Get list of non-None factors."""
        return [
            factor for factor in [
                self.technical, self.sentiment, self.multitimeframe,
                self.volume, self.structure, self.composite
            ]
            if factor is not None
        ]
    
    def count_available_factors(self) -> int:
        """Count available factors."""
        return len(self.get_available_factors())
    
    def get_bullish_factors(self) -> List[FactorScore]:
        """Get factors with bullish direction."""
        return [
            factor for factor in self.get_available_factors()
            if factor.metadata.get("direction") == "bullish"
        ]
    
    def get_bearish_factors(self) -> List[FactorScore]:
        """Get factors with bearish direction."""
        return [
            factor for factor in self.get_available_factors()
            if factor.metadata.get("direction") == "bearish"
        ]


def _create_volume_factor(context: AnalyzerContext) -> Optional[FactorScore]:
    """Create volume factor from context data."""
    volume_analysis = context.volume_analysis or {}
    advanced_metrics = context.advanced_metrics or {}
    
    # Extract volume metrics
    volume_ratio = volume_analysis.get("volume_ratio", 1.0)
    volume_confidence = volume_analysis.get("volume_confidence", 0.5)
    smart_money = advanced_metrics.get("smart_money_activity", {})
    
    # Calculate volume score
    score = 0.5  # Neutral base
    
    # Volume ratio contribution (0-1)
    if volume_ratio > 2.0:
        score += 0.2
    elif volume_ratio > 1.5:
        score += 0.1
    elif volume_ratio < 0.5:
        score -= 0.1
    
    # Volume confidence contribution (0-1)
    score += (volume_confidence - 0.5) * 0.3
    
    # Smart money contribution
    if smart_money:
        smart_money_score = smart_money.get("score", 0.5)
        score += (smart_money_score - 0.5) * 0.2
    
    # Clamp and normalize
    score = max(0.0, min(1.0, score))
    
    # Determine direction
    if score > 0.6:
        direction = "bullish"
        emoji = "🟢"
    elif score < 0.4:
        direction = "bearish"
        emoji = "🔴"
    else:
        direction = "neutral"
        emoji = "⚪"
    
    return FactorScore(
        factor_name="volume_analysis",
        score=score,
        weight=0.20,
        description=f"Volume analysis with ratio {volume_ratio:.2f} and confidence {volume_confidence:.2f}",
        emoji=emoji,
        metadata={
            "direction": direction,
            "volume_ratio": volume_ratio,
            "volume_confidence": volume_confidence,
            "smart_money_score": smart_money.get("score") if smart_money else None,
        }
    )


def _create_structure_factor(context: AnalyzerContext) -> Optional[FactorScore]:
    """Create market structure factor from context data."""
    market_structure = context.market_structure or {}
    advanced_metrics = context.advanced_metrics or {}
    
    # Extract structure metrics
    structure_state = market_structure.get("structure_state", "neutral")
    structure_score = market_structure.get("structure_score", 0.5)
    breadth_metrics = advanced_metrics.get("market_breadth", {})
    
    # Calculate structure score
    score = 0.5  # Neutral base
    
    # Structure state contribution
    if structure_state == "bullish":
        score += 0.2
    elif structure_state == "bearish":
        score -= 0.2
    
    # Structure score contribution
    score += (structure_score - 0.5) * 0.4
    
    # Market breadth contribution
    if breadth_metrics:
        breadth_score = breadth_metrics.get("score", 0.5)
        score += (breadth_score - 0.5) * 0.2
    
    # Clamp and normalize
    score = max(0.0, min(1.0, score))
    
    # Determine direction
    if score > 0.6:
        direction = "bullish"
        emoji = "🟢"
    elif score < 0.4:
        direction = "bearish"
        emoji = "🔴"
    else:
        direction = "neutral"
        emoji = "⚪"
    
    return FactorScore(
        factor_name="market_structure",
        score=score,
        weight=0.15,
        description=f"Market structure {structure_state} with score {structure_score:.2f}",
        emoji=emoji,
        metadata={
            "direction": direction,
            "structure_state": structure_state,
            "structure_score": structure_score,
            "breadth_score": breadth_metrics.get("score") if breadth_metrics else None,
        }
    )


def _create_composite_factor(context: AnalyzerContext) -> Optional[FactorScore]:
    """Create composite factor from advanced metrics."""
    advanced_metrics = context.advanced_metrics or {}
    
    # Extract composite metrics
    composite_indicators = advanced_metrics.get("composite_indicators", {})
    market_context = advanced_metrics.get("market_context", {})
    
    # Calculate composite score
    score = 0.5  # Neutral base
    
    # Composite indicators contribution
    if composite_indicators:
        composite_score = composite_indicators.get("overall_score", 0.5)
        score += (composite_score - 0.5) * 0.5
    
    # Market context contribution
    if market_context:
        context_score = market_context.get("score", 0.5)
        score += (context_score - 0.5) * 0.3
    
    # Clamp and normalize
    score = max(0.0, min(1.0, score))
    
    # Determine direction
    if score > 0.6:
        direction = "bullish"
        emoji = "🟢"
    elif score < 0.4:
        direction = "bearish"
        emoji = "🔴"
    else:
        direction = "neutral"
        emoji = "⚪"
    
    return FactorScore(
        factor_name="composite_analysis",
        score=score,
        weight=0.15,
        description=f"Composite analysis with overall score {score:.2f}",
        emoji=emoji,
        metadata={
            "direction": direction,
            "composite_score": composite_indicators.get("overall_score") if composite_indicators else None,
            "context_score": market_context.get("score") if market_context else None,
        }
    )


def _check_cancellation_triggers(context: AnalyzerContext, factors: SignalFactors) -> List[str]:
    """Check for scenario cancellation triggers."""
    triggers = []
    
    # High risk trigger
    advanced_metrics = context.advanced_metrics or {}
    risk_metrics = advanced_metrics.get("risk_metrics", {})
    if risk_metrics and risk_metrics.get("risk_score", 0) > 0.8:
        triggers.append("High risk score detected")
    
    # Low liquidity trigger
    volume_analysis = context.volume_analysis or {}
    if volume_analysis.get("liquidity_score", 1.0) < 0.2:
        triggers.append("Low liquidity detected")
    
    # Extreme volatility trigger
    indicators = context.indicators or {}
    current_atr = indicators.get("atr", 0)
    current_price = context.current_price
    if current_atr > 0 and current_price > 0:
        volatility_ratio = (current_atr / current_price) * 100
        if volatility_ratio > 5.0:
            triggers.append(f"Extreme volatility: {volatility_ratio:.1f}%")
    
    # Conflicting signals trigger
    bullish_count = len(factors.get_bullish_factors())
    bearish_count = len(factors.get_bearish_factors())
    total_factors = factors.count_available_factors()
    
    if total_factors >= 4 and min(bullish_count, bearish_count) >= 2:
        triggers.append("Strong conflicting signals detected")
    
    return triggers


def _calculate_confidence(
    final_score: float,
    factors: SignalFactors,
    buy_threshold: float,
    sell_threshold: float,
    cancellation_triggers: List[str]
) -> Tuple[int, float]:
    """Calculate confidence level (1-10) and normalized confidence."""
    # Base confidence from score distance from neutral
    if final_score > 0.5:
        # Bullish: distance from 0.5 to 1.0
        base_confidence = (final_score - 0.5) * 20  # 0-10 scale
    else:
        # Bearish: distance from 0.5 to 0.0
        base_confidence = (0.5 - final_score) * 20  # 0-10 scale
    
    # Adjust for factor count
    factor_count = factors.count_available_factors()
    if factor_count >= 5:
        factor_adjustment = 1.2
    elif factor_count >= 3:
        factor_adjustment = 1.0
    else:
        factor_adjustment = 0.7
    
    # Adjust for threshold distance
    if final_score > buy_threshold:
        threshold_bonus = 1.5
    elif final_score < sell_threshold:
        threshold_bonus = 1.5
    else:
        threshold_bonus = 0.8
    
    # Penalty for cancellation triggers
    trigger_penalty = max(0.3, 1.0 - len(cancellation_triggers) * 0.2)
    
    # Calculate final confidence
    confidence = base_confidence * factor_adjustment * threshold_bonus * trigger_penalty
    confidence = max(1.0, min(10.0, confidence))
    
    # Normalized confidence for payload
    normalized_confidence = confidence / 10.0
    
    return int(confidence), normalized_confidence


def _generate_explanation(
    signal_type: str,
    final_score: float,
    factors: SignalFactors,
    confidence: int,
    cancellation_triggers: List[str]
) -> SignalExplanation:
    """Generate detailed explanation for the signal."""
    
    # Primary reason
    if cancellation_triggers:
        primary_reason = f"HOLD due to cancellation triggers: {', '.join(cancellation_triggers)}"
    elif signal_type == "BUY":
        primary_reason = f"Bullish signal with score {final_score:.2f} and confidence {confidence}/10"
    elif signal_type == "SELL":
        primary_reason = f"Bearish signal with score {final_score:.2f} and confidence {confidence}/10"
    else:
        primary_reason = f"Neutral signal with score {final_score:.2f} - insufficient confirmation"
    
    # Supporting factors
    supporting_factors = []
    bullish_factors = factors.get_bullish_factors()
    bearish_factors = factors.get_bearish_factors()
    
    if signal_type == "BUY" and bullish_factors:
        supporting_factors.extend([f"{f.factor_name}: {f.score:.2f}" for f in bullish_factors[:3]])
    elif signal_type == "SELL" and bearish_factors:
        supporting_factors.extend([f"{f.factor_name}: {f.score:.2f}" for f in bearish_factors[:3]])
    
    # Risk factors
    risk_factors = list(cancellation_triggers)
    if len(bullish_factors) >= 2 and len(bearish_factors) >= 2:
        risk_factors.append("Mixed signals across factors")
    
    # Market context
    market_context = f"Signal generated based on {factors.count_available_factors()} factors"
    
    return SignalExplanation(
        primary_reason=primary_reason,
        supporting_factors=supporting_factors,
        risk_factors=risk_factors,
        market_context=market_context,
        metadata={
            "bullish_factors": len(bullish_factors),
            "bearish_factors": len(bearish_factors),
            "neutral_factors": factors.count_available_factors() - len(bullish_factors) - len(bearish_factors),
        }
    )


def generate_trading_signal(
    context: AnalyzerContext,
    config: Optional[SignalConfig] = None
) -> TradingSignalPayload:
    """Generate a comprehensive trading signal combining all analyzer outputs."""
    
    if config is None:
        config = SignalConfig()
    
    # Generate individual factor scores
    factors = SignalFactors()
    
    # Technical analysis (25%)
    try:
        technical_payload = analyze_technical_factors(context)
        if technical_payload.factors:
            # Combine technical factors into weighted average
            tech_score = sum(f.score * f.weight for f in technical_payload.factors) / sum(f.weight for f in technical_payload.factors)
            tech_direction = technical_payload.factors[0].metadata.get("direction", "neutral")
            factors.technical = FactorScore(
                factor_name="technical_analysis",
                score=tech_score,
                weight=config.technical_weight,
                description="Technical analysis combining MACD, RSI, ATR, and Bollinger Bands",
                emoji=technical_payload.factors[0].emoji,
                metadata={"direction": tech_direction, "sub_factors": len(technical_payload.factors)}
            )
    except Exception:
        factors.technical = None
    
    # Sentiment analysis (15%)
    try:
        sentiment_payload = analyze_sentiment_factors(context)
        if sentiment_payload.factors:
            factors.sentiment = sentiment_payload.factors[0]
            factors.sentiment.weight = config.sentiment_weight
    except Exception:
        factors.sentiment = None
    
    # Multi-timeframe analysis (10%)
    try:
        mt_payload = analyze_multitimeframe_factors(context)
        if mt_payload.factors:
            factors.multitimeframe = mt_payload.factors[0]
            factors.multitimeframe.weight = config.multitimeframe_weight
    except Exception:
        factors.multitimeframe = None
    
    # Volume analysis (20%)
    factors.volume = _create_volume_factor(context)
    if factors.volume:
        factors.volume.weight = config.volume_weight
    
    # Market structure (15%)
    factors.structure = _create_structure_factor(context)
    if factors.structure:
        factors.structure.weight = config.structure_weight
    
    # Composite analysis (15%)
    factors.composite = _create_composite_factor(context)
    if factors.composite:
        factors.composite.weight = config.composite_weight
    
    # Calculate final weighted score
    available_factors = factors.get_available_factors()
    if not available_factors:
        final_score = 0.5  # Neutral if no factors
    else:
        total_weight = sum(f.weight for f in available_factors)
        if total_weight > 0:
            final_score = sum(f.score * f.weight for f in available_factors) / total_weight
        else:
            final_score = 0.5
    
    # Get VIX for adaptivity (try to get from context extras)
    vix_value = None
    if context.extras and "market_context" in context.extras:
        market_context = context.extras["market_context"]
        if isinstance(market_context, dict) and "vix" in market_context:
            vix_value = market_context["vix"]
    
    # Get adapted thresholds
    buy_threshold, sell_threshold, min_confidence = config.get_adapted_thresholds(vix_value)
    
    # Check cancellation triggers
    cancellation_triggers = _check_cancellation_triggers(context, factors)
    
    # Determine signal type
    if cancellation_triggers:
        signal_type = "HOLD"
    elif final_score >= buy_threshold and factors.count_available_factors() >= config.min_factors_confirm:
        signal_type = "BUY"
    elif final_score <= sell_threshold and factors.count_available_factors() >= config.min_factors_confirm:
        signal_type = "SELL"
    else:
        signal_type = "HOLD"
    
    # Calculate confidence
    confidence_int, confidence_float = _calculate_confidence(
        final_score, factors, buy_threshold, sell_threshold, cancellation_triggers
    )
    
    # Apply minimum confidence filter
    if confidence_float < min_confidence:
        signal_type = "HOLD"
    
    # Generate explanation
    explanation = _generate_explanation(
        signal_type, final_score, factors, confidence_int, cancellation_triggers
    )
    
    # Create payload
    payload = TradingSignalPayload(
        signal_type=signal_type,
        confidence=confidence_float,
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=available_factors,
        explanation=explanation,
        metadata={
            "final_score": final_score,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "min_confidence": min_confidence,
            "vix_value": vix_value,
            "cancellation_triggers": cancellation_triggers,
            "available_factors": factors.count_available_factors(),
            "config_weights": {
                "technical": config.technical_weight,
                "sentiment": config.sentiment_weight,
                "multitimeframe": config.multitimeframe_weight,
                "volume": config.volume_weight,
                "structure": config.structure_weight,
                "composite": config.composite_weight,
            }
        }
    )

    return payload


class SignalGenerator:
    """Wrapper class for trading signal generation.
    
    Provides an interface for the payload loader to call signal generation
    with a consistent `analyze()` method.
    """
    
    def __init__(self, config: Optional[SignalConfig] = None):
        """Initialize with optional custom configuration.
        
        Args:
            config: SignalConfig instance (uses defaults if not provided)
        """
        self.config = config or SignalConfig()
    
    def analyze(self, context: AnalyzerContext, 
               config: Optional[SignalConfig] = None) -> TradingSignalPayload:
        """Analyze trading context and generate signal.
        
        Args:
            context: AnalyzerContext with market data
            config: Optional SignalConfig to override instance config
            
        Returns:
            TradingSignalPayload with generated signal
        """
        signal_config = config or self.config
        return generate_trading_signal(context, signal_config)