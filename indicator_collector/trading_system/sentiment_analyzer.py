"""Sentiment analyzer combining Alternative.me fear & greed with fundamental metrics."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from ..advanced_metrics import calculate_fundamental_metrics, fetch_fear_greed_index
from ..math_utils import Candle
from .interfaces import FactorScore, JsonDict


def _normalize_fear_greed_score(fear_greed_value: int) -> float:
    """
    Normalize fear & greed index (0-100) to 0.0-1.0 score.
    
    Higher fear & greed values indicate greed (bullish), lower values indicate fear (bearish).
    For sentiment analysis: greed = positive sentiment, fear = negative sentiment.
    
    Args:
        fear_greed_value: Fear & greed index value (0-100)
        
    Returns:
        Normalized score (0.0-1.0)
    """
    # Direct mapping: 0 (Extreme Fear) -> 0.0, 100 (Extreme Greed) -> 1.0
    return fear_greed_value / 100.0


def _analyze_fundamental_sentiment(fundamentals: JsonDict) -> float:
    """
    Analyze fundamental metrics for sentiment scoring.
    
    Args:
        fundamentals: Fundamental metrics from calculate_fundamental_metrics
        
    Returns:
        Normalized sentiment score (0.0-1.0)
    """
    # Extract funding rate sentiment
    funding_data = fundamentals.get("funding_rate", {})
    funding_rate = funding_data.get("current", 0.0)
    
    # Positive funding rate indicates longs paying shorts (bullish sentiment)
    # Negative funding rate indicates shorts paying longs (bearish sentiment)
    # Normalize to 0-1 range where positive rates are bullish
    funding_sentiment = (funding_rate + 0.004) / 0.008  # Map -0.004 to 0.004 -> 0 to 1
    funding_sentiment = max(0.0, min(1.0, funding_sentiment))
    
    # Extract open interest change sentiment
    oi_data = fundamentals.get("open_interest", {})
    oi_change_pct = oi_data.get("change_pct", 0.0)
    
    # Positive OI change indicates growing market interest (bullish)
    # Negative OI change indicates declining interest (bearish)
    # Normalize percentage change to 0-1 range
    oi_sentiment = (oi_change_pct + 50) / 100  # Map -50% to +50% -> 0 to 1
    oi_sentiment = max(0.0, min(1.0, oi_sentiment))
    
    # Extract long/short ratio sentiment
    ls_data = fundamentals.get("long_short_ratio", {})
    long_ratio = ls_data.get("long", 0.5)
    
    # Higher long ratio indicates bullish sentiment
    # This is already in 0-1 range
    ls_sentiment = long_ratio
    
    # Combine fundamental sentiment signals with equal weights
    fundamental_score = (funding_sentiment * 0.3 + oi_sentiment * 0.3 + ls_sentiment * 0.4)
    
    return max(0.0, min(1.0, fundamental_score))


def _calculate_sentiment_direction(score: float) -> str:
    """
    Determine sentiment direction from normalized score.
    
    Args:
        score: Normalized sentiment score (0.0-1.0)
        
    Returns:
        Direction string: "bullish", "bearish", or "neutral"
    """
    if score >= 0.65:
        return "bullish"
    elif score <= 0.35:
        return "bearish"
    else:
        return "neutral"


def _get_sentiment_emoji(direction: str) -> str:
    """Get emoji for sentiment direction."""
    return {
        "bullish": "🟢",
        "bearish": "🔴", 
        "neutral": "⚪"
    }.get(direction, "⚪")


def _calculate_confidence(
    fear_greed_confidence: float,
    fundamentals_confidence: float,
    macro_weight: float,
    derivatives_weight: float
) -> int:
    """
    Calculate overall confidence based on component confidence and weights.
    
    Args:
        fear_greed_confidence: Confidence in fear & greed data (0.0-1.0)
        fundamentals_confidence: Confidence in fundamentals data (0.0-1.0)
        macro_weight: Weight assigned to macro sentiment
        derivatives_weight: Weight assigned to derivatives sentiment
        
    Returns:
        Overall confidence score (0-100)
    """
    # Weighted average of component confidences
    weighted_confidence = (
        fear_greed_confidence * macro_weight +
        fundamentals_confidence * derivatives_weight
    ) / (macro_weight + derivatives_weight)
    
    return int(weighted_confidence * 100)


def _generate_rationale(
    fear_greed_data: JsonDict,
    fundamentals: JsonDict,
    macro_score: float,
    derivatives_score: float,
    direction: str
) -> str:
    """
    Generate human-readable rationale for sentiment analysis.
    
    Args:
        fear_greed_data: Fear & greed index data
        fundamentals: Fundamental metrics data
        macro_score: Macro sentiment score
        derivatives_score: Derivatives sentiment score
        direction: Overall sentiment direction
        
    Returns:
        Human-readable rationale string
    """
    fear_greed_value = fear_greed_data.get("fear_greed_index", 50)
    fear_greed_regime = fear_greed_data.get("regime", "Neutral")
    
    funding_rate = fundamentals.get("funding_rate", {}).get("current", 0.0)
    oi_change = fundamentals.get("open_interest", {}).get("change_pct", 0.0)
    long_ratio = fundamentals.get("long_short_ratio", {}).get("long", 0.5)
    
    rationale_parts = []
    
    # Macro sentiment rationale
    rationale_parts.append(
        f"Macro sentiment: {fear_greed_regime} ({fear_greed_value}/100)"
    )
    
    # Derivatives sentiment rationale
    funding_desc = "positive" if funding_rate > 0.001 else "negative" if funding_rate < -0.001 else "neutral"
    oi_desc = "growing" if oi_change > 5 else "declining" if oi_change < -5 else "stable"
    ls_desc = "bullish" if long_ratio > 0.55 else "bearish" if long_ratio < 0.45 else "balanced"
    
    rationale_parts.append(
        f"Derivatives: {funding_desc} funding ({funding_rate:.4f}), {oi_desc} OI ({oi_change:+.1f}%), {ls_desc} long/short ({long_ratio:.2f})"
    )
    
    # Overall assessment
    if direction == "bullish":
        if macro_score > derivatives_score:
            rationale_parts.append("Overall bullish sentiment driven by macro optimism")
        else:
            rationale_parts.append("Overall bullish sentiment driven by derivatives strength")
    elif direction == "bearish":
        if macro_score < derivatives_score:
            rationale_parts.append("Overall bearish sentiment driven by macro fear")
        else:
            rationale_parts.append("Overall bearish sentiment driven by derivatives weakness")
    else:
        rationale_parts.append("Mixed signals result in neutral sentiment")
    
    return " | ".join(rationale_parts)


def analyze_sentiment_factors(candles: Sequence[Candle]) -> JsonDict:
    """
    Analyze sentiment factors combining Alternative.me fear & greed with fundamental metrics.
    
    This function provides a comprehensive sentiment analysis that accounts for both
    macro sentiment (fear & greed index) and derivatives market sentiment (funding rates,
    open interest, long/short ratios). The final score represents 15% of the overall
    technical analysis weight.
    
    Args:
        candles: Recent price candle data
        
    Returns:
        Dictionary containing:
        - final_score: Normalized sentiment score (0.0-1.0)
        - direction: "bullish", "bearish", or "neutral"
        - confidence: Overall confidence score (0-100)
        - rationale: Human-readable explanation
        - components: Detailed breakdown of macro and derivatives sentiment
        - factor_weights: Weights used for each component
        - factor_scores: Individual normalized scores
        - metadata: Additional analysis metadata
    """
    if not candles:
        return {
            "final_score": 0.5,
            "direction": "neutral",
            "confidence": 0,
            "rationale": "Insufficient data for sentiment analysis",
            "components": {},
            "factor_weights": {},
            "factor_scores": {},
            "metadata": {"error": "No candle data provided"},
        }
    
    # Fetch macro sentiment data (Alternative.me fear & greed)
    fear_greed_data = fetch_fear_greed_index()
    fear_greed_value = fear_greed_data.get("fear_greed_index", 50)
    macro_score = _normalize_fear_greed_score(fear_greed_value)
    
    # Calculate fundamental metrics for derivatives sentiment
    fundamentals = calculate_fundamental_metrics(candles)
    derivatives_score = _analyze_fundamental_sentiment(fundamentals)
    
    # Define weights for macro vs derivatives sentiment
    # Macro sentiment reflects broader market psychology
    # Derivatives sentiment reflects trader positioning and leverage
    macro_weight = 0.6  # 60% weight to macro sentiment
    derivatives_weight = 0.4  # 40% weight to derivatives sentiment
    
    # Calculate weighted sentiment score
    final_score = (
        macro_score * macro_weight +
        derivatives_score * derivatives_weight
    )
    
    # Ensure score is within valid range
    final_score = max(0.0, min(1.0, final_score))
    
    # Determine direction and confidence
    direction = _calculate_sentiment_direction(final_score)
    emoji = _get_sentiment_emoji(direction)
    
    # Calculate confidence based on data availability and quality
    fear_greed_confidence = 1.0 if fear_greed_data.get("source") == "alternative.me" else 0.3
    fundamentals_confidence = 0.8 if fundamentals else 0.0  # High confidence if we have fundamentals
    confidence = _calculate_confidence(fear_greed_confidence, fundamentals_confidence, macro_weight, derivatives_weight)
    
    # Generate rationale
    rationale = _generate_rationale(fear_greed_data, fundamentals, macro_score, derivatives_score, direction)
    
    # Prepare component breakdowns
    components = {
        "macro_sentiment": {
            "score": round(macro_score, 3),
            "weight": macro_weight,
            "data": {
                "fear_greed_index": fear_greed_value,
                "regime": fear_greed_data.get("regime", "Neutral"),
                "source": fear_greed_data.get("source", "unknown"),
            },
            "description": f"Market sentiment from Alternative.me fear & greed index ({fear_greed_data.get('regime', 'Neutral')})",
        },
        "derivatives_sentiment": {
            "score": round(derivatives_score, 3),
            "weight": derivatives_weight,
            "data": {
                "funding_rate": fundamentals.get("funding_rate", {}).get("current", 0.0),
                "funding_rate_annualized": fundamentals.get("funding_rate", {}).get("annualized", 0.0),
                "open_interest_change_pct": fundamentals.get("open_interest", {}).get("change_pct", 0.0),
                "long_ratio": fundamentals.get("long_short_ratio", {}).get("long", 0.5),
                "short_ratio": fundamentals.get("long_short_ratio", {}).get("short", 0.5),
            },
            "description": "Derivatives market sentiment from funding rates, OI, and positioning",
        },
    }
    
    # Factor weights and scores for transparency
    factor_weights = {
        "macro_sentiment": macro_weight,
        "derivatives_sentiment": derivatives_weight,
    }
    
    factor_scores = {
        "macro_sentiment": macro_score,
        "derivatives_sentiment": derivatives_score,
    }
    
    # Metadata
    metadata = {
        "analysis_timestamp": candles[-1].close_time if candles else 0,
        "candle_count": len(candles),
        "data_sources": {
            "fear_greed": fear_greed_data.get("source", "unknown"),
            "fundamentals": "calculated" if fundamentals else "unavailable",
        },
        "confidence_breakdown": {
            "fear_greed_confidence": fear_greed_confidence,
            "fundamentals_confidence": fundamentals_confidence,
        },
        "raw_fear_greed_data": fear_greed_data,
        "raw_fundamentals": fundamentals,
    }
    
    return {
        "final_score": round(final_score, 4),
        "direction": direction,
        "confidence": confidence,
        "rationale": rationale,
        "components": components,
        "factor_weights": factor_weights,
        "factor_scores": {k: round(v, 4) for k, v in factor_scores.items()},
        "metadata": metadata,
        "emoji": emoji,
    }


def create_sentiment_factor_score(candles: Sequence[Candle]) -> FactorScore:
    """
    Create a FactorScore object for sentiment analysis.
    
    This function provides the sentiment analysis in the standard FactorScore format
    for integration with the broader trading system.
    
    Args:
        candles: Recent price candle data
        
    Returns:
        FactorScore object with sentiment analysis results
    """
    sentiment_result = analyze_sentiment_factors(candles)
    
    return FactorScore(
        factor_name="sentiment",
        score=sentiment_result["final_score"],
        weight=0.15,  # 15% weight as specified in requirements
        description=sentiment_result["rationale"],
        emoji=sentiment_result["emoji"],
        metadata={
            "direction": sentiment_result["direction"],
            "confidence": sentiment_result["confidence"],
            "components": sentiment_result["components"],
            "factor_weights": sentiment_result["factor_weights"],
            "factor_scores": sentiment_result["factor_scores"],
            "analysis_metadata": sentiment_result["metadata"],
        },
    )