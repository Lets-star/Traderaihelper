"""Multi-timeframe analyzer evaluating alignment, agreement, and trend force."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from ..math_utils import Candle
from .interfaces import FactorScore, JsonDict


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value between lower and upper bounds."""
    return max(lower, min(upper, value))


def _normalize_to_01(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Normalize a value to 0-1 range."""
    if max_val <= min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return _clamp(normalized, 0.0, 1.0)


def _get_direction_emoji(direction: str) -> str:
    """Get emoji for direction."""
    return {
        "bullish": "🟢",
        "bearish": "🔴",
        "neutral": "⚪"
    }.get(direction, "⚪")


def _calculate_trend_strength_from_candles(
    candles: Sequence[Candle],
    lookback: int = 14,
) -> float:
    """
    Calculate trend strength from candle data.
    
    Uses price momentum and directional movement to determine trend strength.
    
    Args:
        candles: Sequence of candles
        lookback: Period for trend calculation (default 14)
        
    Returns:
        Trend strength (0.0-1.0)
    """
    if not candles or len(candles) < lookback + 1:
        return 0.5
    
    recent_candles = list(candles[-lookback:])
    closes = [c.close for c in recent_candles]
    
    if len(closes) < 2:
        return 0.5
    
    # Calculate directional movement
    up_moves = 0
    down_moves = 0
    total_range = 0.0
    
    for i in range(1, len(closes)):
        move = closes[i] - closes[i-1]
        total_range += abs(move)
        
        if move > 0:
            up_moves += 1
        elif move < 0:
            down_moves += 1
    
    # Calculate momentum
    momentum_direction = (up_moves - down_moves) / lookback if lookback > 0 else 0.0
    momentum_direction = _clamp(momentum_direction, -1.0, 1.0)
    
    # Calculate volatility component (higher volatility = stronger trend)
    avg_move = total_range / lookback if lookback > 0 else 0.0
    avg_price = statistics.fmean(closes) if closes else 1.0
    volatility_ratio = (avg_move / avg_price * 100) if avg_price > 0 else 0.0
    volatility_score = _normalize_to_01(volatility_ratio, 0.0, 5.0)
    
    # Combine momentum direction and volatility
    trend_strength = (momentum_direction + 1.0) / 2.0  # Convert -1..1 to 0..1
    trend_strength = trend_strength * 0.6 + volatility_score * 0.4
    
    return _clamp(trend_strength, 0.0, 1.0)


def _analyze_timeframe_alignment(
    timeframe_directions: Dict[str, str],
) -> Dict[str, object]:
    """
    Analyze alignment of trend directions across timeframes.
    
    Args:
        timeframe_directions: Dict mapping timeframe names to directions (bullish/bearish/neutral)
        
    Returns:
        Dict with alignment metrics
    """
    if not timeframe_directions:
        return {
            "alignment_score": 0.5,
            "aligned_timeframes": 0,
            "conflict_timeframes": 0,
            "alignment_type": "none",
            "conflicting_pairs": [],
        }
    
    bullish_tfs = [tf for tf, dir in timeframe_directions.items() if dir == "bullish"]
    bearish_tfs = [tf for tf, dir in timeframe_directions.items() if dir == "bearish"]
    neutral_tfs = [tf for tf, dir in timeframe_directions.items() if dir == "neutral"]
    
    total_tfs = len(timeframe_directions)
    
    # Calculate alignment score - direction aware
    if bullish_tfs and not bearish_tfs:
        # All aligned bullish
        alignment_ratio = len(bullish_tfs) / total_tfs if total_tfs > 0 else 0.0
        alignment_score = 0.5 + (alignment_ratio * 0.5)  # 0.5-1.0 range for bullish
        alignment_type = "all_bullish"
    elif bearish_tfs and not bullish_tfs:
        # All aligned bearish
        alignment_ratio = len(bearish_tfs) / total_tfs if total_tfs > 0 else 0.0
        alignment_score = 0.5 - (alignment_ratio * 0.5)  # 0.0-0.5 range for bearish
        alignment_type = "all_bearish"
    elif bullish_tfs and bearish_tfs:
        # Conflicting timeframes
        bullish_ratio = len(bullish_tfs) / total_tfs
        alignment_score = 0.5 + (bullish_ratio * 0.5 - 0.25)  # Bias toward bullish/bearish based on ratio
        alignment_type = "conflict"
    else:
        alignment_score = 0.5
        alignment_type = "neutral"
    
    aligned_timeframes = max(len(bullish_tfs), len(bearish_tfs))
    conflict_timeframes = min(len(bullish_tfs), len(bearish_tfs))
    
    return {
        "alignment_score": round(alignment_score, 3),
        "aligned_timeframes": aligned_timeframes,
        "conflict_timeframes": conflict_timeframes,
        "alignment_type": alignment_type,
        "bullish_count": len(bullish_tfs),
        "bearish_count": len(bearish_tfs),
        "neutral_count": len(neutral_tfs),
    }


def _analyze_trend_agreement(
    timeframe_strengths: Dict[str, float],
) -> Dict[str, object]:
    """
    Analyze agreement of trend strength across timeframes.
    
    Args:
        timeframe_strengths: Dict mapping timeframe names to strength values (0-1)
        
    Returns:
        Dict with agreement metrics
    """
    if not timeframe_strengths:
        return {
            "agreement_score": 0.5,
            "strength_variance": 0.0,
            "consensus_strength": 0.5,
            "agreement_type": "none",
        }
    
    strengths = [s for s in timeframe_strengths.values() if isinstance(s, (int, float)) and s == s]
    
    if not strengths:
        return {
            "agreement_score": 0.5,
            "strength_variance": 0.0,
            "consensus_strength": 0.5,
            "agreement_type": "insufficient_data",
        }
    
    # Calculate mean and variance
    mean_strength = statistics.fmean(strengths)
    
    if len(strengths) > 1:
        variance = statistics.variance(strengths)
        std_dev = statistics.stdev(strengths)
    else:
        variance = 0.0
        std_dev = 0.0
    
    # Low variance = high agreement
    # Normalize variance to 0-1 score (lower variance = higher score)
    max_possible_variance = 0.25  # Max variance when some are 0.0 and others are 1.0
    variance_alignment = 1.0 - (variance / max_possible_variance) if max_possible_variance > 0 else 1.0
    variance_alignment = _clamp(variance_alignment, 0.0, 1.0)
    
    # Blend with mean strength to get direction-aware agreement
    # agreement_score: 50% mean strength bias, 50% variance alignment
    # This ensures the direction (mean_strength) has significant influence
    agreement_score = mean_strength * 0.5 + variance_alignment * 0.5
    agreement_score = _clamp(agreement_score, 0.0, 1.0)
    
    # Determine agreement type based on variance
    if std_dev < 0.1:
        agreement_type = "strong"
    elif std_dev < 0.2:
        agreement_type = "moderate"
    else:
        agreement_type = "weak"
    
    return {
        "agreement_score": round(agreement_score, 3),
        "strength_variance": round(variance, 4),
        "std_dev": round(std_dev, 4),
        "consensus_strength": round(mean_strength, 3),
        "agreement_type": agreement_type,
    }


def _analyze_trend_force(
    timeframe_strengths: Dict[str, float],
    timeframe_directions: Dict[str, str],
) -> Dict[str, object]:
    """
    Analyze the force/strength of trends across timeframes.
    
    Args:
        timeframe_strengths: Dict mapping timeframe names to strength values (0-1)
        timeframe_directions: Dict mapping timeframe names to directions
        
    Returns:
        Dict with trend force metrics
    """
    if not timeframe_strengths or not timeframe_directions:
        return {
            "trend_force_score": 0.5,
            "force_type": "none",
            "strong_count": 0,
            "weak_count": 0,
            "avg_force": 0.5,
        }
    
    strong_count = 0
    weak_count = 0
    force_values = []
    
    for tf, strength in timeframe_strengths.items():
        if not isinstance(strength, (int, float)) or strength != strength:  # NaN check
            continue
        
        direction = timeframe_directions.get(tf, "neutral")
        
        # Force is strength weighted by confidence in direction
        if direction != "neutral":
            force = strength
            force_values.append(force)
            
            if force >= 0.7:
                strong_count += 1
            elif force < 0.4:
                weak_count += 1
    
    if not force_values:
        avg_force = 0.5
        trend_force_score = 0.5
        force_type = "insufficient"
    else:
        avg_force = statistics.fmean(force_values)
        trend_force_score = avg_force
        
        if avg_force >= 0.7:
            force_type = "strong"
        elif avg_force >= 0.5:
            force_type = "moderate"
        else:
            force_type = "weak"
    
    return {
        "trend_force_score": round(trend_force_score, 3),
        "force_type": force_type,
        "strong_count": strong_count,
        "weak_count": weak_count,
        "avg_force": round(avg_force, 3),
    }


def _generate_multitf_rationale(
    alignment: Dict[str, object],
    agreement: Dict[str, object],
    trend_force: Dict[str, object],
    direction: str,
) -> str:
    """
    Generate human-readable rationale for multi-timeframe analysis.
    
    Args:
        alignment: Alignment metrics
        agreement: Agreement metrics
        trend_force: Trend force metrics
        direction: Overall direction
        
    Returns:
        Human-readable rationale string
    """
    parts = []
    
    # Alignment rationale
    alignment_type = alignment.get("alignment_type", "none")
    if alignment_type == "all_bullish":
        parts.append("All timeframes aligned bullish")
    elif alignment_type == "all_bearish":
        parts.append("All timeframes aligned bearish")
    elif alignment_type == "conflict":
        bullish = alignment.get("bullish_count", 0)
        bearish = alignment.get("bearish_count", 0)
        parts.append(f"Timeframe conflict: {bullish} bullish vs {bearish} bearish")
    else:
        parts.append("Mixed timeframe signals")
    
    # Agreement rationale
    agreement_type = agreement.get("agreement_type", "weak")
    if agreement_type == "strong":
        parts.append("Strong agreement in trend strength")
    elif agreement_type == "moderate":
        parts.append("Moderate agreement in trend strength")
    else:
        parts.append("Weak agreement in trend strength")
    
    # Trend force rationale
    force_type = trend_force.get("force_type", "weak")
    avg_force = trend_force.get("avg_force", 0.5)
    if force_type == "strong":
        parts.append(f"Strong trend force ({avg_force:.2f})")
    elif force_type == "moderate":
        parts.append(f"Moderate trend force ({avg_force:.2f})")
    else:
        parts.append(f"Weak trend force ({avg_force:.2f})")
    
    return "; ".join(parts)


def _calculate_multitf_confidence(
    alignment: Dict[str, object],
    agreement: Dict[str, object],
    num_timeframes: int,
) -> int:
    """
    Calculate overall confidence for multi-timeframe analysis.
    
    Args:
        alignment: Alignment metrics
        agreement: Agreement metrics
        num_timeframes: Number of timeframes analyzed
        
    Returns:
        Confidence score (0-100)
    """
    if num_timeframes == 0:
        return 0
    
    alignment_score = alignment.get("alignment_score", 0.5)
    agreement_score = agreement.get("agreement_score", 0.5)
    
    # More timeframes analyzed = higher confidence (up to 5)
    num_tf_factor = min(num_timeframes / 5.0, 1.0)
    
    # Combine alignment and agreement with timeframe factor
    confidence = (alignment_score * 0.4 + agreement_score * 0.4 + num_tf_factor * 0.2) * 100
    
    return int(confidence)


def analyze_multitimeframe_factors(
    main_candles: Sequence[Candle],
    multi_timeframe_candles: Dict[str, Sequence[Candle]],
    multi_timeframe_strengths: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """
    Analyze multi-timeframe alignment, agreement, and trend force.
    
    Args:
        main_candles: Candle data for the main timeframe
        multi_timeframe_candles: Dict mapping timeframe names to their candle sequences
        multi_timeframe_strengths: Dict mapping timeframe names to pre-calculated trend strengths
        
    Returns:
        Dict with comprehensive multi-timeframe analysis including:
        - final_score: Normalized score (0.0-1.0)
        - direction: "bullish", "bearish", or "neutral"
        - confidence: 0-100 confidence metric
        - rationale: Human-readable explanation
        - per_timeframe_flags: Per-timeframe analysis details
        - components: Breakdown of alignment, agreement, trend force
        - metadata: Timestamp and analysis metadata
    """
    if not multi_timeframe_candles:
        return {
            "final_score": 0.5,
            "direction": "neutral",
            "confidence": 0,
            "rationale": "No multi-timeframe data available",
            "emoji": "⚪",
            "per_timeframe_flags": {},
            "components": {
                "alignment": {
                    "alignment_score": 0.5,
                    "alignment_type": "none",
                },
                "agreement": {
                    "agreement_score": 0.5,
                    "agreement_type": "none",
                },
                "trend_force": {
                    "trend_force_score": 0.5,
                    "force_type": "none",
                },
            },
            "factor_weights": {
                "alignment": 0.4,
                "agreement": 0.35,
                "trend_force": 0.25,
            },
            "metadata": {
                "timeframe_count": 0,
            },
        }
    
    # Calculate trend strengths and directions for each timeframe
    timeframe_strengths: Dict[str, float] = {}
    timeframe_directions: Dict[str, str] = {}
    per_timeframe_flags: Dict[str, object] = {}
    
    for tf_name, candles in multi_timeframe_candles.items():
        if not candles or len(candles) < 3:
            continue
        
        # Use provided strength or calculate from candles
        if multi_timeframe_strengths and tf_name in multi_timeframe_strengths:
            strength = multi_timeframe_strengths[tf_name]
        else:
            strength = _calculate_trend_strength_from_candles(list(candles), lookback=14)
        
        timeframe_strengths[tf_name] = strength
        
        # Determine direction from strength
        if strength >= 0.65:
            direction = "bullish"
        elif strength <= 0.35:
            direction = "bearish"
        else:
            direction = "neutral"
        
        timeframe_directions[tf_name] = direction
        
        # Per-timeframe flags
        per_timeframe_flags[tf_name] = {
            "strength": round(strength, 3),
            "direction": direction,
            "emoji": _get_direction_emoji(direction),
            "candle_count": len(candles),
        }
    
    num_timeframes = len(timeframe_strengths)
    
    if num_timeframes == 0:
        return {
            "final_score": 0.5,
            "direction": "neutral",
            "confidence": 0,
            "rationale": "Insufficient multi-timeframe data",
            "emoji": "⚪",
            "per_timeframe_flags": {},
            "components": {
                "alignment": {
                    "alignment_score": 0.5,
                    "alignment_type": "none",
                },
                "agreement": {
                    "agreement_score": 0.5,
                    "agreement_type": "none",
                },
                "trend_force": {
                    "trend_force_score": 0.5,
                    "force_type": "none",
                },
            },
            "factor_weights": {
                "alignment": 0.4,
                "agreement": 0.35,
                "trend_force": 0.25,
            },
            "metadata": {
                "timeframe_count": 0,
            },
        }
    
    # Analyze components
    alignment = _analyze_timeframe_alignment(timeframe_directions)
    agreement = _analyze_trend_agreement(timeframe_strengths)
    trend_force = _analyze_trend_force(timeframe_strengths, timeframe_directions)
    
    # Calculate weighted final score
    alignment_weight = 0.4
    agreement_weight = 0.35
    trend_force_weight = 0.25
    
    alignment_score = alignment.get("alignment_score", 0.5)
    agreement_score = agreement.get("agreement_score", 0.5)
    trend_force_score = trend_force.get("trend_force_score", 0.5)
    
    final_score = (
        alignment_score * alignment_weight +
        agreement_score * agreement_weight +
        trend_force_score * trend_force_weight
    )
    final_score = _clamp(final_score, 0.0, 1.0)
    
    # Determine overall direction
    if final_score >= 0.65:
        direction = "bullish"
    elif final_score <= 0.35:
        direction = "bearish"
    else:
        direction = "neutral"
    
    # Calculate confidence
    confidence = _calculate_multitf_confidence(alignment, agreement, num_timeframes)
    
    # Generate rationale
    rationale = _generate_multitf_rationale(alignment, agreement, trend_force, direction)
    
    return {
        "final_score": round(final_score, 3),
        "direction": direction,
        "confidence": confidence,
        "rationale": rationale,
        "emoji": _get_direction_emoji(direction),
        "per_timeframe_flags": per_timeframe_flags,
        "components": {
            "alignment": {
                "alignment_score": alignment.get("alignment_score", 0.5),
                "alignment_type": alignment.get("alignment_type", "none"),
                "bullish_count": alignment.get("bullish_count", 0),
                "bearish_count": alignment.get("bearish_count", 0),
                "neutral_count": alignment.get("neutral_count", 0),
            },
            "agreement": {
                "agreement_score": agreement.get("agreement_score", 0.5),
                "agreement_type": agreement.get("agreement_type", "none"),
                "consensus_strength": agreement.get("consensus_strength", 0.5),
                "std_dev": agreement.get("std_dev", 0.0),
            },
            "trend_force": {
                "trend_force_score": trend_force.get("trend_force_score", 0.5),
                "force_type": trend_force.get("force_type", "none"),
                "strong_count": trend_force.get("strong_count", 0),
                "weak_count": trend_force.get("weak_count", 0),
                "avg_force": trend_force.get("avg_force", 0.5),
            },
        },
        "factor_weights": {
            "alignment": alignment_weight,
            "agreement": agreement_weight,
            "trend_force": trend_force_weight,
        },
        "factor_scores": {
            "alignment": round(alignment_score, 3),
            "agreement": round(agreement_score, 3),
            "trend_force": round(trend_force_score, 3),
        },
        "metadata": {
            "timeframe_count": num_timeframes,
            "timeframes": list(timeframe_strengths.keys()),
        },
    }


def create_multitimeframe_factor_score(
    analysis: Dict[str, object],
) -> FactorScore:
    """
    Create a FactorScore object from multi-timeframe analysis for system integration.
    
    Args:
        analysis: Output from analyze_multitimeframe_factors
        
    Returns:
        FactorScore instance for trading system integration
    """
    return FactorScore(
        factor_name="multitimeframe_alignment",
        score=float(analysis.get("final_score", 0.5)),
        weight=0.10,
        description=analysis.get("rationale", "Multi-timeframe alignment analysis"),
        emoji=analysis.get("emoji", "⚪"),
        metadata={
            "direction": analysis.get("direction", "neutral"),
            "confidence": analysis.get("confidence", 0),
            "per_timeframe_flags": analysis.get("per_timeframe_flags", {}),
            "components": analysis.get("components", {}),
            "factor_weights": analysis.get("factor_weights", {}),
            "factor_scores": analysis.get("factor_scores", {}),
            "metadata": analysis.get("metadata", {}),
        },
    )
