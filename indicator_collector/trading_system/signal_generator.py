"""Trading signal generator that combines analyzer outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
import math
import statistics


from .interfaces import (
    AnalyzerContext,
    FactorScore,
    JsonDict,
    SignalExplanation,
    TradingSignalPayload,
)
from .backtester import ParameterSet
from .technical_analysis import analyze_technical_factors
from .sentiment_analyzer import analyze_sentiment_factors
from .multitimeframe_analyzer import analyze_multitimeframe_factors, create_multitimeframe_factor_score
from .utils import clamp
from ..timeframes import Timeframe


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


def _create_volume_factor(
    context: AnalyzerContext,
    parameter_set: Optional[ParameterSet] = None,
) -> Optional[FactorScore]:
    """Create volume factor from context data and configurable parameters."""

    volume_analysis = context.volume_analysis or {}
    advanced_metrics = context.advanced_metrics or {}
    indicators = context.indicators or {}

    params = parameter_set.get_indicator_group("volume") if parameter_set else {}
    cvd_multiplier = float(params.get("cvd_atr_multiplier", 0.75))
    delta_threshold = float(params.get("delta_imbalance_threshold", 1.2))
    vpvr_threshold = float(params.get("vpvr_poc_share", 0.04))
    smart_money_multiplier = float(params.get("smart_money_multiplier", 1.5))

    component_weight_defaults = {
        "cvd": 0.3,
        "delta": 0.25,
        "vpvr": 0.2,
        "smart_money": 0.25,
    }
    component_weight_spec = params.get("component_weights", {})
    component_weights = {
        key: float(component_weight_spec.get(key, default))
        for key, default in component_weight_defaults.items()
    }
    total_component_weight = sum(component_weights.values()) or 1.0
    normalized_component_weights = {
        key: value / total_component_weight for key, value in component_weights.items()
    }

    context_metrics = volume_analysis.get("context", {})
    volume_ratio = float(context_metrics.get("volume_ratio", 1.0) or 1.0)
    volume_confidence = float(context_metrics.get("volume_confidence", 0.5) or 0.5)
    atr_value = float(indicators.get("atr") or context_metrics.get("atr") or 0.0)

    cvd = volume_analysis.get("cvd", {})
    cvd_change = float(cvd.get("change") or 0.0)
    if atr_value > 0 and cvd_multiplier > 0:
        cvd_normalized = clamp(cvd_change / (atr_value * cvd_multiplier), -2.0, 2.0)
    else:
        cvd_normalized = 0.0
    cvd_component = clamp(0.5 + cvd_normalized * 0.25, 0.0, 1.0)
    cvd_direction = "bullish" if cvd_component > 0.52 else "bearish" if cvd_component < 0.48 else "neutral"

    delta = volume_analysis.get("delta", {})
    delta_latest = float(delta.get("latest") or 0.0)
    delta_average = float(delta.get("average") or 0.0)
    denom = abs(delta_average) * delta_threshold if delta_threshold > 0 else abs(delta_average)
    if denom > 0:
        delta_normalized = clamp(delta_latest / denom, -2.0, 2.0)
    else:
        delta_normalized = 0.0
    delta_component = clamp(0.5 + delta_normalized * 0.25, 0.0, 1.0)
    delta_direction = "bullish" if delta_component > 0.52 else "bearish" if delta_component < 0.48 else "neutral"

    vpvr = volume_analysis.get("vpvr", {})
    total_volume = float(vpvr.get("total_volume") or 0.0)
    levels = vpvr.get("levels") or []
    poc_price = vpvr.get("poc")
    poc_share = 0.0
    if total_volume > 0:
        target_level = None
        if poc_price is not None:
            target_level = next((lvl for lvl in levels if lvl.get("price") == poc_price), None)
        if target_level is None and levels:
            target_level = max(levels, key=lambda lvl: float(lvl.get("volume") or 0.0))
        if target_level:
            poc_share = float(target_level.get("volume", 0.0)) / total_volume
    if vpvr_threshold > 0:
        vpvr_ratio = poc_share / vpvr_threshold
    else:
        vpvr_ratio = 0.0
    vpvr_component = clamp(0.5 + (clamp(vpvr_ratio, 0.0, 2.0) - 1.0) * 0.25, 0.0, 1.0)
    vpvr_direction = "bullish" if vpvr_component > 0.55 else "bearish" if vpvr_component < 0.45 else "neutral"

    smart_money_events = volume_analysis.get("smart_money") or []
    avg_volume = float(context_metrics.get("average_volume") or 0.0)
    qualified_buys = 0
    qualified_sells = 0
    for event in smart_money_events:
        event_volume = float(event.get("volume") or 0.0)
        if avg_volume > 0 and event_volume < avg_volume * smart_money_multiplier:
            continue
        direction = str(event.get("direction", "")).lower()
        if direction == "buy":
            qualified_buys += 1
        elif direction == "sell":
            qualified_sells += 1
    total_qualified = qualified_buys + qualified_sells
    if total_qualified > 0:
        smart_money_bias = (qualified_buys - qualified_sells) / total_qualified
    else:
        smart_money_bias = 0.0
    smart_money_component = clamp(0.5 + smart_money_bias * 0.5, 0.0, 1.0)
    smart_money_direction = "bullish" if smart_money_component > 0.55 else "bearish" if smart_money_component < 0.45 else "neutral"

    components = {
        "cvd": cvd_component,
        "delta": delta_component,
        "vpvr": vpvr_component,
        "smart_money": smart_money_component,
    }
    component_directions = {
        "cvd": cvd_direction,
        "delta": delta_direction,
        "vpvr": vpvr_direction,
        "smart_money": smart_money_direction,
    }

    weighted_score = sum(components[key] * normalized_component_weights[key] for key in components)
    ratio_adjustment = clamp((volume_ratio - 1.0) * 0.1, -0.1, 0.1)
    confidence_adjustment = (volume_confidence - 0.5) * 0.2
    final_score = clamp(weighted_score + ratio_adjustment + confidence_adjustment, 0.0, 1.0)

    if final_score >= 0.58:
        direction = "bullish"
        emoji = "🟢"
    elif final_score <= 0.42:
        direction = "bearish"
        emoji = "🔴"
    else:
        direction = "neutral"
        emoji = "⚪"

    component_confidences = [abs(value - 0.5) * 2 for value in components.values()]
    avg_component_confidence = clamp(
        statistics.fmean(component_confidences) if component_confidences else 0.0,
        0.0,
        1.0,
    )

    metadata = {
        "direction": direction,
        "volume_ratio": volume_ratio,
        "volume_confidence": volume_confidence,
        "components": components,
        "component_weights": normalized_component_weights,
        "component_directions": component_directions,
        "parameters": {
            "cvd_atr_multiplier": cvd_multiplier,
            "delta_imbalance_threshold": delta_threshold,
            "vpvr_poc_share": vpvr_threshold,
            "smart_money_multiplier": smart_money_multiplier,
        },
        "confidence": avg_component_confidence * 100.0,
        "atr": atr_value,
        "poc_share": poc_share,
        "qualified_smart_money": {
            "buy": qualified_buys,
            "sell": qualified_sells,
            "total": total_qualified,
        },
        "smart_money_raw": smart_money_events[:5],
        "advanced_metrics": advanced_metrics.get("smart_money_activity"),
    }

    description = (
        f"Volume composite {final_score:.2f} (ratio {volume_ratio:.2f}, confidence {volume_confidence:.2f})"
    )
    weight = (
        parameter_set.normalized_category_weights().get("volume", 0.20)
        if parameter_set
        else 0.20
    )

    return FactorScore(
        factor_name="volume_analysis",
        score=final_score,
        weight=weight,
        description=description,
        emoji=emoji,
        metadata=metadata,
    )


def _create_structure_factor(
    context: AnalyzerContext,
    parameter_set: Optional[ParameterSet] = None,
) -> Optional[FactorScore]:
    """Create market structure factor from context data and configurable parameters."""

    market_structure = context.market_structure or {}
    advanced_metrics = context.advanced_metrics or {}

    params = parameter_set.get_indicator_group("structure") if parameter_set else {}
    lookback = max(1, int(params.get("lookback", 24)))
    swing_window = max(1, int(params.get("swing_window", 5)))
    trend_window = max(1, int(params.get("trend_window", 12)))
    min_sequence = max(1, int(params.get("min_sequence", 5)))
    atr_distance = float(params.get("atr_distance", 1.0))

    component_weight_defaults = {
        "trend": 0.4,
        "structure": 0.4,
        "liquidity": 0.2,
    }
    component_weight_spec = params.get("component_weights", {})
    component_weights = {
        key: float(component_weight_spec.get(key, default))
        for key, default in component_weight_defaults.items()
    }
    total_component_weight = sum(component_weights.values()) or 1.0
    normalized_component_weights = {
        key: value / total_component_weight for key, value in component_weights.items()
    }

    structure_state = str(market_structure.get("structure_state", "neutral")).lower()
    structure_score = float(market_structure.get("structure_score", 0.5) or 0.5)
    liquidity_score = float(market_structure.get("liquidity_score", 0.5) or 0.5)
    sequence_length = int(market_structure.get("sequence_length", 0) or 0)
    sweep_distance = float(market_structure.get("liquidity_sweep_atr", 0.0) or 0.0)
    breadth_metrics = advanced_metrics.get("market_breadth", {})
    breadth_score = float(breadth_metrics.get("score", 0.5) or 0.5)

    trend_bias_map = {"bullish": 0.65, "bearish": 0.35}
    trend_bias = trend_bias_map.get(structure_state, 0.5)
    trend_ratio = clamp(trend_window / float(lookback), 0.0, 2.0)
    breadth_adjustment = (breadth_score - 0.5) * 0.3
    trend_component = clamp(trend_bias + (trend_ratio - 1.0) * 0.15 + breadth_adjustment, 0.0, 1.0)

    if sequence_length <= 0:
        sequence_length = swing_window
    sequence_ratio = clamp(sequence_length / float(max(min_sequence, 1)), 0.0, 2.0)
    structure_component = clamp(
        0.5 + (structure_score - 0.5) * (1.0 + (sequence_ratio - 1.0) * 0.5),
        0.0,
        1.0,
    )

    if sweep_distance <= 0:
        sweep_distance = atr_distance
    sweep_ratio = clamp(sweep_distance / float(max(atr_distance, 1e-6)), 0.0, 2.0)
    liquidity_component = clamp(
        liquidity_score + (0.5 - sweep_ratio * 0.25),
        0.0,
        1.0,
    )

    components = {
        "trend": trend_component,
        "structure": structure_component,
        "liquidity": liquidity_component,
    }

    final_score = sum(components[key] * normalized_component_weights[key] for key in components)

    if final_score >= 0.58:
        direction = "bullish"
        emoji = "🟢"
    elif final_score <= 0.42:
        direction = "bearish"
        emoji = "🔴"
    else:
        direction = "neutral"
        emoji = "⚪"

    component_confidences = [abs(value - 0.5) * 2 for value in components.values()]
    avg_component_confidence = clamp(
        statistics.fmean(component_confidences) if component_confidences else 0.0,
        0.0,
        1.0,
    )

    metadata = {
        "direction": direction,
        "structure_state": structure_state,
        "structure_score": structure_score,
        "liquidity_score": liquidity_score,
        "components": components,
        "component_weights": normalized_component_weights,
        "parameters": {
            "lookback": lookback,
            "swing_window": swing_window,
            "trend_window": trend_window,
            "min_sequence": min_sequence,
            "atr_distance": atr_distance,
        },
        "sequence_length": sequence_length,
        "sweep_distance_atr": sweep_distance,
        "breadth_score": breadth_score,
        "confidence": avg_component_confidence * 100.0,
    }

    description = (
        f"Market structure {structure_state or 'neutral'} (composite {final_score:.2f})"
    )
    weight = (
        parameter_set.normalized_category_weights().get("market_structure", 0.15)
        if parameter_set
        else 0.15
    )

    return FactorScore(
        factor_name="market_structure",
        score=final_score,
        weight=weight,
        description=description,
        emoji=emoji,
        metadata=metadata,
    )


def _create_composite_factor(
    factors: SignalFactors,
    parameter_set: Optional[ParameterSet] = None,
    composite_params: Optional[Dict[str, Any]] = None,
) -> Optional[FactorScore]:
    """Create composite factor by aggregating category scores with live weights."""

    composite_params = composite_params or {}
    buy_threshold = float(composite_params.get("buy_threshold", 0.6))
    sell_threshold = float(composite_params.get("sell_threshold", 0.4))
    confidence_floor = float(composite_params.get("confidence_floor", 0.3))
    confidence_ceiling = float(composite_params.get("confidence_ceiling", 0.9))
    min_confirmations = int(composite_params.get("min_confirmations", 3))

    weights_map = (
        parameter_set.normalized_category_weights()
        if parameter_set
        else {
            "technical": 0.25,
            "sentiment": 0.15,
            "multitimeframe": 0.10,
            "volume": 0.20,
            "market_structure": 0.15,
        }
    )

    category_factors = {
        "technical": factors.technical,
        "sentiment": factors.sentiment,
        "multitimeframe": factors.multitimeframe,
        "volume": factors.volume,
        "market_structure": factors.structure,
    }

    available_entries = {
        name: factor for name, factor in category_factors.items() if factor is not None
    }
    if not available_entries:
        return None

    weight_sum = sum(weights_map.get(name, 0.0) for name in available_entries)
    if weight_sum <= 0:
        weight_sum = len(available_entries)
        normalized_weights = {name: 1.0 / weight_sum for name in available_entries}
    else:
        normalized_weights = {
            name: weights_map.get(name, 0.0) / weight_sum
            for name in available_entries
        }

    contributions = {
        name: factor.score * normalized_weights[name]
        for name, factor in available_entries.items()
    }
    composite_score = sum(contributions.values())

    bullish_confirmations = 0
    bearish_confirmations = 0
    neutral_confirmations = 0
    per_category_direction: Dict[str, str] = {}
    for name, factor in available_entries.items():
        direction = factor.metadata.get("direction")
        if not direction:
            if factor.score >= 0.55:
                direction = "bullish"
            elif factor.score <= 0.45:
                direction = "bearish"
            else:
                direction = "neutral"
        per_category_direction[name] = direction
        if direction == "bullish":
            bullish_confirmations += 1
        elif direction == "bearish":
            bearish_confirmations += 1
        else:
            neutral_confirmations += 1

    if composite_score >= buy_threshold:
        direction = "bullish"
        emoji = "🟢"
        confirmations = bullish_confirmations
    elif composite_score <= sell_threshold:
        direction = "bearish"
        emoji = "🔴"
        confirmations = bearish_confirmations
    else:
        direction = "neutral"
        emoji = "⚪"
        confirmations = max(bullish_confirmations, bearish_confirmations)

    if confidence_ceiling <= confidence_floor:
        confidence_ceiling = confidence_floor + 1e-6
    normalized_confidence = clamp(
        (composite_score - confidence_floor) / (confidence_ceiling - confidence_floor),
        0.0,
        1.0,
    )

    metadata = {
        "direction": direction,
        "confirmations": confirmations,
        "details": {
            name: {
                "score": factor.score,
                "weight": normalized_weights[name],
                "contribution": contributions[name],
                "direction": per_category_direction[name],
            }
            for name, factor in available_entries.items()
        },
        "parameters": {
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "confidence_floor": confidence_floor,
            "confidence_ceiling": confidence_ceiling,
            "min_confirmations": min_confirmations,
        },
        "normalized_confidence": normalized_confidence,
        "required_confirmations_met": confirmations >= min_confirmations,
    }

    description = f"Composite confirmation {composite_score:.2f} with {confirmations} confirmations"
    weight = (
        parameter_set.normalized_category_weights().get("composite", 0.15)
        if parameter_set
        else 0.15
    )

    return FactorScore(
        factor_name="composite_analysis",
        score=composite_score,
        weight=weight,
        description=description,
        emoji=emoji,
        metadata=metadata,
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
    _ = (factors, buy_threshold, sell_threshold)
    distance = abs(final_score - 0.5) * 2.0
    distance = clamp(distance, 0.0, 1.0)
    confidence_value = round(1 + 9 * distance)

    if cancellation_triggers:
        confidence_value = max(1, confidence_value - len(cancellation_triggers))

    normalized_confidence = confidence_value / 10.0
    return confidence_value, normalized_confidence


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
        primary_reason = f"Bullish composite score {final_score:.2f} with confidence {confidence}/10"
    elif signal_type == "SELL":
        primary_reason = f"Bearish composite score {final_score:.2f} with confidence {confidence}/10"
    else:
        primary_reason = f"Composite score {final_score:.2f} within neutral zone"
    
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
    market_context = "Signal based on composite score"
    
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
    config: Optional[SignalConfig] = None,
    indicator_params: Optional[Dict[str, Any]] = None,
    parameter_set: Optional[ParameterSet] = None,
) -> TradingSignalPayload:
    """Generate a comprehensive trading signal combining all analyzer outputs."""

    if config is None:
        config = SignalConfig()

    indicator_overrides = dict(indicator_params or {})
    base_category_weights = {
        "technical": config.technical_weight,
        "sentiment": config.sentiment_weight,
        "multitimeframe": config.multitimeframe_weight,
        "volume": config.volume_weight,
        "market_structure": config.structure_weight,
        "composite": config.composite_weight,
    }

    timeframe_hint = getattr(context, "timeframe", None)
    if not timeframe_hint and indicator_overrides:
        timeframe_hint = indicator_overrides.get("timeframe")
    try:
        normalized_timeframe = (
            Timeframe.from_value(timeframe_hint).value if timeframe_hint else Timeframe.H1.value
        )
    except Exception:
        normalized_timeframe = Timeframe.H1.value

    active_parameter_set = parameter_set
    if active_parameter_set is None:
        active_parameter_set = ParameterSet(
            weights=dict(base_category_weights),
            indicator_params=indicator_overrides,
            category_weights=dict(base_category_weights),
            timeframe=normalized_timeframe,
            signal_thresholds={
                "buy": float(config.buy_threshold),
                "sell": float(config.sell_threshold),
            },
        )

    resolved_indicator_params = active_parameter_set.indicator_params
    category_weights = active_parameter_set.normalized_category_weights()
    parameter_set = active_parameter_set

    factors = SignalFactors()

    technical_analysis_summary: Optional[Any] = None
    technical_analysis_snapshot: Optional[Dict[str, Any]] = None
    try:
        technical_analysis_summary = analyze_technical_factors(
            context,
            indicator_params=resolved_indicator_params,
        )
    except Exception:
        technical_analysis_summary = None

    if isinstance(technical_analysis_summary, dict) and technical_analysis_summary:
        technical_analysis_snapshot = technical_analysis_summary
        tech_score = float(technical_analysis_summary.get("final_score", 0.5))
        tech_direction = str(technical_analysis_summary.get("direction", "neutral"))
        tech_confidence = float(technical_analysis_summary.get("confidence", 0.0))
        tech_rationale = technical_analysis_summary.get("rationale") or "Technical analysis summary unavailable."
        factor_scores = technical_analysis_summary.get("factor_scores", {})
        factor_metadata = technical_analysis_summary.get("metadata", {})
        emoji_map = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
        factors.technical = FactorScore(
            factor_name="technical_analysis",
            score=clamp(tech_score, 0.0, 1.0),
            weight=category_weights.get("technical", config.technical_weight),
            description=tech_rationale,
            emoji=emoji_map.get(tech_direction, "⚪"),
            metadata={
                "direction": tech_direction,
                "confidence": tech_confidence,
                "factor_scores": factor_scores,
                "factor_weights": technical_analysis_summary.get("factor_weights", {}),
                "components": technical_analysis_summary.get("components", {}),
                "analysis_metadata": factor_metadata,
                "sub_factors": len(factor_scores),
            },
        )
    elif isinstance(technical_analysis_summary, (list, tuple)) and technical_analysis_summary:
        technical_factors: List[FactorScore] = []
        for item in technical_analysis_summary:
            if isinstance(item, FactorScore):
                technical_factors.append(item)
            elif isinstance(item, dict):
                direction = str(item.get("direction", item.get("metadata", {}).get("direction", "neutral")))
                emoji_map = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
                technical_factors.append(
                    FactorScore(
                        factor_name=item.get("factor_name", "technical_analysis"),
                        score=float(item.get("score", 0.5)),
                        weight=float(item.get("weight", 1.0)),
                        description=item.get("description"),
                        emoji=item.get("emoji") or emoji_map.get(direction, "⚪"),
                        metadata=item.get("metadata", {}),
                    )
                )
        if technical_factors:
            total_weight = sum(f.weight for f in technical_factors) or 1.0
            tech_score = sum(f.score * f.weight for f in technical_factors) / total_weight
            primary = technical_factors[0]
            direction = primary.metadata.get("direction", "neutral")
            factors.technical = FactorScore(
                factor_name="technical_analysis",
                score=clamp(tech_score, 0.0, 1.0),
                weight=category_weights.get("technical", config.technical_weight),
                description=primary.description,
                emoji=primary.emoji,
                metadata={
                    "direction": direction,
                    "sub_factors": len(technical_factors),
                },
            )
    else:
        factors.technical = None

    try:
        sentiment_payload = analyze_sentiment_factors(context)
        if sentiment_payload.factors:
            base = sentiment_payload.factors[0]
            factors.sentiment = FactorScore(
                factor_name=base.factor_name,
                score=base.score,
                weight=category_weights.get("sentiment", config.sentiment_weight),
                description=base.description,
                emoji=base.emoji,
                metadata=dict(base.metadata),
            )
    except Exception:
        factors.sentiment = None

    try:
        mt_params = parameter_set.get_indicator_group("multitimeframe") if parameter_set else None
        mt_analysis = analyze_multitimeframe_factors(
            context,
            analysis_params=mt_params,
        )
        mt_factor_scores: List[FactorScore] = []
        if hasattr(mt_analysis, "factors"):
            mt_factor_scores = list(getattr(mt_analysis, "factors"))  # type: ignore[attr-defined]
        elif isinstance(mt_analysis, dict):
            mt_factor_scores = [create_multitimeframe_factor_score(mt_analysis)]
        if mt_factor_scores:
            base = mt_factor_scores[0]
            metadata = dict(base.metadata)
            if isinstance(mt_analysis, dict):
                metadata.setdefault("analysis_summary", mt_analysis)
            factors.multitimeframe = FactorScore(
                factor_name=base.factor_name,
                score=base.score,
                weight=category_weights.get("multitimeframe", config.multitimeframe_weight),
                description=base.description,
                emoji=base.emoji,
                metadata=metadata,
            )
    except Exception:
        factors.multitimeframe = None

    volume_factor = _create_volume_factor(context, parameter_set)
    if volume_factor:
        if not parameter_set:
            volume_factor.weight = category_weights.get("volume", config.volume_weight)
        factors.volume = volume_factor

    structure_factor = _create_structure_factor(context, parameter_set)
    if structure_factor:
        if not parameter_set:
            structure_factor.weight = category_weights.get("market_structure", config.structure_weight)
        factors.structure = structure_factor

    composite_params = parameter_set.get_indicator_group("composite") if parameter_set else None
    composite_factor = _create_composite_factor(factors, parameter_set, composite_params)
    if composite_factor:
        if not parameter_set:
            composite_factor.weight = category_weights.get("composite", config.composite_weight)
        factors.composite = composite_factor

    composite_categories = {
        "technical": factors.technical,
        "market_structure": factors.structure,
        "volume": factors.volume,
        "sentiment": factors.sentiment,
        "multitimeframe": factors.multitimeframe,
    }

    composite_weights_raw = {key: category_weights.get(key, 0.0) for key in composite_categories}
    composite_weight_total = sum(composite_weights_raw.values())
    if composite_weight_total <= 0:
        composite_weights = {key: 1.0 / len(composite_categories) for key in composite_categories}
    else:
        composite_weights = {
            key: composite_weights_raw[key] / composite_weight_total for key in composite_categories
        }

    composite_components: Dict[str, Dict[str, Any]] = {}
    composite_score = 0.0
    for key, factor in composite_categories.items():
        score = factor.score if factor else None
        weight = composite_weights[key]
        contribution = weight * score if score is not None else 0.0
        composite_components[key] = {
            "score": score,
            "weight": weight,
            "contribution": contribution,
        }
        composite_score += contribution

    final_score = clamp(composite_score, 0.0, 1.0)
    available_factors = factors.get_available_factors()

    vix_value = None
    if context.extras and "market_context" in context.extras:
        market_context = context.extras["market_context"]
        if isinstance(market_context, dict) and "vix" in market_context:
            vix_value = market_context["vix"]

    buy_threshold, sell_threshold, min_confidence = config.get_adapted_thresholds(vix_value)

    cancellation_triggers = _check_cancellation_triggers(context, factors)
    if cancellation_triggers:
        signal_type = "HOLD"
    elif final_score >= buy_threshold:
        signal_type = "BUY"
    elif final_score <= sell_threshold:
        signal_type = "SELL"
    else:
        signal_type = "HOLD"

    confidence_int, confidence_float = _calculate_confidence(
        final_score, factors, buy_threshold, sell_threshold, cancellation_triggers
    )

    if confidence_float < min_confidence:
        signal_type = "HOLD"

    explanation = _generate_explanation(
        signal_type, final_score, factors, confidence_int, cancellation_triggers
    )

    analysis_debug = {
        "parameter_hash": parameter_set.params_hash() if parameter_set else None,
        "category_weights": category_weights,
        "factor_breakdown": {
            factor.factor_name: {
                "score": factor.score,
                "weight": factor.weight,
                "direction": factor.metadata.get("direction"),
                "metadata": dict(factor.metadata),
            }
            for factor in available_factors
        },
        "final_score": final_score,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "composite_components": composite_components,
        "composite_weights": composite_weights,
    }

    payload_metadata = {
        "final_score": final_score,
        "composite_score": final_score,
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "min_confidence": min_confidence,
        "vix_value": vix_value,
        "cancellation_triggers": cancellation_triggers,
        "available_factors": factors.count_available_factors(),
        "category_weights": category_weights,
        "technical_analysis_summary": technical_analysis_snapshot,
        "indicator_params": resolved_indicator_params,
        "analysis_debug": analysis_debug,
        "parameter_debug_enabled": parameter_set.debug_enabled if parameter_set else False,
        "parameter_hash": parameter_set.params_hash() if parameter_set else None,
        "composite_components": composite_components,
        "composite_weights": composite_weights,
    }

    payload = TradingSignalPayload(
        signal_type=signal_type,
        confidence=confidence_float,
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=available_factors,
        explanation=explanation,
        metadata=payload_metadata,
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
    
    def analyze(
        self,
        context: AnalyzerContext,
        config: Optional[SignalConfig] = None,
        indicator_params: Optional[Dict[str, Any]] = None,
        parameter_set: Optional[ParameterSet] = None,
    ) -> TradingSignalPayload:
        """Analyze trading context and generate signal.

        Args:
            context: AnalyzerContext with market data
            config: Optional SignalConfig to override instance config
            indicator_params: Optional indicator parameter overrides
            parameter_set: Optional ParameterSet with analyzer configuration

        Returns:
            TradingSignalPayload with generated signal
        """
        signal_config = config or self.config
        return generate_trading_signal(
            context,
            signal_config,
            indicator_params=indicator_params,
            parameter_set=parameter_set,
        )
