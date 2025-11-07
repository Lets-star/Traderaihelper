"""Generate explicit JSON signals from trading analysis results.

This module provides functions to convert trading system analysis results
into the standardized JSON signal format required by the web UI.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .interfaces import TradingSignalPayload
from .signal_schema import TradingSignalSchema, validate_signal_json
from .position_manager import PositionPlan
from ..timeframes import Timeframe

logger = logging.getLogger(__name__)


def generate_signals(
    normalized_payload: Dict[str, Any], 
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate explicit JSON signals from normalized payload and parameters.
    
    Args:
        normalized_payload: Normalized trading payload containing analysis results
        params: Optional parameter set (can be extracted from payload if not provided)
        
    Returns:
        Dictionary containing explicit trading signals in the required JSON format
        
    Raises:
        ValueError: If signal generation fails or validation doesn't pass
    """
    try:
        # Extract signal information from payload
        signal_type = normalized_payload.get("signal_type", "HOLD")
        confidence = normalized_payload.get("confidence", 0.5)
        
        # Extract position plan information
        position_plan = normalized_payload.get("position_plan") or {}
        if hasattr(position_plan, 'to_dict'):
            position_plan = position_plan.to_dict()
        
        entry_price = position_plan.get("entry_price") if position_plan else None
        stop_loss = position_plan.get("stop_loss") if position_plan else None
        take_profits = position_plan.get("take_profits", {}) if position_plan else {}
        position_size = position_plan.get("position_size_pct", 2.0) if position_plan else 2.0
        
        # Extract holding period and rationale
        holding_period = normalized_payload.get("holding_period", "medium")
        explanation = normalized_payload.get("explanation", {})
        rationale = _extract_rationale(explanation)
        
        # Extract weights
        weights = normalized_payload.get("weights", {})
        if not weights:
            weights = _extract_default_weights(normalized_payload)
        
        # Extract timeframe
        timeframe = normalized_payload.get("timeframe", "1h")
        
        # Generate entry prices
        entries = _generate_entries(entry_price, signal_type)
        
        # Generate take profit structure
        tp_structure = _generate_take_profits(take_profits, entry_price, signal_type)
        
        # Generate cancel conditions
        cancel_conditions = _generate_cancel_conditions(normalized_payload)
        
        # Build the signal
        signal_data = {
            "signal": signal_type,
            "confidence": _convert_confidence_to_scale(confidence),
            "entries": entries,
            "stop_loss": stop_loss or _calculate_default_stop_loss(entry_price, signal_type),
            "take_profits": tp_structure,
            "position_size_pct": position_size,
            "holding_period": holding_period,
            "rationale": rationale,
            "cancel_conditions": cancel_conditions,
            "weights": weights,
            "timeframe": timeframe
        }
        
        # Validate against schema
        validated_signal = validate_signal_json(signal_data)
        
        return validated_signal.dict()
        
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise ValueError(f"Failed to generate explicit JSON signals: {e}")


def generate_signals_from_payload(
    signal_payload: TradingSignalPayload,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate explicit JSON signals from TradingSignalPayload.
    
    Args:
        signal_payload: TradingSignalPayload object
        params: Optional parameter set
        
    Returns:
        Dictionary containing explicit trading signals
    """
    # Convert payload to dictionary
    payload_dict = signal_payload.to_dict()
    
    return generate_signals(payload_dict, params)


def _extract_rationale(explanation: Dict[str, Any]) -> List[str]:
    """Extract rationale points from explanation data."""
    rationale = []
    
    if not explanation:
        return ["Signal generated based on technical analysis"]
    
    # Extract from analysis summary
    summary = explanation.get("summary", "")
    if summary:
        rationale.append(f"Analysis: {summary}")
    
    # Extract from factor explanations
    factors = explanation.get("factor_explanations", [])
    for factor in factors[:3]:  # Limit to top 3 factors
        factor_name = factor.get("factor", "Unknown")
        factor_score = factor.get("score", 0)
        factor_reason = factor.get("reason", "")
        
        if factor_reason:
            rationale.append(f"{factor_name} ({factor_score:.2f}): {factor_reason}")
    
    # Extract from overall explanation
    overall = explanation.get("overall_explanation", "")
    if overall and overall not in summary:
        rationale.append(f"Overall: {overall}")
    
    # If still empty, provide default
    if not rationale:
        rationale.append("Signal generated based on technical analysis")
    
    return rationale[:5]  # Limit to 5 rationale points


def _extract_default_weights(normalized_payload: Dict[str, Any]) -> Dict[str, float]:
    """Extract or generate default weights from payload."""
    # Check if weights are in factors
    factors = normalized_payload.get("factors", [])
    if factors:
        weights = {}
        total_weight = 0.0
        
        for factor in factors:
            factor_name = factor.get("factor_name", factor.get("factor", "unknown"))
            factor_weight = factor.get("weight", 0.25)
            weights[factor_name] = factor_weight
            total_weight += factor_weight
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
            return weights
    
    # Check if weights are directly provided
    direct_weights = normalized_payload.get("weights")
    if direct_weights and isinstance(direct_weights, dict):
        total_weight = sum(direct_weights.values())
        if total_weight > 0:
            # Normalize to sum to 1.0
            return {k: v / total_weight for k, v in direct_weights.items()}
    
    # Return default balanced weights
    return {
        "technical": 0.25,
        "volume": 0.25,
        "sentiment": 0.25,
        "market_structure": 0.25,
    }


def _generate_entries(entry_price: Optional[float], signal_type: str) -> List[float]:
    """Generate entry price levels."""
    if entry_price is None:
        return [50000.0]  # Default fallback
    
    # For now, return single entry price
    # Could be extended to generate multiple entry levels
    return [float(entry_price)]


def _generate_take_profits(
    take_profits: Dict[str, Any], 
    entry_price: Optional[float], 
    signal_type: str
) -> Dict[str, float]:
    """Generate take profit structure."""
    if isinstance(take_profits, dict) and take_profits:
        # Extract existing take profits
        tp_structure = {}
        
        # Handle different TP formats
        if "tp1" in take_profits:
            tp_structure["tp1"] = float(take_profits["tp1"])
        if "tp2" in take_profits:
            tp_structure["tp2"] = float(take_profits["tp2"])
        if "tp3" in take_profits:
            tp_structure["tp3"] = float(take_profits["tp3"])
        
        # If we have at least one TP, fill missing ones
        if tp_structure and entry_price:
            tp_structure = _fill_missing_take_profits(tp_structure, entry_price, signal_type)
        
        return tp_structure
    
    # Generate default take profits if entry price is available
    if entry_price:
        return _generate_default_take_profits(entry_price, signal_type)
    
    # Fallback defaults
    return {
        "tp1": 51000.0,
        "tp2": 52000.0,
        "tp3": 53000.0,
    }


def _fill_missing_take_profits(
    existing_tps: Dict[str, float], 
    entry_price: float, 
    signal_type: str
) -> Dict[str, float]:
    """Fill missing take profit levels based on existing ones."""
    tp_structure = existing_tps.copy()
    
    # Calculate default percentages based on signal type
    if signal_type == "BUY":
        tp1_pct = 0.01  # 1%
        tp2_pct = 0.02  # 2%
        tp3_pct = 0.03  # 3%
    elif signal_type == "SELL":
        tp1_pct = -0.01  # -1%
        tp2_pct = -0.02  # -2%
        tp3_pct = -0.03  # -3%
    else:  # HOLD
        tp1_pct = 0.01
        tp2_pct = 0.02
        tp3_pct = 0.03
    
    # Fill missing TPs
    if "tp1" not in tp_structure:
        tp_structure["tp1"] = entry_price * (1 + tp1_pct)
    if "tp2" not in tp_structure:
        tp_structure["tp2"] = entry_price * (1 + tp2_pct)
    if "tp3" not in tp_structure:
        tp_structure["tp3"] = entry_price * (1 + tp3_pct)
    
    return tp_structure


def _generate_default_take_profits(entry_price: float, signal_type: str) -> Dict[str, float]:
    """Generate default take profit levels."""
    if signal_type == "BUY":
        return {
            "tp1": entry_price * 1.01,   # +1%
            "tp2": entry_price * 1.02,   # +2%
            "tp3": entry_price * 1.03,   # +3%
        }
    elif signal_type == "SELL":
        return {
            "tp1": entry_price * 0.99,   # -1%
            "tp2": entry_price * 0.98,   # -2%
            "tp3": entry_price * 0.97,   # -3%
        }
    else:  # HOLD
        return {
            "tp1": entry_price * 1.01,
            "tp2": entry_price * 1.02,
            "tp3": entry_price * 1.03,
        }


def _calculate_default_stop_loss(entry_price: Optional[float], signal_type: str) -> float:
    """Calculate default stop loss if not provided."""
    if entry_price is None:
        return 49000.0  # Default fallback
    
    if signal_type == "BUY":
        return entry_price * 0.98  # -2%
    elif signal_type == "SELL":
        return entry_price * 1.02  # +2%
    else:  # HOLD
        return entry_price * 0.98  # Conservative -2%


def _convert_confidence_to_scale(confidence: float) -> int:
    """Convert confidence from 0-1 scale to 1-10 scale."""
    confidence_1_10 = int(round(confidence * 10))
    return max(1, min(10, confidence_1_10))


def _generate_cancel_conditions(normalized_payload: Dict[str, Any]) -> List[str]:
    """Generate cancel conditions based on payload data."""
    conditions = []
    
    # Add condition based on confidence
    confidence = normalized_payload.get("confidence", 0.5)
    if confidence < 0.3:
        conditions.append("Low confidence - cancel if market conditions change")
    
    # Add condition based on signal strength
    factors = normalized_payload.get("factors", [])
    weak_factors = [f for f in factors if f.get("score", 0) < 0.3]
    if len(weak_factors) > len(factors) / 2:
        conditions.append("Multiple weak factors - monitor closely")
    
    # Add timeframe-specific conditions
    timeframe = normalized_payload.get("timeframe", "1h")
    if timeframe in ["1m", "5m"]:
        conditions.append("High volatility timeframe - be ready to exit quickly")
    
    # Add default condition if no specific ones
    if not conditions:
        conditions.append("Monitor price action and volume")
    
    return conditions[:3]  # Limit to 3 conditions