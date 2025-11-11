"""Generate explicit JSON signals from trading analysis results.

This module provides functions to convert trading system analysis results
into the standardized JSON signal format required by the web UI.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .backtester import DEFAULT_WEIGHTS, indicator_defaults_for
from .interfaces import TradingSignalPayload
from .signal_schema import validate_signal_json
from .utils import clamp, safe_div
from ..timeframes import Timeframe

logger = logging.getLogger(__name__)

_ACTIONABLE_SIGNALS = {"BUY", "SELL"}
_COMPOSITE_CATEGORIES = (
    "technical",
    "market_structure",
    "volume",
    "sentiment",
    "multitimeframe",
)
_FACTOR_CATEGORY_MAP = {
    "technical_analysis": "technical",
    "sentiment": "sentiment",
    "multitimeframe_alignment": "multitimeframe",
    "volume_analysis": "volume",
    "market_structure": "market_structure",
    "composite_analysis": "composite",
}

_DEFAULT_RISK_PER_TRADE = 0.02
_DEBUG_ENABLED = os.getenv("GENERATE_SIGNALS_DEBUG", "0").lower() in {"1", "true", "yes", "on"}


@dataclass
class PlanDetails:
    """Normalized representation of a position plan."""

    valid: bool
    entries: List[float] = field(default_factory=list)
    stop_loss: Optional[float] = None
    take_profits: Dict[str, float] = field(default_factory=dict)
    position_size_pct: Optional[float] = None
    holding_period: str = "medium"
    holding_horizon_bars: Optional[int] = None
    reason: Optional[str] = None
    sanitized_plan: Optional[Dict[str, Any]] = None
    entry_zone: Optional[Dict[str, float]] = None


def generate_signals(
    normalized_payload: Union[TradingSignalPayload, Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate explicit JSON signals from normalized payload and parameters."""

    try:
        payload = _normalize_payload(normalized_payload)
        params_dict = _normalize_params(params)

        original_signal_type = str(payload.get("signal_type", "HOLD")).upper()
        factors = _normalize_factors(payload.get("factors"))
        explanation = payload.get("explanation") or {}
        metadata = payload.get("metadata") or {}
        position_plan = payload.get("position_plan") or {}
        timeframe = _infer_timeframe(payload, metadata, params_dict)
        _ensure_indicator_params(params_dict, timeframe)

        weights = _extract_weights(payload, metadata, params_dict)
        composite_context = _compute_composite_context(factors, weights)
        composite_score = composite_context["score"]

        buy_threshold, sell_threshold = _resolve_composite_thresholds(metadata, params_dict)
        computed_signal = _signal_from_composite(composite_score, buy_threshold, sell_threshold)

        actionable = computed_signal in _ACTIONABLE_SIGNALS
        hold_reasons: List[str] = []

        if computed_signal == "HOLD":
            hold_reasons.append(
                f"Composite score {composite_score:.2f} between thresholds "
                f"(buy ≥ {buy_threshold:.2f}, sell ≤ {sell_threshold:.2f})."
            )

        plan_details = PlanDetails(valid=False, reason="Composite signal not actionable")
        if actionable:
            plan_details = _build_plan_details(
                position_plan,
                computed_signal,
                params_dict,
                metadata,
            )
            if not plan_details.valid:
                actionable = False
                if plan_details.reason:
                    hold_reasons.append(plan_details.reason)
                else:
                    hold_reasons.append("Position plan missing required risk parameters.")

        signal_type = computed_signal if actionable else "HOLD"

        confidence = _convert_confidence(composite_score, actionable)

        if not actionable:
            entries = []
            stop_loss = None
            take_profits = {}
            position_size_pct = None
            holding_period = plan_details.holding_period or _classify_holding_period(None, timeframe)
            plan_output = plan_details.sanitized_plan if plan_details.sanitized_plan else None
        else:
            entries = plan_details.entries
            stop_loss = plan_details.stop_loss
            take_profits = plan_details.take_profits
            position_size_pct = plan_details.position_size_pct
            holding_period = plan_details.holding_period
            plan_output = plan_details.sanitized_plan

        rationale = _build_rationale(
            explanation,
            actionable,
            hold_reasons,
            composite_context,
        )

        cancel_conditions = _build_cancel_conditions(
            explanation,
            metadata.get("cancellation_triggers", []),
            plan_details,
            actionable,
            hold_reasons,
        )

        metadata_block = _build_metadata_block(
            payload,
            timeframe,
            composite_context,
            plan_details,
            actionable,
            buy_threshold,
            sell_threshold,
            composite_score,
            original_signal_type,
        )

        result: Dict[str, Any] = {
            "signal": signal_type,
            "confidence": confidence,
            "entries": entries,
            "stop_loss": stop_loss,
            "take_profits": take_profits,
            "position_size_pct": position_size_pct,
            "holding_period": holding_period,
            "rationale": rationale,
            "cancel_conditions": cancel_conditions,
            "weights": weights,
            "timeframe": timeframe,
            "factors": factors or None,
            "position_plan": plan_output,
            "explanation": explanation or None,
            "metadata": metadata_block,
            "holding_horizon_bars": plan_details.holding_horizon_bars,
            "cancellation_reasons": hold_reasons or metadata.get("cancellation_triggers"),
        }

        if _DEBUG_ENABLED:
            result["debug"] = {
                "actionable": actionable,
                "composite_score": composite_score,
                "computed_signal": computed_signal,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "composite_contributions": composite_context.get("contributions"),
                "composite_weights": composite_context.get("weights"),
                "missing_categories": composite_context.get("missing_categories"),
                "original_signal_type": original_signal_type,
                "reasons": hold_reasons,
            }

        validated = validate_signal_json(result)
        return validated.model_dump()

    except NameError as exc:  # pragma: no cover - diagnostic clarity
        missing = getattr(exc, "name", None)
        details = str(exc)
        if missing:
            message = f"generate_signals failed due to undefined symbol '{missing}'."
        else:
            message = f"generate_signals failed due to undefined symbol: {details}."
        logger.exception(message)
        raise ValueError(message) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("generate_signals encountered an unexpected error")
        raise ValueError(f"Failed to generate explicit JSON signals: {exc}") from exc


def generate_signals_from_payload(
    signal_payload: TradingSignalPayload,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate explicit JSON signals from TradingSignalPayload."""
    return generate_signals(signal_payload.to_dict(), params)


def _normalize_payload(payload: Union[TradingSignalPayload, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(payload, TradingSignalPayload):
        return payload.to_dict()
    if isinstance(payload, dict):
        return dict(payload)
    raise TypeError(f"Unsupported payload type: {type(payload)!r}")


def _normalize_params(params: Optional[Union[Dict[str, Any], Any]]) -> Dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    if hasattr(params, "to_dict") and callable(params.to_dict):
        try:
            return dict(params.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    normalized: Dict[str, Any] = {}
    for key in (
        "weights",
        "indicator_params",
        "timeframe",
        "stop_loss_pct",
        "take_profit_pct",
        "max_position_size_pct",
        "confirmation_threshold",
        "max_risk_per_trade_pct",
        "account_balance",
    ):
        if hasattr(params, key):
            normalized[key] = getattr(params, key)
    return normalized


def _merge_indicator_params(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_indicator_params(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure_indicator_params(params: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
    defaults = indicator_defaults_for(timeframe)
    indicator_params = params.get("indicator_params")
    if not isinstance(indicator_params, dict) or not indicator_params:
        if indicator_params not in (None, {}):
            logger.warning(
                "Invalid indicator_params provided (%s); falling back to defaults for timeframe %s.",
                type(indicator_params).__name__,
                timeframe,
            )
        params["indicator_params"] = defaults
        return defaults
    merged = _merge_indicator_params(defaults, indicator_params)
    params["indicator_params"] = merged
    return merged


def _infer_timeframe(
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> str:
    timeframe = payload.get("timeframe") or metadata.get("timeframe_used")
    if not timeframe:
        timeframe = params.get("timeframe")
    if timeframe:
        return str(timeframe)
    latest = metadata or {}
    return str(latest.get("timeframe", "1h"))


def _normalize_factors(factors_input: Optional[List[Any]]) -> List[Dict[str, Any]]:
    factors: List[Dict[str, Any]] = []
    if not factors_input:
        return factors
    for factor in factors_input:
        if factor is None:
            continue
        if isinstance(factor, dict):
            metadata = factor.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            factors.append(
                {
                    "factor_name": factor.get("factor_name") or factor.get("factor"),
                    "score": _safe_float(factor.get("score")),
                    "weight": _safe_float(factor.get("weight"), default=0.0),
                    "description": factor.get("description"),
                    "emoji": factor.get("emoji"),
                    "metadata": metadata,
                }
            )
        elif hasattr(factor, "to_dict"):
            try:
                factors.append(factor.to_dict())
            except Exception:  # pragma: no cover - defensive
                continue
    return factors





def _build_plan_details(
    position_plan: Dict[str, Any],
    signal_type: str,
    params: Dict[str, Any],
    metadata: Dict[str, Any],
) -> PlanDetails:
    if not isinstance(position_plan, dict) or not position_plan:
        return PlanDetails(valid=False, reason="Position plan unavailable")

    try:
        entry_price = _safe_float(position_plan.get("entry_price"))
        stop_loss = _safe_float(position_plan.get("stop_loss"))
    except (TypeError, ValueError):
        return PlanDetails(valid=False, reason="Invalid entry or stop loss values")

    if entry_price is None or stop_loss is None or entry_price <= 0 or stop_loss <= 0:
        return PlanDetails(valid=False, reason="Entry or stop loss missing")

    if math.isclose(entry_price, stop_loss):
        return PlanDetails(valid=False, reason="Entry and stop loss are identical")

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return PlanDetails(valid=False, reason="Risk distance between entry and stop is zero")

    raw_tp_levels = position_plan.get("take_profit_levels") or []
    plan_metadata = position_plan.get("metadata") or {}
    tp_multipliers = plan_metadata.get("tp_sl_multipliers") or {}

    tp_levels = _sanitize_tp_levels(entry_price, stop_loss, raw_tp_levels, signal_type)
    if len(tp_levels) < 3:
        tp_levels = _compute_tp_levels(entry_price, stop_loss, signal_type, tp_multipliers)

    position_size_usd = _safe_float(position_plan.get("position_size_usd"))
    sizing_factors = plan_metadata.get("sizing_factors", {})
    risk_amount_usd = _safe_float(sizing_factors.get("risk_amount_usd"))

    risk_pct = _safe_float(
        params.get("max_risk_per_trade_pct")
        or params.get("position_config", {}).get("max_risk_per_trade_pct")
        or metadata.get("position_config", {}).get("max_risk_per_trade_pct")
    )
    if risk_pct is None:
        risk_pct = _DEFAULT_RISK_PER_TRADE

    account_balance = _safe_float(
        params.get("account_balance")
        or metadata.get("account_balance")
        or (risk_amount_usd / risk_pct if risk_amount_usd and risk_pct else None)
    )

    max_position_pct = _safe_float(
        params.get("max_position_size_pct")
        or params.get("position_config", {}).get("max_position_size_pct")
        or metadata.get("position_config", {}).get("max_position_size_pct")
    )

    position_size_pct: Optional[float] = None
    if account_balance and position_size_usd:
        try:
            position_size_pct = min(100.0, max(0.0, (position_size_usd / account_balance) * 100.0))
            position_size_pct = round(position_size_pct, 2)
        except ZeroDivisionError:  # pragma: no cover - defensive
            position_size_pct = None

    if position_size_pct is None and position_size_usd and max_position_pct:
        pct_fraction: Optional[float]
        if max_position_pct > 1:
            pct_fraction = max_position_pct / 100.0
        else:
            pct_fraction = max_position_pct
        if pct_fraction and pct_fraction > 0:
            if account_balance is None:
                account_balance = position_size_usd / pct_fraction
            position_size_pct = round(min(100.0, pct_fraction * 100.0), 2)

    holding_horizon_bars = plan_metadata.get("holding_horizon_bars")
    holding_period = _classify_holding_period(holding_horizon_bars, position_plan.get("timeframe") or metadata.get("timeframe_used"))

    entry_zone = _compute_entry_zone(entry_price, plan_metadata.get("atr"), signal_type)

    sanitized_plan: Dict[str, Any] = {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit_levels": [tp_levels.get("tp1"), tp_levels.get("tp2"), tp_levels.get("tp3")],
        "position_size_usd": position_size_usd,
        "leverage": position_plan.get("leverage"),
        "direction": position_plan.get("direction"),
        "notes": position_plan.get("notes"),
        "position_size_pct": position_size_pct,
        "metadata": {
            "holding_horizon_bars": holding_horizon_bars,
            "entry_zone": entry_zone,
            "atr": plan_metadata.get("atr"),
            "risk_amount_usd": risk_amount_usd,
            "account_balance_estimate": account_balance,
            "risk_per_trade_pct": risk_pct,
            "max_position_size_pct": max_position_pct,
        },
    }

    if position_size_pct is None:
        return PlanDetails(
            valid=False,
            reason="Unable to determine position sizing percentage",
            sanitized_plan=sanitized_plan,
            holding_period=holding_period,
            holding_horizon_bars=holding_horizon_bars,
            entry_zone=entry_zone,
        )

    return PlanDetails(
        valid=True,
        entries=[entry_price],
        stop_loss=stop_loss,
        take_profits=tp_levels,
        position_size_pct=position_size_pct,
        holding_period=holding_period,
        holding_horizon_bars=holding_horizon_bars,
        sanitized_plan=sanitized_plan,
        entry_zone=entry_zone,
    )


def _compute_entry_zone(entry_price: float, atr: Optional[float], signal_type: str) -> Optional[Dict[str, float]]:
    if atr is None or atr <= 0:
        return None
    buffer = atr * 0.25
    if signal_type == "BUY":
        lower = entry_price - buffer
        upper = entry_price + buffer * 0.4
    else:
        lower = entry_price - buffer * 0.4
        upper = entry_price + buffer
    if lower >= upper:
        return None
    return {"lower": round(lower, 4), "upper": round(upper, 4)}


def _sanitize_tp_levels(
    entry_price: float,
    stop_loss: float,
    raw_levels: List[Any],
    signal_type: str,
) -> Dict[str, float]:
    cleaned: List[float] = []
    for level in raw_levels:
        value = _safe_float(level)
        if value is None:
            continue
        cleaned.append(value)

    if not cleaned:
        return {}

    cleaned = sorted(cleaned)
    if signal_type == "SELL":
        cleaned = list(reversed(cleaned))

    if signal_type == "BUY":
        cleaned = [level for level in cleaned if level > entry_price]
    else:
        cleaned = [level for level in cleaned if level < entry_price]

    if len(cleaned) < 3:
        return {}

    return {
        "tp1": float(cleaned[0]),
        "tp2": float(cleaned[1]),
        "tp3": float(cleaned[2]),
    }


def _compute_tp_levels(
    entry_price: float,
    stop_loss: float,
    signal_type: str,
    multipliers: Dict[str, Any],
) -> Dict[str, float]:
    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return {}

    defaults = [1.0, 1.8, 3.0]
    levels: Dict[str, float] = {}

    for idx, default in enumerate(defaults, start=1):
        key = f"tp{idx}"
        multiplier = _safe_float(multipliers.get(key), default)
        if multiplier is None or multiplier <= 0:
            multiplier = default
        adjustment = risk_distance * multiplier
        if signal_type == "BUY":
            level = entry_price + adjustment
        else:
            level = entry_price - adjustment
        levels[key] = float(level)

    return levels


def _classify_holding_period(holding_horizon_bars: Optional[int], timeframe: Optional[str]) -> str:
    if holding_horizon_bars is None:
        return "medium"
    try:
        minutes_per_bar = Timeframe.to_minutes(timeframe) if timeframe else Timeframe.to_minutes("1h")
    except Exception:  # pragma: no cover - fallback
        minutes_per_bar = 60
    total_minutes = holding_horizon_bars * minutes_per_bar
    if total_minutes <= 240:  # up to 4 hours
        return "short"
    if total_minutes <= 1440:  # up to 1 day
        return "medium"
    return "long"


def _build_rationale(
    explanation: Dict[str, Any],
    actionable: bool,
    hold_reasons: List[str],
    composite_context: Dict[str, Any],
) -> List[str]:
    points: List[str] = []

    composite_score = composite_context.get("score")
    contributions = composite_context.get("contributions", {})
    category_scores = composite_context.get("category_scores", {})

    if actionable and composite_score is not None:
        positive = [
            (category, contributions.get(category, 0.0))
            for category in _COMPOSITE_CATEGORIES
            if contributions.get(category, 0.0) > 0
        ]
        positive.sort(key=lambda item: item[1], reverse=True)
        if positive:
            driver_parts = [
                f"{_format_category_name(category)} {category_scores.get(category, 0.0):.2f}"
                for category, _ in positive[:3]
            ]
            points.append(
                f"Composite score {composite_score:.2f} driven by {', '.join(driver_parts)}."
            )
        else:
            points.append(f"Composite score {composite_score:.2f} met actionable threshold.")
    else:
        for reason in hold_reasons:
            if reason:
                points.append(reason)

    missing_categories = composite_context.get("missing_categories", [])
    if missing_categories:
        points.append(
            "Missing category data: "
            + ", ".join(_format_category_name(cat) for cat in missing_categories)
            + " (weighted 0)."
        )

    primary_reason = explanation.get("primary_reason")
    if primary_reason:
        points.append(primary_reason)

    supporting = explanation.get("supporting_factors", []) or []
    points.extend([factor for factor in supporting if factor])

    market_context = explanation.get("market_context")
    if market_context:
        points.append(market_context)

    additional = explanation.get("risk_factors", []) or []
    if not actionable:
        points.extend([risk for risk in additional if risk])

    seen: set = set()
    ordered: List[str] = []
    for item in points:
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)

    if not ordered:
        ordered.append("Composite analysis did not produce actionable insight.")

    return ordered[:6]


def _build_cancel_conditions(
    explanation: Dict[str, Any],
    metadata_triggers: List[str],
    plan_details: PlanDetails,
    actionable: bool,
    hold_reasons: List[str],
) -> List[str]:
    cancel_conditions: List[str] = []

    for trigger in metadata_triggers or []:
        if trigger and trigger not in cancel_conditions:
            cancel_conditions.append(trigger)

    risk_factors = explanation.get("risk_factors") or []
    for risk in risk_factors:
        if risk and risk not in cancel_conditions:
            cancel_conditions.append(risk)

    if actionable and plan_details.entry_zone:
        zone = plan_details.entry_zone
        lower = zone.get("lower")
        upper = zone.get("upper")
        if lower is not None:
            cancel_conditions.append(f"Cancel if price closes beyond entry zone ({lower:.2f} - {upper:.2f}).")
    else:
        for reason in hold_reasons:
            if reason and reason not in cancel_conditions:
                cancel_conditions.append(reason)

    return cancel_conditions[:5]


def _extract_weights(
    payload: Dict[str, Any],
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, float]:
    weights_source: Optional[Dict[str, Any]] = None
    for candidate in (
        params.get("weights"),
        payload.get("weights"),
        metadata.get("config_weights"),
    ):
        if isinstance(candidate, dict) and candidate:
            weights_source = candidate
            break

    if not weights_source:
        factors = payload.get("factors") or []
        if factors:
            weights_source = {
                (_FACTOR_CATEGORY_MAP.get(f.get("factor_name"), f.get("factor_name")) or "factor"): _safe_float(f.get("weight"), 0.0)
                for f in factors
            }
        else:
            weights_source = dict(DEFAULT_WEIGHTS)

    numeric_total = sum(
        float(value) for value in weights_source.values() if isinstance(value, (int, float))
    )

    if numeric_total <= 0:
        logger.warning("Category weights sum to zero; falling back to defaults.")
        weights_source = dict(DEFAULT_WEIGHTS)
        numeric_total = sum(DEFAULT_WEIGHTS.values())

    normalized: Dict[str, float] = {}
    for key, value in weights_source.items():
        if not isinstance(value, (int, float)):
            continue
        normalized[key] = safe_div(float(value), numeric_total, default=0.0)

    total_normalized = sum(normalized.values())
    if total_normalized <= 0:
        fallback_total = sum(DEFAULT_WEIGHTS.values())
        normalized = {
            key: safe_div(value, fallback_total, default=0.0)
            for key, value in DEFAULT_WEIGHTS.items()
        }
    elif not math.isclose(total_normalized, 1.0, rel_tol=1e-3):
        normalized = {
            key: safe_div(value, total_normalized, default=0.0)
            for key, value in normalized.items()
        }

    for key in DEFAULT_WEIGHTS:
        normalized.setdefault(key, 0.0)

    return normalized


def _resolve_composite_thresholds(
    metadata: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[float, float]:
    def _lookup(source: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not isinstance(source, dict):
            return None
        return _safe_float(source.get(key))

    buy = _lookup(params, "buy_threshold")
    if buy is None:
        buy = _lookup(metadata, "buy_threshold")
    composite_section = metadata.get("composite")
    if buy is None:
        buy = _lookup(composite_section, "buy_threshold")
    analysis_debug = metadata.get("analysis_debug")
    if buy is None:
        buy = _lookup(analysis_debug, "buy_threshold")

    sell = _lookup(params, "sell_threshold")
    if sell is None:
        sell = _lookup(metadata, "sell_threshold")
    if sell is None:
        sell = _lookup(composite_section, "sell_threshold")
    if sell is None:
        sell = _lookup(analysis_debug, "sell_threshold")

    buy = float(clamp(buy if buy is not None else 0.6, 0.0, 1.0))
    sell = float(clamp(sell if sell is not None else 0.4, 0.0, 1.0))
    return buy, sell


def _signal_from_composite(
    composite_score: float,
    buy_threshold: float,
    sell_threshold: float,
) -> str:
    if composite_score >= buy_threshold:
        return "BUY"
    if composite_score <= sell_threshold:
        return "SELL"
    return "HOLD"


def _extract_category_scores(factors: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    category_values: Dict[str, List[float]] = {}
    category_directions: Dict[str, str] = {}

    for factor in factors:
        name = factor.get("factor_name")
        category = _FACTOR_CATEGORY_MAP.get(name, name)
        if not category:
            continue

        score = factor.get("score")
        if score is not None:
            try:
                category_values.setdefault(category, []).append(float(score))
            except (TypeError, ValueError):
                continue

        metadata = factor.get("metadata") or {}
        direction = metadata.get("direction")
        if direction and category not in category_directions:
            category_directions[category] = direction

    category_data: Dict[str, Dict[str, Any]] = {}
    for category, values in category_values.items():
        if values:
            category_data[category] = {
                "score": sum(values) / len(values),
                "direction": category_directions.get(category),
            }

    for category, direction in category_directions.items():
        category_data.setdefault(category, {"score": None, "direction": direction})

    return category_data


def _compute_composite_context(
    factors: List[Dict[str, Any]],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    category_data = _extract_category_scores(factors)
    filtered_weights = {category: weights.get(category, 0.0) for category in _COMPOSITE_CATEGORIES}
    weight_total = sum(filtered_weights.values())
    if weight_total <= 0:
        normalized_weights = {category: 1.0 / len(_COMPOSITE_CATEGORIES) for category in _COMPOSITE_CATEGORIES}
    else:
        normalized_weights = {
            category: safe_div(filtered_weights[category], weight_total, default=0.0)
            for category in _COMPOSITE_CATEGORIES
        }

    contributions: Dict[str, float] = {}
    missing_categories: List[str] = []
    category_scores: Dict[str, Optional[float]] = {}

    for category in _COMPOSITE_CATEGORIES:
        data = category_data.get(category, {})
        score = data.get("score")
        category_scores[category] = score
        if score is None:
            contributions[category] = 0.0
            missing_categories.append(category)
        else:
            contributions[category] = normalized_weights.get(category, 0.0) * float(score)

    composite_score = clamp(sum(contributions.values()), 0.0, 1.0)
    top_contributors = sorted(
        (
            (category, contribution)
            for category, contribution in contributions.items()
            if contribution > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    directions = {
        category: data.get("direction")
        for category, data in category_data.items()
        if data.get("direction")
    }

    return {
        "score": composite_score,
        "weights": normalized_weights,
        "category_scores": category_scores,
        "contributions": contributions,
        "missing_categories": missing_categories,
        "top_contributors": top_contributors,
        "directions": directions,
    }


def _format_category_name(category: str) -> str:
    return category.replace("_", " ").title()


def _build_metadata_block(
    payload: Dict[str, Any],
    timeframe: str,
    composite_context: Dict[str, Any],
    plan_details: PlanDetails,
    actionable: bool,
    buy_threshold: float,
    sell_threshold: float,
    composite_score: float,
    original_signal_type: str,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "timestamp": payload.get("timestamp"),
        "timeframe": timeframe,
        "actionable": actionable,
        "composite_score": composite_score,
        "composite_weights": composite_context.get("weights"),
        "category_scores": composite_context.get("category_scores"),
        "category_contributions": composite_context.get("contributions"),
        "missing_categories": composite_context.get("missing_categories"),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "original_signal_type": original_signal_type,
    }

    top_contributors = composite_context.get("top_contributors") or []
    if top_contributors:
        metadata["top_contributors"] = [
            {"category": category, "contribution": contribution}
            for category, contribution in top_contributors[:3]
        ]

    directions = composite_context.get("directions")
    if directions:
        metadata["category_directions"] = directions

    if plan_details.holding_horizon_bars is not None:
        metadata["holding_horizon_bars"] = plan_details.holding_horizon_bars
    if plan_details.entry_zone:
        metadata["entry_zone"] = plan_details.entry_zone

    return {key: value for key, value in metadata.items() if value not in (None, {}, [])}


def _convert_confidence(
    composite_score: float,
    actionable: bool,
) -> int:
    distance = clamp(abs(composite_score - 0.5) * 2.0, 0.0, 1.0)
    confidence_value = round(1 + 9 * distance)

    if not actionable:
        confidence_value = max(1, min(confidence_value, 5))

    return int(clamp(float(confidence_value), 1.0, 10.0))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
