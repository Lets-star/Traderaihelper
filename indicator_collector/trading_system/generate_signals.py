"""Generate explicit JSON signals from trading analysis results.

This module provides functions to convert trading system analysis results
into the standardized JSON signal format required by the web UI.
"""

from __future__ import annotations

import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .backtester import DEFAULT_WEIGHTS, DEFAULT_SIGNAL_THRESHOLDS, indicator_defaults_for
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
    warnings: List[str] = field(default_factory=list)


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
        final_indicator_params = _ensure_indicator_params(params_dict, timeframe)

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
                payload,
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
            indicator_params=final_indicator_params,
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
                "neutralized_categories": composite_context.get("neutralized_categories"),
                "skipped_categories": composite_context.get("skipped_categories"),
                "original_signal_type": original_signal_type,
                "plan_warnings": plan_details.warnings,
                "reasons": hold_reasons,
                "indicator_params": deepcopy(final_indicator_params),
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
        "signal_thresholds",
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



def _first_positive(values: Iterable[Any]) -> Optional[float]:
    for value in values:
        numeric = _safe_float(value)
        if numeric is not None and numeric > 0:
            return numeric
    return None



def _build_plan_details(
    payload: Dict[str, Any],
    position_plan: Dict[str, Any],
    signal_type: str,
    params: Dict[str, Any],
    metadata: Dict[str, Any],
) -> PlanDetails:
    plan_warnings: List[str] = []

    plan_source: Dict[str, Any] = dict(position_plan) if isinstance(position_plan, dict) else {}
    if not plan_source:
        synthesized_plan, synth_warnings = _synthesize_plan_from_metadata(payload, signal_type, params, metadata)
        if not synthesized_plan:
            return PlanDetails(valid=False, reason="Position plan unavailable")
        plan_source = synthesized_plan
        plan_warnings.extend(synth_warnings)
    else:
        existing_metadata = plan_source.get("metadata")
        if isinstance(existing_metadata, dict):
            plan_warnings.extend(existing_metadata.get("planning_warnings", []))

    working_plan = dict(plan_source)
    plan_metadata = working_plan.get("metadata") or {}
    entry_price = _safe_float(working_plan.get("entry_price"))
    stop_loss = _safe_float(working_plan.get("stop_loss"))

    if (
        entry_price is None
        or stop_loss is None
        or entry_price <= 0
        or stop_loss <= 0
        or math.isclose(entry_price, stop_loss)
    ):
        synthesized_plan, synth_warnings = _synthesize_plan_from_metadata(payload, signal_type, params, metadata)
        if synthesized_plan:
            working_plan = dict(synthesized_plan)
            plan_metadata = working_plan.get("metadata") or {}
            entry_price = _safe_float(working_plan.get("entry_price"))
            stop_loss = _safe_float(working_plan.get("stop_loss"))
            plan_warnings.extend(synth_warnings)
        else:
            return PlanDetails(valid=False, reason="Entry or stop loss missing")

    if entry_price is None or stop_loss is None or entry_price <= 0 or stop_loss <= 0:
        return PlanDetails(valid=False, reason="Entry or stop loss missing")
    if math.isclose(entry_price, stop_loss):
        return PlanDetails(valid=False, reason="Entry and stop loss are identical")

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return PlanDetails(valid=False, reason="Risk distance between entry and stop is zero")

    raw_tp_levels = working_plan.get("take_profit_levels") or []
    tp_multipliers = plan_metadata.get("tp_sl_multipliers") or {}

    tp_levels = _sanitize_tp_levels(entry_price, stop_loss, raw_tp_levels, signal_type)
    if len(tp_levels) < 3:
        tp_levels = _compute_tp_levels(entry_price, stop_loss, signal_type, tp_multipliers)
        if len(tp_levels) < 3:
            if signal_type == "BUY":
                tp_levels = {
                    "tp1": float(entry_price + risk_distance),
                    "tp2": float(entry_price + risk_distance * 1.8),
                    "tp3": float(entry_price + risk_distance * 3.0),
                }
            else:
                tp_levels = {
                    "tp1": float(entry_price - risk_distance),
                    "tp2": float(entry_price - risk_distance * 1.8),
                    "tp3": float(entry_price - risk_distance * 3.0),
                }
            plan_warnings.append("Fallback take profit levels derived from risk multiples.")

    position_size_usd = _safe_float(working_plan.get("position_size_usd"))
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
        or plan_metadata.get("account_balance_estimate")
        or (risk_amount_usd / risk_pct if risk_amount_usd and risk_pct else None)
    )
    if account_balance is None and position_size_usd and risk_pct:
        try:
            account_balance = position_size_usd / max(risk_pct, 1e-6)
        except ZeroDivisionError:
            account_balance = None

    max_position_pct = _safe_float(
        params.get("max_position_size_pct")
        or params.get("position_config", {}).get("max_position_size_pct")
        or metadata.get("position_config", {}).get("max_position_size_pct")
        or plan_metadata.get("max_position_size_pct")
    )

    if position_size_usd is None and account_balance and risk_pct:
        risk_amount = risk_amount_usd or account_balance * risk_pct
        risk_per_unit = risk_distance / entry_price if entry_price else None
        if risk_per_unit and risk_per_unit > 0:
            position_size_usd = risk_amount / risk_per_unit
            plan_warnings.append("Position size derived from risk parameters.")
        else:
            position_size_usd = account_balance * risk_pct
            plan_warnings.append("Position size fallback applied due to zero risk distance.")

    if max_position_pct and position_size_usd and account_balance:
        pct_fraction = max_position_pct / 100.0 if max_position_pct > 1 else max_position_pct
        if pct_fraction and pct_fraction > 0:
            position_size_usd = min(position_size_usd, account_balance * pct_fraction)

    position_size_pct: Optional[float] = None
    if account_balance and position_size_usd:
        try:
            position_size_pct = min(100.0, max(0.0, (position_size_usd / account_balance) * 100.0))
            position_size_pct = round(position_size_pct, 2)
        except ZeroDivisionError:  # pragma: no cover - defensive
            position_size_pct = None

    if position_size_pct is None and position_size_usd and max_position_pct:
        pct_fraction = max_position_pct / 100.0 if max_position_pct > 1 else max_position_pct
        if pct_fraction and pct_fraction > 0:
            if account_balance is None:
                account_balance = position_size_usd / pct_fraction
            position_size_pct = round(min(100.0, pct_fraction * 100.0), 2)

    holding_horizon_bars = plan_metadata.get("holding_horizon_bars")
    holding_period = _classify_holding_period(
        holding_horizon_bars,
        working_plan.get("timeframe") or metadata.get("timeframe_used"),
    )

    entry_zone = _compute_entry_zone(entry_price, plan_metadata.get("atr"), signal_type)

    deduped_warnings = list(dict.fromkeys([warning for warning in plan_warnings if warning]))

    sanitized_plan: Dict[str, Any] = {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit_levels": [tp_levels.get("tp1"), tp_levels.get("tp2"), tp_levels.get("tp3")],
        "position_size_usd": position_size_usd,
        "leverage": working_plan.get("leverage"),
        "direction": working_plan.get("direction"),
        "notes": working_plan.get("notes"),
        "position_size_pct": position_size_pct,
        "metadata": {
            "holding_horizon_bars": holding_horizon_bars,
            "entry_zone": entry_zone,
            "atr": plan_metadata.get("atr"),
            "risk_amount_usd": risk_amount_usd,
            "account_balance_estimate": account_balance,
            "risk_per_trade_pct": risk_pct,
            "max_position_size_pct": max_position_pct,
            "planning_warnings": deduped_warnings,
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
            warnings=deduped_warnings,
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
        warnings=deduped_warnings,
    )


def _synthesize_plan_from_metadata(
    payload: Dict[str, Any],
    signal_type: str,
    params: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    latest = payload.get("latest") or {}

    entry_candidates = [
        latest.get("close"),
        latest.get("open"),
        latest.get("price"),
        metadata.get("entry_price"),
        metadata.get("last_close"),
        metadata.get("last_price"),
        metadata.get("price"),
    ]
    for factor in payload.get("factors") or []:
        if not isinstance(factor, dict):
            continue
        factor_metadata = factor.get("metadata") or {}
        entry_candidates.append(factor_metadata.get("current_price"))
        entry_candidates.append(factor_metadata.get("price"))

    entry_price = _first_positive(entry_candidates)
    if entry_price is None:
        return None, warnings

    atr_candidates = [
        metadata.get("atr"),
        metadata.get("volatility_atr"),
        latest.get("atr"),
    ]
    indicator_summary = metadata.get("indicator_summary")
    if isinstance(indicator_summary, dict):
        atr_candidates.extend(indicator_summary.values())
    atr_value = _first_positive(atr_candidates)
    if atr_value is None:
        atr_value = max(entry_price * 0.015, 0.5)
        warnings.append("ATR missing; synthetic plan using price-based volatility.")

    direction = "long" if signal_type == "BUY" else "short"
    buffer = max(atr_value, entry_price * 0.006)
    if direction == "long":
        stop_loss = entry_price - buffer
        if stop_loss <= 0:
            stop_loss = max(entry_price * 0.95, 0.0001)
            warnings.append("Synthetic plan adjusted stop loss to remain positive.")
    else:
        stop_loss = entry_price + buffer

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= entry_price * 1e-5:
        adjustment = max(entry_price * 0.01, atr_value)
        if direction == "long":
            stop_loss = max(0.0001, entry_price - adjustment)
        else:
            stop_loss = entry_price + adjustment
        warnings.append("Synthetic plan widened stop due to minimal risk distance.")
        risk_distance = abs(entry_price - stop_loss)

    multipliers = metadata.get("plan_defaults", {}).get("tp_sl_multipliers") or params.get("tp_sl_multipliers") or {}
    tp_levels = _compute_tp_levels(entry_price, stop_loss, signal_type, multipliers)
    if len(tp_levels) < 3:
        if signal_type == "BUY":
            tp_levels = {
                "tp1": float(entry_price + risk_distance),
                "tp2": float(entry_price + risk_distance * 1.8),
                "tp3": float(entry_price + risk_distance * 3.0),
            }
        else:
            tp_levels = {
                "tp1": float(entry_price - risk_distance),
                "tp2": float(entry_price - risk_distance * 1.8),
                "tp3": float(entry_price - risk_distance * 3.0),
            }
        warnings.append("Synthetic take profit levels derived from risk multiples.")

    take_profit_levels = [
        float(tp_levels["tp1"]),
        float(tp_levels["tp2"]),
        float(tp_levels["tp3"]),
    ]

    risk_pct = _safe_float(
        params.get("max_risk_per_trade_pct")
        or params.get("position_config", {}).get("max_risk_per_trade_pct")
        or metadata.get("position_config", {}).get("max_risk_per_trade_pct")
    )
    if risk_pct is None:
        risk_pct = _DEFAULT_RISK_PER_TRADE
        warnings.append("Using default risk per trade for synthetic plan.")

    account_balance = _safe_float(
        params.get("account_balance")
        or metadata.get("account_balance")
        or metadata.get("account_balance_estimate")
    )
    if account_balance is None or account_balance <= 0:
        account_balance = 10000.0
        warnings.append("Synthetic plan assumed $10,000 account balance.")

    risk_amount = account_balance * risk_pct
    risk_per_unit = risk_distance / entry_price if entry_price else None
    if risk_per_unit and risk_per_unit > 0:
        position_size_usd = risk_amount / risk_per_unit
    else:
        position_size_usd = account_balance * risk_pct
        warnings.append("Synthetic plan fallback sizing applied due to zero risk distance.")

    max_position_pct = _safe_float(
        params.get("max_position_size_pct")
        or params.get("position_config", {}).get("max_position_size_pct")
        or metadata.get("position_config", {}).get("max_position_size_pct")
    )
    if max_position_pct:
        pct_fraction = max_position_pct / 100.0 if max_position_pct > 1 else max_position_pct
        if pct_fraction and pct_fraction > 0:
            position_size_usd = min(position_size_usd, account_balance * pct_fraction)

    leverage = _safe_float(
        params.get("position_config", {}).get("default_leverage")
        or metadata.get("position_config", {}).get("default_leverage")
        or metadata.get("leverage")
    )
    if leverage is None or leverage <= 0:
        leverage = 3.0

    plan_metadata = {
        "atr": atr_value,
        "risk_amount_usd": risk_amount,
        "account_balance_estimate": account_balance,
        "risk_per_trade_pct": risk_pct,
        "max_position_size_pct": max_position_pct,
        "planning_warnings": warnings,
    }

    plan = {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit_levels": take_profit_levels,
        "position_size_usd": position_size_usd,
        "leverage": leverage,
        "direction": "long" if signal_type == "BUY" else "short",
        "notes": "Synthetic plan generated from metadata.",
        "metadata": plan_metadata,
    }
    return plan, warnings


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

    neutralized_categories = composite_context.get("neutralized_categories", [])
    for category in neutralized_categories:
        points.append(f"{_format_category_name(category)} data unavailable (neutral contribution).")

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

    threshold_overrides = params.get("signal_thresholds")
    buy = _lookup(threshold_overrides, "buy")
    sell = _lookup(threshold_overrides, "sell")

    if buy is None:
        buy = _lookup(params, "buy_threshold")
    if buy is None:
        buy = _lookup(metadata, "buy_threshold")
    composite_section = metadata.get("composite")
    if buy is None:
        buy = _lookup(composite_section, "buy_threshold")
    analysis_debug = metadata.get("analysis_debug")
    if buy is None:
        buy = _lookup(analysis_debug, "buy_threshold")

    if sell is None:
        sell = _lookup(params, "sell_threshold")
    if sell is None:
        sell = _lookup(metadata, "sell_threshold")
    if sell is None:
        sell = _lookup(composite_section, "sell_threshold")
    if sell is None:
        sell = _lookup(analysis_debug, "sell_threshold")

    default_buy = DEFAULT_SIGNAL_THRESHOLDS["buy"]
    default_sell = DEFAULT_SIGNAL_THRESHOLDS["sell"]

    buy = float(clamp(buy if buy is not None else default_buy, 0.0, 1.0))
    sell = float(clamp(sell if sell is not None else default_sell, 0.0, 1.0))

    if buy <= sell:
        buy, sell = max(buy, sell), min(buy, sell)
        if buy <= sell:
            buy = min(1.0, max(sell + 0.01, default_buy))
        if buy > 1.0:
            buy = 1.0
        if buy <= sell:
            sell = max(0.0, min(buy - 0.01, default_sell))

    if buy <= sell:
        buy = default_buy
        sell = default_sell

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
    neutralized_categories: List[str] = []
    skipped_categories: List[str] = []
    category_scores: Dict[str, Optional[float]] = {}

    for category in _COMPOSITE_CATEGORIES:
        weight = normalized_weights.get(category, 0.0)
        data = category_data.get(category, {})
        score = data.get("score")

        if weight <= 0:
            category_scores[category] = _safe_float(score)
            contributions[category] = 0.0
            if score is None:
                skipped_categories.append(category)
            continue

        if score is None:
            neutral_score = 0.5
            category_scores[category] = neutral_score
            contributions[category] = weight * neutral_score
            neutralized_categories.append(category)
        else:
            numeric_score = float(score)
            category_scores[category] = numeric_score
            contributions[category] = weight * numeric_score

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
        "missing_categories": [],
        "neutralized_categories": neutralized_categories,
        "skipped_categories": skipped_categories,
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
    *,
    indicator_params: Optional[Dict[str, Any]] = None,
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
        "neutralized_categories": composite_context.get("neutralized_categories"),
        "skipped_categories": composite_context.get("skipped_categories"),
        "buy_threshold": buy_threshold,
        "sell_threshold": sell_threshold,
        "original_signal_type": original_signal_type,
    }

    if plan_details.warnings:
        metadata["plan_warnings"] = plan_details.warnings

    if indicator_params:
        metadata["indicator_params"] = deepcopy(indicator_params)

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
    return int(clamp(float(confidence_value), 1.0, 10.0))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
