"""Position manager with risk-based sizing, TP/SL ladders, and diversification limits."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from ..trade_signals import calculate_tp_sl_levels
from .interfaces import (
    AnalyzerContext,
    PositionPlan,
    JsonDict,
)


@dataclass
class PositionManagerConfig:
    """Configuration for position management."""
    
    # Risk management
    max_position_size_usd: float = 1000.0
    max_risk_per_trade_pct: float = 0.02  # 2% risk per trade
    default_leverage: float = 10.0
    commission_rate: float = 0.0006
    
    # TP/SL multipliers
    tp1_multiplier: float = 1.5
    tp2_multiplier: float = 3.0
    tp3_multiplier: float = 5.0
    sl_multiplier: float = 1.0
    
    # Diversification limits
    max_concurrent_same_direction: int = 3
    max_total_positions: int = 10
    
    # Holding horizon (in bars)
    min_holding_bars: int = 5
    max_holding_bars: int = 100
    target_holding_bars: int = 20
    
    # Risk adjustments
    high_volatility_threshold: float = 0.05  # 5% ATR/price ratio
    low_liquidity_threshold: float = 0.2
    high_risk_score_threshold: float = 0.8
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_position_size_usd <= 0:
            raise ValueError("max_position_size_usd must be positive")
        if not 0 < self.max_risk_per_trade_pct <= 0.1:  # Max 10% risk
            raise ValueError("max_risk_per_trade_pct must be between 0 and 0.1")
        if self.default_leverage <= 0:
            raise ValueError("default_leverage must be positive")
        if self.max_concurrent_same_direction <= 0:
            raise ValueError("max_concurrent_same_direction must be positive")
        if self.tp1_multiplier >= self.tp2_multiplier or self.tp2_multiplier >= self.tp3_multiplier:
            raise ValueError("TP multipliers must be increasing")
        if self.sl_multiplier <= 0:
            raise ValueError("sl_multiplier must be positive")
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "max_position_size_usd": self.max_position_size_usd,
            "max_risk_per_trade_pct": self.max_risk_per_trade_pct,
            "default_leverage": self.default_leverage,
            "commission_rate": self.commission_rate,
            "tp1_multiplier": self.tp1_multiplier,
            "tp2_multiplier": self.tp2_multiplier,
            "tp3_multiplier": self.tp3_multiplier,
            "sl_multiplier": self.sl_multiplier,
            "max_concurrent_same_direction": self.max_concurrent_same_direction,
            "max_total_positions": self.max_total_positions,
            "min_holding_bars": self.min_holding_bars,
            "max_holding_bars": self.max_holding_bars,
            "target_holding_bars": self.target_holding_bars,
            "high_volatility_threshold": self.high_volatility_threshold,
            "low_liquidity_threshold": self.low_liquidity_threshold,
            "high_risk_score_threshold": self.high_risk_score_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "PositionManagerConfig":
        """Create from dictionary."""
        return cls(
            max_position_size_usd=float(data.get("max_position_size_usd", 1000.0)),
            max_risk_per_trade_pct=float(data.get("max_risk_per_trade_pct", 0.02)),
            default_leverage=float(data.get("default_leverage", 10.0)),
            commission_rate=float(data.get("commission_rate", 0.0006)),
            tp1_multiplier=float(data.get("tp1_multiplier", 1.5)),
            tp2_multiplier=float(data.get("tp2_multiplier", 3.0)),
            tp3_multiplier=float(data.get("tp3_multiplier", 5.0)),
            sl_multiplier=float(data.get("sl_multiplier", 1.0)),
            max_concurrent_same_direction=int(data.get("max_concurrent_same_direction", 3)),
            max_total_positions=int(data.get("max_total_positions", 10)),
            min_holding_bars=int(data.get("min_holding_bars", 5)),
            max_holding_bars=int(data.get("max_holding_bars", 100)),
            target_holding_bars=int(data.get("target_holding_bars", 20)),
            high_volatility_threshold=float(data.get("high_volatility_threshold", 0.05)),
            low_liquidity_threshold=float(data.get("low_liquidity_threshold", 0.2)),
            high_risk_score_threshold=float(data.get("high_risk_score_threshold", 0.8)),
        )


@dataclass
class DiversificationGuard:
    """Tracks current positions and enforces diversification limits."""
    
    long_positions: List[str] = field(default_factory=list)
    short_positions: List[str] = field(default_factory=list)
    total_positions: int = 0
    
    def can_add_position(self, direction: Literal["long", "short"], symbol: str, 
                        config: PositionManagerConfig) -> Tuple[bool, Optional[str]]:
        """Check if a new position can be added given diversification limits."""
        # Check total position limit
        if self.total_positions >= config.max_total_positions:
            return False, f"Max total positions ({config.max_total_positions}) reached"
        
        # Check same-direction limit
        if direction == "long":
            if len(self.long_positions) >= config.max_concurrent_same_direction:
                return False, f"Max long positions ({config.max_concurrent_same_direction}) reached"
            if symbol in self.long_positions:
                return False, f"Already have long position in {symbol}"
        else:  # short
            if len(self.short_positions) >= config.max_concurrent_same_direction:
                return False, f"Max short positions ({config.max_concurrent_same_direction}) reached"
            if symbol in self.short_positions:
                return False, f"Already have short position in {symbol}"
        
        return True, None
    
    def add_position(self, direction: Literal["long", "short"], symbol: str) -> None:
        """Add a new position to tracking."""
        if direction == "long":
            self.long_positions.append(symbol)
        else:
            self.short_positions.append(symbol)
        self.total_positions += 1
    
    def remove_position(self, direction: Literal["long", "short"], symbol: str) -> None:
        """Remove a position from tracking."""
        if direction == "long" and symbol in self.long_positions:
            self.long_positions.remove(symbol)
            self.total_positions = max(0, self.total_positions - 1)
        elif direction == "short" and symbol in self.short_positions:
            self.short_positions.remove(symbol)
            self.total_positions = max(0, self.total_positions - 1)
    
    def get_positions_by_direction(self, direction: Literal["long", "short"]) -> List[str]:
        """Get positions by direction."""
        if direction == "long":
            return list(self.long_positions)
        else:
            return list(self.short_positions)
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "long_positions": list(self.long_positions),
            "short_positions": list(self.short_positions),
            "total_positions": self.total_positions,
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "DiversificationGuard":
        """Create from dictionary."""
        return cls(
            long_positions=list(data.get("long_positions", [])),
            short_positions=list(data.get("short_positions", [])),
            total_positions=int(data.get("total_positions", 0)),
        )


@dataclass
class PositionSizingResult:
    """Result of position sizing calculation."""
    
    position_size_usd: float
    risk_amount_usd: float
    leverage: float
    quantity: float
    commission_cost: float
    sizing_factors: Dict[str, float] = field(default_factory=dict)
    cancellation_reasons: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class PositionManagerResult:
    """Complete position management result."""
    
    position_plan: Optional[PositionPlan]
    sizing_result: Optional[PositionSizingResult]
    can_trade: bool
    cancellation_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    holding_horizon_bars: Optional[int] = None
    diversification_guard: Optional[DiversificationGuard] = None
    metadata: JsonDict = field(default_factory=dict)


def _safe_float(value: Any) -> Optional[float]:
    """Convert arbitrary values to floats safely."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if math.isnan(numeric) or not math.isfinite(numeric):  # pragma: no cover - defensive
        return None
    return numeric


def _first_valid_float(values: Iterable[Any]) -> Optional[float]:
    """Return the first positive float from an iterable of candidates."""
    for value in values:
        numeric = _safe_float(value)
        if numeric is not None and numeric > 0:
            return numeric
    return None


def _resolve_entry_price(context: AnalyzerContext) -> Tuple[Optional[float], Optional[str]]:
    """Determine an entry price using available context information."""
    candidates: List[Any] = [getattr(context, "current_price", None)]

    ohlcv = context.ohlcv or {}
    for key in ("close", "open", "high", "low"):
        candidates.append(ohlcv.get(key))

    metadata = context.metadata or {}
    for key in (
        "entry_price",
        "last_close",
        "last_price",
        "recent_close",
        "price",
        "close",
        "current_price",
    ):
        candidates.append(metadata.get(key))

    extras = context.extras or {}
    price_context = extras.get("price_context")
    if isinstance(price_context, dict):
        for key in ("last_close", "close", "price", "reference"):
            candidates.append(price_context.get(key))
    for key in ("last_close", "close", "price", "reference_price"):
        candidates.append(extras.get(key))

    for factor in context.historical_signals or []:
        if isinstance(factor, dict):
            candidates.append(factor.get("price"))

    entry_price = _first_valid_float(candidates)
    if entry_price is None:
        return None, "Entry price unavailable from context."
    return entry_price, None


def _resolve_atr_value(context: AnalyzerContext, entry_price: float) -> Tuple[Optional[float], Optional[str]]:
    """Resolve ATR value, falling back to reasonable approximations when needed."""
    indicators = context.indicators or {}
    candidates: List[Any] = [indicators.get("atr")]

    atr_channels = indicators.get("atr_channels")
    if isinstance(atr_channels, dict):
        candidates.extend(atr_channels.values())

    metadata = context.metadata or {}
    for key in ("atr", "atr_value", "average_true_range"):
        candidates.append(metadata.get(key))

    volatility = metadata.get("market_volatility")
    if isinstance(volatility, dict):
        candidates.extend(volatility.values())

    extras = context.extras or {}
    indicator_summary = extras.get("indicator_summary")
    if isinstance(indicator_summary, dict):
        candidates.extend(indicator_summary.values())

    atr_value = _first_valid_float(candidates)
    if atr_value is not None:
        return atr_value, None

    ohlcv = context.ohlcv or {}
    session_high = _safe_float(ohlcv.get("high"))
    session_low = _safe_float(ohlcv.get("low"))
    if session_high is not None and session_low is not None and session_high > session_low:
        fallback = max(session_high - session_low, (entry_price or session_high) * 0.01)
        return fallback, "ATR unavailable; approximated from intrabar range."

    if entry_price and entry_price > 0:
        fallback = entry_price * 0.015
    else:  # pragma: no cover - defensive
        fallback = 1.0
    return fallback, "ATR unavailable; approximated from price-based volatility."


def _extract_structure_level(context: AnalyzerContext, favor: Literal["support", "resistance"]) -> Optional[float]:
    """Attempt to extract structure-based levels for stop placement."""
    candidates: List[Any] = []
    market_structure = context.market_structure or {}

    key_groups = {
        "support": ("recent_support", "nearest_support", "support", "swing_low", "structure_low"),
        "resistance": ("recent_resistance", "nearest_resistance", "resistance", "swing_high", "structure_high"),
    }
    for key in key_groups[favor]:
        candidates.append(market_structure.get(key))

    zone_key = "support_zone" if favor == "support" else "resistance_zone"
    zone = market_structure.get(zone_key)
    if isinstance(zone, dict):
        for key in ("lower", "bottom", "mid", "top", "upper"):
            candidates.append(zone.get(key))

    metadata = context.metadata or {}
    last_levels = metadata.get("last_structure_levels")
    if isinstance(last_levels, dict):
        target_key = "low" if favor == "support" else "high"
        candidates.append(last_levels.get(target_key))

    extras = context.extras or {}
    structure_levels = extras.get("structure_levels")
    if isinstance(structure_levels, dict):
        target_key = "support" if favor == "support" else "resistance"
        candidates.append(structure_levels.get(target_key))

    for zone in context.zones or []:
        zone_type = str(zone.get("type", "")).lower()
        if favor == "support" and any(term in zone_type for term in ("bull", "demand", "support")):
            for key in ("bottom", "lower", "top", "upper"):
                candidates.append(zone.get(key))
        elif favor == "resistance" and any(term in zone_type for term in ("bear", "supply", "resistance")):
            for key in ("top", "upper", "bottom", "lower"):
                candidates.append(zone.get(key))

    numeric_levels = [value for value in (_safe_float(item) for item in candidates) if value is not None]
    if not numeric_levels:
        return None
    return min(numeric_levels) if favor == "support" else max(numeric_levels)


def _compute_stop_loss(
    entry_price: float,
    atr_value: float,
    signal_direction: Literal["long", "short"],
    config: PositionManagerConfig,
    context: AnalyzerContext,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute a stop loss using structure when available, otherwise ATR buffers."""
    buffer = atr_value * config.sl_multiplier if atr_value and atr_value > 0 else None
    if buffer is None or buffer <= 0:
        buffer = max(entry_price * 0.008, 0.5)

    warning: Optional[str] = None
    if signal_direction == "long":
        stop_loss = entry_price - buffer
        structure_level = _extract_structure_level(context, "support")
        if structure_level is not None and structure_level < entry_price:
            stop_loss = min(stop_loss, structure_level - buffer * 0.25)
            warning = "Stop loss aligned with nearby structure support." if warning is None else warning
        if stop_loss <= 0:
            stop_loss = max(entry_price * 0.95, 0.0001)
            warning = "Stop loss adjusted to remain positive." if warning is None else warning
        return stop_loss, warning

    stop_loss = entry_price + buffer
    structure_level = _extract_structure_level(context, "resistance")
    if structure_level is not None and structure_level > entry_price:
        stop_loss = max(stop_loss, structure_level + buffer * 0.25)
        warning = "Stop loss aligned with nearby structure resistance." if warning is None else warning
    return stop_loss, warning


def _compute_tp_levels_from_risk(
    entry_price: float,
    stop_loss: float,
    signal_direction: Literal["long", "short"],
    config: PositionManagerConfig,
) -> Dict[str, float]:
    """Compute TP levels using configured R-multipliers."""
    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return {}

    multipliers = [config.tp1_multiplier, config.tp2_multiplier, config.tp3_multiplier]
    levels: Dict[str, float] = {}
    for idx, mult in enumerate(multipliers, start=1):
        key = f"tp{idx}"
        adjustment = risk_distance * max(mult, 0.1)
        if signal_direction == "long":
            levels[key] = entry_price + adjustment
        else:
            levels[key] = entry_price - adjustment
    return levels


def _build_entry_zone(
    entry_price: float,
    atr_value: Optional[float],
    signal_direction: Literal["long", "short"],
    context: AnalyzerContext,
) -> Optional[Dict[str, float]]:
    """Construct an entry zone using available market structure information."""
    def _format_zone(lower: Any, upper: Any) -> Optional[Dict[str, float]]:
        lower_val = _safe_float(lower)
        upper_val = _safe_float(upper)
        if lower_val is None or upper_val is None:
            return None
        lower_numeric, upper_numeric = sorted((lower_val, upper_val))
        if lower_numeric >= upper_numeric:
            return None
        return {"lower": round(lower_numeric, 4), "upper": round(upper_numeric, 4)}

    market_structure = context.market_structure or {}
    primary_zone = market_structure.get("confluence_zone")
    if isinstance(primary_zone, dict):
        formatted = _format_zone(primary_zone.get("lower") or primary_zone.get("bottom"), primary_zone.get("upper") or primary_zone.get("top"))
        if formatted:
            return formatted

    for zone in (market_structure.get("zones") or []) + (context.zones or []):
        if not isinstance(zone, dict):
            continue
        zone_type = str(zone.get("type", "")).lower()
        if signal_direction == "long" and any(term in zone_type for term in ("bull", "demand", "support")):
            formatted = _format_zone(zone.get("bottom"), zone.get("top"))
            if formatted:
                return formatted
        elif signal_direction == "short" and any(term in zone_type for term in ("bear", "supply", "resistance")):
            formatted = _format_zone(zone.get("bottom"), zone.get("top"))
            if formatted:
                return formatted

    if atr_value is None or atr_value <= 0:
        atr_value = entry_price * 0.01

    buffer = atr_value * 0.25
    if signal_direction == "long":
        lower = entry_price - buffer
        upper = entry_price + buffer * 0.4
    else:
        lower = entry_price - buffer * 0.4
        upper = entry_price + buffer

    lower, upper = sorted((lower, upper))
    if lower >= upper:  # pragma: no cover - defensive
        return None
    return {"lower": round(lower, 4), "upper": round(upper, 4)}


def calculate_risk_based_position_size(
    entry_price: float,
    stop_loss: float,
    account_balance: float,
    risk_per_trade_pct: float,
    leverage: float = 10.0,
    commission_rate: float = 0.0006,
) -> PositionSizingResult:
    """
    Calculate position size based on risk management rules.
    
    Args:
        entry_price: Entry price for the position
        stop_loss: Stop loss price
        account_balance: Total account balance
        risk_per_trade_pct: Percentage of account to risk per trade
        leverage: Leverage multiplier
        commission_rate: Commission rate
    
    Returns:
        PositionSizingResult with calculated position details
    """
    # Calculate risk amount in USD
    risk_amount_usd = account_balance * risk_per_trade_pct
    
    # Calculate price distance to stop loss
    price_distance = abs(entry_price - stop_loss)
    risk_per_unit = price_distance / entry_price
    
    # Calculate position size based on risk
    if risk_per_unit > 0:
        position_size_usd = risk_amount_usd / risk_per_unit
    else:
        position_size_usd = account_balance * 0.1  # Default 10% if no risk defined
    
    # Apply leverage
    notional_value = position_size_usd * leverage
    quantity = notional_value / entry_price
    commission_cost = notional_value * commission_rate
    
    # Calculate sizing factors
    sizing_factors = {
        "risk_per_unit": risk_per_unit,
        "risk_amount_usd": risk_amount_usd,
        "notional_value": notional_value,
        "commission_cost_pct": commission_cost / position_size_usd if position_size_usd > 0 else 0,
    }
    
    return PositionSizingResult(
        position_size_usd=position_size_usd,
        risk_amount_usd=risk_amount_usd,
        leverage=leverage,
        quantity=quantity,
        commission_cost=commission_cost,
        sizing_factors=sizing_factors,
    )


def assess_market_conditions(
    context: AnalyzerContext,
    config: PositionManagerConfig,
) -> Tuple[bool, List[str], List[str]]:
    """Assess market conditions, returning fatal blockers and non-blocking warnings."""
    fatal_reasons: List[str] = []
    warnings: List[str] = []

    current_price = _safe_float(getattr(context, "current_price", None))
    if current_price is None or current_price <= 0:
        fatal_reasons.append("Current price unavailable for position planning.")
    else:
        atr_value = _safe_float((context.indicators or {}).get("atr"))
        if atr_value is not None and atr_value > 0:
            volatility_ratio = atr_value / current_price
            if volatility_ratio > config.high_volatility_threshold * 2:
                warnings.append(
                    f"Extreme volatility detected (ATR ratio {volatility_ratio:.3f})."
                )
            elif volatility_ratio > config.high_volatility_threshold:
                warnings.append(
                    f"High volatility environment (ATR ratio {volatility_ratio:.3f})."
                )
        else:
            warnings.append("ATR missing; will estimate from price data.")

    volume_analysis = context.volume_analysis or {}
    volume_confidence = _safe_float(volume_analysis.get("volume_confidence"))
    if volume_confidence is not None:
        if volume_confidence < config.low_liquidity_threshold:
            warnings.append(
                f"Liquidity caution: confidence {volume_confidence:.3f} below threshold {config.low_liquidity_threshold:.3f}."
            )
    else:
        warnings.append("Volume confidence unavailable; liquidity assessment limited.")

    advanced_metrics = context.advanced_metrics or {}
    market_context = advanced_metrics.get("market_context")
    if isinstance(market_context, dict):
        risk_score = _safe_float(market_context.get("risk_score"))
        if risk_score is not None and risk_score > config.high_risk_score_threshold:
            warnings.append(
                f"Elevated market risk score ({risk_score:.3f}) exceeds threshold {config.high_risk_score_threshold:.3f}."
            )

    return len(fatal_reasons) == 0, fatal_reasons, warnings


def estimate_holding_horizon(
    context: AnalyzerContext,
    config: PositionManagerConfig,
    signal_direction: Literal["long", "short"],
) -> int:
    """
    Estimate optimal holding horizon based on market conditions.
    
    Args:
        context: Market analysis context
        config: Position manager configuration
        signal_direction: Direction of the signal
    
    Returns:
        Estimated holding horizon in bars
    """
    # Base holding period
    base_horizon = config.target_holding_bars
    
    # Adjust based on trend strength
    indicators = context.indicators or {}
    trend_strength = indicators.get("trend_strength", 0.5)
    
    # Stronger trends = longer holds
    trend_adjustment = (trend_strength - 0.5) * 20  # +/- 10 bars
    
    # Adjust based on volatility
    atr = indicators.get("atr", 0)
    current_price = context.current_price
    volatility_adjustment = 0
    if atr and current_price > 0:
        volatility_ratio = atr / current_price
        # Higher volatility = shorter holds
        volatility_adjustment = -(volatility_ratio - 0.02) * 100  # Adjust around 2% baseline
    
    # Adjust based on market structure
    market_structure = context.market_structure or {}
    structure_state = market_structure.get("structure_state", "neutral")
    structure_adjustment = 0
    if structure_state == "trending":
        structure_adjustment = 5  # Extend holds in trending markets
    elif structure_state == "ranging":
        structure_adjustment = -5  # Shorten holds in ranging markets
    
    # Calculate final horizon
    estimated_horizon = int(base_horizon + trend_adjustment + volatility_adjustment + structure_adjustment)
    
    # Clamp to valid range
    return max(config.min_holding_bars, min(config.max_holding_bars, estimated_horizon))


def create_position_plan(
    context: AnalyzerContext,
    signal_direction: Literal["long", "short"],
    config: PositionManagerConfig,
    account_balance: float = 10000.0,
    diversification_guard: Optional[DiversificationGuard] = None,
) -> PositionManagerResult:
    """Create a comprehensive position plan with risk management and diversification checks."""
    config.validate()

    cancellation_reasons: List[str] = []
    warnings: List[str] = []

    if diversification_guard:
        can_add, reason = diversification_guard.can_add_position(signal_direction, context.symbol, config)
        if not can_add:
            cancellation_reasons.append(reason or "Diversification limit reached")
            return PositionManagerResult(
                position_plan=None,
                sizing_result=None,
                can_trade=False,
                cancellation_reasons=cancellation_reasons,
                warnings=warnings,
                diversification_guard=diversification_guard,
            )

    can_trade, fatal_reasons, market_warnings = assess_market_conditions(context, config)
    warnings.extend(market_warnings)
    if not can_trade:
        cancellation_reasons.extend(fatal_reasons)
        metadata = {"warnings": warnings} if warnings else {}
        return PositionManagerResult(
            position_plan=None,
            sizing_result=None,
            can_trade=False,
            cancellation_reasons=cancellation_reasons,
            warnings=warnings,
            diversification_guard=diversification_guard,
            metadata=metadata,
        )

    entry_price, entry_warning = _resolve_entry_price(context)
    if entry_warning:
        warnings.append(entry_warning)
    if entry_price is None or entry_price <= 0:
        cancellation_reasons.append("Unable to determine entry price.")
        metadata = {"warnings": warnings} if warnings else {}
        return PositionManagerResult(
            position_plan=None,
            sizing_result=None,
            can_trade=False,
            cancellation_reasons=cancellation_reasons,
            warnings=warnings,
            diversification_guard=diversification_guard,
            metadata=metadata,
        )

    atr_value, atr_warning = _resolve_atr_value(context, entry_price)
    if atr_warning:
        warnings.append(atr_warning)
    if atr_value is None or atr_value <= 0:
        cancellation_reasons.append("Unable to estimate volatility for stop placement.")
        metadata = {"warnings": warnings} if warnings else {}
        return PositionManagerResult(
            position_plan=None,
            sizing_result=None,
            can_trade=False,
            cancellation_reasons=cancellation_reasons,
            warnings=warnings,
            diversification_guard=diversification_guard,
            metadata=metadata,
        )

    stop_loss, stop_warning = _compute_stop_loss(entry_price, atr_value, signal_direction, config, context)
    if stop_warning:
        warnings.append(stop_warning)
    if stop_loss is None or stop_loss <= 0:
        cancellation_reasons.append("Unable to compute stop loss.")
        metadata = {"warnings": warnings} if warnings else {}
        return PositionManagerResult(
            position_plan=None,
            sizing_result=None,
            can_trade=False,
            cancellation_reasons=cancellation_reasons,
            warnings=warnings,
            diversification_guard=diversification_guard,
            metadata=metadata,
        )

    if math.isclose(stop_loss, entry_price, rel_tol=1e-6):
        fallback_distance = max(entry_price * 0.01, atr_value * max(config.sl_multiplier, 0.5))
        if signal_direction == "long":
            stop_loss = max(0.0001, entry_price - fallback_distance)
        else:
            stop_loss = entry_price + fallback_distance
        warnings.append("Stop loss adjusted to ensure non-zero risk distance.")

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= entry_price * 1e-5:
        fallback_distance = max(entry_price * 0.01, atr_value * max(config.sl_multiplier, 0.5))
        if signal_direction == "long":
            stop_loss = max(0.0001, entry_price - fallback_distance)
        else:
            stop_loss = entry_price + fallback_distance
        warnings.append("Risk distance recalibrated due to insufficient separation.")
        risk_distance = abs(entry_price - stop_loss)

    tp_levels = _compute_tp_levels_from_risk(entry_price, stop_loss, signal_direction, config)
    if not tp_levels:
        fallback_levels = calculate_tp_sl_levels(
            entry_price=entry_price,
            is_long=(signal_direction == "long"),
            atr_value=atr_value,
            tp1_multiplier=config.tp1_multiplier,
            tp2_multiplier=config.tp2_multiplier,
            tp3_multiplier=config.tp3_multiplier,
            sl_multiplier=config.sl_multiplier,
        )
        tp_levels = {
            "tp1": fallback_levels["tp1"],
            "tp2": fallback_levels["tp2"],
            "tp3": fallback_levels["tp3"],
        }
        if signal_direction == "short":
            stop_loss = fallback_levels["sl"]
        warnings.append("Take profit levels generated from ATR defaults.")
    else:
        if signal_direction == "long":
            ordered = sorted(tp_levels.values())
        else:
            ordered = sorted(tp_levels.values(), reverse=True)
        tp_levels = {f"tp{idx}": float(value) for idx, value in enumerate(ordered, start=1)}

    balance_value = _safe_float(account_balance)
    if balance_value is None or balance_value <= 0:
        balance_value = 10000.0
        warnings.append("Account balance unavailable; defaulted to $10,000.")
    balance_value = max(balance_value, config.max_position_size_usd)

    risk_pct = config.max_risk_per_trade_pct
    sizing_result = calculate_risk_based_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        account_balance=balance_value,
        risk_per_trade_pct=risk_pct,
        leverage=config.default_leverage,
        commission_rate=config.commission_rate,
    )

    if sizing_result.position_size_usd <= 0:
        fallback_size = min(balance_value * risk_pct * config.default_leverage, config.max_position_size_usd)
        if fallback_size <= 0:
            fallback_size = config.max_position_size_usd
        sizing_result.position_size_usd = fallback_size
        sizing_result.sizing_factors["position_size_fallback"] = True
        sizing_result.quantity = (fallback_size * config.default_leverage) / entry_price
        sizing_result.commission_cost = (fallback_size * config.default_leverage) * config.commission_rate
        warnings.append("Position size fallback applied due to zero risk distance.")

    final_position_size = min(sizing_result.position_size_usd, config.max_position_size_usd)
    if final_position_size < sizing_result.position_size_usd:
        sizing_result.sizing_factors["size_limited"] = True
        sizing_result.sizing_factors["original_size"] = sizing_result.position_size_usd
        sizing_result.position_size_usd = final_position_size
        sizing_result.quantity = (final_position_size * config.default_leverage) / entry_price
        sizing_result.commission_cost = (final_position_size * config.default_leverage) * config.commission_rate
        warnings.append("Position size limited by configuration.")

    holding_horizon = estimate_holding_horizon(context, config, signal_direction)
    entry_zone = _build_entry_zone(entry_price, atr_value, signal_direction, context)

    take_profit_list = [
        float(tp_levels.get("tp1", entry_price)),
        float(tp_levels.get("tp2", entry_price)),
        float(tp_levels.get("tp3", entry_price)),
    ]

    position_plan = PositionPlan(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_levels=take_profit_list,
        position_size_usd=sizing_result.position_size_usd,
        leverage=config.default_leverage,
        direction=signal_direction,
        notes=f"Holding horizon: {holding_horizon} bars",
        metadata={
            "atr": atr_value,
            "tp_sl_multipliers": {
                "tp1": config.tp1_multiplier,
                "tp2": config.tp2_multiplier,
                "tp3": config.tp3_multiplier,
                "sl": config.sl_multiplier,
            },
            "holding_horizon_bars": holding_horizon,
            "sizing_factors": sizing_result.sizing_factors,
        },
    )

    if entry_zone:
        position_plan.metadata["entry_zone"] = entry_zone
    if warnings:
        position_plan.metadata["planning_warnings"] = warnings

    if diversification_guard:
        diversification_guard.add_position(signal_direction, context.symbol)

    result_metadata: JsonDict = {
        "signal_direction": signal_direction,
        "account_balance": balance_value,
        "final_position_size": sizing_result.position_size_usd,
    }
    if entry_zone:
        result_metadata["entry_zone"] = entry_zone
    if warnings:
        result_metadata["warnings"] = warnings

    return PositionManagerResult(
        position_plan=position_plan,
        sizing_result=sizing_result,
        can_trade=True,
        cancellation_reasons=[],
        warnings=warnings,
        holding_horizon_bars=holding_horizon,
        diversification_guard=diversification_guard,
        metadata=result_metadata,
    )


def create_diversification_guard() -> DiversificationGuard:
    """Create a new diversification guard instance."""
    return DiversificationGuard()


def validate_tp_sl_spacing(
    tp_levels: List[float],
    stop_loss: float,
    entry_price: float,
    min_spacing_pct: float = 0.005,  # 0.5% minimum spacing
) -> Tuple[bool, List[str]]:
    """
    Validate that TP levels and SL have appropriate spacing.
    
    Args:
        tp_levels: List of take profit levels
        stop_loss: Stop loss level
        entry_price: Entry price
        min_spacing_pct: Minimum spacing as percentage of entry price
    
    Returns:
        Tuple of (is_valid, validation_errors)
    """
    errors = []
    min_spacing_abs = entry_price * min_spacing_pct
    
    # Check TP levels spacing
    for i in range(len(tp_levels) - 1):
        spacing = abs(tp_levels[i + 1] - tp_levels[i])
        if spacing < min_spacing_abs:
            errors.append(f"TP{i+1} and TP{i+2} too close: {spacing:.6f} < {min_spacing_abs:.6f}")
    
    # Check TP to SL spacing
    for i, tp in enumerate(tp_levels):
        spacing = abs(tp - stop_loss)
        if spacing < min_spacing_abs:
            errors.append(f"TP{i+1} and SL too close: {spacing:.6f} < {min_spacing_abs:.6f}")
    
    # Check entry to SL spacing
    entry_sl_spacing = abs(entry_price - stop_loss)
    if entry_sl_spacing < min_spacing_abs:
        errors.append(f"Entry and SL too close: {entry_sl_spacing:.6f} < {min_spacing_abs:.6f}")
    
    return len(errors) == 0, errors