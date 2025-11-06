"""Position manager with risk-based sizing, TP/SL ladders, and diversification limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from ..trade_signals import (
    calculate_position_metrics,
    calculate_tp_sl_levels,
)
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
    holding_horizon_bars: Optional[int] = None
    diversification_guard: Optional[DiversificationGuard] = None
    metadata: JsonDict = field(default_factory=dict)


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
) -> Tuple[bool, List[str]]:
    """
    Assess market conditions for position viability.
    
    Args:
        context: Market analysis context
        config: Position manager configuration
    
    Returns:
        Tuple of (can_trade, cancellation_reasons)
    """
    cancellation_reasons = []
    
    # Check volatility
    atr = context.indicators.get("atr", 0)
    current_price = context.current_price
    if atr and current_price > 0:
        volatility_ratio = atr / current_price
        if volatility_ratio > config.high_volatility_threshold:
            cancellation_reasons.append(f"High volatility: {volatility_ratio:.3f} > {config.high_volatility_threshold}")
    
    # Check liquidity (from volume analysis)
    volume_analysis = context.volume_analysis or {}
    volume_confidence = volume_analysis.get("volume_confidence", 0)
    if volume_confidence < config.low_liquidity_threshold:
        cancellation_reasons.append(f"Low liquidity: {volume_confidence:.3f} < {config.low_liquidity_threshold}")
    
    # Check risk score (from advanced metrics)
    advanced_metrics = context.advanced_metrics or {}
    market_context = advanced_metrics.get("market_context", {})
    if isinstance(market_context, dict):
        risk_score = market_context.get("risk_score", 0)
        if risk_score > config.high_risk_score_threshold:
            cancellation_reasons.append(f"High risk score: {risk_score:.3f} > {config.high_risk_score_threshold}")
    
    can_trade = len(cancellation_reasons) == 0
    return can_trade, cancellation_reasons


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
    """
    Create a comprehensive position plan with risk management and diversification checks.
    
    Args:
        context: Market analysis context
        signal_direction: Direction of the trading signal
        config: Position manager configuration
        account_balance: Total account balance for risk calculations
        diversification_guard: Current position tracking
    
    Returns:
        Complete position manager result with plan or cancellation reasons
    """
    config.validate()
    
    cancellation_reasons = []
    
    # Check diversification limits
    if diversification_guard:
        can_add, reason = diversification_guard.can_add_position(
            signal_direction, context.symbol, config
        )
        if not can_add:
            cancellation_reasons.append(reason or "Diversification limit reached")
    
    # Assess market conditions
    can_trade, market_reasons = assess_market_conditions(context, config)
    if not can_trade:
        cancellation_reasons.extend(market_reasons)
    
    if cancellation_reasons:
        return PositionManagerResult(
            position_plan=None,
            sizing_result=None,
            can_trade=False,
            cancellation_reasons=cancellation_reasons,
            diversification_guard=diversification_guard,
        )
    
    # Get current price and ATR
    entry_price = context.current_price
    atr = context.indicators.get("atr", 0)
    
    if not atr or atr == 0:
        cancellation_reasons.append("No valid ATR available for TP/SL calculation")
        return PositionManagerResult(
            position_plan=None,
            sizing_result=None,
            can_trade=False,
            cancellation_reasons=cancellation_reasons,
            diversification_guard=diversification_guard,
        )
    
    # Calculate TP/SL levels
    is_long = signal_direction == "long"
    tp_sl_levels = calculate_tp_sl_levels(
        entry_price=entry_price,
        is_long=is_long,
        atr_value=atr,
        tp1_multiplier=config.tp1_multiplier,
        tp2_multiplier=config.tp2_multiplier,
        tp3_multiplier=config.tp3_multiplier,
        sl_multiplier=config.sl_multiplier,
    )
    
    # Calculate position size based on risk
    sizing_result = calculate_risk_based_position_size(
        entry_price=entry_price,
        stop_loss=tp_sl_levels["sl"],
        account_balance=account_balance,
        risk_per_trade_pct=config.max_risk_per_trade_pct,
        leverage=config.default_leverage,
        commission_rate=config.commission_rate,
    )
    
    # Apply maximum position size limit
    final_position_size = min(sizing_result.position_size_usd, config.max_position_size_usd)
    if final_position_size != sizing_result.position_size_usd:
        sizing_result.sizing_factors["size_limited"] = True
        sizing_result.sizing_factors["original_size"] = sizing_result.position_size_usd
        sizing_result.position_size_usd = final_position_size
        
        # Recalculate other metrics with reduced size
        notional_value = final_position_size * config.default_leverage
        sizing_result.quantity = notional_value / entry_price
        sizing_result.commission_cost = notional_value * config.commission_rate
    
    # Estimate holding horizon
    holding_horizon = estimate_holding_horizon(context, config, signal_direction)
    
    # Create position plan
    position_plan = PositionPlan(
        entry_price=entry_price,
        stop_loss=tp_sl_levels["sl"],
        take_profit_levels=[tp_sl_levels["tp1"], tp_sl_levels["tp2"], tp_sl_levels["tp3"]],
        position_size_usd=final_position_size,
        leverage=config.default_leverage,
        direction=signal_direction,
        notes=f"Holding horizon: {holding_horizon} bars",
        metadata={
            "atr": atr,
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
    
    # Add position to diversification guard
    if diversification_guard:
        diversification_guard.add_position(signal_direction, context.symbol)
    
    return PositionManagerResult(
        position_plan=position_plan,
        sizing_result=sizing_result,
        can_trade=True,
        holding_horizon_bars=holding_horizon,
        diversification_guard=diversification_guard,
        metadata={
            "signal_direction": signal_direction,
            "account_balance": account_balance,
            "final_position_size": final_position_size,
        },
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