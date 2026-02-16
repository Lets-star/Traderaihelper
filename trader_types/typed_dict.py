"""
TypedDict definitions for structured data types.

TypedDict provides type hints for dictionary structures with:
- Required and optional fields
- Specific types for each field
- Better IDE autocomplete and type checking
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
from typing import TypedDict, NotRequired, Required


class SignalPayload(TypedDict, total=False):
    """Trading signal payload structure."""

    signal_id: Required[str]
    symbol: Required[str]
    direction: Required[str]  # "LONG" or "SHORT"
    entry_price: Required[float]
    signal_type: NotRequired[str]  # "BUY", "SELL", or "HOLD"
    take_profit: NotRequired[Union[float, Dict[str, float]]]
    stop_loss: NotRequired[float]
    confidence: NotRequired[float]
    leverage: NotRequired[float]
    quantity: NotRequired[float]
    generated_at: NotRequired[int]  # Unix timestamp in milliseconds
    entries: NotRequired[List[float]]
    take_profits: NotRequired[Dict[str, float]]
    metadata: NotRequired[Dict[str, object]]
    indicators: NotRequired[Dict[str, object]]


class KlineData(TypedDict, total=False):
    """Kline/candlestick data structure."""

    ts: Required[int]  # Open timestamp in milliseconds
    open: Required[float]
    high: Required[float]
    low: Required[float]
    close: Required[float]
    volume: Required[float]
    close_time: NotRequired[int]  # Close timestamp in milliseconds
    quote_volume: NotRequired[float]
    trades: NotRequired[int]
    taker_buy_base: NotRequired[float]
    taker_buy_quote: NotRequired[float]
    # Additional fields for WebSocket data
    is_closed: NotRequired[bool]
    symbol: NotRequired[str]
    interval: NotRequired[str]


class FactorWeight(TypedDict, total=False):
    """Factor weight configuration."""

    technical: NotRequired[float]
    sentiment: NotRequired[float]
    multitimeframe: NotRequired[float]
    volume: NotRequired[float]
    market_structure: NotRequired[float]
    composite: NotRequired[float]


class ExecutionResult(TypedDict, total=False):
    """Signal execution result."""

    status: Required[str]  # "filled", "error", "validation_error", "disabled"
    error: NotRequired[str]
    response_code: NotRequired[int]
    latency_ms: NotRequired[float]
    validation_errors: NotRequired[List[str]]
    order_id: NotRequired[str]
    filled_qty: NotRequired[float]
    filled_price: NotRequired[float]


class PositionData(TypedDict, total=False):
    """Position data structure."""

    symbol: Required[str]
    side: Required[str]  # "Buy" or "Sell"
    size: Required[float]
    entry_price: Required[float]
    mark_price: NotRequired[float]
    unrealized_pnl: NotRequired[float]
    leverage: NotRequired[float]
    liquidation_price: NotRequired[float]
    created_at: NotRequired[int]
    updated_at: NotRequired[int]


class OrderPayload(TypedDict, total=False):
    """Order placement payload."""

    symbol: Required[str]
    side: Required[str]  # "Buy" or "Sell"
    order_type: Required[str]  # "Market", "Limit", etc.
    qty: Required[str]
    price: NotRequired[str]
    take_profit: NotRequired[str]
    stop_loss: NotRequired[str]
    client_order_id: NotRequired[str]
    reduce_only: NotRequired[bool]
    close_on_trigger: NotRequired[bool]


class ChartConfig(TypedDict, total=False):
    """Chart configuration."""

    symbol: Required[str]
    timeframe: Required[str]
    num_bars: NotRequired[int]
    show_indicators: NotRequired[bool]
    height: NotRequired[int]
    theme: NotRequired[str]
    show_volume: NotRequired[bool]
    show_forming_bar: NotRequired[bool]


class WorkerConfig(TypedDict, total=False):
    """Background worker configuration."""

    symbol: Required[str]
    timeframe: Required[str]
    update_bus: NotRequired[object]  # UpdateBus instance
    num_bars: NotRequired[int]
    use_websocket: NotRequired[bool]
    poll_interval_ms: NotRequired[int]


class WebSocketMessage(TypedDict, total=False):
    """WebSocket message structure."""

    type: Required[str]  # "kline", "trade", "error", etc.
    symbol: NotRequired[str]
    interval: NotRequired[str]
    data: NotRequired[Dict[str, object]]
    timestamp: NotRequired[int]
    error: NotRequired[str]


class UpdateMessage(TypedDict, total=False):
    """Update bus message structure."""

    type: Required[str]
    signal_id: NotRequired[str]
    status: NotRequired[str]
    timestamp: NotRequired[float]
    thread_id: NotRequired[Optional[int]]
    latency_ms: NotRequired[float]
    error: NotRequired[str]
    response_code: NotRequired[int]
    data: NotRequired[Dict[str, object]]


class IndicatorValues(TypedDict, total=False):
    """Technical indicator values."""

    rsi: NotRequired[float]
    ema_9: NotRequired[float]
    ema_21: NotRequired[float]
    ema_50: NotRequired[float]
    sma_200: NotRequired[float]
    macd: NotRequired[float]
    macd_signal: NotRequired[float]
    macd_hist: NotRequired[float]
    bb_upper: NotRequired[float]
    bb_middle: NotRequired[float]
    bb_lower: NotRequired[float]
    atr: NotRequired[float]
    volume_sma: NotRequired[float]
    volume_ratio: NotRequired[float]


class SignalAnalysis(TypedDict, total=False):
    """Signal analysis result."""

    signal_id: Required[str]
    symbol: Required[str]
    direction: Required[str]
    confidence: Required[float]
    strength: Required[str]  # "STRONG", "MODERATE", "WEAK"
    factors: NotRequired[Dict[str, float]]
    indicators: NotRequired[IndicatorValues]
    risk_reward: NotRequired[float]
    recommended_position_size: NotRequired[float]


class HealthStatus(TypedDict, total=False):
    """Health check status."""

    component: Required[str]
    status: Required[str]  # "healthy", "degraded", "unhealthy"
    message: NotRequired[str]
    response_time_ms: NotRequired[float]
    checked_at: NotRequired[int]
    details: NotRequired[Dict[str, object]]


class CacheEntry(TypedDict, total=False):
    """Cache entry metadata."""

    key: Required[str]
    created_at: Required[int]
    expires_at: NotRequired[int]
    access_count: NotRequired[int]
    last_accessed: NotRequired[int]
