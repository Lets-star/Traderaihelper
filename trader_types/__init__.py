"""
Type definitions module for improved type safety.

This module provides:
- Protocol types for duck typing (Streamlit components, callbacks)
- TypedDict for structured data (signals, klines, execution results)
- Enum constants for trading (timeframes, signal directions)
- TypeGuard functions for runtime type checking
- Generic types for reusable containers

Example:
    from trader_types import SignalPayload, StreamlitComponent
    from trader_types import Timeframe, is_valid_signal
"""

from __future__ import annotations

# Protocol types
from trader_types.protocols import (
    StreamlitComponent,
    SessionState,
    UpdateCallback,
    SignalExecutorProtocol,
    KlineCallback,
    ChartRenderer,
)

# TypedDict types
from trader_types.typed_dict import (
    SignalPayload,
    KlineData,
    FactorWeight,
    ExecutionResult,
    PositionData,
    OrderPayload,
    ChartConfig,
    WorkerConfig,
    WebSocketMessage,
)

# Enum constants
from trader_types.enums import (
    Timeframe,
    SignalDirection,
    SignalStrength,
    FactorCategory,
    ExecutionStatus,
    WorkerStatus,
    WebSocketStatus,
    OrderSide,
    OrderType,
    OrderStatus,
    HealthStatus,
)

# Type guards
from trader_types.type_guards import (
    is_valid_signal,
    is_kline_data,
    is_streamlit_component,
    is_execution_result,
    is_position_data,
    is_valid_symbol,
    is_valid_leverage,
    is_valid_confidence,
)

# Generic types
from trader_types.generics import (
    UpdateBus,
    Result,
    DataStore,
)

__all__ = [
    # Protocols
    "StreamlitComponent",
    "SessionState",
    "UpdateCallback",
    "SignalExecutorProtocol",
    "KlineCallback",
    "ChartRenderer",
    # TypedDict
    "SignalPayload",
    "KlineData",
    "FactorWeight",
    "ExecutionResult",
    "PositionData",
    "OrderPayload",
    "ChartConfig",
    "WorkerConfig",
    "WebSocketMessage",
    # Enums
    "Timeframe",
    "SignalDirection",
    "SignalStrength",
    "FactorCategory",
    "ExecutionStatus",
    "WorkerStatus",
    "WebSocketStatus",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "HealthStatus",
    # Type guards
    "is_valid_signal",
    "is_kline_data",
    "is_streamlit_component",
    "is_execution_result",
    "is_position_data",
    "is_valid_symbol",
    "is_valid_leverage",
    "is_valid_confidence",
    # Generics
    "UpdateBus",
    "Result",
    "DataStore",
]
