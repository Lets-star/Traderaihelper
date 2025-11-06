"""Trading system core package."""

from .interfaces import (
    AnalyzerContext,
    FactorScore,
    JsonDict,
    OptimizationStats,
    PositionPlan,
    SignalExplanation,
    TradingAnalyzer,
    TradingSignalPayload,
    deserialize_signal_payload,
    parse_collector_payload,
    serialize_signal_payload,
)

__all__ = [
    "AnalyzerContext",
    "FactorScore",
    "JsonDict",
    "OptimizationStats",
    "PositionPlan",
    "SignalExplanation",
    "TradingAnalyzer",
    "TradingSignalPayload",
    "deserialize_signal_payload",
    "parse_collector_payload",
    "serialize_signal_payload",
]
