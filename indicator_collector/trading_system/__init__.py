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
from .market_structure_analyzer import (
    analyze_market_structure,
    calculate_structure_score,
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
    "analyze_market_structure",
    "calculate_structure_score",
    "deserialize_signal_payload",
    "parse_collector_payload",
    "serialize_signal_payload",
]
