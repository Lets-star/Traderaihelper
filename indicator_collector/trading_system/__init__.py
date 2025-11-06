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
from .volume_orderbook_analyzer import (
    analyze_volume_orderbook,
    calculate_mm_confidence_weighted,
    calculate_order_imbalance,
    analyze_smart_money_activity,
    detect_liquidity_zones,
)
from .technical_analysis import (
    analyze_technical_factors,
    analyze_macd,
    analyze_rsi,
    analyze_atr,
    analyze_bollinger_bands,
    detect_divergences,
)
from .sentiment_analyzer import (
    analyze_sentiment_factors,
    create_sentiment_factor_score,
)
from .multitimeframe_analyzer import (
    analyze_multitimeframe_factors,
    create_multitimeframe_factor_score,
)
from .signal_generator import (
    SignalConfig,
    SignalFactors,
    generate_trading_signal,
)

__all__ = [
    "AnalyzerContext",
    "FactorScore",
    "JsonDict",
    "OptimizationStats",
    "PositionPlan",
    "SignalExplanation",
    "SignalConfig",
    "SignalFactors",
    "TradingAnalyzer",
    "TradingSignalPayload",
    "deserialize_signal_payload",
    "parse_collector_payload",
    "serialize_signal_payload",
    "analyze_volume_orderbook",
    "calculate_mm_confidence_weighted",
    "calculate_order_imbalance",
    "analyze_smart_money_activity",
    "detect_liquidity_zones",
    "analyze_technical_factors",
    "analyze_macd",
    "analyze_rsi",
    "analyze_atr",
    "analyze_bollinger_bands",
    "detect_divergences",
    "analyze_sentiment_factors",
    "create_sentiment_factor_score",
    "analyze_multitimeframe_factors",
    "create_multitimeframe_factor_score",
    "generate_trading_signal",
]
