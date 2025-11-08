"""Trading system core package with lazy attribute loading to avoid circular imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, Tuple

_MODULE_EXPORTS: Dict[str, Tuple[str, ...]] = {
    "indicator_collector.timeframes": ("Timeframe", "validate_timeframe"),
    "indicator_collector.trading_system.interfaces": (
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
    ),
    "indicator_collector.trading_system.volume_orderbook_analyzer": (
        "analyze_volume_orderbook",
        "calculate_mm_confidence_weighted",
        "calculate_order_imbalance",
        "analyze_smart_money_activity",
        "detect_liquidity_zones",
    ),
    "indicator_collector.trading_system.technical_analysis": (
        "analyze_technical_factors",
        "analyze_macd",
        "analyze_rsi",
        "analyze_atr",
        "analyze_bollinger_bands",
        "detect_divergences",
    ),
    "indicator_collector.trading_system.sentiment_analyzer": (
        "analyze_sentiment_factors",
        "create_sentiment_factor_score",
    ),
    "indicator_collector.trading_system.multitimeframe_analyzer": (
        "analyze_multitimeframe_factors",
        "create_multitimeframe_factor_score",
    ),
    "indicator_collector.trading_system.signal_generator": (
        "SignalConfig",
        "SignalFactors",
        "generate_trading_signal",
    ),
    "indicator_collector.trading_system.position_manager": (
        "PositionManagerConfig",
        "DiversificationGuard",
        "PositionSizingResult",
        "PositionManagerResult",
        "calculate_risk_based_position_size",
        "assess_market_conditions",
        "estimate_holding_horizon",
        "create_position_plan",
        "create_diversification_guard",
        "validate_tp_sl_spacing",
    ),
    "indicator_collector.trading_system.statistics_optimizer": (
        "SignalOutcome",
        "PerformanceKPIs",
        "WeightAdjustment",
        "OptimizationResult",
        "StatsOptimizerConfig",
        "StatisticsOptimizer",
        "create_stats_optimizer",
        "create_synthetic_outcomes",
    ),
    "indicator_collector.trading_system.trading_system": (
        "MacroBlackoutConfig",
        "TradingConfig",
        "TradingState",
        "TradingOrchestrator",
        "MacroBlackoutFilter",
        "create_trading_orchestrator",
        "create_default_config",
    ),
    "indicator_collector.trading_system.backtester": (
        "BacktestConfig",
        "ParameterSet",
        "BacktestResult",
        "AdaptiveWeightResult",
        "Backtester",
        "indicator_defaults_for",
    ),
    "indicator_collector.trading_system.data_sources": (
        "HistoricalDataSource",
        "BinanceKlinesSource",
        "normalize_timestamp",
        "validate_timestamps_monotonic",
        "validate_no_future_timestamps",
    ),
    "indicator_collector.trading_system.adaptive_weights": (
        "AdaptiveWeightConfig",
        "WeightPerformance",
        "AdaptationReport",
        "AdaptiveWeightManager",
    ),
    "indicator_collector.trading_system.payload_loader": (
        "PayloadProcessor",
        "load_full_payload",
        "load_and_process_payload_dict",
        "validate_and_normalize_payload",
        "extract_trading_context",
        "payload_processor",
    ),
    "indicator_collector.trading_system.generate_signals": (
        "generate_signals",
        "generate_signals_from_payload",
    ),
    "indicator_collector.trading_system.automated_signals": (
        "AutomatedSignalResult",
        "build_payload_from_candles",
        "run_automated_signal_flow",
    ),
    "indicator_collector.trading_system.signal_schema": (
        "TradingSignalSchema",
        "validate_signal_json",
        "is_valid_signal_structure",
    ),
}

_NAME_TO_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str) -> Any:
    try:
        module_name = _NAME_TO_MODULE[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(f"module 'indicator_collector.trading_system' has no attribute {name!r}") from exc

    module = import_module(module_name)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> Iterable[str]:
    return sorted(set(globals()) | set(__all__))
