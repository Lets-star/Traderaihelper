"""Indicator metrics collection toolkit for TradingView PineScript indicators."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

__all__ = [
    "main",
    "Timeframe",
    "validate_timeframe",
    "indicator_defaults_for",
    "load_full_payload",
    "load_and_process_payload_dict",
    "validate_and_normalize_payload",
    "payload_processor",
    "generate_signals",
    "generate_signals_from_payload",
]

_NAME_TO_MODULE: Dict[str, str] = {
    "main": "indicator_collector.cli",
    "Timeframe": "indicator_collector.timeframes",
    "validate_timeframe": "indicator_collector.timeframes",
    "indicator_defaults_for": "indicator_collector.trading_system.backtester",
    "load_full_payload": "indicator_collector.trading_system.payload_loader",
    "load_and_process_payload_dict": "indicator_collector.trading_system.payload_loader",
    "validate_and_normalize_payload": "indicator_collector.trading_system.payload_loader",
    "payload_processor": "indicator_collector.trading_system.payload_loader",
    "generate_signals": "indicator_collector.trading_system.generate_signals",
    "generate_signals_from_payload": "indicator_collector.trading_system.generate_signals",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _NAME_TO_MODULE[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise AttributeError(f"module 'indicator_collector' has no attribute {name!r}") from exc

    module = import_module(module_name)
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
