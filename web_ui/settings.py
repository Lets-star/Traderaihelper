"""Settings and constants for the web UI package.

This module contains all constants, configuration values, and UI helper functions
that are used throughout the web UI package.

Following SRP (Single Responsibility Principle), this module is responsible for:
- Defining UI constants (POPULAR_TOKENS, TIMEFRAMES, etc.)
- Providing UI helper functions (ui_key, num_int, num_float, etc.)
- Formatting functions (format_correlation, format_flow, etc.)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

# =============================================================================
# UI Constants
# =============================================================================

POPULAR_TOKENS = [
    "BINANCE:BTCUSDT",
    "BINANCE:ETHUSDT",
    "BINANCE:BNBUSDT",
    "BINANCE:SOLUSDT",
    "BINANCE:ADAUSDT",
    "BINANCE:XRPUSDT",
    "BINANCE:DOGEUSDT",
    "BINANCE:DOTUSDT",
    "BINANCE:MATICUSDT",
    "BINANCE:AVAXUSDT",
]

TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]

AUTOMATED_SIGNALS_STATE_KEY = "automated_signals_state"

CACHE_VERSION = "binance-real-data-v1"

SYNTHETIC_FLAG_KEYS = {"is_synthetic", "synthetic", "mock", "demo", "paper", "testnet"}
SYNTHETIC_SOURCE_VALUES = {"sample", "demo", "mock", "paper", "testnet", "local"}
SYNTHETIC_MARKER_VALUES = {
    "mock",
    "test",
    "demo",
    "simulated",
    "synthetic",
    "fake",
    "sample",
    "paper",
    "backtest",
    "historical_sim",
    "generated",
    "artificial",
}

FACTOR_CATEGORY_ORDER = [
    "technical",
    "sentiment",
    "multitimeframe",
    "volume",
    "market_structure",
    "composite",
]

FACTOR_NAME_TO_CATEGORY = {
    "technical_analysis": "technical",
    "technical": "technical",
    "sentiment_analysis": "sentiment",
    "sentiment": "sentiment",
    "multitimeframe_alignment": "multitimeframe",
    "multitimeframe": "multitimeframe",
    "volume_analysis": "volume",
    "volume": "volume",
    "market_structure": "market_structure",
    "structure": "market_structure",
    "composite_analysis": "composite",
    "composite": "composite",
}

FACTOR_CATEGORY_LABELS = {
    "technical": "Technical",
    "sentiment": "Sentiment",
    "multitimeframe": "Multi-timeframe",
    "volume": "Volume",
    "market_structure": "Market Structure",
    "composite": "Composite",
}


# =============================================================================
# UI Helper Functions
# =============================================================================

def format_correlation(value: float) -> str:
    """Format correlation value with color coding."""
    if value > 0.7:
        return f"🟢 {value:.3f}"
    elif value > 0.3:
        return f"🟡 {value:.3f}"
    elif value > -0.3:
        return f"⚪ {value:.3f}"
    elif value > -0.7:
        return f"🟠 {value:.3f}"
    else:
        return f"🔴 {value:.3f}"


def format_flow(value: float) -> str:
    """Format flow value with color coding."""
    if abs(value) < 1000:
        return f"⚪ ${value:,.0f}"
    elif value > 0:
        return f"🟢 ${value:,.0f}"
    else:
        return f"🔴 ${value:,.0f}"


def ui_key(prefix: str, label: str) -> str:
    """Generate a unique key for Streamlit widgets.
    
    Args:
        prefix: The prefix for the key (e.g., tab name or section)
        label: The widget label
        
    Returns:
        A standardized unique key combining prefix and label
    """
    label_slug = label.lower().replace(" ", "_").replace("%", "pct")
    return f"{prefix}_{label_slug}"


def stable_hash(payload: Any) -> str:
    """Create a stable SHA1 hash for caching and change detection."""
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str)
    except TypeError:
        serialized = json.dumps(str(payload), sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def num_int(
    label: str,
    *,
    min_v: int,
    value: int,
    max_v: Optional[int] = None,
    step: int = 1,
    key: Optional[str] = None,
    ui: Optional[Any] = None,
    help_text: Optional[str] = None,
    format_str: Optional[str] = "%d",
) -> int:
    """Render an integer-based number input ensuring consistent typing."""
    try:
        import streamlit as st
    except ImportError:
        return value
    
    target = ui if ui is not None else st
    kwargs = {
        "min_value": int(min_v),
        "value": int(value),
        "step": int(step),
    }
    if max_v is not None:
        kwargs["max_value"] = int(max_v)
    if key is not None:
        kwargs["key"] = key
    if help_text is not None:
        kwargs["help"] = help_text
    if format_str is not None:
        kwargs["format"] = format_str
    return int(target.number_input(label, **kwargs))


def num_float(
    label: str,
    *,
    min_v: float,
    value: float,
    max_v: Optional[float] = None,
    step: float = 0.1,
    key: Optional[str] = None,
    ui: Optional[Any] = None,
    help_text: Optional[str] = None,
    format_str: Optional[str] = None,
) -> float:
    """Render a float-based number input ensuring consistent typing."""
    try:
        import streamlit as st
    except ImportError:
        return value
    
    target = ui if ui is not None else st
    kwargs = {
        "min_value": float(min_v),
        "value": float(value),
        "step": float(step),
    }
    if max_v is not None:
        kwargs["max_value"] = float(max_v)
    if key is not None:
        kwargs["key"] = key
    if help_text is not None:
        kwargs["help"] = help_text
    if format_str is not None:
        kwargs["format"] = format_str
    return float(target.number_input(label, **kwargs))


def format_category_label(category: str) -> str:
    """Format a factor category key into a human-readable label."""
    return FACTOR_CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def normalize_factor_category(name: Optional[str]) -> Optional[str]:
    """Normalize a factor category name to standard form."""
    if not name:
        return None
    key = str(name).lower()
    return FACTOR_NAME_TO_CATEGORY.get(key, key)


def normalize_category_weights(weights: Optional[Dict[str, Any]]) -> tuple[Dict[str, float], Dict[str, float]]:
    """Normalize category weights to sum to 1.0.
    
    Returns:
        Tuple of (normalized_weights, raw_values)
    """
    raw_values: Dict[str, float] = {}
    if isinstance(weights, dict):
        for category in FACTOR_CATEGORY_ORDER:
            value = weights.get(category)
            if value is None:
                continue
            try:
                raw_values[category] = float(value)
            except (TypeError, ValueError):
                continue
    total = sum(raw_values.values())
    normalized: Dict[str, float] = {}
    if total > 0:
        needs_normalization = abs(total - 1.0) > 1e-6
        for category, value in raw_values.items():
            normalized[category] = value / total if needs_normalization else value
    for category in FACTOR_CATEGORY_ORDER:
        normalized.setdefault(category, 0.0)
        raw_values.setdefault(category, 0.0)
    return normalized, raw_values


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float with a default fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "POPULAR_TOKENS",
    "TIMEFRAMES",
    "AUTOMATED_SIGNALS_STATE_KEY",
    "CACHE_VERSION",
    "SYNTHETIC_FLAG_KEYS",
    "SYNTHETIC_SOURCE_VALUES",
    "SYNTHETIC_MARKER_VALUES",
    "FACTOR_CATEGORY_ORDER",
    "FACTOR_NAME_TO_CATEGORY",
    "FACTOR_CATEGORY_LABELS",
    # UI Helpers
    "format_correlation",
    "format_flow",
    "ui_key",
    "stable_hash",
    "num_int",
    "num_float",
    "format_category_label",
    "normalize_factor_category",
    "normalize_category_weights",
    "safe_float",
]
