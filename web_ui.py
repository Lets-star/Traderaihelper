#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import numpy as np

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for the web UI. Please install it via 'pip install pandas'.") from e

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest_utils import simulate_backtest
from config_store import ConfigStore
from indicator_collector.collector import collect_metrics
from indicator_collector.indicator_metrics import SimulationSummary
from indicator_collector.real_data_validator import DataValidationError
from indicator_collector.time_series import TimeframeSeries
from indicator_collector.timeframes import Timeframe
from indicator_collector.trade_signals import calculate_position_metrics, calculate_tp_sl_levels
from indicator_collector.trading_system.backtester import (
    DEFAULT_SIGNAL_THRESHOLDS,
    indicator_defaults_for,
)
from automated_signals_worker import AutomatedSignalsWorker
from indicator_collector.trading_system.automated_signals import run_automated_signal_flow
from indicator_collector.trading_system.auto_analyze_worker import (
    AutoAnalyzeWorker,
    floor_closed_bar,
    get_binance_server_time_ms,
    run_analysis,
)
from indicator_collector.trading_system.data_sources.timestamp_utils import normalize_timestamp
from indicator_collector.trading_system.signal_generator import SignalConfig
from indicator_collector.trading_system.signal_schema import is_valid_signal_structure
from signal_executor import SignalExecutor
from update_bus import UpdateBus
from worker_manager import ChartWorkerManager, SignalsWorkerManager

logger = logging.getLogger(__name__)


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


st.set_page_config(
    page_title="Token Charts & Indicators",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def format_category_label(category: str) -> str:
    return FACTOR_CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def normalize_factor_category(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = str(name).lower()
    return FACTOR_NAME_TO_CATEGORY.get(key, key)


def normalize_category_weights(weights: Optional[Dict[str, Any]]) -> tuple[Dict[str, float], Dict[str, float]]:
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_payload_for_real_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove known synthetic markers and enforce Binance metadata."""
    if not isinstance(payload, dict):
        return payload

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            keys_to_remove = []
            for key, value in obj.items():
                key_lower = key.lower()
                if key_lower in SYNTHETIC_FLAG_KEYS:
                    keys_to_remove.append(key)
                    continue

                if isinstance(value, (dict, list)):
                    obj[key] = _clean(value)
                elif isinstance(value, str):
                    value_lower = value.lower()
                    if key_lower in {"source", "exchange"} and value_lower in SYNTHETIC_SOURCE_VALUES:
                        obj[key] = "binance"
                    elif any(marker in value_lower for marker in SYNTHETIC_MARKER_VALUES):
                        obj[key] = "real_market_data"
                    else:
                        obj[key] = value
                else:
                    obj[key] = value

            for key in keys_to_remove:
                obj.pop(key, None)
            return obj
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        if isinstance(obj, str):
            value_lower = obj.lower()
            if any(marker in value_lower for marker in SYNTHETIC_MARKER_VALUES):
                return "real_market_data"
        return obj

    _clean(payload)

    metadata = payload.setdefault("metadata", {})
    metadata["source"] = "binance"
    metadata["exchange"] = metadata.get("exchange", "binance") or "binance"
    metadata["real_data"] = True
    metadata["is_real_data"] = True
    metadata["real_data_validated"] = True
    metadata["data_quality"] = "validated_real_data"

    return payload


@st.cache_data(ttl=300)
def cached_run_automated_signals(
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    params_hash: str,
    weights_hash: str,
    data_version: str,
    signal_config_json: str,
    indicator_params_json: str,
    signal_params_json: str,
) -> Dict[str, Any]:
    """Cache Binance signal generation results for performance."""
    _ = data_version  # ensure cache invalidation when upstream data changes
    start_dt = dt.datetime.fromisoformat(start_iso)
    end_dt = dt.datetime.fromisoformat(end_iso)

    signal_config_payload = json.loads(signal_config_json) if signal_config_json else {}
    indicator_params = json.loads(indicator_params_json) if indicator_params_json else {}
    signal_params = json.loads(signal_params_json) if signal_params_json else {}

    weights = signal_config_payload.get("weights", {})
    signal_config = SignalConfig(
        technical_weight=weights.get("technical", 0.25),
        sentiment_weight=weights.get("sentiment", 0.15),
        multitimeframe_weight=weights.get("multitimeframe", 0.10),
        volume_weight=weights.get("volume", 0.20),
        structure_weight=weights.get("market_structure", 0.15),
        composite_weight=weights.get("composite", 0.0),
        min_factors_confirm=int(signal_config_payload.get("min_confirmations", 3)),
        buy_threshold=float(signal_config_payload.get("buy_threshold", 0.65)),
        sell_threshold=float(signal_config_payload.get("sell_threshold", 0.35)),
        min_confidence=float(signal_config_payload.get("min_confidence", 0.6)),
    )

    indicator_periods = indicator_params.get("rsi", {})
    atr_period = int(indicator_params.get("atr", {}).get("period", 14))
    macd_slow = int(indicator_params.get("macd", {}).get("slow", 26))
    macd_signal = int(indicator_params.get("macd", {}).get("signal", 9))
    rsi_period = int(indicator_periods.get("period", 14))
    min_candles = max(
        30,
        rsi_period + 2,
        atr_period + 2,
        macd_slow + macd_signal,
    )

    result = run_automated_signal_flow(
        symbol,
        timeframe,
        start_dt,
        end_dt,
        validate_real_data=True,
        signal_config=signal_config,
        indicator_params=indicator_params,
        signal_params=signal_params,
        min_candles=min_candles,
    )
    return {
        "candles": result.candles,
        "processed_payload": result.processed_payload,
        "explicit_signal": result.explicit_signal,
        "params_hash": params_hash,
        "weights_hash": weights_hash,
    }


@st.cache_data(ttl=300)
def load_indicator_data(symbol: str, timeframe: str, period: int, token: str, cache_version: str) -> tuple:
    _ = cache_version  # Ensures cache invalidation when version changes
    result = collect_metrics(
        symbol=symbol,
        timeframe=timeframe,
        period=period,
        token=token,
    )
    return result.summary, result.payload, result.main_series


def _rolling_equals(series: pd.Series, window: int, *, method: str) -> pd.Series:
    """Return a boolean mask where values equal the rolling extremum within tolerance."""
    if method not in {"min", "max"}:
        raise ValueError("method must be 'min' or 'max'")
    rolling_window = getattr(series.rolling(window, min_periods=1), method)()
    series_values = series.to_numpy()
    rolling_values = rolling_window.to_numpy()
    mask = (~np.isnan(series_values)) & (~np.isnan(rolling_values))
    result = np.zeros(series_values.shape, dtype=bool)
    result[mask] = np.isclose(series_values[mask], rolling_values[mask], rtol=1e-5, atol=1e-8)
    return pd.Series(result, index=series.index)


def calculate_better_volume_indicator(
    df: pd.DataFrame,
    length: int = 8,
    *,
    use_two_bars: bool = True,
    low_vol_enabled: bool = True,
    climax_up_enabled: bool = True,
    climax_down_enabled: bool = True,
    churn_enabled: bool = True,
    climax_churn_enabled: bool = True,
) -> tuple[pd.Series, list[str]]:
    """Compute BVI bar colors and the accompanying volume average series."""
    if df.empty:
        return pd.Series(dtype=float), []

    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    true_range = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1, skipna=True)

    denom_base = (2.0 + (true_range.pow(2) / 10.0)) * true_range
    denom_up = denom_base + (open_ - close)
    denom_down = denom_base + (close - open_)
    ratio_up = (true_range / denom_up).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    numerator_down = true_range + close - open_
    ratio_down = (numerator_down / denom_down).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    v1 = pd.Series(
        np.where(close >= open_, volume * ratio_up, volume * ratio_down),
        index=df.index,
    )
    v2 = volume - v1
    v3 = v1 + v2

    v4 = v1 * true_range
    v5 = (v1 - v2) * true_range
    v6 = v2 * true_range
    v7 = (v2 - v1) * true_range

    true_range_values = true_range.to_numpy()
    v1_values = v1.to_numpy()
    v2_values = v2.to_numpy()
    v3_values = v3.to_numpy()
    non_zero_range = true_range_values != 0.0

    v8 = pd.Series(
        np.divide(v1_values, true_range_values, out=np.ones_like(v1_values), where=non_zero_range),
        index=df.index,
    )
    v9 = pd.Series(
        np.divide(v1_values - v2_values, true_range_values, out=np.ones_like(v1_values), where=non_zero_range),
        index=df.index,
    )
    v10 = pd.Series(
        np.divide(v2_values, true_range_values, out=np.ones_like(v1_values), where=non_zero_range),
        index=df.index,
    )
    v11 = pd.Series(
        np.divide(v2_values - v1_values, true_range_values, out=np.ones_like(v1_values), where=non_zero_range),
        index=df.index,
    )
    v12 = pd.Series(
        np.divide(v3_values, true_range_values, out=np.ones_like(v1_values), where=non_zero_range),
        index=df.index,
    )

    if use_two_bars:
        v13 = v3 + v3.shift(1)
        highest_high = high.rolling(2, min_periods=1).max()
        lowest_low = low.rolling(2, min_periods=1).min()
        range_two = highest_high - lowest_low

        v14 = (v1 + v1.shift(1)) * range_two
        v15 = (v1 + v1.shift(1) - v2 - v2.shift(1)) * range_two
        v16 = (v2 + v2.shift(1)) * range_two
        v17 = (v2 + v2.shift(1) - v1 - v1.shift(1)) * range_two

        range_two_values = range_two.to_numpy()
        non_zero_range_two = range_two_values != 0.0
        sum_v1 = (v1 + v1.shift(1)).to_numpy()
        sum_v2 = (v2 + v2.shift(1)).to_numpy()
        diff_v12 = (v1 + v1.shift(1) - v2 - v2.shift(1)).to_numpy()
        diff_v21 = (v2 + v2.shift(1) - v1 - v1.shift(1)).to_numpy()
        v18 = pd.Series(
            np.divide(sum_v1, range_two_values, out=np.ones_like(v1_values), where=non_zero_range_two),
            index=df.index,
        )
        v19 = pd.Series(
            np.divide(diff_v12, range_two_values, out=np.ones_like(v1_values), where=non_zero_range_two),
            index=df.index,
        )
        v20 = pd.Series(
            np.divide(sum_v2, range_two_values, out=np.ones_like(v1_values), where=non_zero_range_two),
            index=df.index,
        )
        v21 = pd.Series(
            np.divide(diff_v21, range_two_values, out=np.ones_like(v1_values), where=non_zero_range_two),
            index=df.index,
        )
        v22 = pd.Series(
            np.divide(v13.to_numpy(), range_two_values, out=np.ones_like(v1_values), where=non_zero_range_two),
            index=df.index,
        )
    else:
        v13 = v14 = v15 = v16 = v17 = v18 = v19 = v20 = v21 = v22 = pd.Series(1.0, index=df.index)

    close_greater_open = close > open_
    close_less_open = close < open_
    prev_close = close.shift(1)
    prev_open = open_.shift(1)
    prev_close_greater_prev_open = prev_close > prev_open
    prev_close_less_prev_open = prev_close < prev_open

    c1 = _rolling_equals(v3, length, method="min")
    c2 = _rolling_equals(v4, length, method="max") & close_greater_open
    c3 = _rolling_equals(v5, length, method="max") & close_greater_open
    c4 = _rolling_equals(v6, length, method="max") & close_less_open
    c5 = _rolling_equals(v7, length, method="max") & close_less_open
    c6 = _rolling_equals(v8, length, method="min") & close_less_open
    c7 = _rolling_equals(v9, length, method="min") & close_less_open
    c8 = _rolling_equals(v10, length, method="min") & close_greater_open
    c9 = _rolling_equals(v11, length, method="min") & close_greater_open
    c10 = _rolling_equals(v12, length, method="max")

    if use_two_bars:
        c11 = _rolling_equals(v13, length, method="min") & close_greater_open & prev_close_greater_prev_open
        c12 = _rolling_equals(v14, length, method="max") & close_greater_open & prev_close_greater_prev_open
        c13 = _rolling_equals(v15, length, method="max") & close_greater_open & prev_close_less_prev_open
        c14 = _rolling_equals(v16, length, method="min") & close_less_open & prev_close_less_prev_open
        c15 = _rolling_equals(v17, length, method="min") & close_less_open & prev_close_less_prev_open
        c16 = _rolling_equals(v18, length, method="min") & close_less_open & prev_close_less_prev_open
        c17 = _rolling_equals(v19, length, method="min") & close_greater_open & prev_close_less_prev_open
        c18 = _rolling_equals(v20, length, method="min") & close_greater_open & prev_close_greater_prev_open
        c19 = _rolling_equals(v21, length, method="min") & close_greater_open & prev_close_greater_prev_open
        c20 = _rolling_equals(v22, length, method="min")
    else:
        false_series = pd.Series(False, index=df.index)
        c11 = c12 = c13 = c14 = c15 = c16 = c17 = c18 = c19 = c20 = false_series

    low_vol_color = "#FFFF00"
    climax_up_color = "#FF0000"
    climax_down_color = "#FFFFFF"
    churn_color = "#00FF00"
    climax_churn_color = "#8B008B"
    default_color = "#00FFFF"

    climax_up_condition = (c2 | c3 | c8 | c9 | c12 | c13 | c18 | c19) & climax_up_enabled
    climax_down_condition = (c4 | c5 | c6 | c7 | c14 | c15 | c16 | c17) & climax_down_enabled
    churn_condition = ((c10 & churn_enabled) | c20)
    low_vol_condition = (c1 | c11) & low_vol_enabled
    climax_churn_condition = ((c10 | c20) & (c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9) & climax_churn_enabled)

    climax_up_condition_arr = climax_up_condition.to_numpy()
    climax_down_condition_arr = climax_down_condition.to_numpy()
    churn_condition_arr = churn_condition.to_numpy()

    base_color = np.select(
        [
            climax_up_condition_arr,
            (~climax_up_condition_arr) & climax_down_condition_arr,
            (~climax_up_condition_arr) & (~climax_down_condition_arr) & churn_condition_arr,
        ],
        [climax_up_color, climax_down_color, churn_color],
        default=default_color,
    )

    final_color = np.where(
        climax_churn_condition.to_numpy(),
        climax_churn_color,
        np.where(low_vol_condition.to_numpy(), low_vol_color, base_color),
    )

    volume_sma = volume.rolling(length, min_periods=1).mean()

    return volume_sma, final_color.tolist()


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi.clip(0, 100)


def _compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=period, min_periods=1).mean()
    deviation = series.rolling(window=period, min_periods=1).std(ddof=0)
    upper = middle + std_dev * deviation
    lower = middle - std_dev * deviation
    return middle, upper, lower


def create_realtime_candlestick_chart(
    df: pd.DataFrame,
    *,
    show_bvi: bool = True,
    bvi_length: int = 8,
    atr_channels: Optional[Dict[str, Any]] = None,
    order_blocks: Optional[list] = None,
    show_atr_channels: bool = True,
    show_order_blocks: bool = True,
) -> go.Figure:
    if df.empty:
        raise ValueError("Realtime chart dataframe is empty")

    plot_df = df.copy().sort_values("ts").reset_index(drop=True)
    if "timestamp" not in plot_df.columns:
        plot_df["timestamp"] = pd.to_datetime(plot_df["ts"], unit="ms", utc=True)

    timestamps = plot_df["timestamp"]

    rsi_values = _compute_rsi(plot_df["close"])
    macd_line, macd_signal, macd_hist = _compute_macd(plot_df["close"])
    bb_middle, bb_upper, bb_lower = _compute_bollinger_bands(plot_df["close"])

    if show_bvi:
        bvi_input = plot_df[["open", "high", "low", "close", "volume"]].copy()
        volume_sma, volume_colors = calculate_better_volume_indicator(
            bvi_input,
            length=bvi_length,
            use_two_bars=True,
        )
    else:
        volume_sma = plot_df["volume"].rolling(window=20, min_periods=1).mean()
        volume_colors = [
            "#16a34a" if close >= open_ else "#dc2626"
            for close, open_ in zip(plot_df["close"], plot_df["open"])
        ]

    volume_sma_plot = volume_sma.where(volume_sma.notna(), None).tolist()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.20],
        subplot_titles=("Price & Indicators", "RSI", "MACD", "Volume"),
    )

    atr_channels = atr_channels or {}
    order_blocks = order_blocks or []

    fig.add_trace(
        go.Candlestick(
            x=timestamps,
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1,
        col=1,
    )

    atr_color_map = {
        "atr_trend_1x": "#6366F1",
        "atr_trend_3x": "#22C55E",
        "atr_trend_8x": "#F59E0B",
        "atr_trend_21x": "#EF4444",
    }

    if show_atr_channels and atr_channels:
        for key in sorted(atr_channels.keys()):
            channel = atr_channels.get(key) or {}
            color = atr_color_map.get(key, "rgba(255, 255, 255, 0.6)")
            label_base = key.replace("atr_trend_", "").upper()
            upper_series = channel.get("upper")
            lower_series = channel.get("lower")

            if isinstance(upper_series, pd.Series):
                upper_values = upper_series.where(upper_series.notna(), None).tolist()
            elif isinstance(upper_series, (list, tuple, np.ndarray)):
                upper_values = [value if value == value else None for value in upper_series]
            else:
                upper_values = []

            if isinstance(lower_series, pd.Series):
                lower_values = lower_series.where(lower_series.notna(), None).tolist()
            elif isinstance(lower_series, (list, tuple, np.ndarray)):
                lower_values = [value if value == value else None for value in lower_series]
            else:
                lower_values = []

            if upper_values:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=upper_values,
                        name=f"ATR {label_base}Upper",
                        line=dict(color=color, width=1.5),
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )

            if lower_values:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=lower_values,
                        name=f"ATR {label_base}Lower",
                        line=dict(color=color, width=1.5, dash="dot"),
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=bb_upper,
            name="BB Upper",
            line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=bb_middle,
            name="BB Middle",
            line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=bb_lower,
            name="BB Lower",
            line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
            fill="tonexty",
            fillcolor="rgba(173, 216, 230, 0.1)",
        ),
        row=1,
        col=1,
    )

    if show_order_blocks and order_blocks:
        for zone in order_blocks:
            zone_type = zone.get("zone_type")
            created_idx = int(zone.get("created_index", 0))
            created_idx = max(0, min(created_idx, len(plot_df) - 1))
            start_time = timestamps.iloc[created_idx]
            end_time = timestamps.iloc[-1]
            top = zone.get("top")
            bottom = zone.get("bottom")

            if top is None or bottom is None:
                continue

            color = "rgba(34,197,94,0.18)" if str(zone_type).startswith("Bull") else "rgba(251,146,60,0.18)"
            border = "rgba(34,197,94,0.45)" if str(zone_type).startswith("Bull") else "rgba(251,146,60,0.45)"

            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=end_time,
                y0=min(top, bottom),
                y1=max(top, bottom),
                fillcolor=color,
                line=dict(color=border, width=1),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=rsi_values,
            name="RSI",
            line=dict(color="purple", width=2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=macd_line,
            name="MACD",
            line=dict(color="blue", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=macd_signal,
            name="Signal",
            line=dict(color="orange", width=2),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=macd_hist,
            name="Histogram",
            marker_color=["green" if val >= 0 else "red" for val in macd_hist],
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=plot_df["volume"],
            name="Volume" if not show_bvi else "Better Volume",
            marker_color=volume_colors if volume_colors else "#00FFFF",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=volume_sma_plot,
            name=f"Volume SMA ({bvi_length})",
            line=dict(color="orange", width=2),
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume" if not show_bvi else "Better Volume Indicator", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)

    return fig


def create_candlestick_chart(summary: SimulationSummary, main_series: TimeframeSeries):
    candles = main_series.candles
    
    df = pd.DataFrame([
        {
            "timestamp": dt.datetime.fromtimestamp(c.close_time / 1000),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ])
    
    bvi_length = 8
    volume_sma, volume_colors = calculate_better_volume_indicator(df, length=bvi_length, use_two_bars=True)
    volume_sma_plot = volume_sma.where(volume_sma.notna(), None).tolist()
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.20],
        subplot_titles=("Price & Indicators", "RSI", "MACD", "Better Volume Indicator"),
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1, col=1,
    )
    
    if summary.snapshots and len(summary.snapshots) == len(candles):
        bollinger_upper = [s.bollinger_upper for s in summary.snapshots]
        bollinger_middle = [s.bollinger_middle for s in summary.snapshots]
        bollinger_lower = [s.bollinger_lower for s in summary.snapshots]
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=bollinger_upper,
                name="BB Upper",
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=bollinger_middle,
                name="BB Middle",
                line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=bollinger_lower,
                name="BB Lower",
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.1)",
            ),
            row=1, col=1,
        )
        
        atr_colors = {
            "atr_trend_3x": ("rgba(0, 255, 0, 0.6)", 1),
            "atr_trend_8x": ("rgba(255, 165, 0, 0.6)", 2),
            "atr_trend_21x": ("rgba(255, 0, 0, 0.6)", 3),
        }
        
        for atr_key, (color, width) in atr_colors.items():
            atr_values = [s.atr_channels.get(atr_key) if s.atr_channels else None for s in summary.snapshots]
            if any(v is not None for v in atr_values):
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=atr_values,
                        name=f"ATR {atr_key.replace('atr_trend_', '').replace('x', '')}x",
                        line=dict(color=color, width=width),
                        mode="lines",
                    ),
                    row=1, col=1,
                )
        
        rsi_values = [s.rsi if s.rsi is not None else 50 for s in summary.snapshots]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=rsi_values,
                name="RSI",
                line=dict(color="purple", width=2),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        macd_values = [s.macd if s.macd is not None else 0 for s in summary.snapshots]
        macd_signal = [s.macd_signal if s.macd_signal is not None else 0 for s in summary.snapshots]
        macd_histogram = [s.macd_histogram if s.macd_histogram is not None else 0 for s in summary.snapshots]
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=macd_values,
                name="MACD",
                line=dict(color="blue", width=2),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=macd_signal,
                name="Signal",
                line=dict(color="orange", width=2),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=macd_histogram,
                name="Histogram",
                marker_color=["green" if val >= 0 else "red" for val in macd_histogram],
            ),
            row=3, col=1,
        )
    
    for zone in summary.active_fvg_zones:
        zone_type = zone.zone_type
        color = "rgba(0, 255, 0, 0.2)" if "Bull" in zone_type else "rgba(255, 0, 0, 0.2)"
        
        if zone.created_index < len(df):
            start_time = df.iloc[zone.created_index]["timestamp"]
            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=df["timestamp"].iloc[-1],
                y0=zone.bottom,
                y1=zone.top,
                fillcolor=color,
                line=dict(color=color.replace("0.2", "0.5"), width=1),
                row=1, col=1,
            )
    
    for zone in summary.active_ob_zones:
        zone_type = zone.zone_type
        color = "rgba(0, 0, 255, 0.15)" if "Bull" in zone_type else "rgba(255, 165, 0, 0.15)"
        
        if zone.created_index < len(df):
            start_time = df.iloc[zone.created_index]["timestamp"]
            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=df["timestamp"].iloc[-1],
                y0=zone.bottom,
                y1=zone.top,
                fillcolor=color,
                line=dict(color=color.replace("0.15", "0.5"), width=1, dash="dash"),
                row=1, col=1,
            )
    
    for signal in summary.signals:
        if signal.bar_index < len(df):
            signal_time = df.iloc[signal.bar_index]["timestamp"]
            signal_price = signal.price
            
            if signal.signal_type == "bullish":
                fig.add_trace(
                    go.Scatter(
                        x=[signal_time],
                        y=[signal_price],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=15, color="lime"),
                        name=f"Buy Signal",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[signal_time],
                        y=[signal_price],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=15, color="red"),
                        name=f"Sell Signal",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
    
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Better Volume",
            marker_color=volume_colors if volume_colors else "#00FFFF",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=volume_sma_plot,
            name=f"Volume SMA ({bvi_length})",
            line=dict(color="orange", width=2),
        ),
        row=4, col=1,
    )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Better Volume Indicator", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)
    
    return fig


def create_multi_timeframe_chart(payload: dict):
    mtf_data = payload.get("multi_timeframe", {})
    trend_strength = mtf_data.get("trend_strength", {})
    direction = mtf_data.get("direction", {})
    
    if not trend_strength:
        return None
    
    timeframes = list(trend_strength.keys())
    strengths = list(trend_strength.values())
    directions = [direction.get(tf, "neutral") for tf in timeframes]
    
    colors = []
    for d in directions:
        if d == "bullish":
            colors.append("green")
        elif d == "bearish":
            colors.append("red")
        else:
            colors.append("gray")
    
    fig = go.Figure(data=[
        go.Bar(
            x=timeframes,
            y=strengths,
            marker_color=colors,
            text=[f"{s:.1f}" for s in strengths],
            textposition="outside",
        )
    ])
    
    fig.update_layout(
        title="Multi-Timeframe Trend Strength",
        xaxis_title="Timeframe",
        yaxis_title="Strength (0-100)",
        yaxis_range=[0, 100],
        height=400,
        template="plotly_dark",
    )
    
    return fig


FACTOR_WEIGHT_FIELDS = [
    ("market_structure", "Market Structure"),
    ("technical", "Technical Indicators"),
    ("volume", "Orderbook & Volume"),
    ("sentiment", "Sentiment & Fundamentals"),
    ("multitimeframe", "Multi-timeframe Alignment"),
]


def render_weight_controls(config_store: ConfigStore) -> Dict[str, float]:
    """Render sliders for adjusting category weights and return normalized weights."""
    st.markdown("#### Category Weights")
    weights = config_store.weights()
    columns = st.columns(len(FACTOR_WEIGHT_FIELDS))

    for column, (key, label) in zip(columns, FACTOR_WEIGHT_FIELDS):
        current_value = float(weights.get(key, 0.0))
        with column:
            slider_value = st.slider(
                label,
                min_value=0.0,
                max_value=1.0,
                value=round(current_value, 2),
                step=0.05,
                key=ui_key("weights", key),
            )
        if abs(slider_value - current_value) > 1e-6:
            config_store.update_weight(key, slider_value)

    normalized = config_store.normalized_weights()
    weight_table = pd.DataFrame(
        {
            "Category": [label for _, label in FACTOR_WEIGHT_FIELDS],
            "Weight": [normalized.get(key, 0.0) for key, _ in FACTOR_WEIGHT_FIELDS],
        }
    )
    st.caption("Normalized allocation across confirmation categories")
    st.table(weight_table.set_index("Category"))
    return normalized


def render_indicator_controls(config_store: ConfigStore) -> None:
    """Render indicator parameter controls for MACD, RSI, and ATR."""
    params = config_store.get_indicator_params()

    macd_params = params.get("macd", {})
    macd_cols = st.columns(3)
    fast_length = num_int(
        "MACD Fast Length",
        min_v=2,
        max_v=60,
        value=macd_params.get("fast", 12),
        key=ui_key("indicator", "macd_fast"),
        ui=macd_cols[0],
    )
    slow_length = num_int(
        "MACD Slow Length",
        min_v=fast_length + 1,
        max_v=200,
        value=macd_params.get("slow", 26),
        key=ui_key("indicator", "macd_slow"),
        ui=macd_cols[1],
    )
    signal_length = num_int(
        "MACD Signal Length",
        min_v=1,
        max_v=60,
        value=macd_params.get("signal", 9),
        key=ui_key("indicator", "macd_signal"),
        ui=macd_cols[2],
    )
    config_store.update_indicator_param("macd", "fast", int(fast_length))
    config_store.update_indicator_param("macd", "slow", int(slow_length))
    config_store.update_indicator_param("macd", "signal", int(signal_length))

    rsi_params = params.get("rsi", {})
    rsi_cols = st.columns(3)
    rsi_period = num_int(
        "RSI Period",
        min_v=5,
        max_v=100,
        value=rsi_params.get("period", 14),
        key=ui_key("indicator", "rsi_period"),
        ui=rsi_cols[0],
    )
    rsi_overbought = num_int(
        "RSI Overbought",
        min_v=50,
        max_v=100,
        value=rsi_params.get("overbought", 70),
        key=ui_key("indicator", "rsi_overbought"),
        ui=rsi_cols[1],
    )
    rsi_oversold = num_int(
        "RSI Oversold",
        min_v=0,
        max_v=50,
        value=rsi_params.get("oversold", 30),
        key=ui_key("indicator", "rsi_oversold"),
        ui=rsi_cols[2],
    )
    config_store.update_indicator_param("rsi", "period", int(rsi_period))
    config_store.update_indicator_param("rsi", "overbought", int(rsi_overbought))
    config_store.update_indicator_param("rsi", "oversold", int(rsi_oversold))

    atr_params = params.get("atr", {})
    atr_cols = st.columns(2)
    atr_period = num_int(
        "ATR Period",
        min_v=5,
        max_v=100,
        value=atr_params.get("period", 14),
        key=ui_key("indicator", "atr_period"),
        ui=atr_cols[0],
    )
    atr_mult = num_float(
        "ATR Multiplier",
        min_v=0.5,
        max_v=5.0,
        value=atr_params.get("mult", 1.0),
        step=0.1,
        key=ui_key("indicator", "atr_mult"),
        ui=atr_cols[1],
    )
    config_store.update_indicator_param("atr", "period", int(atr_period))
    config_store.update_indicator_param("atr", "mult", float(atr_mult))

    st.markdown("#### ATR Channels")
    atr_channel_params = params.get("atr_channels", {})
    atr_channel_cols = st.columns(4)
    atr_channel_period = num_int(
        "Channel period",
        min_v=5,
        max_v=100,
        value=int(atr_channel_params.get("period", atr_period)),
        key=ui_key("indicator", "atr_channel_period"),
        ui=atr_channel_cols[0],
    )
    atr_channel_mult_1x = num_float(
        "1x ATR multiplier",
        min_v=0.1,
        max_v=10.0,
        value=float(atr_channel_params.get("mult_1x", atr_mult)),
        step=0.1,
        key=ui_key("indicator", "atr_channel_mult_1x"),
        ui=atr_channel_cols[1],
    )
    atr_channel_mult_2x = num_float(
        "2x ATR multiplier",
        min_v=0.1,
        max_v=10.0,
        value=float(atr_channel_params.get("mult_2x", 2.0)),
        step=0.1,
        key=ui_key("indicator", "atr_channel_mult_2x"),
        ui=atr_channel_cols[2],
    )
    atr_channel_mult_3x = num_float(
        "3x ATR multiplier",
        min_v=0.1,
        max_v=10.0,
        value=float(atr_channel_params.get("mult_3x", 3.0)),
        step=0.1,
        key=ui_key("indicator", "atr_channel_mult_3x"),
        ui=atr_channel_cols[3],
    )
    config_store.update_indicator_param("atr_channels", "period", int(atr_channel_period))
    config_store.update_indicator_param("atr_channels", "mult_1x", float(atr_channel_mult_1x))
    config_store.update_indicator_param("atr_channels", "mult_2x", float(atr_channel_mult_2x))
    config_store.update_indicator_param("atr_channels", "mult_3x", float(atr_channel_mult_3x))

    st.markdown("#### Bollinger Bands")
    bollinger_params = params.get("bollinger", {})
    bollinger_cols = st.columns(3)
    bollinger_period = num_int(
        "Bollinger period",
        min_v=5,
        max_v=200,
        value=int(bollinger_params.get("period", 20)),
        key=ui_key("indicator", "bollinger_period"),
        ui=bollinger_cols[0],
    )
    bollinger_mult = num_float(
        "Band multiplier",
        min_v=0.5,
        max_v=5.0,
        value=float(bollinger_params.get("mult", bollinger_params.get("stddev", 2.0))),
        step=0.1,
        key=ui_key("indicator", "bollinger_mult"),
        ui=bollinger_cols[1],
    )
    bollinger_source_options = ["close", "hlc3", "ohlc4", "open"]
    bollinger_source_value = str(bollinger_params.get("source", "close")).lower()
    if bollinger_source_value not in bollinger_source_options:
        bollinger_source_value = "close"
    bollinger_source = bollinger_cols[2].selectbox(
        "Price source",
        bollinger_source_options,
        index=bollinger_source_options.index(bollinger_source_value),
        key=ui_key("indicator", "bollinger_source"),
    )
    config_store.update_indicator_param("bollinger", "period", int(bollinger_period))
    config_store.update_indicator_param("bollinger", "mult", float(bollinger_mult))
    config_store.update_indicator_param("bollinger", "stddev", float(bollinger_mult))
    config_store.update_indicator_param("bollinger", "source", str(bollinger_source))

    st.markdown("#### Volume & Order Flow")
    volume_params = params.get("volume", {})
    volume_cols = st.columns(2)
    cvd_atr = num_float(
        "CVD acceleration vs ATR (x)",
        min_v=0.1,
        max_v=5.0,
        value=float(volume_params.get("cvd_atr_multiplier", 0.75)),
        step=0.05,
        key=ui_key("indicator", "cvd_atr_multiplier"),
        ui=volume_cols[0],
        help_text="Multiplier applied to ATR when evaluating cumulative volume delta acceleration.",
    )
    delta_threshold = num_float(
        "Delta imbalance threshold",
        min_v=0.1,
        max_v=5.0,
        value=float(volume_params.get("delta_imbalance_threshold", 1.2)),
        step=0.05,
        key=ui_key("indicator", "delta_threshold"),
        ui=volume_cols[1],
        help_text="Relative threshold versus average order flow delta to flag imbalance.",
    )
    volume_cols_2 = st.columns(2)
    poc_share = num_float(
        "POC minimum volume share",
        min_v=0.01,
        max_v=0.20,
        value=float(volume_params.get("vpvr_poc_share", 0.04)),
        step=0.01,
        key=ui_key("indicator", "vpvr_poc_share"),
        ui=volume_cols_2[0],
        help_text="Minimum VPVR share required for the point of control to influence scoring.",
    )
    smart_money_mult = num_float(
        "Smart money spike multiple",
        min_v=1.0,
        max_v=5.0,
        value=float(volume_params.get("smart_money_multiplier", 1.5)),
        step=0.1,
        key=ui_key("indicator", "smart_money_multiplier"),
        ui=volume_cols_2[1],
        help_text="Volume multiple above the rolling average to treat prints as smart money activity.",
    )
    config_store.update_indicator_param("volume", "cvd_atr_multiplier", float(cvd_atr))
    config_store.update_indicator_param("volume", "delta_imbalance_threshold", float(delta_threshold))
    config_store.update_indicator_param("volume", "vpvr_poc_share", float(poc_share))
    config_store.update_indicator_param("volume", "smart_money_multiplier", float(smart_money_mult))

    st.markdown("#### Market Structure")
    structure_params = params.get("structure", {})
    structure_cols = st.columns(4)
    structure_lookback = num_int(
        "Structure lookback (bars)",
        min_v=4,
        max_v=200,
        value=int(structure_params.get("lookback", 24)),
        key=ui_key("indicator", "structure_lookback"),
        ui=structure_cols[0],
    )
    swing_window = num_int(
        "Swing window",
        min_v=2,
        max_v=50,
        value=int(structure_params.get("swing_window", 5)),
        key=ui_key("indicator", "structure_swing_window"),
        ui=structure_cols[1],
    )
    trend_window = num_int(
        "Trend window",
        min_v=2,
        max_v=100,
        value=int(structure_params.get("trend_window", 12)),
        key=ui_key("indicator", "structure_trend_window"),
        ui=structure_cols[2],
    )
    min_sequence = num_int(
        "Minimum sequence",
        min_v=1,
        max_v=20,
        value=int(structure_params.get("min_sequence", 5)),
        key=ui_key("indicator", "structure_min_sequence"),
        ui=structure_cols[3],
    )
    structure_cols_2 = st.columns(1)
    atr_distance = num_float(
        "Liquidity sweep ATR distance",
        min_v=0.1,
        max_v=5.0,
        value=float(structure_params.get("atr_distance", 1.0)),
        step=0.05,
        key=ui_key("indicator", "structure_atr_distance"),
        ui=structure_cols_2[0],
    )
    config_store.update_indicator_param("structure", "lookback", int(structure_lookback))
    config_store.update_indicator_param("structure", "swing_window", int(swing_window))
    config_store.update_indicator_param("structure", "trend_window", int(trend_window))
    config_store.update_indicator_param("structure", "min_sequence", int(min_sequence))
    config_store.update_indicator_param("structure", "atr_distance", float(atr_distance))

    st.markdown("#### Composite Confirmation")
    composite_params = params.get("composite", {})
    composite_cols = st.columns(2)
    composite_buy = num_float(
        "Composite buy threshold",
        min_v=0.0,
        max_v=1.0,
        value=float(composite_params.get("buy_threshold", DEFAULT_SIGNAL_THRESHOLDS["buy"])),
        step=0.01,
        key=ui_key("indicator", "composite_buy_threshold"),
        ui=composite_cols[0],
    )
    composite_sell = num_float(
        "Composite sell threshold",
        min_v=0.0,
        max_v=1.0,
        value=float(composite_params.get("sell_threshold", DEFAULT_SIGNAL_THRESHOLDS["sell"])),
        step=0.01,
        key=ui_key("indicator", "composite_sell_threshold"),
        ui=composite_cols[1],
    )
    composite_cols_2 = st.columns(2)
    confidence_floor = num_float(
        "Confidence floor",
        min_v=0.0,
        max_v=1.0,
        value=float(composite_params.get("confidence_floor", 0.3)),
        step=0.01,
        key=ui_key("indicator", "composite_conf_floor"),
        ui=composite_cols_2[0],
    )
    confidence_ceiling = num_float(
        "Confidence ceiling",
        min_v=0.0,
        max_v=1.0,
        value=float(composite_params.get("confidence_ceiling", 0.9)),
        step=0.01,
        key=ui_key("indicator", "composite_conf_ceiling"),
        ui=composite_cols_2[1],
    )
    min_confirmations = num_int(
        "Min confirmations (composite)",
        min_v=1,
        max_v=6,
        value=int(composite_params.get("min_confirmations", 3)),
        key=ui_key("indicator", "composite_min_confirmations"),
    )
    config_store.update_indicator_param("composite", "buy_threshold", float(composite_buy))
    config_store.update_indicator_param("composite", "sell_threshold", float(composite_sell))
    config_store.update_signal_setting("buy_threshold", float(composite_buy))
    config_store.update_signal_setting("sell_threshold", float(composite_sell))
    config_store.update_indicator_param("composite", "confidence_floor", float(confidence_floor))
    config_store.update_indicator_param("composite", "confidence_ceiling", float(confidence_ceiling))
    config_store.update_indicator_param("composite", "min_confirmations", int(min_confirmations))

    st.markdown("#### Multi-timeframe Alignment")
    mtf_params = params.get("multitimeframe", {})
    mtf_cols = st.columns(2)
    trend_lookback = num_int(
        "Trend lookback (bars)",
        min_v=4,
        max_v=60,
        value=int(mtf_params.get("trend_lookback", 14)),
        key=ui_key("indicator", "mtf_trend_lookback"),
        ui=mtf_cols[0],
    )
    alignment_weight = num_float(
        "Alignment weight",
        min_v=0.0,
        max_v=1.0,
        value=float(mtf_params.get("alignment_weight", 0.4)),
        step=0.05,
        key=ui_key("indicator", "mtf_alignment_weight"),
        ui=mtf_cols[1],
    )
    mtf_cols_2 = st.columns(2)
    agreement_weight = num_float(
        "Agreement weight",
        min_v=0.0,
        max_v=1.0,
        value=float(mtf_params.get("agreement_weight", 0.3)),
        step=0.05,
        key=ui_key("indicator", "mtf_agreement_weight"),
        ui=mtf_cols_2[0],
    )
    force_weight = num_float(
        "Trend force weight",
        min_v=0.0,
        max_v=1.0,
        value=float(mtf_params.get("force_weight", 0.3)),
        step=0.05,
        key=ui_key("indicator", "mtf_force_weight"),
        ui=mtf_cols_2[1],
    )
    config_store.update_indicator_param("multitimeframe", "trend_lookback", int(trend_lookback))
    config_store.update_indicator_param("multitimeframe", "alignment_weight", float(alignment_weight))
    config_store.update_indicator_param("multitimeframe", "agreement_weight", float(agreement_weight))
    config_store.update_indicator_param("multitimeframe", "force_weight", float(force_weight))

    st.markdown("#### Diagnostics")
    debug_enabled = config_store.is_debug_enabled()
    show_debug = st.checkbox(
        "Show analyzer diagnostics in results",
        value=debug_enabled,
        key=ui_key("indicator", "show_debug"),
        help="When enabled, generated signals include per-factor normalization and weighting details.",
    )
    if bool(show_debug) != debug_enabled:
        config_store.update_debug_setting("show_debug", bool(show_debug))


def render_signal_risk_controls(config_store: ConfigStore) -> None:
    """Render controls for risk settings and signal thresholds."""
    risk = config_store.risk_settings()
    risk_cols = st.columns(3)

    account_balance = num_float(
        "Account Balance (USD)",
        min_v=100.0,
        value=risk.get("account_balance", 10_000.0),
        step=100.0,
        key=ui_key("risk", "account_balance"),
        ui=risk_cols[0],
    )
    risk_per_trade_pct = float(risk.get("max_risk_per_trade_pct", 0.02) * 100.0)
    risk_per_trade = num_float(
        "Max Risk per Trade (%)",
        min_v=0.0,
        max_v=10.0,
        value=risk_per_trade_pct,
        step=0.1,
        key=ui_key("risk", "risk_per_trade"),
        ui=risk_cols[1],
    )
    max_position_pct = float(risk.get("max_position_size_pct", 0.05) * 100.0)
    max_position_size = num_float(
        "Max Position Size (%)",
        min_v=0.0,
        max_v=100.0,
        value=max_position_pct,
        step=0.5,
        key=ui_key("risk", "position_size"),
        ui=risk_cols[2],
    )
    config_store.update_risk_setting("account_balance", float(account_balance))
    config_store.update_risk_setting("max_risk_per_trade_pct", float(risk_per_trade) / 100.0)
    config_store.update_risk_setting("max_position_size_pct", float(max_position_size) / 100.0)

    signal_settings = config_store.signal_settings()
    signal_cols = st.columns(3)

    def _sanitize_threshold_pair(buy_val: float, sell_val: float) -> tuple[float, float]:
        buy_val = float(min(max(buy_val, 0.0), 1.0))
        sell_val = float(min(max(sell_val, 0.0), 1.0))
        if buy_val <= sell_val:
            buy_val, sell_val = max(buy_val, sell_val), min(buy_val, sell_val)
            if buy_val <= sell_val:
                buy_val = min(1.0, max(sell_val + 0.01, DEFAULT_SIGNAL_THRESHOLDS["buy"]))
            if buy_val > 1.0:
                buy_val = 1.0
            if buy_val <= sell_val:
                sell_val = max(0.0, min(buy_val - 0.01, DEFAULT_SIGNAL_THRESHOLDS["sell"]))
        if buy_val <= sell_val:
            buy_val = DEFAULT_SIGNAL_THRESHOLDS["buy"]
            sell_val = DEFAULT_SIGNAL_THRESHOLDS["sell"]
        return round(buy_val, 2), round(sell_val, 2)

    min_confirmations = num_int(
        "Min Confirmations",
        min_v=1,
        max_v=5,
        value=signal_settings.get("min_confirmations", 3),
        key=ui_key("signal", "min_confirmations"),
        ui=signal_cols[0],
    )
    buy_threshold_key = ui_key("signal", "buy_threshold")
    buy_threshold = signal_cols[1].slider(
        "Buy Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(signal_settings.get("buy_threshold", DEFAULT_SIGNAL_THRESHOLDS["buy"])),
        step=0.01,
        key=buy_threshold_key,
    )
    sell_threshold_key = ui_key("signal", "sell_threshold")
    sell_threshold = signal_cols[2].slider(
        "Sell Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(signal_settings.get("sell_threshold", DEFAULT_SIGNAL_THRESHOLDS["sell"])),
        step=0.01,
        key=sell_threshold_key,
    )

    sanitized_buy, sanitized_sell = _sanitize_threshold_pair(buy_threshold, sell_threshold)
    if (sanitized_buy, sanitized_sell) != (round(buy_threshold, 2), round(sell_threshold, 2)):
        st.error("Buy threshold must be greater than sell threshold. Values adjusted automatically.")
        st.session_state[buy_threshold_key] = sanitized_buy
        st.session_state[sell_threshold_key] = sanitized_sell
        buy_threshold, sell_threshold = sanitized_buy, sanitized_sell
    else:
        buy_threshold, sell_threshold = sanitized_buy, sanitized_sell

    config_store.update_signal_setting("min_confirmations", int(min_confirmations))
    config_store.update_signal_setting("buy_threshold", float(buy_threshold))
    config_store.update_signal_setting("sell_threshold", float(sell_threshold))

    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=1.0,
        value=float(signal_settings.get("min_confidence", 0.6)),
        step=0.05,
        key=ui_key("signal", "min_confidence"),
    )
    config_store.update_signal_setting("min_confidence", float(min_confidence))


def main():
    st.title("📈 Token Charts & Indicators Dashboard")
    st.markdown("---")
    config_store = ConfigStore.load()
    
    # Initialize Charts tab session state
    if "chart_symbol" not in st.session_state:
        st.session_state.chart_symbol = None
    if "chart_timeframe" not in st.session_state:
        st.session_state.chart_timeframe = None
    if "chart_df" not in st.session_state:
        st.session_state.chart_df = None
    if "chart_indicators" not in st.session_state:
        st.session_state.chart_indicators = None
    if "last_closed_ts" not in st.session_state:
        st.session_state.last_closed_ts = 0
    if "last_closed_ts_per_tf" not in st.session_state:
        st.session_state.last_closed_ts_per_tf = {}
    if "analysis_updated" not in st.session_state:
        st.session_state.analysis_updated = False
    if "worker_running" not in st.session_state:
        st.session_state.worker_running = False
    if "chart_worker" not in st.session_state:
        st.session_state.chart_worker = None
    if "chart_manager_started" not in st.session_state:
        st.session_state.chart_manager_started = False
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    if "bvi_enabled" not in st.session_state:
        st.session_state.bvi_enabled = True
    if "atr_channels_enabled" not in st.session_state:
        st.session_state.atr_channels_enabled = True
    if "order_blocks_enabled" not in st.session_state:
        st.session_state.order_blocks_enabled = True
    if "export_token" not in st.session_state:
        st.session_state.export_token = ""
    
    # Initialize WebSocket and UpdateBus support
    if "use_websocket" not in st.session_state:
        st.session_state.use_websocket = True
    if "chart_update_bus" not in st.session_state:
        from update_bus import UpdateBus
        st.session_state.chart_update_bus = UpdateBus()
    if "signals_update_bus" not in st.session_state:
        from update_bus import UpdateBus
        st.session_state.signals_update_bus = UpdateBus()
    if "chart_worker_manager" not in st.session_state:
        from worker_manager import ChartWorkerManager
        st.session_state.chart_worker_manager = ChartWorkerManager()
    if "signals_worker_manager" not in st.session_state:
        from worker_manager import SignalsWorkerManager
        st.session_state.signals_worker_manager = SignalsWorkerManager()
    
    with st.sidebar:
        st.header("⚙️ Configuration")

        token_mode_index = 0 if config_store.token in POPULAR_TOKENS else 1
        token_input_mode = st.radio(
            "Input Mode",
            ["Select from list", "Custom token"],
            index=token_mode_index,
            key=ui_key("sidebar", "input_mode"),
        )

        if token_input_mode == "Select from list":
            try:
                default_index = POPULAR_TOKENS.index(config_store.token)
            except ValueError:
                default_index = 0
            selected_token_option = st.selectbox(
                "Select Token",
                POPULAR_TOKENS,
                index=default_index,
                key=ui_key("sidebar", "select_token"),
            )
            config_store.set_token(selected_token_option)
        else:
            selected_token_option = st.text_input(
                "Custom Token (e.g., BINANCE:BTCUSDT)",
                config_store.token,
                key=ui_key("sidebar", "custom_token"),
            )
            config_store.set_token(selected_token_option)

        st.subheader("Timeframe & Period")
        try:
            timeframe_index = TIMEFRAMES.index(config_store.timeframe)
        except ValueError:
            timeframe_index = TIMEFRAMES.index("15m")
        selected_timeframe_option = st.selectbox(
            "Timeframe",
            TIMEFRAMES,
            index=timeframe_index,
            key=ui_key("sidebar", "timeframe"),
        )
        config_store.set_timeframe(selected_timeframe_option)

        analysis_period_default = st.session_state.get(
            ui_key("sidebar", "analysis_period_value"), 200
        )
        analysis_period = st.slider(
            "Analysis Period (bars)",
            min_value=50,
            max_value=1000,
            value=int(analysis_period_default),
            step=50,
            key=ui_key("sidebar", "analysis_period"),
        )
        st.session_state[ui_key("sidebar", "analysis_period_value")] = analysis_period

        st.subheader("Realtime Settings")
        use_ws_checkbox = st.checkbox(
            "Use WebSocket streaming",
            value=st.session_state.use_websocket,
            key=ui_key("sidebar", "use_websocket"),
            help="Enable Binance WebSocket streaming for realtime klines and automated signals.",
        )
        st.session_state.use_websocket = use_ws_checkbox

        st.subheader("Export Options")

        export_default = st.session_state.export_token

        export_token = st.text_input(
            "Export Token/ID",
            value=export_default,
            help="Token to identify this analysis session",
            key=ui_key("sidebar", "export_token"),
        )
        st.session_state.export_token = export_token

        analyze_button = st.button(
            "🔄 Analyze", type="primary", width="stretch"
        )

    selected_token = config_store.token
    selected_timeframe = config_store.timeframe
    selected_period = analysis_period

    if analyze_button or "summary" not in st.session_state:
        if analyze_button:
            load_indicator_data.clear()
        with st.spinner(f"Analyzing {selected_token} on {selected_timeframe} timeframe..."):
            try:
                summary, payload, main_series = load_indicator_data(
                    selected_token,
                    selected_timeframe,
                    selected_period,
                    export_token,
                    CACHE_VERSION,
                )
                sanitized_payload = sanitize_payload_for_real_data(payload)
                st.session_state.summary = summary
                st.session_state.payload = sanitized_payload
                st.session_state.main_series = main_series
                st.session_state.export_token = export_token
                st.success("✅ Analysis completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    if "summary" not in st.session_state:
        st.info("👈 Configure parameters in the sidebar and click 'Analyze' to begin.")
        return
    
    summary = st.session_state.summary
    payload = sanitize_payload_for_real_data(st.session_state.payload)
    st.session_state.payload = payload
    main_series = st.session_state.main_series
    
    (
        chart_tab,
        multi_tab,
        latest_tab,
        signals_tab,
        volume_tab,
        structure_tab,
        fundamentals_tab,
        breadth_tab,
        onchain_tab,
        composite_tab,
        patterns_tab,
        trade_tab,
        automated_signals_tab,
        backtest_tab,
        adaptive_tab,
        astrology_tab,
        export_tab,
    ) = st.tabs([
        "📊 Charts",
        "📈 Multi-Timeframe",
        "📋 Latest Metrics",
        "🎯 Signals & Zones",
        "📊 Volume Analysis",
        "🏗️ Market Structure",
        "📈 Fundamentals",
        "🌐 Breadth Indicators",
        "🔗 On-chain Metrics",
        "🧩 Composite Indicators",
        "🌊 Patterns & Waves",
        "🎯 Trade Signals",
        "🤖 Automated Signals",
        "🔬 Backtesting",
        "⚖️ Adaptive Weights",
        "🔮 Astrology",
        "💾 Export",
    ])
    
    with chart_tab:
        from chart_auto_refresh import (
            ChartAutoRefreshWorker,
            compute_chart_indicators,
            fetch_closed_candles,
            invalidate_cache,
            read_chart_state,
            update_chart_state,
        )
        
        st.subheader(f"Price Chart with Indicators - {selected_token}")
        
        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1, 1, 1, 1, 1])
        with ctrl_col1:
            auto_refresh = st.checkbox(
                "🔄 Auto-refresh",
                value=st.session_state.auto_refresh_enabled,
                key="charts_auto_refresh_toggle",
            )
        with ctrl_col2:
            show_forming_bar = st.checkbox(
                "📊 Forming Bar",
                value=st.session_state.get("show_forming_bar", False),
                key="charts_forming_bar_toggle",
                help="Show the currently forming candle (not yet closed)",
            )
            st.session_state.show_forming_bar = show_forming_bar
        with ctrl_col3:
            bvi_enabled = st.checkbox(
                "📊 Better Volume",
                value=st.session_state.bvi_enabled,
                key="charts_bvi_toggle",
            )
            st.session_state.bvi_enabled = bvi_enabled
        with ctrl_col4:
            atr_channels_enabled = st.checkbox(
                "📈 ATR Channels",
                value=st.session_state.atr_channels_enabled,
                key="charts_atr_channels_toggle",
            )
            st.session_state.atr_channels_enabled = atr_channels_enabled
        with ctrl_col5:
            order_blocks_enabled = st.checkbox(
                "🟦 Order Blocks",
                value=st.session_state.order_blocks_enabled,
                key="charts_order_blocks_toggle",
            )
            st.session_state.order_blocks_enabled = order_blocks_enabled
        
        # Status indicator
        if st.session_state.worker_running:
            st.info(f"🟢 Live: Monitoring {selected_token} {selected_timeframe}")
        elif auto_refresh:
            st.caption("Initializing auto-refresh...")
        else:
            st.caption("Auto-refresh disabled")

        chart_container = st.empty()
        
        # Check if symbol or timeframe changed
        symbol_changed = st.session_state.chart_symbol != selected_token
        timeframe_changed = st.session_state.chart_timeframe != selected_timeframe
        
        # Stop existing worker if symbol/timeframe changed or auto-refresh disabled
        if (symbol_changed or timeframe_changed or not auto_refresh):
            # Stop old worker
            if st.session_state.chart_worker is not None:
                try:
                    st.session_state.chart_worker.stop()
                    st.session_state.chart_worker = None
                    st.session_state.worker_running = False
                except Exception as e:
                    logger.warning(f"Failed to stop chart worker: {e}")
            
            # Stop WorkerManager
            try:
                st.session_state.chart_worker_manager.stop()
            except Exception as e:
                logger.warning(f"Failed to stop chart worker manager: {e}")
        
        # Update session state tracking
        st.session_state.auto_refresh_enabled = auto_refresh
        
        # Handle symbol/timeframe change: reset state and invalidate cache
        if symbol_changed or timeframe_changed:
            st.session_state.chart_symbol = selected_token
            st.session_state.chart_timeframe = selected_timeframe
            st.session_state.chart_df = None
            st.session_state.chart_indicators = None
            # Reset last_closed_ts for backward compatibility, but per-tf tracking is used internally
            st.session_state.last_closed_ts = 0
            last_map = st.session_state.get("last_closed_ts_per_tf", {}) or {}
            last_map.pop(f"{selected_token}|{selected_timeframe}", None)
            st.session_state.last_closed_ts_per_tf = last_map
            st.session_state.analysis_updated = False
            
            # Invalidate cache
            try:
                invalidate_cache(selected_token, selected_timeframe)
            except Exception as e:
                logger.warning(f"Failed to invalidate cache: {e}")
            
            # Fetch initial data synchronously
            with st.spinner(f"Loading chart data for {selected_token} {selected_timeframe}..."):
                try:
                    df, last_closed_ts = fetch_closed_candles(
                        symbol=selected_token,
                        timeframe=selected_timeframe,
                        num_bars=selected_period,
                        use_cache=False,
                    )
                    # Update state atomically (indicators computed within update_chart_state)
                    update_chart_state(
                        st.session_state,
                        selected_token,
                        selected_timeframe,
                        df,
                        last_closed_ts,
                        append=False,
                    )
                    st.session_state.analysis_updated = False
                except Exception as e:
                    st.error(f"❌ Failed to load chart data: {str(e)}")
                    st.session_state.chart_df = None
                    st.session_state.chart_indicators = None
        
        # Start worker if auto-refresh is enabled and not already running
        if auto_refresh:
            use_websocket = st.session_state.use_websocket
            manager = st.session_state.chart_worker_manager
            
            # Check if we need to start a new worker
            if not manager.is_running() and st.session_state.chart_worker is None:
                try:
                    # Try WorkerManager with WebSocket
                    if use_websocket:
                        success = manager.start_new(
                            symbol=selected_token,
                            timeframe=selected_timeframe,
                            update_bus=st.session_state.chart_update_bus,
                            use_websocket=True,
                            session_state=st.session_state,
                            num_bars=selected_period,
                        )
                        if success:
                            st.session_state.worker_running = True
                            logger.info(f"Started WebSocket chart worker for {selected_token} {selected_timeframe}")
                        else:
                            # Fallback to REST polling already handled by manager
                            st.session_state.worker_running = True
                    else:
                        # Use REST polling
                        worker = ChartAutoRefreshWorker(
                            symbol=selected_token,
                            timeframe=selected_timeframe,
                            num_bars=selected_period,
                            session_state=st.session_state,
                        )
                        worker.start()
                        st.session_state.chart_worker = worker
                        st.session_state.worker_running = True
                except Exception as e:
                    logger.error(f"Failed to start chart worker: {e}", exc_info=True)
                    st.error(f"❌ Failed to start auto-refresh worker: {str(e)}")
        
        # Poll for updates from WorkerManager and apply to session state
        if auto_refresh and st.session_state.use_websocket:
            manager = st.session_state.chart_worker_manager
            try:
                updates_applied = manager.poll_and_apply(st.session_state)
                if updates_applied:
                    st.session_state.analysis_updated = True
            except Exception as e:
                logger.error(f"Error polling chart worker manager: {e}", exc_info=True)
        
        # Display chart
        if st.session_state.chart_df is not None and not st.session_state.chart_df.empty and auto_refresh:
            # Read chart state safely
            df, indicators, last_closed_close_ms = read_chart_state(st.session_state, selected_token, selected_timeframe)
            
            if df is not None and not df.empty:
                # Display status with UTC timestamp
                if last_closed_close_ms > 0:
                    # last_closed_close_ms is the close_time of the last closed bar
                    last_closed_dt = datetime.fromtimestamp(last_closed_close_ms / 1000, tz=timezone.utc)
                    st.caption(f"📅 Last closed bar (close time): {last_closed_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC | Bars: {len(df)}")
                
                # Build realtime chart from DataFrame with indicators
                try:
                    atr_channels_data = indicators.get("atr_channels", {}) if indicators else {}
                    order_blocks_data = indicators.get("order_blocks", []) if indicators else []
                    
                    fig = create_realtime_candlestick_chart(
                        df,
                        show_bvi=bvi_enabled,
                        atr_channels=atr_channels_data,
                        order_blocks=order_blocks_data,
                        show_atr_channels=atr_channels_enabled,
                        show_order_blocks=order_blocks_enabled,
                    )
                    chart_container.plotly_chart(fig, width="stretch", key="realtime_chart")
                except Exception as e:
                    st.error(f"Failed to render realtime chart: {str(e)}")
                    logger.exception("Chart rendering error")
                    # Fallback to original
                    fig = create_candlestick_chart(summary, main_series)
                    chart_container.plotly_chart(fig, width="stretch", key="fallback_chart")
                
                # Reset analysis_updated flag and rerun if needed
                if st.session_state.analysis_updated:
                    st.session_state.analysis_updated = False
                    st.rerun()
            else:
                # No data available
                st.warning("Waiting for chart data...")
        else:
            # Fallback to original chart
            fig = create_candlestick_chart(summary, main_series)
            chart_container.plotly_chart(fig, width="stretch", key="original_chart")
    
    with multi_tab:
        st.subheader("Multi-Timeframe Analysis")
        mtf_fig = create_multi_timeframe_chart(payload)
        if mtf_fig:
            st.plotly_chart(mtf_fig, width="stretch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trend Strength by Timeframe")
            mtf_data = payload.get("multi_timeframe", {})
            trend_df = pd.DataFrame([
                {"Timeframe": tf, "Strength": f"{val:.2f}"}
                for tf, val in mtf_data.get("trend_strength", {}).items()
            ])
            if not trend_df.empty:
                st.dataframe(trend_df, width="stretch", hide_index=True)
        
        with col2:
            st.markdown("### Direction by Timeframe")
            direction_df = pd.DataFrame([
                {"Timeframe": tf, "Direction": val.upper()}
                for tf, val in mtf_data.get("direction", {}).items()
            ])
            if not direction_df.empty:
                st.dataframe(direction_df, width="stretch", hide_index=True)
        
        if payload.get("multi_symbol"):
            st.markdown("### Multi-Symbol Confirmation")
            multi_sym = payload["multi_symbol"]
            
            sym_col1, sym_col2 = st.columns(2)
            with sym_col1:
                st.markdown("**Signals:**")
                for sym, signal in multi_sym.get("signals", {}).items():
                    color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                    st.write(f"{color} {sym}: **{signal}**")
            
            with sym_col2:
                st.markdown("**Trend Strength:**")
                for sym, strength in multi_sym.get("trend_strength", {}).items():
                    if strength is not None:
                        st.write(f"{sym}: **{strength:.2f}**")
    
    with latest_tab:
        st.subheader("Latest Market Snapshot")
        
        latest = payload.get("latest", {})
        atr_channels = payload.get("atr_channels", {})
        orderbook_data = payload.get("orderbook")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Close Price", f"${latest.get('close', 0):.4f}")
            st.metric("Volume", f"{latest.get('volume', 0):,.0f}")
        
        with col2:
            st.metric("Trend Strength", f"{latest.get('trend_strength', 0):.2f}")
            st.metric("Pattern Score", f"{latest.get('pattern_score', 0):.2f}")
        
        with col3:
            st.metric("Market Sentiment", f"{latest.get('market_sentiment', 0):.2f}")
            st.metric("RSI", f"{latest.get('rsi', 0):.2f}" if latest.get('rsi') else "N/A")
        
        with col4:
            confluence = latest.get('confluence_score', 0)
            confluence_bias = latest.get('confluence_bias', 'neutral')
            confluence_bull = latest.get('confluence_bullish', 0)
            confluence_bear = latest.get('confluence_bearish', 0)
            
            if confluence_bias == 'bullish':
                confluence_color = "🟢"
            elif confluence_bias == 'bearish':
                confluence_color = "🔴"
            else:
                confluence_color = "⚪"
            
            st.metric("Confluence Score", f"{confluence_color} {confluence:.2f}" if confluence else "N/A")
            st.markdown(f"**Bull:** {confluence_bull:.2f} | **Bear:** {confluence_bear:.2f}")
            
            structure = latest.get('structure_state', 'neutral')
            structure_emoji = "🟢" if structure == "bullish" else "🔴" if structure == "bearish" else "⚪"
            st.metric("Structure", f"{structure_emoji} {structure.upper()}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Technical Indicators")
            indicators_df = pd.DataFrame([
                {"Indicator": "MACD", "Value": f"{latest.get('macd', 0):.4f}" if latest.get('macd') else "N/A"},
                {"Indicator": "MACD Signal", "Value": f"{latest.get('macd_signal', 0):.4f}" if latest.get('macd_signal') else "N/A"},
                {"Indicator": "MACD Histogram", "Value": f"{latest.get('macd_histogram', 0):.4f}" if latest.get('macd_histogram') else "N/A"},
                {"Indicator": "Bollinger Upper", "Value": f"{latest.get('bollinger_upper', 0):.4f}" if latest.get('bollinger_upper') else "N/A"},
                {"Indicator": "Bollinger Middle", "Value": f"{latest.get('bollinger_middle', 0):.4f}" if latest.get('bollinger_middle') else "N/A"},
                {"Indicator": "Bollinger Lower", "Value": f"{latest.get('bollinger_lower', 0):.4f}" if latest.get('bollinger_lower') else "N/A"},
            ])
            st.dataframe(indicators_df, width="stretch", hide_index=True)
            
            if atr_channels:
                st.markdown("### ATR Channels")
                atr_df = pd.DataFrame([
                    {"ATR Level": k.replace("atr_trend_", "ATR ").upper(), "Value": f"{v:.4f}" if v is not None else "N/A"}
                    for k, v in atr_channels.items()
                ])
                st.dataframe(atr_df, width="stretch", hide_index=True)
        
        with col2:
            st.markdown("### Performance Statistics")
            success_rates = payload.get("success_rates", {})
            pnl_stats = payload.get("pnl_stats", {})
            
            stats_df = pd.DataFrame([
                {"Metric": "Overall Win Rate", "Value": f"{success_rates.get('overall_win_rate', 0):.2f}%"},
                {"Metric": "Bull Win Rate", "Value": f"{success_rates.get('bull_win_rate', 0):.2f}%"},
                {"Metric": "Bear Win Rate", "Value": f"{success_rates.get('bear_win_rate', 0):.2f}%"},
                {"Metric": "Cumulative PnL", "Value": f"{pnl_stats.get('cum_pnl_pct', 0):.2f}%"},
                {"Metric": "Max Drawdown", "Value": f"{pnl_stats.get('max_drawdown_pct', 0):.2f}%"},
                {"Metric": "Trades Closed", "Value": f"{pnl_stats.get('trades_closed', 0)}"},
            ])
            st.dataframe(stats_df, width="stretch", hide_index=True)
        
        cme_gaps = latest.get("cme_gaps", {})
        if cme_gaps:
            st.markdown("---")
            st.markdown("### 📊 CME Gap Analysis (CME Futures)")
            
            if cme_gaps.get("total_unfilled_gaps", 0) == 0:
                st.info("All CME gaps are currently filled. No outstanding gaps detected near the current price.")
            else:
                gap_col1, gap_col2 = st.columns(2)
                
                with gap_col1:
                    st.markdown("#### Nearest Gaps Above Current Price")
                    gaps_above = cme_gaps.get("nearest_gaps_above", [])
                    if gaps_above:
                        gaps_above_df = pd.DataFrame([
                            {
                                "Type": gap["type"].replace("_", " ").upper(),
                                "Top": f"${gap['gap_top']:.2f}",
                                "Bottom": f"${gap['gap_bottom']:.2f}",
                                "Distance": f"{gap['distance_pct']:.2f}%",
                                "Size": f"{gap['gap_size_pct']:.2f}%"
                            }
                            for gap in gaps_above[:5]
                        ])
                        st.dataframe(gaps_above_df, width="stretch", hide_index=True)
                    else:
                        st.info("No unfilled gaps above current price")
                
                with gap_col2:
                    st.markdown("#### Nearest Gaps Below Current Price")
                    gaps_below = cme_gaps.get("nearest_gaps_below", [])
                    if gaps_below:
                        gaps_below_df = pd.DataFrame([
                            {
                                "Type": gap["type"].replace("_", " ").upper(),
                                "Top": f"${gap['gap_top']:.2f}",
                                "Bottom": f"${gap['gap_bottom']:.2f}",
                                "Distance": f"{gap['distance_pct']:.2f}%",
                                "Size": f"{gap['gap_size_pct']:.2f}%"
                            }
                            for gap in gaps_below[:5]
                        ])
                        st.dataframe(gaps_below_df, width="stretch", hide_index=True)
                    else:
                        st.info("No unfilled gaps below current price")
        
        if orderbook_data:
            st.markdown("---")
            st.markdown("### 📊 Order Book Analysis (Binance)")
            
            ob_col1, ob_col2, ob_col3 = st.columns(3)
            
            with ob_col1:
                best_bid = orderbook_data.get('best_bid')
                st.metric("Best Bid", f"${best_bid:.4f}" if best_bid is not None else "N/A")
                best_ask = orderbook_data.get('best_ask')
                st.metric("Best Ask", f"${best_ask:.4f}" if best_ask is not None else "N/A")
            
            with ob_col2:
                spread = orderbook_data.get('spread')
                st.metric("Spread", f"${spread:.4f}" if spread is not None else "N/A")
                mid_price = orderbook_data.get('mid_price')
                st.metric("Mid Price", f"${mid_price:.4f}" if mid_price is not None else "N/A")
            
            with ob_col3:
                ratio = orderbook_data.get('bid_ask_ratio_top10')
                st.metric("Bid/Ask Ratio (Top 10)", f"{ratio:.2f}" if ratio is not None else "N/A")
                imbalance = orderbook_data.get('volume_imbalance_top10')
                st.metric("Volume Imbalance", f"{imbalance:.2f}" if imbalance is not None else "N/A")
            
            st.markdown("#### Volume at Price Levels")
            price_levels = orderbook_data.get('price_levels', {})
            if price_levels:
                levels_data = []
                for level, data in price_levels.items():
                    ratio_val = None
                    ask_volume = data.get('ask_volume', 0)
                    bid_volume = data.get('bid_volume', 0)
                    if ask_volume:
                        ratio_val = bid_volume / ask_volume
                    levels_data.append({
                        "Level": level,
                        "Bid Volume": f"{bid_volume:.2f}",
                        "Ask Volume": f"{ask_volume:.2f}",
                        "Ratio": f"{ratio_val:.2f}" if ratio_val is not None else "N/A"
                    })
                ob_levels_df = pd.DataFrame(levels_data)
                st.dataframe(ob_levels_df, width="stretch", hide_index=True)
            
            sections = orderbook_data.get('sections', {})
            if sections:
                st.markdown("#### Aggregated Depth (Top Levels)")
                section_rows = []
                bids_sections = sections.get('bids', {})
                asks_sections = sections.get('asks', {})
                for key, label in (('top_5', 'Top 5'), ('top_10', 'Top 10'), ('top_20', 'Top 20')):
                    bid_info = bids_sections.get(key, {})
                    ask_info = asks_sections.get(key, {})
                    section_rows.append({
                        "Levels": label,
                        "Bid Volume": f"{bid_info.get('total_volume', 0):.2f}",
                        "Bid W. Price": f"{bid_info.get('weighted_price'):.4f}" if bid_info.get('weighted_price') is not None else "N/A",
                        "Ask Volume": f"{ask_info.get('total_volume', 0):.2f}",
                        "Ask W. Price": f"{ask_info.get('weighted_price'):.4f}" if ask_info.get('weighted_price') is not None else "N/A",
                    })
                ob_sections_df = pd.DataFrame(section_rows)
                st.dataframe(ob_sections_df, width="stretch", hide_index=True)
            
            aggregated_bins = orderbook_data.get('aggregated_bins', {})
            if aggregated_bins:
                st.markdown("#### Depth by 2% Aggregated Bins")
                summary_rows = []
                for range_label, data in aggregated_bins.items():
                    summary_rows.append({
                        "Range": range_label,
                        "Bid Volume": f"{data.get('total_bid_volume', 0):.2f}",
                        "Ask Volume": f"{data.get('total_ask_volume', 0):.2f}",
                        "Imbalance": f"{(data.get('total_bid_volume', 0) - data.get('total_ask_volume', 0)):.2f}"
                    })
                if summary_rows:
                    agg_summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(agg_summary_df, width="stretch", hide_index=True)
                
                for range_label, data in aggregated_bins.items():
                    with st.expander(f"{range_label} Range Breakdown", expanded=False):
                        bid_bins = data.get('bid_bins_2pct', [])
                        ask_bins = data.get('ask_bins_2pct', [])
                        bid_df = pd.DataFrame([
                            {
                                "Bin": f"{idx * 2}-{(idx + 1) * 2}%",
                                "Orders": bin_info.get('count', 0),
                                "Volume": round(bin_info.get('volume', 0), 2),
                                "Avg Price": f"${bin_info.get('avg_price', 0):.4f}" if bin_info.get('avg_price') else "N/A"
                            }
                            for idx, bin_info in enumerate(bid_bins)
                        ])
                        ask_df = pd.DataFrame([
                            {
                                "Bin": f"{idx * 2}-{(idx + 1) * 2}%",
                                "Orders": bin_info.get('count', 0),
                                "Volume": round(bin_info.get('volume', 0), 2),
                                "Avg Price": f"${bin_info.get('avg_price', 0):.4f}" if bin_info.get('avg_price') else "N/A"
                            }
                            for idx, bin_info in enumerate(ask_bins)
                        ])
                        b_col, a_col = st.columns(2)
                        with b_col:
                            st.markdown("**Bid Bins**")
                            if not bid_df.empty:
                                st.dataframe(bid_df, width="stretch", hide_index=True)
                            else:
                                st.info("No bid volume in this range")
                        with a_col:
                            st.markdown("**Ask Bins**")
                            if not ask_df.empty:
                                st.dataframe(ask_df, width="stretch", hide_index=True)
                            else:
                                st.info("No ask volume in this range")

            advanced = payload.get("advanced", {})
            market_context_data = advanced.get("market_context", {})
            orderbook_context = market_context_data.get("orderbook_context", {})
            mm_activity = orderbook_context.get("market_maker_activity", {})

            if mm_activity and not mm_activity.get("error"):
                st.markdown("---")
                st.markdown("### 🤖 Market Maker Detection (Real-Time)")

                if mm_activity.get("warning"):
                    st.warning(mm_activity["warning"])
                else:
                    mm_detected = mm_activity.get("market_maker_detected", False)
                    confidence = mm_activity.get("confidence", 0)
                    activity_level = mm_activity.get("activity_level", "unknown")

                    mm_col1, mm_col2, mm_col3 = st.columns(3)

                    with mm_col1:
                        status_emoji = "✅" if mm_detected else "❌"
                        st.metric(
                            "Market Maker Detected",
                            f"{status_emoji} {'YES' if mm_detected else 'NO'}"
                        )

                    with mm_col2:
                        st.metric("Confidence", f"{confidence}%")
                        st.progress(confidence / 100)

                    with mm_col3:
                        activity_emoji = "🟢" if activity_level == "high" else "🟡" if activity_level == "medium" else "⚪"
                        st.metric(
                            "Activity Level",
                            f"{activity_emoji} {activity_level.upper()}"
                        )

                    signals = mm_activity.get("signals", [])
                    if signals:
                        st.markdown("#### Detected Signals")
                        signal_tags = " • ".join([f"`{s.replace('_', ' ').title()}`" for s in signals])
                        st.markdown(signal_tags)

                    interpretation = mm_activity.get("interpretation", "")
                    if interpretation:
                        st.info(interpretation)

                    details = mm_activity.get("details", {})

                    with st.expander("📊 Order Walls Analysis", expanded=False):
                        walls = details.get("order_walls", {})
                        wall_col1, wall_col2 = st.columns(2)

                        with wall_col1:
                            st.markdown("**Bid Walls**")
                            bid_walls = walls.get("bid_walls", [])
                            if bid_walls:
                                bid_walls_df = pd.DataFrame([
                                    {
                                        "Price": f"${w['price']:.8f}",
                                        "Volume": f"{w['volume']:.2f}",
                                        "Ratio": f"{w['volume_ratio']:.2f}x",
                                        "Distance": f"{w['distance_from_mid_pct']:.3f}%" if w.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for w in bid_walls
                                ])
                                st.dataframe(bid_walls_df, width="stretch", hide_index=True)
                            else:
                                st.info("No significant bid walls detected")

                        with wall_col2:
                            st.markdown("**Ask Walls**")
                            ask_walls = walls.get("ask_walls", [])
                            if ask_walls:
                                ask_walls_df = pd.DataFrame([
                                    {
                                        "Price": f"${w['price']:.8f}",
                                        "Volume": f"{w['volume']:.2f}",
                                        "Ratio": f"{w['volume_ratio']:.2f}x",
                                        "Distance": f"{w['distance_from_mid_pct']:.3f}%" if w.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for w in ask_walls
                                ])
                                st.dataframe(ask_walls_df, width="stretch", hide_index=True)
                            else:
                                st.info("No significant ask walls detected")

                        wall_pressure = walls.get("wall_pressure", "neutral")
                        wall_emoji = "🟢" if wall_pressure == "bullish" else "🔴" if wall_pressure == "bearish" else "⚪"
                        st.markdown(f"**Wall Pressure:** {wall_emoji} {wall_pressure.upper()}")

                    with st.expander("🔄 Layered Orders Analysis", expanded=False):
                        layers = details.get("layered_orders", {})
                        layering_score = layers.get("layering_score", 0)
                        st.metric("Layering Score", f"{layering_score}/100")
                        st.progress(layering_score / 100)

                        layer_col1, layer_col2 = st.columns(2)

                        with layer_col1:
                            st.markdown("**Bid Layers**")
                            bid_layers = layers.get("bid_layers", [])
                            if bid_layers:
                                bid_layers_df = pd.DataFrame([
                                    {
                                        "Start": f"${l['start_price']:.8f}",
                                        "End": f"${l['end_price']:.8f}",
                                        "Levels": l['levels'],
                                        "Volume": f"{l['total_volume']:.2f}",
                                        "Distance": f"{l['distance_from_mid_pct']:.3f}%" if l.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for l in bid_layers
                                ])
                                st.dataframe(bid_layers_df, width="stretch", hide_index=True)
                            else:
                                st.info("No bid layers detected")

                        with layer_col2:
                            st.markdown("**Ask Layers**")
                            ask_layers = layers.get("ask_layers", [])
                            if ask_layers:
                                ask_layers_df = pd.DataFrame([
                                    {
                                        "Start": f"${l['start_price']:.8f}",
                                        "End": f"${l['end_price']:.8f}",
                                        "Levels": l['levels'],
                                        "Volume": f"{l['total_volume']:.2f}",
                                        "Distance": f"{l['distance_from_mid_pct']:.3f}%" if l.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for l in ask_layers
                                ])
                                st.dataframe(ask_layers_df, width="stretch", hide_index=True)
                            else:
                                st.info("No ask layers detected")

                    with st.expander("🚨 Quote Stuffing Analysis", expanded=False):
                        stuffing = details.get("quote_stuffing", {})
                        stuffing_detected = stuffing.get("stuffing_detected", False)
                        concentration_score = stuffing.get("concentration_score", 0)

                        stuff_col1, stuff_col2 = st.columns(2)

                        with stuff_col1:
                            st.metric("Stuffing Detected", "⚠️ YES" if stuffing_detected else "✅ NO")
                            st.metric("Concentration Score", f"{concentration_score:.2f}/100")

                        with stuff_col2:
                            bid_conc = stuffing.get("bid_concentration", {})
                            ask_conc = stuffing.get("ask_concentration", {})
                            st.metric("Bid Density", f"{bid_conc.get('density', 0):.2f}%")
                            st.metric("Ask Density", f"{ask_conc.get('density', 0):.2f}%")

                    with st.expander("📊 Spread Manipulation Analysis", expanded=False):
                        manipulation = details.get("spread_analysis", {})
                        manip_risk = manipulation.get("manipulation_risk", "unknown")
                        manip_score = manipulation.get("manipulation_score", 0)
                        spread_quality = manipulation.get("spread_quality", "unknown")

                        manip_col1, manip_col2, manip_col3 = st.columns(3)

                        with manip_col1:
                            risk_emoji = "🔴" if manip_risk == "high" else "🟡" if manip_risk == "medium" else "🟢"
                            st.metric("Manipulation Risk", f"{risk_emoji} {manip_risk.upper()}")

                        with manip_col2:
                            st.metric("Manipulation Score", f"{manip_score}/100")

                        with manip_col3:
                            quality_emoji = "🟢" if spread_quality == "good" else "🟡" if spread_quality == "fair" else "🔴"
                            st.metric("Spread Quality", f"{quality_emoji} {spread_quality.upper()}")

                        indicators = manipulation.get("manipulation_indicators", [])
                        if indicators:
                            st.markdown("**Detected Indicators:**")
                            indicator_text = " • ".join([f"`{ind.replace('_', ' ').title()}`" for ind in indicators])
                            st.markdown(indicator_text)
    
    with signals_tab:
        st.subheader("Signals & Zones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trading Signals")
            signals = payload.get("signals", [])
            
            if signals:
                signals_df = pd.DataFrame([
                    {
                        "Type": "🟢 BUY" if s["type"] == "bullish" else "🔴 SELL",
                        "Price": f"${s['price']:.4f}",
                        "Time": s.get("time_iso", "N/A")[:19],
                        "Strength": f"{s.get('strength', 0):.2f}" if s.get('strength') else "N/A",
                    }
                    for s in signals[-20:]
                ])
                st.dataframe(signals_df, width="stretch", hide_index=True)
            else:
                st.info("No signals detected in the analysis period.")
        
        with col2:
            st.markdown("### Active Zones")
            zones = payload.get("zones", [])
            
            if zones:
                zones_df = pd.DataFrame([
                    {
                        "Type": z["type"],
                        "Top": f"{z['top']:.4f}",
                        "Bottom": f"{z['bottom']:.4f}",
                        "Breaker": "✅" if z.get("breaker") else "❌",
                    }
                    for z in zones[:20]
                ])
                st.dataframe(zones_df, width="stretch", hide_index=True)
            else:
                st.info("No active zones detected.")
        
        st.markdown("---")
        st.markdown("### Structure Levels")
        structure_levels = payload.get("last_structure_levels", {})
        if structure_levels:
            struct_col1, struct_col2 = st.columns(2)
            with struct_col1:
                high_level = structure_levels.get("high")
                st.metric("Structure High", f"${high_level:.4f}" if high_level else "N/A")
            with struct_col2:
                low_level = structure_levels.get("low")
                st.metric("Structure Low", f"${low_level:.4f}" if low_level else "N/A")
    
    with volume_tab:
        st.subheader("📊 Volume Analysis")
        advanced = payload.get("advanced", {})
        volume_analysis = advanced.get("volume_analysis", {})
        
        st.markdown("### Volume Profile (VPVR)")
        vpvr = volume_analysis.get("vpvr", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            poc = vpvr.get("poc")
            st.metric("Point of Control (POC)", f"${poc:.4f}" if poc else "N/A")
        with col2:
            va_high = vpvr.get("value_area", {}).get("high")
            st.metric("Value Area High", f"${va_high:.4f}" if va_high else "N/A")
        with col3:
            va_low = vpvr.get("value_area", {}).get("low")
            st.metric("Value Area Low", f"${va_low:.4f}" if va_low else "N/A")
        
        levels = vpvr.get("levels", [])
        if levels:
            st.markdown("#### Top Volume Levels")
            vpvr_df = pd.DataFrame([
                {
                    "Price": f"${level['price']:.4f}",
                    "Volume": f"{level['volume']:,.0f}",
                    "Percentage": f"{level['percentage']:.2f}%"
                }
                for level in levels[:10]
            ])
            st.dataframe(vpvr_df, width="stretch", hide_index=True)
        
        st.markdown("---")
        st.markdown("### Cumulative Volume Delta (CVD)")
        cvd = volume_analysis.get("cvd", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest CVD", f"{cvd.get('latest', 0):,.0f}")
        with col2:
            st.metric("CVD Change", f"{cvd.get('change', 0):,.0f}")
        
        cvd_series = cvd.get("series", [])
        if cvd_series:
            recent_cvd = cvd_series[-10:]
            cvd_df = pd.DataFrame([
                {
                    "Time": entry.get("time_iso", "")[:19],
                    "CVD": f"{entry.get('value', 0):,.0f}",
                    "Delta": f"{entry.get('delta', 0):,.0f}",
                    "Buy Volume": f"{entry.get('buy_volume', 0):,.0f}",
                    "Sell Volume": f"{entry.get('sell_volume', 0):,.0f}"
                }
                for entry in recent_cvd
            ])
            st.dataframe(cvd_df, width="stretch", hide_index=True)
        
        st.markdown("---")
        st.markdown("### Delta Volume (Market vs Limit Orders)")
        delta = volume_analysis.get("delta", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest Delta", f"{delta.get('latest', 0):,.0f}")
        with col2:
            st.metric("Average Delta", f"{delta.get('average', 0):,.0f}")
        
        delta_series = delta.get("series", [])
        if delta_series:
            delta_df = pd.DataFrame([
                {
                    "Time": entry.get("time_iso", "")[:19],
                    "Delta": f"{entry.get('delta', 0):,.0f}",
                    "Market Orders": f"{entry.get('market_orders', 0):,.0f}",
                    "Limit Orders": f"{entry.get('limit_orders', 0):,.0f}",
                    "Imbalance Ratio": "N/A" if entry.get('imbalance_ratio') is None else f"{entry.get('imbalance_ratio', 0):.2f}"
                }
                for entry in delta_series[-10:]
            ])
            st.dataframe(delta_df, width="stretch", hide_index=True)
    
    with structure_tab:
        st.subheader("🏗️ Market Structure")
        market_structure = advanced.get("market_structure", {})
        
        trend = market_structure.get("trend", "neutral")
        trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "⚪"
        st.markdown(f"### Current Trend: {trend_emoji} **{trend.upper()}**")
        
        st.markdown("---")
        st.markdown("### Swing Points")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Higher Highs (HH)")
            swing_points = market_structure.get("swing_points", {})
            hh = swing_points.get("hh", [])
            if hh:
                hh_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in hh
                ])
                st.dataframe(hh_df, width="stretch", hide_index=True)
            else:
                st.info("No HH detected")
            
            st.markdown("#### Lower Highs (LH)")
            lh = swing_points.get("lh", [])
            if lh:
                lh_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in lh
                ])
                st.dataframe(lh_df, width="stretch", hide_index=True)
            else:
                st.info("No LH detected")
        
        with col2:
            st.markdown("#### Higher Lows (HL)")
            hl = swing_points.get("hl", [])
            if hl:
                hl_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in hl
                ])
                st.dataframe(hl_df, width="stretch", hide_index=True)
            else:
                st.info("No HL detected")
            
            st.markdown("#### Lower Lows (LL)")
            ll = swing_points.get("ll", [])
            if ll:
                ll_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in ll
                ])
                st.dataframe(ll_df, width="stretch", hide_index=True)
            else:
                st.info("No LL detected")
        
        st.markdown("---")
        st.markdown("### Key Support & Resistance Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Support Levels")
            key_levels = market_structure.get("key_levels", {})
            support = key_levels.get("support", [])
            if support:
                support_df = pd.DataFrame([
                    {
                        "Price": f"${level['price']:.4f}",
                        "Strength": f"{level['strength']:.2f}"
                    }
                    for level in support
                ])
                st.dataframe(support_df, width="stretch", hide_index=True)
            else:
                st.info("No support levels detected")
        
        with col2:
            st.markdown("#### Resistance Levels")
            resistance = key_levels.get("resistance", [])
            if resistance:
                resistance_df = pd.DataFrame([
                    {
                        "Price": f"${level['price']:.4f}",
                        "Strength": f"{level['strength']:.2f}"
                    }
                    for level in resistance
                ])
                st.dataframe(resistance_df, width="stretch", hide_index=True)
            else:
                st.info("No resistance levels detected")
        
        st.markdown("---")
        st.markdown("### Liquidity Zones")
        liquidity_zones = market_structure.get("liquidity_zones", [])
        if liquidity_zones:
            liq_df = pd.DataFrame([
                {
                    "Type": zone["type"].upper(),
                    "Price": f"${zone['price']:.4f}",
                    "Volume Ratio": f"{zone['volume_ratio']:.4f}"
                }
                for zone in liquidity_zones
            ])
            st.dataframe(liq_df, width="stretch", hide_index=True)
        else:
            st.info("No significant liquidity zones detected")
    
    with fundamentals_tab:
        st.subheader("📈 Fundamental Metrics")
        fundamentals = advanced.get("fundamentals", {})
        
        st.markdown("### Funding Rate")
        funding_rate = fundamentals.get("funding_rate", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Rate", f"{funding_rate.get('current', 0):.4%}")
        with col2:
            st.metric("Predicted Rate", f"{funding_rate.get('predicted', 0):.4%}")
        with col3:
            st.metric("Annualized Rate", f"{funding_rate.get('annualized', 0):.2f}%")
        
        st.markdown("---")
        st.markdown("### Open Interest")
        oi = fundamentals.get("open_interest", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current OI", f"${oi.get('current', 0):,.0f}")
        with col2:
            st.metric("OI Change %", f"{oi.get('change_pct', 0):.2f}%")
        
        st.markdown("---")
        st.markdown("### Long/Short Ratio")
        ls_ratio = fundamentals.get("long_short_ratio", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Long Ratio", f"{ls_ratio.get('long', 0):.3f}")
        with col2:
            st.metric("Short Ratio", f"{ls_ratio.get('short', 0):.3f}")
        with col3:
            ratio_val = ls_ratio.get('ratio', 0)
            st.metric("L/S Ratio", f"{ratio_val:.2f}")
        
        st.markdown("---")
        st.markdown("### Block Trades")
        block_trades = fundamentals.get("block_trades", [])
        if block_trades:
            bt_df = pd.DataFrame([
                {
                    "Time": trade.get("time_iso", "")[:19],
                    "Price": f"${trade['price']:.4f}",
                    "Volume": f"{trade['volume']:,.0f}",
                    "Side": trade["side"].upper()
                }
                for trade in block_trades
            ])
            st.dataframe(bt_df, width="stretch", hide_index=True)
        else:
            st.info("No significant block trades detected")
    
    with breadth_tab:
        st.subheader("🌐 Breadth Indicators")
        advanced_data = payload.get("advanced", {})
        breadth = advanced_data.get("breadth", {})
        
        if not breadth:
            st.info("Breadth data is not available for the selected configuration.")
        else:
            st.markdown(
                "These metrics blend macro sentiment with cross-market behaviour to highlight how broad the current trend really is."
            )
            
            fear_greed = breadth.get("fear_greed_index", 50)
            regime = breadth.get("regime", "Neutral")
            note = breadth.get("note")
            source = breadth.get("source", "unavailable")
            
            sentiment_col, correlation_col, macro_col = st.columns([1.2, 1, 1])
            
            with sentiment_col:
                if fear_greed >= 70:
                    sentiment_emoji = "🟢"
                    sentiment_text = "Extreme Greed"
                elif fear_greed >= 55:
                    sentiment_emoji = "🟡"
                    sentiment_text = "Greed"
                elif fear_greed <= 25:
                    sentiment_emoji = "🔴"
                    sentiment_text = "Extreme Fear"
                elif fear_greed <= 45:
                    sentiment_emoji = "🟠"
                    sentiment_text = "Fear"
                else:
                    sentiment_emoji = "⚪"
                    sentiment_text = "Neutral"
                
                st.metric("Fear & Greed Index", f"{sentiment_emoji} {fear_greed:.1f}")
                st.caption(f"{sentiment_text} — extremes often precede mean-reversion moves.")
                st.progress(fear_greed / 100)
                st.markdown(f"**Regime:** {regime.upper()}")
                if note:
                    st.caption(note)
                st.caption(f"Source: {source}")
            
            with correlation_col:
                st.markdown("### Cross-Market Correlations")
                btc_corr = breadth.get("btc_correlation")
                sp_corr = breadth.get("sp500_correlation")
                
                if btc_corr is None:
                    st.metric("BTC Correlation", "N/A")
                else:
                    st.metric("BTC Correlation", format_correlation(btc_corr))
                st.caption("How closely the asset tracks Bitcoin's daily move.")
                
                if sp_corr is None:
                    st.metric("S&P 500 Correlation", "N/A")
                else:
                    st.metric("S&P 500 Correlation", format_correlation(sp_corr))
                st.caption("Links to traditional risk-on equities.")
            
            with macro_col:
                st.markdown("### Macro Backdrop")
                dxy = breadth.get("dollar_index_dxy")
                if dxy is None:
                    st.metric("Dollar Index (DXY)", "N/A")
                else:
                    if dxy >= 105:
                        dxy_emoji = "🔴"
                    elif dxy >= 100:
                        dxy_emoji = "🟡"
                    else:
                        dxy_emoji = "🟢"
                    st.metric("Dollar Index (DXY)", f"{dxy_emoji} {dxy:.2f}")
                st.caption("A stronger dollar often weighs on crypto risk appetite.")
                
                vix_value = breadth.get("vix_index")
                if vix_value is None:
                    st.metric("Volatility Index (VIX)", "N/A")
                else:
                    if vix_value >= 30:
                        vix_emoji = "🔴"
                    elif vix_value >= 20:
                        vix_emoji = "🟡"
                    else:
                        vix_emoji = "🟢"
                    st.metric("Volatility Index (VIX)", f"{vix_emoji} {vix_value:.2f}")
                st.caption("Elevated volatility signals stress across risk assets.")
                
                yields = breadth.get("treasury_yields", {})
                two_year = yields.get("2y")
                ten_year = yields.get("10y")
                if two_year is not None:
                    st.metric("US 2Y Yield", f"{two_year:.3f}%")
                if ten_year is not None:
                    st.metric("US 10Y Yield", f"{ten_year:.3f}%")
                if two_year is not None and ten_year is not None:
                    curve = two_year - ten_year
                    if curve >= 0:
                        curve_emoji = "🔴"
                    elif curve > -0.3:
                        curve_emoji = "🟡"
                    else:
                        curve_emoji = "🟢"
                    st.metric("Yield Curve (2Y-10Y)", f"{curve_emoji} {curve:.2f}%")
                    st.caption("An inverted curve (2Y > 10Y) signals tightening liquidity conditions.")
    
    with onchain_tab:
        st.subheader("🔗 On-chain Metrics")
        advanced_data = payload.get("advanced", {})
        onchain = advanced_data.get("onchain_metrics", {})
        exchange_flows = onchain.get("exchange_flows", {})
        
        if not exchange_flows:
            st.info("On-chain exchange flow estimates are not available for this asset/timeframe.")
        else:
            st.markdown(
                "Exchange flow estimates highlight whether capital is moving onto exchanges (accumulation) or away from them (distribution)."
            )
            
            net_flow = exchange_flows.get("net_flow", 0.0)
            inflow_usd = exchange_flows.get("inflow", 0.0)
            outflow_usd = exchange_flows.get("outflow", 0.0)
            inflow_asset = exchange_flows.get("inflow_btc", 0.0)
            outflow_asset = exchange_flows.get("outflow_btc", 0.0)
            total_turnover = abs(inflow_usd) + abs(outflow_usd)
            flow_bias = (net_flow / total_turnover * 100) if total_turnover else 0.0
            
            summary_col, bias_col = st.columns(2)
            with summary_col:
                st.metric("Net Flow (USD)", format_flow(net_flow))
                st.caption("Positive values show net accumulation; negative values indicate distribution pressure.")
            with bias_col:
                if total_turnover:
                    if flow_bias >= 5:
                        bias_emoji = "🟢"
                    elif flow_bias <= -5:
                        bias_emoji = "🔴"
                    else:
                        bias_emoji = "⚪"
                    st.metric("Flow Bias", f"{bias_emoji} {flow_bias:+.2f}%")
                else:
                    st.metric("Flow Bias", "N/A")
                st.caption("Net flow relative to total turnover over the analysed window.")
            
            usd_col1, usd_col2 = st.columns(2)
            with usd_col1:
                st.metric("Inflow (USD)", format_flow(inflow_usd))
                st.caption("Buying pressure routed through exchanges.")
            with usd_col2:
                st.metric("Outflow (USD)", format_flow(outflow_usd))
                st.caption("Selling pressure or capital leaving exchanges.")
            
            asset_col1, asset_col2 = st.columns(2)
            with asset_col1:
                st.metric("Inflow (Asset Units)", f"{inflow_asset:.4f}")
            with asset_col2:
                st.metric("Outflow (Asset Units)", f"{outflow_asset:.4f}")
            st.caption("Asset unit estimates use the average price across the last 20 analysed candles.")
    
    with composite_tab:
        st.subheader("🧩 Composite Indicators")
        advanced_data = payload.get("advanced", {})
        composite = advanced_data.get("composite_indicators", {})
        
        if not composite:
            st.info("Composite indicators are not available for this asset/timeframe.")
        else:
            st.markdown(
                "Composite scores summarise liquidity quality, market health, and the risk-adjusted trading signal."
            )
            
            liquidity_score = composite.get("liquidity_score", {})
            market_health = composite.get("market_health_index", {})
            risk_signal = composite.get("risk_adjusted_signal", {})
            
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            
            overall_liquidity = liquidity_score.get("overall")
            with overview_col1:
                if overall_liquidity is None:
                    st.metric("Liquidity Score", "N/A")
                else:
                    if overall_liquidity >= 0.7:
                        liquidity_emoji = "🟢"
                    elif overall_liquidity >= 0.4:
                        liquidity_emoji = "🟡"
                    else:
                        liquidity_emoji = "🔴"
                    st.metric("Liquidity Score", f"{liquidity_emoji} {overall_liquidity:.3f}")
                st.caption("Combines order book depth, spreads, and slippage risk.")
            
            overall_health = market_health.get("overall")
            with overview_col2:
                if overall_health is None:
                    st.metric("Market Health", "N/A")
                else:
                    if overall_health >= 0.7:
                        health_emoji = "🟢"
                    elif overall_health >= 0.4:
                        health_emoji = "🟡"
                    else:
                        health_emoji = "🔴"
                    st.metric("Market Health", f"{health_emoji} {overall_health:.3f}")
                st.caption("Balances volatility stability, volume quality, and momentum consistency.")
            
            with overview_col3:
                final_signal = risk_signal.get("final_signal", "NEUTRAL")
                if final_signal == "BUY":
                    signal_emoji = "🟢"
                elif final_signal == "SELL":
                    signal_emoji = "🔴"
                else:
                    signal_emoji = "⚪"
                st.metric("Final Signal", f"{signal_emoji} {final_signal}")
                confidence = risk_signal.get("confidence")
                if confidence is not None:
                    st.metric("Signal Confidence", f"{confidence:.3f}")
                    st.progress(confidence)
                risk_adjustment = risk_signal.get("risk_adjustment")
                if risk_adjustment is not None:
                    st.caption(f"Risk adjustment applied: {risk_adjustment:+.3f}")
                raw_signal = risk_signal.get("raw_signal")
                if raw_signal and raw_signal != final_signal:
                    st.caption(f"Raw trend signal was {raw_signal}; adjustments tempered the outcome.")
            
            st.markdown("---")
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("### Liquidity Components")
                liq_rows = []
                for key, label in (
                    ("depth_quality", "Depth Quality"),
                    ("spread_efficiency", "Spread Efficiency"),
                    ("slippage_risk", "Slippage Risk"),
                ):
                    value = liquidity_score.get(key)
                    if value is not None:
                        liq_rows.append({"Component": label, "Score": f"{value:.3f}"})
                if liq_rows:
                    st.dataframe(pd.DataFrame(liq_rows), width="stretch", hide_index=True)
                else:
                    st.info("No liquidity breakdown available.")
            
            with detail_col2:
                st.markdown("### Market Health Components")
                health_rows = []
                for key, label in (
                    ("volatility_stability", "Volatility Stability"),
                    ("volume_quality", "Volume Quality"),
                    ("momentum_consistency", "Momentum Consistency"),
                ):
                    value = market_health.get(key)
                    if value is not None:
                        health_rows.append({"Component": label, "Score": f"{value:.3f}"})
                if health_rows:
                    st.dataframe(pd.DataFrame(health_rows), width="stretch", hide_index=True)
                else:
                    st.info("No market health breakdown available.")
            
            risk_factors = risk_signal.get("risk_factors", [])
            if risk_factors:
                readable = [factor.replace("_", " ").title() for factor in risk_factors]
                st.markdown("### Risk Factors Considered")
                st.write(" • ".join(readable))
            else:
                st.caption("No additional risk factors flagged in this analysis.")
    
    with patterns_tab:
        st.subheader("🌊 Patterns & Waves")
        patterns = advanced.get("patterns", {})
        
        st.markdown("### Elliott Wave Analysis")
        elliott = patterns.get("elliott", {}) or {}
        current_wave = elliott.get("current_wave") or {}
        total_waves = elliott.get("total_waves") or elliott.get("wave_count", 0)
        current_wave_label = (
            current_wave.get("label")
            or elliott.get("current_wave_label")
            or elliott.get("label", "Unknown")
        )
        current_direction = current_wave.get("direction")
        pivot_structure = (
            current_wave.get("structure_label")
            or current_wave.get("structure")
            or elliott.get("structure", "Unknown")
        )
        structure_regime = elliott.get("structure", "Unknown")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wave Count", total_waves if total_waves is not None else 0)
        with col2:
            delta_label = current_direction.title() if isinstance(current_direction, str) else None
            st.metric("Current Wave", current_wave_label, delta=delta_label)
        with col3:
            pivot_display = str(pivot_structure).replace("_", " ").title() if pivot_structure else "Unknown"
            regime_display = str(structure_regime).replace("_", " ").title() if structure_regime else "—"
            st.metric("Pivot Structure", pivot_display, delta=regime_display if regime_display else None)

        pivot_price = safe_float(current_wave.get("price"))
        pivot_time = current_wave.get("time_iso")
        caption_parts = []
        if pivot_price is not None:
            caption_parts.append(f"Price ${pivot_price:.4f}")
        if pivot_time:
            caption_parts.append(pivot_time.replace("T", " ")[:19])
        if caption_parts:
            st.caption("Last pivot: " + " • ".join(caption_parts))
        
        pivot_points = elliott.get("pivot_points", [])
        if pivot_points:
            st.markdown("#### Pivot Points")
            pivot_df = pd.DataFrame([
                {
                    "Time": point.get("time_iso", "")[:19],
                    "Price": f"${point.get('price', 0):.4f}",
                    "Type": point.get("type", ""),
                }
                for point in pivot_points
            ])
            st.dataframe(
                pivot_df,
                width="stretch",
                hide_index=True,
                key=ui_key(
                    "patterns",
                    f"pivot_points_{config_store.symbol}_{config_store.timeframe}"
                ),
            )
        
        st.markdown("---")
        st.markdown("### Orderbook Clusters")
        clusters = patterns.get("orderbook_clusters", [])
        if clusters:
            cluster_df = pd.DataFrame([
                {
                    "Side": cluster["side"].upper(),
                    "Price": f"${cluster['price']:.4f}",
                    "Volume": f"{cluster['volume']:,.2f}",
                    "Strength": f"{cluster['strength']:.2f}x",
                }
                for cluster in clusters
            ])
            st.dataframe(
                cluster_df,
                width="stretch",
                hide_index=True,
                key=ui_key(
                    "patterns",
                    f"clusters_{config_store.symbol}_{config_store.timeframe}"
                ),
            )
        else:
            st.info("No significant orderbook clusters detected")
        
        st.markdown("---")
        st.markdown("### Liquidity Anomalies")
        anomalies = patterns.get("liquidity_anomalies", [])
        if anomalies:
            anom_df = pd.DataFrame([
                {
                    "Time": anom.get("time_iso", "")[:19],
                    "Type": anom["type"].upper().replace("_", " "),
                    "Price": f"${anom['price']:.4f}",
                    "Severity": f"{anom['severity']:.2f}x",
                    "Description": anom["description"],
                }
                for anom in anomalies
            ])
            st.dataframe(
                anom_df,
                width="stretch",
                hide_index=True,
                key=ui_key(
                    "patterns",
                    f"liquidity_anomalies_{config_store.symbol}_{config_store.timeframe}"
                ),
            )
        else:
            st.info("No liquidity anomalies detected")
    
    with trade_tab:
        st.subheader("🎯 Trade Signal Calculator")
        trade_plan = advanced.get("trade_plan", {})
        signal_analysis = advanced.get("signal_analysis", {})
        
        if signal_analysis:
            st.markdown("### 📊 Historical Signal Performance")
            
            bullish_stats = signal_analysis.get("bullish", {})
            bearish_stats = signal_analysis.get("bearish", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Bullish Signals")
                bull_df = pd.DataFrame([
                    {"Metric": "Total Signals", "Value": f"{bullish_stats.get('total_signals', 0):,}"},
                    {"Metric": "TP1 Hit Rate", "Value": f"{bullish_stats.get('tp1_rate_pct', 0):.2f}%"},
                    {"Metric": "TP2 Hit Rate", "Value": f"{bullish_stats.get('tp2_rate_pct', 0):.2f}%"},
                    {"Metric": "TP3 Hit Rate", "Value": f"{bullish_stats.get('tp3_rate_pct', 0):.2f}%"},
                    {"Metric": "SL Hit Rate", "Value": f"{bullish_stats.get('sl_rate_pct', 0):.2f}%"},
                    {"Metric": "Win Rate", "Value": f"{bullish_stats.get('overall_win_rate_pct', 0):.2f}%"},
                    {"Metric": "Avg Bars to TP1", "Value": f"{bullish_stats.get('avg_bars_to_tp1', 0):.1f}"},
                ])
                st.dataframe(bull_df, width="stretch", hide_index=True)
            
            with col2:
                st.markdown("#### 🔴 Bearish Signals")
                bear_df = pd.DataFrame([
                    {"Metric": "Total Signals", "Value": f"{bearish_stats.get('total_signals', 0):,}"},
                    {"Metric": "TP1 Hit Rate", "Value": f"{bearish_stats.get('tp1_rate_pct', 0):.2f}%"},
                    {"Metric": "TP2 Hit Rate", "Value": f"{bearish_stats.get('tp2_rate_pct', 0):.2f}%"},
                    {"Metric": "TP3 Hit Rate", "Value": f"{bearish_stats.get('tp3_rate_pct', 0):.2f}%"},
                    {"Metric": "SL Hit Rate", "Value": f"{bearish_stats.get('sl_rate_pct', 0):.2f}%"},
                    {"Metric": "Win Rate", "Value": f"{bearish_stats.get('overall_win_rate_pct', 0):.2f}%"},
                    {"Metric": "Avg Bars to TP1", "Value": f"{bearish_stats.get('avg_bars_to_tp1', 0):.1f}"},
                ])
                st.dataframe(bear_df, width="stretch", hide_index=True)
            
            st.markdown("---")
        
        if not trade_plan:
            st.warning("No trade plan available. Generate signals first.")
        else:
            signal = trade_plan.get("signal", {})
            risk = trade_plan.get("risk", {})
            position = trade_plan.get("position", {})
            targets = trade_plan.get("targets", [])
            
            signal_type = signal.get("type", "NEUTRAL")
            if signal_type == "BUY":
                st.success(f"### 🟢 BUY SIGNAL")
            elif signal_type == "SELL":
                st.error(f"### 🔴 SELL SIGNAL")
            else:
                st.info(f"### ⚪ NO ACTIVE SIGNAL")
            
            st.markdown("---")
            st.markdown("### 🎛️ Position Size Calculator")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_position_size = num_float(
                    "Position Size (USD)",
                    min_v=10.0,
                    max_v=100000.0,
                    value=risk.get('risk_amount', 100.0),
                    step=10.0,
                    key="custom_position_size",
                )
            with col2:
                custom_leverage = st.slider(
                    "Leverage",
                    min_value=1,
                    max_value=125,
                    value=int(position.get('leverage', 10)),
                    step=1,
                    key="custom_leverage"
                )
            with col3:
                commission_rate = position.get('commission_rate', 0.0006)
                st.metric("Commission Rate", f"{commission_rate * 100:.02f}%")
            
            entry_price = signal.get('entry_price', 0)
            stop_loss = risk.get('stop_loss', 0)
            atr_value = risk.get('atr', 0)
            
            if entry_price and stop_loss and atr_value:
                is_long = entry_price > stop_loss
                custom_metrics = calculate_position_metrics(
                    entry_price,
                    custom_position_size,
                    custom_leverage,
                    commission_rate
                )
                
                custom_levels = calculate_tp_sl_levels(
                    entry_price,
                    is_long,
                    atr_value
                )
                
                risk_per_unit = abs(entry_price - stop_loss)
                quantity = custom_metrics['quantity']
                max_loss = risk_per_unit * quantity + custom_metrics['entry_commission']
                
                st.markdown("---")
                st.markdown("### Entry & Risk Parameters")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry Price", f"${entry_price:.4f}")
                    st.metric("Stop Loss", f"${custom_levels['sl']:.4f}")
                with col2:
                    st.metric("ATR Value", f"${atr_value:.4f}")
                    st.metric("Risk Amount", f"${custom_position_size:.2f}")
                with col3:
                    st.metric("Max Loss", f"${max_loss:.2f}")
                    st.metric("Quantity", f"{quantity:.4f}")
                
                st.markdown("---")
                st.markdown(f"### Position Details ({custom_leverage}x Leverage)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Position Size", f"${custom_position_size:.2f}")
                with col2:
                    st.metric("Notional Value", f"${custom_metrics['notional_value']:.2f}")
                with col3:
                    st.metric("Est. Commission", f"${custom_metrics['entry_commission']:.2f}")
                
                st.markdown("---")
                st.markdown("### Take Profit Targets (ATR-based)")
                
                custom_targets = []
                for tp_key in ['tp1', 'tp2', 'tp3']:
                    tp_price = custom_levels.get(tp_key)
                    if tp_price:
                        gross = abs(tp_price - entry_price) * quantity
                        net = gross - custom_metrics['entry_commission'] * 2
                        custom_targets.append({
                            "Target": tp_key.upper(),
                            "Price": f"${tp_price:.4f}",
                            "Gross P&L": f"${gross:.2f}",
                            "Net P&L": f"${net:.2f}",
                            "R:R": f"{(gross / max_loss):.2f}x" if max_loss else "N/A"
                        })
                
                if custom_targets:
                    custom_targets_df = pd.DataFrame(custom_targets)
                    st.dataframe(custom_targets_df, width="stretch", hide_index=True)
            
            st.markdown("---")
            st.info("⚠️ **Disclaimer:** This is a calculated trade plan based on ATR channels. Always manage your risk and use proper position sizing.")
    
    with automated_signals_tab:
        st.subheader("🤖 Automated Signals")
        state = st.session_state.setdefault(AUTOMATED_SIGNALS_STATE_KEY, {})
        
        if "signal_executor" not in st.session_state:
            st.session_state.signal_executor = SignalExecutor(update_bus=getattr(st.session_state, "signals_update_bus", None))
        executor = st.session_state.signal_executor

        cache_identity = f"{config_store.symbol}|{config_store.timeframe}"
        previous_identity = state.get("cache_identity")
        if previous_identity and previous_identity != cache_identity:
            cached_run_automated_signals.clear()
            state["end_time_user_set"] = False
            state.pop("auto_end_timestamp", None)
            state.pop("auto_end_time_ms", None)
            state.pop("analysis_updated", None)
            auto_toggle_key = ui_key("automated_signals", "auto_advance_end")
            st.session_state.pop(auto_toggle_key, None)
            
            # Stop existing worker if any
            existing_worker = getattr(st.session_state, "automated_signals_worker", None)
            if existing_worker is not None:
                try:
                    existing_worker.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop existing worker: {e}")
                st.session_state.automated_signals_worker = None
                st.session_state.automated_signals_worker_running = False
        state["cache_identity"] = cache_identity

        # Initialize core state flags
        state.setdefault("end_time_user_set", False)
        state.setdefault("fetch_needed", False)
        state.setdefault("analysis_updated", False)
        state.setdefault("error", None)
        state.setdefault("result", None)

        # Initialize auto end-time if not yet set
        if "auto_end_time_ms" not in state:
            state["auto_end_time_ms"] = int(config_store.end_datetime.timestamp() * 1000)

        st.caption(
            f"Symbol: **{config_store.symbol.upper()}** · Timeframe: **{config_store.timeframe.upper()}**"
        )

        normalized_weights = render_weight_controls(config_store)

        with st.expander("Indicator Parameters", expanded=False):
            render_indicator_controls(config_store)

        with st.expander("Risk & Signal Settings", expanded=False):
            render_signal_risk_controls(config_store)

        with st.expander("ByBit Execution", expanded=False):
            st.warning("⚠️ Testnet only recommended. Use at your own risk.")
            
            exec_enabled = st.toggle("Enable ByBit Execution", value=executor.enabled, key="bybit_enabled")
            
            c1, c2 = st.columns(2)
            with c1:
                api_key = st.text_input("API Key", value=executor.api_key, type="password", key="bybit_key")
                is_testnet = st.selectbox("Network", ["Testnet", "Mainnet"], index=0 if executor.testnet else 1, key="bybit_net") == "Testnet"
                dry_run = st.checkbox("Dry Run (Log only)", value=executor.dry_run, key="bybit_dry")
            with c2:
                api_secret = st.text_input("API Secret", value=executor.api_secret, type="password", key="bybit_secret")
                leverage = st.number_input("Default Leverage", min_value=1, max_value=125, value=executor.default_leverage, key="bybit_lev")
                pos_mult = st.number_input("Pos Size Multiplier", min_value=0.1, max_value=10.0, value=executor.pos_size_multiplier, step=0.1, key="bybit_pos")

            executor.configure(exec_enabled, api_key, api_secret, is_testnet, int(leverage), float(pos_mult), dry_run)

            st.markdown("### Recent Executions")
            if os.path.exists(SignalExecutor.LOG_FILE):
                try:
                    logs = pd.read_csv(SignalExecutor.LOG_FILE)
                    if not logs.empty and "latency_ms" in logs.columns:
                        logs["latency_ms"] = pd.to_numeric(logs["latency_ms"], errors='coerce')
                        avg_lat = logs["latency_ms"].mean()
                        p95 = logs["latency_ms"].quantile(0.95)
                        col_lat1, col_lat2 = st.columns(2)
                        col_lat1.metric("Avg Latency", f"{avg_lat:.0f} ms")
                        col_lat2.metric("95th % Latency", f"{p95:.0f} ms")
                    
                    st.dataframe(logs.tail(10).sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
                except Exception:
                    st.info("No execution logs found.")

        # Auto-advance toggle for End time
        auto_advance_key = ui_key("automated_signals", "auto_advance_end")
        if auto_advance_key not in st.session_state:
            st.session_state[auto_advance_key] = not state.get("end_time_user_set", False)
        auto_advance_end = st.checkbox(
            "🔄 Auto-advance End time to latest TF boundary (with auto-refresh)",
            value=st.session_state[auto_advance_key],
            key=auto_advance_key,
            help=f"When enabled, End time automatically updates to the latest closed {config_store.timeframe.upper()} bar boundary and signals refresh automatically without user action",
        )
        
        # Update end_time_user_set based on checkbox
        prev_end_time_user_set = state.get("end_time_user_set", False)
        state["end_time_user_set"] = not auto_advance_end
        
        # Start or stop worker based on auto-advance state
        worker = getattr(st.session_state, "automated_signals_worker", None)
        if auto_advance_end and not prev_end_time_user_set:
            # Just enabled auto-advance - start worker if not running
            if worker is None or not getattr(st.session_state, "automated_signals_worker_running", False):
                # We'll start the worker after we have all config ready (below)
                pass
        elif not auto_advance_end and prev_end_time_user_set:
            # Just disabled auto-advance - stop worker
            if worker is not None:
                try:
                    worker.stop()
                    logger.info("Stopped automated signals worker (user disabled auto-advance)")
                except Exception as e:
                    logger.warning(f"Failed to stop worker: {e}")
                st.session_state.automated_signals_worker = None
        
        # Widget keys for date/time controls
        end_date_key = ui_key("automated_signals", "end_date")
        end_time_key = ui_key("automated_signals", "end_time")
        
        date_cols = st.columns(2)
        start_dt = config_store.start_datetime
        
        # Determine the effective end datetime based on auto-advance state
        if auto_advance_end:
            auto_end_ms = state.get("auto_end_time_ms") or 0
            if auto_end_ms > 0:
                end_dt = dt.datetime.fromtimestamp(auto_end_ms / 1000, tz=dt.timezone.utc)
            else:
                end_dt = config_store.end_datetime
            auto_end_date = end_dt.date()
            auto_end_time = end_dt.time().replace(microsecond=0)
            if st.session_state.get(end_date_key) != auto_end_date:
                st.session_state[end_date_key] = auto_end_date
            if st.session_state.get(end_time_key) != auto_end_time:
                st.session_state[end_time_key] = auto_end_time
        else:
            end_dt = config_store.end_datetime

        with date_cols[0]:
            start_date = st.date_input(
                "Start date (UTC)",
                value=start_dt.date(),
                key=ui_key("automated_signals", "start_date"),
            )
            start_time = st.time_input(
                "Start time (UTC)",
                value=start_dt.time().replace(microsecond=0),
                key=ui_key("automated_signals", "start_time"),
            )

        with date_cols[1]:
            end_date = st.date_input(
                "End date (UTC)",
                value=end_dt.date(),
                key=ui_key("automated_signals", "end_date"),
            )
            end_time = st.time_input(
                "End time (UTC)",
                value=end_dt.time().replace(microsecond=0),
                key=ui_key("automated_signals", "end_time"),
            )

        updated_start = dt.datetime.combine(start_date, start_time, tzinfo=dt.timezone.utc)
        updated_end = dt.datetime.combine(end_date, end_time, tzinfo=dt.timezone.utc)
        
        # Detect if user explicitly changed end time (only when auto-advance is enabled)
        # If auto-advance is disabled, the user already has control, so no need to override
        if auto_advance_end and updated_end != end_dt:
            # User modified the time even though auto-advance was on - disable auto-advance
            state["end_time_user_set"] = True
            st.session_state[auto_advance_key] = False
            logger.info(f"User explicitly set end time to {updated_end}, disabling auto-advance")
            # Stop worker since auto-advance was disabled
            worker = getattr(st.session_state, "automated_signals_worker", None)
            if worker is not None:
                try:
                    worker.stop()
                except Exception as e:
                    logger.warning(f"Failed to stop worker: {e}")
                st.session_state.automated_signals_worker = None

        if updated_end <= updated_start:
            state["result"] = None
            state["error"] = "End time must be after start time."
        else:
            config_store.set_date_range(updated_start, updated_end)
            start_dt = config_store.start_datetime
            end_dt = config_store.end_datetime

            signal_config = config_store.build_signal_config()
            indicator_params = config_store.get_indicator_params()
            signal_params = config_store.build_signal_params()
            signal_config_payload = {
                "weights": {
                    "technical": signal_config.technical_weight,
                    "sentiment": signal_config.sentiment_weight,
                    "multitimeframe": signal_config.multitimeframe_weight,
                    "volume": signal_config.volume_weight,
                    "market_structure": signal_config.structure_weight,
                    "composite": signal_config.composite_weight,
                },
                "min_confirmations": signal_config.min_factors_confirm,
                "buy_threshold": signal_config.buy_threshold,
                "sell_threshold": signal_config.sell_threshold,
                "min_confidence": signal_config.min_confidence,
            }

            # Store latest config in state for worker reuse
            state["weights"] = normalized_weights
            state["signal_params"] = signal_params
            state["signal_config"] = signal_config_payload
            state["indicator_params"] = indicator_params

            # Prepare timestamps and hashes for refresh decisions
            start_iso = config_store.start_iso()
            end_iso = config_store.end_iso()
            state["start_datetime_iso"] = start_iso
            params_hash = stable_hash(indicator_params)
            weights_hash = stable_hash(signal_config_payload.get("weights", {}))

            # Determine if we need a manual refresh (initial load or config changes)
            prev_params_hash = state.get("params_hash")
            prev_weights_hash = state.get("weights_hash")
            prev_start_iso = state.get("cached_start_iso")
            prev_end_iso = state.get("cached_end_iso")
            needs_manual_refresh = (
                state.get("result") is None
                or not auto_advance_end
                or prev_params_hash != params_hash
                or prev_weights_hash != weights_hash
                or prev_start_iso != start_iso
                or (not auto_advance_end and prev_end_iso != end_iso)
            )

            if needs_manual_refresh:
                # Fetch manually (initial fetch or manual mode or config change)
                try:
                    with st.spinner(
                        f"Fetching Binance candles for {config_store.symbol.upper()} on {config_store.timeframe.upper()}..."
                    ):
                        result_dict = cached_run_automated_signals(
                            config_store.symbol,
                            config_store.timeframe,
                            start_iso,
                            end_iso,
                            params_hash,
                            weights_hash,
                            CACHE_VERSION,
                            json.dumps(signal_config_payload, sort_keys=True),
                            json.dumps(indicator_params, sort_keys=True),
                            json.dumps(signal_params, sort_keys=True),
                        )

                    if not is_valid_signal_structure(result_dict["explicit_signal"]):
                        raise ValueError("Generated signal does not match required schema")

                    final_indicator_params = (
                        result_dict.get("explicit_signal", {})
                        .get("metadata", {})
                        .get("indicator_params")
                        or result_dict.get("processed_payload", {})
                        .get("metadata", {})
                        .get("indicator_params")
                        or indicator_params
                    )

                    state.update(
                        {
                            "result": result_dict,
                            "error": None,
                            "candles": result_dict.get("candles", []),
                            "processed_payload": result_dict.get("processed_payload"),
                            "explicit_signal": result_dict.get("explicit_signal"),
                            "params_hash": params_hash,
                            "weights_hash": weights_hash,
                        }
                    )
                    # Store final indicator params separately
                    state["indicator_params"] = final_indicator_params
                    state["cached_start_iso"] = start_iso
                    state["cached_end_iso"] = end_iso
                    
                    # Initialize auto_end_time_ms from result
                    if auto_advance_end:
                        candles = result_dict.get("candles", [])
                        if candles:
                            last_ts = candles[-1].get("ts")
                            from chart_auto_refresh import TIMEFRAME_TO_MS
                            tf_ms = TIMEFRAME_TO_MS.get(config_store.timeframe, 3_600_000)
                            # Calculate close_time of last candle
                            state["auto_end_time_ms"] = int(last_ts) + tf_ms
                except DataValidationError as exc:
                    state["result"] = None
                    state["error"] = f"Data validation failed: {exc}"
                except Exception as exc:
                    state["result"] = None
                    state["error"] = str(exc)

            # Start worker if auto-advance is enabled and worker is not running
            if auto_advance_end and state.get("result") is not None:
                use_websocket = st.session_state.use_websocket
                manager = st.session_state.signals_worker_manager

                worker = getattr(st.session_state, "automated_signals_worker", None)
                if worker is None or not getattr(st.session_state, "automated_signals_worker_running", False):
                    try:
                        if use_websocket and not manager.is_running():
                            # Try WorkerManager with WebSocket
                            success = manager.start_new(
                                symbol=config_store.symbol,
                                timeframe=config_store.timeframe,
                                update_bus=st.session_state.signals_update_bus,
                                signal_config_payload=signal_config_payload,
                                indicator_params=indicator_params,
                                signal_params=signal_params,
                                use_websocket=True,
                                session_state=st.session_state,
                            )
                            if success:
                                logger.info(f"Started WebSocket signals worker for {config_store.symbol} {config_store.timeframe}")
                        else:
                            # Use REST polling
                            worker = AutomatedSignalsWorker(
                                symbol=config_store.symbol,
                                timeframe=config_store.timeframe,
                                session_state=st.session_state,
                                signal_config_payload=signal_config_payload,
                                indicator_params=indicator_params,
                                signal_params=signal_params,
                            )
                            worker.start()
                            st.session_state.automated_signals_worker = worker
                            logger.info(f"Started REST polling signals worker for {config_store.symbol} {config_store.timeframe}")
                    except Exception as e:
                        logger.error(f"Failed to start automated signals worker: {e}", exc_info=True)
                        state["error"] = f"Failed to start auto-refresh worker: {e}"
                else:
                    # Update existing worker config
                    try:
                        worker.update_config(signal_config_payload, indicator_params, signal_params)
                    except Exception as e:
                        logger.warning(f"Failed to update worker config: {e}")

            # Poll for updates from WorkerManager and apply to session state
            if auto_advance_end and st.session_state.use_websocket:
                manager = st.session_state.signals_worker_manager
                try:
                    updates_applied = manager.poll_and_apply(st.session_state)
                    if updates_applied:
                        state["analysis_updated"] = True
                except Exception as e:
                    logger.error(f"Error polling signals worker manager: {e}", exc_info=True)

            # Consume analysis_updated flag and trigger rerun if needed
            if auto_advance_end and state.get("analysis_updated", False):
                state["analysis_updated"] = False
                logger.info("Analysis updated via worker - triggering UI refresh")
                st.rerun()
            else:
                state["analysis_updated"] = False

            error_message = state.get("error")
            result = state.get("result")

            if error_message:
                st.error(error_message)
            elif result:
                candles = result.get("candles", [])
                if candles:
                    first_ts = candles[0].get("ts")
                    last_ts = candles[-1].get("ts")
                    try:
                        result_start = dt.datetime.fromtimestamp(float(first_ts) / 1000.0, tz=dt.timezone.utc)
                    except (TypeError, ValueError):
                        result_start = config_store.start_datetime
                    try:
                        result_end = dt.datetime.fromtimestamp(float(last_ts) / 1000.0, tz=dt.timezone.utc)
                    except (TypeError, ValueError):
                        result_end = config_store.end_datetime
                else:
                    result_start = config_store.start_datetime
                    result_end = config_store.end_datetime

                st.caption(
                    f"Source: Binance | {config_store.symbol.upper()} {config_store.timeframe.upper()} | {len(candles)} candles "
                    f"from {result_start.strftime('%Y-%m-%d %H:%M UTC')} to {result_end.strftime('%Y-%m-%d %H:%M UTC')}"
                )

                signal_data = result["explicit_signal"]
                processed_signal = result["processed_payload"]

                raw_weight_data = signal_data.get("weights") or state.get("weights")
                normalized_weights_map, raw_weights_map = normalize_category_weights(raw_weight_data)
                has_weight_data = any(raw_weights_map.get(category, 0.0) > 0 for category in FACTOR_CATEGORY_ORDER)
            if not has_weight_data:
                has_weight_data = any(normalized_weights_map.get(category, 0.0) > 0 for category in FACTOR_CATEGORY_ORDER)

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                signal_type = signal_data.get("signal", "HOLD")
                if signal_type == "BUY":
                    st.success("## 🟢 BUY SIGNAL")
                elif signal_type == "SELL":
                    st.error("## 🔴 SELL SIGNAL")
                else:
                    st.info("## ⚪ HOLD")

            with col2:
                confidence = signal_data.get("confidence", 5)
                if confidence >= 8:
                    st.metric("Confidence", f"{confidence}/10", "🟢 High")
                elif confidence >= 5:
                    st.metric("Confidence", f"{confidence}/10", "🟡 Medium")
                else:
                    st.metric("Confidence", f"{confidence}/10", "⚪ Low")

            with col3:
                timeframe_value = str(signal_data.get("timeframe") or config_store.timeframe).upper()
                holding_period = str(signal_data.get("holding_period", "medium")).title()
                st.metric("Timeframe", timeframe_value)
                st.metric("Holding Period", holding_period)

            st.markdown("---")

            actionable = signal_data.get("signal") in {"BUY", "SELL"}

            detail_left, detail_right = st.columns(2)

            with detail_left:
                if actionable:
                    st.markdown("### 📈 Entry & Exit Levels")

                    entries = signal_data.get("entries") or []
                    metadata_summary = signal_data.get("metadata", {})
                    entry_zone: Dict[str, Any] = {}
                    if isinstance(metadata_summary, dict):
                        entry_zone = metadata_summary.get("entry_zone", {}) or {}

                    if entries:
                        st.write("**Entry Levels:**")
                        for i, entry in enumerate(entries[:3], 1):
                            st.write(f"  Entry {i}: ${entry:.4f}")
                    else:
                        st.warning("Entry levels were not provided by the analyzers.")

                    if entry_zone:
                        lower = entry_zone.get("lower")
                        upper = entry_zone.get("upper")
                        if lower is not None and upper is not None:
                            st.write(f"**Entry Confluence Zone:** ${lower:.4f} - ${upper:.4f}")

                    stop_loss = signal_data.get("stop_loss")
                    if stop_loss is not None:
                        st.write(f"**Stop Loss:** ${stop_loss:.4f}")

                    take_profits = signal_data.get("take_profits", {}) or {}
                    if take_profits:
                        st.write("**Take Profits:**")
                        for tp_key, tp_price in take_profits.items():
                            st.write(f"  {tp_key.upper()}: ${tp_price:.4f}")
                else:
                    st.markdown("### 🤔 Hold Rationale")
                    hold_reasons = signal_data.get("cancellation_reasons") or signal_data.get("rationale") or []
                    if hold_reasons:
                        for idx, reason in enumerate(hold_reasons, 1):
                            st.write(f"{idx}. {reason}")
                    else:
                        st.info("Signal is currently on HOLD while the composite score remains neutral.")

            with detail_right:
                if actionable:
                    st.markdown("### 📊 Position & Risk")

                    position_size = signal_data.get("position_size_pct")
                    if position_size is not None:
                        st.write(f"**Position Size:** {position_size:.1f}%")

                    if has_weight_data:
                        st.write("**Component Weights:**")
                        for category in FACTOR_CATEGORY_ORDER:
                            weight_value = normalized_weights_map.get(category, 0.0)
                            if weight_value <= 0:
                                continue
                            st.write(f"  {format_category_label(category)}: {weight_value:.2f}")
                else:
                    st.markdown("### 📊 Component Weights")
                    if has_weight_data:
                        for category in FACTOR_CATEGORY_ORDER:
                            weight_value = normalized_weights_map.get(category, 0.0)
                            if weight_value <= 0:
                                continue
                            st.write(f"• {format_category_label(category)}: {weight_value:.2f}")
                    else:
                        st.write("No component weights available.")

            composite_meta = signal_data.get("metadata", {}) or {}
            composite_score = composite_meta.get("composite_score")
            if composite_score is not None:
                st.markdown("### 🧩 Composite Breakdown")
                st.metric("Composite Score", f"{composite_score:.2f}")
                st.caption(
                    f"Buy ≥ {composite_meta.get('buy_threshold', DEFAULT_SIGNAL_THRESHOLDS['buy']):.2f} · "
                    f"Sell ≤ {composite_meta.get('sell_threshold', DEFAULT_SIGNAL_THRESHOLDS['sell']):.2f}"
                )

                top_contributors = composite_meta.get("top_contributors") or []
                if top_contributors:
                    readable = [
                        f"{item.get('category', '').replace('_', ' ').title()}: {item.get('contribution', 0.0):.3f}"
                        for item in top_contributors
                    ]
                    st.markdown("**Top Drivers:** " + ", ".join(readable))

                weights = composite_meta.get("composite_weights") or {}
                scores = composite_meta.get("category_scores") or {}
                contributions = composite_meta.get("category_contributions") or {}
                st.markdown("**Category Detail**")
                for category in ["technical", "market_structure", "volume", "sentiment", "multitimeframe"]:
                    weight = weights.get(category)
                    score = scores.get(category)
                    contribution = contributions.get(category)
                    parts = []
                    if weight is not None:
                        parts.append(f"weight {weight:.2f}")
                    if score is not None:
                        parts.append(f"score {score:.2f}")
                    if contribution is not None:
                        parts.append(f"contribution {contribution:.3f}")
                    if parts:
                        st.write(f"• {category.replace('_', ' ').title()}: " + ", ".join(parts))

                missing_categories = composite_meta.get("missing_categories") or []
                if missing_categories:
                    st.warning(
                        "Composite weights falling back for missing categories: "
                        + ", ".join(cat.replace("_", " ").title() for cat in missing_categories)
                    )

            mt_meta = (processed_signal.get("multi_timeframe") or {}).get("metadata", {}) if processed_signal else {}
            mt_note = mt_meta.get("note") or mt_meta.get("notes")
            if mt_note:
                st.info(f"Multi-timeframe analysis: {mt_note}")
            mt_missing = mt_meta.get("missing_timeframes") or []
            if mt_missing:
                st.caption(
                    "Missing timeframe data: "
                    + ", ".join(tf.upper() for tf in mt_missing)
                )

            st.markdown("---")

            rationale = signal_data.get("rationale", [])
            if rationale:
                st.markdown("### 💡 Signal Rationale")
                for i, point in enumerate(rationale, 1):
                    st.write(f"{i}. {point}")

            cancel_conditions = signal_data.get("cancel_conditions", [])
            if cancel_conditions:
                st.markdown("### ⚠️ Cancel Conditions")
                for condition in cancel_conditions:
                    st.write(f"• {condition}")

            with st.expander("🔧 Processing Information", expanded=False):
                processing_info = processed_signal.get("metadata", {})
                st.write(f"**Processor:** {processing_info.get('payload_processor', 'Unknown')}")
                st.write(f"**Timeframe Used:** {processing_info.get('timeframe_used', 'Unknown')}")
                st.write(f"**Real Data Validated:** {processing_info.get('real_data_validated', False)}")
                st.write(f"**Source Data Quality:** {processing_info.get('source_data_quality', 'Unknown')}")
                st.write("**Signal Format:** Explicit JSON Schema v1.0")

                debug_payload = signal_data.get("debug")
                if debug_payload:
                    st.write("**Debug Details:**")
                    st.json(debug_payload, expanded=False)

            analyzer_indicator_params = signal_data.get("metadata", {}).get("indicator_params") or processed_signal.get("metadata", {}).get("indicator_params")
            with st.expander("🛠 Analyzer Inputs", expanded=False):
                st.write("**Category Weights**")
                if raw_weight_data or has_weight_data:
                    weight_rows = []
                    for category in FACTOR_CATEGORY_ORDER:
                        normalized_value = normalized_weights_map.get(category, 0.0)
                        raw_value = raw_weights_map.get(category, 0.0)
                        row = {
                            "Category": format_category_label(category),
                            "Normalized Weight": f"{normalized_value:.2f}",
                        }
                        if raw_weight_data:
                            row["Raw Weight"] = f"{raw_value:.2f}"
                        weight_rows.append(row)
                    weight_df = pd.DataFrame(weight_rows)
                    st.dataframe(
                        weight_df,
                        width="stretch",
                        hide_index=True,
                        key=ui_key(
                            "automated_signals",
                            f"category_weights_{config_store.symbol}_{config_store.timeframe}"
                        ),
                    )
                else:
                    st.write("No weight data available.")

                if analyzer_indicator_params:
                    st.write("**Indicator Parameters**")
                    st.json(analyzer_indicator_params)

                if result:
                    st.write("**Parameter Hash:**", result.get("params_hash"))
                    st.write("**Weights Hash:**", result.get("weights_hash"))
                else:
                    st.write("**Parameter Hash:**", state.get("params_hash"))
                    st.write("**Weights Hash:**", state.get("weights_hash"))

            # Factor Analysis
            factors = signal_data.get("factors", []) or []
            factor_entries: Dict[str, Dict[str, Any]] = {}
            for factor in factors:
                if not isinstance(factor, dict):
                    continue
                category = normalize_factor_category(factor.get("factor_name"))
                if not category or category not in FACTOR_CATEGORY_ORDER:
                    continue
                score = safe_float(factor.get("score"))
                metadata = factor.get("metadata") or {}
                direction = metadata.get("direction")
                confidence = safe_float(metadata.get("confidence"))
                entry = factor_entries.get(category)
                if entry is None or (score is not None and entry.get("score") is not None and abs(score - 0.5) > abs(entry["score"] - 0.5)):
                    factor_entries[category] = {
                        "score": score,
                        "emoji": factor.get("emoji", "⚪"),
                        "description": factor.get("description", ""),
                        "direction": direction,
                        "confidence": confidence,
                    }

            factors_rows = []
            for category in FACTOR_CATEGORY_ORDER:
                entry = factor_entries.get(category)
                weight_value = normalized_weights_map.get(category, 0.0)
                if entry and entry.get("score") is not None:
                    row = {
                        "Category": format_category_label(category),
                        "Score": f"{entry['score']:.2f}",
                        "Weight": f"{weight_value:.2f}",
                        "Direction": entry.get("direction", "").title() if entry.get("direction") else "—",
                        "Emoji": entry.get("emoji", "⚪") or "⚪",
                        "Description": entry.get("description", ""),
                        "Confidence": f"{entry['confidence']:.1f}" if entry.get("confidence") is not None else "—",
                    }
                    factors_rows.append(row)
                elif has_weight_data and weight_value > 0:
                    factors_rows.append(
                        {
                            "Category": format_category_label(category),
                            "Score": "N/A",
                            "Weight": f"{weight_value:.2f}",
                            "Direction": "Neutral",
                            "Emoji": "⚪",
                            "Description": "No factor data available",
                            "Confidence": "—",
                        }
                    )

            if factors_rows:
                st.markdown("### 📊 Factor Analysis")
                factors_df = pd.DataFrame(factors_rows)
                st.dataframe(
                    factors_df,
                    width="stretch",
                    hide_index=True,
                    key=ui_key(
                        "automated_signals",
                        f"factors_{config_store.symbol}_{config_store.timeframe}"
                    ),
                )

            st.markdown("---")

            # Position Plan
            position_plan = signal_data.get("position_plan", {})
            if position_plan:
                st.markdown("### 💼 Position Plan")

                plan_col1, plan_col2, plan_col3, plan_col4 = st.columns(4)

                entry_price = position_plan.get("entry_price", 0.0)
                with plan_col1:
                    st.metric("Entry Price", f"${entry_price:.4f}")

                position_size_usd = position_plan.get("position_size_usd", 0.0)
                with plan_col2:
                    st.metric(
                        "Position Size",
                        f"${position_size_usd:.2f}" if position_size_usd else "N/A",
                    )

                direction = position_plan.get("direction", "N/A")
                with plan_col3:
                    st.metric("Direction", direction.upper() if direction else "N/A")

                leverage = position_plan.get("leverage", 1.0)
                with plan_col4:
                    st.metric("Leverage", f"{leverage:.1f}x" if leverage else "N/A")

                st.markdown("#### TP/SL Ladder")

                ladder_col1, ladder_col2 = st.columns(2)

                with ladder_col1:
                    stop_loss = position_plan.get("stop_loss", 0.0)
                    st.write(f"**Stop Loss:** ${stop_loss:.4f}" if stop_loss else "**Stop Loss:** N/A")

                    if entry_price and stop_loss:
                        risk_distance = abs(entry_price - stop_loss)
                        risk_pct = (risk_distance / entry_price) * 100
                        st.write(f"Risk Distance: ${risk_distance:.4f} ({risk_pct:.2f}%)")

                with ladder_col2:
                    take_profit_levels = position_plan.get("take_profit_levels", [])
                    if take_profit_levels:
                        for idx, tp_level in enumerate(take_profit_levels, 1):
                            if entry_price and tp_level:
                                profit_pct = ((tp_level - entry_price) / entry_price) * 100
                                st.write(f"TP{idx}: ${tp_level:.4f} ({profit_pct:+.2f}%)")
                    else:
                        st.write("No TP levels defined")

                if position_plan.get("risk_reward_ratio"):
                    rrr = position_plan.get("risk_reward_ratio")
                    st.markdown(f"**Risk/Reward Ratio:** {rrr:.2f}:1")

                if position_plan.get("max_risk_pct"):
                    max_risk = position_plan.get("max_risk_pct")
                    st.markdown(f"**Max Risk %:** {max_risk * 100:.2f}%")

            st.markdown("---")

            if signal_data.get("holding_horizon_bars"):
                holding_horizon = signal_data.get("holding_horizon_bars")
                st.markdown("### ⏱️ Holding Horizon")
                st.info(f"**Estimated Holding Period:** {holding_horizon} bars")

            explanation = signal_data.get("explanation", {})
            if explanation:
                st.markdown("### 📝 Signal Rationale")

                primary_reason = explanation.get("primary_reason", "")
                if primary_reason:
                    st.markdown(f"**Primary Reason:** {primary_reason}")

                supporting_factors = explanation.get("supporting_factors", [])
                if supporting_factors:
                    st.markdown("**Supporting Factors:**")
                    for factor in supporting_factors:
                        st.write(f"• {factor}")

                risk_factors = explanation.get("risk_factors", [])
                if risk_factors:
                    st.markdown("**Risk Factors:**")
                    for risk in risk_factors:
                        st.write(f"⚠️ {risk}")

                market_context = explanation.get("market_context", "")
                if market_context:
                    st.markdown(f"**Market Context:** {market_context}")

            st.markdown("---")

            cancellation_reasons = signal_data.get("cancellation_reasons", []) or []
            plan_warnings = (signal_data.get("metadata", {}) or {}).get("plan_warnings", []) or []
            if signal_data.get("signal") == "HOLD" and cancellation_reasons:
                st.warning("### ⛔ Blocking Reasons")
                for reason in cancellation_reasons:
                    st.write(f"• {reason}")
            else:
                advisory_notes = list(dict.fromkeys(plan_warnings + cancellation_reasons))
                if advisory_notes:
                    st.info("### ℹ️ Signal Notes")
                    for note in advisory_notes:
                        st.write(f"• {note}")

            st.markdown("---")

            optimization_stats = signal_data.get("optimization_stats", {})
            if optimization_stats:
                st.markdown("### 📈 Performance Metrics")

                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                with perf_col1:
                    win_rate = optimization_stats.get("backtest_win_rate")
                    if win_rate is not None:
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                with perf_col2:
                    profit_factor = optimization_stats.get("profit_factor")
                    if profit_factor is not None:
                        st.metric("Profit Factor", f"{profit_factor:.2f}")

                with perf_col3:
                    sharpe_ratio = optimization_stats.get("sharpe_ratio")
                    if sharpe_ratio is not None:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                with perf_col4:
                    total_signals = optimization_stats.get("total_signals", 0)
                    st.metric("Total Signals", total_signals)

                perf_extra_col1, perf_extra_col2, perf_extra_col3 = st.columns(3)

                with perf_extra_col1:
                    avg_profit = optimization_stats.get("avg_profit_pct")
                    if avg_profit is not None:
                        st.metric("Avg Profit %", f"{avg_profit:.2f}%")

                with perf_extra_col2:
                    avg_loss = optimization_stats.get("avg_loss_pct")
                    if avg_loss is not None:
                        st.metric("Avg Loss %", f"{avg_loss:.2f}%")

                with perf_extra_col3:
                    profitable = optimization_stats.get("profitable_signals", 0)
                    losing = optimization_stats.get("losing_signals", 0)
                    st.metric("Profitable Signals", f"{profitable}/{profitable + losing}")

                st.markdown("---")
                st.info(
                    "💡 **Note:** This automated signals tab generates trading system analysis using real Binance data. "
                    "Ensure your trading system exports signals in the expected JSON format for full functionality."
                )
            else:
                st.info("Configure symbol, timeframe, and date range to run automated signals with real Binance data.")
    
    with backtest_tab:
        st.subheader("🔬 Backtesting")
        state = st.session_state.setdefault(AUTOMATED_SIGNALS_STATE_KEY, {})

        st.caption(
            "Backtest uses the same symbol, timeframe, and configuration as the automated signal analysis."
        )

        backtest_settings = config_store.backtest_settings()
        settings_cols = st.columns(3)
        max_bars = num_int(
            "Max Bars",
            min_v=120,
            max_v=5000,
            value=backtest_settings.get("max_bars", 320),
            step=20,
            key=ui_key("backtest", "max_bars"),
            ui=settings_cols[0],
        )
        signal_step = num_int(
            "Signal Step",
            min_v=1,
            max_v=24,
            value=backtest_settings.get("step", 1),
            key=ui_key("backtest", "step"),
            ui=settings_cols[1],
        )
        holding_bars = num_int(
            "Holding Horizon (bars)",
            min_v=5,
            max_v=240,
            value=backtest_settings.get("holding_bars", 24),
            key=ui_key("backtest", "holding_bars"),
            ui=settings_cols[2],
        )
        config_store.update_backtest_setting("max_bars", int(max_bars))
        config_store.update_backtest_setting("step", int(signal_step))
        config_store.update_backtest_setting("holding_bars", int(holding_bars))

        misc_cols = st.columns(2)
        min_trades = num_int(
            "Minimum Trades",
            min_v=1,
            max_v=200,
            value=backtest_settings.get("min_trades", 5),
            key=ui_key("backtest", "min_trades"),
            ui=misc_cols[0],
        )
        config_store.update_backtest_setting("min_trades", int(min_trades))

        candles = state.get("candles")
        if not candles:
            st.info("Run the automated signals analysis to fetch Binance candles before backtesting.")
        else:
            try:
                signal_config = config_store.build_signal_config()
                indicator_params = config_store.get_indicator_params()
                signal_params = config_store.build_signal_params()
                backtest_result = simulate_backtest(
                    candles,
                    config_store.symbol,
                    config_store.timeframe,
                    signal_config,
                    indicator_params,
                    signal_params,
                    max_bars=int(max_bars),
                    step=int(signal_step),
                    holding_bars=int(holding_bars),
                    min_required_bars=max(int(holding_bars) * 3, 120),
                )
            except ValueError as exc:
                st.warning(str(exc))
            else:
                trade_count = len(backtest_result.trades)
                if trade_count < int(min_trades):
                    st.warning(
                        f"Backtest generated {trade_count} trades, fewer than the minimum of {int(min_trades)}. "
                        "Consider expanding the date range or adjusting holding parameters."
                    )

                metric_cols = st.columns(5)
                metric_cols[0].metric("Win Rate", f"{backtest_result.win_rate:.1f}%")
                metric_cols[1].metric("Total Return", f"{backtest_result.total_return_pct:.1f}%")
                metric_cols[2].metric("Profit Factor", f"{backtest_result.profit_factor:.2f}")
                metric_cols[3].metric("Max Drawdown", f"{backtest_result.max_drawdown_pct:.1f}%")
                metric_cols[4].metric("Sharpe Ratio", f"{backtest_result.sharpe_ratio:.2f}")
                st.metric("Average R Multiple", f"{backtest_result.average_r_multiple:.2f}")

                equity_df = pd.DataFrame(backtest_result.equity_curve, columns=["timestamp", "equity"])
                if not equity_df.empty:
                    equity_df["time"] = pd.to_datetime(equity_df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC")
                    equity_fig = go.Figure(
                        go.Scatter(x=equity_df["time"], y=equity_df["equity"], mode="lines", name="Equity")
                    )
                    equity_fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Time",
                        yaxis_title="Equity (USD)",
                        height=400,
                        template="plotly_dark",
                    )
                    st.plotly_chart(
                        equity_fig,
                        width="stretch",
                        key=ui_key(
                            "automated_signals",
                            f"equity_curve_{config_store.symbol}_{config_store.timeframe}"
                        ),
                    )

                if trade_count:
                    trades_df = pd.DataFrame(
                        [
                            {
                                "Direction": trade.direction,
                                "Entry": trade.entry_price,
                                "Exit": trade.exit_price,
                                "Return %": trade.return_pct,
                                "R Multiple": trade.r_multiple,
                                "Outcome": trade.outcome,
                                "Entry Time": dt.datetime.fromtimestamp(trade.entry_timestamp / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                                "Exit Time": dt.datetime.fromtimestamp(trade.exit_timestamp / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            }
                            for trade in backtest_result.trades
                        ]
                    )
                    st.markdown("### Trades Summary")
                    st.dataframe(
                        trades_df,
                        width="stretch",
                        key=ui_key(
                            "automated_signals",
                            f"backtest_trades_{config_store.symbol}_{config_store.timeframe}"
                        ),
                    )
                else:
                    st.info("No trades met the criteria within the selected horizon.")

    with adaptive_tab:
        st.subheader("⚖️ Adaptive Weight Management")

        try:
            from indicator_collector.trading_system import (
                AdaptiveWeightConfig,
                AdaptiveWeightManager,
                Backtester,
                ParameterSet,
            )
            
            st.markdown("""
            ### 🧠 Adaptive Weight System
            
            Automatically adjust signal weights based on rolling performance metrics
            to optimize trading system performance over time.
            """)
            
            # Adaptive weight configuration
            with st.expander("⚙️ Adaptive Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    rolling_window = num_int(
                        "Rolling Window (days)",
                        min_v=7,
                        max_v=90,
                        value=30,
                        help_text="Number of days to consider for performance tracking",
                        key=ui_key("adaptive_tab", "rolling_window"),
                    )
                    min_signals = num_int(
                        "Min Signals for Adaptation",
                        min_v=10,
                        max_v=200,
                        value=50,
                        help_text="Minimum signals required before adapting weights",
                        key=ui_key("adaptive_tab", "min_signals"),
                    )

                    adaptation_strategy = st.selectbox(
                        "Adaptation Strategy",
                        ["performance_based", "volatility_adjusted", "hybrid"],
                        index=2,
                        help="Method for calculating weight adjustments",
                        key=ui_key("adaptive_tab", "adaptation_strategy"),
                    )
                
                with col2:
                    target_win_rate = st.slider(
                        "Target Win Rate",
                        min_value=0.3,
                        max_value=0.9,
                        value=0.55,
                        help="Target win rate for adaptation triggers",
                        key=ui_key("adaptive_tab", "target_win_rate")
                    )
                    target_profit_factor = st.slider(
                        "Target Profit Factor",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        help="Target profit factor for adaptation triggers",
                        key=ui_key("adaptive_tab", "target_profit_factor")
                    )
                    adaptation_threshold = st.slider(
                        "Adaptation Threshold",
                        min_value=0.01,
                        max_value=0.2,
                        value=0.05,
                        help="Minimum performance improvement to trigger adaptation",
                        key=ui_key("adaptive_tab", "adaptation_threshold")
                    )
            
            # Current weights display
            with st.expander("📊 Current Weights", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Initial Weights**")
                    tech_weight = st.slider("Technical", 0.0, 1.0, 0.4, 0.05, key=ui_key("adaptive_tab", "tech_weight"))
                    vol_weight = st.slider("Volume", 0.0, 1.0, 0.3, 0.05, key=ui_key("adaptive_tab", "vol_weight"))
                    sent_weight = st.slider("Sentiment", 0.0, 1.0, 0.2, 0.05, key=ui_key("adaptive_tab", "sent_weight"))
                    struct_weight = st.slider("Market Structure", 0.0, 1.0, 0.1, 0.05, key=ui_key("adaptive_tab", "struct_weight"))
                
                with col2:
                    st.markdown("**Weight Constraints**")
                    min_weight = st.slider("Min Weight per Factor", 0.01, 0.2, 0.05, 0.01, key=ui_key("adaptive_tab", "min_weight"))
                    max_weight = st.slider("Max Weight per Factor", 0.2, 0.8, 0.5, 0.05, key=ui_key("adaptive_tab", "max_weight"))
                    max_change = st.slider("Max Change pct", 0.1, 0.5, 0.3, 0.05, key=ui_key("adaptive_tab", "max_change"))
            
            # Run adaptation
            run_adaptation = st.button("🔄 Run Adaptive Analysis", type="primary", width="stretch")
            
            if run_adaptation:
                with st.spinner("Running adaptive weight analysis..."):
                    try:
                        # Create adaptive weight manager
                        config = AdaptiveWeightConfig(
                            rolling_window_days=int(rolling_window),
                            min_signals_for_adaptation=int(min_signals),
                            adaptation_strategy=adaptation_strategy,
                            target_win_rate=target_win_rate,
                            target_profit_factor=target_profit_factor,
                            adaptation_threshold=adaptation_threshold,
                            min_weight_per_factor=min_weight,
                            max_weight_per_factor=max_weight,
                            max_weight_change_pct=max_change,
                        )
                        
                        manager = AdaptiveWeightManager(config)
                        
                        # Initialize weights
                        total_weight = tech_weight + vol_weight + sent_weight + struct_weight
                        if total_weight > 0:
                            initial_weights = {
                                "technical": tech_weight / total_weight,
                                "volume": vol_weight / total_weight,
                                "sentiment": sent_weight / total_weight,
                                "market_structure": struct_weight / total_weight,
                            }
                        else:
                            initial_weights = {"technical": 0.25, "volume": 0.25, "sentiment": 0.25, "market_structure": 0.25}
                        
                        manager.initialize_weights(initial_weights)
                        
                        # Create sample signal outcomes (in real implementation, this would be historical data)
                        import random
                        
                        outcomes = []
                        base_timestamp = int((dt.datetime.now() - dt.timedelta(days=rolling_window)).timestamp() * 1000)
                        
                        for i in range(int(min_signals * 2)):  # Generate more than minimum
                            success = random.random() > 0.4  # 60% win rate
                            pnl = random.uniform(1.0, 5.0) if success else random.uniform(-3.0, -0.5)
                            
                            outcome = {
                                "signal_type": random.choice(["BUY", "SELL"]),
                                "entry_price": 50000 + random.uniform(-5000, 5000),
                                "exit_price": None,
                                "entry_timestamp": base_timestamp + i * 86400000,
                                "exit_timestamp": base_timestamp + i * 86400000 + 86400000,
                                "pnl_pct": pnl,
                                "holding_bars": random.randint(1, 100),
                                "success": success,
                                "factors": [
                                    {"factor_name": "technical", "score": random.uniform(0.3, 0.9)},
                                    {"factor_name": "volume", "score": random.uniform(0.3, 0.9)},
                                    {"factor_name": "sentiment", "score": random.uniform(0.3, 0.9)},
                                ],
                            }
                            outcomes.append(outcome)
                        
                        # Update manager with outcomes
                        from indicator_collector.trading_system.statistics_optimizer import SignalOutcome
                        signal_outcomes = []
                        
                        for outcome_data in outcomes:
                            outcome = SignalOutcome(
                                signal_type=outcome_data["signal_type"],
                                entry_price=outcome_data["entry_price"],
                                exit_price=outcome_data["exit_price"],
                                entry_timestamp=outcome_data["entry_timestamp"],
                                exit_timestamp=outcome_data["exit_timestamp"],
                                pnl_pct=outcome_data["pnl_pct"],
                                holding_bars=outcome_data["holding_bars"],
                                success=outcome_data["success"],
                                factors=outcome_data["factors"],
                            )
                            signal_outcomes.append(outcome)
                        
                        manager.update_signal_outcomes(signal_outcomes)
                        
                        # Check if adaptation should be performed
                        should_adapt, reason = manager.should_adapt()
                        
                        st.info(f"📊 Adaptation Analysis: {reason}")
                        
                        if should_adapt:
                            # Perform adaptation
                            adaptation_report = manager.adapt_weights()
                            
                            st.success("✅ Weight adaptation completed!")
                            
                            # Display adaptation results
                            st.markdown("### 🔄 Adaptation Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Adaptation Summary**")
                                st.write(f"• Reason: {adaptation_report.adaptation_reason}")
                                st.write(f"• Confidence: {adaptation_report.confidence_score:.3f}")
                                st.write(f"• Expected Improvement: {adaptation_report.expected_improvement:.4f}")
                                st.write(f"• Factors Adjusted: {', '.join(adaptation_report.factors_adjusted)}")
                            
                            with col2:
                                st.markdown("**Performance Before**")
                                before = adaptation_report.performance_before
                                st.write(f"• Win Rate: {before.win_rate:.3f}")
                                st.write(f"• Profit Factor: {before.profit_factor:.3f}")
                                st.write(f"• Sharpe Ratio: {before.sharpe_ratio:.3f}")
                                st.write(f"• Max Drawdown: {before.max_drawdown_pct:.3f}")
                            
                            # Weight changes
                            st.markdown("### 📊 Weight Changes")
                            
                            weight_changes = []
                            for factor in adaptation_report.factors_adjusted:
                                old_weight = adaptation_report.original_weights.get(factor, 0)
                                new_weight = adaptation_report.new_weights.get(factor, 0)
                                change = new_weight - old_weight
                                weight_changes.append({
                                    "Factor": factor.replace("_", " ").title(),
                                    "Before": f"{old_weight:.3f}",
                                    "After": f"{new_weight:.3f}",
                                    "Change": f"{change:+.3f}",
                                    "Change %": f"{change/old_weight*100:+.1f}%" if old_weight > 0 else "N/A"
                                })
                            
                            if weight_changes:
                                weight_df = pd.DataFrame(weight_changes)
                                st.dataframe(weight_df, width="stretch", hide_index=True)
                            
                            # Performance report
                            st.markdown("### 📈 Performance Report")
                            report = manager.generate_performance_report()
                            
                            # Summary metrics
                            summary = report["summary"]
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Signals", summary["total_signals_analyzed"])
                            with col2:
                                st.metric("Total Adaptations", summary["total_adaptations"])
                            with col3:
                                current_wr = summary["recent_kpis"]["win_rate"]
                                st.metric("Current Win Rate", f"{current_wr:.3f}")
                            with col4:
                                current_pf = summary["recent_kpis"]["profit_factor"]
                                st.metric("Current Profit Factor", f"{current_pf:.3f}")
                            
                            # Performance vs targets
                            st.markdown("**Performance vs Targets**")
                            perf_vs_targets = report["performance_vs_targets"]
                            
                            for metric, data in perf_vs_targets.items():
                                current = data["current"]
                                target = data["target"]
                                gap = data["gap"]
                                status = "✅" if gap >= 0 else "❌"
                                
                                st.write(f"{status} **{metric.replace('_', ' ').title()}:** {current:.3f} (target: {target:.3f})")
                            
                            # Recommendations
                            st.markdown("### 💡 Recommendations")
                            recommendations = report["recommendations"]
                            
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")
                            
                        else:
                            st.info("ℹ️ No adaptation needed at this time.")
                            
                            # Display current performance
                            st.markdown("### 📊 Current Performance")
                            current_kpis = manager._calculate_recent_kpis()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Win Rate", f"{current_kpis.win_rate:.3f}")
                            with col2:
                                st.metric("Profit Factor", f"{current_kpis.profit_factor:.3f}")
                            with col3:
                                st.metric("Sharpe Ratio", f"{current_kpis.sharpe_ratio:.3f}")
                            with col4:
                                st.metric("Max Drawdown", f"{current_kpis.max_drawdown_pct:.3f}")
                        
                        # Display weight performance
                        st.markdown("### 📊 Factor Performance")
                        weight_performance = manager.get_weight_performance()
                        
                        perf_data = []
                        for factor_name, perf in weight_performance.items():
                            perf_data.append({
                                "Factor": factor_name.replace("_", " ").title(),
                                "Current Weight": f"{perf.current_weight:.3f}",
                                "Win Rate": f"{perf.rolling_win_rate:.3f}",
                                "Profit Factor": f"{perf.rolling_profit_factor:.3f}",
                                "Sharpe": f"{perf.rolling_sharpe:.3f}",
                                "Consistency": f"{perf.consistency_score:.3f}",
                                "Adaptations": perf.adaptation_count,
                            })
                        
                        if perf_data:
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(perf_df, width="stretch", hide_index=True)
                        
                    except Exception as e:
                        st.error(f"❌ Adaptive analysis failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except ImportError as e:
            st.error(f"❌ Adaptive weight components not available: {str(e)}")
            st.info("Please ensure the adaptive weight management modules are properly installed.")
    
    with astrology_tab:
        st.subheader("🔮 Astrology & Celestial Cycles")
        astrology = payload.get("astrology", {})
        
        if not astrology:
            st.info("Astrology metrics are not available for this analysis.")
        else:
            confluence = astrology.get("confluence", {})
            moon = astrology.get("moon", {})
            mercury = astrology.get("mercury", {})
            jupiter = astrology.get("jupiter", {})
            
            st.markdown("### Overall Celestial Confluence")
            
            conf_col1, conf_col2 = st.columns([1, 2])
            
            with conf_col1:
                st.metric(
                    "Confluence Score",
                    f"{confluence.get('score', 0):.2f}",
                    delta=f"{confluence.get('signal', 'neutral').upper()}"
                )
                st.markdown(f"### {confluence.get('signal_color', '⚪')}")
            
            with conf_col2:
                recommendation = confluence.get("recommendation", "No specific recommendation available.")
                st.info(f"**Trading Recommendation:** {recommendation}")
            
            if confluence.get("factors"):
                st.markdown("#### Active Celestial Factors")
                for factor in confluence["factors"]:
                    st.write(f"• {factor}")
            
            st.markdown("---")
            
            moon_col, mercury_col = st.columns(2)
            
            with moon_col:
                st.markdown("### 🌕 Moon Cycle Analysis")
                st.metric("Current Phase", moon.get("phase_name", "Unknown"))
                st.metric("Illumination", f"{moon.get('illumination_pct', 0):.1f}%")
                
                volatility_ind = moon.get("volatility_indication", "moderate")
                if volatility_ind == "high":
                    vol_color = "🔴"
                    vol_text = "HIGH (expect increased volatility)"
                elif volatility_ind == "moderate":
                    vol_color = "🟡"
                    vol_text = "MODERATE (normal volatility expected)"
                else:
                    vol_color = "🟢"
                    vol_text = "LOW (reduced volatility expected)"
                
                st.markdown(f"**Volatility Indication:** {vol_color} {vol_text}")
                st.markdown(f"**Trading Bias:** {moon.get('trading_bias', 'neutral').title()}")
                
                st.markdown("---")
                st.markdown("**Upcoming Moon Events:**")
                st.write(f"• Full Moon in **{moon.get('days_to_full_moon', 0):.1f}** days ({moon.get('next_full_moon', 'N/A')[:10]})")
                st.write(f"• New Moon in **{moon.get('days_to_new_moon', 0):.1f}** days ({moon.get('next_new_moon', 'N/A')[:10]})")
                
                st.markdown("---")
                with st.expander("ℹ️ Moon Cycle Trading Context"):
                    st.markdown("""
                    **Full Moon & New Moon periods** often coincide with volatility peaks in crypto markets.
                    - **Full Moon**: Peak emotions, potential tops
                    - **New Moon**: Fresh starts, potential bottoms
                    - **Waxing Moon**: Growing phase, accumulation
                    - **Waning Moon**: Declining phase, distribution
                    """)
            
            with mercury_col:
                st.markdown("### ☿ Mercury Cycle (Trading Planet)")
                st.metric("Current Phase", mercury.get("phase_name", "Unknown"))
                st.metric("Cycle Position", f"{mercury.get('cycle_position_pct', 0):.1f}%")
                
                volume_ind = mercury.get("volume_indication", "moderate")
                if volume_ind == "high":
                    vol_color = "🟢"
                    vol_text = "HIGH (peak trading activity)"
                elif volume_ind == "increasing":
                    vol_color = "🟡"
                    vol_text = "INCREASING (building momentum)"
                elif volume_ind == "decreasing":
                    vol_color = "🟠"
                    vol_text = "DECREASING (slowing activity)"
                else:
                    vol_color = "🔴"
                    vol_text = "LOW (reduced trading)"
                
                st.markdown(f"**Volume Indication:** {vol_color} {vol_text}")
                st.markdown(f"**Recommendation:** {mercury.get('trading_recommendation', 'No specific recommendation')}")
                
                st.markdown("---")
                st.markdown("**Next Mercury Peak:**")
                st.write(f"• In **{mercury.get('days_to_peak_activity', 0):.1f}** days")
                st.write(f"• Date: {mercury.get('next_peak_date', 'N/A')[:10]}")
                
                st.markdown("---")
                with st.expander("ℹ️ Mercury Cycle Trading Context"):
                    st.markdown("""
                    **Mercury's 88-day cycle** correlates with trading volume patterns:
                    - **Direct Motion Peak**: Highest trading activity, good liquidity
                    - **Retrograde**: Lower activity, consolidation periods
                    - **Post-Retrograde**: Recovery, new opportunities emerging
                    """)
            
            st.markdown("---")
            
            st.markdown("### ♃ Jupiter 12-Year Cycle & Bitcoin Halvings")
            
            jup_col1, jup_col2, jup_col3 = st.columns(3)
            
            with jup_col1:
                st.markdown("#### Jupiter Cycle")
                st.metric("Phase", jupiter.get("jupiter_phase", "Unknown"))
                st.metric("Position", f"{jupiter.get('jupiter_cycle_position_pct', 0):.1f}%")
                
                correlation = jupiter.get("market_correlation", "neutral")
                if "strongly bullish" in correlation:
                    corr_emoji = "🟢🟢🟢"
                elif "bullish" in correlation:
                    corr_emoji = "🟢🟢"
                elif "bearish" in correlation:
                    corr_emoji = "🔴"
                else:
                    corr_emoji = "⚪"
                
                st.markdown(f"**Market Correlation:** {corr_emoji} {correlation.title()}")
            
            with jup_col2:
                st.markdown("#### Bitcoin Halving Cycle")
                st.metric("Current Epoch", f"#{jupiter.get('current_halving_epoch', 0)}")
                st.metric("Halving Phase", jupiter.get("halving_phase", "Unknown"))
                st.metric("Phase Progress", f"{jupiter.get('halving_cycle_position_pct', 0):.1f}%")
            
            with jup_col3:
                st.markdown("#### Timeline")
                st.metric("Days Since Halving", f"{jupiter.get('days_since_last_halving', 0):,}")
                st.metric("Days to Next", f"{jupiter.get('days_to_next_halving', 0):,}")
                st.write(f"**Next Halving:** {jupiter.get('next_halving_date', 'N/A')[:10]}")
            
            st.markdown("---")
            st.markdown(f"**Jupiter Recommendation:** {jupiter.get('recommendation', 'No specific recommendation')}")
            
            st.markdown("---")
            with st.expander("ℹ️ Jupiter & Bitcoin Halving Correlation"):
                st.markdown("""
                **Jupiter's 12-year cycle** aligns remarkably with Bitcoin's 4-year halving cycles:
                
                - **Jupiter Expansion (Year 1-6)**: Coincides with post-halving bull markets
                - **Jupiter Peak**: Often aligns with cycle tops
                - **Jupiter Contraction (Year 7-12)**: Coincides with bear markets and accumulation
                
                **Bitcoin Halving Phases:**
                - **Post-Halving Accumulation (0-12 months)**: Build positions
                - **Bull Market Phase (12-24 months)**: Major uptrend
                - **Euphoria & Distribution (24-36 months)**: Take profits
                - **Pre-Halving Bear (36-48 months)**: Accumulation opportunity
                
                This pattern has repeated across 3+ cycles, making it a useful contextual indicator.
                """)
            
            st.markdown("---")
            st.warning("⚠️ **Disclaimer:** Astrology-based analysis is provided for contextual reference only and should not be the sole basis for trading decisions. Always combine with technical analysis, fundamental research, and proper risk management.")
        
    with export_tab:
        st.subheader("💾 Export Analysis Data")
        
        st.markdown("### Current Session")
        metadata = payload.get("metadata", {})
        
        export_info_col1, export_info_col2 = st.columns(2)
        with export_info_col1:
            st.write(f"**Symbol:** {metadata.get('symbol', 'N/A')}")
            st.write(f"**Timeframe:** {metadata.get('timeframe', 'N/A')}")
            st.write(f"**Period:** {metadata.get('period', 'N/A')} bars")
        
        with export_info_col2:
            st.write(f"**Export Token:** {metadata.get('token', 'N/A')}")
            st.write(f"**Generated:** {metadata.get('generated_at', 'N/A')[:19]}")
        
        st.markdown("---")
        
        st.markdown("### Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_str = json.dumps(payload, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{selected_token.replace(':', '_')}_{selected_timeframe}_{export_token}.json",
                mime="application/json",
                width="stretch",
            )
        
        with col2:
            latest = payload.get("latest", {})
            csv_data = f"""Symbol,Timeframe,Period,Close,Trend_Strength,Pattern_Score,Sentiment,Structure,Confluence_Score,RSI,MACD
{metadata.get('symbol')},{metadata.get('timeframe')},{metadata.get('period')},{latest.get('close')},{latest.get('trend_strength')},{latest.get('pattern_score')},{latest.get('market_sentiment')},{latest.get('structure_state')},{latest.get('confluence_score')},{latest.get('rsi')},{latest.get('macd')}
"""
            st.download_button(
                label="📥 Download CSV (Latest)",
                data=csv_data,
                file_name=f"{selected_token.replace(':', '_')}_{selected_timeframe}_latest.csv",
                mime="text/csv",
                width="stretch",
            )
        
        with col3:
            advanced = payload.get("advanced", {})
            advanced_rows = []
            if advanced:
                volume_analysis = advanced.get("volume_analysis", {})
                vpvr = volume_analysis.get("vpvr", {})
                advanced_rows.append(("VPVR POC", vpvr.get("poc")))
                advanced_rows.append(("Value Area High", vpvr.get("value_area", {}).get("high")))
                advanced_rows.append(("Value Area Low", vpvr.get("value_area", {}).get("low")))
                advanced_rows.append(("CVD Latest", volume_analysis.get("cvd", {}).get("latest")))
                advanced_rows.append(("CVD Change", volume_analysis.get("cvd", {}).get("change")))
                advanced_rows.append(("Delta Latest", volume_analysis.get("delta", {}).get("latest")))
                advanced_rows.append(("Delta Average", volume_analysis.get("delta", {}).get("average")))
                market_structure = advanced.get("market_structure", {})
                advanced_rows.append(("Structure Trend", market_structure.get("trend")))
                fundamentals = advanced.get("fundamentals", {})
                advanced_rows.append(("Funding Rate", fundamentals.get("funding_rate", {}).get("current")))
                advanced_rows.append(("Open Interest", fundamentals.get("open_interest", {}).get("current")))
                advanced_rows.append(("OI Change %", fundamentals.get("open_interest", {}).get("change_pct")))
                advanced_rows.append(("Long/Short Ratio", fundamentals.get("long_short_ratio", {}).get("ratio")))
                breadth = advanced.get("breadth", {})
                advanced_rows.append(("BTC Dominance", breadth.get("btc_dominance")))
                advanced_rows.append(("Fear & Greed", breadth.get("fear_greed_index")))
                trade_plan = advanced.get("trade_plan", {})
                signal = trade_plan.get("signal", {})
                risk = trade_plan.get("risk", {})
                advanced_rows.append(("Signal Type", signal.get("type")))
                advanced_rows.append(("Entry Price", signal.get("entry_price")))
                advanced_rows.append(("Stop Loss", risk.get("stop_loss")))
                advanced_rows.append(("ATR", risk.get("atr")))
            
            if advanced_rows:
                adv_csv = "Metric,Value\n" + "\n".join([
                    f"{metric},{value if value is not None else ''}"
                    for metric, value in advanced_rows
                ])
                st.download_button(
                    label="📥 Download CSV (Advanced Summary)",
                    data=adv_csv,
                    file_name=f"{selected_token.replace(':', '_')}_{selected_timeframe}_advanced.csv",
                    mime="text/csv",
                    width="stretch",
                )
            else:
                st.caption("Advanced metrics unavailable for export.")
        
        st.markdown("---")
        
        with st.expander("📄 View Full JSON Payload"):
            st.json(payload)


if __name__ == "__main__":
    main()
