"""Utilities for generating automated trading signals from Binance data."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from indicator_collector.timeframes import Timeframe

from .data_sources.binance_source import BinanceKlinesSource
from .data_sources.timestamp_utils import (
    ensure_utc_datetime,
    get_last_closed_candle_ts,
    floor_to_interval,
)
from .generate_signals import generate_signals
from .payload_loader import load_full_payload
from .signal_generator import SignalConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutomatedSignalResult:
    """Container for automated signal generation outputs."""

    candles: List[Dict[str, Any]]
    processed_payload: Dict[str, Any]
    explicit_signal: Dict[str, Any]


_MULTI_TIMEFRAME_RELATIONS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    Timeframe.M1.value: (("same", Timeframe.M1.value), ("higher", Timeframe.M5.value)),
    Timeframe.M5.value: (("lower", Timeframe.M1.value), ("same", Timeframe.M5.value), ("higher", Timeframe.M15.value)),
    Timeframe.M15.value: (("lower", Timeframe.M5.value), ("same", Timeframe.M15.value), ("higher", Timeframe.H1.value)),
    Timeframe.H1.value: (("lower", Timeframe.M15.value), ("same", Timeframe.H1.value), ("higher", Timeframe.H4.value)),
    Timeframe.H3.value: (("lower", Timeframe.H1.value), ("same", Timeframe.H3.value), ("higher", Timeframe.H4.value)),
    Timeframe.H4.value: (("lower", Timeframe.H1.value), ("same", Timeframe.H4.value), ("higher", Timeframe.D1.value)),
    Timeframe.D1.value: (("lower", Timeframe.H4.value), ("same", Timeframe.D1.value)),
}

_MIN_MTF_BARS_BY_RELATION: Dict[str, int] = {
    "lower": 100,
    "same": 80,
    "higher": 50,
}

_MTF_CACHE: Dict[Tuple[str, str, int, int, int], List[Dict[str, Any]]] = {}
_MTF_PARAMS_HASH = hashlib.sha1(
    json.dumps(
        {
            "relations": {
                key: [list(relation) for relation in relations]
                for key, relations in _MULTI_TIMEFRAME_RELATIONS.items()
            },
            "min_bars": _MIN_MTF_BARS_BY_RELATION,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def _normalize_symbol(symbol: str) -> str:
    if not symbol:
        raise ValueError("Symbol must be provided")
    return symbol.strip().upper()


def _to_dataframe(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No candle data provided")
    df = pd.DataFrame(records)
    required_cols = {"ts", "open", "high", "low", "close", "volume"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {sorted(missing)}")
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="raise")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df.sort_values("ts").reset_index(drop=True)


def _dataframe_to_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "ts": int(row.ts),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in df.itertuples()
    ]


def _adjacent_timeframe_relations(base_timeframe: Timeframe) -> Tuple[Tuple[str, str], ...]:
    return _MULTI_TIMEFRAME_RELATIONS.get(base_timeframe.value, (("same", base_timeframe.value),))


def _relation_min_bars(relation: str) -> int:
    return _MIN_MTF_BARS_BY_RELATION.get(relation, 50)


def _align_end_timestamp(last_closed_ts: int, timeframe_value: str) -> int:
    tf_enum = Timeframe.from_value(timeframe_value)
    interval_ms = tf_enum.to_milliseconds()
    if interval_ms <= 0:
        return last_closed_ts
    aligned = floor_to_interval(last_closed_ts, interval_ms)
    if last_closed_ts % interval_ms == 0:
        aligned = last_closed_ts
    return aligned


def _load_multi_timeframe_series(
    source: BinanceKlinesSource,
    symbol: str,
    timeframe_value: str,
    start_dt: datetime,
    end_dt: datetime,
    min_bars: int,
) -> Tuple[Optional[List[Dict[str, Any]]], bool, Optional[str]]:
    tf_enum = Timeframe.from_value(timeframe_value)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    if start_ms >= end_ms:
        interval_ms = tf_enum.to_milliseconds()
        span_ms = interval_ms * max(min_bars or 1, 1)
        start_ms = max(0, end_ms - span_ms)
        start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    cache_key = (symbol, tf_enum.value, start_ms, end_ms, int(min_bars))
    cached = _MTF_CACHE.get(cache_key)
    if cached is not None:
        return cached, True, None
    try:
        df = source.load_candles(symbol, tf_enum, start_dt, end_dt)
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning(
            "Failed to load %s candles for %s between %s and %s: %s",
            tf_enum.value,
            symbol,
            start_dt.isoformat(),
            end_dt.isoformat(),
            exc,
        )
        return None, False, str(exc)
    df = df.sort_values("ts").reset_index(drop=True)
    candles = _dataframe_to_candles(df)
    _MTF_CACHE[cache_key] = candles
    return candles, False, None


def _prepare_multi_timeframe_payload(
    source: BinanceKlinesSource,
    symbol: str,
    base_timeframe: Timeframe,
    base_candles: Sequence[Dict[str, Any]],
    last_closed_ts: int,
    analysis_start: datetime,
    analysis_end: datetime,
) -> Dict[str, Any]:
    relations = _adjacent_timeframe_relations(base_timeframe)
    requested = [tf for _, tf in relations]
    candles: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[str] = []
    errors: Dict[str, str] = {}
    cache_hits: Dict[str, bool] = {}
    fetched_bars: Dict[str, int] = {}

    for relation, tf_value in relations:
        min_bars = _relation_min_bars(relation)
        if relation == "same":
            window = max(min_bars * 2, min_bars + 20)
            sliced = list(base_candles[-window:]) if window > 0 else list(base_candles)
            candles[tf_value] = sliced
            fetched_bars[tf_value] = len(sliced)
            cache_hits[tf_value] = True
            continue

        aligned_end_ms = _align_end_timestamp(last_closed_ts, tf_value)
        if aligned_end_ms <= 0:
            missing.append(tf_value)
            continue

        end_dt_aligned = datetime.fromtimestamp(aligned_end_ms / 1000, tz=timezone.utc)
        adjusted_end = min(end_dt_aligned, analysis_end)
        interval_ms = Timeframe.from_value(tf_value).to_milliseconds()
        extra_bars = max(min_bars + 10, int(min_bars * 1.25))
        span_ms = interval_ms * extra_bars
        start_ms = max(0, aligned_end_ms - span_ms)
        start_dt_candidate = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        candles_list, cache_hit, error_message = _load_multi_timeframe_series(
            source,
            symbol,
            tf_value,
            start_dt_candidate,
            adjusted_end,
            min_bars,
        )
        cache_hits[tf_value] = cache_hit
        if candles_list is None:
            missing.append(tf_value)
            if error_message:
                errors[tf_value] = error_message
            continue
        fetched_count = len(candles_list)
        fetched_bars[tf_value] = fetched_count
        if fetched_count < min_bars:
            missing.append(tf_value)
            continue
        candles[tf_value] = candles_list

    for tf_value in requested:
        cache_hits.setdefault(tf_value, False)
        fetched_bars.setdefault(tf_value, 0)

    metadata: Dict[str, Any] = {
        "base_timeframe": base_timeframe.value,
        "requested_timeframes": requested,
        "fetched_timeframes": sorted(candles.keys()),
        "missing_timeframes": sorted(set(missing)),
        "min_bars": {tf: _relation_min_bars(rel) for rel, tf in relations},
        "fetched_bars": fetched_bars,
        "cache_hits": cache_hits,
        "effective_end_ts": int(last_closed_ts),
        "params_hash": _MTF_PARAMS_HASH,
        "analysis_window": {
            "start_ts": int(analysis_start.timestamp() * 1000),
            "end_ts": int(analysis_end.timestamp() * 1000),
        },
    }
    if errors:
        metadata["errors"] = errors
    if metadata["missing_timeframes"]:
        metadata["note"] = (
            "Missing multi-timeframe data for "
            + ", ".join(metadata["missing_timeframes"])
            + "; treated as neutral."
        )

    return {
        "candles": candles,
        "metadata": metadata,
        "trend_strength": {},
    }


def build_payload_from_candles(
    symbol: str,
    timeframe: str,
    candles: Sequence[Dict[str, Any]],
    *,
    multi_timeframe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize raw candle records into a trading system payload.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT").
        timeframe: Timeframe string (e.g., "1h", "3h").
        candles: Sequence of candle dictionaries containing ts/open/high/low/close/volume.

    Returns:
        Payload dictionary compatible with ``load_full_payload``.
    """
    normalized_symbol = _normalize_symbol(symbol)
    tf = Timeframe.from_value(timeframe)

    df = _to_dataframe(candles)
    last_closed_ts = get_last_closed_candle_ts(df, tf)
    last_candle = df.iloc[-1]

    payload_candles = _dataframe_to_candles(df)

    metadata: Dict[str, Any] = {
        "source": "binance",
        "exchange": "binance",
        "symbol": normalized_symbol,
        "timeframe": tf.value,
        "granularity": tf.value,
        "timestamp": last_closed_ts,
        "start_timestamp": int(df.iloc[0].ts),
        "end_timestamp": last_closed_ts,
        "bar_count": len(payload_candles),
        "data_quality": "binance_historical",
        "real_data": True,
        "is_real_data": True,
        "real_data_validated": False,
        "timezone": "UTC",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    latest: Dict[str, Any] = {
        "timestamp": last_closed_ts,
        "timeframe": tf.value,
        "symbol": normalized_symbol,
        "open": float(last_candle.open),
        "high": float(last_candle.high),
        "low": float(last_candle.low),
        "close": float(last_candle.close),
        "volume": float(last_candle.volume),
        "open_time": int(last_candle.ts),
        "open_time_iso": datetime.fromtimestamp(
            int(last_candle.ts) / 1000, tz=timezone.utc
        ).isoformat(),
        "close_time_iso": datetime.fromtimestamp(
            last_closed_ts / 1000, tz=timezone.utc
        ).isoformat(),
    }

    return {
        "metadata": metadata,
        "latest": latest,
        "candles": payload_candles,
        "advanced": {},
        "multi_timeframe": multi_timeframe or {},
        "zones": [],
        "signals": [],
    }


def run_automated_signal_flow(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    *,
    data_source: Optional[BinanceKlinesSource] = None,
    validate_real_data: bool = True,
    min_candles: int = 30,
    signal_config: Optional[SignalConfig] = None,
    indicator_params: Optional[Dict[str, Any]] = None,
    signal_params: Optional[Dict[str, Any]] = None,
) -> AutomatedSignalResult:
    """Fetch Binance candles and generate trading signals.

    Args:
        symbol: Trading symbol.
        timeframe: Requested timeframe.
        start: Start datetime (inclusive).
        end: End datetime (inclusive).
        data_source: Optional Binance data source override (useful for tests).
        validate_real_data: Whether to run ``RealDataValidator`` during processing.
        min_candles: Minimum number of candles required to generate signals.
        signal_config: Optional ``SignalConfig`` overrides for analyzer weights and thresholds.
        indicator_params: Optional indicator parameter overrides (e.g., MACD/RSI/ATR).
        signal_params: Optional explicit signal generation parameters (risk settings, etc.).

    Returns:
        ``AutomatedSignalResult`` containing candles, processed payload, and explicit signal.
    """
    start_utc = ensure_utc_datetime(start)
    end_utc = ensure_utc_datetime(end)
    if start_utc >= end_utc:
        raise ValueError("Start time must be before end time for automated signals")

    tf = Timeframe.from_value(timeframe)
    symbol_norm = _normalize_symbol(symbol)

    source = data_source or BinanceKlinesSource()
    df = source.load_candles(symbol_norm, tf, start_utc, end_utc)
    if df is None or df.empty:
        raise ValueError(f"No Binance candles returned for {symbol_norm} {tf.value}")

    df = df.sort_values("ts").reset_index(drop=True)
    candles = _dataframe_to_candles(df)
    if len(candles) < min_candles:
        raise ValueError(
            f"Insufficient candles ({len(candles)}) for {symbol_norm} {tf.value}; "
            f"need at least {min_candles}"
        )

    last_closed_ts = get_last_closed_candle_ts(df, tf)
    try:
        multi_timeframe_payload = _prepare_multi_timeframe_payload(
            source,
            symbol_norm,
            tf,
            candles,
            last_closed_ts,
            start_utc,
            end_utc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to prepare multi-timeframe payload: %s", exc)
        multi_timeframe_payload = {"candles": {}, "metadata": {"note": str(exc)}, "trend_strength": {}}

    payload = build_payload_from_candles(
        symbol_norm,
        tf.value,
        candles,
        multi_timeframe=multi_timeframe_payload,
    )
    processed_payload = load_full_payload(
        payload,
        timeframe=tf.value,
        validate_real_data=validate_real_data,
        signal_config=signal_config,
        indicator_params=indicator_params,
    ).to_dict()

    params: Dict[str, Any] = {}
    if signal_params:
        params.update(signal_params)
    if indicator_params and "indicator_params" not in params:
        params["indicator_params"] = indicator_params
    if signal_config:
        params.setdefault("weights", {
            "technical": signal_config.technical_weight,
            "sentiment": signal_config.sentiment_weight,
            "multitimeframe": signal_config.multitimeframe_weight,
            "volume": signal_config.volume_weight,
            "market_structure": signal_config.structure_weight,
            "composite": signal_config.composite_weight,
        })

    composite_overrides = (indicator_params or {}).get("composite") if indicator_params else None
    if isinstance(composite_overrides, dict):
        buy_override = composite_overrides.get("buy_threshold")
        sell_override = composite_overrides.get("sell_threshold")
        if buy_override is not None or sell_override is not None:
            thresholds = dict(params.get("signal_thresholds") or {})
            metadata = processed_payload.setdefault("metadata", {})
            composite_meta = metadata.get("composite")
            if not isinstance(composite_meta, dict):
                composite_meta = {}
                metadata["composite"] = composite_meta
            if buy_override is not None:
                buy_value = float(buy_override)
                thresholds["buy"] = buy_value
                metadata["buy_threshold"] = buy_value
                composite_meta["buy_threshold"] = buy_value
            if sell_override is not None:
                sell_value = float(sell_override)
                thresholds["sell"] = sell_value
                metadata["sell_threshold"] = sell_value
                composite_meta["sell_threshold"] = sell_value
            if thresholds:
                params["signal_thresholds"] = thresholds

    explicit_signal = generate_signals(processed_payload, params=params or None)

    return AutomatedSignalResult(
        candles=candles,
        processed_payload=processed_payload,
        explicit_signal=explicit_signal,
    )
