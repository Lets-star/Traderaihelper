"""Utilities for generating automated trading signals from Binance data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from indicator_collector.timeframes import Timeframe

from .data_sources.binance_source import BinanceKlinesSource
from .data_sources.timestamp_utils import get_last_closed_candle_ts
from .generate_signals import generate_signals
from .payload_loader import load_full_payload


@dataclass(frozen=True)
class AutomatedSignalResult:
    """Container for automated signal generation outputs."""

    candles: List[Dict[str, Any]]
    processed_payload: Dict[str, Any]
    explicit_signal: Dict[str, Any]


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


def build_payload_from_candles(
    symbol: str,
    timeframe: str,
    candles: Sequence[Dict[str, Any]],
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
        "multi_timeframe": {},
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

    Returns:
        ``AutomatedSignalResult`` containing candles, processed payload, and explicit signal.
    """
    if start >= end:
        raise ValueError("Start time must be before end time for automated signals")

    tf = Timeframe.from_value(timeframe)
    symbol_norm = _normalize_symbol(symbol)

    source = data_source or BinanceKlinesSource()
    df = source.load_candles(symbol_norm, tf, start, end)
    if df is None or df.empty:
        raise ValueError(f"No Binance candles returned for {symbol_norm} {tf.value}")

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    df = df[(df["ts"] >= start_ms) & (df["ts"] <= end_ms)].copy()
    if df.empty:
        raise ValueError(
            f"No candles within requested range for {symbol_norm} {tf.value}"
        )

    df = df.sort_values("ts").reset_index(drop=True)
    candles = _dataframe_to_candles(df)
    if len(candles) < min_candles:
        raise ValueError(
            f"Insufficient candles ({len(candles)}) for {symbol_norm} {tf.value}; "
            f"need at least {min_candles}"
        )

    payload = build_payload_from_candles(symbol_norm, tf.value, candles)
    processed_payload = load_full_payload(
        payload,
        timeframe=tf.value,
        validate_real_data=validate_real_data,
    ).to_dict()

    explicit_signal = generate_signals(processed_payload)

    return AutomatedSignalResult(
        candles=candles,
        processed_payload=processed_payload,
        explicit_signal=explicit_signal,
    )
