from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from ..timeframes import Timeframe
from .data_sources.timestamp_utils import normalize_timestamp, validate_no_future_timestamps


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class BacktestPayload:
    """Lightweight payload wrapper compatible with Backtester expectations."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        metadata = data.get("metadata", {})
        self.symbol: Optional[str] = data.get("symbol") or metadata.get("symbol")
        self.timeframe: Optional[str] = data.get("timeframe") or metadata.get("timeframe")

    @property
    def timestamp(self) -> int:
        return int(self._data.get("timestamp", 0) or 0)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)


def build_backtest_payloads_from_candles(
    candles: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    display_symbol: Optional[str] = None,
    source: str = "binance",
    exchange: str = "binance",
) -> List[BacktestPayload]:
    """Convert a dataframe of OHLCV data into normalized backtest payloads."""

    if candles.empty:
        return []

    required_columns = {"ts", "open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(candles.columns)
    if missing_columns:
        raise ValueError(f"Missing candle columns: {', '.join(sorted(missing_columns))}")

    tf_enum = Timeframe.from_value(timeframe)
    timeframe_ms = tf_enum.to_milliseconds()
    timeframe_minutes = tf_enum.to_minutes_instance()

    sorted_candles = candles.sort_values("ts").reset_index(drop=True)
    volume_median = float(sorted_candles["volume"].median()) or 1.0

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    tolerance_ms = 60 * 1000

    payloads: List[BacktestPayload] = []
    timestamps: List[int] = []

    for _, row in sorted_candles.iterrows():
        try:
            open_time = normalize_timestamp(row["ts"])
        except ValueError:
            continue
        try:
            close_time = normalize_timestamp(open_time + timeframe_ms)
        except ValueError:
            continue

        if close_time > now_ms + tolerance_ms:
            continue

        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        volume = float(row["volume"])

        if close_price <= 0:
            continue

        price_change_pct = 0.0
        if open_price > 0:
            price_change_pct = (close_price - open_price) / open_price * 100

        signal_type = "NEUTRAL"
        if price_change_pct > 0.05:
            signal_type = "BUY"
        elif price_change_pct < -0.05:
            signal_type = "SELL"

        range_pct = 0.0
        if open_price > 0:
            range_pct = max(0.0, (high_price - low_price) / open_price)

        volume_ratio = 0.0
        if volume_median > 0:
            volume_ratio = max(0.0, volume / volume_median)

        trend_score = _clamp(0.5 + price_change_pct / 20)
        momentum_score = _clamp(abs(price_change_pct) / 10)
        volatility_score = _clamp(range_pct / 0.05)
        volume_score = _clamp(volume_ratio / 3)

        confidence = _clamp((trend_score + momentum_score + volume_score) / 3)

        factors = [
            {"factor_name": "trend", "score": round(trend_score, 4), "weight": 1.0},
            {"factor_name": "momentum", "score": round(momentum_score, 4), "weight": 1.0},
            {"factor_name": "volume", "score": round(volume_score, 4), "weight": 1.0},
        ]

        timestamp_iso = datetime.fromtimestamp(close_time / 1000, tz=timezone.utc).isoformat()

        metadata = {
            "symbol": symbol,
            "full_symbol": display_symbol or symbol,
            "timeframe": tf_enum.value,
            "timeframe_minutes": timeframe_minutes,
            "granularity": tf_enum.value,
            "source": source,
            "exchange": exchange,
            "method": "historical_backtest",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "timestamp": close_time,
            "timestamp_iso": timestamp_iso,
            "data_quality": "binance_klines",
            "data_points": len(sorted_candles),
            "price_change_pct": price_change_pct,
            "volume_ratio": volume_ratio,
        }

        latest = {
            "timestamp": close_time,
            "time_iso": timestamp_iso,
            "timeframe": tf_enum.value,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "price_change_pct": price_change_pct,
            "volume_ratio": volume_ratio,
            "signal": signal_type if signal_type != "NEUTRAL" else None,
        }

        payload_dict: Dict[str, Any] = {
            "timestamp": close_time,
            "signal_type": signal_type,
            "entry_price": close_price,
            "confidence": round(confidence, 4),
            "symbol": display_symbol or symbol,
            "timeframe": tf_enum.value,
            "factors": factors,
            "metadata": metadata,
            "latest": latest,
        }

        payloads.append(BacktestPayload(payload_dict))
        timestamps.append(close_time)

    if not payloads:
        return []

    validate_no_future_timestamps(timestamps)
    return payloads
