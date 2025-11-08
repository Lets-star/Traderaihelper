"""Reusable helpers for collecting indicator metrics."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

from .types import CollectionResult
from .data_fetcher import (
    fetch_klines,
    fetch_order_book,
    fetch_and_validate_klines,
    create_source_metadata_dict,
)
from .advanced_metrics import compute_advanced_metrics
from .cme_gap import get_nearest_cme_gaps
from .astrology import get_astrology_metrics
from .indicator_metrics import (
    IndicatorSettings,
    IndicatorSimulator,
    SimulationSummary,
    summary_to_payload,
)
from .math_utils import Candle
from .time_series import MetricPoint, TimeframeMetricSeries, TimeframeSeries

DEFAULT_MULTI_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


def compute_trend_strength_series(
    simulator: IndicatorSimulator,
    series: TimeframeSeries,
    length: int,
) -> TimeframeMetricSeries:
    closes = [c.close for c in series.candles]
    strength = simulator._calculate_trend_strength_series(closes, length)
    points = [MetricPoint(candle.close_time, value) for candle, value in zip(series.candles, strength)]
    return TimeframeMetricSeries(points)


def safe_fetch_candles(symbol: str, timeframe: str, limit: int, context: str, 
                       validate_data: bool = True) -> List[Candle]:
    try:
        if validate_data:
            candles, metadata = fetch_and_validate_klines(symbol, timeframe, limit)
            return candles
        else:
            return fetch_klines(symbol, timeframe, limit)
    except RuntimeError as exc:
        print(f"[warning] {context}: {exc}", file=sys.stderr)
        return []


def collect_metrics(
    symbol: str,
    timeframe: str,
    period: int,
    token: str,
    *,
    multi_symbol: Optional[Sequence[str]] = None,
    disable_multi_symbol: bool = False,
    additional_timeframes: Optional[Sequence[str]] = None,
    validate_real_data: bool = True,
) -> CollectionResult:
    period_limit = min(max(period + 50, 200), 1000)
    
    # Fetch main timeframe with validation if enabled
    if validate_real_data:
        main_candles, main_metadata = fetch_and_validate_klines(symbol, timeframe, period_limit)
    else:
        main_candles = fetch_klines(symbol, timeframe, period_limit)
        main_metadata = create_source_metadata_dict("binance", "binance", symbol, timeframe, 
                                                  method="direct", validation_disabled=True)
    
    if len(main_candles) < period:
        raise RuntimeError(
            f"Requested period {period} but only received {len(main_candles)} bars for {symbol} {timeframe}"
        )
    main_slice = main_candles[-period:]
    main_series = TimeframeSeries(main_slice)
    reference_price = main_series.candles[-1].close if main_series.candles else 0.0

    timeframe_keys = list(DEFAULT_MULTI_TIMEFRAMES)
    if additional_timeframes:
        timeframe_keys.extend(additional_timeframes)
    timeframe_keys = list(dict.fromkeys(timeframe_keys))

    multi_timeframe_series: Dict[str, TimeframeSeries] = {}
    for tf in timeframe_keys:
        candles_tf = safe_fetch_candles(symbol, tf, max(period, 300), f"timeframe {tf}", validate_real_data)
        if len(candles_tf) < 3:
            continue
        multi_timeframe_series[tf] = TimeframeSeries(candles_tf)

    multi_symbol_series: Dict[str, TimeframeSeries] = {}
    if not disable_multi_symbol:
        symbols = list(multi_symbol)[:3] if multi_symbol else ["BINANCE:ETHUSDT", "BINANCE:SOLUSDT"]
        for sym in symbols:
            candles_sym = safe_fetch_candles(sym, timeframe, period + 50, f"multi-symbol {sym}", validate_real_data)
            if len(candles_sym) < 3:
                continue
            multi_symbol_series[sym] = TimeframeSeries(candles_sym)

    settings = IndicatorSettings()
    multi_timeframe_strength: Dict[str, TimeframeMetricSeries] = {}

    simulator = IndicatorSimulator(
        settings,
        main_series,
        multi_timeframe_series,
        multi_timeframe_strength,
        multi_symbol_series,
    )

    for tf, series in multi_timeframe_series.items():
        metric_series = compute_trend_strength_series(simulator, series, settings.trend_strength_period)
        multi_timeframe_strength[tf] = metric_series

    for sym, series in multi_symbol_series.items():
        metric_series = compute_trend_strength_series(simulator, series, settings.trend_strength_period)
        multi_timeframe_strength[f"{sym}_trend"] = metric_series

    summary = simulator.run()

    try:
        orderbook_data = fetch_order_book(symbol, limit=1000)
    except RuntimeError as exc:
        print(f"[warning] Failed to fetch real orderbook from Binance: {exc}", file=sys.stderr)
        print("[warning] Market maker detection requires real orderbook data", file=sys.stderr)
        orderbook_data = None

    summary.orderbook_data = orderbook_data
    advanced_data = compute_advanced_metrics(summary, main_series.candles)
    payload = summary_to_payload(summary, symbol, timeframe, period, token)
    payload["advanced"] = advanced_data

    # Add source metadata to payload
    payload["metadata"].update(main_metadata)
    
    # Add real data validation status
    payload["metadata"]["real_data_validated"] = validate_real_data
    if validate_real_data:
        payload["metadata"]["data_quality"] = "validated_real_data"
    else:
        payload["metadata"]["data_quality"] = "not_validated"

    cme_gap_data = get_nearest_cme_gaps(symbol, reference_price)
    payload.setdefault("latest", {})["cme_gaps"] = cme_gap_data
    
    latest_timestamp = main_series.candles[-1].close_time if main_series.candles else None
    if latest_timestamp:
        astrology_data = get_astrology_metrics(latest_timestamp)
        payload["astrology"] = astrology_data

    return CollectionResult(
        payload=payload,
        summary=summary,
        main_series=main_series,
        multi_timeframe_series=multi_timeframe_series,
        multi_timeframe_strength=multi_timeframe_strength,
        multi_symbol_series=multi_symbol_series,
    )
