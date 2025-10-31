"""Reusable helpers for collecting indicator metrics."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .data_fetcher import (
    fetch_klines,
    fetch_order_book,
    generate_synthetic_candles,
    generate_synthetic_order_book,
)
from .indicator_metrics import (
    IndicatorSettings,
    IndicatorSimulator,
    SimulationSummary,
    summary_to_payload,
)
from .math_utils import Candle
from .time_series import MetricPoint, TimeframeMetricSeries, TimeframeSeries

DEFAULT_MULTI_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


@dataclass
class CollectionResult:
    payload: Dict[str, object]
    summary: SimulationSummary
    main_series: TimeframeSeries
    multi_timeframe_series: Dict[str, TimeframeSeries]
    multi_timeframe_strength: Dict[str, TimeframeMetricSeries]
    multi_symbol_series: Dict[str, TimeframeSeries]


def compute_trend_strength_series(
    simulator: IndicatorSimulator,
    series: TimeframeSeries,
    length: int,
) -> TimeframeMetricSeries:
    closes = [c.close for c in series.candles]
    strength = simulator._calculate_trend_strength_series(closes, length)
    points = [MetricPoint(candle.close_time, value) for candle, value in zip(series.candles, strength)]
    return TimeframeMetricSeries(points)


def fetch_or_generate(symbol: str, timeframe: str, limit: int, offline: bool, context: str) -> List[Candle]:
    if offline:
        return generate_synthetic_candles(symbol, timeframe, limit)
    try:
        return fetch_klines(symbol, timeframe, limit)
    except RuntimeError as exc:
        print(f"[fallback] {context}: {exc}. Using synthetic data instead.", file=sys.stderr)
        return generate_synthetic_candles(symbol, timeframe, limit)


def collect_metrics(
    symbol: str,
    timeframe: str,
    period: int,
    token: str,
    *,
    offline: bool = False,
    multi_symbol: Optional[Sequence[str]] = None,
    disable_multi_symbol: bool = False,
    additional_timeframes: Optional[Sequence[str]] = None,
) -> CollectionResult:
    period_limit = min(max(period + 50, 200), 1000)
    main_candles = fetch_or_generate(symbol, timeframe, period_limit, offline, "primary timeframe")
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
        candles_tf = fetch_or_generate(symbol, tf, max(period, 300), offline, f"timeframe {tf}")
        if len(candles_tf) < 3:
            continue
        multi_timeframe_series[tf] = TimeframeSeries(candles_tf)

    multi_symbol_series: Dict[str, TimeframeSeries] = {}
    if not disable_multi_symbol:
        symbols = list(multi_symbol)[:3] if multi_symbol else ["BINANCE:ETHUSDT", "BINANCE:SOLUSDT"]
        for sym in symbols:
            candles_sym = fetch_or_generate(sym, timeframe, period + 50, offline, f"multi-symbol {sym}")
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
    
    orderbook_data = None
    if offline:
        orderbook_data = generate_synthetic_order_book(symbol, reference_price, limit=100)
    else:
        try:
            orderbook_data = fetch_order_book(symbol, limit=100)
        except RuntimeError as exc:
            print(f"[warning] Failed to fetch orderbook: {exc}", file=sys.stderr)
            orderbook_data = generate_synthetic_order_book(symbol, reference_price, limit=100)
    
    summary.orderbook_data = orderbook_data
    payload = summary_to_payload(summary, symbol, timeframe, period, token)

    return CollectionResult(
        payload=payload,
        summary=summary,
        main_series=main_series,
        multi_timeframe_series=multi_timeframe_series,
        multi_timeframe_strength=multi_timeframe_strength,
        multi_symbol_series=multi_symbol_series,
    )
