from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from .data_fetcher import fetch_klines, generate_synthetic_candles
from .indicator_metrics import IndicatorSettings, IndicatorSimulator, MetricPoint, TimeframeMetricSeries, TimeframeSeries, summary_to_payload
from .math_utils import Candle

DEFAULT_MULTI_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


def compute_trend_strength_series(simulator: IndicatorSimulator, series: TimeframeSeries, length: int) -> TimeframeMetricSeries:
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


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect metrics from FVG & Order Block Sync Pro indicator logic")
    parser.add_argument("--symbol", default="BINANCE:BTCUSDT", help="Primary symbol to analyse (exchange prefix optional)")
    parser.add_argument("--timeframe", default="15m", help="Primary timeframe (e.g. 15m, 1h, 4h)")
    parser.add_argument("--period", type=int, default=500, help="Number of bars to process")
    parser.add_argument("--token", required=True, help="Authentication token to include in the payload")
    parser.add_argument("--output", help="Optional path to write JSON payload")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic synthetic data instead of downloading from Binance",
    )
    parser.add_argument(
        "--multi-symbol",
        nargs="*",
        default=["BINANCE:ETHUSDT", "BINANCE:SOLUSDT"],
        help="Additional symbols for multi-symbol analysis",
    )
    parser.add_argument(
        "--disable-multi-symbol",
        action="store_true",
        help="Disable multi-symbol analysis even if symbols are provided",
    )
    parser.add_argument(
        "--additional-timeframes",
        nargs="*",
        help="Extra timeframes to include besides the defaults (e.g. 30m 2h)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    offline = bool(args.offline)

    period_limit = min(max(args.period + 50, 200), 1000)
    main_candles = fetch_or_generate(args.symbol, args.timeframe, period_limit, offline, "primary timeframe")
    if len(main_candles) < args.period:
        raise RuntimeError(
            f"Requested period {args.period} but only received {len(main_candles)} bars for {args.symbol} {args.timeframe}"
        )
    main_slice = main_candles[-args.period :]
    main_series = TimeframeSeries(main_slice)

    timeframe_keys = list(DEFAULT_MULTI_TIMEFRAMES)
    if args.additional_timeframes:
        timeframe_keys.extend(args.additional_timeframes)
    timeframe_keys = list(dict.fromkeys(timeframe_keys))  # remove duplicates preserving order

    multi_timeframe_series: Dict[str, TimeframeSeries] = {}
    for tf in timeframe_keys:
        candles_tf = fetch_or_generate(args.symbol, tf, max(args.period, 300), offline, f"timeframe {tf}")
        if len(candles_tf) < 3:
            raise RuntimeError(f"Not enough data available for {args.symbol} {tf}")
        multi_timeframe_series[tf] = TimeframeSeries(candles_tf)

    multi_symbol_series: Dict[str, TimeframeSeries] = {}
    if not args.disable_multi_symbol:
        for sym in args.multi_symbol[:3]:
            candles_sym = fetch_or_generate(sym, args.timeframe, args.period + 50, offline, f"multi-symbol {sym}")
            if len(candles_sym) < 3:
                print(f"[skip] multi-symbol {sym}: insufficient data after fallback.", file=sys.stderr)
                continue
            multi_symbol_series[sym] = TimeframeSeries(candles_sym)

    settings = IndicatorSettings()
    # Placeholders that will be populated after simulator creation
    multi_timeframe_strength: Dict[str, TimeframeMetricSeries] = {}

    simulator = IndicatorSimulator(settings, main_series, multi_timeframe_series, multi_timeframe_strength, multi_symbol_series)

    for tf, series in multi_timeframe_series.items():
        metric_series = compute_trend_strength_series(simulator, series, settings.trend_strength_period)
        multi_timeframe_strength[tf] = metric_series

    for sym, series in multi_symbol_series.items():
        metric_series = compute_trend_strength_series(simulator, series, settings.trend_strength_period)
        multi_timeframe_strength[f"{sym}_trend"] = metric_series

    summary = simulator.run()
    payload = summary_to_payload(summary, args.symbol, args.timeframe, args.period, args.token)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
