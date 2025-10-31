from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .collector import collect_metrics


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

    result = collect_metrics(
        symbol=args.symbol,
        timeframe=args.timeframe,
        period=args.period,
        token=args.token,
        offline=bool(args.offline),
        multi_symbol=args.multi_symbol,
        disable_multi_symbol=args.disable_multi_symbol,
        additional_timeframes=args.additional_timeframes,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.write_text(json.dumps(result.payload, indent=2))
    else:
        print(json.dumps(result.payload, indent=2))


if __name__ == "__main__":
    main()
