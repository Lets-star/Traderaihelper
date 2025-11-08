from datetime import datetime, timedelta, timezone

import pandas as pd

from indicator_collector.trading_system import (
    BacktestConfig,
    Backtester,
    ParameterSet,
    build_backtest_payloads_from_candles,
)


def test_backtester_loads_binance_payloads_without_insufficient_error():
    """Backtester should accept payloads derived from Binance candles without raising errors."""
    now = datetime.now(tz=timezone.utc)
    base_time = now - timedelta(hours=1101)

    rows = []
    for index in range(1100):
        open_time = base_time + timedelta(hours=index)
        ts_ms = int(open_time.timestamp() * 1000)
        price = 50000.0 + index * 5.0
        rows.append(
            {
                "ts": ts_ms,
                "open": price,
                "high": price + 50,
                "low": price - 50,
                "close": price + 10,
                "volume": 100 + index * 0.5,
            }
        )

    candles = pd.DataFrame(rows)

    payloads = build_backtest_payloads_from_candles(
        candles=candles,
        symbol="BTCUSDT",
        timeframe="1h",
        display_symbol="BINANCE:BTCUSDT",
    )

    config = BacktestConfig(
        validate_real_data=False,
        min_data_points_per_timeframe={"1h": 1000},
    )
    backtester = Backtester(config)

    loaded_count = backtester.load_historical_data(payloads, symbol="BTCUSDT", timeframe="1h")

    assert loaded_count >= 1000

    params = ParameterSet(timeframe="1h")
    result = backtester.run_backtest(params)
    assert result.parameter_set.timeframe == "1h"
