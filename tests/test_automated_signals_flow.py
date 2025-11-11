from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from indicator_collector.trading_system.automated_signals import run_automated_signal_flow


class _StubBinanceSource:
    """Stub data source returning deterministic candles for testing."""

    def __init__(self, frames: Dict[str, pd.DataFrame]) -> None:
        self._frames = frames
        self.calls: list[tuple[str, str, datetime, datetime]] = []

    def load_candles(self, symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:  # noqa: ANN001
        key = timeframe.value if hasattr(timeframe, "value") else str(timeframe)
        self.calls.append((symbol, key, start, end))
        frame = self._frames.get(key)
        if frame is None:
            raise ValueError(f"No frame registered for timeframe {key}")
        return frame.copy()


def _build_candles(start: datetime, interval_seconds: int, count: int) -> pd.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    interval_ms = interval_seconds * 1000
    rows = []
    price = 100.0
    for idx in range(count):
        ts = start_ms + idx * interval_ms
        open_price = price
        high = open_price + 1.0
        low = open_price - 1.0
        close = open_price + 0.5
        volume = 10.0 + idx
        rows.append({
            "ts": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        price += 0.2
    return pd.DataFrame(rows)


def test_run_automated_signal_flow_with_stubbed_binance_source() -> None:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(minutes=90)
    frame = _build_candles(start, interval_seconds=60, count=90)

    stub_source = _StubBinanceSource(frame)

    result = run_automated_signal_flow(
        symbol="BTCUSDT",
        timeframe="1m",
        start=start,
        end=now,
        data_source=stub_source,
        validate_real_data=True,
    )

    assert len(result.candles) == 90

    metadata = result.processed_payload["metadata"]
    assert metadata["source"] == "binance"
    assert metadata["real_data_validated"] is True
    assert metadata["symbol"] == "BTCUSDT"

    signal = result.explicit_signal
    assert signal["signal"] in {"BUY", "SELL", "HOLD"}

    metadata_values = [value for value in metadata.values() if isinstance(value, str)]
    assert not any("synthetic" in value.lower() for value in metadata_values)
