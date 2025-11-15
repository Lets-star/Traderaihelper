from __future__ import annotations

import math

import pytest

from indicator_collector.trading_system.backtester import (
    ParameterSet,
    indicator_defaults_for,
)
from indicator_collector.timeframes import Timeframe


@pytest.mark.parametrize(
    "timeframe",
    [
        Timeframe.M5.value,
        Timeframe.M15.value,
        Timeframe.H1.value,
        Timeframe.H3.value,
        Timeframe.H4.value,
        Timeframe.D1.value,
    ],
)
def test_indicator_defaults_include_atr_channels_and_bollinger(timeframe: str) -> None:
    defaults = indicator_defaults_for(timeframe)

    atr_channels = defaults.get("atr_channels")
    assert isinstance(atr_channels, dict)
    assert atr_channels.get("period") and atr_channels["period"] > 0
    for key in ("mult_1x", "mult_2x", "mult_3x"):
        assert key in atr_channels
        assert atr_channels[key] > 0

    bollinger = defaults.get("bollinger")
    assert isinstance(bollinger, dict)
    assert bollinger.get("period") and bollinger["period"] > 0
    assert bollinger.get("mult") and bollinger["mult"] > 0
    assert bollinger.get("stddev") and bollinger["stddev"] > 0
    assert bollinger.get("source")


def test_parameter_set_merges_indicator_overrides() -> None:
    overrides = {
        "bollinger": {"period": 18, "source": "hlc3"},
        "atr_channels": {"mult_3x": 4.5},
    }
    params = ParameterSet(timeframe=Timeframe.M15.value, indicator_params=overrides)

    merged = params.indicator_params
    assert merged["bollinger"]["period"] == 18
    assert merged["bollinger"]["source"] == "hlc3"
    assert math.isclose(merged["atr_channels"]["mult_3x"], 4.5)
    assert math.isclose(
        merged["bollinger"]["mult"],
        indicator_defaults_for(Timeframe.M15.value)["bollinger"]["mult"],
    )


def test_parameter_set_signal_thresholds_sanitized() -> None:
    params = ParameterSet(signal_thresholds={"buy": 0.2, "sell": 0.6})
    thresholds = params.signal_thresholds
    assert thresholds["buy"] > thresholds["sell"]
    assert math.isclose(thresholds["buy"], 0.6, rel_tol=1e-6)
    assert math.isclose(thresholds["sell"], 0.2, rel_tol=1e-6)
