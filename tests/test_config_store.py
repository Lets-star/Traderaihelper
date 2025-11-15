from __future__ import annotations

import math

import pytest

from config_store import ConfigStore


@pytest.fixture()
def store() -> ConfigStore:
    state = ConfigStore._default_state()
    return ConfigStore(state)


def test_normalized_weights_sum_to_one(store: ConfigStore) -> None:
    store.update_weight("technical", 0.5)
    store.update_weight("volume", 0.25)
    store.update_weight("sentiment", 0.25)

    normalized = store.normalized_weights()
    total = sum(normalized.values())
    assert math.isclose(total, 1.0, rel_tol=1e-6)


def test_signal_config_reflects_store_settings(store: ConfigStore) -> None:
    store.update_weight("market_structure", 0.4)
    store.update_weight("technical", 0.3)
    store.update_weight("volume", 0.15)
    store.update_weight("sentiment", 0.1)
    store.update_weight("multitimeframe", 0.05)

    store.update_signal_setting("min_confirmations", 4)
    store.update_signal_setting("buy_threshold", 0.7)
    store.update_signal_setting("sell_threshold", 0.3)
    store.update_signal_setting("min_confidence", 0.8)

    config = store.build_signal_config()
    weight_sum = (
        config.technical_weight
        + config.sentiment_weight
        + config.multitimeframe_weight
        + config.volume_weight
        + config.structure_weight
        + config.composite_weight
    )
    assert math.isclose(weight_sum, 1.0, rel_tol=1e-6)
    assert config.min_factors_confirm == 4
    assert math.isclose(config.buy_threshold, 0.7)
    assert math.isclose(config.sell_threshold, 0.3)
    assert math.isclose(config.min_confidence, 0.8)


def test_indicator_params_are_timeframe_specific(store: ConfigStore) -> None:
    initial_params = store.get_indicator_params()
    default_fast = initial_params["macd"]["fast"]

    store.update_indicator_param("macd", "fast", default_fast + 2)
    updated_params = store.get_indicator_params()
    assert updated_params["macd"]["fast"] == default_fast + 2

    store.set_timeframe("4h")
    params_4h = store.get_indicator_params()
    assert params_4h["macd"]["fast"] != default_fast + 2

    store.set_timeframe("1h")
    reverted_params = store.get_indicator_params()
    assert reverted_params["macd"]["fast"] == default_fast + 2


def test_state_persistence(store: ConfigStore) -> None:
    underlying = store._state  # access for testing persistence
    store.set_token("BINANCE:SOLUSDT")
    assert underlying["token"] == "BINANCE:SOLUSDT"

    store.update_signal_setting("min_confidence", 0.75)
    assert math.isclose(underlying["signal"]["min_confidence"], 0.75)


def test_build_signal_params_uses_decimal_percentages(store: ConfigStore) -> None:
    store.update_risk_setting("max_risk_per_trade_pct", 0.05)
    store.update_risk_setting("max_position_size_pct", 0.1)
    params = store.build_signal_params()
    assert math.isclose(params["max_risk_per_trade_pct"], 0.05)
    assert math.isclose(params["max_position_size_pct"], 0.1)
    thresholds = params.get("signal_thresholds")
    assert thresholds
    assert math.isclose(thresholds["buy"], store.signal_settings()["buy_threshold"])
    assert math.isclose(thresholds["sell"], store.signal_settings()["sell_threshold"])
