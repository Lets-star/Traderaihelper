from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Dict

import pytest
import streamlit as real_st

from config_store import ConfigStore


class FakeColumn:
    """Lightweight proxy to validate column widget interactions."""

    def __init__(self, parent: "FakeStreamlit") -> None:
        self._parent = parent

    def number_input(self, label: str, **kwargs: Any) -> Any:
        return self._parent._handle_number_input(label, kwargs)

    def slider(self, label: str, **kwargs: Any) -> Any:
        return self._parent._handle_slider(label, kwargs)

    def selectbox(self, label: str, options: list[Any], **kwargs: Any) -> Any:
        return self._parent.selectbox(label, options, **kwargs)


class FakeStreamlit:
    """Minimal Streamlit stub that enforces consistent numeric typing."""

    def __init__(self) -> None:
        self.number_input_calls: list[tuple[str, Any, Dict[str, Any]]] = []
        self.slider_calls: list[tuple[str, Dict[str, Any]]] = []
        self.selectbox_calls: list[tuple[str, list[Any], Dict[str, Any]]] = []
        self.session_state: Dict[str, Any] = {}

    # Core widget handlers -------------------------------------------------
    def _handle_number_input(self, label: str, kwargs: Dict[str, Any]) -> Any:
        value = kwargs["value"]
        min_value = kwargs["min_value"]
        step_value = kwargs["step"]

        if isinstance(value, bool):
            raise AssertionError(f"Boolean values are not supported for {label}")

        if isinstance(value, int):
            assert isinstance(min_value, int), f"Expected int min_value for {label}"
            assert isinstance(step_value, int), f"Expected int step for {label}"
            max_value = kwargs.get("max_value")
            if max_value is not None:
                assert isinstance(max_value, int), f"Expected int max_value for {label}"
            result = int(value)
        elif isinstance(value, float):
            assert isinstance(min_value, float), f"Expected float min_value for {label}"
            assert isinstance(step_value, float), f"Expected float step for {label}"
            max_value = kwargs.get("max_value")
            if max_value is not None:
                assert isinstance(max_value, float), f"Expected float max_value for {label}"
            result = float(value)
        else:
            raise AssertionError(f"Unsupported value type {type(value)!r} for {label}")

        self.number_input_calls.append((label, result, kwargs))
        return result

    def _handle_slider(self, label: str, kwargs: Dict[str, Any]) -> Any:
        self.slider_calls.append((label, kwargs))
        key = kwargs.get("key")
        if key is not None:
            self.session_state[key] = kwargs["value"]
        return kwargs["value"]

    # Streamlit-like API ---------------------------------------------------
    def columns(self, count: int) -> list[FakeColumn]:
        return [FakeColumn(self) for _ in range(count)]

    def number_input(self, label: str, **kwargs: Any) -> Any:
        return self._handle_number_input(label, kwargs)

    def slider(self, label: str, **kwargs: Any) -> Any:
        return self._handle_slider(label, kwargs)

    def selectbox(self, label: str, options: list[Any], **kwargs: Any) -> Any:
        self.selectbox_calls.append((label, options, kwargs))
        index = kwargs.get("index", 0)
        if not isinstance(index, int) or not (0 <= index < len(options)):
            index = 0
        value = options[index]
        key = kwargs.get("key")
        if key is not None:
            self.session_state[key] = value
        return value

    def cache_data(self, *args: Any, **kwargs: Any):  # pragma: no cover - caching disabled for tests
        def decorator(func: Any) -> Any:
            return func

        return decorator

    # No-op st methods used elsewhere in the module -----------------------
    def info(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def metric(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def data_frame(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def dataframe(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def markdown(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def caption(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def table(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def plotly_chart(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def subheader(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def metric_row(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - non-critical
        return None

    def expander(self, *args: Any, **kwargs: Any) -> "FakeStreamlit":  # pragma: no cover - minimal support
        return self

    def __enter__(self) -> "FakeStreamlit":  # pragma: no cover - minimal support
        return self

    def __exit__(self, *exc: Any) -> None:  # pragma: no cover - minimal support
        return None


@pytest.fixture()
def web_ui_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Reload ``web_ui`` with a harmless ``set_page_config`` stubbed out."""
    monkeypatch.setattr(real_st, "set_page_config", lambda *args, **kwargs: None)
    import web_ui

    return importlib.reload(web_ui)


def test_indicator_controls_use_consistent_number_input_types(
    monkeypatch: pytest.MonkeyPatch, web_ui_module: ModuleType
) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(web_ui_module, "st", fake_st)

    store = ConfigStore(ConfigStore._default_state())
    web_ui_module.render_indicator_controls(store)

    # MACD (3) + RSI (3) + ATR (2) + ATR Channels (4) + Bollinger (2)
    assert len(fake_st.number_input_calls) == 14


def test_signal_controls_produce_expected_numeric_types(
    monkeypatch: pytest.MonkeyPatch, web_ui_module: ModuleType
) -> None:
    fake_st = FakeStreamlit()
    monkeypatch.setattr(web_ui_module, "st", fake_st)

    store = ConfigStore(ConfigStore._default_state())
    web_ui_module.render_indicator_controls(store)
    web_ui_module.render_signal_risk_controls(store)

    indicator_params = store.get_indicator_params()
    assert isinstance(indicator_params["macd"]["fast"], int)
    assert isinstance(indicator_params["macd"]["slow"], int)
    assert isinstance(indicator_params["macd"]["signal"], int)
    assert isinstance(indicator_params["rsi"]["period"], int)
    assert isinstance(indicator_params["rsi"]["overbought"], int)
    assert isinstance(indicator_params["rsi"]["oversold"], int)
    assert isinstance(indicator_params["atr"]["period"], int)
    assert isinstance(indicator_params["atr"]["mult"], float)
    atr_channels = indicator_params["atr_channels"]
    assert isinstance(atr_channels["period"], int)
    assert isinstance(atr_channels["mult_1x"], float)
    assert isinstance(atr_channels["mult_2x"], float)
    assert isinstance(atr_channels["mult_3x"], float)
    bollinger = indicator_params["bollinger"]
    assert isinstance(bollinger["period"], int)
    assert isinstance(bollinger["mult"], float)
    assert isinstance(bollinger["source"], str)

    risk_settings = store.risk_settings()
    assert isinstance(risk_settings["account_balance"], float)
    assert isinstance(risk_settings["max_risk_per_trade_pct"], float)
    assert isinstance(risk_settings["max_position_size_pct"], float)

    signal_settings = store.signal_settings()
    assert isinstance(signal_settings["min_confirmations"], int)
    assert isinstance(signal_settings["buy_threshold"], float)
    assert isinstance(signal_settings["sell_threshold"], float)
    assert isinstance(signal_settings["min_confidence"], float)
