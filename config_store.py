from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

import streamlit as st

from indicator_collector.trading_system.backtester import (
    DEFAULT_SIGNAL_THRESHOLDS,
    indicator_defaults_for,
)
from indicator_collector.trading_system.signal_generator import SignalConfig

DEFAULT_TOKEN = "BINANCE:BTCUSDT"
DEFAULT_TIMEFRAME = "1h"

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical": 0.3,
    "volume": 0.2,
    "sentiment": 0.15,
    "market_structure": 0.2,
    "multitimeframe": 0.15,
}

_DEFAULT_RISK: Dict[str, float] = {
    "account_balance": 10_000.0,
    "max_risk_per_trade_pct": 0.02,
    "max_position_size_pct": 0.05,
}

_DEFAULT_SIGNAL_SETTINGS: Dict[str, Any] = {
    "min_confirmations": 3,
    "buy_threshold": 0.65,
    "sell_threshold": 0.35,
    "min_confidence": 0.6,
}

_DEFAULT_BACKTEST_SETTINGS: Dict[str, Any] = {
    "max_bars": 320,
    "step": 1,
    "holding_bars": 24,
    "min_trades": 5,
}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)


def _parse_iso(value: str, fallback: dt.datetime) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:  # pragma: no cover - defensive
        return fallback


class ConfigStore:
    """Shared configuration persisted in ``st.session_state``."""

    STATE_KEY = "global_config_store"

    def __init__(self, state: Dict[str, Any]) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------
    @classmethod
    def _default_state(cls) -> Dict[str, Any]:
        end = _utc_now()
        start = end - dt.timedelta(days=3)
        return {
            "token": DEFAULT_TOKEN,
            "timeframe": DEFAULT_TIMEFRAME,
            "date_range": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "weights": {**_DEFAULT_WEIGHTS},
            "indicator_overrides": {},
            "risk": {**_DEFAULT_RISK},
            "signal": {**_DEFAULT_SIGNAL_SETTINGS},
            "backtest": {**_DEFAULT_BACKTEST_SETTINGS},
            "debug": {"show_debug": False},
        }

    @classmethod
    def load(cls) -> "ConfigStore":
        if cls.STATE_KEY not in st.session_state:
            st.session_state[cls.STATE_KEY] = cls._default_state()
        return cls(st.session_state[cls.STATE_KEY])

    # ------------------------------------------------------------------
    # Core configuration helpers
    # ------------------------------------------------------------------
    @property
    def token(self) -> str:
        return self._state.get("token", DEFAULT_TOKEN)

    def set_token(self, value: str) -> None:
        self._state["token"] = value.strip() or DEFAULT_TOKEN

    @property
    def symbol(self) -> str:
        token = self.token
        if ":" in token:
            return token.split(":", 1)[1]
        return token

    @property
    def timeframe(self) -> str:
        return self._state.get("timeframe", DEFAULT_TIMEFRAME)

    def set_timeframe(self, timeframe: str) -> None:
        tf = timeframe or DEFAULT_TIMEFRAME
        if tf != self._state.get("timeframe"):
            self._state["timeframe"] = tf
            # Ensure defaults exist for new timeframe
            overrides = self._state.setdefault("indicator_overrides", {})
            overrides.setdefault(tf, {})

    # ------------------------------------------------------------------
    # Date range handling
    # ------------------------------------------------------------------
    def _date_range(self) -> Dict[str, str]:
        return self._state.setdefault("date_range", {})

    @property
    def start_datetime(self) -> dt.datetime:
        default_start = _utc_now() - dt.timedelta(days=3)
        return _parse_iso(self._date_range().get("start", ""), default_start)

    @property
    def end_datetime(self) -> dt.datetime:
        default_end = _utc_now()
        return _parse_iso(self._date_range().get("end", ""), default_end)

    def set_date_range(self, start: dt.datetime, end: dt.datetime) -> None:
        if start >= end:
            start = end - dt.timedelta(hours=1)
        self._state["date_range"] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    # ------------------------------------------------------------------
    # Weights and normalization
    # ------------------------------------------------------------------
    def weights(self) -> Dict[str, float]:
        weights = self._state.setdefault("weights", {**_DEFAULT_WEIGHTS})
        # Ensure all keys exist
        for key, value in _DEFAULT_WEIGHTS.items():
            weights.setdefault(key, value)
        return {key: float(max(0.0, value)) for key, value in weights.items()}

    def update_weight(self, key: str, value: float) -> None:
        weights = self.weights()
        weights[key] = max(0.0, float(value))
        self._state["weights"] = weights

    def normalized_weights(self) -> Dict[str, float]:
        weights = self.weights()
        total = sum(weights.values())
        if total <= 0:
            weights = {**_DEFAULT_WEIGHTS}
            total = sum(weights.values())
        normalized = {key: value / total for key, value in weights.items()}
        normalized["composite"] = 0.0
        return normalized

    def weights_signature(self) -> str:
        weights = self.normalized_weights()
        return str(sorted(weights.items()))

    # ------------------------------------------------------------------
    # Indicator parameters
    # ------------------------------------------------------------------
    def _default_indicator_params(self, timeframe: Optional[str] = None) -> Dict[str, Any]:
        tf = timeframe or self.timeframe
        defaults = indicator_defaults_for(tf)
        macd_defaults = defaults.get("macd", {})
        rsi_defaults = defaults.get("rsi", {})
        atr_defaults = defaults.get("atr", {})
        atr_channel_defaults = defaults.get("atr_channels", {})
        bollinger_defaults = defaults.get("bollinger", {})
        volume_defaults = defaults.get("volume", {})
        structure_defaults = defaults.get("structure", {})
        composite_defaults = defaults.get("composite", {})
        multitimeframe_defaults = defaults.get("multitimeframe", {})
        return {
            "macd": {
                "fast": int(macd_defaults.get("fast", 12)),
                "slow": int(macd_defaults.get("slow", 26)),
                "signal": int(macd_defaults.get("signal", 9)),
            },
            "rsi": {
                "period": int(rsi_defaults.get("period", 14)),
                "overbought": float(rsi_defaults.get("overbought", 70.0)),
                "oversold": float(rsi_defaults.get("oversold", 30.0)),
            },
            "atr": {
                "period": int(atr_defaults.get("period", 14)),
                "mult": float(atr_defaults.get("mult", atr_defaults.get("mult_1x", 1.0))),
            },
            "atr_channels": {
                "period": int(atr_channel_defaults.get("period", atr_defaults.get("period", 14))),
                "mult_1x": float(atr_channel_defaults.get("mult_1x", atr_defaults.get("mult", 1.0))),
                "mult_2x": float(atr_channel_defaults.get("mult_2x", 2.0)),
                "mult_3x": float(atr_channel_defaults.get("mult_3x", 3.0)),
            },
            "bollinger": {
                "period": int(bollinger_defaults.get("period", 20)),
                "mult": float(bollinger_defaults.get("mult", bollinger_defaults.get("stddev", 2.0))),
                "stddev": float(bollinger_defaults.get("stddev", bollinger_defaults.get("mult", 2.0))),
                "source": str(bollinger_defaults.get("source", "close")),
            },
            "volume": {
                "ma_period": int(volume_defaults.get("ma_period", 20)),
                "cvd_atr_multiplier": float(volume_defaults.get("cvd_atr_multiplier", 0.75)),
                "delta_imbalance_threshold": float(volume_defaults.get("delta_imbalance_threshold", 1.2)),
                "vpvr_poc_share": float(volume_defaults.get("vpvr_poc_share", 0.04)),
                "smart_money_multiplier": float(volume_defaults.get("smart_money_multiplier", 1.5)),
            },
            "structure": {
                "lookback": int(structure_defaults.get("lookback", 24)),
                "swing_window": int(structure_defaults.get("swing_window", 5)),
                "trend_window": int(structure_defaults.get("trend_window", 12)),
                "min_sequence": int(structure_defaults.get("min_sequence", 5)),
                "atr_distance": float(structure_defaults.get("atr_distance", 1.0)),
            },
            "composite": {
                "buy_threshold": float(composite_defaults.get("buy_threshold", DEFAULT_SIGNAL_THRESHOLDS["buy"])),
                "sell_threshold": float(composite_defaults.get("sell_threshold", DEFAULT_SIGNAL_THRESHOLDS["sell"])),
                "confidence_floor": float(composite_defaults.get("confidence_floor", 0.3)),
                "confidence_ceiling": float(composite_defaults.get("confidence_ceiling", 0.9)),
                "min_confirmations": int(composite_defaults.get("min_confirmations", 3)),
            },
            "multitimeframe": {
                "trend_lookback": int(multitimeframe_defaults.get("trend_lookback", 14)),
                "alignment_weight": float(multitimeframe_defaults.get("alignment_weight", 0.4)),
                "agreement_weight": float(multitimeframe_defaults.get("agreement_weight", 0.3)),
                "force_weight": float(multitimeframe_defaults.get("force_weight", 0.3)),
            },
        }

    def get_indicator_params(self, timeframe: Optional[str] = None) -> Dict[str, Any]:
        tf = timeframe or self.timeframe
        defaults = self._default_indicator_params(tf)
        overrides = self._state.setdefault("indicator_overrides", {}).get(tf, {})
        params = {key: {**value} for key, value in defaults.items()}
        for indicator, settings in overrides.items():
            if indicator not in params:
                params[indicator] = {}
            for setting, setting_value in settings.items():
                params[indicator][setting] = setting_value
        return params

    def reset_indicator_params(self, timeframe: Optional[str] = None) -> None:
        tf = timeframe or self.timeframe
        overrides = self._state.setdefault("indicator_overrides", {})
        overrides.pop(tf, None)

    def update_indicator_param(self, indicator: str, field: str, value: float) -> None:
        tf = self.timeframe
        overrides = self._state.setdefault("indicator_overrides", {})
        tf_overrides = overrides.setdefault(tf, {})
        indicator_override = tf_overrides.setdefault(indicator, {})
        indicator_override[field] = value

    def indicator_signature(self, timeframe: Optional[str] = None) -> str:
        params = self.get_indicator_params(timeframe)
        return str(sorted((k, tuple(sorted(v.items()))) for k, v in params.items()))

    # ------------------------------------------------------------------
    # Debug / diagnostics
    # ------------------------------------------------------------------
    def debug_settings(self) -> Dict[str, Any]:
        settings = self._state.setdefault("debug", {"show_debug": False})
        settings.setdefault("show_debug", False)
        return settings

    def update_debug_setting(self, key: str, value: Any) -> None:
        settings = self.debug_settings()
        settings[key] = value
        self._state["debug"] = settings

    def is_debug_enabled(self) -> bool:
        return bool(self.debug_settings().get("show_debug", False))

    # ------------------------------------------------------------------
    # Risk & signal params
    # ------------------------------------------------------------------
    def risk_settings(self) -> Dict[str, float]:
        risk = self._state.setdefault("risk", {**_DEFAULT_RISK})
        for key, default_value in _DEFAULT_RISK.items():
            risk.setdefault(key, default_value)
        return risk

    def update_risk_setting(self, key: str, value: float) -> None:
        risk = self.risk_settings()
        risk[key] = max(0.0, float(value))
        self._state["risk"] = risk

    def signal_settings(self) -> Dict[str, Any]:
        settings = self._state.setdefault("signal", {**_DEFAULT_SIGNAL_SETTINGS})
        for key, default_value in _DEFAULT_SIGNAL_SETTINGS.items():
            settings.setdefault(key, default_value)
        return settings

    def update_signal_setting(self, key: str, value: float) -> None:
        settings = self.signal_settings()
        settings[key] = value
        self._state["signal"] = settings

    def signal_params_signature(self) -> str:
        settings = self.signal_settings()
        risk = self.risk_settings()
        buy_threshold = float(settings.get("buy_threshold", DEFAULT_SIGNAL_THRESHOLDS["buy"]))
        sell_threshold = float(settings.get("sell_threshold", DEFAULT_SIGNAL_THRESHOLDS["sell"]))
        params = {
            "max_risk_per_trade_pct": risk.get("max_risk_per_trade_pct", 0.02),
            "account_balance": risk.get("account_balance", 10_000.0),
            "max_position_size_pct": risk.get("max_position_size_pct", 0.05),
            "min_confirmations": settings.get("min_confirmations", 3),
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "signal_thresholds": (buy_threshold, sell_threshold),
            "min_confidence": settings.get("min_confidence", 0.6),
        }
        return str(sorted(params.items()))

    def build_signal_config(self) -> SignalConfig:
        settings = self.signal_settings()
        weights = self.normalized_weights()
        return SignalConfig(
            technical_weight=weights["technical"],
            sentiment_weight=weights["sentiment"],
            multitimeframe_weight=weights["multitimeframe"],
            volume_weight=weights["volume"],
            structure_weight=weights["market_structure"],
            composite_weight=weights.get("composite", 0.0),
            min_factors_confirm=int(settings.get("min_confirmations", 3)),
            buy_threshold=float(settings.get("buy_threshold", 0.65)),
            sell_threshold=float(settings.get("sell_threshold", 0.35)),
            min_confidence=float(settings.get("min_confidence", 0.6)),
        )

    def build_signal_params(self) -> Dict[str, Any]:
        risk = self.risk_settings()
        settings = self.signal_settings()
        thresholds = {
            "buy": float(settings.get("buy_threshold", DEFAULT_SIGNAL_THRESHOLDS["buy"])),
            "sell": float(settings.get("sell_threshold", DEFAULT_SIGNAL_THRESHOLDS["sell"])),
        }
        return {
            "max_risk_per_trade_pct": float(risk.get("max_risk_per_trade_pct", 0.02)),
            "account_balance": float(risk.get("account_balance", 10_000.0)),
            "max_position_size_pct": float(risk.get("max_position_size_pct", 0.05)),
            "signal_thresholds": thresholds,
        }

    # ------------------------------------------------------------------
    # Backtest settings
    # ------------------------------------------------------------------
    def backtest_settings(self) -> Dict[str, Any]:
        settings = self._state.setdefault("backtest", {**_DEFAULT_BACKTEST_SETTINGS})
        for key, default_value in _DEFAULT_BACKTEST_SETTINGS.items():
            settings.setdefault(key, default_value)
        return settings

    def update_backtest_setting(self, key: str, value: float) -> None:
        settings = self.backtest_settings()
        settings[key] = value
        self._state["backtest"] = settings

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def to_serializable(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "date_range": {
                "start": self.start_datetime.isoformat(),
                "end": self.end_datetime.isoformat(),
            },
            "weights": self.normalized_weights(),
            "indicator_params": self.get_indicator_params(),
            "risk": self.risk_settings(),
            "signal": self.signal_settings(),
            "backtest": self.backtest_settings(),
            "debug": self.debug_settings(),
        }

    def cache_payload(self) -> Dict[str, str]:
        return {
            "weights": self.weights_signature(),
            "indicator": self.indicator_signature(),
            "signal": self.signal_params_signature(),
            "debug": str(self.is_debug_enabled()),
        }

    def start_iso(self) -> str:
        return self.start_datetime.isoformat()

    def end_iso(self) -> str:
        return self.end_datetime.isoformat()


__all__ = ["ConfigStore"]
