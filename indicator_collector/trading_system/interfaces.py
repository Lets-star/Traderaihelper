"""Trading system core interfaces and dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Protocol

JsonDict = Dict[str, Any]


def _copy_dict(mapping: Optional[Mapping[str, Any]]) -> JsonDict:
    return dict(mapping) if mapping else {}


@dataclass
class FactorScore:
    """Individual factor score contributing to a trading decision."""

    factor_name: str
    score: float
    weight: float = 1.0
    description: Optional[str] = None
    emoji: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "factor_name": self.factor_name,
            "score": self.score,
            "weight": self.weight,
            "description": self.description,
            "emoji": self.emoji,
        }
        if self.metadata:
            data["metadata"] = _copy_dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FactorScore":
        return cls(
            factor_name=str(data.get("factor_name", "")),
            score=float(data.get("score", 0.0)),
            weight=float(data.get("weight", 1.0)),
            description=data.get("description"),
            emoji=data.get("emoji"),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class PositionPlan:
    """Position sizing and risk management plan."""

    entry_price: float
    stop_loss: Optional[float] = None
    take_profit_levels: List[float] = field(default_factory=list)
    position_size_usd: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    max_risk_pct: Optional[float] = None
    leverage: Optional[float] = None
    direction: Optional[Literal["long", "short", "flat"]] = None
    notes: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_levels": list(self.take_profit_levels),
            "position_size_usd": self.position_size_usd,
            "risk_reward_ratio": self.risk_reward_ratio,
            "max_risk_pct": self.max_risk_pct,
            "leverage": self.leverage,
            "direction": self.direction,
            "notes": self.notes,
        }
        if self.metadata:
            data["metadata"] = _copy_dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PositionPlan":
        return cls(
            entry_price=float(data.get("entry_price", 0.0)),
            stop_loss=data.get("stop_loss"),
            take_profit_levels=list(data.get("take_profit_levels", []) or []),
            position_size_usd=data.get("position_size_usd"),
            risk_reward_ratio=data.get("risk_reward_ratio"),
            max_risk_pct=data.get("max_risk_pct"),
            leverage=data.get("leverage"),
            direction=data.get("direction"),
            notes=data.get("notes"),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class SignalExplanation:
    """Detailed explanation for a trading signal."""

    primary_reason: str
    supporting_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    market_context: Optional[str] = None
    notes: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "primary_reason": self.primary_reason,
            "supporting_factors": list(self.supporting_factors),
            "risk_factors": list(self.risk_factors),
            "market_context": self.market_context,
            "notes": self.notes,
        }
        if self.metadata:
            data["metadata"] = _copy_dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalExplanation":
        return cls(
            primary_reason=str(data.get("primary_reason", "")),
            supporting_factors=list(data.get("supporting_factors", []) or []),
            risk_factors=list(data.get("risk_factors", []) or []),
            market_context=data.get("market_context"),
            notes=data.get("notes"),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class OptimizationStats:
    """Statistics generated from optimization or backtesting."""

    backtest_win_rate: Optional[float] = None
    avg_profit_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    total_signals: int = 0
    profitable_signals: int = 0
    losing_signals: int = 0
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "backtest_win_rate": self.backtest_win_rate,
            "avg_profit_pct": self.avg_profit_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "total_signals": self.total_signals,
            "profitable_signals": self.profitable_signals,
            "losing_signals": self.losing_signals,
        }
        if self.metadata:
            data["metadata"] = _copy_dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizationStats":
        return cls(
            backtest_win_rate=data.get("backtest_win_rate"),
            avg_profit_pct=data.get("avg_profit_pct"),
            avg_loss_pct=data.get("avg_loss_pct"),
            sharpe_ratio=data.get("sharpe_ratio"),
            total_signals=int(data.get("total_signals", 0) or 0),
            profitable_signals=int(data.get("profitable_signals", 0) or 0),
            losing_signals=int(data.get("losing_signals", 0) or 0),
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class TradingSignalPayload:
    """Complete trading signal with analysis and planning."""

    signal_type: Literal["BUY", "SELL", "NEUTRAL"]
    confidence: float
    timestamp: int
    symbol: str
    timeframe: str
    factors: List[FactorScore] = field(default_factory=list)
    position_plan: Optional[PositionPlan] = None
    explanation: Optional[SignalExplanation] = None
    optimization_stats: Optional[OptimizationStats] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "signal_type": self.signal_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "factors": [factor.to_dict() for factor in self.factors],
            "position_plan": self.position_plan.to_dict() if self.position_plan else None,
            "explanation": self.explanation.to_dict() if self.explanation else None,
            "optimization_stats": self.optimization_stats.to_dict() if self.optimization_stats else None,
        }
        if self.metadata:
            data["metadata"] = _copy_dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradingSignalPayload":
        return cls(
            signal_type=data.get("signal_type", "NEUTRAL"),
            confidence=float(data.get("confidence", 0.0)),
            timestamp=int(data.get("timestamp", 0) or 0),
            symbol=str(data.get("symbol", "")),
            timeframe=str(data.get("timeframe", "")),
            factors=[FactorScore.from_dict(item) for item in data.get("factors", [])],
            position_plan=PositionPlan.from_dict(data["position_plan"]) if data.get("position_plan") else None,
            explanation=SignalExplanation.from_dict(data["explanation"]) if data.get("explanation") else None,
            optimization_stats=OptimizationStats.from_dict(data["optimization_stats"]) if data.get("optimization_stats") else None,
            metadata=_copy_dict(data.get("metadata")),
        )


@dataclass
class AnalyzerContext:
    """Context information required by analyzers."""

    symbol: str
    timeframe: str
    timestamp: int
    current_price: float
    ohlcv: JsonDict
    indicators: JsonDict
    multi_timeframe: JsonDict = field(default_factory=dict)
    success_rates: JsonDict = field(default_factory=dict)
    pnl_stats: JsonDict = field(default_factory=dict)
    volume_analysis: JsonDict = field(default_factory=dict)
    market_structure: JsonDict = field(default_factory=dict)
    zones: List[JsonDict] = field(default_factory=list)
    historical_signals: List[JsonDict] = field(default_factory=list)
    advanced_metrics: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    extras: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "current_price": self.current_price,
            "ohlcv": _copy_dict(self.ohlcv),
            "indicators": _copy_dict(self.indicators),
            "multi_timeframe": _copy_dict(self.multi_timeframe),
            "success_rates": _copy_dict(self.success_rates),
            "pnl_stats": _copy_dict(self.pnl_stats),
            "volume_analysis": _copy_dict(self.volume_analysis),
            "market_structure": _copy_dict(self.market_structure),
            "zones": [dict(zone) for zone in self.zones],
            "historical_signals": [dict(signal) for signal in self.historical_signals],
            "advanced_metrics": _copy_dict(self.advanced_metrics),
            "metadata": _copy_dict(self.metadata),
            "extras": _copy_dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalyzerContext":
        return cls(
            symbol=str(data.get("symbol", "")),
            timeframe=str(data.get("timeframe", "")),
            timestamp=int(data.get("timestamp", 0) or 0),
            current_price=float(data.get("current_price", 0.0)),
            ohlcv=_copy_dict(data.get("ohlcv")),
            indicators=_copy_dict(data.get("indicators")),
            multi_timeframe=_copy_dict(data.get("multi_timeframe")),
            success_rates=_copy_dict(data.get("success_rates")),
            pnl_stats=_copy_dict(data.get("pnl_stats")),
            volume_analysis=_copy_dict(data.get("volume_analysis")),
            market_structure=_copy_dict(data.get("market_structure")),
            zones=[dict(item) for item in data.get("zones", [])],
            historical_signals=[dict(item) for item in data.get("historical_signals", [])],
            advanced_metrics=_copy_dict(data.get("advanced_metrics")),
            metadata=_copy_dict(data.get("metadata")),
            extras=_copy_dict(data.get("extras")),
        )


class TradingAnalyzer(Protocol):
    """Protocol definition for trading system analyzers."""

    def analyze(self, context: AnalyzerContext) -> TradingSignalPayload:
        """Produce a trading signal for the provided context."""

    def optimize(self, history: Iterable[AnalyzerContext]) -> Optional[OptimizationStats]:
        """Optionally optimize the analyzer using historical contexts."""


_COLLECTOR_INDICATOR_KEYS = (
    "trend_strength",
    "pattern_score",
    "market_sentiment",
    "structure_state",
    "structure_event",
    "volume_confirmed",
    "volume_ratio",
    "volume_confidence",
    "confluence_score",
    "confluence_bias",
    "confluence_bullish",
    "confluence_bearish",
    "signal",
    "rsi",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    "atr",
    "atr_channels",
    "vwap",
    "sma_fast",
    "sma_slow",
    "rsi_divergence",
    "macd_divergence",
)


def parse_collector_payload(collector_payload: Mapping[str, Any]) -> AnalyzerContext:
    """Convert a collector payload dictionary into an AnalyzerContext instance."""

    metadata = _copy_dict(collector_payload.get("metadata"))
    latest = _copy_dict(collector_payload.get("latest"))
    advanced = _copy_dict(collector_payload.get("advanced"))

    ohlcv = {
        "open": float(latest.get("open", 0.0) or 0.0),
        "high": float(latest.get("high", 0.0) or 0.0),
        "low": float(latest.get("low", 0.0) or 0.0),
        "close": float(latest.get("close", 0.0) or 0.0),
        "volume": float(latest.get("volume", 0.0) or 0.0),
    }

    indicators: JsonDict = {key: latest.get(key) for key in _COLLECTOR_INDICATOR_KEYS if key in latest}

    if "atr_channels" in indicators and isinstance(indicators["atr_channels"], Mapping):
        indicators["atr_channels"] = dict(indicators["atr_channels"])

    extras = {
        "astrology": collector_payload.get("astrology"),
        "definitions": collector_payload.get("definitions"),
        "orderbook": collector_payload.get("orderbook"),
        "cme_gaps": latest.get("cme_gaps"),
        "trade_plan": advanced.get("trade_plan"),
        "composite_indicators": advanced.get("composite_indicators"),
        "market_context": advanced.get("market_context"),
        "signal_analysis": advanced.get("signal_analysis"),
        "candles": collector_payload.get("candles"),
    }
    extras = {key: value for key, value in extras.items() if value not in (None, {}, [])}

    return AnalyzerContext(
        symbol=str(metadata.get("symbol", "")),
        timeframe=str(metadata.get("timeframe", "")),
        timestamp=int(latest.get("timestamp", 0) or 0),
        current_price=float(latest.get("close", 0.0) or 0.0),
        ohlcv=ohlcv,
        indicators=indicators,
        multi_timeframe=_copy_dict(collector_payload.get("multi_timeframe")),
        success_rates=_copy_dict(collector_payload.get("success_rates")),
        pnl_stats=_copy_dict(collector_payload.get("pnl_stats")),
        volume_analysis=advanced.get("volume_analysis", {}),
        market_structure=advanced.get("market_structure", {}),
        zones=[dict(item) for item in collector_payload.get("zones", [])],
        historical_signals=[dict(item) for item in collector_payload.get("signals", [])],
        advanced_metrics=advanced,
        metadata=metadata,
        extras=extras,
    )


def serialize_signal_payload(signal: TradingSignalPayload) -> JsonDict:
    """Serialize a TradingSignalPayload into a JSON-compatible dictionary."""

    return signal.to_dict()


def deserialize_signal_payload(data: Mapping[str, Any]) -> TradingSignalPayload:
    """Deserialize a JSON-compatible dictionary into TradingSignalPayload."""

    return TradingSignalPayload.from_dict(data)
