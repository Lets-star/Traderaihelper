"""Signal schema definitions and validation for automated trading signals.

This module defines the expected JSON structure for automated signals output
and provides validation functions to ensure schema compliance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ALLOWED_SIGNALS = {"BUY", "SELL", "HOLD"}
_ALLOWED_HOLDING_PERIODS = {"short", "medium", "long"}
_ACTIONABLE_SIGNALS = {"BUY", "SELL"}


class TradingSignalSchema(BaseModel):
    """Schema for automated trading signals output."""

    model_config = ConfigDict(extra="allow")

    signal: str = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    confidence: int = Field(..., ge=1, le=10, description="Confidence level from 1 to 10")
    entries: List[float] = Field(default_factory=list, description="Entry price levels")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price level")
    take_profits: Dict[str, float] = Field(default_factory=dict, description="Take profit levels with tp1, tp2, tp3 keys")
    position_size_pct: Optional[float] = Field(default=None, ge=0, le=100, description="Position size as percentage of portfolio")
    holding_period: str = Field(..., description="Expected holding period: short, medium, or long")
    rationale: List[str] = Field(default_factory=list, description="List of rationale points for the signal")
    cancel_conditions: List[str] = Field(default_factory=list, description="Conditions that would cancel the signal")
    weights: Dict[str, float] = Field(default_factory=dict, description="Signal component weights")
    timeframe: str = Field(..., description="Trading timeframe used for analysis")

    # Optional diagnostic data
    factors: Optional[List[Dict[str, Any]]] = Field(default=None, description="Factor breakdown contributing to the signal")
    position_plan: Optional[Dict[str, Any]] = Field(default=None, description="Normalized position plan data")
    explanation: Optional[Dict[str, Any]] = Field(default=None, description="Detailed explanation metadata")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Auxiliary metadata for the signal")
    debug: Optional[Dict[str, Any]] = Field(default=None, description="Debug payload when verbose mode enabled")
    holding_horizon_bars: Optional[int] = Field(default=None, description="Estimated holding horizon in bars")
    cancellation_reasons: Optional[List[str]] = Field(default=None, description="Reasons the signal may be cancelled or downgraded")

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, value: str) -> str:
        if value not in _ALLOWED_SIGNALS:
            raise ValueError(f"Signal must be one of {_ALLOWED_SIGNALS}")
        return value

    @field_validator("confidence")
    @classmethod
    def normalize_confidence(cls, value: int) -> int:
        return int(value)

    @field_validator("entries", mode="before")
    @classmethod
    def normalize_entries(cls, value: Optional[Any]) -> List[float]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            return [float(value)]
        normalized: List[float] = []
        for item in value:
            if item is None:
                continue
            normalized.append(float(item))
        return normalized

    @field_validator("take_profits", mode="before")
    @classmethod
    def normalize_take_profits(cls, value: Optional[Any]) -> Dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {key: float(val) for key, val in value.items() if val is not None}
        raise TypeError("take_profits must be a mapping of levels")

    @field_validator("rationale", mode="before")
    @classmethod
    def normalize_rationale(cls, value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return [str(value)]

    @field_validator("cancel_conditions", mode="before")
    @classmethod
    def normalize_cancel_conditions(cls, value: Optional[Any]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return [str(value)]

    @field_validator("cancellation_reasons", mode="before")
    @classmethod
    def normalize_cancellation_reasons(cls, value: Optional[Any]) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            reasons = [str(item) for item in value if item]
            return reasons or None
        return [str(value)]

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: Dict[str, float]) -> Dict[str, float]:
        if not value:
            raise ValueError("Weights cannot be empty")
        total = sum(float(v) for v in value.values())
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to approximately 1.0, got {total}")
        return {k: float(v) for k, v in value.items()}

    @field_validator("holding_period")
    @classmethod
    def validate_holding_period(cls, value: str) -> str:
        if value not in _ALLOWED_HOLDING_PERIODS:
            raise ValueError(f"Holding period must be one of {_ALLOWED_HOLDING_PERIODS}")
        return value

    @field_validator("position_size_pct")
    @classmethod
    def validate_position_size_pct(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if value < 0 or value > 100:
            raise ValueError("position_size_pct must be between 0 and 100")
        return float(value)

    @field_validator("stop_loss")
    @classmethod
    def validate_stop_loss(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("stop_loss must be positive when provided")
        return float(value)

    @field_validator("take_profits")
    @classmethod
    def ensure_positive_tps(cls, value: Dict[str, float]) -> Dict[str, float]:
        for key, tp_value in value.items():
            if tp_value is not None and tp_value <= 0:
                raise ValueError(f"Take profit {key} must be positive")
        return value

    @model_validator(mode="after")
    def validate_consistency(self) -> "TradingSignalSchema":
        actionable = self.signal in _ACTIONABLE_SIGNALS
        if actionable:
            if not self.entries:
                raise ValueError("Actionable signals must include at least one entry level")
            required_keys = {"tp1", "tp2", "tp3"}
            if not required_keys.issubset(self.take_profits.keys()):
                raise ValueError("Actionable signals require tp1, tp2, and tp3 take profit levels")
            if self.stop_loss is None:
                raise ValueError("Actionable signals require a stop loss level")
            if self.position_size_pct is None:
                raise ValueError("Actionable signals require position_size_pct")
            if not self.rationale:
                raise ValueError("Actionable signals require at least one rationale entry")
        else:
            # Non-actionable signals may omit levels; ensure structure is coherent
            self.entries = list(self.entries)
        return self


def validate_signal_json(signal_data: Union[Dict[str, Any], str]) -> TradingSignalSchema:
    """Validate signal data against the schema."""
    if isinstance(signal_data, str):
        try:
            signal_data = json.loads(signal_data)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(f"Invalid JSON signal data: {exc.msg}", exc.doc, exc.pos)

    try:
        return TradingSignalSchema(**signal_data)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Signal validation failed: {exc}") from exc


def create_signal_schema_validator() -> Dict[str, Any]:
    """Create a JSON schema validator for signal data."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "signal",
            "confidence",
            "entries",
            "stop_loss",
            "take_profits",
            "position_size_pct",
            "holding_period",
            "rationale",
            "weights",
            "timeframe",
        ],
        "properties": {
            "signal": {
                "type": "string",
                "enum": ["BUY", "SELL", "HOLD"],
                "description": "Trading signal direction",
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Confidence level (1-10)",
            },
            "entries": {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
                "description": "Entry price levels",
            },
            "stop_loss": {
                "type": ["number", "null"],
                "minimum": 0,
                "description": "Stop loss price level",
            },
            "take_profits": {
                "type": "object",
                "properties": {
                    "tp1": {"type": "number", "minimum": 0},
                    "tp2": {"type": "number", "minimum": 0},
                    "tp3": {"type": "number", "minimum": 0},
                },
                "description": "Take profit levels",
            },
            "position_size_pct": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 100,
                "description": "Position size as percentage",
            },
            "holding_period": {
                "type": "string",
                "enum": ["short", "medium", "long"],
                "description": "Expected holding period",
            },
            "rationale": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Rationale for the signal",
            },
            "cancel_conditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions that would cancel the signal",
            },
            "weights": {
                "type": "object",
                "description": "Component weights (should sum to 1.0)",
            },
            "timeframe": {
                "type": "string",
                "description": "Trading timeframe used",
            },
        },
    }


def is_valid_signal_structure(data: Union[Dict[str, Any], str]) -> bool:
    """Quick structural check for trading signal data."""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        required_fields = [
            "signal",
            "confidence",
            "entries",
            "stop_loss",
            "take_profits",
            "position_size_pct",
            "holding_period",
            "rationale",
            "weights",
            "timeframe",
        ]
        return all(field in data for field in required_fields)
    except Exception:  # pragma: no cover - defensive
        return False
