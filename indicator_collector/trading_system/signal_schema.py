"""Signal schema definitions and validation for automated trading signals.

This module defines the expected JSON structure for automated signals output
and provides validation functions to ensure schema compliance.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class TradingSignalSchema(BaseModel):
    """Schema for automated trading signals output."""
    
    signal: str = Field(..., description="Trading signal: BUY, SELL, or HOLD")
    confidence: int = Field(..., ge=1, le=10, description="Confidence level from 1 to 10")
    entries: List[float] = Field(..., min_length=1, description="Entry price levels")
    stop_loss: float = Field(..., gt=0, description="Stop loss price level")
    take_profits: Dict[str, float] = Field(..., description="Take profit levels with tp1, tp2, tp3 keys")
    position_size_pct: float = Field(..., ge=0, le=100, description="Position size as percentage of portfolio")
    holding_period: str = Field(..., description="Expected holding period: short, medium, or long")
    rationale: List[str] = Field(..., min_length=1, description="List of rationale points for the signal")
    cancel_conditions: List[str] = Field(default_factory=list, description="Conditions that would cancel the signal")
    weights: Dict[str, float] = Field(..., description="Signal component weights")
    timeframe: str = Field(..., description="Trading timeframe used for analysis")
    
    @field_validator('signal')
    @classmethod
    def validate_signal(cls, v):
        """Validate signal value."""
        allowed_signals = {'BUY', 'SELL', 'HOLD'}
        if v not in allowed_signals:
            raise ValueError(f"Signal must be one of {allowed_signals}")
        return v
    
    @field_validator('holding_period')
    @classmethod
    def validate_holding_period(cls, v):
        """Validate holding period value."""
        allowed_periods = {'short', 'medium', 'long'}
        if v not in allowed_periods:
            raise ValueError(f"Holding period must be one of {allowed_periods}")
        return v
    
    @field_validator('take_profits')
    @classmethod
    def validate_take_profits(cls, v):
        """Validate take profits structure."""
        required_keys = {'tp1', 'tp2', 'tp3'}
        if not set(v.keys()).issuperset(required_keys):
            raise ValueError(f"Take profits must contain keys: {required_keys}")
        
        for key, value in v.items():
            if key.startswith('tp') and value <= 0:
                raise ValueError(f"Take profit {key} must be positive")
        
        return v
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        """Validate weights sum to approximately 1.0."""
        if not v:
            raise ValueError("Weights cannot be empty")
        
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to approximately 1.0, got {total}")
        
        return v


def validate_signal_json(signal_data: Union[Dict[str, Any], str]) -> TradingSignalSchema:
    """
    Validate signal data against the schema.
    
    Args:
        signal_data: Signal data as dictionary or JSON string
        
    Returns:
        Validated TradingSignalSchema object
        
    Raises:
        ValueError: If validation fails
        json.JSONDecodeError: If JSON string is malformed
    """
    if isinstance(signal_data, str):
        try:
            signal_data = json.loads(signal_data)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON signal data: {e.msg}", e.doc, e.pos)
    
    try:
        return TradingSignalSchema(**signal_data)
    except Exception as e:
        raise ValueError(f"Signal validation failed: {e}")


def create_signal_schema_validator():
    """
    Create a JSON schema validator for signal data.
    
    Returns:
        Dictionary containing the JSON schema
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "signal", "confidence", "entries", "stop_loss", 
            "take_profits", "position_size_pct", "holding_period",
            "rationale", "weights", "timeframe"
        ],
        "properties": {
            "signal": {
                "type": "string",
                "enum": ["BUY", "SELL", "HOLD"],
                "description": "Trading signal direction"
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Confidence level (1-10)"
            },
            "entries": {
                "type": "array",
                "items": {"type": "number", "minimum": 0},
                "minItems": 1,
                "description": "Entry price levels"
            },
            "stop_loss": {
                "type": "number",
                "minimum": 0,
                "description": "Stop loss price level"
            },
            "take_profits": {
                "type": "object",
                "required": ["tp1", "tp2", "tp3"],
                "properties": {
                    "tp1": {"type": "number", "minimum": 0},
                    "tp2": {"type": "number", "minimum": 0},
                    "tp3": {"type": "number", "minimum": 0}
                },
                "description": "Take profit levels"
            },
            "position_size_pct": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Position size as percentage"
            },
            "holding_period": {
                "type": "string",
                "enum": ["short", "medium", "long"],
                "description": "Expected holding period"
            },
            "rationale": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Rationale for the signal"
            },
            "cancel_conditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions that would cancel the signal"
            },
            "weights": {
                "type": "object",
                "description": "Component weights (should sum to 1.0)"
            },
            "timeframe": {
                "type": "string",
                "description": "Trading timeframe used"
            }
        }
    }


def is_valid_signal_structure(data: Union[Dict[str, Any], str]) -> bool:
    """
    Quick check if data has the basic structure of a trading signal.
    
    Args:
        data: Data to check
        
    Returns:
        True if structure looks valid, False otherwise
    """
    try:
        if isinstance(data, str):
            data = json.loads(data)
        
        required_fields = [
            "signal", "confidence", "entries", "stop_loss",
            "take_profits", "position_size_pct", "holding_period",
            "rationale", "weights", "timeframe"
        ]
        
        return all(field in data for field in required_fields)
    except:
        return False