"""
TypeGuard functions for runtime type checking.

TypeGuard allows narrowing types based on runtime checks,
enabling safer code when working with untyped data (API responses, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from typing import TypeGuard

from trader_types.protocols import StreamlitComponent
from trader_types.typed_dict import (
    SignalPayload,
    KlineData,
    ExecutionResult,
    PositionData,
    OrderPayload,
)


def is_valid_signal(payload: dict) -> TypeGuard[SignalPayload]:
    """
    Check if dict is a valid SignalPayload.
    
    Args:
        payload: Dictionary to validate
        
    Returns:
        True if payload has all required signal fields
    """
    if not isinstance(payload, dict):
        return False
    
    # Required fields
    required = ["signal_id", "symbol", "direction", "entry_price"]
    for field in required:
        if field not in payload:
            return False
    
    # Validate types
    if not isinstance(payload.get("signal_id"), str):
        return False
    if not isinstance(payload.get("symbol"), str):
        return False
    if not isinstance(payload.get("direction"), str):
        return False
    if payload.get("direction") not in ("LONG", "SHORT"):
        return False
    
    try:
        float(payload.get("entry_price", 0))
    except (TypeError, ValueError):
        return False
    
    return True


def is_kline_data(data: dict) -> TypeGuard[KlineData]:
    """
    Check if dict is valid KlineData.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if data has all required kline fields
    """
    if not isinstance(data, dict):
        return False
    
    # Required fields
    required = ["ts", "open", "high", "low", "close", "volume"]
    for field in required:
        if field not in data:
            return False
    
    # Validate numeric fields
    numeric_fields = ["open", "high", "low", "close", "volume"]
    for field in numeric_fields:
        try:
            float(data[field])
        except (TypeError, ValueError):
            return False
    
    # Validate timestamp
    try:
        int(data["ts"])
    except (TypeError, ValueError):
        return False
    
    return True


def is_execution_result(data: dict) -> TypeGuard[ExecutionResult]:
    """
    Check if dict is valid ExecutionResult.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if data has all required execution result fields
    """
    if not isinstance(data, dict):
        return False
    
    # Required fields
    if "status" not in data:
        return False
    
    # Validate status is a string
    if not isinstance(data.get("status"), str):
        return False
    
    return True


def is_position_data(data: dict) -> TypeGuard[PositionData]:
    """
    Check if dict is valid PositionData.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if data has all required position fields
    """
    if not isinstance(data, dict):
        return False
    
    # Required fields
    required = ["symbol", "side", "size", "entry_price"]
    for field in required:
        if field not in data:
            return False
    
    # Validate types
    if not isinstance(data.get("symbol"), str):
        return False
    if not isinstance(data.get("side"), str):
        return False
    
    # Validate numeric fields
    numeric_fields = ["size", "entry_price"]
    for field in numeric_fields:
        try:
            float(data[field])
        except (TypeError, ValueError):
            return False
    
    return True


def is_order_payload(data: dict) -> TypeGuard[OrderPayload]:
    """
    Check if dict is valid OrderPayload.
    
    Args:
        data: Dictionary to validate
        
    Returns:
        True if data has all required order fields
    """
    if not isinstance(data, dict):
        return False
    
    # Required fields
    required = ["symbol", "side", "order_type", "qty"]
    for field in required:
        if field not in data:
            return False
    
    # Validate types
    if not isinstance(data.get("symbol"), str):
        return False
    if not isinstance(data.get("side"), str):
        return False
    if not isinstance(data.get("order_type"), str):
        return False
    if not isinstance(data.get("qty"), (str, int, float)):
        return False
    
    return True


def is_streamlit_component(obj: Any) -> TypeGuard[StreamlitComponent]:
    """
    Check if object implements StreamlitComponent protocol.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object has required Streamlit component methods
    """
    if obj is None:
        return False
    
    required_methods = [
        "number_input",
        "selectbox",
        "button",
        "checkbox",
        "text_input",
    ]
    
    for method in required_methods:
        if not hasattr(obj, method):
            return False
        if not callable(getattr(obj, method)):
            return False
    
    return True


def is_list_of_signals(data: list) -> TypeGuard[List[SignalPayload]]:
    """
    Check if list contains only valid SignalPayload objects.
    
    Args:
        data: List to validate
        
    Returns:
        True if all items are valid SignalPayload
    """
    if not isinstance(data, list):
        return False
    
    return all(is_valid_signal(item) for item in data if isinstance(item, dict))


def is_numeric(value: Any) -> TypeGuard[Union[int, float]]:
    """
    Check if value is numeric (int or float).
    
    Args:
        value: Value to check
        
    Returns:
        True if value is int or float (not bool)
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def is_non_empty_string(value: Any) -> TypeGuard[str]:
    """
    Check if value is a non-empty string.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a non-empty string
    """
    return isinstance(value, str) and len(value.strip()) > 0


def is_valid_symbol(symbol: Any) -> TypeGuard[str]:
    """
    Check if symbol is a valid trading symbol.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        True if symbol is a valid format (e.g., BTCUSDT)
    """
    if not isinstance(symbol, str):
        return False
    
    # Basic validation: uppercase letters and numbers, at least 3 chars
    if len(symbol) < 3:
        return False
    
    # Check for valid characters
    if not symbol.isalnum():
        return False
    
    return True


def is_valid_leverage(leverage: Any) -> TypeGuard[float]:
    """
    Check if leverage is a valid value (1-125).
    
    Args:
        leverage: Leverage value to validate
        
    Returns:
        True if leverage is within valid range
    """
    try:
        lev = float(leverage)
        return 1 <= lev <= 125
    except (TypeError, ValueError):
        return False


def is_valid_confidence(confidence: Any) -> TypeGuard[float]:
    """
    Check if confidence is a valid value (0-1).
    
    Args:
        confidence: Confidence value to validate
        
    Returns:
        True if confidence is within valid range
    """
    try:
        conf = float(confidence)
        return 0 <= conf <= 1
    except (TypeError, ValueError):
        return False


def has_required_keys(data: dict, required: List[str]) -> bool:
    """
    Check if dict has all required keys.
    
    Args:
        data: Dictionary to check
        required: List of required keys
        
    Returns:
        True if all required keys are present
    """
    if not isinstance(data, dict):
        return False
    return all(key in data for key in required)


def is_dict_with_keys(data: Any, keys: List[str]) -> TypeGuard[Dict[str, Any]]:
    """
    Check if value is a dict with specific keys.
    
    Args:
        data: Value to check
        keys: Required keys
        
    Returns:
        True if data is dict with all specified keys
    """
    if not isinstance(data, dict):
        return False
    return all(key in data for key in keys)
