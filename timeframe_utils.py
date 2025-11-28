# timeframe_utils.py

from typing import Dict

TIMEFRAME_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "3h": 10_800_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

def map_tf_to_ms(tf: str) -> int:
    """Convert timeframe string to milliseconds."""
    return TIMEFRAME_TO_MS.get(tf, 60_000)

def get_boundary(now_ms: int, tf_ms: int, tolerance_ms: int = 1500) -> int:
    """Calculate last closed bar timestamp."""
    return ((now_ms - tolerance_ms) // tf_ms) * tf_ms

def floor_closed_bar_local(now_ms: int, tf_ms: int, tol_ms: int = 60_000) -> int:
    """
    Calculate the close_time of the last closed bar boundary.
    
    This returns the close_time (not open_time) of the last closed candle.
    For a candle with open_time T, its close_time is T + tf_ms.
    
    Args:
        now_ms: Current time in milliseconds (UTC)
        tf_ms: Timeframe interval in milliseconds
        tol_ms: Tolerance in milliseconds (default 60s)
        
    Returns:
        close_time of the last closed bar in milliseconds (UTC)
    """
    if tf_ms <= 0:
        return now_ms
    
    effective_now = max(now_ms - tol_ms, 0)
    last_closed_close_ms = (effective_now // tf_ms) * tf_ms
    return last_closed_close_ms
