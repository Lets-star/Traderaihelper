"""CME Gap detection for Bitcoin futures."""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone, timedelta

from .math_utils import Candle


def detect_cme_gaps(candles: List[Candle]) -> List[Dict[str, object]]:
    """
    Detect CME gaps in historical price data.
    
    CME futures markets close on weekends, creating potential gaps when
    the market reopens on Monday if the opening price differs significantly
    from Friday's close.
    
    Returns a list of detected gaps with their fill status.
    """
    gaps = []
    
    for i in range(1, len(candles)):
        prev_candle = candles[i - 1]
        curr_candle = candles[i]
        
        prev_time = datetime.fromtimestamp(prev_candle.close_time / 1000, tz=timezone.utc)
        curr_time = datetime.fromtimestamp(curr_candle.open_time / 1000, tz=timezone.utc)
        
        time_diff_hours = (curr_candle.open_time - prev_candle.close_time) / (1000 * 3600)
        
        if time_diff_hours > 24:
            gap_up = curr_candle.open > prev_candle.close
            gap_down = curr_candle.open < prev_candle.close
            
            if gap_up:
                gap_top = curr_candle.open
                gap_bottom = prev_candle.close
                gap_size = gap_top - gap_bottom
                gap_size_pct = (gap_size / prev_candle.close) * 100
                
                if gap_size_pct > 0.1:
                    is_filled = False
                    filled_at_index = None
                    for j in range(i + 1, len(candles)):
                        if candles[j].low <= gap_bottom:
                            is_filled = True
                            filled_at_index = j
                            break
                    
                    gaps.append({
                        "type": "gap_up",
                        "created_index": i,
                        "created_timestamp": curr_candle.open_time,
                        "created_time_iso": curr_time.isoformat(),
                        "gap_top": gap_top,
                        "gap_bottom": gap_bottom,
                        "gap_size": gap_size,
                        "gap_size_pct": gap_size_pct,
                        "is_filled": is_filled,
                        "filled_at_index": filled_at_index,
                        "filled_at_timestamp": candles[filled_at_index].close_time if filled_at_index else None,
                    })
            
            elif gap_down:
                gap_top = prev_candle.close
                gap_bottom = curr_candle.open
                gap_size = gap_top - gap_bottom
                gap_size_pct = (gap_size / prev_candle.close) * 100
                
                if gap_size_pct > 0.1:
                    is_filled = False
                    filled_at_index = None
                    for j in range(i + 1, len(candles)):
                        if candles[j].high >= gap_top:
                            is_filled = True
                            filled_at_index = j
                            break
                    
                    gaps.append({
                        "type": "gap_down",
                        "created_index": i,
                        "created_timestamp": curr_candle.open_time,
                        "created_time_iso": curr_time.isoformat(),
                        "gap_top": gap_top,
                        "gap_bottom": gap_bottom,
                        "gap_size": gap_size,
                        "gap_size_pct": gap_size_pct,
                        "is_filled": is_filled,
                        "filled_at_index": filled_at_index,
                        "filled_at_timestamp": candles[filled_at_index].close_time if filled_at_index else None,
                    })
    
    return gaps


def get_nearest_cme_gaps(candles: List[Candle], current_price: float, max_gaps: int = 5) -> Dict[str, object]:
    """
    Get the nearest unfilled CME gaps above and below current price.
    
    Args:
        candles: Historical price data
        current_price: Current market price
        max_gaps: Maximum number of gaps to return in each direction
        
    Returns:
        Dictionary containing nearest gaps above and below current price
    """
    all_gaps = detect_cme_gaps(candles)
    
    unfilled_gaps = [gap for gap in all_gaps if not gap["is_filled"]]
    
    gaps_above = []
    gaps_below = []
    
    for gap in unfilled_gaps:
        gap_bottom = gap["gap_bottom"]
        gap_top = gap["gap_top"]
        gap_mid = (gap_top + gap_bottom) / 2
        
        if gap_bottom > current_price:
            distance_pct = ((gap_bottom - current_price) / current_price) * 100
            gaps_above.append({
                **gap,
                "distance_to_price": gap_bottom - current_price,
                "distance_pct": distance_pct,
            })
        elif gap_top < current_price:
            distance_pct = ((current_price - gap_top) / current_price) * 100
            gaps_below.append({
                **gap,
                "distance_to_price": current_price - gap_top,
                "distance_pct": distance_pct,
            })
        else:
            distance_pct = 0.0
            gaps_below.append({
                **gap,
                "distance_to_price": 0.0,
                "distance_pct": distance_pct,
                "currently_inside": True,
            })
    
    gaps_above.sort(key=lambda x: x["distance_to_price"])
    gaps_below.sort(key=lambda x: x["distance_to_price"])
    
    return {
        "total_unfilled_gaps": len(unfilled_gaps),
        "total_gaps_above": len(gaps_above),
        "total_gaps_below": len(gaps_below),
        "nearest_gaps_above": gaps_above[:max_gaps],
        "nearest_gaps_below": gaps_below[:max_gaps],
        "all_unfilled_gaps": len(unfilled_gaps),
    }
