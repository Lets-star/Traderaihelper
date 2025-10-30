from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Sequence

from .math_utils import Candle


class TimeframeSeries:
    """Provides fast lookup for candles by timestamp."""

    def __init__(self, candles: Sequence[Candle]):
        if not candles:
            raise ValueError("TimeframeSeries requires at least one candle")
        ordered = sorted(candles, key=lambda c: c.close_time)
        self._candles: List[Candle] = list(ordered)
        self._close_times: List[int] = [c.close_time for c in self._candles]

    def candle_at(self, timestamp: int) -> Candle:
        """Return the latest candle whose close time is <= timestamp."""
        idx = bisect.bisect_right(self._close_times, timestamp) - 1
        if idx < 0:
            idx = 0
        return self._candles[idx]

    def latest(self) -> Candle:
        return self._candles[-1]

    @property
    def candles(self) -> Sequence[Candle]:
        return self._candles


@dataclass(frozen=True)
class MetricPoint:
    timestamp: int
    value: float


class TimeframeMetricSeries:
    def __init__(self, points: Sequence[MetricPoint]):
        if not points:
            raise ValueError("TimeframeMetricSeries requires at least one point")
        ordered = sorted(points, key=lambda p: p.timestamp)
        self._points: List[MetricPoint] = list(ordered)
        self._timestamps: List[int] = [p.timestamp for p in self._points]

    def value_at(self, timestamp: int) -> float:
        idx = bisect.bisect_right(self._timestamps, timestamp) - 1
        if idx < 0:
            idx = 0
        return self._points[idx].value

    def latest(self) -> float:
        return self._points[-1].value

