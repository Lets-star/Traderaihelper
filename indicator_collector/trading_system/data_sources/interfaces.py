"""Base interfaces for historical data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from ...timeframes import Timeframe


class HistoricalDataSource(ABC):
    """Abstract base class for historical data sources."""

    @abstractmethod
    def load_candles(
        self,
        symbol: str,
        timeframe: Timeframe | str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Load historical OHLCV candles from the data source.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            timeframe: Timeframe for candles
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            DataFrame with columns: ts (UTC ms), open, high, low, close, volume
            Timestamps should be strictly increasing and numeric types.

        Raises:
            ValueError: If data cannot be loaded or is invalid
        """
        pass
