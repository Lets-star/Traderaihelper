"""
Enum constants for trading and application states.

These enums provide:
- Type-safe string constants
- IDE autocomplete support
- Validation of valid values
"""

from __future__ import annotations

from enum import Enum, IntEnum


class Timeframe(str, Enum):
    """Valid chart timeframes."""

    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_3 = "3h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    @classmethod
    def from_string(cls, value: str) -> "Timeframe":
        """Create Timeframe from string, with validation."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid timeframe: {value}. Valid options: {[t.value for t in cls]}")

    @property
    def milliseconds(self) -> int:
        """Get timeframe duration in milliseconds."""
        mapping = {
            Timeframe.MINUTE_1: 60_000,
            Timeframe.MINUTE_3: 180_000,
            Timeframe.MINUTE_5: 300_000,
            Timeframe.MINUTE_15: 900_000,
            Timeframe.MINUTE_30: 1_800_000,
            Timeframe.HOUR_1: 3_600_000,
            Timeframe.HOUR_2: 7_200_000,
            Timeframe.HOUR_3: 10_800_000,
            Timeframe.HOUR_4: 14_400_000,
            Timeframe.HOUR_6: 21_600_000,
            Timeframe.HOUR_8: 28_800_000,
            Timeframe.HOUR_12: 43_200_000,
            Timeframe.DAY_1: 86_400_000,
            Timeframe.DAY_3: 259_200_000,
            Timeframe.WEEK_1: 604_800_000,
            Timeframe.MONTH_1: 2_592_000_000,
        }
        return mapping[self]

    @property
    def is_short(self) -> bool:
        """Check if this is a short timeframe (<= 15m)."""
        return self in {
            Timeframe.MINUTE_1,
            Timeframe.MINUTE_3,
            Timeframe.MINUTE_5,
            Timeframe.MINUTE_15,
        }

    @property
    def is_medium(self) -> bool:
        """Check if this is a medium timeframe (30m - 4h)."""
        return self in {
            Timeframe.MINUTE_30,
            Timeframe.HOUR_1,
            Timeframe.HOUR_2,
            Timeframe.HOUR_3,
            Timeframe.HOUR_4,
        }

    @property
    def is_long(self) -> bool:
        """Check if this is a long timeframe (>= 6h)."""
        return self in {
            Timeframe.HOUR_6,
            Timeframe.HOUR_8,
            Timeframe.HOUR_12,
            Timeframe.DAY_1,
            Timeframe.DAY_3,
            Timeframe.WEEK_1,
            Timeframe.MONTH_1,
        }


class SignalDirection(str, Enum):
    """Signal direction (trade side)."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def order_side(self) -> str:
        """Get corresponding order side."""
        return "Buy" if self == SignalDirection.LONG else "Sell"

    @property
    def opposite(self) -> "SignalDirection":
        """Get opposite direction."""
        return SignalDirection.SHORT if self == SignalDirection.LONG else SignalDirection.LONG


class SignalStrength(str, Enum):
    """Signal strength levels."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    HOLD = "HOLD"

    @classmethod
    def from_confidence(cls, confidence: float) -> "SignalStrength":
        """Determine strength from confidence score."""
        if confidence >= 0.8:
            return cls.STRONG
        elif confidence >= 0.6:
            return cls.MODERATE
        elif confidence >= 0.4:
            return cls.WEAK
        else:
            return cls.HOLD


class FactorCategory(str, Enum):
    """Signal factor categories."""

    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MULTITIMEFRAME = "multitimeframe"
    VOLUME = "volume"
    MARKET_STRUCTURE = "market_structure"
    COMPOSITE = "composite"

    @classmethod
    def get_default_weights(cls) -> dict[str, float]:
        """Get default factor weights."""
        return {
            cls.TECHNICAL.value: 0.30,
            cls.SENTIMENT.value: 0.20,
            cls.MULTITIMEFRAME.value: 0.20,
            cls.VOLUME.value: 0.15,
            cls.MARKET_STRUCTURE.value: 0.10,
            cls.COMPOSITE.value: 0.05,
        }


class ExecutionStatus(str, Enum):
    """Signal execution status."""

    PENDING = "pending"
    VALIDATING = "validating"
    SUBMITTING = "submitting"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    DRY_RUN = "dry_run"
    DISABLED = "disabled"
    VALIDATION_ERROR = "validation_error"


class WorkerStatus(str, Enum):
    """Background worker status."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class WebSocketStatus(str, Enum):
    """WebSocket connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    CLOSED = "closed"


class OrderSide(str, Enum):
    """Order side (Buy/Sell)."""

    BUY = "Buy"
    SELL = "Sell"

    @classmethod
    def from_direction(cls, direction: SignalDirection) -> "OrderSide":
        """Get order side from signal direction."""
        return cls.BUY if direction == SignalDirection.LONG else cls.SELL


class OrderType(str, Enum):
    """Order types."""

    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_MARKET = "StopMarket"
    TAKE_PROFIT = "TakeProfit"
    TAKE_PROFIT_MARKET = "TakeProfitMarket"
    TRAILING_STOP = "TrailingStop"


class OrderStatus(str, Enum):
    """Order status values."""

    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"


class SignalType(str, Enum):
    """Signal types."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CacheLevel(IntEnum):
    """Cache priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def numeric_level(self) -> int:
        """Get numeric logging level."""
        import logging
        return getattr(logging, self.value)
