"""
Pydantic models for data validation across the Traderaihelper system.

This module defines data models with comprehensive validation for:
- Trading signals (BUY/SELL/HOLD)
- Order placement and management
- Position tracking
- API credentials
- Processed signal validation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SignalType(str, Enum):
    """Valid signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderSide(str, Enum):
    """Valid order sides."""
    BUY = "Buy"
    SELL = "Sell"


class OrderType(str, Enum):
    """Valid order types."""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_MARKET = "StopMarket"
    TAKE_PROFIT = "TakeProfit"
    TAKE_PROFIT_MARKET = "TakeProfitMarket"
    TRAILING_STOP = "TrailingStop"


class Direction(str, Enum):
    """Valid trade directions."""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(str, Enum):
    """Order status values."""
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"


class Credentials(BaseModel):
    """
    API credentials with validation.

    Attributes:
        api_key: API key for exchange
        api_secret: API secret for exchange
        exchange: Exchange name (e.g., 'bybit', 'binance')
        testnet: Whether to use testnet
    """

    api_key: str = Field(..., min_length=10, description="API key for exchange")
    api_secret: str = Field(..., min_length=10, description="API secret for exchange")
    exchange: str = Field(default="bybit", description="Exchange name")
    testnet: bool = Field(default=True, description="Use testnet if True")

    @field_validator("exchange")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """Validate exchange name."""
        valid_exchanges = ["bybit", "binance", "okx", "kucoin"]
        v_lower = v.lower()
        if v_lower not in valid_exchanges:
            raise ValueError(f"Exchange must be one of: {valid_exchanges}")
        return v_lower

    @field_validator("api_key", "api_secret")
    @classmethod
    def validate_not_whitespace(cls, v: str) -> str:
        """Validate credential is not just whitespace."""
        if v.strip() != v:
            raise ValueError("Credentials must not contain leading/trailing whitespace")
        return v

    def is_valid(self) -> bool:
        """Check if credentials are properly configured."""
        return bool(
            self.api_key and
            self.api_secret and
            len(self.api_key) >= 10 and
            len(self.api_secret) >= 10
        )


class Signal(BaseModel):
    """
    Trading signal with comprehensive validation.

    Attributes:
        signal_id: Unique signal identifier
        signal_type: Type of signal (BUY/SELL/HOLD)
        symbol: Trading symbol
        direction: Trade direction (LONG/SHORT)
        entry_price: Entry price
        take_profit: Take profit price(s) - can be single value or dict with levels
        stop_loss: Stop loss price
        confidence: Signal confidence score (0-1)
        leverage: Leverage multiplier
        quantity: Order quantity
        generated_at: Timestamp when signal was generated
        metadata: Additional signal metadata
        indicators: Indicator values that generated the signal
    """

    signal_id: str = Field(..., description="Unique signal identifier")
    signal_type: SignalType = Field(..., description="Type of signal")
    symbol: str = Field(..., min_length=2, description="Trading symbol")
    direction: Direction = Field(..., description="Trade direction")
    entry_price: float = Field(..., gt=0, description="Entry price")
    take_profit: Union[float, Dict[str, float]] = Field(..., description="Take profit price(s)")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    confidence: float = Field(default=0.5, ge=0, le=1, description="Signal confidence (0-1)")
    leverage: float = Field(default=5.0, gt=0, le=125, description="Leverage multiplier")
    quantity: float = Field(default=0.001, gt=0, description="Order quantity")
    generated_at: int = Field(..., description="Unix timestamp in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    indicators: Dict[str, Any] = Field(default_factory=dict, description="Indicator values")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        if len(v) < 3:
            raise ValueError("Symbol must be at least 3 characters long")
        return v.upper()

    @field_validator("take_profit")
    @classmethod
    def validate_take_profit(cls, v: Union[float, Dict[str, float]]) -> Union[float, Dict[str, float]]:
        """Validate take profit values."""
        if isinstance(v, float):
            if v <= 0:
                raise ValueError("Take profit must be positive")
            return v
        elif isinstance(v, dict):
            for key, value in v.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Take profit level {key} must be positive number")
            return v
        else:
            raise ValueError("Take profit must be float or dict")

    @model_validator(mode="after")
    def validate_price_relationship(self) -> "Signal":
        """Validate price relationships based on direction."""
        if self.signal_type == SignalType.BUY and self.direction == Direction.LONG:
            # For LONG: entry < take_profit, entry > stop_loss
            if isinstance(self.take_profit, float):
                if not (self.entry_price < self.take_profit):
                    raise ValueError("For LONG trades, entry_price must be less than take_profit")
            if not (self.entry_price > self.stop_loss):
                raise ValueError("For LONG trades, entry_price must be greater than stop_loss")

        elif self.signal_type == SignalType.SELL and self.direction == Direction.SHORT:
            # For SHORT: entry > take_profit, entry < stop_loss
            if isinstance(self.take_profit, float):
                if not (self.entry_price > self.take_profit):
                    raise ValueError("For SHORT trades, entry_price must be greater than take_profit")
            if not (self.entry_price < self.stop_loss):
                raise ValueError("For SHORT trades, entry_price must be less than stop_loss")

        return self

    def is_executable(self) -> bool:
        """Check if signal is executable (not HOLD)."""
        return self.signal_type != SignalType.HOLD

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
            "leverage": self.leverage,
            "quantity": self.quantity,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
            "indicators": self.indicators,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create Signal from dictionary."""
        # Handle enum conversion
        if "signal_type" in data and isinstance(data["signal_type"], str):
            data["signal_type"] = SignalType(data["signal_type"])
        if "direction" in data and isinstance(data["direction"], str):
            data["direction"] = Direction(data["direction"])
        return cls(**data)


class Order(BaseModel):
    """
    Order with comprehensive validation.

    Attributes:
        order_id: Unique order identifier
        client_order_id: Client-side order identifier
        symbol: Trading symbol
        side: Order side (Buy/Sell)
        order_type: Order type
        quantity: Order quantity
        price: Order price (required for Limit orders)
        stop_loss: Stop loss price
        take_profit: Take profit price
        reduce_only: Reduce position only flag
        close_on_trigger: Close position on trigger
        status: Order status
        created_at: Order creation timestamp
        updated_at: Last update timestamp
    """

    order_id: Optional[str] = Field(default=None, description="Exchange order ID")
    client_order_id: Optional[str] = Field(default=None, max_length=36, description="Client order ID")
    symbol: str = Field(..., min_length=2, description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(default=None, gt=0, description="Order price")
    stop_loss: Optional[float] = Field(default=None, gt=0, description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, gt=0, description="Take profit price")
    reduce_only: bool = Field(default=False, description="Reduce position only")
    close_on_trigger: bool = Field(default=False, description="Close on trigger")
    status: OrderStatus = Field(default=OrderStatus.NEW, description="Order status")
    created_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="Creation timestamp")
    updated_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="Update timestamp")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        if len(v) < 3:
            raise ValueError("Symbol must be at least 3 characters long")
        return v.upper()

    @field_validator("client_order_id")
    @classmethod
    def validate_client_order_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate client order ID length."""
        if v is not None and len(v) > 36:
            raise ValueError("Client order ID must be at most 36 characters")
        return v

    @model_validator(mode="after")
    def validate_limit_order_price(self) -> "Order":
        """Validate that Limit orders have a price."""
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Price is required for Limit orders")
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reduce_only": self.reduce_only,
            "close_on_trigger": self.close_on_trigger,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Position(BaseModel):
    """
    Position tracking with validation.

    Attributes:
        symbol: Trading symbol
        side: Position side (Buy=LONG, Sell=SHORT)
        size: Position size (positive for LONG, negative for SHORT)
        entry_price: Average entry price
        mark_price: Current mark price
        unrealized_pnl: Unrealized PnL
        leverage: Current leverage
        liquidation_price: Liquidation price
        created_at: Position creation timestamp
        updated_at: Last update timestamp
    """

    symbol: str = Field(..., min_length=2, description="Trading symbol")
    side: str = Field(..., description="Position side (Buy=LONG, Sell=SHORT)")
    size: float = Field(..., description="Position size")
    entry_price: float = Field(..., gt=0, description="Average entry price")
    mark_price: Optional[float] = Field(default=None, gt=0, description="Current mark price")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized PnL")
    leverage: float = Field(default=1.0, gt=0, le=125, description="Current leverage")
    liquidation_price: Optional[float] = Field(default=None, gt=0, description="Liquidation price")
    created_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="Creation timestamp")
    updated_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="Update timestamp")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate trading symbol format."""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        if len(v) < 3:
            raise ValueError("Symbol must be at least 3 characters long")
        return v.upper()

    @property
    def is_long(self) -> bool:
        """Check if position is LONG."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Check if position is SHORT."""
        return self.size < 0

    def calculate_pnl_percentage(self) -> float:
        """Calculate PnL as percentage of position value."""
        if self.mark_price is None or self.entry_price == 0:
            return 0.0

        if self.is_long:
            price_change = (self.mark_price - self.entry_price) / self.entry_price
        else:
            price_change = (self.entry_price - self.mark_price) / self.entry_price

        return price_change * 100 * self.leverage

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "unrealized_pnl": self.unrealized_pnl,
            "leverage": self.leverage,
            "liquidation_price": self.liquidation_price,
            "pnl_percentage": self.calculate_pnl_percentage(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ProcessedSignal(BaseModel):
    """
    Processed signal validation model.

    This validates the structure and content of processed signals
    before execution or further processing.

    Attributes:
        signal: The Signal object
        processed_at: Processing timestamp
        validated: Whether the signal passed validation
        validation_errors: List of validation errors (if any)
        execution_attempts: Number of execution attempts
        last_attempt_at: Last execution attempt timestamp
        execution_status: Current execution status
        execution_result: Result of last execution attempt
    """

    signal: Signal = Field(..., description="The signal object")
    processed_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="Processing timestamp")
    validated: bool = Field(default=False, description="Validation status")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    execution_attempts: int = Field(default=0, ge=0, description="Number of execution attempts")
    last_attempt_at: Optional[int] = Field(default=None, description="Last execution attempt timestamp")
    execution_status: str = Field(default="pending", description="Execution status")
    execution_result: Optional[Dict[str, Any]] = Field(default=None, description="Execution result")

    def validate(self) -> bool:
        """
        Validate the signal structure and content.

        Returns:
            True if validation passes, False otherwise
        """
        errors = []

        # Check if signal is executable
        if not self.signal.is_executable():
            errors.append("Signal type is HOLD, not executable")

        # Validate confidence threshold
        if self.signal.confidence < 0.6:
            errors.append(f"Signal confidence {self.signal.confidence} below minimum threshold 0.6")

        # Validate price relationships
        try:
            # Signal model already validates this via model_validator
            pass
        except ValueError as e:
            errors.append(f"Price relationship error: {e}")

        # Validate entry price is reasonable
        if self.signal.entry_price <= 0:
            errors.append("Entry price must be positive")

        # Validate stop loss and take profit
        if self.signal.stop_loss <= 0:
            errors.append("Stop loss must be positive")

        if isinstance(self.signal.take_profit, float):
            if self.signal.take_profit <= 0:
                errors.append("Take profit must be positive")
        elif isinstance(self.signal.take_profit, dict):
            if not self.signal.take_profit:
                errors.append("Take profit levels cannot be empty")

        # Validate leverage is reasonable
        if self.signal.leverage < 1 or self.signal.leverage > 125:
            errors.append(f"Leverage {self.signal.leverage} must be between 1 and 125")

        # Validate quantity
        if self.signal.quantity <= 0:
            errors.append("Quantity must be positive")

        # Validate timestamp
        if self.signal.generated_at <= 0:
            errors.append("Signal generated_at timestamp must be positive")

        # Check if signal is too old (older than 5 minutes)
        current_time = int(datetime.utcnow().timestamp() * 1000)
        signal_age_ms = current_time - self.signal.generated_at
        if signal_age_ms > 300000:  # 5 minutes
            errors.append(f"Signal is too old: {signal_age_ms / 1000:.1f} seconds")

        # Update validation status
        self.validation_errors = errors
        self.validated = len(errors) == 0

        return self.validated

    def increment_execution_attempt(self) -> None:
        """Increment execution attempt counter."""
        self.execution_attempts += 1
        self.last_attempt_at = int(datetime.utcnow().timestamp() * 1000)

    def set_execution_result(self, result: Dict[str, Any], status: str) -> None:
        """
        Set execution result and status.

        Args:
            result: Execution result dictionary
            status: Execution status string
        """
        self.execution_result = result
        self.execution_status = status
        self.last_attempt_at = int(datetime.utcnow().timestamp() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert processed signal to dictionary."""
        return {
            "signal": self.signal.to_dict(),
            "processed_at": self.processed_at,
            "validated": self.validated,
            "validation_errors": self.validation_errors,
            "execution_attempts": self.execution_attempts,
            "last_attempt_at": self.last_attempt_at,
            "execution_status": self.execution_status,
            "execution_result": self.execution_result,
        }


class HealthStatus(str, Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck(BaseModel):
    """
    Health check result model.

    Attributes:
        component: Component name
        status: Health status
        message: Status message
        details: Additional details
        checked_at: Check timestamp
        response_time_ms: Response time in milliseconds
    """

    component: str = Field(..., description="Component name")
    status: HealthStatus = Field(default=HealthStatus.UNKNOWN, description="Health status")
    message: str = Field(default="", description="Status message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    checked_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp() * 1000), description="Check timestamp")
    response_time_ms: Optional[float] = Field(default=None, description="Response time in ms")

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at,
            "response_time_ms": self.response_time_ms,
        }
