# Pydantic Settings and Type Safety Guide

This guide explains how to use the new centralized configuration and type safety features.

## Overview

The codebase now includes:
- **Pydantic Settings** for centralized configuration management
- **Protocol types** for duck typing (Streamlit components, callbacks)
- **TypedDict** for structured data (signals, klines)
- **TypeGuard functions** for runtime type checking
- **Generic types** for reusable containers
- **Enum constants** for type-safe constants

## Configuration (config module)

### Basic Usage

```python
from config import AppSettings, get_settings

# Create settings instance
settings = AppSettings()

# Or use the global singleton
settings = get_settings()

# Access ByBit credentials
api_key, api_secret = settings.get_bybit_credentials()

# Access nested settings
leverage = settings.bybit.default_leverage
testnet = settings.bybit.testnet
confidence_threshold = settings.trading.min_confidence
default_timeframe = settings.ui.chart_default_timeframe
```

### Environment Variables

All settings can be configured via environment variables:

```bash
# ByBit settings
export BYBIT_API_KEY="your_api_key"
export BYBIT_API_SECRET="your_api_secret"
export BYBIT_TESTNET="true"
export BYBIT_DEFAULT_LEVERAGE="5"

# Trading settings
export TRADING_MAX_LEVERAGE="20"
export TRADING_MIN_CONFIDENCE="0.6"
export TRADING_DRY_RUN="false"

# UI settings
export UI_CHART_DEFAULT_TIMEFRAME="1h"
export UI_ENABLE_AUTO_REFRESH="true"
```

### Streamlit Secrets Integration

Settings automatically load from Streamlit secrets:

```toml
# .streamlit/secrets.toml
[bybit]
api_key = "your_api_key"
api_secret = "your_api_secret"
testnet = true
default_leverage = 5
```

## Types (types module)

### Protocol Types

Use Protocol types for duck typing Streamlit components:

```python
from typing import Optional
from trader_types import StreamlitComponent

def render_control(
    label: str,
    ui: Optional[StreamlitComponent] = None
) -> float:
    """Render a control that works with st or any container."""
    target = ui if ui is not None else st
    return target.number_input(label)

# Works with st
value = render_control("My Input")

# Works with columns
left, right = st.columns(2)
value = render_control("My Input", ui=left)
```

### TypedDict

Use TypedDict for structured signal data:

```python
from trader_types import SignalPayload, KlineData

# Type-safe signal creation
signal: SignalPayload = {
    "signal_id": "sig-123",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "entry_price": 50000.0,
    "take_profit": 55000.0,
    "stop_loss": 48000.0,
}

# Type-safe kline data
kline: KlineData = {
    "ts": 1700000000000,
    "open": 50000.0,
    "high": 51000.0,
    "low": 49000.0,
    "close": 50500.0,
    "volume": 100.5,
}
```

### TypeGuard Functions

Use TypeGuard for runtime type checking:

```python
from trader_types import is_valid_signal, is_kline_data

# API response validation
response = api.fetch_signal()
if is_valid_signal(response):
    # Type-safe access - IDE knows this is SignalPayload
    print(response["signal_id"])
    print(response["entry_price"])
else:
    logger.error("Invalid signal format")

# Kline validation
kline_data = websocket.get_message()
if is_kline_data(kline_data):
    # Type-safe access
    print(f"Close: {kline_data['close']}")
```

### Enum Types

Use Enums for type-safe constants:

```python
from trader_types import Timeframe, SignalDirection, OrderSide, ExecutionStatus

# Timeframe with helper properties
tf = Timeframe.from_string("1h")
if tf.is_short:
    poll_interval = 1000  # 1 second for short timeframes
elif tf.is_long:
    poll_interval = 5000  # 5 seconds for long timeframes

# Get milliseconds for calculations
ms = tf.milliseconds  # 3600000 for 1h

# Signal direction with conversions
direction = SignalDirection.LONG
order_side = direction.order_side  # "Buy"
opposite = direction.opposite  # SignalDirection.SHORT

# Order side from direction
order_side = OrderSide.from_direction(SignalDirection.SHORT)  # OrderSide.SELL

# Execution status
status = ExecutionStatus.FILLED
if status == ExecutionStatus.FILLED:
    print("Order filled!")
```

### Generic Types

Use Generic types for type-safe containers:

```python
from trader_types import UpdateBus, Result, DataStore
from trader_types.typed_dict import UpdateMessage

# Type-safe update bus
bus: UpdateBus[UpdateMessage] = UpdateBus()
bus.publish({"type": "EXECUTION_UPDATE", "status": "filled"})
updates = bus.drain()  # List[UpdateMessage]

# Type-safe result handling
def fetch_data() -> Result[KlineData, RequestException]:
    try:
        data = api.fetch_klines()
        return Result.ok(data)
    except RequestException as e:
        return Result.err(e)

result = fetch_data()
if result.is_ok:
    data = result.unwrap()  # Type is KlineData
else:
    error = result.error  # Type is RequestException

# Type-safe data store
cache: DataStore[str, KlineData] = DataStore(max_size=1000)
cache.set("BTCUSDT:1h", kline_data)
kline = cache.get("BTCUSDT:1h")  # Optional[KlineData]

# Compute if not exists
kline = cache.get_or_compute("BTCUSDT:1h", lambda: fetch_from_api())
```

## Migration Guide

### From os.getenv to Settings

**Before:**
```python
import os

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
```

**After:**
```python
from config import AppSettings

settings = AppSettings.from_secrets()
api_key = settings.bybit.api_key
api_secret = settings.bybit.api_secret
testnet = settings.bybit.testnet
```

### From Dict[str, Any] to TypedDict

**Before:**
```python
signal: Dict[str, Any] = {
    "signal_id": "123",
    "symbol": "BTCUSDT",
}
# No IDE autocomplete, no type checking
```

**After:**
```python
from trader_types import SignalPayload

signal: SignalPayload = {
    "signal_id": "123",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "entry_price": 50000.0,
}
# Full IDE autocomplete and type checking
```

### From Any to Protocol

**Before:**
```python
def num_float(label: str, ui: Optional[Any] = None) -> float:
    target = ui if ui is not None else st
    return target.number_input(label)
```

**After:**
```python
from trader_types import StreamlitComponent

def num_float(
    label: str,
    ui: Optional[StreamlitComponent] = None
) -> float:
    target = ui if ui is not None else st
    return target.number_input(label)
```

## Testing

Run the configuration and types tests:

```bash
# Run config tests
pytest tests/test_config.py -v

# Run types tests
pytest tests/test_types.py -v

# Run all tests
pytest tests/ -v
```

## Best Practices

1. **Always use Settings** instead of os.getenv for configuration
2. **Use TypedDict** for structured data like signals and klines
3. **Use TypeGuard** for runtime validation of external data
4. **Use Enums** instead of string literals for constants
5. **Use Protocols** for duck typing instead of Any
6. **Use Generics** for type-safe reusable containers

## Validation

All settings include validation:

```python
from config import ByBitSettings

# This will raise ValueError
ByBitSettings(default_leverage=200)  # Too high
ByBitSettings(api_key="short")  # Too short

# This will work
settings = ByBitSettings(default_leverage=20)
```

## Backward Compatibility

The existing code continues to work:
- `UpdateBus` still works with `Dict[str, Any]`
- `SecretManager` is now a thin wrapper around Settings
- Existing function signatures are preserved

New code should use the type-safe alternatives:
- `TypedUpdateBus[T]` for type-safe message passing
- Direct `AppSettings` instead of `SecretManager`
