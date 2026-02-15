# Quick Reference Guide - Traderaihelper Extended Fixes

## Key Changes Summary

| Component | Old Approach | New Approach |
|-----------|-------------|--------------|
| ByBitClient | Async/await with aiohttp | Synchronous with requests + threading |
| SignalExecutor | asyncio.run() | ThreadPoolExecutor |
| API Credentials | Constructor parameters | st.secrets + env fallback |
| CSV Logging | Single lock | Dedicated locks per component |
| Error Handling | Generic Exception | Specific exception types |
| Rate Limits | Basic retry | Exponential backoff + connection pool |
| Validation | Minimal | Comprehensive for all inputs |

## API Cheat Sheet

### SecretManager

```python
from utils.secret_manager import SecretManager

# Check if credentials available
has_creds = SecretManager.has_bybit_credentials()

# Get credentials
api_key, api_secret = SecretManager.get_bybit_credentials()

# Get full config
config = SecretManager.get_bybit_config()

# Validate format
valid = SecretManager.validate_credential_format(api_key, api_secret)

# Mask for logging
masked = SecretManager.mask_credential(api_key)
```

### ByBitClient

```python
from bybit_client import ByBitClient

# Context manager (recommended)
with ByBitClient(api_key, api_secret, testnet=True) as client:
    # Place order with validation
    result = client.place_order(
        symbol="BTCUSDT",
        side="Buy",
        qty=0.001,
        order_type="Market"
    )

    # Get position
    pos = client.get_position("BTCUSDT")

    # Set leverage
    client.set_leverage("BTCUSDT", 5)
```

### SignalExecutor

```python
from signal_executor import SignalExecutor

# Create executor
executor = SignalExecutor(update_bus)

# Configure (uses st.secrets automatically)
executor.configure(
    enabled=True,
    testnet=True,
    leverage=5,
    dry_run=False
)

# Execute signal
signal = {
    "signal_id": "sig_123",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "entry_price": 47000,
    "quantity": 0.001
}
executor.execute_signal(signal)
```

## Validation Rules

### Symbol
- Must be non-empty string
- Minimum 3 characters
- Automatically uppercased

### Side
- Must be "Buy" or "Sell" (case-insensitive)

### Order Type
- Valid types: Market, Limit, Stop, StopMarket, TakeProfit, TakeProfitMarket, TrailingStop

### Quantity
- Must be positive number
- Maximum 1,000,000

### Price
- Must be positive (if provided)
- Maximum 1,000,000

### Leverage
- Must be between 0 and 125

### Signal Validation
Required fields:
- signal_id (string)
- symbol (string)
- direction (LONG/SHORT)
- entry_price (positive number)

Optional fields:
- take_profit (positive number)
- stop_loss (positive number)
- quantity (positive number, default 0.001)
- leverage (number, default from config)

## Configuration

### st.secrets.toml

```toml
# Option 1: Direct keys
BYBIT_API_KEY = "your_key"
BYBIT_API_SECRET = "your_secret"

# Option 2: Nested (recommended)
[bybit]
api_key = "your_key"
api_secret = "your_secret"
testnet = true
default_leverage = 5
pos_size_multiplier = 1.0
```

### Environment Variables

```bash
BYBIT_API_KEY="your_key"
BYBIT_API_SECRET="your_secret"
BYBIT_TESTNET="true"
BYBIT_DEFAULT_LEVERAGE="5"
BYBIT_POS_SIZE_MULTIPLIER="1.0"
```

## Error Handling Patterns

### Value Errors (Validation)
```python
try:
    client.place_order(symbol="BTCUSDT", side="Buy", qty=-1)
except ValueError as e:
    print(f"Validation failed: {e}")
```

### Request Errors (Network)
```python
try:
    result = client.place_order(...)
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.ConnectionError:
    print("Connection failed")
```

### Business Logic Errors
```python
result = client.place_order(...)
if result.get("retCode") != 0:
    print(f"API error: {result.get('retMsg')}")
```

## Rate Limiting

Automatic retry with exponential backoff:
- Attempt 1: No delay
- Attempt 2: 1 second delay
- Attempt 3: 2 second delay
- Attempt 4: 4 second delay

Max retries: 3 (total 4 attempts)

## Connection Pooling

Default configuration:
- Pool size: 10 connections
- Max retries per connection: 3
- Retry status codes: 429, 500, 502, 503, 504
- Backoff factor: 0.5

Custom configuration:
```python
with ByBitClient(
    api_key, api_secret,
    connection_pool_size=20  # Custom pool size
) as client:
    ...
```

## Logging

### Log Files

1. **bybit_api_log.csv** - All API calls
   - Columns: timestamp, method, endpoint, symbol, side, qty, order_type, price, status, ret_code, ret_msg, latency_ms, attempt, error_details

2. **trade_execution_log.csv** - Signal executions
   - Columns: timestamp, signal_id, symbol, direction, qty, entry_price, take_profit, stop_loss, leverage, status, response_code, latency_ms, error_msg, validation_errors, thread_id

### Thread Safety
All CSV writes protected by `threading.Lock`

## Performance Tips

1. **Reuse clients**: Use context manager for automatic cleanup
2. **Batch operations**: ThreadPoolExecutor handles concurrent signals
3. **Monitor latency**: Check `latency_ms` in logs
4. **Watch rate limits**: Log 429 responses indicate hitting limits

## Common Issues

### Issue: "API credentials not found"
**Solution**: Set st.secrets or environment variables

### Issue: "Validation failed: Quantity must be positive"
**Solution**: Check quantity value is > 0

### Issue: "Rate limit exceeded"
**Solution**: Automatic retry, reduce request frequency

### Issue: "Invalid leverage"
**Solution**: Leverage must be 0-125

## Testing

```python
# Dry run mode (no actual trades)
executor.configure(enabled=True, dry_run=True)

# Testnet mode
executor.configure(enabled=True, testnet=True, dry_run=False)

# Validate signal without executing
errors = executor._validate_signal(signal)
if not errors:
    executor.execute_signal(signal)
```

## Migration

### Old → New

```python
# OLD
client = ByBitClient(key, secret)
result = await client.place_order(...)
await client.close()

# NEW
with ByBitClient(key, secret) as client:
    result = client.place_order(...)
```

```python
# OLD
executor.configure(enabled=True, api_key=key, api_secret=secret)

# NEW
executor.configure(enabled=True)  # Uses st.secrets
```

## Best Practices

1. **Always use context managers** for ByBitClient
2. **Use dry_run mode** when testing
3. **Validate signals** before execution
4. **Monitor log files** for errors and latency
5. **Set reasonable leverage** (start with 5x or lower)
6. **Use testnet first** before mainnet
7. **Check credentials format** before use
8. **Handle specific exceptions** rather than generic Exception

## Support

For issues or questions:
1. Check log files for detailed error messages
2. Verify credentials in st.secrets.toml
3. Ensure all required dependencies are installed
4. Review validation rules in code

## Dependencies

Required (already in pyproject.toml):
- requests>=2.31.0
- streamlit>=1.28.0

No additional dependencies needed.
