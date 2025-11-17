# Real Data Policy

## 🚨 Important: No Synthetic Data for Trading

This trading system is designed to work **exclusively with real market data**. The use of synthetic, generated, or mock data for actual trading decisions is **strictly prohibited**.

## What is Prohibited

❌ **DO NOT use:**
- Randomly generated price data
- Synthetic signal outcomes
- Mock market data
- Simulated trading results (except for testing/backtesting with real historical data)
- Artificial data from any source

## What is Required

✅ **DO use:**
- Real market data from exchanges (Binance, Coinbase, Kraken, etc.)
- Historical price data from verified sources
- Actual trading signal outcomes from real trades
- Real orderbook and volume data

## Data Validation

The system includes built-in validation to detect and reject synthetic data:

### RealDataValidator

The `RealDataValidator` class (`indicator_collector/real_data_validator.py`) performs:

1. **Source Validation**: Checks for proper exchange metadata
2. **Timestamp Validation**: Ensures timestamps are realistic and not in the future
3. **Synthetic Marker Detection**: Scans for keywords like "mock", "test", "synthetic", "fake", "sample", "demo", "simulated"
4. **Time Continuity**: Validates that data timestamps align properly with the timeframe
5. **Data Freshness**: Rejects stale data (older than 24 hours for live trading)

### Usage Example

```python
from indicator_collector.real_data_validator import RealDataValidator

validator = RealDataValidator()

# Validate payload
try:
    validator.validate_payload_sources(payload)
    validator.ensure_no_synthetic_flags(payload)
    validator.validate_time_continuity(payload, timeframe="1h")
    print("✓ Data validation passed - using real market data")
except DataValidationError as e:
    print(f"✗ Data validation failed: {e}")
    # DO NOT proceed with trading
```

## Demo and Testing

### Demo Scripts

Demo scripts (`demo_*.py`) are provided for **demonstration purposes only** and include warnings:

```python
⚠️ WARNING: This demo uses synthetic data for demonstration purposes only.
For real trading or production use, you MUST use actual market data from exchanges.
Synthetic data should NEVER be used for live trading decisions.
```

### Testing

Unit tests (`tests/test_*.py`) may use synthetic data for testing the codebase, but this is:
- Clearly marked as test data
- Never used for actual trading
- Isolated to the test environment

## Loading Real Data

### Signal Outcomes

When using the Statistics Optimizer or Adaptive Weights feature, provide real historical outcomes:

```python
# Load from file containing real trading results
from indicator_collector.trading_system.statistics_optimizer import SignalOutcome
import json

with open('real_signal_outcomes.json', 'r') as f:
    data = json.load(f)
    outcomes = [SignalOutcome.from_dict(item) for item in data]

# Add to optimizer
for outcome in outcomes:
    optimizer.add_signal_outcome(outcome)
```

Example file structure: `samples/example_signal_outcomes.json`

### Market Data

Always fetch data from real exchanges:

```python
from indicator_collector.trading_system.data_sources import BinanceKlinesSource

# Real market data from Binance
source = BinanceKlinesSource()
candles = source.fetch_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_time=start_timestamp,
    limit=100
)
```

## Consequences of Violating This Policy

Using synthetic data for real trading can lead to:

1. **Inaccurate Trading Decisions**: Synthetic data doesn't reflect real market conditions
2. **Financial Losses**: Models trained on fake data will fail in real markets
3. **System Failures**: Validation errors and rejected signals
4. **Regulatory Issues**: Trading systems must use real data for compliance

## Checklist Before Trading

Before enabling live trading, verify:

- [ ] All data sources point to real exchanges (Binance, Coinbase, etc.)
- [ ] No "demo", "test", or "mock" markers in metadata
- [ ] Timestamps are current and realistic
- [ ] Historical data comes from actual trading logs
- [ ] Validation passes without errors
- [ ] No calls to `create_synthetic_*` functions (except in tests)

## Getting Help

If you need to use real market data:

1. **Binance API**: https://binance-docs.github.io/apidocs/
2. **Historical Data**: Use the `BinanceKlinesSource` class
3. **Data Validation**: Review `indicator_collector/real_data_validator.py`
4. **Signal Logging**: Implement proper logging for signal outcomes

## Summary

**Remember**: Trading with real money requires real data. Always validate your data sources and never bypass the real data validator for production trading.

For questions or issues, please review:
- `DEVELOPMENT.md` - Development guidelines
- `QUICKSTART.md` - Getting started guide
- `samples/` - Example data structures
