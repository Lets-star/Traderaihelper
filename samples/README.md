# Samples and Documentation

This directory contains sample JSON files, schemas, configuration examples, and documentation for the automated trading system.

## Files Overview

### JSON Schemas

#### `trading_signal_schema.json`
Complete JSON Schema (Draft-7) for trading signal payloads. Use this to:
- Validate signal JSON structures
- Generate documentation from schema
- Understand the complete signal format
- Build API clients that consume signals

**Key sections:**
- `signal_type`: BUY, SELL, or NEUTRAL
- `factors`: Individual scoring factors with weights
- `position_plan`: Entry, SL, TP levels and position sizing
- `explanation`: Signal rationale and risk factors
- `optimization_stats`: Historical performance metrics

### Example Signals

#### `example_buy_signal.json`
Complete example of a BUY signal with:
- High confidence (78%)
- Multiple bullish factors (RSI oversold, MACD crossover, multi-timeframe alignment)
- Full position plan with TP ladder
- Performance statistics

**Use case:** Understanding a bullish trading signal, testing JSON parsing

#### `example_sell_signal.json`
Complete example of a SELL signal with:
- Good confidence (71%)
- Multiple bearish factors (RSI overbought, divergence, resistance rejection)
- Short position plan with profit targets
- Risk factors documented

**Use case:** Understanding a bearish trading signal, testing short position logic

#### `example_neutral_signal.json`
Complete example of a NEUTRAL signal with:
- Mixed confidence (45%)
- No clear directional bias
- No position plan (zero allocation)
- Cancellation triggers documented

**Use case:** Understanding signal rejection, consolidation scenarios

### Configuration Examples

#### `macro_filter_config.json`
Market regime-based macro filter configurations for adapting signals to different conditions:

**Included configurations:**
- `normal_conditions`: Default configuration
- `high_volatility`: When VIX > 30, stricter requirements
- `low_volatility`: When VIX < 15, more relaxed requirements
- `bull_market`: Strong uptrend bias
- `bear_market`: Strong downtrend bias
- `news_event`: High volatility around major news

**Each configuration includes:**
- Factor weight adjustments
- Threshold modifications
- Risk tolerance parameters
- Sample Python implementation code

**Use case:** Adapting trading system to current market conditions, testing configuration loading

### Guides and Documentation

#### `BACKTESTING_WORKFLOW.md`
Complete step-by-step guide for backtesting trading strategies:

**Sections:**
1. Data Preparation - Gathering historical OHLCV data
2. Signal Generation - Creating your strategy and generating signals
3. Outcome Recording - Evaluating signal profitability
4. Performance Analysis - Calculating metrics and optimizing

**Includes:**
- Full code examples for each phase
- Complete strategy implementation example
- P&L calculation logic
- Performance metrics interpretation
- Best practices and common pitfalls

**Use case:** Testing your trading strategy, understanding backtesting workflow, optimization

#### `TEST_COVERAGE_GUIDE.md`
Comprehensive guide to running tests and understanding coverage:

**Sections:**
- Quick start for running tests
- Test organization and structure
- Running specific tests by file, class, or keyword
- Understanding coverage reports
- CI/CD integration
- Adding new tests
- Troubleshooting

**Coverage targets:**
- Core trading system: >95%
- Analysis modules: >90%
- Utility modules: >85%
- UI/CLI: >80%

**Use case:** Understanding test suite, running tests locally, improving coverage

## Quick Start

### Validate Sample Files

```bash
# Test JSON schema validity
python -c "import json; json.load(open('samples/trading_signal_schema.json'))"

# Test example signals
python -c "import json; print(json.load(open('samples/example_buy_signal.json'))['signal_type'])"
python -c "import json; print(json.load(open('samples/example_sell_signal.json'))['signal_type'])"
python -c "import json; print(json.load(open('samples/example_neutral_signal.json'))['signal_type'])"

# Test configuration
python -c "import json; print(list(json.load(open('samples/macro_filter_config.json'))['configurations'].keys()))"
```

### Load and Use Configurations

```python
import json
from indicator_collector.trading_system import SignalGenerator, SignalConfig

# Load macro filter config
with open('samples/macro_filter_config.json') as f:
    config_data = json.load(f)

# Get high volatility configuration
high_vol = config_data['configurations']['high_volatility']

# Create SignalConfig from it
config = SignalConfig(
    technical_weight=high_vol['technical_weight'],
    sentiment_weight=high_vol['sentiment_weight'],
    multitimeframe_weight=high_vol['multitimeframe_weight'],
    volume_weight=high_vol['volume_weight'],
    structure_weight=high_vol['structure_weight'],
    composite_weight=high_vol['composite_weight'],
    buy_threshold=high_vol['buy_threshold'],
    sell_threshold=high_vol['sell_threshold'],
    min_factors_confirm=high_vol['min_factors_confirm'],
)

# Use it
generator = SignalGenerator(config=config)
```

### Validate a Signal Against Schema

```python
import json
import jsonschema

# Load schema
with open('samples/trading_signal_schema.json') as f:
    schema = json.load(f)

# Load signal
with open('samples/example_buy_signal.json') as f:
    signal = json.load(f)

# Validate
try:
    jsonschema.validate(signal, schema)
    print("✓ Signal is valid")
except jsonschema.ValidationError as e:
    print(f"✗ Signal validation failed: {e.message}")
```

### Run Backtesting

```bash
# Study the backtesting workflow
cat samples/BACKTESTING_WORKFLOW.md

# Implement your strategy (see examples in BACKTESTING_WORKFLOW.md)
# Then run:
python your_backtest_script.py
```

### Run Tests with Coverage

```bash
# Run all tests with coverage
pytest tests/ -v --cov=indicator_collector --cov-report=term-missing

# Read coverage guide
cat samples/TEST_COVERAGE_GUIDE.md

# Generate HTML report
pytest tests/ --cov=indicator_collector --cov-report=html
open htmlcov/index.html
```

## File Structure Reference

```
samples/
├── README.md                          # This file
├── trading_signal_schema.json         # JSON Schema for signals
├── example_buy_signal.json            # Example BUY signal
├── example_sell_signal.json           # Example SELL signal
├── example_neutral_signal.json        # Example NEUTRAL signal
├── macro_filter_config.json           # Market regime configurations
├── BACKTESTING_WORKFLOW.md            # Complete backtesting guide
└── TEST_COVERAGE_GUIDE.md             # Test coverage and testing guide
```

## Integration Points

### Web UI
The web UI's "Automated Signals" tab displays signals in the format defined by `trading_signal_schema.json`.

See `web_ui.py` for the implementation:
```python
# Tab reads from payload["automated_signals"]
signal = payload.get("automated_signals", {})

# Displays all fields from the signal schema
```

### Data Collection
The collector can emit signals in the JSON format. Add to your collector:

```python
from indicator_collector.trading_system import SignalGenerator

signal = generator.generate_signal(context)
payload["automated_signals"] = signal.to_dict()
```

### API Integration
Any API client can validate incoming signals against the schema:

```python
import jsonschema
jsonschema.validate(incoming_signal, schema)
```

### CLI Integration
Export signals as JSON for downstream processing:

```bash
python main.py --token test --symbol BINANCE:BTCUSDT --timeframe 1h --period 500 \
    --output data.json
```

## Performance Characteristics

### Signal Generation
- Time: ~50-100ms per signal
- Memory: ~5-10MB for complete context

### Position Calculation
- Time: ~20-50ms per position
- Memory: <1MB

### Statistics Update
- Time: ~5-10ms per outcome record
- Memory: Grows with history size

## Validation and Testing

All sample files are automatically validated by CI:
- JSON schema syntax checking
- Example signal validation against schema
- Configuration file validation
- Python import checks

See `.github/workflows/tests.yml` for details.

## Adding New Samples

When adding new sample files:

1. Create JSON with clear, realistic data
2. Validate against appropriate schema
3. Add comments explaining the scenario
4. Include in documentation
5. Update CI workflow if needed
6. Test JSON validity: `python -c "import json; json.load(open('file.json'))"`

## Common Use Cases

### "I want to understand the signal format"
→ Read `trading_signal_schema.json` and `example_buy_signal.json`

### "I want to implement custom signal handling"
→ Read `trading_signal_schema.json` and look at `web_ui.py` implementation

### "I want to backtest my strategy"
→ Follow `BACKTESTING_WORKFLOW.md` step by step

### "I want to adapt to market conditions"
→ Load configuration from `macro_filter_config.json`

### "I want to understand test coverage"
→ Read `TEST_COVERAGE_GUIDE.md` and run `pytest tests/`

### "I want to see a real signal example"
→ Pick from `example_buy_signal.json`, `example_sell_signal.json`, or `example_neutral_signal.json`

## Support

For issues with:
- **Signal format**: Check `trading_signal_schema.json`
- **Examples**: See appropriate `example_*.json` file
- **Backtesting**: Follow `BACKTESTING_WORKFLOW.md`
- **Testing**: See `TEST_COVERAGE_GUIDE.md`
- **Configuration**: Review `macro_filter_config.json`

For general issues, check the main project README.md or AUTOMATED_SIGNALS_INTEGRATION.md.
