# Test Coverage Guide

This document explains how to run tests, generate coverage reports, and interpret the results.

## Quick Start

### Run All Tests

```bash
# Basic test run
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=indicator_collector --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=indicator_collector --cov-report=html

# Generate XML coverage report (for CI/CD)
pytest tests/ --cov=indicator_collector --cov-report=xml
```

### View Coverage Report

```bash
# Open HTML report in browser (after running coverage)
open htmlcov/index.html

# View in terminal
coverage report -m
```

## Test Organization

The test suite is organized into two sections:

### Main Test Directory (`tests/`)

Core tests for the trading system components:

- **`test_signal_generator.py`** (30+ tests)
  - Tests: signal generation, confidence calculation, factor weighting
  - Coverage: SignalGenerator, SignalConfig
  - Status: All tests passing ✓

- **`test_position_manager.py`** (43 tests)
  - Tests: position sizing, TP/SL calculations, risk management
  - Coverage: PositionManager, PositionManagerConfig
  - Status: All tests passing ✓

- **`test_statistics_optimizer.py`** (40+ tests)
  - Tests: signal outcome recording, performance metrics, weight optimization
  - Coverage: StatisticsOptimizer
  - Status: All tests passing ✓

- **`test_web_ui_automated_signals.py`** (80+ tests)
  - Tests: signal JSON serialization, web UI formatting, data validation
  - Coverage: Web UI display, JSON parsing, metrics calculation
  - Status: All tests passing ✓

### Root Test Files

Individual test files for specific modules:

- `test_technical_analysis.py` - Technical indicator tests
- `test_sentiment_analyzer.py` - Sentiment analysis tests
- `test_multitimeframe_analyzer.py` - Multi-timeframe trend tests
- `test_volume_orderbook_analyzer.py` - Volume and orderbook tests
- `test_trading_system.py` - Integration tests

## Running Specific Tests

### By Test File

```bash
# Run only signal generator tests
pytest tests/test_signal_generator.py -v

# Run only position manager tests
pytest tests/test_position_manager.py -v

# Run only web UI tests
pytest tests/test_web_ui_automated_signals.py -v
```

### By Test Class

```bash
# Run specific test class
pytest tests/test_signal_generator.py::TestSignalGeneration -v

# Run specific test method
pytest tests/test_signal_generator.py::TestSignalGeneration::test_buy_signal -v
```

### By Keyword

```bash
# Run all tests matching keyword
pytest tests/ -k "confidence" -v

# Run all signal generator tests
pytest tests/ -k "signal" -v

# Run all position tests
pytest tests/ -k "position" -v
```

## Coverage Report Interpretation

### Terminal Coverage Report

```
Name                              Stmts   Miss  Cover
----------------------------------------------------
indicator_collector/__init__.py       8      2    75%
indicator_collector/cli.py           25      5    80%
indicator_collector/collector.py     45      3    93%
----------------------------------------------------
Total                              500     45    91%
```

- **Stmts**: Number of statements (lines of executable code)
- **Miss**: Number of statements not executed by tests
- **Cover**: Coverage percentage (higher is better)

Target: **>90% coverage** for core modules

### Missing Line Report

```
pytest tests/ --cov=indicator_collector --cov-report=term-missing
```

Shows which lines are not covered:

```
indicator_collector/signal_generator.py:125-145,201
```

This means lines 125-145 and 201 are not executed by tests.

## Coverage Goals by Module

### Core Trading System (Target: 95%+)

- `trading_system/interfaces.py` - Data structures
- `trading_system/signal_generator.py` - Signal generation
- `trading_system/position_manager.py` - Position sizing
- `trading_system/statistics_optimizer.py` - Performance tracking

### Analysis Modules (Target: 90%+)

- `trading_system/technical_analysis.py` - Technical indicators
- `trading_system/sentiment_analyzer.py` - Sentiment analysis
- `trading_system/multitimeframe_analyzer.py` - Multi-timeframe analysis
- `trading_system/volume_orderbook_analyzer.py` - Volume analysis

### Utility Modules (Target: 85%+)

- `data_fetcher.py` - Data fetching
- `indicator_metrics.py` - Indicator calculations
- `market_context.py` - Market context analysis
- `advanced_metrics.py` - Advanced metrics

### UI and CLI (Target: 80%+)

- `web_ui.py` - Streamlit interface (complex, partial coverage acceptable)
- `cli.py` - Command-line interface
- `collector.py` - Main collector logic

## Continuous Integration Coverage

### GitHub Actions Workflow

The CI pipeline (``.github/workflows/tests.yml`) runs:

1. **Unit tests** on Python 3.10, 3.11, 3.12
2. **Coverage collection** with multiple Python versions
3. **JSON schema validation** for sample files
4. **Syntax checking** for all modules

### Accessing CI Results

```bash
# View workflow runs
https://github.com/YOUR_ORG/indicator-collector/actions

# Download coverage artifacts
# (Available in workflow run details)
```

## Adding New Tests

### Test Template

```python
"""Tests for new_module."""

import pytest
from indicator_collector.trading_system import YourComponent


class TestYourComponent:
    """Tests for YourComponent."""
    
    @pytest.fixture
    def component(self):
        """Create component instance."""
        return YourComponent(param1="value")
    
    def test_basic_functionality(self, component):
        """Test basic functionality."""
        result = component.do_something()
        assert result is not None
    
    def test_edge_case(self, component):
        """Test edge case."""
        with pytest.raises(ValueError):
            component.do_something(invalid_input=True)
```

### Ensure Coverage

When adding new functionality:

1. Write tests before or alongside the code
2. Run coverage: `pytest tests/ --cov=indicator_collector`
3. Aim for >90% coverage on new code
4. Document any intentionally uncovered lines

## Benchmarking and Performance Tests

### Run Performance Tests

```bash
# Run with timing information
pytest tests/ -v --durations=10

# Profile specific tests
python -m cProfile -s cumulative -m pytest tests/test_signal_generator.py
```

### Performance Targets

- Signal generation: <100ms per signal
- Position calculation: <50ms per position
- Statistics update: <10ms per outcome record

## Troubleshooting Test Issues

### Tests Failing with "No module named"

```bash
# Ensure package is installed in editable mode
pip install -e .

# Or add project to Python path
export PYTHONPATH=/home/engine/project:$PYTHONPATH
```

### Coverage Report Shows 0% for Some Modules

```bash
# Verify module is importable
python -c "from indicator_collector import module_name"

# Check if module has test imports
grep -r "from indicator_collector import" tests/
```

### Flaky Tests (Intermittent Failures)

- Check for time-dependent logic
- Mock external services (Binance API)
- Use fixed random seeds: `pytest --randomly-seed=12345`

## Best Practices

1. **Organize Tests**: Group related tests in classes
2. **Use Fixtures**: Share setup code with pytest fixtures
3. **Parametrize**: Use `@pytest.mark.parametrize` for multiple inputs
4. **Mock External**: Mock Binance API calls for deterministic tests
5. **Test Names**: Clear names that describe what's being tested
6. **DRY**: Don't repeat test logic
7. **Fast**: Tests should run in <5 seconds total
8. **Deterministic**: No random failures, same results every run

## Example: Complete Test Suite

```python
"""Example comprehensive test suite."""

import pytest
import json
from indicator_collector.trading_system import (
    SignalGenerator,
    AnalyzerContext,
    PositionManager,
)


class TestCompleteWorkflow:
    """Integration tests for complete trading workflow."""
    
    @pytest.fixture
    def generator(self):
        return SignalGenerator()
    
    @pytest.fixture
    def position_manager(self):
        return PositionManager(account_size_usd=10000)
    
    @pytest.fixture
    def context(self):
        return AnalyzerContext(
            symbol="BTCUSDT",
            timeframe="1h",
            current_price=45000.0,
            current_time=1699000000000,
            indicators={"rsi": 28.0},
        )
    
    def test_generate_and_position(self, generator, position_manager, context):
        """Test signal generation and position planning."""
        signal = generator.generate_signal(context)
        
        if signal.signal_type == "BUY":
            position = position_manager.plan_position(
                signal_type="BUY",
                current_price=context.current_price,
                stop_loss_price=44000.0,
                entry_price=context.current_price,
            )
            
            assert position.position_size_usd > 0
            assert position.risk_reward_ratio > 0
    
    def test_json_serialization(self, generator, context):
        """Test signal JSON serialization."""
        signal = generator.generate_signal(context)
        signal_dict = signal.to_dict()
        
        # Verify it's JSON serializable
        json_str = json.dumps(signal_dict)
        assert json_str is not None
        
        # Verify it can be parsed back
        parsed = json.loads(json_str)
        assert parsed['signal_type'] in ['BUY', 'SELL', 'NEUTRAL']
```

## Continuous Improvement

### Monitor Coverage Trends

```bash
# Generate coverage report for each commit
git log --oneline | head -5 | while read commit; do
    git checkout $commit
    pytest --cov=indicator_collector --cov-report=xml
    cp coverage.xml coverage_${commit:0:7}.xml
done
```

### Set Coverage Gates

```bash
# Fail if coverage drops below 85%
pytest tests/ --cov=indicator_collector --cov-fail-under=85
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Best Practices for Test Coverage](https://testdriven.io/)
