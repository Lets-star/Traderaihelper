# Development Guide

This guide provides information for developers contributing to the Indicator Collector project.

## Project Structure

```
indicator_collector/
├── __init__.py                 # Package initialization
├── __main__.py                 # Entry point for module execution
├── cli.py                      # Command-line interface
├── collector.py                # Main data collection logic
├── data_fetcher.py             # Binance API integration
├── indicator_metrics.py        # Core indicator calculations
├── advanced_metrics.py         # Advanced market analytics
├── math_utils.py               # Mathematical utilities
├── trade_signals.py            # Signal evaluation
├── market_context.py           # Market context analysis
├── market_maker_detection.py   # Market maker detection algorithms
├── astrology.py                # Astrological metrics
├── cme_gap.py                  # CME gap analysis with caching
├── time_series.py              # Time series utilities
└── trading_system/             # Automated trading system
    ├── __init__.py             # Exports for trading system
    ├── interfaces.py           # Core dataclasses and protocols
    ├── signal_generator.py     # Signal generation engine
    ├── technical_analysis.py   # Technical indicators
    ├── sentiment_analyzer.py   # Sentiment analysis
    ├── multitimeframe_analyzer.py  # Multi-timeframe analysis
    ├── volume_orderbook_analyzer.py # Volume analysis
    ├── position_manager.py     # Position sizing and risk
    └── statistics_optimizer.py # Performance tracking and optimization

tests/
├── test_signal_generator.py    # Signal generator tests
├── test_position_manager.py    # Position manager tests
├── test_statistics_optimizer.py # Statistics optimizer tests
└── test_web_ui_automated_signals.py # Web UI tests

samples/
├── trading_signal_schema.json  # JSON Schema for signals
├── example_buy_signal.json     # Example BUY signal
├── example_sell_signal.json    # Example SELL signal
├── example_neutral_signal.json # Example NEUTRAL signal
├── macro_filter_config.json    # Market regime configurations
├── BACKTESTING_WORKFLOW.md     # Backtesting guide
└── TEST_COVERAGE_GUIDE.md      # Testing guide
```

## Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd indicator_collector
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package in editable mode with development extras
pip install -e ".[dev]"

# For a runtime-only environment
pip install -e .
```

### 4. Verify Installation

```bash
# Test imports
python -c "from indicator_collector import *; print('✓ Package imports successfully')"

# Run a quick test
pytest tests/test_signal_generator.py::TestSignalGeneration::test_buy_signal -v
```

## Code Style and Conventions

### Python Version
- **Target**: Python 3.12
- **Minimum**: Python 3.10
- Use type hints for all functions
- Use `from __future__ import annotations` for forward references

### Import Order
1. Standard library
2. Third-party libraries
3. Local imports

Example:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..math_utils import Candle
from .interfaces import FactorScore, JsonDict
```

### Docstring Style
Use Google-style docstrings for public functions:

```python
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: List of closing prices (most recent last).
        period: Number of periods for RSI calculation (default: 14).
    
    Returns:
        RSI value between 0 and 100.
    
    Raises:
        ValueError: If prices list is too short for the period.
    """
```

### Module Docstrings
Every module should have a module-level docstring:

```python
"""Brief description of what this module does.

Longer explanation if needed.
"""

from __future__ import annotations
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `SignalGenerator`)
- Functions: `snake_case` (e.g., `calculate_rsi`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_RSI_PERIOD`)
- Private: prefix with `_` (e.g., `_internal_method`)

### Type Hints
Always use type hints:

```python
# ✓ Good
def process_signal(signal: TradingSignalPayload, config: SignalConfig) -> bool:
    """Process a trading signal."""
    pass

# ✗ Avoid
def process_signal(signal, config):
    pass
```

### Data Structures
Use dataclasses for structured data:

```python
@dataclass
class MyData:
    """Description of MyData."""
    
    field1: str
    field2: int = 10  # Default value
    field3: Optional[List[float]] = None
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=indicator_collector --cov-report=term-missing

# Run specific test file
pytest tests/test_signal_generator.py -v

# Run specific test class
pytest tests/test_signal_generator.py::TestSignalGeneration -v

# Run specific test method
pytest tests/test_signal_generator.py::TestSignalGeneration::test_buy_signal -v
```

### Test Coverage Goals

- **Core trading system**: >95%
- **Analysis modules**: >90%
- **Utility modules**: >85%
- **UI/CLI**: >80%

### Writing Tests

Test files should be in `tests/` directory with `test_` prefix:

```python
"""Tests for my_module."""

import pytest
from indicator_collector.my_module import MyClass


class TestMyClass:
    """Tests for MyClass."""
    
    @pytest.fixture
    def instance(self):
        """Fixture providing MyClass instance."""
        return MyClass(param="value")
    
    def test_basic_functionality(self, instance):
        """Test that basic functionality works."""
        result = instance.do_something()
        assert result is not None
    
    @pytest.mark.parametrize("input,expected", [
        ("a", 1),
        ("b", 2),
    ])
    def test_with_parameters(self, instance, input, expected):
        """Test with multiple inputs."""
        result = instance.process(input)
        assert result == expected
    
    def test_error_handling(self, instance):
        """Test error handling."""
        with pytest.raises(ValueError):
            instance.process(invalid=True)
```

## Git Workflow

### Branch Naming
- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Maintenance: `chore/description`

### Commit Messages
Use clear, concise commit messages:

```
type(scope): Brief description

Longer explanation if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Requests
1. Create a feature branch
2. Make changes
3. Run tests locally: `pytest tests/ -v`
4. Commit changes with clear messages
5. Push to remote
6. Create PR with description
7. Address review feedback
8. Merge when approved

## Performance Considerations

### Optimization Areas

1. **Signal Generation** (<100ms target)
   - Profile with: `python -m cProfile -s cumulative -m pytest`
   - Cache expensive calculations
   - Use numpy for numerical operations

2. **Position Calculation** (<50ms target)
   - Minimize object allocation
   - Batch process multiple positions

3. **Data Fetching** (network I/O bound)
   - Implement caching (see `cme_gap.py` for example)
   - Use connection pooling
   - Handle rate limiting gracefully

### Profiling

```bash
# Profile a specific test
python -m cProfile -s cumulative -m pytest tests/test_signal_generator.py::TestSignalGeneration::test_buy_signal

# Generate profile stats
python -c "
import cProfile
import pstats
from indicator_collector.trading_system import SignalGenerator

profiler = cProfile.Profile()
profiler.enable()

# Code to profile here
generator = SignalGenerator()
for i in range(1000):
    signal = generator.generate_signal(context)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
"
```

## Documentation

### Adding Documentation

1. **Module docstrings**: Brief description at top of file
2. **Function docstrings**: Google-style for public functions
3. **Inline comments**: For complex logic
4. **Type hints**: Always include return types
5. **Examples**: Include usage examples in docstrings

### Building Docs

Documentation files are in Markdown:
- `README.md` - Main project documentation
- `QUICKSTART.md` - Quick start guide
- `DEVELOPMENT.md` - This file
- `AUTOMATED_SIGNALS_INTEGRATION.md` - Integration guide
- `AUTOMATED_SIGNALS_TAB_USAGE.md` - User guide
- `samples/` - Sample files and guides

### Updating Documentation

When making changes:
1. Update relevant docs
2. Update CHANGELOG if significant
3. Add examples if introducing new features
4. Verify links work
5. Check markdown formatting

## Deployment

### Release Process

1. Update version in relevant files
2. Update CHANGELOG
3. Create release notes
4. Tag release in git
5. Run final tests
6. Deploy to production

### Production Checklist

- [ ] All tests passing
- [ ] Code review completed
- [ ] Documentation updated
- [ ] Performance benchmarks acceptable
- [ ] No security issues
- [ ] No breaking changes (or documented)

## Troubleshooting Development Issues

### "ModuleNotFoundError"

```bash
# Ensure package is in PYTHONPATH
export PYTHONPATH=/path/to/project:$PYTHONPATH

# Or use editable install
pip install -e .
```

### "No module named pytest"

```bash
pip install pytest pytest-cov pytest-mock
```

### "Binance API errors in tests"

Tests should mock Binance API calls:
```python
@pytest.fixture
def mock_binance(monkeypatch):
    def mock_fetch(*args, **kwargs):
        return [{"time": 0, "close": 100.0, ...}]
    
    monkeypatch.setattr("indicator_collector.data_fetcher.fetch_binance_ohlcv", mock_fetch)
```

### "Test coverage not improving"

1. Check what lines aren't covered: `pytest --cov-report=term-missing`
2. Add tests for uncovered paths
3. Consider if code is actually needed
4. Document intentionally uncovered code

## Adding New Features

### Checklist for New Features

- [ ] Create feature branch
- [ ] Implement feature with tests
- [ ] Write docstrings (module, functions, classes)
- [ ] Add type hints to all functions
- [ ] Ensure >90% coverage on new code
- [ ] Update README if user-facing
- [ ] Add example if appropriate
- [ ] Run all tests: `pytest tests/ -v`
- [ ] Check coverage: `pytest tests/ --cov=indicator_collector`
- [ ] Commit with clear message
- [ ] Create PR with description
- [ ] Address review feedback

### Example: Adding a New Analyzer

```python
"""New trading analyzer for custom signals."""

from __future__ import annotations

from typing import Iterable, Optional

from .interfaces import (
    TradingAnalyzer,
    AnalyzerContext,
    TradingSignalPayload,
    OptimizationStats,
    FactorScore,
    SignalExplanation,
    PositionPlan,
)


class MyCustomAnalyzer(TradingAnalyzer):
    """Custom trading analyzer implementation."""
    
    def __init__(self, param1: str = "default"):
        """Initialize analyzer.
        
        Args:
            param1: Configuration parameter.
        """
        self.param1 = param1
    
    def analyze(self, context: AnalyzerContext) -> TradingSignalPayload:
        """Analyze context and generate signal.
        
        Args:
            context: Current market context.
        
        Returns:
            Trading signal with factors and position plan.
        """
        # Your analysis logic here
        return TradingSignalPayload(
            signal_type="BUY",
            confidence=0.75,
            timestamp=context.current_time,
            factors=[
                FactorScore(
                    factor_name="my_factor",
                    score=75.0,
                    weight=1.0,
                )
            ],
            position_plan=PositionPlan(),
            explanation=SignalExplanation(primary_reason="Custom analysis"),
        )
    
    def optimize(self, history: Iterable[AnalyzerContext]) -> Optional[OptimizationStats]:
        """Optimize analyzer parameters based on history.
        
        Args:
            history: Historical contexts.
        
        Returns:
            Optimization statistics or None if insufficient data.
        """
        return None


# Add tests in tests/test_my_custom_analyzer.py
```

## Resources

- **Python 3.12**: https://docs.python.org/3.12/
- **Type hints**: https://peps.python.org/pep-0484/
- **pytest**: https://docs.pytest.org/
- **Git**: https://git-scm.com/doc
- **Streamlit**: https://docs.streamlit.io/

## Getting Help

1. Check existing documentation
2. Search GitHub issues
3. Review similar code in repository
4. Ask in project discussions
5. Create an issue with details

## Code Review Checklist

When reviewing code:
- [ ] Tests included and passing
- [ ] Type hints present
- [ ] Docstrings clear and complete
- [ ] No obvious performance issues
- [ ] Follows project conventions
- [ ] No breaking changes
- [ ] Documentation updated if needed
- [ ] Commit messages clear
