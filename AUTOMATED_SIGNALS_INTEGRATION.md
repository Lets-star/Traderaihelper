# Automated Signals Tab - Integration Guide

## Overview

This guide explains how to integrate the Automated Signals tab with your trading system, data collectors, and analyzers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Collector (collect_metrics)              │
│                   (produces payload JSON)                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (web_ui.py)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   🤖 Automated Signals Tab                           │  │
│  │  (reads trading_signals / automated_signals sections)│  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             ↑
         ┌───────────────────┴───────────────────┐
         │                                       │
         ↓                                       ↓
   ┌──────────────────┐             ┌──────────────────────┐
   │ Signal Generator │             │ Position Manager     │
   │ (signal.py)      │             │ (position_manager.py)│
   │                  │             │                      │
   │ - Combines       │             │ - Calculates sizing  │
   │   factors        │             │ - TP/SL ladders      │
   │ - Generates      │             │ - Holding horizon    │
   │   BUY/SELL       │             │ - Risk checks        │
   └──────────────────┘             └──────────────────────┘
         ↑                                    ↑
         │                                    │
         └────────────────┬──────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ↓                                 ↓
   ┌──────────────────┐        ┌──────────────────────┐
   │ Statistics       │        │ Analysis Modules     │
   │ Optimizer        │        │ (technical, sentiment)│
   │ (stats_opt.py)   │        │                      │
   │                  │        │ - Calculate scores   │
   │ - Track outcomes │        │ - Generate factors   │
   │ - Calculate KPIs │        │ - Estimate confidence│
   └──────────────────┘        └──────────────────────┘
```

## Integration Steps

### Step 1: Generate Trading Signals

Use the signal_generator to create trading signals:

```python
from indicator_collector.trading_system import (
    signal_generator,
    AnalyzerContext,
)

# Create context from market data
context = AnalyzerContext(
    symbol="BTCUSDT",
    timeframe="1h",
    timestamp=int(datetime.now().timestamp() * 1000),
    current_price=45000.0,
    ohlcv={
        "open": 44900.0,
        "high": 45100.0,
        "low": 44800.0,
        "close": 45000.0,
        "volume": 1000000.0,
    },
    indicators={
        "rsi": 28.5,
        "macd": -50.0,
        "atr": 500.0,
        "trend_strength": 65.0,
    },
    # ... other context fields
)

# Generate signal
signal = signal_generator.generate_signal(context, weights={
    "technical": 1.0,
    "sentiment": 0.8,
    # ... other weights
})
```

### Step 2: Create Position Plan

Use the position manager to create a position plan:

```python
from indicator_collector.trading_system import (
    create_position_plan,
    PositionManagerConfig,
    create_diversification_guard,
)

config = PositionManagerConfig(
    max_position_size_usd=1000.0,
    max_risk_per_trade_pct=0.02,
    max_concurrent_same_direction=3,
)

guard = create_diversification_guard()

result = create_position_plan(
    context=context,
    signal_direction=signal.signal_type.lower(),
    config=config,
    account_balance=10000.0,
    diversification_guard=guard,
)

if result.can_trade:
    position_plan = result.position_plan
    holding_horizon = result.holding_horizon_bars
```

### Step 3: Add Performance Metrics

Track signal performance using the statistics optimizer:

```python
from indicator_collector.trading_system import (
    create_stats_optimizer,
    StatsOptimizerConfig,
)

config = StatsOptimizerConfig(
    min_win_rate_target=0.60,
    min_profit_factor_target=1.8,
)

optimizer = create_stats_optimizer(config)

# When signal closes, record outcome
outcome = SignalOutcome(
    signal_type=signal.signal_type,
    entry_price=position_plan.entry_price,
    exit_price=exit_price,
    pnl_pct=pnl_percentage,
    success=(pnl_percentage > 0),
    factors=[
        {"factor_name": factor.factor_name, "score": factor.score}
        for factor in signal.factors
    ],
)

optimizer.add_signal_outcome(outcome)

# Calculate KPIs
kpis = optimizer.calculate_kpis()

# Get optimization stats
optimization_stats = OptimizationStats(
    backtest_win_rate=kpis.win_rate * 100,
    avg_profit_pct=kpis.avg_profit_pct,
    avg_loss_pct=kpis.avg_loss_pct,
    sharpe_ratio=kpis.sharpe_ratio,
    total_signals=kpis.total_signals,
    profitable_signals=kpis.profitable_signals,
    losing_signals=kpis.losing_signals,
)
```

### Step 4: Integrate with Collector

Add trading signals to the collector payload:

```python
def collect_metrics(symbol: str, timeframe: str, ...) -> CollectorResult:
    # ... existing collection code ...
    
    # Generate trading signal
    trading_signal = generate_signal_for_context(context)
    
    # Build payload
    payload = {
        "metadata": metadata,
        "latest": latest_data,
        "advanced": advanced_data,
        # ... other sections ...
        
        # NEW: Add trading signals section
        "automated_signals": trading_signal.to_dict() if trading_signal else None,
    }
    
    return CollectorResult(
        summary=summary,
        payload=payload,
        main_series=series,
    )
```

### Step 5: Display in Web UI

The web_ui.py automatically displays the `automated_signals` section in the new tab. No changes needed!

The tab looks for signals in this order:
1. `payload["automated_signals"]`
2. `payload["trading_signals"]`
3. `payload["advanced"]["trade_plan"]` (fallback)

## Data Format Specification

### Complete Signal Object

```json
{
  "signal_type": "BUY",
  "confidence": 0.85,
  "timestamp": 1699000000000,
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "factors": [
    {
      "factor_name": "rsi_oversold",
      "score": 85.0,
      "weight": 2.0,
      "description": "RSI at 28 (oversold)",
      "emoji": "🟢",
      "metadata": {
        "rsi_value": 28.0,
        "threshold": 30.0
      }
    }
  ],
  "position_plan": {
    "entry_price": 45000.0,
    "stop_loss": 44000.0,
    "take_profit_levels": [46000.0, 47000.0, 48000.0],
    "position_size_usd": 1000.0,
    "risk_reward_ratio": 2.0,
    "max_risk_pct": 0.02,
    "leverage": 10.0,
    "direction": "long",
    "notes": "Holding horizon: 24 bars"
  },
  "explanation": {
    "primary_reason": "RSI oversold at 28, expecting bounce",
    "supporting_factors": [
      "Price near support level",
      "Volume increasing"
    ],
    "risk_factors": [
      "Could continue lower in strong downtrend",
      "Liquidation risk on leverage"
    ],
    "market_context": "Timeframe: 1h, Price: $45,000"
  },
  "optimization_stats": {
    "backtest_win_rate": 62.5,
    "avg_profit_pct": 2.1,
    "avg_loss_pct": -0.8,
    "sharpe_ratio": 1.8,
    "total_signals": 40,
    "profitable_signals": 25,
    "losing_signals": 15
  },
  "metadata": {}
}
```

### Minimal Signal Object

For cases where you only have basic data:

```json
{
  "signal_type": "BUY",
  "confidence": 0.8,
  "timestamp": 1699000000000,
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "factors": []
}
```

The web UI will gracefully handle missing optional fields.

## Configuration Examples

### Example 1: Simple RSI-Based Signal

```python
# Simple RSI analyzer
context = create_context_from_market_data(...)

rsi = context.indicators.get("rsi")
if rsi < 30:
    signal = TradingSignalPayload(
        signal_type="BUY",
        confidence=min(1.0, (30 - rsi) / 30),
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=[
            FactorScore(
                factor_name="rsi_oversold",
                score=100 - (rsi / 30 * 100),
                weight=1.0,
                description=f"RSI at {rsi:.1f}",
                emoji="🟢",
            )
        ],
    )
```

### Example 2: Multi-Factor Signal

```python
# Combine multiple factors
factors = []

# Technical factor
technical_score = calculate_technical_score(context)
factors.append(FactorScore(
    factor_name="technical_analysis",
    score=technical_score,
    weight=2.0,
    description="Multiple indicators aligned",
))

# Sentiment factor
sentiment_score = get_fear_greed_score()
factors.append(FactorScore(
    factor_name="sentiment",
    score=sentiment_score,
    weight=1.5,
    description=f"Fear & Greed at {sentiment_score:.0f}",
))

# Calculate combined confidence
total_weight = sum(f.weight for f in factors)
weighted_score = sum(f.score * f.weight for f in factors) / total_weight
confidence = min(1.0, weighted_score / 100)

signal = TradingSignalPayload(
    signal_type="BUY" if weighted_score > 60 else "NEUTRAL",
    confidence=confidence,
    timestamp=context.timestamp,
    symbol=context.symbol,
    timeframe=context.timeframe,
    factors=factors,
)
```

### Example 3: Signal with Position Plan

```python
# Create signal with complete position plan
atr = context.indicators.get("atr", context.current_price * 0.02)
entry_price = context.current_price
stop_loss = entry_price - (atr * 1.5)
tp_levels = [
    entry_price + (atr * 2.0),
    entry_price + (atr * 3.0),
    entry_price + (atr * 5.0),
]

signal = TradingSignalPayload(
    signal_type="BUY",
    confidence=0.85,
    timestamp=context.timestamp,
    symbol=context.symbol,
    timeframe=context.timeframe,
    factors=[...],
    position_plan=PositionPlan(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_levels=tp_levels,
        position_size_usd=1000.0,
        risk_reward_ratio=2.0,
        max_risk_pct=0.02,
        leverage=10.0,
        direction="long",
    ),
)
```

## Testing Integration

### Unit Tests

```python
def test_signal_generation():
    """Test signal generation."""
    context = create_test_context()
    signal = generate_signal(context)
    
    assert signal.signal_type in ["BUY", "SELL", "NEUTRAL"]
    assert 0 <= signal.confidence <= 1
    assert signal.symbol == context.symbol
    assert signal.timeframe == context.timeframe
    assert signal.factors is not None
    assert signal.position_plan is not None or signal.signal_type == "NEUTRAL"

def test_signal_serialization():
    """Test signal can be serialized to JSON."""
    signal = create_sample_signal()
    json_str = json.dumps(signal.to_dict())
    data = json.loads(json_str)
    
    # Verify round-trip
    signal2 = TradingSignalPayload.from_dict(data)
    assert signal.signal_type == signal2.signal_type
    assert signal.confidence == signal2.confidence
```

### Integration Tests

```python
def test_collector_includes_signals():
    """Test collector payload includes signals."""
    result = collect_metrics(
        symbol="BTCUSDT",
        timeframe="1h",
        period=100,
        token="test",
    )
    
    payload = result.payload
    assert "automated_signals" in payload or "trading_signals" in payload

def test_web_ui_displays_signals(tmpdir):
    """Test web UI can display signals."""
    payload = create_test_payload_with_signals()
    
    # Verify data structure
    signal_data = payload.get("automated_signals", {})
    assert signal_data.get("signal_type") in ["BUY", "SELL", "NEUTRAL"]
    assert "position_plan" in signal_data
    assert "explanation" in signal_data
```

## Performance Considerations

### Optimization

1. **Cache Signal Calculations:** Cache complex factor calculations
   ```python
   @lru_cache(maxsize=128)
   def calculate_technical_score(price_data):
       # Expensive calculation
       return score
   ```

2. **Lazy Load Performance Data:** Only calculate stats when needed
   ```python
   @property
   def optimization_stats(self):
       if self._stats_cache is None:
           self._stats_cache = self._calculate_stats()
       return self._stats_cache
   ```

3. **Batch Signal Generation:** Generate signals for multiple symbols efficiently
   ```python
   signals = []
   for symbol in symbols:
       signal = generate_signal(get_context(symbol))
       signals.append(signal)
   ```

### Caching Strategy

```python
# Cache configuration
SIGNAL_CACHE_TTL = 300  # 5 minutes
POSITION_PLAN_CACHE_TTL = 600  # 10 minutes
STATS_CACHE_TTL = 3600  # 1 hour

# Invalidation triggers
CACHE_INVALIDATE_ON = [
    "new_candle",
    "high_volatility",
    "market_hours_change",
]
```

## Troubleshooting Integration

### Signals Not Appearing

1. Check `payload["automated_signals"]` exists:
   ```python
   print("automated_signals" in payload)
   print(payload.get("automated_signals"))
   ```

2. Verify signal structure:
   ```python
   signal = payload["automated_signals"]
   required_fields = ["signal_type", "confidence", "timestamp", "symbol"]
   for field in required_fields:
       assert field in signal, f"Missing {field}"
   ```

3. Check web UI logs:
   ```bash
   streamlit run web_ui.py --logger.level=debug
   ```

### Performance Issues

1. Profile signal generation:
   ```python
   import cProfile
   cProfile.run('generate_signal(context)')
   ```

2. Check for slow calculations:
   ```python
   import time
   start = time.time()
   signal = generate_signal(context)
   print(f"Signal generation: {time.time() - start:.3f}s")
   ```

### Data Inconsistencies

1. Validate signal data:
   ```python
   def validate_signal(signal: TradingSignalPayload) -> bool:
       if signal.signal_type not in ["BUY", "SELL", "NEUTRAL"]:
           return False
       if not 0 <= signal.confidence <= 1:
           return False
       if signal.position_plan:
           if signal.position_plan.entry_price <= 0:
               return False
       return True
   ```

2. Add logging:
   ```python
   logger.info(f"Generated {signal.signal_type} signal")
   logger.debug(f"Confidence: {signal.confidence:.2%}")
   logger.debug(f"Position size: ${signal.position_plan.position_size_usd}")
   ```

## API Reference

### Signal Generation

```python
def generate_signal(
    context: AnalyzerContext,
    weights: Dict[str, float] = None,
    min_factors: int = 1,
) -> TradingSignalPayload:
    """
    Generate a trading signal from market context.
    
    Args:
        context: Market data and analysis context
        weights: Factor weights for combination
        min_factors: Minimum factors for valid signal
    
    Returns:
        TradingSignalPayload with complete signal
    """
```

### Position Planning

```python
def create_position_plan(
    context: AnalyzerContext,
    signal_direction: str,
    config: PositionManagerConfig,
    account_balance: float,
    diversification_guard: DiversificationGuard = None,
) -> PositionManagerResult:
    """
    Create a position plan for a signal.
    
    Args:
        context: Market context
        signal_direction: "long" or "short"
        config: Position manager configuration
        account_balance: Account balance in USD
        diversification_guard: Tracks current positions
    
    Returns:
        PositionManagerResult with plan or rejection reasons
    """
```

### Performance Tracking

```python
def create_stats_optimizer(
    config: StatsOptimizerConfig = None,
) -> StatisticsOptimizer:
    """
    Create a statistics optimizer for tracking signal performance.
    
    Args:
        config: Optimizer configuration
    
    Returns:
        StatisticsOptimizer instance
    """

def add_signal_outcome(
    optimizer: StatisticsOptimizer,
    outcome: SignalOutcome,
) -> None:
    """Track an outcome for a signal."""

def calculate_kpis(
    optimizer: StatisticsOptimizer,
) -> PerformanceKPIs:
    """Calculate comprehensive performance metrics."""
```

## Version Compatibility

- **Trading System Version:** 1.0+
- **Web UI Version:** Compatible with latest
- **Python Version:** 3.9+
- **Dependencies:** streamlit, plotly, pandas

## Support & Documentation

- **API Reference:** See `indicator_collector/trading_system/interfaces.py`
- **Examples:** See `example_trading_system_usage.py`
- **Tests:** See `tests/test_web_ui_automated_signals.py`
- **User Guide:** See `AUTOMATED_SIGNALS_TAB_USAGE.md`

---

**Version:** 1.0  
**Last Updated:** 2024
