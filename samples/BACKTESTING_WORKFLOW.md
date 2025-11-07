# Backtesting Workflow Guide

This guide walks through the complete backtesting workflow for testing trading strategies using the automated trading system.

## Overview

The backtesting workflow consists of four main phases:

1. **Data Preparation**: Gather and format historical market data
2. **Signal Generation**: Generate trading signals for each bar
3. **Outcome Recording**: Record whether each signal was profitable
4. **Performance Analysis**: Calculate metrics and optimize weights

## Phase 1: Data Preparation

### Gather Historical OHLCV Data

```python
from indicator_collector.data_fetcher import fetch_binance_ohlcv

# Fetch historical data from Binance
symbol = "BTCUSDT"
timeframe = "1h"
bars_needed = 1000

ohlcv_data = fetch_binance_ohlcv(
    symbol=f"BINANCE:{symbol}",
    timeframe=timeframe,
    limit=bars_needed,
    offline=False  # Use real data, not synthetic
)

print(f"Fetched {len(ohlcv_data)} bars of {symbol} {timeframe} data")
```

### Compute Technical Indicators

```python
from indicator_collector.indicator_metrics import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
)

# Process each bar and calculate indicators
indicators_history = []

for i, bar in enumerate(ohlcv_data):
    if i < 20:  # Skip bars with insufficient lookback
        continue
    
    recent_closes = [b['close'] for b in ohlcv_data[i-20:i+1]]
    recent_highs = [b['high'] for b in ohlcv_data[i-20:i+1]]
    recent_lows = [b['low'] for b in ohlcv_data[i-20:i+1]]
    
    indicators = {
        'rsi': calculate_rsi(recent_closes, period=14),
        'macd': calculate_macd(recent_closes),
        'bollinger': calculate_bollinger_bands(recent_closes, period=20),
        'atr': calculate_atr(recent_highs, recent_lows, recent_closes),
    }
    
    indicators_history.append({
        'bar': bar,
        'indicators': indicators,
    })

print(f"Calculated indicators for {len(indicators_history)} bars")
```

## Phase 2: Signal Generation

### Create Your Trading Strategy

```python
from indicator_collector.trading_system import (
    TradingAnalyzer,
    AnalyzerContext,
    TradingSignalPayload,
    FactorScore,
)

class MyRSIStrategy(TradingAnalyzer):
    """Simple RSI-based strategy for demonstration."""
    
    def __init__(self, rsi_oversold=30, rsi_overbought=70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def analyze(self, context: AnalyzerContext) -> TradingSignalPayload:
        """Generate signal based on RSI."""
        rsi = context.indicators.get('rsi')
        
        if rsi is None:
            return self._neutral_signal(context, "No RSI data")
        
        if rsi < self.rsi_oversold:
            return self._buy_signal(context, rsi)
        elif rsi > self.rsi_overbought:
            return self._sell_signal(context, rsi)
        else:
            return self._neutral_signal(context, f"RSI neutral: {rsi:.1f}")
    
    def _buy_signal(self, context: AnalyzerContext, rsi: float) -> TradingSignalPayload:
        """Create BUY signal."""
        from indicator_collector.trading_system import (
            SignalExplanation,
            PositionPlan,
        )
        
        atr = context.indicators.get('atr', context.current_price * 0.02)
        entry_price = context.current_price
        stop_loss = entry_price - (atr * 2)
        tp_levels = [
            entry_price + (atr * 2),
            entry_price + (atr * 4),
            entry_price + (atr * 6),
        ]
        
        return TradingSignalPayload(
            signal_type="BUY",
            confidence=min((self.rsi_oversold - rsi) / self.rsi_oversold, 1.0),
            timestamp=context.current_time,
            factors=[
                FactorScore(
                    factor_name="rsi_oversold",
                    score=100.0 * (1.0 - rsi / self.rsi_oversold),
                    weight=2.0,
                    description=f"RSI: {rsi:.1f}",
                    emoji="🟢",
                )
            ],
            position_plan=PositionPlan(
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                take_profit_levels=tp_levels,
            ),
            explanation=SignalExplanation(
                primary_reason=f"RSI oversold at {rsi:.1f}",
                supporting_factors=["Potential bounce from support"],
                risk_factors=["Could break support further"],
            ),
        )
    
    def _sell_signal(self, context: AnalyzerContext, rsi: float) -> TradingSignalPayload:
        # Similar to _buy_signal but for shorts
        pass
    
    def _neutral_signal(self, context: AnalyzerContext, reason: str) -> TradingSignalPayload:
        from indicator_collector.trading_system import SignalExplanation, PositionPlan
        
        return TradingSignalPayload(
            signal_type="NEUTRAL",
            confidence=0.5,
            timestamp=context.current_time,
            factors=[],
            position_plan=PositionPlan(),
            explanation=SignalExplanation(primary_reason=reason),
        )
```

### Generate Signals for All Historical Bars

```python
from indicator_collector.trading_system import AnalyzerContext

strategy = MyRSIStrategy()
signals = []

for bar_data in indicators_history:
    bar = bar_data['bar']
    indicators = bar_data['indicators']
    
    # Create context
    context = AnalyzerContext(
        symbol="BTCUSDT",
        timeframe="1h",
        current_price=bar['close'],
        current_time=int(bar['time'] * 1000),  # Convert to milliseconds
        indicators=indicators,
    )
    
    # Generate signal
    signal = strategy.analyze(context)
    signals.append({
        'bar': bar,
        'signal': signal,
    })

print(f"Generated {len(signals)} signals")

# Show signal distribution
buy_count = sum(1 for s in signals if s['signal'].signal_type == 'BUY')
sell_count = sum(1 for s in signals if s['signal'].signal_type == 'SELL')
neutral_count = sum(1 for s in signals if s['signal'].signal_type == 'NEUTRAL')

print(f"Signal distribution: {buy_count} BUY, {sell_count} SELL, {neutral_count} NEUTRAL")
```

## Phase 3: Outcome Recording

### Evaluate Signal Performance

```python
from indicator_collector.trading_system import StatisticsOptimizer

optimizer = StatisticsOptimizer()

# For each signal, determine if it was profitable
for i, signal_data in enumerate(signals[:-1]):  # Leave last bar for lookahead
    signal = signal_data['signal']
    current_bar = signal_data['bar']
    next_bars = ohlcv_data[i+1:min(i+20, len(ohlcv_data))]  # Look ahead 20 bars
    
    if signal.signal_type == 'NEUTRAL':
        continue
    
    # Calculate P&L for this signal
    entry_price = signal.position_plan.entry_price or current_bar['close']
    
    # Find exit (either TP or SL hit)
    exited = False
    pnl_pct = 0
    
    for lookahead_bar in next_bars:
        if signal.signal_type == 'BUY':
            # Check for TP or SL
            if signal.position_plan.take_profit_levels:
                if lookahead_bar['high'] >= signal.position_plan.take_profit_levels[0]:
                    pnl_pct = (signal.position_plan.take_profit_levels[0] - entry_price) / entry_price * 100
                    exited = True
                    break
            
            if signal.position_plan.stop_loss_price:
                if lookahead_bar['low'] <= signal.position_plan.stop_loss_price:
                    pnl_pct = (signal.position_plan.stop_loss_price - entry_price) / entry_price * 100
                    exited = True
                    break
        
        elif signal.signal_type == 'SELL':
            # Similar logic for short
            pass
    
    # Record outcome
    outcome = 1.0 if pnl_pct > 0 else -1.0
    
    optimizer.record_signal_outcome(
        signal_type=signal.signal_type,
        pnl_pct=pnl_pct,
        context=None,  # Can pass context if available
    )

print("Outcomes recorded")
```

## Phase 4: Performance Analysis

### Calculate Performance Metrics

```python
stats = optimizer.get_statistics()

print("=== Backtest Results ===")
print(f"Total Signals: {stats.total_signals}")
print(f"Profitable: {stats.profitable_signals}")
print(f"Losing: {stats.losing_signals}")
print(f"Win Rate: {stats.backtest_win_rate:.1f}%")
print(f"Average Profit: {stats.avg_profit_pct:.2f}%")
print(f"Average Loss: {stats.avg_loss_pct:.2f}%")
print(f"Profit Factor: {stats.profit_factor:.2f}")
print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
```

### Optimize Strategy Weights

```python
# The StatisticsOptimizer can suggest weight adjustments
suggestions = optimizer.suggest_weight_adjustments()

print("=== Weight Optimization Suggestions ===")
for factor, adjustment in suggestions.items():
    print(f"{factor}: {adjustment:+.2f}")

# Apply suggestions to strategy
# Example:
# if suggestions.get('rsi_oversold', 0) > 0:
#     strategy.rsi_oversold -= 5  # Make threshold more sensitive
```

## Complete Backtesting Example

```python
def run_backtest(
    symbol: str,
    timeframe: str,
    strategy_class,
    bars_needed: int = 1000,
    test_period_days: int = 30,
):
    """Run complete backtest of a strategy."""
    
    from indicator_collector.data_fetcher import fetch_binance_ohlcv
    from indicator_collector.trading_system import StatisticsOptimizer, AnalyzerContext
    
    print(f"Starting backtest for {symbol} {timeframe}...")
    
    # Phase 1: Data
    print("Phase 1: Fetching historical data...")
    ohlcv_data = fetch_binance_ohlcv(
        symbol=f"BINANCE:{symbol}",
        timeframe=timeframe,
        limit=bars_needed,
    )
    
    # Phase 2: Signals
    print("Phase 2: Generating signals...")
    strategy = strategy_class()
    optimizer = StatisticsOptimizer()
    
    signals = []
    for i in range(20, len(ohlcv_data)-1):
        bar = ohlcv_data[i]
        
        context = AnalyzerContext(
            symbol=symbol,
            timeframe=timeframe,
            current_price=bar['close'],
            current_time=int(bar['time'] * 1000),
            indicators={'rsi': calculate_rsi([b['close'] for b in ohlcv_data[max(0,i-14):i+1]])},
        )
        
        signal = strategy.analyze(context)
        signals.append((bar, signal))
    
    # Phase 3 & 4: Outcomes and Analysis
    print("Phase 3-4: Recording outcomes and analyzing...")
    
    for i, (bar, signal) in enumerate(signals[:-1]):
        if signal.signal_type == 'NEUTRAL':
            continue
        
        # Simplified outcome: random P&L for demo
        # In real backtest, calculate from lookahead bars
        pnl = 1.0 if i % 2 == 0 else -1.0
        optimizer.record_signal_outcome(signal.signal_type, pnl, None)
    
    stats = optimizer.get_statistics()
    
    print("\n=== Backtest Results ===")
    print(f"Total Signals: {stats.total_signals}")
    print(f"Win Rate: {stats.backtest_win_rate:.1f}%")
    print(f"Profit Factor: {stats.profit_factor:.2f}")
    print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
    
    return stats

# Run backtest
results = run_backtest("BTCUSDT", "1h", MyRSIStrategy)
```

## Best Practices

1. **Use Sufficient Lookback**: Ensure you have enough historical data for indicator calculation (at least 100 bars minimum, 500+ recommended)

2. **Avoid Lookahead Bias**: Never use future data when generating signals

3. **Commission and Slippage**: Consider adding realistic fees:
   ```python
   commission_pct = 0.1  # 0.1% per trade
   slippage_pct = 0.05   # 0.05% execution slippage
   
   pnl_after_fees = pnl_pct - (2 * commission_pct + slippage_pct)
   ```

4. **Multiple Timeframes**: Test across different timeframes to ensure robustness

5. **Out-of-Sample Testing**: Reserve recent data for validation without curve-fitting

6. **Optimize Carefully**: Don't over-optimize on historical data (overfitting risk)

7. **Monitor Regime Changes**: Markets change; periodically re-backtest and re-optimize

## Sample Backtesting Commands

```bash
# Run basic backtest
python3 -c "from samples.backtest_example import run_backtest; run_backtest('BTCUSDT', '1h')"

# Run backtest with multiple symbols
python3 -c "
from samples.backtest_example import run_backtest
for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
    print(f'\nBacktesting {symbol}...')
    run_backtest(symbol, '1h')
"
```

## Troubleshooting

### "No data available"
- Check internet connection for Binance access
- Use offline mode with synthetic data for testing

### "Insufficient bars for indicators"
- Increase `bars_needed` parameter (e.g., 500 or 1000)
- Ensure you have at least 20 bars before generating first signal

### "Unrealistic results"
- Check for lookahead bias
- Add realistic commission/slippage
- Verify signal generation logic
- Check outcome calculation

## Next Steps

1. Implement your own strategy class inheriting from `TradingAnalyzer`
2. Run backtest on your strategy
3. Review performance metrics
4. Optimize weights using `StatisticsOptimizer`
5. Deploy to live trading with appropriate risk management
