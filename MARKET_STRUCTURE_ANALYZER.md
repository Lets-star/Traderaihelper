# Market Structure Analyzer

## Overview

The Market Structure Analyzer is a comprehensive tool for analyzing price action and market structure patterns in cryptocurrency trading. It leverages the `advanced_metrics.calculate_market_structure` function to identify swing points, support/resistance levels, liquidity zones, and classify market trends.

## Features

- **Swing Point Detection**: Identifies Higher Highs (HH), Higher Lows (HL), Lower Highs (LH), and Lower Lows (LL)
- **Trend Classification**: Automatically classifies market as bullish, bearish, or neutral/sideways
- **Support/Resistance Levels**: Identifies key price levels with strength ratings
- **Liquidity Zones**: Detects areas of high liquidity based on volume profile
- **Normalized Scoring**: Produces a 0-1 score representing market structure strength
- **Structured Highlights**: Generates human-readable explanations of the analysis
- **30% Weight Factor**: Designed to contribute 30% to overall trading signal confidence

## Installation

The analyzer is part of the `indicator_collector.trading_system` package:

```python
from indicator_collector.trading_system import (
    analyze_market_structure,
    calculate_structure_score,
)
```

## Usage

### Standalone Usage with Candles

```python
from indicator_collector.math_utils import Candle
from indicator_collector.trading_system import calculate_structure_score

# Create or fetch candle data
candles = [...]  # List of Candle objects

# Calculate structure score
factor_score = calculate_structure_score(candles)

print(f"Score: {factor_score.score:.3f}")
print(f"Trend: {factor_score.metadata['trend']}")
print(f"Description: {factor_score.description}")
```

### Usage with AnalyzerContext

```python
from indicator_collector.trading_system import (
    AnalyzerContext,
    analyze_market_structure,
)

# Create context (typically from collector output)
context = AnalyzerContext(
    symbol='BTCUSDT',
    timeframe='1h',
    timestamp=1699000000000,
    current_price=50000.0,
    ohlcv={'open': 49900, 'high': 50100, 'low': 49800, 'close': 50000, 'volume': 1000000},
    indicators={},
    extras={'candles': candles},  # Provide historical candles
)

# Analyze market structure
factor_score = analyze_market_structure(context)
```

### Accessing Results

```python
# Basic information
factor_score.factor_name  # 'market_structure'
factor_score.score        # 0.0 to 1.0 (0=bearish, 0.5=neutral, 1=bullish)
factor_score.weight       # 0.3 (30% contribution)
factor_score.emoji        # '🟢' (bullish), '⚪' (neutral), '🔴' (bearish)
factor_score.description  # Human-readable description

# Metadata
metadata = factor_score.metadata
trend = metadata['trend']  # 'bullish', 'bearish', or 'neutral'

# Swing point counts
swing_points = metadata['swing_points']
hh_count = swing_points['hh_count']  # Higher Highs
hl_count = swing_points['hl_count']  # Higher Lows
lh_count = swing_points['lh_count']  # Lower Highs
ll_count = swing_points['ll_count']  # Lower Lows

# Key levels
key_levels = metadata['key_levels']
support_count = key_levels['support_count']
resistance_count = key_levels['resistance_count']

# Highlights (list of strings explaining the analysis)
for highlight in metadata['highlights']:
    print(f"• {highlight}")

# Score components (detailed breakdown)
components = metadata['score_components']
print(f"Base: {components['base_score']}")
print(f"Swing Adjustment: {components['swing_adjustment']}")
print(f"S/R Adjustment: {components['sr_adjustment']}")
print(f"Clarity: {components['clarity']}")
```

## Score Interpretation

The analyzer produces a normalized score between 0 and 1:

### Strong Bullish (0.80 - 1.00) 🟢
- Clear Higher Highs and Higher Lows pattern
- Strong support levels identified
- High pattern clarity
- **Trading Implication**: Favorable for long positions; look for pullbacks to support

### Moderately Bullish (0.60 - 0.79) 🟢
- Bullish trend with some mixed signals
- Good swing structure but less clear
- **Trading Implication**: Consider long positions with tight stops; monitor for continuation

### Neutral/Sideways (0.40 - 0.59) ⚪
- No clear directional bias
- Equal distribution of bullish/bearish swings
- Range-bound market
- **Trading Implication**: Wait for breakout or trade the range boundaries

### Moderately Bearish (0.21 - 0.39) 🔴
- Bearish trend with some mixed signals
- Lower Highs and Lower Lows present
- **Trading Implication**: Consider short positions or stay out; watch for reversal signals

### Strong Bearish (0.00 - 0.20) 🔴
- Clear Lower Highs and Lower Lows pattern
- Strong resistance levels identified
- High pattern clarity
- **Trading Implication**: Favorable for short positions; look for rallies to fade

## How It Works

### 1. Swing Point Detection

The analyzer uses the `calculate_market_structure` function which:
1. Identifies swing highs where price is higher than surrounding candles within a lookback window
2. Identifies swing lows where price is lower than surrounding candles
3. Labels consecutive swing points as HH/HL (bullish) or LH/LL (bearish)

### 2. Trend Classification

Trends are classified based on swing point patterns:
- **Bullish**: Presence of both HH and HL points
- **Bearish**: Presence of both LH and LL points
- **Neutral**: No clear HH/HL or LH/LL pattern

### 3. Score Calculation

The final score is computed from multiple components:

```python
normalized_score = base_score + swing_adjustment + sr_adjustment + liquidity_bonus
```

Where:
- **Base Score**: 0.75 (bullish), 0.50 (neutral), 0.25 (bearish)
- **Swing Adjustment**: ±0.15 based on swing point dominance
- **S/R Adjustment**: ±0.10 based on support vs resistance ratio
- **Liquidity Bonus**: Up to +0.05 for strong liquidity zones

### 4. Highlights Generation

The analyzer generates structured highlights explaining:
- Trend confirmation with swing point counts
- Swing dominance (bullish vs bearish)
- Key support and resistance levels
- Liquidity zone presence
- Pattern clarity assessment
- Trading recommendations based on score

## Integration with Trading System

The Market Structure Analyzer is designed to integrate with a larger trading system:

### Weight Contribution
- **30% weight** in overall signal confidence
- Combine with other factors:
  - Volume analysis (CVD, smart money, liquidity)
  - Momentum indicators (RSI, MACD, divergence)
  - Market breadth and sentiment
  - On-chain metrics

### Example Integration

```python
# Calculate multiple factors
structure_factor = calculate_structure_score(candles)  # weight=0.3
volume_factor = calculate_volume_score(candles)        # weight=0.25
momentum_factor = calculate_momentum_score(candles)     # weight=0.25
sentiment_factor = calculate_sentiment_score(data)      # weight=0.20

# Weighted average
total_score = (
    structure_factor.score * structure_factor.weight +
    volume_factor.score * volume_factor.weight +
    momentum_factor.score * momentum_factor.weight +
    sentiment_factor.score * sentiment_factor.weight
)

# Generate trading signal
if total_score >= 0.7:
    signal = "BUY"
elif total_score <= 0.3:
    signal = "SELL"
else:
    signal = "NEUTRAL"
```

## Testing

Comprehensive tests are included in `test_trading_system.py`:

```bash
python test_trading_system.py
```

Tests cover:
- ✅ Bullish structure detection (HH/HL patterns)
- ✅ Bearish structure detection (LH/LL patterns)
- ✅ Neutral/sideways structure detection
- ✅ Usage with AnalyzerContext
- ✅ Edge cases (insufficient data, empty candles)

All tests use synthetic candle data to verify correct pattern detection.

## Examples

See `example_market_structure_analyzer.py` for complete working examples:

```bash
python example_market_structure_analyzer.py
```

Examples demonstrate:
1. Standalone usage with candles
2. Usage with AnalyzerContext
3. Score interpretation and trading implications

## API Reference

### `calculate_structure_score(candles, volume_analysis=None)`

Standalone function to calculate market structure score from candles.

**Parameters:**
- `candles` (Sequence[Candle]): Historical price data
- `volume_analysis` (Optional[Dict]): Volume analysis data for liquidity zones

**Returns:**
- `FactorScore`: Score object with metadata

### `analyze_market_structure(context)`

Analyze market structure from an AnalyzerContext.

**Parameters:**
- `context` (AnalyzerContext): Trading context with OHLCV, indicators, and optionally candles

**Returns:**
- `FactorScore`: Score object with metadata

## Advanced Usage

### Custom Score Interpretation

```python
factor_score = calculate_structure_score(candles)
score = factor_score.score

if score >= 0.8:
    position_size = "Full size"
    confidence = "High"
elif score >= 0.6:
    position_size = "Half size"
    confidence = "Medium"
else:
    position_size = "Stay out"
    confidence = "Low"

print(f"Confidence: {confidence}, Position: {position_size}")
```

### Extracting Specific Details

```python
# Get detailed market structure data
structure_data = factor_score.metadata['market_structure_data']
swing_points = structure_data['swing_points']

# Access individual swing points
for hh in swing_points['hh']:
    print(f"Higher High at {hh['price']:.2f} on {hh['time_iso']}")

for hl in swing_points['hl']:
    print(f"Higher Low at {hl['price']:.2f} on {hl['time_iso']}")

# Get support/resistance levels
key_levels = structure_data['key_levels']
for support in key_levels['support']:
    print(f"Support: ${support['price']:.2f} (strength: {support['strength']:.2f})")
```

## Limitations

1. **Lookback Period**: Swing point detection requires sufficient candles (minimum 7, recommended 50+)
2. **Lagging Nature**: Structure confirmation occurs after swing points form
3. **False Signals**: Choppy markets may produce unclear patterns (check `clarity` metric)
4. **No Volume Required**: Basic analysis works without volume, but liquidity zones require volume data

## Best Practices

1. **Combine with Other Factors**: Don't rely solely on structure; use as part of a multi-factor system
2. **Check Pattern Clarity**: Review `score_components['clarity']` to assess pattern reliability
3. **Read Highlights**: Use structured highlights to understand the reasoning behind the score
4. **Appropriate Timeframes**: Works best on 1h, 4h, and 1d timeframes for swing trading
5. **Monitor Trend Changes**: Watch for transitions between bullish/bearish/neutral states

## Contributing

When extending or modifying the analyzer:
1. Maintain the 0-1 normalized score range
2. Keep the 30% weight constant
3. Update tests for any scoring logic changes
4. Document new highlights in the README
5. Ensure backward compatibility with AnalyzerContext

## License

Part of the indicator_collector project.
