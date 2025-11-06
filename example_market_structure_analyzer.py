"""Example usage of the market structure analyzer."""

from __future__ import annotations

from indicator_collector.math_utils import Candle
from indicator_collector.trading_system import (
    AnalyzerContext,
    analyze_market_structure,
    calculate_structure_score,
)


def create_example_bullish_candles() -> list[Candle]:
    """Create example bullish candles for demonstration."""
    base_time = 1699000000000
    base_price = 50000.0
    
    swing_pattern = [
        # First swing: rise
        (0, 100, 150), (100, 200, 250), (200, 300, 350), (300, 400, 450), (400, 500, 550),
        # Pullback
        (500, 400, 450), (400, 350, 400), (350, 300, 350),
        # Second swing: rise to higher high
        (300, 400, 450), (400, 500, 550), (500, 600, 650), (600, 700, 750), (700, 800, 850),
        # Pullback to higher low
        (800, 700, 750), (700, 650, 700), (650, 600, 650),
        # Third swing: rise to even higher high
        (600, 700, 750), (700, 800, 850), (800, 900, 950), (900, 1000, 1050), (1000, 1100, 1150),
    ]
    
    candles = []
    for i, (low_offset, close_offset, high_offset) in enumerate(swing_pattern):
        time_offset = i * 3600000
        candles.append(Candle(
            open_time=base_time + time_offset,
            close_time=base_time + time_offset + 3600000,
            open=base_price + close_offset - 25,
            high=base_price + high_offset,
            low=base_price + low_offset,
            close=base_price + close_offset,
            volume=1000000.0 + (i % 5) * 200000,
        ))
    
    return candles


def example_standalone_usage():
    """Example: Using calculate_structure_score directly with candles."""
    print("=" * 70)
    print("Example 1: Standalone Usage with Candles")
    print("=" * 70)
    
    candles = create_example_bullish_candles()
    
    # Calculate structure score directly from candles
    factor_score = calculate_structure_score(candles)
    
    print(f"\nMarket Structure Analysis:")
    print(f"  Factor Name: {factor_score.factor_name}")
    print(f"  Score: {factor_score.score:.3f} (0=bearish, 0.5=neutral, 1=bullish)")
    print(f"  Weight: {factor_score.weight} (30% of total signal)")
    print(f"  Emoji: {factor_score.emoji}")
    print(f"  Description: {factor_score.description}")
    
    # Access metadata
    metadata = factor_score.metadata
    print(f"\n  Trend: {metadata['trend']}")
    print(f"  Swing Points:")
    print(f"    - Higher Highs (HH): {metadata['swing_points']['hh_count']}")
    print(f"    - Higher Lows (HL): {metadata['swing_points']['hl_count']}")
    print(f"    - Lower Highs (LH): {metadata['swing_points']['lh_count']}")
    print(f"    - Lower Lows (LL): {metadata['swing_points']['ll_count']}")
    
    print(f"\n  Key Levels:")
    print(f"    - Support Levels: {metadata['key_levels']['support_count']}")
    print(f"    - Resistance Levels: {metadata['key_levels']['resistance_count']}")
    print(f"    - Liquidity Zones: {metadata['liquidity_zones_count']}")
    
    # Display highlights
    print(f"\n  Highlights:")
    for highlight in metadata['highlights']:
        print(f"    • {highlight}")
    
    # Display score components
    print(f"\n  Score Breakdown:")
    components = metadata['score_components']
    print(f"    - Base Score: {components['base_score']:.3f}")
    print(f"    - Swing Adjustment: {components['swing_adjustment']:+.3f}")
    print(f"    - S/R Adjustment: {components['sr_adjustment']:+.3f}")
    print(f"    - Liquidity Bonus: {components['liquidity_bonus']:+.3f}")
    print(f"    - Pattern Clarity: {components['clarity']:.3f}")


def example_with_context():
    """Example: Using analyze_market_structure with AnalyzerContext."""
    print("\n" + "=" * 70)
    print("Example 2: Usage with AnalyzerContext")
    print("=" * 70)
    
    candles = create_example_bullish_candles()
    
    # Create an AnalyzerContext (as would come from the collector)
    context = AnalyzerContext(
        symbol='BTCUSDT',
        timeframe='1h',
        timestamp=candles[-1].close_time,
        current_price=candles[-1].close,
        ohlcv={
            'open': candles[-1].open,
            'high': candles[-1].high,
            'low': candles[-1].low,
            'close': candles[-1].close,
            'volume': candles[-1].volume,
        },
        indicators={
            'rsi': 65.0,
            'macd': 50.0,
        },
        # Provide candles in extras for analysis
        extras={'candles': candles},
    )
    
    # Analyze using context
    factor_score = analyze_market_structure(context)
    
    print(f"\nMarket Structure Analysis for {context.symbol} ({context.timeframe}):")
    print(f"  Current Price: ${context.current_price:,.2f}")
    print(f"  Structure Score: {factor_score.score:.3f} {factor_score.emoji}")
    print(f"  {factor_score.description}")
    
    print(f"\n  Top 3 Highlights:")
    for i, highlight in enumerate(factor_score.metadata['highlights'][:3], 1):
        print(f"    {i}. {highlight}")


def example_interpretation():
    """Example: Interpreting scores for trading decisions."""
    print("\n" + "=" * 70)
    print("Example 3: Score Interpretation Guide")
    print("=" * 70)
    
    print("\nScore Ranges and Interpretations:")
    print("  0.80 - 1.00 🟢 Strong Bullish")
    print("    → Clear HH/HL pattern with strong support")
    print("    → Favorable for long positions")
    print("    → Look for pullbacks to enter")
    
    print("\n  0.60 - 0.79 🟢 Moderately Bullish")
    print("    → Bullish trend with some mixed signals")
    print("    → Consider long positions with tight stops")
    print("    → Monitor for continuation or reversal")
    
    print("\n  0.40 - 0.59 ⚪ Neutral/Sideways")
    print("    → No clear directional bias")
    print("    → Range-bound market")
    print("    → Wait for breakout or trade the range")
    
    print("\n  0.21 - 0.39 🔴 Moderately Bearish")
    print("    → Bearish trend with some mixed signals")
    print("    → Consider short positions or stay out")
    print("    → Monitor for reversal signals")
    
    print("\n  0.00 - 0.20 🔴 Strong Bearish")
    print("    → Clear LH/LL pattern with strong resistance")
    print("    → Favorable for short positions")
    print("    → Look for rallies to fade")
    
    print("\nIntegration with Trading System:")
    print("  • Weight: 30% of total signal confidence")
    print("  • Combine with:")
    print("    - Volume analysis (liquidity, CVD, smart money)")
    print("    - Momentum indicators (RSI, MACD)")
    print("    - Market breadth and sentiment")
    print("  • Use highlights to understand the 'why' behind the score")


if __name__ == "__main__":
    print("\n🏗️  Market Structure Analyzer - Usage Examples\n")
    
    example_standalone_usage()
    example_with_context()
    example_interpretation()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print()
