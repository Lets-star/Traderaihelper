#!/usr/bin/env python3
"""
Demonstration of the sentiment analyzer module.

This script shows how to use the sentiment analyzer to combine Alternative.me
fear & greed data with fundamental metrics to produce a comprehensive sentiment score.
"""

from indicator_collector.trading_system import analyze_sentiment_factors, create_sentiment_factor_score
from indicator_collector.math_utils import Candle


def create_sample_candles(count: int = 100, trend: str = "bullish") -> list[Candle]:
    """Create sample candle data for demonstration."""
    candles = []
    price = 50000.0
    
    for i in range(count):
        if trend == "bullish":
            price += 50
        elif trend == "bearish":
            price -= 50
        else:  # neutral
            import math
            price += 20 * math.sin(i * 0.1)
        
        candles.append(Candle(
            open_time=1609459200000 + i * 60000,
            close_time=1609459260000 + i * 60000,
            open=price - 10,
            high=price + 20,
            low=price - 20,
            close=price,
            volume=100.0 + i * 2,
        ))
    
    return candles


def main():
    """Demonstrate sentiment analyzer functionality."""
    print("=" * 60)
    print("Sentiment Analyzer Demonstration")
    print("=" * 60)
    
    # Test with different market conditions
    scenarios = [
        ("Bullish Market", "bullish"),
        ("Bearish Market", "bearish"),
        ("Neutral/Sideways Market", "neutral"),
    ]
    
    for scenario_name, trend in scenarios:
        print(f"\n{scenario_name}:")
        print("-" * 40)
        
        # Create sample data
        candles = create_sample_candles(50, trend)
        
        # Analyze sentiment
        result = analyze_sentiment_factors(candles)
        
        # Display results
        print(f"Sentiment Score: {result['final_score']:.3f}")
        print(f"Direction: {result['direction']} {result['emoji']}")
        print(f"Confidence: {result['confidence']}/100")
        
        # Show component breakdown
        print("\nComponent Breakdown:")
        for component_name, component_data in result['components'].items():
            print(f"  {component_name.replace('_', ' ').title()}: "
                  f"{component_data['score']:.3f} (weight: {component_data['weight']:.1f})")
        
        # Show rationale
        print(f"\nRationale: {result['rationale']}")
        
        # Create FactorScore for integration
        factor_score = create_sentiment_factor_score(candles)
        print(f"\nFactorScore Integration:")
        print(f"  Factor: {factor_score.factor_name}")
        print(f"  Weight: {factor_score.weight:.1%}")
        print(f"  Score: {factor_score.score:.3f}")
        print(f"  Description: {factor_score.description}")
    
    print("\n" + "=" * 60)
    print("Key Features:")
    print("• Combines Alternative.me fear & greed index (60% weight)")
    print("• Integrates fundamental metrics: funding, OI, L/S ratio (40% weight)")
    print("• Provides 15% contribution to overall technical analysis")
    print("• Handles API failures with graceful fallbacks")
    print("• Returns detailed component breakdown and rationale")
    print("• Normalized scores (0.0-1.0) with confidence metrics")
    print("=" * 60)


if __name__ == "__main__":
    main()