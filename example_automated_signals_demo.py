#!/usr/bin/env python3
"""
Example: Generate automated trading signals and integrate with web UI.

This script demonstrates:
1. Creating trading signals using the signal generator
2. Adding position plans with risk management
3. Calculating performance metrics
4. Formatting signals for web UI display
"""

from datetime import datetime
import json

from indicator_collector.trading_system import (
    AnalyzerContext,
    FactorScore,
    OptimizationStats,
    PositionPlan,
    SignalExplanation,
    TradingSignalPayload,
)


def create_sample_context() -> AnalyzerContext:
    """Create a sample market context."""
    return AnalyzerContext(
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
            "bollinger_position": 0.2,
        },
        multi_timeframe={
            "4h": {"trend": "down", "strength": 55},
            "1d": {"trend": "down", "strength": 70},
        },
        success_rates={
            "bullish_signals": 62.5,
            "bearish_signals": 58.0,
        },
    )


def generate_buy_signal() -> TradingSignalPayload:
    """Generate a sample BUY signal."""
    context = create_sample_context()
    
    # Create factors
    factors = [
        FactorScore(
            factor_name="rsi_oversold",
            score=85.0,
            weight=2.0,
            description="RSI at 28.5 (oversold threshold: 30)",
            emoji="🟢",
        ),
        FactorScore(
            factor_name="trend_strength",
            score=65.0,
            weight=1.5,
            description="Downtrend showing weakening momentum",
        ),
        FactorScore(
            factor_name="volume_confirmation",
            score=70.0,
            weight=1.0,
            description="Volume increasing on lower prices",
        ),
    ]
    
    # Create position plan
    entry_price = context.current_price
    atr = context.indicators["atr"]
    stop_loss = entry_price - (atr * 1.5)
    
    position_plan = PositionPlan(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_levels=[
            entry_price + (atr * 2.0),
            entry_price + (atr * 3.0),
            entry_price + (atr * 5.0),
        ],
        position_size_usd=1000.0,
        risk_reward_ratio=2.0,
        max_risk_pct=0.02,
        leverage=10.0,
        direction="long",
    )
    
    # Create explanation
    explanation = SignalExplanation(
        primary_reason="RSI oversold at 28.5, indicating strong bounce setup",
        supporting_factors=[
            "Price near support level at $44,800",
            "Volume increasing on lower prices",
            "MACD showing divergence (lower lows with higher RSI)",
        ],
        risk_factors=[
            "Could continue lower in strong downtrend",
            "Liquidation risk with 10x leverage on volatile asset",
            "News risk - watch for market-moving announcements",
        ],
        market_context="Timeframe: 1h, Price: $45,000, Trend: Down but weakening",
    )
    
    # Create optimization stats
    optimization_stats = OptimizationStats(
        backtest_win_rate=62.5,
        avg_profit_pct=2.1,
        avg_loss_pct=-0.8,
        sharpe_ratio=1.8,
        total_signals=40,
        profitable_signals=25,
        losing_signals=15,
    )
    
    # Create signal
    signal = TradingSignalPayload(
        signal_type="BUY",
        confidence=0.85,
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=factors,
        position_plan=position_plan,
        explanation=explanation,
        optimization_stats=optimization_stats,
    )
    
    return signal


def generate_sell_signal() -> TradingSignalPayload:
    """Generate a sample SELL signal."""
    context = create_sample_context()
    context.current_price = 47500.0
    context.indicators["rsi"] = 75.0
    
    factors = [
        FactorScore(
            factor_name="rsi_overbought",
            score=80.0,
            weight=2.0,
            description="RSI at 75.0 (overbought threshold: 70)",
            emoji="🔴",
        ),
        FactorScore(
            factor_name="resistance_rejection",
            score=72.0,
            weight=1.5,
            description="Price rejected at resistance level",
        ),
    ]
    
    entry_price = context.current_price
    atr = context.indicators["atr"]
    stop_loss = entry_price + (atr * 1.5)
    
    position_plan = PositionPlan(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_levels=[
            entry_price - (atr * 2.0),
            entry_price - (atr * 3.0),
            entry_price - (atr * 5.0),
        ],
        position_size_usd=1500.0,
        risk_reward_ratio=2.5,
        max_risk_pct=0.02,
        leverage=10.0,
        direction="short",
    )
    
    explanation = SignalExplanation(
        primary_reason="RSI overbought at 75, indicating reversal setup",
        supporting_factors=[
            "Price rejected at $47,500 resistance",
            "Divergence: higher prices but declining volume",
        ],
        risk_factors=[
            "Could break higher in strong uptrend",
            "Funding rates are high (funding risk)",
        ],
        market_context="Timeframe: 1h, Price: $47,500, Trend: Up but exhausting",
    )
    
    optimization_stats = OptimizationStats(
        backtest_win_rate=58.0,
        avg_profit_pct=1.8,
        avg_loss_pct=-1.2,
        sharpe_ratio=1.2,
        total_signals=35,
        profitable_signals=20,
        losing_signals=15,
    )
    
    signal = TradingSignalPayload(
        signal_type="SELL",
        confidence=0.78,
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=factors,
        position_plan=position_plan,
        explanation=explanation,
        optimization_stats=optimization_stats,
    )
    
    return signal


def generate_neutral_signal() -> TradingSignalPayload:
    """Generate a NEUTRAL (no trade) signal."""
    context = create_sample_context()
    
    explanation = SignalExplanation(
        primary_reason="Waiting for better setup - mixed signals",
        supporting_factors=[
            "RSI in neutral zone (28-70)",
            "MACD not aligned",
        ],
        risk_factors=[
            "No clear trend direction",
            "High market volatility",
        ],
        market_context="Timeframe: 1h - No clear directional bias",
    )
    
    signal = TradingSignalPayload(
        signal_type="NEUTRAL",
        confidence=0.0,
        timestamp=context.timestamp,
        symbol=context.symbol,
        timeframe=context.timeframe,
        factors=[],
        position_plan=None,
        explanation=explanation,
    )
    
    return signal


def create_payload_with_signals(signal: TradingSignalPayload) -> dict:
    """Create a collector payload with the signal."""
    return {
        "metadata": {
            "symbol": signal.symbol,
            "timeframe": signal.timeframe,
            "period": 100,
            "token": "demo",
            "generated_at": datetime.now().isoformat(),
        },
        "latest": {
            "close": 45000.0,
            "open": 44900.0,
            "high": 45100.0,
            "low": 44800.0,
            "volume": 1000000.0,
            "rsi": 28.5,
            "macd": -50.0,
            "trend_strength": 65.0,
        },
        "advanced": {},
        "multi_timeframe": {},
        "zones": [],
        "automated_signals": signal.to_dict(),
    }


def main():
    """Run examples."""
    print("=" * 70)
    print("AUTOMATED SIGNALS DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: BUY Signal
    print("\n1. GENERATING BUY SIGNAL")
    print("-" * 70)
    buy_signal = generate_buy_signal()
    print(f"Signal Type: {buy_signal.signal_type}")
    print(f"Confidence: {buy_signal.confidence * 100:.1f}%")
    print(f"Factors: {len(buy_signal.factors)}")
    for factor in buy_signal.factors:
        print(f"  - {factor.factor_name}: {factor.score:.1f} (weight: {factor.weight})")
    if buy_signal.position_plan:
        plan = buy_signal.position_plan
        print(f"Position Size: ${plan.position_size_usd:.2f}")
        print(f"Entry: ${plan.entry_price:.2f}")
        print(f"Stop Loss: ${plan.stop_loss:.2f}")
        print(f"TP Levels: {len(plan.take_profit_levels)}")
    if buy_signal.optimization_stats:
        stats = buy_signal.optimization_stats
        print(f"Historical Win Rate: {stats.backtest_win_rate:.1f}%")
        print(f"Sharpe Ratio: {stats.sharpe_ratio:.2f}")
    
    # Example 2: SELL Signal
    print("\n2. GENERATING SELL SIGNAL")
    print("-" * 70)
    sell_signal = generate_sell_signal()
    print(f"Signal Type: {sell_signal.signal_type}")
    print(f"Confidence: {sell_signal.confidence * 100:.1f}%")
    print(f"Factors: {len(sell_signal.factors)}")
    if sell_signal.position_plan:
        plan = sell_signal.position_plan
        print(f"Direction: {plan.direction.upper()}")
        print(f"Position Size: ${plan.position_size_usd:.2f}")
    
    # Example 3: NEUTRAL Signal
    print("\n3. GENERATING NEUTRAL SIGNAL")
    print("-" * 70)
    neutral_signal = generate_neutral_signal()
    print(f"Signal Type: {neutral_signal.signal_type}")
    print(f"Confidence: {neutral_signal.confidence * 100:.1f}%")
    print(f"Position Plan: {neutral_signal.position_plan is not None}")
    print(f"Primary Reason: {neutral_signal.explanation.primary_reason}")
    
    # Example 4: JSON for Web UI
    print("\n4. WEB UI PAYLOAD FORMAT")
    print("-" * 70)
    payload = create_payload_with_signals(buy_signal)
    print("Payload structure:")
    print(f"  - metadata: {list(payload['metadata'].keys())}")
    print(f"  - latest: {list(payload['latest'].keys())}")
    print(f"  - automated_signals: {list(payload['automated_signals'].keys())}")
    
    print("\nJSON export (abbreviated):")
    signal_dict = payload["automated_signals"]
    print(f"  signal_type: {signal_dict['signal_type']}")
    print(f"  confidence: {signal_dict['confidence']}")
    print(f"  symbol: {signal_dict['symbol']}")
    print(f"  timeframe: {signal_dict['timeframe']}")
    print(f"  factors: {len(signal_dict['factors'])} factors")
    print(f"  position_plan: {signal_dict['position_plan'] is not None}")
    print(f"  optimization_stats: {signal_dict['optimization_stats'] is not None}")
    
    # Example 5: Full JSON
    print("\n5. FULL SIGNAL JSON")
    print("-" * 70)
    print(json.dumps(payload["automated_signals"], indent=2)[:500] + "\n... (truncated)")
    
    # Example 6: Web UI Integration
    print("\n6. WEB UI INTEGRATION EXAMPLE")
    print("-" * 70)
    print("""
To integrate with the web UI:

1. Generate signals in your data collection:
   signal = generate_trading_signal(context)
   payload["automated_signals"] = signal.to_dict()

2. Access in web UI (automatic):
   # web_ui.py will read:
   signal_data = payload.get("automated_signals", {})
   
3. The 🤖 Automated Signals tab will display:
   - Signal type with emoji and color
   - Confidence as percentage with level indicator
   - Factor analysis table
   - Position plan with entry, SL, TP levels
   - Risk calculations
   - Holding horizon
   - Signal rationale
   - Performance metrics
   - Cancellation reasons (if applicable)

4. Users can:
   - View signal details
   - Understand signal rationale
   - Check position sizing
   - Review historical performance
   - Export signal JSON
""")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Generate signals from your trading system")
    print("2. Add automated_signals to collector payload")
    print("3. Run streamlit to view in web UI")
    print("4. Navigate to 🤖 Automated Signals tab")
    print("\nFor more information:")
    print("- AUTOMATED_SIGNALS_TAB_USAGE.md - User guide")
    print("- AUTOMATED_SIGNALS_INTEGRATION.md - Developer guide")
    print("- QA_AUTOMATED_SIGNALS_TAB.md - QA checklist")
    print("\n")


if __name__ == "__main__":
    main()
