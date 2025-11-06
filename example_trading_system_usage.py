"""Example usage of the trading system core interfaces."""

from __future__ import annotations

import json
from typing import Iterable, Optional

from indicator_collector.trading_system import (
    AnalyzerContext,
    FactorScore,
    OptimizationStats,
    PositionPlan,
    SignalExplanation,
    TradingAnalyzer,
    TradingSignalPayload,
    parse_collector_payload,
    serialize_signal_payload,
)


class SimpleRSIAnalyzer:
    """
    Simple example analyzer using RSI.
    
    This demonstrates implementing the TradingAnalyzer protocol.
    """
    
    def __init__(self, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def analyze(self, context: AnalyzerContext) -> TradingSignalPayload:
        """Generate trading signal based on RSI."""
        rsi = context.indicators.get("rsi")
        
        if rsi is None:
            return self._neutral_signal(context, "No RSI data available")
        
        if rsi < self.rsi_oversold:
            return self._buy_signal(context, rsi)
        elif rsi > self.rsi_overbought:
            return self._sell_signal(context, rsi)
        else:
            return self._neutral_signal(context, f"RSI in neutral zone: {rsi:.1f}")
    
    def optimize(self, history: Iterable[AnalyzerContext]) -> Optional[OptimizationStats]:
        """Optimize analyzer parameters (simplified example)."""
        total = 0
        profitable = 0
        
        for ctx in history:
            signal = self.analyze(ctx)
            if signal.signal_type != "NEUTRAL":
                total += 1
                if signal.confidence > 0.7:
                    profitable += 1
        
        if total == 0:
            return None
        
        win_rate = (profitable / total) * 100
        
        return OptimizationStats(
            backtest_win_rate=win_rate,
            avg_profit_pct=1.5,
            avg_loss_pct=-0.8,
            sharpe_ratio=1.2,
            total_signals=total,
            profitable_signals=profitable,
            losing_signals=total - profitable,
        )
    
    def _buy_signal(self, context: AnalyzerContext, rsi: float) -> TradingSignalPayload:
        """Create BUY signal."""
        atr = context.indicators.get("atr", context.current_price * 0.02)
        
        factors = [
            FactorScore(
                factor_name="rsi_oversold",
                score=100.0 - (rsi / self.rsi_oversold * 100.0),
                weight=2.0,
                description=f"RSI at {rsi:.1f} (oversold threshold: {self.rsi_oversold})",
                emoji="🟢",
            )
        ]
        
        trend_strength = context.indicators.get("trend_strength", 50.0)
        if trend_strength:
            factors.append(
                FactorScore(
                    factor_name="trend_strength",
                    score=trend_strength,
                    weight=1.0,
                    description=f"Trend strength: {trend_strength:.1f}",
                )
            )
        
        confidence = min((self.rsi_oversold - rsi) / self.rsi_oversold, 1.0)
        
        return TradingSignalPayload(
            signal_type="BUY",
            confidence=confidence,
            timestamp=context.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            factors=factors,
            position_plan=PositionPlan(
                entry_price=context.current_price,
                stop_loss=context.current_price - (atr * 1.5),
                take_profit_levels=[
                    context.current_price + (atr * 2.0),
                    context.current_price + (atr * 3.0),
                    context.current_price + (atr * 5.0),
                ],
                position_size_usd=1000.0,
                risk_reward_ratio=2.0,
                max_risk_pct=2.0,
                leverage=10.0,
                direction="long",
            ),
            explanation=SignalExplanation(
                primary_reason=f"RSI oversold at {rsi:.1f}",
                supporting_factors=[
                    "Price approaching oversold territory",
                    "Potential bounce setup",
                ],
                risk_factors=[
                    "May continue lower in strong downtrend",
                    "Low RSI can persist in bear markets",
                ],
                market_context=f"Timeframe: {context.timeframe}, Price: ${context.current_price:.2f}",
            ),
        )
    
    def _sell_signal(self, context: AnalyzerContext, rsi: float) -> TradingSignalPayload:
        """Create SELL signal."""
        atr = context.indicators.get("atr", context.current_price * 0.02)
        
        factors = [
            FactorScore(
                factor_name="rsi_overbought",
                score=(rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought) * 100.0,
                weight=2.0,
                description=f"RSI at {rsi:.1f} (overbought threshold: {self.rsi_overbought})",
                emoji="🔴",
            )
        ]
        
        confidence = min((rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought), 1.0)
        
        return TradingSignalPayload(
            signal_type="SELL",
            confidence=confidence,
            timestamp=context.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            factors=factors,
            position_plan=PositionPlan(
                entry_price=context.current_price,
                stop_loss=context.current_price + (atr * 1.5),
                take_profit_levels=[
                    context.current_price - (atr * 2.0),
                    context.current_price - (atr * 3.0),
                    context.current_price - (atr * 5.0),
                ],
                position_size_usd=1000.0,
                risk_reward_ratio=2.0,
                max_risk_pct=2.0,
                leverage=10.0,
                direction="short",
            ),
            explanation=SignalExplanation(
                primary_reason=f"RSI overbought at {rsi:.1f}",
                supporting_factors=[
                    "Price in overbought territory",
                    "Potential reversal setup",
                ],
                risk_factors=[
                    "May continue higher in strong uptrend",
                    "High RSI can persist in bull markets",
                ],
                market_context=f"Timeframe: {context.timeframe}, Price: ${context.current_price:.2f}",
            ),
        )
    
    def _neutral_signal(self, context: AnalyzerContext, reason: str) -> TradingSignalPayload:
        """Create NEUTRAL signal."""
        return TradingSignalPayload(
            signal_type="NEUTRAL",
            confidence=0.0,
            timestamp=context.timestamp,
            symbol=context.symbol,
            timeframe=context.timeframe,
            factors=[],
            position_plan=None,
            explanation=SignalExplanation(
                primary_reason=reason,
                supporting_factors=[],
                risk_factors=[],
                market_context=f"Waiting for better setup",
            ),
        )


def example_from_collector_payload():
    """Example: Parse collector payload and generate signal."""
    print("=" * 60)
    print("Example 1: From Collector Payload")
    print("=" * 60)
    
    sample_payload = {
        "metadata": {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "period": 100,
        },
        "latest": {
            "timestamp": 1699000000000,
            "close": 50000.0,
            "open": 49900.0,
            "high": 50100.0,
            "low": 49800.0,
            "volume": 1000000.0,
            "rsi": 28.5,
            "macd": -50.0,
            "trend_strength": 45.0,
            "atr": 500.0,
            "structure_state": "bearish",
        },
        "advanced": {
            "volume_analysis": {},
            "market_structure": {},
        },
        "multi_timeframe": {},
        "zones": [],
        "signals": [],
    }
    
    context = parse_collector_payload(sample_payload)
    
    print(f"\nParsed Context:")
    print(f"  Symbol: {context.symbol}")
    print(f"  Timeframe: {context.timeframe}")
    print(f"  Price: ${context.current_price:.2f}")
    print(f"  RSI: {context.indicators.get('rsi')}")
    print(f"  Trend Strength: {context.indicators.get('trend_strength')}")
    
    analyzer = SimpleRSIAnalyzer()
    signal = analyzer.analyze(context)
    
    print(f"\nGenerated Signal:")
    print(f"  Type: {signal.signal_type}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Explanation: {signal.explanation.primary_reason}")
    
    if signal.position_plan:
        print(f"\nPosition Plan:")
        print(f"  Entry: ${signal.position_plan.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.position_plan.stop_loss:.2f}")
        print(f"  Take Profits: {[f'${tp:.2f}' for tp in signal.position_plan.take_profit_levels]}")
    
    serialized = serialize_signal_payload(signal)
    print(f"\nJSON Serialized: {json.dumps(serialized, indent=2)[:500]}...")


def example_direct_usage():
    """Example: Direct usage without collector."""
    print("\n" + "=" * 60)
    print("Example 2: Direct Context Creation")
    print("=" * 60)
    
    context = AnalyzerContext(
        symbol="ETHUSDT",
        timeframe="15m",
        timestamp=1699000000000,
        current_price=3000.0,
        ohlcv={
            "open": 2995.0,
            "high": 3010.0,
            "low": 2990.0,
            "close": 3000.0,
            "volume": 500000.0,
        },
        indicators={
            "rsi": 75.5,
            "macd": 25.0,
            "trend_strength": 80.0,
            "atr": 30.0,
        },
    )
    
    print(f"\nContext:")
    print(f"  Symbol: {context.symbol}")
    print(f"  Price: ${context.current_price:.2f}")
    print(f"  RSI: {context.indicators['rsi']}")
    
    analyzer = SimpleRSIAnalyzer(rsi_oversold=30.0, rsi_overbought=70.0)
    signal = analyzer.analyze(context)
    
    print(f"\nSignal:")
    print(f"  Type: {signal.signal_type}")
    print(f"  Confidence: {signal.confidence:.2%}")
    
    if signal.factors:
        print(f"\nFactors:")
        for factor in signal.factors:
            print(f"  - {factor.factor_name}: {factor.score:.1f} {factor.emoji or ''}")
            print(f"    {factor.description}")


def example_optimization():
    """Example: Optimizer usage."""
    print("\n" + "=" * 60)
    print("Example 3: Optimization")
    print("=" * 60)
    
    history = [
        AnalyzerContext(
            symbol="BTCUSDT",
            timeframe="1h",
            timestamp=1699000000000 + i * 3600000,
            current_price=50000.0 + (i * 100),
            ohlcv={"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0},
            indicators={"rsi": 20.0 + (i * 5)},
        )
        for i in range(20)
    ]
    
    analyzer = SimpleRSIAnalyzer()
    stats = analyzer.optimize(history)
    
    if stats:
        print(f"\nOptimization Results:")
        print(f"  Total Signals: {stats.total_signals}")
        print(f"  Win Rate: {stats.backtest_win_rate:.1f}%")
        print(f"  Profitable: {stats.profitable_signals}")
        print(f"  Losing: {stats.losing_signals}")
        print(f"  Sharpe Ratio: {stats.sharpe_ratio:.2f}")


if __name__ == "__main__":
    print("\n🔧 Trading System Core - Usage Examples\n")
    
    example_from_collector_payload()
    example_direct_usage()
    example_optimization()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)
    print()
