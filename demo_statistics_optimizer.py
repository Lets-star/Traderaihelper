#!/usr/bin/env python3
"""
Demonstration script for the Statistics Optimizer module.

This script shows how to use the stats optimizer to:
1. Track signal outcomes
2. Calculate performance KPIs
3. Optimize factor weights
4. Ingest historical data
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicator_collector.trading_system.statistics_optimizer import (
    SignalOutcome,
    PerformanceKPIs,
    WeightAdjustment,
    OptimizationResult,
    StatsOptimizerConfig,
    StatisticsOptimizer,
    create_stats_optimizer,
    create_synthetic_outcomes,
)


def main():
    """Demonstrate statistics optimizer functionality."""
    
    print("=" * 60)
    print("STATISTICS OPTIMIZER DEMONSTRATION")
    print("=" * 60)
    
    # 1. Create optimizer with custom configuration
    print("\n1. Creating Statistics Optimizer...")
    config = StatsOptimizerConfig(
        min_win_rate_target=0.60,
        min_profit_factor_target=1.8,
        max_drawdown_target=0.15,
        min_sharpe_target=1.2,
        min_signals_for_analysis=25,
        weight_adjustment_factor=0.15
    )
    optimizer = StatisticsOptimizer(config)
    print("✓ Optimizer created with custom targets")
    
    # 2. Add synthetic signal outcomes
    print("\n2. Adding synthetic signal outcomes...")
    outcomes = create_synthetic_outcomes(count=40)
    for outcome in outcomes:
        optimizer.add_signal_outcome(outcome)
    print(f"✓ Added {len(outcomes)} signal outcomes")
    
    # 3. Calculate current performance KPIs
    print("\n3. Calculating Performance KPIs...")
    kpis = optimizer.calculate_kpis()
    print(f"   Total Signals: {kpis.total_signals}")
    print(f"   Win Rate: {kpis.win_rate:.2%}")
    print(f"   Profit Factor: {kpis.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {kpis.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {kpis.max_drawdown_pct:.2%}")
    print(f"   Avg Profit: {kpis.avg_profit_pct:.2f}%")
    print(f"   Avg Loss: {kpis.avg_loss_pct:.2f}%")
    print(f"   Total Return: {kpis.total_return_pct:.2f}%")
    
    # 4. Suggest weight adjustments
    print("\n4. Analyzing Factor Performance...")
    current_weights = {
        'technical': 1.0,
        'sentiment': 0.8,
        'volume': 1.2,
        'multitimeframe': 0.9
    }
    
    print("   Current factor weights:")
    for factor, weight in current_weights.items():
        print(f"     {factor}: {weight:.2f}")
    
    adjustments = optimizer.suggest_weight_adjustments(current_weights)
    if adjustments:
        print("   Suggested adjustments:")
        for adj in adjustments:
            direction = "↑" if adj.suggested_weight > adj.current_weight else "↓"
            print(f"     {adj.factor_name}: {adj.current_weight:.2f} → {adj.suggested_weight:.2f} {direction}")
            print(f"       Reason: {adj.adjustment_reason}")
            print(f"       Confidence: {adj.confidence:.1%}")
    else:
        print("   No significant adjustments recommended")
    
    # 5. Run full optimization
    print("\n5. Running Weight Optimization...")
    result = optimizer.optimize_weights(current_weights)
    
    print(f"   Can meet targets: {result.can_meet_targets}")
    print(f"   Optimization score: {result.optimization_score:.1f}/100")
    
    if result.weight_adjustments:
        print("   Final weight recommendations:")
        for adj in result.weight_adjustments:
            direction = "↑" if adj.suggested_weight > adj.current_weight else "↓"
            print(f"     {adj.factor_name}: {adj.current_weight:.2f} → {adj.suggested_weight:.2f} {direction}")
    
    print("   Recommendations:")
    for rec in result.recommendations:
        print(f"     • {rec}")
    
    # 6. Demonstrate historical data ingestion
    print("\n6. Demonstrating Historical Data Ingestion...")
    
    # Create sample historical log format
    historical_data = [
        {
            'signal': {
                'signal_type': 'BUY',
                'entry_price': 45000.0,
                'timestamp': 1640995200,
                'factors': [
                    {'factor_name': 'technical', 'score': 0.8},
                    {'factor_name': 'sentiment', 'score': 0.6}
                ]
            },
            'outcome': {
                'exit_price': 46800.0,
                'pnl_pct': 4.0,
                'success': True,
                'holding_bars': 48,
                'exit_timestamp': 1641174400
            }
        },
        {
            'signal': {
                'signal_type': 'SELL',
                'entry_price': 47000.0,
                'timestamp': 1641260800,
                'factors': [
                    {'factor_name': 'volume', 'score': -0.3},
                    {'factor_name': 'multitimeframe', 'score': -0.5}
                ]
            },
            'outcome': {
                'exit_price': 45150.0,
                'pnl_pct': -3.94,
                'success': False,
                'holding_bars': 24,
                'exit_timestamp': 1641347200
            }
        }
    ]
    
    new_optimizer = create_stats_optimizer()
    ingested = new_optimizer.ingest_historical_logs(historical_data)
    print(f"   Ingested {ingested} historical entries")
    
    historical_kpis = new_optimizer.calculate_kpis()
    print(f"   Historical win rate: {historical_kpis.win_rate:.2%}")
    print(f"   Historical profit factor: {historical_kpis.profit_factor:.2f}")
    
    # 7. Show integration with trading system
    print("\n7. Integration with Trading System...")
    optimization_stats = optimizer.get_optimization_stats()
    print("   Optimization stats compatible with trading system:")
    print(f"     Backtest win rate: {optimization_stats.backtest_win_rate:.2%}")
    print(f"     Avg profit: {optimization_stats.avg_profit_pct:.2f}%")
    print(f"     Avg loss: {optimization_stats.avg_loss_pct:.2f}%")
    print(f"     Sharpe ratio: {optimization_stats.sharpe_ratio:.2f}")
    print(f"     Total signals: {optimization_stats.total_signals}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nThe Statistics Optimizer provides:")
    print("• Comprehensive KPI tracking (win rate, profit factor, Sharpe, drawdown)")
    print("• Factor performance analysis and weight optimization")
    print("• Historical data ingestion for backtesting")
    print("• Performance target validation")
    print("• Integration with existing trading system interfaces")
    print("\nAll functions are properly exported in trading_system.__init__.py")


if __name__ == "__main__":
    main()