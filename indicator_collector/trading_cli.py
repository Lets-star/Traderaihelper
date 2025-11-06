"""CLI interface for the trading system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .collector import collect_metrics
from .trading_system import (
    TradingConfig,
    TradingOrchestrator,
    create_trading_orchestrator,
    create_default_config,
)


def create_trading_parser() -> argparse.ArgumentParser:
    """Create argument parser for trading commands."""
    parser = argparse.ArgumentParser(description="Trading system orchestrator CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Trading commands")
    
    # Live trading command
    live_parser = subparsers.add_parser("live", help="Run live trading analysis")
    live_parser.add_argument("--symbol", default="BINANCE:BTCUSDT", help="Symbol to analyze")
    live_parser.add_argument("--timeframe", default="15m", help="Timeframe for analysis")
    live_parser.add_argument("--period", type=int, default=500, help="Number of bars to analyze")
    live_parser.add_argument("--token", required=True, help="Authentication token")
    live_parser.add_argument("--config", help="Path to trading configuration file")
    live_parser.add_argument("--output", help="Output file for trading signal")
    live_parser.add_argument("--account-balance", type=float, default=10000.0, help="Account balance")
    live_parser.add_argument("--mode", choices=["live", "backtest"], default="live", help="Trading mode")
    live_parser.add_argument("--data-dir", default="./trading_data", help="Data directory for state")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument("--input", required=True, help="Input JSON file with collector results")
    backtest_parser.add_argument("--config", help="Path to trading configuration file")
    backtest_parser.add_argument("--output", help="Output file for backtest results")
    backtest_parser.add_argument("--account-balance", type=float, default=10000.0, help="Account balance")
    backtest_parser.add_argument("--data-dir", default="./backtest_data", help="Data directory for backtest")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--action", choices=["show", "create"], required=True, help="Config action")
    config_parser.add_argument("--output", help="Output file for configuration")
    config_parser.add_argument("--input", help="Input configuration file")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize trading weights")
    optimize_parser.add_argument("--config", help="Path to trading configuration file")
    optimize_parser.add_argument("--data-dir", default="./trading_data", help="Data directory")
    optimize_parser.add_argument("--weights", help="Current weights as JSON string")
    optimize_parser.add_argument("--output", help="Output file for optimization results")
    
    # Outcomes command
    outcomes_parser = subparsers.add_parser("outcomes", help="Manage signal outcomes")
    outcomes_parser.add_argument("--action", choices=["add", "load", "show"], required=True, help="Action")
    outcomes_parser.add_argument("--data-dir", default="./trading_data", help="Data directory")
    outcomes_parser.add_argument("--file", help="File path for outcomes")
    outcomes_parser.add_argument("--outcome", help="Signal outcome as JSON string")
    
    # State command
    state_parser = subparsers.add_parser("state", help="State management")
    state_parser.add_argument("--action", choices=["show", "export", "import"], required=True, help="State action")
    state_parser.add_argument("--data-dir", default="./trading_data", help="Data directory")
    state_parser.add_argument("--file", help="File path for state operations")
    
    return parser


def load_or_create_config(config_path: Optional[str], **overrides) -> TradingConfig:
    """Load configuration from file or create default with overrides."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = TradingConfig.from_dict(config_data)
    else:
        config = create_default_config()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def cmd_live(args: argparse.Namespace) -> None:
    """Run live trading analysis."""
    config = load_or_create_config(
        args.config,
        account_balance=args.account_balance,
        mode=args.mode,
        data_dir=args.data_dir,
    )
    
    orchestrator = create_trading_orchestrator(config)
    
    # Collect metrics
    result = collect_metrics(
        symbol=args.symbol,
        timeframe=args.timeframe,
        period=args.period,
        token=args.token,
    )
    
    # Process through trading system
    signal = orchestrator.process_collection_result(result)
    
    # Output result
    signal_data = signal.to_dict()
    if args.output:
        output_path = Path(args.output).expanduser()
        with open(output_path, 'w') as f:
            json.dump(signal_data, f, indent=2)
        print(f"Trading signal saved to {output_path}")
    else:
        print(json.dumps(signal_data, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtesting on collector results."""
    config = load_or_create_config(
        args.config,
        account_balance=args.account_balance,
        mode="backtest",
        data_dir=args.data_dir,
    )
    
    orchestrator = create_trading_orchestrator(config)
    
    # Load input data
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    results = []
    
    if isinstance(input_data, list):
        # Multiple collector results
        for item in input_data:
            # Convert dict to CollectionResult-like structure
            # For now, we'll create a mock CollectionResult
            from .trading_system.interfaces import parse_collector_payload
            context = parse_collector_payload(item)
            
            # Create a minimal CollectionResult
            class MockCollectionResult:
                def __init__(self, payload):
                    self.payload = payload
            
            mock_result = MockCollectionResult(item)
            signal = orchestrator.process_collection_result(mock_result)
            results.append(signal.to_dict())
    else:
        # Single collector result
        from .trading_system.interfaces import parse_collector_payload
        context = parse_collector_payload(input_data)
        
        class MockCollectionResult:
            def __init__(self, payload):
                self.payload = payload
        
        mock_result = MockCollectionResult(input_data)
        signal = orchestrator.process_collection_result(mock_result)
        results.append(signal.to_dict())
    
    # Output results
    output_data = {
        "config": config.to_dict(),
        "results": results,
        "performance": orchestrator.get_performance_stats(),
    }
    
    if args.output:
        output_path = Path(args.output).expanduser()
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Backtest results saved to {output_path}")
    else:
        print(json.dumps(output_data, indent=2))


def cmd_config(args: argparse.Namespace) -> None:
    """Handle configuration commands."""
    if args.action == "show":
        if args.input:
            with open(args.input, 'r') as f:
                config_data = json.load(f)
            print(json.dumps(config_data, indent=2))
        else:
            config = create_default_config()
            print(json.dumps(config.to_dict(), indent=2))
    
    elif args.action == "create":
        config = create_default_config()
        config_data = config.to_dict()
        
        if args.output:
            output_path = Path(args.output).expanduser()
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Default configuration saved to {output_path}")
        else:
            print(json.dumps(config_data, indent=2))


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run weight optimization."""
    config = load_or_create_config(args.config, data_dir=args.data_dir)
    orchestrator = create_trading_orchestrator(config)
    
    # Parse current weights
    current_weights = {}
    if args.weights:
        current_weights = json.loads(args.weights)
    else:
        # Use default weights
        current_weights = {
            "technical": 1.0,
            "sentiment": 0.8,
            "multitimeframe": 0.6,
            "volume": 0.7,
        }
    
    # Run optimization
    result = orchestrator.optimize_weights(current_weights)
    
    output_data = {
        "current_weights": current_weights,
        "optimization_result": result.to_dict(),
        "performance": orchestrator.get_performance_stats(),
    }
    
    if args.output:
        output_path = Path(args.output).expanduser()
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Optimization results saved to {output_path}")
    else:
        print(json.dumps(output_data, indent=2))


def cmd_outcomes(args: argparse.Namespace) -> None:
    """Handle signal outcomes."""
    config = load_or_create_config(None, data_dir=args.data_dir)
    orchestrator = create_trading_orchestrator(config)
    
    if args.action == "add":
        if args.outcome:
            from .trading_system import SignalOutcome
            outcome_data = json.loads(args.outcome)
            outcome = SignalOutcome.from_dict(outcome_data)
            orchestrator.add_signal_outcome(outcome)
            print("Signal outcome added successfully")
        else:
            print("Error: --outcome required for add action")
            sys.exit(1)
    
    elif args.action == "load":
        if args.file:
            ingested = orchestrator.load_historical_outcomes(args.file)
            print(f"Loaded {ingested} historical outcomes")
        else:
            print("Error: --file required for load action")
            sys.exit(1)
    
    elif args.action == "show":
        outcomes = orchestrator._load_outcomes()
        print(json.dumps([outcome.to_dict() for outcome in outcomes], indent=2))


def cmd_state(args: argparse.Namespace) -> None:
    """Handle state management."""
    config = load_or_create_config(None, data_dir=args.data_dir)
    orchestrator = create_trading_orchestrator(config)
    
    if args.action == "show":
        state_data = orchestrator.get_performance_stats()
        print(json.dumps(state_data, indent=2))
    
    elif args.action == "export":
        if args.file:
            orchestrator.export_state(args.file)
            print(f"State exported to {args.file}")
        else:
            print("Error: --file required for export action")
            sys.exit(1)
    
    elif args.action == "import":
        if args.file:
            orchestrator.import_state(args.file)
            print(f"State imported from {args.file}")
        else:
            print("Error: --file required for import action")
            sys.exit(1)


def main(argv: List[str] | None = None) -> None:
    """Main CLI entry point for trading system."""
    parser = create_trading_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "live":
            cmd_live(args)
        elif args.command == "backtest":
            cmd_backtest(args)
        elif args.command == "config":
            cmd_config(args)
        elif args.command == "optimize":
            cmd_optimize(args)
        elif args.command == "outcomes":
            cmd_outcomes(args)
        elif args.command == "state":
            cmd_state(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()