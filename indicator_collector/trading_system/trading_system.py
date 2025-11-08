"""Trading system orchestrator for coordinating all trading components."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple, Union
from unittest.mock import Mock

from .interfaces import (
    AnalyzerContext,
    JsonDict,
    PositionPlan,
    SignalExplanation,
    TradingSignalPayload,
    deserialize_signal_payload,
    parse_collector_payload,
    serialize_signal_payload,
)
from .position_manager import (
    DiversificationGuard,
    PositionManagerConfig,
    PositionManagerResult,
    create_diversification_guard,
    create_position_plan,
)
from .signal_generator import SignalConfig, SignalFactors, generate_trading_signal
from .statistics_optimizer import (
    OptimizationResult,
    SignalOutcome,
    StatsOptimizerConfig,
    StatisticsOptimizer,
    create_stats_optimizer,
    create_synthetic_outcomes,
)
from .technical_analysis import analyze_technical_factors
from .sentiment_analyzer import analyze_sentiment_factors
from .multitimeframe_analyzer import analyze_multitimeframe_factors
from .volume_orderbook_analyzer import analyze_volume_orderbook

if TYPE_CHECKING:
    from ..types import CollectionResult


@dataclass
class MacroBlackoutConfig:
    """Configuration for macro news blackout periods."""
    
    enabled: bool = True
    blackout_periods: List[Tuple[time, time]] = field(default_factory=lambda: [
        (time(8, 30), time(10, 0)),   # US CPI/PPI releases
        (time(13, 30), time(15, 0)),  # FOMC announcements
    ])
    blackout_weekdays: Set[int] = field(default_factory=lambda: {1, 2, 3, 4, 5})  # Mon-Fri
    timezone: str = "UTC"
    custom_blackout_dates: Set[str] = field(default_factory=set)  # YYYY-MM-DD format
    
    def to_dict(self) -> JsonDict:
        return {
            "enabled": self.enabled,
            "blackout_periods": [(start.isoformat(), end.isoformat()) for start, end in self.blackout_periods],
            "blackout_weekdays": list(self.blackout_weekdays),
            "timezone": self.timezone,
            "custom_blackout_dates": list(self.custom_blackout_dates),
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "MacroBlackoutConfig":
        periods = []
        for start_str, end_str in data.get("blackout_periods", []):
            start = time.fromisoformat(start_str)
            end = time.fromisoformat(end_str)
            periods.append((start, end))
        
        return cls(
            enabled=bool(data.get("enabled", True)),
            blackout_periods=periods,
            blackout_weekdays=set(data.get("blackout_weekdays", [1, 2, 3, 4, 5])),
            timezone=str(data.get("timezone", "UTC")),
            custom_blackout_dates=set(data.get("custom_blackout_dates", [])),
        )


@dataclass
class TradingConfig:
    """Complete configuration for the trading orchestrator."""
    
    # Basic settings
    account_balance: float = 10000.0
    mode: Literal["live", "backtest"] = "live"
    
    # Signal generation
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    
    # Position management
    position_config: PositionManagerConfig = field(default_factory=PositionManagerConfig)
    
    # Statistics optimization
    stats_config: StatsOptimizerConfig = field(default_factory=StatsOptimizerConfig)
    
    # Macro blackout filter
    macro_config: MacroBlackoutConfig = field(default_factory=MacroBlackoutConfig)
    
    # I/O settings
    data_dir: str = "./trading_data"
    state_file: str = "trading_state.json"
    signals_file: str = "trading_signals.json"
    outcomes_file: str = "signal_outcomes.json"
    
    def to_dict(self) -> JsonDict:
        return {
            "account_balance": self.account_balance,
            "mode": self.mode,
            "signal_config": self.signal_config.to_dict(),
            "position_config": self.position_config.to_dict(),
            "stats_config": self.stats_config.to_dict(),
            "macro_config": self.macro_config.to_dict(),
            "data_dir": self.data_dir,
            "state_file": self.state_file,
            "signals_file": self.signals_file,
            "outcomes_file": self.outcomes_file,
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "TradingConfig":
        return cls(
            account_balance=float(data.get("account_balance", 10000.0)),
            mode=data.get("mode", "live"),
            signal_config=SignalConfig.from_dict(data.get("signal_config", {})),
            position_config=PositionManagerConfig.from_dict(data.get("position_config", {})),
            stats_config=StatsOptimizerConfig.from_dict(data.get("stats_config", {})),
            macro_config=MacroBlackoutConfig.from_dict(data.get("macro_config", {})),
            data_dir=str(data.get("data_dir", "./trading_data")),
            state_file=str(data.get("state_file", "trading_state.json")),
            signals_file=str(data.get("signals_file", "trading_signals.json")),
            outcomes_file=str(data.get("outcomes_file", "signal_outcomes.json")),
        )


class MacroBlackoutFilter:
    """Filter for macro news blackout periods."""
    
    def __init__(self, config: MacroBlackoutConfig):
        self.config = config
    
    def is_blackout_period(self, timestamp: int) -> bool:
        """Check if the given timestamp falls within a blackout period."""
        if not self.config.enabled:
            return False
        
        # Convert timestamp to datetime
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        # Check custom blackout dates first
        date_str = dt.strftime("%Y-%m-%d")
        if date_str in self.config.custom_blackout_dates:
            return True
        
        # Check weekday
        if dt.weekday() not in self.config.blackout_weekdays:
            return False
        
        # Check time periods
        current_time = dt.time()
        for start_time, end_time in self.config.blackout_periods:
            if start_time <= current_time <= end_time:
                return True
        
        return False


@dataclass
class TradingState:
    """Persistent state for the trading orchestrator."""
    
    diversification_guard: DiversificationGuard
    last_signal_timestamp: Optional[int] = None
    last_optimization_timestamp: Optional[int] = None
    signal_count: int = 0
    metadata: JsonDict = field(default_factory=dict)
    
    def to_dict(self) -> JsonDict:
        return {
            "diversification_guard": self.diversification_guard.to_dict(),
            "last_signal_timestamp": self.last_signal_timestamp,
            "last_optimization_timestamp": self.last_optimization_timestamp,
            "signal_count": self.signal_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "TradingState":
        return cls(
            diversification_guard=DiversificationGuard.from_dict(data.get("diversification_guard", {})),
            last_signal_timestamp=data.get("last_signal_timestamp"),
            last_optimization_timestamp=data.get("last_optimization_timestamp"),
            signal_count=int(data.get("signal_count", 0)),
            metadata=data.get("metadata", {}),
        )


class TradingOrchestrator:
    """Main orchestrator for the trading system."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.macro_filter = MacroBlackoutFilter(config.macro_config)
        self.stats_optimizer = create_stats_optimizer(config.stats_config)
        
        # Initialize state
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / config.state_file
        self.signals_file = self.data_dir / config.signals_file
        self.outcomes_file = self.data_dir / config.outcomes_file
        
        # Load or create state
        self.state = self._load_state()
        
        # Track recent signals for optimization
        self._recent_signals: List[TradingSignalPayload] = []
        self._recent_outcomes: List[SignalOutcome] = []
    
    def _load_state(self) -> TradingState:
        """Load trading state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return TradingState.from_dict(data)
            except Exception as e:
                print(f"Warning: Failed to load state file: {e}")
        
        # Create new state
        return TradingState(diversification_guard=create_diversification_guard())
    
    def _save_state(self) -> None:
        """Save trading state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save state file: {e}")
    
    def _save_signal(self, signal: TradingSignalPayload) -> None:
        """Save a trading signal to file."""
        signals = []
        if self.signals_file.exists():
            try:
                with open(self.signals_file, 'r') as f:
                    signals = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load signals file: {e}")
        
        signals.append(serialize_signal_payload(signal))
        
        try:
            with open(self.signals_file, 'w') as f:
                json.dump(signals, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save signals file: {e}")
    
    def _load_outcomes(self) -> List[SignalOutcome]:
        """Load signal outcomes from file."""
        if not self.outcomes_file.exists():
            return []
        
        try:
            with open(self.outcomes_file, 'r') as f:
                data = json.load(f)
            return [SignalOutcome.from_dict(item) for item in data]
        except Exception as e:
            print(f"Warning: Failed to load outcomes file: {e}")
            return []
    
    def _save_outcomes(self, outcomes: List[SignalOutcome]) -> None:
        """Save signal outcomes to file."""
        try:
            with open(self.outcomes_file, 'w') as f:
                json.dump([outcome.to_dict() for outcome in outcomes], f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save outcomes file: {e}")
    
    def process_collection_result(self, result: "CollectionResult") -> TradingSignalPayload:
        """Process a CollectionResult and generate a complete trading signal."""
        
        # Parse collector result into analyzer context
        context = parse_collector_payload(result.payload)
        
        # Check macro blackout filter
        if self.macro_filter.is_blackout_period(context.timestamp):
            return TradingSignalPayload(
                signal_type="NEUTRAL",
                confidence=0.0,
                timestamp=context.timestamp,
                symbol=context.symbol,
                timeframe=context.timeframe,
                explanation=SignalExplanation(
                    primary_reason="Macro news blackout period - no trading signals generated",
                    supporting_factors=["Blackout filter active"],
                    risk_factors=["High volatility expected during macro releases"],
                ),
            )
        
        # Generate trading signal
        signal_payload = generate_trading_signal(
            context=context,
            config=self.config.signal_config,
        )
        
        # Generate position plan if we have a trading signal
        if signal_payload.signal_type in ["BUY", "SELL"]:
            position_result = create_position_plan(
                context=context,
                signal_direction="long" if signal_payload.signal_type == "BUY" else "short",
                config=self.config.position_config,
                account_balance=self.config.account_balance,
                diversification_guard=self.state.diversification_guard,
            )
            
            if position_result.can_trade:
                signal_payload.position_plan = position_result.position_plan
                signal_payload.explanation = position_result.explanation
                
                # Update diversification guard
                if signal_payload.position_plan:
                    direction = signal_payload.position_plan.direction
                    symbol = context.symbol
                    self.state.diversification_guard.add_position(symbol, direction)
            else:
                # Override signal to NEUTRAL if position plan fails
                signal_payload.signal_type = "NEUTRAL"
                signal_payload.confidence = 0.0
                signal_payload.explanation = SignalExplanation(
                    primary_reason="Position plan rejected",
                    supporting_factors=[],
                    risk_factors=position_result.cancellation_reasons,
                )
        
        # Add optimization stats if available
        if self.state.signal_count > 0:
            kpis = self.stats_optimizer.calculate_kpis()
            signal_payload.optimization_stats = kpis.to_optimization_stats()
        
        # Update state
        self.state.last_signal_timestamp = signal_payload.timestamp
        self.state.signal_count += 1
        self._save_state()
        
        # Save signal
        self._save_signal(signal_payload)
        
        # Track for optimization
        self._recent_signals.append(signal_payload)
        if len(self._recent_signals) > 100:  # Keep last 100 signals
            self._recent_signals.pop(0)
        
        return signal_payload
    
    def add_signal_outcome(self, outcome: SignalOutcome) -> None:
        """Add a signal outcome for performance tracking."""
        self.stats_optimizer.add_signal_outcome(outcome)
        
        # Track recent outcomes
        self._recent_outcomes.append(outcome)
        if len(self._recent_outcomes) > 100:  # Keep last 100 outcomes
            self._recent_outcomes.pop(0)
        
        # Save to file
        outcomes = self._load_outcomes()
        outcomes.append(outcome)
        self._save_outcomes(outcomes)
    
    def optimize_weights(self, current_weights: Dict[str, float]) -> OptimizationResult:
        """Optimize trading weights based on historical performance."""
        result = self.stats_optimizer.optimize_weights(current_weights)
        
        # Update optimization timestamp
        if result.can_meet_targets:
            self.state.last_optimization_timestamp = int(datetime.now().timestamp() * 1000)
            self._save_state()
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        kpis = self.stats_optimizer.calculate_kpis()
        
        return {
            "kpis": kpis.to_dict(),
            "state": self.state.to_dict(),
            "recent_signals_count": len(self._recent_signals),
            "recent_outcomes_count": len(self._recent_outcomes),
        }
    
    def load_historical_outcomes(self, file_path: str) -> int:
        """Load historical signal outcomes from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            outcomes = [SignalOutcome.from_dict(item) for item in data]
            ingested = self.stats_optimizer.ingest_historical_logs(outcomes)
            
            # Update current outcomes file
            current_outcomes = self._load_outcomes()
            current_outcomes.extend(outcomes)
            self._save_outcomes(current_outcomes)
            
            return ingested
        except Exception as e:
            print(f"Error loading historical outcomes: {e}")
            return 0
    
    def export_state(self, file_path: str) -> None:
        """Export complete trading state to a file."""
        export_data = {
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "performance": self.get_performance_stats(),
            "recent_signals": [serialize_signal_payload(s) for s in self._recent_signals],
            "recent_outcomes": [outcome.to_dict() for outcome in self._recent_outcomes],
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_state(self, file_path: str) -> None:
        """Import trading state from a file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Import config if present
        if "config" in data:
            self.config = TradingConfig.from_dict(data["config"])
            self.macro_filter = MacroBlackoutFilter(self.config.macro_config)
        
        # Import state if present
        if "state" in data:
            self.state = TradingState.from_dict(data["state"])
        
        # Import recent data if present
        if "recent_signals" in data:
            self._recent_signals = [deserialize_signal_payload(s) for s in data["recent_signals"]]
        
        if "recent_outcomes" in data:
            outcomes = [SignalOutcome.from_dict(item) for item in data["recent_outcomes"]]
            self._recent_outcomes = outcomes
            for outcome in outcomes:
                self.stats_optimizer.add_signal_outcome(outcome)
        
        self._save_state()


def create_trading_orchestrator(config: Optional[TradingConfig] = None) -> TradingOrchestrator:
    """Create a trading orchestrator with default or provided configuration."""
    if config is None:
        config = TradingConfig()
    return TradingOrchestrator(config)


def create_default_config() -> TradingConfig:
    """Create a default trading configuration."""
    return TradingConfig()