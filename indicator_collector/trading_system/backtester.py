"""Backtesting engine for real-only trading signals with performance optimization.

This module provides comprehensive backtesting capabilities that operate strictly
on validated real data, with support for walk-forward analysis, parameter optimization,
and adaptive weight adjustment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from ..real_data_validator import validate_real_data_payload, DataValidationError
from ..timeframes import Timeframe
from .interfaces import JsonDict, TradingSignalPayload
from .statistics_optimizer import (
    PerformanceKPIs,
    SignalOutcome,
    StatsOptimizerConfig,
    StatisticsOptimizer,
)

if TYPE_CHECKING:
    from .data_sources import HistoricalDataSource

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "technical": 0.25,
    "volume": 0.20,
    "sentiment": 0.15,
    "market_structure": 0.15,
    "multitimeframe": 0.10,
    "composite": 0.15,
}

DEFAULT_SIGNAL_THRESHOLDS: Dict[str, float] = {
    "buy": 0.65,
    "sell": 0.35,
}

_BASE_INDICATOR_DEFAULTS: Dict[str, Any] = {
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
    "atr": {"period": 14, "mult": 1.0},
    "atr_channels": {"period": 14, "mult_1x": 1.0, "mult_2x": 2.0, "mult_3x": 3.0},
    "bollinger": {"period": 20, "mult": 2.0, "source": "close", "stddev": 2.0},
    "volume": {
        "ma_period": 20,
        "cvd_atr_multiplier": 0.75,
        "delta_imbalance_threshold": 1.2,
        "vpvr_poc_share": 0.04,
        "smart_money_multiplier": 1.5,
        "component_weights": {
            "cvd": 0.3,
            "delta": 0.25,
            "vpvr": 0.2,
            "smart_money": 0.25,
        },
    },
    "structure": {
        "lookback": 24,
        "swing_window": 5,
        "trend_window": 12,
        "min_sequence": 5,
        "atr_distance": 1.0,
        "component_weights": {
            "trend": 0.4,
            "structure": 0.4,
            "liquidity": 0.2,
        },
    },
    "multitimeframe": {
        "trend_lookback": 14,
        "alignment_weight": 0.4,
        "agreement_weight": 0.3,
        "force_weight": 0.3,
    },
    "composite": {
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "confidence_floor": 0.3,
        "confidence_ceiling": 0.9,
        "min_confirmations": 3,
    },
}

_TIMEFRAME_INDICATOR_OVERRIDES: Dict[str, Dict[str, Any]] = {
    Timeframe.M5.value: {
        "atr": {"mult": 0.6},
        "atr_channels": {"mult_1x": 0.8, "mult_2x": 1.6, "mult_3x": 2.4},
        "structure": {"lookback": 36},
    },
    Timeframe.M15.value: {
        "atr": {"mult": 0.8},
        "structure": {"lookback": 32},
    },
    Timeframe.H3.value: {
        "atr": {"mult": 1.3},
        "atr_channels": {"mult_1x": 1.3, "mult_2x": 2.6, "mult_3x": 3.9},
        "structure": {"lookback": 20},
    },
    Timeframe.H4.value: {
        "atr": {"mult": 1.5},
        "rsi": {"period": 16},
        "structure": {"lookback": 18},
    },
    Timeframe.D1.value: {
        "rsi": {"period": 21, "overbought": 65, "oversold": 35},
        "atr": {"mult": 2.0},
        "atr_channels": {"mult_1x": 2.0, "mult_2x": 4.0, "mult_3x": 6.0},
        "structure": {"lookback": 30},
        "bollinger": {"period": 20, "mult": 2.5, "stddev": 2.5},
    },
}

_SUPPORTED_INDICATORS = set(_BASE_INDICATOR_DEFAULTS.keys())
_AVAILABLE_TIMEFRAMES = {
    Timeframe.M1.value,
    Timeframe.M5.value,
    Timeframe.M15.value,
    Timeframe.H1.value,
    Timeframe.H3.value,
    Timeframe.H4.value,
    Timeframe.D1.value,
}

_MISSING_INDICATOR_DEBUG_LOGGED: set[tuple[str, tuple[str, ...]]] = set()


def _normalize_timeframe_key(timeframe: Union[str, Timeframe]) -> str:
    """Normalize timeframe input to standard enum value."""
    if timeframe is None:
        return Timeframe.H1.value
    try:
        tf = Timeframe.from_value(timeframe)
        return tf.value
    except (ValueError, TypeError):
        return Timeframe.H1.value


def _deep_merge_indicator_params(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep-merge indicator parameter dictionaries."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_indicator_params(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def indicator_defaults_for(timeframe: Union[str, Timeframe]) -> Dict[str, Any]:
    """Return indicator parameter defaults for the given timeframe.

    Falls back to 1h defaults when the requested timeframe is unsupported.
    """
    normalized = _normalize_timeframe_key(timeframe)
    if normalized not in _AVAILABLE_TIMEFRAMES:
        normalized = Timeframe.H1.value
    defaults = deepcopy(_BASE_INDICATOR_DEFAULTS)
    overrides = _TIMEFRAME_INDICATOR_OVERRIDES.get(normalized)
    if overrides:
        defaults = _deep_merge_indicator_params(defaults, overrides)
    return defaults


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""
    
    # Data requirements
    lookback_days: int = 730  # 2 years default
    min_data_points: int = 1000
    min_data_points_per_timeframe: Optional[Dict[str, int]] = None
    data_source: Optional[HistoricalDataSource] = None
    validate_real_data: bool = True
    
    # Split configuration
    split_method: Literal["walk_forward", "k_fold", "time_split"] = "walk_forward"
    train_ratio: float = 0.7
    n_folds: int = 5
    
    # Performance targets
    target_win_rate: float = 0.55
    target_profit_factor: float = 1.5
    max_drawdown_target: float = 0.25
    target_sharpe: float = 1.0
    
    # Position limits
    max_concurrent_same_direction: int = 3
    min_confirmation_categories: int = 3
    
    # Search parameters
    search_method: Literal["grid", "random", "bayesian"] = "grid"
    max_iterations: int = 100
    random_seed: Optional[int] = None
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "lookback_days": self.lookback_days,
            "min_data_points": self.min_data_points,
            "min_data_points_per_timeframe": self.min_data_points_per_timeframe or {},
            "validate_real_data": self.validate_real_data,
            "split_method": self.split_method,
            "train_ratio": self.train_ratio,
            "n_folds": self.n_folds,
            "target_win_rate": self.target_win_rate,
            "target_profit_factor": self.target_profit_factor,
            "max_drawdown_target": self.max_drawdown_target,
            "target_sharpe": self.target_sharpe,
            "max_concurrent_same_direction": self.max_concurrent_same_direction,
            "min_confirmation_categories": self.min_confirmation_categories,
            "search_method": self.search_method,
            "max_iterations": self.max_iterations,
            "random_seed": self.random_seed,
        }


@dataclass
class ParameterSet:
    """Parameter bundle used for backtesting, optimization, and live analysis.

    Attributes:
        weights: Raw weighting coefficients configured for the trading factors.
        indicator_params: Indicator configuration keyed by indicator name.
        category_weights: Optional category weights overriding ``weights`` for aggregation.
        timeframe: Primary timeframe the parameters are tuned for.
        stop_loss_pct: Percentage stop loss applied to each position.
        take_profit_pct: Percentage take profit applied to each position.
        max_position_size_pct: Maximum position size as a fraction of equity.
        confirmation_threshold: Minimum aggregate factor score required to act on a signal.
        debug_enabled: Whether analyzer diagnostics should be attached to outputs.
        extras: Arbitrary metadata attached to the parameter bundle.
    """

    weights: Dict[str, float] = field(default_factory=lambda: deepcopy(DEFAULT_WEIGHTS))
    indicator_params: Dict[str, Any] = field(default_factory=dict)
    category_weights: Dict[str, float] = field(default_factory=dict)
    timeframe: Union[str, Timeframe] = Timeframe.H1.value
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_position_size_pct: float = 0.05
    confirmation_threshold: float = 0.6
    signal_thresholds: Dict[str, float] = field(default_factory=lambda: deepcopy(DEFAULT_SIGNAL_THRESHOLDS))
    debug_enabled: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    _normalized_category_weights: Dict[str, float] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and complete parameter configuration after initialization."""
        self.timeframe = _normalize_timeframe_key(self.timeframe)
        if self.timeframe not in _AVAILABLE_TIMEFRAMES:
            logger.warning(
                "ParameterSet received unsupported timeframe '%s'; defaulting to '%s'",
                self.timeframe,
                Timeframe.H1.value,
            )
            self.timeframe = Timeframe.H1.value

        fallback_weights = dict(DEFAULT_WEIGHTS)
        fallback_weights.setdefault("composite", 0.0)

        self.weights = self._sanitize_weights(self.weights, fallback_weights)
        category_source = self.category_weights or self.weights
        self.category_weights = self._sanitize_weights(category_source, fallback_weights)
        self._normalized_category_weights = self._normalize_weight_map(self.category_weights)

        if not isinstance(self.extras, dict):
            self.extras = {}

        defaults = indicator_defaults_for(self.timeframe)
        user_params = self.indicator_params if isinstance(self.indicator_params, dict) else {}
        if self.indicator_params and not isinstance(self.indicator_params, dict):
            logger.warning(
                "ParameterSet indicator_params must be a dict; received %s. Resetting to defaults.",
                type(self.indicator_params).__name__,
            )
            user_params = {}

        missing_keys: List[str] = []
        if user_params:
            missing_keys = [key for key in defaults if key not in user_params]
            if missing_keys:
                log_key = (self.timeframe, tuple(sorted(missing_keys)))
                if log_key not in _MISSING_INDICATOR_DEBUG_LOGGED:
                    logger.debug(
                        "ParameterSet auto-filled missing indicator params for timeframe '%s': %s",
                        self.timeframe,
                        ", ".join(sorted(missing_keys)),
                    )
                    _MISSING_INDICATOR_DEBUG_LOGGED.add(log_key)
        unsupported_keys = [key for key in user_params if key not in defaults]
        if unsupported_keys:
            logger.warning(
                "ParameterSet received unsupported indicator params: %s",
                ", ".join(sorted(unsupported_keys)),
            )

        merged_params = deepcopy(defaults)
        for key, value in user_params.items():
            if isinstance(value, dict) and isinstance(merged_params.get(key), dict):
                merged_params[key] = _deep_merge_indicator_params(merged_params[key], value)
            else:
                merged_params[key] = deepcopy(value)

        self.indicator_params = merged_params
        self.signal_thresholds = self._sanitize_signal_thresholds(self.signal_thresholds)

    @staticmethod
    def _sanitize_weights(weights: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, float]:
        """Sanitize a weight mapping ensuring non-negative values and fallback defaults."""
        sanitized: Dict[str, float] = {}
        if isinstance(weights, dict):
            for key, value in weights.items():
                try:
                    sanitized[key] = max(0.0, float(value))
                except (TypeError, ValueError):
                    continue
        if not sanitized:
            sanitized = deepcopy(fallback)
        else:
            for key, value in fallback.items():
                sanitized.setdefault(key, float(value))
            if sum(sanitized.values()) <= 0:
                sanitized = deepcopy(fallback)
        return sanitized

    @staticmethod
    def _sanitize_signal_thresholds(thresholds: Dict[str, Any]) -> Dict[str, float]:
        """Sanitize signal thresholds ensuring buy > sell within [0, 1]."""
        sanitized: Dict[str, float] = {}
        if isinstance(thresholds, dict):
            for key in ("buy", "sell"):
                value = thresholds.get(key)
                try:
                    sanitized[key] = float(value)
                except (TypeError, ValueError):
                    continue

        buy = sanitized.get("buy", DEFAULT_SIGNAL_THRESHOLDS["buy"])
        sell = sanitized.get("sell", DEFAULT_SIGNAL_THRESHOLDS["sell"])

        buy = min(max(buy, 0.0), 1.0)
        sell = min(max(sell, 0.0), 1.0)

        if buy <= sell:
            # Attempt to swap if that yields a valid ordering
            buy, sell = max(buy, sell), min(buy, sell)
            if buy <= sell:
                buy = min(1.0, max(sell + 0.01, DEFAULT_SIGNAL_THRESHOLDS["buy"]))
            if buy > 1.0:
                buy = 1.0
            if buy <= sell:
                sell = max(0.0, min(buy - 0.01, DEFAULT_SIGNAL_THRESHOLDS["sell"]))

        if buy <= sell:
            buy = DEFAULT_SIGNAL_THRESHOLDS["buy"]
            sell = DEFAULT_SIGNAL_THRESHOLDS["sell"]

        return {"buy": round(buy, 4), "sell": round(sell, 4)}

    @staticmethod
    def _normalize_weight_map(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {key: 0.0 for key in weights}
        return {key: value / total for key, value in weights.items()}

    def normalized_category_weights(self) -> Dict[str, float]:
        """Return normalized category weights (summing to 1)."""
        return dict(self._normalized_category_weights)

    def get_indicator_group(self, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a defensive copy of an indicator parameter group."""
        group = self.indicator_params.get(name, {})
        if not isinstance(group, dict):
            return deepcopy(default or {})
        return deepcopy(group)

    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "weights": dict(self.weights),
            "indicator_params": deepcopy(self.indicator_params),
            "category_weights": dict(self.category_weights),
            "timeframe": self.timeframe,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_position_size_pct": self.max_position_size_pct,
            "confirmation_threshold": self.confirmation_threshold,
            "signal_thresholds": deepcopy(self.signal_thresholds),
            "debug_enabled": self.debug_enabled,
            "extras": deepcopy(self.extras),
        }

    def to_signature_payload(self) -> Dict[str, Any]:
        """Create a stable payload for cache/dirty detection."""
        return {
            "timeframe": self.timeframe,
            "category_weights": self.normalized_category_weights(),
            "indicator_params": self.indicator_params,
            "signal_thresholds": self.signal_thresholds,
            "debug_enabled": self.debug_enabled,
            "extras": self.extras,
        }

    def params_hash(self) -> str:
        """Stable hash representing the parameter configuration."""
        payload = json.dumps(self.to_signature_payload(), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ParameterSet":
        """Create from dictionary."""
        weights = data.get("weights", {})
        if not isinstance(weights, dict):
            weights = dict(weights)
        indicator_params = data.get("indicator_params", {})
        if not isinstance(indicator_params, dict):
            indicator_params = dict(indicator_params)
        category_weights = data.get("category_weights", {})
        if not isinstance(category_weights, dict):
            category_weights = dict(category_weights)
        extras = data.get("extras", {})
        if not isinstance(extras, dict):
            extras = {}
        return cls(
            weights=weights,
            indicator_params=deepcopy(indicator_params),
            category_weights=category_weights,
            timeframe=data.get("timeframe", Timeframe.HOUR_1.value),
            stop_loss_pct=float(data.get("stop_loss_pct", 2.0)),
            take_profit_pct=float(data.get("take_profit_pct", 4.0)),
            max_position_size_pct=float(data.get("max_position_size_pct", 0.05)),
            confirmation_threshold=float(data.get("confirmation_threshold", 0.6)),
            signal_thresholds=data.get("signal_thresholds", {}),
            debug_enabled=bool(data.get("debug_enabled", False)),
            extras=extras,
        )


@dataclass
class BacktestResult:
    """Result of a backtesting run."""
    
    parameter_set: ParameterSet
    train_kpis: PerformanceKPIs
    test_kpis: PerformanceKPIs
    train_results: List[SignalOutcome]
    test_results: List[SignalOutcome]
    targets_met: bool
    optimization_score: float
    execution_time_seconds: float
    metadata: JsonDict = field(default_factory=dict)
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "parameter_set": self.parameter_set.to_dict(),
            "train_kpis": self.train_kpis.to_dict(),
            "test_kpis": self.test_kpis.to_dict(),
            "train_results": [r.to_dict() for r in self.train_results],
            "test_results": [r.to_dict() for r in self.test_results],
            "targets_met": self.targets_met,
            "optimization_score": self.optimization_score,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": dict(self.metadata),
        }


@dataclass
class AdaptiveWeightResult:
    """Result of adaptive weight optimization."""
    
    original_weights: Dict[str, float]
    adapted_weights: Dict[str, float]
    performance_improvement: float
    adaptation_reason: str
    rolling_window_days: int
    confidence_score: float
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "original_weights": dict(self.original_weights),
            "adapted_weights": dict(self.adapted_weights),
            "performance_improvement": self.performance_improvement,
            "adaptation_reason": self.adaptation_reason,
            "rolling_window_days": self.rolling_window_days,
            "confidence_score": self.confidence_score,
        }


class Backtester:
    """Comprehensive backtesting engine for real-only signals."""
    
    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        """Initialize backtester."""
        self.config = config or BacktestConfig()
        self._historical_data: List[TradingSignalPayload] = []
        self._optimizer = StatisticsOptimizer(StatsOptimizerConfig(
            min_win_rate_target=self.config.target_win_rate,
            min_profit_factor_target=self.config.target_profit_factor,
            max_drawdown_target=self.config.max_drawdown_target,
            min_sharpe_target=self.config.target_sharpe,
        ))
    
    def load_historical_data(
        self, 
        data_source: Union[str, List[TradingSignalPayload]],
        symbol: str = "BTCUSDT",
        timeframe: str = "1h"
    ) -> int:
        """Load historical data for backtesting."""
        if isinstance(data_source, str):
            # Load from file
            try:
                with open(data_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    payloads = [self._parse_payload(item) for item in data]
                else:
                    payloads = [self._parse_payload(data)]
                
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to load historical data from {data_source}: {e}")
        
        else:
            payloads = data_source
        
        # Validate real data
        if self.config.validate_real_data:
            validated_payloads = []
            for payload in payloads:
                try:
                    validate_real_data_payload(payload.to_dict() if hasattr(payload, 'to_dict') else payload, timeframe)
                    validated_payloads.append(payload)
                except DataValidationError as e:
                    # Skip invalid data
                    continue
            payloads = validated_payloads
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_days)
        cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
        
        filtered_payloads = []
        for payload in payloads:
            timestamp = getattr(payload, 'timestamp', 0)
            if isinstance(timestamp, str):
                timestamp = int(datetime.fromisoformat(timestamp).timestamp() * 1000)
            
            if timestamp >= cutoff_timestamp:
                filtered_payloads.append(payload)
        
        # Check minimum data requirements
        if len(filtered_payloads) < self.config.min_data_points:
            raise ValueError(f"Insufficient data: {len(filtered_payloads)} < {self.config.min_data_points}")
        
        self._historical_data = filtered_payloads
        return len(filtered_payloads)
    
    def run_backtest(self, parameter_set: ParameterSet) -> BacktestResult:
        """Run backtest with given parameter set."""
        import time
        start_time = time.time()
        
        # Split data
        train_data, test_data = self._split_data(self._historical_data)
        
        # Run on training data
        train_results = self._simulate_trading(train_data, parameter_set)
        self._optimizer = StatisticsOptimizer()  # Reset optimizer
        for result in train_results:
            self._optimizer.add_signal_outcome(result)
        train_kpis = self._optimizer.calculate_kpis()
        
        # Run on test data
        test_results = self._simulate_trading(test_data, parameter_set)
        test_optimizer = StatisticsOptimizer()
        for result in test_results:
            test_optimizer.add_signal_outcome(result)
        test_kpis = test_optimizer.calculate_kpis()
        
        # Check if targets are met
        targets_met = self._check_targets_met(test_kpis)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(test_kpis)
        
        execution_time = time.time() - start_time
        
        return BacktestResult(
            parameter_set=parameter_set,
            train_kpis=train_kpis,
            test_kpis=test_kpis,
            train_results=train_results,
            test_results=test_results,
            targets_met=targets_met,
            optimization_score=optimization_score,
            execution_time_seconds=execution_time,
        )
    
    def optimize_parameters(
        self, 
        search_space: Optional[Dict[str, Any]] = None
    ) -> Tuple[ParameterSet, BacktestResult]:
        """Optimize parameters using search method."""
        if search_space is None:
            search_space = self._get_default_search_space()
        
        best_result: Optional[BacktestResult] = None
        best_parameter_set: Optional[ParameterSet] = None
        
        if self.config.search_method == "grid":
            parameter_sets = self._generate_grid_search(search_space)
        elif self.config.search_method == "random":
            parameter_sets = self._generate_random_search(search_space)
        else:
            raise ValueError(f"Unsupported search method: {self.config.search_method}")
        
        for i, param_set in enumerate(parameter_sets[:self.config.max_iterations]):
            try:
                result = self.run_backtest(param_set)
                
                if best_result is None or result.optimization_score > best_result.optimization_score:
                    best_result = result
                    best_parameter_set = param_set
                
                # Early stop if targets met with good score
                if result.targets_met and result.optimization_score > 0.8:
                    break
                    
            except Exception as e:
                # Skip invalid parameter sets
                continue
        
        if best_result is None:
            raise RuntimeError("No valid parameter sets found during optimization")
        
        return best_parameter_set, best_result
    
    def adapt_weights(
        self, 
        current_weights: Dict[str, float],
        rolling_window_days: int = 30
    ) -> AdaptiveWeightResult:
        """Adapt weights based on recent performance."""
        # Get recent data
        cutoff_date = datetime.now() - timedelta(days=rolling_window_days)
        cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
        
        recent_data = [
            payload for payload in self._historical_data
            if getattr(payload, 'timestamp', 0) >= cutoff_timestamp
        ]
        
        if len(recent_data) < 50:
            return AdaptiveWeightResult(
                original_weights=current_weights,
                adapted_weights=current_weights,
                performance_improvement=0.0,
                adaptation_reason="Insufficient recent data for adaptation",
                rolling_window_days=rolling_window_days,
                confidence_score=0.0,
            )
        
        # Test current weights
        current_params = ParameterSet(weights=current_weights)
        current_result = self.run_backtest(current_params)
        
        # Get weight suggestions from optimizer
        self._optimizer = StatisticsOptimizer()
        for outcome in current_result.test_results:
            self._optimizer.add_signal_outcome(outcome)
        
        suggestions = self._optimizer.suggest_weight_adjustments(current_weights)
        
        if not suggestions:
            return AdaptiveWeightResult(
                original_weights=current_weights,
                adapted_weights=current_weights,
                performance_improvement=0.0,
                adaptation_reason="No weight adjustments suggested",
                rolling_window_days=rolling_window_days,
                confidence_score=0.0,
            )
        
        # Apply suggested adjustments
        adapted_weights = dict(current_weights)
        total_impact = 0.0
        
        for suggestion in suggestions:
            adapted_weights[suggestion.factor_name] = suggestion.suggested_weight
            total_impact += abs(suggestion.performance_impact)
        
        # Normalize weights
        total_weight = sum(adapted_weights.values())
        if total_weight > 0:
            adapted_weights = {k: v / total_weight for k, v in adapted_weights.items()}
        
        # Test adapted weights
        adapted_params = ParameterSet(weights=adapted_weights)
        adapted_result = self.run_backtest(adapted_params)
        
        # Calculate improvement
        current_score = current_result.test_kpis.win_rate * current_result.test_kpis.profit_factor
        adapted_score = adapted_result.test_kpis.win_rate * adapted_result.test_kpis.profit_factor
        performance_improvement = (adapted_score - current_score) / current_score if current_score > 0 else 0.0
        
        # Calculate confidence
        confidence = min(1.0, len(recent_data) / 500) * min(1.0, total_impact / 10.0)
        
        adaptation_reason = f"Adapted {len(suggestions)} weights based on {len(recent_data)} recent signals"
        
        return AdaptiveWeightResult(
            original_weights=current_weights,
            adapted_weights=adapted_weights,
            performance_improvement=performance_improvement,
            adaptation_reason=adaptation_reason,
            rolling_window_days=rolling_window_days,
            confidence_score=confidence,
        )
    
    def generate_report(self, results: List[BacktestResult]) -> JsonDict:
        """Generate comprehensive backtesting report."""
        if not results:
            return {"error": "No results to report"}
        
        # Find best result
        best_result = max(results, key=lambda r: r.optimization_score)
        
        # Aggregate statistics
        all_test_kpis = [r.test_kpis for r in results]
        avg_win_rate = statistics.mean([k.win_rate for k in all_test_kpis])
        avg_profit_factor = statistics.mean([k.profit_factor for k in all_test_kpis])
        avg_sharpe = statistics.mean([k.sharpe_ratio for k in all_test_kpis])
        avg_max_dd = statistics.mean([k.max_drawdown_pct for k in all_test_kpis])
        
        # Target achievement rates
        targets_achieved = sum(1 for r in results if r.targets_met)
        achievement_rate = targets_achieved / len(results)
        
        # Parameter analysis
        weight_analysis = self._analyze_parameter_importance(results)
        
        return {
            "summary": {
                "total_runs": len(results),
                "targets_achieved": targets_achieved,
                "achievement_rate": achievement_rate,
                "best_score": best_result.optimization_score,
                "avg_execution_time": statistics.mean([r.execution_time_seconds for r in results]),
            },
            "performance_metrics": {
                "avg_win_rate": avg_win_rate,
                "avg_profit_factor": avg_profit_factor,
                "avg_sharpe_ratio": avg_sharpe,
                "avg_max_drawdown": avg_max_dd,
                "target_win_rate": self.config.target_win_rate,
                "target_profit_factor": self.config.target_profit_factor,
                "target_sharpe": self.config.target_sharpe,
                "max_drawdown_target": self.config.max_drawdown_target,
            },
            "best_parameters": best_result.parameter_set.to_dict(),
            "best_performance": best_result.test_kpis.to_dict(),
            "parameter_importance": weight_analysis,
            "detailed_results": [r.to_dict() for r in results[:10]],  # Top 10 results
        }
    
    def save_results(self, results: List[BacktestResult], filepath: str) -> None:
        """Save backtesting results to file."""
        data = {
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in results],
            "report": self.generate_report(results),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _parse_payload(self, data: JsonDict) -> TradingSignalPayload:
        """Parse payload from dictionary."""
        # This is a simplified parser - in practice, you'd use the actual payload loader
        if hasattr(data, 'to_dict'):
            return data
        
        # Create a simple payload-like object
        class SimplePayload:
            def __init__(self, d: JsonDict):
                for k, v in d.items():
                    setattr(self, k, v)
            
            def to_dict(self) -> JsonDict:
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        return SimplePayload(data)
    
    def _split_data(self, data: List[TradingSignalPayload]) -> Tuple[List[TradingSignalPayload], List[TradingSignalPayload]]:
        """Split data based on configured method."""
        if self.config.split_method == "time_split":
            split_point = int(len(data) * self.config.train_ratio)
            return data[:split_point], data[split_point:]
        
        elif self.config.split_method == "walk_forward":
            # Simple walk-forward with 70/30 split
            split_point = int(len(data) * self.config.train_ratio)
            return data[:split_point], data[split_point:]
        
        elif self.config.split_method == "k_fold":
            # For k-fold, return first fold as train, rest as test
            fold_size = len(data) // self.config.n_folds
            return data[:fold_size], data[fold_size:]
        
        else:
            raise ValueError(f"Unsupported split method: {self.config.split_method}")
    
    def _simulate_trading(self, data: List[TradingSignalPayload], params: ParameterSet) -> List[SignalOutcome]:
        """Simulate trading with given parameters."""
        outcomes = []
        open_positions: Dict[str, Dict[str, Any]] = {}
        
        for i, payload in enumerate(data):
            try:
                # Extract signal data
                signal_data = payload.to_dict() if hasattr(payload, 'to_dict') else payload.__dict__
                signal_type = signal_data.get("signal_type", "NEUTRAL")
                entry_price = float(signal_data.get("entry_price", signal_data.get("metadata", {}).get("entry_price", 0.0)))
                timestamp = int(signal_data.get("timestamp", i))
                
                if signal_type == "NEUTRAL" or entry_price <= 0:
                    continue
                
                # Check confirmation requirements
                factors = signal_data.get("factors", [])
                if len(factors) < self.config.min_confirmation_categories:
                    continue
                
                # Check position limits
                direction = "long" if signal_type == "BUY" else "short"
                open_count = sum(1 for pos in open_positions.values() if pos["direction"] == direction)
                if open_count >= self.config.max_concurrent_same_direction:
                    continue
                
                # Calculate position size
                position_size = entry_price * params.max_position_size_pct
                
                # Calculate stop loss and take profit
                if direction == "long":
                    stop_loss = entry_price * (1 - params.stop_loss_pct / 100)
                    take_profit = entry_price * (1 + params.take_profit_pct / 100)
                else:
                    stop_loss = entry_price * (1 + params.stop_loss_pct / 100)
                    take_profit = entry_price * (1 - params.take_profit_pct / 100)
                
                # Open position
                position_id = f"{direction}_{timestamp}_{i}"
                open_positions[position_id] = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_timestamp": timestamp,
                    "signal_type": signal_type,
                    "factors": factors,
                }
                
            except Exception:
                # Skip invalid entries
                continue
        
        # Simulate position exits (simplified - in practice, you'd use future price data)
        for pos_id, position in open_positions.items():
            # Simulate random exit outcomes for demonstration
            import random
            random.seed(hash(pos_id))
            
            # Simulate exit price based on random walk
            entry_price = position["entry_price"]
            direction = position["direction"]
            
            # Simulate market movement
            price_change_pct = random.gauss(0, 2)  # 2% volatility
            exit_price = entry_price * (1 + price_change_pct / 100)
            
            # Check if stop loss or take profit would have been hit
            if direction == "long":
                if exit_price <= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                elif exit_price >= position["take_profit"]:
                    exit_price = position["take_profit"]
            else:
                if exit_price >= position["stop_loss"]:
                    exit_price = position["stop_loss"]
                elif exit_price <= position["take_profit"]:
                    exit_price = position["take_profit"]
            
            # Calculate P&L
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            success = pnl_pct > 0
            
            outcome = SignalOutcome(
                signal_type=position["signal_type"],
                entry_price=entry_price,
                exit_price=exit_price,
                entry_timestamp=position["entry_timestamp"],
                exit_timestamp=position["entry_timestamp"] + 86400000,  # 1 day later
                pnl_pct=pnl_pct,
                holding_bars=24,  # Assuming 1h timeframe
                success=success,
                factors=position["factors"],
            )
            
            outcomes.append(outcome)
        
        return outcomes
    
    def _check_targets_met(self, kpis: PerformanceKPIs) -> bool:
        """Check if performance targets are met."""
        return (
            kpis.win_rate >= self.config.target_win_rate and
            kpis.profit_factor >= self.config.target_profit_factor and
            kpis.max_drawdown_pct <= self.config.max_drawdown_target and
            kpis.sharpe_ratio >= self.config.target_sharpe
        )
    
    def _calculate_optimization_score(self, kpis: PerformanceKPIs) -> float:
        """Calculate optimization score based on KPIs."""
        # Weighted score based on how well targets are met
        win_rate_score = min(1.0, kpis.win_rate / self.config.target_win_rate)
        profit_factor_score = min(1.0, kpis.profit_factor / self.config.target_profit_factor)
        sharpe_score = min(1.0, kpis.sharpe_ratio / self.config.target_sharpe)
        drawdown_score = 1.0 - min(1.0, kpis.max_drawdown_pct / self.config.max_drawdown_target)
        
        # Weighted average
        weights = [0.3, 0.3, 0.25, 0.15]  # win_rate, profit_factor, sharpe, drawdown
        scores = [win_rate_score, profit_factor_score, sharpe_score, drawdown_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default parameter search space."""
        return {
            "weights": {
                "technical": (0.1, 0.5),
                "volume": (0.1, 0.4),
                "sentiment": (0.05, 0.3),
                "onchain": (0.05, 0.3),
                "market_structure": (0.1, 0.4),
            },
            "stop_loss_pct": (1.0, 5.0),
            "take_profit_pct": (2.0, 8.0),
            "max_position_size_pct": (0.01, 0.1),
            "confirmation_threshold": (0.5, 0.8),
        }
    
    def _generate_grid_search(self, search_space: Dict[str, Any]) -> List[ParameterSet]:
        """Generate parameter sets for grid search."""
        import itertools
        
        # Generate grid for each parameter
        weight_grids = {}
        for param, (min_val, max_val) in search_space["weights"].items():
            weight_grids[param] = [min_val + i * (max_val - min_val) / 4 for i in range(5)]  # 5 points
        
        # Generate grids for other parameters
        sl_grid = [search_space["stop_loss_pct"][0] + i * (search_space["stop_loss_pct"][1] - search_space["stop_loss_pct"][0]) / 4 for i in range(5)]
        tp_grid = [search_space["take_profit_pct"][0] + i * (search_space["take_profit_pct"][1] - search_space["take_profit_pct"][0]) / 4 for i in range(5)]
        size_grid = [search_space["max_position_size_pct"][0] + i * (search_space["max_position_size_pct"][1] - search_space["max_position_size_pct"][0]) / 4 for i in range(5)]
        conf_grid = [search_space["confirmation_threshold"][0] + i * (search_space["confirmation_threshold"][1] - search_space["confirmation_threshold"][0]) / 4 for i in range(5)]
        
        # Generate all combinations
        parameter_sets = []
        
        # Sample combinations to avoid explosion
        import random
        if self.config.random_seed:
            random.seed(self.config.random_seed)
        
        for _ in range(min(50, len(list(itertools.product(*[sl_grid, tp_grid, size_grid, conf_grid]))))):
            sl = random.choice(sl_grid)
            tp = random.choice(tp_grid)
            size = random.choice(size_grid)
            conf = random.choice(conf_grid)
            
            # Generate weights
            weights = {}
            for param, grid in weight_grids.items():
                weights[param] = random.choice(grid)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            parameter_sets.append(ParameterSet(
                weights=weights,
                indicator_params={},
                stop_loss_pct=sl,
                take_profit_pct=tp,
                max_position_size_pct=size,
                confirmation_threshold=conf,
            ))
        
        return parameter_sets
    
    def _generate_random_search(self, search_space: Dict[str, Any]) -> List[ParameterSet]:
        """Generate parameter sets for random search."""
        import random
        if self.config.random_seed:
            random.seed(self.config.random_seed)
        
        parameter_sets = []
        
        for _ in range(self.config.max_iterations):
            # Random weights
            weights = {}
            for param, (min_val, max_val) in search_space["weights"].items():
                weights[param] = random.uniform(min_val, max_val)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Random other parameters
            parameter_sets.append(ParameterSet(
                weights=weights,
                indicator_params={},
                stop_loss_pct=random.uniform(*search_space["stop_loss_pct"]),
                take_profit_pct=random.uniform(*search_space["take_profit_pct"]),
                max_position_size_pct=random.uniform(*search_space["max_position_size_pct"]),
                confirmation_threshold=random.uniform(*search_space["confirmation_threshold"]),
            ))
        
        return parameter_sets
    
    def _analyze_parameter_importance(self, results: List[BacktestResult]) -> JsonDict:
        """Analyze parameter importance across results."""
        if len(results) < 5:
            return {"error": "Insufficient results for analysis"}
        
        # Correlate parameters with performance
        scores = [r.optimization_score for r in results]
        
        weight_importance = {}
        for weight_name in results[0].parameter_set.weights.keys():
            weight_values = [r.parameter_set.weights.get(weight_name, 0) for r in results]
            correlation = self._calculate_correlation(weight_values, scores)
            weight_importance[weight_name] = correlation
        
        # Sort by importance
        sorted_importance = dict(sorted(weight_importance.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return {
            "weight_importance": sorted_importance,
            "total_results": len(results),
            "analysis_method": "pearson_correlation",
        }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            return statistics.correlation(x, y)
        except statistics.StatisticsError:
            return 0.0