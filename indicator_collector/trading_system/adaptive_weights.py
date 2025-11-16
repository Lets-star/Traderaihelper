"""Adaptive weighting system for trading signal optimization.

This module provides adaptive weight adjustment based on rolling performance
metrics, with automatic rebalancing and performance tracking.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

from .backtester import Backtester, BacktestConfig, ParameterSet, AdaptiveWeightResult
from .interfaces import JsonDict
from .statistics_optimizer import PerformanceKPIs, SignalOutcome, StatisticsOptimizer


@dataclass
class AdaptiveWeightConfig:
    """Configuration for adaptive weighting system."""
    
    # Performance tracking
    rolling_window_days: int = 30
    min_signals_for_adaptation: int = 50
    adaptation_threshold: float = 0.05  # Minimum performance improvement to adapt
    
    # Weight constraints
    min_weight_per_factor: float = 0.05
    max_weight_per_factor: float = 0.5
    max_weight_change_pct: float = 0.3  # Maximum change per adaptation
    
    # Adaptation strategy
    adaptation_strategy: Literal["performance_based", "volatility_adjusted", "hybrid"] = "hybrid"
    rebalance_frequency_days: int = 7
    confidence_threshold: float = 0.6
    
    # Performance targets
    target_win_rate: float = 0.55
    target_profit_factor: float = 1.5
    target_sharpe: float = 1.0
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "rolling_window_days": self.rolling_window_days,
            "min_signals_for_adaptation": self.min_signals_for_adaptation,
            "adaptation_threshold": self.adaptation_threshold,
            "min_weight_per_factor": self.min_weight_per_factor,
            "max_weight_per_factor": self.max_weight_per_factor,
            "max_weight_change_pct": self.max_weight_change_pct,
            "adaptation_strategy": self.adaptation_strategy,
            "rebalance_frequency_days": self.rebalance_frequency_days,
            "confidence_threshold": self.confidence_threshold,
            "target_win_rate": self.target_win_rate,
            "target_profit_factor": self.target_profit_factor,
            "target_sharpe": self.target_sharpe,
        }


@dataclass
class WeightPerformance:
    """Performance tracking for individual weights."""
    
    factor_name: str
    current_weight: float
    rolling_win_rate: float
    rolling_profit_factor: float
    rolling_sharpe: float
    volatility_score: float
    consistency_score: float
    last_adaptation_date: Optional[datetime] = None
    adaptation_count: int = 0
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "factor_name": self.factor_name,
            "current_weight": self.current_weight,
            "rolling_win_rate": self.rolling_win_rate,
            "rolling_profit_factor": self.rolling_profit_factor,
            "rolling_sharpe": self.rolling_sharpe,
            "volatility_score": self.volatility_score,
            "consistency_score": self.consistency_score,
            "last_adaptation_date": self.last_adaptation_date.isoformat() if self.last_adaptation_date else None,
            "adaptation_count": self.adaptation_count,
        }


@dataclass
class AdaptationReport:
    """Report on weight adaptation results."""
    
    adaptation_date: datetime
    original_weights: Dict[str, float]
    new_weights: Dict[str, float]
    performance_before: PerformanceKPIs
    performance_after: Optional[PerformanceKPIs]
    adaptation_reason: str
    confidence_score: float
    expected_improvement: float
    factors_adjusted: List[str]
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "adaptation_date": self.adaptation_date.isoformat(),
            "original_weights": dict(self.original_weights),
            "new_weights": dict(self.new_weights),
            "performance_before": self.performance_before.to_dict(),
            "performance_after": self.performance_after.to_dict() if self.performance_after else None,
            "adaptation_reason": self.adaptation_reason,
            "confidence_score": self.confidence_score,
            "expected_improvement": self.expected_improvement,
            "factors_adjusted": list(self.factors_adjusted),
        }


class AdaptiveWeightManager:
    """Manages adaptive weight adjustments based on performance."""
    
    def __init__(self, config: Optional[AdaptiveWeightConfig] = None) -> None:
        """Initialize adaptive weight manager."""
        self.config = config or AdaptiveWeightConfig()
        self._backtester: Optional[Backtester] = None
        self._weight_performance: Dict[str, WeightPerformance] = {}
        self._adaptation_history: List[AdaptationReport] = []
        self._signal_history: List[SignalOutcome] = []
        self._last_rebalance_date: Optional[datetime] = None
    
    def set_backtester(self, backtester: Backtester) -> None:
        """Set backtester for performance evaluation."""
        self._backtester = backtester
    
    def initialize_weights(self, initial_weights: Dict[str, float]) -> None:
        """Initialize weight performance tracking."""
        for factor_name, weight in initial_weights.items():
            self._weight_performance[factor_name] = WeightPerformance(
                factor_name=factor_name,
                current_weight=weight,
                rolling_win_rate=0.0,
                rolling_profit_factor=0.0,
                rolling_sharpe=0.0,
                volatility_score=0.0,
                consistency_score=0.0,
            )
    
    def update_signal_outcomes(self, outcomes: List[SignalOutcome]) -> None:
        """Update signal outcomes for performance tracking."""
        self._signal_history.extend(outcomes)
        
        # Keep only recent history (rolling window)
        cutoff_date = datetime.now() - timedelta(days=self.config.rolling_window_days)
        cutoff_timestamp = int(cutoff_date.timestamp() * 1000)
        
        self._signal_history = [
            outcome for outcome in self._signal_history
            if outcome.entry_timestamp >= cutoff_timestamp
        ]
        
        # Update weight performance metrics
        self._update_weight_performance()
    
    def should_adapt(self) -> Tuple[bool, str]:
        """Check if adaptation should be performed."""
        if not self._signal_history:
            return False, "No signal history available"
        
        if len(self._signal_history) < self.config.min_signals_for_adaptation:
            return False, f"Insufficient signals: {len(self._signal_history)} < {self.config.min_signals_for_adaptation}"
        
        # Check rebalance frequency
        if self._last_rebalance_date:
            days_since_rebalance = (datetime.now() - self._last_rebalance_date).days
            if days_since_rebalance < self.config.rebalance_frequency_days:
                return False, f"Too soon since last rebalance: {days_since_rebalance} < {self.config.rebalance_frequency_days}"
        
        # Check if performance is below targets
        recent_kpis = self._calculate_recent_kpis()
        
        if recent_kpis.win_rate < self.config.target_win_rate:
            return True, f"Win rate below target: {recent_kpis.win_rate:.3f} < {self.config.target_win_rate}"
        
        if recent_kpis.profit_factor < self.config.target_profit_factor:
            return True, f"Profit factor below target: {recent_kpis.profit_factor:.3f} < {self.config.target_profit_factor}"
        
        if recent_kpis.sharpe_ratio < self.config.target_sharpe:
            return True, f"Sharpe ratio below target: {self.config.target_sharpe:.3f} < {self.config.target_sharpe}"
        
        # Check if any factor is underperforming significantly
        for factor_name, perf in self._weight_performance.items():
            if perf.rolling_win_rate < self.config.target_win_rate * 0.7:  # 30% below target
                return True, f"Factor {factor_name} underperforming: {perf.rolling_win_rate:.3f}"
        
        return False, "Performance targets met"
    
    def adapt_weights(self) -> AdaptationReport:
        """Perform adaptive weight adjustment."""
        
        # Get current weights
        current_weights = {name: perf.current_weight for name, perf in self._weight_performance.items()}
        
        # Calculate adaptation based on strategy
        if self.config.adaptation_strategy == "performance_based":
            new_weights, reason = self._adapt_performance_based(current_weights)
        elif self.config.adaptation_strategy == "volatility_adjusted":
            new_weights, reason = self._adapt_volatility_adjusted(current_weights)
        else:  # hybrid
            new_weights, reason = self._adapt_hybrid(current_weights)
        
        # Validate new weights
        new_weights = self._validate_weights(new_weights, current_weights)
        
        # Calculate expected improvement
        performance_before = self._calculate_recent_kpis()
        expected_improvement = self._estimate_improvement(current_weights, new_weights)
        
        # Create adaptation report
        report = AdaptationReport(
            adaptation_date=datetime.now(),
            original_weights=current_weights,
            new_weights=new_weights,
            performance_before=performance_before,
            performance_after=None,  # Will be updated after next evaluation
            adaptation_reason=reason,
            confidence_score=self._calculate_confidence_score(),
            expected_improvement=expected_improvement,
            factors_adjusted=[k for k, v in new_weights.items() if abs(v - current_weights.get(k, 0)) > 0.01],
        )
        
        # Update weight performance
        for factor_name, new_weight in new_weights.items():
            if factor_name in self._weight_performance:
                self._weight_performance[factor_name].current_weight = new_weight
                self._weight_performance[factor_name].last_adaptation_date = datetime.now()
                self._weight_performance[factor_name].adaptation_count += 1
        
        # Record adaptation
        self._adaptation_history.append(report)
        self._last_rebalance_date = datetime.now()
        
        return report
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current adaptive weights."""
        return {name: perf.current_weight for name, perf in self._weight_performance.items()}
    
    def get_weight_performance(self) -> Dict[str, WeightPerformance]:
        """Get performance metrics for all weights."""
        return dict(self._weight_performance)
    
    def get_adaptation_history(self) -> List[AdaptationReport]:
        """Get history of weight adaptations."""
        return list(self._adaptation_history)
    
    def generate_performance_report(self) -> JsonDict:
        """Generate comprehensive performance report."""
        current_weights = self.get_current_weights()
        recent_kpis = self._calculate_recent_kpis()
        
        # Weight analysis
        weight_analysis = {}
        for factor_name, perf in self._weight_performance.items():
            weight_analysis[factor_name] = {
                "current_weight": perf.current_weight,
                "performance_score": (perf.rolling_win_rate + perf.rolling_profit_factor / 5 + perf.rolling_sharpe / 5) / 3,
                "stability": perf.consistency_score,
                "adaptations": perf.adaptation_count,
            }
        
        # Adaptation analysis
        if self._adaptation_history:
            total_adaptations = len(self._adaptation_history)
            avg_confidence = statistics.mean([r.confidence_score for r in self._adaptation_history])
            most_common_reason = max(
                set(r.adaptation_reason for r in self._adaptation_history),
                key=self._adaptation_history.count
            )
        else:
            total_adaptations = 0
            avg_confidence = 0.0
            most_common_reason = "No adaptations yet"
        
        return {
            "summary": {
                "current_weights": current_weights,
                "recent_kpis": recent_kpis.to_dict(),
                "total_signals_analyzed": len(self._signal_history),
                "total_adaptations": total_adaptations,
                "last_adaptation": self._adaptation_history[-1].adaptation_date.isoformat() if self._adaptation_history else None,
            },
            "weight_analysis": weight_analysis,
            "adaptation_analysis": {
                "total_adaptations": total_adaptations,
                "average_confidence": avg_confidence,
                "most_common_reason": most_common_reason,
                "adaptation_frequency": total_adaptations / max(1, len(self._adaptation_history)),
            },
            "performance_vs_targets": {
                "win_rate": {
                    "current": recent_kpis.win_rate,
                    "target": self.config.target_win_rate,
                    "gap": recent_kpis.win_rate - self.config.target_win_rate,
                },
                "profit_factor": {
                    "current": recent_kpis.profit_factor,
                    "target": self.config.target_profit_factor,
                    "gap": recent_kpis.profit_factor - self.config.target_profit_factor,
                },
                "sharpe_ratio": {
                    "current": recent_kpis.sharpe_ratio,
                    "target": self.config.target_sharpe,
                    "gap": recent_kpis.sharpe_ratio - self.config.target_sharpe,
                },
            },
            "recommendations": self._generate_recommendations(),
        }
    
    def save_adaptation_history(self, filepath: str) -> None:
        """Save adaptation history to file."""
        data = {
            "config": self.config.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "current_weights": self.get_current_weights(),
            "weight_performance": {k: v.to_dict() for k, v in self._weight_performance.items()},
            "adaptation_history": [r.to_dict() for r in self._adaptation_history],
            "performance_report": self.generate_performance_report(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _update_weight_performance(self) -> None:
        """Update performance metrics for each weight."""
        if not self._signal_history:
            return
        
        # Group outcomes by factor
        factor_outcomes: Dict[str, List[SignalOutcome]] = {}
        for outcome in self._signal_history:
            for factor in outcome.factors:
                factor_name = factor.get("factor_name", "")
                if factor_name and factor_name in self._weight_performance:
                    if factor_name not in factor_outcomes:
                        factor_outcomes[factor_name] = []
                    factor_outcomes[factor_name].append(outcome)
        
        # Update metrics for each factor
        for factor_name, outcomes in factor_outcomes.items():
            if factor_name not in self._weight_performance:
                continue
            
            completed_outcomes = [o for o in outcomes if o.success is not None]
            if not completed_outcomes:
                continue
            
            # Calculate rolling metrics
            win_rate = sum(1 for o in completed_outcomes if o.success) / len(completed_outcomes)
            
            profits = [o.pnl_pct for o in completed_outcomes if o.success and o.pnl_pct is not None]
            losses = [o.pnl_pct for o in completed_outcomes if not o.success and o.pnl_pct is not None]
            
            total_profit = sum(profits) if profits else 0.0
            total_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
            
            # Calculate Sharpe-like ratio
            returns = [o.pnl_pct for o in completed_outcomes if o.pnl_pct is not None]
            sharpe_ratio = self._calculate_sharpe_ratio(returns) if len(returns) > 1 else 0.0
            
            # Calculate volatility and consistency
            volatility_score = statistics.stdev(returns) if len(returns) > 1 else 0.0
            consistency_score = 1.0 - min(1.0, volatility_score / 5.0)  # Normalize to 0-1
            
            # Update performance
            perf = self._weight_performance[factor_name]
            perf.rolling_win_rate = win_rate
            perf.rolling_profit_factor = profit_factor
            perf.rolling_sharpe = sharpe_ratio
            perf.volatility_score = volatility_score
            perf.consistency_score = consistency_score
    
    def _adapt_performance_based(self, current_weights: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """Adapt weights based on performance metrics."""
        new_weights = dict(current_weights)
        adjustments = []
        
        for factor_name, perf in self._weight_performance.items():
            if factor_name not in current_weights:
                continue
            
            current_weight = current_weights[factor_name]
            
            # Calculate performance score
            performance_score = (
                perf.rolling_win_rate * 0.4 +
                min(1.0, perf.rolling_profit_factor / 2.0) * 0.3 +
                min(1.0, perf.rolling_sharpe / 2.0) * 0.2 +
                perf.consistency_score * 0.1
            )
            
            # Adjust weight based on performance
            if performance_score > 0.7:  # High performer
                adjustment_factor = 1.1
                reason = f"High performance score: {performance_score:.3f}"
            elif performance_score < 0.4:  # Low performer
                adjustment_factor = 0.9
                reason = f"Low performance score: {performance_score:.3f}"
            else:
                continue  # No adjustment
            
            new_weight = current_weight * adjustment_factor
            new_weights[factor_name] = new_weight
            adjustments.append(f"{factor_name}: {current_weight:.3f} -> {new_weight:.3f}")
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        reason = f"Performance-based adaptation: {', '.join(adjustments)}"
        return new_weights, reason
    
    def _adapt_volatility_adjusted(self, current_weights: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """Adapt weights with volatility adjustment."""
        new_weights = dict(current_weights)
        adjustments = []
        
        for factor_name, perf in self._weight_performance.items():
            if factor_name not in current_weights:
                continue
            
            current_weight = current_weights[factor_name]
            
            # Adjust based on volatility (lower volatility = higher weight)
            volatility_adjustment = 1.0 - (perf.volatility_score / 10.0)  # Normalize
            volatility_adjustment = max(0.8, min(1.2, volatility_adjustment))
            
            # Also consider performance
            performance_multiplier = 1.0
            if perf.rolling_win_rate > self.config.target_win_rate:
                performance_multiplier = 1.05
            elif perf.rolling_win_rate < self.config.target_win_rate * 0.8:
                performance_multiplier = 0.95
            
            # Combined adjustment
            adjustment_factor = volatility_adjustment * performance_multiplier
            new_weight = current_weight * adjustment_factor
            
            if abs(new_weight - current_weight) > 0.01:
                new_weights[factor_name] = new_weight
                adjustments.append(f"{factor_name}: {current_weight:.3f} -> {new_weight:.3f}")
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        reason = f"Volatility-adjusted adaptation: {', '.join(adjustments)}"
        return new_weights, reason
    
    def _adapt_hybrid(self, current_weights: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """Adapt weights using hybrid approach."""
        # Get both adaptations
        perf_weights, perf_reason = self._adapt_performance_based(current_weights)
        vol_weights, vol_reason = self._adapt_volatility_adjusted(current_weights)
        
        # Combine with weighted average
        new_weights = {}
        adjustments = []
        
        for factor_name in current_weights:
            perf_weight = perf_weights.get(factor_name, current_weights[factor_name])
            vol_weight = vol_weights.get(factor_name, current_weights[factor_name])
            
            # Weight more towards performance (60%) than volatility (40%)
            combined_weight = perf_weight * 0.6 + vol_weight * 0.4
            new_weights[factor_name] = combined_weight
            
            if abs(combined_weight - current_weights[factor_name]) > 0.01:
                adjustments.append(f"{factor_name}: {current_weights[factor_name]:.3f} -> {combined_weight:.3f}")
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        reason = f"Hybrid adaptation (60% performance, 40% volatility): {', '.join(adjustments)}"
        return new_weights, reason
    
    def _validate_weights(self, new_weights: Dict[str, float], original_weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and constrain weight adjustments."""
        validated_weights = {}
        
        for factor_name, new_weight in new_weights.items():
            original_weight = original_weights.get(factor_name, 0.0)
            
            # Check min/max constraints
            constrained_weight = max(self.config.min_weight_per_factor, 
                                    min(self.config.max_weight_per_factor, new_weight))
            
            # Check maximum change constraint
            max_change = original_weight * self.config.max_weight_change_pct
            if abs(constrained_weight - original_weight) > max_change:
                if constrained_weight > original_weight:
                    constrained_weight = original_weight + max_change
                else:
                    constrained_weight = original_weight - max_change
            
            validated_weights[factor_name] = constrained_weight
        
        # Normalize to ensure sum = 1.0
        total_weight = sum(validated_weights.values())
        if total_weight > 0:
            validated_weights = {k: v / total_weight for k, v in validated_weights.items()}
        
        return validated_weights
    
    def _calculate_recent_kpis(self) -> PerformanceKPIs:
        """Calculate recent performance KPIs."""
        if not self._signal_history:
            return PerformanceKPIs()
        
        optimizer = StatisticsOptimizer()
        for outcome in self._signal_history:
            optimizer.add_signal_outcome(outcome)
        
        return optimizer.calculate_kpis()
    
    def _estimate_improvement(self, old_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
        """Estimate performance improvement from weight changes."""
        # Simple estimate based on factor performance
        total_improvement = 0.0
        
        for factor_name, new_weight in new_weights.items():
            if factor_name in self._weight_performance and factor_name in old_weights:
                perf = self._weight_performance[factor_name]
                old_weight = old_weights[factor_name]
                
                # Estimate improvement based on factor performance
                factor_performance = (perf.rolling_win_rate + perf.rolling_profit_factor / 5) / 2
                weight_change = new_weight - old_weight
                total_improvement += weight_change * factor_performance
        
        return total_improvement
    
    def _calculate_confidence_score(self) -> float:
        """Calculate confidence score for adaptation."""
        if not self._signal_history:
            return 0.0
        
        # Base confidence on signal count
        signal_confidence = min(1.0, len(self._signal_history) / self.config.min_signals_for_adaptation)
        
        # Adjust based on consistency of performance
        consistency_scores = [perf.consistency_score for perf in self._weight_performance.values()]
        avg_consistency = statistics.mean(consistency_scores) if consistency_scores else 0.0
        
        # Combine confidence factors
        confidence = (signal_confidence * 0.6 + avg_consistency * 0.4)
        
        return min(1.0, confidence)
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for returns."""
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        return_std = statistics.stdev(returns)
        
        if return_std == 0:
            return 0.0 if avg_return <= 0 else float('inf')
        
        return avg_return / return_std
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current performance."""
        recommendations = []
        recent_kpis = self._calculate_recent_kpis()
        
        if recent_kpis.win_rate < self.config.target_win_rate:
            recommendations.append(
                f"Consider increasing confirmation requirements to improve win rate "
                f"(current: {recent_kpis.win_rate:.3f}, target: {self.config.target_win_rate:.3f})"
            )
        
        if recent_kpis.profit_factor < self.config.target_profit_factor:
            recommendations.append(
                f"Profit factor below target - review stop-loss and take-profit levels "
                f"(current: {recent_kpis.profit_factor:.3f}, target: {self.config.target_profit_factor:.3f})"
            )
        
        if recent_kpis.max_drawdown_pct > 0.2:  # 20% drawdown
            recommendations.append(
                f"High maximum drawdown detected - consider reducing position sizes "
                f"(current: {recent_kpis.max_drawdown_pct:.3f})"
            )
        
        # Check for underperforming factors
        for factor_name, perf in self._weight_performance.items():
            if perf.rolling_win_rate < 0.4:  # Less than 40% win rate
                recommendations.append(
                    f"Factor '{factor_name}' underperforming with {perf.rolling_win_rate:.3f} win rate "
                    f"- consider reducing weight or disabling"
                )
        
        if not recommendations:
            recommendations.append("Performance is meeting targets - continue current strategy")
        
        return recommendations