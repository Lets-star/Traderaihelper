"""Trading system statistics optimizer for performance tracking and weight optimization."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .interfaces import (
    AnalyzerContext,
    JsonDict,
    OptimizationStats,
    TradingSignalPayload,
)


@dataclass
class SignalOutcome:
    """Outcome of a trading signal with performance metrics."""
    
    signal_type: Literal["BUY", "SELL", "NEUTRAL"]
    entry_price: float
    exit_price: Optional[float] = None
    entry_timestamp: int = 0
    exit_timestamp: Optional[int] = None
    timestamp: int = 0  # Signal generation timestamp
    symbol: str = ""   # Trading symbol
    pnl_pct: Optional[float] = None
    holding_bars: Optional[int] = None
    success: Optional[bool] = None
    factors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary for serialization."""
        return {
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_timestamp": self.entry_timestamp,
            "exit_timestamp": self.exit_timestamp,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "pnl_pct": self.pnl_pct,
            "holding_bars": self.holding_bars,
            "success": self.success,
            "factors": list(self.factors),
            "metadata": dict(self.metadata),
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "SignalOutcome":
        """Create from dictionary."""
        return cls(
            signal_type=data.get("signal_type", "NEUTRAL"),
            entry_price=float(data.get("entry_price", 0.0)),
            exit_price=data.get("exit_price"),
            entry_timestamp=int(data.get("entry_timestamp", 0) or 0),
            exit_timestamp=data.get("exit_timestamp"),
            timestamp=int(data.get("timestamp", 0) or 0),
            symbol=str(data.get("symbol", "")),
            pnl_pct=data.get("pnl_pct"),
            holding_bars=data.get("holding_bars"),
            success=data.get("success"),
            factors=list(data.get("factors", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PerformanceKPIs:
    """Key Performance Indicators for trading system performance."""
    
    total_signals: int = 0
    profitable_signals: int = 0
    losing_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win_pct: float = 0.0
    largest_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_holding_bars: float = 0.0
    total_return_pct: float = 0.0
    volatility_pct: float = 0.0
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "total_signals": self.total_signals,
            "profitable_signals": self.profitable_signals,
            "losing_signals": self.losing_signals,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_profit_pct": self.avg_profit_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "largest_win_pct": self.largest_win_pct,
            "largest_loss_pct": self.largest_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "avg_holding_bars": self.avg_holding_bars,
            "total_return_pct": self.total_return_pct,
            "volatility_pct": self.volatility_pct,
        }
    
    def to_optimization_stats(self) -> OptimizationStats:
        """Convert to OptimizationStats for compatibility."""
        return OptimizationStats(
            backtest_win_rate=self.win_rate,
            avg_profit_pct=self.avg_profit_pct,
            avg_loss_pct=self.avg_loss_pct,
            sharpe_ratio=self.sharpe_ratio,
            total_signals=self.total_signals,
            profitable_signals=self.profitable_signals,
            losing_signals=self.losing_signals,
        )


@dataclass
class WeightAdjustment:
    """Suggested weight adjustment for a factor."""
    
    factor_name: str
    current_weight: float
    suggested_weight: float
    adjustment_reason: str
    confidence: float
    performance_impact: float
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "factor_name": self.factor_name,
            "current_weight": self.current_weight,
            "suggested_weight": self.suggested_weight,
            "adjustment_reason": self.adjustment_reason,
            "confidence": self.confidence,
            "performance_impact": self.performance_impact,
        }


@dataclass
class OptimizationResult:
    """Result of weight optimization with performance targets."""
    
    current_kpis: PerformanceKPIs
    target_kpis: PerformanceKPIs
    projected_kpis: PerformanceKPIs
    weight_adjustments: List[WeightAdjustment]
    can_meet_targets: bool
    optimization_score: float
    recommendations: List[str]
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "current_kpis": self.current_kpis.to_dict(),
            "target_kpis": self.target_kpis.to_dict(),
            "projected_kpis": self.projected_kpis.to_dict(),
            "weight_adjustments": [adj.to_dict() for adj in self.weight_adjustments],
            "can_meet_targets": self.can_meet_targets,
            "optimization_score": self.optimization_score,
            "recommendations": list(self.recommendations),
        }


@dataclass
class StatsOptimizerConfig:
    """Configuration for statistics optimizer."""
    
    min_win_rate_target: float = 0.55
    min_profit_factor_target: float = 1.5
    max_drawdown_target: float = 0.20
    min_sharpe_target: float = 1.0
    min_signals_for_analysis: int = 30
    weight_adjustment_factor: float = 0.1
    factor_performance_window: int = 50
    enable_volatility_filter: bool = True
    max_volatility_pct: float = 2.0
    
    def to_dict(self) -> JsonDict:
        """Convert to dictionary."""
        return {
            "min_win_rate_target": self.min_win_rate_target,
            "min_profit_factor_target": self.min_profit_factor_target,
            "max_drawdown_target": self.max_drawdown_target,
            "min_sharpe_target": self.min_sharpe_target,
            "min_signals_for_analysis": self.min_signals_for_analysis,
            "weight_adjustment_factor": self.weight_adjustment_factor,
            "factor_performance_window": self.factor_performance_window,
            "enable_volatility_filter": self.enable_volatility_filter,
            "max_volatility_pct": self.max_volatility_pct,
        }
    
    @classmethod
    def from_dict(cls, data: JsonDict) -> "StatsOptimizerConfig":
        """Create from dictionary."""
        return cls(
            min_win_rate_target=float(data.get("min_win_rate_target", 0.55)),
            min_profit_factor_target=float(data.get("min_profit_factor_target", 1.5)),
            max_drawdown_target=float(data.get("max_drawdown_target", 0.20)),
            min_sharpe_target=float(data.get("min_sharpe_target", 1.0)),
            min_signals_for_analysis=int(data.get("min_signals_for_analysis", 30)),
            weight_adjustment_factor=float(data.get("weight_adjustment_factor", 0.1)),
            factor_performance_window=int(data.get("factor_performance_window", 50)),
            enable_volatility_filter=bool(data.get("enable_volatility_filter", True)),
            max_volatility_pct=float(data.get("max_volatility_pct", 2.0)),
        )


class StatisticsOptimizer:
    """Statistics optimizer for tracking signal outcomes and optimizing weights."""
    
    def __init__(self, config: Optional[StatsOptimizerConfig] = None) -> None:
        """Initialize statistics optimizer."""
        self.config = config or StatsOptimizerConfig()
        self._outcomes: List[SignalOutcome] = []
        self._factor_performance: Dict[str, List[float]] = {}
    
    def add_signal_outcome(self, outcome: SignalOutcome) -> None:
        """Add a signal outcome for tracking."""
        self._outcomes.append(outcome)
        
        # Track factor performance
        for factor in outcome.factors:
            factor_name = factor.get("factor_name", "")
            if factor_name and outcome.pnl_pct is not None:
                if factor_name not in self._factor_performance:
                    self._factor_performance[factor_name] = []
                self._factor_performance[factor_name].append(outcome.pnl_pct)
                
                # Keep only recent performance
                if len(self._factor_performance[factor_name]) > self.config.factor_performance_window:
                    self._factor_performance[factor_name] = self._factor_performance[factor_name][-self.config.factor_performance_window:]
    
    def calculate_kpis(self) -> PerformanceKPIs:
        """Calculate current performance KPIs."""
        if not self._outcomes:
            return PerformanceKPIs()
        
        completed_outcomes = [o for o in self._outcomes if o.success is not None]
        if not completed_outcomes:
            return PerformanceKPIs()
        
        total_signals = len(completed_outcomes)
        profitable_signals = sum(1 for o in completed_outcomes if o.success)
        losing_signals = total_signals - profitable_signals
        
        win_rate = profitable_signals / total_signals if total_signals > 0 else 0.0
        
        # Calculate P&L metrics
        profits = [o.pnl_pct for o in completed_outcomes if o.success and o.pnl_pct is not None]
        losses = [o.pnl_pct for o in completed_outcomes if not o.success and o.pnl_pct is not None]
        
        avg_profit_pct = statistics.mean(profits) if profits else 0.0
        avg_loss_pct = statistics.mean(losses) if losses else 0.0
        largest_win_pct = max(profits) if profits else 0.0
        largest_loss_pct = min(losses) if losses else 0.0
        
        # Calculate profit factor
        total_profit = sum(profits) if profits else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
        
        # Calculate drawdown
        returns = [o.pnl_pct for o in completed_outcomes if o.pnl_pct is not None]
        max_drawdown_pct = self._calculate_max_drawdown(returns) if returns else 0.0
        
        # Calculate Sharpe and Sortino ratios
        sharpe_ratio = self._calculate_sharpe_ratio(returns) if returns else 0.0
        sortino_ratio = self._calculate_sortino_ratio(returns) if returns else 0.0
        
        # Calculate holding period
        holding_periods = [o.holding_bars for o in completed_outcomes if o.holding_bars is not None]
        avg_holding_bars = statistics.mean(holding_periods) if holding_periods else 0.0
        
        # Calculate total return and volatility
        total_return_pct = sum(returns) if returns else 0.0
        volatility_pct = statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        return PerformanceKPIs(
            total_signals=total_signals,
            profitable_signals=profitable_signals,
            losing_signals=losing_signals,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_profit_pct=avg_profit_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win_pct=largest_win_pct,
            largest_loss_pct=largest_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            avg_holding_bars=avg_holding_bars,
            total_return_pct=total_return_pct,
            volatility_pct=volatility_pct,
        )
    
    def suggest_weight_adjustments(self, current_weights: Dict[str, float]) -> List[WeightAdjustment]:
        """Suggest weight adjustments based on factor performance."""
        adjustments = []
        
        for factor_name, current_weight in current_weights.items():
            if factor_name not in self._factor_performance:
                continue
            
            performances = self._factor_performance[factor_name]
            if len(performances) < 5:  # Need sufficient data
                continue
            
            avg_performance = statistics.mean(performances)
            performance_std = statistics.stdev(performances) if len(performances) > 1 else 0.0
            
            # Calculate adjustment based on performance
            if avg_performance > 0.5:  # Strong performer
                adjustment_factor = 1 + self.config.weight_adjustment_factor
                reason = f"Strong average performance: {avg_performance:.2f}%"
                confidence = min(0.9, len(performances) / 50)
            elif avg_performance < -0.2:  # Poor performer
                adjustment_factor = 1 - self.config.weight_adjustment_factor
                reason = f"Poor average performance: {avg_performance:.2f}%"
                confidence = min(0.8, len(performances) / 50)
            else:
                continue  # No adjustment needed
            
            suggested_weight = current_weight * adjustment_factor
            suggested_weight = max(0.0, min(2.0, suggested_weight))  # Keep weights reasonable
            
            # Estimate performance impact
            performance_impact = (suggested_weight - current_weight) * avg_performance
            
            adjustments.append(WeightAdjustment(
                factor_name=factor_name,
                current_weight=current_weight,
                suggested_weight=suggested_weight,
                adjustment_reason=reason,
                confidence=confidence,
                performance_impact=performance_impact,
            ))
        
        return adjustments
    
    def optimize_weights(
        self, 
        current_weights: Dict[str, float],
        target_kpis: Optional[PerformanceKPIs] = None
    ) -> OptimizationResult:
        """Optimize weights to meet performance targets."""
        current_kpis = self.calculate_kpis()
        
        if target_kpis is None:
            target_kpis = PerformanceKPIs(
                win_rate=self.config.min_win_rate_target,
                profit_factor=self.config.min_profit_factor_target,
                max_drawdown_pct=self.config.max_drawdown_target,
                sharpe_ratio=self.config.min_sharpe_target,
            )
        
        # Check if we have enough data
        if current_kpis.total_signals < self.config.min_signals_for_analysis:
            return OptimizationResult(
                current_kpis=current_kpis,
                target_kpis=target_kpis,
                projected_kpis=current_kpis,
                weight_adjustments=[],
                can_meet_targets=False,
                optimization_score=0.0,
                recommendations=[f"Need at least {self.config.min_signals_for_analysis} signals for optimization"],
            )
        
        # Get weight adjustments
        adjustments = self.suggest_weight_adjustments(current_weights)
        
        # Project KPIs with adjustments
        projected_kpis = self._project_kpis_with_adjustments(current_kpis, adjustments)
        
        # Check if targets can be met
        can_meet_targets = self._check_targets_met(projected_kpis, target_kpis)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(current_kpis, target_kpis, projected_kpis)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(current_kpis, target_kpis, adjustments)
        
        return OptimizationResult(
            current_kpis=current_kpis,
            target_kpis=target_kpis,
            projected_kpis=projected_kpis,
            weight_adjustments=adjustments,
            can_meet_targets=can_meet_targets,
            optimization_score=optimization_score,
            recommendations=recommendations,
        )
    
    def ingest_historical_logs(self, log_data: List[JsonDict]) -> int:
        """Ingest historical signal logs for backtesting."""
        ingested_count = 0
        
        for log_entry in log_data:
            try:
                # Extract signal data
                signal_data = log_entry.get("signal", {})
                outcome_data = log_entry.get("outcome", {})
                
                if not signal_data:
                    continue
                
                outcome = SignalOutcome(
                    signal_type=signal_data.get("signal_type", "NEUTRAL"),
                    entry_price=float(signal_data.get("entry_price", 0.0)),
                    exit_price=outcome_data.get("exit_price"),
                    entry_timestamp=int(signal_data.get("timestamp", 0) or 0),
                    exit_timestamp=outcome_data.get("exit_timestamp"),
                    pnl_pct=outcome_data.get("pnl_pct"),
                    holding_bars=outcome_data.get("holding_bars"),
                    success=outcome_data.get("success"),
                    factors=signal_data.get("factors", []),
                    metadata=log_entry.get("metadata", {}),
                )
                
                self.add_signal_outcome(outcome)
                ingested_count += 1
                
            except (ValueError, TypeError, KeyError) as e:
                # Skip invalid entries
                continue
        
        return ingested_count
    
    def get_optimization_stats(self) -> OptimizationStats:
        """Get optimization stats for compatibility with trading system."""
        kpis = self.calculate_kpis()
        
        return OptimizationStats(
            backtest_win_rate=kpis.win_rate,
            avg_profit_pct=kpis.avg_profit_pct,
            avg_loss_pct=kpis.avg_loss_pct,
            sharpe_ratio=kpis.sharpe_ratio,
            total_signals=kpis.total_signals,
            profitable_signals=kpis.profitable_signals,
            losing_signals=kpis.losing_signals,
        )
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        if not returns:
            return 0.0
        
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        
        for ret in returns:
            cumulative += ret
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        return_std = statistics.stdev(returns)
        
        if return_std == 0:
            return 0.0 if avg_return <= 0 else float('inf')
        
        # Annualize (assuming daily returns)
        annual_return = avg_return * 252
        annual_std = return_std * math.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_std
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        avg_return = statistics.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if avg_return > risk_free_rate else 0.0
        
        if len(negative_returns) < 2:
            # Not enough negative returns for standard deviation
            return 0.0 if avg_return <= risk_free_rate else float('inf')
        
        downside_std = statistics.stdev(negative_returns)
        
        if downside_std == 0:
            return 0.0 if avg_return <= risk_free_rate else float('inf')
        
        # Annualize
        annual_return = avg_return * 252
        annual_downside_std = downside_std * math.sqrt(252)
        
        return (annual_return - risk_free_rate) / annual_downside_std
    
    def _project_kpis_with_adjustments(self, current_kpis: PerformanceKPIs, adjustments: List[WeightAdjustment]) -> PerformanceKPIs:
        """Project KPIs after applying weight adjustments."""
        if not adjustments:
            return current_kpis
        
        # Simple projection based on performance impact
        total_impact = sum(adj.performance_impact for adj in adjustments)
        
        # Adjust metrics based on projected improvement
        improvement_factor = 1.0 + (total_impact / 100.0) if total_impact != 0 else 1.0
        
        projected_win_rate = min(1.0, current_kpis.win_rate * improvement_factor)
        projected_profit_factor = current_kpis.profit_factor * improvement_factor
        projected_sharpe = current_kpis.sharpe_ratio * improvement_factor
        
        return PerformanceKPIs(
            total_signals=current_kpis.total_signals,
            profitable_signals=int(current_kpis.total_signals * projected_win_rate),
            losing_signals=current_kpis.total_signals - int(current_kpis.total_signals * projected_win_rate),
            win_rate=projected_win_rate,
            profit_factor=projected_profit_factor,
            avg_profit_pct=current_kpis.avg_profit_pct * improvement_factor,
            avg_loss_pct=current_kpis.avg_loss_pct / improvement_factor,  # Losses should decrease
            largest_win_pct=current_kpis.largest_win_pct,
            largest_loss_pct=current_kpis.largest_loss_pct,
            max_drawdown_pct=current_kpis.max_drawdown_pct / improvement_factor,
            sharpe_ratio=projected_sharpe,
            sortino_ratio=current_kpis.sortino_ratio * improvement_factor,
            avg_holding_bars=current_kpis.avg_holding_bars,
            total_return_pct=current_kpis.total_return_pct * improvement_factor,
            volatility_pct=current_kpis.volatility_pct,
        )
    
    def _check_targets_met(self, kpis: PerformanceKPIs, targets: PerformanceKPIs) -> bool:
        """Check if performance targets are met."""
        return (
            kpis.win_rate >= targets.win_rate and
            kpis.profit_factor >= targets.profit_factor and
            kpis.max_drawdown_pct <= targets.max_drawdown_pct and
            kpis.sharpe_ratio >= targets.sharpe_ratio
        )
    
    def _calculate_optimization_score(self, current: PerformanceKPIs, targets: PerformanceKPIs, projected: PerformanceKPIs) -> float:
        """Calculate optimization score (0-100)."""
        score = 0.0
        
        # Win rate contribution
        if targets.win_rate > 0:
            win_rate_score = min(100, (projected.win_rate / targets.win_rate) * 100)
            score += win_rate_score * 0.3
        
        # Profit factor contribution
        if targets.profit_factor > 0:
            profit_factor_score = min(100, (projected.profit_factor / targets.profit_factor) * 100)
            score += profit_factor_score * 0.3
        
        # Sharpe ratio contribution
        if targets.sharpe_ratio > 0:
            sharpe_score = min(100, (projected.sharpe_ratio / targets.sharpe_ratio) * 100)
            score += sharpe_score * 0.2
        
        # Drawdown contribution (lower is better)
        if targets.max_drawdown_pct > 0:
            drawdown_score = min(100, (targets.max_drawdown_pct / max(projected.max_drawdown_pct, 0.001)) * 100)
            score += drawdown_score * 0.2
        
        return min(100, score)
    
    def _generate_recommendations(self, current: PerformanceKPIs, targets: PerformanceKPIs, adjustments: List[WeightAdjustment]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if current.win_rate < targets.win_rate:
            recommendations.append(f"Win rate ({current.win_rate:.1%}) below target ({targets.win_rate:.1%})")
        
        if current.profit_factor < targets.profit_factor:
            recommendations.append(f"Profit factor ({current.profit_factor:.2f}) below target ({targets.profit_factor:.2f})")
        
        if current.max_drawdown_pct > targets.max_drawdown_pct:
            recommendations.append(f"Max drawdown ({current.max_drawdown_pct:.1%}) exceeds target ({targets.max_drawdown_pct:.1%})")
        
        if current.sharpe_ratio < targets.sharpe_ratio:
            recommendations.append(f"Sharpe ratio ({current.sharpe_ratio:.2f}) below target ({targets.sharpe_ratio:.2f})")
        
        if not adjustments:
            recommendations.append("No significant factor performance differences detected")
        else:
            high_confidence_adj = [adj for adj in adjustments if adj.confidence > 0.7]
            if high_confidence_adj:
                recommendations.append(f"Consider {len(high_confidence_adj)} high-confidence weight adjustments")
        
        if current.total_signals < self.config.min_signals_for_analysis:
            recommendations.append(f"Collect more signals (need {self.config.min_signals_for_analysis}, have {current.total_signals})")
        
        if current.volatility_pct > self.config.max_volatility_pct and self.config.enable_volatility_filter:
            recommendations.append(f"High volatility detected ({current.volatility_pct:.1%}), consider reducing position sizes")
        
        return recommendations


def create_stats_optimizer(config: Optional[StatsOptimizerConfig] = None) -> StatisticsOptimizer:
    """Create a new statistics optimizer instance."""
    return StatisticsOptimizer(config)


def create_synthetic_outcomes(count: int = 100) -> List[SignalOutcome]:
    """Create synthetic signal outcomes for testing."""
    import random
    
    outcomes = []
    factors = ["technical", "sentiment", "volume", "multitimeframe"]
    
    for i in range(count):
        # Random signal type
        signal_type = random.choice(["BUY", "SELL", "NEUTRAL"])
        
        # Random entry price
        entry_price = random.uniform(100, 1000)
        
        # Random outcome
        success = random.random() > 0.4  # 60% win rate
        pnl_pct = random.uniform(0.5, 5.0) if success else random.uniform(-5.0, -0.5)
        
        # Random holding period
        holding_bars = random.randint(1, 50)
        
        # Random factors
        signal_factors = []
        for factor in factors:
            signal_factors.append({
                "factor_name": factor,
                "score": random.uniform(-1, 1),
                "weight": random.uniform(0.5, 1.5),
            })
        
        outcome = SignalOutcome(
            signal_type=signal_type,
            entry_price=entry_price,
            exit_price=entry_price * (1 + pnl_pct / 100),
            entry_timestamp=1640995200 + i * 3600,  # Start from 2022-01-01, hourly
            exit_timestamp=1640995200 + i * 3600 + holding_bars * 3600,
            pnl_pct=pnl_pct,
            holding_bars=holding_bars,
            success=success,
            factors=signal_factors,
        )
        
        outcomes.append(outcome)
    
    return outcomes