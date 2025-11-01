"""Enhanced trade signal analysis with TP levels and position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from .math_utils import Candle


@dataclass
class TradeSignalStats:
    """Detailed statistics for trade signals with multiple TP levels."""
    total_signals: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    tp3_hits: int = 0
    sl_hits: int = 0
    still_open: int = 0
    
    avg_bars_to_tp1: float = 0.0
    avg_bars_to_tp2: float = 0.0
    avg_bars_to_tp3: float = 0.0
    avg_bars_to_sl: float = 0.0
    
    @property
    def tp1_rate(self) -> float:
        return (self.tp1_hits / self.total_signals * 100) if self.total_signals else 0.0
    
    @property
    def tp2_rate(self) -> float:
        return (self.tp2_hits / self.total_signals * 100) if self.total_signals else 0.0
    
    @property
    def tp3_rate(self) -> float:
        return (self.tp3_hits / self.total_signals * 100) if self.total_signals else 0.0
    
    @property
    def sl_rate(self) -> float:
        return (self.sl_hits / self.total_signals * 100) if self.total_signals else 0.0
    
    @property
    def win_rate(self) -> float:
        winners = self.tp1_hits + self.tp2_hits + self.tp3_hits
        total_closed = winners + self.sl_hits
        return (winners / total_closed * 100) if total_closed else 0.0


def calculate_position_metrics(
    entry_price: float,
    position_size_usd: float,
    leverage: float = 10.0,
    commission_rate: float = 0.0006,
) -> Dict[str, object]:
    """
    Calculate position metrics with commission.
    
    Args:
        entry_price: Entry price for the position
        position_size_usd: Position size in USD
        leverage: Leverage multiplier
        commission_rate: Commission rate (0.06% = 0.0006 for maker/taker)
    
    Returns:
        Dictionary with position details and commission
    """
    notional_value = position_size_usd * leverage
    quantity = notional_value / entry_price
    
    entry_commission = notional_value * commission_rate
    
    return {
        "position_size_usd": position_size_usd,
        "leverage": leverage,
        "notional_value": notional_value,
        "quantity": quantity,
        "entry_commission": entry_commission,
        "commission_rate": commission_rate,
    }


def calculate_tp_sl_levels(
    entry_price: float,
    is_long: bool,
    atr_value: float,
    tp1_multiplier: float = 1.5,
    tp2_multiplier: float = 3.0,
    tp3_multiplier: float = 5.0,
    sl_multiplier: float = 1.0,
) -> Dict[str, float]:
    """
    Calculate take profit and stop loss levels based on ATR.
    
    Args:
        entry_price: Entry price
        is_long: True for long position, False for short
        atr_value: ATR value for volatility measurement
        tp1_multiplier: ATR multiplier for TP1
        tp2_multiplier: ATR multiplier for TP2
        tp3_multiplier: ATR multiplier for TP3
        sl_multiplier: ATR multiplier for stop loss
    
    Returns:
        Dictionary with TP and SL levels
    """
    if is_long:
        tp1 = entry_price + (atr_value * tp1_multiplier)
        tp2 = entry_price + (atr_value * tp2_multiplier)
        tp3 = entry_price + (atr_value * tp3_multiplier)
        sl = entry_price - (atr_value * sl_multiplier)
    else:
        tp1 = entry_price - (atr_value * tp1_multiplier)
        tp2 = entry_price - (atr_value * tp2_multiplier)
        tp3 = entry_price - (atr_value * tp3_multiplier)
        sl = entry_price + (atr_value * sl_multiplier)
    
    return {
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": sl,
    }


def evaluate_signal_performance(
    signals: List[Dict[str, object]],
    candles: Sequence[Candle],
    atr_values: Sequence[float],
    lookback_bars: int = 50,
) -> Tuple[TradeSignalStats, TradeSignalStats]:
    """
    Evaluate signal performance with multiple TP levels.
    
    Returns:
        Tuple of (bullish_stats, bearish_stats)
    """
    bull_stats = TradeSignalStats()
    bear_stats = TradeSignalStats()
    
    tp1_bars_bull = []
    tp2_bars_bull = []
    tp3_bars_bull = []
    sl_bars_bull = []
    
    tp1_bars_bear = []
    tp2_bars_bear = []
    tp3_bars_bear = []
    sl_bars_bear = []
    
    for signal in signals:
        signal_index = signal.get("bar_index", 0)
        signal_type = signal.get("type", "")
        entry_price = signal.get("price", 0)
        
        if signal_index >= len(candles) or signal_index >= len(atr_values):
            continue
        
        atr_val = atr_values[signal_index]
        if not atr_val or atr_val == 0:
            continue
        
        is_long = signal_type == "bullish"
        stats = bull_stats if is_long else bear_stats
        tp_bars_lists = (tp1_bars_bull, tp2_bars_bull, tp3_bars_bull, sl_bars_bull) if is_long else (
            tp1_bars_bear, tp2_bars_bear, tp3_bars_bear, sl_bars_bear
        )
        
        levels = calculate_tp_sl_levels(entry_price, is_long, atr_val)
        
        stats.total_signals += 1
        
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        sl_hit = False
        
        max_lookback = min(signal_index + lookback_bars, len(candles))
        
        for i in range(signal_index + 1, max_lookback):
            candle = candles[i]
            bars_elapsed = i - signal_index
            
            if is_long:
                if not tp1_hit and candle.high >= levels["tp1"]:
                    tp1_hit = True
                    stats.tp1_hits += 1
                    tp_bars_lists[0].append(bars_elapsed)
                
                if not tp2_hit and candle.high >= levels["tp2"]:
                    tp2_hit = True
                    stats.tp2_hits += 1
                    tp_bars_lists[1].append(bars_elapsed)
                
                if not tp3_hit and candle.high >= levels["tp3"]:
                    tp3_hit = True
                    stats.tp3_hits += 1
                    tp_bars_lists[2].append(bars_elapsed)
                
                if not sl_hit and candle.low <= levels["sl"]:
                    sl_hit = True
                    stats.sl_hits += 1
                    tp_bars_lists[3].append(bars_elapsed)
                    break
            else:
                if not tp1_hit and candle.low <= levels["tp1"]:
                    tp1_hit = True
                    stats.tp1_hits += 1
                    tp_bars_lists[0].append(bars_elapsed)
                
                if not tp2_hit and candle.low <= levels["tp2"]:
                    tp2_hit = True
                    stats.tp2_hits += 1
                    tp_bars_lists[1].append(bars_elapsed)
                
                if not tp3_hit and candle.low <= levels["tp3"]:
                    tp3_hit = True
                    stats.tp3_hits += 1
                    tp_bars_lists[2].append(bars_elapsed)
                
                if not sl_hit and candle.high >= levels["sl"]:
                    sl_hit = True
                    stats.sl_hits += 1
                    tp_bars_lists[3].append(bars_elapsed)
                    break
        
        if not tp1_hit and not tp2_hit and not tp3_hit and not sl_hit:
            stats.still_open += 1
    
    if tp1_bars_bull:
        bull_stats.avg_bars_to_tp1 = sum(tp1_bars_bull) / len(tp1_bars_bull)
    if tp2_bars_bull:
        bull_stats.avg_bars_to_tp2 = sum(tp2_bars_bull) / len(tp2_bars_bull)
    if tp3_bars_bull:
        bull_stats.avg_bars_to_tp3 = sum(tp3_bars_bull) / len(tp3_bars_bull)
    if sl_bars_bull:
        bull_stats.avg_bars_to_sl = sum(sl_bars_bull) / len(sl_bars_bull)
    
    if tp1_bars_bear:
        bear_stats.avg_bars_to_tp1 = sum(tp1_bars_bear) / len(tp1_bars_bear)
    if tp2_bars_bear:
        bear_stats.avg_bars_to_tp2 = sum(tp2_bars_bear) / len(tp2_bars_bear)
    if tp3_bars_bear:
        bear_stats.avg_bars_to_tp3 = sum(tp3_bars_bear) / len(tp3_bars_bear)
    if sl_bars_bear:
        bear_stats.avg_bars_to_sl = sum(sl_bars_bear) / len(sl_bars_bear)
    
    return bull_stats, bear_stats


def format_stats_to_dict(stats: TradeSignalStats, signal_type: str) -> Dict[str, object]:
    """Format TradeSignalStats to dictionary for JSON serialization."""
    return {
        "signal_type": signal_type,
        "total_signals": stats.total_signals,
        "tp1_hits": stats.tp1_hits,
        "tp2_hits": stats.tp2_hits,
        "tp3_hits": stats.tp3_hits,
        "sl_hits": stats.sl_hits,
        "still_open": stats.still_open,
        "tp1_rate_pct": round(stats.tp1_rate, 2),
        "tp2_rate_pct": round(stats.tp2_rate, 2),
        "tp3_rate_pct": round(stats.tp3_rate, 2),
        "sl_rate_pct": round(stats.sl_rate, 2),
        "overall_win_rate_pct": round(stats.win_rate, 2),
        "avg_bars_to_tp1": round(stats.avg_bars_to_tp1, 1),
        "avg_bars_to_tp2": round(stats.avg_bars_to_tp2, 1),
        "avg_bars_to_tp3": round(stats.avg_bars_to_tp3, 1),
        "avg_bars_to_sl": round(stats.avg_bars_to_sl, 1),
    }
