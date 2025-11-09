from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from indicator_collector.trading_system.automated_signals import build_payload_from_candles
from indicator_collector.trading_system.generate_signals import generate_signals
from indicator_collector.trading_system.payload_loader import load_full_payload
from indicator_collector.trading_system.signal_generator import SignalConfig

Candles = Sequence[Dict[str, Any]]


@dataclass
class TradeResult:
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    entry_timestamp: int
    exit_timestamp: int
    outcome: str
    r_multiple: float
    return_pct: float


@dataclass
class BacktestResults:
    trades: List[TradeResult]
    equity_curve: List[Tuple[int, float]]
    total_return_pct: float
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    average_r_multiple: float


def _evaluate_trade(
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    future_candles: Sequence[Dict[str, Any]],
    risk_pct: float,
) -> Tuple[str, float, float, int]:
    if stop_loss is None or take_profit is None:
        return "invalid", entry_price, 0.0, 0

    exit_price = entry_price
    exit_timestamp = future_candles[-1]["ts"] if future_candles else 0
    outcome = "timeout"

    for candle in future_candles:
        high = float(candle.get("high", 0.0))
        low = float(candle.get("low", 0.0))
        open_price = float(candle.get("open", entry_price))
        ts = int(candle.get("ts", exit_timestamp))

        if direction == "BUY":
            hit_stop = low <= stop_loss
            hit_target = high >= take_profit
            if hit_stop and hit_target:
                # Resolve ambiguous intrabar move by comparing distance from open
                if abs(open_price - stop_loss) <= abs(take_profit - open_price):
                    hit_target = False
                else:
                    hit_stop = False
            if hit_stop:
                exit_price = stop_loss
                exit_timestamp = ts
                outcome = "loss"
                break
            if hit_target:
                exit_price = take_profit
                exit_timestamp = ts
                outcome = "win"
                break
        else:  # SELL
            hit_stop = high >= stop_loss
            hit_target = low <= take_profit
            if hit_stop and hit_target:
                if abs(stop_loss - open_price) <= abs(open_price - take_profit):
                    hit_target = False
                else:
                    hit_stop = False
            if hit_stop:
                exit_price = stop_loss
                exit_timestamp = ts
                outcome = "loss"
                break
            if hit_target:
                exit_price = take_profit
                exit_timestamp = ts
                outcome = "win"
                break

    if outcome == "timeout" and future_candles:
        last_close = float(future_candles[-1].get("close", exit_price))
        exit_price = last_close
        exit_timestamp = int(future_candles[-1].get("ts", exit_timestamp))

    risk_distance = abs(entry_price - stop_loss)
    if risk_distance <= 0:
        return "invalid", exit_price, 0.0, exit_timestamp

    if direction == "BUY":
        r_multiple = (exit_price - entry_price) / risk_distance
    else:
        r_multiple = (entry_price - exit_price) / risk_distance

    return_pct = r_multiple * risk_pct * 100.0
    return outcome, exit_price, return_pct, exit_timestamp


def _max_drawdown(equity_points: List[Tuple[int, float]]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for _, value in equity_points:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, drawdown)
    return max_dd * 100.0


def _sharpe_ratio(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        return 0.0
    return (mean / std_dev) * math.sqrt(len(returns))


def simulate_backtest(
    candles: Candles,
    symbol: str,
    timeframe: str,
    signal_config: SignalConfig,
    indicator_params: Dict[str, Any],
    signal_params: Dict[str, Any],
    *,
    max_bars: int = 320,
    step: int = 1,
    holding_bars: int = 24,
    min_required_bars: int = 120,
) -> BacktestResults:
    if not candles:
        raise ValueError("No candle data provided for backtest")

    sliced_candles = list(candles)[-max_bars:]
    if len(sliced_candles) < min_required_bars:
        raise ValueError(
            f"Insufficient candles ({len(sliced_candles)}) for backtest; need at least {min_required_bars}"
        )

    trades: List[TradeResult] = []
    risk_pct = float(signal_params.get("max_risk_per_trade_pct", 0.02))

    for idx in range(min_required_bars, len(sliced_candles) - 1, max(step, 1)):
        window = sliced_candles[: idx + 1]
        payload = build_payload_from_candles(symbol, timeframe, window)
        signal_payload = load_full_payload(
            payload,
            timeframe=timeframe,
            validate_real_data=False,
            signal_config=signal_config,
            indicator_params=indicator_params,
        )
        processed_payload = signal_payload.to_dict()

        signal_generation_params = dict(signal_params or {})
        signal_generation_params.setdefault("indicator_params", indicator_params)
        signal_generation_params.setdefault("weights", {
            "technical": signal_config.technical_weight,
            "sentiment": signal_config.sentiment_weight,
            "multitimeframe": signal_config.multitimeframe_weight,
            "volume": signal_config.volume_weight,
            "market_structure": signal_config.structure_weight,
            "composite": signal_config.composite_weight,
        })

        explicit_signal = generate_signals(processed_payload, params=signal_generation_params)
        signal_type = explicit_signal.get("signal", "HOLD")
        entries = explicit_signal.get("entries") or []
        stop_loss = explicit_signal.get("stop_loss")
        take_profits = explicit_signal.get("take_profits", {}) or {}

        if signal_type not in {"BUY", "SELL"}:
            continue
        if not entries or stop_loss is None:
            continue

        tp_key = "tp1"
        take_profit = None
        if isinstance(take_profits, dict):
            take_profit = take_profits.get(tp_key)
            if take_profit is None and take_profits:
                # fallback to first key
                first_key = sorted(take_profits.keys())[0]
                take_profit = take_profits[first_key]
        if take_profit is None:
            continue

        future_slice = sliced_candles[idx + 1 : idx + 1 + holding_bars]
        outcome, exit_price, return_pct, exit_timestamp = _evaluate_trade(
            signal_type,
            float(entries[0]),
            float(stop_loss),
            float(take_profit),
            future_slice,
            risk_pct,
        )

        if outcome == "invalid":
            continue

        trades.append(
            TradeResult(
                direction=signal_type,
                entry_price=float(entries[0]),
                exit_price=float(exit_price),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                entry_timestamp=int(window[-1]["ts"]),
                exit_timestamp=int(exit_timestamp),
                outcome=outcome,
                r_multiple=(return_pct / (risk_pct * 100.0)) if risk_pct > 0 else 0.0,
                return_pct=return_pct,
            )
        )

    account_balance = float(signal_params.get("account_balance", 10_000.0))
    equity = account_balance
    equity_curve: List[Tuple[int, float]] = []
    returns: List[float] = []

    for trade in trades:
        equity_curve.append((trade.entry_timestamp, equity))
        equity *= 1 + (trade.return_pct / 100.0)
        equity_curve.append((trade.exit_timestamp, equity))
        returns.append(trade.return_pct / 100.0)

    if not equity_curve:
        equity_curve.append((int(sliced_candles[-1]["ts"]), equity))

    wins = [trade for trade in trades if trade.outcome == "win"]
    losses = [trade for trade in trades if trade.outcome == "loss"]
    sum_gains = sum(trade.return_pct for trade in wins)
    sum_losses = sum(trade.return_pct for trade in losses)

    profit_factor = 0.0
    if sum_losses < 0:
        profit_factor = abs(sum_gains / sum_losses) if sum_losses != 0 else 0.0
    elif sum_gains > 0:
        profit_factor = float("inf")

    win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0
    total_return_pct = ((equity / account_balance) - 1.0) * 100.0
    max_drawdown_pct = _max_drawdown(equity_curve)
    sharpe_ratio = _sharpe_ratio(returns)

    avg_r = sum(trade.r_multiple for trade in trades) / len(trades) if trades else 0.0

    return BacktestResults(
        trades=trades,
        equity_curve=equity_curve,
        total_return_pct=total_return_pct,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        average_r_multiple=avg_r,
    )


__all__ = ["BacktestResults", "TradeResult", "simulate_backtest"]
