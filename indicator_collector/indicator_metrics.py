from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .data_fetcher import timeframe_to_minutes
from .math_utils import (
    Candle,
    ema,
    highest,
    lowest,
    mom,
    rma,
    rsi,
    sma,
)
from .time_series import MetricPoint, TimeframeMetricSeries, TimeframeSeries

ZoneType = Literal["BullFVG", "BearFVG", "BullOB", "BearOB"]


@dataclass
class Zone:
    zone_type: ZoneType
    top: float
    bottom: float
    breaker: bool = False
    created_index: int = 0


@dataclass
class SignalRecord:
    bar_index: int
    timestamp: int
    signal_type: Literal["bullish", "bearish"]
    price: float
    strength: Optional[float]


@dataclass
class IndicatorSettings:
    min_fvg_size: float = 0.001
    enable_multi_symbol: bool = True
    enable_pattern_recognition: bool = True
    pattern_lookback: int = 100
    min_pattern_accuracy: float = 65.0
    enable_sentiment: bool = True
    sentiment_period: int = 20
    weight_structure: float = 2.0
    weight_volume: float = 1.5
    weight_timeframe: float = 1.0
    weight_pattern: float = 2.5
    weight_sentiment: float = 1.2
    use_volume_filter: bool = True
    volume_multiplier: float = 1.5
    volume_lookback: int = 20
    use_structure_filter: bool = True
    structure_lookback: int = 10
    min_structure_move: float = 0.005
    trend_strength_period: int = 14
    ma_fast: int = 20
    ma_slow: int = 50
    banker_volume_threshold: float = 2.0
    banker_wick_ratio: float = 0.6
    banker_body_ratio: float = 0.3
    success_lookahead_bars: int = 10
    success_threshold_pct: float = 1.0
    max_array_size: int = 100
    show_signals: bool = True
    show_fvg: bool = True
    show_ob: bool = True


@dataclass
class ConfluenceContext:
    is_bullish: bool
    struct_bull: bool
    struct_bear: bool
    struct_neutral: bool
    volume_confirmed: bool
    multi_tf_bools: Dict[str, bool]
    pattern_prediction: Optional[float]
    market_sentiment: Optional[float]


@dataclass
class PnLStats:
    trades_closed: int = 0
    cum_pnl_pct: float = 0.0
    equity_peak_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    active_dir: int = 0
    active_entry: Optional[float] = None
    active_entry_bar: Optional[int] = None
    last_trade_pnl_pct: Optional[float] = None


@dataclass
class SuccessStats:
    total_bull_signals: int = 0
    successful_bull_signals: int = 0
    total_bear_signals: int = 0
    successful_bear_signals: int = 0

    @property
    def bull_win_rate(self) -> float:
        if self.total_bull_signals == 0:
            return 0.0
        return (self.successful_bull_signals / self.total_bull_signals) * 100

    @property
    def bear_win_rate(self) -> float:
        if self.total_bear_signals == 0:
            return 0.0
        return (self.successful_bear_signals / self.total_bear_signals) * 100

    @property
    def overall_win_rate(self) -> float:
        total = self.total_bull_signals + self.total_bear_signals
        if total == 0:
            return 0.0
        successes = self.successful_bull_signals + self.successful_bear_signals
        return (successes / total) * 100


@dataclass
class MarketSnapshot:
    timestamp: int
    close: float
    open: float
    high: float
    low: float
    volume: float
    trend_strength: float
    pattern_score: float
    sentiment: float
    structure_state: Literal["bullish", "bearish", "neutral"]
    structure_event: Optional[str]
    volume_confirmed: bool
    volume_ratio: Optional[float]
    confluence_score: Optional[float]
    signal: Optional[str]


@dataclass
class MultiSymbolSnapshot:
    signals: Dict[str, Literal["BUY", "SELL", "NEUTRAL"]]
    trend_strength: Dict[str, Optional[float]]


@dataclass
class SimulationSummary:
    snapshots: List[MarketSnapshot]
    signals: List[SignalRecord]
    pnl: PnLStats
    success: SuccessStats
    active_fvg_zones: List[Zone]
    active_ob_zones: List[Zone]
    last_structure_levels: Dict[str, Optional[float]]
    multi_timeframe_trend: Dict[str, float]
    multi_timeframe_direction: Dict[str, Literal["bullish", "bearish", "neutral"]]
    market_sentiment: float
    pattern_prediction: float
    multi_symbol: Optional[MultiSymbolSnapshot]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class IndicatorSimulator:
    def __init__(
        self,
        settings: IndicatorSettings,
        main_series: TimeframeSeries,
        multi_timeframe_series: Dict[str, TimeframeSeries],
        multi_timeframe_strength: Dict[str, TimeframeMetricSeries],
        multi_symbol_series: Optional[Dict[str, TimeframeSeries]] = None,
    ) -> None:
        self.settings = settings
        self.main_series = main_series
        self.multi_timeframe_series = multi_timeframe_series
        self.multi_timeframe_strength = multi_timeframe_strength
        self.multi_symbol_series = multi_symbol_series or {}

    def run(self) -> SimulationSummary:
        candles = list(self.main_series.candles)
        opens = [c.open for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        timestamps = [c.close_time for c in candles]

        ema_fast_series = ema(closes, self.settings.ma_fast)
        ema_slow_series = ema(closes, self.settings.ma_slow)
        rsi14 = rsi(closes, 14)
        momentum14 = mom(closes, 14)
        volume_sma = sma(volumes, self.settings.volume_lookback)
        volume_sma20 = sma(volumes, 20)
        trend_strength_series = self._calculate_trend_strength_series(closes, self.settings.trend_strength_period)
        sentiment_series = self._calculate_sentiment_series(opens, closes, volumes, self.settings.sentiment_period, rsi_values=rsi(closes, self.settings.sentiment_period))
        pattern_scores = self._calculate_pattern_scores(opens, highs, closes, volumes, ema_fast_series, ema_slow_series, rsi14, momentum14, volume_sma20)

        snapshots: List[MarketSnapshot] = []
        signals: List[SignalRecord] = []
        pnl = PnLStats()
        success = SuccessStats()

        active_fvg_zones: List[Zone] = []
        active_ob_zones: List[Zone] = []

        last_swing_high: Optional[float] = None
        last_swing_low: Optional[float] = None
        last_bos_high: Optional[float] = None
        last_bos_low: Optional[float] = None

        last_bull_fvg_top: Optional[float] = None
        last_bull_fvg_bottom: Optional[float] = None
        last_bear_fvg_top: Optional[float] = None
        last_bear_fvg_bottom: Optional[float] = None
        last_bull_ob_top: Optional[float] = None
        last_bull_ob_bottom: Optional[float] = None
        last_bear_ob_top: Optional[float] = None
        last_bear_ob_bottom: Optional[float] = None

        bull_signal_queue: List[int] = []
        bear_signal_queue: List[int] = []

        max_index = len(candles)
        for i in range(max_index):
            # Update zones lifecycle before calculations
            self._update_active_zones(active_fvg_zones, highs[i], lows[i])
            self._update_active_zones(active_ob_zones, highs[i], lows[i])

            bull_fvg, bull_fvg_top, bull_fvg_bottom, bear_fvg, bear_fvg_top, bear_fvg_bottom = self._detect_fvg(i, lows, highs)
            bull_ob, bull_ob_top, bull_ob_bottom, bear_ob, bear_ob_top, bear_ob_bottom = self._detect_order_block(i, opens, closes, highs, lows)

            if bull_fvg and self.settings.show_fvg:
                zone = Zone("BullFVG", bull_fvg_top, bull_fvg_bottom, created_index=i)
                active_fvg_zones.append(zone)
                last_bull_fvg_top = bull_fvg_top
                last_bull_fvg_bottom = bull_fvg_bottom

            if bear_fvg and self.settings.show_fvg:
                zone = Zone("BearFVG", bear_fvg_top, bear_fvg_bottom, created_index=i)
                active_fvg_zones.append(zone)
                last_bear_fvg_top = bear_fvg_top
                last_bear_fvg_bottom = bear_fvg_bottom

            if bull_ob and self.settings.show_ob:
                zone = Zone("BullOB", bull_ob_top, bull_ob_bottom, created_index=i)
                active_ob_zones.append(zone)
                last_bull_ob_top = bull_ob_top
                last_bull_ob_bottom = bull_ob_bottom

            if bear_ob and self.settings.show_ob:
                zone = Zone("BearOB", bear_ob_top, bear_ob_bottom, created_index=i)
                active_ob_zones.append(zone)
                last_bear_ob_top = bear_ob_top
                last_bear_ob_bottom = bear_ob_bottom

            swing_high, swing_high_price, swing_low, swing_low_price = self._detect_swings(i, highs, lows)
            if swing_high:
                last_swing_high = swing_high_price
            if swing_low:
                last_swing_low = swing_low_price

            structure_state = self._detect_market_structure(
                i,
                closes,
                highs,
                lows,
                self.settings.structure_lookback,
                self.settings.min_structure_move,
            )
            struct_bullish, struct_bearish, struct_neutral, bos_bull, bos_bear, choch_bull, choch_bear, new_bos_high, new_bos_low = structure_state

            if new_bos_high is not None:
                last_bos_high = new_bos_high
            if new_bos_low is not None:
                last_bos_low = new_bos_low

            volume_confirmed = self._volume_confirmed(i, volumes, volume_sma)
            volume_ratio = None
            if i < len(volume_sma) and not math.isnan(volume_sma[i]) and volume_sma[i] != 0:
                volume_ratio = volumes[i] / volume_sma[i]

            sentiment_value = sentiment_series[i] if i < len(sentiment_series) else 50.0
            pattern_score = pattern_scores[i] if i < len(pattern_scores) else 50.0

            multi_tf_flags, _ = self._multi_timeframe_context(timestamps[i])

            confluence_score = None
            signal_label: Optional[str] = None

            bullish_sync = False
            bearish_sync = False
            if self.settings.show_signals and i > 0:
                prev_close = closes[i - 1]
                prev_open = opens[i - 1]
                if last_bull_fvg_top is not None and last_bull_ob_top is not None:
                    in_fvg = self._price_in_range(prev_close, last_bull_fvg_top, last_bull_fvg_bottom)
                    in_ob = self._price_in_range(prev_close, last_bull_ob_top, last_bull_ob_bottom)
                    if in_fvg and in_ob and prev_close > prev_open and self._structure_filter(True, struct_bullish, struct_bearish, struct_neutral):
                        bullish_sync = True

                if last_bear_fvg_top is not None and last_bear_ob_top is not None:
                    in_fvg = self._price_in_range(prev_close, last_bear_fvg_top, last_bear_fvg_bottom)
                    in_ob = self._price_in_range(prev_close, last_bear_ob_top, last_bear_ob_bottom)
                    if in_fvg and in_ob and prev_close < prev_open and self._structure_filter(False, struct_bullish, struct_bearish, struct_neutral):
                        bearish_sync = True

            if bullish_sync or bearish_sync:
                confluence_score = self._calculate_confluence_score(
                    ConfluenceContext(
                        is_bullish=bullish_sync,
                        struct_bull=struct_bullish,
                        struct_bear=struct_bearish,
                        struct_neutral=struct_neutral,
                        volume_confirmed=volume_confirmed,
                        multi_tf_bools=multi_tf_flags,
                        pattern_prediction=pattern_score if self.settings.enable_pattern_recognition else None,
                        market_sentiment=sentiment_value if self.settings.enable_sentiment else None,
                    )
                )

            if bullish_sync:
                signal_label = "BUY"
                success.total_bull_signals += 1
                bull_signal_queue.append(i)
                if len(bull_signal_queue) > self.settings.max_array_size:
                    bull_signal_queue.pop(0)
                signals.append(
                    SignalRecord(
                        bar_index=i,
                        timestamp=timestamps[i],
                        signal_type="bullish",
                        price=closes[i],
                        strength=confluence_score,
                    )
                )
            elif bearish_sync:
                signal_label = "SELL"
                success.total_bear_signals += 1
                bear_signal_queue.append(i)
                if len(bear_signal_queue) > self.settings.max_array_size:
                    bear_signal_queue.pop(0)
                signals.append(
                    SignalRecord(
                        bar_index=i,
                        timestamp=timestamps[i],
                        signal_type="bearish",
                        price=closes[i],
                        strength=confluence_score,
                    )
                )

            # Update trade stats
            self._update_trades(pnl, bullish_sync, bearish_sync, confluence_score, closes[i], i, choch_bull, choch_bear)

            # Success evaluation window
            self._evaluate_signal_success(i, closes, success, bull_signal_queue, bear_signal_queue)

            structure_event = None
            if choch_bull:
                structure_event = "CHOCH_BULL"
            elif choch_bear:
                structure_event = "CHOCH_BEAR"
            elif bos_bull:
                structure_event = "BOS_BULL"
            elif bos_bear:
                structure_event = "BOS_BEAR"

            struct_state_label: Literal["bullish", "bearish", "neutral"]
            if struct_bullish:
                struct_state_label = "bullish"
            elif struct_bearish:
                struct_state_label = "bearish"
            else:
                struct_state_label = "neutral"

            snapshots.append(
                MarketSnapshot(
                    timestamp=timestamps[i],
                    close=closes[i],
                    open=opens[i],
                    high=highs[i],
                    low=lows[i],
                    volume=volumes[i],
                    trend_strength=trend_strength_series[i],
                    pattern_score=pattern_score,
                    sentiment=sentiment_value,
                    structure_state=struct_state_label,
                    structure_event=structure_event,
                    volume_confirmed=volume_confirmed,
                    volume_ratio=volume_ratio,
                    confluence_score=confluence_score,
                    signal=signal_label,
                )
            )

        multi_symbol_snapshot = None
        if self.settings.enable_multi_symbol and self.multi_symbol_series:
            signals_map: Dict[str, Literal["BUY", "SELL", "NEUTRAL"]] = {}
            strength_map: Dict[str, Optional[float]] = {}
            latest_timestamp = timestamps[-1]
            for symbol, series in self.multi_symbol_series.items():
                candle = series.candle_at(latest_timestamp)
                if candle.close > candle.open:
                    signals_map[symbol] = "BUY"
                elif candle.close < candle.open:
                    signals_map[symbol] = "SELL"
                else:
                    signals_map[symbol] = "NEUTRAL"
                strength_series = self.multi_timeframe_strength.get(f"{symbol}_trend")
                strength_map[symbol] = strength_series.value_at(latest_timestamp) if strength_series else None
            multi_symbol_snapshot = MultiSymbolSnapshot(signals=signals_map, trend_strength=strength_map)

        multi_tf_strength_latest = {
            tf: series.value_at(timestamps[-1]) for tf, series in self.multi_timeframe_strength.items() if not tf.endswith("_trend")
        }
        multi_tf_direction_latest = {}
        latest_timestamp = timestamps[-1]
        for tf, series in self.multi_timeframe_series.items():
            candle = series.candle_at(latest_timestamp)
            if candle.close > candle.open:
                multi_tf_direction_latest[tf] = "bullish"
            elif candle.close < candle.open:
                multi_tf_direction_latest[tf] = "bearish"
            else:
                multi_tf_direction_latest[tf] = "neutral"

        return SimulationSummary(
            snapshots=snapshots,
            signals=signals,
            pnl=pnl,
            success=success,
            active_fvg_zones=active_fvg_zones,
            active_ob_zones=active_ob_zones,
            last_structure_levels={"high": last_bos_high, "low": last_bos_low},
            multi_timeframe_trend=multi_tf_strength_latest,
            multi_timeframe_direction=multi_tf_direction_latest,
            market_sentiment=sentiment_series[-1] if sentiment_series else 50.0,
            pattern_prediction=pattern_scores[-1] if pattern_scores else 50.0,
            multi_symbol=multi_symbol_snapshot,
        )

    # --- Helper methods -------------------------------------------------

    def _calculate_trend_strength_series(self, closes: Sequence[float], length: int) -> List[float]:
        up_moves: List[float] = [0.0]
        down_moves: List[float] = [0.0]
        for i in range(1, len(closes)):
            change = closes[i] - closes[i - 1]
            up_moves.append(max(change, 0.0))
            down_moves.append(max(-change, 0.0))
        smooth_up = rma(up_moves, length)
        smooth_down = rma(down_moves, length)
        momentum_series = mom(closes, length)
        result: List[float] = []
        for i in range(len(closes)):
            up = smooth_up[i]
            down = smooth_down[i]
            if up + down != 0:
                strength_component = 100 * up / (up + down)
            else:
                strength_component = 50.0

            momentum = momentum_series[i]
            if math.isnan(momentum) or closes[i] == 0:
                momentum_strength = 50.0
            elif momentum > 0:
                momentum_strength = 50 + (momentum / closes[i]) * 100
            else:
                momentum_strength = 50 - (abs(momentum) / closes[i]) * 100
            momentum_strength = _clamp(momentum_strength, 0.0, 100.0)
            result.append((strength_component + momentum_strength) / 2)
        return result

    def _calculate_sentiment_series(
        self,
        opens: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        period: int,
        rsi_values: Sequence[float],
    ) -> List[float]:
        sentiment: List[float] = []
        for i in range(len(closes)):
            if i < period:
                sentiment.append(50.0)
                continue
            price_momentum = rsi_values[i]
            up_bars = 0
            down_bars = 0
            start = i - period + 1
            for j in range(start, i + 1):
                if closes[j] > opens[j]:
                    up_bars += 1
                elif closes[j] < opens[j]:
                    down_bars += 1
            if up_bars > down_bars:
                breadth = (up_bars / period) * 100
            else:
                breadth = (1 - down_bars / period) * 100
            sentiment.append((price_momentum + breadth) / 2)
        return sentiment

    def _calculate_pattern_scores(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        ema_fast: Sequence[float],
        ema_slow: Sequence[float],
        rsi_values: Sequence[float],
        momentum_values: Sequence[float],
        volume_sma20: Sequence[float],
    ) -> List[float]:
        scores: List[float] = []
        for i in range(len(closes)):
            score = 0.0
            if closes[i] > opens[i]:
                score += 10
            if i < len(volume_sma20) and not math.isnan(volume_sma20[i]) and volumes[i] > volume_sma20[i]:
                score += 10
            if i < len(ema_fast) and closes[i] > ema_fast[i]:
                score += 10
            if i < len(rsi_values) and rsi_values[i] > 50:
                score += 10
            if i > 0 and highs[i] > highs[i - 1]:
                score += 10
            momentum = momentum_values[i] if i < len(momentum_values) else float("nan")
            if not math.isnan(momentum) and momentum > 0:
                score += 20
            fast = ema_fast[i] if i < len(ema_fast) else closes[i]
            slow = ema_slow[i] if i < len(ema_slow) else closes[i]
            if closes[i] > fast and fast > slow:
                score += 30
            elif closes[i] < fast and fast < slow:
                score += 0
            else:
                score += 15
            scores.append(min(100.0, score))
        return scores

    def _detect_fvg(
        self,
        index: int,
        lows: Sequence[float],
        highs: Sequence[float],
    ) -> Tuple[bool, Optional[float], Optional[float], bool, Optional[float], Optional[float]]:
        if index < 2:
            return (False, None, None, False, None, None)
        bullish = lows[index] > highs[index - 2] and abs(lows[index] - highs[index - 2]) > self.settings.min_fvg_size
        bearish = highs[index] < lows[index - 2] and abs(lows[index - 2] - highs[index]) > self.settings.min_fvg_size
        bull_top = lows[index] if bullish else None
        bull_bottom = highs[index - 2] if bullish else None
        bear_top = lows[index - 2] if bearish else None
        bear_bottom = highs[index] if bearish else None
        return bullish, bull_top, bull_bottom, bearish, bear_top, bear_bottom

    def _detect_order_block(
        self,
        index: int,
        opens: Sequence[float],
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
    ) -> Tuple[bool, Optional[float], Optional[float], bool, Optional[float], Optional[float]]:
        if index < 3:
            return (False, None, None, False, None, None)
        bullish = False
        bullish_top = bullish_bottom = None
        bearish = False
        bearish_top = bearish_bottom = None
        if closes[index - 1] > opens[index - 1] and closes[index - 2] < opens[index - 2] and closes[index - 3] < opens[index - 3]:
            if (closes[index - 1] - opens[index - 1]) > (highs[index - 1] - lows[index - 1]) * 0.7:
                bullish = True
                bullish_top = highs[index - 1]
                bullish_bottom = lows[index - 1]
        if closes[index - 1] < opens[index - 1] and closes[index - 2] > opens[index - 2] and closes[index - 3] > opens[index - 3]:
            if (opens[index - 1] - closes[index - 1]) > (highs[index - 1] - lows[index - 1]) * 0.7:
                bearish = True
                bearish_top = highs[index - 1]
                bearish_bottom = lows[index - 1]
        return bullish, bullish_top, bullish_bottom, bearish, bearish_top, bearish_bottom

    def _detect_swings(
        self,
        index: int,
        highs: Sequence[float],
        lows: Sequence[float],
    ) -> Tuple[bool, Optional[float], bool, Optional[float]]:
        if index < 2:
            return (False, None, False, None)
        swing_high = highs[index - 1] > highs[index - 2] and highs[index - 1] > highs[index]
        swing_low = lows[index - 1] < lows[index - 2] and lows[index - 1] < lows[index]
        return (
            swing_high,
            highs[index - 1] if swing_high else None,
            swing_low,
            lows[index - 1] if swing_low else None,
        )

    def _detect_market_structure(
        self,
        index: int,
        closes: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        lookback: int,
        min_move: float,
    ) -> Tuple[bool, bool, bool, bool, bool, bool, bool, Optional[float], Optional[float]]:
        if index < lookback + 1:
            return (False, False, True, False, False, False, False, None, None)
        prev_index = index - 1
        recent_high_prev = highest(highs, lookback, prev_index)
        recent_low_prev = lowest(lows, lookback, prev_index)

        bullish_bos = closes[index] > recent_high_prev and (closes[index] - recent_high_prev) / max(recent_high_prev, 1e-12) > min_move
        bearish_bos = closes[index] < recent_low_prev and (recent_low_prev - closes[index]) / max(recent_low_prev, 1e-12) > min_move

        prev_trend_up = index >= 10 and closes[index - 1] > closes[index - 5] and closes[index - 5] > closes[index - 10]
        prev_trend_down = index >= 10 and closes[index - 1] < closes[index - 5] and closes[index - 5] < closes[index - 10]

        bullish_choch = prev_trend_down and closes[index] > recent_high_prev
        bearish_choch = prev_trend_up and closes[index] < recent_low_prev

        struct_bull = bullish_bos or bullish_choch
        struct_bear = bearish_bos or bearish_choch
        struct_neutral = not struct_bull and not struct_bear

        new_bos_high = recent_high_prev if bullish_bos else None
        new_bos_low = recent_low_prev if bearish_bos else None

        return (
            struct_bull,
            struct_bear,
            struct_neutral,
            bullish_bos,
            bearish_bos,
            bullish_choch,
            bearish_choch,
            new_bos_high,
            new_bos_low,
        )

    def _volume_confirmed(self, index: int, volumes: Sequence[float], volume_sma: Sequence[float]) -> bool:
        if not self.settings.use_volume_filter:
            return True
        if index >= len(volume_sma):
            return False
        avg = volume_sma[index]
        if math.isnan(avg) or avg == 0:
            return False
        return volumes[index] > avg * self.settings.volume_multiplier

    def _multi_timeframe_context(self, timestamp: int) -> Tuple[Dict[str, bool], Dict[str, float]]:
        direction_flags: Dict[str, bool] = {}
        strengths: Dict[str, float] = {}
        for name, series in self.multi_timeframe_series.items():
            candle = series.candle_at(timestamp)
            direction_flags[name] = candle.close > candle.open
        for name, series in self.multi_timeframe_strength.items():
            if name.endswith("_trend"):
                continue
            strengths[name] = series.value_at(timestamp)
        return direction_flags, strengths

    def _calculate_confluence_score(self, ctx: ConfluenceContext) -> float:
        score = 0.0
        max_score = 0.0
        structure_score = 0.0
        if ctx.is_bullish and ctx.struct_bull:
            structure_score = 10
        elif not ctx.is_bullish and ctx.struct_bear:
            structure_score = 10
        elif ctx.struct_neutral:
            structure_score = 5
        score += structure_score * self.settings.weight_structure / 10
        max_score += self.settings.weight_structure

        volume_score = 10 if ctx.volume_confirmed else 0
        score += volume_score * self.settings.weight_volume / 10
        max_score += self.settings.weight_volume

        tf_alignment_score = 0.0
        for _, is_bull in ctx.multi_tf_bools.items():
            tf_alignment_score += 2 if is_bull == ctx.is_bullish else 0
        score += tf_alignment_score * self.settings.weight_timeframe / 10
        max_score += self.settings.weight_timeframe

        if self.settings.enable_pattern_recognition and ctx.pattern_prediction is not None:
            if ctx.pattern_prediction > self.settings.min_pattern_accuracy:
                pattern_score = 10
            else:
                pattern_score = ctx.pattern_prediction / 10
            score += pattern_score * self.settings.weight_pattern / 10
            max_score += self.settings.weight_pattern

        if self.settings.enable_sentiment and ctx.market_sentiment is not None:
            if ctx.is_bullish:
                sentiment_score = 10 if ctx.market_sentiment > 70 else 5 if ctx.market_sentiment > 50 else 0
            else:
                sentiment_score = 10 if ctx.market_sentiment < 30 else 5 if ctx.market_sentiment < 50 else 0
            score += sentiment_score * self.settings.weight_sentiment / 10
            max_score += self.settings.weight_sentiment

        if max_score == 0:
            return 0.0
        return _clamp((score / max_score) * 10, 0.0, 10.0)

    def _structure_filter(self, is_bullish: bool, struct_bull: bool, struct_bear: bool, struct_neutral: bool) -> bool:
        if not self.settings.use_structure_filter:
            return True
        if is_bullish:
            return struct_bull or struct_neutral
        return struct_bear or struct_neutral

    def _price_in_range(self, price: float, top: Optional[float], bottom: Optional[float]) -> bool:
        if top is None or bottom is None:
            return False
        upper = max(top, bottom)
        lower = min(top, bottom)
        return lower <= price <= upper

    def _update_active_zones(self, zones: List[Zone], high: float, low: float) -> None:
        to_remove: List[int] = []
        for idx, zone in enumerate(zones):
            if not zone.breaker:
                if zone.zone_type == "BullFVG" and low < zone.bottom:
                    zone.breaker = True
                elif zone.zone_type == "BearFVG" and high > zone.top:
                    zone.breaker = True
                elif zone.zone_type == "BullOB" and low < zone.bottom:
                    zone.breaker = True
                elif zone.zone_type == "BearOB" and high > zone.top:
                    zone.breaker = True
            else:
                if zone.zone_type == "BullFVG" and high > zone.top:
                    to_remove.append(idx)
                elif zone.zone_type == "BearFVG" and low < zone.bottom:
                    to_remove.append(idx)
                elif zone.zone_type == "BullOB" and high > zone.top:
                    to_remove.append(idx)
                elif zone.zone_type == "BearOB" and low < zone.bottom:
                    to_remove.append(idx)
        for idx in reversed(to_remove):
            zones.pop(idx)

    def _update_trades(
        self,
        pnl: PnLStats,
        bullish_sync: bool,
        bearish_sync: bool,
        confluence_score: Optional[float],
        close_price: float,
        index: int,
        choch_bull: bool,
        choch_bear: bool,
    ) -> None:
        if pnl.active_dir == 0:
            if bullish_sync:
                pnl.active_dir = 1
                pnl.active_entry = close_price
                pnl.active_entry_bar = index
            elif bearish_sync:
                pnl.active_dir = -1
                pnl.active_entry = close_price
                pnl.active_entry_bar = index
            return

        if pnl.active_dir == 1 and choch_bear:
            trade_pnl = (close_price / pnl.active_entry - 1.0) * 100
            pnl.cum_pnl_pct += trade_pnl
            pnl.equity_peak_pct = max(pnl.equity_peak_pct, pnl.cum_pnl_pct)
            pnl.max_drawdown_pct = max(pnl.max_drawdown_pct, pnl.equity_peak_pct - pnl.cum_pnl_pct)
            pnl.trades_closed += 1
            pnl.last_trade_pnl_pct = trade_pnl
            pnl.active_dir = 0
            pnl.active_entry = None
            pnl.active_entry_bar = None
        elif pnl.active_dir == -1 and choch_bull:
            trade_pnl = (pnl.active_entry / close_price - 1.0) * 100
            pnl.cum_pnl_pct += trade_pnl
            pnl.equity_peak_pct = max(pnl.equity_peak_pct, pnl.cum_pnl_pct)
            pnl.max_drawdown_pct = max(pnl.max_drawdown_pct, pnl.equity_peak_pct - pnl.cum_pnl_pct)
            pnl.trades_closed += 1
            pnl.last_trade_pnl_pct = trade_pnl
            pnl.active_dir = 0
            pnl.active_entry = None
            pnl.active_entry_bar = None

    def _evaluate_signal_success(
        self,
        index: int,
        closes: Sequence[float],
        success: SuccessStats,
        bull_queue: List[int],
        bear_queue: List[int],
    ) -> None:
        while bull_queue:
            signal_index = bull_queue[0]
            if index - signal_index >= self.settings.success_lookahead_bars:
                entry_price = closes[signal_index]
                if closes[index] > entry_price * (1 + self.settings.success_threshold_pct / 100):
                    success.successful_bull_signals += 1
                bull_queue.pop(0)
            else:
                break
        while bear_queue:
            signal_index = bear_queue[0]
            if index - signal_index >= self.settings.success_lookahead_bars:
                entry_price = closes[signal_index]
                if closes[index] < entry_price * (1 - self.settings.success_threshold_pct / 100):
                    success.successful_bear_signals += 1
                bear_queue.pop(0)
            else:
                break


def summary_to_payload(
    summary: SimulationSummary,
    symbol: str,
    timeframe: str,
    period: int,
    token: str,
) -> Dict[str, object]:
    latest_snapshot = summary.snapshots[-1]
    generated_at = datetime.now(tz=timezone.utc).isoformat()

    signals_payload = [
        {
            "bar_index": s.bar_index,
            "timestamp": s.timestamp,
            "time_iso": datetime.fromtimestamp(s.timestamp / 1000, tz=timezone.utc).isoformat(),
            "type": s.signal_type,
            "price": s.price,
            "strength": s.strength,
        }
        for s in summary.signals
    ]

    active_zones_payload = [
        {
            "type": zone.zone_type,
            "top": zone.top,
            "bottom": zone.bottom,
            "breaker": zone.breaker,
            "created_bar_index": zone.created_index,
        }
        for zone in summary.active_fvg_zones + summary.active_ob_zones
    ]

    success_payload = {
        "total_bull_signals": summary.success.total_bull_signals,
        "successful_bull_signals": summary.success.successful_bull_signals,
        "total_bear_signals": summary.success.total_bear_signals,
        "successful_bear_signals": summary.success.successful_bear_signals,
        "bull_win_rate": summary.success.bull_win_rate,
        "bear_win_rate": summary.success.bear_win_rate,
        "overall_win_rate": summary.success.overall_win_rate,
    }

    pnl_payload = {
        "trades_closed": summary.pnl.trades_closed,
        "cum_pnl_pct": summary.pnl.cum_pnl_pct,
        "equity_peak_pct": summary.pnl.equity_peak_pct,
        "max_drawdown_pct": summary.pnl.max_drawdown_pct,
        "active_direction": summary.pnl.active_dir,
        "active_entry": summary.pnl.active_entry,
        "last_trade_pnl_pct": summary.pnl.last_trade_pnl_pct,
    }

    multi_symbol_payload = None
    if summary.multi_symbol:
        multi_symbol_payload = {
            "signals": summary.multi_symbol.signals,
            "trend_strength": summary.multi_symbol.trend_strength,
        }

    definitions = {
        "trend_strength": "Composite measure (0-100) combining directional movement and momentum.",
        "pattern_score": "Rule-based pattern recognition score inspired by the TradingView indicator.",
        "sentiment": "Momentum and breadth based market sentiment estimate (0-100).",
        "structure_state": "Overall market structure bias derived from BOS/CHOCH analysis.",
        "volume_confirmed": "Whether current volume exceeds average volume by the configured multiplier.",
        "confluence_score": "Weighted confluence score (0-10) aggregating structure, volume, timeframe alignment, pattern and sentiment.",
        "signal": "Latest signal classification derived from FVG and Order Block overlap logic.",
        "success_rates": "Historical win rates for bullish/bearish signals based on look-ahead evaluation.",
        "pnl_stats": "Cumulative PnL stats assuming CHOCH-based exits.",
    }

    payload: Dict[str, object] = {
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "timeframe_minutes": timeframe_to_minutes(timeframe),
            "period": period,
            "token": token,
            "generated_at": generated_at,
        },
        "latest": {
            "timestamp": latest_snapshot.timestamp,
            "time_iso": datetime.fromtimestamp(latest_snapshot.timestamp / 1000, tz=timezone.utc).isoformat(),
            "close": latest_snapshot.close,
            "open": latest_snapshot.open,
            "high": latest_snapshot.high,
            "low": latest_snapshot.low,
            "volume": latest_snapshot.volume,
            "trend_strength": latest_snapshot.trend_strength,
            "pattern_score": latest_snapshot.pattern_score,
            "market_sentiment": latest_snapshot.sentiment,
            "structure_state": latest_snapshot.structure_state,
            "structure_event": latest_snapshot.structure_event,
            "volume_confirmed": latest_snapshot.volume_confirmed,
            "volume_ratio": latest_snapshot.volume_ratio,
            "confluence_score": latest_snapshot.confluence_score,
            "signal": latest_snapshot.signal,
        },
        "multi_timeframe": {
            "trend_strength": summary.multi_timeframe_trend,
            "direction": summary.multi_timeframe_direction,
        },
        "zones": active_zones_payload,
        "signals": signals_payload,
        "success_rates": success_payload,
        "pnl_stats": pnl_payload,
        "last_structure_levels": summary.last_structure_levels,
        "multi_symbol": multi_symbol_payload,
        "definitions": definitions,
    }

    return payload

