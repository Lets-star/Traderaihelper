from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


def _validate_length(length: int) -> None:
    if length <= 0:
        raise ValueError("length must be positive")


def sma(values: Sequence[float], length: int) -> List[float]:
    _validate_length(length)
    result: List[float] = []
    window_sum = 0.0
    for i, value in enumerate(values):
        window_sum += value
        if i >= length:
            window_sum -= values[i - length]
        if i + 1 < length:
            result.append(float("nan"))
        else:
            result.append(window_sum / length)
    return result


def ema(values: Sequence[float], length: int) -> List[float]:
    _validate_length(length)
    result: List[float] = []
    alpha = 2.0 / (length + 1.0)
    ema_prev = None
    for value in values:
        if ema_prev is None:
            ema_prev = value
        else:
            ema_prev = alpha * value + (1 - alpha) * ema_prev
        result.append(ema_prev)
    return result


def rma(values: Sequence[float], length: int) -> List[float]:
    _validate_length(length)
    alpha = 1.0 / length
    result: List[float] = []
    prev = None
    for value in values:
        if prev is None:
            prev = value
        else:
            prev = alpha * value + (1 - alpha) * prev
        result.append(prev)
    return result


def mom(values: Sequence[float], length: int) -> List[float]:
    _validate_length(length)
    result: List[float] = []
    for i, value in enumerate(values):
        if i < length:
            result.append(float("nan"))
        else:
            result.append(value - values[i - length])
    return result


def rsi(values: Sequence[float], length: int) -> List[float]:
    _validate_length(length)
    gains: List[float] = [0.0]
    losses: List[float] = [0.0]

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = rma(gains, length)
    avg_loss = rma(losses, length)

    rsi_values: List[float] = []
    for gain, loss in zip(avg_gain, avg_loss):
        if loss == 0:
            rsi_values.append(100.0)
            continue
        rs = gain / loss
        rsi_values.append(100 - (100 / (1 + rs)))
    return rsi_values


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], length: int) -> List[float]:
    _validate_length(length)
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low and close must be the same length")

    true_ranges: List[float] = []
    for i in range(len(high)):
        if i == 0:
            true_ranges.append(high[i] - low[i])
        else:
            high_low = high[i] - low[i]
            high_close = abs(high[i] - close[i - 1])
            low_close = abs(low[i] - close[i - 1])
            true_ranges.append(max(high_low, high_close, low_close))
    return rma(true_ranges, length)


def highest(values: Sequence[float], length: int, index: int) -> float:
    _validate_length(length)
    if index < 0:
        raise ValueError("index must be non-negative")
    start = max(0, index - length + 1)
    end = index + 1
    return max(values[start:end])


def lowest(values: Sequence[float], length: int, index: int) -> float:
    _validate_length(length)
    if index < 0:
        raise ValueError("index must be non-negative")
    start = max(0, index - length + 1)
    end = index + 1
    return min(values[start:end])


def macd(
    values: Sequence[float],
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
) -> tuple[List[float], List[float], List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns:
        tuple of (macd_line, signal_line, histogram)
    """
    _validate_length(fast_length)
    _validate_length(slow_length)
    _validate_length(signal_length)
    
    fast_ema = ema(values, fast_length)
    slow_ema = ema(values, slow_length)
    
    macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
    signal_line = ema(macd_line, signal_length)
    histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
    
    return macd_line, signal_line, histogram


def bollinger_bands(
    values: Sequence[float],
    length: int = 20,
    mult: float = 2.0,
) -> tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        tuple of (upper_band, middle_band, lower_band)
    """
    _validate_length(length)
    
    middle_band = sma(values, length)
    
    # Calculate standard deviation
    std_dev: List[float] = []
    for i in range(len(values)):
        if i + 1 < length:
            std_dev.append(float("nan"))
        else:
            window_start = i - length + 1
            window = values[window_start : i + 1]
            mean = middle_band[i]
            variance = sum((x - mean) ** 2 for x in window) / length
            std_dev.append(variance ** 0.5)
    
    upper_band = [middle + mult * std for middle, std in zip(middle_band, std_dev)]
    lower_band = [middle - mult * std for middle, std in zip(middle_band, std_dev)]
    
    return upper_band, middle_band, lower_band


def vwap(prices: Sequence[float], volumes: Sequence[float]) -> List[float]:
    """
    Calculate Volume Weighted Average Price (VWAP).
    
    Returns cumulative VWAP for each period.
    """
    if len(prices) != len(volumes):
        raise ValueError("prices and volumes must be the same length")
    
    result: List[float] = []
    cumulative_pv = 0.0
    cumulative_v = 0.0
    
    for price, volume in zip(prices, volumes):
        cumulative_pv += price * volume
        cumulative_v += volume
        if cumulative_v > 0:
            result.append(cumulative_pv / cumulative_v)
        else:
            result.append(float("nan"))
    
    return result


def detect_divergence(
    price_values: Sequence[float],
    indicator_values: Sequence[float],
    lookback: int = 14
) -> List[str]:
    """
    Detect divergences between price and a momentum indicator.
    
    The algorithm uses rolling lookback windows to compare recent swing highs/lows
    in price against corresponding highs/lows in the indicator series. It flags
    both regular and hidden divergences.
    """
    _validate_length(lookback)
    if len(price_values) != len(indicator_values):
        raise ValueError("price_values and indicator_values must be the same length")

    result: List[str] = []
    extended = lookback * 2

    for idx in range(len(price_values)):
        if idx + 1 < extended:
            result.append("none")
            continue

        window_start = idx + 1 - extended
        mid_point = idx + 1 - lookback
        prev_prices = price_values[window_start:mid_point]
        curr_prices = price_values[mid_point:idx + 1]
        prev_indicator = indicator_values[window_start:mid_point]
        curr_indicator = indicator_values[mid_point:idx + 1]

        if not prev_prices or not curr_prices:
            result.append("none")
            continue

        prev_high = max(prev_prices)
        prev_low = min(prev_prices)
        curr_high = max(curr_prices)
        curr_low = min(curr_prices)

        prev_ind_high = max(prev_indicator)
        prev_ind_low = min(prev_indicator)
        curr_ind_high = max(curr_indicator)
        curr_ind_low = min(curr_indicator)

        divergence = "none"

        if curr_low < prev_low and curr_ind_low >= prev_ind_low:
            divergence = "bullish_divergence"
        elif curr_high > prev_high and curr_ind_high <= prev_ind_high:
            divergence = "bearish_divergence"
        elif curr_low > prev_low and curr_ind_low < prev_ind_low:
            divergence = "hidden_bullish"
        elif curr_high < prev_high and curr_ind_high > prev_ind_high:
            divergence = "hidden_bearish"

        result.append(divergence)

    return result


@dataclass(frozen=True)
class Candle:
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float

