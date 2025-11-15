"""Technical analysis module using MACD, RSI, ATR, Bollinger Bands, and divergence detection."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from indicator_collector import math_utils
from .utils import clamp


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .interfaces import AnalyzerContext


def _normalize_to_01(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Normalize a value to 0-1 range."""
    if max_val <= min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return clamp(normalized, 0.0, 1.0)


def _safe_int(value: Any, default: int) -> int:
    """Safely convert a value to int with a fallback default."""
    try:
        result = int(value)
        return result if result > 0 else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    """Safely convert a value to float with a fallback default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def analyze_macd(
    candles: List[Dict[str, float]],
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
) -> Dict[str, object]:
    """
    Analyze MACD indicator from candle data.
    
    Returns score, direction, and rationale based on MACD position and histogram.
    """
    if not candles or len(candles) < 30:
        return {
            "macd_score": 0.5,
            "macd_direction": "neutral",
            "confidence": 0.0,
            "rationale": "Insufficient data for MACD analysis",
            "macd_value": 0.0,
            "signal_value": 0.0,
            "histogram": 0.0,
            "momentum": "neutral",
        }
    
    closes = [float(c.get("close", 0)) for c in candles]
    
    try:
        macd_line, signal_line, histogram = math_utils.macd(
            closes,
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
        )
    except (ValueError, IndexError):
        return {
            "macd_score": 0.5,
            "macd_direction": "neutral",
            "confidence": 0.0,
            "rationale": "MACD calculation failed",
            "macd_value": 0.0,
            "signal_value": 0.0,
            "histogram": 0.0,
            "momentum": "neutral",
        }
    
    # Get latest values
    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    current_histogram = histogram[-1]
    
    # Check for NaN or invalid values
    if not all(isinstance(v, (int, float)) and v == v for v in [current_macd, current_signal, current_histogram]):
        return {
            "macd_score": 0.5,
            "macd_direction": "neutral",
            "confidence": 0.0,
            "rationale": "Invalid MACD values",
            "macd_value": 0.0,
            "signal_value": 0.0,
            "histogram": 0.0,
            "momentum": "neutral",
        }
    
    # Get previous histogram for momentum
    prev_histogram = histogram[-2] if len(histogram) > 1 else current_histogram
    histogram_momentum = histogram[-1] - prev_histogram if len(histogram) > 1 else 0.0
    
    # Determine direction and score
    macd_direction = "neutral"
    macd_score = 0.5
    confidence = 50.0
    momentum = "neutral"
    rationale_parts = []
    
    if current_histogram > 0:
        macd_direction = "bullish"
        if current_macd > current_signal:
            macd_score = 0.7
            confidence = 70.0
            momentum = "strengthening" if histogram_momentum > 0 else "weakening"
            rationale_parts.append("MACD above signal line (bullish)")
            if histogram_momentum > 0:
                rationale_parts.append("Histogram increasing (gaining momentum)")
                macd_score = 0.8
                confidence = 80.0
        else:
            macd_score = 0.6
            confidence = 60.0
            rationale_parts.append("MACD positive but below signal (early bullish)")
    elif current_histogram < 0:
        macd_direction = "bearish"
        if current_macd < current_signal:
            macd_score = 0.3
            confidence = 70.0
            momentum = "strengthening" if histogram_momentum < 0 else "weakening"
            rationale_parts.append("MACD below signal line (bearish)")
            if histogram_momentum < 0:
                rationale_parts.append("Histogram decreasing (gaining downside momentum)")
                macd_score = 0.2
                confidence = 80.0
        else:
            macd_score = 0.4
            confidence = 60.0
            rationale_parts.append("MACD negative but above signal (early bearish)")
    else:
        rationale_parts.append("MACD near signal line (neutral)")
        confidence = 40.0
    
    rationale = "; ".join(rationale_parts) if rationale_parts else "MACD neutral"
    
    return {
        "macd_score": round(macd_score, 3),
        "macd_direction": macd_direction,
        "confidence": round(confidence, 2),
        "rationale": rationale,
        "macd_value": round(current_macd, 4),
        "signal_value": round(current_signal, 4),
        "histogram": round(current_histogram, 4),
        "momentum": momentum,
    }




def analyze_rsi(
    candles: List[Dict[str, float]],
    period: int = 14,
    threshold_overbought: float = 70.0,
    threshold_oversold: float = 30.0,
) -> Dict[str, object]:
    """
    Analyze RSI indicator from candle data.
    
    Returns score based on RSI level, with extremes indicating potential reversals.
    """
    if not candles or len(candles) < max(16, period + 2):
        return {
            "rsi_score": 0.5,
            "rsi_direction": "neutral",
            "confidence": 0.0,
            "rationale": "Insufficient data for RSI analysis",
            "rsi_value": 50.0,
            "rsi_state": "neutral",
        }
    
    closes = [float(c.get("close", 0)) for c in candles]
    
    try:
        rsi_values = math_utils.rsi(closes, length=period)
    except (ValueError, IndexError):
        return {
            "rsi_score": 0.5,
            "rsi_direction": "neutral",
            "confidence": 0.0,
            "rationale": "RSI calculation failed",
            "rsi_value": 50.0,
            "rsi_state": "neutral",
        }
    
    current_rsi = rsi_values[-1]
    
    if not isinstance(current_rsi, (int, float)) or current_rsi != current_rsi:  # NaN check
        return {
            "rsi_score": 0.5,
            "rsi_direction": "neutral",
            "confidence": 0.0,
            "rationale": "Invalid RSI value",
            "rsi_value": 50.0,
            "rsi_state": "neutral",
        }
    
    # Determine state and score
    rsi_direction = "neutral"
    rsi_score = 0.5
    confidence = 0.0
    rsi_state = "neutral"
    rationale = ""
    
    if current_rsi >= threshold_overbought:
        rsi_direction = "bearish"
        rsi_score = _normalize_to_01(current_rsi, threshold_overbought, 100.0)
        rsi_score = 1.0 - rsi_score  # Invert: higher RSI = lower score
        confidence = min(abs(current_rsi - 50.0), 100.0) * 0.7
        rsi_state = "overbought"
        rationale = f"RSI overbought at {current_rsi:.1f} (potential reversal)"
    elif current_rsi <= threshold_oversold:
        rsi_direction = "bullish"
        rsi_score = _normalize_to_01(current_rsi, 0.0, threshold_oversold)
        confidence = min(abs(50.0 - current_rsi), 100.0) * 0.7
        rsi_state = "oversold"
        rationale = f"RSI oversold at {current_rsi:.1f} (potential reversal)"
    else:
        rsi_score = _normalize_to_01(current_rsi, 0.0, 100.0)
        confidence = 50.0
        if current_rsi > 50:
            rsi_direction = "bullish"
            rationale = f"RSI above neutral at {current_rsi:.1f}"
        else:
            rsi_direction = "bearish"
            rationale = f"RSI below neutral at {current_rsi:.1f}"
    
    return {
        "rsi_score": round(rsi_score, 3),
        "rsi_direction": rsi_direction,
        "confidence": round(confidence, 2),
        "rationale": rationale,
        "rsi_value": round(current_rsi, 2),
        "rsi_state": rsi_state,
    }



def analyze_atr(
    candles: List[Dict[str, float]],
    period: int = 14,
    multiplier: float = 1.0,
    *,
    channel_period: Optional[int] = None,
    channel_multipliers: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """
    Analyze ATR (Average True Range) to assess volatility context.
    
    Returns volatility level and channel boundaries.
    """
    min_period = max(period, channel_period or period)
    if not candles or len(candles) < max(16, min_period + 2):
        return {
            "atr_score": 0.5,
            "atr_volatility": "neutral",
            "confidence": 0.0,
            "rationale": "Insufficient data for ATR analysis",
            "atr_value": 0.0,
            "atr_channels": {
                "upper": 0.0,
                "lower": 0.0,
                "width": 0.0,
                "levels": {},
            },
        }
    
    highs = [float(c.get("high", 0)) for c in candles]
    lows = [float(c.get("low", 0)) for c in candles]
    closes = [float(c.get("close", 0)) for c in candles]
    
    if not (len(highs) == len(lows) == len(closes)):
        return {
            "atr_score": 0.5,
            "atr_volatility": "neutral",
            "confidence": 0.0,
            "rationale": "Mismatched candle data",
            "atr_value": 0.0,
            "atr_channels": {
                "upper": 0.0,
                "lower": 0.0,
                "width": 0.0,
                "levels": {},
            },
        }
    
    try:
        atr_values = math_utils.atr(highs, lows, closes, length=period)
    except (ValueError, IndexError):
        return {
            "atr_score": 0.5,
            "atr_volatility": "neutral",
            "confidence": 0.0,
            "rationale": "ATR calculation failed",
            "atr_value": 0.0,
            "atr_channels": {
                "upper": 0.0,
                "lower": 0.0,
                "width": 0.0,
                "levels": {},
            },
        }
    
    raw_atr = atr_values[-1]
    current_atr = raw_atr * multiplier
    current_close = closes[-1]
    
    if not isinstance(current_atr, (int, float)) or current_atr != current_atr:  # NaN check
        return {
            "atr_score": 0.5,
            "atr_volatility": "neutral",
            "confidence": 0.0,
            "rationale": "Invalid ATR value",
            "atr_value": 0.0,
            "atr_channels": {
                "upper": 0.0,
                "lower": 0.0,
                "width": 0.0,
                "levels": {},
            },
        }
    
    # Calculate ATR percentage relative to price
    atr_percent = (current_atr / current_close * 100) if current_close > 0 else 0.0
    
    # Get historical ATR for context
    historical_atr = [float(v) for v in atr_values if isinstance(v, (int, float)) and v == v]
    avg_atr = statistics.fmean(historical_atr) if len(historical_atr) >= 5 else raw_atr
    atr_ma = statistics.fmean(historical_atr[-20:]) if len(historical_atr) >= 20 else avg_atr
    
    # Determine volatility state
    atr_volatility = "neutral"
    atr_score = 0.5
    confidence = 50.0
    rationale_parts = []
    
    atr_ratio = raw_atr / atr_ma if atr_ma > 0 else 1.0
    
    if atr_ratio > 1.3:
        atr_volatility = "high"
        atr_score = 0.7
        confidence = 70.0
        rationale_parts.append(f"High volatility (ATR {atr_percent:.2f}% of price)")
    elif atr_ratio > 1.1:
        atr_volatility = "elevated"
        atr_score = 0.6
        confidence = 60.0
        rationale_parts.append(f"Elevated volatility (ATR {atr_percent:.2f}% of price)")
    elif atr_ratio < 0.7:
        atr_volatility = "low"
        atr_score = 0.4
        confidence = 70.0
        rationale_parts.append(f"Low volatility (ATR {atr_percent:.2f}% of price)")
    else:
        atr_volatility = "normal"
        atr_score = 0.5
        confidence = 50.0
        rationale_parts.append(f"Normal volatility (ATR {atr_percent:.2f}% of price)")
    
    # Calculate ATR channels
    level_multipliers: Dict[str, float] = {}
    if isinstance(channel_multipliers, dict):
        for key, value in channel_multipliers.items():
            try:
                level_multipliers[key] = float(value)
            except (TypeError, ValueError):
                continue
    if not level_multipliers:
        level_multipliers["mult_1x"] = float(multiplier)
    channel_len = channel_period or period
    try:
        channel_atr_values = (
            atr_values if channel_len == period else math_utils.atr(highs, lows, closes, length=channel_len)
        )
        raw_channel_atr = channel_atr_values[-1]
    except (ValueError, IndexError):
        raw_channel_atr = raw_atr
    levels: Dict[str, Dict[str, float]] = {}
    for key, mult in level_multipliers.items():
        offset = raw_channel_atr * mult
        levels[key] = {
            "upper": round(current_close + offset, 4),
            "lower": round(current_close - offset, 4),
            "width": round(offset * 2, 4),
        }
    primary_level = levels.get("mult_1x") or next(iter(levels.values()))
    upper_channel = primary_level["upper"]
    lower_channel = primary_level["lower"]
    channel_width = primary_level["width"]
    
    rationale = "; ".join(rationale_parts) if rationale_parts else "ATR neutral"
    
    return {
        "atr_score": round(atr_score, 3),
        "atr_volatility": atr_volatility,
        "confidence": round(confidence, 2),
        "rationale": rationale,
        "atr_value": round(current_atr, 4),
        "atr_percent": round(atr_percent, 2),
        "atr_channels": {
            "upper": upper_channel,
            "lower": lower_channel,
            "width": channel_width,
            "levels": levels,
        },
        "metadata": {
            "raw_atr": round(raw_atr, 4),
            "multiplier": multiplier,
            "period": period,
            "channel_period": channel_len,
            "channel_multipliers": level_multipliers,
        },
    }



def analyze_bollinger_bands(
    candles: List[Dict[str, float]],
    period: int = 20,
    mult: float = 2.0,
    source: str = "close",
) -> Dict[str, object]:
    """
    Analyze Bollinger Bands for squeeze/breakout and mean reversion signals.
    
    Returns score based on bands position and price proximity.
    """
    if not candles or len(candles) < max(21, period + 1):
        return {
            "bollinger_score": 0.5,
            "bollinger_state": "neutral",
            "confidence": 0.0,
            "rationale": "Insufficient data for Bollinger analysis",
            "price_position": 0.5,
            "band_squeeze": 0.0,
            "band_width_percent": 0.0,
        }
    
    def _extract_series() -> List[float]:
        normalized_source = source.lower()
        if normalized_source in {"close", "c"}:
            return [float(c.get("close", 0.0)) for c in candles]
        if normalized_source in {"ohlc4", "avg"}:
            return [
                float(
                    (c.get("open", 0.0) + c.get("high", 0.0) + c.get("low", 0.0) + c.get("close", 0.0))
                    / 4.0
                )
                for c in candles
            ]
        if normalized_source in {"hlc3", "typical"}:
            return [
                float((c.get("high", 0.0) + c.get("low", 0.0) + c.get("close", 0.0)) / 3.0)
                for c in candles
            ]
        if normalized_source in {"open", "o"}:
            return [float(c.get("open", 0.0)) for c in candles]
        return [float(c.get("close", 0.0)) for c in candles]
    
    series = _extract_series()
    
    try:
        upper_band, middle_band, lower_band = math_utils.bollinger_bands(series, length=period, mult=mult)
    except (ValueError, IndexError):
        return {
            "bollinger_score": 0.5,
            "bollinger_state": "neutral",
            "confidence": 0.0,
            "rationale": "Bollinger Bands calculation failed",
            "price_position": 0.5,
            "band_squeeze": 0.0,
            "band_width_percent": 0.0,
        }
    
    current_upper = upper_band[-1]
    current_middle = middle_band[-1]
    current_lower = lower_band[-1]
    current_close = series[-1]
    
    # Validate values
    if not all(isinstance(v, (int, float)) and v == v for v in [current_upper, current_middle, current_lower]):
        return {
            "bollinger_score": 0.5,
            "bollinger_state": "neutral",
            "confidence": 0.0,
            "rationale": "Invalid Bollinger Bands values",
            "price_position": 0.5,
            "band_squeeze": 0.0,
            "band_width_percent": 0.0,
        }
    
    # Calculate band width
    band_width = current_upper - current_lower
    band_width_percent = (band_width / current_middle * 100) if current_middle > 0 else 0.0
    
    # Get historical band widths for squeeze detection
    historical_widths = []
    for i in range(max(0, len(upper_band) - 20), len(upper_band)):
        if isinstance(upper_band[i], (int, float)) and isinstance(lower_band[i], (int, float)):
            if upper_band[i] == upper_band[i] and lower_band[i] == lower_band[i]:
                historical_widths.append(upper_band[i] - lower_band[i])
    
    avg_width = statistics.fmean(historical_widths) if historical_widths else band_width
    
    # Calculate price position within bands (0 = lower, 1 = upper)
    if band_width > 0:
        price_position = (current_close - current_lower) / band_width
        price_position = clamp(price_position, 0.0, 1.0)
    else:
        price_position = 0.5
    
    # Determine state
    bollinger_state = "neutral"
    bollinger_score = 0.5
    confidence = 50.0
    rationale_parts = []
    
    # Squeeze detection
    squeeze_ratio = band_width / avg_width if avg_width > 0 else 1.0
    band_squeeze = clamp(1.0 - squeeze_ratio, 0.0, 1.0)
    
    if squeeze_ratio < 0.7:
        bollinger_state = "squeeze"
        confidence = 70.0
        rationale_parts.append("Bollinger Bands squeezed (potential breakout)")
    elif squeeze_ratio > 1.3:
        confidence = 60.0
        rationale_parts.append("Bollinger Bands expanded (high volatility)")
    
    # Price position analysis
    if price_position > 0.8:
        bollinger_score = 0.7
        if bollinger_state != "squeeze":
            bollinger_state = "near_upper"
        rationale_parts.append("Price near upper band (bullish)")
    elif price_position < 0.2:
        bollinger_score = 0.3
        if bollinger_state != "squeeze":
            bollinger_state = "near_lower"
        rationale_parts.append("Price near lower band (bearish)")
    else:
        bollinger_state = "mean_reversion" if bollinger_state != "squeeze" else "squeeze"
        rationale_parts.append("Price in middle band zone")
    
    rationale = "; ".join(rationale_parts) if rationale_parts else "Bollinger Bands neutral"
    
    return {
        "bollinger_score": round(bollinger_score, 3),
        "bollinger_state": bollinger_state,
        "confidence": round(confidence, 2),
        "rationale": rationale,
        "price_position": round(price_position, 3),
        "band_squeeze": round(band_squeeze, 3),
        "band_width_percent": round(band_width_percent, 2),
        "upper_band": round(current_upper, 4),
        "middle_band": round(current_middle, 4),
        "lower_band": round(current_lower, 4),
        "metadata": {
            "period": period,
            "mult": mult,
            "source": source,
            "series_value": round(current_close, 4),
        },
    }



def detect_divergences(
    candles: List[Dict[str, float]],
) -> Dict[str, object]:
    """
    Detect bullish and bearish divergences using RSI.
    
    Compares price structure with RSI momentum for divergence signals.
    """
    if not candles or len(candles) < 30:
        return {
            "divergence_score": 0.5,
            "divergence_type": "none",
            "confidence": 0.0,
            "rationale": "Insufficient data for divergence analysis",
            "price_divergence": "none",
            "rsi_divergence": "none",
        }
    
    closes = [float(c.get("close", 0)) for c in candles]
    
    try:
        rsi_values = math_utils.rsi(closes, length=14)
        divergence_results = math_utils.detect_divergence(closes, rsi_values, lookback=14)
    except (ValueError, IndexError):
        return {
            "divergence_score": 0.5,
            "divergence_type": "none",
            "confidence": 0.0,
            "rationale": "Divergence detection failed",
            "price_divergence": "none",
            "rsi_divergence": "none",
        }
    
    current_divergence = divergence_results[-1] if divergence_results else "none"
    
    # Calculate divergence strength
    divergence_type = "none"
    divergence_score = 0.5
    confidence = 0.0
    rationale = ""
    
    if current_divergence == "bullish_divergence":
        divergence_type = "bullish"
        divergence_score = 0.75
        confidence = 75.0
        rationale = "Regular bullish divergence detected (price lower, RSI higher)"
    elif current_divergence == "bearish_divergence":
        divergence_type = "bearish"
        divergence_score = 0.25
        confidence = 75.0
        rationale = "Regular bearish divergence detected (price higher, RSI lower)"
    elif current_divergence == "hidden_bullish":
        divergence_type = "hidden_bullish"
        divergence_score = 0.65
        confidence = 60.0
        rationale = "Hidden bullish divergence detected (price higher, RSI lower)"
    elif current_divergence == "hidden_bearish":
        divergence_type = "hidden_bearish"
        divergence_score = 0.35
        confidence = 60.0
        rationale = "Hidden bearish divergence detected (price lower, RSI higher)"
    else:
        rationale = "No divergence detected"
        confidence = 30.0
    
    return {
        "divergence_score": round(divergence_score, 3),
        "divergence_type": divergence_type,
        "confidence": round(confidence, 2),
        "rationale": rationale,
        "price_divergence": "none",
        "rsi_divergence": current_divergence,
    }



def analyze_technical_factors(
    data: Union[List[Dict[str, float]], "AnalyzerContext"],
    indicator_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, object]:
    """
    Comprehensive technical analysis combining MACD, RSI, ATR, Bollinger Bands, and divergences.
    
    Supports both raw candle lists and ``AnalyzerContext`` inputs. Indicator parameters can be
    overridden via ``indicator_params`` or ``context.extras['indicator_params']``.
    """
    context: Optional["AnalyzerContext"] = None
    candles: List[Dict[str, float]] = []

    if isinstance(data, list):
        candles = [c for c in data if isinstance(c, dict)]
    else:
        try:
            from .interfaces import AnalyzerContext as _AnalyzerContext  # type: ignore
        except Exception:  # pragma: no cover - defensive
            _AnalyzerContext = None  # type: ignore
        if _AnalyzerContext is not None and isinstance(data, _AnalyzerContext):  # type: ignore[arg-type]
            context = data  # type: ignore[assignment]
            extras = context.extras if isinstance(context.extras, dict) else {}
            metadata_dict = context.metadata if isinstance(context.metadata, dict) else {}
            candle_source = extras.get("candles")
            if not isinstance(candle_source, list):
                candle_source = metadata_dict.get("candles")
            if isinstance(candle_source, list):
                candles = [c for c in candle_source if isinstance(c, dict)]
        elif isinstance(data, dict):
            candles = [data]

    # Ensure candle ordering and structure
    sanitized_candles: List[Dict[str, float]] = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        sanitized_candles.append(candle)
    candles = sanitized_candles

    if candles and all(isinstance(candle.get("ts"), (int, float)) for candle in candles):
        candles = sorted(candles, key=lambda item: item.get("ts", 0))

    params_source: Dict[str, Any]
    params_source = indicator_params if isinstance(indicator_params, dict) else {}
    if context and isinstance(context.extras, dict):
        contextual_params = context.extras.get("indicator_params")
        if isinstance(contextual_params, dict):
            merged = dict(contextual_params)
            merged.update(params_source)
            params_source = merged

    resolved_params = {
        "macd": {
            "fast": _safe_int(params_source.get("macd", {}).get("fast"), 12),
            "slow": _safe_int(params_source.get("macd", {}).get("slow"), 26),
            "signal": _safe_int(params_source.get("macd", {}).get("signal"), 9),
        },
        "rsi": {
            "period": _safe_int(params_source.get("rsi", {}).get("period"), 14),
            "overbought": _safe_float(params_source.get("rsi", {}).get("overbought"), 70.0),
            "oversold": _safe_float(params_source.get("rsi", {}).get("oversold"), 30.0),
        },
        "atr": {
            "period": _safe_int(params_source.get("atr", {}).get("period"), 14),
            "mult": _safe_float(params_source.get("atr", {}).get("mult"), 1.0),
        },
        "atr_channels": {
            "period": _safe_int(params_source.get("atr_channels", {}).get("period"), _safe_int(params_source.get("atr", {}).get("period"), 14)),
            "multipliers": {
                key: _safe_float(value, default)
                for key, value, default in [
                    ("mult_1x", params_source.get("atr_channels", {}).get("mult_1x"), 1.0),
                    ("mult_2x", params_source.get("atr_channels", {}).get("mult_2x"), 2.0),
                    ("mult_3x", params_source.get("atr_channels", {}).get("mult_3x"), 3.0),
                ]
            },
        },
        "bollinger": {
            "period": _safe_int(params_source.get("bollinger", {}).get("period"), 20),
            "mult": _safe_float(
                params_source.get("bollinger", {}).get("mult"),
                _safe_float(params_source.get("bollinger", {}).get("stddev"), 2.0),
            ),
            "stddev": _safe_float(params_source.get("bollinger", {}).get("stddev"), 2.0),
            "source": str(params_source.get("bollinger", {}).get("source", "close")),
        },
    }

    resolved_params["atr_channels"]["multipliers"] = {
        key: value
        for key, value in resolved_params["atr_channels"]["multipliers"].items()
        if isinstance(value, (int, float))
    } or {"mult_1x": resolved_params["atr"]["mult"]}
    resolved_params["atr_channels"].update(resolved_params["atr_channels"]["multipliers"])

    min_required_candles = max(
        30,
        resolved_params["rsi"]["period"] + 2,
        resolved_params["atr"]["period"] + 2,
        resolved_params["atr_channels"]["period"] + 2,
        resolved_params["bollinger"]["period"] + 1,
        resolved_params["macd"]["slow"] + resolved_params["macd"]["signal"],
    )

    analysis_timestamp: Optional[int] = None
    if candles:
        ts_value = candles[-1].get("ts")
        if isinstance(ts_value, (int, float)):
            analysis_timestamp = int(ts_value)
    elif context is not None:
        potential_ts = getattr(context, "timestamp", None)
        if isinstance(potential_ts, (int, float)):
            analysis_timestamp = int(potential_ts)

    if len(candles) < min_required_candles:
        return {
            "final_score": 0.5,
            "direction": "neutral",
            "confidence": 0.0,
            "rationale": "Insufficient candle data for technical analysis",
            "components": {},
            "factor_weights": {},
            "factor_scores": {},
            "metadata": {
                "total_candles": len(candles),
                "analysis_timestamp": analysis_timestamp,
                "indicator_params": resolved_params,
                "context_symbol": getattr(context, "symbol", None),
                "context_timeframe": getattr(context, "timeframe", None),
            },
        }

    # Analyze each component with resolved parameters
    macd_analysis = analyze_macd(
        candles,
        fast_length=resolved_params["macd"]["fast"],
        slow_length=resolved_params["macd"]["slow"],
        signal_length=resolved_params["macd"]["signal"],
    )
    rsi_analysis = analyze_rsi(
        candles,
        period=resolved_params["rsi"]["period"],
        threshold_overbought=resolved_params["rsi"]["overbought"],
        threshold_oversold=resolved_params["rsi"]["oversold"],
    )
    atr_analysis = analyze_atr(
        candles,
        period=resolved_params["atr"]["period"],
        multiplier=resolved_params["atr"]["mult"],
        channel_period=resolved_params["atr_channels"]["period"],
        channel_multipliers=resolved_params["atr_channels"]["multipliers"],
    )
    bollinger_analysis = analyze_bollinger_bands(
        candles,
        period=resolved_params["bollinger"]["period"],
        mult=resolved_params["bollinger"]["mult"],
        source=resolved_params["bollinger"]["source"],
    )
    divergence_analysis = detect_divergences(candles)

    # Extract normalized scores
    macd_score = float(macd_analysis.get("macd_score", 0.5))
    rsi_score = float(rsi_analysis.get("rsi_score", 0.5))
    atr_score = float(atr_analysis.get("atr_score", 0.5))
    bollinger_score = float(bollinger_analysis.get("bollinger_score", 0.5))
    divergence_score = float(divergence_analysis.get("divergence_score", 0.5))

    weights = {
        "macd": 0.25,
        "rsi": 0.25,
        "atr": 0.15,
        "bollinger": 0.20,
        "divergence": 0.15,
    }

    weighted_score = (
        macd_score * weights["macd"]
        + rsi_score * weights["rsi"]
        + atr_score * weights["atr"]
        + bollinger_score * weights["bollinger"]
        + divergence_score * weights["divergence"]
    )

    final_score = clamp(weighted_score, 0.0, 1.0)

    bullish_votes = 0
    bearish_votes = 0
    neutral_votes = 0

    if str(macd_analysis.get("macd_direction", "")) == "bullish":
        bullish_votes += 1
    elif str(macd_analysis.get("macd_direction", "")) == "bearish":
        bearish_votes += 1
    else:
        neutral_votes += 1

    if str(rsi_analysis.get("rsi_direction", "")) == "bullish":
        bullish_votes += 1
    elif str(rsi_analysis.get("rsi_direction", "")) == "bearish":
        bearish_votes += 1
    else:
        neutral_votes += 1

    bollinger_state = str(bollinger_analysis.get("bollinger_state", ""))
    if bollinger_state.startswith("near_upper"):
        bullish_votes += 1
    elif bollinger_state.startswith("near_lower"):
        bearish_votes += 1
    else:
        neutral_votes += 1

    divergence_type = str(divergence_analysis.get("divergence_type", ""))
    if divergence_type in ("bullish", "hidden_bullish"):
        bullish_votes += 1
    elif divergence_type in ("bearish", "hidden_bearish"):
        bearish_votes += 1
    else:
        neutral_votes += 1

    if bullish_votes > bearish_votes:
        direction = "bullish"
    elif bearish_votes > bullish_votes:
        direction = "bearish"
    else:
        direction = "neutral"

    rationale_parts: List[str] = []
    if macd_analysis.get("macd_direction") == "bullish":
        rationale_parts.append(f"✓ MACD bullish ({macd_analysis.get('momentum')})")
    elif macd_analysis.get("macd_direction") == "bearish":
        rationale_parts.append(f"✗ MACD bearish ({macd_analysis.get('momentum')})")

    if rsi_analysis.get("rsi_state"):
        rationale_parts.append(
            f"RSI {rsi_analysis.get('rsi_state')} ({rsi_analysis.get('rsi_value'):.1f})"
        )

    if atr_analysis.get("atr_volatility") != "normal":
        rationale_parts.append(f"ATR {atr_analysis.get('atr_volatility')}")

    if bollinger_state and any(token in bollinger_state for token in ("squeeze", "upper", "lower")):
        rationale_parts.append(f"Bollinger {bollinger_state}")

    if divergence_type != "none":
        rationale_parts.append(f"Divergence: {divergence_type}")

    final_rationale = "; ".join(rationale_parts) if rationale_parts else "Neutral technical setup"

    avg_confidence = statistics.fmean(
        [
            float(macd_analysis.get("confidence", 50)),
            float(rsi_analysis.get("confidence", 50)),
            float(atr_analysis.get("confidence", 50)),
            float(bollinger_analysis.get("confidence", 50)),
            float(divergence_analysis.get("confidence", 30)),
        ]
    )

    return {
        "final_score": round(final_score, 3),
        "direction": direction,
        "confidence": round(avg_confidence, 2),
        "rationale": final_rationale,
        "components": {
            "macd": macd_analysis,
            "rsi": rsi_analysis,
            "atr": atr_analysis,
            "bollinger": bollinger_analysis,
            "divergence": divergence_analysis,
        },
        "factor_weights": weights,
        "factor_scores": {
            "macd": round(macd_score, 3),
            "rsi": round(rsi_score, 3),
            "atr": round(atr_score, 3),
            "bollinger": round(bollinger_score, 3),
            "divergence": round(divergence_score, 3),
        },
        "metadata": {
            "total_candles": len(candles),
            "analysis_timestamp": analysis_timestamp,
            "indicator_params": resolved_params,
            "context_symbol": getattr(context, "symbol", None),
            "context_timeframe": getattr(context, "timeframe", None),
        },
    }

