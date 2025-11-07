"""Technical analysis module using MACD, RSI, ATR, Bollinger Bands, and divergence detection."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence

from indicator_collector import math_utils


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a value between lower and upper bounds."""
    return max(lower, min(upper, value))


def _normalize_to_01(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Normalize a value to 0-1 range."""
    if max_val <= min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return _clamp(normalized, 0.0, 1.0)


def analyze_macd(
    candles: List[Dict[str, float]],
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
        macd_line, signal_line, histogram = math_utils.macd(closes)
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
    histogram_momentum = current_histogram - prev_histogram if len(histogram) > 1 else 0.0
    
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
    threshold_overbought: float = 70.0,
    threshold_oversold: float = 30.0,
) -> Dict[str, object]:
    """
    Analyze RSI indicator from candle data.
    
    Returns score based on RSI level, with extremes indicating potential reversals.
    """
    if not candles or len(candles) < 16:
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
        rsi_values = math_utils.rsi(closes, length=14)
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
) -> Dict[str, object]:
    """
    Analyze ATR (Average True Range) to assess volatility context.
    
    Returns volatility level and channel boundaries.
    """
    if not candles or len(candles) < 16:
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
            },
        }
    
    try:
        atr_values = math_utils.atr(highs, lows, closes, length=14)
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
            },
        }
    
    current_atr = atr_values[-1]
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
            },
        }
    
    # Calculate ATR percentage relative to price
    atr_percent = (current_atr / current_close * 100) if current_close > 0 else 0.0
    
    # Get historical ATR for context
    historical_atr = [v for v in atr_values if isinstance(v, (int, float)) and v == v]
    avg_atr = statistics.fmean(historical_atr) if len(historical_atr) >= 5 else current_atr
    atr_ma = statistics.fmean(historical_atr[-20:]) if len(historical_atr) >= 20 else avg_atr
    
    # Determine volatility state
    atr_volatility = "neutral"
    atr_score = 0.5
    confidence = 50.0
    rationale_parts = []
    
    atr_ratio = current_atr / atr_ma if atr_ma > 0 else 1.0
    
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
    upper_channel = round(current_close + current_atr, 4)
    lower_channel = round(current_close - current_atr, 4)
    channel_width = round(current_atr * 2, 4)
    
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
        },
    }



def analyze_bollinger_bands(
    candles: List[Dict[str, float]],
) -> Dict[str, object]:
    """
    Analyze Bollinger Bands for squeeze/breakout and mean reversion signals.
    
    Returns score based on bands position and price proximity.
    """
    if not candles or len(candles) < 21:
        return {
            "bollinger_score": 0.5,
            "bollinger_state": "neutral",
            "confidence": 0.0,
            "rationale": "Insufficient data for Bollinger analysis",
            "price_position": 0.5,
            "band_squeeze": 0.0,
            "band_width_percent": 0.0,
        }
    
    closes = [float(c.get("close", 0)) for c in candles]
    
    try:
        upper_band, middle_band, lower_band = math_utils.bollinger_bands(closes, length=20, mult=2.0)
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
    current_close = closes[-1]
    
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
        price_position = _clamp(price_position, 0.0, 1.0)
    else:
        price_position = 0.5
    
    # Determine state
    bollinger_state = "neutral"
    bollinger_score = 0.5
    confidence = 50.0
    rationale_parts = []
    
    # Squeeze detection
    squeeze_ratio = band_width / avg_width if avg_width > 0 else 1.0
    band_squeeze = _clamp(1.0 - squeeze_ratio, 0.0, 1.0)
    
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
    candles: List[Dict[str, float]],
) -> Dict[str, object]:
    """
    Comprehensive technical analysis combining MACD, RSI, ATR, Bollinger Bands, and divergences.
    
    Returns normalized technical factor score (0-1) with detailed breakdown.
    """
    if not candles or len(candles) < 30:
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
                "analysis_timestamp": None,
            },
        }
    
    # Analyze each component
    macd_analysis = analyze_macd(candles)
    rsi_analysis = analyze_rsi(candles)
    atr_analysis = analyze_atr(candles)
    bollinger_analysis = analyze_bollinger_bands(candles)
    divergence_analysis = detect_divergences(candles)
    
    # Extract normalized scores
    macd_score = float(macd_analysis.get("macd_score", 0.5))
    rsi_score = float(rsi_analysis.get("rsi_score", 0.5))
    atr_score = float(atr_analysis.get("atr_score", 0.5))
    bollinger_score = float(bollinger_analysis.get("bollinger_score", 0.5))
    divergence_score = float(divergence_analysis.get("divergence_score", 0.5))
    
    # Define component weights (sum to 1.0)
    weights = {
        "macd": 0.25,
        "rsi": 0.25,
        "atr": 0.15,
        "bollinger": 0.20,
        "divergence": 0.15,
    }
    
    # Calculate weighted score
    weighted_score = (
        macd_score * weights["macd"] +
        rsi_score * weights["rsi"] +
        atr_score * weights["atr"] +
        bollinger_score * weights["bollinger"] +
        divergence_score * weights["divergence"]
    )
    
    final_score = _clamp(weighted_score, 0.0, 1.0)
    
    # Determine direction based on weighted signals
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
    
    if str(bollinger_analysis.get("bollinger_state", "")).startswith("near_upper"):
        bullish_votes += 1
    elif str(bollinger_analysis.get("bollinger_state", "")).startswith("near_lower"):
        bearish_votes += 1
    else:
        neutral_votes += 1
    
    if str(divergence_analysis.get("divergence_type", "")) in ("bullish", "hidden_bullish"):
        bullish_votes += 1
    elif str(divergence_analysis.get("divergence_type", "")) in ("bearish", "hidden_bearish"):
        bearish_votes += 1
    else:
        neutral_votes += 1
    
    if bullish_votes > bearish_votes:
        direction = "bullish"
    elif bearish_votes > bullish_votes:
        direction = "bearish"
    else:
        direction = "neutral"
    
    # Build rationale
    rationale_parts = []
    
    if macd_analysis.get("macd_direction") == "bullish":
        rationale_parts.append(f"✓ MACD bullish ({macd_analysis.get('momentum')})")
    elif macd_analysis.get("macd_direction") == "bearish":
        rationale_parts.append(f"✗ MACD bearish ({macd_analysis.get('momentum')})")
    
    if rsi_analysis.get("rsi_state"):
        rationale_parts.append(f"RSI {rsi_analysis.get('rsi_state')} ({rsi_analysis.get('rsi_value'):.1f})")
    
    if atr_analysis.get("atr_volatility") != "normal":
        rationale_parts.append(f"ATR {atr_analysis.get('atr_volatility')}")
    
    bollinger_state = str(bollinger_analysis.get("bollinger_state", "neutral"))
    if "squeeze" in bollinger_state or "upper" in bollinger_state or "lower" in bollinger_state:
        rationale_parts.append(f"Bollinger {bollinger_state}")
    
    if divergence_analysis.get("divergence_type") != "none":
        rationale_parts.append(f"Divergence: {divergence_analysis.get('divergence_type')}")
    
    final_rationale = "; ".join(rationale_parts) if rationale_parts else "Neutral technical setup"
    
    # Calculate average confidence
    avg_confidence = statistics.fmean([
        float(macd_analysis.get("confidence", 50)),
        float(rsi_analysis.get("confidence", 50)),
        float(atr_analysis.get("confidence", 50)),
        float(bollinger_analysis.get("confidence", 50)),
        float(divergence_analysis.get("confidence", 30)),
    ])
    
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
            "analysis_timestamp": None,
        },
    }

