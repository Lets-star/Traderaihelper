"""Astrology-based market cycle analysis for crypto trading."""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional


def get_moon_phase(timestamp_ms: int) -> Dict[str, object]:
    """
    Calculate moon phase and volatility indication.
    
    Full moon and new moon periods often coincide with volatility peaks.
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    
    year = dt.year
    month = dt.month
    day = dt.day
    
    if month < 3:
        year -= 1
        month += 12
    
    a = year // 100
    b = a // 4
    c = 2 - a + b
    e = int(365.25 * (year + 4716))
    f = int(30.6001 * (month + 1))
    jd = c + day + e + f - 1524.5
    
    days_since_new = jd - 2451549.5
    new_moons = days_since_new / 29.53
    
    phase = (new_moons - int(new_moons)) * 29.53
    
    if phase < 1.84566:
        phase_name = "New Moon"
        volatility_indication = "high"
        trading_bias = "growth"
    elif phase < 5.53699:
        phase_name = "Waxing Crescent"
        volatility_indication = "moderate"
        trading_bias = "accumulation"
    elif phase < 9.22831:
        phase_name = "First Quarter"
        volatility_indication = "moderate"
        trading_bias = "growth"
    elif phase < 12.91963:
        phase_name = "Waxing Gibbous"
        volatility_indication = "moderate"
        trading_bias = "growth"
    elif phase < 16.61096:
        phase_name = "Full Moon"
        volatility_indication = "high"
        trading_bias = "peak"
    elif phase < 20.30228:
        phase_name = "Waning Gibbous"
        volatility_indication = "moderate"
        trading_bias = "distribution"
    elif phase < 23.99361:
        phase_name = "Last Quarter"
        volatility_indication = "moderate"
        trading_bias = "consolidation"
    else:
        phase_name = "Waning Crescent"
        volatility_indication = "moderate"
        trading_bias = "accumulation"
    
    illumination = 0.5 * (1 - math.cos(phase * 2 * math.pi / 29.53))
    
    days_to_next_full = (14.765 - phase) % 29.53
    days_to_next_new = (29.53 - phase) % 29.53
    
    next_full_moon = dt + timedelta(days=days_to_next_full)
    next_new_moon = dt + timedelta(days=days_to_next_new)
    
    return {
        "phase_name": phase_name,
        "phase_days": round(phase, 2),
        "illumination_pct": round(illumination * 100, 2),
        "volatility_indication": volatility_indication,
        "trading_bias": trading_bias,
        "next_full_moon": next_full_moon.isoformat(),
        "next_new_moon": next_new_moon.isoformat(),
        "days_to_full_moon": round(days_to_next_full, 1),
        "days_to_new_moon": round(days_to_next_new, 1),
    }


def get_mercury_cycle(timestamp_ms: int) -> Dict[str, object]:
    """
    Calculate Mercury cycles - the planet of trade and commerce.
    
    Mercury cycles of approximately 88 days coincide with periods of
    increased trading volume and market activity.
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    
    reference_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    days_since_reference = (dt - reference_date).days
    
    mercury_orbital_period = 88
    cycle_position = (days_since_reference % mercury_orbital_period) / mercury_orbital_period
    
    if cycle_position < 0.25:
        phase_name = "Direct Motion Early"
        volume_indication = "increasing"
        trading_recommendation = "accumulation phase"
    elif cycle_position < 0.4:
        phase_name = "Direct Motion Peak"
        volume_indication = "high"
        trading_recommendation = "active trading period"
    elif cycle_position < 0.6:
        phase_name = "Retrograde Shadow"
        volume_indication = "decreasing"
        trading_recommendation = "caution advised"
    elif cycle_position < 0.75:
        phase_name = "Retrograde Motion"
        volume_indication = "low"
        trading_recommendation = "consolidation period"
    else:
        phase_name = "Post-Retrograde Recovery"
        volume_indication = "increasing"
        trading_recommendation = "opportunity emerging"
    
    days_in_cycle = days_since_reference % mercury_orbital_period
    days_to_next_peak = (mercury_orbital_period * 0.4 - days_in_cycle) % mercury_orbital_period
    
    next_peak_date = dt + timedelta(days=days_to_next_peak)
    
    return {
        "phase_name": phase_name,
        "cycle_position_pct": round(cycle_position * 100, 2),
        "days_in_cycle": days_in_cycle,
        "volume_indication": volume_indication,
        "trading_recommendation": trading_recommendation,
        "days_to_peak_activity": round(days_to_next_peak, 1),
        "next_peak_date": next_peak_date.isoformat(),
    }


def get_jupiter_cycle(timestamp_ms: int) -> Dict[str, object]:
    """
    Calculate Jupiter's 12-year cycle correlation with Bitcoin halvings.
    
    Jupiter's expansion/contraction cycle aligns with Bitcoin's 4-year
    halving cycle (3 halvings per Jupiter cycle).
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    
    bitcoin_genesis = datetime(2009, 1, 3, tzinfo=timezone.utc)
    first_halving = datetime(2012, 11, 28, tzinfo=timezone.utc)
    second_halving = datetime(2016, 7, 9, tzinfo=timezone.utc)
    third_halving = datetime(2020, 5, 11, tzinfo=timezone.utc)
    fourth_halving = datetime(2024, 4, 20, tzinfo=timezone.utc)
    fifth_halving_estimate = datetime(2028, 4, 15, tzinfo=timezone.utc)
    
    halving_dates = [
        first_halving,
        second_halving,
        third_halving,
        fourth_halving,
        fifth_halving_estimate,
    ]
    
    current_halving_epoch = None
    next_halving_date = None
    days_since_last_halving = 0
    days_to_next_halving = 0
    
    for i, halving_date in enumerate(halving_dates):
        if dt >= halving_date:
            current_halving_epoch = i + 1
            days_since_last_halving = (dt - halving_date).days
        else:
            next_halving_date = halving_date
            days_to_next_halving = (halving_date - dt).days
            break
    
    if next_halving_date is None:
        next_halving_date = fifth_halving_estimate
        days_to_next_halving = (fifth_halving_estimate - dt).days
    
    jupiter_period_years = 11.86
    jupiter_period_days = jupiter_period_years * 365.25
    
    days_since_genesis = (dt - bitcoin_genesis).days
    jupiter_cycle_position = (days_since_genesis % jupiter_period_days) / jupiter_period_days
    
    if jupiter_cycle_position < 0.25:
        jupiter_phase = "Expansion Phase 1"
        market_correlation = "bullish"
        recommendation = "accumulation - growth phase beginning"
    elif jupiter_cycle_position < 0.5:
        jupiter_phase = "Peak Expansion"
        market_correlation = "strongly bullish"
        recommendation = "active growth - halving effect strong"
    elif jupiter_cycle_position < 0.75:
        jupiter_phase = "Contraction Phase 1"
        market_correlation = "corrective"
        recommendation = "consolidation - expect pullbacks"
    else:
        jupiter_phase = "Deep Contraction"
        market_correlation = "bearish to neutral"
        recommendation = "accumulation - cycle bottoming"
    
    if current_halving_epoch:
        halving_cycle_position = (days_since_last_halving / 1460.0) if days_since_last_halving < 1460 else 1.0
        
        if halving_cycle_position < 0.2:
            halving_phase = "Post-Halving Accumulation"
        elif halving_cycle_position < 0.5:
            halving_phase = "Bull Market Phase"
        elif halving_cycle_position < 0.8:
            halving_phase = "Euphoria & Distribution"
        else:
            halving_phase = "Pre-Halving Bear Market"
    else:
        halving_cycle_position = 0
        halving_phase = "Pre-Genesis"
    
    return {
        "jupiter_phase": jupiter_phase,
        "jupiter_cycle_position_pct": round(jupiter_cycle_position * 100, 2),
        "market_correlation": market_correlation,
        "recommendation": recommendation,
        "current_halving_epoch": current_halving_epoch or 0,
        "halving_phase": halving_phase,
        "halving_cycle_position_pct": round(halving_cycle_position * 100, 2) if current_halving_epoch else 0,
        "days_since_last_halving": days_since_last_halving,
        "days_to_next_halving": days_to_next_halving,
        "next_halving_date": next_halving_date.isoformat(),
    }


def get_astrology_metrics(timestamp_ms: int) -> Dict[str, object]:
    """
    Get comprehensive astrology-based market analysis.
    
    Combines moon phases, Mercury cycles, and Jupiter cycle correlations
    to provide trading insights based on celestial patterns.
    """
    moon_data = get_moon_phase(timestamp_ms)
    mercury_data = get_mercury_cycle(timestamp_ms)
    jupiter_data = get_jupiter_cycle(timestamp_ms)
    
    confluence_score = 0.0
    factors = []
    
    if moon_data["volatility_indication"] == "high":
        confluence_score += 2.0
        factors.append(f"{moon_data['phase_name']} - expect volatility")
    
    if mercury_data["volume_indication"] == "high":
        confluence_score += 2.5
        factors.append("Mercury peak - high volume period")
    elif mercury_data["volume_indication"] == "low":
        confluence_score -= 1.0
        factors.append("Mercury retrograde - low activity")
    
    if jupiter_data["market_correlation"] == "strongly bullish":
        confluence_score += 3.0
        factors.append("Jupiter expansion peak - strong growth")
    elif jupiter_data["market_correlation"] == "bullish":
        confluence_score += 2.0
        factors.append("Jupiter expansion - growth phase")
    elif jupiter_data["market_correlation"] == "bearish to neutral":
        confluence_score -= 1.5
        factors.append("Jupiter contraction - cycle bottoming")
    
    if jupiter_data["halving_phase"] == "Bull Market Phase":
        confluence_score += 2.5
        factors.append("Post-halving bull phase")
    elif jupiter_data["halving_phase"] == "Euphoria & Distribution":
        confluence_score += 1.0
        factors.append("Late cycle - take profits")
    elif jupiter_data["halving_phase"] == "Pre-Halving Bear Market":
        confluence_score -= 0.5
        factors.append("Pre-halving accumulation opportunity")
    
    if confluence_score > 5:
        overall_signal = "strongly bullish"
        signal_color = "🟢🟢🟢"
    elif confluence_score > 2:
        overall_signal = "bullish"
        signal_color = "🟢🟢"
    elif confluence_score > 0:
        overall_signal = "mildly bullish"
        signal_color = "🟢"
    elif confluence_score > -2:
        overall_signal = "neutral"
        signal_color = "⚪"
    elif confluence_score > -4:
        overall_signal = "mildly bearish"
        signal_color = "🔴"
    else:
        overall_signal = "bearish"
        signal_color = "🔴🔴"
    
    trading_recommendation = ""
    if overall_signal == "strongly bullish":
        trading_recommendation = "Strong bullish alignment across celestial cycles. Consider accumulation and long positions with proper risk management."
    elif overall_signal == "bullish":
        trading_recommendation = "Bullish celestial alignment supports growth. Look for long opportunities on pullbacks."
    elif overall_signal == "mildly bullish":
        trading_recommendation = "Slightly bullish bias. Favor longs but remain cautious with position sizing."
    elif overall_signal == "neutral":
        trading_recommendation = "Neutral celestial influence. Trade based on technical signals only."
    elif overall_signal == "mildly bearish":
        trading_recommendation = "Slightly bearish bias. Consider reducing leverage and taking profits on longs."
    else:
        trading_recommendation = "Bearish celestial alignment. Exercise caution and consider defensive positioning."
    
    return {
        "moon": moon_data,
        "mercury": mercury_data,
        "jupiter": jupiter_data,
        "confluence": {
            "score": round(confluence_score, 2),
            "signal": overall_signal,
            "signal_color": signal_color,
            "factors": factors,
            "recommendation": trading_recommendation,
        },
    }
