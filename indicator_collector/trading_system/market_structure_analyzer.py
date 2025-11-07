"""Market structure analyzer for trading system."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..advanced_metrics import calculate_market_structure, detect_liquidity_zones
from ..math_utils import Candle
from .interfaces import AnalyzerContext, FactorScore, JsonDict


def _reconstruct_candles_from_context(context: AnalyzerContext) -> List[Candle]:
    """
    Reconstruct candle data from AnalyzerContext.
    
    This is a simplified reconstruction that creates a single candle
    from the OHLCV data in the context. For full analysis, historical
    candles should be provided in context.extras['candles'].
    """
    if 'candles' in context.extras:
        return context.extras['candles']
    
    # Create a single candle from current OHLCV
    ohlcv = context.ohlcv
    return [
        Candle(
            open_time=context.timestamp - 3600000,  # 1 hour ago
            close_time=context.timestamp,
            open=ohlcv.get('open', context.current_price),
            high=ohlcv.get('high', context.current_price),
            low=ohlcv.get('low', context.current_price),
            close=ohlcv.get('close', context.current_price),
            volume=ohlcv.get('volume', 0.0),
        )
    ]


def analyze_market_structure(context: AnalyzerContext) -> FactorScore:
    """
    Analyze market structure leveraging advanced_metrics.calculate_market_structure.
    
    Analyzes:
    - Swing points (HH/HL for bullish, LH/LL for bearish)
    - Liquidity zones
    - Support/Resistance levels
    - Sideways/ranging structure
    
    Returns:
        FactorScore with normalized score (0-1) and 30% weight
    """
    # Get market structure from context if available
    if context.market_structure and context.market_structure.get('trend'):
        market_structure = context.market_structure
    else:
        # Calculate it from candles
        candles = _reconstruct_candles_from_context(context)
        market_structure = calculate_market_structure(candles)
    
    # Get volume analysis for liquidity zones
    volume_analysis = context.volume_analysis or {}
    
    # Detect liquidity zones if we have volume data
    liquidity_zones: List[JsonDict] = []
    if volume_analysis and context.current_price:
        liquidity_zones = detect_liquidity_zones(volume_analysis, context.current_price)
    
    # Analyze trend and swing points
    trend = market_structure.get('trend', 'neutral')
    swing_points = market_structure.get('swing_points', {})
    key_levels = market_structure.get('key_levels', {})
    
    hh_points = swing_points.get('hh', [])
    hl_points = swing_points.get('hl', [])
    lh_points = swing_points.get('lh', [])
    ll_points = swing_points.get('ll', [])
    
    support_levels = key_levels.get('support', [])
    resistance_levels = key_levels.get('resistance', [])
    
    # Calculate score components
    score_components = _calculate_score_components(
        trend=trend,
        hh_count=len(hh_points),
        hl_count=len(hl_points),
        lh_count=len(lh_points),
        ll_count=len(ll_points),
        support_count=len(support_levels),
        resistance_count=len(resistance_levels),
        liquidity_zone_count=len(liquidity_zones),
    )
    
    # Calculate normalized score (0-1)
    normalized_score = score_components['normalized_score']
    
    # Generate highlights
    highlights = _generate_highlights(
        trend=trend,
        hh_points=hh_points,
        hl_points=hl_points,
        lh_points=lh_points,
        ll_points=ll_points,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        liquidity_zones=liquidity_zones,
        score_components=score_components,
    )
    
    # Determine emoji based on trend
    if trend == 'bullish':
        emoji = '🟢'
    elif trend == 'bearish':
        emoji = '🔴'
    else:
        emoji = '⚪'
    
    # Create description
    description = f"{trend.capitalize()} structure"
    if trend == 'bullish':
        description += f" with {len(hh_points)} HH and {len(hl_points)} HL"
    elif trend == 'bearish':
        description += f" with {len(lh_points)} LH and {len(ll_points)} LL"
    else:
        description += " (sideways/ranging)"
    
    return FactorScore(
        factor_name='market_structure',
        score=normalized_score,
        weight=0.3,  # 30% weight as specified
        description=description,
        emoji=emoji,
        metadata={
            'trend': trend,
            'swing_points': {
                'hh_count': len(hh_points),
                'hl_count': len(hl_points),
                'lh_count': len(lh_points),
                'll_count': len(ll_points),
            },
            'key_levels': {
                'support_count': len(support_levels),
                'resistance_count': len(resistance_levels),
            },
            'liquidity_zones_count': len(liquidity_zones),
            'highlights': highlights,
            'score_components': score_components,
            'market_structure_data': market_structure,
        },
    )


def _calculate_score_components(
    trend: str,
    hh_count: int,
    hl_count: int,
    lh_count: int,
    ll_count: int,
    support_count: int,
    resistance_count: int,
    liquidity_zone_count: int,
) -> Dict[str, float]:
    """
    Calculate score components for market structure.
    
    Returns a normalized score between 0 and 1 where:
    - 1.0 = Strong bullish structure (HH/HL pattern, strong support)
    - 0.5 = Neutral/sideways structure
    - 0.0 = Strong bearish structure (LH/LL pattern, strong resistance)
    """
    # Base score from trend
    if trend == 'bullish':
        base_score = 0.75
    elif trend == 'bearish':
        base_score = 0.25
    else:
        base_score = 0.5
    
    # Swing point strength (bullish patterns increase score, bearish decrease)
    bullish_swing_strength = (hh_count + hl_count) / 10.0  # Normalize to 0-1 range
    bearish_swing_strength = (lh_count + ll_count) / 10.0
    swing_score_adjustment = (bullish_swing_strength - bearish_swing_strength) * 0.15
    
    # Support/resistance strength
    total_levels = support_count + resistance_count
    if total_levels > 0:
        support_ratio = support_count / total_levels
        # More support levels = bullish, more resistance = bearish
        sr_adjustment = (support_ratio - 0.5) * 0.1
    else:
        sr_adjustment = 0.0
    
    # Liquidity zones (more zones = stronger structure)
    liquidity_score = min(liquidity_zone_count / 5.0, 1.0) * 0.05
    
    # Calculate final normalized score
    raw_score = base_score + swing_score_adjustment + sr_adjustment + liquidity_score
    normalized_score = max(0.0, min(1.0, raw_score))
    
    # Structure clarity (how clear is the pattern)
    if trend == 'bullish':
        clarity = min((hh_count + hl_count) / 8.0, 1.0)
    elif trend == 'bearish':
        clarity = min((lh_count + ll_count) / 8.0, 1.0)
    else:
        # Sideways - look for equal support/resistance
        clarity = 1.0 - abs(support_count - resistance_count) / max(total_levels, 1)
    
    return {
        'normalized_score': round(normalized_score, 3),
        'base_score': round(base_score, 3),
        'swing_adjustment': round(swing_score_adjustment, 3),
        'sr_adjustment': round(sr_adjustment, 3),
        'liquidity_bonus': round(liquidity_score, 3),
        'clarity': round(clarity, 3),
    }


def _generate_highlights(
    trend: str,
    hh_points: List[JsonDict],
    hl_points: List[JsonDict],
    lh_points: List[JsonDict],
    ll_points: List[JsonDict],
    support_levels: List[JsonDict],
    resistance_levels: List[JsonDict],
    liquidity_zones: List[JsonDict],
    score_components: Dict[str, float],
) -> List[str]:
    """Generate structured highlights for the market structure analysis."""
    highlights = []
    
    # Trend analysis
    if trend == 'bullish':
        highlights.append(f"📈 Bullish trend confirmed with {len(hh_points)} Higher Highs and {len(hl_points)} Higher Lows")
    elif trend == 'bearish':
        highlights.append(f"📉 Bearish trend confirmed with {len(lh_points)} Lower Highs and {len(ll_points)} Lower Lows")
    else:
        highlights.append("➡️ Sideways/ranging market structure detected")
    
    # Swing point details
    total_bullish_swings = len(hh_points) + len(hl_points)
    total_bearish_swings = len(lh_points) + len(ll_points)
    
    if total_bullish_swings > total_bearish_swings:
        highlights.append(f"✅ Bullish swing dominance: {total_bullish_swings} vs {total_bearish_swings} bearish")
    elif total_bearish_swings > total_bullish_swings:
        highlights.append(f"❌ Bearish swing dominance: {total_bearish_swings} vs {total_bullish_swings} bullish")
    else:
        highlights.append(f"⚖️ Balanced swing structure: {total_bullish_swings} bullish, {total_bearish_swings} bearish")
    
    # Support/Resistance levels
    if support_levels:
        strongest_support = support_levels[0]
        highlights.append(f"🛡️ Key support at {strongest_support['price']:.4f} (strength: {strongest_support['strength']:.2f})")
    
    if resistance_levels:
        strongest_resistance = resistance_levels[0]
        highlights.append(f"🚧 Key resistance at {strongest_resistance['price']:.4f} (strength: {strongest_resistance['strength']:.2f})")
    
    # Liquidity zones
    if liquidity_zones:
        zone_types = {}
        for zone in liquidity_zones:
            zone_type = zone.get('type', 'unknown')
            zone_types[zone_type] = zone_types.get(zone_type, 0) + 1
        
        zone_summary = ', '.join([f"{count} {ztype}" for ztype, count in zone_types.items()])
        highlights.append(f"💧 Liquidity zones detected: {zone_summary}")
    
    # Structure clarity
    clarity = score_components.get('clarity', 0.5)
    if clarity >= 0.7:
        highlights.append(f"✨ Clear structure pattern (clarity: {clarity:.2f})")
    elif clarity < 0.5:
        highlights.append(f"⚠️ Unclear structure (clarity: {clarity:.2f}) - choppy price action")
    
    # Score interpretation
    score = score_components['normalized_score']
    if score >= 0.7:
        highlights.append("🎯 Strong bullish structure - favorable for long positions")
    elif score <= 0.3:
        highlights.append("🎯 Strong bearish structure - favorable for short positions")
    else:
        highlights.append("⏸️ Neutral structure - wait for clearer setup")
    
    return highlights


def calculate_structure_score(
    candles: Sequence[Candle],
    volume_analysis: Optional[Dict] = None,
) -> FactorScore:
    """
    Standalone function to calculate market structure score from candles.
    
    This is useful for direct integration without AnalyzerContext.
    
    Args:
        candles: Sequence of Candle objects
        volume_analysis: Optional volume analysis dict
        
    Returns:
        FactorScore with market structure analysis
    """
    if not candles:
        return FactorScore(
            factor_name='market_structure',
            score=0.5,
            weight=0.3,
            description='Insufficient data for analysis',
            emoji='⚪',
            metadata={},
        )
    
    market_structure = calculate_market_structure(candles)
    
    # Detect liquidity zones if volume analysis provided
    liquidity_zones: List[JsonDict] = []
    if volume_analysis:
        last_close = candles[-1].close
        liquidity_zones = detect_liquidity_zones(volume_analysis, last_close)
    
    # Extract components
    trend = market_structure.get('trend', 'neutral')
    swing_points = market_structure.get('swing_points', {})
    key_levels = market_structure.get('key_levels', {})
    
    hh_points = swing_points.get('hh', [])
    hl_points = swing_points.get('hl', [])
    lh_points = swing_points.get('lh', [])
    ll_points = swing_points.get('ll', [])
    
    support_levels = key_levels.get('support', [])
    resistance_levels = key_levels.get('resistance', [])
    
    # Calculate score
    score_components = _calculate_score_components(
        trend=trend,
        hh_count=len(hh_points),
        hl_count=len(hl_points),
        lh_count=len(lh_points),
        ll_count=len(ll_points),
        support_count=len(support_levels),
        resistance_count=len(resistance_levels),
        liquidity_zone_count=len(liquidity_zones),
    )
    
    # Generate highlights
    highlights = _generate_highlights(
        trend=trend,
        hh_points=hh_points,
        hl_points=hl_points,
        lh_points=lh_points,
        ll_points=ll_points,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        liquidity_zones=liquidity_zones,
        score_components=score_components,
    )
    
    # Determine emoji
    if trend == 'bullish':
        emoji = '🟢'
    elif trend == 'bearish':
        emoji = '🔴'
    else:
        emoji = '⚪'
    
    # Create description
    description = f"{trend.capitalize()} structure"
    if trend == 'bullish':
        description += f" with {len(hh_points)} HH and {len(hl_points)} HL"
    elif trend == 'bearish':
        description += f" with {len(lh_points)} LH and {len(ll_points)} LL"
    else:
        description += " (sideways/ranging)"
    
    return FactorScore(
        factor_name='market_structure',
        score=score_components['normalized_score'],
        weight=0.3,
        description=description,
        emoji=emoji,
        metadata={
            'trend': trend,
            'swing_points': {
                'hh_count': len(hh_points),
                'hl_count': len(hl_points),
                'lh_count': len(lh_points),
                'll_count': len(ll_points),
            },
            'key_levels': {
                'support_count': len(support_levels),
                'resistance_count': len(resistance_levels),
            },
            'liquidity_zones_count': len(liquidity_zones),
            'highlights': highlights,
            'score_components': score_components,
            'market_structure_data': market_structure,
        },
    )
