"""Chart visualization functions for the web UI package.

This module contains all chart creation and visualization logic.

Following SRP (Single Responsibility Principle), this module is responsible for:
- Chart indicator calculations (RSI, MACD, Bollinger Bands)
- Volume indicator calculations (BVI)
- Chart creation functions (candlestick, realtime, multi-timeframe)
- Technical analysis visualization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass

from indicator_collector.indicator_metrics import SimulationSummary
from indicator_collector.time_series import TimeframeSeries


# =============================================================================
# Technical Indicators Calculation Functions
# =============================================================================

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Compute MACD (Moving Average Convergence Divergence) indicator."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def _compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """Compute Bollinger Bands indicator."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return {
        "middle": sma,
        "upper": upper,
        "lower": lower,
    }


def _rolling_equals(series: pd.Series, window: int, *, method: str) -> pd.Series:
    """Rolling comparison function for series analysis."""
    if method == "all":
        return series.rolling(window=window).apply(lambda x: x.nunique() == 1, raw=True)
    elif method == "any":
        return series.rolling(window=window).apply(lambda x: x.nunique() > 1, raw=True)
    else:
        return series.rolling(window=window).apply(lambda x: x.iloc[-1] in x.iloc[:-1].values if len(x) > 1 else False, raw=True)


def calculate_better_volume_indicator(
    df: pd.DataFrame,
    period: int = 20,
    bvi_period: int = 14,
    use_bvi: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Better Volume Indicator (BVI) with enhanced volume analysis.
    
    The Better Volume Indicator uses price and volume relationship to identify:
    - Climax Chaks (high volume at trend ends)
    - High Volume Breakouts
    - Volume Divergences
    - Low Volume Consolidations
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for volume calculations
        bvi_period: Period for BVI specific calculations
        use_bvi: Whether to use BVI or standard volume analysis
        
    Returns:
        Tuple of (bvi_series, volume_profile_series)
    """
    if df.empty or len(df) < period + bvi_period:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Calculate price changes and ranges
    price_change = df['close'].diff()
    price_range = df['high'] - df['low']
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Calculate volume components
    volume_ma = df['volume'].rolling(window=period).mean()
    volume_std = df['volume'].rolling(window=period).std()
    volume_zscore = (df['volume'] - volume_ma) / volume_std
    
    # Price-Volume relationship indicators
    pv_correlation = df['close'].rolling(window=period).corr(df['volume'])
    price_volatility = price_range / df['close']
    volume_intensity = df['volume'] / (price_volatility + 1e-8)
    
    # Better Volume Indicator components
    volume_surge = df['volume'] > (volume_ma + volume_std * 1.5)
    volume_dry = df['volume'] < (volume_ma - volume_std * 0.5)
    
    # Climax volume detection
    climax_up = (price_change > 0) & volume_surge
    climax_down = (price_change < 0) & volume_surge
    
    # Breakout detection
    price_breakout = price_range > price_range.rolling(window=period).quantile(0.8)
    volume_breakout = volume_surge & price_breakout
    
    # Volume divergence
    price_up = df['close'] > df['close'].rolling(window=period).mean()
    volume_down = df['volume'] < volume_ma
    divergence_bull = price_up & volume_down
    
    price_down = df['close'] < df['close'].rolling(window=period).mean()
    volume_up = df['volume'] > volume_ma
    divergence_bear = price_down & volume_up
    
    # Combine BVI components
    bvi_components = {
        'climax_up': climax_up.astype(int) * 1.0,
        'climax_down': climax_down.astype(int) * -1.0,
        'breakout': volume_breakout.astype(int) * 0.5,
        'divergence_bull': divergence_bull.astype(int) * 0.3,
        'divergence_bear': divergence_bear.astype(int) * -0.3,
        'high_volume': (volume_zscore > 1.5).astype(int) * 0.2,
        'low_volume': (volume_zscore < -1.5).astype(int) * -0.1,
    }
    
    if use_bvi:
        bvi_series = sum(bvi_components.values())
        # Normalize BVI to reasonable range
        bvi_series = bvi_series.clip(-2.0, 2.0)
    else:
        # Simple volume oscillator
        bvi_series = volume_zscore
    
    # Volume Profile calculation
    # Divide price range into bins and calculate volume distribution
    price_bins = np.linspace(df['low'].min(), df['high'].max(), 50)
    volume_profile = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        if pd.isna(df.iloc[i-1:i+1]['volume']).any():
            continue
            
        price_at_volume = df.iloc[i]['close']
        volume_at_price = df.iloc[i]['volume']
        
        # Find closest price bin
        closest_bin_idx = np.digitize(price_at_volume, price_bins) - 1
        if 0 <= closest_bin_idx < len(price_bins) - 1:
            volume_profile.iloc[i] = volume_at_price
    
    volume_profile_ma = volume_profile.rolling(window=period).mean()
    
    return bvi_series, volume_profile_ma


# =============================================================================
# Chart Creation Functions
# =============================================================================

def create_realtime_candlestick_chart(
    df: pd.DataFrame,
    chart_height: int = 700,
    timeframe: str = "15m",
    show_forming_bar: bool = False,
    bvi_enabled: bool = True,
    atr_channels_enabled: bool = True,
    order_blocks_enabled: bool = True,
) -> go.Figure:
    """Create a real-time candlestick chart with indicators.
    
    Args:
        df: DataFrame with OHLCV data
        chart_height: Height of the chart in pixels
        timeframe: Chart timeframe for context
        show_forming_bar: Whether to show the currently forming bar
        bvi_enabled: Whether to enable Better Volume Indicator
        atr_channels_enabled: Whether to show ATR channels
        order_blocks_enabled: Whether to show order blocks
        
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Create empty figure
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.15, 0.15],
            subplot_titles=("Price Chart", "Volume", "RSI")
        )
        fig.update_layout(height=chart_height, title="No Data Available")
        return fig
    
    # Sort by timestamp to ensure proper order
    df = df.sort_values('timestamp').copy()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.15, 0.15],
        subplot_titles=("Price Chart", "Volume", "RSI"),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff0044',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff0044',
            showlegend=True,
        ),
        row=1, col=1
    )
    
    # Calculate and add moving averages
    if len(df) >= 20:
        sma_20 = df['close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_20,
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    if len(df) >= 50:
        sma_50 = df['close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_50,
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Add RSI indicator
    if len(df) >= 14:
        rsi = _compute_rsi(df['close'])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
    
    # Add volume bars
    if 'volume' in df.columns:
        colors = ['green' if close >= open_ else 'red' 
                 for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7,
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Add Better Volume Indicator if enabled
    if bvi_enabled and 'volume' in df.columns and len(df) >= 20:
        bvi_series, volume_profile = calculate_better_volume_indicator(df)
        if not bvi_series.isna().all():
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=bvi_series,
                    mode='lines',
                    name='BVI',
                    line=dict(color='cyan', width=1),
                    opacity=0.8
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        height=chart_height,
        title=f"Realtime Chart - {timeframe}",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="RSI",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update candlestick axis
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        row=1, col=1
    )
    
    # Update volume axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        row=2, col=1
    )
    
    # Update RSI axis
    fig.update_yaxes(
        range=[0, 100],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        row=3, col=1
    )
    
    return fig


def create_candlestick_chart(summary: SimulationSummary, main_series: TimeframeSeries) -> go.Figure:
    """Create a detailed candlestick chart from simulation summary.
    
    Args:
        summary: SimulationSummary with analysis results
        main_series: TimeframeSeries with OHLCV data
        
    Returns:
        Plotly figure object
    """
    if not main_series or main_series.data.empty:
        # Create empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No Chart Data Available",
            height=600,
            showlegend=False
        )
        return fig
    
    df = main_series.data.copy()
    if df.empty:
        # Create empty figure
        fig = go.Figure()
        fig.update_layout(
            title="No Chart Data Available",
            height=600,
            showlegend=False
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.15, 0.15, 0.1],
        subplot_titles=("Price Chart", "Volume", "RSI", "MACD"),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Price",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff0044',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff0044',
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if len(df) >= 20:
        sma_20 = df['close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_20,
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
    
    if len(df) >= 50:
        sma_50 = df['close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sma_50,
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands if available
    if len(df) >= 20:
        bb_bands = _compute_bollinger_bands(df['close'])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb_bands['upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb_bands['middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='gray', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb_bands['lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Add volume
    if 'volume' in df.columns:
        colors = ['green' if close >= open_ else 'red' 
                 for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Add RSI
    if len(df) >= 14:
        rsi = _compute_rsi(df['close'])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi,
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1.5)
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Add MACD
    if len(df) >= 26:
        macd_data = _compute_macd(df['close'])
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd_data['macd'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=1.5)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd_data['signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=1.5)
            ),
            row=4, col=1
        )
        
        # MACD histogram
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=macd_data['histogram'],
                name='MACD Histogram',
                marker_color='gray',
                opacity=0.7
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        title=f"Trading Chart - {main_series.symbol} ({main_series.timeframe})",
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="RSI",
        yaxis4_title="MACD",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    for row in range(1, 5):
        fig.update_xaxes(
            rangeslider_visible=False,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            row=row, col=1
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            row=row, col=1
        )
    
    return fig


def create_multi_timeframe_chart(payload: dict) -> go.Figure:
    """Create a multi-timeframe analysis chart.
    
    Args:
        payload: Dictionary containing analysis results and data
        
    Returns:
        Plotly figure object
    """
    # Extract data from payload
    main_tf_data = payload.get('main_timeframe', {})
    higher_tf_data = payload.get('higher_timeframe', {})
    
    if not main_tf_data or not main_tf_data.get('data'):
        fig = go.Figure()
        fig.update_layout(
            title="No Multi-Timeframe Data Available",
            height=600,
            showlegend=False
        )
        return fig
    
    # Create figure with multiple timeframes
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        row_heights=[0.7, 0.15, 0.15],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        subplot_titles=(
            "Primary Timeframe Price", "Higher Timeframe Price",
            "Primary Volume", "Higher Volume",
            "Primary RSI", "Higher RSI"
        )
    )
    
    # Process main timeframe data
    main_df = main_tf_data.get('data', pd.DataFrame())
    if not main_df.empty:
        # Main timeframe candlestick
        fig.add_trace(
            go.Candlestick(
                x=main_df.index,
                open=main_df['open'],
                high=main_df['high'],
                low=main_df['low'],
                close=main_df['close'],
                name="Main TF",
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff0044'
            ),
            row=1, col=1
        )
        
        # Main timeframe volume
        if 'volume' in main_df.columns:
            colors = ['green' if close >= open_ else 'red' 
                     for close, open_ in zip(main_df['close'], main_df['open'])]
            fig.add_trace(
                go.Bar(
                    x=main_df.index,
                    y=main_df['volume'],
                    name="Main Vol",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Main timeframe RSI
        if len(main_df) >= 14:
            main_rsi = _compute_rsi(main_df['close'])
            fig.add_trace(
                go.Scatter(
                    x=main_df.index,
                    y=main_rsi,
                    mode='lines',
                    name="Main RSI",
                    line=dict(color='blue', width=1.5)
                ),
                row=3, col=1
            )
            
            # Add RSI levels for main timeframe
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=1)
    
    # Process higher timeframe data
    if higher_tf_data and higher_tf_data.get('data'):
        higher_df = higher_tf_data.get('data', pd.DataFrame())
        if not higher_df.empty:
            # Higher timeframe candlestick (smaller subplot)
            fig.add_trace(
                go.Candlestick(
                    x=higher_df.index,
                    open=higher_df['open'],
                    high=higher_df['high'],
                    low=higher_df['low'],
                    close=higher_df['close'],
                    name="Higher TF",
                    increasing_line_color='#00aa88',
                    decreasing_line_color='#aa0044',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Higher timeframe volume
            if 'volume' in higher_df.columns:
                colors = ['green' if close >= open_ else 'red' 
                         for close, open_ in zip(higher_df['close'], higher_df['open'])]
                fig.add_trace(
                    go.Bar(
                        x=higher_df.index,
                        y=higher_df['volume'],
                        name="Higher Vol",
                        marker_color=colors,
                        opacity=0.7,
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Higher timeframe RSI
            if len(higher_df) >= 14:
                higher_rsi = _compute_rsi(higher_df['close'])
                fig.add_trace(
                    go.Scatter(
                        x=higher_df.index,
                        y=higher_rsi,
                        mode='lines',
                        name="Higher RSI",
                        line=dict(color='orange', width=1.5),
                        showlegend=False
                    ),
                    row=3, col=2
                )
                
                # Add RSI levels for higher timeframe
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=3, col=2)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=3, col=2)
    
    # Update layout
    fig.update_layout(
        height=900,
        title="Multi-Timeframe Analysis",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes for all subplots
    for row in range(1, 4):
        for col in range(1, 3):
            fig.update_xaxes(
                rangeslider_visible=False,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.3)',
                row=row, col=col
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.3)',
                row=row, col=col
            )
    
    return fig


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Technical indicators
    "_compute_rsi",
    "_compute_macd",
    "_compute_bollinger_bands",
    "_rolling_equals",
    "calculate_better_volume_indicator",
    # Chart creation functions
    "create_realtime_candlestick_chart",
    "create_candlestick_chart",
    "create_multi_timeframe_chart",
]