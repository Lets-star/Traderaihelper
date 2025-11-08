#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for the web UI. Please install it via 'pip install pandas'.") from e

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from indicator_collector.collector import collect_metrics
from indicator_collector.indicator_metrics import SimulationSummary
from indicator_collector.real_data_validator import DataValidationError
from indicator_collector.time_series import TimeframeSeries
from indicator_collector.timeframes import Timeframe
from indicator_collector.trade_signals import calculate_position_metrics, calculate_tp_sl_levels
from indicator_collector.trading_system import indicator_defaults_for
from indicator_collector.trading_system.automated_signals import run_automated_signal_flow
from indicator_collector.trading_system.signal_schema import is_valid_signal_structure


def format_correlation(value: float) -> str:
    """Format correlation value with color coding."""
    if value > 0.7:
        return f"🟢 {value:.3f}"
    elif value > 0.3:
        return f"🟡 {value:.3f}"
    elif value > -0.3:
        return f"⚪ {value:.3f}"
    elif value > -0.7:
        return f"🟠 {value:.3f}"
    else:
        return f"🔴 {value:.3f}"


def format_flow(value: float) -> str:
    """Format flow value with color coding."""
    if abs(value) < 1000:
        return f"⚪ ${value:,.0f}"
    elif value > 0:
        return f"🟢 ${value:,.0f}"
    else:
        return f"🔴 ${value:,.0f}"


def ui_key(prefix: str, label: str) -> str:
    """Generate a unique key for Streamlit widgets.
    
    Args:
        prefix: The prefix for the key (e.g., tab name or section)
        label: The widget label
        
    Returns:
        A standardized unique key combining prefix and label
    """
    label_slug = label.lower().replace(" ", "_").replace("%", "pct")
    return f"{prefix}_{label_slug}"


st.set_page_config(
    page_title="Token Charts & Indicators",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

POPULAR_TOKENS = [
    "BINANCE:BTCUSDT",
    "BINANCE:ETHUSDT",
    "BINANCE:BNBUSDT",
    "BINANCE:SOLUSDT",
    "BINANCE:ADAUSDT",
    "BINANCE:XRPUSDT",
    "BINANCE:DOGEUSDT",
    "BINANCE:DOTUSDT",
    "BINANCE:MATICUSDT",
    "BINANCE:AVAXUSDT",
]

TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]
AUTOMATED_SIGNALS_TIMEFRAMES = ["1m", "5m", "15m", "1h", "3h", "4h", "1d"]
AUTOMATED_SIGNALS_STATE_KEY = "automated_signals_state"

CACHE_VERSION = "binance-real-data-v1"
SYNTHETIC_FLAG_KEYS = {"is_synthetic", "synthetic", "mock", "demo", "paper", "testnet"}
SYNTHETIC_SOURCE_VALUES = {"sample", "demo", "mock", "paper", "testnet", "local"}
SYNTHETIC_MARKER_VALUES = {
    "mock",
    "test",
    "demo",
    "simulated",
    "synthetic",
    "fake",
    "sample",
    "paper",
    "backtest",
    "historical_sim",
    "generated",
    "artificial",
}


def sanitize_payload_for_real_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove known synthetic markers and enforce Binance metadata."""
    if not isinstance(payload, dict):
        return payload

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            keys_to_remove = []
            for key, value in obj.items():
                key_lower = key.lower()
                if key_lower in SYNTHETIC_FLAG_KEYS:
                    keys_to_remove.append(key)
                    continue

                if isinstance(value, (dict, list)):
                    obj[key] = _clean(value)
                elif isinstance(value, str):
                    value_lower = value.lower()
                    if key_lower in {"source", "exchange"} and value_lower in SYNTHETIC_SOURCE_VALUES:
                        obj[key] = "binance"
                    elif any(marker in value_lower for marker in SYNTHETIC_MARKER_VALUES):
                        obj[key] = "real_market_data"
                    else:
                        obj[key] = value
                else:
                    obj[key] = value

            for key in keys_to_remove:
                obj.pop(key, None)
            return obj
        if isinstance(obj, list):
            return [_clean(item) for item in obj]
        if isinstance(obj, str):
            value_lower = obj.lower()
            if any(marker in value_lower for marker in SYNTHETIC_MARKER_VALUES):
                return "real_market_data"
        return obj

    _clean(payload)

    metadata = payload.setdefault("metadata", {})
    metadata["source"] = "binance"
    metadata["exchange"] = metadata.get("exchange", "binance") or "binance"
    metadata["real_data"] = True
    metadata["is_real_data"] = True
    metadata["real_data_validated"] = True
    metadata["data_quality"] = "validated_real_data"

    return payload


@st.cache_data(ttl=300)
def cached_run_automated_signals(
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    source: str = "binance",
) -> Dict[str, Any]:
    """Cache Binance signal generation results for performance."""
    _ = source  # Included to differentiate cache keys if needed
    start_dt = datetime.fromisoformat(start_iso)
    end_dt = datetime.fromisoformat(end_iso)
    result = run_automated_signal_flow(
        symbol,
        timeframe,
        start_dt,
        end_dt,
        validate_real_data=True,
    )
    return {
        "candles": result.candles,
        "processed_payload": result.processed_payload,
        "explicit_signal": result.explicit_signal,
    }


@st.cache_data(ttl=300)
def load_indicator_data(symbol: str, timeframe: str, period: int, token: str, cache_version: str) -> tuple:
    _ = cache_version  # Ensures cache invalidation when version changes
    result = collect_metrics(
        symbol=symbol,
        timeframe=timeframe,
        period=period,
        token=token,
    )
    return result.summary, result.payload, result.main_series


def create_candlestick_chart(summary: SimulationSummary, main_series: TimeframeSeries):
    candles = main_series.candles
    
    df = pd.DataFrame([
        {
            "timestamp": datetime.fromtimestamp(c.close_time / 1000),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ])
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.20],
        subplot_titles=("Price & Indicators", "RSI", "MACD", "Volume"),
    )
    
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ),
        row=1, col=1,
    )
    
    if summary.snapshots and len(summary.snapshots) == len(candles):
        bollinger_upper = [s.bollinger_upper for s in summary.snapshots]
        bollinger_middle = [s.bollinger_middle for s in summary.snapshots]
        bollinger_lower = [s.bollinger_lower for s in summary.snapshots]
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=bollinger_upper,
                name="BB Upper",
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=bollinger_middle,
                name="BB Middle",
                line=dict(color="rgba(255, 255, 255, 0.5)", width=1, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=bollinger_lower,
                name="BB Lower",
                line=dict(color="rgba(173, 216, 230, 0.5)", width=1),
                fill="tonexty",
                fillcolor="rgba(173, 216, 230, 0.1)",
            ),
            row=1, col=1,
        )
        
        atr_colors = {
            "atr_trend_3x": ("rgba(0, 255, 0, 0.6)", 1),
            "atr_trend_8x": ("rgba(255, 165, 0, 0.6)", 2),
            "atr_trend_21x": ("rgba(255, 0, 0, 0.6)", 3),
        }
        
        for atr_key, (color, width) in atr_colors.items():
            atr_values = [s.atr_channels.get(atr_key) if s.atr_channels else None for s in summary.snapshots]
            if any(v is not None for v in atr_values):
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=atr_values,
                        name=f"ATR {atr_key.replace('atr_trend_', '').replace('x', '')}x",
                        line=dict(color=color, width=width),
                        mode="lines",
                    ),
                    row=1, col=1,
                )
        
        rsi_values = [s.rsi if s.rsi is not None else 50 for s in summary.snapshots]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=rsi_values,
                name="RSI",
                line=dict(color="purple", width=2),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
        
        macd_values = [s.macd if s.macd is not None else 0 for s in summary.snapshots]
        macd_signal = [s.macd_signal if s.macd_signal is not None else 0 for s in summary.snapshots]
        macd_histogram = [s.macd_histogram if s.macd_histogram is not None else 0 for s in summary.snapshots]
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=macd_values,
                name="MACD",
                line=dict(color="blue", width=2),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=macd_signal,
                name="Signal",
                line=dict(color="orange", width=2),
            ),
            row=3, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=macd_histogram,
                name="Histogram",
                marker_color=["green" if val >= 0 else "red" for val in macd_histogram],
            ),
            row=3, col=1,
        )
    
    for zone in summary.active_fvg_zones:
        zone_type = zone.zone_type
        color = "rgba(0, 255, 0, 0.2)" if "Bull" in zone_type else "rgba(255, 0, 0, 0.2)"
        
        if zone.created_index < len(df):
            start_time = df.iloc[zone.created_index]["timestamp"]
            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=df["timestamp"].iloc[-1],
                y0=zone.bottom,
                y1=zone.top,
                fillcolor=color,
                line=dict(color=color.replace("0.2", "0.5"), width=1),
                row=1, col=1,
            )
    
    for zone in summary.active_ob_zones:
        zone_type = zone.zone_type
        color = "rgba(0, 0, 255, 0.15)" if "Bull" in zone_type else "rgba(255, 165, 0, 0.15)"
        
        if zone.created_index < len(df):
            start_time = df.iloc[zone.created_index]["timestamp"]
            fig.add_shape(
                type="rect",
                x0=start_time,
                x1=df["timestamp"].iloc[-1],
                y0=zone.bottom,
                y1=zone.top,
                fillcolor=color,
                line=dict(color=color.replace("0.15", "0.5"), width=1, dash="dash"),
                row=1, col=1,
            )
    
    for signal in summary.signals:
        if signal.bar_index < len(df):
            signal_time = df.iloc[signal.bar_index]["timestamp"]
            signal_price = signal.price
            
            if signal.signal_type == "bullish":
                fig.add_trace(
                    go.Scatter(
                        x=[signal_time],
                        y=[signal_price],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=15, color="lime"),
                        name=f"Buy Signal",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[signal_time],
                        y=[signal_price],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=15, color="red"),
                        name=f"Sell Signal",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )
    
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["volume"],
            name="Volume",
            marker_color="rgba(100, 150, 255, 0.5)",
        ),
        row=4, col=1,
    )
    
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=4, col=1)
    
    return fig


def create_multi_timeframe_chart(payload: dict):
    mtf_data = payload.get("multi_timeframe", {})
    trend_strength = mtf_data.get("trend_strength", {})
    direction = mtf_data.get("direction", {})
    
    if not trend_strength:
        return None
    
    timeframes = list(trend_strength.keys())
    strengths = list(trend_strength.values())
    directions = [direction.get(tf, "neutral") for tf in timeframes]
    
    colors = []
    for d in directions:
        if d == "bullish":
            colors.append("green")
        elif d == "bearish":
            colors.append("red")
        else:
            colors.append("gray")
    
    fig = go.Figure(data=[
        go.Bar(
            x=timeframes,
            y=strengths,
            marker_color=colors,
            text=[f"{s:.1f}" for s in strengths],
            textposition="outside",
        )
    ])
    
    fig.update_layout(
        title="Multi-Timeframe Trend Strength",
        xaxis_title="Timeframe",
        yaxis_title="Strength (0-100)",
        yaxis_range=[0, 100],
        height=400,
        template="plotly_dark",
    )
    
    return fig


def main():
    st.title("📈 Token Charts & Indicators Dashboard")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Token Selection")
        token_input_mode = st.radio("Input Mode", ["Select from list", "Custom token"], key=ui_key("sidebar", "input_mode"))

        if token_input_mode == "Select from list":
            selected_token = st.selectbox("Select Token", POPULAR_TOKENS, index=0, key=ui_key("sidebar", "select_token"))
        else:
            selected_token = st.text_input("Custom Token (e.g., BINANCE:BTCUSDT)", "BINANCE:BTCUSDT", key=ui_key("sidebar", "custom_token"))

        st.subheader("Timeframe & Period")
        selected_timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("15m"), key=ui_key("sidebar", "timeframe"))
        selected_period = st.slider("Analysis Period (bars)", min_value=50, max_value=1000, value=200, step=50, key=ui_key("sidebar", "analysis_period"))
        
        st.subheader("Export Options")
        export_token = st.text_input("Export Token/ID", value="export-session-001", help="Token to identify this analysis session", key=ui_key("sidebar", "export_token"))
        
        analyze_button = st.button("🔄 Analyze", type="primary", use_container_width=True)
    
    if analyze_button or "summary" not in st.session_state:
        if analyze_button:
            load_indicator_data.clear()
        with st.spinner(f"Analyzing {selected_token} on {selected_timeframe} timeframe..."):
            try:
                summary, payload, main_series = load_indicator_data(
                    selected_token,
                    selected_timeframe,
                    selected_period,
                    export_token,
                    CACHE_VERSION,
                )
                sanitized_payload = sanitize_payload_for_real_data(payload)
                st.session_state.summary = summary
                st.session_state.payload = sanitized_payload
                st.session_state.main_series = main_series
                st.session_state.export_token = export_token
                st.success("✅ Analysis completed successfully!")
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                return
    
    if "summary" not in st.session_state:
        st.info("👈 Configure parameters in the sidebar and click 'Analyze' to begin.")
        return
    
    summary = st.session_state.summary
    payload = sanitize_payload_for_real_data(st.session_state.payload)
    st.session_state.payload = payload
    main_series = st.session_state.main_series
    
    (
        chart_tab,
        multi_tab,
        latest_tab,
        signals_tab,
        volume_tab,
        structure_tab,
        fundamentals_tab,
        breadth_tab,
        onchain_tab,
        composite_tab,
        patterns_tab,
        trade_tab,
        automated_signals_tab,
        backtest_tab,
        adaptive_tab,
        astrology_tab,
        export_tab,
    ) = st.tabs([
        "📊 Charts",
        "📈 Multi-Timeframe",
        "📋 Latest Metrics",
        "🎯 Signals & Zones",
        "📊 Volume Analysis",
        "🏗️ Market Structure",
        "📈 Fundamentals",
        "🌐 Breadth Indicators",
        "🔗 On-chain Metrics",
        "🧩 Composite Indicators",
        "🌊 Patterns & Waves",
        "🎯 Trade Signals",
        "🤖 Automated Signals",
        "🔬 Backtesting",
        "⚖️ Adaptive Weights",
        "🔮 Astrology",
        "💾 Export",
    ])
    
    with chart_tab:
        st.subheader(f"Price Chart with Indicators - {selected_token}")
        fig = create_candlestick_chart(summary, main_series)
        st.plotly_chart(fig, use_container_width=True)
    
    with multi_tab:
        st.subheader("Multi-Timeframe Analysis")
        mtf_fig = create_multi_timeframe_chart(payload)
        if mtf_fig:
            st.plotly_chart(mtf_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trend Strength by Timeframe")
            mtf_data = payload.get("multi_timeframe", {})
            trend_df = pd.DataFrame([
                {"Timeframe": tf, "Strength": f"{val:.2f}"}
                for tf, val in mtf_data.get("trend_strength", {}).items()
            ])
            if not trend_df.empty:
                st.dataframe(trend_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Direction by Timeframe")
            direction_df = pd.DataFrame([
                {"Timeframe": tf, "Direction": val.upper()}
                for tf, val in mtf_data.get("direction", {}).items()
            ])
            if not direction_df.empty:
                st.dataframe(direction_df, use_container_width=True, hide_index=True)
        
        if payload.get("multi_symbol"):
            st.markdown("### Multi-Symbol Confirmation")
            multi_sym = payload["multi_symbol"]
            
            sym_col1, sym_col2 = st.columns(2)
            with sym_col1:
                st.markdown("**Signals:**")
                for sym, signal in multi_sym.get("signals", {}).items():
                    color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
                    st.write(f"{color} {sym}: **{signal}**")
            
            with sym_col2:
                st.markdown("**Trend Strength:**")
                for sym, strength in multi_sym.get("trend_strength", {}).items():
                    if strength is not None:
                        st.write(f"{sym}: **{strength:.2f}**")
    
    with latest_tab:
        st.subheader("Latest Market Snapshot")
        
        latest = payload.get("latest", {})
        atr_channels = payload.get("atr_channels", {})
        orderbook_data = payload.get("orderbook")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Close Price", f"${latest.get('close', 0):.4f}")
            st.metric("Volume", f"{latest.get('volume', 0):,.0f}")
        
        with col2:
            st.metric("Trend Strength", f"{latest.get('trend_strength', 0):.2f}")
            st.metric("Pattern Score", f"{latest.get('pattern_score', 0):.2f}")
        
        with col3:
            st.metric("Market Sentiment", f"{latest.get('market_sentiment', 0):.2f}")
            st.metric("RSI", f"{latest.get('rsi', 0):.2f}" if latest.get('rsi') else "N/A")
        
        with col4:
            confluence = latest.get('confluence_score', 0)
            confluence_bias = latest.get('confluence_bias', 'neutral')
            confluence_bull = latest.get('confluence_bullish', 0)
            confluence_bear = latest.get('confluence_bearish', 0)
            
            if confluence_bias == 'bullish':
                confluence_color = "🟢"
            elif confluence_bias == 'bearish':
                confluence_color = "🔴"
            else:
                confluence_color = "⚪"
            
            st.metric("Confluence Score", f"{confluence_color} {confluence:.2f}" if confluence else "N/A")
            st.markdown(f"**Bull:** {confluence_bull:.2f} | **Bear:** {confluence_bear:.2f}")
            
            structure = latest.get('structure_state', 'neutral')
            structure_emoji = "🟢" if structure == "bullish" else "🔴" if structure == "bearish" else "⚪"
            st.metric("Structure", f"{structure_emoji} {structure.upper()}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Technical Indicators")
            indicators_df = pd.DataFrame([
                {"Indicator": "MACD", "Value": f"{latest.get('macd', 0):.4f}" if latest.get('macd') else "N/A"},
                {"Indicator": "MACD Signal", "Value": f"{latest.get('macd_signal', 0):.4f}" if latest.get('macd_signal') else "N/A"},
                {"Indicator": "MACD Histogram", "Value": f"{latest.get('macd_histogram', 0):.4f}" if latest.get('macd_histogram') else "N/A"},
                {"Indicator": "Bollinger Upper", "Value": f"{latest.get('bollinger_upper', 0):.4f}" if latest.get('bollinger_upper') else "N/A"},
                {"Indicator": "Bollinger Middle", "Value": f"{latest.get('bollinger_middle', 0):.4f}" if latest.get('bollinger_middle') else "N/A"},
                {"Indicator": "Bollinger Lower", "Value": f"{latest.get('bollinger_lower', 0):.4f}" if latest.get('bollinger_lower') else "N/A"},
            ])
            st.dataframe(indicators_df, use_container_width=True, hide_index=True)
            
            if atr_channels:
                st.markdown("### ATR Channels")
                atr_df = pd.DataFrame([
                    {"ATR Level": k.replace("atr_trend_", "ATR ").upper(), "Value": f"{v:.4f}" if v is not None else "N/A"}
                    for k, v in atr_channels.items()
                ])
                st.dataframe(atr_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### Performance Statistics")
            success_rates = payload.get("success_rates", {})
            pnl_stats = payload.get("pnl_stats", {})
            
            stats_df = pd.DataFrame([
                {"Metric": "Overall Win Rate", "Value": f"{success_rates.get('overall_win_rate', 0):.2f}%"},
                {"Metric": "Bull Win Rate", "Value": f"{success_rates.get('bull_win_rate', 0):.2f}%"},
                {"Metric": "Bear Win Rate", "Value": f"{success_rates.get('bear_win_rate', 0):.2f}%"},
                {"Metric": "Cumulative PnL", "Value": f"{pnl_stats.get('cum_pnl_pct', 0):.2f}%"},
                {"Metric": "Max Drawdown", "Value": f"{pnl_stats.get('max_drawdown_pct', 0):.2f}%"},
                {"Metric": "Trades Closed", "Value": f"{pnl_stats.get('trades_closed', 0)}"},
            ])
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        cme_gaps = latest.get("cme_gaps", {})
        if cme_gaps:
            st.markdown("---")
            st.markdown("### 📊 CME Gap Analysis (CME Futures)")
            
            if cme_gaps.get("total_unfilled_gaps", 0) == 0:
                st.info("All CME gaps are currently filled. No outstanding gaps detected near the current price.")
            else:
                gap_col1, gap_col2 = st.columns(2)
                
                with gap_col1:
                    st.markdown("#### Nearest Gaps Above Current Price")
                    gaps_above = cme_gaps.get("nearest_gaps_above", [])
                    if gaps_above:
                        gaps_above_df = pd.DataFrame([
                            {
                                "Type": gap["type"].replace("_", " ").upper(),
                                "Top": f"${gap['gap_top']:.2f}",
                                "Bottom": f"${gap['gap_bottom']:.2f}",
                                "Distance": f"{gap['distance_pct']:.2f}%",
                                "Size": f"{gap['gap_size_pct']:.2f}%"
                            }
                            for gap in gaps_above[:5]
                        ])
                        st.dataframe(gaps_above_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No unfilled gaps above current price")
                
                with gap_col2:
                    st.markdown("#### Nearest Gaps Below Current Price")
                    gaps_below = cme_gaps.get("nearest_gaps_below", [])
                    if gaps_below:
                        gaps_below_df = pd.DataFrame([
                            {
                                "Type": gap["type"].replace("_", " ").upper(),
                                "Top": f"${gap['gap_top']:.2f}",
                                "Bottom": f"${gap['gap_bottom']:.2f}",
                                "Distance": f"{gap['distance_pct']:.2f}%",
                                "Size": f"{gap['gap_size_pct']:.2f}%"
                            }
                            for gap in gaps_below[:5]
                        ])
                        st.dataframe(gaps_below_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No unfilled gaps below current price")
        
        if orderbook_data:
            st.markdown("---")
            st.markdown("### 📊 Order Book Analysis (Binance)")
            
            ob_col1, ob_col2, ob_col3 = st.columns(3)
            
            with ob_col1:
                best_bid = orderbook_data.get('best_bid')
                st.metric("Best Bid", f"${best_bid:.4f}" if best_bid is not None else "N/A")
                best_ask = orderbook_data.get('best_ask')
                st.metric("Best Ask", f"${best_ask:.4f}" if best_ask is not None else "N/A")
            
            with ob_col2:
                spread = orderbook_data.get('spread')
                st.metric("Spread", f"${spread:.4f}" if spread is not None else "N/A")
                mid_price = orderbook_data.get('mid_price')
                st.metric("Mid Price", f"${mid_price:.4f}" if mid_price is not None else "N/A")
            
            with ob_col3:
                ratio = orderbook_data.get('bid_ask_ratio_top10')
                st.metric("Bid/Ask Ratio (Top 10)", f"{ratio:.2f}" if ratio is not None else "N/A")
                imbalance = orderbook_data.get('volume_imbalance_top10')
                st.metric("Volume Imbalance", f"{imbalance:.2f}" if imbalance is not None else "N/A")
            
            st.markdown("#### Volume at Price Levels")
            price_levels = orderbook_data.get('price_levels', {})
            if price_levels:
                levels_data = []
                for level, data in price_levels.items():
                    ratio_val = None
                    ask_volume = data.get('ask_volume', 0)
                    bid_volume = data.get('bid_volume', 0)
                    if ask_volume:
                        ratio_val = bid_volume / ask_volume
                    levels_data.append({
                        "Level": level,
                        "Bid Volume": f"{bid_volume:.2f}",
                        "Ask Volume": f"{ask_volume:.2f}",
                        "Ratio": f"{ratio_val:.2f}" if ratio_val is not None else "N/A"
                    })
                ob_levels_df = pd.DataFrame(levels_data)
                st.dataframe(ob_levels_df, use_container_width=True, hide_index=True)
            
            sections = orderbook_data.get('sections', {})
            if sections:
                st.markdown("#### Aggregated Depth (Top Levels)")
                section_rows = []
                bids_sections = sections.get('bids', {})
                asks_sections = sections.get('asks', {})
                for key, label in (('top_5', 'Top 5'), ('top_10', 'Top 10'), ('top_20', 'Top 20')):
                    bid_info = bids_sections.get(key, {})
                    ask_info = asks_sections.get(key, {})
                    section_rows.append({
                        "Levels": label,
                        "Bid Volume": f"{bid_info.get('total_volume', 0):.2f}",
                        "Bid W. Price": f"{bid_info.get('weighted_price'):.4f}" if bid_info.get('weighted_price') is not None else "N/A",
                        "Ask Volume": f"{ask_info.get('total_volume', 0):.2f}",
                        "Ask W. Price": f"{ask_info.get('weighted_price'):.4f}" if ask_info.get('weighted_price') is not None else "N/A",
                    })
                ob_sections_df = pd.DataFrame(section_rows)
                st.dataframe(ob_sections_df, use_container_width=True, hide_index=True)
            
            aggregated_bins = orderbook_data.get('aggregated_bins', {})
            if aggregated_bins:
                st.markdown("#### Depth by 2% Aggregated Bins")
                summary_rows = []
                for range_label, data in aggregated_bins.items():
                    summary_rows.append({
                        "Range": range_label,
                        "Bid Volume": f"{data.get('total_bid_volume', 0):.2f}",
                        "Ask Volume": f"{data.get('total_ask_volume', 0):.2f}",
                        "Imbalance": f"{(data.get('total_bid_volume', 0) - data.get('total_ask_volume', 0)):.2f}"
                    })
                if summary_rows:
                    agg_summary_df = pd.DataFrame(summary_rows)
                    st.dataframe(agg_summary_df, use_container_width=True, hide_index=True)
                
                for range_label, data in aggregated_bins.items():
                    with st.expander(f"{range_label} Range Breakdown", expanded=False):
                        bid_bins = data.get('bid_bins_2pct', [])
                        ask_bins = data.get('ask_bins_2pct', [])
                        bid_df = pd.DataFrame([
                            {
                                "Bin": f"{idx * 2}-{(idx + 1) * 2}%",
                                "Orders": bin_info.get('count', 0),
                                "Volume": round(bin_info.get('volume', 0), 2),
                                "Avg Price": f"${bin_info.get('avg_price', 0):.4f}" if bin_info.get('avg_price') else "N/A"
                            }
                            for idx, bin_info in enumerate(bid_bins)
                        ])
                        ask_df = pd.DataFrame([
                            {
                                "Bin": f"{idx * 2}-{(idx + 1) * 2}%",
                                "Orders": bin_info.get('count', 0),
                                "Volume": round(bin_info.get('volume', 0), 2),
                                "Avg Price": f"${bin_info.get('avg_price', 0):.4f}" if bin_info.get('avg_price') else "N/A"
                            }
                            for idx, bin_info in enumerate(ask_bins)
                        ])
                        b_col, a_col = st.columns(2)
                        with b_col:
                            st.markdown("**Bid Bins**")
                            if not bid_df.empty:
                                st.dataframe(bid_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No bid volume in this range")
                        with a_col:
                            st.markdown("**Ask Bins**")
                            if not ask_df.empty:
                                st.dataframe(ask_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No ask volume in this range")

            advanced = payload.get("advanced", {})
            market_context_data = advanced.get("market_context", {})
            orderbook_context = market_context_data.get("orderbook_context", {})
            mm_activity = orderbook_context.get("market_maker_activity", {})

            if mm_activity and not mm_activity.get("error"):
                st.markdown("---")
                st.markdown("### 🤖 Market Maker Detection (Real-Time)")

                if mm_activity.get("warning"):
                    st.warning(mm_activity["warning"])
                else:
                    mm_detected = mm_activity.get("market_maker_detected", False)
                    confidence = mm_activity.get("confidence", 0)
                    activity_level = mm_activity.get("activity_level", "unknown")

                    mm_col1, mm_col2, mm_col3 = st.columns(3)

                    with mm_col1:
                        status_emoji = "✅" if mm_detected else "❌"
                        st.metric(
                            "Market Maker Detected",
                            f"{status_emoji} {'YES' if mm_detected else 'NO'}"
                        )

                    with mm_col2:
                        st.metric("Confidence", f"{confidence}%")
                        st.progress(confidence / 100)

                    with mm_col3:
                        activity_emoji = "🟢" if activity_level == "high" else "🟡" if activity_level == "medium" else "⚪"
                        st.metric(
                            "Activity Level",
                            f"{activity_emoji} {activity_level.upper()}"
                        )

                    signals = mm_activity.get("signals", [])
                    if signals:
                        st.markdown("#### Detected Signals")
                        signal_tags = " • ".join([f"`{s.replace('_', ' ').title()}`" for s in signals])
                        st.markdown(signal_tags)

                    interpretation = mm_activity.get("interpretation", "")
                    if interpretation:
                        st.info(interpretation)

                    details = mm_activity.get("details", {})

                    with st.expander("📊 Order Walls Analysis", expanded=False):
                        walls = details.get("order_walls", {})
                        wall_col1, wall_col2 = st.columns(2)

                        with wall_col1:
                            st.markdown("**Bid Walls**")
                            bid_walls = walls.get("bid_walls", [])
                            if bid_walls:
                                bid_walls_df = pd.DataFrame([
                                    {
                                        "Price": f"${w['price']:.8f}",
                                        "Volume": f"{w['volume']:.2f}",
                                        "Ratio": f"{w['volume_ratio']:.2f}x",
                                        "Distance": f"{w['distance_from_mid_pct']:.3f}%" if w.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for w in bid_walls
                                ])
                                st.dataframe(bid_walls_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No significant bid walls detected")

                        with wall_col2:
                            st.markdown("**Ask Walls**")
                            ask_walls = walls.get("ask_walls", [])
                            if ask_walls:
                                ask_walls_df = pd.DataFrame([
                                    {
                                        "Price": f"${w['price']:.8f}",
                                        "Volume": f"{w['volume']:.2f}",
                                        "Ratio": f"{w['volume_ratio']:.2f}x",
                                        "Distance": f"{w['distance_from_mid_pct']:.3f}%" if w.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for w in ask_walls
                                ])
                                st.dataframe(ask_walls_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No significant ask walls detected")

                        wall_pressure = walls.get("wall_pressure", "neutral")
                        wall_emoji = "🟢" if wall_pressure == "bullish" else "🔴" if wall_pressure == "bearish" else "⚪"
                        st.markdown(f"**Wall Pressure:** {wall_emoji} {wall_pressure.upper()}")

                    with st.expander("🔄 Layered Orders Analysis", expanded=False):
                        layers = details.get("layered_orders", {})
                        layering_score = layers.get("layering_score", 0)
                        st.metric("Layering Score", f"{layering_score}/100")
                        st.progress(layering_score / 100)

                        layer_col1, layer_col2 = st.columns(2)

                        with layer_col1:
                            st.markdown("**Bid Layers**")
                            bid_layers = layers.get("bid_layers", [])
                            if bid_layers:
                                bid_layers_df = pd.DataFrame([
                                    {
                                        "Start": f"${l['start_price']:.8f}",
                                        "End": f"${l['end_price']:.8f}",
                                        "Levels": l['levels'],
                                        "Volume": f"{l['total_volume']:.2f}",
                                        "Distance": f"{l['distance_from_mid_pct']:.3f}%" if l.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for l in bid_layers
                                ])
                                st.dataframe(bid_layers_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No bid layers detected")

                        with layer_col2:
                            st.markdown("**Ask Layers**")
                            ask_layers = layers.get("ask_layers", [])
                            if ask_layers:
                                ask_layers_df = pd.DataFrame([
                                    {
                                        "Start": f"${l['start_price']:.8f}",
                                        "End": f"${l['end_price']:.8f}",
                                        "Levels": l['levels'],
                                        "Volume": f"{l['total_volume']:.2f}",
                                        "Distance": f"{l['distance_from_mid_pct']:.3f}%" if l.get('distance_from_mid_pct') else "N/A",
                                    }
                                    for l in ask_layers
                                ])
                                st.dataframe(ask_layers_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("No ask layers detected")

                    with st.expander("🚨 Quote Stuffing Analysis", expanded=False):
                        stuffing = details.get("quote_stuffing", {})
                        stuffing_detected = stuffing.get("stuffing_detected", False)
                        concentration_score = stuffing.get("concentration_score", 0)

                        stuff_col1, stuff_col2 = st.columns(2)

                        with stuff_col1:
                            st.metric("Stuffing Detected", "⚠️ YES" if stuffing_detected else "✅ NO")
                            st.metric("Concentration Score", f"{concentration_score:.2f}/100")

                        with stuff_col2:
                            bid_conc = stuffing.get("bid_concentration", {})
                            ask_conc = stuffing.get("ask_concentration", {})
                            st.metric("Bid Density", f"{bid_conc.get('density', 0):.2f}%")
                            st.metric("Ask Density", f"{ask_conc.get('density', 0):.2f}%")

                    with st.expander("📊 Spread Manipulation Analysis", expanded=False):
                        manipulation = details.get("spread_analysis", {})
                        manip_risk = manipulation.get("manipulation_risk", "unknown")
                        manip_score = manipulation.get("manipulation_score", 0)
                        spread_quality = manipulation.get("spread_quality", "unknown")

                        manip_col1, manip_col2, manip_col3 = st.columns(3)

                        with manip_col1:
                            risk_emoji = "🔴" if manip_risk == "high" else "🟡" if manip_risk == "medium" else "🟢"
                            st.metric("Manipulation Risk", f"{risk_emoji} {manip_risk.upper()}")

                        with manip_col2:
                            st.metric("Manipulation Score", f"{manip_score}/100")

                        with manip_col3:
                            quality_emoji = "🟢" if spread_quality == "good" else "🟡" if spread_quality == "fair" else "🔴"
                            st.metric("Spread Quality", f"{quality_emoji} {spread_quality.upper()}")

                        indicators = manipulation.get("manipulation_indicators", [])
                        if indicators:
                            st.markdown("**Detected Indicators:**")
                            indicator_text = " • ".join([f"`{ind.replace('_', ' ').title()}`" for ind in indicators])
                            st.markdown(indicator_text)
    
    with signals_tab:
        st.subheader("Signals & Zones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trading Signals")
            signals = payload.get("signals", [])
            
            if signals:
                signals_df = pd.DataFrame([
                    {
                        "Type": "🟢 BUY" if s["type"] == "bullish" else "🔴 SELL",
                        "Price": f"${s['price']:.4f}",
                        "Time": s.get("time_iso", "N/A")[:19],
                        "Strength": f"{s.get('strength', 0):.2f}" if s.get('strength') else "N/A",
                    }
                    for s in signals[-20:]
                ])
                st.dataframe(signals_df, use_container_width=True, hide_index=True)
            else:
                st.info("No signals detected in the analysis period.")
        
        with col2:
            st.markdown("### Active Zones")
            zones = payload.get("zones", [])
            
            if zones:
                zones_df = pd.DataFrame([
                    {
                        "Type": z["type"],
                        "Top": f"{z['top']:.4f}",
                        "Bottom": f"{z['bottom']:.4f}",
                        "Breaker": "✅" if z.get("breaker") else "❌",
                    }
                    for z in zones[:20]
                ])
                st.dataframe(zones_df, use_container_width=True, hide_index=True)
            else:
                st.info("No active zones detected.")
        
        st.markdown("---")
        st.markdown("### Structure Levels")
        structure_levels = payload.get("last_structure_levels", {})
        if structure_levels:
            struct_col1, struct_col2 = st.columns(2)
            with struct_col1:
                high_level = structure_levels.get("high")
                st.metric("Structure High", f"${high_level:.4f}" if high_level else "N/A")
            with struct_col2:
                low_level = structure_levels.get("low")
                st.metric("Structure Low", f"${low_level:.4f}" if low_level else "N/A")
    
    with volume_tab:
        st.subheader("📊 Volume Analysis")
        advanced = payload.get("advanced", {})
        volume_analysis = advanced.get("volume_analysis", {})
        
        st.markdown("### Volume Profile (VPVR)")
        vpvr = volume_analysis.get("vpvr", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            poc = vpvr.get("poc")
            st.metric("Point of Control (POC)", f"${poc:.4f}" if poc else "N/A")
        with col2:
            va_high = vpvr.get("value_area", {}).get("high")
            st.metric("Value Area High", f"${va_high:.4f}" if va_high else "N/A")
        with col3:
            va_low = vpvr.get("value_area", {}).get("low")
            st.metric("Value Area Low", f"${va_low:.4f}" if va_low else "N/A")
        
        levels = vpvr.get("levels", [])
        if levels:
            st.markdown("#### Top Volume Levels")
            vpvr_df = pd.DataFrame([
                {
                    "Price": f"${level['price']:.4f}",
                    "Volume": f"{level['volume']:,.0f}",
                    "Percentage": f"{level['percentage']:.2f}%"
                }
                for level in levels[:10]
            ])
            st.dataframe(vpvr_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Cumulative Volume Delta (CVD)")
        cvd = volume_analysis.get("cvd", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest CVD", f"{cvd.get('latest', 0):,.0f}")
        with col2:
            st.metric("CVD Change", f"{cvd.get('change', 0):,.0f}")
        
        cvd_series = cvd.get("series", [])
        if cvd_series:
            recent_cvd = cvd_series[-10:]
            cvd_df = pd.DataFrame([
                {
                    "Time": entry.get("time_iso", "")[:19],
                    "CVD": f"{entry.get('value', 0):,.0f}",
                    "Delta": f"{entry.get('delta', 0):,.0f}",
                    "Buy Volume": f"{entry.get('buy_volume', 0):,.0f}",
                    "Sell Volume": f"{entry.get('sell_volume', 0):,.0f}"
                }
                for entry in recent_cvd
            ])
            st.dataframe(cvd_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Delta Volume (Market vs Limit Orders)")
        delta = volume_analysis.get("delta", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest Delta", f"{delta.get('latest', 0):,.0f}")
        with col2:
            st.metric("Average Delta", f"{delta.get('average', 0):,.0f}")
        
        delta_series = delta.get("series", [])
        if delta_series:
            delta_df = pd.DataFrame([
                {
                    "Time": entry.get("time_iso", "")[:19],
                    "Delta": f"{entry.get('delta', 0):,.0f}",
                    "Market Orders": f"{entry.get('market_orders', 0):,.0f}",
                    "Limit Orders": f"{entry.get('limit_orders', 0):,.0f}",
                    "Imbalance Ratio": "N/A" if entry.get('imbalance_ratio') is None else f"{entry.get('imbalance_ratio', 0):.2f}"
                }
                for entry in delta_series[-10:]
            ])
            st.dataframe(delta_df, use_container_width=True, hide_index=True)
    
    with structure_tab:
        st.subheader("🏗️ Market Structure")
        market_structure = advanced.get("market_structure", {})
        
        trend = market_structure.get("trend", "neutral")
        trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "⚪"
        st.markdown(f"### Current Trend: {trend_emoji} **{trend.upper()}**")
        
        st.markdown("---")
        st.markdown("### Swing Points")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Higher Highs (HH)")
            swing_points = market_structure.get("swing_points", {})
            hh = swing_points.get("hh", [])
            if hh:
                hh_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in hh
                ])
                st.dataframe(hh_df, use_container_width=True, hide_index=True)
            else:
                st.info("No HH detected")
            
            st.markdown("#### Lower Highs (LH)")
            lh = swing_points.get("lh", [])
            if lh:
                lh_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in lh
                ])
                st.dataframe(lh_df, use_container_width=True, hide_index=True)
            else:
                st.info("No LH detected")
        
        with col2:
            st.markdown("#### Higher Lows (HL)")
            hl = swing_points.get("hl", [])
            if hl:
                hl_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in hl
                ])
                st.dataframe(hl_df, use_container_width=True, hide_index=True)
            else:
                st.info("No HL detected")
            
            st.markdown("#### Lower Lows (LL)")
            ll = swing_points.get("ll", [])
            if ll:
                ll_df = pd.DataFrame([
                    {
                        "Time": point.get("time_iso", "")[:19],
                        "Price": f"${point.get('price', 0):.4f}",
                        "Type": point.get("structure", "")
                    }
                    for point in ll
                ])
                st.dataframe(ll_df, use_container_width=True, hide_index=True)
            else:
                st.info("No LL detected")
        
        st.markdown("---")
        st.markdown("### Key Support & Resistance Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Support Levels")
            key_levels = market_structure.get("key_levels", {})
            support = key_levels.get("support", [])
            if support:
                support_df = pd.DataFrame([
                    {
                        "Price": f"${level['price']:.4f}",
                        "Strength": f"{level['strength']:.2f}"
                    }
                    for level in support
                ])
                st.dataframe(support_df, use_container_width=True, hide_index=True)
            else:
                st.info("No support levels detected")
        
        with col2:
            st.markdown("#### Resistance Levels")
            resistance = key_levels.get("resistance", [])
            if resistance:
                resistance_df = pd.DataFrame([
                    {
                        "Price": f"${level['price']:.4f}",
                        "Strength": f"{level['strength']:.2f}"
                    }
                    for level in resistance
                ])
                st.dataframe(resistance_df, use_container_width=True, hide_index=True)
            else:
                st.info("No resistance levels detected")
        
        st.markdown("---")
        st.markdown("### Liquidity Zones")
        liquidity_zones = market_structure.get("liquidity_zones", [])
        if liquidity_zones:
            liq_df = pd.DataFrame([
                {
                    "Type": zone["type"].upper(),
                    "Price": f"${zone['price']:.4f}",
                    "Volume Ratio": f"{zone['volume_ratio']:.4f}"
                }
                for zone in liquidity_zones
            ])
            st.dataframe(liq_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant liquidity zones detected")
    
    with fundamentals_tab:
        st.subheader("📈 Fundamental Metrics")
        fundamentals = advanced.get("fundamentals", {})
        
        st.markdown("### Funding Rate")
        funding_rate = fundamentals.get("funding_rate", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Rate", f"{funding_rate.get('current', 0):.4%}")
        with col2:
            st.metric("Predicted Rate", f"{funding_rate.get('predicted', 0):.4%}")
        with col3:
            st.metric("Annualized Rate", f"{funding_rate.get('annualized', 0):.2f}%")
        
        st.markdown("---")
        st.markdown("### Open Interest")
        oi = fundamentals.get("open_interest", {})
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current OI", f"${oi.get('current', 0):,.0f}")
        with col2:
            st.metric("OI Change %", f"{oi.get('change_pct', 0):.2f}%")
        
        st.markdown("---")
        st.markdown("### Long/Short Ratio")
        ls_ratio = fundamentals.get("long_short_ratio", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Long Ratio", f"{ls_ratio.get('long', 0):.3f}")
        with col2:
            st.metric("Short Ratio", f"{ls_ratio.get('short', 0):.3f}")
        with col3:
            ratio_val = ls_ratio.get('ratio', 0)
            st.metric("L/S Ratio", f"{ratio_val:.2f}")
        
        st.markdown("---")
        st.markdown("### Block Trades")
        block_trades = fundamentals.get("block_trades", [])
        if block_trades:
            bt_df = pd.DataFrame([
                {
                    "Time": trade.get("time_iso", "")[:19],
                    "Price": f"${trade['price']:.4f}",
                    "Volume": f"{trade['volume']:,.0f}",
                    "Side": trade["side"].upper()
                }
                for trade in block_trades
            ])
            st.dataframe(bt_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant block trades detected")
    
    with breadth_tab:
        st.subheader("🌐 Breadth Indicators")
        advanced_data = payload.get("advanced", {})
        breadth = advanced_data.get("breadth", {})
        
        if not breadth:
            st.info("Breadth data is not available for the selected configuration.")
        else:
            st.markdown(
                "These metrics blend macro sentiment with cross-market behaviour to highlight how broad the current trend really is."
            )
            
            fear_greed = breadth.get("fear_greed_index", 50)
            regime = breadth.get("regime", "Neutral")
            note = breadth.get("note")
            source = breadth.get("source", "unavailable")
            
            sentiment_col, correlation_col, macro_col = st.columns([1.2, 1, 1])
            
            with sentiment_col:
                if fear_greed >= 70:
                    sentiment_emoji = "🟢"
                    sentiment_text = "Extreme Greed"
                elif fear_greed >= 55:
                    sentiment_emoji = "🟡"
                    sentiment_text = "Greed"
                elif fear_greed <= 25:
                    sentiment_emoji = "🔴"
                    sentiment_text = "Extreme Fear"
                elif fear_greed <= 45:
                    sentiment_emoji = "🟠"
                    sentiment_text = "Fear"
                else:
                    sentiment_emoji = "⚪"
                    sentiment_text = "Neutral"
                
                st.metric("Fear & Greed Index", f"{sentiment_emoji} {fear_greed:.1f}")
                st.caption(f"{sentiment_text} — extremes often precede mean-reversion moves.")
                st.progress(fear_greed / 100)
                st.markdown(f"**Regime:** {regime.upper()}")
                if note:
                    st.caption(note)
                st.caption(f"Source: {source}")
            
            with correlation_col:
                st.markdown("### Cross-Market Correlations")
                btc_corr = breadth.get("btc_correlation")
                sp_corr = breadth.get("sp500_correlation")
                
                if btc_corr is None:
                    st.metric("BTC Correlation", "N/A")
                else:
                    st.metric("BTC Correlation", format_correlation(btc_corr))
                st.caption("How closely the asset tracks Bitcoin's daily move.")
                
                if sp_corr is None:
                    st.metric("S&P 500 Correlation", "N/A")
                else:
                    st.metric("S&P 500 Correlation", format_correlation(sp_corr))
                st.caption("Links to traditional risk-on equities.")
            
            with macro_col:
                st.markdown("### Macro Backdrop")
                dxy = breadth.get("dollar_index_dxy")
                if dxy is None:
                    st.metric("Dollar Index (DXY)", "N/A")
                else:
                    if dxy >= 105:
                        dxy_emoji = "🔴"
                    elif dxy >= 100:
                        dxy_emoji = "🟡"
                    else:
                        dxy_emoji = "🟢"
                    st.metric("Dollar Index (DXY)", f"{dxy_emoji} {dxy:.2f}")
                st.caption("A stronger dollar often weighs on crypto risk appetite.")
                
                vix_value = breadth.get("vix_index")
                if vix_value is None:
                    st.metric("Volatility Index (VIX)", "N/A")
                else:
                    if vix_value >= 30:
                        vix_emoji = "🔴"
                    elif vix_value >= 20:
                        vix_emoji = "🟡"
                    else:
                        vix_emoji = "🟢"
                    st.metric("Volatility Index (VIX)", f"{vix_emoji} {vix_value:.2f}")
                st.caption("Elevated volatility signals stress across risk assets.")
                
                yields = breadth.get("treasury_yields", {})
                two_year = yields.get("2y")
                ten_year = yields.get("10y")
                if two_year is not None:
                    st.metric("US 2Y Yield", f"{two_year:.3f}%")
                if ten_year is not None:
                    st.metric("US 10Y Yield", f"{ten_year:.3f}%")
                if two_year is not None and ten_year is not None:
                    curve = two_year - ten_year
                    if curve >= 0:
                        curve_emoji = "🔴"
                    elif curve > -0.3:
                        curve_emoji = "🟡"
                    else:
                        curve_emoji = "🟢"
                    st.metric("Yield Curve (2Y-10Y)", f"{curve_emoji} {curve:.2f}%")
                    st.caption("An inverted curve (2Y > 10Y) signals tightening liquidity conditions.")
    
    with onchain_tab:
        st.subheader("🔗 On-chain Metrics")
        advanced_data = payload.get("advanced", {})
        onchain = advanced_data.get("onchain_metrics", {})
        exchange_flows = onchain.get("exchange_flows", {})
        
        if not exchange_flows:
            st.info("On-chain exchange flow estimates are not available for this asset/timeframe.")
        else:
            st.markdown(
                "Exchange flow estimates highlight whether capital is moving onto exchanges (accumulation) or away from them (distribution)."
            )
            
            net_flow = exchange_flows.get("net_flow", 0.0)
            inflow_usd = exchange_flows.get("inflow", 0.0)
            outflow_usd = exchange_flows.get("outflow", 0.0)
            inflow_asset = exchange_flows.get("inflow_btc", 0.0)
            outflow_asset = exchange_flows.get("outflow_btc", 0.0)
            total_turnover = abs(inflow_usd) + abs(outflow_usd)
            flow_bias = (net_flow / total_turnover * 100) if total_turnover else 0.0
            
            summary_col, bias_col = st.columns(2)
            with summary_col:
                st.metric("Net Flow (USD)", format_flow(net_flow))
                st.caption("Positive values show net accumulation; negative values indicate distribution pressure.")
            with bias_col:
                if total_turnover:
                    if flow_bias >= 5:
                        bias_emoji = "🟢"
                    elif flow_bias <= -5:
                        bias_emoji = "🔴"
                    else:
                        bias_emoji = "⚪"
                    st.metric("Flow Bias", f"{bias_emoji} {flow_bias:+.2f}%")
                else:
                    st.metric("Flow Bias", "N/A")
                st.caption("Net flow relative to total turnover over the analysed window.")
            
            usd_col1, usd_col2 = st.columns(2)
            with usd_col1:
                st.metric("Inflow (USD)", format_flow(inflow_usd))
                st.caption("Buying pressure routed through exchanges.")
            with usd_col2:
                st.metric("Outflow (USD)", format_flow(outflow_usd))
                st.caption("Selling pressure or capital leaving exchanges.")
            
            asset_col1, asset_col2 = st.columns(2)
            with asset_col1:
                st.metric("Inflow (Asset Units)", f"{inflow_asset:.4f}")
            with asset_col2:
                st.metric("Outflow (Asset Units)", f"{outflow_asset:.4f}")
            st.caption("Asset unit estimates use the average price across the last 20 analysed candles.")
    
    with composite_tab:
        st.subheader("🧩 Composite Indicators")
        advanced_data = payload.get("advanced", {})
        composite = advanced_data.get("composite_indicators", {})
        
        if not composite:
            st.info("Composite indicators are not available for this asset/timeframe.")
        else:
            st.markdown(
                "Composite scores summarise liquidity quality, market health, and the risk-adjusted trading signal."
            )
            
            liquidity_score = composite.get("liquidity_score", {})
            market_health = composite.get("market_health_index", {})
            risk_signal = composite.get("risk_adjusted_signal", {})
            
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            
            overall_liquidity = liquidity_score.get("overall")
            with overview_col1:
                if overall_liquidity is None:
                    st.metric("Liquidity Score", "N/A")
                else:
                    if overall_liquidity >= 0.7:
                        liquidity_emoji = "🟢"
                    elif overall_liquidity >= 0.4:
                        liquidity_emoji = "🟡"
                    else:
                        liquidity_emoji = "🔴"
                    st.metric("Liquidity Score", f"{liquidity_emoji} {overall_liquidity:.3f}")
                st.caption("Combines order book depth, spreads, and slippage risk.")
            
            overall_health = market_health.get("overall")
            with overview_col2:
                if overall_health is None:
                    st.metric("Market Health", "N/A")
                else:
                    if overall_health >= 0.7:
                        health_emoji = "🟢"
                    elif overall_health >= 0.4:
                        health_emoji = "🟡"
                    else:
                        health_emoji = "🔴"
                    st.metric("Market Health", f"{health_emoji} {overall_health:.3f}")
                st.caption("Balances volatility stability, volume quality, and momentum consistency.")
            
            with overview_col3:
                final_signal = risk_signal.get("final_signal", "NEUTRAL")
                if final_signal == "BUY":
                    signal_emoji = "🟢"
                elif final_signal == "SELL":
                    signal_emoji = "🔴"
                else:
                    signal_emoji = "⚪"
                st.metric("Final Signal", f"{signal_emoji} {final_signal}")
                confidence = risk_signal.get("confidence")
                if confidence is not None:
                    st.metric("Signal Confidence", f"{confidence:.3f}")
                    st.progress(confidence)
                risk_adjustment = risk_signal.get("risk_adjustment")
                if risk_adjustment is not None:
                    st.caption(f"Risk adjustment applied: {risk_adjustment:+.3f}")
                raw_signal = risk_signal.get("raw_signal")
                if raw_signal and raw_signal != final_signal:
                    st.caption(f"Raw trend signal was {raw_signal}; adjustments tempered the outcome.")
            
            st.markdown("---")
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("### Liquidity Components")
                liq_rows = []
                for key, label in (
                    ("depth_quality", "Depth Quality"),
                    ("spread_efficiency", "Spread Efficiency"),
                    ("slippage_risk", "Slippage Risk"),
                ):
                    value = liquidity_score.get(key)
                    if value is not None:
                        liq_rows.append({"Component": label, "Score": f"{value:.3f}"})
                if liq_rows:
                    st.dataframe(pd.DataFrame(liq_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No liquidity breakdown available.")
            
            with detail_col2:
                st.markdown("### Market Health Components")
                health_rows = []
                for key, label in (
                    ("volatility_stability", "Volatility Stability"),
                    ("volume_quality", "Volume Quality"),
                    ("momentum_consistency", "Momentum Consistency"),
                ):
                    value = market_health.get(key)
                    if value is not None:
                        health_rows.append({"Component": label, "Score": f"{value:.3f}"})
                if health_rows:
                    st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No market health breakdown available.")
            
            risk_factors = risk_signal.get("risk_factors", [])
            if risk_factors:
                readable = [factor.replace("_", " ").title() for factor in risk_factors]
                st.markdown("### Risk Factors Considered")
                st.write(" • ".join(readable))
            else:
                st.caption("No additional risk factors flagged in this analysis.")
    
    with patterns_tab:
        st.subheader("🌊 Patterns & Waves")
        patterns = advanced.get("patterns", {})
        
        st.markdown("### Elliott Wave Analysis")
        elliott = patterns.get("elliott", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wave Count", elliott.get("wave_count", 0))
        with col2:
            st.metric("Current Wave", elliott.get("label", "Unknown"))
        with col3:
            st.metric("Structure Type", elliott.get("structure", "Unknown").upper())
        
        pivot_points = elliott.get("pivot_points", [])
        if pivot_points:
            st.markdown("#### Pivot Points")
            pivot_df = pd.DataFrame([
                {
                    "Time": point.get("time_iso", "")[:19],
                    "Price": f"${point.get('price', 0):.4f}",
                    "Type": point.get("type", "")
                }
                for point in pivot_points
            ])
            st.dataframe(pivot_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Orderbook Clusters")
        clusters = patterns.get("orderbook_clusters", [])
        if clusters:
            cluster_df = pd.DataFrame([
                {
                    "Side": cluster["side"].upper(),
                    "Price": f"${cluster['price']:.4f}",
                    "Volume": f"{cluster['volume']:,.2f}",
                    "Strength": f"{cluster['strength']:.2f}x"
                }
                for cluster in clusters
            ])
            st.dataframe(cluster_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant orderbook clusters detected")
        
        st.markdown("---")
        st.markdown("### Liquidity Anomalies")
        anomalies = patterns.get("liquidity_anomalies", [])
        if anomalies:
            anom_df = pd.DataFrame([
                {
                    "Time": anom.get("time_iso", "")[:19],
                    "Type": anom["type"].upper().replace("_", " "),
                    "Price": f"${anom['price']:.4f}",
                    "Severity": f"{anom['severity']:.2f}x",
                    "Description": anom["description"]
                }
                for anom in anomalies
            ])
            st.dataframe(anom_df, use_container_width=True, hide_index=True)
        else:
            st.info("No liquidity anomalies detected")
    
    with trade_tab:
        st.subheader("🎯 Trade Signal Calculator")
        trade_plan = advanced.get("trade_plan", {})
        signal_analysis = advanced.get("signal_analysis", {})
        
        if signal_analysis:
            st.markdown("### 📊 Historical Signal Performance")
            
            bullish_stats = signal_analysis.get("bullish", {})
            bearish_stats = signal_analysis.get("bearish", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🟢 Bullish Signals")
                bull_df = pd.DataFrame([
                    {"Metric": "Total Signals", "Value": bullish_stats.get("total_signals", 0)},
                    {"Metric": "TP1 Hit Rate", "Value": f"{bullish_stats.get('tp1_rate_pct', 0):.2f}%"},
                    {"Metric": "TP2 Hit Rate", "Value": f"{bullish_stats.get('tp2_rate_pct', 0):.2f}%"},
                    {"Metric": "TP3 Hit Rate", "Value": f"{bullish_stats.get('tp3_rate_pct', 0):.2f}%"},
                    {"Metric": "SL Hit Rate", "Value": f"{bullish_stats.get('sl_rate_pct', 0):.2f}%"},
                    {"Metric": "Win Rate", "Value": f"{bullish_stats.get('overall_win_rate_pct', 0):.2f}%"},
                    {"Metric": "Avg Bars to TP1", "Value": f"{bullish_stats.get('avg_bars_to_tp1', 0):.1f}"},
                ])
                st.dataframe(bull_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### 🔴 Bearish Signals")
                bear_df = pd.DataFrame([
                    {"Metric": "Total Signals", "Value": bearish_stats.get("total_signals", 0)},
                    {"Metric": "TP1 Hit Rate", "Value": f"{bearish_stats.get('tp1_rate_pct', 0):.2f}%"},
                    {"Metric": "TP2 Hit Rate", "Value": f"{bearish_stats.get('tp2_rate_pct', 0):.2f}%"},
                    {"Metric": "TP3 Hit Rate", "Value": f"{bearish_stats.get('tp3_rate_pct', 0):.2f}%"},
                    {"Metric": "SL Hit Rate", "Value": f"{bearish_stats.get('sl_rate_pct', 0):.2f}%"},
                    {"Metric": "Win Rate", "Value": f"{bearish_stats.get('overall_win_rate_pct', 0):.2f}%"},
                    {"Metric": "Avg Bars to TP1", "Value": f"{bearish_stats.get('avg_bars_to_tp1', 0):.1f}"},
                ])
                st.dataframe(bear_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
        
        if not trade_plan:
            st.warning("No trade plan available. Generate signals first.")
        else:
            signal = trade_plan.get("signal", {})
            risk = trade_plan.get("risk", {})
            position = trade_plan.get("position", {})
            targets = trade_plan.get("targets", [])
            
            signal_type = signal.get("type", "NEUTRAL")
            if signal_type == "BUY":
                st.success(f"### 🟢 BUY SIGNAL")
            elif signal_type == "SELL":
                st.error(f"### 🔴 SELL SIGNAL")
            else:
                st.info(f"### ⚪ NO ACTIVE SIGNAL")
            
            st.markdown("---")
            st.markdown("### 🎛️ Position Size Calculator")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                custom_position_size = st.number_input(
                    "Position Size (USD)",
                    min_value=10.0,
                    max_value=100000.0,
                    value=float(risk.get('risk_amount', 100)),
                    step=10.0,
                    key="custom_position_size"
                )
            with col2:
                custom_leverage = st.slider(
                    "Leverage",
                    min_value=1,
                    max_value=125,
                    value=int(position.get('leverage', 10)),
                    step=1,
                    key="custom_leverage"
                )
            with col3:
                commission_rate = position.get('commission_rate', 0.0006)
                st.metric("Commission Rate", f"{commission_rate * 100:.02f}%")
            
            entry_price = signal.get('entry_price', 0)
            stop_loss = risk.get('stop_loss', 0)
            atr_value = risk.get('atr', 0)
            
            if entry_price and stop_loss and atr_value:
                is_long = entry_price > stop_loss
                custom_metrics = calculate_position_metrics(
                    entry_price,
                    custom_position_size,
                    custom_leverage,
                    commission_rate
                )
                
                custom_levels = calculate_tp_sl_levels(
                    entry_price,
                    is_long,
                    atr_value
                )
                
                risk_per_unit = abs(entry_price - stop_loss)
                quantity = custom_metrics['quantity']
                max_loss = risk_per_unit * quantity + custom_metrics['entry_commission']
                
                st.markdown("---")
                st.markdown("### Entry & Risk Parameters")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Entry Price", f"${entry_price:.4f}")
                    st.metric("Stop Loss", f"${custom_levels['sl']:.4f}")
                with col2:
                    st.metric("ATR Value", f"${atr_value:.4f}")
                    st.metric("Risk Amount", f"${custom_position_size:.2f}")
                with col3:
                    st.metric("Max Loss", f"${max_loss:.2f}")
                    st.metric("Quantity", f"{quantity:.4f}")
                
                st.markdown("---")
                st.markdown(f"### Position Details ({custom_leverage}x Leverage)")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Position Size", f"${custom_position_size:.2f}")
                with col2:
                    st.metric("Notional Value", f"${custom_metrics['notional_value']:.2f}")
                with col3:
                    st.metric("Est. Commission", f"${custom_metrics['entry_commission']:.2f}")
                
                st.markdown("---")
                st.markdown("### Take Profit Targets (ATR-based)")
                
                custom_targets = []
                for tp_key in ['tp1', 'tp2', 'tp3']:
                    tp_price = custom_levels.get(tp_key)
                    if tp_price:
                        gross = abs(tp_price - entry_price) * quantity
                        net = gross - custom_metrics['entry_commission'] * 2
                        custom_targets.append({
                            "Target": tp_key.upper(),
                            "Price": f"${tp_price:.4f}",
                            "Gross P&L": f"${gross:.2f}",
                            "Net P&L": f"${net:.2f}",
                            "R:R": f"{(gross / max_loss):.2f}x" if max_loss else "N/A"
                        })
                
                if custom_targets:
                    custom_targets_df = pd.DataFrame(custom_targets)
                    st.dataframe(custom_targets_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.info("⚠️ **Disclaimer:** This is a calculated trade plan based on ATR channels. Always manage your risk and use proper position sizing.")
    
    with automated_signals_tab:
        st.subheader("🤖 Automated Trading Signals")

        state: Dict[str, Any] = st.session_state.setdefault(AUTOMATED_SIGNALS_STATE_KEY, {})

        default_end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        default_start = default_end - timedelta(hours=72)

        inputs_defaults = state.get("inputs")
        symbol_default = "BTCUSDT"
        timeframe_default = "1h"
        if inputs_defaults:
            symbol_default = inputs_defaults.get("symbol", symbol_default)
            timeframe_default = inputs_defaults.get("timeframe", timeframe_default)
            try:
                default_start = datetime.fromisoformat(
                    inputs_defaults.get("start", default_start.isoformat())
                )
                default_end = datetime.fromisoformat(
                    inputs_defaults.get("end", default_end.isoformat())
                )
            except (TypeError, ValueError):
                default_start = default_end - timedelta(hours=72)

        if default_start >= default_end:
            default_start = default_end - timedelta(hours=72)

        col_symbol, col_timeframe = st.columns([2, 1])
        with col_symbol:
            symbol_input = st.text_input(
                "Symbol (Binance spot)",
                value=symbol_default,
                help="Enter the Binance spot symbol, e.g. BTCUSDT",
                key=ui_key("automated_signals", "symbol"),
            )
        with col_timeframe:
            try:
                timeframe_index = AUTOMATED_SIGNALS_TIMEFRAMES.index(timeframe_default)
            except ValueError:
                timeframe_index = AUTOMATED_SIGNALS_TIMEFRAMES.index("1h")
            timeframe_selection = st.selectbox(
                "Timeframe",
                AUTOMATED_SIGNALS_TIMEFRAMES,
                index=timeframe_index,
                key=ui_key("automated_signals", "timeframe"),
            )

        start_col, end_col = st.columns(2)
        with start_col:
            start_date = st.date_input(
                "Start date (UTC)",
                value=default_start.date(),
                key=ui_key("automated_signals", "start_date"),
            )
            start_time = st.time_input(
                "Start time (UTC)",
                value=default_start.time().replace(microsecond=0),
                key=ui_key("automated_signals", "start_time"),
            )
        with end_col:
            end_date = st.date_input(
                "End date (UTC)",
                value=default_end.date(),
                key=ui_key("automated_signals", "end_date"),
            )
            end_time = st.time_input(
                "End time (UTC)",
                value=default_end.time().replace(microsecond=0),
                key=ui_key("automated_signals", "end_time"),
            )

        start_dt = datetime.combine(start_date, start_time, tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, end_time, tzinfo=timezone.utc)

        symbol_clean = symbol_input.strip().upper()
        try:
            tf_minutes = Timeframe.to_minutes(timeframe_selection)
        except ValueError:
            tf_minutes = 0

        duration_minutes = max((end_dt - start_dt).total_seconds() / 60, 0.0)
        estimated_bars = duration_minutes / tf_minutes if tf_minutes else 0.0

        if start_dt >= end_dt:
            st.warning("Start time must be before end time.")
        if estimated_bars < 30:
            st.info("Select a range with at least 30 completed candles to generate signals.")

        current_inputs_snapshot = {
            "symbol": symbol_clean,
            "timeframe": timeframe_selection,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        }
        if "inputs" not in state:
            state["inputs"] = current_inputs_snapshot

        inputs_valid = bool(symbol_clean) and start_dt < end_dt and estimated_bars >= 30

        run_button = st.button(
            "🚀 Run with Binance Data",
            type="primary",
            use_container_width=True,
            disabled=not inputs_valid,
        )

        if run_button:
            previous_inputs = state.get("inputs")
            if previous_inputs and previous_inputs != current_inputs_snapshot:
                cached_run_automated_signals.clear()
            try:
                with st.spinner(
                    f"Fetching Binance candles for {symbol_clean} on {timeframe_selection} timeframe..."
                ):
                    result_dict = cached_run_automated_signals(
                        symbol_clean,
                        timeframe_selection,
                        current_inputs_snapshot["start"],
                        current_inputs_snapshot["end"],
                    )
                if not is_valid_signal_structure(result_dict["explicit_signal"]):
                    raise ValueError("Generated signal does not match required schema")
                state["inputs"] = current_inputs_snapshot
                state["result"] = result_dict
                state["error"] = None
            except DataValidationError as exc:
                state["inputs"] = current_inputs_snapshot
                state["result"] = None
                state["error"] = f"Data validation failed: {exc}"
            except Exception as exc:
                state["inputs"] = current_inputs_snapshot
                state["result"] = None
                state["error"] = str(exc)

        error_message = state.get("error")
        result = state.get("result")

        if error_message:
            st.error(error_message)
        elif result:
            inputs_used = state.get("inputs", current_inputs_snapshot)
            candles = result.get("candles", [])
            try:
                result_start = datetime.fromisoformat(
                    inputs_used.get("start", current_inputs_snapshot["start"])
                )
                result_end = datetime.fromisoformat(
                    inputs_used.get("end", current_inputs_snapshot["end"])
                )
            except (TypeError, ValueError):
                result_start = start_dt
                result_end = end_dt

            st.caption(
                f"Source: Binance | {inputs_used.get('symbol', symbol_clean)} "
                f"{inputs_used.get('timeframe', timeframe_selection)} | {len(candles)} candles "
                f"from {result_start.strftime('%Y-%m-%d %H:%M UTC')} to {result_end.strftime('%Y-%m-%d %H:%M UTC')}"
            )

            signal_data = result["explicit_signal"]
            processed_signal = result["processed_payload"]

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                signal_type = signal_data.get("signal", "HOLD")
                if signal_type == "BUY":
                    st.success("## 🟢 BUY SIGNAL")
                elif signal_type == "SELL":
                    st.error("## 🔴 SELL SIGNAL")
                else:
                    st.info("## ⚪ HOLD")

            with col2:
                confidence = signal_data.get("confidence", 5)
                if confidence >= 8:
                    st.metric("Confidence", f"{confidence}/10", "🟢 High")
                elif confidence >= 5:
                    st.metric("Confidence", f"{confidence}/10", "🟡 Medium")
                else:
                    st.metric("Confidence", f"{confidence}/10", "⚪ Low")

            with col3:
                timeframe_value = str(
                    signal_data.get(
                        "timeframe", inputs_used.get("timeframe", timeframe_selection)
                    )
                ).upper()
                holding_period = str(signal_data.get("holding_period", "medium")).title()
                st.metric("Timeframe", timeframe_value)
                st.metric("Holding Period", holding_period)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 Entry & Exit Levels")

                entries = signal_data.get("entries", [])
                if entries:
                    st.write("**Entry Levels:**")
                    for i, entry in enumerate(entries[:3], 1):
                        st.write(f"  Entry {i}: ${entry:.4f}")

                stop_loss = signal_data.get("stop_loss")
                if stop_loss:
                    st.write(f"**Stop Loss:** ${stop_loss:.4f}")

                take_profits = signal_data.get("take_profits", {})
                if take_profits:
                    st.write("**Take Profits:**")
                    for tp_key, tp_price in take_profits.items():
                        st.write(f"  {tp_key.upper()}: ${tp_price:.4f}")

            with col2:
                st.markdown("### 📊 Position & Risk")

                position_size = signal_data.get("position_size_pct")
                if position_size:
                    st.write(f"**Position Size:** {position_size:.1f}%")

                weights = signal_data.get("weights", {})
                if weights:
                    st.write("**Component Weights:**")
                    for component, weight in weights.items():
                        st.write(f"  {component.title()}: {weight:.2f}")

            st.markdown("---")

            rationale = signal_data.get("rationale", [])
            if rationale:
                st.markdown("### 💡 Signal Rationale")
                for i, point in enumerate(rationale, 1):
                    st.write(f"{i}. {point}")

            cancel_conditions = signal_data.get("cancel_conditions", [])
            if cancel_conditions:
                st.markdown("### ⚠️ Cancel Conditions")
                for condition in cancel_conditions:
                    st.write(f"• {condition}")

            with st.expander("🔧 Processing Information", expanded=False):
                processing_info = processed_signal.get("metadata", {})
                st.write(f"**Processor:** {processing_info.get('payload_processor', 'Unknown')}")
                st.write(f"**Timeframe Used:** {processing_info.get('timeframe_used', 'Unknown')}")
                st.write(f"**Real Data Validated:** {processing_info.get('real_data_validated', False)}")
                st.write(f"**Source Data Quality:** {processing_info.get('source_data_quality', 'Unknown')}")
                st.write("**Signal Format:** Explicit JSON Schema v1.0")

            # Factor Analysis
            factors = signal_data.get("factors", [])
            if factors:
                st.markdown("### 📊 Factor Analysis")

                factors_data = []
                for factor in factors:
                    if isinstance(factor, dict):
                        factors_data.append(
                            {
                                "Factor": factor.get("factor_name", "Unknown"),
                                "Score": f"{factor.get('score', 0):.2f}",
                                "Weight": f"{factor.get('weight', 1.0):.2f}",
                                "Emoji": factor.get("emoji", "⚪"),
                                "Description": factor.get("description", ""),
                            }
                        )

                if factors_data:
                    factors_df = pd.DataFrame(factors_data)
                    st.dataframe(factors_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Position Plan
            position_plan = signal_data.get("position_plan", {})
            if position_plan:
                st.markdown("### 💼 Position Plan")

                plan_col1, plan_col2, plan_col3, plan_col4 = st.columns(4)

                entry_price = position_plan.get("entry_price", 0.0)
                with plan_col1:
                    st.metric("Entry Price", f"${entry_price:.4f}")

                position_size_usd = position_plan.get("position_size_usd", 0.0)
                with plan_col2:
                    st.metric(
                        "Position Size",
                        f"${position_size_usd:.2f}" if position_size_usd else "N/A",
                    )

                direction = position_plan.get("direction", "N/A")
                with plan_col3:
                    st.metric("Direction", direction.upper() if direction else "N/A")

                leverage = position_plan.get("leverage", 1.0)
                with plan_col4:
                    st.metric("Leverage", f"{leverage:.1f}x" if leverage else "N/A")

                st.markdown("#### TP/SL Ladder")

                ladder_col1, ladder_col2 = st.columns(2)

                with ladder_col1:
                    stop_loss = position_plan.get("stop_loss", 0.0)
                    st.write(f"**Stop Loss:** ${stop_loss:.4f}" if stop_loss else "**Stop Loss:** N/A")

                    if entry_price and stop_loss:
                        risk_distance = abs(entry_price - stop_loss)
                        risk_pct = (risk_distance / entry_price) * 100
                        st.write(f"Risk Distance: ${risk_distance:.4f} ({risk_pct:.2f}%)")

                with ladder_col2:
                    take_profit_levels = position_plan.get("take_profit_levels", [])
                    if take_profit_levels:
                        for idx, tp_level in enumerate(take_profit_levels, 1):
                            if entry_price and tp_level:
                                profit_pct = ((tp_level - entry_price) / entry_price) * 100
                                st.write(f"TP{idx}: ${tp_level:.4f} ({profit_pct:+.2f}%)")
                    else:
                        st.write("No TP levels defined")

                if position_plan.get("risk_reward_ratio"):
                    rrr = position_plan.get("risk_reward_ratio")
                    st.markdown(f"**Risk/Reward Ratio:** {rrr:.2f}:1")

                if position_plan.get("max_risk_pct"):
                    max_risk = position_plan.get("max_risk_pct")
                    st.markdown(f"**Max Risk %:** {max_risk * 100:.2f}%")

            st.markdown("---")

            if signal_data.get("holding_horizon_bars"):
                holding_horizon = signal_data.get("holding_horizon_bars")
                st.markdown("### ⏱️ Holding Horizon")
                st.info(f"**Estimated Holding Period:** {holding_horizon} bars")

            explanation = signal_data.get("explanation", {})
            if explanation:
                st.markdown("### 📝 Signal Rationale")

                primary_reason = explanation.get("primary_reason", "")
                if primary_reason:
                    st.markdown(f"**Primary Reason:** {primary_reason}")

                supporting_factors = explanation.get("supporting_factors", [])
                if supporting_factors:
                    st.markdown("**Supporting Factors:**")
                    for factor in supporting_factors:
                        st.write(f"• {factor}")

                risk_factors = explanation.get("risk_factors", [])
                if risk_factors:
                    st.markdown("**Risk Factors:**")
                    for risk in risk_factors:
                        st.write(f"⚠️ {risk}")

                market_context = explanation.get("market_context", "")
                if market_context:
                    st.markdown(f"**Market Context:** {market_context}")

            st.markdown("---")

            if signal_data.get("cancellation_reasons"):
                cancellation_reasons = signal_data.get("cancellation_reasons", [])
                st.warning("### ⛔ Signal Rejection Reasons")
                for reason in cancellation_reasons:
                    st.write(f"• {reason}")

            st.markdown("---")

            optimization_stats = signal_data.get("optimization_stats", {})
            if optimization_stats:
                st.markdown("### 📈 Performance Metrics")

                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

                with perf_col1:
                    win_rate = optimization_stats.get("backtest_win_rate")
                    if win_rate is not None:
                        st.metric("Win Rate", f"{win_rate:.1f}%")

                with perf_col2:
                    profit_factor = optimization_stats.get("profit_factor")
                    if profit_factor is not None:
                        st.metric("Profit Factor", f"{profit_factor:.2f}")

                with perf_col3:
                    sharpe_ratio = optimization_stats.get("sharpe_ratio")
                    if sharpe_ratio is not None:
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                with perf_col4:
                    total_signals = optimization_stats.get("total_signals", 0)
                    st.metric("Total Signals", total_signals)

                perf_extra_col1, perf_extra_col2, perf_extra_col3 = st.columns(3)

                with perf_extra_col1:
                    avg_profit = optimization_stats.get("avg_profit_pct")
                    if avg_profit is not None:
                        st.metric("Avg Profit %", f"{avg_profit:.2f}%")

                with perf_extra_col2:
                    avg_loss = optimization_stats.get("avg_loss_pct")
                    if avg_loss is not None:
                        st.metric("Avg Loss %", f"{avg_loss:.2f}%")

                with perf_extra_col3:
                    profitable = optimization_stats.get("profitable_signals", 0)
                    losing = optimization_stats.get("losing_signals", 0)
                    st.metric("Profitable Signals", f"{profitable}/{profitable + losing}")

            st.markdown("---")
            st.info(
                "💡 **Note:** This automated signals tab generates trading system analysis using real Binance data. "
                "Ensure your trading system exports signals in the expected JSON format for full functionality."
            )
        else:
            st.info("Configure symbol, timeframe, and date range to run automated signals with real Binance data.")
    
    with backtest_tab:
        st.subheader("🔬 Backtesting Engine")
        
        # Import backtesting components
        try:
            from indicator_collector.trading_system import (
                Backtester, BacktestConfig, ParameterSet, 
                PerformanceKPIs, BacktestResult
            )
            
            st.markdown("""
            ### 🎯 Backtesting Configuration
            
            Configure and run backtests on historical data to validate trading strategies
            and optimize parameters for target performance metrics.
            """)
            
            # Backtest configuration
            with st.expander("⚙️ Backtest Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    lookback_days = st.number_input(
                        "Lookback Period (days)", 
                        min_value=30, 
                        max_value=3650, 
                        value=730,
                        help="Number of days of historical data to use",
                        key=ui_key("backtest_tab", "lookback_days")
                    )
                    split_method = st.selectbox(
                        "Data Split Method",
                        ["walk_forward", "time_split", "k_fold"],
                        index=0,
                        help="Method for splitting training and test data",
                        key=ui_key("backtest_tab", "split_method")
                    )
                    train_ratio = st.slider(
                        "Training Ratio", 
                        min_value=0.5, 
                        max_value=0.9, 
                        value=0.7,
                        help="Proportion of data used for training",
                        key=ui_key("backtest_tab", "train_ratio")
                    )
                
                with col2:
                    target_win_rate = st.slider(
                        "Target Win Rate", 
                        min_value=0.3, 
                        max_value=0.9, 
                        value=0.55,
                        help="Target win rate for optimization",
                        key=ui_key("backtest_tab", "target_win_rate")
                    )
                    target_profit_factor = st.slider(
                        "Target Profit Factor", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=1.5,
                        help="Target profit factor (profit/loss ratio)",
                        key=ui_key("backtest_tab", "target_profit_factor")
                    )
                    target_sharpe = st.slider(
                        "Target Sharpe Ratio", 
                        min_value=0.5, 
                        max_value=3.0, 
                        value=1.0,
                        help="Target Sharpe ratio for risk-adjusted returns",
                        key=ui_key("backtest_tab", "target_sharpe")
                    )
            
            # Parameter configuration
            with st.expander("🎛️ Parameter Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Signal Weights**")
                    technical_weight = st.slider("Technical Analysis", 0.0, 1.0, 0.4, 0.05, key=ui_key("backtest_tab", "technical_weight"))
                    volume_weight = st.slider("Volume Analysis", 0.0, 1.0, 0.3, 0.05, key=ui_key("backtest_tab", "volume_weight"))
                    sentiment_weight = st.slider("Sentiment Analysis", 0.0, 1.0, 0.2, 0.05, key=ui_key("backtest_tab", "sentiment_weight"))
                    structure_weight = st.slider("Market Structure", 0.0, 1.0, 0.1, 0.05, key=ui_key("backtest_tab", "structure_weight"))
                
                with col2:
                    st.markdown("**Risk Parameters**")
                    stop_loss_pct = st.slider("Stop Loss pct", 0.5, 5.0, 2.0, 0.5, key=ui_key("backtest_tab", "stop_loss_pct"))
                    take_profit_pct = st.slider("Take Profit pct", 1.0, 10.0, 4.0, 0.5, key=ui_key("backtest_tab", "take_profit_pct"))
                    max_position_size = st.slider("Max Position Size pct", 0.01, 0.2, 0.05, 0.01, key=ui_key("backtest_tab", "max_position_size"))
                    confirmation_threshold = st.slider("Confirmation Threshold", 0.3, 0.9, 0.6, 0.1, key=ui_key("backtest_tab", "confirmation_threshold"))
            
            # Optimization settings
            with st.expander("🔍 Optimization Settings"):
                col1, col2 = st.columns(2)
                
                with col1:
                    search_method = st.selectbox(
                        "Search Method",
                        ["grid", "random"],
                        index=0,
                        help="Method for parameter optimization",
                        key=ui_key("backtest_tab", "search_method")
                    )
                    max_iterations = st.number_input(
                        "Max Iterations",
                        min_value=5,
                        max_value=200,
                        value=20,
                        help="Maximum number of optimization iterations",
                        key=ui_key("backtest_tab", "max_iterations")
                    )
                
                with col2:
                    validate_real_data = st.checkbox(
                        "Validate Real Data Only",
                        value=True,
                        help="Only use validated real market data",
                        key=ui_key("backtest_tab", "validate_real_data")
                    )
                    min_confirmations = st.number_input(
                        "Min Confirmation Categories",
                        min_value=1,
                        max_value=5,
                        value=3,
                        help="Minimum number of confirming factors required",
                        key=ui_key("backtest_tab", "min_confirmations")
                    )
            
            # Run backtest button
            run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
            
            if run_backtest:
                with st.spinner("Running backtest..."):
                    try:
                        # Create backtest configuration
                        config = BacktestConfig(
                            lookback_days=int(lookback_days),
                            split_method=split_method,
                            train_ratio=train_ratio,
                            target_win_rate=target_win_rate,
                            target_profit_factor=target_profit_factor,
                            target_sharpe=target_sharpe,
                            search_method=search_method,
                            max_iterations=int(max_iterations),
                            validate_real_data=validate_real_data,
                            min_confirmation_categories=int(min_confirmations),
                        )
                        
                        # Create parameter set
                        total_weight = technical_weight + volume_weight + sentiment_weight + structure_weight
                        if total_weight > 0:
                            weights = {
                                "technical": technical_weight / total_weight,
                                "volume": volume_weight / total_weight,
                                "sentiment": sentiment_weight / total_weight,
                                "market_structure": structure_weight / total_weight,
                            }
                        else:
                            weights = {"technical": 0.25, "volume": 0.25, "sentiment": 0.25, "market_structure": 0.25}
                        
                        indicator_params = indicator_defaults_for(selected_timeframe)
                        params = ParameterSet(
                            weights=weights,
                            indicator_params=indicator_params,
                            timeframe=selected_timeframe,
                            stop_loss_pct=stop_loss_pct,
                            take_profit_pct=take_profit_pct,
                            max_position_size_pct=max_position_size,
                            confirmation_threshold=confirmation_threshold,
                        )
                        
                        # Create backtester and run
                        backtester = Backtester(config)
                        
                        # Create sample historical data (in real implementation, this would load from file/database)
                        st.info("📊 Using sample historical data for demonstration. In production, this would load actual historical signals.")
                        
                        # Generate sample data
                        from datetime import datetime, timedelta
                        import random
                        
                        sample_payloads = []
                        base_timestamp = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
                        
                        for i in range(min(500, int(lookback_days * 0.7))):
                            signal_type = random.choice(["BUY", "SELL", "NEUTRAL"])
                            price = 50000 + random.uniform(-5000, 5000)
                            
                            sample_payload = {
                                "timestamp": base_timestamp + i * 86400000,  # Daily
                                "signal_type": signal_type,
                                "entry_price": price,
                                "factors": [
                                    {"factor_name": "technical", "score": random.uniform(0.3, 0.9)},
                                    {"factor_name": "volume", "score": random.uniform(0.3, 0.9)},
                                    {"factor_name": "sentiment", "score": random.uniform(0.3, 0.9)},
                                ] if signal_type != "NEUTRAL" else [],
                            }
                            sample_payloads.append(sample_payload)
                        
                        # Load data
                        backtester.load_historical_data(sample_payloads)
                        
                        # Run backtest
                        result = backtester.run_backtest(params)
                        
                        # Display results
                        st.success("✅ Backtest completed successfully!")
                        
                        # Performance metrics
                        st.markdown("### 📊 Performance Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Win Rate",
                                f"{result.test_kpis.win_rate:.3f}",
                                delta=f"{result.test_kpis.win_rate - target_win_rate:+.3f}",
                                delta_color="normal" if result.test_kpis.win_rate >= target_win_rate else "inverse"
                            )
                        
                        with col2:
                            st.metric(
                                "Profit Factor",
                                f"{result.test_kpis.profit_factor:.3f}",
                                delta=f"{result.test_kpis.profit_factor - target_profit_factor:+.3f}",
                                delta_color="normal" if result.test_kpis.profit_factor >= target_profit_factor else "inverse"
                            )
                        
                        with col3:
                            st.metric(
                                "Sharpe Ratio",
                                f"{result.test_kpis.sharpe_ratio:.3f}",
                                delta=f"{result.test_kpis.sharpe_ratio - target_sharpe:+.3f}",
                                delta_color="normal" if result.test_kpis.sharpe_ratio >= target_sharpe else "inverse"
                            )
                        
                        with col4:
                            st.metric(
                                "Max Drawdown",
                                f"{result.test_kpis.max_drawdown_pct:.3f}",
                                delta=None,
                                delta_color="normal"
                            )
                        
                        # Detailed metrics
                        st.markdown("### 📈 Detailed Metrics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Training Performance**")
                            st.write(f"• Total Signals: {result.train_kpis.total_signals}")
                            st.write(f"• Profitable: {result.train_kpis.profitable_signals}")
                            st.write(f"• Win Rate: {result.train_kpis.win_rate:.3f}")
                            st.write(f"• Profit Factor: {result.train_kpis.profit_factor:.3f}")
                            st.write(f"• Avg Profit: {result.train_kpis.avg_profit_pct:.3f}%")
                            st.write(f"• Avg Loss: {result.train_kpis.avg_loss_pct:.3f}%")
                        
                        with col2:
                            st.markdown("**Test Performance**")
                            st.write(f"• Total Signals: {result.test_kpis.total_signals}")
                            st.write(f"• Profitable: {result.test_kpis.profitable_signals}")
                            st.write(f"• Win Rate: {result.test_kpis.win_rate:.3f}")
                            st.write(f"• Profit Factor: {result.test_kpis.profit_factor:.3f}")
                            st.write(f"• Avg Profit: {result.test_kpis.avg_profit_pct:.3f}%")
                            st.write(f"• Avg Loss: {result.test_kpis.avg_loss_pct:.3f}%")
                        
                        # Target achievement
                        st.markdown("### 🎯 Target Achievement")
                        targets_met = (
                            result.test_kpis.win_rate >= target_win_rate and
                            result.test_kpis.profit_factor >= target_profit_factor and
                            result.test_kpis.sharpe_ratio >= target_sharpe and
                            result.test_kpis.max_drawdown_pct <= 0.25
                        )
                        
                        if targets_met:
                            st.success("🎉 All targets achieved!")
                        else:
                            st.warning("⚠️ Some targets not met. Consider parameter optimization.")
                        
                        # Optimization score
                        st.markdown("### 📊 Optimization Score")
                        st.progress(result.optimization_score)
                        st.write(f"Score: {result.optimization_score:.4f}")
                        
                        # Execution info
                        st.markdown("### ℹ️ Execution Info")
                        st.write(f"• Execution Time: {result.execution_time_seconds:.2f} seconds")
                        st.write(f"• Data Points Used: {len(sample_payloads)}")
                        st.write(f"• Split Method: {split_method}")
                        st.write(f"• Search Method: {search_method}")
                        
                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except ImportError as e:
            st.error(f"❌ Backtesting components not available: {str(e)}")
            st.info("Please ensure the trading system backtesting modules are properly installed.")
    
    with adaptive_tab:
        st.subheader("⚖️ Adaptive Weight Management")
        
        try:
            from indicator_collector.trading_system import (
                AdaptiveWeightManager, AdaptiveWeightConfig,
                Backtester, ParameterSet
            )
            
            st.markdown("""
            ### 🧠 Adaptive Weight System
            
            Automatically adjust signal weights based on rolling performance metrics
            to optimize trading system performance over time.
            """)
            
            # Adaptive weight configuration
            with st.expander("⚙️ Adaptive Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    rolling_window = st.number_input(
                        "Rolling Window (days)",
                        min_value=7,
                        max_value=90,
                        value=30,
                        help="Number of days to consider for performance tracking",
                        key=ui_key("adaptive_tab", "rolling_window")
                    )
                    min_signals = st.number_input(
                        "Min Signals for Adaptation",
                        min_value=10,
                        max_value=200,
                        value=50,
                        help="Minimum signals required before adapting weights",
                        key=ui_key("adaptive_tab", "min_signals")
                    )
                    adaptation_strategy = st.selectbox(
                        "Adaptation Strategy",
                        ["performance_based", "volatility_adjusted", "hybrid"],
                        index=2,
                        help="Method for calculating weight adjustments",
                        key=ui_key("adaptive_tab", "adaptation_strategy")
                    )
                
                with col2:
                    target_win_rate = st.slider(
                        "Target Win Rate",
                        min_value=0.3,
                        max_value=0.9,
                        value=0.55,
                        help="Target win rate for adaptation triggers",
                        key=ui_key("adaptive_tab", "target_win_rate")
                    )
                    target_profit_factor = st.slider(
                        "Target Profit Factor",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        help="Target profit factor for adaptation triggers",
                        key=ui_key("adaptive_tab", "target_profit_factor")
                    )
                    adaptation_threshold = st.slider(
                        "Adaptation Threshold",
                        min_value=0.01,
                        max_value=0.2,
                        value=0.05,
                        help="Minimum performance improvement to trigger adaptation",
                        key=ui_key("adaptive_tab", "adaptation_threshold")
                    )
            
            # Current weights display
            with st.expander("📊 Current Weights", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Initial Weights**")
                    tech_weight = st.slider("Technical", 0.0, 1.0, 0.4, 0.05, key=ui_key("adaptive_tab", "tech_weight"))
                    vol_weight = st.slider("Volume", 0.0, 1.0, 0.3, 0.05, key=ui_key("adaptive_tab", "vol_weight"))
                    sent_weight = st.slider("Sentiment", 0.0, 1.0, 0.2, 0.05, key=ui_key("adaptive_tab", "sent_weight"))
                    struct_weight = st.slider("Market Structure", 0.0, 1.0, 0.1, 0.05, key=ui_key("adaptive_tab", "struct_weight"))
                
                with col2:
                    st.markdown("**Weight Constraints**")
                    min_weight = st.slider("Min Weight per Factor", 0.01, 0.2, 0.05, 0.01, key=ui_key("adaptive_tab", "min_weight"))
                    max_weight = st.slider("Max Weight per Factor", 0.2, 0.8, 0.5, 0.05, key=ui_key("adaptive_tab", "max_weight"))
                    max_change = st.slider("Max Change pct", 0.1, 0.5, 0.3, 0.05, key=ui_key("adaptive_tab", "max_change"))
            
            # Run adaptation
            run_adaptation = st.button("🔄 Run Adaptive Analysis", type="primary", use_container_width=True)
            
            if run_adaptation:
                with st.spinner("Running adaptive weight analysis..."):
                    try:
                        # Create adaptive weight manager
                        config = AdaptiveWeightConfig(
                            rolling_window_days=int(rolling_window),
                            min_signals_for_adaptation=int(min_signals),
                            adaptation_strategy=adaptation_strategy,
                            target_win_rate=target_win_rate,
                            target_profit_factor=target_profit_factor,
                            adaptation_threshold=adaptation_threshold,
                            min_weight_per_factor=min_weight,
                            max_weight_per_factor=max_weight,
                            max_weight_change_pct=max_change,
                        )
                        
                        manager = AdaptiveWeightManager(config)
                        
                        # Initialize weights
                        total_weight = tech_weight + vol_weight + sent_weight + struct_weight
                        if total_weight > 0:
                            initial_weights = {
                                "technical": tech_weight / total_weight,
                                "volume": vol_weight / total_weight,
                                "sentiment": sent_weight / total_weight,
                                "market_structure": struct_weight / total_weight,
                            }
                        else:
                            initial_weights = {"technical": 0.25, "volume": 0.25, "sentiment": 0.25, "market_structure": 0.25}
                        
                        manager.initialize_weights(initial_weights)
                        
                        # Create sample signal outcomes (in real implementation, this would be historical data)
                        import random
                        from datetime import datetime, timedelta
                        
                        outcomes = []
                        base_timestamp = int((datetime.now() - timedelta(days=rolling_window)).timestamp() * 1000)
                        
                        for i in range(int(min_signals * 2)):  # Generate more than minimum
                            success = random.random() > 0.4  # 60% win rate
                            pnl = random.uniform(1.0, 5.0) if success else random.uniform(-3.0, -0.5)
                            
                            outcome = {
                                "signal_type": random.choice(["BUY", "SELL"]),
                                "entry_price": 50000 + random.uniform(-5000, 5000),
                                "exit_price": None,
                                "entry_timestamp": base_timestamp + i * 86400000,
                                "exit_timestamp": base_timestamp + i * 86400000 + 86400000,
                                "pnl_pct": pnl,
                                "holding_bars": random.randint(1, 100),
                                "success": success,
                                "factors": [
                                    {"factor_name": "technical", "score": random.uniform(0.3, 0.9)},
                                    {"factor_name": "volume", "score": random.uniform(0.3, 0.9)},
                                    {"factor_name": "sentiment", "score": random.uniform(0.3, 0.9)},
                                ],
                            }
                            outcomes.append(outcome)
                        
                        # Update manager with outcomes
                        from indicator_collector.trading_system.statistics_optimizer import SignalOutcome
                        signal_outcomes = []
                        
                        for outcome_data in outcomes:
                            outcome = SignalOutcome(
                                signal_type=outcome_data["signal_type"],
                                entry_price=outcome_data["entry_price"],
                                exit_price=outcome_data["exit_price"],
                                entry_timestamp=outcome_data["entry_timestamp"],
                                exit_timestamp=outcome_data["exit_timestamp"],
                                pnl_pct=outcome_data["pnl_pct"],
                                holding_bars=outcome_data["holding_bars"],
                                success=outcome_data["success"],
                                factors=outcome_data["factors"],
                            )
                            signal_outcomes.append(outcome)
                        
                        manager.update_signal_outcomes(signal_outcomes)
                        
                        # Check if adaptation should be performed
                        should_adapt, reason = manager.should_adapt()
                        
                        st.info(f"📊 Adaptation Analysis: {reason}")
                        
                        if should_adapt:
                            # Perform adaptation
                            adaptation_report = manager.adapt_weights()
                            
                            st.success("✅ Weight adaptation completed!")
                            
                            # Display adaptation results
                            st.markdown("### 🔄 Adaptation Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Adaptation Summary**")
                                st.write(f"• Reason: {adaptation_report.adaptation_reason}")
                                st.write(f"• Confidence: {adaptation_report.confidence_score:.3f}")
                                st.write(f"• Expected Improvement: {adaptation_report.expected_improvement:.4f}")
                                st.write(f"• Factors Adjusted: {', '.join(adaptation_report.factors_adjusted)}")
                            
                            with col2:
                                st.markdown("**Performance Before**")
                                before = adaptation_report.performance_before
                                st.write(f"• Win Rate: {before.win_rate:.3f}")
                                st.write(f"• Profit Factor: {before.profit_factor:.3f}")
                                st.write(f"• Sharpe Ratio: {before.sharpe_ratio:.3f}")
                                st.write(f"• Max Drawdown: {before.max_drawdown_pct:.3f}")
                            
                            # Weight changes
                            st.markdown("### 📊 Weight Changes")
                            
                            weight_changes = []
                            for factor in adaptation_report.factors_adjusted:
                                old_weight = adaptation_report.original_weights.get(factor, 0)
                                new_weight = adaptation_report.new_weights.get(factor, 0)
                                change = new_weight - old_weight
                                weight_changes.append({
                                    "Factor": factor.replace("_", " ").title(),
                                    "Before": f"{old_weight:.3f}",
                                    "After": f"{new_weight:.3f}",
                                    "Change": f"{change:+.3f}",
                                    "Change %": f"{change/old_weight*100:+.1f}%" if old_weight > 0 else "N/A"
                                })
                            
                            if weight_changes:
                                weight_df = pd.DataFrame(weight_changes)
                                st.dataframe(weight_df, use_container_width=True, hide_index=True)
                            
                            # Performance report
                            st.markdown("### 📈 Performance Report")
                            report = manager.generate_performance_report()
                            
                            # Summary metrics
                            summary = report["summary"]
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Signals", summary["total_signals_analyzed"])
                            with col2:
                                st.metric("Total Adaptations", summary["total_adaptations"])
                            with col3:
                                current_wr = summary["recent_kpis"]["win_rate"]
                                st.metric("Current Win Rate", f"{current_wr:.3f}")
                            with col4:
                                current_pf = summary["recent_kpis"]["profit_factor"]
                                st.metric("Current Profit Factor", f"{current_pf:.3f}")
                            
                            # Performance vs targets
                            st.markdown("**Performance vs Targets**")
                            perf_vs_targets = report["performance_vs_targets"]
                            
                            for metric, data in perf_vs_targets.items():
                                current = data["current"]
                                target = data["target"]
                                gap = data["gap"]
                                status = "✅" if gap >= 0 else "❌"
                                
                                st.write(f"{status} **{metric.replace('_', ' ').title()}:** {current:.3f} (target: {target:.3f})")
                            
                            # Recommendations
                            st.markdown("### 💡 Recommendations")
                            recommendations = report["recommendations"]
                            
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")
                            
                        else:
                            st.info("ℹ️ No adaptation needed at this time.")
                            
                            # Display current performance
                            st.markdown("### 📊 Current Performance")
                            current_kpis = manager._calculate_recent_kpis()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Win Rate", f"{current_kpis.win_rate:.3f}")
                            with col2:
                                st.metric("Profit Factor", f"{current_kpis.profit_factor:.3f}")
                            with col3:
                                st.metric("Sharpe Ratio", f"{current_kpis.sharpe_ratio:.3f}")
                            with col4:
                                st.metric("Max Drawdown", f"{current_kpis.max_drawdown_pct:.3f}")
                        
                        # Display weight performance
                        st.markdown("### 📊 Factor Performance")
                        weight_performance = manager.get_weight_performance()
                        
                        perf_data = []
                        for factor_name, perf in weight_performance.items():
                            perf_data.append({
                                "Factor": factor_name.replace("_", " ").title(),
                                "Current Weight": f"{perf.current_weight:.3f}",
                                "Win Rate": f"{perf.rolling_win_rate:.3f}",
                                "Profit Factor": f"{perf.rolling_profit_factor:.3f}",
                                "Sharpe": f"{perf.rolling_sharpe:.3f}",
                                "Consistency": f"{perf.consistency_score:.3f}",
                                "Adaptations": perf.adaptation_count,
                            })
                        
                        if perf_data:
                            perf_df = pd.DataFrame(perf_data)
                            st.dataframe(perf_df, use_container_width=True, hide_index=True)
                        
                    except Exception as e:
                        st.error(f"❌ Adaptive analysis failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        except ImportError as e:
            st.error(f"❌ Adaptive weight components not available: {str(e)}")
            st.info("Please ensure the adaptive weight management modules are properly installed.")
    
    with astrology_tab:
        st.subheader("🔮 Astrology & Celestial Cycles")
        astrology = payload.get("astrology", {})
        
        if not astrology:
            st.info("Astrology metrics are not available for this analysis.")
        else:
            confluence = astrology.get("confluence", {})
            moon = astrology.get("moon", {})
            mercury = astrology.get("mercury", {})
            jupiter = astrology.get("jupiter", {})
            
            st.markdown("### Overall Celestial Confluence")
            
            conf_col1, conf_col2 = st.columns([1, 2])
            
            with conf_col1:
                st.metric(
                    "Confluence Score",
                    f"{confluence.get('score', 0):.2f}",
                    delta=f"{confluence.get('signal', 'neutral').upper()}"
                )
                st.markdown(f"### {confluence.get('signal_color', '⚪')}")
            
            with conf_col2:
                recommendation = confluence.get("recommendation", "No specific recommendation available.")
                st.info(f"**Trading Recommendation:** {recommendation}")
            
            if confluence.get("factors"):
                st.markdown("#### Active Celestial Factors")
                for factor in confluence["factors"]:
                    st.write(f"• {factor}")
            
            st.markdown("---")
            
            moon_col, mercury_col = st.columns(2)
            
            with moon_col:
                st.markdown("### 🌕 Moon Cycle Analysis")
                st.metric("Current Phase", moon.get("phase_name", "Unknown"))
                st.metric("Illumination", f"{moon.get('illumination_pct', 0):.1f}%")
                
                volatility_ind = moon.get("volatility_indication", "moderate")
                if volatility_ind == "high":
                    vol_color = "🔴"
                    vol_text = "HIGH (expect increased volatility)"
                elif volatility_ind == "moderate":
                    vol_color = "🟡"
                    vol_text = "MODERATE (normal volatility expected)"
                else:
                    vol_color = "🟢"
                    vol_text = "LOW (reduced volatility expected)"
                
                st.markdown(f"**Volatility Indication:** {vol_color} {vol_text}")
                st.markdown(f"**Trading Bias:** {moon.get('trading_bias', 'neutral').title()}")
                
                st.markdown("---")
                st.markdown("**Upcoming Moon Events:**")
                st.write(f"• Full Moon in **{moon.get('days_to_full_moon', 0):.1f}** days ({moon.get('next_full_moon', 'N/A')[:10]})")
                st.write(f"• New Moon in **{moon.get('days_to_new_moon', 0):.1f}** days ({moon.get('next_new_moon', 'N/A')[:10]})")
                
                st.markdown("---")
                with st.expander("ℹ️ Moon Cycle Trading Context"):
                    st.markdown("""
                    **Full Moon & New Moon periods** often coincide with volatility peaks in crypto markets.
                    - **Full Moon**: Peak emotions, potential tops
                    - **New Moon**: Fresh starts, potential bottoms
                    - **Waxing Moon**: Growing phase, accumulation
                    - **Waning Moon**: Declining phase, distribution
                    """)
            
            with mercury_col:
                st.markdown("### ☿ Mercury Cycle (Trading Planet)")
                st.metric("Current Phase", mercury.get("phase_name", "Unknown"))
                st.metric("Cycle Position", f"{mercury.get('cycle_position_pct', 0):.1f}%")
                
                volume_ind = mercury.get("volume_indication", "moderate")
                if volume_ind == "high":
                    vol_color = "🟢"
                    vol_text = "HIGH (peak trading activity)"
                elif volume_ind == "increasing":
                    vol_color = "🟡"
                    vol_text = "INCREASING (building momentum)"
                elif volume_ind == "decreasing":
                    vol_color = "🟠"
                    vol_text = "DECREASING (slowing activity)"
                else:
                    vol_color = "🔴"
                    vol_text = "LOW (reduced trading)"
                
                st.markdown(f"**Volume Indication:** {vol_color} {vol_text}")
                st.markdown(f"**Recommendation:** {mercury.get('trading_recommendation', 'No specific recommendation')}")
                
                st.markdown("---")
                st.markdown("**Next Mercury Peak:**")
                st.write(f"• In **{mercury.get('days_to_peak_activity', 0):.1f}** days")
                st.write(f"• Date: {mercury.get('next_peak_date', 'N/A')[:10]}")
                
                st.markdown("---")
                with st.expander("ℹ️ Mercury Cycle Trading Context"):
                    st.markdown("""
                    **Mercury's 88-day cycle** correlates with trading volume patterns:
                    - **Direct Motion Peak**: Highest trading activity, good liquidity
                    - **Retrograde**: Lower activity, consolidation periods
                    - **Post-Retrograde**: Recovery, new opportunities emerging
                    """)
            
            st.markdown("---")
            
            st.markdown("### ♃ Jupiter 12-Year Cycle & Bitcoin Halvings")
            
            jup_col1, jup_col2, jup_col3 = st.columns(3)
            
            with jup_col1:
                st.markdown("#### Jupiter Cycle")
                st.metric("Phase", jupiter.get("jupiter_phase", "Unknown"))
                st.metric("Position", f"{jupiter.get('jupiter_cycle_position_pct', 0):.1f}%")
                
                correlation = jupiter.get("market_correlation", "neutral")
                if "strongly bullish" in correlation:
                    corr_emoji = "🟢🟢🟢"
                elif "bullish" in correlation:
                    corr_emoji = "🟢🟢"
                elif "bearish" in correlation:
                    corr_emoji = "🔴"
                else:
                    corr_emoji = "⚪"
                
                st.markdown(f"**Market Correlation:** {corr_emoji} {correlation.title()}")
            
            with jup_col2:
                st.markdown("#### Bitcoin Halving Cycle")
                st.metric("Current Epoch", f"#{jupiter.get('current_halving_epoch', 0)}")
                st.metric("Halving Phase", jupiter.get("halving_phase", "Unknown"))
                st.metric("Phase Progress", f"{jupiter.get('halving_cycle_position_pct', 0):.1f}%")
            
            with jup_col3:
                st.markdown("#### Timeline")
                st.metric("Days Since Halving", f"{jupiter.get('days_since_last_halving', 0):,}")
                st.metric("Days to Next", f"{jupiter.get('days_to_next_halving', 0):,}")
                st.write(f"**Next Halving:** {jupiter.get('next_halving_date', 'N/A')[:10]}")
            
            st.markdown("---")
            st.markdown(f"**Jupiter Recommendation:** {jupiter.get('recommendation', 'No specific recommendation')}")
            
            st.markdown("---")
            with st.expander("ℹ️ Jupiter & Bitcoin Halving Correlation"):
                st.markdown("""
                **Jupiter's 12-year cycle** aligns remarkably with Bitcoin's 4-year halving cycles:
                
                - **Jupiter Expansion (Year 1-6)**: Coincides with post-halving bull markets
                - **Jupiter Peak**: Often aligns with cycle tops
                - **Jupiter Contraction (Year 7-12)**: Coincides with bear markets and accumulation
                
                **Bitcoin Halving Phases:**
                - **Post-Halving Accumulation (0-12 months)**: Build positions
                - **Bull Market Phase (12-24 months)**: Major uptrend
                - **Euphoria & Distribution (24-36 months)**: Take profits
                - **Pre-Halving Bear (36-48 months)**: Accumulation opportunity
                
                This pattern has repeated across 3+ cycles, making it a useful contextual indicator.
                """)
            
            st.markdown("---")
            st.warning("⚠️ **Disclaimer:** Astrology-based analysis is provided for contextual reference only and should not be the sole basis for trading decisions. Always combine with technical analysis, fundamental research, and proper risk management.")
        
    with export_tab:
        st.subheader("💾 Export Analysis Data")
        
        st.markdown("### Current Session")
        metadata = payload.get("metadata", {})
        
        export_info_col1, export_info_col2 = st.columns(2)
        with export_info_col1:
            st.write(f"**Symbol:** {metadata.get('symbol', 'N/A')}")
            st.write(f"**Timeframe:** {metadata.get('timeframe', 'N/A')}")
            st.write(f"**Period:** {metadata.get('period', 'N/A')} bars")
        
        with export_info_col2:
            st.write(f"**Export Token:** {metadata.get('token', 'N/A')}")
            st.write(f"**Generated:** {metadata.get('generated_at', 'N/A')[:19]}")
        
        st.markdown("---")
        
        st.markdown("### Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_str = json.dumps(payload, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{selected_token.replace(':', '_')}_{selected_timeframe}_{export_token}.json",
                mime="application/json",
                use_container_width=True,
            )
        
        with col2:
            latest = payload.get("latest", {})
            csv_data = f"""Symbol,Timeframe,Period,Close,Trend_Strength,Pattern_Score,Sentiment,Structure,Confluence_Score,RSI,MACD
{metadata.get('symbol')},{metadata.get('timeframe')},{metadata.get('period')},{latest.get('close')},{latest.get('trend_strength')},{latest.get('pattern_score')},{latest.get('market_sentiment')},{latest.get('structure_state')},{latest.get('confluence_score')},{latest.get('rsi')},{latest.get('macd')}
"""
            st.download_button(
                label="📥 Download CSV (Latest)",
                data=csv_data,
                file_name=f"{selected_token.replace(':', '_')}_{selected_timeframe}_latest.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with col3:
            advanced = payload.get("advanced", {})
            advanced_rows = []
            if advanced:
                volume_analysis = advanced.get("volume_analysis", {})
                vpvr = volume_analysis.get("vpvr", {})
                advanced_rows.append(("VPVR POC", vpvr.get("poc")))
                advanced_rows.append(("Value Area High", vpvr.get("value_area", {}).get("high")))
                advanced_rows.append(("Value Area Low", vpvr.get("value_area", {}).get("low")))
                advanced_rows.append(("CVD Latest", volume_analysis.get("cvd", {}).get("latest")))
                advanced_rows.append(("CVD Change", volume_analysis.get("cvd", {}).get("change")))
                advanced_rows.append(("Delta Latest", volume_analysis.get("delta", {}).get("latest")))
                advanced_rows.append(("Delta Average", volume_analysis.get("delta", {}).get("average")))
                market_structure = advanced.get("market_structure", {})
                advanced_rows.append(("Structure Trend", market_structure.get("trend")))
                fundamentals = advanced.get("fundamentals", {})
                advanced_rows.append(("Funding Rate", fundamentals.get("funding_rate", {}).get("current")))
                advanced_rows.append(("Open Interest", fundamentals.get("open_interest", {}).get("current")))
                advanced_rows.append(("OI Change %", fundamentals.get("open_interest", {}).get("change_pct")))
                advanced_rows.append(("Long/Short Ratio", fundamentals.get("long_short_ratio", {}).get("ratio")))
                breadth = advanced.get("breadth", {})
                advanced_rows.append(("BTC Dominance", breadth.get("btc_dominance")))
                advanced_rows.append(("Fear & Greed", breadth.get("fear_greed_index")))
                trade_plan = advanced.get("trade_plan", {})
                signal = trade_plan.get("signal", {})
                risk = trade_plan.get("risk", {})
                advanced_rows.append(("Signal Type", signal.get("type")))
                advanced_rows.append(("Entry Price", signal.get("entry_price")))
                advanced_rows.append(("Stop Loss", risk.get("stop_loss")))
                advanced_rows.append(("ATR", risk.get("atr")))
            
            if advanced_rows:
                adv_csv = "Metric,Value\n" + "\n".join([
                    f"{metric},{value if value is not None else ''}"
                    for metric, value in advanced_rows
                ])
                st.download_button(
                    label="📥 Download CSV (Advanced Summary)",
                    data=adv_csv,
                    file_name=f"{selected_token.replace(':', '_')}_{selected_timeframe}_advanced.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.caption("Advanced metrics unavailable for export.")
        
        st.markdown("---")
        
        with st.expander("📄 View Full JSON Payload"):
            st.json(payload)


if __name__ == "__main__":
    main()
