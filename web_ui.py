#!/usr/bin/env python3

import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from indicator_collector.collector import collect_metrics
from indicator_collector.indicator_metrics import SimulationSummary
from indicator_collector.time_series import TimeframeSeries

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

TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]



@st.cache_data(ttl=300)
def load_indicator_data(symbol: str, timeframe: str, period: int, offline: bool, token: str) -> tuple:
    result = collect_metrics(
        symbol=symbol,
        timeframe=timeframe,
        period=period,
        token=token,
        offline=offline,
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
        token_input_mode = st.radio("Input Mode", ["Select from list", "Custom token"])
        
        if token_input_mode == "Select from list":
            selected_token = st.selectbox("Select Token", POPULAR_TOKENS, index=0)
        else:
            selected_token = st.text_input("Custom Token (e.g., BINANCE:BTCUSDT)", "BINANCE:BTCUSDT")
        
        st.subheader("Timeframe & Period")
        selected_timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=TIMEFRAMES.index("15m"))
        selected_period = st.slider("Analysis Period (bars)", min_value=50, max_value=1000, value=200, step=50)
        
        st.subheader("Data Source")
        offline_mode = st.checkbox("Offline Mode (Synthetic Data)", value=False, help="Use synthetic data instead of fetching from Binance")
        
        st.subheader("Export Options")
        export_token = st.text_input("Export Token/ID", value="export-session-001", help="Token to identify this analysis session")
        
        analyze_button = st.button("🔄 Analyze", type="primary", use_container_width=True)
    
    if analyze_button or "summary" not in st.session_state:
        with st.spinner(f"Analyzing {selected_token} on {selected_timeframe} timeframe..."):
            try:
                summary, payload, main_series = load_indicator_data(
                    selected_token,
                    selected_timeframe,
                    selected_period,
                    offline_mode,
                    export_token,
                )
                st.session_state.summary = summary
                st.session_state.payload = payload
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
    payload = st.session_state.payload
    main_series = st.session_state.main_series
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 Charts", 
        "📈 Multi-Timeframe", 
        "📋 Latest Metrics", 
        "🎯 Signals & Zones", 
        "📊 Volume Analysis",
        "🏗️ Market Structure",
        "📈 Fundamentals",
        "🌐 Breadth Indicators",
        "🌊 Patterns & Waves",
        "🎯 Trade Signals",
        "💾 Export"
    ])
    
    with tab1:
        st.subheader(f"Price Chart with Indicators - {selected_token}")
        fig = create_candlestick_chart(summary, main_series)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
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
    
    with tab3:
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
            confluence_color = "🟢" if confluence and confluence > 6 else "🟡" if confluence and confluence > 4 else "🔴"
            st.metric("Confluence Score", f"{confluence_color} {confluence:.2f}" if confluence else "N/A")
            
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
    
    with tab4:
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
    
    with tab5:
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
    
    with tab6:
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
    
    with tab7:
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
    
    with tab8:
        st.subheader("🌐 Breadth Indicators")
        breadth = advanced.get("breadth", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Market Dominance & Correlations")
            st.metric("BTC Dominance", f"{breadth.get('btc_dominance', 0):.2f}%")
            st.metric("S&P500 Correlation", f"{breadth.get('sp500_correlation', 0):.2f}")
            st.metric("NASDAQ Correlation", f"{breadth.get('nasdaq_correlation', 0):.2f}")
        
        with col2:
            st.markdown("### Fear & Greed Index")
            fear_greed = breadth.get("fear_greed_index", 50)
            regime = breadth.get("regime", "Neutral")
            
            if fear_greed >= 70:
                emoji = "😱"
                color = "green"
            elif fear_greed >= 55:
                emoji = "🤑"
                color = "lightgreen"
            elif fear_greed <= 30:
                emoji = "😨"
                color = "red"
            elif fear_greed <= 45:
                emoji = "😟"
                color = "orange"
            else:
                emoji = "😐"
                color = "gray"
            
            st.markdown(f"## {emoji} {fear_greed:.1f}")
            st.markdown(f"**Regime: {regime}**")
            
            st.progress(fear_greed / 100)
    
    with tab9:
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
    
    with tab10:
        st.subheader("🎯 Trade Signal Calculator")
        trade_plan = advanced.get("trade_plan", {})
        
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
            st.markdown("### Entry & Risk Parameters")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entry Price", f"${signal.get('entry_price', 0):.4f}")
                st.metric("Stop Loss", f"${risk.get('stop_loss', 0):.4f}")
            with col2:
                st.metric("ATR Value", f"${risk.get('atr', 0):.4f}")
                st.metric("Risk Amount", f"${risk.get('risk_amount', 0):.2f}")
            with col3:
                st.metric("Max Loss", f"${risk.get('max_loss', 0):.2f}")
                st.metric("Reward/Risk", f"{position.get('reward_risk', 0):.2f}x")
            
            st.markdown("---")
            st.markdown("### Position Details (10x Leverage)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Position Size", f"{position.get('position_size', 0):.4f}")
            with col2:
                st.metric("Notional Value", f"${position.get('notional', 0):.2f}")
            with col3:
                st.metric("Est. Commission", f"${position.get('commission_estimate', 0):.2f}")
            
            st.markdown("---")
            st.markdown("### Take Profit Targets")
            
            if targets:
                targets_df = pd.DataFrame([
                    {
                        "Target": f"TP{idx+1}",
                        "Price": f"${target['price']:.4f}",
                        "Gross P&L": f"${target['gross_pnl']:.2f}",
                        "Net P&L": f"${target['net_pnl']:.2f}"
                    }
                    for idx, target in enumerate(targets)
                ])
                st.dataframe(targets_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.info("⚠️ **Disclaimer:** This is a calculated trade plan based on ATR channels. Always manage your risk and use proper position sizing.")
    
    with tab11:
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
