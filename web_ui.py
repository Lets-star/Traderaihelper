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
from indicator_collector.trade_signals import calculate_position_metrics, calculate_tp_sl_levels

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
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
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
        "🔮 Astrology",
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
    
    with tab11:
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
        
    with tab12:
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
