"""Web UI package - Modular Streamlit application.

This package contains a refactored version of the monolithic web_ui.py file,
split into modules following SRP (Single Responsibility Principle).

Modules:
- settings: UI constants and helper functions
- state_manager: Session state management
- charts: Chart visualization functions
- signals: Signal management functions
- callbacks: Event handler callbacks

Usage:
    from web_ui import main
    main()

Or for backward compatibility:
    import web_ui  # Shows deprecation warning
    web_ui.main()
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Any, Dict, Optional

# Set page config must be first Streamlit command
import streamlit as st

st.set_page_config(
    page_title="Token Charts & Indicators",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import all submodules
from web_ui import settings
from web_ui.state_manager import SessionStateManager
from web_ui import charts
from web_ui import signals
from web_ui import callbacks

# Import additional modules needed for main
from config_store import ConfigStore
from indicator_collector.collector import collect_metrics
from indicator_collector.indicator_metrics import SimulationSummary
from indicator_collector.time_series import TimeframeSeries
from indicator_collector.timeframes import Timeframe
from indicator_collector.trade_signals import calculate_position_metrics, calculate_tp_sl_levels
from indicator_collector.trading_system.backtester import (
    DEFAULT_SIGNAL_THRESHOLDS,
    indicator_defaults_for,
)
from automated_signals_worker import AutomatedSignalsWorker
from chart_auto_refresh import compute_atr_channels
from indicator_collector.trading_system.auto_analyze_worker import (
    AutoAnalyzeWorker,
    floor_closed_bar,
    get_binance_server_time_ms,
    run_analysis,
)
from indicator_collector.trading_system.data_sources.timestamp_utils import normalize_timestamp
from indicator_collector.trading_system.signal_generator import SignalConfig
from indicator_collector.trading_system.signal_schema import is_valid_signal_structure
from signal_executor import SignalExecutor
from update_bus import UpdateBus
from worker_manager import ChartWorkerManager, SignalsWorkerManager


logger = logging.getLogger(__name__)


# =============================================================================
# Re-export all public APIs for backward compatibility
# =============================================================================

# Settings module exports
__all__ = settings.__all__ + [
    "safe_rerun",
    "get_api_credentials_from_secrets",
    "cached_run_automated_signals",
    "load_indicator_data",
    "calculate_better_volume_indicator",
    "create_realtime_candlestick_chart",
    "create_candlestick_chart",
    "create_multi_timeframe_chart",
    "render_weight_controls",
    "render_indicator_controls",
    "render_signal_risk_controls",
    "main",
    # Also export modules for direct access
    "settings",
    "state_manager",
    "charts",
    "signals",
    "callbacks",
    "SessionStateManager",
]


# =============================================================================
# Utility Functions (from original web_ui.py)
# =============================================================================

def safe_rerun():
    """Safely rerun Streamlit app with error handling."""
    try:
        st.rerun()
    except Exception as e:
        logger.debug(f"Cannot rerun: no ScriptRunContext or other error: {e}")


def get_api_credentials_from_secrets() -> tuple[str, str]:
    """
    Retrieve API credentials from st.secrets with fallback to empty strings.

    Returns:
        Tuple of (api_key, api_secret)
    """
    try:
        bybit_secrets = st.secrets.get("bybit", {})
        api_key = bybit_secrets.get("api_key", "")
        api_secret = bybit_secrets.get("api_secret", "")
        return str(api_key), str(api_secret)
    except Exception:
        return "", ""


# =============================================================================
# UI Render Functions (from original web_ui.py)
# =============================================================================

def render_weight_controls(config_store: ConfigStore) -> Dict[str, float]:
    """Render factor weight controls in the sidebar."""
    st.markdown("### 📊 Factor Weights")
    
    weights = {}
    
    # Technical weight
    weights['technical'] = settings.num_float(
        "Technical",
        min_v=0.0,
        value=config_store.get_factor_weight('technical', 0.30),
        max_v=1.0,
        step=0.05,
        key=settings.ui_key("weights", "technical"),
        help_text="Weight for technical analysis signals"
    )
    
    # Sentiment weight
    weights['sentiment'] = settings.num_float(
        "Sentiment",
        min_v=0.0,
        value=config_store.get_factor_weight('sentiment', 0.20),
        max_v=1.0,
        step=0.05,
        key=settings.ui_key("weights", "sentiment"),
        help_text="Weight for sentiment analysis signals"
    )
    
    # Multi-timeframe weight
    weights['multitimeframe'] = settings.num_float(
        "Multi-timeframe",
        min_v=0.0,
        value=config_store.get_factor_weight('multitimeframe', 0.20),
        max_v=1.0,
        step=0.05,
        key=settings.ui_key("weights", "multitimeframe"),
        help_text="Weight for multi-timeframe alignment signals"
    )
    
    # Volume weight
    weights['volume'] = settings.num_float(
        "Volume",
        min_v=0.0,
        value=config_store.get_factor_weight('volume', 0.15),
        max_v=1.0,
        step=0.05,
        key=settings.ui_key("weights", "volume"),
        help_text="Weight for volume analysis signals"
    )
    
    # Market structure weight
    weights['market_structure'] = settings.num_float(
        "Market Structure",
        min_v=0.0,
        value=config_store.get_factor_weight('market_structure', 0.10),
        max_v=1.0,
        step=0.05,
        key=settings.ui_key("weights", "market_structure"),
        help_text="Weight for market structure signals"
    )
    
    # Composite weight
    weights['composite'] = settings.num_float(
        "Composite",
        min_v=0.0,
        value=config_store.get_factor_weight('composite', 0.05),
        max_v=1.0,
        step=0.05,
        key=settings.ui_key("weights", "composite"),
        help_text="Weight for composite indicator signals"
    )
    
    # Normalize weights
    normalized, raw = settings.normalize_category_weights(weights)
    
    # Show normalization info
    total = sum(raw.values())
    if total > 0 and abs(total - 1.0) > 1e-6:
        st.caption(f"⚠️ Weights normalized to sum to 1.0 (total: {total:.2f})")
    
    return normalized


def render_indicator_controls(config_store: ConfigStore) -> None:
    """Render indicator controls in the sidebar."""
    st.markdown("### 📈 Indicators")
    
    # RSI settings
    with st.expander("RSI Settings", expanded=False):
        config_store.rsi_period = settings.num_int(
            "RSI Period",
            min_v=1,
            value=config_store.rsi_period or 14,
            max_v=50,
            key=settings.ui_key("indicator", "rsi_period"),
            help_text="Period for RSI calculation"
        )
        
        config_store.rsi_overbought = settings.num_int(
            "RSI Overbought",
            min_v=50,
            value=config_store.rsi_overbought or 70,
            max_v=100,
            key=settings.ui_key("indicator", "rsi_overbought"),
            help_text="RSI overbought threshold"
        )
        
        config_store.rsi_oversold = settings.num_int(
            "RSI Oversold",
            min_v=0,
            value=config_store.rsi_oversold or 30,
            max_v=50,
            key=settings.ui_key("indicator", "rsi_oversold"),
            help_text="RSI oversold threshold"
        )
    
    # MACD settings
    with st.expander("MACD Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            config_store.macd_fast = settings.num_int(
                "Fast Period",
                min_v=1,
                value=config_store.macd_fast or 12,
                max_v=50,
                key=settings.ui_key("indicator", "macd_fast")
            )
        with col2:
            config_store.macd_slow = settings.num_int(
                "Slow Period",
                min_v=1,
                value=config_store.macd_slow or 26,
                max_v=100,
                key=settings.ui_key("indicator", "macd_slow")
            )
        with col3:
            config_store.macd_signal = settings.num_int(
                "Signal Period",
                min_v=1,
                value=config_store.macd_signal or 9,
                max_v=50,
                key=settings.ui_key("indicator", "macd_signal")
            )
    
    # Bollinger Bands settings
    with st.expander("Bollinger Bands Settings", expanded=False):
        config_store.bb_period = settings.num_int(
            "BB Period",
            min_v=1,
            value=config_store.bb_period or 20,
            max_v=100,
            key=settings.ui_key("indicator", "bb_period"),
            help_text="Period for Bollinger Bands calculation"
        )
        
        config_store.bb_std = settings.num_float(
            "BB Std Dev",
            min_v=0.1,
            value=config_store.bb_std or 2.0,
            max_v=5.0,
            step=0.1,
            key=settings.ui_key("indicator", "bb_std"),
            help_text="Standard deviation multiplier for Bollinger Bands"
        )


def render_signal_risk_controls(config_store: ConfigStore) -> None:
    """Render signal risk controls in the sidebar."""
    st.markdown("### 🎯 Risk & Signals")
    
    # Signal thresholds
    with st.expander("Signal Thresholds", expanded=False):
        config_store.signal_threshold = settings.num_float(
            "Signal Threshold",
            min_v=0.0,
            value=config_store.signal_threshold or 0.5,
            max_v=1.0,
            step=0.05,
            key=settings.ui_key("risk", "signal_threshold"),
            help_text="Minimum confidence threshold for signals"
        )
        
        config_store.min_signal_strength = settings.num_float(
            "Min Signal Strength",
            min_v=0.0,
            value=config_store.min_signal_strength or 0.3,
            max_v=1.0,
            step=0.05,
            key=settings.ui_key("risk", "min_signal_strength"),
            help_text="Minimum strength for signal generation"
        )
    
    # Risk management
    with st.expander("Risk Management", expanded=False):
        config_store.default_risk_percent = settings.num_float(
            "Risk per Trade (%)",
            min_v=0.1,
            value=config_store.default_risk_percent or 1.0,
            max_v=10.0,
            step=0.1,
            key=settings.ui_key("risk", "default_risk_percent"),
            help_text="Percentage of account to risk per trade"
        )
        
        config_store.max_position_size = settings.num_float(
            "Max Position Size (%)",
            min_v=1.0,
            value=config_store.max_position_size or 10.0,
            max_v=100.0,
            step=1.0,
            key=settings.ui_key("risk", "max_position_size"),
            help_text="Maximum position size as percentage of account"
        )
        
        config_store.default_stop_loss_pct = settings.num_float(
            "Default Stop Loss (%)",
            min_v=0.1,
            value=config_store.default_stop_loss_pct or 2.0,
            max_v=20.0,
            step=0.1,
            key=settings.ui_key("risk", "default_stop_loss_pct"),
            help_text="Default stop loss percentage"
        )
        
        config_store.default_take_profit_pct = settings.num_float(
            "Default Take Profit (%)",
            min_v=0.1,
            value=config_store.default_take_profit_pct or 4.0,
            max_v=50.0,
            step=0.1,
            key=settings.ui_key("risk", "default_take_profit_pct"),
            help_text="Default take profit percentage"
        )


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    st.title("📈 Token Charts & Indicators Dashboard")
    st.markdown("---")
    config_store = ConfigStore.load()
    
    # Initialize session state
    SessionStateManager.init_all()
    
    # Also initialize individual chart states for backward compatibility
    if "chart_symbol" not in st.session_state:
        st.session_state.chart_symbol = None
    if "chart_timeframe" not in st.session_state:
        st.session_state.chart_timeframe = None
    if "chart_df" not in st.session_state:
        st.session_state.chart_df = None
    if "chart_indicators" not in st.session_state:
        st.session_state.chart_indicators = None
    if "last_closed_ts" not in st.session_state:
        st.session_state.last_closed_ts = 0
    if "last_closed_ts_per_tf" not in st.session_state:
        st.session_state.last_closed_ts_per_tf = {}
    if "analysis_updated" not in st.session_state:
        st.session_state.analysis_updated = False
    if "worker_running" not in st.session_state:
        st.session_state.worker_running = False
    if "chart_worker" not in st.session_state:
        st.session_state.chart_worker = None
    if "chart_manager_started" not in st.session_state:
        st.session_state.chart_manager_started = False
    if "auto_refresh_enabled" not in st.session_state:
        st.session_state.auto_refresh_enabled = False
    if "bvi_enabled" not in st.session_state:
        st.session_state.bvi_enabled = True
    if "atr_channels_enabled" not in st.session_state:
        st.session_state.atr_channels_enabled = True
    if "order_blocks_enabled" not in st.session_state:
        st.session_state.order_blocks_enabled = True
    if "export_token" not in st.session_state:
        st.session_state.export_token = ""
    if "use_websocket" not in st.session_state:
        st.session_state.use_websocket = True
    
    # Initialize update buses if not present
    if "chart_update_bus" not in st.session_state:
        st.session_state.chart_update_bus = UpdateBus()
    if "signals_update_bus" not in st.session_state:
        st.session_state.signals_update_bus = UpdateBus()
    
    # Initialize worker managers if not present
    if "chart_worker_manager" not in st.session_state:
        st.session_state.chart_worker_manager = ChartWorkerManager()
    if "signals_worker_manager" not in st.session_state:
        st.session_state.signals_worker_manager = SignalsWorkerManager()
    
    # Initialize automated signals state
    if "automated_signals_state" not in st.session_state:
        st.session_state.automated_signals_state = None
    if "signal_executor" not in st.session_state:
        st.session_state.signal_executor = None
    
    with st.sidebar:
        st.header("⚙️ Configuration")

        token_mode_index = 0 if config_store.token in settings.POPULAR_TOKENS else 1
        token_input_mode = st.radio(
            "Input Mode",
            ["Select from list", "Custom token"],
            index=token_mode_index,
            key=settings.ui_key("sidebar", "input_mode"),
        )

        if token_input_mode == "Select from list":
            try:
                default_index = settings.POPULAR_TOKENS.index(config_store.token)
            except ValueError:
                default_index = 0
            selected_token_option = st.selectbox(
                "Select Token",
                settings.POPULAR_TOKENS,
                index=default_index,
                key=settings.ui_key("sidebar", "select_token"),
            )
            config_store.set_token(selected_token_option)
        else:
            selected_token_option = st.text_input(
                "Custom Token (e.g., BINANCE:BTCUSDT)",
                config_store.token,
                key=settings.ui_key("sidebar", "custom_token"),
            )
            config_store.set_token(selected_token_option)

        st.subheader("Timeframe & Period")
        try:
            timeframe_index = settings.TIMEFRAMES.index(config_store.timeframe)
        except ValueError:
            timeframe_index = settings.TIMEFRAMES.index("15m")
        selected_timeframe_option = st.selectbox(
            "Timeframe",
            settings.TIMEFRAMES,
            index=timeframe_index,
            key=settings.ui_key("sidebar", "timeframe"),
        )
        config_store.set_timeframe(selected_timeframe_option)

        # Analysis period
        analysis_period_default = st.session_state.get(
            settings.ui_key("sidebar", "analysis_period_value"), 200
        )
        config_store.analysis_period = settings.num_int(
            "Analysis Period",
            min_v=10,
            value=config_store.analysis_period or analysis_period_default,
            max_v=1000,
            step=10,
            key=settings.ui_key("sidebar", "analysis_period_value"),
            help_text="Number of candles to analyze",
        )

        selected_token = config_store.token
        selected_timeframe = config_store.timeframe

        # Render factor weight controls
        factor_weights = render_weight_controls(config_store)
        config_store.set_factor_weights(factor_weights)

        # Render indicator controls
        render_indicator_controls(config_store)

        # Render signal risk controls
        render_signal_risk_controls(config_store)

        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            try:
                config_store.save()
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Failed to save configuration: {e}")

    # Initial analysis on load
    with st.spinner(f"Analyzing {selected_token} on {selected_timeframe} timeframe..."):
        try:
            summary, main_series, payload = run_analysis(
                token=selected_token,
                timeframe=selected_timeframe,
                period=config_store.analysis_period,
                config=config_store.to_dict(),
                factor_weights=factor_weights,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

    # Create tabs
    (
        chart_tab,
        multitf_tab,
        latest_metrics_tab,
        signals_zones_tab,
        volume_tab,
        structure_tab,
        fundamentals_tab,
        breadth_tab,
        onchain_tab,
        composite_tab,
        patterns_tab,
        trade_signals_tab,
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
        from chart_auto_refresh import (
            ChartAutoRefreshWorker,
            compute_chart_indicators,
            fetch_closed_candles,
            invalidate_cache,
            read_chart_state,
            update_chart_state,
        )
        
        st.subheader(f"Price Chart with Indicators - {selected_token}")
        
        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])
        with ctrl_col1:
            show_forming_bar = st.checkbox(
                "📊 Forming Bar",
                value=st.session_state.get("show_forming_bar", False),
                key="charts_forming_bar_toggle",
                help="Show the currently forming candle (not yet closed)",
            )
            st.session_state.show_forming_bar = show_forming_bar
        with ctrl_col2:
            bvi_enabled = st.checkbox(
                "📊 Better Volume",
                value=st.session_state.bvi_enabled,
                key="charts_bvi_toggle",
                help="Show Better Volume Indicator",
            )
            st.session_state.bvi_enabled = bvi_enabled
        with ctrl_col3:
            atr_channels_enabled = st.checkbox(
                "📈 ATR Channels",
                value=st.session_state.atr_channels_enabled,
                key="charts_atr_toggle",
                help="Show ATR Channels",
            )
            st.session_state.atr_channels_enabled = atr_channels_enabled
        with ctrl_col4:
            order_blocks_enabled = st.checkbox(
                "🧱 Order Blocks",
                value=st.session_state.order_blocks_enabled,
                key="charts_ob_toggle",
                help="Show Order Blocks",
            )
            st.session_state.order_blocks_enabled = order_blocks_enabled
        
        # Auto-refresh controls
        auto_col1, auto_col2 = st.columns([1, 2])
        with auto_col1:
            auto_refresh_enabled = st.checkbox(
                "🔄 Auto-refresh",
                value=st.session_state.auto_refresh_enabled,
                key="charts_auto_refresh_toggle",
                help="Automatically refresh chart data",
            )
            st.session_state.auto_refresh_enabled = auto_refresh_enabled
        
        with auto_col2:
            st.info(f"📡 Using timeframe: {selected_timeframe}")
        
        # Render the chart
        with st.spinner(f"Loading chart data for {selected_token} {selected_timeframe}..."):
            # Fetch chart data
            try:
                df = fetch_closed_candles(selected_token, selected_timeframe, config_store.analysis_period)
                
                if df is not None and not df.empty:
                    # Create chart
                    fig = charts.create_realtime_candlestick_chart(
                        df,
                        chart_height=700,
                        timeframe=selected_timeframe,
                        show_forming_bar=show_forming_bar,
                        bvi_enabled=bvi_enabled,
                        atr_channels_enabled=atr_channels_enabled,
                        order_blocks_enabled=order_blocks_enabled,
                    )
                    
                    # Render chart
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            'displayModeBar': True,
                            'scrollZoom': True,
                            'responsive': True,
                        }
                    )
                    
                    # Store in session state
                    st.session_state.chart_df = df
                    
                else:
                    st.warning("No chart data available")
                    
            except Exception as e:
                st.error(f"Error loading chart: {e}")
                logger.exception("Chart loading error")
    
    # Continue with other tabs - simplified for now since full implementation is in original
    with multitf_tab:
        st.subheader("📈 Multi-Timeframe Analysis")
        st.info("Multi-timeframe analysis is available in the full implementation")
        
    with latest_metrics_tab:
        st.subheader("📋 Latest Metrics")
        if summary:
            st.json(summary.to_dict() if hasattr(summary, 'to_dict') else str(summary))
        else:
            st.info("No metrics available")
        
    with signals_zones_tab:
        st.subheader("🎯 Signals & Zones")
        st.info("Signals and zones are available in the full implementation")
        
    with automated_signals_tab:
        # Use the signals module
        signals_tab = signals.AutomatedSignalsTab(config_store)
        signals_tab.render({'auto_advance_end': st.session_state.get('auto_advance_end', False), 'config_store': config_store})
        
    # Additional tabs would be implemented similarly...
    with trade_signals_tab:
        st.subheader("🎯 Trade Signals")
        st.info("Trade signals are available in the full implementation")
        
    with backtest_tab:
        st.subheader("🔬 Backtesting")
        st.info("Backtesting is available in the full implementation")
        
    with export_tab:
        st.subheader("💾 Export Data")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("Export as JSON"):
                callbacks.on_export_data("json")
        
        with export_col2:
            if st.button("Export as CSV"):
                callbacks.on_export_data("csv")
        
        # Clear cache button
        st.markdown("---")
        if st.button("🗑️ Clear Cache"):
            callbacks.on_clear_cache()
    
    # Auto-refresh polling loop
    if st.session_state.auto_refresh_enabled:
        import time
        import threading
        
        # Determine polling interval based on timeframe
        tf_to_seconds = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080
        }
        poll_interval = tf_to_seconds.get(selected_timeframe, 15)
        if poll_interval < 60:
            poll_interval = 1  # 1 second for short timeframes
        elif poll_interval >= 3600:
            poll_interval = 5  # 5 seconds for longer timeframes
        else:
            poll_interval = 1  # default to 1 second
        
        # Placeholder for polling loop - in full implementation this would be handled by ChartAutoRefreshWorker
        st.empty()


# =============================================================================
# Package Initialization
# =============================================================================

def _init_package():
    """Initialize the package."""
    logger.info("Web UI package initialized")
    
    # Set up any global configurations here
    pass


# Run initialization
_init_package()
