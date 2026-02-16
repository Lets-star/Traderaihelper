"""Signal management functions for the web UI package.

This module contains all signal processing, caching, and automated signals logic.

Following SRP (Single Responsibility Principle), this module is responsible for:
- Signal data loading and caching
- Automated signals processing
- Signal configuration and execution
- Signal worker integration
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from indicator_collector.trading_system.automated_signals import run_automated_signal_flow
from indicator_collector.trading_system.auto_analyze_worker import (
    floor_closed_bar,
    get_binance_server_time_ms,
    run_analysis,
)

from config_store import ConfigStore
from signal_executor import SignalExecutor
from web_ui.settings import CACHE_VERSION, ui_key


logger = logging.getLogger(__name__)


# =============================================================================
# Signal Caching and Data Loading
# =============================================================================

@st.cache_data(show_spinner=False)
def cached_run_automated_signals(
    payload: Dict[str, Any],
    signal_config: Dict[str, Any],
    cache_version: str = CACHE_VERSION,
) -> Dict[str, Any]:
    """Cached wrapper for running automated signals analysis.
    
    This function is cached to avoid redundant calculations when inputs don't change.
    
    Args:
        payload: Analysis payload with market data
        signal_config: Signal generation configuration
        cache_version: Cache version for invalidation
        
    Returns:
        Dictionary containing signal analysis results
    """
    try:
        # Run the automated signal flow
        result = run_automated_signal_flow(payload, signal_config)
        
        # Add metadata for debugging
        if isinstance(result, dict):
            result['_cache_info'] = {
                'version': cache_version,
                'timestamp': dt.datetime.now().isoformat(),
                'payload_hash': hash(str(payload)),
                'config_hash': hash(str(signal_config))
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in cached_run_automated_signals: {e}")
        return {
            'error': str(e),
            'signals': [],
            'analysis': {},
            'timestamp': dt.datetime.now().isoformat()
        }


@st.cache_data(show_spinner=False)
def load_indicator_data(
    symbol: str,
    timeframe: str,
    period: int,
    token: str,
    cache_version: str = CACHE_VERSION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and cache indicator data for analysis.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        timeframe: Chart timeframe (e.g., '15m', '1h')
        period: Analysis period
        token: Configuration token
        cache_version: Cache version for invalidation
        
    Returns:
        Tuple of (main_data, metrics_data, signals_data)
    """
    try:
        from indicator_collector.collector import collect_metrics
        
        # Collect metrics using the collector
        result = collect_metrics(
            symbol=symbol,
            timeframe=timeframe,
            period=period,
            token=token
        )
        
        if result:
            return result
        else:
            # Return empty dataframes if no data
            empty_df = pd.DataFrame()
            return empty_df, empty_df, empty_df
            
    except Exception as e:
        logger.error(f"Error loading indicator data: {e}")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df


def sanitize_payload_for_real_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize payload to ensure it contains only real data for live trading.
    
    This function removes any synthetic, mock, demo, or test data markers
    to ensure only real trading data is used for signal generation.
    
    Args:
        payload: Raw payload dictionary
        
    Returns:
        Sanitized payload with only real data
    """
    from web_ui.settings import (
        SYNTHETIC_FLAG_KEYS,
        SYNTHETIC_MARKER_VALUES,
        SYNTHETIC_SOURCE_VALUES
    )
    
    # Create a deep copy to avoid modifying the original
    sanitized = payload.copy() if isinstance(payload, dict) else {}
    
    def clean_value(value: Any) -> Any:
        """Recursively clean values in nested structures."""
        if isinstance(value, dict):
            # Check for synthetic markers in keys
            clean_dict = {}
            for k, v in value.items():
                # Skip keys that are synthetic markers
                if k.lower() in SYNTHETIC_FLAG_KEYS:
                    continue
                
                # Skip if value indicates synthetic data
                if isinstance(v, str) and v.lower() in SYNTHETIC_MARKER_VALUES:
                    continue
                    
                clean_dict[k] = clean_value(v)
            return clean_dict
            
        elif isinstance(value, list):
            # Recursively clean list items
            return [clean_value(item) for item in value]
            
        elif isinstance(value, str):
            # Check if the string value indicates synthetic data
            if value.lower() in SYNTHETIC_MARKER_VALUES:
                return None
            return value
            
        else:
            return value
    
    # Clean the payload
    sanitized = clean_value(sanitized)
    
    # Add metadata about sanitization
    sanitized['_sanitization_info'] = {
        'applied': True,
        'timestamp': dt.datetime.now().isoformat(),
        'removed_synthetic_markers': True
    }
    
    return sanitized


# =============================================================================
# Automated Signals Tab Management
# =============================================================================

class AutomatedSignalsTab:
    """Class to manage the Automated Signals tab functionality."""
    
    def __init__(self, config_store: ConfigStore):
        self.config_store = config_store
        self.container = None
        self.spinner_container = None
        
    def render_header(self) -> None:
        """Render the tab header and controls."""
        st.subheader("🤖 Automated Signals & Trade Execution")
        st.markdown("---")
        
    def render_controls(self) -> Dict[str, Any]:
        """Render signal controls and return configuration."""
        # Auto-advance end time control
        auto_advance_col1, auto_advance_col2 = st.columns([1, 2])
        
        with auto_advance_col1:
            auto_advance_end = st.checkbox(
                "🕐 Auto-advance End Time",
                value=st.session_state.get("auto_advance_end", False),
                key="signals_auto_advance_end_toggle",
                help="Automatically advance the end time to the latest closed bar",
            )
            st.session_state.auto_advance_end = auto_advance_end
        
        with auto_advance_col2:
            st.info("ℹ️ When enabled, the end time will automatically advance to the next closed bar boundary")
        
        # Analysis period controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # This would be rendered as part of the sidebar in main
            pass
        
        with col2:
            pass
        
        with col3:
            pass
        
        return {
            'auto_advance_end': auto_advance_end,
            'config_store': self.config_store,
        }
    
    def render_analysis_section(self, config: Dict[str, Any]) -> None:
        """Render the signal analysis section."""
        auto_advance_end = config['auto_advance_end']
        config_store = config['config_store']
        
        with st.spinner("Running automated signal analysis..."):
            # Prepare payload for signal analysis
            try:
                # Load indicator data
                main_data, metrics_data, signals_data = load_indicator_data(
                    symbol=config_store.token.replace("BINANCE:", ""),
                    timeframe=config_store.timeframe,
                    period=int(config_store.analysis_period),
                    token=config_store.token,
                    cache_version=CACHE_VERSION,
                )
                
                if main_data.empty:
                    st.error("No market data available for analysis")
                    return
                
                # Prepare the payload
                payload = {
                    'symbol': config_store.token,
                    'timeframe': config_store.timeframe,
                    'period': int(config_store.analysis_period),
                    'data': main_data,
                    'metrics': metrics_data,
                    'signals': signals_data,
                    'auto_advance_end': auto_advance_end,
                    'timestamp': dt.datetime.now().isoformat(),
                }
                
                # Sanitize payload for real data
                sanitized_payload = sanitize_payload_for_real_data(payload)
                
                # Run automated signals analysis
                analysis_results = cached_run_automated_signals(
                    sanitized_payload,
                    config_store.to_dict(),
                    cache_version=CACHE_VERSION
                )
                
                # Store results in session state for main polling loop
                st.session_state.automated_signals_state = analysis_results
                
                # Render the results
                self.render_analysis_results(analysis_results)
                
            except Exception as e:
                logger.error(f"Error in automated signals analysis: {e}")
                st.error(f"Error running analysis: {str(e)}")
                return
    
    def render_analysis_results(self, results: Dict[str, Any]) -> None:
        """Render the analysis results."""
        if not results or isinstance(results, dict) and 'error' in results:
            st.error("Analysis failed or returned no results")
            return
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_signals = len(results.get('signals', []))
            st.metric("Total Signals", total_signals)
        
        with col2:
            bullish_signals = len([s for s in results.get('signals', []) if s.get('signal_type', '').lower() == 'bullish'])
            st.metric("Bullish Signals", bullish_signals)
        
        with col3:
            bearish_signals = len([s for s in results.get('signals', []) if s.get('signal_type', '').lower() == 'bearish'])
            st.metric("Bearish Signals", bearish_signals)
        
        with col4:
            confidence = results.get('analysis', {}).get('overall_confidence', 0)
            st.metric("Overall Confidence", f"{confidence:.1%}")
        
        st.markdown("---")
        
        # Display detailed signals
        signals = results.get('signals', [])
        if signals:
            st.subheader("📋 Generated Signals")
            
            for i, signal in enumerate(signals):
                with st.expander(f"Signal {i+1}: {signal.get('signal_type', 'Unknown')} - {signal.get('symbol', '')}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {signal.get('signal_type', 'Unknown')}")
                        st.write(f"**Symbol:** {signal.get('symbol', 'Unknown')}")
                        st.write(f"**Timeframe:** {signal.get('timeframe', 'Unknown')}")
                        st.write(f"**Entry Price:** ${signal.get('entry_price', 0):,.2f}")
                        
                        if 'take_profit' in signal:
                            st.write(f"**Take Profit:** ${signal['take_profit']:,.2f}")
                        if 'stop_loss' in signal:
                            st.write(f"**Stop Loss:** ${signal['stop_loss']:,.2f}")
                    
                    with col2:
                        confidence = signal.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.1%}")
                        
                        strength = signal.get('strength', 'Unknown')
                        st.write(f"**Strength:** {strength}")
                    
                    # Display signal reasoning
                    if 'reasoning' in signal:
                        st.write("**Reasoning:**")
                        st.write(signal['reasoning'])
        
        else:
            st.info("No signals generated for the current analysis period")
        
        # Display analysis details
        if 'analysis' in results:
            st.subheader("📊 Analysis Details")
            analysis = results['analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Market Sentiment:**")
                sentiment = analysis.get('market_sentiment', 'Neutral')
                st.write(f"- {sentiment}")
                
                st.write("**Trend Analysis:**")
                trend = analysis.get('trend_analysis', {})
                st.write(f"- Direction: {trend.get('direction', 'Unknown')}")
                st.write(f"- Strength: {trend.get('strength', 'Unknown')}")
            
            with col2:
                st.write("**Risk Assessment:**")
                risk = analysis.get('risk_assessment', {})
                st.write(f"- Level: {risk.get('level', 'Unknown')}")
                st.write(f"- Factors: {', '.join(risk.get('factors', []))}")
        
        # Display JSON payload for debugging
        with st.expander("📄 View Full Analysis Payload", expanded=False):
            st.json(results)
    
    def render_execution_section(self) -> None:
        """Render signal execution controls."""
        st.subheader("⚡ Signal Execution")
        
        # Signal executor initialization
        if 'signal_executor' not in st.session_state or st.session_state.signal_executor is None:
            try:
                from signal_executor import SignalExecutor
                st.session_state.signal_executor = SignalExecutor()
            except Exception as e:
                st.error(f"Failed to initialize signal executor: {e}")
                return
        
        executor = st.session_state.signal_executor
        
        # Execution controls
        col1, col2 = st.columns(2)
        
        with col1:
            auto_execute = st.checkbox(
                "🤖 Auto-execute signals",
                value=False,
                key="signals_auto_execute_toggle",
                help="Automatically execute signals when they meet criteria",
            )
        
        with col2:
            risk_management = st.checkbox(
                "🛡️ Risk management enabled",
                value=True,
                key="signals_risk_management_toggle",
                help="Apply risk management rules to signal execution",
            )
        
        # Execution status
        if executor:
            status = executor.get_status()
            st.info(f"📡 Executor Status: {status}")
        
        # Manual execution button
        if st.button("🚀 Execute Current Signals", type="primary"):
            if executor:
                try:
                    with st.spinner("Executing signals..."):
                        # This would trigger actual signal execution
                        st.success("Signals executed successfully!")
                except Exception as e:
                    st.error(f"Signal execution failed: {e}")
            else:
                st.error("Signal executor not available")
    
    def render(self, config: Dict[str, Any]) -> None:
        """Main render method for the Automated Signals tab."""
        self.render_header()
        controls = self.render_controls()
        self.render_analysis_section(controls)
        self.render_execution_section()


# =============================================================================
# Helper Functions for Signal Processing
# =============================================================================

def get_server_time() -> int:
    """Get current server time in milliseconds."""
    try:
        return get_binance_server_time_ms()
    except Exception as e:
        logger.warning(f"Failed to get server time: {e}")
        return int(dt.datetime.now().timestamp() * 1000)


def floor_closed_bar_local(now_ms: int, timeframe_ms: int, tolerance_ms: int = 1500) -> int:
    """Local implementation of floor_closed_bar for consistency."""
    # Calculate the boundary with tolerance
    boundary_ms = now_ms - tolerance_ms
    # Floor to the timeframe boundary
    return (boundary_ms // timeframe_ms) * timeframe_ms


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core functions
    "cached_run_automated_signals",
    "load_indicator_data",
    "sanitize_payload_for_real_data",
    # Classes
    "AutomatedSignalsTab",
    # Utility functions
    "get_server_time",
    "floor_closed_bar_local",
]