"""Session state management for the web UI package.

This module provides a centralized SessionStateManager class that handles
all session state operations in a type-safe manner.

Following SRP (Single Responsibility Principle), this module is responsible for:
- Managing session state initialization
- Providing type-safe getters/setters for session state
- Handling chart data store integration
- Managing worker lifecycle state
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class SessionStateManager:
    """Centralized session state manager for web UI.
    
    This class encapsulates all st.session_state access and provides
    type-safe methods for reading and writing session state values.
    """
    
    # Default session state keys
    CHART_SYMBOL_KEY = "chart_symbol"
    CHART_TIMEFRAME_KEY = "chart_timeframe"
    CHART_DF_KEY = "chart_df"
    CHART_INDICATORS_KEY = "chart_indicators"
    LAST_CLOSED_TS_KEY = "last_closed_ts"
    LAST_CLOSED_TS_PER_TF_KEY = "last_closed_ts_per_tf"
    ANALYSIS_UPDATED_KEY = "analysis_updated"
    WORKER_RUNNING_KEY = "worker_running"
    CHART_WORKER_KEY = "chart_worker"
    CHART_MANAGER_STARTED_KEY = "chart_manager_started"
    AUTO_REFRESH_ENABLED_KEY = "auto_refresh_enabled"
    BVI_ENABLED_KEY = "bvi_enabled"
    ATR_CHANNELS_ENABLED_KEY = "atr_channels_enabled"
    ORDER_BLOCKS_ENABLED_KEY = "order_blocks_enabled"
    EXPORT_TOKEN_KEY = "export_token"
    USE_WEBSOCKET_KEY = "use_websocket"
    CHART_UPDATE_BUS_KEY = "chart_update_bus"
    SIGNALS_UPDATE_BUS_KEY = "signals_update_bus"
    CHART_WORKER_MANAGER_KEY = "chart_worker_manager"
    SIGNALS_WORKER_MANAGER_KEY = "signals_worker_manager"
    
    # Keys for automated signals
    AUTOMATED_SIGNALS_STATE_KEY = "automated_signals_state"
    SIGNAL_EXECUTOR_KEY = "signal_executor"
    SIGNALS_WORKER_KEY = "signals_worker"
    AUTO_ADVANCE_END_KEY = "auto_advance_end"
    
    @staticmethod
    def init_chart_state() -> None:
        """Initialize all chart-related session state values."""
        try:
            import streamlit as st
        except ImportError:
            return
            
        defaults = {
            SessionStateManager.CHART_SYMBOL_KEY: None,
            SessionStateManager.CHART_TIMEFRAME_KEY: None,
            SessionStateManager.CHART_DF_KEY: None,
            SessionStateManager.CHART_INDICATORS_KEY: None,
            SessionStateManager.LAST_CLOSED_TS_KEY: 0,
            SessionStateManager.LAST_CLOSED_TS_PER_TF_KEY: {},
            SessionStateManager.ANALYSIS_UPDATED_KEY: False,
            SessionStateManager.WORKER_RUNNING_KEY: False,
            SessionStateManager.CHART_WORKER_KEY: None,
            SessionStateManager.CHART_MANAGER_STARTED_KEY: False,
            SessionStateManager.AUTO_REFRESH_ENABLED_KEY: False,
            SessionStateManager.BVI_ENABLED_KEY: True,
            SessionStateManager.ATR_CHANNELS_ENABLED_KEY: True,
            SessionStateManager.ORDER_BLOCKS_ENABLED_KEY: True,
            SessionStateManager.EXPORT_TOKEN_KEY: "",
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def init_websocket_support() -> None:
        """Initialize WebSocket and UpdateBus support."""
        try:
            import streamlit as st
            from update_bus import UpdateBus
            from worker_manager import ChartWorkerManager, SignalsWorkerManager
        except ImportError:
            return
        
        if SessionStateManager.USE_WEBSOCKET_KEY not in st.session_state:
            st.session_state[SessionStateManager.USE_WEBSOCKET_KEY] = True
        
        if SessionStateManager.CHART_UPDATE_BUS_KEY not in st.session_state:
            st.session_state[SessionStateManager.CHART_UPDATE_BUS_KEY] = UpdateBus()
        
        if SessionStateManager.SIGNALS_UPDATE_BUS_KEY not in st.session_state:
            st.session_state[SessionStateManager.SIGNALS_UPDATE_BUS_KEY] = UpdateBus()
        
        if SessionStateManager.CHART_WORKER_MANAGER_KEY not in st.session_state:
            st.session_state[SessionStateManager.CHART_WORKER_MANAGER_KEY] = ChartWorkerManager()
        
        if SessionStateManager.SIGNALS_WORKER_MANAGER_KEY not in st.session_state:
            st.session_state[SessionStateManager.SIGNALS_WORKER_MANAGER_KEY] = SignalsWorkerManager()
    
    @staticmethod
    def init_automated_signals_state() -> None:
        """Initialize automated signals session state."""
        try:
            import streamlit as st
        except ImportError:
            return
            
        defaults = {
            SessionStateManager.AUTOMATED_SIGNALS_STATE_KEY: None,
            SessionStateManager.SIGNAL_EXECUTOR_KEY: None,
            SessionStateManager.SIGNALS_WORKER_KEY: None,
            SessionStateManager.AUTO_ADVANCE_END_KEY: False,
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def init_all() -> None:
        """Initialize all session state values."""
        SessionStateManager.init_chart_state()
        SessionStateManager.init_websocket_support()
        SessionStateManager.init_automated_signals_state()
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get a value from session state with optional default."""
        try:
            import streamlit as st
        except ImportError:
            return default
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set a value in session state."""
        try:
            import streamlit as st
        except ImportError:
            return
        st.session_state[key] = value
    
    @staticmethod
    def get_chart_symbol() -> Optional[str]:
        """Get the current chart symbol."""
        return SessionStateManager.get(SessionStateManager.CHART_SYMBOL_KEY)
    
    @staticmethod
    def set_chart_symbol(symbol: str) -> None:
        """Set the current chart symbol."""
        SessionStateManager.set(SessionStateManager.CHART_SYMBOL_KEY, symbol)
    
    @staticmethod
    def get_chart_timeframe() -> Optional[str]:
        """Get the current chart timeframe."""
        return SessionStateManager.get(SessionStateManager.CHART_TIMEFRAME_KEY)
    
    @staticmethod
    def set_chart_timeframe(timeframe: str) -> None:
        """Set the current chart timeframe."""
        SessionStateManager.set(SessionStateManager.CHART_TIMEFRAME_KEY, timeframe)
    
    @staticmethod
    def get_chart_df() -> Any:
        """Get the current chart dataframe."""
        return SessionStateManager.get(SessionStateManager.CHART_DF_KEY)
    
    @staticmethod
    def set_chart_df(df: Any) -> None:
        """Set the current chart dataframe."""
        SessionStateManager.set(SessionStateManager.CHART_DF_KEY, df)
    
    @staticmethod
    def is_analysis_updated() -> bool:
        """Check if analysis was updated."""
        return SessionStateManager.get(SessionStateManager.ANALYSIS_UPDATED_KEY, False)
    
    @staticmethod
    def set_analysis_updated(updated: bool) -> None:
        """Set analysis updated flag."""
        SessionStateManager.set(SessionStateManager.ANALYSIS_UPDATED_KEY, updated)
    
    @staticmethod
    def get_auto_refresh_enabled() -> bool:
        """Check if auto-refresh is enabled."""
        return SessionStateManager.get(SessionStateManager.AUTO_REFRESH_ENABLED_KEY, False)
    
    @staticmethod
    def set_auto_refresh_enabled(enabled: bool) -> None:
        """Set auto-refresh enabled flag."""
        SessionStateManager.set(SessionStateManager.AUTO_REFRESH_ENABLED_KEY, enabled)
    
    @staticmethod
    def get_chart_worker() -> Any:
        """Get the chart worker instance."""
        return SessionStateManager.get(SessionStateManager.CHART_WORKER_KEY)
    
    @staticmethod
    def set_chart_worker(worker: Any) -> None:
        """Set the chart worker instance."""
        SessionStateManager.set(SessionStateManager.CHART_WORKER_KEY, worker)
    
    @staticmethod
    def get_chart_worker_manager() -> Any:
        """Get the chart worker manager instance."""
        return SessionStateManager.get(SessionStateManager.CHART_WORKER_MANAGER_KEY)
    
    @staticmethod
    def get_signals_worker_manager() -> Any:
        """Get the signals worker manager instance."""
        return SessionStateManager.get(SessionStateManager.SIGNALS_WORKER_MANAGER_KEY)
    
    @staticmethod
    def get_chart_update_bus() -> Any:
        """Get the chart update bus instance."""
        return SessionStateManager.get(SessionStateManager.CHART_UPDATE_BUS_KEY)
    
    @staticmethod
    def get_signals_update_bus() -> Any:
        """Get the signals update bus instance."""
        return SessionStateManager.get(SessionStateManager.SIGNALS_UPDATE_BUS_KEY)
    
    @staticmethod
    def get_automated_signals_state() -> Any:
        """Get the automated signals state."""
        return SessionStateManager.get(SessionStateManager.AUTOMATED_SIGNALS_STATE_KEY)
    
    @staticmethod
    def set_automated_signals_state(state: Any) -> None:
        """Set the automated signals state."""
        SessionStateManager.set(SessionStateManager.AUTOMATED_SIGNALS_STATE_KEY, state)
    
    @staticmethod
    def get_signal_executor() -> Any:
        """Get the signal executor instance."""
        return SessionStateManager.get(SessionStateManager.SIGNAL_EXECUTOR_KEY)
    
    @staticmethod
    def set_signal_executor(executor: Any) -> None:
        """Set the signal executor instance."""
        SessionStateManager.set(SessionStateManager.SIGNAL_EXECUTOR_KEY, executor)
    
    @staticmethod
    def get_auto_advance_end() -> bool:
        """Check if auto-advance end is enabled."""
        return SessionStateManager.get(SessionStateManager.AUTO_ADVANCE_END_KEY, False)
    
    @staticmethod
    def set_auto_advance_end(enabled: bool) -> None:
        """Set auto-advance end flag."""
        SessionStateManager.set(SessionStateManager.AUTO_ADVANCE_END_KEY, enabled)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "SessionStateManager",
]
