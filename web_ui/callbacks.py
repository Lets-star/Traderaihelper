"""Callback handlers for the web UI package.

This module contains all event handler callbacks used in the web UI.

Following SRP (Single Responsibility Principle), this module is responsible for:
- Button click handlers
- Form submission handlers
- Widget change callbacks
- Event handler registration
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import streamlit as st


logger = logging.getLogger(__name__)


# =============================================================================
# Callback Registry
# =============================================================================

class CallbackRegistry:
    """Central registry for managing event callbacks.
    
    This class provides a way to register, manage, and execute callbacks
    for various UI events in the application.
    """
    
    def __init__(self):
        self._callbacks: Dict[str, Callable] = {}
        self._callback_data: Dict[str, Any] = {}
    
    def register(self, name: str, callback: Callable, data: Optional[Any] = None) -> None:
        """Register a callback.
        
        Args:
            name: Unique name for the callback
            callback: The callback function
            data: Optional data to pass to the callback
        """
        self._callbacks[name] = callback
        if data is not None:
            self._callback_data[name] = data
        logger.debug(f"Registered callback: {name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a callback.
        
        Args:
            name: Name of the callback to unregister
        """
        if name in self._callbacks:
            del self._callbacks[name]
        if name in self._callback_data:
            del self._callback_data[name]
        logger.debug(f"Unregistered callback: {name}")
    
    def execute(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered callback.
        
        Args:
            name: Name of the callback to execute
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
            
        Returns:
            The return value of the callback, or None if not found
        """
        if name not in self._callbacks:
            logger.warning(f"Callback not found: {name}")
            return None
        
        callback = self._callbacks[name]
        data = self._callback_data.get(name)
        
        try:
            if data is not None:
                return callback(data, *args, **kwargs)
            else:
                return callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing callback '{name}': {e}")
            return None
    
    def has_callback(self, name: str) -> bool:
        """Check if a callback is registered.
        
        Args:
            name: Name of the callback to check
            
        Returns:
            True if the callback is registered, False otherwise
        """
        return name in self._callbacks
    
    def get_callback(self, name: str) -> Optional[Callable]:
        """Get a registered callback.
        
        Args:
            name: Name of the callback to get
            
        Returns:
            The callback function, or None if not found
        """
        return self._callbacks.get(name)


# Global callback registry
_callback_registry = CallbackRegistry()


def get_callback_registry() -> CallbackRegistry:
    """Get the global callback registry instance."""
    return _callback_registry


# =============================================================================
# Chart Callbacks
# =============================================================================

def on_chart_symbol_change() -> None:
    """Callback when chart symbol changes."""
    logger.info("Chart symbol changed")
    # Invalidate any cached chart data
    try:
        from chart_auto_refresh import invalidate_cache
        invalidate_cache()
    except ImportError:
        pass
    
    # Reset chart state
    st.session_state.chart_df = None
    st.session_state.chart_indicators = None
    st.session_state.last_closed_ts = 0


def on_chart_timeframe_change() -> None:
    """Callback when chart timeframe changes."""
    logger.info("Chart timeframe changed")
    # Invalidate cached data for old timeframe
    try:
        from chart_auto_refresh import invalidate_cache
        invalidate_cache()
    except ImportError:
        pass
    
    # Reset chart data
    st.session_state.chart_df = None
    st.session_state.chart_indicators = None
    st.session_state.last_closed_ts = 0


def on_chart_update() -> None:
    """Callback when chart update is triggered."""
    logger.debug("Chart update triggered")
    # This is called when user requests a chart refresh
    # The actual update logic is handled by the chart worker


def on_indicator_toggle(toggle_name: str) -> None:
    """Callback when an indicator toggle changes.
    
    Args:
        toggle_name: Name of the toggle that changed
    """
    logger.info(f"Indicator toggle changed: {toggle_name}")
    # Store the toggle state
    st.session_state[f"indicator_{toggle_name}_enabled"] = True


def on_auto_refresh_toggle() -> None:
    """Callback when auto-refresh toggle changes."""
    enabled = st.session_state.get("auto_refresh_enabled", False)
    logger.info(f"Auto-refresh toggled: {enabled}")
    
    if enabled:
        # Start the chart worker
        from web_ui.state_manager import SessionStateManager
        if SessionStateManager.get_chart_worker() is None:
            try:
                from chart_auto_refresh import ChartAutoRefreshWorker
                worker = ChartAutoRefreshWorker()
                SessionStateManager.set_chart_worker(worker)
                logger.info("Chart auto-refresh worker started")
            except Exception as e:
                logger.error(f"Failed to start chart worker: {e}")
                st.session_state.auto_refresh_enabled = False
    else:
        # Stop the chart worker
        worker = SessionStateManager.get_chart_worker()
        if worker is not None:
            try:
                worker.stop()
                logger.info("Chart auto-refresh worker stopped")
            except Exception as e:
                logger.error(f"Failed to stop chart worker: {e}")
        SessionStateManager.set_chart_worker(None)


# =============================================================================
# Signal Callbacks
# =============================================================================

def on_signal_update() -> None:
    """Callback when signal update is triggered."""
    logger.debug("Signal update triggered")
    # This is called when user requests a signal refresh
    # The actual update logic is handled by the signals worker


def on_auto_advance_end_toggle() -> None:
    """Callback when auto-advance end toggle changes."""
    enabled = st.session_state.get("auto_advance_end", False)
    logger.info(f"Auto-advance end toggled: {enabled}")
    
    if enabled:
        # Start the signals worker
        if st.session_state.get("signals_worker") is None:
            try:
                from automated_signals_worker import AutomatedSignalsWorker
                worker = AutomatedSignalsWorker()
                st.session_state.signals_worker = worker
                logger.info("Signals auto-advance worker started")
            except Exception as e:
                logger.error(f"Failed to start signals worker: {e}")
                st.session_state.auto_advance_end = False
    else:
        # Stop the signals worker
        worker = st.session_state.get("signals_worker")
        if worker is not None:
            try:
                worker.stop()
                logger.info("Signals auto-advance worker stopped")
            except Exception as e:
                logger.error(f"Failed to stop signals worker: {e}")
        st.session_state.signals_worker = None


def on_signal_execution(signal_data: Dict[str, Any]) -> None:
    """Execute a trading signal.
    
    Args:
        signal_data: Dictionary containing signal details
    """
    logger.info(f"Executing signal: {signal_data.get('signal_type')} for {signal_data.get('symbol')}")
    
    try:
        executor = st.session_state.get("signal_executor")
        if executor is None:
            from signal_executor import SignalExecutor
            executor = SignalExecutor()
            st.session_state.signal_executor = executor
        
        # Execute the signal
        result = executor.execute(signal_data)
        
        if result.get('success'):
            st.success(f"Signal executed successfully: {result.get('message')}")
        else:
            st.error(f"Signal execution failed: {result.get('message')}")
            
    except Exception as e:
        logger.error(f"Error executing signal: {e}")
        st.error(f"Signal execution error: {str(e)}")


def on_signal_dismiss(signal_id: str) -> None:
    """Dismiss a signal.
    
    Args:
        signal_id: ID of the signal to dismiss
    """
    logger.info(f"Dismissing signal: {signal_id}")
    
    # Remove the signal from session state
    if 'dismissed_signals' not in st.session_state:
        st.session_state.dismissed_signals = []
    
    st.session_state.dismissed_signals.append(signal_id)


# =============================================================================
# Configuration Callbacks
# =============================================================================

def on_config_update(section: str, key: str, value: Any) -> None:
    """Callback when configuration is updated.
    
    Args:
        section: Configuration section
        key: Configuration key
        value: New value
    """
    logger.info(f"Config update: {section}.{key} = {value}")
    
    try:
        from config_store import ConfigStore
        config_store = ConfigStore.load()
        
        # Update the config
        if hasattr(config_store, f"set_{key}"):
            getattr(config_store, f"set_{key}")(value)
        elif hasattr(config_store, 'set_value'):
            config_store.set_value(section, key, value)
        
        # Save the config
        config_store.save()
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        st.error(f"Failed to save configuration: {str(e)}")


def on_weights_update(weights: Dict[str, float]) -> None:
    """Callback when factor weights are updated.
    
    Args:
        weights: Dictionary of factor weights
    """
    logger.info(f"Weights updated: {weights}")
    
    try:
        from config_store import ConfigStore
        config_store = ConfigStore.load()
        
        # Update weights in config
        config_store.set_factor_weights(weights)
        config_store.save()
        
    except Exception as e:
        logger.error(f"Error saving weights: {e}")
        st.error(f"Failed to save weights: {str(e)}")


def on_indicator_params_update(params: Dict[str, Any]) -> None:
    """Callback when indicator parameters are updated.
    
    Args:
        params: Dictionary of indicator parameters
    """
    logger.info(f"Indicator params updated: {params}")
    
    try:
        from config_store import ConfigStore
        config_store = ConfigStore.load()
        
        # Update indicator parameters in config
        config_store.set_indicator_params(params)
        config_store.save()
        
    except Exception as e:
        logger.error(f"Error saving indicator params: {e}")
        st.error(f"Failed to save indicator parameters: {str(e)}")


# =============================================================================
# Export Callbacks
# =============================================================================

def on_export_data(export_type: str) -> None:
    """Export data to file.
    
    Args:
        export_type: Type of export (json, csv, etc.)
    """
    logger.info(f"Export requested: {export_type}")
    
    try:
        if export_type == "json":
            # Export chart data as JSON
            if 'chart_df' in st.session_state and st.session_state.chart_df is not None:
                import json
                from datetime import datetime
                
                df = st.session_state.chart_df
                data_dict = df.to_dict(orient='records')
                
                # Create downloadable JSON
                json_str = json.dumps(data_dict, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        elif export_type == "csv":
            # Export chart data as CSV
            if 'chart_df' in st.session_state and st.session_state.chart_df is not None:
                import csv
                from datetime import datetime
                
                df = st.session_state.chart_df
                csv_str = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        st.error(f"Failed to export data: {str(e)}")


# =============================================================================
# Utility Callbacks
# =============================================================================

def on_rerun() -> None:
    """Trigger a Streamlit rerun."""
    try:
        st.rerun()
    except Exception as e:
        logger.debug(f"Cannot rerun: {e}")


def on_clear_cache() -> None:
    """Clear all cached data."""
    logger.info("Clearing cache")
    
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Invalidate custom caches
        try:
            from chart_auto_refresh import invalidate_cache
            invalidate_cache()
        except ImportError:
            pass
        
        st.success("Cache cleared successfully!")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        st.error(f"Failed to clear cache: {str(e)}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Registry
    "CallbackRegistry",
    "get_callback_registry",
    # Chart callbacks
    "on_chart_symbol_change",
    "on_chart_timeframe_change",
    "on_chart_update",
    "on_indicator_toggle",
    "on_auto_refresh_toggle",
    # Signal callbacks
    "on_signal_update",
    "on_auto_advance_end_toggle",
    "on_signal_execution",
    "on_signal_dismiss",
    # Config callbacks
    "on_config_update",
    "on_weights_update",
    "on_indicator_params_update",
    # Export callbacks
    "on_export_data",
    # Utility callbacks
    "on_rerun",
    "on_clear_cache",
]
