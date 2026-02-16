#!/usr/bin/env python3
"""
Legacy web_ui.py - Backward compatibility wrapper.

This module provides backward compatibility for the refactored web_ui package.
When imported directly, it shows a deprecation warning and redirects to the new package.

Usage (deprecated):
    import web_ui
    web_ui.main()

Recommended usage:
    from web_ui import main
    main()

Or:
    import web_ui
    web_ui.web_ui.main()  # Explicitly access the package
"""

from __future__ import annotations

import warnings
from typing import Any


def _show_deprecation_warning():
    """Show deprecation warning about direct web_ui.py usage."""
    warnings.warn(
        "Importing web_ui.py directly is deprecated. "
        "The monolithic web_ui.py has been refactored into a modular web_ui/ package. "
        "Please use 'from web_ui import main' instead.\n\n"
        "The new package structure:\n"
        "- web_ui.settings: UI constants and helper functions\n"
        "- web_ui.state_manager: Session state management\n"
        "- web_ui.charts: Chart visualization functions\n"
        "- web_ui.signals: Signal management functions\n"
        "- web_ui.callbacks: Event handler callbacks\n\n"
        "For backward compatibility, this module will continue to work, "
        "but please migrate to the new package structure.",
        DeprecationWarning,
        stacklevel=2
    )


# Show deprecation warning when this module is imported
_show_deprecation_warning()

# Import everything from the new web_ui package for backward compatibility
try:
    from web_ui import (
        # All the functions that were originally in web_ui.py
        safe_rerun,
        get_api_credentials_from_secrets,
        format_correlation,
        format_flow,
        ui_key,
        stable_hash,
        num_int,
        num_float,
        format_category_label,
        normalize_factor_category,
        normalize_category_weights,
        safe_float,
        cached_run_automated_signals,
        load_indicator_data,
        calculate_better_volume_indicator,
        create_realtime_candlestick_chart,
        create_candlestick_chart,
        create_multi_timeframe_chart,
        render_weight_controls,
        render_indicator_controls,
        render_signal_risk_controls,
        # Also re-export the modules
        settings,
        state_manager,
        charts,
        signals,
        callbacks,
        SessionStateManager,
        # And all constants
        POPULAR_TOKENS,
        TIMEFRAMES,
        AUTOMATED_SIGNALS_STATE_KEY,
        CACHE_VERSION,
        SYNTHETIC_FLAG_KEYS,
        SYNTHETIC_SOURCE_VALUES,
        SYNTHETIC_MARKER_VALUES,
        FACTOR_CATEGORY_ORDER,
        FACTOR_NAME_TO_CATEGORY,
        FACTOR_CATEGORY_LABELS,
        main,
    )
except ImportError as e:
    raise ImportError(
        f"Failed to import from web_ui package: {e}\n"
        "The modular web_ui package structure may not be properly installed or accessible."
    ) from e


# Re-export everything for compatibility
__all__ = [
    # Core functions
    'main',
    'safe_rerun',
    'get_api_credentials_from_secrets',
    'format_correlation',
    'format_flow',
    'ui_key',
    'stable_hash',
    'num_int',
    'num_float',
    'format_category_label',
    'normalize_factor_category',
    'normalize_category_weights',
    'safe_float',
    'cached_run_automated_signals',
    'load_indicator_data',
    'calculate_better_volume_indicator',
    'create_realtime_candlestick_chart',
    'create_candlestick_chart',
    'create_multi_timeframe_chart',
    'render_weight_controls',
    'render_indicator_controls',
    'render_signal_risk_controls',
    # Modules
    'settings',
    'state_manager',
    'charts',
    'signals',
    'callbacks',
    # Classes
    'SessionStateManager',
    # Constants
    'POPULAR_TOKENS',
    'TIMEFRAMES',
    'AUTOMATED_SIGNALS_STATE_KEY',
    'CACHE_VERSION',
    'SYNTHETIC_FLAG_KEYS',
    'SYNTHETIC_SOURCE_VALUES',
    'SYNTHETIC_MARKER_VALUES',
    'FACTOR_CATEGORY_ORDER',
    'FACTOR_NAME_TO_CATEGORY',
    'FACTOR_CATEGORY_LABELS',
]


# Legacy compatibility - make this module behave exactly like the original
# This allows existing code that imports web_ui.py to continue working

def _legacy_getattr(name: str) -> Any:
    """Provide legacy access to module attributes."""
    if hasattr(globals(), name):
        return globals()[name]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Override __getattr__ for dynamic attribute access
def __getattr__(name: str) -> Any:
    """Get attribute with backward compatibility."""
    # Try the current module first
    try:
        return _legacy_getattr(name)
    except AttributeError:
        pass
    
    # If not found, try importing from web_ui package
    try:
        import web_ui
        if hasattr(web_ui, name):
            return getattr(web_ui, name)
    except ImportError:
        pass
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Legacy execution check
if __name__ == "__main__":
    # Legacy execution - run the main function
    main()