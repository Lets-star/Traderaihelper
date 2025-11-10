"""Trading system payload loader and processor.

This module provides functions to automatically load and process complete JSON payloads
for the trading system with real data validation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Union

from ..real_data_validator import (
    DataValidationError,
    RealDataValidator,
    load_and_validate_json_payload,
)
from .interfaces import (
    AnalyzerContext,
    TradingSignalPayload,
    parse_collector_payload,
    deserialize_signal_payload,
)
from .backtester import ParameterSet
from .signal_generator import SignalConfig, SignalGenerator, generate_trading_signal
from .position_manager import create_position_plan
from .statistics_optimizer import StatisticsOptimizer
from ..timeframes import Timeframe, timeframe_params


class PayloadProcessor:
    """Processes and routes trading payloads through the trading system."""
    
    def __init__(self):
        self.validator = RealDataValidator()
        self.signal_generator = SignalGenerator()
        self.statistics_optimizer = StatisticsOptimizer()
    
    def load_full_payload(
        self,
        json_data: Union[str, Dict[str, Any]],
        timeframe: Optional[str] = None,
        validate_real_data: bool = True,
        signal_config: Optional[SignalConfig] = None,
        indicator_params: Optional[Dict[str, Any]] = None,
        parameter_set: Optional[ParameterSet] = None,
    ) -> TradingSignalPayload:
        """
        Automatically load, validate, and process a complete JSON payload.
        
        Args:
            json_data: JSON string or dictionary containing trading data
            timeframe: Trading timeframe (extracted from payload if not provided)
            validate_real_data: Whether to validate real data constraints
            signal_config: Optional signal configuration overrides (weights, thresholds)
            indicator_params: Optional indicator parameter overrides keyed by indicator
            parameter_set: Optional ParameterSet propagated to analyzers
            
        Returns:
            Processed TradingSignalPayload with complete analysis
            
        Raises:
            DataValidationError: If real data validation fails
            ValueError: If payload format is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        # Load and parse JSON
        if isinstance(json_data, str):
            try:
                payload_dict = json.loads(json_data)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(f"Invalid JSON: {e.msg}", e.doc, e.pos)
        else:
            payload_dict = json_data
        
        # Extract timeframe if not provided
        if timeframe is None:
            metadata = payload_dict.get("metadata", {})
            timeframe = metadata.get("timeframe")
            if not timeframe:
                latest = payload_dict.get("latest", {})
                timeframe = latest.get("timeframe")
            
            if not timeframe:
                raise ValueError("Timeframe not found in payload and not provided as parameter")
        
        # Validate timeframe
        if not Timeframe.is_supported(timeframe):
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Real data validation if enabled
        if validate_real_data:
            self.validator.validate_payload_sources(payload_dict)
            self.validator.ensure_no_synthetic_flags(payload_dict)
            self.validator.validate_time_continuity(payload_dict, timeframe)
        
        # Parse into AnalyzerContext
        try:
            context = parse_collector_payload(payload_dict)
        except Exception as e:
            raise ValueError(f"Failed to parse collector payload: {e}")
        
        # Update context with timeframe-specific parameters
        self._apply_timeframe_parameters(context, timeframe)

        if indicator_params:
            if not context.extras:
                context.extras = {}
            existing_params = context.extras.get("indicator_params")
            merged_params = dict(existing_params) if isinstance(existing_params, dict) else {}
            merged_params.update(indicator_params)
            context.extras["indicator_params"] = merged_params
        
        # Generate trading signal
        try:
            signal_payload = self.signal_generator.analyze(
                context,
                config=signal_config,
                indicator_params=indicator_params,
            )
        except Exception as e:
            raise ValueError(f"Signal generation failed: {e}")
        
        # Enhance with position management
        try:
            # Create position plan from signal and context
            from .position_manager import PositionManagerConfig
            config = PositionManagerConfig()
            position_result = create_position_plan(
                context=context,
                signal_direction="long" if signal_payload.signal_type == "BUY" else "short",
                config=config
            )
            if position_result and position_result.position_plan:
                signal_payload.position_plan = position_result.position_plan
        except Exception as e:
            # Position management is optional, log but continue
            print(f"[warning] Position management failed: {e}")
        
        # Apply statistical optimization if historical data available
        try:
            if context.historical_signals:
                optimization_stats = self.statistics_optimizer.optimize([context])
                if optimization_stats:
                    signal_payload.optimization_stats = optimization_stats
        except Exception as e:
            # Optimization is optional, log but continue
            print(f"[warning] Statistical optimization failed: {e}")
        
        # Add processing metadata
        signal_payload.metadata.update({
            "payload_processor": "PayloadProcessor",
            "timeframe_used": timeframe,
            "real_data_validated": validate_real_data,
            "processing_timestamp": context.timestamp,
            "source_data_quality": payload_dict.get("metadata", {}).get("data_quality", "unknown"),
        })
        
        return signal_payload
    
    def _apply_timeframe_parameters(self, context: AnalyzerContext, timeframe: str) -> None:
        """Apply timeframe-specific parameters to the context."""
        try:
            # Get timeframe-specific parameters
            params = timeframe_params.get_parameters(timeframe)
            
            # Update context with timeframe parameters
            if not context.extras:
                context.extras = {}
            
            context.extras["timeframe_parameters"] = params
            
            # Update metadata
            if not context.metadata:
                context.metadata = {}
            
            context.metadata.update({
                "timeframe": timeframe,
                "timeframe_minutes": Timeframe.to_minutes(timeframe),
                "timeframe_display": Timeframe.from_value(timeframe).get_display_name(),
            })
            
        except Exception as e:
            print(f"[warning] Failed to apply timeframe parameters: {e}")
    
    def process_payload_to_dict(self, json_data: Union[str, Dict[str, Any]], 
                              timeframe: Optional[str] = None,
                              validate_real_data: bool = True) -> Dict[str, Any]:
        """
        Process payload and return as dictionary for backward compatibility.
        
        Args:
            json_data: JSON string or dictionary
            timeframe: Trading timeframe
            validate_real_data: Whether to validate real data
            
        Returns:
            Processed payload as dictionary
        """
        signal_payload = self.load_full_payload(json_data, timeframe, validate_real_data)
        return signal_payload.to_dict()


# Global processor instance for easy access
payload_processor = PayloadProcessor()


def load_full_payload(
    json_data: Union[str, Dict[str, Any]],
    timeframe: Optional[str] = None,
    validate_real_data: bool = True,
    signal_config: Optional[SignalConfig] = None,
    indicator_params: Optional[Dict[str, Any]] = None,
) -> TradingSignalPayload:
    """Convenience function to load and process a complete JSON payload.

    Args:
        json_data: JSON string or dictionary containing trading data
        timeframe: Trading timeframe (extracted from payload if not provided)
        validate_real_data: Whether to validate real data constraints
        signal_config: Optional signal configuration overrides
        indicator_params: Optional indicator parameter overrides (e.g., MACD/RSI/ATR)

    Returns:
        Processed TradingSignalPayload with complete analysis

    Raises:
        DataValidationError: If real data validation fails
        ValueError: If payload format is invalid
    """
    return payload_processor.load_full_payload(
        json_data,
        timeframe,
        validate_real_data,
        signal_config=signal_config,
        indicator_params=indicator_params,
    )


def load_and_process_payload_dict(
    json_data: Union[str, Dict[str, Any]],
    timeframe: Optional[str] = None,
    validate_real_data: bool = True,
    signal_config: Optional[SignalConfig] = None,
    indicator_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function to load and process payload as dictionary."""
    payload = payload_processor.load_full_payload(
        json_data,
        timeframe,
        validate_real_data,
        signal_config=signal_config,
        indicator_params=indicator_params,
    )
    return payload.to_dict()


def validate_and_normalize_payload(json_data: Union[str, Dict[str, Any]], 
                               timeframe: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate and normalize payload without full processing.
    
    Args:
        json_data: JSON string or dictionary
        timeframe: Trading timeframe
        
    Returns:
        Validated and normalized payload dictionary
    """
    # Load JSON if needed
    if isinstance(json_data, str):
        payload_dict = json.loads(json_data)
    else:
        payload_dict = json_data
    
    # Extract timeframe if not provided
    if timeframe is None:
        metadata = payload_dict.get("metadata", {})
        timeframe = metadata.get("timeframe")
        if not timeframe:
            latest = payload_dict.get("latest", {})
            timeframe = latest.get("timeframe")
    
    if timeframe:
        # Validate timeframe
        if not Timeframe.is_supported(timeframe):
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Basic real data validation
        validator = RealDataValidator()
        validator.validate_payload_sources(payload_dict)
        validator.validate_time_continuity(payload_dict, timeframe)
    
    return payload_dict


def extract_trading_context(payload_dict: Dict[str, Any]) -> AnalyzerContext:
    """
    Extract trading context from payload dictionary.
    
    Args:
        payload_dict: Payload dictionary
        
    Returns:
        AnalyzerContext for further processing
    """
    return parse_collector_payload(payload_dict)