"""Real data validation utilities for trading system inputs.

This module provides validation functions to ensure only real market data
is processed by the trading system, rejecting synthetic/mock data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


logger = logging.getLogger(__name__)


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes (local implementation to avoid circular imports)."""
    mapping = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "3h": 180,
        "4h": 240,
        "6h": 360,
        "12h": 720,
        "1d": 1440,
        "1w": 10080,
    }
    return mapping.get(timeframe, 0)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class DataSource(Enum):
    """Supported data sources."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BITFINEX = "bitfinex"
    UNKNOWN = "unknown"


class RealDataValidator:
    """Validates that trading payloads contain only real market data."""
    
    # Known synthetic/mock data markers
    SYNTHETIC_MARKERS = {
        "mock", "test", "demo", "simulated", "synthetic", "fake", "sample",
        "paper", "backtest", "historical_sim", "generated", "artificial"
    }
    
    # Known synthetic data sources
    SYNTHETIC_SOURCES = {
        "testnet", "paper_trading", "demo_api", "simulator", "backtest_engine"
    }
    
    # Required fields for real data validation
    REQUIRED_SOURCE_FIELDS = {
        "source", "exchange", "timestamp", "granularity"
    }
    
    def __init__(self, marker_preview_limit: int = 3):
        self.validation_errors: List[str] = []
        self.marker_preview_limit = max(0, marker_preview_limit)
    
    def validate_payload_sources(self, payload: Dict[str, Any]) -> bool:
        """
        Validate that payload contains proper source metadata.
        
        Args:
            payload: Trading signal payload dictionary
            
        Returns:
            True if all sources are valid real data sources
            
        Raises:
            DataValidationError: If validation fails
        """
        self.validation_errors.clear()
        
        # Check metadata section
        metadata = payload.get("metadata", {})
        if not metadata:
            raise DataValidationError("Missing metadata section in payload")
        
        # Validate source information
        source = metadata.get("source", "").lower()
        exchange = metadata.get("exchange", "").lower()
        
        if not source or not exchange:
            raise DataValidationError(
                "Missing source or exchange information in metadata",
                {"source": source, "exchange": exchange}
            )
        
        # Check for synthetic markers in source/exchange
        if self._contains_synthetic_markers(source) or self._contains_synthetic_markers(exchange):
            raise DataValidationError(
                f"Synthetic data detected: source={source}, exchange={exchange}",
                {"source": source, "exchange": exchange}
            )
        
        # Validate timestamp
        timestamp = metadata.get("timestamp", 0)
        if not self._is_valid_timestamp(timestamp):
            raise DataValidationError(
                f"Invalid timestamp: {timestamp}",
                {"timestamp": timestamp}
            )
        
        # Validate data freshness and continuity
        latest_data = payload.get("latest", {})
        if latest_data:
            latest_timestamp = latest_data.get("timestamp", 0)
            if not self._is_valid_timestamp(latest_timestamp):
                raise DataValidationError(
                    f"Invalid latest timestamp: {latest_timestamp}",
                    {"latest_timestamp": latest_timestamp}
                )
            
            # Check timestamp continuity
            if abs(timestamp - latest_timestamp) > 300000:  # 5 minutes in ms
                self.validation_errors.append(
                    f"Large timestamp gap: metadata={timestamp}, latest={latest_timestamp}"
                )
        
        # Validate OHLCV data if present
        self._validate_ohlcv_data(latest_data)
        
        # Validate orderbook data if present
        orderbook = payload.get("orderbook", {})
        if orderbook:
            self._validate_orderbook_data(orderbook)
        
        # Validate multi-timeframe data if present
        mtf_data = payload.get("multi_timeframe", {})
        if mtf_data:
            self._validate_multitimeframe_data(mtf_data)
        
        # Check for synthetic flags throughout payload
        self.ensure_no_synthetic_flags(payload)
        
        if self.validation_errors:
            raise DataValidationError(
                f"Validation completed with {len(self.validation_errors)} warnings",
                {"warnings": self.validation_errors}
            )
        
        return True
    
    def ensure_no_synthetic_flags(self, payload: Dict[str, Any]) -> bool:
        """
        Scan entire payload for synthetic data markers.
        
        Args:
            payload: Trading signal payload dictionary
            
        Returns:
            True if no synthetic markers found
            
        Raises:
            DataValidationError: If synthetic markers detected
        """
        synthetic_flags_found = []
        
        def _scan_for_markers(obj: Any, path: str = "") -> None:
            if isinstance(obj, str):
                if self._contains_synthetic_markers(obj):
                    synthetic_flags_found.append(f"{path}: '{obj}'")
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    _scan_for_markers(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    _scan_for_markers(item, new_path)
        
        _scan_for_markers(payload)
        
        if synthetic_flags_found:
            preview = synthetic_flags_found[: self.marker_preview_limit]
            preview_text = ", ".join(preview)
            message = f"Synthetic data markers detected: {len(synthetic_flags_found)} instances"
            if preview:
                message += f" (e.g., {preview_text})"
            logger.debug("Synthetic data markers detected: %s", preview_text or "no preview")
            raise DataValidationError(
                message,
                {
                    "synthetic_flags": synthetic_flags_found,
                    "preview": preview,
                },
            )
        
        return True
    
    def validate_time_continuity(self, payload: Dict[str, Any], timeframe: str) -> bool:
        """
        Validate timestamp continuity and plausibility for given timeframe.
        
        Args:
            payload: Trading signal payload dictionary
            timeframe: Trading timeframe (e.g., "1m", "5m", "1h", "3h")
            
        Returns:
            True if time continuity is valid
            
        Raises:
            DataValidationError: If time continuity issues detected
        """
        timeframe_minutes = timeframe_to_minutes(timeframe)
        timeframe_ms = timeframe_minutes * 60 * 1000
        
        # Check metadata timestamp
        metadata = payload.get("metadata", {})
        metadata_timestamp = metadata.get("timestamp", 0)
        
        # Check latest data timestamp
        latest = payload.get("latest", {})
        latest_timestamp = latest.get("timestamp", 0)
        
        # Validate timestamp ranges
        current_time = datetime.now().timestamp() * 1000
        
        if metadata_timestamp > current_time + 60000:  # 1 minute future tolerance
            raise DataValidationError(
                f"Metadata timestamp is in the future: {metadata_timestamp}",
                {"metadata_timestamp": metadata_timestamp, "current_time": current_time}
            )
        
        if latest_timestamp > current_time + 60000:
            raise DataValidationError(
                f"Latest data timestamp is in the future: {latest_timestamp}",
                {"latest_timestamp": latest_timestamp, "current_time": current_time}
            )
        
        # Check for stale data (older than 24 hours)
        stale_threshold = current_time - (24 * 60 * 60 * 1000)  # 24 hours ago
        
        if latest_timestamp < stale_threshold:
            raise DataValidationError(
                f"Data is too old: {datetime.fromtimestamp(latest_timestamp/1000)}",
                {"latest_timestamp": latest_timestamp, "stale_threshold": stale_threshold}
            )
        
        # Validate timeframe alignment
        if latest_timestamp > 0:
            # Check if timestamp aligns with timeframe boundaries
            timeframe_start = (latest_timestamp // timeframe_ms) * timeframe_ms
            
            # Allow some tolerance for real-world data
            tolerance = timeframe_ms // 10  # 10% of timeframe
            if abs(latest_timestamp - timeframe_start) > tolerance:
                self.validation_errors.append(
                    f"Timestamp not aligned with timeframe: {latest_timestamp}, "
                    f"expected near {timeframe_start} for {timeframe}"
                )
        
        # Check multi-timeframe continuity if present
        mtf_data = payload.get("multi_timeframe", {})
        if mtf_data:
            self._validate_mtf_time_continuity(mtf_data, current_time)
        
        if self.validation_errors:
            raise DataValidationError(
                f"Time continuity validation completed with {len(self.validation_errors)} warnings",
                {"warnings": self.validation_errors}
            )
        
        return True
    
    def _contains_synthetic_markers(self, text: str) -> bool:
        """Check if text contains synthetic data markers."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.SYNTHETIC_MARKERS)
    
    def _is_valid_timestamp(self, timestamp: Union[int, float]) -> bool:
        """Check if timestamp is plausible."""
        if not isinstance(timestamp, (int, float)):
            return False
        
        # Check if timestamp is in reasonable range (2020-2030)
        min_timestamp = datetime(2020, 1, 1).timestamp() * 1000
        max_timestamp = datetime(2030, 1, 1).timestamp() * 1000
        
        return min_timestamp <= timestamp <= max_timestamp
    
    def _validate_ohlcv_data(self, ohlcv: Dict[str, Any]) -> None:
        """Validate OHLCV data for plausibility."""
        required_fields = {"open", "high", "low", "close", "volume"}
        
        for field in required_fields:
            if field not in ohlcv:
                self.validation_errors.append(f"Missing OHLCV field: {field}")
                continue
            
            value = ohlcv[field]
            if not isinstance(value, (int, float)) or value < 0:
                self.validation_errors.append(f"Invalid {field} value: {value}")
        
        # Validate OHLC relationships
        if all(field in ohlcv for field in ["open", "high", "low", "close"]):
            o, h, l, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
            
            if not (l <= o <= h and l <= c <= h):
                self.validation_errors.append(
                    f"OHLC relationship violation: O={o}, H={h}, L={l}, C={c}"
                )
            
            # Check for zero prices
            if any(price == 0 for price in [o, h, l, c]):
                self.validation_errors.append("Zero price detected in OHLC data")
    
    def _validate_orderbook_data(self, orderbook: Dict[str, Any]) -> None:
        """Validate orderbook data."""
        source = orderbook.get("source", "").lower()
        if self._contains_synthetic_markers(source):
            raise DataValidationError(
                f"Synthetic orderbook source detected: {source}",
                {"orderbook_source": source}
            )
        
        # Validate bids/asks structure
        bids = orderbook.get("raw_levels", {}).get("bids", [])
        asks = orderbook.get("raw_levels", {}).get("asks", [])
        
        if not bids or not asks:
            self.validation_errors.append("Empty orderbook bids or asks")
            return
        
        # Validate bid/ask price ordering
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        
        if best_bid and best_ask and best_bid >= best_ask:
            self.validation_errors.append(
                f"Invalid bid-ask spread: bid={best_bid}, ask={best_ask}"
            )
    
    def _validate_multitimeframe_data(self, mtf_data: Dict[str, Any]) -> None:
        """Validate multi-timeframe data consistency."""
        validated_candles = False
        candles_section = mtf_data.get("candles")
        if isinstance(candles_section, dict) and candles_section:
            validated_candles = True
            for tf, entries in candles_section.items():
                if not isinstance(entries, list) or not entries:
                    self.validation_errors.append(
                        f"Multi-timeframe candles missing or empty for timeframe: {tf}"
                    )
                    continue
                previous_ts: Optional[int] = None
                for candle in entries:
                    if not isinstance(candle, dict):
                        self.validation_errors.append(
                            f"Invalid candle structure for timeframe {tf}: {candle!r}"
                        )
                        break
                    ts = candle.get("ts")
                    if not isinstance(ts, (int, float)) or not self._is_valid_timestamp(ts):
                        self.validation_errors.append(
                            f"Invalid timestamp in multi-timeframe candles for {tf}: {ts}"
                        )
                        break
                    if previous_ts is not None and ts <= previous_ts:
                        self.validation_errors.append(
                            f"Non-monotonic timestamps in multi-timeframe candles for {tf}"
                        )
                        break
                    previous_ts = int(ts)
                    for price_field in ("open", "high", "low", "close"):
                        value = candle.get(price_field)
                        if value is not None and (not isinstance(value, (int, float)) or value < 0):
                            self.validation_errors.append(
                                f"Invalid {price_field} in multi-timeframe candles for {tf}: {value}"
                            )
                            break
                volume_value = entries[-1].get("volume") if entries else None
                if volume_value is not None and (not isinstance(volume_value, (int, float)) or volume_value < 0):
                    self.validation_errors.append(
                        f"Invalid volume in multi-timeframe candles for {tf}: {volume_value}"
                    )
        
        trend_strength = mtf_data.get("trend_strength", {})
        direction = mtf_data.get("direction", {})
        
        if trend_strength or direction:
            for tf in trend_strength.keys():
                if tf not in direction:
                    self.validation_errors.append(
                        f"Missing direction data for timeframe: {tf}"
                    )
                
                strength = trend_strength[tf]
                if not isinstance(strength, (int, float)) or not (0 <= strength <= 100):
                    self.validation_errors.append(
                        f"Invalid trend strength for {tf}: {strength}"
                    )
        elif not validated_candles:
            self.validation_errors.append("Multi-timeframe data missing candles and trend strength information")
    
    def _validate_mtf_time_continuity(self, mtf_data: Dict[str, Any], current_time: float) -> None:
        """Validate time continuity in multi-timeframe data."""
        # This would check that MTF data timestamps are consistent
        # For now, just check if MTF data exists and has reasonable structure
        if not isinstance(mtf_data, dict):
            self.validation_errors.append("Multi-timeframe data is not a dictionary")
            return
        
        # Check for timestamp fields in MTF data
        for key, value in mtf_data.items():
            if isinstance(value, dict) and "timestamp" in value:
                timestamp = value["timestamp"]
                if not self._is_valid_timestamp(timestamp):
                    self.validation_errors.append(
                        f"Invalid timestamp in MTF data {key}: {timestamp}"
                    )


def validate_real_data_payload(payload: Dict[str, Any], timeframe: str) -> bool:
    """
    Convenience function to validate a complete payload for real data.
    
    Args:
        payload: Trading signal payload dictionary
        timeframe: Trading timeframe
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    validator = RealDataValidator()
    
    # Validate sources and metadata
    validator.validate_payload_sources(payload)
    
    # Ensure no synthetic flags
    validator.ensure_no_synthetic_flags(payload)
    
    # Validate time continuity
    validator.validate_time_continuity(payload, timeframe)
    
    return True


def load_and_validate_json_payload(json_data: Union[str, Dict[str, Any]], timeframe: str) -> Dict[str, Any]:
    """
    Load JSON data and validate it contains only real data.
    
    Args:
        json_data: JSON string or dictionary
        timeframe: Trading timeframe
        
    Returns:
        Validated payload dictionary
        
    Raises:
        DataValidationError: If validation fails
        json.JSONDecodeError: If JSON is invalid
    """
    if isinstance(json_data, str):
        payload = json.loads(json_data)
    else:
        payload = json_data
    
    validate_real_data_payload(payload, timeframe)
    return payload