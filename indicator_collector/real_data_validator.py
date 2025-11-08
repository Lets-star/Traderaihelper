"""Real data validation utilities for trading system inputs.

This module provides validation functions to ensure only real market data
is processed by the trading system, rejecting synthetic/mock data.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


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
        "testnet", "paper_trading", "demo_api", "simulator", "backtest_engine", "local"
    }
    
    SYNTHETIC_VALUE_TOKENS = SYNTHETIC_MARKERS.union(SYNTHETIC_SOURCES).union({
        "demo_feed",
        "demo_source",
        "sandbox",
        "simulation",
        "papertrade",
        "paper-trading",
    })
    
    SYNTHETIC_KEY_TOKENS = {"synthetic", "mock", "demo", "sample", "paper", "testnet", "sandbox", "simulation"}
    SYNTHETIC_BOOLEAN_KEYS = {
        "is_synthetic",
        "synthetic_mode",
        "use_mock_data",
        "mock_mode",
        "demo_mode",
        "use_demo_data",
        "is_demo",
        "paper_trading",
        "is_paper_trading",
        "is_testnet",
        "testnet_mode",
        "sandbox_mode",
    }
    SYNTHETIC_SOURCE_KEYS = {"source", "exchange", "provider", "data_source"}
    MAX_FLAG_PATHS = 20
    VALUE_PREVIEW_LIMIT = 80
    
    # Required fields for real data validation
    REQUIRED_SOURCE_FIELDS = {
        "source", "exchange", "timestamp", "granularity"
    }
    
    def __init__(self):
        self.validation_errors: List[str] = []
    
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
        flagged_paths: List[str] = []
        total_flags = 0

        def add_flag(path: str, flag_type: str, value: Any = None) -> None:
            nonlocal total_flags
            total_flags += 1
            if len(flagged_paths) >= self.MAX_FLAG_PATHS:
                return
            if value is not None and not isinstance(value, (dict, list)):
                if isinstance(value, str):
                    value_clean = value.strip()
                    if len(value_clean) > self.VALUE_PREVIEW_LIMIT:
                        value_clean = value_clean[: self.VALUE_PREVIEW_LIMIT - 3] + "..."
                    flagged_paths.append(f"{path}='{value_clean}'")
                else:
                    flagged_paths.append(f"{path}={value!r}")
            elif flag_type == "key":
                flagged_paths.append(f"{path} (key)")
            else:
                flagged_paths.append(path)

        def scan(obj: Any, current_path: str) -> None:
            if isinstance(obj, dict):
                keys_to_remove: List[Any] = []
                for key, value in list(obj.items()):
                    key_str = str(key)
                    key_lower = key_str.lower()
                    next_path = f"{current_path}.{key_str}" if current_path != "$" else f"$." + key_str
                    key_tokens = self._tokenize_text(key_lower)
                    if key_lower in self.SYNTHETIC_BOOLEAN_KEYS or any(
                        token in self.SYNTHETIC_KEY_TOKENS for token in key_tokens
                    ):
                        add_flag(next_path, "key", value if key_lower in self.SYNTHETIC_BOOLEAN_KEYS else None)
                        keys_to_remove.append(key)
                        continue
                    if key_lower in self.SYNTHETIC_SOURCE_KEYS and isinstance(value, str):
                        if self._string_has_synthetic_marker(value):
                            add_flag(next_path, "value", value)
                            keys_to_remove.append(key)
                            continue
                    if isinstance(value, str):
                        if self._string_has_synthetic_marker(value):
                            add_flag(next_path, "value", value)
                    if isinstance(value, (dict, list)):
                        scan(value, next_path)
                for key in keys_to_remove:
                    obj.pop(key, None)
            elif isinstance(obj, list):
                for index, item in enumerate(obj):
                    next_path = f"{current_path}[{index}]"
                    if isinstance(item, str):
                        if self._string_has_synthetic_marker(item):
                            add_flag(next_path, "value", item)
                    if isinstance(item, (dict, list)):
                        scan(item, next_path)

        scan(payload, "$")

        if total_flags:
            display_count = min(len(flagged_paths), self.MAX_FLAG_PATHS)
            preview = ", ".join(flagged_paths[: min(display_count, 5)])
            logger.debug(
                "RealDataValidator detected %s synthetic marker(s): %s",
                total_flags,
                flagged_paths[: display_count],
            )
            raise DataValidationError(
                f"Synthetic data detected: {total_flags} markers found (showing first {display_count}). Paths: {preview}",
                {"flag_count": total_flags, "synthetic_flags": flagged_paths},
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
        if not isinstance(text, str):
            return False
        return self._string_has_synthetic_marker(text)
    
    def _string_has_synthetic_marker(self, text: str) -> bool:
        """Determine whether a string contains synthetic markers."""
        text_lower = text.lower().strip()
        if not text_lower:
            return False
        if text_lower in self.SYNTHETIC_VALUE_TOKENS:
            return True
        tokens = self._tokenize_text(text_lower)
        return any(token in self.SYNTHETIC_VALUE_TOKENS for token in tokens)
    
    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        return [token for token in TOKEN_SPLIT_RE.split(text) if token]
    
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
        # Check that we have consistent data across timeframes
        trend_strength = mtf_data.get("trend_strength", {})
        direction = mtf_data.get("direction", {})
        
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