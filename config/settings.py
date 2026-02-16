"""
Pydantic Settings for centralized configuration management.

This module provides type-safe configuration with:
- Environment variable loading
- Validation for credentials and trading parameters
- st.secrets integration
- Sensible defaults for all settings
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings, SettingsConfigDict


class BaseSettings(PydanticBaseSettings):
    """Base settings class with common configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class ByBitSettings(BaseSettings):
    """ByBit API and trading configuration."""

    model_config = SettingsConfigDict(env_prefix="BYBIT_")

    api_key: Optional[str] = Field(
        default=None,
        min_length=10,
        description="ByBit API key",
    )
    api_secret: Optional[str] = Field(
        default=None,
        min_length=10,
        description="ByBit API secret",
    )
    testnet: bool = Field(
        default=True,
        description="Use ByBit testnet",
    )
    default_leverage: int = Field(
        default=5,
        ge=1,
        le=125,
        description="Default leverage for positions",
    )
    pos_size_multiplier: float = Field(
        default=1.0,
        gt=0,
        le=10,
        description="Position size multiplier",
    )
    recv_window_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Request receive window in milliseconds",
    )

    @field_validator("api_key", "api_secret")
    @classmethod
    def validate_credential_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate credential format."""
        if v is None:
            return v
        if len(v) < 10:
            raise ValueError("API credential must be at least 10 characters")
        # Check for valid characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("API credential contains invalid characters")
        return v

    @field_validator("api_key", "api_secret")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Strip whitespace from credentials."""
        if v is not None:
            return v.strip()
        return v

    @model_validator(mode="after")
    def check_credentials_match(self) -> Self:
        """Ensure both credentials are provided or both are None."""
        if (self.api_key is None) != (self.api_secret is None):
            raise ValueError("Both api_key and api_secret must be provided together")
        return self

    def is_configured(self) -> bool:
        """Check if both API credentials are configured."""
        return self.api_key is not None and self.api_secret is not None

    def get_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get API credentials as tuple."""
        return self.api_key, self.api_secret


class TradingSettings(BaseSettings):
    """Trading strategy and risk management settings."""

    model_config = SettingsConfigDict(env_prefix="TRADING_")

    max_leverage: int = Field(
        default=20,
        ge=1,
        le=125,
        description="Maximum allowed leverage",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum signal confidence threshold",
    )
    default_position_size: float = Field(
        default=0.001,
        gt=0,
        description="Default position size in base currency",
    )
    max_position_size: float = Field(
        default=1.0,
        gt=0,
        description="Maximum position size in base currency",
    )
    risk_per_trade_pct: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Risk percentage per trade",
    )
    stop_loss_atr_multiplier: float = Field(
        default=2.0,
        gt=0,
        le=5.0,
        description="ATR multiplier for stop loss calculation",
    )
    take_profit_atr_multiplier: float = Field(
        default=3.0,
        gt=0,
        le=10.0,
        description="ATR multiplier for take profit calculation",
    )
    enable_trailing_stop: bool = Field(
        default=True,
        description="Enable trailing stop feature",
    )
    trailing_stop_activation_pct: float = Field(
        default=1.0,
        gt=0,
        le=5.0,
        description="Profit percentage to activate trailing stop",
    )
    dry_run: bool = Field(
        default=False,
        description="Execute trades in dry-run mode (no real orders)",
    )


class UISettings(BaseSettings):
    """Streamlit UI configuration."""

    model_config = SettingsConfigDict(env_prefix="UI_")

    chart_default_timeframe: str = Field(
        default="1h",
        pattern=r"^[1-9][0-9]*[mhdw]$",
        description="Default chart timeframe",
    )
    chart_default_bars: int = Field(
        default=200,
        ge=10,
        le=1000,
        description="Default number of bars to display",
    )
    chart_theme: str = Field(
        default="plotly_dark",
        description="Chart color theme",
    )
    enable_auto_refresh: bool = Field(
        default=True,
        description="Enable auto-refresh for charts",
    )
    auto_refresh_interval_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Auto-refresh interval in seconds",
    )
    show_forming_bar: bool = Field(
        default=True,
        description="Show forming (incomplete) candle",
    )
    max_concurrent_workers: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent background workers",
    )
    popular_tokens: List[str] = Field(
        default=[
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOGEUSDT",
        ],
        description="List of popular trading tokens",
    )
    available_timeframes: List[str] = Field(
        default=["1m", "5m", "15m", "1h", "4h", "1d"],
        description="Available chart timeframes",
    )

    @field_validator("chart_default_timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format."""
        import re
        if not re.match(r"^[1-9][0-9]*[mhdw]$", v):
            raise ValueError("Timeframe must match pattern like 1m, 5m, 1h, 4h, 1d, 1w")
        return v


class WorkerSettings(BaseSettings):
    """Background worker configuration."""

    model_config = SettingsConfigDict(env_prefix="WORKER_")

    websocket_reconnect_max: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum WebSocket reconnection attempts",
    )
    websocket_backoff_ms: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Initial WebSocket reconnection backoff in ms",
    )
    websocket_max_backoff_ms: int = Field(
        default=30000,
        ge=1000,
        le=60000,
        description="Maximum WebSocket reconnection backoff in ms",
    )
    signal_execution_timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Signal execution timeout in seconds",
    )
    max_worker_threads: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum worker threads for signal execution",
    )
    worker_poll_interval_short_ms: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Worker poll interval for short timeframes (<=15m)",
    )
    worker_poll_interval_long_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Worker poll interval for long timeframes (>=1h)",
    )
    enable_websocket: bool = Field(
        default=True,
        description="Enable WebSocket connections for real-time data",
    )


class CacheSettings(BaseSettings):
    """Cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    ttl_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Default cache TTL in seconds",
    )
    max_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum cache entries",
    )
    enable_persistence: bool = Field(
        default=False,
        description="Enable cache persistence to disk",
    )
    persistence_path: str = Field(
        default=".cache",
        description="Cache persistence directory",
    )
    klines_cache_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum klines cache entries per symbol/timeframe",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    level: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    format: str = Field(
        default="structured",
        description="Log format: structured or simple",
    )
    enable_file_logging: bool = Field(
        default=True,
        description="Enable logging to file",
    )
    log_file: str = Field(
        default="trading.log",
        description="Log file path",
    )
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum log file size in MB",
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of backup log files to keep",
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics collection",
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v_upper


class AppSettings(BaseSettings):
    """Root application settings container."""

    model_config = SettingsConfigDict(
        env_prefix="TRADER_",
        env_nested_delimiter="__",
    )

    bybit: ByBitSettings = Field(default_factory=ByBitSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    ui: UISettings = Field(default_factory=UISettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    app_name: str = Field(
        default="Trader AI Helper",
        description="Application name",
    )
    version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    @classmethod
    def from_secrets(cls) -> "AppSettings":
        """Create settings from Streamlit secrets."""
        try:
            import streamlit as st

            secrets_dict: Dict[str, Any] = {}

            # Try to get bybit credentials
            if hasattr(st, 'secrets') and st.secrets:
                bybit_secrets = st.secrets.get("bybit", {})
                if bybit_secrets:
                    secrets_dict["bybit"] = {
                        "api_key": bybit_secrets.get("api_key"),
                        "api_secret": bybit_secrets.get("api_secret"),
                        "testnet": bybit_secrets.get("testnet", True),
                        "default_leverage": bybit_secrets.get("default_leverage", 5),
                        "pos_size_multiplier": bybit_secrets.get("pos_size_multiplier", 1.0),
                    }

                # Direct environment variables
                if "BYBIT_API_KEY" in st.secrets:
                    if "bybit" not in secrets_dict:
                        secrets_dict["bybit"] = {}
                    secrets_dict["bybit"]["api_key"] = st.secrets["BYBIT_API_KEY"]
                if "BYBIT_API_SECRET" in st.secrets:
                    if "bybit" not in secrets_dict:
                        secrets_dict["bybit"] = {}
                    secrets_dict["bybit"]["api_secret"] = st.secrets["BYBIT_API_SECRET"]

            return cls(**secrets_dict)

        except ImportError:
            # Streamlit not available, use environment variables
            return cls()

    def get_bybit_credentials(self) -> tuple[Optional[str], Optional[str]]:
        """Get ByBit API credentials."""
        return self.bybit.get_credentials()

    def is_bybit_configured(self) -> bool:
        """Check if ByBit is properly configured."""
        return self.bybit.is_configured()


# Global settings instance (lazy-loaded)
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = AppSettings.from_secrets()
    return _settings


def reload_settings() -> AppSettings:
    """Reload settings from environment/secrets."""
    global _settings
    _settings = AppSettings.from_secrets()
    return _settings
