"""
Secrets manager for secure configuration management.

This module provides utilities for managing sensitive configuration
with support for Streamlit secrets, environment variables, and validation.

Supported configuration sources (in order of precedence):
1. Streamlit secrets (st.secrets)
2. Environment variables
3. Default values
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional

from models import Credentials

logger = logging.getLogger(__name__)


class SecretsManager:
    """
    Secure secrets manager with multiple source support.

    Manages sensitive configuration with the following priority:
    1. Streamlit secrets (st.secrets) - available in Streamlit apps
    2. Environment variables - available in all environments
    3. Default values - fallback when not configured
    """

    def __init__(self):
        """Initialize secrets manager."""
        self._lock = threading.RLock()
        self._secrets: Dict[str, Any] = {}
        self._st_secrets_available = False

        # Try to import Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets'):
                self._st_secrets_available = True
                logger.info("Streamlit secrets available")
        except ImportError:
            logger.debug("Streamlit not available, using environment variables only")
        except Exception as e:
            logger.warning(f"Failed to initialize Streamlit secrets: {e}")

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get a secret value from available sources.

        Checks sources in order:
        1. Streamlit secrets (st.secrets[key])
        2. Environment variable (os.environ[key])
        3. Default value

        Args:
            key: Secret key (supports nested keys with dots, e.g., "bybit.api_key")
            default: Default value if not found

        Returns:
            Secret value or default
        """
        with self._lock:
            # Try Streamlit secrets first
            if self._st_secrets_available:
                try:
                    import streamlit as st
                    value = self._get_nested(st.secrets, key)
                    if value is not None:
                        logger.debug(f"Found {key} in Streamlit secrets")
                        return value
                except Exception as e:
                    logger.debug(f"Failed to get {key} from Streamlit secrets: {e}")

            # Try environment variables
            env_key = key.upper().replace('.', '_')
            env_value = os.environ.get(env_key)
            if env_value is not None:
                logger.debug(f"Found {key} in environment variables ({env_key})")
                return env_value

            # Return default
            return default

    def _get_nested(self, data: Dict[str, Any], key: str) -> Optional[Any]:
        """
        Get nested value from dictionary using dot notation.

        Args:
            data: Source dictionary
            key: Dot-separated key path

        Returns:
            Nested value or None if not found
        """
        parts = key.split('.')
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def get_credentials(
        self,
        exchange: str = "bybit",
        testnet: bool = True
    ) -> Optional[Credentials]:
        """
        Get API credentials for an exchange.

        Looks for credentials in:
        1. st.secrets.{exchange}.api_key / st.secrets.{exchange}.api_secret
        2. {EXCHANGE}_API_KEY / {EXCHANGE}_API_SECRET environment variables

        Args:
            exchange: Exchange name (e.g., "bybit", "binance")
            testnet: Whether to use testnet

        Returns:
            Credentials object or None if not configured
        """
        api_key = self.get(f"{exchange}.api_key")
        api_secret = self.get(f"{exchange}.api_secret")

        if not api_key or not api_secret:
            logger.warning(f"Credentials not found for exchange: {exchange}")
            return None

        try:
            credentials = Credentials(
                api_key=api_key,
                api_secret=api_secret,
                exchange=exchange,
                testnet=testnet,
            )

            logger.info(f"Loaded credentials for {exchange} (testnet={testnet})")
            return credentials

        except Exception as e:
            logger.error(f"Failed to create credentials for {exchange}: {e}")
            return None

    def get_bybit_credentials(self, testnet: bool = True) -> Optional[Credentials]:
        """
        Get ByBit API credentials.

        Args:
            testnet: Whether to use testnet

        Returns:
            Credentials object or None if not configured
        """
        return self.get_credentials("bybit", testnet)

    def get_binance_credentials(self, testnet: bool = True) -> Optional[Credentials]:
        """
        Get Binance API credentials.

        Args:
            testnet: Whether to use testnet

        Returns:
            Credentials object or None if not configured
        """
        return self.get_credentials("binance", testnet)

    def set(self, key: str, value: Any) -> None:
        """
        Set a secret value in memory (not persisted).

        This is useful for testing or temporary override.

        Args:
            key: Secret key
            value: Secret value
        """
        with self._lock:
            self._secrets[key] = value
            logger.debug(f"Set secret in memory: {key}")

    def get_all(self) -> Dict[str, Any]:
        """
        Get all secrets from all sources.

        Returns:
            Dictionary of all secrets (from all sources combined)
        """
        result: Dict[str, Any] = {}

        # Add in-memory secrets
        with self._lock:
            result.update(self._secrets)

        # Add environment variables (excluding some system ones)
        env_keys_to_exclude = {'PATH', 'HOME', 'USER', 'SHELL', 'PWD'}
        for key, value in os.environ.items():
            if key not in env_keys_to_exclude and not key.startswith('_'):
                result[f"env.{key}"] = value

        return result

    def validate_required_secrets(self, required_keys: List[str]) -> tuple[bool, List[str]]:
        """
        Validate that all required secrets are configured.

        Args:
            required_keys: List of required secret keys

        Returns:
            Tuple of (all_valid, missing_keys)
        """
        missing: List[str] = []

        for key in required_keys:
            value = self.get(key)
            if value is None:
                missing.append(key)

        all_valid = len(missing) == 0

        if not all_valid:
            logger.warning(f"Missing required secrets: {missing}")

        return all_valid, missing

    def get_env_example(self, keys: Optional[List[str]] = None) -> str:
        """
        Generate example .env file content.

        Args:
            keys: List of keys to include (or all if None)

        Returns:
            .env file content as string
        """
        if keys is None:
            keys = [
                "bybit.api_key",
                "bybit.api_secret",
                "binance.api_key",
                "binance.api_secret",
            ]

        lines = ["# TraderAIHelper Environment Variables", ""]
        for key in keys:
            env_key = key.upper().replace('.', '_')
            lines.append(f"# {key}")
            lines.append(f"{env_key}=")
            lines.append("")

        return "\n".join(lines)


# Global secrets manager instance
_global_secrets_manager: Optional[SecretsManager] = None
_global_manager_lock = threading.Lock()


def get_secrets_manager() -> SecretsManager:
    """
    Get the global secrets manager instance.

    Returns:
        SecretsManager instance
    """
    global _global_secrets_manager

    with _global_manager_lock:
        if _global_secrets_manager is None:
            _global_secrets_manager = SecretsManager()
            logger.info("Initialized global secrets manager")

        return _global_secrets_manager


def get_secret(key: str, default: Optional[Any] = None) -> Optional[Any]:
    """
    Convenience function to get a secret value.

    Args:
        key: Secret key
        default: Default value if not found

    Returns:
        Secret value or default
    """
    return get_secrets_manager().get(key, default)


def get_credentials(
    exchange: str = "bybit",
    testnet: bool = True
) -> Optional[Credentials]:
    """
    Convenience function to get API credentials.

    Args:
        exchange: Exchange name
        testnet: Whether to use testnet

    Returns:
        Credentials object or None if not configured
    """
    return get_secrets_manager().get_credentials(exchange, testnet)
