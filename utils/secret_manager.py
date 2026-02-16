"""
Utility functions for secure credential management using Pydantic Settings.

Provides fallback to environment variables when st.secrets is not available.
This module is now a thin wrapper around the config module for backward compatibility.
"""

import os
import logging
from typing import Optional, Dict, Any

from config import AppSettings

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Manages secure access to credentials via Pydantic Settings with st.secrets fallback.
    
    This class is now a thin wrapper around the config.AppSettings class
    for backward compatibility. New code should use AppSettings directly.
    """

    @staticmethod
    def get_bybit_credentials() -> tuple[Optional[str], Optional[str]]:
        """
        Get ByBit API credentials from Pydantic Settings.

        Returns:
            Tuple of (api_key, api_secret) or (None, None) if not found
        """
        settings = AppSettings.from_secrets()
        return settings.get_bybit_credentials()

    @staticmethod
    def get_bybit_config() -> Dict[str, Any]:
        """
        Get complete ByBit configuration from Pydantic Settings.

        Returns:
            Dictionary with configuration including api_key, api_secret, testnet, etc.
        """
        settings = AppSettings.from_secrets()
        
        if not settings.is_bybit_configured():
            logger.warning("ByBit configuration not found in settings")
            return {}
        
        return {
            "api_key": settings.bybit.api_key,
            "api_secret": settings.bybit.api_secret,
            "testnet": settings.bybit.testnet,
            "default_leverage": settings.bybit.default_leverage,
            "pos_size_multiplier": settings.bybit.pos_size_multiplier,
        }

    @staticmethod
    def validate_credential_format(api_key: str, api_secret: str) -> bool:
        """
        Validate API credential format.

        Args:
            api_key: API key to validate
            api_secret: API secret to validate

        Returns:
            True if credentials appear valid, False otherwise
        """
        if not api_key or not api_secret:
            return False

        if len(api_key) < 10 or len(api_secret) < 10:
            logger.warning("API credentials appear too short")
            return False

        # Basic format check (alphanumeric with some special chars)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            logger.warning("API key contains invalid characters")
            return False

        if not re.match(r'^[a-zA-Z0-9_-]+$', api_secret):
            logger.warning("API secret contains invalid characters")
            return False

        return True

    @staticmethod
    def get_secret(key: str, default: Any = None) -> Any:
        """
        Get a secret value from st.secrets with environment fallback.

        Args:
            key: Secret key (supports dot notation for nested keys, e.g., 'bybit.api_key')
            default: Default value if not found

        Returns:
            Secret value or default
        """
        try:
            import streamlit as st

            if hasattr(st, 'secrets') and st.secrets:
                # Handle nested keys with dot notation
                if '.' in key:
                    parts = key.split('.')
                    value = st.secrets
                    for part in parts:
                        if isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    if value is not None:
                        return value
                else:
                    if key in st.secrets:
                        return st.secrets[key]

        except ImportError:
            logger.debug(f"Streamlit not available for secret: {key}")
        except AttributeError:
            logger.debug(f"st.secrets not available for secret: {key}")
        except Exception as e:
            logger.warning(f"Error accessing st.secrets for key {key}: {e}")

        # Fallback to environment variables
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)

        if env_value is not None:
            logger.debug(f"Using environment variable for {key}")
            return env_value

        logger.debug(f"Secret {key} not found, using default")
        return default

    @staticmethod
    def has_bybit_credentials() -> bool:
        """
        Check if ByBit credentials are available.

        Returns:
            True if credentials are configured, False otherwise
        """
        api_key, api_secret = SecretManager.get_bybit_credentials()
        return bool(api_key and api_secret)

    @staticmethod
    def mask_credential(credential: str, visible_chars: int = 4) -> str:
        """
        Mask a credential for safe logging/display.

        Args:
            credential: Credential to mask
            visible_chars: Number of characters to show at start and end

        Returns:
            Masked credential string
        """
        if not credential or len(credential) <= visible_chars * 2:
            return "***"

        start = credential[:visible_chars]
        end = credential[-visible_chars:]
        masked_length = len(credential) - (visible_chars * 2)
        return f"{start}{'*' * masked_length}{end}"
