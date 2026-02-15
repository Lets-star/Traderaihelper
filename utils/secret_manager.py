"""
Utility functions for secure credential management using st.secrets.

Provides fallback to environment variables when st.secrets is not available.
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Manages secure access to credentials via st.secrets with environment fallback.
    """

    @staticmethod
    def get_bybit_credentials() -> tuple[Optional[str], Optional[str]]:
        """
        Get ByBit API credentials from st.secrets or environment.

        Returns:
            Tuple of (api_key, api_secret) or (None, None) if not found
        """
        try:
            import streamlit as st

            # Try st.secrets first - direct keys
            if hasattr(st, 'secrets') and st.secrets:
                api_key = st.secrets.get("BYBIT_API_KEY")
                api_secret = st.secrets.get("BYBIT_API_SECRET")

                if api_key and api_secret:
                    logger.info("Using ByBit credentials from st.secrets (direct)")
                    return api_key, api_secret

                # Try bybit section in secrets
                api_key = st.secrets.get("bybit", {}).get("api_key")
                api_secret = st.secrets.get("bybit", {}).get("api_secret")

                if api_key and api_secret:
                    logger.info("Using ByBit credentials from st.secrets[bybit]")
                    return api_key, api_secret

        except ImportError:
            logger.debug("Streamlit not available, using environment variables")
        except AttributeError:
            logger.debug("st.secrets not available, using environment variables")
        except KeyError:
            logger.debug("ByBit credentials not in st.secrets, using environment variables")
        except Exception as e:
            logger.warning(f"Error accessing st.secrets: {e}")

        # Fallback to environment variables
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if api_key and api_secret:
            logger.info("Using ByBit credentials from environment variables")
            return api_key, api_secret

        logger.warning("ByBit credentials not found in st.secrets or environment")
        return None, None

    @staticmethod
    def get_bybit_config() -> Dict[str, Any]:
        """
        Get complete ByBit configuration from st.secrets or environment.

        Returns:
            Dictionary with configuration including api_key, api_secret, testnet, etc.
        """
        config: Dict[str, Any] = {}

        try:
            import streamlit as st

            if hasattr(st, 'secrets') and st.secrets:
                # Try direct keys first
                if "BYBIT_API_KEY" in st.secrets and "BYBIT_API_SECRET" in st.secrets:
                    config["api_key"] = st.secrets["BYBIT_API_KEY"]
                    config["api_secret"] = st.secrets["BYBIT_API_SECRET"]
                    config["testnet"] = st.secrets.get("BYBIT_TESTNET", True)
                    config["default_leverage"] = st.secrets.get("BYBIT_DEFAULT_LEVERAGE", 5)
                    config["pos_size_multiplier"] = st.secrets.get("BYBIT_POS_SIZE_MULTIPLIER", 1.0)
                    logger.info("Using ByBit config from st.secrets (direct)")
                    return config

                # Try bybit section
                bybit_section = st.secrets.get("bybit")
                if bybit_section and isinstance(bybit_section, dict):
                    config["api_key"] = bybit_section.get("api_key")
                    config["api_secret"] = bybit_section.get("api_secret")
                    config["testnet"] = bybit_section.get("testnet", True)
                    config["default_leverage"] = bybit_section.get("default_leverage", 5)
                    config["pos_size_multiplier"] = bybit_section.get("pos_size_multiplier", 1.0)
                    logger.info("Using ByBit config from st.secrets[bybit]")
                    return config

        except ImportError:
            logger.debug("Streamlit not available, using environment variables")
        except AttributeError:
            logger.debug("st.secrets not available, using environment variables")
        except Exception as e:
            logger.warning(f"Error accessing st.secrets: {e}")

        # Fallback to environment variables
        config["api_key"] = os.getenv("BYBIT_API_KEY")
        config["api_secret"] = os.getenv("BYBIT_API_SECRET")
        config["testnet"] = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
        config["default_leverage"] = int(os.getenv("BYBIT_DEFAULT_LEVERAGE", "5"))
        config["pos_size_multiplier"] = float(os.getenv("BYBIT_POS_SIZE_MULTIPLIER", "1.0"))

        if config["api_key"] and config["api_secret"]:
            logger.info("Using ByBit config from environment variables")
            return config

        logger.warning("ByBit configuration not found in st.secrets or environment")
        return {}

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
