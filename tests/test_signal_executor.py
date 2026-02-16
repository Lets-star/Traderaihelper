"""Tests for SignalExecutor bug fixes."""

from __future__ import annotations

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from signal_executor import SignalExecutor


class TestSignalExecutorCredentials:
    """Test SignalExecutor._get_api_credentials error handling."""

    def test_credentials_from_streamlit_secrets(self):
        """Test getting credentials from st.secrets (direct keys)."""
        executor = SignalExecutor()

        with patch('signal_executor.st') as mock_st:
            mock_st.secrets = {
                "BYBIT_API_KEY": "test_key_123456789",
                "BYBIT_API_SECRET": "test_secret_123456789"
            }

            api_key, api_secret = executor._get_api_credentials()

            assert api_key == "test_key_123456789"
            assert api_secret == "test_secret_123456789"

    def test_credentials_from_streamlit_sections(self):
        """Test getting credentials from st.secrets[bybit] section."""
        executor = SignalExecutor()

        with patch('signal_executor.st') as mock_st:
            mock_st.secrets = {
                "bybit": {
                    "api_key": "test_key_123456789",
                    "api_secret": "test_secret_123456789"
                }
            }

            api_key, api_secret = executor._get_api_credentials()

            assert api_key == "test_key_123456789"
            assert api_secret == "test_secret_123456789"

    def test_credentials_fallback_to_env_vars(self):
        """Test fallback to environment variables when Streamlit unavailable."""
        executor = SignalExecutor()

        with patch.dict(os.environ, {
            "BYBIT_API_KEY": "env_key_123456789",
            "BYBIT_API_SECRET": "env_secret_123456789"
        }, clear=False):
            # Simulate ImportError
            with patch('signal_executor.st', side_effect=ImportError):
                api_key, api_secret = executor._get_api_credentials()

                assert api_key == "env_key_123456789"
                assert api_secret == "env_secret_123456789"

    def test_credentials_prefer_streamlit_over_env(self):
        """Test that Streamlit secrets are preferred over environment variables."""
        executor = SignalExecutor()

        with patch('signal_executor.st') as mock_st:
            mock_st.secrets = {
                "BYBIT_API_KEY": "st_key_123456789",
                "BYBIT_API_SECRET": "st_secret_123456789"
            }

            with patch.dict(os.environ, {
                "BYBIT_API_KEY": "env_key_123456789",
                "BYBIT_API_SECRET": "env_secret_123456789"
            }, clear=False):
                api_key, api_secret = executor._get_api_credentials()

                # Should prefer Streamlit
                assert api_key == "st_key_123456789"
                assert api_secret == "st_secret_123456789"

    def test_credentials_no_credentials_raises_error(self):
        """Test that ValueError is raised when no credentials found."""
        executor = SignalExecutor()

        with patch.dict(os.environ, {}, clear=False):
            with patch('signal_executor.st', side_effect=ImportError):
                with pytest.raises(ValueError, match="API credentials not found"):
                    executor._get_api_credentials()

    def test_credentials_env_vars_with_streamlit_import_error(self):
        """Test that env vars work even when Streamlit import fails."""
        executor = SignalExecutor()

        # This is the critical fix: env vars should work even if Streamlit can't be imported
        with patch.dict(os.environ, {
            "BYBIT_API_KEY": "env_key_123456789",
            "BYBIT_API_SECRET": "env_secret_123456789"
        }, clear=False):
            # Patch to simulate ImportError when importing streamlit
            import builtins
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == 'streamlit':
                    raise ImportError("Streamlit not available")
                return real_import(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=mock_import):
                api_key, api_secret = executor._get_api_credentials()

                assert api_key == "env_key_123456789"
                assert api_secret == "env_secret_123456789"


class TestSignalExecutorIsPositionOpen:
    """Test SignalExecutor.is_position_open None comparison fix."""

    def test_is_position_open_with_none_retcode(self):
        """Test that is_position_open returns False when retCode is None."""
        executor = SignalExecutor()

        # Mock get_position to return error response with None retCode
        with patch.object(executor, 'get_position') as mock_get_position:
            mock_get_position.return_value = {
                "retCode": None,
                "retMsg": "Error occurred"
            }

            result = executor.is_position_open("BTCUSDT")

            # Should return False, not True (the bug was that None != 0 returns True)
            assert result is False

    def test_is_position_open_with_nonzero_retcode(self):
        """Test that is_position_open returns False when retCode != 0."""
        executor = SignalExecutor()

        with patch.object(executor, 'get_position') as mock_get_position:
            mock_get_position.return_value = {
                "retCode": 10001,
                "retMsg": "Invalid symbol"
            }

            result = executor.is_position_open("BTCUSDT")

            assert result is False

    def test_is_position_open_with_zero_retcode_no_positions(self):
        """Test that is_position_open returns False when no positions exist."""
        executor = SignalExecutor()

        with patch.object(executor, 'get_position') as mock_get_position:
            mock_get_position.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": []
                }
            }

            result = executor.is_position_open("BTCUSDT")

            assert result is False

    def test_is_position_open_with_zero_retcode_with_positions(self):
        """Test that is_position_open returns True when position exists."""
        executor = SignalExecutor()

        with patch.object(executor, 'get_position') as mock_get_position:
            mock_get_position.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "size": "0.001",
                            "side": "Buy"
                        }
                    ]
                }
            }

            result = executor.is_position_open("BTCUSDT")

            assert result is True

    def test_is_position_open_with_zero_size_position(self):
        """Test that is_position_open returns False when position size is zero."""
        executor = SignalExecutor()

        with patch.object(executor, 'get_position') as mock_get_position:
            mock_get_position.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "size": "0",
                            "side": "Buy"
                        }
                    ]
                }
            }

            result = executor.is_position_open("BTCUSDT")

            assert result is False

    def test_is_position_open_with_missing_size_field(self):
        """Test that is_position_open returns False when size field is missing."""
        executor = SignalExecutor()

        with patch.object(executor, 'get_position') as mock_get_position:
            mock_get_position.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "side": "Buy"
                        }
                    ]
                }
            }

            result = executor.is_position_open("BTCUSDT")

            assert result is False
