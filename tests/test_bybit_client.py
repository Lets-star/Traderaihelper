"""Tests for ByBitClient.validate_credentials enhancement."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

from bybit_client import ByBitClient


class TestByBitClientValidateCredentials:
    """Test ByBitClient.validate_credentials with real API validation."""

    def test_validate_credentials_with_empty_keys(self):
        """Test that validation fails with empty keys."""
        client = ByBitClient(api_key="", api_secret="")

        result = client.validate_credentials()

        assert result is False

    def test_validate_credentials_with_short_keys(self):
        """Test that validation fails with keys that are too short."""
        client = ByBitClient(api_key="short", api_secret="short")

        result = client.validate_credentials()

        assert result is False

    def test_validate_credentials_with_valid_api_call(self):
        """Test that validation succeeds with valid credentials and successful API call."""
        client = ByBitClient(
            api_key="valid_key_123456789",
            api_secret="valid_secret_123456789"
        )

        # Mock get_wallet_balance to return success
        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "list": [
                        {
                            "coin": [
                                {"coin": "USDT", "walletBalance": "1000.0"}
                            ]
                        }
                    ]
                }
            }

            result = client.validate_credentials()

            assert result is True
            mock_balance.assert_called_once_with(account_type="UNIFIED")

    def test_validate_credentials_with_api_error(self):
        """Test that validation fails when API returns error."""
        client = ByBitClient(
            api_key="invalid_key_123456789",
            api_secret="invalid_secret_123456789"
        )

        # Mock get_wallet_balance to return error
        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.return_value = {
                "retCode": 10001,
                "retMsg": "Invalid API keys",
                "result": {}
            }

            result = client.validate_credentials()

            assert result is False
            mock_balance.assert_called_once_with(account_type="UNIFIED")

    def test_validate_credentials_with_api_exception(self):
        """Test that validation fails when API call raises exception."""
        client = ByBitClient(
            api_key="error_key_123456789",
            api_secret="error_secret_123456789"
        )

        # Mock get_wallet_balance to raise exception
        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.side_effect = Exception("Network error")

            result = client.validate_credentials()

            assert result is False
            mock_balance.assert_called_once_with(account_type="UNIFIED")

    def test_validate_credentials_with_none_retcode(self):
        """Test that validation fails when API returns None retCode."""
        client = ByBitClient(
            api_key="partial_key_123456789",
            api_secret="partial_secret_123456789"
        )

        # Mock get_wallet_balance to return response without retCode
        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.return_value = {
                "retMsg": "Partial response",
                "result": {}
            }

            result = client.validate_credentials()

            assert result is False
            mock_balance.assert_called_once_with(account_type="UNIFIED")

    def test_validate_credentials_format_check_only(self):
        """Test that format validation happens before API call."""
        client = ByBitClient(api_key="", api_secret="")

        # Mock get_wallet_balance - should NOT be called due to format check
        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {}
            }

            result = client.validate_credentials()

            assert result is False
            # API call should not be made if format check fails
            mock_balance.assert_not_called()

    def test_validate_credentials_logs_success(self):
        """Test that successful validation is logged."""
        client = ByBitClient(
            api_key="valid_key_123456789",
            api_secret="valid_secret_123456789"
        )

        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.return_value = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {}
            }

            with patch('bybit_client.logger') as mock_logger:
                client.validate_credentials()

                # Verify success log
                mock_logger.info.assert_any_call(
                    "ByBit API credentials validated successfully"
                )

    def test_validate_credentials_logs_failure(self):
        """Test that failed validation is logged."""
        client = ByBitClient(
            api_key="invalid_key_123456789",
            api_secret="invalid_secret_123456789"
        )

        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.return_value = {
                "retCode": 10001,
                "retMsg": "Invalid API keys",
                "result": {}
            }

            with patch('bybit_client.logger') as mock_logger:
                client.validate_credentials()

                # Verify warning log with details
                mock_logger.warning.assert_any_call(
                    "ByBit API credential validation failed: Invalid API keys (retCode: 10001)"
                )

    def test_validate_credentials_logs_exception(self):
        """Test that validation exceptions are logged."""
        client = ByBitClient(
            api_key="error_key_123456789",
            api_secret="error_secret_123456789"
        )

        with patch.object(client, 'get_wallet_balance') as mock_balance:
            mock_balance.side_effect = Exception("Network timeout")

            with patch('bybit_client.logger') as mock_logger:
                client.validate_credentials()

                # Verify warning log with exception
                mock_logger.warning.assert_any_call(
                    "ByBit API credential validation error: Network timeout"
                )
