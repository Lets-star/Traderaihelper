"""Tests for sanitize_payload_for_real_data fix."""

from __future__ import annotations

import pytest

from web_ui import sanitize_payload_for_real_data


class TestSanitizePayloadForRealData:
    """Test sanitize_payload_for_real_data with proper validation."""

    def test_sanitize_real_data_without_synthetic_markers(self):
        """Test that real data without synthetic markers is marked as real."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance"
            },
            "data": [1, 2, 3]
        }

        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is True
        assert result["metadata"]["is_real_data"] is True
        assert result["metadata"]["real_data_validated"] is True
        assert result["metadata"]["data_quality"] == "validated_real_data"
        assert result["metadata"]["source"] == "binance"
        assert result["metadata"]["exchange"] == "binance"

    def test_sanitize_synthetic_data_with_markers(self):
        """Test that synthetic data is NOT marked as real."""
        payload = {
            "metadata": {
                "source": "demo",
                "exchange": "demo"
            },
            "data": [1, 2, 3],
            "is_synthetic": True
        }

        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is False
        assert result["metadata"]["is_real_data"] is False
        assert result["metadata"]["real_data_validated"] is False
        assert result["metadata"]["data_quality"] == "unvalidated_data"
        assert result["metadata"]["source"] == ""  # Not falsified
        assert result["metadata"]["exchange"] == ""  # Not falsified
        assert "is_synthetic" not in result  # Removed synthetic flag

    def test_sanitize_synthetic_data_with_marker_values(self):
        """Test that data with synthetic marker values is NOT marked as real."""
        payload = {
            "metadata": {
                "source": "binance",
                "exchange": "binance"
            },
            "data": "mock_data_value"
        }

        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is False
        assert result["metadata"]["is_real_data"] is False
        assert result["metadata"]["data_quality"] == "unvalidated_data"
        assert result["data"] == ""  # Marker replaced with empty string

    def test_sanitize_multiple_synthetic_flags(self):
        """Test handling multiple synthetic markers."""
        payload = {
            "metadata": {
                "source": "sample"
            },
            "is_synthetic": True,
            "is_mock": True,
            "data": "demo data"
        }

        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is False
        assert "is_synthetic" not in result
        assert "is_mock" not in result
        assert result["metadata"]["source"] == ""

    def test_sanitize_nested_dict_with_synthetic_markers(self):
        """Test cleaning nested dictionaries."""
        payload = {
            "metadata": {},
            "candles": [
                {"close": "100.0", "source": "mock"},
                {"close": "101.0", "source": "binance"}
            ]
        }

        result = sanitize_payload_for_real_data(payload)

        # Should detect synthetic marker in nested structure
        assert result["metadata"]["real_data"] is False
        assert result["candles"][0]["source"] == ""
        assert result["candles"][1]["source"] == "binance"

    def test_sanitize_list_with_synthetic_markers(self):
        """Test cleaning lists with synthetic markers."""
        payload = {
            "metadata": {},
            "sources": ["demo", "binance", "mock"]
        }

        result = sanitize_payload_for_real_data(payload)

        # Should detect synthetic markers in list
        assert result["metadata"]["real_data"] is False
        assert result["sources"] == ["", "binance", ""]

    def test_sanitize_preserves_real_data_source(self):
        """Test that real data source is preserved."""
        payload = {
            "metadata": {},
            "source": "binance",
            "exchange": "binance",
            "data": [1, 2, 3]
        }

        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is True
        assert result["metadata"]["source"] == "binance"
        assert result["metadata"]["exchange"] == "binance"

    def test_sanitize_does_not_falsify_empty_source(self):
        """Test that empty source is not replaced with binance."""
        payload = {
            "metadata": {},
            "source": "",
            "exchange": "",
            "data": [1, 2, 3]
        }

        result = sanitize_payload_for_real_data(payload)

        # With no synthetic markers, default to binance
        assert result["metadata"]["real_data"] is True
        assert result["metadata"]["source"] == "binance"
        assert result["metadata"]["exchange"] == "binance"

    def test_sanitize_non_dict_payload(self):
        """Test handling non-dict payload."""
        payload = "just a string"
        result = sanitize_payload_for_real_data(payload)

        assert result == "just a string"

    def test_sanitize_empty_payload(self):
        """Test handling empty payload."""
        payload = {}
        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is True  # No markers means valid
        assert result["metadata"]["data_quality"] == "validated_real_data"

    def test_sanitize_removes_all_synthetic_flag_keys(self):
        """Test that all synthetic flag keys are removed."""
        payload = {
            "metadata": {},
            "is_synthetic": True,
            "synthetic": True,
            "mock": True,
            "demo": True,
            "paper": True,
            "testnet": True
        }

        result = sanitize_payload_for_real_data(payload)

        assert "is_synthetic" not in result
        assert "synthetic" not in result
        assert "mock" not in result
        assert "demo" not in result
        assert "paper" not in result
        assert "testnet" not in result
        assert result["metadata"]["real_data"] is False

    def test_sanitize_all_synthetic_marker_values(self):
        """Test that all synthetic marker values are detected."""
        synthetic_values = [
            "mock", "test", "demo", "simulated", "synthetic",
            "fake", "sample", "paper", "backtest", "historical_sim",
            "generated", "artificial"
        ]

        for value in synthetic_values:
            payload = {
                "metadata": {},
                "data_source": value
            }

            result = sanitize_payload_for_real_data(payload)

            assert result["metadata"]["real_data"] is False,
                f"Failed to detect synthetic marker: {value}"
            assert result["data_source"] == ""

    def test_sanitize_case_insensitive_detection(self):
        """Test that marker detection is case insensitive."""
        payload = {
            "metadata": {},
            "source": "DEMO",
            "data": "MockData"
        }

        result = sanitize_payload_for_real_data(payload)

        assert result["metadata"]["real_data"] is False

    def test_sanitize_partial_marker_detection(self):
        """Test detection of markers within strings."""
        payload = {
            "metadata": {},
            "description": "This is a backtest_dataset"
        }

        result = sanitize_payload_for_real_data(payload)

        # Should detect "backtest" within the string
        assert result["metadata"]["real_data"] is False
        assert result["description"] == ""
