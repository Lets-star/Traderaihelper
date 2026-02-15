#!/usr/bin/env python3
"""
Test script for ByBit integration and SignalExecutor.
Tests the trading functionality without requiring actual API keys.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bybit_client import ByBitClient
from signal_executor import SignalExecutor
from update_bus import UpdateBus


def test_bybit_client_validation():
    """Test ByBit client credential validation."""
    print("\n=== Testing ByBitClient Validation ===")
    
    # Test with empty credentials
    client = ByBitClient("", "", testnet=True)
    assert not client.validate_credentials(), "Empty credentials should be invalid"
    print("✓ Empty credentials correctly rejected")
    
    # Test with short credentials
    client = ByBitClient("short", "secret", testnet=True)
    assert not client.validate_credentials(), "Short credentials should be invalid"
    print("✓ Short credentials correctly rejected")
    
    # Test with valid-looking credentials
    client = ByBitClient("X" * 20, "Y" * 30, testnet=True)
    assert client.validate_credentials(), "Valid credentials should be accepted"
    print("✓ Valid credentials correctly accepted")
    
    print("✅ ByBitClient validation tests passed!")


def test_signal_executor_init():
    """Test SignalExecutor initialization."""
    print("\n=== Testing SignalExecutor Initialization ===")
    
    update_bus = UpdateBus()
    executor = SignalExecutor(update_bus=update_bus)
    
    assert not executor.enabled, "Executor should start disabled"
    assert executor.testnet, "Executor should default to testnet"
    assert executor.dry_run is False, "Executor should default to live mode"
    print("✓ SignalExecutor initialized correctly")
    
    # Test configuration
    executor.configure(
        enabled=True,
        api_key="test_key_12345",
        api_secret="test_secret_12345",
        testnet=True,
        leverage=10,
        pos_size_multiplier=1.5,
        dry_run=True
    )
    
    assert executor.enabled, "Executor should be enabled after configure"
    assert executor.dry_run, "Executor should be in dry run mode"
    assert executor.default_leverage == 10, "Leverage should be set to 10"
    print("✓ SignalExecutor configuration works correctly")
    
    print("✅ SignalExecutor initialization tests passed!")


def test_signal_executor_dry_run():
    """Test SignalExecutor dry run execution."""
    print("\n=== Testing SignalExecutor Dry Run ===")
    
    update_bus = UpdateBus()
    executor = SignalExecutor(update_bus=update_bus)
    
    executor.configure(
        enabled=True,
        api_key="test_key",
        api_secret="test_secret",
        testnet=True,
        leverage=5,
        pos_size_multiplier=1.0,
        dry_run=True
    )
    
    # Create a test signal
    signal = {
        "signal_id": "test_signal_001",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 50000.0,
        "take_profit": 51000.0,
        "stop_loss": 49000.0,
        "quantity": 0.01,
        "leverage": 5
    }
    
    # Execute signal (dry run should work without API keys)
    try:
        executor.execute_signal(signal)
        print("✓ Dry run execution initiated")
        
        # Check update bus for pending status
        updates = update_bus.drain()
        if updates:
            print(f"✓ Update bus received {len(updates)} updates")
            for update in updates:
                print(f"  - Type: {update.get('type')}, Status: {update.get('status')}")
        
        print("✅ Dry run test passed!")
    except Exception as e:
        print(f"✗ Dry run execution failed: {e}")
        raise


def test_update_bus():
    """Test UpdateBus functionality."""
    print("\n=== Testing UpdateBus ===")
    
    bus = UpdateBus(max_size=100)
    
    # Test publish
    result = bus.publish({"type": "TEST", "data": "hello"})
    assert result, "Publish should succeed"
    print("✓ Publish works correctly")
    
    # Test has_updates
    assert bus.has_updates(), "Should have pending updates"
    print("✓ has_updates works correctly")
    
    # Test drain
    updates = bus.drain()
    assert len(updates) == 1, "Should drain 1 update"
    assert updates[0]["type"] == "TEST", "Update should have correct type"
    print("✓ Drain works correctly")
    
    # Test empty drain
    updates = bus.drain()
    assert len(updates) == 0, "Should have no updates after drain"
    print("✓ Empty drain works correctly")
    
    # Test invalid publish
    result = bus.publish("not a dict")
    assert not result, "Invalid publish should fail"
    print("✓ Invalid publish correctly rejected")
    
    result = bus.publish({"no_type": "missing"})
    assert not result, "Publish without type should fail"
    print("✓ Missing type correctly rejected")
    
    print("✅ UpdateBus tests passed!")


def test_signal_structure():
    """Test that signal structure is validated."""
    print("\n=== Testing Signal Structure ===")
    
    executor = SignalExecutor()
    
    # Test signal with minimal fields
    minimal_signal = {
        "symbol": "ETHUSDT",
        "direction": "SHORT"
    }
    
    # Should not raise an error
    try:
        executor.execute_signal(minimal_signal)
        print("✓ Minimal signal accepted")
    except Exception as e:
        print(f"✗ Minimal signal failed: {e}")
        raise
    
    # Test signal with all fields
    full_signal = {
        "signal_id": "test_002",
        "symbol": "ETHUSDT",
        "direction": "SHORT",
        "entry_price": 3000.0,
        "take_profit": 2800.0,
        "stop_loss": 3200.0,
        "quantity": 0.1,
        "leverage": 10
    }
    
    try:
        executor.execute_signal(full_signal)
        print("✓ Full signal accepted")
    except Exception as e:
        print(f"✗ Full signal failed: {e}")
        raise
    
    print("✅ Signal structure tests passed!")


def test_log_file_creation():
    """Test that trade log file is created."""
    print("\n=== Testing Log File Creation ===")
    
    log_file = "trade_execution_log.csv"
    
    # Remove existing log file if present
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Create executor which should create log file
    executor = SignalExecutor()
    
    assert os.path.exists(log_file), "Log file should be created"
    
    with open(log_file, 'r') as f:
        content = f.read()
        assert "timestamp" in content, "Log file should have header"
        assert "signal_id" in content, "Log file should have signal_id column"
        assert "status" in content, "Log file should have status column"
    
    print("✓ Log file created with correct headers")
    print("✅ Log file creation test passed!")


async def test_bybit_client_methods():
    """Test ByBit client methods (requires API keys)."""
    print("\n=== Testing ByBitClient Methods (Mock) ===")
    
    # We can't test actual API calls without credentials,
    # but we can test that the methods exist and have correct signatures
    client = ByBitClient("test_key_1234567890", "test_secret_1234567890", testnet=True)
    
    # Check methods exist
    methods = [
        'set_leverage',
        'place_order',
        'cancel_order',
        'get_order_status',
        'get_position',
        'get_wallet_balance',
        'get_tickers',
        'get_open_orders',
        'validate_credentials'
    ]
    
    for method in methods:
        assert hasattr(client, method), f"Client should have {method} method"
        print(f"✓ Method {method} exists")
    
    print("✅ ByBitClient methods test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("BYBIT INTEGRATION TEST SUITE")
    print("=" * 60)
    
    try:
        test_bybit_client_validation()
        test_signal_executor_init()
        test_update_bus()
        test_signal_structure()
        test_log_file_creation()
        test_signal_executor_dry_run()
        
        # Run async tests
        asyncio.run(test_bybit_client_methods())
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
