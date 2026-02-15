"""
Example demonstrating the updated Traderaihelper fixes.

This script shows how to use the refactored components with:
- Thread-safe synchronous API client
- st.secrets integration
- Context manager support
- Parameter validation
- Rate limit handling
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.secret_manager import SecretManager
from bybit_client import ByBitClient
from signal_executor import SignalExecutor
from update_bus import UpdateBus


def example_secret_manager():
    """Example of using SecretManager for credential management."""
    print("=" * 60)
    print("Example 1: SecretManager Usage")
    print("=" * 60)

    # Check if credentials are available
    if SecretManager.has_bybit_credentials():
        print("✓ ByBit credentials are configured")

        # Get credentials (will use st.secrets or environment)
        api_key, api_secret = SecretManager.get_bybit_credentials()

        # Mask for display
        masked_key = SecretManager.mask_credential(api_key or "")
        print(f"  API Key: {masked_key}")
        print(f"  API Secret: {SecretManager.mask_credential(api_secret or "")}")

        # Get full configuration
        config = SecretManager.get_bybit_config()
        print(f"  Testnet: {config.get('testnet', True)}")
        print(f"  Default Leverage: {config.get('default_leverage', 5)}")
    else:
        print("✗ ByBit credentials not found")
        print("  Set st.secrets[BYBIT_API_KEY] and st.secrets[BYBIT_API_SECRET]")
        print("  Or environment variables BYBIT_API_KEY and BYBIT_API_SECRET")

    print()


def example_bybit_client():
    """Example of using ByBitClient with context manager."""
    print("=" * 60)
    print("Example 2: ByBitClient with Context Manager")
    print("=" * 60)

    # Get credentials
    api_key, api_secret = SecretManager.get_bybit_credentials()
    config = SecretManager.get_bybit_config()

    if not api_key or not api_secret:
        print("Skipping ByBitClient example (no credentials)")
        print()
        return

    try:
        # Use context manager for automatic resource cleanup
        with ByBitClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=config.get("testnet", True),
            connection_pool_size=10,
            log_trades=True
        ) as client:

            print("✓ Client created with context manager")
            print(f"  Base URL: {client.base_url}")
            print(f"  Connection Pool: {client.connection_pool_size}")

            # Validate credentials
            if client.validate_credentials():
                print("✓ Credentials validated")
            else:
                print("✗ Credential validation failed")

            # Example: Get wallet balance
            print("\nFetching wallet balance...")
            balance_result = client.get_wallet_balance(account_type="UNIFIED")

            if balance_result.get("retCode") == 0:
                print("✓ Wallet balance fetched successfully")
                result = balance_result.get("result", {})
                if result.get("list"):
                    account = result["list"][0]
                    coin = account.get("coin", [])
                    for c in coin[:3]:  # Show first 3 coins
                        print(f"  {c.get('coin')}: {c.get('walletBalance')} {c.get('coin')}")
            else:
                print(f"✗ Failed to fetch balance: {balance_result.get('retMsg')}")

            # Example: Get position for a symbol
            print("\nFetching position for BTCUSDT...")
            position_result = client.get_position("BTCUSDT")

            if position_result.get("retCode") == 0:
                print("✓ Position fetched successfully")
                positions = position_result.get("result", {}).get("list", [])
                if positions and positions[0].get("size") != "0":
                    pos = positions[0]
                    print(f"  Symbol: {pos.get('symbol')}")
                    print(f"  Size: {pos.get('size')}")
                    print(f"  Side: {pos.get('side')}")
                else:
                    print("  No open position")
            else:
                print(f"✗ Failed to fetch position: {position_result.get('retMsg')}")

    except ValueError as e:
        print(f"✗ Validation error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


def example_order_validation():
    """Example of parameter validation."""
    print("=" * 60)
    print("Example 3: Parameter Validation")
    print("=" * 60)

    api_key, api_secret = SecretManager.get_bybit_credentials()
    config = SecretManager.get_bybit_config()

    if not api_key or not api_secret:
        print("Skipping validation example (no credentials)")
        print()
        return

    try:
        with ByBitClient(api_key, api_secret, config.get("testnet", True)) as client:

            # Test valid parameters
            print("Testing valid parameters...")
            try:
                # This should succeed
                client._validate_symbol("BTCUSDT")
                client._validate_side("Buy")
                client._validate_order_type("Market")
                qty = client._validate_quantity("0.001")
                price = client._validate_price(50000)
                print("✓ All valid parameters accepted")
            except ValueError as e:
                print(f"✗ Unexpected validation error: {e}")

            # Test invalid parameters
            print("\nTesting invalid parameters...")

            # Invalid symbol
            try:
                client._validate_symbol("bt")  # Too short
                print("✗ Should have rejected short symbol")
            except ValueError as e:
                print(f"✓ Rejected short symbol: {e}")

            # Invalid side
            try:
                client._validate_side("UP")  # Invalid side
                print("✗ Should have rejected invalid side")
            except ValueError as e:
                print(f"✓ Rejected invalid side: {e}")

            # Invalid quantity
            try:
                client._validate_quantity(-0.001)  # Negative
                print("✗ Should have rejected negative quantity")
            except ValueError as e:
                print(f"✓ Rejected negative quantity: {e}")

            # Invalid price
            try:
                client._validate_price(0)  # Zero
                print("✗ Should have rejected zero price")
            except ValueError as e:
                print(f"✓ Rejected zero price: {e}")

            # Limit order without price
            try:
                client._validate_order_type("Limit")
                # The actual place_order checks this
                print("  Note: Limit order requires price (checked in place_order)")
            except ValueError:
                pass

    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print()


def example_signal_executor():
    """Example of using SignalExecutor with validation."""
    print("=" * 60)
    print("Example 4: SignalExecutor with Validation")
    print("=" * 60)

    # Create executor with update bus
    update_bus = UpdateBus()
    executor = SignalExecutor(update_bus=update_bus)

    # Test signal validation
    print("Testing signal validation...")

    # Valid signal
    valid_signal = {
        "signal_id": "test_sig_001",
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 47000.0,
        "take_profit": 50000.0,
        "stop_loss": 45000.0,
        "quantity": 0.001,
        "leverage": 5
    }

    errors = executor._validate_signal(valid_signal)
    if errors:
        print(f"✗ Valid signal rejected: {errors}")
    else:
        print("✓ Valid signal accepted")

    # Invalid signals
    print("\nTesting invalid signals...")

    # Missing required fields
    invalid_signal_1 = {
        "symbol": "BTCUSDT",
        "direction": "LONG"
        # Missing: signal_id, entry_price
    }
    errors = executor._validate_signal(invalid_signal_1)
    if errors:
        print(f"✓ Rejected signal with missing fields: {', '.join(errors[:2])}")

    # Invalid direction
    invalid_signal_2 = valid_signal.copy()
    invalid_signal_2["direction"] = "UP"
    errors = executor._validate_signal(invalid_signal_2)
    if errors:
        print(f"✓ Rejected invalid direction: {errors[0]}")

    # Negative price
    invalid_signal_3 = valid_signal.copy()
    invalid_signal_3["entry_price"] = -1000
    errors = executor._validate_signal(invalid_signal_3)
    if errors:
        print(f"✓ Rejected negative price: {errors[0]}")

    # Configure executor (will use st.secrets if available)
    print("\nConfiguring executor...")
    api_key, api_secret = SecretManager.get_bybit_credentials()
    config = SecretManager.get_bybit_config()

    if api_key and api_secret:
        executor.configure(
            enabled=True,
            testnet=config.get("testnet", True),
            leverage=5,
            pos_size_multiplier=1.0,
            dry_run=True  # Use dry run for safety
        )
        print("✓ Executor configured in dry-run mode")
    else:
        print("✗ Cannot configure executor (no credentials)")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TRADERAIHELPER EXTENDED FIXES - DEMONSTRATION")
    print("=" * 60)
    print()

    example_secret_manager()
    example_bybit_client()
    example_order_validation()
    example_signal_executor()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
