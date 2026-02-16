"""
Example usage of refactored TraderAIHelper components.

This example demonstrates:
1. Using SecretsManager for secure credential access
2. Using HealthChecker for system monitoring
3. Using CacheManager for caching
4. Using SignalExecutor for signal execution
5. Using Pydantic models for data validation
"""

import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_secrets_manager():
    """Example: Using SecretsManager for credential management."""
    print("\n" + "="*70)
    print("Example 1: Secrets Manager")
    print("="*70)

    from secrets_manager import get_secrets_manager

    # Get the global secrets manager
    secrets = get_secrets_manager()

    # Get a secret value (checks st.secrets, then env vars, then default)
    api_key = secrets.get("bybit.api_key", default="default_key_placeholder")
    print(f"API Key (masked): {api_key[:8]}..." if len(api_key) > 8 else "Not configured")

    # Get credentials object
    credentials = secrets.get_bybit_credentials(testnet=True)
    if credentials:
        print(f"✓ Credentials loaded for {credentials.exchange}")
        print(f"  Testnet: {credentials.testnet}")
        print(f"  Valid: {credentials.is_valid()}")
    else:
        print("✗ No credentials configured")
        print("  Set BYBIT_API_KEY and BYBIT_API_SECRET environment variables")

    # Generate example .env file
    print("\nExample .env file:")
    print(secrets.get_env_example(["bybit.api_key", "bybit.api_secret"]))


def example_health_checker():
    """Example: Using HealthChecker for system monitoring."""
    print("\n" + "="*70)
    print("Example 2: Health Checker")
    print("="*70)

    from health_checker import HealthChecker
    from secrets_manager import get_secrets_manager

    # Create health checker
    checker = HealthChecker(check_interval=60)

    # Check system resources
    print("\nChecking system resources...")
    health = checker.check_system_resources()
    print(f"  Status: {health.status.value}")
    print(f"  Message: {health.message}")
    if health.details:
        if 'memory_percent' in health.details:
            print(f"  Memory: {health.details['memory_percent']:.1f}%")
        if 'cpu_percent' in health.details:
            print(f"  CPU: {health.details['cpu_percent']:.1f}%")

    # Check API connection (without credentials)
    print("\nChecking ByBit API connection...")
    health = checker.check_api_connection(
        "https://api-testnet.bybit.com",
        timeout=5
    )
    print(f"  Status: {health.status.value}")
    print(f"  Message: {health.message}")
    if health.response_time_ms:
        print(f"  Response time: {health.response_time_ms:.2f}ms")

    # Get health summary
    summary = checker.get_summary()
    print(f"\nHealth Summary:")
    print(f"  Overall status: {summary['status']}")
    print(f"  Components checked: {summary['counts']['total']}")


def example_cache_manager():
    """Example: Using CacheManager for caching."""
    print("\n" + "="*70)
    print("Example 3: Cache Manager")
    print("="*70)

    from cache_manager import get_cache_manager, cached
    import time

    # Get the global cache manager
    cache_mgr = get_cache_manager()

    # Get a specific cache
    market_cache = cache_mgr.get_cache("market_data")

    # Set and get values
    print("\nBasic cache operations:")
    market_cache.set("BTCUSDT_price", 50000.0, ttl=30)
    market_cache.set("ETHUSDT_price", 3000.0, ttl=30)

    btc_price = market_cache.get("BTCUSDT_price")
    print(f"  BTC Price: ${btc_price:,.2f}")

    eth_price = market_cache.get("ETHUSDT_price")
    print(f"  ETH Price: ${eth_price:,.2f}")

    non_existent = market_cache.get("DOGEUSDT_price")
    print(f"  DOGE Price (cached): {non_existent}")

    # Check cache statistics
    stats = market_cache.get_stats()
    print(f"\nCache statistics:")
    print(f"  Size: {stats['size']} / {stats['max_size']}")
    print(f"  Hit rate: {stats['hit_rate_percent']:.1f}%")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")

    # Use cached decorator
    print("\nUsing @cached decorator:")

    @cached(cache_name="indicators", ttl=60)
    def expensive_calculation(x: float) -> float:
        """Simulate expensive calculation."""
        time.sleep(0.1)  # Simulate work
        return x * x

    # First call - computes
    start = time.time()
    result1 = expensive_calculation(42.0)
    time1 = (time.time() - start) * 1000
    print(f"  First call: {result1} ({time1:.2f}ms)")

    # Second call - from cache
    start = time.time()
    result2 = expensive_calculation(42.0)
    time2 = (time.time() - start) * 1000
    print(f"  Second call: {result2} ({time2:.2f}ms) [from cache]")

    # Get all cache summary
    summary = cache_mgr.get_summary()
    print(f"\nAll caches summary:")
    print(f"  Total caches: {summary['cache_count']}")
    print(f"  Total entries: {summary['total_size']}")
    print(f"  Overall hit rate: {summary['overall_hit_rate_percent']:.1f}%")


def example_pydantic_models():
    """Example: Using Pydantic models for validation."""
    print("\n" + "="*70)
    print("Example 4: Pydantic Models")
    print("="*70)

    from models import Signal, SignalType, Direction, Credentials, ProcessedSignal

    # Create a valid signal
    print("\nCreating a valid signal:")
    signal = Signal(
        signal_id="btc_signal_001",
        signal_type=SignalType.BUY,
        symbol="BTCUSDT",
        direction=Direction.LONG,
        entry_price=50000.0,
        take_profit=52000.0,
        stop_loss=49000.0,
        confidence=0.85,
        leverage=10,
        quantity=0.001,
        generated_at=int(datetime.utcnow().timestamp() * 1000),
    )
    print(f"  Signal ID: {signal.signal_id}")
    print(f"  Type: {signal.signal_type.value}")
    print(f"  Symbol: {signal.symbol}")
    print(f"  Direction: {signal.direction.value}")
    print(f"  Entry: ${signal.entry_price:,.2f}")
    print(f"  Take Profit: ${signal.take_profit:,.2f}")
    print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"  Confidence: {signal.confidence:.2f}")
    print(f"  Leverage: {signal.leverage}x")
    print(f"  Executable: {signal.is_executable()}")

    # Validate the signal
    print("\nValidating signal...")
    processed = ProcessedSignal(signal=signal)
    is_valid = processed.validate()
    print(f"  Valid: {is_valid}")
    if processed.validation_errors:
        for error in processed.validation_errors:
            print(f"  Error: {error}")
    else:
        print("  No validation errors")

    # Try to create invalid signal (will raise ValidationError)
    print("\nTrying to create invalid signal...")
    try:
        invalid_signal = Signal(
            signal_id="invalid_001",
            signal_type=SignalType.BUY,
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry_price=50000.0,
            take_profit=48000.0,  # Invalid: TP < entry for LONG
            stop_loss=49000.0,
            confidence=0.85,
            leverage=10,
            quantity=0.001,
            generated_at=int(datetime.utcnow().timestamp() * 1000),
        )
    except Exception as e:
        print(f"  Validation error (expected): {e}")

    # Create credentials
    print("\nCreating credentials:")
    creds = Credentials(
        api_key="test_api_key_1234567890",
        api_secret="test_api_secret_1234567890",
        exchange="bybit",
        testnet=True,
    )
    print(f"  Exchange: {creds.exchange}")
    print(f"  Testnet: {creds.testnet}")
    print(f"  Valid: {creds.is_valid()}")


def example_signal_executor():
    """Example: Using SignalExecutor for signal execution."""
    print("\n" + "="*70)
    print("Example 5: Signal Executor")
    print("="*70)

    from signal_executor import SignalExecutor
    from secrets_manager import get_secrets_manager

    # Get credentials
    secrets_mgr = get_secrets_manager()
    credentials = secrets_mgr.get_bybit_credentials(testnet=True)

    # Create executor
    executor = SignalExecutor(
        credentials=credentials,
        enabled=False,  # Disabled by default for safety
        log_file="example_executions.csv",
        max_execution_threads=3,
        rate_limit_delay=1.0,
    )

    print(f"\nSignal Executor created:")
    print(f"  Enabled: {executor.enabled}")
    print(f"  Log file: {executor.log_file}")
    print(f"  Max threads: {executor.max_execution_threads}")
    print(f"  Rate limit delay: {executor.rate_limit_delay}s")

    # Get statistics
    stats = executor.get_statistics()
    print(f"\nExecution Statistics:")
    print(f"  Total executions: {stats['total_executions']}")
    print(f"  Successful: {stats['successful_executions']}")
    print(f"  Failed: {stats['failed_executions']}")
    print(f"  Success rate: {stats['success_rate_percent']:.1f}%")
    print(f"  Active executions: {stats['active_executions']}")

    print("\nNote: Signal execution is disabled for safety.")
    print("To enable, set enabled=True and configure API credentials.")

    # Note: In production, you would call:
    # executor.execute_signal(signal_payload, wait=False)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TRADERAIHELPER REFACTORED COMPONENTS EXAMPLES")
    print("="*70)

    try:
        example_secrets_manager()
        example_health_checker()
        example_cache_manager()
        example_pydantic_models()
        example_signal_executor()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nFor production usage:")
        print("1. Set environment variables: BYBIT_API_KEY, BYBIT_API_SECRET")
        print("2. Configure .streamlit/secrets.toml for Streamlit secrets")
        print("3. Enable signal executor with enabled=True")
        print("4. Integrate health checks into monitoring")
        print("5. Add caching to expensive operations")
        print("6. Use Pydantic models for all data validation")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
