"""Smoke tests ensuring key modules can be imported without circular dependencies."""


def test_import_indicator_collector_root() -> None:
    import indicator_collector  # noqa: F401

    from indicator_collector import load_full_payload, main

    assert callable(main)
    assert callable(load_full_payload)


def test_import_trading_system_package() -> None:
    import indicator_collector.trading_system as trading_system

    assert hasattr(trading_system, "TradingOrchestrator")
    assert hasattr(trading_system, "generate_signals")


def test_collection_result_type_exposed() -> None:
    from indicator_collector.types import CollectionResult

    assert CollectionResult.__name__ == "CollectionResult"
