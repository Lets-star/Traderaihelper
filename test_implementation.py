#!/usr/bin/env python3
"""
Comprehensive test script for the timeframe API and explicit JSON signals implementation.

This script tests the main requirements from the ticket:
1) Timeframe API normalization with validate_timeframe method
2) 3h timeframe support end-to-end
3) Explicit JSON signals output
"""

import sys
sys.path.insert(0, '/home/engine/project')

from indicator_collector.trading_system import (
    Timeframe, validate_timeframe, indicator_defaults_for,
    generate_signals, is_valid_signal_structure,
    load_and_process_payload_dict
)

def test_timeframe_api():
    """Test Timeframe API with aliases support."""
    print("=== Testing Timeframe API ===")
    
    # Test from_value with aliases
    assert Timeframe.from_value("3h") == Timeframe.H3
    assert Timeframe.from_value("180m") == Timeframe.H3
    assert Timeframe.from_value("1h") == Timeframe.H1
    assert Timeframe.from_value("60m") == Timeframe.H1
    assert Timeframe.from_value("5m") == Timeframe.M5
    assert Timeframe.from_value("1d") == Timeframe.D1
    assert Timeframe.from_value("24h") == Timeframe.D1
    
    # Test validate_timeframe method
    assert Timeframe.validate_timeframe("3h") == Timeframe.H3
    assert Timeframe.validate_timeframe("180m") == Timeframe.H3
    
    # Test is_supported
    assert Timeframe.is_supported("3h") == True
    assert Timeframe.is_supported("180m") == True
    assert Timeframe.is_supported("invalid") == False
    
    # Test to_minutes class method
    assert Timeframe.to_minutes("3h") == 180
    assert Timeframe.to_minutes("180m") == 180
    assert Timeframe.to_minutes("1h") == 60
    
    print("✅ Timeframe API tests passed!")


def test_3h_timeframe_support():
    """Test 3h timeframe support end-to-end."""
    print("\n=== Testing 3h Timeframe Support ===")
    
    # Test indicator defaults for 3h
    h3_defaults = indicator_defaults_for("3h")
    h1_defaults = indicator_defaults_for("1h")
    
    # Should be different for 3h
    assert h3_defaults != h1_defaults
    assert h3_defaults['atr']['mult'] == 1.3
    assert h1_defaults['atr']['mult'] == 1.0
    
    print(f"✅ 3h ATR multiplier: {h3_defaults['atr']['mult']}")
    print(f"✅ 1h ATR multiplier: {h1_defaults['atr']['mult']}")
    
    # Test processing 3h payload
    payload_3h = {
        'metadata': {'timeframe': '3h', 'source': 'binance'},
        'latest': {
            'timestamp': 1640995200000,
            'close': 50000.0,
            'volume': 300.5,
            'rsi': 55.0,
            'macd': 10.5,
            'atr': 450.0
        }
    }
    
    result = load_and_process_payload_dict(payload_3h, '3h', validate_real_data=False)
    assert result['timeframe'] == '3h'
    
    print("✅ 3h timeframe processing successful!")


def test_explicit_json_signals():
    """Test explicit JSON signals generation."""
    print("\n=== Testing Explicit JSON Signals ===")
    
    # Test payload
    test_payload = {
        'metadata': {'timeframe': '1h', 'source': 'binance'},
        'latest': {
            'timestamp': 1640995200000,
            'close': 50000.0,
            'volume': 100.5,
            'rsi': 55.0,
            'macd': 10.5,
            'atr': 150.0
        }
    }
    
    # Process payload
    result = load_and_process_payload_dict(test_payload, '1h', validate_real_data=False)
    
    # Generate explicit signal
    signal = generate_signals(result)
    
    # Validate signal structure
    assert is_valid_signal_structure(signal) is True
    
    # Check required fields
    required_fields = [
        'signal', 'confidence', 'entries', 'stop_loss',
        'take_profits', 'position_size_pct', 'holding_period',
        'rationale', 'weights', 'timeframe'
    ]
    
    for field in required_fields:
        assert field in signal, f"Missing required field: {field}"
    
    # Validate field types and values
    assert signal['signal'] in ['BUY', 'SELL', 'HOLD']
    assert isinstance(signal['confidence'], int)
    assert 1 <= signal['confidence'] <= 10
    assert isinstance(signal['entries'], list)
    assert signal['holding_period'] in ['short', 'medium', 'long']
    assert isinstance(signal['rationale'], list)
    assert isinstance(signal['weights'], dict)
    
    if signal['signal'] in {'BUY', 'SELL'}:
        assert len(signal['entries']) >= 1
        assert isinstance(signal['stop_loss'], (int, float))
        assert signal['stop_loss'] > 0
        assert isinstance(signal['take_profits'], dict)
        assert {'tp1', 'tp2', 'tp3'}.issubset(signal['take_profits'].keys())
        assert isinstance(signal['position_size_pct'], (int, float))
        assert 0 <= signal['position_size_pct'] <= 100
        assert len(signal['rationale']) >= 1
    else:
        # HOLD signal should not fabricate execution levels
        assert signal['entries'] == []
        assert signal['stop_loss'] is None
        assert signal['take_profits'] == {}
        assert signal['position_size_pct'] is None
        assert len(signal['rationale']) >= 1
    
    # Check weights sum to approximately 1.0
    weight_sum = sum(signal['weights'].values())
    assert abs(weight_sum - 1.0) < 0.01, f"Weights sum to {weight_sum}, expected ~1.0"
    
    print("✅ Explicit JSON signals generated successfully!")
    print(f"✅ Signal: {signal['signal']} (confidence: {signal['confidence']}/10)")
    if signal['signal'] in {'BUY', 'SELL'}:
        print(f"✅ Entry: ${signal['entries'][0]:.2f}")
        print(f"✅ Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"✅ Take Profits: {signal['take_profits']}")
        print(f"✅ Position Size: {signal['position_size_pct']:.1f}%")
    else:
        print("✅ No actionable levels returned; signal remains on HOLD")
    print(f"✅ Weights: {signal['weights']}")


def test_web_ui_integration():
    """Test that the web UI can consume the new signals."""
    print("\n=== Testing Web UI Integration ===")
    
    # Test that we can import all required functions for web UI
    try:
        from indicator_collector.trading_system import (
            load_and_process_payload_dict,
            generate_signals,
            is_valid_signal_structure
        )
        print("✅ All web UI imports successful!")
    except ImportError as e:
        print(f"❌ Web UI import failed: {e}")
        return False
    
    # Test signal generation with realistic data
    realistic_payload = {
        'metadata': {
            'timeframe': '3h',
            'source': 'binance',
            'symbol': 'BTCUSDT'
        },
        'latest': {
            'timestamp': 1640995200000,
            'open': 50000.0,
            'high': 50100.0,
            'low': 49900.0,
            'close': 50050.0,
            'volume': 300.5,
            'rsi': 55.0,
            'macd': 10.5,
            'atr': 450.0
        },
        'indicators': {
            'trend_strength': 65.0,
            'pattern_score': 70.0,
            'market_sentiment': 60.0
        }
    }
    
    try:
        processed = load_and_process_payload_dict(realistic_payload, '3h', validate_real_data=False)
        explicit_signal = generate_signals(processed)
        
        if is_valid_signal_structure(explicit_signal):
            print("✅ Web UI integration test passed!")
            print(f"✅ Generated {explicit_signal['signal']} signal for {explicit_signal['timeframe']} timeframe")
            return True
        else:
            print("❌ Generated signal failed validation")
            return False
            
    except Exception as e:
        print(f"❌ Web UI integration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running Comprehensive Implementation Tests\n")
    
    try:
        test_timeframe_api()
        test_3h_timeframe_support()
        test_explicit_json_signals()
        web_ui_success = test_web_ui_integration()
        
        if web_ui_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("\n✅ Acceptance Criteria Met:")
            print("  ✅ No AttributeError for Timeframe.validate_timeframe")
            print("  ✅ Timeframe API is consistent with aliases support")
            print("  ✅ Automated signals produce explicit, schema-valid JSON")
            print("  ✅ 3h timeframe fully supported end-to-end")
            print("  ✅ Web UI can consume explicit JSON signals")
            return True
        else:
            print("\n❌ Some tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)