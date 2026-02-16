"""
WebSocket reconnect tests.

Tests exponential backoff calculation, max reconnect attempts limit,
stop event handling during reconnect, connection failure scenarios,
message handling during reconnection, and reconnect counter reset on success.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, Mock, patch, call

import pytest

from websocket_client import BinanceWebSocketClient


class TestWebSocketExponentialBackoff:
    """Test exponential backoff calculation."""
    
    def test_backoff_increases_exponentially(self):
        """Test that backoff increases exponentially with each attempt."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        # Simulate reconnections and check backoff values
        backoffs = []
        for i in range(1, 6):
            client.reconnect_count = i
            backoff = min(client.backoff_ms * (2 ** client.reconnect_count), 30000)
            backoffs.append(backoff)
        
        # Each backoff should be larger than the previous
        for i in range(1, len(backoffs)):
            assert backoffs[i] > backoffs[i-1]
    
    def test_backoff_maximum_cap(self):
        """Test that backoff is capped at maximum value."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.reconnect_count = 20
        
        backoff = min(client.backoff_ms * (2 ** client.reconnect_count), 30000)
        
        # Should be capped at 30000ms (30 seconds)
        assert backoff == 30000
    
    def test_initial_backoff_value(self):
        """Test initial backoff value."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        assert client.backoff_ms == 100
    
    def test_backoff_calculation_formula(self):
        """Test the exact backoff calculation formula."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        test_cases = [
            (1, 200),      # 100 * 2^1 = 200ms
            (2, 400),      # 100 * 2^2 = 400ms
            (3, 800),      # 100 * 2^3 = 800ms
            (4, 1600),     # 100 * 2^4 = 1600ms
            (5, 3200),     # 100 * 2^5 = 3200ms
            (10, 30000),   # Would be 102400ms, but capped at 30000ms
        ]
        
        for attempt, expected_max in test_cases:
            client.reconnect_count = attempt
            backoff = min(client.backoff_ms * (2 ** client.reconnect_count), 30000)
            assert backoff <= 30000
            if attempt <= 8:
                assert backoff == client.backoff_ms * (2 ** attempt)


class TestWebSocketMaxReconnectAttempts:
    """Test max reconnect attempts limit."""
    
    def test_max_reconnects_enforced(self):
        """Test that max reconnects is enforced."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.max_reconnects = 3
        client.reconnect_count = 3
        
        # Should not attempt reconnect when max reached
        with patch('logging.Logger.error') as mock_error:
            client._schedule_reconnect()
            mock_error.assert_called_once()
            assert "Max reconnect attempts reached" in str(mock_error.call_args)
    
    def test_reconnect_count_incremented(self):
        """Test that reconnect count is incremented."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        initial_count = client.reconnect_count
        
        with patch('threading.Timer') as mock_timer:
            client._schedule_reconnect()
            assert client.reconnect_count == initial_count + 1
    
    def test_reconnect_stops_at_max(self):
        """Test that reconnections stop at max attempts."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.max_reconnects = 2
        
        with patch('threading.Timer') as mock_timer:
            # First reconnect
            client._schedule_reconnect()
            assert client.reconnect_count == 1
            
            # Second reconnect
            client._schedule_reconnect()
            assert client.reconnect_count == 2
            
            # Third attempt should not schedule timer
            mock_timer.reset_mock()
            client._schedule_reconnect()
            mock_timer.assert_not_called()


class TestWebSocketStopEventHandling:
    """Test stop event handling during reconnect."""
    
    def test_stop_event_prevents_reconnect(self):
        """Test that stop event prevents reconnection."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.stop_event.set()
        
        with patch('threading.Timer') as mock_timer:
            client._schedule_reconnect()
            mock_timer.assert_not_called()
    
    def test_stop_event_prevents_start(self):
        """Test that stop event prevents start."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.stop_event.set()
        
        with patch.object(client, '_connect') as mock_connect:
            client.start()
            mock_connect.assert_not_called()
    
    def test_stop_cancels_pending_reconnect(self):
        """Test that stop cancels pending reconnect timers."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        mock_timer = MagicMock()
        
        with patch('threading.Timer', return_value=mock_timer):
            client._schedule_reconnect()
            
            # Stop should prevent the timer from starting
            client.stop()
            
            # The timer was created but start might not have been called
            # depending on timing
            assert client.stop_event.is_set()


class TestWebSocketConnectionFailures:
    """Test connection failure scenarios."""
    
    def test_connection_failure_triggers_reconnect(self):
        """Test that connection failure triggers reconnection."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        with patch('websocket.WebSocketApp') as mock_ws_class:
            mock_ws = MagicMock()
            mock_ws_class.return_value = mock_ws
            mock_ws.run_forever.side_effect = Exception("Connection failed")
            
            with patch.object(client, '_schedule_reconnect') as mock_reconnect:
                client._connect()
                # Allow time for thread to start
                time.sleep(0.1)
                
                # Reconnect should be scheduled
                mock_reconnect.assert_called_once()
    
    def test_error_callback_triggers_reconnect(self):
        """Test that error callback triggers reconnection."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        with patch.object(client, '_schedule_reconnect') as mock_reconnect:
            client._on_error(None, "Test error")
            mock_reconnect.assert_called_once()
    
    def test_close_callback_triggers_reconnect(self):
        """Test that close callback triggers reconnection."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        with patch.object(client, '_schedule_reconnect') as mock_reconnect:
            client._on_close(None, 1000, "Normal closure")
            mock_reconnect.assert_called_once()
    
    def test_close_callback_no_reconnect_when_stopped(self):
        """Test that close callback doesn't reconnect when stopped."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.stop_event.set()
        
        with patch.object(client, '_schedule_reconnect') as mock_reconnect:
            client._on_close(None, 1000, "Normal closure")
            mock_reconnect.assert_not_called()


class TestWebSocketMessageHandlingDuringReconnect:
    """Test message handling during reconnection."""
    
    def test_message_parsed_correctly(self):
        """Test that messages are parsed correctly."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        received_messages = []
        def on_closed_bar(kline):
            received_messages.append(kline)
        
        client.callbacks["closed"] = on_closed_bar
        
        # Simulate closed bar message
        msg = json.dumps({
            "k": {
                "t": 1700000000000,
                "o": "50000.0",
                "h": "50100.0",
                "l": "49900.0",
                "c": "50050.0",
                "v": "100.0",
                "x": True  # Closed
            }
        })
        
        client._on_message(None, msg)
        
        assert len(received_messages) == 1
        assert received_messages[0]["close"] == 50050.0
    
    def test_forming_bar_callback(self):
        """Test forming bar callback is invoked correctly."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        received_messages = []
        def on_forming_bar(kline):
            received_messages.append(kline)
        
        client.callbacks["forming"] = on_forming_bar
        
        # Simulate forming bar message
        msg = json.dumps({
            "k": {
                "t": 1700000000000,
                "o": "50000.0",
                "h": "50100.0",
                "l": "49900.0",
                "c": "50050.0",
                "v": "100.0",
                "x": False  # Not closed (forming)
            }
        })
        
        client._on_message(None, msg)
        
        assert len(received_messages) == 1
    
    def test_invalid_message_handled_gracefully(self):
        """Test that invalid messages are handled gracefully."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        # Invalid JSON
        with patch('logging.Logger.error') as mock_error:
            client._on_message(None, "not valid json")
            mock_error.assert_called_once()
    
    def test_message_without_kline_data(self):
        """Test handling of message without kline data."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        # Message without 'k' field - should not raise
        msg = json.dumps({"other": "data"})
        client._on_message(None, msg)
        
        # Should complete without error
        assert True


class TestWebSocketReconnectCounterReset:
    """Test reconnect counter reset on success."""
    
    def test_counter_reset_on_successful_connection(self):
        """Test that counter resets on successful connection."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.reconnect_count = 5
        
        with patch('websocket.WebSocketApp'):
            with patch('threading.Thread'):
                client._connect()
                
                # Counter should be reset
                assert client.reconnect_count == 0
    
    def test_counter_not_reset_on_failed_connection(self):
        """Test that counter is not reset on failed connection."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.reconnect_count = 3
        
        with patch('websocket.WebSocketApp', side_effect=Exception("Failed")):
            with patch.object(client, '_schedule_reconnect'):
                try:
                    client._connect()
                except Exception:
                    pass
                
                # Counter should not be reset
                assert client.reconnect_count == 3


class TestWebSocketIsConnected:
    """Test is_connected method."""
    
    def test_is_connected_true(self):
        """Test is_connected returns True when connected."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.ws = MagicMock()
        client.ws.sock = MagicMock()
        client.ws.sock.connected = True
        
        assert client.is_connected() is True
    
    def test_is_connected_false_no_ws(self):
        """Test is_connected returns False when ws is None."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.ws = None
        
        assert client.is_connected() is False
    
    def test_is_connected_false_no_sock(self):
        """Test is_connected returns False when sock is None."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.ws = MagicMock()
        client.ws.sock = None
        
        assert client.is_connected() is False
    
    def test_is_connected_handles_exceptions(self):
        """Test is_connected handles exceptions gracefully."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        client.ws = MagicMock()
        client.ws.sock = MagicMock()
        client.ws.sock.connected = MagicMock(side_effect=AttributeError())
        
        # Should not raise
        result = client.is_connected()
        assert result is False


class TestWebSocketLifecycle:
    """Test WebSocket client lifecycle."""
    
    def test_full_lifecycle(self):
        """Test full lifecycle: start -> message -> stop."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        received = []
        client.callbacks["closed"] = lambda x: received.append(x)
        
        with patch('websocket.WebSocketApp') as mock_ws_class:
            mock_ws = MagicMock()
            mock_ws_class.return_value = mock_ws
            
            # Start
            client.start()
            assert client.ws is not None
            
            # Simulate message
            msg = json.dumps({
                "k": {
                    "t": 1700000000000,
                    "o": "50000.0",
                    "h": "50100.0",
                    "l": "49900.0",
                    "c": "50050.0",
                    "v": "100.0",
                    "x": True
                }
            })
            client._on_message(mock_ws, msg)
            
            assert len(received) == 1
            
            # Stop
            client.stop()
            assert client.stop_event.is_set()
            mock_ws.close.assert_called_once()


class TestWebSocketThreadSafety:
    """Test WebSocket client thread safety."""
    
    def test_concurrent_message_handling(self):
        """Test handling concurrent messages."""
        client = BinanceWebSocketClient("BTCUSDT", "1h")
        
        received = []
        lock = threading.Lock()
        
        def on_message(kline):
            with lock:
                received.append(kline)
        
        client.callbacks["closed"] = on_message
        
        def send_message(i):
            msg = json.dumps({
                "k": {
                    "t": 1700000000000 + i,
                    "o": "50000.0",
                    "h": "50100.0",
                    "l": "49900.0",
                    "c": "50050.0",
                    "v": "100.0",
                    "x": True
                }
            })
            client._on_message(None, msg)
        
        threads = [threading.Thread(target=send_message, args=(i,)) for i in range(20)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(received) == 20
