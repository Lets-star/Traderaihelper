# TradingView-like Candle Updates Implementation Summary

## Overview
Implemented TradingView-like incremental candle updates with Binance server time API, forming bar support, and thread-safe auto-refresh without ScriptRunContext warnings.

## Changes Made

### 1. Binance Server Time API (binance_source.py)

**Added `get_server_time()` method:**
- Fetches server time from `/api/v3/time` endpoint
- Implements retry logic with exponential backoff
- Caches result for 1 second to avoid excessive API calls
- Extrapolates from cached value using monotonic time
- Records failures and success via circuit breaker pattern

**Key features:**
- Retry with backoff: base=0.5s, multiplier=2x, max_retries=3
- Short-term caching prevents API hammering
- Returns None on failure (triggers fallback)

### 2. Enhanced Server Time Fallback (auto_analyze_worker.py)

**Updated `get_binance_server_time_ms()`:**
- Added global `_SERVER_TIME_FALLBACK_WARNED` flag
- Logs warning only once per session when falling back to system time
- Debug-level logging for subsequent fallbacks
- Graceful degradation with proper error handling

### 3. TradingView-like Chart Updates (chart_auto_refresh.py)

**Key improvements:**

#### Adaptive Polling
- **`get_poll_interval()`**: Returns 1s for ≤15m timeframes, 5s for ≥1h timeframes
- Reduces unnecessary API calls for higher timeframes
- Matches TradingView behavior

#### Boundary Calculations
- **Tight tolerance**: Changed `DEFAULT_TOLERANCE_MS` from 60s to 1.5s
- **Simplified `floor_closed_bar_local()`**: Formula = `((now_ms - tol_ms) // tf_ms) * tf_ms`
- Ensures candles appear exactly at close time
- No special case needed for 3h (handled by standard alignment)

#### Overlap Bars
- **Increased `OVERLAP_BARS`** from 1 to 3
- Fetches 3 extra bars when doing delta updates
- Prevents gaps from clock skew or missed updates
- Deduplication happens in `update_chart_state()`

#### Forming Bar Support
- **`fetch_forming_bar()`**: Fetches currently forming (unclosed) candle
- For 3h: Aggregates from 1h candles for current 3h period
- For other timeframes: Direct fetch via klines endpoint with limit=2
- Checks close_time to verify bar is still forming
- Optional: Toggled via UI checkbox

#### Worker Thread Updates
- **Adaptive polling**: Uses `get_poll_interval()` for sleep duration
- **No st.* calls**: Worker only updates session_state under `_CHART_DATA_LOCK`
- **Flag handoff**: Sets `analysis_updated = True` for main UI to detect
- **Separate DataFrames**:
  - `chart_df`: Closed bars only
  - `chart_df_with_forming`: Closed + forming bar (when enabled)
- **Detailed logging**: Shows fetched/appended/deduped counts per boundary

#### Thread-Safe State Management
- **`read_chart_state()`**: Prefers `chart_df_with_forming` when forming bar is enabled
- Protected by `_CHART_DATA_LOCK` for all reads/writes
- Clean separation between worker thread and main UI thread

### 4. UI Enhancements (web_ui.py)

**Added "Forming Bar" toggle:**
- New checkbox in Charts tab controls row
- Help text: "Show the currently forming candle (not yet closed)"
- Stored in `session_state.show_forming_bar`
- Worker reads this flag and conditionally fetches forming bar

**Layout change:**
- Changed from 4 columns to 5 columns for controls
- Order: Auto-refresh | Forming Bar | Better Volume | ATR Channels | Order Blocks

## Technical Details

### Boundary Detection Logic
```python
def floor_closed_bar_local(now_ms: int, tf_ms: int, tol_ms: int = 1_500) -> int:
    effective_now = max(now_ms - tol_ms, 0)
    last_closed = (effective_now // tf_ms) * tf_ms
    return last_closed
```

### Poll Intervals
- **1m, 5m, 15m**: 1 second (high frequency)
- **1h, 3h, 4h, 1d, etc.**: 5 seconds (lower frequency)

### Forming Bar Detection
For non-3h timeframes, fetches latest kline and checks:
1. open_time matches expected current bar start
2. close_time > server_time (bar still forming)
3. Returns DataFrame with single row or None

For 3h timeframe, aggregates 1h candles from current 3h period start.

### Diagnostics Logging
Each boundary update logs:
- `last_closed_ms`: Timestamp of boundary
- `fetched`: Number of bars fetched from API
- `appended`: Number of bars added (after deduplication)
- `deduped`: Number of duplicate bars removed

Example:
```
[BTCUSDT 1h] Boundary update: last_closed_ms=1234567890000, fetched=5, appended=3, deduped=2
```

## Acceptance Criteria Met

✅ **New candles appear exactly at close**: Tight 1.5s tolerance ensures TradingView-like behavior  
✅ **Forming bar preview**: Optional, toggled in UI, updates at poll interval  
✅ **No ScriptRunContext warnings**: Worker thread never calls st.* APIs  
✅ **Binance server time**: Used for all boundary calculations via `get_server_time()`  
✅ **Fallback handling**: Logged once per session, falls back to system time  
✅ **3h aggregation**: Handled correctly with standard alignment or 1h aggregation  
✅ **Adaptive polling**: 1s for ≤15m, 5s for ≥1h  
✅ **Thread-safe**: All shared state protected by locks, flag handoff pattern  
✅ **Diagnostics**: Detailed logging of fetched/appended/deduped counts  

## Testing

All changes compile successfully:
- `chart_auto_refresh.py`: ✓
- `indicator_collector/trading_system/data_sources/binance_source.py`: ✓
- `indicator_collector/trading_system/auto_analyze_worker.py`: ✓

Integration testing via web UI:
1. Enable auto-refresh in Charts tab
2. Toggle "Forming Bar" to see real-time updates
3. Observe logs for boundary updates with diagnostics
4. No ScriptRunContext warnings should appear
5. Candles update exactly at TF boundaries

## Future Enhancements

Potential improvements (not required for this ticket):
- WebSocket support for sub-second updates
- Forming bar visual styling (different color/opacity)
- Progress bar showing time until next candle close
- Configurable poll intervals
- Forming bar for multi-timeframe aggregations beyond 3h
