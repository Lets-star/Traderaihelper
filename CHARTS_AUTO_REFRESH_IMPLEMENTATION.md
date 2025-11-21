# Charts Tab Auto-Refresh Implementation Summary

## Overview
Implemented auto-refresh functionality for the Charts tab that updates the Plotly chart and displayed data automatically when:
1. A new bar closes for the selected timeframe
2. The user changes symbol/timeframe/period

## Implementation Details

### 1. New Module: `chart_auto_refresh.py`

Created a dedicated module containing:

- **`floor_closed_bar_local()`**: Calculates the timestamp of the last closed bar with special handling for 3h timeframe alignment (00:00, 03:00, 06:00, etc.)
- **`fetch_closed_candles()`**: Fetches only CLOSED bars using BinanceKlinesSource, with caching support keyed by (symbol, timeframe, start_ms, end_ms)
- **`invalidate_cache()`**: Clears cache entries for a specific symbol/timeframe combination
- **`ChartAutoRefreshWorker`**: Background daemon thread that:
  - Monitors Binance server time
  - Detects new closed bars
  - Fetches updated candle data
  - Updates session state with new data
  - Sleeps until next boundary with max(5s, next_boundary - now)

### 2. Updated `web_ui.py`

#### Session State Initialization
Added new session state fields in `main()`:
- `chart_symbol`: Current chart symbol
- `chart_timeframe`: Current chart timeframe
- `chart_period`: Current number of bars to display
- `chart_df`: DataFrame containing closed candle data
- `last_closed_ts`: Timestamp of last closed bar in milliseconds
- `analysis_updated`: Flag indicating data has been updated by worker
- `worker_running`: Worker thread status
- `chart_worker`: Reference to ChartAutoRefreshWorker instance
- `auto_refresh_enabled`: User toggle for auto-refresh feature
- `bvi_enabled`: User toggle for Better Volume Indicator

#### New Chart Rendering Function
Added `create_realtime_candlestick_chart()`:
- Creates chart directly from DataFrame (ts, open, high, low, close, volume)
- Computes RSI, MACD, and Bollinger Bands on-the-fly
- Supports Better Volume Indicator coloring when enabled
- Independent of SimulationSummary/TimeframeSeries objects

#### Charts Tab Implementation
Completely redesigned the Charts tab with:

1. **Control Row**: Checkboxes for auto-refresh and BVI toggle, live status indicator

2. **Stable Containers**: 
   - `chart_box_placeholder` for the chart
   - `chart_status_placeholder` for status messages
   - Prevents DOM errors and flicker

3. **Change Detection**:
   - Detects symbol, timeframe, or period changes
   - Stops worker when changes detected or auto-refresh disabled
   - Resets state and invalidates cache (only for symbol/timeframe changes)
   - Fetches initial data synchronously before starting worker

4. **Worker Management**:
   - Starts ChartAutoRefreshWorker when auto-refresh enabled
   - Worker runs in daemon thread
   - Automatically stops/restarts on configuration changes

5. **Chart Rendering**:
   - Uses cached DataFrame when available
   - Falls back to original chart from SimulationSummary
   - Updates status text with last closed bar timestamp and bar count
   - No page rerun needed (worker updates session state asynchronously)

### 3. Cache Management

- Candles cached by (symbol, timeframe, start_ms, last_closed_ts)
- Thread-safe with lock protection
- Cache invalidated only on symbol/timeframe change (not period change)
- Supports `use_cache` parameter for forced fresh fetches

### 4. Closed Bar Logic

#### For Standard Timeframes:
```
current_bar_start = (now_ms // tf_ms) * tf_ms
last_closed = current_bar_start - tf_ms
```

#### For 3h Timeframe:
```
day_start_ms = (now_ms // 86_400_000) * 86_400_000
elapsed_from_day_start = now_ms - day_start_ms
current_3h_index = elapsed_from_day_start // tf_ms
current_bar_start = day_start_ms + (current_3h_index * tf_ms)
last_closed = current_bar_start - tf_ms
```

With tolerance check to avoid edge cases near boundaries.

### 5. Supported Timeframes

All standard timeframes supported:
- 1m, 3m, 5m, 15m, 30m
- 1h, 2h, 3h, 4h, 6h, 8h, 12h
- 1d, 3d, 1w

Special handling for 3h with proper UTC alignment.

## Usage

1. Navigate to the Charts tab
2. Enable "🔄 Auto-refresh on new bars" checkbox
3. Chart will automatically update when new bars close
4. Change symbol/timeframe/period to see instant updates
5. Toggle "📊 Better Volume Indicator" to enable/disable BVI coloring

## Benefits

- **No Manual Refresh**: Charts update automatically on new closed bars
- **Instant Feedback**: Symbol/timeframe changes trigger immediate data fetch
- **Stable UI**: No page-wide reruns or flicker
- **Efficient**: Uses cached data when possible, only fetches closed bars
- **Flexible**: Works with all timeframes including 3h aggregation
- **Robust**: Background worker handles errors gracefully
- **Clean Separation**: Auto-refresh logic isolated in dedicated module

## Testing Checklist

- [x] Syntax validation (Python compile)
- [ ] Manual smoke test: Switch between 15m ↔ 1h ↔ 3h
- [ ] Verify chart updates on next closed bar
- [ ] Verify no flicker or DOM errors
- [ ] Verify 3h timestamps aligned to 00:00, 03:00, etc.
- [ ] Verify cache invalidation on symbol/timeframe change
- [ ] Verify worker stops/starts correctly
- [ ] Verify fallback to original chart when needed
- [ ] Verify Better Volume Indicator toggle works
- [ ] Verify status text displays correctly

## Notes

- Worker runs as daemon thread (auto-cleanup on app exit)
- Only one worker per session (duplicate prevention)
- Cache invalidation strategy optimized per instructions
- Session state used for communication between worker and main thread
- BinanceKlinesSource handles all data fetching with built-in retries
