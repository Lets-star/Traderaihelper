# Binance Network Robustness & Auto-Refresh Implementation

## Summary

This document describes the hardening of Binance API fetching to handle connection failures, network errors, and the addition of timeframe-based auto-refresh for analyzers.

## Part A: Network Robustness for BinanceKlinesSource

### 1. Enhanced HTTP Client

**Changes:**
- Migrated from `urllib` to `requests` library for better timeout and proxy support
- Added configurable connection timeout (default: 5s) and read timeout (default: 20s)
- Added custom User-Agent header: `indicator-collector/1.0`
- Added automatic proxy support via `HTTP_PROXY` and `HTTPS_PROXY` environment variables

**Usage:**
```python
source = BinanceKlinesSource(
    connect_timeout=5.0,
    read_timeout=20.0,
    user_agent="my-app/1.0"
)
```

### 2. Retry Logic with Exponential Backoff & Jitter

**Changes:**
- Enhanced retry logic with exponential backoff
- Added jitter (randomization) to prevent thundering herd (default: 75% of base delay)
- Retry on:
  - HTTP 429 (Rate Limited)
  - HTTP 5xx (Server Errors)
  - Connection errors (errno 10061, errno 111, WinError 10061)
  - Timeout errors (connect and read)

**Formula:**
```
delay = backoff_base * (2 ^ (attempt - 1)) + random(0, jitter_factor * base_delay)
```

**Usage:**
```python
source = BinanceKlinesSource(
    max_retries=3,
    backoff_base=0.5,
    backoff_jitter=0.75
)
```

### 3. Circuit Breaker Pattern

**Changes:**
- Added circuit breaker to prevent repeated failures to the same endpoint
- Trips after `max_retries` consecutive failures
- Cooldown period (default: 30s) before retrying
- Automatic reset after cooldown

**Usage:**
```python
source = BinanceKlinesSource(
    enable_circuit_breaker=True,
    circuit_breaker_cooldown=30.0
)
```

**Status:**
- Circuit open: Endpoint is temporarily disabled
- Circuit closed: Endpoint is healthy and available

### 4. Health Checks & Endpoint Fallback

**Changes:**
- Added `/api/v3/ping` preflight check
- Added `/api/v3/time` to get accurate server time
- Health check cache (default TTL: 45s) to avoid excessive checks
- Automatic fallback to secondary URLs on failure

**Default URLs:**
1. `https://api.binance.com` (primary)
2. `https://api1.binance.com` (fallback)
3. `https://api2.binance.com` (fallback)

**Configuration:**
```python
# Via constructor
source = BinanceKlinesSource(
    base_url="https://custom.binance.com",
    fallback_urls=["https://backup1.com", "https://backup2.com"]
)

# Via environment variables
export BINANCE_API_BASE_URL="https://custom.binance.com"
export BINANCE_API_FALLBACK_URLS="https://backup1.com,https://backup2.com"
```

### 5. Connection Refused Handling

**Changes:**
- Specific detection of connection refused errors:
  - WinError 10061 (Windows)
  - errno 111 (Linux)
  - Generic "connection refused" messages
- Actionable error messages with guidance:
  - Suggests setting `BINANCE_API_BASE_URL`
  - Suggests configuring HTTP proxy
  - Provides clear explanation of the issue

**Example Error Message:**
```
Connection refused ([WinError 10061] No connection could be made...). 
Binance API may be blocking direct access. Try setting BINANCE_API_BASE_URL 
or configuring an HTTP(S) proxy.
```

### 6. Data Caching for Graceful Degradation

**Changes:**
- Automatic caching of successful fetches per symbol/timeframe
- On failure, returns cached data instead of crashing
- Cache metadata attached to DataFrame via `.attrs`:
  ```python
  df.attrs["binance_status"] = {
      "active_base_url": "https://api.binance.com",
      "used_cache": False,
      "fetched_at": "2024-01-01T10:00:00Z",
      "effective_end_ms": 1704110400000
  }
  ```

**Behavior:**
1. First request succeeds → cache stored
2. Second request fails → cached data returned with warning
3. Third request succeeds → cache updated

### 7. Logging and Status Tracking

**Changes:**
- Structured logging for all retry attempts
- Last error code and message tracked
- Status dict with connection state:
  ```python
  source._last_status = {
      "status": "failure",
      "base_url": "https://api.binance.com",
      "retryable": True,
      "consecutive_failures": 2,
      "error": "Connection refused..."
  }
  ```

## Part B: Timeframe-based Auto-Refresh

### 1. Auto-Refresh Module

**New File:** `indicator_collector/trading_system/auto_refresh.py`

**Features:**
- Refresh interval mapping per timeframe:
  - 1m: 5s refresh
  - 5m: 30s refresh
  - 15m: 60s refresh
  - 1h: 120s refresh
  - 3h: 300s refresh (5 minutes)
  - 4h: 300s refresh (5 minutes)
  - 1d: 600s refresh (10 minutes)

- New closed bar detection:
  ```python
  has_new_closed_bar(last_closed_ts, current_server_time_ms, timeframe)
  ```

- Next bar close calculation:
  ```python
  next_close_ms = calculate_next_bar_close_ms(current_time_ms, timeframe)
  ```

- ETA formatting:
  ```python
  format_eta_seconds(seconds) → "1h 30m 5s"
  ```

### 2. AutoRefreshState Class

**Usage:**
```python
from indicator_collector.trading_system.auto_refresh import AutoRefreshState

state = AutoRefreshState("BTCUSDT", "1h")
state.enabled = True

# Update after fetching data
state.update_last_closed(last_closed_ts)

# Check if refresh needed
if state.should_refresh(current_server_time_ms):
    # Fetch new data
    pass

# Get ETA for next update
seconds, formatted = state.get_next_update_eta()
print(f"Next update in: {formatted}")
```

### 3. Streamlit Integration

**Dependencies Added:**
- `requests>=2.31.0`
- `streamlit-autorefresh>=1.0.1`

**Integration Points:**
To integrate into `web_ui.py`:

```python
from streamlit_autorefresh import st_autorefresh
from indicator_collector.trading_system.auto_refresh import (
    AutoRefreshState,
    get_refresh_interval_ms,
)

# In session state
if "autorefresh_state" not in st.session_state:
    st.session_state.autorefresh_state = {}

# Get or create state
key = f"{symbol}_{timeframe}"
if key not in st.session_state.autorefresh_state:
    st.session_state.autorefresh_state[key] = AutoRefreshState(symbol, timeframe)

state = st.session_state.autorefresh_state[key]

# UI controls
state.enabled = st.toggle("Auto-refresh", value=state.enabled)

# Auto-refresh trigger
if state.enabled:
    interval_ms = get_refresh_interval_ms(timeframe)
    st_autorefresh(interval=interval_ms, key=f"refresh_{key}")

    # Check if new bar available
    if state.should_refresh():
        # Trigger data fetch and recompute
        pass

# Status display
if state.last_closed_ts:
    seconds, eta_str = state.get_next_update_eta()
    st.info(f"⏱️ Next update in: {eta_str}")
```

## Part C: Testing

### 1. New Test Files

**`tests/test_binance_robustness.py`:**
- Circuit breaker tests
- Health check tests
- Fallback URL tests
- Retry logic tests (429, 5xx, connection errors)
- Error message formatting tests
- Caching tests

**`tests/test_auto_refresh.py`:**
- Refresh interval mapping tests
- Next bar calculation tests
- New bar detection tests
- ETA formatting tests
- AutoRefreshState tests

### 2. Updated Test Files

**`tests/test_binance_source.py`:**
- Removed obsolete urllib-based tests
- Maintained integration tests with mocked `_fetch_klines_batch`

### 3. Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_binance_robustness.py
pytest tests/test_auto_refresh.py

# Run with coverage
pytest --cov=indicator_collector tests/
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_BASE_URL` | Primary API endpoint | `https://api.binance.com` |
| `BINANCE_API_FALLBACK_URLS` | Comma-separated fallback URLs | `https://api1.binance.com,https://api2.binance.com` |
| `HTTP_PROXY` | HTTP proxy URL | None |
| `HTTPS_PROXY` | HTTPS proxy URL | None |

### BinanceKlinesSource Parameters

```python
BinanceKlinesSource(
    # Authentication (optional for public endpoints)
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    
    # Rate limiting
    rate_limit_delay: float = 0.1,  # 100ms between requests
    
    # Retry configuration
    max_retries: int = 3,
    backoff_base: float = 0.5,  # seconds
    backoff_jitter: float = 0.75,  # 75% of base delay
    
    # Network configuration
    connect_timeout: float = 5.0,  # seconds
    read_timeout: float = 20.0,  # seconds
    user_agent: str = "indicator-collector/1.0",
    
    # URL configuration
    base_url: Optional[str] = None,
    fallback_urls: Optional[List[str]] = None,
    
    # Circuit breaker
    enable_circuit_breaker: bool = True,
    circuit_breaker_cooldown: float = 30.0,  # seconds
    
    # Health checks
    healthcheck_ttl: float = 45.0,  # seconds
    
    # Testing
    sleep_func: Optional[Callable] = None,
)
```

## Migration Guide

### For Existing Code

Existing code continues to work without changes. The enhancements are backward compatible:

```python
# Old code (still works)
source = BinanceKlinesSource()
df = source.load_candles("BTCUSDT", "1h", start, end)

# New code (with enhancements)
source = BinanceKlinesSource(
    max_retries=5,
    enable_circuit_breaker=True,
    base_url=os.environ.get("BINANCE_API_BASE_URL"),
)
df = source.load_candles("BTCUSDT", "1h", start, end)

# Check if cached data was used
if df.attrs.get("binance_status", {}).get("used_cache"):
    st.warning("Using cached data due to network issues")
```

### For Web UI Integration

To add auto-refresh to an analyzer page:

1. Import dependencies:
```python
from streamlit_autorefresh import st_autorefresh
from indicator_collector.trading_system.auto_refresh import AutoRefreshState
```

2. Initialize state in session:
```python
if "autorefresh" not in st.session_state:
    st.session_state.autorefresh = {}
```

3. Create state for symbol/timeframe:
```python
key = f"{symbol}_{timeframe}"
if key not in st.session_state.autorefresh:
    st.session_state.autorefresh[key] = AutoRefreshState(symbol, timeframe)
state = st.session_state.autorefresh[key]
```

4. Add UI controls:
```python
col1, col2 = st.columns([1, 4])
with col1:
    state.enabled = st.toggle("Auto-refresh", value=state.enabled)
with col2:
    if state.last_closed_ts:
        seconds, eta = state.get_next_update_eta()
        st.caption(f"Next update: {eta}")
```

5. Implement auto-refresh:
```python
if state.enabled:
    interval_ms = get_refresh_interval_ms(timeframe)
    st_autorefresh(interval=interval_ms, key=f"refresh_{key}")
    
    if state.should_refresh():
        # Fetch data and update state
        df = source.load_candles(...)
        last_closed_ts = df["ts"].iloc[-1] + tf.to_milliseconds()
        state.update_last_closed(last_closed_ts)
```

## Troubleshooting

### Connection Refused Errors

**Symptom:** `[WinError 10061] No connection could be made`

**Solutions:**
1. Check if Binance is blocking your IP or region
2. Use a proxy:
   ```bash
   export HTTPS_PROXY="http://proxy.example.com:8080"
   ```
3. Use a fallback URL:
   ```bash
   export BINANCE_API_BASE_URL="https://api1.binance.com"
   ```

### Rate Limiting (429)

**Symptom:** HTTP 429 errors even with backoff

**Solutions:**
1. Increase `rate_limit_delay`:
   ```python
   source = BinanceKlinesSource(rate_limit_delay=0.5)  # 500ms
   ```
2. Reduce request frequency
3. Use API key for higher limits

### Timeout Errors

**Symptom:** Frequent timeout errors

**Solutions:**
1. Increase timeouts:
   ```python
   source = BinanceKlinesSource(
       connect_timeout=10.0,
       read_timeout=30.0
   )
   ```
2. Check network connection
3. Use a more stable proxy or endpoint

### Circuit Breaker Tripped

**Symptom:** "circuit breaker active" messages

**Solutions:**
1. Wait for cooldown period (default: 30s)
2. Fix underlying network issues
3. Use fallback URLs
4. Disable circuit breaker temporarily (not recommended):
   ```python
   source = BinanceKlinesSource(enable_circuit_breaker=False)
   ```

## Performance Considerations

### Memory Usage

- Cache stores one DataFrame per symbol/timeframe
- Typical memory per cached DataFrame: ~100KB - 1MB depending on data size
- Recommendation: Monitor memory if caching many symbols

### Network Usage

- Health checks add 2 requests per source initialization (ping + time)
- Health checks cached for 45s by default
- Rate limit delay prevents overwhelming the API

### Refresh Intervals

- Lower refresh intervals = more responsive but higher load
- Higher refresh intervals = less responsive but lower load
- Current intervals are optimized for balance

## Future Enhancements

Potential improvements not included in this implementation:

1. **Adaptive backoff:** Adjust backoff based on error type and response headers
2. **Request queuing:** Queue requests during circuit breaker cooldown
3. **Multi-region failover:** Automatically detect and switch to regional endpoints
4. **WebSocket integration:** Use WebSocket for real-time updates instead of polling
5. **Distributed caching:** Share cache across multiple instances (Redis, etc.)
6. **Metric collection:** Export retry/failure metrics to monitoring systems

## References

- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Exponential Backoff with Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- [requests Library Documentation](https://requests.readthedocs.io/)
- [streamlit-autorefresh](https://github.com/kmcgrady/streamlit-autorefresh)
