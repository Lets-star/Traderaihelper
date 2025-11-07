# Binance API Integration for Real Historical Candles

## Overview
This document describes the implementation of Binance API integration for loading real historical OHLCV data with support for 3h aggregation, timestamp normalization, and validation.

## Implementation Details

### 1. Data Source Architecture

#### New Module: `indicator_collector/trading_system/data_sources/`

Created a modular data sources framework:

- **`interfaces.py`** - Abstract base class `HistoricalDataSource` for all data source implementations
  - `load_candles(symbol, timeframe, start, end) -> pd.DataFrame`
  - Returns normalized DataFrame with columns: `ts` (UTC ms), `open`, `high`, `low`, `close`, `volume`

- **`binance_source.py`** - `BinanceKlinesSource` implementation
  - Direct support for: 1m, 5m, 15m, 1h, 4h, 1d
  - 3h implemented via aggregation (3×1h candles)
  - Pagination support for large date ranges (1000 candles per request)
  - Retry logic with exponential backoff for rate limiting
  - Rate limit respect (configurable delay between requests)

- **`timestamp_utils.py`** - Timestamp normalization and validation
  - `normalize_timestamp()` - Auto-detects seconds vs milliseconds, validates range
  - `validate_timestamps_monotonic()` - Ensures strictly increasing timestamps
  - `validate_no_future_timestamps()` - Rejects future data with 1-minute tolerance

### 2. Binance Data Fetching

#### Features:
- **Pagination**: Fetches large date ranges by requesting up to 1000 candles per batch
- **Retry Logic**: Implements exponential backoff (2× multiplier) with configurable max retries
- **Rate Limiting**: Respects Binance API rate limits (0.1s default delay between requests)
- **Error Handling**: Distinguishes between network errors and rate limits (429)

#### 3h Aggregation:
- Fetches 1h candles and aggregates into 3h periods
- Aligned to UTC hour boundaries: 00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00
- Proper OHLCV stitching:
  - Open: from first 1h candle
  - High: maximum across 3 candles
  - Low: minimum across 3 candles
  - Close: from last 1h candle
  - Volume: sum across 3 candles

### 3. Data Validation & Normalization

Comprehensive validation pipeline:

1. **Type Validation**: Ensures all OHLCV values are numeric
2. **NaN Detection**: Rejects any NaN or None values
3. **Timestamp Validation**:
   - Normalizes to milliseconds
   - Validates monotonicity (strictly increasing)
   - Rejects future timestamps
   - Rejects timestamps outside 2020-2030 range
   - Rejects zero/negative timestamps
4. **OHLCV Relationships**:
   - Checks: low ≤ open ≤ high
   - Checks: low ≤ close ≤ high
5. **Price Validation**: Rejects zero prices
6. **Volume Validation**: Rejects negative volume

### 4. Backtester Integration

Extended `BacktestConfig` with:
- `data_source: Optional[HistoricalDataSource]` - Pluggable data source
- `min_data_points_per_timeframe: Optional[Dict[str, int]]` - Per-timeframe minimums
  - Example: `{"1h": 1000, "3h": 400, "4h": 300, "15m": 2000}`

This allows:
- Using real Binance data instead of demo data
- Validating sufficient data before backtesting
- Timeframe-specific data sufficiency checks

### 5. API & Exports

All new components exported via `indicator_collector/trading_system/__init__.py`:

```python
from .data_sources import (
    HistoricalDataSource,
    BinanceKlinesSource,
    normalize_timestamp,
    validate_timestamps_monotonic,
    validate_no_future_timestamps,
)
```

Usage example:
```python
from indicator_collector.trading_system import BinanceKlinesSource
from datetime import datetime, timedelta
from indicator_collector.timeframes import Timeframe

source = BinanceKlinesSource()
df = source.load_candles(
    symbol="BTCUSDT",
    timeframe="3h",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 1, 31)
)
```

### 6. Comprehensive Test Suite

Created `tests/test_binance_source.py` with 27 tests covering:

#### Timestamp Utilities (11 tests):
- Milliseconds/seconds conversion
- Zero/negative/NaN rejection
- Out-of-range detection
- Monotonicity validation
- Future timestamp detection

#### Data Conversion (3 tests):
- Binance API format → DataFrame conversion
- 1h → 3h aggregation correctness
- UTC hour alignment

#### Data Validation (6 tests):
- Valid data acceptance
- Missing column detection
- NaN rejection
- Zero price rejection
- Negative volume rejection
- OHLC relationship violations

#### API Fetching (3 tests):
- Successful batch fetch
- Rate limit retry (429 handling)
- Max retries exceeded failure

#### Integration Tests (4 tests):
- Loading 1h candles
- Loading 3h candles (aggregated)
- Empty data error handling
- Pagination for large date ranges

## Testing Results

All tests passing with deprecation warnings (datetime.utcnow() → datetime.now(datetime.UTC)):
```
11 passed (timestamp utilities)
3 passed (data conversion)
6 passed (data validation)
3 passed (API fetching)
4 passed (integration)
Total: 27 tests passing
```

## Key Design Decisions

1. **DataFrame Format**: Uses pandas DataFrame for consistency with existing codebase
2. **Timestamp Normalization**: Auto-detection of seconds vs milliseconds for robustness
3. **3h Aggregation**: Client-side aggregation for flexibility and offline support
4. **Retry Strategy**: Exponential backoff respects Binance rate limits without aggressive hammering
5. **Validation First**: All validation happens before returning data to ensure quality
6. **Type-Checked Imports**: Uses TYPE_CHECKING to avoid circular imports

## Future Enhancements

- Support for additional exchanges (Coinbase, Kraken, etc.)
- Caching layer for repeated requests
- WebSocket support for real-time data
- Multi-threading for parallel requests
- Database persistence for historical data

## Acceptance Criteria Met

✓ Binance data source module with HistoricalDataSource base class
✓ Real OHLCV loading (1m, 5m, 15m, 1h, 4h, 1d)
✓ 3h support via aggregation with proper OHLCV stitching
✓ Timestamp normalization (sec/ms auto-detect)
✓ Validation: rejects 0/negative/NaN, verifies monotonicity, checks for future data
✓ Pagination with backoff and retries
✓ Rate limit respect (0.1s delay configurable)
✓ Comprehensive test coverage
✓ Integration with BacktestConfig for real data paths
✓ All exports in trading_system module
✓ Supports per-timeframe min_data_points configuration
✓ RealDataValidator integration ready
