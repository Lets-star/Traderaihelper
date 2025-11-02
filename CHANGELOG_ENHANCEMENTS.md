# Market Context & Data Quality Enhancements

## Overview
This document summarizes the major enhancements made to the TradingView Indicator Metrics Collector to address data representation issues and add comprehensive market context analysis.

## 🎯 Key Improvements

### 1. Market Context Analysis (MOST IMPORTANT)

#### VWAP (Volume Weighted Average Price)
- **Implementation**: `indicator_collector/market_context.py::calculate_vwap_levels()`
- **Features**:
  - Calculated from typical price ((H+L+C)/3) weighted by volume
  - Standard deviation bands (upper/lower)
  - Distance percentage from current price
  - Historical series (last 50 periods)
- **Access**: `payload['latest']['vwap']` and `payload['advanced']['market_context']['vwap']`

#### Cumulative Delta 24H
- **Implementation**: `indicator_collector/market_context.py::calculate_cumulative_delta_24h()`
- **Features**:
  - Buy/sell pressure analysis over last 24 hours
  - Delta momentum (recent vs earlier periods)
  - Total buy/sell volume breakdown
  - Net delta percentage
  - Time-series data for tracking
- **Access**: `payload['advanced']['market_context']['cumulative_delta_24h']`

#### Liquidation Heatmap
- **Implementation**: `indicator_collector/market_context.py::calculate_liquidation_heatmap()`
- **Features**:
  - Liquidation clusters for leverage levels: 5x, 10x, 20x, 25x, 50x, 100x
  - Estimated volumes at each liquidation price
  - Separate long/short liquidation prices
  - High-risk zone identification
- **Access**: `payload['advanced']['market_context']['liquidation_heatmap']`

#### Orderbook Cluster Data
- **Enhanced Depth**: Increased from 100 to 500-1000 levels
- **Implementation**: `indicator_collector/data_fetcher.py::fetch_order_book()` + `market_context.py::analyze_orderbook_context()`
- **Features**:
  - Market maker presence indicators
  - Liquidity skew (bid/ask imbalance)
  - Stability score (book depth vs recent volume)
  - Liquidity shelves (concentrated order zones)
  - Aggregated bins at 5%, 10%, 20% ranges
- **Access**: `payload['advanced']['market_context']['orderbook_context']`

### 2. Time Pattern Analysis

#### Trading Session Breakdown
- **Implementation**: `indicator_collector/market_context.py::analyze_trading_session()`
- **Features**:
  - Asian session (00:00-08:00 UTC): volume, price change, candle count
  - European session (08:00-13:00 UTC): volume, price change, candle count
  - US session (13:00-21:00 UTC): volume, price change, candle count
  - Current session detection
  - Volume percentage distribution
- **Access**: `payload['advanced']['market_context']['trading_sessions']`

### 3. Fundamental Metrics

#### Stablecoin Flows
- **Implementation**: `indicator_collector/market_context.py::analyze_stablecoin_flows()`
- **Features**:
  - USDT/USDC inflow/outflow estimates
  - Net flow calculation
  - Flow momentum classification (strong_inflow, weak_inflow, neutral, weak_outflow, strong_outflow)
  - Based on volume patterns and price action
- **Access**: `payload['advanced']['fundamentals']['stablecoin_flows']`

#### ETH Network Activity
- **Implementation**: `indicator_collector/market_context.py::analyze_eth_network_activity()`
- **Features**:
  - Gas price estimates (Gwei)
  - Network utilization percentage
  - Transaction count estimates
  - Congestion level classification
  - Average block time
- **Access**: `payload['advanced']['fundamentals']['eth_network']`

### 4. Technical Indicator Enhancements

#### VWAP Integration
- **Implementation**: `indicator_collector/math_utils.py::vwap()`
- Added to main simulation loop in `indicator_metrics.py`
- Available in every `MarketSnapshot`
- **Access**: `payload['latest']['vwap']`

#### SMA Support
- **Implementation**: Enhanced existing `math_utils.py::sma()`
- Added SMA calculations for fast (20) and slow (50) periods alongside EMA
- **Access**: `payload['latest']['sma_fast']`, `payload['latest']['sma_slow']`

#### RSI Divergence Detection
- **Implementation**: `indicator_collector/math_utils.py::detect_divergence()`
- **Types Detected**:
  - Regular bullish divergence (price lower low, RSI higher low)
  - Regular bearish divergence (price higher high, RSI lower high)
  - Hidden bullish divergence (price higher low, RSI lower low)
  - Hidden bearish divergence (price lower high, RSI higher high)
- **Access**: `payload['latest']['rsi_divergence']`, `payload['advanced']['divergences']`

#### MACD Divergence Detection
- Same implementation as RSI, applied to MACD line
- **Access**: `payload['latest']['macd_divergence']`, `payload['advanced']['divergences']`

### 5. Volume Confirmation Improvements

#### Enhanced Volume Analysis
- **Implementation**: `indicator_collector/indicator_metrics.py::_volume_confirmed()`
- **Improvements**:
  - Primary check: volume > SMA * multiplier
  - Fallback: recent 5-bar window analysis
  - Dynamic threshold: 70% of multiplier with 1.05x minimum
  - Z-score outlier detection (>1.2σ)
  - Handles edge cases with missing/NaN data

#### Volume Confidence Score
- **Implementation**: Added to `MarketSnapshot` and volume analysis
- **Features**:
  - Normalized 0-1 score
  - Based on volume ratio distribution
  - Considers recent volume statistics (mean, median, stdev)
  - Outlier score for anomaly detection
- **Access**: `payload['latest']['volume_confidence']`, `payload['advanced']['volume_analysis']['context']`

#### Smart Money Detection
- **Implementation**: `indicator_collector/advanced_metrics.py::calculate_volume_analysis()`
- **Features**:
  - Identifies volume outliers (>2x median)
  - Tracks institutional-sized trades
  - Direction classification (buy/sell)
  - Volume ratio vs threshold
  - Last 40 bars analyzed, top 10 events returned
- **Access**: `payload['advanced']['volume_analysis']['smart_money']`

### 6. Data Representation Fixes

#### Orderbook Depth
- **Before**: 100 levels
- **After**: 500-1000 levels
- **Files Changed**:
  - `indicator_collector/data_fetcher.py` (fetch_order_book, generate_synthetic_order_book)
  - `indicator_collector/collector.py` (increased limit parameter)

#### Volume Confirmation
- **Before**: Simple threshold check, often returned `false`
- **After**: Multi-stage validation with fallback mechanisms
- **Result**: More accurate volume confirmation, better signal quality

#### Moving Averages
- **Before**: Only EMA available
- **After**: Both SMA and EMA calculated
- **Benefits**: More traditional technical analysis support, VWAP added

#### Divergences
- **Before**: Not tracked
- **After**: RSI and MACD divergences automatically detected
- **Benefits**: Early reversal signals, hidden divergence patterns identified

## 📊 New Data Structure

### Market Context Section
```json
{
  "advanced": {
    "market_context": {
      "vwap": {
        "vwap": 131.50,
        "vwap_upper": 146.42,
        "vwap_lower": 116.59,
        "vwap_distance_pct": -6.1,
        "std_dev": 14.91,
        "vwap_series": [...]
      },
      "cumulative_delta_24h": {
        "cumulative_delta": 1234.56,
        "delta_momentum": 45.67,
        "buy_volume_24h": 12345.67,
        "sell_volume_24h": 11111.11,
        "net_delta_pct": 10.5,
        "delta_series": [...]
      },
      "liquidation_heatmap": {
        "liquidation_zones": [
          {
            "leverage": 10,
            "long_liquidation_price": 90000,
            "long_estimated_volume": 5000000,
            "short_liquidation_price": 110000,
            "short_estimated_volume": 4500000
          }
        ],
        "high_risk_long": 90000,
        "high_risk_short": 110000
      },
      "trading_sessions": {
        "asian_session": {"volume": 14508.6, "volume_pct": 46.0, "price_change_pct": -21.42},
        "european_session": {"volume": 7212.53, "volume_pct": 22.87, "price_change_pct": -14.04},
        "us_session": {"volume": 9821.77, "volume_pct": 31.14, "price_change_pct": -23.92},
        "current_session": "european"
      },
      "orderbook_context": {
        "maker_presence": {"bids": 4.48, "asks": 6.39, "depth_levels": {"bids": 500, "asks": 500}},
        "liquidity_skew": 135.38,
        "stability_score": 158.07,
        "liquidity_shelves": [...]
      }
    }
  }
}
```

### Enhanced Latest Snapshot
```json
{
  "latest": {
    "vwap": 131.50,
    "sma_fast": 162.69,
    "sma_slow": 149.78,
    "volume_confirmed": false,
    "volume_ratio": 1.26,
    "volume_confidence": 0.61,
    "rsi_divergence": "hidden_bullish",
    "macd_divergence": null
  }
}
```

### Volume Analysis Enhancement
```json
{
  "advanced": {
    "volume_analysis": {
      "context": {
        "latest_volume": 189.45,
        "average_volume": 152.8,
        "median_volume": 148.2,
        "volume_ratio": 1.24,
        "outlier_score": 1.85,
        "volume_confidence": 0.573
      },
      "smart_money": [
        {
          "timestamp": 1761934091745,
          "price": 124.71,
          "volume": 227.69,
          "direction": "sell",
          "volume_ratio": 2.15
        }
      ]
    }
  }
}
```

## 🔧 Technical Implementation Details

### Files Modified
1. **indicator_collector/math_utils.py**
   - Added `vwap()` function
   - Added `detect_divergence()` function
   - Enhanced divergence detection algorithm

2. **indicator_collector/indicator_metrics.py**
   - Added VWAP, SMA calculations to simulation
   - Enhanced `_volume_confirmed()` with multi-stage validation
   - Added divergence tracking for RSI and MACD
   - Extended `MarketSnapshot` dataclass with new fields
   - Updated `summary_to_payload()` with new data

3. **indicator_collector/advanced_metrics.py**
   - Enhanced `calculate_volume_analysis()` with smart money detection
   - Integrated new market context modules
   - Added divergence aggregation

4. **indicator_collector/data_fetcher.py**
   - Increased orderbook depth from 100 to 500 (default), max 1000

5. **indicator_collector/collector.py**
   - Updated orderbook fetch limits

6. **indicator_collector/market_context.py** (NEW)
   - `calculate_vwap_levels()` - VWAP with bands
   - `calculate_cumulative_delta_24h()` - Buy/sell pressure
   - `calculate_liquidation_heatmap()` - Liquidation clusters
   - `analyze_trading_session()` - Session breakdown
   - `analyze_stablecoin_flows()` - Stablecoin metrics
   - `analyze_eth_network_activity()` - Network metrics
   - `analyze_orderbook_context()` - Enhanced orderbook analysis

7. **README.md**
   - Updated feature list with new capabilities

## 🎓 Usage Examples

### Accessing VWAP Data
```python
import json
data = json.load(open('output.json'))

# Latest VWAP
vwap = data['latest']['vwap']
vwap_upper = data['advanced']['market_context']['vwap']['vwap_upper']
vwap_lower = data['advanced']['market_context']['vwap']['vwap_lower']
distance_pct = data['advanced']['market_context']['vwap']['vwap_distance_pct']
```

### Checking Volume Quality
```python
# Multiple volume indicators
volume_confirmed = data['latest']['volume_confirmed']
volume_ratio = data['latest']['volume_ratio']
volume_confidence = data['latest']['volume_confidence']

# Smart money activity
smart_money_events = data['advanced']['volume_analysis']['smart_money']
```

### Liquidation Analysis
```python
liquidation_map = data['advanced']['market_context']['liquidation_heatmap']
high_risk_long = liquidation_map['high_risk_long']
high_risk_short = liquidation_map['high_risk_short']

for zone in liquidation_map['liquidation_zones']:
    print(f"{zone['leverage']}x Long Liq: {zone['long_liquidation_price']}")
```

### Session Trading Patterns
```python
sessions = data['advanced']['market_context']['trading_sessions']
current = sessions['current_session']
us_volume_pct = sessions['us_session']['volume_pct']
```

## ✅ Testing

All enhancements have been tested with:
- Offline synthetic data generation
- Live Binance data integration
- Multiple timeframes (15m, 1h, 4h, 1d)
- Various symbols (BTCUSDT, ETHUSDT, etc.)

Sample test:
```bash
python3 main.py --symbol BINANCE:ETHUSDT --timeframe 1h --period 200 --token test --offline --output test.json
```

## 📝 Notes

- All synthetic data generation uses deterministic seeding for reproducibility
- Stablecoin and ETH network metrics are estimates based on volume patterns
- Liquidation heatmap calculations assume standard leverage and margin requirements
- Session times are in UTC (Asian: 00-08, European: 08-13, US: 13-21)
- Volume confidence score ranges from 0.0 (no confidence) to 1.0 (high confidence)
- Divergences only appear in output when detected (not "none")

## 🚀 Future Enhancements

Potential areas for further development:
- Real-time stablecoin flow tracking via blockchain APIs
- Actual ETH network metrics from Etherscan/Infura
- On-chain liquidation data from DeFi protocols
- Machine learning for volume pattern classification
- Multi-exchange orderbook aggregation
- Historical divergence success rates
