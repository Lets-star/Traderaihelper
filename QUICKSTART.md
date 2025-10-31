# Quick Start Guide

## Web Dashboard (Recommended)

The web dashboard provides an interactive interface to visualize token charts with technical indicators.

### Features:
- 📊 **Interactive Charts**: Candlestick charts with Bollinger Bands, RSI, MACD
- 🎯 **Trading Zones**: Visual display of Fair Value Gaps (FVG) and Order Blocks (OB)
- 📈 **Multi-Timeframe Analysis**: Compare trend strength across 5m, 15m, 1h, 4h, 1d
- 🎪 **Signal Detection**: Bullish/bearish signals with confluence scores
- 💾 **Export**: Download analysis as JSON or CSV

### Steps:

1. **Install Dependencies** (first time only)
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Dashboard**
   ```bash
   ./run_web_ui.sh
   # Or directly: streamlit run web_ui.py
   ```

3. **Configure Analysis** (in the sidebar):
   - Select a token (e.g., BINANCE:BTCUSDT)
   - Choose timeframe (15m, 1h, 4h, etc.)
   - Set analysis period (50-1000 bars)
   - Enable "Offline Mode" if Binance is unavailable

4. **Click "Analyze"** to load data and visualize indicators

5. **Explore Tabs**:
   - **Charts**: Main candlestick chart with all indicators
   - **Multi-Timeframe**: Trend analysis across different timeframes
   - **Latest Metrics**: Current market snapshot and statistics
   - **Signals & Zones**: Trading signals and active FVG/OB zones
   - **Export**: Download data in JSON or CSV format

## Command Line Interface

For automation and batch processing:

```bash
# Basic usage
python3 main.py --token my-token --symbol BINANCE:BTCUSDT --timeframe 15m --period 500

# Save to file
python3 main.py --token my-token --symbol BINANCE:ETHUSDT --timeframe 1h --period 600 --output eth_analysis.json

# Offline mode (synthetic data)
python3 main.py --token test --symbol BINANCE:BTCUSDT --timeframe 15m --period 200 --offline

# With multi-symbol confirmation
python3 main.py --token test --symbol BINANCE:BTCUSDT --timeframe 15m --period 500 \
  --multi-symbol BINANCE:ETHUSDT BINANCE:SOLUSDT

# Disable multi-symbol analysis
python3 main.py --token test --symbol BINANCE:BTCUSDT --timeframe 15m --period 500 --disable-multi-symbol
```

## Understanding the Output

### Indicators Displayed:
- **Bollinger Bands**: Price volatility bands (upper, middle, lower)
- **RSI (Relative Strength Index)**: Momentum indicator (0-100)
- **MACD**: Trend-following momentum indicator
- **Volume**: Trading volume bars
- **FVG Zones**: Green (bullish) or red (bearish) transparent rectangles
- **Order Block Zones**: Blue (bullish) or orange (bearish) dashed rectangles
- **Signals**: Green triangle up (buy) or red triangle down (sell)

### Key Metrics:
- **Trend Strength**: 0-100 score combining directional movement and momentum
- **Pattern Score**: Pattern recognition score based on candlestick patterns
- **Sentiment**: Market sentiment estimate (0-100)
- **Confluence Score**: Weighted score (0-10) combining multiple factors
- **Structure State**: Overall market bias (bullish/bearish/neutral)

## Tips

- **Start with 15m timeframe and 200-500 bars** for a good balance
- **Enable Offline Mode** if you're testing or have network issues
- **Check Multi-Timeframe tab** to see alignment across different periods
- **Export JSON** for programmatic processing or AI analysis
- **Use CSV export** for quick viewing in Excel/spreadsheets

## Troubleshooting

### "Streamlit not found"
Install dependencies: `pip install -r requirements.txt`

### "No data available"
- Check your internet connection
- Try offline mode: add `--offline` flag or check "Offline Mode" in UI
- Verify symbol format: should be `BINANCE:SYMBOL` (e.g., `BINANCE:BTCUSDT`)

### Binance rate limiting
- Wait a few minutes between requests
- Use offline mode for testing
- Consider increasing cache TTL in code

### Dependencies conflict
Use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
