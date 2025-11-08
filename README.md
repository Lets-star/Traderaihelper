# TradingView Indicator Metrics Collector

This project provides both a **web-based dashboard** and a **command line utility** that emulates the logic of the "FVG & Order Block Sync Pro - Enhanced" TradingView indicator. It collects the indicator's derived metrics for a chosen symbol, timeframe, and analysis period, visualizes the results with interactive charts, and makes them available as a machine-friendly JSON payload.

## Features

- **📊 Interactive Web Dashboard**: Visualize token charts with indicators, zones, and signals in real-time
  - Candlestick charts with Bollinger Bands, RSI, and MACD indicators
  - **ATR-based trailing channels** (3x, 8x, 21x multipliers) overlaid on price chart
  - Fair Value Gaps (FVG) and Order Block (OB) zones displayed on charts
  - Multi-timeframe trend analysis with visual strength indicators
  - **Enhanced Confluence Score** calculation showing both bullish and bearish bias components
  - **Deep orderbook analysis** (up to 1000 levels) with 2% aggregated bins for 5%, 10%, 20% ranges
  - **CME Gap Analysis** tracking unfilled CME futures gaps above and below current price
  - **Advanced Trade Signal Statistics** with TP1/TP2/TP3 success rates and SL hit rates
  - **Position Size Calculator** with adjustable leverage and real-time commission calculation
  - **Astrology & Celestial Cycles** integration (Moon phases, Mercury trading cycles, Jupiter/Bitcoin halving correlation)
  - Trading signals with weighted confluence scores
  - Export data in JSON and CSV formats
  - Easy token, timeframe, and period selection

- **🎯 Market Context Analysis (Enhanced)**:
  - **VWAP (Volume Weighted Average Price)** - Key intraday level with standard deviation bands
  - **Cumulative Delta 24H** - Buy/sell pressure analysis over last 24 hours
  - **Liquidation Heatmap** - Estimated liquidation clusters at various leverage levels (5x-100x)
  - **Trading Session Analysis** - Activity breakdown by Asian/European/US market hours
  - **Orderbook Depth Context** - Real Binance orderbook ingestion with market maker detection, liquidity skew, and stability scores
  - **Smart Money Detection** - Volume outlier analysis flagging institutional-sized trades

- **📈 Enhanced Technical Analysis**:
  - **SMA Support** - Simple Moving Averages alongside existing EMA calculations
  - **RSI Divergence Detection** - Regular and hidden divergences automatically identified
  - **MACD Divergence Detection** - Momentum divergences with price action
  - **Volume Confidence Score** - Normalized 0-1 score indicating volume reliability
  - **Improved Volume Confirmation** - Multi-threshold volume analysis with fallback logic

- **🤖 Real-Time Market Maker Detection**:
  - **Order Walls Detection** - Identifies large orders (3.5x+ average) at key price levels
  - **Layered Orders Analysis** - Detects multiple consecutive orders (market making patterns)
  - **Quote Stuffing Detection** - Flags suspicious order concentration and manipulation
  - **Spread Manipulation Analysis** - Monitors bid-ask spread quality and manipulation indicators
  - **Activity Confidence Scoring** - 0-100% confidence in market maker presence
  - **Real Binance Data Only** - All detection algorithms use live orderbook data (up to 1000 levels)

- **💰 Fundamental Metrics**:
  - **Stablecoin Flow Analysis** - USDT/USDC inflow/outflow estimates and momentum
  - **ETH Network Activity** - Gas prices, network utilization, transaction count estimates
  - **Funding Rates & Open Interest** - Synthetic estimates based on market dynamics
  - **Long/Short Ratio Analysis** - Position imbalance detection

- **🔧 Command Line Interface**: Batch processing and automation
  - Fetches historical OHLCV data directly from Binance for the selected symbol and timeframe
  - Reproduces the indicator's calculations, including:
    - Multi-timeframe trend strength and directional alignment
    - Market structure (BOS/CHOCH) detection
    - Fair Value Gaps (FVG) and Order Block (OB) zone tracking
    - Pattern recognition and sentiment estimation
    - Signal generation with weighted confluence scoring
    - Trade performance statistics and signal success rates
  - Optional multi-symbol confirmation across up to three additional pairs
  - Outputs a comprehensive JSON document with both raw metric values and human-readable definitions

## Requirements

- Python 3.10 or higher
- Dependencies: `streamlit`, `plotly`, `pandas` (installed automatically via `pyproject.toml`)
- Internet access to reach the Binance public REST API for market data (or use offline mode with synthetic data)

## Installation

1. (Optional) Create and activate a virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install the project in editable mode (includes test extras)
   ```bash
   pip install -e ".[dev]"
   ```
   For a runtime-only environment, use `pip install -e .`.

## Running the Web Dashboard

Start the dashboard with the helper script:

```bash
./run_web_ui.sh
```

Once Streamlit launches, open the provided URL (defaults to [http://localhost:8501](http://localhost:8501)). Configure the token, timeframe, and analysis period from the sidebar, then press **Analyze** to load charts, indicators, and export options. Enable *Offline Mode* if Binance data is not accessible.

## Command Line Usage

```
usage: main.py [-h] [--symbol SYMBOL] [--timeframe TIMEFRAME] [--period PERIOD]
               --token TOKEN [--output OUTPUT]
               [--multi-symbol [MULTI_SYMBOL ...]] [--disable-multi-symbol]
               [--additional-timeframes [ADDITIONAL_TIMEFRAMES ...]]
```

### Arguments

- `--symbol`: Main symbol to analyse (default: `BINANCE:BTCUSDT`).
- `--timeframe`: Chart timeframe (default: `15m`).
- `--period`: Number of bars to include in the analysis window (default: `500`).
- `--token`: Required string that is echoed in the output payload. Use this to tag the request for downstream services.
- `--output`: Optional path to a file where the JSON payload will be written. If omitted, the payload prints to stdout.
- `--offline`: Generate deterministic synthetic OHLCV data instead of requesting it from Binance (useful when network access is restricted).
- `--multi-symbol`: Up to three additional symbols for multi-symbol confirmation logic (default: `BINANCE:ETHUSDT BINANCE:SOLUSDT`).
- `--disable-multi-symbol`: Skip fetching and evaluating extra symbols even if the flag above is present.
- `--additional-timeframes`: Add more comparison timeframes beyond the built-in set (`5m`, `15m`, `1h`, `4h`, `1d`).

### Example

```bash
python3 main.py \
  --symbol BINANCE:BTCUSDT \
  --timeframe 15m \
  --period 600 \
  --token sample-token-123 \
  --output btcusdt_metrics.json
```

The command above writes a JSON payload summarising the indicator metrics for the latest 600 bars on the 15-minute timeframe. The payload includes definitions so downstream consumers can interpret each measurement without referencing the original indicator code.

## Output Structure

The generated JSON contains the following top-level keys:

- `metadata`: Basic context (symbol, timeframe, requested period, token, timestamp).
- `latest`: Snapshot of the most recent bar with calculated indicator metrics.
- `multi_timeframe`: Trend strength and direction for each supporting timeframe.
- `zones`: Active FVG and OB zones that remain on the chart.
- `signals`: History of detected bullish/bearish signals with their confluence scores.
- `success_rates`: Win-rate statistics based on the indicator's success lookahead logic.
- `pnl_stats`: Aggregate performance figures assuming CHOCH-based exits.
- `last_structure_levels`: Latest BOS levels derived from structure analysis.
- `multi_symbol`: Optional snapshot summarising alignment across additional symbols.
- `definitions`: Short explanations of each major metric category.

## Automated Trading System

The project includes a complete automated trading system for programmatic signal generation and position management:

### Core Components

- **Signal Generator** (`trading_system/signal_generator.py`): Combines multi-factor analysis into trading decisions
- **Technical Analysis** (`trading_system/technical_analysis.py`): Computes technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Sentiment Analyzer** (`trading_system/sentiment_analyzer.py`): Evaluates market sentiment from multiple data sources
- **Multi-Timeframe Analyzer** (`trading_system/multitimeframe_analyzer.py`): Analyzes trends across different timeframes
- **Volume Orderbook Analyzer** (`trading_system/volume_orderbook_analyzer.py`): Analyzes volume and orderbook depth
- **Position Manager** (`trading_system/position_manager.py`): Calculates position sizing with risk management
- **Statistics Optimizer** (`trading_system/statistics_optimizer.py`): Tracks performance and optimizes weights

### Quick Integration Example

```python
from indicator_collector.trading_system import (
    SignalGenerator,
    AnalyzerContext,
    TradingSignalPayload,
)

# Create signal generator with custom configuration
generator = SignalGenerator(
    technical_weight=0.25,
    sentiment_weight=0.15,
    multitimeframe_weight=0.10,
    volume_weight=0.20,
    structure_weight=0.15,
    composite_weight=0.15,
)

# Create analyzer context with market data
context = AnalyzerContext(
    symbol="BTCUSDT",
    timeframe="1h",
    current_price=45000.0,
    current_time=1699000000000,
    indicators={
        "rsi": 65.0,
        "macd_histogram": 0.15,
        "bollinger_position": 0.6,
        "atr": 300.0,
    },
)

# Generate signal
signal = generator.generate_signal(context)

# Access signal data
print(f"Signal: {signal.signal_type}")  # BUY, SELL, or NEUTRAL
print(f"Confidence: {signal.confidence:.2%}")
print(f"Position Plan: {signal.position_plan}")
```

### Position Sizing and Risk Management

The position manager automatically calculates:
- Entry, Stop Loss, and Take Profit levels
- Position size based on account risk tolerance
- Take Profit ladder levels for scaling out
- Risk-to-Reward ratio calculation

### Backtesting Workflow

For backtesting your trading strategy, see `samples/BACKTESTING_WORKFLOW.md` for a complete step-by-step guide.

### JSON Signal Format

See `samples/trading_signal_schema.json` for the complete signal structure.

### Macro Filter Configuration

Configure macro-level filters to adapt signals to market conditions. See `samples/macro_filter_config.json` for examples.

## Notes

- Binance imposes rate limits; avoid rapid repeated requests.
- If Binance data is unavailable, the CLI automatically falls back to deterministic synthetic candles (or you can force this with `--offline`).
- The calculations are deterministic and self-contained, so no indicator code runs remotely.
- All computations are performed locally once OHLCV data has been downloaded (or generated).
- See `AUTOMATED_SIGNALS_INTEGRATION.md` for complete developer integration guide.

## Requirements

- Python 3.12 or higher (3.10 minimum for compatibility)
- Dependencies: `streamlit`, `plotly`, `pandas` (installed automatically with requirements.txt)
- Development: `pytest>=7.4.0`, `pytest-cov>=4.1.0` (for testing)

## License

This project is provided under the MIT License. See the `LICENSE` file if present or consult the repository maintainers for details.
