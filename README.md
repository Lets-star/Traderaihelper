# TradingView Indicator Metrics Collector

This project provides both a **web-based dashboard** and a **command line utility** that emulates the logic of the "FVG & Order Block Sync Pro - Enhanced" TradingView indicator. It collects the indicator's derived metrics for a chosen symbol, timeframe, and analysis period, visualizes the results with interactive charts, and makes them available as a machine-friendly JSON payload.

## Features

- **📊 Interactive Web Dashboard**: Visualize token charts with indicators, zones, and signals in real-time
  - Candlestick charts with Bollinger Bands, RSI, and MACD indicators
  - Fair Value Gaps (FVG) and Order Block (OB) zones displayed on charts
  - Multi-timeframe trend analysis with visual strength indicators
  - Trading signals with confluence scores
  - Export data in JSON and CSV formats
  - Easy token, timeframe, and period selection

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
- Dependencies: `streamlit`, `plotly`, `pandas` (installed automatically with requirements.txt)
- Internet access to reach the Binance public REST API for market data (or use offline mode with synthetic data)

## Installation

1. (Optional) Create and activate a virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install project dependencies
   ```bash
   pip install -r requirements.txt
   ```

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

## Notes

- Binance imposes rate limits; avoid rapid repeated requests.
- If Binance data is unavailable, the CLI automatically falls back to deterministic synthetic candles (or you can force this with `--offline`).
- The calculations are deterministic and self-contained, so no indicator code runs remotely.
- All computations are performed locally once OHLCV data has been downloaded (or generated).

## License

This project is provided under the MIT License. See the `LICENSE` file if present or consult the repository maintainers for details.
