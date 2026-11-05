# Strategy-Sandbox: Vectorized Trading Strategy Backtester & Forward-Testing Engine

**Strategy-Sandbox** is a high-performance quantitative research and paper-trading web application built with **Streamlit**, **`yfinance`**, **vectorized `pandas`/`numpy`**, and **`APScheduler`**. It enables quantitative developers and traders to dynamically write custom Python trading logic, run sub-10ms vectorized historical backtests against a Buy-and-Hold benchmark, and deploy strategies to an asynchronous background forward-testing ("paper trading") engine.

---

## 1. Key Capabilities & Architecture

```text
Strategy-Sandbox/
├── app.py               # Streamlit UI: Strategy Input Window, Instant Backtester, Paper Trading Monitor
├── backtester.py        # Vectorized pandas/numpy execution engine & technical indicator library
├── data_loader.py       # Two-tier (RAM + local disk pickle) TTL-cached yfinance market data fetcher
├── forward_tester.py    # Background APScheduler daemon + SQLite persistence for live paper trading
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

### Fast Multi-Layer Market Data Caching (`data_loader.py`)
- Fetches historical and latest daily/intraday OHLCV market bars via `yfinance`.
- Implements a **two-tier TTL cache**:
  1. **In-Memory (`RLock`-protected RAM dictionary):** Sub-millisecond lookups when tweaking strategy parameters on the same ticker and timeframe.
  2. **On-Disk Pickle Cache (`.market_cache/`):** Persists normalized OHLCV DataFrames across process restarts to eliminate redundant Yahoo Finance API calls.
  3. **Streamlit `@st.cache_data` Layer:** Prevents UI re-fetches during interactive widget updates.

### Zero-Loop Vectorized Backtesting Engine (`backtester.py`)
- Executes custom user Python code dynamically in a sandboxed namespace with pre-loaded vectorized technical indicators (`sma`, `ema`, `rsi`, `bollinger_bands`, `macd`, `atr`).
- **Lookahead-Bias Protection:** Target signals computed on bar $t$'s close are strictly shifted by 1 bar (`target_pos.shift(1).fillna(0)`) so trades execute on bar $t+1$.
- **Realistic Transaction Costs:** Applies configurable commission and slippage in basis points (`commission_bps`) proportional to position turnover (` executed_pos.diff().abs() `).
- **Institutional Performance Metrics:** Calculates Total Return, Buy-and-Hold Benchmark Return, Excess Return (Alpha), CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Max Drawdown, Win Rate, Profit Factor, Exposure Time, and a full round-trip Trade Log.

### Background Forward-Testing / Paper Trading Engine (`forward_tester.py`)
- Powered by `APScheduler` (`BackgroundScheduler`, with an automatic Python daemon-thread fallback) initialized as a singleton via `@st.cache_resource`.
- **How the Background Task Works:**
  1. **Strategy Deployment:** When you click **"Activate Forward-Test"** in the UI, the strategy's Python code, ticker, capital allocation, and deployment baseline are saved to a local thread-safe SQLite database (`forward_testing_state.db`).
  2. **Continuous Asynchronous Polling:** Every 60 seconds (or immediately when clicking **"Sync Now"**), the background scheduler wakes up independently of the UI thread, queries `yfinance` for the latest daily market bars of every `ACTIVE` strategy, and executes the strategy's Python logic against the updated rolling window.
  3. **Mark-to-Market PnL & Paper Ledger:** The worker updates each strategy's current signal (`LONG`, `SHORT`, `FLAT`), mark-to-market paper equity, cumulative return, forward PnL since deployment, trailing 90-day equity curve, and timestamped paper fill ledger—even while you are actively backtesting completely different tickers and strategies in the main UI.

---

## 2. Writing Custom Strategies in the UI

In the **Custom Python Strategy Logic** editor, you have direct access to:
- `df`: Normalized OHLCV `pandas.DataFrame` with columns `Open`, `High`, `Low`, `Close`, `Volume`.
- `pd` (`pandas`) and `np` (`numpy`).
- Built-in vectorized indicators: `sma(series, window)`, `ema(series, span)`, `rsi(series, period=14)`, `bollinger_bands(series, window=20, num_std=2.0)`, `macd(series, fast=12, slow=26, signal=9)`, and `atr(df, period=14)`.

### Option A: Define Boolean `entries` and `exits`
```python
fast_ma = sma(df['Close'], 20)
slow_ma = sma(df['Close'], 50)

entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
```

### Option B: Assign Target Weights Directly to `df['signal']` (`1.0` = Long, `0.0` = Flat, `-1.0` = Short)
```python
macd_line, signal_line, hist = macd(df['Close'], fast=12, slow=26, signal=9)
trend_ema = ema(df['Close'], span=100)

df['signal'] = np.where(
    (macd_line > signal_line) & (df['Close'] > trend_ema), 1.0,
    np.where((macd_line < signal_line) & (df['Close'] < trend_ema), -1.0, 0.0)
)
```

---

## 3. Running the Application Locally

### Prerequisites
- Python 3.9+

### Installation
```bash
cd Strategy-Sandbox
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Launch the Streamlit Web Interface
```bash
python3 -m streamlit run app.py
```
The web interface will open automatically at `http://localhost:8501`.
