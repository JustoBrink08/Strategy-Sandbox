"""
Vectorized Backtesting Engine using high-performance pandas and numpy operations.
Executes custom user Python strategy logic against OHLCV data without lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vectorized Technical Indicator Library (exposed to custom user strategies)
# ---------------------------------------------------------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.fillna(50.0)


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper_band, middle_band, lower_band)."""
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Preset Strategy Templates for the UI Strategy Input Window
# ---------------------------------------------------------------------------

PRESET_STRATEGIES: Dict[str, str] = {
    "Golden Cross (SMA 20 / 50 Trend Following)": """# Strategy: Golden Cross Trend Following
# Available helpers: df (OHLCV DataFrame), pd, np, sma, ema, rsi, bollinger_bands, macd, atr
# Define `entries` & `exits` (boolean Series) OR assign `df['signal']` (1=Long, 0=Flat, -1=Short)

fast_ma = sma(df['Close'], 20)
slow_ma = sma(df['Close'], 50)

entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
""",
    "RSI Mean Reversion (Oversold / Overbought)": """# Strategy: RSI(14) Mean Reversion
# Buys when RSI crosses below 30 (oversold) and exits when RSI crosses above 70 (overbought)

rsi_14 = rsi(df['Close'], period=14)

entries = rsi_14 < 30
exits = rsi_14 > 70
""",
    "Bollinger Band Volatility Breakout": """# Strategy: Bollinger Band Breakout (20-day, 2 StdDev)
# Enters long when price closes above upper band; exits when price reverts below the 20-day midline

upper, mid, lower = bollinger_bands(df['Close'], window=20, num_std=2.0)

entries = df['Close'] > upper
exits = df['Close'] < mid
""",
    "MACD Momentum + Trend Filter (Long/Short)": """# Strategy: MACD Momentum with 200-EMA Regime Filter
# Directly assigns `df['signal']`: 1 for Long, -1 for Short, 0 for Flat

macd_line, signal_line, hist = macd(df['Close'], fast=12, slow=26, signal=9)
trend_ema = ema(df['Close'], span=100)

df['signal'] = np.where(
    (macd_line > signal_line) & (df['Close'] > trend_ema), 1.0,
    np.where((macd_line < signal_line) & (df['Close'] < trend_ema), -1.0, 0.0)
)
""",
}


# ---------------------------------------------------------------------------
# Backtest Result Data Structure
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    ticker: str
    timeseries: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    latest_position: float
    latest_signal_label: str
    execution_time_ms: float


# ---------------------------------------------------------------------------
# Vectorized Execution Engine
# ---------------------------------------------------------------------------

def _signals_from_entries_exits(entries: pd.Series, exits: pd.Series, index: pd.Index) -> pd.Series:
    """
    Convert boolean entry and exit series into a continuous position series (1.0 in-market, 0.0 flat)
    using purely vectorized pandas operations.
    """
    ent = pd.Series(entries, index=index).fillna(False).astype(bool)
    ext = pd.Series(exits, index=index).fillna(False).astype(bool)

    raw = pd.Series(np.nan, index=index, dtype=float)
    raw.loc[ext] = 0.0
    raw.loc[ent] = 1.0  # If both true on same bar, entry takes precedence or can be overridden
    raw.loc[ent & ext] = 0.0
    return raw.ffill().fillna(0.0)


def execute_strategy_code(df: pd.DataFrame, strategy_code: str) -> pd.Series:
    """
    Execute user-supplied Python strategy code against a copy of `df` and return
    a vectorized target position Series aligned to `df.index` (values in [-1.0, 1.0]).

    Supported user code patterns:
    1. Setting `entries` (and optionally `exits`) boolean Series.
    2. Assigning `df['signal']` or `df['position']` or variable `signal` / `positions`.
    3. Defining a `strategy(df)` function that returns a Series or (entries, exits) tuple.
    """
    if df is None or df.empty:
        raise ValueError("Market DataFrame is empty; cannot execute strategy.")

    work_df = df.copy()
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
        "ValueError": ValueError,
    }

    local_env: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "df": work_df,
        "pd": pd,
        "np": np,
        "sma": sma,
        "ema": ema,
        "rsi": rsi,
        "bollinger_bands": bollinger_bands,
        "macd": macd,
        "atr": atr,
    }

    exec(compile(strategy_code, "<strategy_input>", "exec"), local_env, local_env)

    # 1. Check if user defined a `strategy(df)` function
    if "strategy" in local_env and callable(local_env["strategy"]):
        fn_out = local_env["strategy"](work_df)
        if isinstance(fn_out, tuple) and len(fn_out) == 2:
            return _signals_from_entries_exits(fn_out[0], fn_out[1], work_df.index)
        if isinstance(fn_out, pd.Series):
            return fn_out.reindex(work_df.index).fillna(0.0).astype(float).clip(-1.0, 1.0)
        if isinstance(fn_out, pd.DataFrame):
            work_df = fn_out

    # 2. Check if user defined `entries` / `exits` variables or columns
    entries = local_env.get("entries", work_df.get("entries"))
    exits = local_env.get("exits", work_df.get("exits"))
    if entries is not None:
        if exits is None:
            exits = pd.Series(False, index=work_df.index)
        return _signals_from_entries_exits(entries, exits, work_df.index)

    # 3. Check if user set `signal` or `position` variable or DataFrame column
    for candidate_key in ("position", "positions", "signal", "signals"):
        if candidate_key in local_env and isinstance(local_env[candidate_key], (pd.Series, np.ndarray, list)):
            s = pd.Series(local_env[candidate_key], index=work_df.index)
            return s.fillna(0.0).astype(float).clip(-1.0, 1.0)
        if candidate_key in work_df.columns:
            return work_df[candidate_key].fillna(0.0).astype(float).clip(-1.0, 1.0)

    raise ValueError(
        "Strategy code must define either (`entries` and `exits`), `df['signal']`, "
        "`df['position']`, `signal`, or a `strategy(df)` function."
    )


def _extract_trades(
    index: pd.Index,
    close: pd.Series,
    executed_pos: pd.Series,
    strategy_net_returns: pd.Series,
) -> pd.DataFrame:
    """Extract discrete round-trip trade records from vectorized position series."""
    pos_diff = executed_pos.diff().fillna(executed_pos)
    change_indices = np.flatnonzero(pos_diff.values != 0)

    trades: List[Dict[str, Any]] = []
    pos_vals = executed_pos.values
    close_vals = close.values
    ret_vals = strategy_net_returns.values

    for i, start_idx in enumerate(change_indices):
        direction = pos_vals[start_idx]
        if direction == 0.0:
            continue
        end_idx = change_indices[i + 1] if (i + 1) < len(change_indices) else len(pos_vals) - 1
        entry_date = index[start_idx]
        exit_date = index[end_idx]
        entry_price = float(close_vals[max(0, start_idx - 1)])
        exit_price = float(close_vals[end_idx])
        slice_rets = ret_vals[start_idx : end_idx + 1]
        compounded_ret = float(np.prod(1.0 + slice_rets) - 1.0)
        bars_held = int(end_idx - start_idx + 1)

        trades.append(
            {
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Direction": "LONG" if direction > 0 else "SHORT",
                "Entry Price": round(entry_price, 4),
                "Exit Price": round(exit_price, 4),
                "Return (%)": round(compounded_ret * 100.0, 2),
                "Bars Held": bars_held,
            }
        )

    return pd.DataFrame(trades)


def run_vectorized_backtest(
    df: pd.DataFrame,
    strategy_code: str,
    ticker: str = "SPY",
    initial_capital: float = 100_000.0,
    commission_bps: float = 5.0,
    risk_free_rate: float = 0.04,
    bars_per_year: int = 252,
) -> BacktestResult:
    """
    Execute a high-speed vectorized backtest on historical OHLCV data.

    - Strictly shifts target positions by 1 bar (`target_pos.shift(1)`) so signals computed
      at bar t close are entered during bar t+1.
    - Deducts proportional transaction costs (`commission_bps`) on position turnover.
    - Benchmarks against a fully invested Buy-and-Hold baseline over the identical window.
    """
    import time

    t0 = time.perf_counter()

    target_pos = execute_strategy_code(df, strategy_code)
    close = df["Close"].astype(float)
    asset_returns = close.pct_change().fillna(0.0)

    # Executed position is shifted by 1 bar to eliminate lookahead bias
    executed_pos = target_pos.shift(1).fillna(0.0)

    # Turnover occurs whenever executed position changes
    turnover = executed_pos.diff().abs().fillna(executed_pos.abs())
    cost_rate = commission_bps / 10_000.0
    transaction_costs = turnover * cost_rate

    strategy_gross_returns = executed_pos * asset_returns
    strategy_net_returns = strategy_gross_returns - transaction_costs

    # Equity curves
    strategy_cum_growth = (1.0 + strategy_net_returns).cumprod()
    benchmark_cum_growth = (1.0 + asset_returns).cumprod()

    strategy_equity = initial_capital * strategy_cum_growth
    benchmark_equity = initial_capital * benchmark_cum_growth

    # Drawdown curves
    strategy_peak = strategy_equity.cummax()
    strategy_drawdown = (strategy_equity / strategy_peak) - 1.0

    benchmark_peak = benchmark_equity.cummax()
    benchmark_drawdown = (benchmark_equity / benchmark_peak) - 1.0

    # Assemble output timeseries DataFrame
    ts = pd.DataFrame(
        {
            "Open": df["Open"],
            "High": df["High"],
            "Low": df["Low"],
            "Close": close,
            "Volume": df.get("Volume", pd.Series(0.0, index=df.index)),
            "Signal": target_pos,
            "Position": executed_pos,
            "Position_Change": executed_pos.diff().fillna(0.0),
            "Asset_Return": asset_returns,
            "Strategy_Return": strategy_net_returns,
            "Strategy_Equity": strategy_equity,
            "Benchmark_Equity": benchmark_equity,
            "Strategy_Drawdown": strategy_drawdown * 100.0,
            "Benchmark_Drawdown": benchmark_drawdown * 100.0,
        },
        index=df.index,
    )

    trades_df = _extract_trades(df.index, close, executed_pos, strategy_net_returns)

    # Compute Quantitative Performance Metrics
    n_bars = max(1, len(ts))
    years = n_bars / float(bars_per_year)

    total_ret = float(strategy_cum_growth.iloc[-1] - 1.0)
    bench_ret = float(benchmark_cum_growth.iloc[-1] - 1.0)
    alpha_ret = total_ret - bench_ret

    cagr = float((strategy_cum_growth.iloc[-1] ** (1.0 / max(years, 1e-6))) - 1.0) if strategy_cum_growth.iloc[-1] > 0 else -1.0
    bench_cagr = float((benchmark_cum_growth.iloc[-1] ** (1.0 / max(years, 1e-6))) - 1.0) if benchmark_cum_growth.iloc[-1] > 0 else -1.0

    ann_vol = float(strategy_net_returns.std(ddof=1) * np.sqrt(bars_per_year)) if n_bars > 1 else 0.0
    bench_vol = float(asset_returns.std(ddof=1) * np.sqrt(bars_per_year)) if n_bars > 1 else 0.0

    daily_rf = risk_free_rate / float(bars_per_year)
    excess_daily = strategy_net_returns - daily_rf
    sharpe = (
        float(excess_daily.mean() / strategy_net_returns.std(ddof=1) * np.sqrt(bars_per_year))
        if ann_vol > 1e-9
        else 0.0
    )

    downside = excess_daily[excess_daily < 0.0]
    downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(bars_per_year)) if len(downside) > 0 else 0.0
    sortino = float((excess_daily.mean() * bars_per_year) / downside_std) if downside_std > 1e-9 else 0.0

    max_dd = float(strategy_drawdown.min())
    bench_max_dd = float(benchmark_drawdown.min())
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

    if not trades_df.empty:
        wins = trades_df[trades_df["Return (%)"] > 0]
        losses = trades_df[trades_df["Return (%)"] <= 0]
        win_rate = float(len(wins) / len(trades_df)) * 100.0
        gross_profit = float(wins["Return (%)"].sum()) if not wins.empty else 0.0
        gross_loss = float(abs(losses["Return (%)"].sum())) if not losses.empty else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0 else 0.0)
    else:
        win_rate = 0.0
        profit_factor = 0.0

    exposure_pct = float((executed_pos != 0).mean() * 100.0)
    latest_pos = float(target_pos.iloc[-1])
    latest_label = "LONG" if latest_pos > 0 else ("SHORT" if latest_pos < 0 else "FLAT")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    metrics = {
        "Initial Capital ($)": round(initial_capital, 2),
        "Final Equity ($)": round(float(strategy_equity.iloc[-1]), 2),
        "Benchmark Final Equity ($)": round(float(benchmark_equity.iloc[-1]), 2),
        "Total Return (%)": round(total_ret * 100.0, 2),
        "Benchmark Return (%)": round(bench_ret * 100.0, 2),
        "Excess Return / Alpha (%)": round(alpha_ret * 100.0, 2),
        "CAGR (%)": round(cagr * 100.0, 2),
        "Benchmark CAGR (%)": round(bench_cagr * 100.0, 2),
        "Annualized Volatility (%)": round(ann_vol * 100.0, 2),
        "Benchmark Volatility (%)": round(bench_vol * 100.0, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Sortino Ratio": round(sortino, 2),
        "Calmar Ratio": round(calmar, 2),
        "Max Drawdown (%)": round(max_dd * 100.0, 2),
        "Benchmark Max Drawdown (%)": round(bench_max_dd * 100.0, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else 99.99,
        "Total Trades": int(len(trades_df)),
        "Exposure Time (%)": round(exposure_pct, 2),
        "Execution Time (ms)": round(elapsed_ms, 2),
    }

    return BacktestResult(
        ticker=ticker.upper(),
        timeseries=ts,
        trades=trades_df,
        metrics=metrics,
        latest_position=latest_pos,
        latest_signal_label=latest_label,
        execution_time_ms=round(elapsed_ms, 2),
    )
