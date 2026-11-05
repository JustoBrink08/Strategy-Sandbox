"""
Streamlit Frontend for the High-Performance Quantitative Backtesting & Forward-Testing Engine.
Run locally with: streamlit run app.py
"""

from __future__ import annotations

import site
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

# Automatically attach local .venv site-packages if Streamlit was launched from global Python
for _venv_sp in (Path(__file__).resolve().parent / ".venv" / "lib").glob("python*/site-packages"):
    if _venv_sp.exists() and str(_venv_sp) not in sys.path:
        site.addsitedir(str(_venv_sp))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from backtester import PRESET_STRATEGIES, BacktestResult, run_vectorized_backtest
from data_loader import clear_market_cache, fetch_market_data
from forward_tester import ForwardTestingEngine

st.set_page_config(
    page_title="QuantPulse | Vectorized Backtest & Forward-Test Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Singleton Background Forward-Testing Scheduler & Cached Market Data Loader
# ---------------------------------------------------------------------------

@st.cache_resource
def get_forward_engine() -> ForwardTestingEngine:
    """Initialize and persist the background APScheduler paper-trading engine across UI reruns."""
    return ForwardTestingEngine.get_instance(poll_interval_seconds=60)


@st.cache_data(ttl=900, show_spinner=False)
def cached_load_data(ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Streamlit memory-cached wrapper around disk/RAM cached yfinance loader."""
    return fetch_market_data(
        ticker=ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        ttl_seconds=900,
    )


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------

def build_equity_and_drawdown_chart(result: BacktestResult) -> go.Figure:
    ts = result.timeseries

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            f"Cumulative Equity Curve ($) — {result.ticker} Strategy vs. Buy & Hold Benchmark",
            "Underlying Asset Price & Executed Trade Signals",
            "Underwater Drawdown (%)",
        ),
    )

    # Row 1: Equity Curves
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Strategy_Equity"],
            mode="lines",
            name="Strategy Equity ($)",
            line=dict(color="#00E676", width=2.4),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Benchmark_Equity"],
            mode="lines",
            name=f"Buy & Hold ({result.ticker})",
            line=dict(color="#90A4AE", width=1.8, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Row 2: Asset Close Price + Entry/Exit Markers
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Close"],
            mode="lines",
            name=f"{result.ticker} Close Price",
            line=dict(color="#42A5F5", width=1.6),
        ),
        row=2,
        col=1,
    )

    buys = ts[ts["Position_Change"] > 0]
    sells = ts[ts["Position_Change"] < 0]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Close"],
                mode="markers",
                name="Buy / Long Entry",
                marker=dict(symbol="triangle-up", size=10, color="#00E676"),
            ),
            row=2,
            col=1,
        )
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Close"],
                mode="markers",
                name="Sell / Exit",
                marker=dict(symbol="triangle-down", size=10, color="#FF5252"),
            ),
            row=2,
            col=1,
        )

    # Row 3: Underwater Drawdown
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Strategy_Drawdown"],
            mode="lines",
            fill="tozeroy",
            name="Strategy Drawdown (%)",
            line=dict(color="#FF5252", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts["Benchmark_Drawdown"],
            mode="lines",
            name="Benchmark Drawdown (%)",
            line=dict(color="#78909C", width=1.2, dash="dot"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=720,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# Main Application UI
# ---------------------------------------------------------------------------

def main() -> None:
    fwd_engine = get_forward_engine()
    active_strategies = fwd_engine.list_strategies()

    # Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Backtest Parameters")
        ticker = st.text_input("Ticker Symbol", value="SPY", help="Any valid Yahoo Finance ticker (e.g., SPY, AAPL, NVDA, BTC-USD, QQQ)").strip().upper()

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365 * 3))
        with col_d2:
            end_date = st.date_input("End Date", value=date.today())

        interval = st.selectbox("Bar Interval", options=["1d", "1h", "1wk"], index=0)
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=100_000.0, step=10_000.0)
        commission_bps = st.slider("Commission & Slippage (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        risk_free_rate_pct = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.25)

        st.divider()
        st.subheader("📡 Background Forward-Tester")
        status_label = "🟢 RUNNING" if fwd_engine.is_running else "🔴 STOPPED"
        st.caption(f"**Engine Status:** {status_label}")
        st.caption(f"**Scheduler:** `{fwd_engine.scheduler_backend}`")
        st.caption(f"**Active Paper Strategies:** `{sum(1 for s in active_strategies if s['status'] == 'ACTIVE')}`")
        st.caption(f"**Last Background Sync:** `{fwd_engine.last_global_sync or 'Pending next tick'}`")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("🔄 Sync Now", use_container_width=True, help="Force an immediate yfinance fetch & PnL update for all active forward-tested strategies"):
                n_synced = fwd_engine.sync_all_active_strategies(force_refresh=True)
                st.toast(f"Synced {n_synced} active forward-testing strategies!", icon="✅")
                st.rerun()
        with col_s2:
            if st.button("🧹 Clear Cache", use_container_width=True, help="Clear local disk and RAM market data caches"):
                removed = clear_market_cache()
                st.cache_data.clear()
                st.toast(f"Cleared cache ({removed} files removed).", icon="🧹")

    # Header
    st.title("📈 QuantPulse: Vectorized Strategy Backtester & Live Paper Trader")
    st.markdown(
        "Write custom vectorized Python trading strategies, backtest them instantaneously against cached `yfinance` data, "
        "and deploy winning strategies to the background **Forward-Testing Engine** (`APScheduler`) for continuous paper trading."
    )

    tab_backtest, tab_forward = st.tabs(
        [
            "⚡ Instant Vectorized Backtester",
            f"🛰️ Background Forward-Testing Monitor ({len(active_strategies)} Strategies)",
        ]
    )

    # =======================================================================
    # TAB 1: STRATEGY INPUT WINDOW & INSTANT VECTORIZED BACKTESTER
    # =======================================================================
    with tab_backtest:
        col_preset, col_help = st.columns([2, 3])
        with col_preset:
            preset_name = st.selectbox(
                "Load Preset Strategy Template",
                options=list(PRESET_STRATEGIES.keys()),
                index=0,
            )
        with col_help:
            with st.expander("📖 Strategy API Quick Reference (Available Variables & Vectorized Indicators)"):
                st.markdown(
                    """
                    - **`df`**: Historical OHLCV `pandas.DataFrame` (`Open`, `High`, `Low`, `Close`, `Volume`).
                    - **Built-in Vectorized Indicators**:
                      - `sma(series, window)` — Simple Moving Average
                      - `ema(series, span)` — Exponential Moving Average
                      - `rsi(series, period=14)` — Wilder's Relative Strength Index
                      - `bollinger_bands(series, window=20, num_std=2.0)` — returns `(upper, mid, lower)`
                      - `macd(series, fast=12, slow=26, signal=9)` — returns `(macd_line, signal_line, hist)`
                      - `atr(df, period=14)` — Average True Range
                    - **How to output signals**:
                      1. Define boolean series `entries` and `exits` (for long entry/exit signals), **OR**
                      2. Assign target weights directly to `df['signal']` (`1.0` = Long, `0.0` = Flat, `-1.0` = Short).
                    - **Zero Lookahead Bias**: Target positions computed on bar $t$ close are automatically executed on bar $t+1$.
                    """
                )

        if "current_preset" not in st.session_state or st.session_state.current_preset != preset_name:
            st.session_state.current_preset = preset_name
            st.session_state.strategy_code_input = PRESET_STRATEGIES[preset_name]

        strategy_code = st.text_area(
            "🐍 Custom Python Strategy Logic (Vectorized)",
            value=st.session_state.strategy_code_input,
            height=220,
            key="strategy_editor",
            help="Write or paste custom vectorized pandas/numpy strategy logic here. Click 'Run Instant Backtest' or press Ctrl+Enter.",
        )
        st.session_state.strategy_code_input = strategy_code

        run_col, deploy_col = st.columns([1, 2])
        with run_col:
            run_triggered = st.button("🚀 Run Instant Backtest", type="primary", use_container_width=True)

        # Always run backtest automatically on load or when button clicked
        if run_triggered or "last_bt_result" not in st.session_state or True:
            try:
                with st.spinner(f"Fetching {ticker} market data & executing vectorized backtest..."):
                    df_market = cached_load_data(
                        ticker=ticker,
                        start_date=str(start_date),
                        end_date=str(end_date),
                        interval=interval,
                    )
                    if df_market.empty:
                        st.error(f"No historical data returned by `yfinance` for ticker `{ticker}` in the selected range.")
                        st.stop()

                    bars_per_year = 252 if interval == "1d" else (252 * 6.5 if interval == "1h" else 52)
                    bt_result = run_vectorized_backtest(
                        df=df_market,
                        strategy_code=strategy_code,
                        ticker=ticker,
                        initial_capital=initial_capital,
                        commission_bps=commission_bps,
                        risk_free_rate=risk_free_rate_pct / 100.0,
                        bars_per_year=int(bars_per_year),
                    )
                    st.session_state.last_bt_result = bt_result
            except Exception as exc:
                st.error(f"**Strategy Execution Error:** `{exc}`")
                st.stop()

        bt: BacktestResult = st.session_state.last_bt_result
        m = bt.metrics

        # Top KPI Metrics Row
        st.subheader(
            f"📊 Backtest Performance Summary — `{bt.ticker}` "
            f"(Executed in `{bt.execution_time_ms:.2f} ms` | Current Signal: `{bt.latest_signal_label}`)"
        )

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric(
            "Total Strategy Return",
            f"{m['Total Return (%)']:+.2f}%",
            delta=f"{m['Excess Return / Alpha (%)']:+.2f}% vs Benchmark",
        )
        k2.metric(
            "Buy & Hold Benchmark",
            f"{m['Benchmark Return (%)']:+.2f}%",
            delta=f"${m['Benchmark Final Equity ($)']:,.0f} Final",
            delta_color="off",
        )
        k3.metric(
            "Sharpe Ratio",
            f"{m['Sharpe Ratio']:.2f}",
            delta=f"Sortino: {m['Sortino Ratio']:.2f}",
            delta_color="off",
        )
        k4.metric(
            "Max Drawdown",
            f"{m['Max Drawdown (%)']:.2f}%",
            delta=f"Bench DD: {m['Benchmark Max Drawdown (%)']:.2f}%",
            delta_color="inverse",
        )
        k5.metric(
            "Win Rate",
            f"{m['Win Rate (%)']:.1f}%",
            delta=f"Profit Factor: {m['Profit Factor']:.2f}",
            delta_color="off",
        )
        k6.metric(
            "Final Strategy Equity",
            f"${m['Final Equity ($)']:,.2f}",
            delta=f"{m['Total Trades']} Trades ({m['Exposure Time (%)']:.0f}% Time in Market)",
            delta_color="off",
        )

        # Interactive Equity & Drawdown Chart
        st.plotly_chart(build_equity_and_drawdown_chart(bt), use_container_width=True)

        # Side-by-side Performance Comparison Table & Trade Log
        col_metrics, col_trades = st.columns([1, 1.4])
        with col_metrics:
            st.markdown("#### 📋 Strategy vs. Benchmark Scorecard")
            comparison_df = pd.DataFrame(
                [
                    {"Metric": "Initial Capital", "Strategy": f"${m['Initial Capital ($)']:,.2f}", "Buy & Hold": f"${m['Initial Capital ($)']:,.2f}"},
                    {"Metric": "Final Equity", "Strategy": f"${m['Final Equity ($)']:,.2f}", "Buy & Hold": f"${m['Benchmark Final Equity ($)']:,.2f}"},
                    {"Metric": "Cumulative Return", "Strategy": f"{m['Total Return (%)']:+.2f}%", "Buy & Hold": f"{m['Benchmark Return (%)']:+.2f}%"},
                    {"Metric": "CAGR (Annualized)", "Strategy": f"{m['CAGR (%)']:+.2f}%", "Buy & Hold": f"{m['Benchmark CAGR (%)']:+.2f}%"},
                    {"Metric": "Annualized Volatility", "Strategy": f"{m['Annualized Volatility (%)']:.2f}%", "Buy & Hold": f"{m['Benchmark Volatility (%)']:.2f}%"},
                    {"Metric": "Max Drawdown", "Strategy": f"{m['Max Drawdown (%)']:.2f}%", "Buy & Hold": f"{m['Benchmark Max Drawdown (%)']:.2f}%"},
                    {"Metric": "Sharpe Ratio", "Strategy": f"{m['Sharpe Ratio']:.2f}", "Buy & Hold": "—"},
                    {"Metric": "Sortino Ratio", "Strategy": f"{m['Sortino Ratio']:.2f}", "Buy & Hold": "—"},
                    {"Metric": "Calmar Ratio", "Strategy": f"{m['Calmar Ratio']:.2f}", "Buy & Hold": "—"},
                    {"Metric": "Win Rate / Profit Factor", "Strategy": f"{m['Win Rate (%)']:.1f}% / {m['Profit Factor']:.2f}", "Buy & Hold": "—"},
                ]
            )
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        with col_trades:
            st.markdown(f"#### 🧾 Executed Trade Log ({len(bt.trades)} Round-Trip Trades)")
            if bt.trades.empty:
                st.info("No position changes were triggered during the selected date window.")
            else:
                st.dataframe(bt.trades, use_container_width=True, hide_index=True, height=350)

        st.divider()

        # Deploy to Background Forward-Testing Section
        st.subheader("🛰️ Deploy Strategy to Background Forward-Testing (Paper Trading)")
        st.caption(
            "Save this strategy to the background `APScheduler` engine. It will continue running in the background, "
            "fetching new daily bars from `yfinance` and updating paper PnL even while you test other strategies in the UI."
        )
        with st.form("deploy_forward_test_form", clear_on_submit=False):
            f_col1, f_col2, f_col3, f_col4 = st.columns([2.2, 1, 1.2, 1.2])
            with f_col1:
                deploy_name = st.text_input(
                    "Strategy Name",
                    value=f"{ticker} - {preset_name.split('(')[0].strip()}",
                )
            with f_col2:
                deploy_ticker = st.text_input("Paper Ticker", value=ticker)
            with f_col3:
                deploy_capital = st.number_input("Allocated Paper Capital ($)", value=float(initial_capital), step=10_000.0)
            with f_col4:
                st.write("")
                st.write("")
                deploy_submitted = st.form_submit_button("➕ Activate Forward-Test", type="primary", use_container_width=True)

            if deploy_submitted:
                new_id = fwd_engine.register_strategy(
                    name=deploy_name,
                    ticker=deploy_ticker,
                    code=strategy_code,
                    initial_capital=deploy_capital,
                    commission_bps=commission_bps,
                    interval="1d",
                )
                st.success(
                    f"Deployed **{deploy_name}** (`{deploy_ticker}`, ID: `{new_id}`) to the background paper-trading scheduler! "
                    "Switch to the **Background Forward-Testing Monitor** tab to track live signals and PnL."
                )
                st.rerun()

    # =======================================================================
    # TAB 2: BACKGROUND FORWARD-TESTING (PAPER TRADING) MONITOR
    # =======================================================================
    with tab_forward:
        st.subheader("🛰️ Active Forward-Testing & Paper Trading Strategies")
        st.markdown(
            f"The background worker (`{fwd_engine.scheduler_backend}`) polls `yfinance` every "
            f"**{fwd_engine.poll_interval_seconds} seconds**, evaluates each active strategy's Python code on the latest daily bars, "
            "and persists paper equity and position updates to SQLite (`forward_testing_state.db`)."
        )

        if not active_strategies:
            st.info(
                "No strategies are currently registered for background forward-testing. "
                "Use the **Deploy Strategy to Background Forward-Testing** panel in the Backtester tab to add one!"
            )
        else:
            # Portfolio-wide rollup KPIs
            total_alloc = sum(float(s["initial_capital"]) for s in active_strategies)
            total_eq = sum(float(s["current_equity"]) for s in active_strategies)
            total_cum_pnl = total_eq - total_alloc
            total_fwd_pnl = sum(float(s["forward_pnl"]) for s in active_strategies)

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Total Paper Capital Allocated", f"${total_alloc:,.2f}")
            p2.metric("Current Marked-to-Market Equity", f"${total_eq:,.2f}", delta=f"${total_cum_pnl:+,.2f} Cumulative")
            p3.metric("Forward Paper PnL (Since Deployment)", f"${total_fwd_pnl:+,.2f}")
            p4.metric("Background Sync Cycles Completed", f"{fwd_engine.total_sync_cycles}")

            # Overview Table
            summary_rows = []
            for s in active_strategies:
                summary_rows.append(
                    {
                        "ID": s["strategy_id"],
                        "Strategy Name": s["name"],
                        "Ticker": s["ticker"],
                        "Status": s["status"],
                        "Current Signal": s["current_signal"],
                        "Latest Price": f"${float(s['latest_price'] or 0.0):,.2f}",
                        "Paper Equity ($)": f"${float(s['current_equity']):,.2f}",
                        "Cumulative Return (%)": f"{float(s['cumulative_return_pct']):+.2f}%",
                        "Forward PnL ($)": f"${float(s['forward_pnl']):+,.2f}",
                        "Sharpe": f"{float(s['sharpe_ratio']):.2f}",
                        "Last Synced": s["last_updated_at"],
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # Detailed Cards per Forward-Tested Strategy
            for s in active_strategies:
                sid = s["strategy_id"]
                sig_badge = (
                    "🟢 LONG" if s["current_signal"] == "LONG"
                    else ("🔴 SHORT" if s["current_signal"] == "SHORT" else "⚪ FLAT")
                )
                with st.expander(
                    f"📌 {s['name']} ({s['ticker']}) — Signal: {sig_badge} | "
                    f"Equity: ${float(s['current_equity']):,.2f} ({float(s['cumulative_return_pct']):+.2f}%) | "
                    f"Status: {s['status']}",
                    expanded=True,
                ):
                    if s.get("last_error"):
                        st.error(f"Last evaluation error: `{s['last_error']}`")

                    c_chart, c_meta = st.columns([2.2, 1.2])
                    with c_chart:
                        eq_hist = s.get("equity_history", [])
                        if eq_hist:
                            hist_df = pd.DataFrame(eq_hist)
                            fig_fwd = go.Figure()
                            fig_fwd.add_trace(
                                go.Scatter(
                                    x=hist_df["date"],
                                    y=hist_df["strategy_equity"],
                                    mode="lines",
                                    name="Paper Strategy Equity ($)",
                                    line=dict(color="#00E676", width=2.2),
                                )
                            )
                            fig_fwd.add_trace(
                                go.Scatter(
                                    x=hist_df["date"],
                                    y=hist_df["benchmark_equity"],
                                    mode="lines",
                                    name=f"{s['ticker']} Buy & Hold ($)",
                                    line=dict(color="#90A4AE", width=1.5, dash="dot"),
                                )
                            )
                            fig_fwd.update_layout(
                                title=f"Trailing 90-Day Paper Equity Progression — {s['name']}",
                                height=280,
                                template="plotly_dark",
                                margin=dict(l=15, r=15, t=35, b=15),
                                hovermode="x unified",
                            )
                            st.plotly_chart(fig_fwd, use_container_width=True, key=f"fwd_chart_{sid}")

                    with c_meta:
                        st.markdown("**Paper Trading Ledger (Recent Signals & Bar Updates)**")
                        ledger = s.get("paper_ledger", [])
                        if ledger:
                            st.dataframe(pd.DataFrame(ledger), use_container_width=True, hide_index=True, height=180)

                        b1, b2, b3 = st.columns(3)
                        with b1:
                            if s["status"] == "ACTIVE":
                                if st.button("⏸️ Pause", key=f"pause_{sid}", use_container_width=True):
                                    fwd_engine.set_strategy_status(sid, "PAUSED")
                                    st.rerun()
                            else:
                                if st.button("▶️ Resume", key=f"resume_{sid}", use_container_width=True):
                                    fwd_engine.set_strategy_status(sid, "ACTIVE")
                                    fwd_engine.evaluate_single_strategy(sid, force_refresh=True)
                                    st.rerun()
                        with b2:
                            if st.button("🔄 Update Bar", key=f"sync_{sid}", use_container_width=True):
                                fwd_engine.evaluate_single_strategy(sid, force_refresh=True)
                                st.toast(f"Updated {s['name']}!", icon="📈")
                                st.rerun()
                        with b3:
                            if st.button("🗑️ Delete", key=f"del_{sid}", use_container_width=True):
                                fwd_engine.delete_strategy(sid)
                                st.rerun()

                    with st.popover(f"🔍 View Saved Strategy Code (`{sid}`)"):
                        st.code(s["code"], language="python")


if __name__ == "__main__":
    main()
