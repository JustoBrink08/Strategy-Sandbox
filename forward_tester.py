"""
Background Forward-Testing ("Paper Trading") Engine.
Uses APScheduler (with daemon thread fallback) and SQLite persistence to continuously
evaluate active strategies on fresh daily yfinance data and update paper PnL in the background.
"""

from __future__ import annotations

import json
import logging
import site
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

for _venv_sp in (Path(__file__).resolve().parent / ".venv" / "lib").glob("python*/site-packages"):
    if _venv_sp.exists() and str(_venv_sp) not in sys.path:
        site.addsitedir(str(_venv_sp))

import pandas as pd

from backtester import run_vectorized_backtest
from data_loader import fetch_market_data

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ImportError:
    BackgroundScheduler = None  # type: ignore

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "forward_testing_state.db"


class ForwardTestingEngine:
    """
    Singleton background engine that persists active paper-trading strategies and
    periodically fetches new market data to update their forward-test PnL and equity history.
    """

    _instance: Optional["ForwardTestingEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Path = DB_PATH, poll_interval_seconds: int = 60) -> None:
        self.db_path = db_path
        self.poll_interval_seconds = poll_interval_seconds
        self._db_lock = threading.RLock()
        self._scheduler: Any = None
        self._fallback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False
        self.scheduler_backend = "uninitialized"
        self.last_global_sync: Optional[str] = None
        self.total_sync_cycles: int = 0

        self._init_db()

    @classmethod
    def get_instance(cls, poll_interval_seconds: int = 60) -> "ForwardTestingEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(poll_interval_seconds=poll_interval_seconds)
                cls._instance.start()
            return cls._instance

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS active_strategies (
                        strategy_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        interval TEXT NOT NULL DEFAULT '1d',
                        code TEXT NOT NULL,
                        initial_capital REAL NOT NULL,
                        commission_bps REAL NOT NULL DEFAULT 5.0,
                        status TEXT NOT NULL DEFAULT 'ACTIVE',
                        created_at TEXT NOT NULL,
                        deployed_bar_date TEXT,
                        deployment_price REAL,
                        latest_bar_date TEXT,
                        latest_price REAL,
                        current_position REAL NOT NULL DEFAULT 0.0,
                        current_signal TEXT NOT NULL DEFAULT 'FLAT',
                        current_equity REAL NOT NULL,
                        cumulative_pnl REAL NOT NULL DEFAULT 0.0,
                        cumulative_return_pct REAL NOT NULL DEFAULT 0.0,
                        forward_pnl REAL NOT NULL DEFAULT 0.0,
                        forward_return_pct REAL NOT NULL DEFAULT 0.0,
                        sharpe_ratio REAL NOT NULL DEFAULT 0.0,
                        max_drawdown_pct REAL NOT NULL DEFAULT 0.0,
                        total_trades INTEGER NOT NULL DEFAULT 0,
                        last_updated_at TEXT,
                        last_error TEXT,
                        equity_history_json TEXT NOT NULL DEFAULT '[]',
                        paper_ledger_json TEXT NOT NULL DEFAULT '[]'
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Background Scheduler Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background APScheduler (or fallback daemon thread) if not already running."""
        if self.is_running:
            return

        if BackgroundScheduler is not None:
            self._scheduler = BackgroundScheduler(daemon=True)
            self._scheduler.add_job(
                self.sync_all_active_strategies,
                trigger="interval",
                seconds=self.poll_interval_seconds,
                id="paper_trading_sync_job",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            self._scheduler.start()
            self.scheduler_backend = "APScheduler (BackgroundScheduler)"
            self.is_running = True
        else:
            self._stop_event.clear()
            self._fallback_thread = threading.Thread(
                target=self._fallback_loop,
                name="ForwardTestingWorker",
                daemon=True,
            )
            self._fallback_thread.start()
            self.scheduler_backend = "Python Daemon Thread"
            self.is_running = True

    def _fallback_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sync_all_active_strategies()
            except Exception as exc:
                logger.error("Background forward-test error: %s", exc)
            self._stop_event.wait(self.poll_interval_seconds)

    def stop(self) -> None:
        if not self.is_running:
            return
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        if self._fallback_thread is not None:
            self._stop_event.set()
        self.is_running = False

    # ------------------------------------------------------------------
    # Strategy Registration & Management
    # ------------------------------------------------------------------

    def register_strategy(
        self,
        name: str,
        ticker: str,
        code: str,
        initial_capital: float = 100_000.0,
        commission_bps: float = 5.0,
        interval: str = "1d",
    ) -> str:
        """
        Register a new strategy for continuous background forward-testing and perform
        its initial paper-trading evaluation immediately.
        """
        strategy_id = str(uuid.uuid4())[:8]
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """
                    INSERT INTO active_strategies (
                        strategy_id, name, ticker, interval, code,
                        initial_capital, commission_bps, status, created_at,
                        current_equity, last_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
                    """,
                    (
                        strategy_id,
                        name.strip(),
                        ticker.strip().upper(),
                        interval,
                        code,
                        float(initial_capital),
                        float(commission_bps),
                        now_iso,
                        float(initial_capital),
                        now_iso,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        # Run initial evaluation synchronously so the UI reflects live state immediately
        self.evaluate_single_strategy(strategy_id, force_refresh=False)
        return strategy_id

    def set_strategy_status(self, strategy_id: str, status: str) -> None:
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE active_strategies SET status = ? WHERE strategy_id = ?",
                    (status.upper(), strategy_id),
                )
                conn.commit()
            finally:
                conn.close()

    def delete_strategy(self, strategy_id: str) -> None:
        with self._db_lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM active_strategies WHERE strategy_id = ?", (strategy_id,))
                conn.commit()
            finally:
                conn.close()

    def list_strategies(self) -> List[Dict[str, Any]]:
        with self._db_lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM active_strategies ORDER BY created_at DESC"
                ).fetchall()
                result = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["equity_history"] = json.loads(item.get("equity_history_json") or "[]")
                    except Exception:
                        item["equity_history"] = []
                    try:
                        item["paper_ledger"] = json.loads(item.get("paper_ledger_json") or "[]")
                    except Exception:
                        item["paper_ledger"] = []
                    result.append(item)
                return result
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Background Evaluation & Paper-Trading Simulation Logic
    # ------------------------------------------------------------------

    def sync_all_active_strategies(self, force_refresh: bool = False) -> int:
        """
        Executed periodically by APScheduler in the background (or on-demand from the UI).
        Fetches latest market bars for all ACTIVE strategies and updates their PnL.
        """
        strategies = self.list_strategies()
        updated_count = 0
        for strat in strategies:
            if strat["status"] == "ACTIVE":
                self.evaluate_single_strategy(strat["strategy_id"], force_refresh=force_refresh)
                updated_count += 1

        self.last_global_sync = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.total_sync_cycles += 1
        return updated_count

    def evaluate_single_strategy(self, strategy_id: str, force_refresh: bool = False) -> None:
        """Fetch latest rolling market data and update a single strategy's paper PnL and ledger."""
        with self._db_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM active_strategies WHERE strategy_id = ?", (strategy_id,)
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return

        strat = dict(row)
        ticker = strat["ticker"]
        interval = strat["interval"]
        code = strat["code"]
        initial_capital = float(strat["initial_capital"])
        commission_bps = float(strat["commission_bps"])
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # Fetch trailing 1 year of daily bars (or 60d for hourly) to warm up indicators
            lookback_days = 365 if interval == "1d" else 60
            start_dt = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            df = fetch_market_data(
                ticker=ticker,
                start=start_dt,
                interval=interval,
                ttl_seconds=300,
                force_refresh=force_refresh,
            )
            if df.empty:
                raise ValueError(f"No market data returned for {ticker}.")

            bt = run_vectorized_backtest(
                df=df,
                strategy_code=code,
                ticker=ticker,
                initial_capital=initial_capital,
                commission_bps=commission_bps,
            )

            ts = bt.timeseries
            latest_bar_dt = str(ts.index[-1])[:19]
            latest_price = float(ts["Close"].iloc[-1])
            current_pos = float(bt.latest_position)
            current_sig = bt.latest_signal_label
            current_equity = float(ts["Strategy_Equity"].iloc[-1])
            cum_pnl = current_equity - initial_capital
            cum_ret_pct = float(bt.metrics["Total Return (%)"])

            # Track deployment baseline so we can also report forward-only PnL since deployment
            deployed_bar_date = strat["deployed_bar_date"] or latest_bar_dt
            deployment_price = (
                float(strat["deployment_price"])
                if strat["deployment_price"] is not None
                else latest_price
            )

            # Compute forward PnL since deployment bar (plus latest daily bar PnL for paper simulation)
            if deployed_bar_date in ts.index.astype(str):
                mask = ts.index.astype(str) >= deployed_bar_date
                forward_slice = ts.loc[mask, "Strategy_Return"]
                if len(forward_slice) > 1:
                    fwd_growth = float((1.0 + forward_slice.iloc[1:]).prod() - 1.0)
                else:
                    fwd_growth = float(ts["Strategy_Return"].iloc[-1])
            else:
                fwd_growth = float(ts["Strategy_Return"].iloc[-1])

            forward_pnl = initial_capital * fwd_growth
            forward_return_pct = fwd_growth * 100.0

            # Build compact trailing 90-bar equity curve for fast UI rendering
            tail_ts = ts.tail(90)
            equity_history = [
                {
                    "date": str(idx)[:10],
                    "strategy_equity": round(float(r["Strategy_Equity"]), 2),
                    "benchmark_equity": round(float(r["Benchmark_Equity"]), 2),
                    "close": round(float(r["Close"]), 2),
                    "signal": float(r["Signal"]),
                }
                for idx, r in tail_ts.iterrows()
            ]

            # Maintain paper trading fill / signal ledger
            try:
                ledger = json.loads(strat.get("paper_ledger_json") or "[]")
            except Exception:
                ledger = []

            prev_sig = strat.get("current_signal")
            if not ledger or prev_sig != current_sig or strat.get("latest_bar_date") != latest_bar_dt:
                ledger.insert(
                    0,
                    {
                        "timestamp": now_iso,
                        "bar_date": latest_bar_dt[:10],
                        "signal": current_sig,
                        "position": current_pos,
                        "price": round(latest_price, 2),
                        "equity": round(current_equity, 2),
                        "daily_return_pct": round(float(ts["Strategy_Return"].iloc[-1]) * 100.0, 2),
                    },
                )
                ledger = ledger[:50]  # Keep most recent 50 paper events

            with self._db_lock:
                conn = self._get_conn()
                try:
                    conn.execute(
                        """
                        UPDATE active_strategies
                        SET deployed_bar_date = ?,
                            deployment_price = ?,
                            latest_bar_date = ?,
                            latest_price = ?,
                            current_position = ?,
                            current_signal = ?,
                            current_equity = ?,
                            cumulative_pnl = ?,
                            cumulative_return_pct = ?,
                            forward_pnl = ?,
                            forward_return_pct = ?,
                            sharpe_ratio = ?,
                            max_drawdown_pct = ?,
                            total_trades = ?,
                            last_updated_at = ?,
                            last_error = NULL,
                            equity_history_json = ?,
                            paper_ledger_json = ?
                        WHERE strategy_id = ?
                        """,
                        (
                            deployed_bar_date,
                            deployment_price,
                            latest_bar_dt,
                            round(latest_price, 4),
                            current_pos,
                            current_sig,
                            round(current_equity, 2),
                            round(cum_pnl, 2),
                            round(cum_ret_pct, 2),
                            round(forward_pnl, 2),
                            round(forward_return_pct, 2),
                            float(bt.metrics["Sharpe Ratio"]),
                            float(bt.metrics["Max Drawdown (%)"]),
                            int(bt.metrics["Total Trades"]),
                            now_iso,
                            json.dumps(equity_history),
                            json.dumps(ledger),
                            strategy_id,
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

        except Exception as exc:
            logger.warning("Failed evaluating strategy %s: %s", strategy_id, exc)
            with self._db_lock:
                conn = self._get_conn()
                try:
                    conn.execute(
                        """
                        UPDATE active_strategies
                        SET last_updated_at = ?, last_error = ?
                        WHERE strategy_id = ?
                        """,
                        (now_iso, str(exc), strategy_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
