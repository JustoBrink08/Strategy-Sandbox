"""
High-performance market data loader with multi-layer caching (in-memory TTL + local disk cache).
"""

from __future__ import annotations

import hashlib
import logging
import site
import sys
import time
from datetime import date, datetime
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Tuple

for _venv_sp in (Path(__file__).resolve().parent / ".venv" / "lib").glob("python*/site-packages"):
    if _venv_sp.exists() and str(_venv_sp) not in sys.path:
        site.addsitedir(str(_venv_sp))

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / ".market_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MEM_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_CACHE_LOCK = RLock()
DEFAULT_TTL_SECONDS = 900  # 15 minutes default cache lifetime


def _make_cache_key(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    interval: str,
    period: Optional[str] = None,
) -> str:
    raw = f"{ticker.upper().strip()}|{start or ''}|{end or ''}|{interval}|{period or ''}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe_ticker = "".join(c for c in ticker.upper() if c.isalnum() or c in ("-", "_"))
    return f"{safe_ticker}_{interval}_{digest}"


def _normalize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardize yfinance output into a clean single-level OHLCV DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Flatten MultiIndex columns returned by newer yfinance versions
    if isinstance(out.columns, pd.MultiIndex):
        if ticker.upper() in out.columns.get_level_values(-1):
            out = out.xs(ticker.upper(), axis=1, level=-1, drop_level=True)
        else:
            out.columns = out.columns.get_level_values(0)

    # Normalize column names to Title Case
    rename_map = {}
    for col in out.columns:
        col_str = str(col).strip()
        lower = col_str.lower()
        if lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower == "close":
            rename_map[col] = "Close"
        elif lower in ("adj close", "adj_close", "adjclose"):
            rename_map[col] = "Adj Close"
        elif lower == "volume":
            rename_map[col] = "Volume"

    out = out.rename(columns=rename_map)

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    required = ["Open", "High", "Low", "Close"]
    for req in required:
        if req not in out.columns and "Close" in out.columns:
            out[req] = out["Close"]

    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    # Ensure timezone-naive DatetimeIndex sorted chronologically
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["Close"])

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in out.columns]
    return out[keep_cols].astype(float)


def fetch_market_data(
    ticker: str,
    start: Optional[str | date | datetime] = None,
    end: Optional[str | date | datetime] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data via yfinance with two-tier caching (RAM + Disk CSV/pkl).
    Ensures ultra-fast execution without redundant network calls.
    """
    clean_ticker = ticker.strip().upper()
    start_str = str(start)[:10] if start is not None else None
    end_str = str(end)[:10] if end is not None else None

    cache_key = _make_cache_key(clean_ticker, start_str, end_str, interval, period)
    disk_path = CACHE_DIR / f"{cache_key}.pkl"
    now = time.time()

    if not force_refresh:
        # 1. Check in-memory cache first (sub-millisecond)
        with _CACHE_LOCK:
            cached = _MEM_CACHE.get(cache_key)
            if cached is not None:
                ts, cached_df = cached
                if now - ts <= ttl_seconds and not cached_df.empty:
                    return cached_df.copy()

        # 2. Check disk cache
        if disk_path.exists():
            mtime = disk_path.stat().st_mtime
            if now - mtime <= ttl_seconds:
                try:
                    disk_df = pd.read_pickle(disk_path)
                    if isinstance(disk_df, pd.DataFrame) and not disk_df.empty:
                        with _CACHE_LOCK:
                            _MEM_CACHE[cache_key] = (mtime, disk_df)
                        return disk_df.copy()
                except Exception as exc:
                    logger.warning("Failed reading disk cache %s: %s", disk_path, exc)

    if yf is None:
        raise RuntimeError("yfinance is not installed. Please run `pip install -r requirements.txt`.")

    # 3. Fetch from yfinance
    dl_kwargs = {
        "tickers": clean_ticker,
        "interval": interval,
        "progress": False,
        "auto_adjust": True,
    }
    if period:
        dl_kwargs["period"] = period
    else:
        if start_str:
            dl_kwargs["start"] = start_str
        if end_str:
            dl_kwargs["end"] = end_str

    raw_df = yf.download(**dl_kwargs)
    norm_df = _normalize_ohlcv(raw_df, clean_ticker)

    if norm_df.empty:
        # Fallback to Ticker.history if yf.download returned empty
        ticker_obj = yf.Ticker(clean_ticker)
        hist_kwargs = {"interval": interval, "auto_adjust": True}
        if period:
            hist_kwargs["period"] = period
        else:
            if start_str:
                hist_kwargs["start"] = start_str
            if end_str:
                hist_kwargs["end"] = end_str
        raw_df = ticker_obj.history(**hist_kwargs)
        norm_df = _normalize_ohlcv(raw_df, clean_ticker)

    if not norm_df.empty:
        with _CACHE_LOCK:
            _MEM_CACHE[cache_key] = (now, norm_df)
        try:
            norm_df.to_pickle(disk_path)
        except Exception as exc:
            logger.warning("Failed writing disk cache %s: %s", disk_path, exc)

    return norm_df.copy()


def clear_market_cache() -> int:
    """Clear both in-memory and on-disk market data caches. Returns number of disk files removed."""
    with _CACHE_LOCK:
        _MEM_CACHE.clear()
    removed = 0
    for file in CACHE_DIR.glob("*.pkl"):
        try:
            file.unlink()
            removed += 1
        except OSError:
            pass
    return removed
