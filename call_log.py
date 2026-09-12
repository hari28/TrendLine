"""Persistent log of "calls" -- Above 200 EMA / Golden Cross signals -- logged
the moment they first qualify in universe_digest.py (the same instant a
Telegram alert would fire), so the Streamlit Call Performance tab can look
back at what was signalled, when, and how it has performed since.

Only these two bullish modes are logged (below_ma/death-cross/volume/pattern
hits are not, and a Death Cross within golden_cross's combined scan is
skipped too) -- this is deliberately narrower than what Telegram actually
alerts on, per the user's explicit ask to track "above 200 ema and golden
cross" calls specifically.
"""
import csv
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from screener import load_frame

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_PATH = os.path.join(DATA_DIR, "call_log.csv")
COLUMNS = ["timestamp_ist", "date", "time", "symbol", "segment", "signal_type", "timeframe", "close_at_call"]

LOGGED_MODES = {"above_ma", "golden_cross"}
SIGNAL_LABELS = {"above_ma": "Above 200 EMA", "golden_cross": "Golden Cross"}


def log_call(now_ist: datetime, symbol: str, segment: str, signal_type: str, timeframe: str, close: float) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS)
        writer.writerow([
            now_ist.strftime("%Y-%m-%dT%H:%M:%S"), now_ist.strftime("%Y-%m-%d"), now_ist.strftime("%H:%M:%S"),
            symbol, segment, signal_type, timeframe, f"{close:.2f}",
        ])


def load_calls() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(LOG_PATH, parse_dates=["date"])
    return df


def _latest_close(symbol: str) -> float | None:
    """Latest daily close, cache-only -- used to mark logged calls to market.
    Never force-refreshes, so this stays fast even for a long call history."""
    frame = load_frame(symbol, "1D", force_refresh=False)
    if frame is None or frame.empty:
        return None
    return float(frame["Close"].iloc[-1])


def with_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Adds current_close / change_pct / days_since columns. Looks up each
    distinct symbol's latest close once, not per row, since a symbol can have
    several logged calls."""
    if df.empty:
        out = df.copy()
        out["current_close"] = pd.Series(dtype=float)
        out["change_pct"] = pd.Series(dtype=float)
        out["days_since"] = pd.Series(dtype="int64")
        return out

    df = df.copy()
    latest = {sym: _latest_close(sym) for sym in df["symbol"].unique()}

    df["current_close"] = df["symbol"].map(latest)
    df["change_pct"] = (df["current_close"] - df["close_at_call"]) / df["close_at_call"] * 100.0
    now_naive = datetime.now(IST).replace(tzinfo=None)
    df["days_since"] = (now_naive - pd.to_datetime(df["timestamp_ist"])).dt.days
    return df
