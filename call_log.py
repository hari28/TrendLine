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

# Target/SL applied to every logged call (both signal types, per user's explicit
# choice) -- reuses Golden Cross's own live-tracked rule (see backtest.py's
# TARGET_PCT/STOP_PCT, the same numbers paper_trading.py's golden_cross strategy
# runs on) even for Above 200 EMA calls, which have no target/SL rule of their
# own -- it's just a state filter, not a trade setup. Entry = the call's own
# logged close (no separate 5% trigger step like backtest.py's Golden Cross
# entry -- the call is already the qualifying event here).
TARGET_PCT = 0.25
STOP_PCT = 0.05


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


def _walk_to_exit(symbol: str, call_date, target_price: float, sl_price: float):
    """Daily bars strictly after call_date -- first bar whose High/Low reaches
    target or stop wins (stop wins if a single bar touches both, the same
    conservative convention backtest.py uses). Returns (sold_price, exit_date,
    exit_reason), all None if still open (neither hit yet in cached history)."""
    frame = load_frame(symbol, "1D", force_refresh=False)
    if frame is None or frame.empty:
        return None, None, None
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.set_index(pd.to_datetime(frame.index, errors="raise"))
        except (ValueError, TypeError):
            return None, None, None  # cached index is corrupt for this symbol -- treat as still open
    future = frame[frame.index.date > call_date]
    for ts, row in future.iterrows():
        if row["Low"] <= sl_price:
            return sl_price, ts.date(), "Stop-loss"
        if row["High"] >= target_price:
            return target_price, ts.date(), "Target"
    return None, None, None


def with_target_sl_exit(df: pd.DataFrame, target_pct: float = TARGET_PCT,
                         stop_pct: float = STOP_PCT) -> pd.DataFrame:
    """Adds entry_price / target_price / sl_price / sold_at_price / exit_date /
    exit_reason columns. entry_price is just close_at_call (the call itself is
    the qualifying event, no separate trigger step)."""
    if df.empty:
        out = df.copy()
        for col in ["entry_price", "target_price", "sl_price", "sold_at_price", "exit_date", "exit_reason"]:
            out[col] = pd.Series(dtype=object)
        return out

    out = df.copy()
    out["entry_price"] = out["close_at_call"]
    out["target_price"] = out["entry_price"] * (1 + target_pct)
    out["sl_price"] = out["entry_price"] * (1 - stop_pct)

    sold_prices, exit_dates, exit_reasons = [], [], []
    for r in out.itertuples(index=False):
        sold, exit_date, reason = _walk_to_exit(r.symbol, r.date.date(), r.target_price, r.sl_price)
        sold_prices.append(sold)
        exit_dates.append(exit_date)
        exit_reasons.append(reason)
    out["sold_at_price"] = sold_prices
    out["exit_date"] = exit_dates
    out["exit_reason"] = exit_reasons
    return out
