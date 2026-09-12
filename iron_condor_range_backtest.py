"""Backtest for the "Intraday Range Iron Condor" strategy -- the non-expiry-
day analog of iron_condor_backtest.py's 0DTE strategy. Read that module's
docstring first: the same fundamental caveat applies here (no real options
data anywhere in this app -- synthetic Black-Scholes pricing off the index
history, ~60-day 15-min history ceiling from Yahoo).

Why this needs a DIFFERENT model, not just the 0DTE one on a different day:
with several days left to expiry, there's barely any theta decay within one
session, so 0DTE's edge (harvest the accelerating decay into the close)
mostly isn't there. This strategy instead harvests two different things:

  1. Intraday IMPLIED VOLATILITY CRUSH -- vol is typically elevated right at
     the open (pricing in overnight/weekend uncertainty) and compresses as
     the session's range establishes itself. Modeled here by estimating an
     empirical "vol shape by time of day" from historical 15-min bars: each
     time-of-day bucket's own realized variance (intraday bar-to-bar, with
     the overnight gap excluded) relative to the session average, then
     applied as a MULTIPLIER on the same daily realized-vol estimate used
     elsewhere in this app. Entry gets priced with an elevated sigma, an
     early exit gets priced with a reduced one -- isolating the VEGA effect,
     since days-to-expiry (and so the time-value component) barely changes
     across a few intraday hours.
  2. Intraday RANGE containment -- strikes are set off the recent average
     daily range from the day's open, not off delta (delta on a multi-day
     option barely moves within a single session, so a delta target isn't a
     meaningful strike-selection tool here the way it is for 0DTE).

Entry: ~09:45 AM, on any day that is NOT the chosen expiry weekday (that
  day is what iron_condor_backtest.py's 0DTE strategy already covers).
Exit: ~13:15 (the vol-crush edge is front-loaded to the first half of the
  session -- holding longer just adds days-to-expiry gamma risk without
  more of the edge being harvested) or a stop-loss at STOP_LOSS_MULTIPLE x
  credit, whichever comes first. Never held overnight, same as the 0DTE
  version.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

from iron_condor_backtest import (
    EXPIRY_WEEKDAY_MAP, INSTRUMENT_CONFIG, TRADING_DAYS_PER_YEAR,
    _condor_cost_to_close, _realized_vol_series, simulate_portfolio,  # noqa: F401 (re-exported)
)
from screener import load_frame

ENTRY_TIME = "09:45"
EXIT_TIME = "13:15"
STOP_LOSS_MULTIPLE = 1.4
RANGE_LOOKBACK = 10  # trading days, for the average-daily-range strike offset
MIN_DAYS_TO_EXPIRY = 2  # skip 1-day-to-expiry entries -- see note below

# Diagnostic (found while validating this module against real Nifty 50 data):
# 1-day-to-expiry entries (e.g. Wednesday into a Thursday expiry) behaved far
# worse than 2-4 day entries in backtesting -- 41.7% win rate / -0.76% avg
# return vs. 91-100% win rate / +17-28% avg return for 2-4 days out. The
# range-based strikes stay roughly the same width regardless of days left,
# but a 1-day option has much less extrinsic value at that same strike
# distance -- so the credit collected shrinks a lot while the full-session
# range risk doesn't, a genuinely worse risk/reward day for this specific
# setup. MIN_DAYS_TO_EXPIRY skips those entries by default.


def _avg_range_series(daily: pd.DataFrame, lookback: int = RANGE_LOOKBACK) -> pd.Series:
    daily_range = daily["High"] - daily["Low"]
    return daily_range.shift(1).rolling(window=lookback, min_periods=lookback).mean()


def _time_of_day_vol_multiplier(intraday: pd.DataFrame) -> dict:
    """{"HH:MM": multiplier} on the day's overall realized vol, built from
    every historical 15-min bar's own bucket variance relative to the
    session average. The overnight return (yesterday's last bar to today's
    first) is excluded per day via groupby-shift, so an overnight gap never
    inflates the first bucket's apparent intraday volatility."""
    df = intraday.sort_index()
    prev_close = df["Close"].groupby(df.index.date).shift(1)
    log_ret = np.log(df["Close"] / prev_close)
    bucket_var = pd.Series(log_ret.values ** 2, index=df.index.strftime("%H:%M")).groupby(level=0).mean()
    bucket_var = bucket_var.dropna()
    if bucket_var.empty:
        return {}
    overall = bucket_var.mean()
    if overall <= 0:
        return {}
    return (bucket_var / overall).apply(np.sqrt).to_dict()


def _trading_days_until_expiry(entry_date, target_weekday: int) -> int:
    d = entry_date + timedelta(days=1)
    trading_days = 0
    while True:
        if d.weekday() < 5:
            trading_days += 1
        if d.weekday() == target_weekday:
            return trading_days
        d += timedelta(days=1)


def _round_to_step(value: float, step: int) -> int:
    return int(round(value / step) * step)


def _simulate_day(day_bars: pd.DataFrame, sigma_daily: float, vol_profile: dict, avg_range: float,
                   expiry_days: int, cfg: dict) -> dict | None:
    entry_matches = day_bars[day_bars.index.strftime("%H:%M") == ENTRY_TIME]
    if entry_matches.empty or pd.isna(sigma_daily) or sigma_daily <= 0 or pd.isna(avg_range) or avg_range <= 0:
        return None
    entry_pos = day_bars.index.get_loc(entry_matches.index[0])
    day_open = float(day_bars["Open"].iloc[0])
    spot_entry = float(day_bars["Close"].iloc[entry_pos])

    step, width = cfg["strike_step"], cfg["spread_width"]
    short_call_k = _round_to_step(day_open + avg_range, step)
    short_put_k = _round_to_step(day_open - avg_range, step)
    if short_call_k <= spot_entry or short_put_k >= spot_entry:
        return None  # spot already outside the intended range by entry time -- skip
    long_call_k, long_put_k = short_call_k + width, short_put_k - width

    t_years = expiry_days / TRADING_DAYS_PER_YEAR
    sigma_entry = sigma_daily * vol_profile.get(ENTRY_TIME, 1.0)
    credit = _condor_cost_to_close(spot_entry, t_years, sigma_entry, short_call_k, long_call_k, short_put_k, long_put_k)
    if credit <= 0:
        return None
    max_loss = width - credit
    if max_loss <= 0:
        return None
    stop_loss_value = credit * STOP_LOSS_MULTIPLE

    exit_reason = exit_spot = exit_time = exit_cost = None
    for i in range(entry_pos + 1, len(day_bars)):
        t_str = day_bars.index[i].strftime("%H:%M")
        spot_now = float(day_bars["Close"].iloc[i])
        sigma_now = sigma_daily * vol_profile.get(t_str, 1.0)
        cost_now = _condor_cost_to_close(spot_now, t_years, sigma_now, short_call_k, long_call_k, short_put_k,
                                          long_put_k)
        hit_stop = cost_now >= stop_loss_value
        if hit_stop or t_str >= EXIT_TIME:
            exit_reason = "Stop-loss" if hit_stop else "Time-based exit"
            exit_spot, exit_time, exit_cost = spot_now, day_bars.index[i], cost_now
            break

    if exit_reason is None:
        exit_time = day_bars.index[-1]
        exit_spot = float(day_bars["Close"].iloc[-1])
        sigma_last = sigma_daily * vol_profile.get(exit_time.strftime("%H:%M"), 1.0)
        exit_cost = _condor_cost_to_close(exit_spot, t_years, sigma_last, short_call_k, long_call_k, short_put_k,
                                           long_put_k)
        exit_reason = "Time-based exit"

    pnl = credit - exit_cost
    return {
        "EntryDate": day_bars.index[entry_pos].date(), "ExitDate": exit_time.date(),
        "EntryTime": day_bars.index[entry_pos], "ExitTime": exit_time,
        "SpotEntry": round(spot_entry, 2), "SpotExit": round(exit_spot, 2),
        "ShortCall": short_call_k, "LongCall": long_call_k,
        "ShortPut": short_put_k, "LongPut": long_put_k,
        "DaysToExpiry": expiry_days, "SigmaEntryPct": round(sigma_entry * 100, 1),
        "CreditPts": round(credit, 2), "MaxLossPts": round(max_loss, 2), "PnLPts": round(pnl, 2),
        "RiskPct": round(max_loss / spot_entry * 100, 2),
        "ReturnPct": round(pnl / max_loss * 100, 2),
        "ExitReason": exit_reason,
    }


def backtest_range_condor(symbol: str, expiry_weekday: str = "Thursday", start_date: str = "2020-01-01",
                           force_refresh: bool = False) -> list[dict]:
    if symbol not in INSTRUMENT_CONFIG:
        raise ValueError(f"Unsupported symbol {symbol!r} -- choose one of {list(INSTRUMENT_CONFIG)}")
    cfg = INSTRUMENT_CONFIG[symbol]
    target_weekday = EXPIRY_WEEKDAY_MAP[expiry_weekday]

    daily = load_frame(symbol, "1D", force_refresh=force_refresh)
    intraday = load_frame(symbol, "15MIN", force_refresh=force_refresh)
    if daily is None or intraday is None or intraday.empty:
        return []

    vol_profile = _time_of_day_vol_multiplier(intraday)
    sigma_by_date = {ts.date(): float(v) for ts, v in _realized_vol_series(daily["Close"]).shift(1).items()
                      if pd.notna(v)}
    range_by_date = {ts.date(): float(v) for ts, v in _avg_range_series(daily).items() if pd.notna(v)}
    start = pd.Timestamp(start_date).date()

    trades = []
    for date, day_bars in intraday.groupby(intraday.index.date):
        if date.weekday() == target_weekday or date.weekday() >= 5 or date < start:
            continue
        sigma_daily = sigma_by_date.get(date)
        avg_range = range_by_date.get(date)
        if sigma_daily is None or avg_range is None:
            continue
        expiry_days = _trading_days_until_expiry(date, target_weekday)
        if expiry_days < MIN_DAYS_TO_EXPIRY:
            continue
        trade = _simulate_day(day_bars, sigma_daily, vol_profile, avg_range, expiry_days, cfg)
        if trade:
            trade["Symbol"] = symbol
            trades.append(trade)

    return trades


def run_backtest(symbol: str, expiry_weekday: str = "Thursday", start_date: str = "2020-01-01",
                  force_refresh: bool = False) -> pd.DataFrame:
    trades = backtest_range_condor(symbol, expiry_weekday, start_date, force_refresh=force_refresh)
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values("EntryDate").reset_index(drop=True)
    return df
