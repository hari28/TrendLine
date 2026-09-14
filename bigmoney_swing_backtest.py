"""An APPROXIMATION of a "big-money accumulation, anticipation entry" swing
strategy, inspired by CA Afzal Lokhandwala's PUBLICLY STATED trading
philosophy (price/volume-only technical analysis -- no fundamentals; swing
entries taken in ANTICIPATION of a breakout rather than after it confirms;
no intraday, no options; 1-4 week holds targeting 10-30% gains; end-of-day
only scanning, no all-day screen time).

IMPORTANT -- this is NOT his actual system. His specific mechanical rules
(what precisely counts as "big money" price/volume behavior, his exact
anticipation-entry trigger, his stop-loss method) are taught in his paid
Champion Trading Course and are not published anywhere publicly. Every
public source (his own website, interviews, YouTube) describes only the
philosophy above, in general terms -- there is no public specification to
implement faithfully here, unlike this app's other strategies (CPR+EMA, EMA
Scalping) which came from an exact user-supplied rule set. This module is
TrendLine's OWN mechanical translation of that public philosophy, built
just to be testable. Read every result as evidence about THIS specific
approximation, never as evidence about his real, undisclosed system.

Rules implemented:
- Trend filter: Close > 50 EMA -- an established uptrend ("big money" is
  already in the stock, this isn't a bottom-picking strategy).
- Setup ("anticipation" zone): Close is within NEAR_HIGH_PCT of the recent
  HIGH_LOOKBACK-day high, but has NOT yet closed above it -- i.e. coiling
  just under resistance, not chasing an already-confirmed breakout the way
  this app's other breakout strategy does (swing_backtest.py's Strategy A,
  which enters ON the close-above-level day).
- Trigger (the "big money" proxy): that same day's volume is at least
  VOLUME_PICKUP_MULTIPLE x the trailing 20-day average -- a volume pickup
  AHEAD of the actual breakout. This is the closest mechanical stand-in
  available without a real order-flow/delivery-volume feed; it is a guess
  at the spirit of "big investors accumulating," not his real signal.
- Entry: at that trigger day's own close.
- Stop-loss: ATR-based (Entry - ATR_MULTIPLE * ATR(14)), the same
  house-style convention used by every other backtest in this app (see
  swing_backtest.py) -- skip the trade if that implies more than
  MAX_RISK_PCT risk, rather than tightening an already-thin stop.
- Target: a single target_pct, exposed as a parameter and constrained to
  the publicly stated 10-30% range (default 20%, the midpoint).
- Time-based exit: if neither stop nor target is hit within
  MAX_HOLDING_DAYS trading days (~4 weeks), force-close at that day's
  close -- matches the stated 1-4 week holding period; this is never meant
  to become a multi-month hold.
- One trade per symbol at a time.
"""
import pandas as pd

from indicators import atr, ema
from screener import load_frame
from swing_backtest import simulate_portfolio, summarize  # noqa: F401 (re-exported) -- EntryDate < ExitDate
                                                            # always holds here (multi-day swing holds), so
                                                            # the date-only event ordering is safe to reuse
                                                            # as-is (unlike scalping_backtest.py's same-day
                                                            # trades, which needed a timestamp-based version).

TREND_EMA = 50
HIGH_LOOKBACK = 20
NEAR_HIGH_PCT = 5.0             # "anticipation zone" = within 5% below the recent high, not yet through it
VOL_AVG_PERIOD = 20
VOLUME_PICKUP_MULTIPLE = 1.5    # the "big money" volume-pickup proxy
ATR_PERIOD = 14
ATR_MULTIPLE = 1.5
MAX_RISK_PCT = 4.5
MAX_HOLDING_DAYS = 20           # ~4 trading weeks
MIN_TARGET_PCT, MAX_TARGET_PCT = 10.0, 30.0
DEFAULT_TARGET_PCT = 20.0


def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["EMATrend"] = ema(df["Close"], TREND_EMA)
    df["ATR14"] = atr(df, ATR_PERIOD)
    # both exclude the current bar so today's own move never inflates its own baseline/level
    df["RollingHigh"] = df["Close"].shift(1).rolling(window=HIGH_LOOKBACK, min_periods=HIGH_LOOKBACK).max()
    df["VolAvg"] = df["Volume"].shift(1).rolling(window=VOL_AVG_PERIOD, min_periods=VOL_AVG_PERIOD).mean()
    return df


def backtest_symbol(symbol: str, segment: str, target_pct: float = DEFAULT_TARGET_PCT,
                     start_date: str = "2020-01-01", force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    if frame is None or len(frame) < TREND_EMA + HIGH_LOOKBACK + 5:
        return []

    df = _prepare_indicators(frame)
    df = df.dropna(subset=["EMATrend", "ATR14", "RollingHigh", "VolAvg"])
    df = df[df.index >= pd.Timestamp(start_date)]
    if df.empty:
        return []

    trades = []
    in_trade_until = -1
    n = len(df)
    for i in range(n):
        if i <= in_trade_until:
            continue
        row = df.iloc[i]
        if row["Close"] <= row["EMATrend"]:
            continue
        near_high_floor = row["RollingHigh"] * (1 - NEAR_HIGH_PCT / 100.0)
        if not (near_high_floor <= row["Close"] < row["RollingHigh"]):
            continue
        if row["Volume"] < row["VolAvg"] * VOLUME_PICKUP_MULTIPLE:
            continue

        entry_price = float(row["Close"])
        atr_value = float(row["ATR14"])
        if pd.isna(atr_value) or atr_value <= 0:
            continue
        risk_distance = ATR_MULTIPLE * atr_value
        risk_pct = risk_distance / entry_price * 100.0
        if risk_pct <= 0 or risk_pct > MAX_RISK_PCT:
            continue
        stop_price = entry_price - risk_distance
        target_price = entry_price * (1 + target_pct / 100.0)

        exit_idx = exit_price = exit_reason = None
        for k in range(i + 1, min(i + 1 + MAX_HOLDING_DAYS, n)):
            low, high = df["Low"].iloc[k], df["High"].iloc[k]
            hit_stop, hit_target = low <= stop_price, high >= target_price
            if hit_stop:
                exit_idx, exit_price, exit_reason = k, stop_price, "Stop-loss"
                break
            if hit_target:
                exit_idx, exit_price, exit_reason = k, target_price, "Target"
                break

        if exit_idx is None:
            last_k = min(i + MAX_HOLDING_DAYS, n - 1)
            if last_k > i:
                exit_idx, exit_reason = last_k, "Time exit (~4 weeks)"
            else:
                exit_idx, exit_reason = i, "Open (end of data)"
            exit_price = float(df["Close"].iloc[exit_idx])

        pct_return = (exit_price - entry_price) / entry_price * 100.0
        trades.append({
            "Symbol": symbol,
            "Segment": segment,
            "EntryDate": df.index[i].date(),
            "EntryPrice": round(entry_price, 2),
            "StopPrice": round(stop_price, 2),
            "TargetPrice": round(target_price, 2),
            "RiskPct": round(risk_pct, 2),
            "ExitDate": df.index[exit_idx].date(),
            "ExitPrice": round(float(exit_price), 2),
            "ExitReason": exit_reason,
            "ReturnPct": round(pct_return, 2),
            "HoldingDays": exit_idx - i,
        })
        in_trade_until = exit_idx

    return trades


def run_backtest(symbols: list[tuple[str, str]], target_pct: float = DEFAULT_TARGET_PCT,
                  start_date: str = "2020-01-01", progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    all_trades = []
    total = len(symbols)
    for i, (symbol, segment) in enumerate(symbols):
        if progress_cb:
            progress_cb(i, total, symbol)
        all_trades.extend(backtest_symbol(symbol, segment, target_pct=target_pct, start_date=start_date,
                                           force_refresh=force_refresh))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values("EntryDate").reset_index(drop=True)
    return trades
