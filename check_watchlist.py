#!/usr/bin/env python3
"""Standalone entry point for the launchd scheduled job -- NOT part of the
Streamlit process. Invoke with the venv's python, e.g.:

    venv/bin/python3 check_watchlist.py

Runs several things, all gated to NSE-relevant hours (IST) so it's harmless
to leave the launchd job loaded permanently -- see README.md "Universe
Digest" for details:
  1. The full-universe scan digest (universe_digest.py) -- during trading
     hours for its 15-Minute/1-Hour combos, plus a short window after close
     (15:30-16:00) for its once-daily 1D/1W/1M combos. It self-gates its own
     cadence internally, so it's safe to call every cycle.
  2. The ad-hoc "alert me on this scan" cycle (screener_alert.py) -- only
     during live trading hours (Mon-Fri 9:15-15:30).
  3. Paper trading (paper_trading.py) -- all eight strategies are checked
     once, in the 15:30-16:00 post-close window (seven hold positions
     across days and get a fresh entry/exit check; EMA Scalping's trades
     resolve within the same session, so its check just logs the day's
     already-resolved trades), since a day's own bar isn't meaningfully
     final before then.

A file lock prevents two invocations from ever running concurrently -- the
first-ever universe digest cycle has to cold-fetch ~500 stocks and can take
longer than the 15-minute launchd interval, and overlapping runs could race
on the shared state JSON files.
"""
import fcntl
import os
import sys
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)  # ensure local imports resolve regardless of launchd's cwd

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import universe_digest
import screener_alert
import cpr_alert
import paper_trading

LOG_PATH = os.path.join(PROJECT_ROOT, "data", "watchlist_check.log")
LOCK_PATH = os.path.join(PROJECT_ROOT, "data", ".check_watchlist.lock")
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 30)
DAILY_DIGEST_END = dtime(16, 0)


def _is_trading_hours(now_ist: datetime) -> bool:
    if now_ist.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def _is_eligible_window(now_ist: datetime) -> bool:
    """Broader than trading hours -- also covers the post-close window the
    universe digest's daily combos run in."""
    if now_ist.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_ist.time() <= DAILY_DIGEST_END


def _is_post_close_window(now_ist: datetime) -> bool:
    """15:30-16:00 IST -- same window cpr_alert.py's daily check uses. Paper
    trading's daily-bar strategies wait for this window rather than running
    from market open, since a day's own daily bar isn't meaningfully final
    (or even mostly formed) until after the close."""
    if now_ist.weekday() >= 5:
        return False
    return MARKET_CLOSE <= now_ist.time() <= DAILY_DIGEST_END


def _log(line: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    now_ist = datetime.now(IST)

    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    lock_file = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _log(f"{now_ist:%Y-%m-%d %H:%M:%S %z} SKIPPED (previous run still in progress)")
        return

    try:
        if not _is_eligible_window(now_ist):
            _log(f"{now_ist:%Y-%m-%d %H:%M:%S %z} SKIPPED (outside eligible window)")
            return

        parts = [f"{now_ist:%Y-%m-%d %H:%M:%S %z}"]

        if _is_trading_hours(now_ist):
            screener_alert_result = screener_alert.run_cycle(now_ist)
            if screener_alert_result["ran"]:
                parts.append(f"screener_alert: sent={screener_alert_result['sent']}")
            else:
                parts.append(f"screener_alert: skipped ({screener_alert_result['reason']})")
        else:
            parts.append("screener_alert: skipped (outside trading hours)")

        digest = universe_digest.run_cycle(now_ist, send_alerts=True)
        if digest["ran"]:
            b = digest["buckets"]
            parts.append(f"digest: hits={digest['hits']} messages_sent={digest.get('messages_sent', 0)} "
                         f"buckets=[15min={b['15min']} hourly={b['hourly']} daily={b['daily']}]")
        else:
            parts.append(f"digest: skipped ({digest.get('reason', 'n/a')})")

        cpr_result = cpr_alert.run_cycle(now_ist, send_alerts=True)
        if cpr_result["ran"]:
            parts.append(f"cpr: hits={cpr_result['hits']} messages_sent={cpr_result.get('messages_sent', 0)}")
        else:
            parts.append(f"cpr: skipped ({cpr_result.get('reason', 'n/a')})")

        if _is_post_close_window(now_ist):
            for strat in paper_trading.DAILY_BAR_STRATEGIES:
                try:
                    r = paper_trading.run_daily_cycle(strat)
                    parts.append(f"paper[{strat}]: {r}")
                except Exception as e:
                    parts.append(f"paper[{strat}]: ERROR {e}")
            for strat in paper_trading.SAME_DAY_STRATEGIES:
                try:
                    r = paper_trading.run_same_day_cycle(strat)
                    parts.append(f"paper[{strat}]: {r}")
                except Exception as e:
                    parts.append(f"paper[{strat}]: ERROR {e}")

        _log(" | ".join(parts))
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


if __name__ == "__main__":
    main()
