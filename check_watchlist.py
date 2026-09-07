#!/usr/bin/env python3
"""Standalone entry point for the launchd scheduled job -- NOT part of the
Streamlit process. Invoke with the venv's python, e.g.:

    venv/bin/python3 check_watchlist.py

Runs one watchlist check-and-alert pass, gated to NSE market hours (IST,
Mon-Fri 9:15-15:30) so it's harmless to leave the launchd job loaded
permanently -- see README.md for the launchd plist and install steps.
"""
import os
import sys
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)  # ensure local imports resolve regardless of launchd's cwd

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import watchlist

LOG_PATH = os.path.join(PROJECT_ROOT, "data", "watchlist_check.log")
IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN, MARKET_CLOSE = dtime(9, 15), dtime(15, 30)


def _is_market_hours(now_ist: datetime) -> bool:
    if now_ist.weekday() >= 5:  # Sat=5, Sun=6
        return False
    return MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE


def _log(line: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def main():
    now_ist = datetime.now(IST)
    if not _is_market_hours(now_ist):
        _log(f"{now_ist:%Y-%m-%d %H:%M:%S %z} SKIPPED (outside NSE market hours)")
        return

    results = watchlist.check_all(send_alerts=True)
    triggered = [r for r in results if r.get("triggered")]
    failed = [r for r in results if not r.get("ok")]
    sent = sum(1 for r in triggered if r.get("alert_sent"))
    _log(f"{now_ist:%Y-%m-%d %H:%M:%S %z} checked={len(results)} triggered={len(triggered)} "
         f"alerts_sent={sent} failed={len(failed)}")


if __name__ == "__main__":
    main()
