"""Scheduled daily CPR alert: scans the full universe once per day and posts
NEWLY-narrow-CPR symbols to Telegram.

Unlike universe_digest.py's other scan modes, this runs on its own once-daily
cadence rather than 15-Minute/1-Hour buckets: CPR is computed from the last
*completed* session (see cpr.py's _basis_row) and is fixed for the whole next
session, so rescanning intraday would just recompute the same numbers.

Fixed scan configuration:
  - Universe: All three combined (Nifty 100 + Midcap 150 + Smallcap 250)
  - "Narrow" threshold: cpr.py's own WIDTH_NARROW_MAX (kept as the single
    source of truth so the Streamlit tab and this alert never drift apart)
  - force_refresh: OFF, same rationale as universe_digest.py -- daily cache
    TTLs already keep the prior session's OHLC fresh enough for this.
"""
import html
import json
import os
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import nse_bhavcopy

CPR_WIDTH_LT = 0.05  # Narrow CPR: 0% < width < 0.05% (same as the Narrow CPR tab)
from telegram_bot import send_telegram_message

logger = logging.getLogger("trendline.cpr_alert")

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_PATH = os.path.join(DATA_DIR, "cpr_state.json")

ALL_UNIVERSES = ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"]

MARKET_CLOSE = dtime(18, 30)  # Bhavcopy download time (IST)
DAILY_WINDOW_END = dtime(23, 59)

# --- Telegram cadence: exactly two messages per trading day -------------------
# Evening: sent once, as soon as today's NSE Bhavcopy CSV is actually published
#          (polled every 15 min from MARKET_CLOSE by check_watchlist.py).
# Morning: sent once at ~08:50 IST by the trendline-morning.timer (check_watchlist.py --morning).
EVENING_SEND_WHEN_EMPTY = True  # still notify (short "no new symbols" note) when Bhavcopy is out but nothing newly qualifies
MORNING_TIME = dtime(8, 50)
MORNING_WINDOW_END = dtime(11, 0)  # if the VM was down at 08:50, still send until this time
MORNING_MAX_ROWS = 60


def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to read %s (%s) -- falling back to default", path, e)
        return default


def _save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _format_message(now_ist: datetime, details: list[dict]) -> str:
    header = (f"{now_ist:%d-%b-%Y %H:%M} IST\n\n"
              f"🎯 Narrow CPR — newly qualifying for next session")
    rows = [(d["symbol"], f"Width {d['width_pct']:.2f}%") for d in details]
    sym_width = max((len(s) for s, _ in rows), default=0)
    table_text = "\n".join(f"{s.ljust(sym_width)}  {w}" for s, w in rows)
    return f"{html.escape(header)}\n<pre>{html.escape(table_text)}</pre>"


def _format_empty_message(now_ist: datetime, total_narrow: int, asof: str) -> str:
    return html.escape(f"{now_ist:%d-%b-%Y %H:%M} IST\n\n"
                       f"🎯 Narrow CPR — {asof} Bhavcopy is out. "
                       f"No newly qualifying symbols for the next session "
                       f"({total_narrow} narrow in total).")


def _format_morning_message(now_ist: datetime, symbols: list[str], widths: dict, asof: str) -> str:
    ordered = sorted(symbols, key=lambda s: (widths.get(s) is None, widths.get(s, 0.0), s))
    shown = ordered[:MORNING_MAX_ROWS]
    header = (f"{now_ist:%d-%b-%Y %H:%M} IST\n\n"
              f"🌅 Narrow CPR — watchlist for today (based on {asof} Bhavcopy, {len(symbols)} symbols)")
    rows = [(s, f"Width {widths[s]:.2f}%" if s in widths else "") for s in shown]
    sym_width = max((len(s) for s, _ in rows), default=0)
    table_text = "\n".join(f"{s.ljust(sym_width)}  {w}".rstrip() for s, w in rows)
    more = len(ordered) - len(shown)
    footer = f"\n… and {more} more (narrowest first)" if more > 0 else ""
    return f"{html.escape(header)}\n<pre>{html.escape(table_text)}</pre>{html.escape(footer)}"


def run_cycle(now_ist: datetime, send_alerts: bool = True) -> dict:
    """EVENING message. Runs at most once per calendar day, in the post-close window
    (after 18:30 IST), and only once NSE's Bhavcopy CSV for TODAY has actually been
    published -- until then every 15-min cycle just polls and returns quietly.
    Sends exactly one Telegram message per day (newly-qualifying symbols, or a short
    "nothing new" note when EVENING_SEND_WHEN_EMPTY). If the send fails, state is NOT
    saved, so the next cycle retries instead of silently dropping the message."""
    state = _load_json(STATE_PATH, {"narrow_symbols": [], "last_run": None})
    state.setdefault("narrow_symbols", [])

    day_key = now_ist.strftime("%Y-%m-%d")
    in_daily_window = MARKET_CLOSE <= now_ist.time() <= DAILY_WINDOW_END
    already_ran_today = state.get("last_run") == day_key

    if not (in_daily_window and not already_ran_today):
        return {"ran": False, "reason": "outside daily window or already ran today", "hits": 0}

    is_first_run = state.get("last_run") is None

    try:
        results = nse_bhavcopy.scan_bhavcopy_cpr(fo_only=False, now_ist=now_ist)
        if results.empty or results["AsOf"].iloc[0].date() != now_ist.date():
            return {"ran": False, "reason": "today's Bhavcopy not available yet", "hits": 0}
    except Exception as e:
        logger.error("CPR universe scan failed: %s", e)
        return {"ran": False, "reason": f"scan failed: {e}", "hits": 0}

    current_narrow = {}
    if not results.empty:
        narrow = results[(results["WidthPct"] > 0) & (results["WidthPct"] < CPR_WIDTH_LT)]
        current_narrow = {r.Symbol: float(r.WidthPct) for r in narrow.itertuples()}

    prev_keys = set(state["narrow_symbols"])
    current_keys = set(current_narrow.keys())
    new_keys = set() if is_first_run else (current_keys - prev_keys)
    asof = results["AsOf"].iloc[0].strftime("%d-%b-%Y")

    messages_sent = 0
    if send_alerts and not is_first_run:
        if new_keys:
            details = [{"symbol": s, "width_pct": current_narrow[s]} for s in sorted(new_keys)]
            msg = _format_message(now_ist, details)
        elif EVENING_SEND_WHEN_EMPTY:
            msg = _format_empty_message(now_ist, len(current_keys), asof)
        else:
            msg = None
        if msg is not None:
            if not send_telegram_message(msg, parse_mode="HTML"):
                return {"ran": True, "hits": len(new_keys), "messages_sent": 0,
                        "reason": "telegram send failed -- will retry next cycle"}
            messages_sent = 1

    state["narrow_symbols"] = sorted(current_keys)
    state["narrow_details"] = {k: round(v, 4) for k, v in current_narrow.items()}  # used by the morning message
    state["asof"] = asof
    state["last_run"] = day_key
    _save_json(STATE_PATH, state)

    return {"ran": True, "hits": len(new_keys), "messages_sent": messages_sent}


def run_morning(now_ist: datetime, send_alerts: bool = True) -> dict:
    """MORNING message (~08:50 IST, Mon-Fri): today's Narrow CPR watchlist, i.e. the
    symbols from the last published Bhavcopy (CPR is fixed for the whole next session).
    No network calls -- reads the state the evening run saved. Once per day."""
    if now_ist.weekday() >= 5:
        return {"ran": False, "reason": "weekend"}
    if not (MORNING_TIME <= now_ist.time() <= MORNING_WINDOW_END):
        return {"ran": False, "reason": "outside morning window"}

    state = _load_json(STATE_PATH, {})
    day_key = now_ist.strftime("%Y-%m-%d")
    if state.get("last_morning") == day_key:
        return {"ran": False, "reason": "morning message already sent today"}

    symbols = state.get("narrow_symbols") or []
    if not symbols:
        return {"ran": False, "reason": "no Narrow CPR data saved yet"}

    msg = _format_morning_message(now_ist, symbols, state.get("narrow_details") or {},
                                  state.get("asof") or "last session")
    if send_alerts:
        if not send_telegram_message(msg, parse_mode="HTML"):
            return {"ran": True, "messages_sent": 0, "reason": "telegram send failed"}
    state["last_morning"] = day_key
    _save_json(STATE_PATH, state)
    return {"ran": True, "messages_sent": 1 if send_alerts else 0, "symbols": len(symbols)}
