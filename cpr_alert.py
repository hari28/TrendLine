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

from constituents import get_all_symbols
from cpr import scan_universe_cpr, WIDTH_NARROW_MAX
from telegram_bot import send_telegram_message

logger = logging.getLogger("trendline.cpr_alert")

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_PATH = os.path.join(DATA_DIR, "cpr_state.json")

ALL_UNIVERSES = ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"]

MARKET_CLOSE = dtime(15, 30)
DAILY_WINDOW_END = dtime(16, 0)


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


def run_cycle(now_ist: datetime, send_alerts: bool = True) -> dict:
    """Runs at most once per calendar day, in the post-close window
    (15:30-16:00 IST) -- same window universe_digest.py uses for its daily
    combos, since that's when the day's session first becomes "completed".
    Self-gates on its own state, so it's safe to call every 15-min cycle."""
    state = _load_json(STATE_PATH, {"narrow_symbols": [], "last_run": None})
    state.setdefault("narrow_symbols", [])

    day_key = now_ist.strftime("%Y-%m-%d")
    in_daily_window = MARKET_CLOSE <= now_ist.time() <= DAILY_WINDOW_END
    already_ran_today = state.get("last_run") == day_key

    if not (in_daily_window and not already_ran_today):
        return {"ran": False, "reason": "outside daily window or already ran today", "hits": 0}

    is_first_run = state.get("last_run") is None

    try:
        symbols_df = get_all_symbols(ALL_UNIVERSES, force_refresh=False)[["Symbol", "Segment"]]
        results = scan_universe_cpr(symbols_df, force_refresh=False)
    except Exception as e:
        logger.error("CPR universe scan failed: %s", e)
        return {"ran": False, "reason": f"scan failed: {e}", "hits": 0}

    current_narrow = {}
    if not results.empty:
        narrow = results[results["WidthPct"] <= WIDTH_NARROW_MAX]
        current_narrow = {r.Symbol: float(r.WidthPct) for r in narrow.itertuples()}

    prev_keys = set(state["narrow_symbols"])
    current_keys = set(current_narrow.keys())
    new_keys = set() if is_first_run else (current_keys - prev_keys)

    state["narrow_symbols"] = sorted(current_keys)
    state["last_run"] = day_key
    _save_json(STATE_PATH, state)

    messages_sent = 0
    if new_keys and send_alerts:
        details = [{"symbol": s, "width_pct": current_narrow[s]} for s in sorted(new_keys)]
        if send_telegram_message(_format_message(now_ist, details), parse_mode="HTML"):
            messages_sent = 1

    return {"ran": True, "hits": len(new_keys), "messages_sent": messages_sent}
