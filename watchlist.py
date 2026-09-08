"""Persistent watchlist: symbols saved with the scan rule they were added under,
so a headless background check (see check_watchlist.py) can re-evaluate each one
without a Streamlit sidebar to read settings from.

Two JSON files under data/ (same convention as constituents.py / market_data.py's
CSV caches, just JSON since these aren't NSE feed data):
  - watchlist.json       -- the list of watched entries
  - watchlist_state.json -- last-known status per entry, used to detect a
                             transition INTO a triggerable condition, not just
                             "is it currently true" (which would re-alert every check).

NOTE on concurrency: the Streamlit app and the launchd background job can both
write these files around the same moment. Writes are atomic (temp file + os.replace)
so a reader never sees a half-written file, but it's last-writer-wins on the whole
file, not a merge -- acceptable for a single-user local tool, not solved further here.
"""
import json
import os
import uuid
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from screener import scan_symbol, scan_symbol_cross, scan_symbol_volume, scan_symbol_pattern, apply_band, apply_band_below
from telegram_bot import send_telegram_message

logger = logging.getLogger("trendline.watchlist")

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
STATE_PATH = os.path.join(DATA_DIR, "watchlist_state.json")

SCAN_MODE_LABELS = {
    "above_ma": "Above 200 MA",
    "below_ma": "Below 200 MA",
    "golden_cross": "Golden Cross / Death Cross (50 vs 200)",
    "unusual_volume": "Unusual Volume (Buying/Selling Spike)",
    "chart_pattern": "Chart Patterns (Triangle / Channel / Flag & Pole)",
}
SCAN_MODE_CODES = {v: k for k, v in SCAN_MODE_LABELS.items()}

# Statuses that count as "interesting" for above_ma / below_ma / golden_cross / unusual_volume.
# chart_pattern is handled separately (set-diff on Pattern:Direction hits).
TRIGGER_VALUES = {
    "above_ma": {"In band"},
    "below_ma": {"In band"},
    "golden_cross": {"Golden Cross", "Death Cross"},
    "unusual_volume": {"Unusual Buying", "Unusual Selling", "Volume Spike (Flat)"},
}

GOLDEN_CROSS_LONG_TERM_NOTE = (
    "Note: a Golden Cross on the Weekly or Monthly timeframe has historically often "
    "preceded the stock roughly doubling -- not guaranteed, just a pattern worth watching."
)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

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


def load_watchlist() -> list[dict]:
    return _load_json(WATCHLIST_PATH, [])


def save_watchlist(entries: list[dict]) -> None:
    _save_json(WATCHLIST_PATH, entries)


def load_state() -> dict:
    return _load_json(STATE_PATH, {})


def save_state(state: dict) -> None:
    _save_json(STATE_PATH, state)


# ---------------------------------------------------------------------------
# Add / remove
# ---------------------------------------------------------------------------

def add_entry(symbol: str, segment: str, universe: str, timeframe: str, ma_type: str,
              scan_mode: str, params: dict) -> dict:
    entry = {
        "id": f"wl_{uuid.uuid4().hex[:12]}",
        "symbol": symbol,
        "segment": segment,
        "universe": universe,
        "timeframe": timeframe,
        "ma_type": ma_type,
        "scan_mode": scan_mode,
        "params": params,
        "added_at": datetime.now(IST).isoformat(),
    }
    entries = load_watchlist()
    entries.append(entry)
    save_watchlist(entries)
    return entry


def remove_entry(entry_id: str) -> None:
    entries = [e for e in load_watchlist() if e["id"] != entry_id]
    save_watchlist(entries)
    state = load_state()
    state.pop(entry_id, None)
    save_state(state)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_entry(entry: dict, force_refresh: bool = False) -> dict:
    """Re-run the entry's saved scan rule for its symbol. Never raises -- returns
    {"ok": False, "reason": ...} on any failure (missing data, bad params, etc.)."""
    try:
        mode = entry["scan_mode"]
        params = entry.get("params", {})
        symbol, segment = entry["symbol"], entry["segment"]
        timeframe, ma_type = entry["timeframe"], entry["ma_type"]

        if mode == "above_ma":
            row = scan_symbol(symbol, segment, timeframe, ma_type, force_refresh=force_refresh)
            if row is None:
                return {"ok": False, "reason": "no data", "status": None, "raw": None}
            df = apply_band(pd.DataFrame([row]), params["band_low"], params["band_high"])
            status, raw = df["Status"].iloc[0], row

        elif mode == "below_ma":
            row = scan_symbol(symbol, segment, timeframe, ma_type, force_refresh=force_refresh)
            if row is None:
                return {"ok": False, "reason": "no data", "status": None, "raw": None}
            df = apply_band_below(pd.DataFrame([row]), params["band_low"], params["band_high"])
            status, raw = df["Status"].iloc[0], row

        elif mode == "golden_cross":
            row = scan_symbol_cross(symbol, segment, timeframe, ma_type, params["lookback"],
                                     force_refresh=force_refresh)
            if row is None:
                return {"ok": False, "reason": "no data", "status": None, "raw": None}
            status, raw = row["CrossType"], row

        elif mode == "unusual_volume":
            row = scan_symbol_volume(symbol, segment, timeframe, params["avg_period"],
                                      params["spike_multiple"], force_refresh=force_refresh)
            if row is None:
                return {"ok": False, "reason": "no data", "status": None, "raw": None}
            status, raw = row["Activity"], row

        elif mode == "chart_pattern":
            hits = scan_symbol_pattern(symbol, segment, timeframe, params["pattern_types"],
                                        params["pattern_lookback"], params["pole_min_move_pct"],
                                        force_refresh=force_refresh)
            status = "|".join(sorted(f"{h['Pattern']}:{h['Direction']}" for h in hits)) if hits else "None"
            raw = hits

        else:
            return {"ok": False, "reason": f"unknown scan_mode '{mode}'", "status": None, "raw": None}

        return {"ok": True, "status": status, "raw": raw, "checked_at": datetime.now(IST).isoformat()}

    except KeyError as e:
        return {"ok": False, "reason": f"missing param {e}", "status": None, "raw": None}
    except Exception as e:
        return {"ok": False, "reason": f"evaluation error: {e}", "status": None, "raw": None}


def _is_new_trigger(entry: dict, prev_status: str | None, current_status: str) -> tuple[bool, str | None]:
    """Alert only on a transition INTO a triggerable status -- not on staying there
    across repeat checks. First-ever check for an entry (prev_status is None) never
    alerts; it just records the baseline."""
    if prev_status is None:
        return False, None

    mode = entry["scan_mode"]
    symbol = entry["symbol"]

    if mode == "chart_pattern":
        prev_set = set(prev_status.split("|")) if prev_status != "None" else set()
        cur_set = set(current_status.split("|")) if current_status != "None" else set()
        new_hits = cur_set - prev_set
        if new_hits:
            return True, f"TrendLine alert: {symbol} -- new pattern breakout: {', '.join(sorted(new_hits))}"
        return False, None

    trigger_vals = TRIGGER_VALUES.get(mode, set())
    if current_status in trigger_vals and current_status != prev_status:
        label = SCAN_MODE_LABELS.get(mode, mode)
        message = f"TrendLine alert: {symbol} -- {label} -> {current_status}"
        if mode == "golden_cross" and current_status == "Golden Cross" and entry["timeframe"] in ("1W", "1M"):
            message += f"\n\n{GOLDEN_CROSS_LONG_TERM_NOTE}"
        return True, message
    return False, None


def check_all(send_alerts: bool = True, force_refresh: bool = False) -> list[dict]:
    """Evaluate every watchlist entry, diff against stored state, send a Telegram
    alert for each newly-triggered entry, persist the new state, and return
    per-entry results for display. One bad entry never aborts the batch -- this
    runs unattended under launchd."""
    entries = load_watchlist()
    state = load_state()
    results = []

    for entry in entries:
        eid = entry["id"]
        try:
            eval_result = evaluate_entry(entry, force_refresh=force_refresh)
        except Exception as e:
            eval_result = {"ok": False, "reason": f"unexpected error: {e}", "status": None, "raw": None}

        triggered, message = False, None
        if eval_result["ok"]:
            prev = state.get(eid)
            prev_status = prev["status"] if prev else None
            triggered, message = _is_new_trigger(entry, prev_status, eval_result["status"])
            state[eid] = {"status": eval_result["status"], "checked_at": eval_result.get("checked_at")}

        eval_result["entry"] = entry
        eval_result["triggered"] = triggered
        eval_result["message"] = message
        if triggered and send_alerts:
            eval_result["alert_sent"] = send_telegram_message(message)

        results.append(eval_result)

    save_state(state)
    return results
