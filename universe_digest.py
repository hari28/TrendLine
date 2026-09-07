"""Scheduled full-universe scan digest: scans the full "All" universe (Nifty
100 + Midcap 150 + Smallcap 250 union) across every scan mode, on multiple
timeframes, and posts NEWLY-matching symbols to Telegram. Distinct from
watchlist.py, which only tracks individually-added symbols -- this covers
everyone.

Fixed scan configuration (per user's request):
  - Universe: All three combined (redundant to scan Large/Mid/Small
    separately since All is their union)
  - Scan modes: Above 200 MA, Golden/Death Cross, Unusual Volume, Chart
    Patterns (Triangle/Channel/Flag & Pole)
  - MA type: EMA only
  - Above-MA band: 0-2%
  - Minimum volume filter: 500,000
  - force_refresh: OFF -- deliberately, see README "Universe Digest" section
    for why (force-refreshing ~500 stocks x multiple timeframes every 15
    minutes would take far longer than 15 minutes per cycle and risks
    Yahoo/NSE rate-limiting the whole app). Existing per-timeframe cache TTLs
    already keep data fresh enough for this cadence.

Cadence (to avoid re-scanning data that hasn't changed):
  - 15-Minute combos: every cycle this module is called during market hours
  - 1 Hour combos: at most once per wall-clock hour
  - 1D/1W/1M combos: at most once per day, in the post-close window (these
    timeframes share a single daily fetch per symbol -- 1W/1M are resampled
    from the same daily bars, not fetched separately)
"""
import html
import json
import os
import logging
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

from constituents import get_all_symbols
from screener import (scan_universe, scan_universe_cross, scan_universe_volume, scan_universe_pattern,
                       apply_band)
from telegram_bot import send_telegram_message

logger = logging.getLogger("trendline.universe_digest")

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATE_PATH = os.path.join(DATA_DIR, "universe_digest_state.json")

ALL_UNIVERSES = ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"]

MA_TYPE = "EMA"
MIN_VOLUME = 500000
ABOVE_MA_BAND = (0.0, 2.0)
CROSS_LOOKBACK = 5
VOLUME_AVG_PERIOD = 20
VOLUME_SPIKE_MULTIPLE = 2.0
PATTERN_TYPES = ["Triangle", "Channel", "Flag & Pole"]
PATTERN_LOOKBACK = 80
PATTERN_POLE_MIN_MOVE_PCT = 8.0

SCAN_MODES = ["above_ma", "golden_cross", "unusual_volume", "chart_pattern"]
MODE_LABELS = {
    "above_ma": f"Above 200 {MA_TYPE}",
    "golden_cross": "Golden Cross / Death Cross",
    "unusual_volume": "Unusual Volume",
    "chart_pattern": "Chart Patterns",
}
TF_LABELS = {"15MIN": "15 Minute", "1H": "1 Hour", "1D": "1 Day", "1W": "1 Week", "1M": "1 Month"}

MARKET_OPEN = dtime(9, 15)
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


def _qualifying_symbols(mode: str, timeframe: str, symbols_df: pd.DataFrame) -> dict:
    """Returns {key: detail_line} for symbols currently qualifying under `mode`
    on `timeframe`. Never raises -- an empty dict on any scan failure."""
    try:
        if mode == "above_ma":
            results = scan_universe(symbols_df, timeframe, MA_TYPE, force_refresh=False)
            if results.empty:
                return {}
            results = results[results["Volume"] >= MIN_VOLUME]
            results = apply_band(results, *ABOVE_MA_BAND)
            hits = results[results["Status"] == "In band"]
            return {r.Symbol: f"{r.Symbol}: In band ({r.PctAbove:.2f}% above {MA_TYPE}200)"
                    for r in hits.itertuples()}

        if mode == "golden_cross":
            results = scan_universe_cross(symbols_df, timeframe, MA_TYPE, CROSS_LOOKBACK, force_refresh=False)
            if results.empty:
                return {}
            results = results[results["Volume"] >= MIN_VOLUME]
            hits = results[results["CrossType"].isin(["Golden Cross", "Death Cross"])]
            return {r.Symbol: f"{r.Symbol}: {r.CrossType}" for r in hits.itertuples()}

        if mode == "unusual_volume":
            results = scan_universe_volume(symbols_df, timeframe, VOLUME_AVG_PERIOD, VOLUME_SPIKE_MULTIPLE,
                                            force_refresh=False)
            if results.empty:
                return {}
            results = results[results["Volume"] >= MIN_VOLUME]
            hits = results[results["Activity"].isin(["Unusual Buying", "Unusual Selling", "Volume Spike (Flat)"])]
            return {r.Symbol: f"{r.Symbol}: {r.Activity}" for r in hits.itertuples()}

        if mode == "chart_pattern":
            results = scan_universe_pattern(symbols_df, timeframe, PATTERN_TYPES, PATTERN_LOOKBACK,
                                             PATTERN_POLE_MIN_MOVE_PCT, force_refresh=False)
            if results.empty:
                return {}
            results = results[results["Volume"] >= MIN_VOLUME]
            return {f"{r.Symbol}:{r.Pattern}:{r.Direction}": f"{r.Symbol}: {r.Pattern} ({r.Direction})"
                    for r in results.itertuples()}

    except Exception as e:
        logger.error("Scan failed for mode=%s timeframe=%s: %s", mode, timeframe, e)

    return {}


def _check_combo(mode: str, timeframe: str, symbols_df: pd.DataFrame, state: dict) -> list:
    """Diffs current qualifying symbols against stored state; returns detail
    lines for NEWLY qualifying ones. First-ever check for a combo never
    alerts -- it just records the baseline (same philosophy as watchlist.py)."""
    combo_key = f"{mode}:{timeframe}"
    current = _qualifying_symbols(mode, timeframe, symbols_df)
    current_keys = set(current.keys())
    is_first_run = combo_key not in state["matches"]

    prev_keys = set(state["matches"].get(combo_key, []))
    new_keys = set() if is_first_run else (current_keys - prev_keys)
    state["matches"][combo_key] = sorted(current_keys)

    return [(mode, timeframe, current[k]) for k in sorted(new_keys)]


def _group_hits(hits: list) -> dict:
    by_mode_tf = {}
    for mode, timeframe, detail in hits:
        by_mode_tf.setdefault((mode, timeframe), []).append(detail)
    return by_mode_tf


def _format_section(now_ist: datetime, index: int, mode: str, timeframe: str, details: list) -> str:
    """One rule's hits as a header + a monospace, column-aligned table (via
    Telegram's HTML <pre> tag) -- e.g. "RELIANCE  In band (1.20% above
    EMA200)". Sent as its own message so it never risks truncation breaking
    the HTML (which a single giant combined message could)."""
    header = f"{now_ist:%d-%b-%Y %H:%M} IST\n\n{index}) {MODE_LABELS.get(mode, mode)}\nTime Frame - {TF_LABELS.get(timeframe, timeframe)}"

    rows = []
    for d in details:
        symbol, _, rest = d.partition(": ")
        rows.append((symbol, rest))
    sym_width = max((len(s) for s, _ in rows), default=0)
    table_text = "\n".join(f"{s.ljust(sym_width)}  {r}" for s, r in rows)

    return f"{html.escape(header)}\n<pre>{html.escape(table_text)}</pre>"


def run_cycle(now_ist: datetime, send_alerts: bool = True) -> dict:
    """Runs whichever cadence buckets are due this cycle, sends one combined
    Telegram digest if anything newly qualified, and returns a summary dict
    for logging. Safe to call every 15 minutes -- internally self-gates."""
    state = _load_json(STATE_PATH, {"matches": {}, "cadence": {}})
    state.setdefault("matches", {})
    state.setdefault("cadence", {})

    in_trading_hours = MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE
    in_daily_window = MARKET_CLOSE <= now_ist.time() <= DAILY_WINDOW_END

    hour_key = now_ist.strftime("%Y-%m-%dT%H")
    day_key = now_ist.strftime("%Y-%m-%d")

    run_15min = in_trading_hours
    run_hourly = in_trading_hours and state["cadence"].get("last_hourly_run") != hour_key
    run_daily = in_daily_window and state["cadence"].get("last_daily_run") != day_key

    if not (run_15min or run_hourly or run_daily):
        return {"ran": False, "reason": "outside all cadence windows", "hits": 0}

    symbols_df = get_all_symbols(ALL_UNIVERSES, force_refresh=False)[["Symbol", "Segment"]]

    all_hits = []
    for mode in SCAN_MODES:
        if run_15min:
            all_hits.extend(_check_combo(mode, "15MIN", symbols_df, state))
        if run_hourly:
            all_hits.extend(_check_combo(mode, "1H", symbols_df, state))
        if run_daily:
            for tf in ("1D", "1W", "1M"):
                all_hits.extend(_check_combo(mode, tf, symbols_df, state))

    if run_hourly:
        state["cadence"]["last_hourly_run"] = hour_key
    if run_daily:
        state["cadence"]["last_daily_run"] = day_key

    _save_json(STATE_PATH, state)

    messages_sent = 0
    if all_hits and send_alerts:
        by_mode_tf = _group_hits(all_hits)
        for i, ((mode, timeframe), details) in enumerate(by_mode_tf.items(), start=1):
            msg = _format_section(now_ist, i, mode, timeframe, details)
            if send_telegram_message(msg, parse_mode="HTML"):
                messages_sent += 1

    return {
        "ran": True, "hits": len(all_hits), "messages_sent": messages_sent,
        "buckets": {"15min": run_15min, "hourly": run_hourly, "daily": run_daily},
    }
