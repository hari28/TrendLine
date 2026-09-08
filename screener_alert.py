"""Ad-hoc "alert me on this scan" support for the Screener tab.

Unlike universe_digest.py (a fixed full-universe scan across every mode) or
watchlist.py (per-symbol tracking), this lets the user pick ANY scan
configuration in the Screener sidebar -- any universe (including Indices),
scan mode, timeframe, and settings -- click "Run scan", and have that exact
configuration re-run and posted to Telegram every 15 minutes (via the same
check_watchlist.py background job) until market close that day.

Only one active scan is tracked at a time: the most recent "Run scan" click
replaces whatever was there before. And unlike the digest (which only alerts
on newly-qualifying symbols), this always sends the FULL current snapshot
each cycle -- the user asked for that explicitly, since they want to see the
complete picture every 15 minutes, not just deltas.
"""
import html
import json
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

from constituents import get_all_symbols, INDEX_UNIVERSE_LABEL
from screener import (scan_universe, scan_universe_cross, scan_universe_volume, scan_universe_pattern,
                       apply_band, apply_band_below)
from telegram_bot import send_telegram_message

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CONFIG_PATH = os.path.join(DATA_DIR, "active_scan_config.json")

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def set_active_scan(config: dict) -> None:
    """Persist this scan configuration so the background job keeps re-running
    it every 15 minutes for the rest of today's trading session."""
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = dict(config)
    payload["date"] = datetime.now(IST).strftime("%Y-%m-%d")
    tmp_path = f"{CONFIG_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, CONFIG_PATH)


def get_active_scan() -> dict | None:
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _run_scan(config: dict, symbols_df: pd.DataFrame) -> pd.DataFrame:
    """Re-runs this config's scan mode fresh (used by the 15-min background
    cycle -- the interactive app instead passes its own just-computed results
    into _qualifying_lines directly, to avoid scanning twice)."""
    mode = config["scan_mode"]
    timeframe = config["timeframe"]
    ma_type = config["ma_type"]

    if mode in ("Above 200 MA", "Below 200 MA"):
        return scan_universe(symbols_df, timeframe, ma_type, force_refresh=False)
    if mode.startswith("Golden Cross"):
        return scan_universe_cross(symbols_df, timeframe, ma_type, config["lookback"], force_refresh=False)
    if mode.startswith("Unusual Volume"):
        return scan_universe_volume(symbols_df, timeframe, config["avg_period"], config["spike_multiple"],
                                     force_refresh=False)
    return scan_universe_pattern(symbols_df, timeframe, config["pattern_types"], config["pattern_lookback"],
                                  config["pole_min_move_pct"], force_refresh=False)


def qualifying_hits(config: dict, results: pd.DataFrame) -> list[dict]:
    """Same "what counts as a hit" rules as universe_digest.py's
    _qualifying_symbols, applied to this config's own params instead of the
    digest's fixed ones. Returns one dict per hit: {symbol, segment, close,
    detail} -- kept structured (not pre-joined into a string) so
    _format_message can lay each one out as its own labeled, multi-line block.
    Public: also used by the Telegram Alerts tab to preview what a manual
    send would contain before the user clicks the button."""
    if results.empty:
        return []
    mode = config["scan_mode"]
    ma_type = config["ma_type"]
    min_volume = config.get("min_volume", 0)
    results = results[results["Volume"] >= min_volume]

    if mode == "Above 200 MA":
        results = apply_band(results, config["band_low"], config["band_high"])
        hits = results[results["Status"] == "In band"]
        return [{"symbol": r.Symbol, "segment": r.Segment, "close": r.Close,
                  "detail": f"In band — {r.PctAbove:.2f}% above {ma_type}200",
                  "ma_value": f"{ma_type}200: {getattr(r, f'{ma_type}200'):.2f}"} for r in hits.itertuples()]

    if mode == "Below 200 MA":
        results = apply_band_below(results, config["band_low"], config["band_high"])
        hits = results[results["Status"] == "In band"]
        return [{"symbol": r.Symbol, "segment": r.Segment, "close": r.Close,
                  "detail": f"Short — {-r.PctAbove:.2f}% below {ma_type}200",
                  "ma_value": f"{ma_type}200: {getattr(r, f'{ma_type}200'):.2f}"} for r in hits.itertuples()]

    if mode.startswith("Golden Cross"):
        hits = results[results["CrossType"].isin(["Golden Cross", "Death Cross"])]
        return [{"symbol": r.Symbol, "segment": r.Segment, "close": r.Close, "detail": r.CrossType,
                  "ma_value": f"{ma_type}50: {getattr(r, f'{ma_type}50'):.2f} / "
                               f"{ma_type}200: {getattr(r, f'{ma_type}200'):.2f}"} for r in hits.itertuples()]

    if mode.startswith("Unusual Volume"):
        hits = results[results["Activity"].isin(["Unusual Buying", "Unusual Selling", "Volume Spike (Flat)"])]
        return [{"symbol": r.Symbol, "segment": r.Segment, "close": r.Close, "detail": r.Activity,
                  "ma_value": "-"} for r in hits.itertuples()]

    return [{"symbol": r.Symbol, "segment": r.Segment, "close": r.Close,
              "detail": f"{r.Pattern} ({r.Direction})", "ma_value": "-"} for r in results.itertuples()]


def _format_message(now_ist: datetime, config: dict, hits: list[dict]) -> str:
    header_lines = [
        f"📅 {now_ist:%d-%b-%Y %H:%M} IST",
        f"🔎 Screener Alert — {html.escape(config['scan_mode'])}",
        f"🌐 Universe: {html.escape(config['universe_choice'])}",
        f"⏱ Timeframe: {html.escape(config['timeframe_label'])} | MA: {html.escape(config['ma_type'])}",
    ]
    header = "\n".join(header_lines)

    if not hits:
        return f"{header}\n\nNo symbols currently match this scan's conditions."

    # A single-universe scan already names its universe in the header, so
    # repeating it per row is only useful for "All", where each hit can come
    # from a different cap segment.
    show_segment_suffix = len(config["segments"]) > 1
    blocks = []
    for h in hits:
        symbol = html.escape(h["symbol"])
        if h["segment"] == INDEX_UNIVERSE_LABEL:
            kind_line = f"📊 <b>Index: {symbol}</b>"
        elif show_segment_suffix:
            kind_line = f"📈 <b>Stock: {symbol}</b> ({html.escape(h['segment'])})"
        else:
            kind_line = f"📈 <b>Stock: {symbol}</b>"
        detail = html.escape(h["detail"])
        close = f"{h['close']:,.2f}"
        lines = [kind_line, detail]
        if h.get("ma_value") and h["ma_value"] != "-":
            lines.append(html.escape(h["ma_value"]))
        lines.append(f"Close: {close}")
        blocks.append("\n".join(lines))

    count_line = f"{len(hits)} match{'es' if len(hits) != 1 else ''}:"
    body = "\n\n".join(blocks)
    return f"{header}\n\n{count_line}\n\n{body}"


def send_snapshot_now(config: dict, results: pd.DataFrame | None = None) -> bool:
    """Sends one Telegram message with the full current snapshot for `config`.
    If `results` isn't given (the 15-min background path), runs the scan
    fresh first."""
    if results is None:
        symbols_df = get_all_symbols(config["segments"], force_refresh=False)[["Symbol", "Segment"]]
        results = _run_scan(config, symbols_df)
    hits = qualifying_hits(config, results)
    msg = _format_message(datetime.now(IST), config, hits)
    return send_telegram_message(msg, parse_mode="HTML")


def run_cycle(now_ist: datetime) -> dict:
    """Called every 15 min by check_watchlist.py during trading hours. Only
    acts if a scan was activated today and we're inside market hours --
    otherwise this is a no-op (which is how it naturally stops at close and
    stays dormant until the user runs a new scan on a later day)."""
    config = get_active_scan()
    if not config:
        return {"ran": False, "reason": "no active scan"}
    if config.get("date") != now_ist.strftime("%Y-%m-%d"):
        return {"ran": False, "reason": "active scan is from a previous day"}
    if not (MARKET_OPEN <= now_ist.time() <= MARKET_CLOSE):
        return {"ran": False, "reason": "outside market hours"}

    sent = send_snapshot_now(config)
    return {"ran": True, "sent": sent}
