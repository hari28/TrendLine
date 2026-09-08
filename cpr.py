"""Central Pivot Range (CPR) and Narrow CPR screening.

CPR is computed from one completed daily session's High/Low/Close and
projects three levels for the *next* session:
    Pivot = (High + Low + Close) / 3
    BC    = (High + Low) / 2
    TC    = 2 * Pivot - BC
Width% = |TC - BC| / Pivot * 100 -- how tight the band is. CPR traders read
a narrow band as compressed volatility and a higher-probability breakout
day; a wide band suggests a range-bound day instead.

This always uses the most recently completed daily bar from the app's
existing EOD-cached price feed (screener.load_frame on the "1D" timeframe)
-- there's no live intraday LTP/day-high/day-low tracking here, matching
how the rest of the app already works off cached daily data rather than a
streaming quote feed.
"""
import pandas as pd
from screener import load_frame

WIDTH_NARROW_MAX = 0.5
WIDTH_NORMAL_MAX = 1.0


def compute_cpr(high: float, low: float, close: float) -> tuple[float, float, float, float]:
    """Return (pivot, tc, bc, width_pct) for one session's H/L/C.

    TC (Top Central) and BC (Bottom Central) are *positional* labels, not the
    raw formula outputs -- when Close sits in the lower half of the day's
    range, the raw "2*Pivot - BC" term comes out below the raw BC term, so
    the two must be swapped to keep TC >= BC. Every real CPR calculator does
    this; skipping it silently reverses TC/BC on any day Close < (H+L)/2."""
    pivot = (high + low + close) / 3.0
    bc_raw = (high + low) / 2.0
    tc_raw = 2 * pivot - bc_raw
    tc, bc = max(tc_raw, bc_raw), min(tc_raw, bc_raw)
    width_pct = abs(tc - bc) / pivot * 100.0 if pivot else float("nan")
    return pivot, tc, bc, width_pct


def width_category(width_pct: float) -> str:
    if pd.isna(width_pct):
        return "Unknown"
    if width_pct <= WIDTH_NARROW_MAX:
        return "Narrow"
    if width_pct <= WIDTH_NORMAL_MAX:
        return "Normal"
    return "Wide"


def _basis_row(frame: pd.DataFrame) -> pd.Series:
    """The most recent *fully completed* session. Today's own cached row can be an
    unfinished placeholder (Open=High=Low=Close, just today's last traded price
    echoed) whenever the feed hasn't finalized the day's bar yet -- even after
    market close -- so it's always excluded, not just during market hours."""
    today = pd.Timestamp.now().normalize()
    prior = frame[frame.index < today]
    return prior.iloc[-1] if not prior.empty else frame.iloc[-1]


def scan_symbol_cpr(symbol: str, segment: str, force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    if frame is None or frame.empty:
        return None

    last = _basis_row(frame)
    high, low, close = float(last["High"]), float(last["Low"]), float(last["Close"])
    pivot, tc, bc, width_pct = compute_cpr(high, low, close)

    return {
        "Symbol": symbol,
        "Segment": segment,
        "PrevClose": close,
        "PrevHigh": high,
        "PrevLow": low,
        "Range": high - low,
        "Pivot": pivot,
        "TC": tc,
        "BC": bc,
        "WidthPct": width_pct,
        "Category": width_category(width_pct),
        "AsOf": last.name,
    }


def scan_universe_cpr(symbols: pd.DataFrame, progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    """symbols: DataFrame with columns Symbol, Segment. progress_cb(i, total, symbol) optional.

    Adds IsStale/LatestAsOf columns: Yahoo Finance occasionally lags a session's bar for a
    chunk of symbols while having it for others (verified: not a local cache issue -- querying
    Yahoo directly shows the same gap), so any row whose basis session is older than the most
    RECENT session seen anywhere in this scan is flagged. Using the max (not the most common)
    date as the reference is deliberate: if most of the universe is stuck on a stale date and
    only a few symbols have the latest one, the stale majority should still be flagged, not the
    fresh minority."""
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_cpr(r.Symbol, r.Segment, force_refresh=force_refresh)
        if result:
            rows.append(result)
    df = pd.DataFrame(rows)
    if not df.empty:
        latest = df["AsOf"].max()
        df["LatestAsOf"] = latest
        df["IsStale"] = df["AsOf"] < latest
    return df
