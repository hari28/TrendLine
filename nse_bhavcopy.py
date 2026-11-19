"""NSE Full Bhavcopy & Security Deliverable data -> Narrow CPR source.

File: https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{DDMMYYYY}.csv
(NSE's "Full Bhavcopy and Security Deliverable data" report, trading days only,
published after market close, typically ~17:30-18:00 IST).
Downloads are cached under data/bhavcopy/. Only SERIES == EQ rows are used.
"""
import os
import time
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

IST = ZoneInfo("Asia/Kolkata")
DOWNLOAD_AFTER = dtime(18, 30)  # IST
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "bhavcopy")
URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
HOME = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,application/octet-stream,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports",
}


def _path(d) -> str:
    return os.path.join(CACHE_DIR, f"sec_bhavdata_full_{d:%d%m%Y}.csv")


def _download(d) -> bool:
    """Download the file for date d. False if NSE has no file (holiday / not yet published)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(HOME, timeout=15)  # pick up session cookies
    except Exception:
        pass
    r = s.get(URL.format(d=f"{d:%d%m%Y}"), timeout=60)
    if r.status_code != 200 or len(r.content) < 10_000 or b"SYMBOL" not in r.content[:300]:
        return False
    tmp = _path(d) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, _path(d))
    return True


def ensure_bhavcopy(d, retries: int = 1, wait: int = 60) -> bool:
    if os.path.exists(_path(d)):
        return True
    for i in range(retries):
        try:
            if _download(d):
                return True
        except Exception as e:
            print(f"[bhavcopy] download {d} failed: {e}")
        if i < retries - 1:
            time.sleep(wait)
    return False


def download_if_due(now_ist: datetime | None = None, retries: int = 1) -> dict:
    """Scheduler hook: after 18:30 IST, fetch today's file once. Weekends/holidays just miss."""
    now_ist = now_ist or datetime.now(IST)
    today = now_ist.date()
    if now_ist.time() < DOWNLOAD_AFTER or today.weekday() >= 5:
        return {"downloaded": False, "date": today, "reason": "before 18:30 IST or weekend"}
    if os.path.exists(_path(today)):
        return {"downloaded": False, "date": today, "reason": "already cached"}
    ok = ensure_bhavcopy(today, retries=retries)
    return {"downloaded": ok, "date": today, "reason": "ok" if ok else "not published / holiday"}


def latest_bhavcopy(now_ist: datetime | None = None, lookback_days: int = 10):
    """Most recent trading day's file (cached, else downloaded). Returns (date, DataFrame)."""
    now_ist = now_ist or datetime.now(IST)
    d = now_ist.date()
    for _ in range(lookback_days):
        if d.weekday() < 5 and ensure_bhavcopy(d):
            df = pd.read_csv(_path(d))
            df.columns = [c.strip() for c in df.columns]
            return d, df
        d -= timedelta(days=1)
    raise RuntimeError("No NSE Bhavcopy file could be downloaded/found in the last "
                       f"{lookback_days} days. Check network access to nsearchives.nseindia.com.")


def scan_bhavcopy_cpr(fo_only: bool = False, now_ist: datetime | None = None) -> pd.DataFrame:
    """Same columns as cpr.scan_universe_cpr. fo_only=True keeps only F&O stocks."""
    from cpr import compute_cpr, width_category
    d, df = latest_bhavcopy(now_ist)
    for c in ("SERIES", "SYMBOL"):
        df[c] = df[c].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    for c in ("HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE"])
    df = df[(df["HIGH_PRICE"] > 0) & (df["LOW_PRICE"] > 0) & (df["CLOSE_PRICE"] > 0)]
    if fo_only:
        import fo_universe
        df = df[df["SYMBOL"].isin(fo_universe.get_fo_symbols())]
    asof = pd.Timestamp(d)
    rows = []
    for r in df.itertuples(index=False):
        h, l, c = float(r.HIGH_PRICE), float(r.LOW_PRICE), float(r.CLOSE_PRICE)
        pivot, tc, bc, w = compute_cpr(h, l, c)
        rows.append({"Symbol": r.SYMBOL, "Segment": "NSE EQ", "PrevClose": c, "PrevHigh": h,
                     "PrevLow": l, "Range": h - l, "Pivot": pivot, "TC": tc, "BC": bc,
                     "WidthPct": w, "Category": width_category(w), "AsOf": asof,
                     "LatestAsOf": asof, "IsStale": False})
    return pd.DataFrame(rows)
