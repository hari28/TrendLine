"""Market-wide institutional activity signals from NSE's public feeds.

Important honesty note: NSE does not publish which specific stock an FII or
DII bought/sold — that FII/DII split only exists as a market-wide daily
aggregate (cash market, in Rs. Crore). The closest free per-stock proxy for
"unusual large buying/selling" is the Bulk/Block deals feed, which discloses
individual large trades (with the counterparty's name) but isn't officially
tagged by investor category (FII/DII/other) the way the aggregate figure is.
"""
import os
import time
import io
import requests
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
FII_DII_CACHE_MAX_AGE = 6 * 60 * 60   # 6h -- this is EOD data, published once a day
DEALS_CACHE_MAX_AGE = 60 * 60         # 1h

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _session() -> requests.Session:
    """NSE's endpoints reject requests without cookies from a prior page visit."""
    s = requests.Session()
    s.get("https://www.nseindia.com", headers=_HEADERS, timeout=15)
    return s


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def _is_fresh(path: str, max_age: int) -> bool:
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) <= max_age


def get_fii_dii_activity() -> pd.DataFrame | None:
    """Today's aggregate FII/FPI and DII cash-market buy/sell/net (Rs. Crore).
    Returns None if the feed can't be reached (NSE occasionally rate-limits)."""
    path = _cache_path("fii_dii.csv")
    if _is_fresh(path, FII_DII_CACHE_MAX_AGE):
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        s = _session()
        r = s.get("https://www.nseindia.com/api/fiidiiTradeReact", headers=_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        for col in ("buyValue", "sellValue", "netValue"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        print(f"[market_data] FII/DII fetch failed: {e}")
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return None
        return None


def _get_deals_csv(url: str, cache_name: str) -> pd.DataFrame | None:
    path = _cache_path(cache_name)
    if _is_fresh(path, DEALS_CACHE_MAX_AGE):
        try:
            return pd.read_csv(path)
        except Exception:
            pass

    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        s = _session()
        r = s.get(url, headers=_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip() for c in df.columns]
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        print(f"[market_data] deals fetch failed ({cache_name}): {e}")
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return None
        return None


def get_bulk_deals() -> pd.DataFrame | None:
    """Latest session's NSE bulk deals (>0.5% of listed shares in a single trade)."""
    return _get_deals_csv("https://nsearchives.nseindia.com/content/equities/bulk.csv", "bulk_deals.csv")


def get_block_deals() -> pd.DataFrame | None:
    """Latest session's NSE block deals (large single trades, min. 5 lakh shares or Rs 5 Cr)."""
    return _get_deals_csv("https://nsearchives.nseindia.com/content/equities/block.csv", "block_deals.csv")


def deals_for_symbols(symbols: list[str]) -> pd.DataFrame:
    """Combine bulk + block deals, filtered to the given symbol list."""
    frames = []
    bulk = get_bulk_deals()
    if bulk is not None:
        bulk = bulk.copy()
        bulk["Deal Type"] = "Bulk"
        frames.append(bulk)
    block = get_block_deals()
    if block is not None:
        block = block.copy()
        block["Deal Type"] = "Block"
        frames.append(block)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["Symbol"] = combined["Symbol"].astype(str).str.strip()
    symbol_set = set(symbols)
    return combined[combined["Symbol"].isin(symbol_set)].reset_index(drop=True)
