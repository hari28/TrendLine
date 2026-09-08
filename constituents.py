"""Fetch Nifty Smallcap 250 / Midcap 150 constituent lists from NSE.

NSE publishes the official index constituent CSVs at nsearchives.nseindia.com.
These change slowly (index reshuffles happen ~twice a year), so results are
cached to disk and only re-fetched when stale.
"""
import os
import time
import requests
import pandas as pd

from indices import get_index_universe

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 1 week

INDEX_UNIVERSE_LABEL = "Indices (Nifty, Bank Nifty & Sectoral)"

INDEX_URLS = {
    "Nifty 100 (Large Cap)": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty Midcap 150": "https://nsearchives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "https://nsearchives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _cache_path(index_name: str) -> str:
    safe = index_name.lower().replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def get_constituents(index_name: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame with columns: Company Name, Industry, Symbol, Series, ISIN Code."""
    if index_name == INDEX_UNIVERSE_LABEL:
        return get_index_universe()

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(index_name)

    is_stale = True
    if os.path.exists(path) and not force_refresh:
        age = time.time() - os.path.getmtime(path)
        is_stale = age > CACHE_MAX_AGE_SECONDS

    if is_stale:
        url = INDEX_URLS[index_name]
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            if os.path.exists(path):
                # Fall back to whatever we have cached, even if stale.
                print(f"[constituents] live fetch failed ({e}), using cached copy for {index_name}")
            else:
                raise RuntimeError(
                    f"Could not download {index_name} constituents and no cache exists. "
                    f"Download manually from {url} and save as {path}"
                ) from e

    df = pd.read_csv(path)
    df["Symbol"] = df["Symbol"].str.strip()
    return df


def get_all_symbols(segments: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """Combine selected segments into one DataFrame with a 'Segment' column, de-duplicated on Symbol."""
    frames = []
    for seg in segments:
        df = get_constituents(seg, force_refresh=force_refresh)
        df = df.copy()
        df["Segment"] = seg
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
    return combined
