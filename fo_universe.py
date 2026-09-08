"""NSE's list of individual-security F&O (equity derivatives) underlyings.

Fetched from NSE's own public market-lot archive (the same file NSE
publishes for F&O lot sizes), which lists BOTH index derivatives (NIFTY,
BANKNIFTY, ...) and individual-stock derivatives in one CSV, in that order.
Only the individual-stock section is kept -- "F&O Stocks Only" means stocks
with equity derivatives available, not index derivatives.

Cached the same way constituents.py caches index lists: F&O underlyings
change slowly (NSE reviews/reshuffles the list periodically, not daily), so
a stale cache is fine for up to a week.
"""
import os
import time
import requests
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_PATH = os.path.join(CACHE_DIR, "fo_individual_securities.csv")
CACHE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 1 week

FO_MKTLOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

MARKER = "Individual Securities"


def get_fo_symbols(force_refresh: bool = False) -> set[str]:
    """Return the set of NSE trading symbols that currently have equity
    derivatives (F&O) available -- individual stocks only, not index
    derivatives like NIFTY/BANKNIFTY."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    is_stale = True
    if os.path.exists(CACHE_PATH) and not force_refresh:
        age = time.time() - os.path.getmtime(CACHE_PATH)
        is_stale = age > CACHE_MAX_AGE_SECONDS

    if is_stale:
        try:
            resp = requests.get(FO_MKTLOTS_URL, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            with open(CACHE_PATH, "wb") as f:
                f.write(resp.content)
        except Exception as e:
            if not os.path.exists(CACHE_PATH):
                raise RuntimeError(
                    f"Could not download the F&O underlying list and no cache exists. "
                    f"Download manually from {FO_MKTLOTS_URL} and save as {CACHE_PATH}"
                ) from e
            print(f"[fo_universe] live fetch failed ({e}), using cached copy")

    df = pd.read_csv(CACHE_PATH)
    df.columns = [c.strip() for c in df.columns]
    df["UNDERLYING"] = df["UNDERLYING"].astype(str).str.strip()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    marker_idx = df.index[df["UNDERLYING"].str.contains(MARKER, case=False, na=False)]
    if marker_idx.empty:
        # Format changed unexpectedly -- fail loud rather than silently return nothing/everything.
        raise RuntimeError(
            f"Couldn't find the '{MARKER}' section marker in the F&O underlying list -- "
            "NSE may have changed the file format."
        )

    stock_rows = df.iloc[marker_idx[0] + 1:]
    return set(stock_rows["SYMBOL"].tolist())
