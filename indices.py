"""Static list of major NSE indices, mapped to their yfinance tickers.

Unlike the Nifty 100/Midcap 150/Smallcap 250 stock universes, index levels
aren't published as a downloadable constituent CSV, and Yahoo doesn't follow
the plain ".NS" convention for indices (some use "^XXXX", others use
"XXXX.NS") -- so this is a fixed, hand-verified list rather than a fetched
one. Every ticker below was checked live against yfinance and cross-checked
against a real quote before being added.
"""
import pandas as pd

INDEX_YF_TICKERS = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY FIN SERVICE": "NIFTY_FIN_SERVICE.NS",
    "NIFTY MIDCAP 50": "^NSEMDCP50",
    "NIFTY NEXT 50": "^NSMIDCP",
    "NIFTY MIDCAP 150": "NIFTYMIDCAP150.NS",
    "NIFTY 500": "^CRSLDX",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY PVT BANK": "NIFTY_PVT_BANK.NS",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRA": "^CNXINFRA",
}
# NIFTY SMLCAP 50 / NIFTY OIL & GAS / NIFTY CONSR DURABLE / NIFTY HEALTHCARE were
# tried and dropped -- Yahoo only serves a live quote for these (period="max"
# fails, and even wide periods return 0-1 daily bars), so MA/crossover scans
# can never produce a result for them.


def get_index_universe() -> pd.DataFrame:
    """Same column shape as constituents.get_constituents(), so it plugs
    into the existing Symbol/Segment scan pipeline unchanged."""
    rows = [
        {"Company Name": symbol, "Industry": "Index", "Symbol": symbol,
         "Series": "INDEX", "ISIN Code": ""}
        for symbol in INDEX_YF_TICKERS
    ]
    return pd.DataFrame(rows)
