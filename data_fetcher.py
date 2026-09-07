"""Fetch + disk-cache OHLCV history for NSE symbols via yfinance.

Two kinds of history are cached separately:
- Daily ("period=max"): used directly for the 1D timeframe, and resampled for
  1W / 1M. 200-month MAs need ~17 years of monthly bars, so we always pull
  full history rather than a fixed window.
- Intraday bars at a chosen interval, cached per interval. Yahoo enforces a
  hard cap on how much history each interval can return, and finer intervals
  go stale faster, so both the fetch period and the cache TTL vary by interval:
    1m  -> max 8 days of history (Yahoo hard limit)  -> cached 5 min
    5m  -> max 60 days of history (Yahoo hard limit)  -> cached 15 min
    15m -> max 60 days of history (Yahoo hard limit)  -> cached 30 min
    60m -> max ~729 days of history (Yahoo hard limit) -> cached 1 hour
  60m is also the base for the 4H timeframe (resampled, see indicators.py).
"""
import os
import time
import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DAILY_CACHE_MAX_AGE_SECONDS = 12 * 60 * 60   # EOD data: refetch at most twice a day

INTRADAY_PERIOD_BY_INTERVAL = {"1m": "7d", "5m": "60d", "15m": "60d", "60m": "729d"}
INTRADAY_CACHE_MAX_AGE_BY_INTERVAL = {
    "1m": 5 * 60, "5m": 15 * 60, "15m": 30 * 60, "60m": 60 * 60,
}


def _cache_path(symbol: str, suffix: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}{suffix}.csv")


def _load_cache(path: str, max_age: int, force_refresh: bool = False) -> pd.DataFrame | None:
    if force_refresh:
        return None
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > max_age:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception:
        return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["Close"])
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_daily_history(symbol: str, retries: int = 2, pause: float = 0.4,
                         force_refresh: bool = False) -> pd.DataFrame | None:
    """Full daily OHLCV history (DatetimeIndex) for NSE:SYMBOL, or None on failure."""
    path = _cache_path(symbol, "")
    cached = _load_cache(path, DAILY_CACHE_MAX_AGE_SECONDS, force_refresh=force_refresh)
    if cached is not None:
        return cached

    os.makedirs(CACHE_DIR, exist_ok=True)
    ticker = f"{symbol}.NS"
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(ticker).history(period="max", auto_adjust=True)
            if df is None or df.empty:
                return None
            df = _clean(df)
            df.to_csv(path)
            return df
        except Exception as e:
            last_err = e
            time.sleep(pause * (attempt + 1))
    print(f"[data_fetcher] failed to fetch daily {symbol}: {last_err}")
    return None


def fetch_intraday_history(symbol: str, interval: str = "60m", retries: int = 2,
                            pause: float = 0.4, force_refresh: bool = False) -> pd.DataFrame | None:
    """Intraday OHLCV bars at the given yfinance interval ("1m"/"5m"/"15m"/"60m")
    for NSE:SYMBOL, or None on failure. History length and cache TTL both depend
    on the interval -- see INTRADAY_PERIOD_BY_INTERVAL / INTRADAY_CACHE_MAX_AGE_BY_INTERVAL."""
    path = _cache_path(symbol, f"_{interval}")
    max_age = INTRADAY_CACHE_MAX_AGE_BY_INTERVAL.get(interval, 60 * 60)
    cached = _load_cache(path, max_age, force_refresh=force_refresh)
    if cached is not None:
        return cached

    os.makedirs(CACHE_DIR, exist_ok=True)
    ticker = f"{symbol}.NS"
    period = INTRADAY_PERIOD_BY_INTERVAL.get(interval, "60d")
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            if df is None or df.empty:
                return None
            df = _clean(df)
            df.to_csv(path)
            return df
        except Exception as e:
            last_err = e
            time.sleep(pause * (attempt + 1))
    print(f"[data_fetcher] failed to fetch intraday({interval}) {symbol}: {last_err}")
    return None
