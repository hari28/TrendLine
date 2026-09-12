"""EMA / SMA helpers and timeframe construction (1MIN, 5MIN, 15MIN, 1H, 4H, 1D, 1W, 1M)."""
import pandas as pd

_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, Wilder-smoothed (an EMA with alpha=1/period, the
    classic ATR convention) -- a volatility-adjusted distance, used to size
    stop-losses off each stock's own recent range instead of a flat %."""
    high, low, prev_close = frame["High"], frame["Low"], frame["Close"].shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def resample_ohlc(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a daily OHLCV DataFrame (DatetimeIndex) to weekly ('W-FRI') or monthly ('ME')."""
    return daily.resample(rule).agg(_AGG).dropna(subset=["Close"])


def resample_intraday(intraday: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate finer intraday bars into coarser blocks (e.g. 60m -> 4h), anchored to
    each trading day's open (9:15 IST) so blocks never straddle the overnight/weekend gap."""
    if intraday.empty:
        return intraday
    chunks = []
    for _, day_rows in intraday.groupby(intraday.index.date):
        chunks.append(day_rows.resample(rule, origin="start").agg(_AGG).dropna(subset=["Close"]))
    return pd.concat(chunks) if chunks else intraday.iloc[0:0]


# timeframe key -> spec. "fetch_interval" is the yfinance interval string to fetch when
# kind == "intraday" (None when kind == "daily", which uses full daily history instead).
# "resample" (optional) is applied AFTER fetching the base interval/daily data.
TIMEFRAMES = {
    "1MIN":  {"label": "1 Minute",  "kind": "intraday", "fetch_interval": "1m",  "resample": None},
    "5MIN":  {"label": "5 Minute",  "kind": "intraday", "fetch_interval": "5m",  "resample": None},
    "15MIN": {"label": "15 Minute", "kind": "intraday", "fetch_interval": "15m", "resample": None},
    "1H":    {"label": "1 Hour",    "kind": "intraday", "fetch_interval": "60m", "resample": None},
    "4H":    {"label": "4 Hour",    "kind": "intraday", "fetch_interval": "60m", "resample": "4h"},
    "1D":    {"label": "1 Day",     "kind": "daily",    "fetch_interval": None,  "resample": None},
    "1W":    {"label": "1 Week",    "kind": "daily",    "fetch_interval": None,  "resample": "W-FRI"},
    "1M":    {"label": "1 Month",   "kind": "daily",    "fetch_interval": None,  "resample": "ME"},
}


def build_frame(timeframe: str, base: pd.DataFrame | None) -> pd.DataFrame:
    """base: the raw fetched frame -- full daily history for a "daily" timeframe, or
    intraday history at that timeframe's fetch_interval for an "intraday" one."""
    if base is None:
        return pd.DataFrame(columns=list(_AGG))
    resample = TIMEFRAMES[timeframe]["resample"]
    if resample == "4h":
        return resample_intraday(base, "4h")
    if resample in ("W-FRI", "ME"):
        return resample_ohlc(base, resample)
    return base
