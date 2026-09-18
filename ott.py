"""Optimized Trend Tracker (OTT) by Anil Ozeksi / Kivanc Ozbilgic -- ported from the
published Pine Script v4 source (study("Optimized Trend Tracker","OTT")).

The OTT is a trailing-stop-style trend line built on top of a moving average: the MA
is offset up/down by `percent`% to form long/short "stop" levels, a direction flips
between them exactly like a SuperTrend/Chandelier stop, and the plotted OTT value is
that stop level nudged a further `percent`% beyond the MA -- then, per the original
script, plotted 2 bars behind the live MAvg/price (`OTT[2]`), which is what gives the
published indicator its smoother, less whippy look.
"""
import math

import numpy as np
import pandas as pd

from indicators import sma, ema

MA_TYPES = ["VAR", "SMA", "EMA", "WMA", "TMA", "WWMA", "ZLEMA", "TSF"]


def _wma(src: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)

    def _w(window):
        return np.dot(window, weights) / weights.sum()

    return src.rolling(length).apply(_w, raw=True)


def _tma(src: pd.Series, length: int) -> pd.Series:
    return sma(sma(src, math.ceil(length / 2)), math.floor(length / 2) + 1)


def _var_ma(src: pd.Series, length: int) -> pd.Series:
    """Pine's VAR (Chande VIDYA-style, CMO-weighted EMA) -- the script's default MA."""
    valpha = 2 / (length + 1)
    delta = src.diff()
    vud = delta.clip(lower=0).rolling(9, min_periods=1).sum()
    vdd = (-delta.clip(upper=0)).rolling(9, min_periods=1).sum()
    vcmo = ((vud - vdd) / (vud + vdd)).fillna(0.0)

    s = src.to_numpy()
    a = vcmo.abs().to_numpy() * valpha
    out = np.empty(len(s))
    prev = 0.0
    for i in range(len(s)):
        out[i] = a[i] * s[i] + (1 - a[i]) * prev
        prev = out[i]
    return pd.Series(out, index=src.index)


def _wwma(src: pd.Series, length: int) -> pd.Series:
    return src.ewm(alpha=1 / length, adjust=False).mean()


def _zlema(src: pd.Series, length: int) -> pd.Series:
    lag = length // 2 if length % 2 == 0 else (length - 1) // 2
    zx_data = src + (src - src.shift(lag))
    return zx_data.ewm(span=length, adjust=False).mean()


def _linreg_endpoint(src: pd.Series, length: int) -> pd.Series:
    """Value of the least-squares regression line, fit over the trailing `length`
    bars, evaluated at the last (most recent) bar of that window -- TradingView's
    `linreg(src, length, 0)`."""
    x = np.arange(length, dtype=float)

    def _fit(window):
        slope, intercept = np.polyfit(x, window, 1)
        return intercept + slope * (length - 1)

    return src.rolling(length).apply(_fit, raw=True)


def _tsf(src: pd.Series, length: int) -> pd.Series:
    lrc = _linreg_endpoint(src, length)
    lrs = lrc - lrc.shift(1)
    return lrc + lrs


def get_ma(src: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "SMA":
        return sma(src, length)
    if ma_type == "EMA":
        return ema(src, length)
    if ma_type == "WMA":
        return _wma(src, length)
    if ma_type == "TMA":
        return _tma(src, length)
    if ma_type == "WWMA":
        return _wwma(src, length)
    if ma_type == "ZLEMA":
        return _zlema(src, length)
    if ma_type == "TSF":
        return _tsf(src, length)
    return _var_ma(src, length)  # "VAR", and the fallback default


def calc_ott(full_frame: pd.DataFrame, length: int = 2, percent: float = 1.4,
             ma_type: str = "VAR") -> pd.DataFrame:
    """full_frame: complete OHLC history (DatetimeIndex), so the MA/stop levels are
    accurate from the first displayed bar. Returns a frame aligned to full_frame.index:
      MAvg     -- the underlying moving average ("Support Line" in the original script)
      OTT      -- the OTT trend line, already shifted 2 bars (the script's `OTT[2]`
                  plotting convention)
      trend_up -- OTT[2] > OTT[3], i.e. this OTT value rose vs. the previous bar's --
                  the script's own color-change condition
    """
    src = full_frame["Close"]
    ma = get_ma(src, length, ma_type)
    fark = ma * percent * 0.01

    ma_v = ma.to_numpy()
    long_raw = (ma - fark).to_numpy()
    short_raw = (ma + fark).to_numpy()
    n = len(ma_v)

    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)
    mt = np.full(n, np.nan)

    for i in range(n):
        prev_long = long_stop[i - 1] if i > 0 and not np.isnan(long_stop[i - 1]) else long_raw[i]
        prev_short = short_stop[i - 1] if i > 0 and not np.isnan(short_stop[i - 1]) else short_raw[i]
        long_stop[i] = max(long_raw[i], prev_long) if ma_v[i] > prev_long else long_raw[i]
        short_stop[i] = min(short_raw[i], prev_short) if ma_v[i] < prev_short else short_raw[i]

        prev_dir = direction[i - 1] if i > 0 else 1
        if prev_dir == -1 and ma_v[i] > prev_short:
            direction[i] = 1
        elif prev_dir == 1 and ma_v[i] < prev_long:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        mt[i] = long_stop[i] if direction[i] == 1 else short_stop[i]

    ott_raw = np.where(ma_v > mt, mt * (200 + percent) / 200, mt * (200 - percent) / 200)
    ott = pd.Series(ott_raw, index=full_frame.index).shift(2)

    out = pd.DataFrame({"MAvg": ma, "OTT": ott}, index=full_frame.index)
    out["trend_up"] = out["OTT"] > out["OTT"].shift(1)
    return out


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))
