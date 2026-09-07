"""Two scan modes, both on ONE chosen timeframe (1H, 4H, 1D, 1W, 1M):

- scan_universe / apply_band: is the latest close just above the 200-period
  EMA/SMA?
- scan_universe_cross / bucket_cross: has the 50-period EMA/SMA recently
  crossed the 200-period EMA/SMA (golden cross = up, death cross = down)?
"""
import pandas as pd
from indicators import sma, ema, build_frame, TIMEFRAMES
from data_fetcher import fetch_daily_history, fetch_intraday_history
from patterns import detect_triangle, detect_channel, detect_flag_pole

PERIOD = 200
FAST_PERIOD = 50
SLOW_PERIOD = 200


def _ma(close: pd.Series, ma_type: str, period: int) -> pd.Series:
    return ema(close, period) if ma_type == "EMA" else sma(close, period)


def load_frame(symbol: str, timeframe: str, force_refresh: bool = False) -> pd.DataFrame | None:
    spec = TIMEFRAMES[timeframe]
    if spec["kind"] == "daily":
        base = fetch_daily_history(symbol, force_refresh=force_refresh)
    else:
        base = fetch_intraday_history(symbol, interval=spec["fetch_interval"], force_refresh=force_refresh)
    frame = build_frame(timeframe, base)
    if frame is None or frame.empty or len(frame) < 20:
        return None
    return frame


# ---------------------------------------------------------------------------
# Mode 1: just above the 200 MA
# ---------------------------------------------------------------------------

def scan_symbol(symbol: str, segment: str, timeframe: str, ma_type: str,
                 force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None:
        return None

    close = frame["Close"]
    has_enough_history = len(close) >= PERIOD
    ma_value = _ma(close, ma_type, PERIOD).iloc[-1]
    last_close = close.iloc[-1]

    pct_above = float("nan") if pd.isna(ma_value) else (last_close - ma_value) / ma_value * 100.0

    return {
        "Symbol": symbol,
        "Segment": segment,
        "Close": last_close,
        "AsOf": frame.index[-1],
        "Volume": frame["Volume"].iloc[-1],
        f"{ma_type}200": ma_value,
        "PctAbove": pct_above,
        "HasEnoughHistory": has_enough_history,
        "BarsAvailable": len(close),
    }


def scan_universe(symbols: pd.DataFrame, timeframe: str, ma_type: str, progress_cb=None,
                   force_refresh: bool = False) -> pd.DataFrame:
    """symbols: DataFrame with columns Symbol, Segment. progress_cb(i, total, symbol) optional."""
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol(r.Symbol, r.Segment, timeframe, ma_type, force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


def apply_band(df: pd.DataFrame, band_low: float, band_high: float) -> pd.DataFrame:
    """Split rows into In Band / Above Band / Below Band / Insufficient history."""
    df = df.copy()

    def _status(r):
        if not r["HasEnoughHistory"] or pd.isna(r["PctAbove"]):
            return "Insufficient history"
        if r["PctAbove"] < band_low:
            return "Below band"
        if r["PctAbove"] <= band_high:
            return "In band"
        return "Above band"

    df["Status"] = df.apply(_status, axis=1)
    return df


# ---------------------------------------------------------------------------
# Mode 2: golden cross / death cross (50 MA vs 200 MA)
# ---------------------------------------------------------------------------

def _cross_type(close: pd.Series, ma_type: str, lookback: int) -> tuple[str, float, float]:
    """Return (cross_type, fast_value, slow_value). cross_type is one of
    'Golden Cross', 'Death Cross', 'No recent cross', or 'Insufficient history'."""
    if len(close) < SLOW_PERIOD:
        return "Insufficient history", float("nan"), float("nan")

    fast = _ma(close, ma_type, FAST_PERIOD)
    slow = _ma(close, ma_type, SLOW_PERIOD)
    diff = (fast - slow).dropna()
    if diff.empty:
        return "Insufficient history", float("nan"), float("nan")

    fast_val, slow_val = fast.iloc[-1], slow.iloc[-1]
    window = diff.iloc[-(lookback + 1):] if len(diff) > lookback else diff
    vals = window.values

    golden_event = any(vals[i - 1] <= 0 and vals[i] > 0 for i in range(1, len(vals)))
    death_event = any(vals[i - 1] >= 0 and vals[i] < 0 for i in range(1, len(vals)))
    currently_bullish = fast_val > slow_val

    if golden_event and currently_bullish:
        return "Golden Cross", fast_val, slow_val
    if death_event and not currently_bullish:
        return "Death Cross", fast_val, slow_val
    return "No recent cross", fast_val, slow_val


def scan_symbol_cross(symbol: str, segment: str, timeframe: str, ma_type: str, lookback: int,
                       force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None:
        return None

    close = frame["Close"]
    cross_type, fast_val, slow_val = _cross_type(close, ma_type, lookback)

    return {
        "Symbol": symbol,
        "Segment": segment,
        "Close": close.iloc[-1],
        "AsOf": frame.index[-1],
        "Volume": frame["Volume"].iloc[-1],
        f"{ma_type}50": fast_val,
        f"{ma_type}200": slow_val,
        "CrossType": cross_type,
        "BarsAvailable": len(close),
    }


def scan_universe_cross(symbols: pd.DataFrame, timeframe: str, ma_type: str, lookback: int,
                         progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_cross(r.Symbol, r.Segment, timeframe, ma_type, lookback, force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mode 3: unusual volume (a volume spike vs. the recent average, direction
# inferred from the price move on that same bar) -- the closest free,
# per-stock proxy for "big buying or selling" activity. This is NOT the same
# as FII/DII activity, which NSE only discloses as a market-wide aggregate.
# ---------------------------------------------------------------------------

def scan_symbol_volume(symbol: str, segment: str, timeframe: str, avg_period: int,
                        spike_multiple: float, min_price_move_pct: float = 0.2,
                        force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None or len(frame) < avg_period + 2:
        return None

    close = frame["Close"]
    volume = frame["Volume"]
    # baseline excludes the current bar so a spike doesn't inflate its own average
    baseline = volume.shift(1).rolling(window=avg_period, min_periods=avg_period).mean()
    avg_volume = baseline.iloc[-1]
    current_volume = volume.iloc[-1]
    has_enough_history = not pd.isna(avg_volume) and avg_volume > 0

    if not has_enough_history:
        return {
            "Symbol": symbol, "Segment": segment, "Close": close.iloc[-1], "AsOf": frame.index[-1],
            "Volume": current_volume, "AvgVolume": float("nan"), "VolumeRatio": float("nan"),
            "PriceChangePct": float("nan"), "Activity": "Insufficient history",
        }

    volume_ratio = current_volume / avg_volume
    price_change_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100.0 if len(close) > 1 else float("nan")

    if volume_ratio < spike_multiple:
        activity = "Normal"
    elif pd.isna(price_change_pct):
        activity = "Volume Spike (Flat)"
    elif price_change_pct >= min_price_move_pct:
        activity = "Unusual Buying"
    elif price_change_pct <= -min_price_move_pct:
        activity = "Unusual Selling"
    else:
        activity = "Volume Spike (Flat)"

    return {
        "Symbol": symbol,
        "Segment": segment,
        "Close": close.iloc[-1],
        "AsOf": frame.index[-1],
        "Volume": current_volume,
        "AvgVolume": avg_volume,
        "VolumeRatio": volume_ratio,
        "PriceChangePct": price_change_pct,
        "Activity": activity,
    }


def scan_universe_volume(symbols: pd.DataFrame, timeframe: str, avg_period: int, spike_multiple: float,
                          progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_volume(r.Symbol, r.Segment, timeframe, avg_period, spike_multiple,
                                     force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mode 4: chart pattern breakouts (Triangle / Channel / Flag & Pole) --
# heuristic, best-effort detection. See patterns.py for the caveats. Unlike
# the other modes, only actual hits are returned -- there's no meaningful
# "no pattern" row to tabulate for hundreds of stocks.
# ---------------------------------------------------------------------------

def scan_symbol_pattern(symbol: str, segment: str, timeframe: str, pattern_types: list,
                         lookback: int, pole_min_move_pct: float, force_refresh: bool = False) -> list:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None:
        return []

    hits = []
    for ptype in pattern_types:
        if ptype == "Triangle":
            r = detect_triangle(frame, lookback=lookback)
        elif ptype == "Channel":
            r = detect_channel(frame, lookback=lookback)
        elif ptype == "Flag & Pole":
            r = detect_flag_pole(frame, pole_min_move_pct=pole_min_move_pct)
        else:
            continue
        if r.get("found"):
            hits.append({
                "Symbol": symbol,
                "Segment": segment,
                "Close": frame["Close"].iloc[-1],
                "AsOf": frame.index[-1],
                "Volume": frame["Volume"].iloc[-1],
                "Pattern": r["pattern"],
                "Direction": r["direction"],
            })
    return hits


PATTERN_COLUMNS = ["Symbol", "Segment", "Close", "AsOf", "Volume", "Pattern", "Direction"]


def scan_universe_pattern(symbols: pd.DataFrame, timeframe: str, pattern_types: list, lookback: int,
                           pole_min_move_pct: float, progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        rows.extend(scan_symbol_pattern(r.Symbol, r.Segment, timeframe, pattern_types, lookback, pole_min_move_pct,
                                         force_refresh=force_refresh))
    return pd.DataFrame(rows, columns=PATTERN_COLUMNS)
