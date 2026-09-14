"""Two scan modes, both on ONE chosen timeframe (1H, 4H, 1D, 1W, 1M):

- scan_universe / apply_band: is the latest close just above the 200-period
  EMA/SMA?
- scan_universe_cross / bucket_cross: has the 50-period EMA/SMA recently
  crossed the 200-period EMA/SMA (golden cross = up, death cross = down)?
"""
import pandas as pd
from indicators import sma, ema, rsi, build_frame, TIMEFRAMES
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
    ma_series = _ma(close, ma_type, PERIOD)
    ma_value = ma_series.iloc[-1]
    last_close = close.iloc[-1]

    pct_above = float("nan") if pd.isna(ma_value) else (last_close - ma_value) / ma_value * 100.0

    # "Fresh cross" = the previous bar closed on the other side of the MA and this
    # bar closed on this side -- not just "currently above/below by some %", which
    # stays true for as long as the stock lingers there. False (not NaN) whenever
    # either bar's MA/close isn't available, so it never silently counts as a cross.
    prev_close = close.iloc[-2] if len(close) > 1 else float("nan")
    prev_ma_value = ma_series.iloc[-2] if len(ma_series) > 1 else float("nan")
    have_prev = not pd.isna(prev_close) and not pd.isna(prev_ma_value) and not pd.isna(ma_value)
    crossed_above = bool(have_prev and prev_close <= prev_ma_value and last_close > ma_value)
    crossed_below = bool(have_prev and prev_close >= prev_ma_value and last_close < ma_value)

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
        "CrossedAbove": crossed_above,
        "CrossedBelow": crossed_below,
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
    """Split rows into In Band / Above Band / Below Band / Insufficient history.
    "In band" requires BOTH being within the %-band AND a fresh cross (previous
    bar closed at/below the MA, this bar closed above it) -- a stock that's simply
    been sitting inside the band for a while, with no crossover today, falls
    through to "Above band" instead so it never fires a fresh alert."""
    df = df.copy()

    def _status(r):
        if not r["HasEnoughHistory"] or pd.isna(r["PctAbove"]):
            return "Insufficient history"
        if r["PctAbove"] < band_low:
            return "Below band"
        if r["PctAbove"] <= band_high and r.get("CrossedAbove", False):
            return "In band"
        return "Above band"

    df["Status"] = df.apply(_status, axis=1)
    return df


def apply_band_below(df: pd.DataFrame, band_low: float, band_high: float) -> pd.DataFrame:
    """Mirror of apply_band for short setups: band_low/band_high are read as % BELOW
    the 200 MA instead of % above -- e.g. (0, 3) means "just broke down, 0-3% under the
    MA", the short-side equivalent of apply_band's "just crossed above" long entry zone.
    Same fresh-cross requirement as apply_band: "In band" needs the previous bar to
    have closed at/above the MA and this bar to have closed below it, not just
    "currently sitting a little under it"."""
    df = df.copy()

    def _status(r):
        if not r["HasEnoughHistory"] or pd.isna(r["PctAbove"]):
            return "Insufficient history"
        pct_below = -r["PctAbove"]
        if pct_below < band_low:
            return "Not below MA"
        if pct_below <= band_high and r.get("CrossedBelow", False):
            return "In band"
        return "Extended below"

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


# ---------------------------------------------------------------------------
# Mode 5: RSI range -- flags stocks whose RSI(length) currently sits between
# a lower and upper bound (default period 14, bounds 30/70 -- i.e. "neither
# overbought nor oversold"). The same 14/30/70 values are reused by
# scalping_backtest.py's entry filter, so "avoid an already-extended RSI" is
# applied consistently everywhere RSI gates an entry in this app.
# ---------------------------------------------------------------------------

def scan_symbol_rsi(symbol: str, segment: str, timeframe: str, rsi_period: int, rsi_min: float,
                     rsi_max: float, force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None or len(frame) < rsi_period + 2:
        return None

    close = frame["Close"]
    rsi_value = rsi(close, rsi_period).iloc[-1]

    if pd.isna(rsi_value):
        status = "Insufficient history"
    elif rsi_value < rsi_min:
        status = "Oversold"
    elif rsi_value > rsi_max:
        status = "Overbought"
    else:
        status = "In Range"

    return {
        "Symbol": symbol,
        "Segment": segment,
        "Close": close.iloc[-1],
        "AsOf": frame.index[-1],
        "Volume": frame["Volume"].iloc[-1],
        "RSI": rsi_value,
        "Status": status,
    }


def scan_universe_rsi(symbols: pd.DataFrame, timeframe: str, rsi_period: int, rsi_min: float, rsi_max: float,
                       progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_rsi(r.Symbol, r.Segment, timeframe, rsi_period, rsi_min, rsi_max,
                                  force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mode 6: Aged All-Time-High breakout -- flags stocks that have just CONFIRMED
# a breakout (min_confirm_bars consecutive closes above the level, not a
# single-candle cross that can be a whipsaw/fakeout) over an all-time high
# that is itself at least min_age_years old -- a genuinely multi-year-dormant
# level finally giving way, not just any ordinary new high. The classic chart
# shape this looks for: a big high set years ago, a long multi-year base/
# decline under it, then price finally reclaiming and holding above it.
# ---------------------------------------------------------------------------

def scan_symbol_aged_ath(symbol: str, segment: str, timeframe: str, min_age_years: float,
                          min_confirm_bars: int = 2, force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None or len(frame) < max(30, min_confirm_bars + 5):
        return None

    n = min_confirm_bars
    high, close, volume = frame["High"], frame["Close"], frame["Volume"]
    last_close = float(close.iloc[-1])
    last_date = frame.index[-1]

    # The ATH is computed over every bar EXCLUDING the last N (the
    # confirmation window itself), so the breakout attempt can never inflate
    # its own breakout level -- "age" naturally comes out ~0 for a stock
    # that's simply making routine fresh highs, without needing a separate
    # "exclude recent window" rule.
    prior_high = high.iloc[:-n]
    if prior_high.empty:
        return {"Symbol": symbol, "Segment": segment, "Close": last_close, "AsOf": last_date,
                "Volume": float(volume.iloc[-1]), "ATH": float("nan"), "ATHDate": pd.NaT,
                "AgeYears": float("nan"), "PctFromATH": float("nan"), "Status": "Insufficient history"}

    ath_idx = prior_high.idxmax()  # first (oldest) occurrence if tied -- the ORIGINAL time it was set
    ath_value = float(prior_high.loc[ath_idx])
    age_years = (last_date - ath_idx).days / 365.25

    recent_closes = close.iloc[-n:]
    all_confirmed = bool((recent_closes > ath_value).all())  # every one of the last N candles closed above
    currently_above = last_close > ath_value

    # "Freshly confirmed" = the Nth consecutive close-above just completed on
    # THIS bar -- the bar immediately before the N-bar confirmation window
    # was NOT above the level, so this fires exactly once (the day
    # confirmation completes), not on every later day the stock stays up.
    if len(close) > n:
        bar_before_window = float(close.iloc[-(n + 1)])
        freshly_confirmed = all_confirmed and bar_before_window <= ath_value
    else:
        freshly_confirmed = all_confirmed  # not enough history before the window to check "freshness"

    if age_years < min_age_years:
        status = "ATH Too Recent"
    elif freshly_confirmed:
        status = "Fresh Aged Breakout"
    elif all_confirmed:
        status = "Above Aged ATH"
    elif currently_above:
        status = "Breakout Forming"  # above the level, but hasn't held for N candles yet -- not confirmed
    else:
        status = "Below ATH"

    return {
        "Symbol": symbol,
        "Segment": segment,
        "Close": last_close,
        "AsOf": last_date,
        "Volume": float(volume.iloc[-1]),
        "ATH": ath_value,
        "ATHDate": ath_idx,
        "AgeYears": round(age_years, 1),
        "PctFromATH": round((last_close - ath_value) / ath_value * 100.0, 2),
        "Status": status,
    }


def scan_universe_aged_ath(symbols: pd.DataFrame, timeframe: str, min_age_years: float,
                            min_confirm_bars: int = 2, progress_cb=None,
                            force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_aged_ath(r.Symbol, r.Segment, timeframe, min_age_years,
                                       min_confirm_bars=min_confirm_bars, force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)
