"""Market Structure: Higher-High/Higher-Low (uptrend) vs. Lower-High/Lower-Low
(downtrend) swing analysis, plus Character Change (CHoCH) detection -- the
classic price-action / Smart Money Concepts framework:

    Uptrend   = a sequence of Higher Highs (HH) + Higher Lows (HL)
    Downtrend = a sequence of Lower Highs (LH) + Lower Lows (LL)
    Character Change = the structure's floor/ceiling breaks:
      - In an uptrend, a Close below the last confirmed Higher Low means the
        HH/HL pattern just failed -- character changes to Downtrend.
      - In a downtrend, a Close above the last confirmed Lower High means the
        LH/LL pattern just failed -- character changes to Uptrend.

A "Trending" signal additionally requires price to be above (uptrend) or
below (downtrend) BOTH the 10 and 20-period MA -- structure alone can be
noisy; the MA filter is this scan's confirmation that the trend is live, not
just a lagging pattern of old swing points.

Swing points are fractal pivots (SWING_ORDER bars lower/higher on each side,
same technique as patterns.py's _swing_points) -- so the most recent
SWING_ORDER bars can never be confirmed as swings yet; there's an inherent
lag on any fractal-based indicator.
"""
import pandas as pd

from screener import load_frame
from indicators import sma, ema

SWING_ORDER = 2
MA_FAST, MA_SLOW = 10, 20


def _ma(close: pd.Series, ma_type: str, period: int) -> pd.Series:
    return ema(close, period) if ma_type == "EMA" else sma(close, period)


def find_swings(high: pd.Series, low: pd.Series, order: int) -> tuple[pd.Series, pd.Series]:
    """Returns (is_swing_high, is_swing_low) boolean Series aligned to the index."""
    n = len(high)
    swing_high = pd.Series(False, index=high.index)
    swing_low = pd.Series(False, index=low.index)
    h, l = high.to_numpy(), low.to_numpy()
    for i in range(order, n - order):
        window_h = h[i - order:i + order + 1]
        if h[i] == window_h.max() and (window_h == h[i]).sum() == 1:
            swing_high.iloc[i] = True
        window_l = l[i - order:i + order + 1]
        if l[i] == window_l.min() and (window_l == l[i]).sum() == 1:
            swing_low.iloc[i] = True
    return swing_high, swing_low


def analyze_structure(frame: pd.DataFrame, order: int = SWING_ORDER) -> dict:
    """Walks the frame bar-by-bar, tracking swing labels and Close-crossing-level
    Character Change events. Returns:
      - trend: "Up" / "Down" / None (not enough swings yet for a baseline)
      - last_swing_high / last_swing_low: most recent confirmed swing prices
      - last_high_label / last_low_label: "HH"/"LH" and "HL"/"LL" (None until
        a second swing of that type has been seen)
      - choch_bars_ago: bars since the most recent Character Change event
        (None if none has happened in this history)
      - choch_type: "Up" or "Down" -- the direction of that most recent flip
      - swing_points: chronological list of {"index": int, "price": float,
        "kind": "high"/"low", "label": str|None} for charting

    `trend` is STATELESS -- recomputed fresh from only the two most recent
    confirmed swing labels (HH+HL = Up, LH+LL = Down, anything else = None /
    "no clear structure right now"). It does NOT stick to "Up" just because a
    trend was confirmed earlier and nothing has technically broken it yet --
    the moment the next swing label stops matching, `trend` reflects that
    honestly instead of persisting a stale read.

    Character Change is recorded only when this clean classification actually
    FLIPS from one confirmed direction to the other (Up -> Down or vice
    versa) -- e.g. Up, then a Lower Low prints (temporarily Mixed while only
    one side has broken), then a Lower High confirms it -> CHoCH to Down is
    recorded at that confirming swing. A single swing breaking one side while
    the other hasn't confirmed the reversal yet shows as "Mixed", not a firm
    Character Change -- this avoids flagging a one-swing wobble (e.g. a brief
    spike through an old swing high) as a full trend reversal."""
    swing_high, swing_low = find_swings(frame["High"], frame["Low"], order)

    def _classify(high_label, low_label):
        if high_label == "HH" and low_label == "HL":
            return "Up"
        if high_label == "LH" and low_label == "LL":
            return "Down"
        return None

    last_high_price = last_low_price = None
    last_high_label = last_low_label = None
    established_trend = None  # last CONFIRMED clean Up/Down classification, for detecting a flip
    choch_index = None
    choch_type = None
    swing_points = []

    for i in range(len(frame)):
        touched = False
        if swing_high.iloc[i]:
            h = frame["High"].iloc[i]
            last_high_label = None if last_high_price is None else ("HH" if h > last_high_price else "LH")
            last_high_price = h
            swing_points.append({"index": i, "price": h, "kind": "high", "label": last_high_label})
            touched = True
        if swing_low.iloc[i]:
            lo = frame["Low"].iloc[i]
            last_low_label = None if last_low_price is None else ("HL" if lo > last_low_price else "LL")
            last_low_price = lo
            swing_points.append({"index": i, "price": lo, "kind": "low", "label": last_low_label})
            touched = True

        if touched:
            new_trend = _classify(last_high_label, last_low_label)
            if new_trend is not None and established_trend is not None and new_trend != established_trend:
                choch_index, choch_type = i, new_trend
            if new_trend is not None:
                established_trend = new_trend

    trend = _classify(last_high_label, last_low_label)
    choch_bars_ago = None if choch_index is None else (len(frame) - 1 - choch_index)

    return {
        "trend": trend,
        "last_swing_high": last_high_price,
        "last_swing_low": last_low_price,
        "last_high_label": last_high_label,
        "last_low_label": last_low_label,
        "choch_bars_ago": choch_bars_ago,
        "choch_type": choch_type,
        "swing_points": swing_points,
    }


def trend_history(frame: pd.DataFrame, order: int = SWING_ORDER) -> pd.Series:
    """Same one-pass swing walk as analyze_structure, but returns the classified
    trend ("Up"/"Down"/None, forward-filled between swing confirmations) at
    EVERY bar instead of only the final snapshot. analyze_structure only needs
    "what does structure look like today" (the Market Structure tab); a
    backtest needs "what did it look like as of each historical day", so this
    is a separate function rather than changing analyze_structure's contract."""
    swing_high, swing_low = find_swings(frame["High"], frame["Low"], order)

    def _classify(high_label, low_label):
        if high_label == "HH" and low_label == "HL":
            return "Up"
        if high_label == "LH" and low_label == "LL":
            return "Down"
        return None

    last_high_price = last_low_price = None
    last_high_label = last_low_label = None
    current = None
    trend = [None] * len(frame)

    for i in range(len(frame)):
        if swing_high.iloc[i]:
            h = frame["High"].iloc[i]
            last_high_label = None if last_high_price is None else ("HH" if h > last_high_price else "LH")
            last_high_price = h
        if swing_low.iloc[i]:
            lo = frame["Low"].iloc[i]
            last_low_label = None if last_low_price is None else ("HL" if lo > last_low_price else "LL")
            last_low_price = lo
        classified = _classify(last_high_label, last_low_label)
        if classified is not None:
            current = classified
        trend[i] = current

    return pd.Series(trend, index=frame.index)


def scan_symbol_structure(symbol: str, segment: str, timeframe: str, ma_type: str, order: int,
                           choch_lookback: int, force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None:
        return None

    close = frame["Close"]
    min_bars_needed = order * 2 + MA_SLOW + 5
    if len(close) < min_bars_needed:
        return {
            "Symbol": symbol, "Segment": segment, "Close": close.iloc[-1], "AsOf": frame.index[-1],
            "Volume": frame["Volume"].iloc[-1], f"{ma_type}{MA_FAST}": float("nan"),
            f"{ma_type}{MA_SLOW}": float("nan"), "Structure": "-", "Signal": "Insufficient history",
            "BarsAvailable": len(close),
        }

    result = analyze_structure(frame, order)
    trend = result["trend"]

    ma_fast_val = _ma(close, ma_type, MA_FAST).iloc[-1]
    ma_slow_val = _ma(close, ma_type, MA_SLOW).iloc[-1]
    last_close = close.iloc[-1]
    above_mas = last_close > ma_fast_val > 0 and last_close > ma_slow_val if pd.notna(ma_fast_val) and pd.notna(ma_slow_val) else False
    below_mas = last_close < ma_fast_val and last_close < ma_slow_val if pd.notna(ma_fast_val) and pd.notna(ma_slow_val) else False

    is_recent_choch = result["choch_bars_ago"] is not None and result["choch_bars_ago"] <= choch_lookback

    if is_recent_choch:
        signal = f"Character Change → {result['choch_type']}trend"
    elif trend == "Up" and above_mas:
        signal = "Uptrend"
    elif trend == "Up":
        signal = f"Uptrend (below {ma_type}{MA_FAST}/{MA_SLOW})"
    elif trend == "Down" and below_mas:
        signal = "Downtrend"
    elif trend == "Down":
        signal = f"Downtrend (above {ma_type}{MA_FAST}/{MA_SLOW})"
    else:
        signal = "No clear structure"

    structure_label = "-"
    if result["last_high_label"] and result["last_low_label"]:
        structure_label = f"{result['last_high_label']}/{result['last_low_label']}"

    return {
        "Symbol": symbol,
        "Segment": segment,
        "Close": last_close,
        "AsOf": frame.index[-1],
        "Volume": frame["Volume"].iloc[-1],
        f"{ma_type}{MA_FAST}": ma_fast_val,
        f"{ma_type}{MA_SLOW}": ma_slow_val,
        "Structure": structure_label,
        "Signal": signal,
        "BarsAvailable": len(close),
    }


def scan_universe_structure(symbols: pd.DataFrame, timeframe: str, ma_type: str, order: int, choch_lookback: int,
                             progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols)
    for i, r in enumerate(symbols.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_structure(r.Symbol, r.Segment, timeframe, ma_type, order, choch_lookback,
                                        force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)
