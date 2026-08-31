"""Best-effort chart pattern detection: Triangle, Channel, and Flag & Pole
breakouts.

These are heuristic, rule-based approximations of patterns a chartist draws
by eye. There is no single agreed-upon algorithm for any of the three --
this implements one reasonable, commonly-used definition for each, built on
swing-point trendlines. Expect false positives/negatives; treat results as
candidates to confirm visually on the chart, not certainties.
"""
import numpy as np
import pandas as pd

SWING_ORDER = 3          # bars on each side to qualify as a swing high/low
MIN_PIVOTS = 2           # minimum swing points needed to fit a trendline
FRESH_BREAKOUT_BARS = 3  # a breakout must have happened within this many bars to count as "fresh"


def _swing_points(series: pd.Series, order: int = SWING_ORDER, kind: str = "high") -> list:
    """Positional indices (0-based, into `series`) of local maxima/minima."""
    vals = series.values
    n = len(vals)
    idx = []
    for i in range(order, n - order):
        window = vals[i - order: i + order + 1]
        if kind == "high" and vals[i] == window.max() and (window == vals[i]).sum() == 1:
            idx.append(i)
        elif kind == "low" and vals[i] == window.min() and (window == vals[i]).sum() == 1:
            idx.append(i)
    return idx


def _fit_line(x: list, y: np.ndarray):
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def _line_value(fit, x):
    slope, intercept = fit
    return slope * x + intercept


def detect_triangle(frame: pd.DataFrame, lookback: int = 80) -> dict:
    """Upper trendline (swing highs) flat-or-falling, lower trendline (swing lows)
    flat-or-rising, and the gap between them narrowing -- classic triangle
    consolidation. A fresh breakout is the latest close crossing outside the
    triangle within the last few bars."""
    window = frame.tail(lookback).reset_index(drop=False)
    n = len(window)
    if n < 20:
        return {"found": False}

    high, low, close = window["High"], window["Low"], window["Close"]
    highs_idx = _swing_points(high, kind="high")
    lows_idx = _swing_points(low, kind="low")
    if len(highs_idx) < MIN_PIVOTS or len(lows_idx) < MIN_PIVOTS:
        return {"found": False}

    upper_fit = _fit_line(highs_idx, high.iloc[highs_idx].values)
    lower_fit = _fit_line(lows_idx, low.iloc[lows_idx].values)
    if upper_fit is None or lower_fit is None:
        return {"found": False}

    price_scale = close.mean()
    upper_slope, lower_slope = upper_fit[0], lower_fit[0]

    gap_start = _line_value(upper_fit, 0) - _line_value(lower_fit, 0)
    gap_end = _line_value(upper_fit, n - 1) - _line_value(lower_fit, n - 1)
    converging = gap_end < gap_start * 0.75 and gap_start > 0

    flat_or_falling_top = upper_slope <= 0.0005 * price_scale
    flat_or_rising_bottom = lower_slope >= -0.0005 * price_scale
    if not (converging and flat_or_falling_top and flat_or_rising_bottom):
        return {"found": False}

    last_close = close.iloc[-1]
    upper_now = _line_value(upper_fit, n - 1)
    lower_now = _line_value(lower_fit, n - 1)

    direction = None
    for back in range(FRESH_BREAKOUT_BARS):
        i = n - 1 - back
        c = close.iloc[i]
        u, l = _line_value(upper_fit, i), _line_value(lower_fit, i)
        if c > u:
            direction = "Bullish breakout"
        elif c < l:
            direction = "Bearish breakdown"
        if direction:
            break
    if direction is None:
        return {"found": False}

    return {
        "found": True, "pattern": "Triangle", "direction": direction,
        "last_close": last_close, "upper_now": upper_now, "lower_now": lower_now,
        "pivots": len(highs_idx) + len(lows_idx),
    }


def detect_channel(frame: pd.DataFrame, lookback: int = 80) -> dict:
    """Upper and lower trendlines roughly PARALLEL (not converging) -- an
    ascending, descending, or horizontal channel. A fresh breakout is the
    latest close crossing outside either line within the last few bars."""
    window = frame.tail(lookback).reset_index(drop=False)
    n = len(window)
    if n < 20:
        return {"found": False}

    high, low, close = window["High"], window["Low"], window["Close"]
    highs_idx = _swing_points(high, kind="high")
    lows_idx = _swing_points(low, kind="low")
    if len(highs_idx) < MIN_PIVOTS or len(lows_idx) < MIN_PIVOTS:
        return {"found": False}

    upper_fit = _fit_line(highs_idx, high.iloc[highs_idx].values)
    lower_fit = _fit_line(lows_idx, low.iloc[lows_idx].values)
    if upper_fit is None or lower_fit is None:
        return {"found": False}

    upper_slope, lower_slope = upper_fit[0], lower_fit[0]
    price_scale = close.mean()
    denom = max(abs(upper_slope), abs(lower_slope), 1e-9)
    parallel = abs(upper_slope - lower_slope) / denom < 0.5
    trending = abs(upper_slope) > 0.0003 * price_scale or abs(lower_slope) > 0.0003 * price_scale

    gap_start = _line_value(upper_fit, 0) - _line_value(lower_fit, 0)
    gap_end = _line_value(upper_fit, n - 1) - _line_value(lower_fit, n - 1)
    not_converging = gap_start > 0 and gap_end > gap_start * 0.6

    if not (parallel and not_converging):
        return {"found": False}

    last_close = close.iloc[-1]
    upper_now = _line_value(upper_fit, n - 1)
    lower_now = _line_value(lower_fit, n - 1)

    direction = None
    for back in range(FRESH_BREAKOUT_BARS):
        i = n - 1 - back
        c = close.iloc[i]
        u, l = _line_value(upper_fit, i), _line_value(lower_fit, i)
        if c > u:
            direction = "Bullish breakout"
        elif c < l:
            direction = "Bearish breakdown"
        if direction:
            break
    if direction is None:
        return {"found": False}

    slope_label = "Ascending" if lower_slope > 0.0003 * price_scale else \
                  "Descending" if upper_slope < -0.0003 * price_scale else "Horizontal"

    return {
        "found": True, "pattern": f"{slope_label} Channel", "direction": direction,
        "last_close": last_close, "upper_now": upper_now, "lower_now": lower_now,
        "pivots": len(highs_idx) + len(lows_idx), "trending": trending,
    }


def detect_flag_pole(frame: pd.DataFrame, pole_bars: int = 10, flag_bars: int = 15,
                      pole_min_move_pct: float = 8.0) -> dict:
    """A sharp directional move (the pole) followed by a short, tight
    consolidation (the flag) that drifts sideways or mildly against the pole.
    A fresh breakout is the latest close extending beyond the flag in the
    pole's original direction."""
    total = pole_bars + flag_bars
    window = frame.tail(total + 5).reset_index(drop=False)
    if len(window) < total:
        return {"found": False}

    window = window.tail(total).reset_index(drop=True)
    pole = window.iloc[:pole_bars]
    flag = window.iloc[pole_bars:]

    pole_start, pole_end = pole["Close"].iloc[0], pole["Close"].iloc[-1]
    if pole_start <= 0:
        return {"found": False}
    pole_move_pct = (pole_end - pole_start) / pole_start * 100.0

    if abs(pole_move_pct) < pole_min_move_pct:
        return {"found": False}
    bullish_pole = pole_move_pct > 0

    flag_range = flag["High"].max() - flag["Low"].min()
    pole_range = pole["High"].max() - pole["Low"].min()
    if pole_range <= 0 or flag_range > pole_range * 0.7:
        return {"found": False}  # flag isn't tight/contained enough vs. the pole's range

    flag_close_fit = _fit_line(list(range(len(flag))), flag["Close"].values)
    if flag_close_fit is None:
        return {"found": False}
    flag_slope = flag_close_fit[0]
    price_scale = flag["Close"].mean()
    # flag should drift flat or mildly against the pole, not continue exploding in the same direction
    if bullish_pole and flag_slope > 0.01 * price_scale:
        return {"found": False}
    if not bullish_pole and flag_slope < -0.01 * price_scale:
        return {"found": False}

    # fresh breakout: within the last few bars, a close exceeded the flag's
    # prior extreme (excluding itself) in the pole's original direction
    breakout = False
    for back in range(min(FRESH_BREAKOUT_BARS, len(flag) - 1)):
        i = len(flag) - 1 - back
        prior_high = flag["High"].iloc[:i].max()
        prior_low = flag["Low"].iloc[:i].min()
        c = flag["Close"].iloc[i]
        if bullish_pole and c > prior_high:
            breakout = True
            break
        if not bullish_pole and c < prior_low:
            breakout = True
            break
    if not breakout:
        return {"found": False}

    last_close = flag["Close"].iloc[-1]
    flag_high, flag_low = flag["High"].max(), flag["Low"].min()

    return {
        "found": True, "pattern": "Flag & Pole", "direction": "Bullish breakout" if bullish_pole else "Bearish breakdown",
        "last_close": last_close, "pole_move_pct": pole_move_pct,
        "flag_high": flag_high, "flag_low": flag_low,
    }


DETECTORS = {
    "Triangle": detect_triangle,
    "Channel": detect_channel,
    "Flag & Pole": detect_flag_pole,
}
