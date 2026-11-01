"""Fab4: a four-indicator trend + momentum system (Supertrend, RSI, Pivot R1/S1, Bollinger Band).

Source: a podcast interview with an options trader (user-supplied transcript, 2026-11-01). The rules,
as stated in the transcript, for a LONG (call-buying) setup, all evaluated on the bar's CLOSE:

  1. Trend      -- Close above Supertrend (10, 3)            -> "the escalator is going up"
  2. Momentum   -- RSI(14) above 70                          -> buy strength, not "overbought"
  3. Structure  -- Close above the Traditional pivot R1      -> at least one resistance broken
  4. Strength   -- Close above the upper Bollinger Band (20, 2 SD) -> a "super candle" (~5% of bars)

  Exit: Supertrend flips (close crosses back below it). The Supertrend line is also the initial
  stop-loss and the trailing stop -- no fixed target. The transcript calls it an intraday system
  (3/5-min, also works on 15m/30m/1h), so intraday positions are squared off at the session's last bar.

SHORT (put-buying) side: the transcript does NOT give the put rules (they're behind a paid course).
This module offers the mirror image -- Close below Supertrend, RSI below 30, Close below S1, Close
below the lower band -- as a clearly-labelled assumption, off by default in the UI.

Pivots: TradingView "Pivot Points Standard", Type = Traditional, Pivots Timeframe = Auto, which
anchors on the PREVIOUS period: daily pivots for intraday <= 15 min, weekly for 30m-4h, monthly for
1D, yearly for 1W/1M.

All indicator conventions match TradingView: Supertrend on HL2 with Wilder (RMA) ATR; RSI Wilder;
Bollinger SMA basis with population standard deviation.

Backtest scope: signals are on the UNDERLYING (stock / index) price, entry and exit at bar close.
It does not price the option itself (the transcript's strike rule -- a premium of ~1% of the
underlying -- is shown in the UI as a guide only). Costs are a flat round-trip % you choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import rsi as _rsi, resample_intraday

DEFAULTS = {
    "st_period": 10, "st_mult": 3.0,
    "rsi_period": 14, "rsi_long": 70.0, "rsi_short": 30.0,
    "bb_period": 20, "bb_sd": 2.0,
}

# Ablation ladder, as built up step by step in the transcript's backtest
LADDER = [
    ("Supertrend only", ("trend",)),
    ("+ RSI", ("trend", "rsi")),
    ("+ R1 / S1 pivot", ("trend", "rsi", "pivot")),
    ("+ Bollinger (Fab4)", ("trend", "rsi", "pivot", "bb")),
]
ALL_CONDS = ("trend", "rsi", "pivot", "bb")


# --------------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------------

def supertrend(frame: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """TradingView ta.supertrend(): returns columns ST (the line) and Dir (+1 uptrend, -1 downtrend)."""
    high, low, close = frame["High"].to_numpy(float), frame["Low"].to_numpy(float), frame["Close"].to_numpy(float)
    n = len(close)
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()

    hl2 = (high + low) / 2
    up_basic, dn_basic = hl2 - multiplier * atr, hl2 + multiplier * atr
    lower, upper = np.full(n, np.nan), np.full(n, np.nan)
    st, direction = np.full(n, np.nan), np.zeros(n, dtype=int)

    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if i == 0 or np.isnan(lower[i - 1]):
            lower[i], upper[i] = up_basic[i], dn_basic[i]
            direction[i] = -1  # TV starts in "down" until price proves otherwise
        else:
            lower[i] = up_basic[i] if (up_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]) else lower[i - 1]
            upper[i] = dn_basic[i] if (dn_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]) else upper[i - 1]
            if st[i - 1] == upper[i - 1]:
                direction[i] = 1 if close[i] > upper[i] else -1
            else:
                direction[i] = -1 if close[i] < lower[i] else 1
        st[i] = lower[i] if direction[i] == 1 else upper[i]

    return pd.DataFrame({"ST": st, "Dir": direction}, index=frame.index)


def bollinger(close: pd.Series, period: int = 20, sd: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)  # TradingView uses population SD
    return pd.DataFrame({"BB_mid": mid, "BB_up": mid + sd * dev, "BB_lo": mid - sd * dev}, index=close.index)


def pivot_anchor(timeframe: str) -> str:
    """TradingView 'Auto' pivots timeframe for a chart timeframe key (see indicators.TIMEFRAMES)."""
    if timeframe in ("1MIN", "3MIN", "5MIN", "15MIN"):
        return "D"
    if timeframe in ("30MIN", "1H", "4H"):
        return "W"
    if timeframe == "1D":
        return "M"
    return "Y"


def pivot_levels(frame: pd.DataFrame, anchor: str = "D") -> pd.DataFrame:
    """Traditional floor pivots for each bar, from the PREVIOUS anchor period's High/Low/Close."""
    idx = frame.index
    if anchor == "D":
        key = idx.normalize()
    elif anchor == "W":
        key = idx.to_period("W-FRI").start_time
    elif anchor == "M":
        key = idx.to_period("M").start_time
    else:
        key = idx.to_period("Y").start_time
    key = pd.Index(key)
    g = frame.groupby(key).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last")).shift(1)
    p = (g["H"] + g["L"] + g["C"]) / 3
    rng = g["H"] - g["L"]
    lv = pd.DataFrame({
        "P": p, "R1": 2 * p - g["L"], "S1": 2 * p - g["H"],
        "R2": p + rng, "S2": p - rng,
        "R3": g["H"] + 2 * (p - g["L"]), "S3": g["L"] - 2 * (g["H"] - p),
    })
    out = lv.reindex(key)
    out.index = idx
    return out


def compute(frame: pd.DataFrame, timeframe: str = "5MIN", params: dict | None = None) -> pd.DataFrame:
    """All Fab4 indicator columns plus each condition as a boolean, long (L_*) and short (S_*)."""
    p = {**DEFAULTS, **(params or {})}
    close = frame["Close"]
    out = pd.concat([
        supertrend(frame, int(p["st_period"]), float(p["st_mult"])),
        bollinger(close, int(p["bb_period"]), float(p["bb_sd"])),
        pivot_levels(frame, pivot_anchor(timeframe)),
    ], axis=1)
    out["RSI"] = _rsi(close, int(p["rsi_period"]))

    out["L_trend"] = (out["Dir"] == 1) & (close > out["ST"])
    out["L_rsi"] = out["RSI"] > p["rsi_long"]
    out["L_pivot"] = close > out["R1"]
    out["L_bb"] = close > out["BB_up"]
    out["S_trend"] = (out["Dir"] == -1) & (close < out["ST"])
    out["S_rsi"] = out["RSI"] < p["rsi_short"]
    out["S_pivot"] = close < out["S1"]
    out["S_bb"] = close < out["BB_lo"]
    # "Below R1" contrast run from the transcript: same setup but the pivot rule inverted
    out["L_pivot_inv"] = close <= out["R1"]
    out["S_pivot_inv"] = close >= out["S1"]
    return out


def conditions_mask(ind: pd.DataFrame, side: str, conds: tuple, invert_pivot: bool = False) -> pd.Series:
    pre = "L_" if side == "Long" else "S_"
    mask = pd.Series(True, index=ind.index)
    for c in conds:
        col = f"{pre}{c}_inv" if (c == "pivot" and invert_pivot) else f"{pre}{c}"
        mask &= ind[col].fillna(False).astype(bool)
    return mask


# --------------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------------

def to_ist_naive(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    if getattr(frame.index, "tz", None) is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return frame


def is_intraday(frame: pd.DataFrame) -> bool:
    if frame is None or len(frame) < 3:
        return False
    return bool((pd.Series(frame.index).diff().dropna() < pd.Timedelta(hours=20)).mean() > 0.5)


def kite_frame(symbol: str, minutes: int = 5) -> pd.DataFrame | None:
    """Cached Kite 5-min snapshot (data/kite_5min), optionally resampled to 15/30/60 min."""
    from oliver_ma_cross_backtest import load_kite_5min
    df = load_kite_5min(symbol)
    if df is None or df.empty:
        return None
    df = to_ist_naive(df)[["Open", "High", "Low", "Close", "Volume"]]
    if minutes != 5:
        df = resample_intraday(df, f"{minutes}min")
    return df


KITE_TF = {5: "5MIN", 15: "15MIN", 30: "30MIN", 60: "1H"}


# --------------------------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------------------------

def simulate(frame: pd.DataFrame, ind: pd.DataFrame, side: str = "Long", conds: tuple = ALL_CONDS,
             invert_pivot: bool = False, square_off: bool = True, cost_pct: float = 0.0,
             symbol: str = "") -> list[dict]:
    """One position at a time. Enter at the close of the first bar where every condition holds;
    exit at the close of the bar where Supertrend flips against the trade, or (intraday,
    square_off) at the close of the session's last bar. No entry on a session's last bar."""
    entry_mask = conditions_mask(ind, side, conds, invert_pivot).to_numpy()
    close = frame["Close"].to_numpy(float)
    st_dir = ind["Dir"].to_numpy()
    st_line = ind["ST"].to_numpy(float)
    times = frame.index
    n = len(frame)

    intraday = square_off and is_intraday(frame)
    if intraday:
        day = times.normalize()
        last_of_day = np.r_[day[1:] != day[:-1], False]
        # The final bar only counts as a session close if it's at the usual closing time
        # (otherwise the data simply ends mid-session and the trade is still open).
        closes = times[:-1][last_of_day[:-1]]
        if len(closes):
            usual_close = pd.Series(closes.time).mode().iloc[0]
            last_of_day[-1] = times[-1].time() >= usual_close
    else:
        last_of_day = np.zeros(n, dtype=bool)

    want = 1 if side == "Long" else -1
    trades, pos = [], None
    for i in range(n):
        if pos is not None:
            reason = None
            if st_dir[i] == -want:
                reason = "Supertrend flip"
            elif last_of_day[i]:
                reason = "Square-off (EOD)"
            if reason:
                trades.append(_close_trade(pos, i, close[i], times[i], reason, side, cost_pct, symbol))
                pos = None
            continue
        if entry_mask[i] and not last_of_day[i]:
            pos = {"i": i, "price": close[i], "time": times[i], "stop": st_line[i]}

    if pos is not None:
        trades.append(_close_trade(pos, n - 1, close[-1], times[-1], "Open (end of data)", side, cost_pct, symbol))
    return trades


def _close_trade(pos, i, price, ts, reason, side, cost_pct, symbol) -> dict:
    sign = 1 if side == "Long" else -1
    pts = (price - pos["price"]) * sign
    ret = pts / pos["price"] * 100 - cost_pct
    return {
        "Symbol": symbol, "Direction": side, "EntryTime": pos["time"], "EntryPrice": round(pos["price"], 2),
        "InitialStop": round(float(pos["stop"]), 2) if pd.notna(pos["stop"]) else None,
        "ExitTime": ts, "ExitPrice": round(float(price), 2), "ExitReason": reason,
        "Bars": i - pos["i"], "Points": round(float(pts), 2), "ReturnPct": round(float(ret), 3),
        "Premium~1%": round(pos["price"] * 0.01, 1),
    }


def backtest(frames: dict, timeframe: str, params: dict | None = None, sides=("Long",),
             conds: tuple = ALL_CONDS, invert_pivot: bool = False, square_off: bool = True,
             cost_pct: float = 0.0) -> pd.DataFrame:
    """frames: {symbol: OHLCV frame}. Returns every trade across symbols and sides."""
    rows = []
    for sym, fr in frames.items():
        if fr is None or len(fr) < 40:
            continue
        ind = compute(fr, timeframe, params)
        for side in sides:
            rows += simulate(fr, ind, side, conds, invert_pivot, square_off, cost_pct, sym)
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    empty = {"Trades": 0, "Win rate %": np.nan, "Total return %": np.nan, "Total points": np.nan,
             "Avg win %": np.nan, "Avg loss %": np.nan, "Payoff (W/L)": np.nan, "Profit factor": np.nan,
             "Max DD %": np.nan, "Top-5 share of profit %": np.nan}
    if trades is None or trades.empty:
        return empty
    closed = trades[trades["ExitReason"] != "Open (end of data)"]
    if closed.empty:
        return empty
    r = closed.sort_values("ExitTime")["ReturnPct"]
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    curve = r.cumsum()
    dd = (curve - curve.cummax()).min()
    total = r.sum()
    top5 = r.nlargest(5).sum()
    return {
        "Trades": int(len(r)),
        "Win rate %": round(len(wins) / len(r) * 100, 1),
        "Total return %": round(total, 2),
        "Total points": round(closed["Points"].sum(), 1),
        "Avg win %": round(wins.mean(), 3) if len(wins) else np.nan,
        "Avg loss %": round(losses.mean(), 3) if len(losses) else np.nan,
        "Payoff (W/L)": round(wins.mean() / -losses.mean(), 2) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "Profit factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else np.nan,
        "Max DD %": round(float(dd), 2),
        "Top-5 share of profit %": round(top5 / gross_win * 100, 1) if gross_win > 0 else np.nan,
    }


def ladder(frames: dict, timeframe: str, params: dict | None = None, sides=("Long",),
           square_off: bool = True, cost_pct: float = 0.0) -> pd.DataFrame:
    """The transcript's step-by-step build-up: Supertrend -> +RSI -> +R1 -> +Bollinger, plus the
    'same setup but below R1' contrast. One summary row per step."""
    computed = {s: compute(f, timeframe, params) for s, f in frames.items() if f is not None and len(f) >= 40}
    steps = [(name, conds, False) for name, conds in LADDER]
    steps.append(("Fab4 but BELOW R1 / above S1 (contrast)", ALL_CONDS, True))
    out = []
    for name, conds, inv in steps:
        rows = []
        for sym, ind in computed.items():
            for side in sides:
                rows += simulate(frames[sym], ind, side, conds, inv, square_off, cost_pct, sym)
        out.append({"Step": name, **summarize(pd.DataFrame(rows))})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
# Screener
# --------------------------------------------------------------------------------------------

def scan_symbol(symbol: str, segment: str, frame: pd.DataFrame | None, timeframe: str, lookback: int,
                params: dict | None = None, sides=("Long",), square_off: bool = True) -> dict | None:
    if frame is None or len(frame) < 40:
        return None
    ind = compute(frame, timeframe, params)
    last = ind.iloc[-1]
    close = float(frame["Close"].iloc[-1])
    trades = []
    for side in sides:
        trades += simulate(frame, ind, side, ALL_CONDS, False, square_off, 0.0, symbol)
    n = len(frame)
    recent = [t for t in trades if n - 1 - frame.index.get_loc(t["EntryTime"]) < lookback]
    tick = lambda b: "✅" if bool(b) else "·"
    base = {
        "Symbol": symbol, "Segment": segment, "Close": round(close, 2), "AsOf": frame.index[-1],
        "Trend": "Up" if last["Dir"] == 1 else ("Down" if last["Dir"] == -1 else ""),
        "Supertrend": round(float(last["ST"]), 2) if pd.notna(last["ST"]) else None,
        "RSI": round(float(last["RSI"]), 1) if pd.notna(last["RSI"]) else None,
        "R1": round(float(last["R1"]), 2) if pd.notna(last["R1"]) else None,
        "S1": round(float(last["S1"]), 2) if pd.notna(last["S1"]) else None,
        "Long now": "".join(tick(last[f"L_{c}"]) for c in ALL_CONDS),
        "Short now": "".join(tick(last[f"S_{c}"]) for c in ALL_CONDS),
    }
    if not recent:
        return {**base, "Signal": "None", "BarsAgo": np.nan, "Status": "", "Entry": None, "P&L %": None}
    t = max(recent, key=lambda t: t["EntryTime"])
    return {**base, "Signal": t["Direction"], "BarsAgo": n - 1 - frame.index.get_loc(t["EntryTime"]),
            "Status": "Open" if t["ExitReason"] == "Open (end of data)" else f"Exited ({t['ExitReason']})",
            "Entry": t["EntryPrice"], "P&L %": t["ReturnPct"]}
