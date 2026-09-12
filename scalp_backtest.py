"""Backtest for the 15-min/5-min multi-timeframe scalping strategy discussed
with the user: 15-min CPR breakout sets the directional bias, 5-min 9-EMA
pullback-and-reclaim times the entry.

Data source: TrendLine's own Yahoo-based intraday fetcher (screener.load_frame),
same as cpr_ema_backtest_intraday.py -- runs locally, no Kite MCP context cost.
Yahoo retains ~60 days of 5-min/15-min history, so this covers roughly the
last 60 trading days, not a multi-year window -- same statistical caveat as
the earlier 15-min run (expect a modest trade count, treat as directional,
not definitive).

Rules:
- Trend/bias filter (15-min): bias is LONG while the most recently completed
  15-min bar closed above that day's CPR Top (TC); SHORT while it closed
  below that day's CPR Bottom (BC). CPR is the usual DAILY indicator
  (computed once per day from the prior day's daily H/L/C) -- same
  cpr.compute_cpr used throughout this app. No bias (no new entries) when
  neither condition holds.
- Entry (5-min), only in the direction of the current 15-min bias:
  LONG when a 5-min candle closes back above its 9 EMA after the previous
  5-min candle closed at/below it, and the entry candle itself is bullish
  (close > open) -- a pullback-and-reclaim. SHORT is the mirror image.
- Session window: entries only allowed 09:15-11:15 IST (highest-volume
  window), per the discussed strategy.
- SL: the lowest low (long) / highest high (short) of the prior 3 five-min
  bars including the entry bar.
- Exit: fixed 1.5R target, OR an earlier trail-exit if a 5-min candle closes
  back through the 9 EMA against the trade direction, OR the SL -- whichever
  comes first, checked bar-by-bar on 5-min candles.
- Max 5 new entries per symbol per day (per the discussed overtrading cap).
- One open trade per symbol at a time.
"""
import pandas as pd

from cpr import compute_cpr
from indicators import ema
from screener import load_frame

FAST_EMA = 9
TARGET_R = 1.5
MAX_TRADES_PER_DAY = 5
SESSION_START, SESSION_END = "09:15", "11:15"


def _daily_cpr_by_date(symbol: str) -> dict:
    daily = load_frame(symbol, "1D")
    if daily is None or len(daily) < 2:
        return {}
    prev_high, prev_low, prev_close = daily["High"].shift(1), daily["Low"].shift(1), daily["Close"].shift(1)
    out = {}
    for date, h, l, c in zip(daily.index, prev_high, prev_low, prev_close):
        if pd.isna(h) or pd.isna(l) or pd.isna(c):
            continue
        _, tc, bc, _ = compute_cpr(h, l, c)
        out[date.date()] = (tc, bc)
    return out


def _bias_series_15m(symbol: str, cpr_by_date: dict) -> pd.Series:
    """Returns a Series indexed by 15-min bar timestamp -> 'Long'/'Short'/None,
    representing the bias established by THAT bar's close (effective from the
    next bar onward when merged into the 5-min series)."""
    df15 = load_frame(symbol, "15MIN")
    if df15 is None:
        return pd.Series(dtype=object)
    bias = []
    for ts, close in df15["Close"].items():
        levels = cpr_by_date.get(ts.date())
        if levels is None:
            bias.append(None)
            continue
        tc, bc = levels
        if close > tc:
            bias.append("Long")
        elif close < bc:
            bias.append("Short")
        else:
            bias.append(None)
    return pd.Series(bias, index=df15.index)


def backtest_symbol(symbol: str, segment: str, force_refresh: bool = False) -> list[dict]:
    cpr_by_date = _daily_cpr_by_date(symbol)
    if not cpr_by_date:
        return []

    bias_15m = _bias_series_15m(symbol, cpr_by_date)
    if bias_15m.empty:
        return []

    df5 = load_frame(symbol, "5MIN", force_refresh=force_refresh)
    if df5 is None or len(df5) < FAST_EMA + 5:
        return []

    df = df5.copy()
    df["EMA9"] = ema(df["Close"], FAST_EMA)
    # merge_asof needs sorted DatetimeIndex-as-column
    bias_df = bias_15m.rename("Bias").reset_index().rename(columns={bias_15m.index.name or "index": "Time"})
    df = df.reset_index().rename(columns={df.index.name or "index": "Time"})
    df["Time"] = pd.to_datetime(df["Time"]).astype("datetime64[ns]")
    bias_df["Time"] = pd.to_datetime(bias_df["Time"]).astype("datetime64[ns]")
    df = pd.merge_asof(df.sort_values("Time"), bias_df.sort_values("Time"), on="Time", direction="backward")
    df = df.set_index("Time")
    df = df.dropna(subset=["EMA9"])
    if df.empty:
        return []

    times = df.index.time
    session_mask = (pd.Series(times, index=df.index).astype(str) >= SESSION_START) & \
                   (pd.Series(times, index=df.index).astype(str) <= SESSION_END)

    trades = []
    in_trade_until = None
    trades_today = {}

    idx = df.index
    closes, opens, highs, lows, ema9s, bias_col = (
        df["Close"], df["Open"], df["High"], df["Low"], df["EMA9"], df["Bias"]
    )

    for i in range(1, len(idx)):
        ts = idx[i]
        if in_trade_until is not None and ts <= in_trade_until:
            continue
        if not session_mask.loc[ts]:
            continue

        day = ts.date()
        if trades_today.get(day, 0) >= MAX_TRADES_PER_DAY:
            continue

        bias = bias_col.iloc[i]
        if bias not in ("Long", "Short"):
            continue

        prev_close, prev_ema = closes.iloc[i - 1], ema9s.iloc[i - 1]
        close, open_, ema_now = closes.iloc[i], opens.iloc[i], ema9s.iloc[i]

        if bias == "Long":
            signal = (prev_close <= prev_ema) and (close > ema_now) and (close > open_)
        else:
            signal = (prev_close >= prev_ema) and (close < ema_now) and (close < open_)
        if not signal:
            continue

        lookback_lo = lows.iloc[max(0, i - 2):i + 1].min()
        lookback_hi = highs.iloc[max(0, i - 2):i + 1].max()
        entry_price = float(close)
        entry_time = ts
        direction = bias

        if direction == "Long":
            sl_price = float(lookback_lo)
            if sl_price >= entry_price:
                continue
            r = entry_price - sl_price
            target = entry_price + TARGET_R * r
        else:
            sl_price = float(lookback_hi)
            if sl_price <= entry_price:
                continue
            r = sl_price - entry_price
            target = entry_price - TARGET_R * r

        exit_time, exit_reason, exit_price = None, None, None
        for j in range(i + 1, len(idx)):
            ts2 = idx[j]
            low2, high2, close2, ema2 = lows.iloc[j], highs.iloc[j], closes.iloc[j], ema9s.iloc[j]

            if direction == "Long":
                hit_sl = low2 <= sl_price
                hit_target = high2 >= target
                trail_exit = close2 < ema2
            else:
                hit_sl = high2 >= sl_price
                hit_target = low2 <= target
                trail_exit = close2 > ema2

            if hit_sl:
                exit_time, exit_reason, exit_price = ts2, "Stop-loss", sl_price
                break
            if hit_target:
                exit_time, exit_reason, exit_price = ts2, "Target (1.5R)", target
                break
            if trail_exit:
                exit_time, exit_reason, exit_price = ts2, "Trail exit (9 EMA)", float(close2)
                break

        if exit_time is None:
            exit_time = idx[-1]
            exit_price = float(closes.iloc[-1])
            exit_reason = "Open (end of data)"

        ret_pct = (exit_price - entry_price) / entry_price * 100.0 * (1 if direction == "Long" else -1)

        trades.append({
            "Symbol": symbol,
            "Segment": segment,
            "Direction": direction,
            "EntryTime": entry_time,
            "EntryPrice": round(entry_price, 2),
            "SLPrice": round(sl_price, 2),
            "Target": round(target, 2),
            "ExitTime": exit_time,
            "ExitReason": exit_reason,
            "ExitPrice": round(exit_price, 2),
            "ReturnPct": round(ret_pct, 2),
        })
        in_trade_until = exit_time
        trades_today[day] = trades_today.get(day, 0) + 1

    return trades


def run_backtest(symbols: list[tuple[str, str]], progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    all_trades = []
    total = len(symbols)
    for i, (symbol, segment) in enumerate(symbols):
        if progress_cb:
            progress_cb(i, total, symbol)
        all_trades.extend(backtest_symbol(symbol, segment, force_refresh=force_refresh))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values("EntryTime").reset_index(drop=True)
    return trades


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_trades": 0, "win_rate_pct": float("nan"), "avg_return_pct": float("nan"),
                "avg_win_pct": float("nan"), "avg_loss_pct": float("nan"), "max_drawdown_pct": float("nan")}
    wins = trades[trades["ReturnPct"] > 0]
    losses = trades[trades["ReturnPct"] <= 0]
    ordered = trades.sort_values("EntryTime")
    equity = (1 + ordered["ReturnPct"] / 100).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100
    return {
        "total_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "avg_return_pct": round(trades["ReturnPct"].mean(), 2),
        "avg_win_pct": round(wins["ReturnPct"].mean(), 2) if len(wins) else float("nan"),
        "avg_loss_pct": round(losses["ReturnPct"].mean(), 2) if len(losses) else float("nan"),
        "final_multiple": round(float(equity.iloc[-1]), 3),
        "peak_multiple": round(float(equity.max()), 3),
        "max_drawdown_pct": round(float(drawdown.min()), 2) if not drawdown.empty else float("nan"),
    }
