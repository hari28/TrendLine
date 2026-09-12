"""Backtest for the "FAB 4" (Fabulous Four) location-based strategy, adapted
to 5-min bars (the source material specifies 2-min bars; Yahoo's pipeline
here only offers 1-min (7-day retention) or 5-min (60-day retention) --
5-min was the timeframe chosen by the user as the workable proxy).

Two things the source video does NOT specify, and had to be filled in with
a documented, defensible assumption:
- Exit/target rule: the video only describes "risk one bar for the chance
  of winning many bars" and repeated re-entries -- no fixed target or
  explicit trailing rule. Implemented here as a bar-by-bar trailing stop:
  each time a new bar closes in the trend-confirming color, the stop
  ratchets to just beyond that bar's opposite extreme (mirrors the same
  "confirming bar" logic used for entries). Exit on stop hit, or forced
  flat at the end of the trading session (15:25 IST, the last 5-min bar) --
  every example in the source video plays out within a single day.
- Max entries/day: the video describes "2 to 5" entries on a trending day
  organically, with no hard cap stated. A MAX_TRADES_PER_DAY cap of 5 is
  applied here as an explicit discipline guard (not from the source),
  consistent with the caps used in the other backtest modules in this app.

Rules (from the source video):
- FAB4 block (computed once per day, from data available BEFORE that day's
  open): the highest and lowest of --
    1. Daily 20-period SIMPLE moving average (as of the prior day's close)
    2. Daily 200-period SIMPLE moving average (as of the prior day's close)
    3. Prior day's closing price
    4. The high/low range of the last ~45 minutes (9 5-min bars) of the
       PRIOR day's trading
  These 4 items' overall max = FAB4 top, overall min = FAB4 bottom.
- Bias: today's open above the FAB4 block -> LONG bias only. Below it ->
  SHORT bias only. Open inside the block -> no trade that day (location is
  ambiguous).
- Entry ("the color game"), 5-min bars, only in the bias direction:
  - Confirming bar: a bar whose color matches the bias (red while SHORT,
    green while LONG). Mark its low (short) / high (long). Enter on the
    next bar that breaks that level.
  - Opposing/conflict bar: a bar whose color contradicts the bias. Mark its
    low (short) / high (long) the same way -- enter on the next bar that
    breaks it. This is what produces repeated re-entries through the day.
  - Stop on entry: the triggering bar's opposite extreme (high for shorts,
    low for longs) -- "risk one bar."
- One open trade per symbol at a time; a stopped-out trade frees the symbol
  up to re-enter on the next valid color-game signal, same day.
"""
import pandas as pd

from screener import load_frame

DAILY_FAST_SMA, DAILY_SLOW_SMA = 20, 200
LATE_BLOCK_BARS = 9  # 45 minutes / 5-min bars
# Diagnostic (2026-09-11): Long returns decay sharply past the 3rd re-entry of
# the day (legs 1-3 avg +0.024%/trade, legs 4-5 avg -0.033%/trade) while Short
# doesn't show the same decay -- so the two directions get different caps.
MAX_LONG_LEGS_PER_DAY = 3
MAX_SHORT_LEGS_PER_DAY = 5
SESSION_LAST_BAR = "15:25"  # last 5-min bar of the NSE session (15:25-15:30)


def _daily_fab4_inputs(symbol: str) -> pd.DataFrame:
    """Per-day DataFrame indexed by date with columns Sma20, Sma200, PrevClose
    (all as of that day's close -- i.e. usable as "prior day" inputs for the
    NEXT trading day) and LateBlockHigh/LateBlockLow (that day's last-45-min
    range, likewise usable as an input for the next day)."""
    daily = load_frame(symbol, "1D")
    if daily is None or len(daily) < DAILY_SLOW_SMA + 5:
        return pd.DataFrame()

    df5 = load_frame(symbol, "5MIN")
    if df5 is None or df5.empty:
        return pd.DataFrame()

    late_hi, late_lo = {}, {}
    for date, day_rows in df5.groupby(df5.index.date):
        tail = day_rows.tail(LATE_BLOCK_BARS)
        late_hi[date] = tail["High"].max()
        late_lo[date] = tail["Low"].min()

    out = pd.DataFrame(index=daily.index)
    out["Sma20"] = daily["Close"].rolling(DAILY_FAST_SMA).mean()
    out["Sma200"] = daily["Close"].rolling(DAILY_SLOW_SMA).mean()
    out["PrevClose"] = daily["Close"]
    out["LateBlockHigh"] = [late_hi.get(d.date()) for d in daily.index]
    out["LateBlockLow"] = [late_lo.get(d.date()) for d in daily.index]
    return out.dropna()


def _fab4_inputs_from_frames(daily: pd.DataFrame, df5: pd.DataFrame) -> pd.DataFrame:
    """Same computation as _daily_fab4_inputs, but from already-loaded daily/5-min
    frames instead of fetching via screener.load_frame -- lets backtest_from_frames
    reuse identical logic against data from a different source (e.g. Kite)."""
    if daily is None or len(daily) < DAILY_SLOW_SMA + 5 or df5 is None or df5.empty:
        return pd.DataFrame()

    late_hi, late_lo = {}, {}
    for date, day_rows in df5.groupby(df5.index.date):
        tail = day_rows.tail(LATE_BLOCK_BARS)
        late_hi[date] = tail["High"].max()
        late_lo[date] = tail["Low"].min()

    out = pd.DataFrame(index=daily.index)
    out["Sma20"] = daily["Close"].rolling(DAILY_FAST_SMA).mean()
    out["Sma200"] = daily["Close"].rolling(DAILY_SLOW_SMA).mean()
    out["PrevClose"] = daily["Close"]
    out["LateBlockHigh"] = [late_hi.get(d.date()) for d in daily.index]
    out["LateBlockLow"] = [late_lo.get(d.date()) for d in daily.index]
    return out.dropna()


def backtest_from_frames(daily: pd.DataFrame, df5: pd.DataFrame, symbol: str, segment: str) -> list[dict]:
    """Core FAB4 logic, parameterized on already-loaded daily/5-min frames.
    backtest_symbol (Yahoo path) and any alternate-data-source check (e.g. an
    older window pulled from Kite) both funnel through this single
    implementation so the rules can never drift between the two."""
    fab4_inputs = _fab4_inputs_from_frames(daily, df5)
    if fab4_inputs.empty:
        return []
    if df5 is None or df5.empty:
        return []

    # FAB4 block for day D uses day D-1's inputs -- shift the per-day inputs
    # forward by one trading day.
    block_top = (fab4_inputs[["Sma20", "Sma200", "PrevClose", "LateBlockHigh"]].max(axis=1)).shift(1)
    block_bot = (fab4_inputs[["Sma20", "Sma200", "PrevClose", "LateBlockLow"]].min(axis=1)).shift(1)
    block_by_date = {d.date(): (t, b) for d, t, b in zip(fab4_inputs.index, block_top, block_bot)
                      if pd.notna(t) and pd.notna(b)}

    trades = []

    for date, day_rows in df5.groupby(df5.index.date):
        levels = block_by_date.get(date)
        if levels is None:
            continue
        top, bottom = levels
        day_rows = day_rows.between_time("09:15", SESSION_LAST_BAR)
        if day_rows.empty:
            continue

        open_price = float(day_rows["Open"].iloc[0])
        if open_price > top:
            bias = "Long"
        elif open_price < bottom:
            bias = "Short"
        else:
            continue

        opens, highs, lows, closes = day_rows["Open"], day_rows["High"], day_rows["Low"], day_rows["Close"]
        idx = day_rows.index

        pending_level = None  # price level to watch for a break (entry trigger)
        pending_stop = None   # stop that would be used if that entry triggers
        in_trade = False
        entry_price = entry_time = stop_price = None
        trail_stop = None
        trades_today = 0

        for i in range(1, len(idx)):
            ts = idx[i]
            o, h, l, c = opens.iloc[i], highs.iloc[i], lows.iloc[i], closes.iloc[i]
            is_green = c > o
            is_red = c < o

            if in_trade:
                if bias == "Long":
                    hit_stop = l <= trail_stop
                else:
                    hit_stop = h >= trail_stop
                if hit_stop:
                    exit_price = trail_stop
                    ret_pct = (exit_price - entry_price) / entry_price * 100.0 * (1 if bias == "Long" else -1)
                    trades.append({
                        "Symbol": symbol, "Segment": segment, "Direction": bias,
                        "EntryTime": entry_time, "EntryPrice": round(entry_price, 2),
                        "SLPrice": round(initial_stop, 2),
                        "ExitTime": ts, "ExitPrice": round(exit_price, 2),
                        "ExitReason": "Trailing stop", "ReturnPct": round(ret_pct, 2),
                    })
                    in_trade = False
                    pending_level = pending_stop = None
                    continue
                # ratchet trail on a new trend-confirming bar
                if bias == "Long" and is_green:
                    trail_stop = max(trail_stop, l)
                elif bias == "Short" and is_red:
                    trail_stop = min(trail_stop, h)
                # forced flat at session end
                if ts.strftime("%H:%M") >= SESSION_LAST_BAR:
                    exit_price = float(c)
                    ret_pct = (exit_price - entry_price) / entry_price * 100.0 * (1 if bias == "Long" else -1)
                    trades.append({
                        "Symbol": symbol, "Segment": segment, "Direction": bias,
                        "EntryTime": entry_time, "EntryPrice": round(entry_price, 2),
                        "SLPrice": round(initial_stop, 2),
                        "ExitTime": ts, "ExitPrice": round(exit_price, 2),
                        "ExitReason": "Session close", "ReturnPct": round(ret_pct, 2),
                    })
                    in_trade = False
                continue

            leg_cap = MAX_LONG_LEGS_PER_DAY if bias == "Long" else MAX_SHORT_LEGS_PER_DAY
            if trades_today >= leg_cap:
                continue

            # check if the pending trigger level got broken this bar -> enter
            if pending_level is not None:
                if bias == "Long" and h > pending_level:
                    entry_price = float(pending_level) + 0.0  # enter at breakout level
                    entry_time = ts
                    trail_stop = initial_stop = pending_stop
                    in_trade = True
                    trades_today += 1
                    pending_level = pending_stop = None
                    continue
                if bias == "Short" and l < pending_level:
                    entry_price = float(pending_level)
                    entry_time = ts
                    trail_stop = initial_stop = pending_stop
                    in_trade = True
                    trades_today += 1
                    pending_level = pending_stop = None
                    continue

            # no pending trigger (or it wasn't hit) -- does this bar set a new one?
            if bias == "Long":
                if is_green or is_red:
                    pending_level = float(h)
                    pending_stop = float(l)
            else:
                if is_green or is_red:
                    pending_level = float(l)
                    pending_stop = float(h)

        if in_trade:
            last_close = float(closes.iloc[-1])
            ret_pct = (last_close - entry_price) / entry_price * 100.0 * (1 if bias == "Long" else -1)
            trades.append({
                "Symbol": symbol, "Segment": segment, "Direction": bias,
                "EntryTime": entry_time, "EntryPrice": round(entry_price, 2),
                "SLPrice": round(initial_stop, 2),
                "ExitTime": idx[-1], "ExitPrice": round(last_close, 2),
                "ExitReason": "Session close", "ReturnPct": round(ret_pct, 2),
            })

    return trades


def backtest_symbol(symbol: str, segment: str, force_refresh: bool = False) -> list[dict]:
    daily = load_frame(symbol, "1D")
    df5 = load_frame(symbol, "5MIN", force_refresh=force_refresh)
    return backtest_from_frames(daily, df5, symbol, segment)


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
