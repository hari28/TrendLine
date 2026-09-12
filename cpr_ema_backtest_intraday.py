"""15-minute intraday variant of the CPR + EMA backtest.

Same core strategy as cpr_ema_backtest.py, with one change the user asked
for: the trend filter is no longer "close above/below the 200 EMA" but a
persisting EMA20-vs-EMA200 relationship on the 15-min series itself --
long only while EMA20 > EMA200, short only while EMA20 < EMA200 (state-based,
not a one-shot crossover trigger -- clarified with the user). Periods are
kept literally 20/200 bars (also clarified with the user), which on 15-min
bars is a much shorter real-time lookback than on daily bars (~8 trading
days vs. ~10 months) -- this is a short-term momentum filter now, not a
long-term trend filter.

Data-source note: the Kite MCP bridge was ruled out for bulk intraday
backtesting (its 15-min endpoint caps each request at ~200 days, and every
candle would have to flow through the assistant's own conversation context
-- see cpr_ema_backtest.py's docstring). This module sidesteps that entirely
by reusing TrendLine's own Yahoo-based intraday fetcher
(screener.load_frame / data_fetcher.fetch_intraday_history), which runs as
a normal local call with no context cost. The real constraint here is
Yahoo's own retention: only ~60 days of 15-min history are available, so
this covers roughly the last 60 trading days, not 2020-present -- expect a
much smaller trade sample than the daily backtest.

Rules (same as cpr_ema_backtest.py except the trend filter, above):
- Entry: a 15-min bar CLOSES above the day's CPR Top (TC) while EMA20 >
  EMA200 (long), or CLOSES below the day's CPR Bottom (BC) while EMA20 <
  EMA200 (short). CPR is still a DAILY indicator -- computed once per
  trading day from the PRIOR day's daily H/L/C and held constant across
  every 15-min bar of that day.
- SL: CPR BC (long) / TC (short) from the entry day, fixed for the trade's
  life.
- Exits: 50% booked at 1R, remaining 50% booked at the midpoint between 1R
  and 2R (1.5R) -- same exit rule as the current daily version. Checked
  bar-by-bar on 15-min highs/lows rather than daily ones.
- One trade per symbol at a time.
"""
import pandas as pd

from cpr import compute_cpr
from indicators import ema
from screener import load_frame

FAST_EMA, SLOW_EMA = 20, 200
BOOK_1R_FRACTION = 0.50
BOOK_MID_FRACTION = round(1 - BOOK_1R_FRACTION, 4)  # 0.50, booked at 1.5R


def _daily_cpr_by_date(symbol: str) -> dict:
    """date -> (tc, bc) for that trading day, from the PRIOR daily session's H/L/C."""
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


def _intraday_indicators(frame: pd.DataFrame, cpr_by_date: dict) -> pd.DataFrame:
    df = frame.copy()
    df["EMA20"] = ema(df["Close"], FAST_EMA)
    df["EMA200"] = ema(df["Close"], SLOW_EMA)
    tcs, bcs = [], []
    for ts in df.index:
        levels = cpr_by_date.get(ts.date())
        if levels is None:
            tcs.append(float("nan")); bcs.append(float("nan"))
        else:
            tcs.append(levels[0]); bcs.append(levels[1])
    df["TC"], df["BC"] = tcs, bcs
    return df


def backtest_symbol(symbol: str, segment: str, force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "15MIN", force_refresh=force_refresh)
    if frame is None or len(frame) < SLOW_EMA + 5:
        return []

    cpr_by_date = _daily_cpr_by_date(symbol)
    if not cpr_by_date:
        return []

    df = _intraday_indicators(frame, cpr_by_date)
    df = df.dropna(subset=["EMA200", "TC", "BC"])
    if df.empty:
        return []

    trades = []
    in_trade_until = None  # timestamp label; a new signal is ignored while <= this

    idx = df.index
    for i, ts in enumerate(idx):
        if in_trade_until is not None and ts <= in_trade_until:
            continue

        row = df.loc[ts]
        close = row["Close"]
        ema20, ema200 = row["EMA20"], row["EMA200"]
        tc, bc = row["TC"], row["BC"]

        long_trend_ok = ema20 > ema200
        short_trend_ok = ema20 < ema200
        long_signal = long_trend_ok and close > tc
        short_signal = short_trend_ok and close < bc
        if not (long_signal or short_signal):
            continue
        direction = "Long" if long_signal else "Short"

        entry_price = float(close)
        entry_time = ts

        if direction == "Long":
            sl_price = float(bc)
            if sl_price >= entry_price:
                continue  # degenerate CPR -- no usable risk distance
            r = entry_price - sl_price
            target_1r = entry_price + r
            target_mid = entry_price + 1.5 * r
        else:
            sl_price = float(tc)
            if sl_price <= entry_price:
                continue
            r = sl_price - entry_price
            target_1r = entry_price - r
            target_mid = entry_price - 1.5 * r

        booked_1r = False
        qty_remaining = 1.0
        weighted_return = 0.0
        exit_time, exit_reason = None, None

        for ts2 in idx[i + 1:]:
            row2 = df.loc[ts2]
            low2, high2 = row2["Low"], row2["High"]

            if direction == "Long":
                hit_sl = low2 <= sl_price
                hit_1r = high2 >= target_1r
                hit_mid = high2 >= target_mid
            else:
                hit_sl = high2 >= sl_price
                hit_1r = low2 <= target_1r
                hit_mid = low2 <= target_mid

            if not booked_1r:
                if hit_sl:
                    weighted_return += qty_remaining * (sl_price - entry_price) / entry_price * 100.0 \
                        * (1 if direction == "Long" else -1)
                    exit_time, exit_reason, qty_remaining = ts2, "Stop-loss (full)", 0.0
                    break
                if hit_1r:
                    weighted_return += BOOK_1R_FRACTION * (target_1r - entry_price) / entry_price * 100.0 \
                        * (1 if direction == "Long" else -1)
                    qty_remaining -= BOOK_1R_FRACTION
                    booked_1r = True
            else:
                if hit_sl:
                    weighted_return += qty_remaining * (sl_price - entry_price) / entry_price * 100.0 \
                        * (1 if direction == "Long" else -1)
                    exit_time, exit_reason, qty_remaining = ts2, "Stop-loss (after 1R)", 0.0
                    break
                if hit_mid:
                    weighted_return += qty_remaining * (target_mid - entry_price) / entry_price * 100.0 \
                        * (1 if direction == "Long" else -1)
                    exit_time, exit_reason, qty_remaining = ts2, "Booked at 1.5R (final)", 0.0
                    break

        if exit_time is None:
            last_ts = idx[-1]
            last_close = float(df.loc[last_ts, "Close"])
            weighted_return += qty_remaining * (last_close - entry_price) / entry_price * 100.0 \
                * (1 if direction == "Long" else -1)
            exit_time, exit_reason, qty_remaining = last_ts, "Open (end of data)", 0.0

        trades.append({
            "Symbol": symbol,
            "Segment": segment,
            "Direction": direction,
            "EntryTime": entry_time,
            "EntryPrice": round(entry_price, 2),
            "SLPrice": round(sl_price, 2),
            "Target1R": round(target_1r, 2),
            "TargetMid": round(target_mid, 2),
            "ExitTime": exit_time,
            "ExitReason": exit_reason,
            "Booked1R": booked_1r,
            "ReturnPct": round(weighted_return, 2),
        })
        in_trade_until = exit_time

    return trades


def run_backtest(symbols: list[tuple[str, str]], progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    """symbols: list of (Symbol, Segment) tuples."""
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
