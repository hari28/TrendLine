"""Backtest for the CPR + EMA swing/positional strategy the user supplied.

IMPORTANT scoping note (read before trusting the numbers): the user asked for
this on a 15-minute timeframe. That is not feasible end-to-end through the
Kite MCP bridge available here -- Kite's 15-minute historical endpoint caps
each request at ~200 days, so a single symbol's full 2020-to-date history
needs ~13 chunked calls, and tracking every trade's stop-loss/1R/2R/trailing
exit at 15-min precision for its whole holding period (days to weeks) would
require thousands of additional calls system-wide. Each call's candle data
has to pass through the assistant's own context, so this is not achievable
in one sitting.

What this module actually does instead: runs the full rule set (CPR,
20/50/200 EMA trend filter, breakout entry, conservative SL, 1R/2R partial
booking, 20 EMA trailing exit) on DAILY bars, using the free/cached Yahoo
daily history already used everywhere else in this app. This is a faithful,
zero-extra-API-cost implementation of the strategy's rules as literally
specified (the exit rules already reference "a daily candle closes below the
20 EMA" -- CPR itself is a daily-only indicator) -- just not tied to 15-min
candle precision for entry timing or intrabar SL/target detection.

Rules implemented, as clarified by the user:
- Trend filter: close above 200 EMA. Alignment (20 > 50 > 200 EMA) is
  scored but NOT required to pass -- the user's rules call it "preferred",
  not mandatory.
- Entry: LONG when the daily candle CLOSES above that day's CPR Top (TC).
  SHORT when it CLOSES below that day's CPR Bottom (BC). CPR levels are
  computed from the PRIOR day's H/L/C (see cpr.py::compute_cpr) -- CPR is a
  pivot-style level projected for "today" from yesterday's session.
- No subjective "rejection candle" or volume-expansion entry (per user's
  "objective breakout only" choice) -- just the close-above/below-CPR-level
  rule with the EMA trend filter.
- SL: conservative -- CPR Bottom (BC) for longs, CPR Top (TC) for shorts,
  both from the breakout day itself. Never moved after entry (per the
  "never widen the SL" rule -- this implementation never moves it at all).
- Exits (updated 2026-09-10, per user): 50% booked at 1R, remaining 50%
  booked at the midpoint between 1R and 2R (i.e. 1.5R). No trailing leg --
  the position is fully closed by 1.5R. The hard SL from entry still
  applies to the remaining 50% between the 1R and 1.5R exits.
- One trade per symbol at a time: a new signal is ignored while a prior
  trade on that symbol is still open.
"""
import pandas as pd

from cpr import compute_cpr
from indicators import ema
from screener import load_frame

FAST_EMA, MID_EMA, SLOW_EMA = 20, 50, 200
BOOK_1R_FRACTION = 0.50
BOOK_MID_FRACTION = round(1 - BOOK_1R_FRACTION, 4)  # 0.50, booked at 1.5R


def _daily_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA20/50/200 and this-day's CPR (Pivot/TC/BC, from the PRIOR day's H/L/C)
    to a daily OHLCV frame."""
    df = frame.copy()
    df["EMA20"] = ema(df["Close"], FAST_EMA)
    df["EMA50"] = ema(df["Close"], MID_EMA)
    df["EMA200"] = ema(df["Close"], SLOW_EMA)

    prev_high, prev_low, prev_close = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    pivots, tcs, bcs = [], [], []
    for h, l, c in zip(prev_high, prev_low, prev_close):
        if pd.isna(h) or pd.isna(l) or pd.isna(c):
            pivots.append(float("nan")); tcs.append(float("nan")); bcs.append(float("nan"))
            continue
        pivot, tc, bc, _ = compute_cpr(h, l, c)
        pivots.append(pivot); tcs.append(tc); bcs.append(bc)
    df["Pivot"], df["TC"], df["BC"] = pivots, tcs, bcs
    return df


def backtest_symbol(symbol: str, segment: str, start_date: str = "2020-01-01",
                     force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    if frame is None or len(frame) < SLOW_EMA + 5:
        return []

    df = _daily_indicators(frame)
    df = df[df.index >= pd.Timestamp(start_date)]
    df = df.dropna(subset=["EMA200", "TC", "BC"])
    if df.empty:
        return []

    trades = []
    in_trade_until = None  # index label; a new signal is ignored while <= this

    idx = df.index
    for i, day in enumerate(idx):
        if in_trade_until is not None and day <= in_trade_until:
            continue

        row = df.loc[day]
        close, high, low = row["Close"], row["High"], row["Low"]
        ema20, ema50, ema200 = row["EMA20"], row["EMA50"], row["EMA200"]
        tc, bc = row["TC"], row["BC"]

        long_trend_ok = close > ema200
        short_trend_ok = close < ema200
        long_signal = long_trend_ok and close > tc
        short_signal = short_trend_ok and close < bc
        if not (long_signal or short_signal):
            continue
        # if both somehow true (shouldn't happen -- TC >= BC always), prefer long
        direction = "Long" if long_signal else "Short"

        entry_price = float(close)
        entry_date = day
        aligned = (ema20 > ema50 > ema200) if direction == "Long" else (ema20 < ema50 < ema200)

        if direction == "Long":
            sl_price = float(bc)
            if sl_price >= entry_price:
                continue  # degenerate CPR (BC above/at entry) -- no usable risk distance
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
        exit_date, exit_reason = None, None

        for day2 in idx[i + 1:]:
            row2 = df.loc[day2]
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
                    exit_date, exit_reason, qty_remaining = day2, "Stop-loss (full)", 0.0
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
                    exit_date, exit_reason, qty_remaining = day2, "Stop-loss (after 1R)", 0.0
                    break
                if hit_mid:
                    weighted_return += qty_remaining * (target_mid - entry_price) / entry_price * 100.0 \
                        * (1 if direction == "Long" else -1)
                    exit_date, exit_reason, qty_remaining = day2, "Booked at 1.5R (final)", 0.0
                    break

        if exit_date is None:
            last_day = idx[-1]
            last_close = float(df.loc[last_day, "Close"])
            weighted_return += qty_remaining * (last_close - entry_price) / entry_price * 100.0 \
                * (1 if direction == "Long" else -1)
            exit_date, exit_reason, qty_remaining = last_day, "Open (end of data)", 0.0

        trades.append({
            "Symbol": symbol,
            "Segment": segment,
            "Direction": direction,
            "EntryDate": entry_date.date(),
            "EntryPrice": round(entry_price, 2),
            "SLPrice": round(sl_price, 2),
            "Target1R": round(target_1r, 2),
            "TargetMid": round(target_mid, 2),
            "EMAAligned": bool(aligned),
            "ExitDate": exit_date.date(),
            "ExitReason": exit_reason,
            "Booked1R": booked_1r,
            "ReturnPct": round(weighted_return, 2),
        })
        in_trade_until = exit_date

    return trades


def run_backtest(symbols: list[tuple[str, str]], start_date: str = "2020-01-01", progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    """symbols: list of (Symbol, Segment) tuples."""
    all_trades = []
    total = len(symbols)
    for i, (symbol, segment) in enumerate(symbols):
        if progress_cb:
            progress_cb(i, total, symbol)
        all_trades.extend(backtest_symbol(symbol, segment, start_date=start_date, force_refresh=force_refresh))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values("EntryDate").reset_index(drop=True)
    return trades


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_trades": 0, "win_rate_pct": float("nan"), "avg_return_pct": float("nan"),
                "avg_win_pct": float("nan"), "avg_loss_pct": float("nan"), "max_drawdown_pct": float("nan")}

    wins = trades[trades["ReturnPct"] > 0]
    losses = trades[trades["ReturnPct"] <= 0]

    ordered = trades.sort_values("EntryDate")
    equity = (1 + ordered["ReturnPct"] / 100).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    return {
        "total_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "avg_return_pct": round(trades["ReturnPct"].mean(), 2),
        "avg_win_pct": round(wins["ReturnPct"].mean(), 2) if len(wins) else float("nan"),
        "avg_loss_pct": round(losses["ReturnPct"].mean(), 2) if len(losses) else float("nan"),
        "max_drawdown_pct": round(float(drawdown.min()), 2) if not drawdown.empty else float("nan"),
    }
