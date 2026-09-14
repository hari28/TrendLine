"""Backtest for a 9/21 EMA-cross scalping strategy with an RSI filter, on
1-minute or 5-minute intraday bars -- the classic technical-indicator combo
for scalping (fast EMA for momentum, slow EMA for the micro-trend, RSI(14)
to confirm which side is in control).

Rules, as specified:
- Indicators: 9-period EMA (fast), 21-period EMA (slow), RSI(14).
- Entry -- LONG: the 9 EMA crosses above the 21 EMA on a bar where price is
  ALSO trading above both EMAs and RSI(14) is BETWEEN RSI_LONG_MIN (50) and
  RSI_MAX (70), all on the same bar. Entered at that trigger bar's own
  CLOSE (not the next bar's open) -- "enter on the close of the trigger
  candle", per the source rules. SHORT is the exact mirror: cross down,
  price below both EMAs, RSI between RSI_MIN (30) and RSI_SHORT_MAX (50).
  The >50/<50 half of this comes from the original scalping rules (RSI
  confirms which side is in control); the 30/70 half is the same
  overbought/oversold screen exposed on the Screener tab's RSI Range mode
  (screener.py::scan_symbol_rsi) -- added on top so an entry is never taken
  once RSI has already run into overbought/oversold territory (a likely
  already-extended move, not a fresh one).
- Stop-loss: a small buffer beyond the recent swing low (longs) / swing high
  (shorts), where "recent" = the last SWING_LOOKBACK_BARS bars up to and
  including the trigger bar, restricted to the SAME trading session (a
  swing point from the tail of a prior day isn't a meaningful intraday
  level for a same-day scalp). "A few ticks" has no literal tick size
  available here (no real-time market-depth feed) -- approximated as
  STOP_BUFFER_PCT of price, a disclosed simplification.
- Target: risk x rr_multiple. The source material offers "a quick 1:1 or
  1:1.5" -- both are exposed as a parameter, default 1.5.
- Whichever of stop/target is hit first wins; on a bar where both levels
  fall inside that bar's High/Low range (only known from OHLC, not the
  actual intrabar path), the stop-loss is assumed to hit first -- the same
  conservative tie-break used by every other backtest module in this app.
  If neither is hit before the session's last bar, the trade is force-
  closed at that bar's close ("Session close") -- scalping is explicitly an
  intraday-only strategy, so this backtest never carries a position across
  an overnight gap.
- One trade per symbol at a time -- a new signal is ignored while a prior
  trade on that symbol is still open.

Data source note: TrendLine's existing Yahoo intraday fetcher caps 1-minute
history at ~7 days and 5-minute at ~60 days (see data_fetcher.py), so this
backtest necessarily covers a short recent window, not a multi-year one --
same constraint already disclosed on the other intraday backtests in this
app (see cpr_ema_backtest_intraday.py).
"""
import pandas as pd

from indicators import ema, rsi
from screener import load_frame
from swing_backtest import summarize  # noqa: F401 (re-exported for callers) -- order-independent, safe to reuse

FAST_EMA, SLOW_EMA = 9, 21
RSI_PERIOD = 14
RSI_MIN = 30.0   # oversold floor -- same default as the Screener tab's RSI Range mode
RSI_MAX = 70.0   # overbought ceiling -- same default as the Screener tab's RSI Range mode
RSI_LONG_MIN = 50.0
RSI_SHORT_MAX = 50.0
SWING_LOOKBACK_BARS = 10
STOP_BUFFER_PCT = 0.05  # "a few ticks" beyond the swing point -- see module docstring
DEFAULT_RR_MULTIPLE = 1.5
VALID_TIMEFRAMES = ("1MIN", "5MIN")


def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["EMAFast"] = ema(df["Close"], FAST_EMA)
    df["EMASlow"] = ema(df["Close"], SLOW_EMA)
    df["RSI"] = rsi(df["Close"], RSI_PERIOD)
    df["TradingDate"] = df.index.date
    return df


def _swing_stop(df: pd.DataFrame, i: int, direction: str) -> float:
    """Lowest low (Long) / highest high (Short) of the last SWING_LOOKBACK_BARS
    bars up to and including bar i, never reaching back past bar i's own
    trading day."""
    day = df["TradingDate"].iloc[i]
    start = i
    while start > 0 and (i - start) < SWING_LOOKBACK_BARS - 1 and df["TradingDate"].iloc[start - 1] == day:
        start -= 1
    window = df.iloc[start:i + 1]
    if direction == "Long":
        return float(window["Low"].min()) * (1 - STOP_BUFFER_PCT / 100.0)
    return float(window["High"].max()) * (1 + STOP_BUFFER_PCT / 100.0)


def _scan_exit(df: pd.DataFrame, entry_idx: int, entry_day, stop_price: float, target_price: float,
               direction: str) -> tuple:
    last_idx = entry_idx
    for k in range(entry_idx + 1, len(df)):
        if df["TradingDate"].iloc[k] != entry_day:
            break
        last_idx = k
        low, high = df["Low"].iloc[k], df["High"].iloc[k]
        if direction == "Long":
            hit_stop, hit_target = low <= stop_price, high >= target_price
        else:
            hit_stop, hit_target = high >= stop_price, low <= target_price
        if hit_stop:
            return k, stop_price, "Stop-loss"
        if hit_target:
            return k, target_price, "Target"
    return last_idx, float(df["Close"].iloc[last_idx]), "Session close"


def backtest_symbol(symbol: str, segment: str, timeframe: str = "5MIN",
                     rr_multiple: float = DEFAULT_RR_MULTIPLE, force_refresh: bool = False) -> list[dict]:
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {VALID_TIMEFRAMES}, got {timeframe!r}")
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None or len(frame) < SLOW_EMA + RSI_PERIOD + SWING_LOOKBACK_BARS:
        return []

    df = _prepare_indicators(frame)
    df = df.dropna(subset=["EMAFast", "EMASlow", "RSI"])
    if df.empty:
        return []
    df = df.reset_index(names="Timestamp")

    diff = (df["EMAFast"] - df["EMASlow"]).to_numpy()
    n = len(df)

    trades = []
    in_trade_until = -1
    for i in range(1, n):
        if i <= in_trade_until:
            continue
        prev_diff, cur_diff = diff[i - 1], diff[i]
        if pd.isna(prev_diff) or pd.isna(cur_diff):
            continue

        row = df.iloc[i]
        direction = None
        if prev_diff <= 0 and cur_diff > 0 and row["Close"] > row["EMAFast"] and row["Close"] > row["EMASlow"] \
                and RSI_LONG_MIN < row["RSI"] <= RSI_MAX:
            direction = "Long"
        elif prev_diff >= 0 and cur_diff < 0 and row["Close"] < row["EMAFast"] and row["Close"] < row["EMASlow"] \
                and RSI_MIN <= row["RSI"] < RSI_SHORT_MAX:
            direction = "Short"
        if direction is None:
            continue

        entry_price = float(row["Close"])
        stop_price = _swing_stop(df, i, direction)
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            continue
        risk_pct = risk / entry_price * 100.0
        reward_pct = risk_pct * rr_multiple
        target_price = entry_price + risk * rr_multiple if direction == "Long" else entry_price - risk * rr_multiple

        exit_idx, exit_price, exit_reason = _scan_exit(
            df, i, row["TradingDate"], stop_price, target_price, direction)
        pct_return = (exit_price - entry_price) / entry_price * 100.0 if direction == "Long" \
            else (entry_price - exit_price) / entry_price * 100.0

        entry_ts, exit_ts = row["Timestamp"], df["Timestamp"].iloc[exit_idx]
        trades.append({
            "Symbol": symbol,
            "Segment": segment,
            "Direction": direction,
            "EntryDate": entry_ts.date(),
            "EntryTime": entry_ts.strftime("%H:%M"),
            "EntryPrice": round(entry_price, 2),
            "StopPrice": round(stop_price, 2),
            "TargetPrice": round(target_price, 2),
            "RiskPct": round(risk_pct, 2),
            "RewardPct": round(reward_pct, 2),
            "ExitDate": exit_ts.date(),
            "ExitTime": exit_ts.strftime("%H:%M"),
            "ExitPrice": round(float(exit_price), 2),
            "ExitReason": exit_reason,
            "ReturnPct": round(pct_return, 2),
            "HoldingBars": exit_idx - i,
        })
        in_trade_until = exit_idx

    return trades


def run_backtest(symbols: list[tuple[str, str]], timeframe: str = "5MIN",
                  rr_multiple: float = DEFAULT_RR_MULTIPLE, progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    all_trades = []
    total = len(symbols)
    for i, (symbol, segment) in enumerate(symbols):
        if progress_cb:
            progress_cb(i, total, symbol)
        all_trades.extend(backtest_symbol(symbol, segment, timeframe=timeframe, rr_multiple=rr_multiple,
                                           force_refresh=force_refresh))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["EntryDate", "EntryTime"]).reset_index(drop=True)
    return trades


def simulate_portfolio(trades: pd.DataFrame, risk_per_trade_pct: float = 1.0, max_concurrent: int = 20,
                        starting_capital: float = 100.0) -> dict:
    """Same accounting model as swing_backtest.simulate_portfolio (fixed
    fractional risk per trade sized off that trade's own stop distance,
    capped at an equal-weight cash allocation of capital / max_concurrent,
    at most max_concurrent positions open at once, equity marked only when a
    position closes) -- but ordered by full ENTRY/EXIT TIMESTAMP, not just
    calendar date. Scalp trades routinely open AND close within the same
    trading day (often several times per symbol), unlike this app's
    daily-bar strategies where EntryDate < ExitDate always -- reusing the
    date-only event ordering from swing_backtest.simulate_portfolio here
    silently misorders same-day trades (an exit can sort before its own
    entry once several trades share a date), which corrupts the capital
    ledger. Confirmed this by running it against real scalping output
    before writing this dedicated version."""
    if trades.empty:
        return {"final_capital": starting_capital, "total_return_pct": 0.0, "max_drawdown_pct": float("nan"),
                "trades_taken": 0, "trades_skipped": 0, "equity_curve": pd.Series(dtype=float)}

    closed = trades.reset_index(drop=True).copy()
    closed["EntryTS"] = pd.to_datetime(closed["EntryDate"].astype(str) + " " + closed["EntryTime"])
    closed["ExitTS"] = pd.to_datetime(closed["ExitDate"].astype(str) + " " + closed["ExitTime"])

    # priority 0 = exits, 1 = entries -- on a tied timestamp, free a slot/
    # capital from a closing trade before considering a new entry at that
    # same instant (same convention as swing_backtest.simulate_portfolio).
    events = []
    for idx, row in closed.iterrows():
        events.append((row["EntryTS"], 1, idx))
        events.append((row["ExitTS"], 0, idx))
    events.sort(key=lambda e: (e[0], e[1]))

    capital = starting_capital
    open_positions = {}
    equity_dates, equity_values = [], []
    taken = skipped = 0

    for ts, kind, idx in events:
        row = closed.iloc[idx]
        if kind == 1:  # entry
            if len(open_positions) >= max_concurrent:
                skipped += 1
                continue
            risk_amount = capital * risk_per_trade_pct / 100.0
            risk_pct = row["RiskPct"] if row["RiskPct"] > 0 else 0.01
            cash_cap = capital / max_concurrent
            position_size = min(risk_amount / (risk_pct / 100.0), cash_cap)
            open_positions[idx] = position_size * (row["ReturnPct"] / 100.0)
            taken += 1
        else:
            pnl_amount = open_positions.pop(idx, None)
            if pnl_amount is None:
                continue  # this trade's entry was skipped (no capacity) -- nothing to realize
            capital += pnl_amount
            equity_dates.append(ts)
            equity_values.append(capital)

    equity_curve = pd.Series(equity_values, index=equity_dates)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max * 100

    return {
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - starting_capital) / starting_capital * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()), 2) if not drawdown.empty else float("nan"),
        "trades_taken": taken,
        "trades_skipped": skipped,
        "equity_curve": equity_curve,
    }
