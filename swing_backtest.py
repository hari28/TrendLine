"""Backtests for two positional-swing entry strategies, both built around one
shared risk/reward rule the user specified: minimum 5% target, 1:2 risk:reward,
minimal drawdown. Long-only (both strategies are defined for uptrends).

Shared trade-management rule (see _finalize_stop_target):
  - Stop-loss = ATR-BASED, not structural and not a flat %: Stop = Entry -
    ATR_MULTIPLE * ATR(14). This scales the stop to each stock's own recent
    volatility -- a quiet stock gets a tighter stop, a volatile one gets a
    wider one, both scaled off actual evidence instead of one-size-fits-all.
    (An earlier version used the CPR bottom / pullback low as the stop,
    tightened to a flat 2.5% cap when wider -- that tightening ignored the
    stock's real volatility and risked stopping out on ordinary noise for
    anything more volatile than the cap. Switched to ATR per the user's
    explicit request for "any other recommendation" instead of that cap.)
  - If the ATR-implied risk exceeds MAX_RISK_PCT, the trade is SKIPPED, not
    tightened -- an artificially shrunk stop gets whipsawed by normal noise,
    which is worse than not taking the trade.
  - Target = max(RR_MULTIPLE * actual_risk_pct, MIN_REWARD_PCT) -- so the 1:2
    ratio and the 5% floor never fight each other; whichever is bigger wins.
  - Whichever of stop/target is touched first exits the trade; if both would
    be touched within the same bar (only known from that bar's High/Low, not
    the intrabar path), the stop-loss is assumed to hit first -- the
    conservative assumption, same as backtest.py/cpr_ema_backtest.py.
  - One trade per symbol at a time -- a new signal is ignored while a prior
    trade on that symbol is still open.

Strategy A -- "Volatility Contraction Breakout":
  Trend filter: Close > 200 EMA.
  Setup: Narrow CPR that day (see cpr.py -- Width% <= WIDTH_NARROW_MAX).
  Trigger: Close breaks above that day's CPR Top (TC), on volume >=
    VOLUME_SPIKE_MULTIPLE x the trailing 20-day average (confirms real
    participation, not a low-volume drift above the level).
  Stop: ATR-based (see above) -- the CPR level still defines the setup and
    entry trigger, just no longer the stop distance.

Strategy B -- "Trend Pullback Entry":
  Trend filter: Market Structure shows an established uptrend (HH/HL, via
    structure.py::trend_history) AND Close > 200 EMA.
  Setup: a pullback day whose Low touches within PULLBACK_TOUCH_PCT of the
    rising 20 EMA.
  Trigger: within PULLBACK_MAX_WAIT_BARS, a later day's Close reclaims back
    above the pullback day's High.
  Stop: ATR-based (see above) -- the pullback low still defines the setup,
    just no longer the stop distance.
"""
import pandas as pd

from cpr import compute_cpr, width_category
from indicators import atr, ema
from screener import load_frame
from structure import trend_history, SWING_ORDER

FAST_EMA, MID_EMA, SLOW_EMA = 20, 50, 200
VOL_AVG_PERIOD = 20
VOLUME_SPIKE_MULTIPLE = 1.5
PULLBACK_TOUCH_PCT = 1.0       # Low within 1% of (at or below) the rising 20 EMA
PULLBACK_MAX_WAIT_BARS = 5     # give up waiting for the reclaim after this many bars

ATR_PERIOD = 14
ATR_MULTIPLE = 1.5             # Stop = Entry - ATR_MULTIPLE * ATR(14)
MAX_RISK_PCT = 4.5             # skip the trade if the ATR-implied risk exceeds this
MIN_REWARD_PCT = 5.0           # target never falls below this
RR_MULTIPLE = 2.0              # target = max(RR_MULTIPLE * risk, MIN_REWARD_PCT)

STRATEGY_LABELS = {
    "breakout": "Volatility Contraction Breakout",
    "pullback": "Trend Pullback Entry",
}


def _prepare_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA20/50/200 and this-day's CPR (Pivot/TC/BC/Width%, projected from
    the PRIOR day's H/L/C -- same convention as cpr.py/cpr_ema_backtest.py)
    plus a trailing 20-day average volume (excluding the current bar, so a
    spike day doesn't inflate its own baseline)."""
    df = frame.copy()
    df["EMA20"] = ema(df["Close"], FAST_EMA)
    df["EMA50"] = ema(df["Close"], MID_EMA)
    df["EMA200"] = ema(df["Close"], SLOW_EMA)
    df["ATR14"] = atr(df, ATR_PERIOD)
    df["VolAvg20"] = df["Volume"].shift(1).rolling(window=VOL_AVG_PERIOD, min_periods=VOL_AVG_PERIOD).mean()

    prev_high, prev_low, prev_close = df["High"].shift(1), df["Low"].shift(1), df["Close"].shift(1)
    pivots, tcs, bcs, widths = [], [], [], []
    for h, l, c in zip(prev_high, prev_low, prev_close):
        if pd.isna(h) or pd.isna(l) or pd.isna(c):
            pivots.append(float("nan")); tcs.append(float("nan")); bcs.append(float("nan"))
            widths.append(float("nan"))
            continue
        pivot, tc, bc, width_pct = compute_cpr(h, l, c)
        pivots.append(pivot); tcs.append(tc); bcs.append(bc); widths.append(width_pct)
    df["Pivot"], df["TC"], df["BC"], df["WidthPct"] = pivots, tcs, bcs, widths
    return df


def _finalize_stop_target(entry_price: float, atr_value: float, direction: str = "Long") -> tuple | None:
    """Stop = Entry -/+ ATR_MULTIPLE * ATR(14), scaled to this stock's own
    recent volatility. If that implies more risk than MAX_RISK_PCT, the trade
    is SKIPPED (returns None) rather than the stop being tightened to fit --
    an artificially shrunk stop just gets whipsawed by ordinary noise on a
    stock this volatile. Target = max(RR_MULTIPLE * actual_risk, MIN_REWARD_PCT)."""
    if pd.isna(atr_value) or atr_value <= 0:
        return None
    risk_distance = ATR_MULTIPLE * atr_value
    risk_pct = risk_distance / entry_price * 100.0
    if risk_pct <= 0 or risk_pct > MAX_RISK_PCT:
        return None
    reward_pct = max(RR_MULTIPLE * risk_pct, MIN_REWARD_PCT)
    if direction == "Long":
        stop_price = entry_price - risk_distance
        target_price = entry_price * (1 + reward_pct / 100.0)
    else:
        stop_price = entry_price + risk_distance
        target_price = entry_price * (1 - reward_pct / 100.0)
    return stop_price, target_price, risk_pct, reward_pct


def _scan_exit(df: pd.DataFrame, start_idx: int, stop_price: float, target_price: float,
               direction: str = "Long") -> tuple:
    for k in range(start_idx, len(df)):
        low, high = df["Low"].iloc[k], df["High"].iloc[k]
        if direction == "Long":
            hit_stop, hit_target = low <= stop_price, high >= target_price
        else:
            hit_stop, hit_target = high >= stop_price, low <= target_price
        if hit_stop:
            return k, stop_price, "Stop-loss"
        if hit_target:
            return k, target_price, "Target"
    return None, None, None


def _make_trade(symbol, segment, strategy, entry_idx, entry_date, entry_price, stop_price, target_price,
                 risk_pct, reward_pct, df) -> dict:
    exit_idx, exit_price, exit_reason = _scan_exit(df, entry_idx + 1, stop_price, target_price, "Long")
    if exit_idx is None:
        exit_idx = len(df) - 1
        exit_price = float(df["Close"].iloc[-1])
        exit_reason = "Open (end of data)"
    pct_return = (exit_price - entry_price) / entry_price * 100.0
    return {
        "Strategy": STRATEGY_LABELS[strategy],
        "Symbol": symbol,
        "Segment": segment,
        "EntryDate": entry_date,
        "EntryPrice": round(entry_price, 2),
        "StopPrice": round(stop_price, 2),
        "TargetPrice": round(target_price, 2),
        "RiskPct": round(risk_pct, 2),
        "RewardPct": round(reward_pct, 2),
        "ExitDate": df.index[exit_idx].date(),
        "ExitPrice": round(float(exit_price), 2),
        "ExitReason": exit_reason,
        "ReturnPct": round(float(pct_return), 2),
        "HoldingBars": exit_idx - entry_idx,
    }, exit_idx


def backtest_symbol_breakout(symbol: str, segment: str, start_date: str = "2020-01-01",
                              force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    if frame is None or len(frame) < SLOW_EMA + VOL_AVG_PERIOD + 5:
        return []
    df = _prepare_indicators(frame)
    df = df.dropna(subset=["EMA200", "TC", "BC", "VolAvg20", "ATR14"])
    df = df[df.index >= pd.Timestamp(start_date)]
    if df.empty:
        return []

    trades = []
    in_trade_until = -1
    for i in range(len(df)):
        if i <= in_trade_until:
            continue
        row = df.iloc[i]
        if row["Close"] <= row["EMA200"]:
            continue
        if width_category(row["WidthPct"]) != "Narrow":
            continue
        if row["Close"] <= row["TC"]:
            continue
        if row["Volume"] < row["VolAvg20"] * VOLUME_SPIKE_MULTIPLE:
            continue

        entry_price = float(row["Close"])
        result = _finalize_stop_target(entry_price, float(row["ATR14"]), "Long")
        if result is None:
            continue
        stop_price, target_price, risk_pct, reward_pct = result

        trade, exit_idx = _make_trade(symbol, segment, "breakout", i, df.index[i].date(), entry_price,
                                       stop_price, target_price, risk_pct, reward_pct, df)
        trades.append(trade)
        in_trade_until = exit_idx

    return trades


def backtest_symbol_pullback(symbol: str, segment: str, start_date: str = "2020-01-01",
                              force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    if frame is None or len(frame) < SLOW_EMA + 5:
        return []
    df = _prepare_indicators(frame)
    trend = trend_history(frame, SWING_ORDER).reindex(frame.index)
    df = df.join(trend.rename("Trend"))
    df = df.dropna(subset=["EMA200", "EMA20", "ATR14"])
    df = df[df.index >= pd.Timestamp(start_date)]
    if df.empty:
        return []

    trades = []
    n = len(df)
    in_trade_until = -1
    p = 0
    while p < n:
        if p <= in_trade_until:
            p += 1
            continue
        row = df.iloc[p]
        is_pullback = (
            row["Trend"] == "Up" and row["Close"] > row["EMA200"]
            and row["Low"] <= row["EMA20"] * (1 + PULLBACK_TOUCH_PCT / 100.0)
        )
        if not is_pullback:
            p += 1
            continue

        pullback_high = float(row["High"])
        entry_idx = None
        for e in range(p + 1, min(p + 1 + PULLBACK_MAX_WAIT_BARS, n)):
            if df["Close"].iloc[e] > pullback_high:
                entry_idx = e
                break
        if entry_idx is None:
            p += 1
            continue

        entry_price = float(df["Close"].iloc[entry_idx])
        result = _finalize_stop_target(entry_price, float(df["ATR14"].iloc[entry_idx]), "Long")
        if result is None:
            p = entry_idx + 1
            continue
        stop_price, target_price, risk_pct, reward_pct = result

        trade, exit_idx = _make_trade(symbol, segment, "pullback", entry_idx, df.index[entry_idx].date(),
                                       entry_price, stop_price, target_price, risk_pct, reward_pct, df)
        trades.append(trade)
        in_trade_until = exit_idx
        p = exit_idx + 1

    return trades


def run_backtest(strategy: str, symbols: list[tuple[str, str]], start_date: str = "2020-01-01", progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    """strategy: "breakout", "pullback", or "both". symbols: list of (Symbol, Segment)."""
    fn_by_strategy = {"breakout": backtest_symbol_breakout, "pullback": backtest_symbol_pullback}
    strategies = ["breakout", "pullback"] if strategy == "both" else [strategy]

    all_trades = []
    total = len(symbols) * len(strategies)
    done = 0
    for strat in strategies:
        fn = fn_by_strategy[strat]
        for symbol, segment in symbols:
            if progress_cb:
                progress_cb(done, total, symbol)
            all_trades.extend(fn(symbol, segment, start_date=start_date, force_refresh=force_refresh))
            done += 1

    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["Strategy", "EntryDate"]).reset_index(drop=True)
    return trades


def simulate_portfolio(trades: pd.DataFrame, risk_per_trade_pct: float = 1.0, max_concurrent: int = 20,
                        starting_capital: float = 100.0) -> dict:
    """A realistic drawdown read needs position sizing and concurrency limits --
    summarize()'s equity curve compounds every trade sequentially as if 100% of
    capital were staked on each signal in turn, which blows up nonsensically
    once thousands of trades' dates overlap across symbols (as they do here).

    This instead risks a fixed `risk_per_trade_pct` of CURRENT capital per
    trade, sized off that trade's own stop distance -- BUT capped at an
    equal-weight cash allocation of capital / max_concurrent, since this is a
    plain cash equity account with no margin: a very tight stop (e.g. 0.3%)
    must NOT be levered up to hit the risk target, it just risks less than
    that target. (An earlier version of this function skipped that cap and
    scaled position size purely off the risk target -- on tight-stop trades
    that implied 5-10x leverage on a single name, which produced nonsense
    results in both directions once compounded across thousands of trades.)
    At most `max_concurrent` positions open at once -- a new signal beyond
    that cap is skipped (no slot/capital available), same as a real trader
    would have to pass on it. Equity updates only when a position closes
    (realized P&L only, no intra-trade mark-to-market) -- a simplification,
    but a far more honest drawdown figure than the naive full-compounding
    chain."""
    closed = trades[trades["ExitReason"] != "Open (end of data)"].copy()
    if closed.empty:
        return {"final_capital": starting_capital, "total_return_pct": 0.0, "max_drawdown_pct": float("nan"),
                "trades_taken": 0, "trades_skipped": 0, "equity_curve": pd.Series(dtype=float)}

    closed["EntryDate"] = pd.to_datetime(closed["EntryDate"])
    closed["ExitDate"] = pd.to_datetime(closed["ExitDate"])
    closed = closed.reset_index(drop=True)

    # priority 0 = exits, 1 = entries -- on a tied date, free a slot/capital
    # from a closing trade before considering a new entry that same day.
    events = []
    for idx, row in closed.iterrows():
        events.append((row["EntryDate"], 1, "entry", idx))
        events.append((row["ExitDate"], 0, "exit", idx))
    events.sort(key=lambda e: (e[0], e[1]))

    capital = starting_capital
    open_positions = {}
    equity_dates, equity_values = [], []
    taken = skipped = 0

    for date, _, kind, idx in events:
        row = closed.iloc[idx]
        if kind == "entry":
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
            equity_dates.append(date)
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


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_trades": 0, "closed_trades": 0, "open_trades": 0, "win_rate_pct": float("nan"),
                "avg_return_pct": float("nan"), "avg_win_pct": float("nan"), "avg_loss_pct": float("nan"),
                "expectancy_pct": float("nan"), "max_drawdown_pct": float("nan")}

    closed = trades[trades["ExitReason"] != "Open (end of data)"]
    wins = closed[closed["ReturnPct"] > 0]
    losses = closed[closed["ReturnPct"] <= 0]

    ordered = closed.sort_values("EntryDate")
    equity = (1 + ordered["ReturnPct"] / 100).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    win_rate = len(wins) / len(closed) if len(closed) else float("nan")
    avg_win = wins["ReturnPct"].mean() if len(wins) else 0.0
    avg_loss = losses["ReturnPct"].mean() if len(losses) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss if len(closed) else float("nan")

    return {
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "open_trades": len(trades) - len(closed),
        "win_rate_pct": round(win_rate * 100, 2) if len(closed) else float("nan"),
        "avg_return_pct": round(closed["ReturnPct"].mean(), 2) if len(closed) else float("nan"),
        "avg_win_pct": round(avg_win, 2) if len(wins) else float("nan"),
        "avg_loss_pct": round(avg_loss, 2) if len(losses) else float("nan"),
        "expectancy_pct": round(expectancy, 2) if len(closed) else float("nan"),
        "max_drawdown_pct": round(float(drawdown.min()), 2) if not drawdown.empty else float("nan"),
    }
