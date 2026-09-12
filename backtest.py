"""Backtest for the Golden Cross screener strategy (50/200 EMA cross).

Rules (as specified by the user, now configurable -- defaults are the
original weekly spec):
- Timeframe: weekly bars by default; any TIMEFRAMES key from indicators.py
  works (e.g. "1D" for daily).
- Signal: 50-period EMA crosses above the 200-period EMA on that timeframe
  (same event the Golden Cross scan mode detects, see
  screener.py::_cross_type).
- Entry: only once price trades `entry_trigger_pct` above the cross bar's
  close (a breakout confirmation, not the cross itself). Fill is assumed at
  the trigger price, on the first bar whose High reaches it.
- Exit: `target_pct` target or `stop_pct` stop-loss from entry, whichever is
  hit first. If both are touched within the same bar (only known from
  High/Low, not the intrabar path), the stop-loss is assumed to hit first --
  the conservative assumption.
- One trade per symbol at a time: a new cross signal is ignored while a
  prior trade on that symbol is still open.
"""
import pandas as pd

from constituents import get_all_symbols
from screener import load_frame, _ma, FAST_PERIOD, SLOW_PERIOD

MA_TYPE = "EMA"
TIMEFRAME = "1W"
ENTRY_TRIGGER_PCT = 0.05
TARGET_PCT = 0.25
STOP_PCT = 0.05


def find_golden_crosses(frame: pd.DataFrame, ma_type: str = MA_TYPE) -> list[int]:
    """Integer positions in `frame` where the fast MA crosses above the slow MA."""
    close = frame["Close"]
    fast = _ma(close, ma_type, FAST_PERIOD)
    slow = _ma(close, ma_type, SLOW_PERIOD)
    diff = fast - slow

    events = []
    for i in range(1, len(diff)):
        prev, cur = diff.iloc[i - 1], diff.iloc[i]
        if pd.isna(prev) or pd.isna(cur):
            continue
        if prev <= 0 and cur > 0:
            events.append(i)
    return events


def backtest_symbol(symbol: str, segment: str, timeframe: str = TIMEFRAME,
                     entry_trigger_pct: float = ENTRY_TRIGGER_PCT, target_pct: float = TARGET_PCT,
                     stop_pct: float = STOP_PCT, force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, timeframe, force_refresh=force_refresh)
    if frame is None or len(frame) < SLOW_PERIOD + 5:
        return []

    trades = []
    in_trade_until = -1

    for cross_idx in find_golden_crosses(frame):
        if cross_idx <= in_trade_until:
            continue

        cross_close = frame["Close"].iloc[cross_idx]
        trigger_price = cross_close * (1 + entry_trigger_pct)

        entry_idx = None
        for j in range(cross_idx + 1, len(frame)):
            if frame["High"].iloc[j] >= trigger_price:
                entry_idx = j
                break
        if entry_idx is None:
            continue

        entry_price = trigger_price
        target_price = entry_price * (1 + target_pct)
        stop_price = entry_price * (1 - stop_pct)

        exit_idx, exit_price, exit_reason = None, None, None
        for k in range(entry_idx, len(frame)):
            low, high = frame["Low"].iloc[k], frame["High"].iloc[k]
            hit_stop = low <= stop_price
            hit_target = high >= target_price
            if hit_stop:
                exit_idx, exit_price, exit_reason = k, stop_price, "Stop-loss"
                break
            if hit_target:
                exit_idx, exit_price, exit_reason = k, target_price, "Target"
                break

        if exit_idx is None:
            exit_idx = len(frame) - 1
            exit_price = frame["Close"].iloc[-1]
            exit_reason = "Open (end of data)"

        pct_return = (exit_price - entry_price) / entry_price * 100.0
        trades.append({
            "Symbol": symbol,
            "Segment": segment,
            "CrossDate": frame.index[cross_idx].date(),
            "EntryDate": frame.index[entry_idx].date(),
            "EntryPrice": round(float(entry_price), 2),
            "ExitDate": frame.index[exit_idx].date(),
            "ExitPrice": round(float(exit_price), 2),
            "ExitReason": exit_reason,
            "ReturnPct": round(float(pct_return), 2),
            "HoldingBars": exit_idx - entry_idx,
        })
        in_trade_until = exit_idx

    return trades


def run_backtest(segments: list[str], timeframe: str = TIMEFRAME, entry_trigger_pct: float = ENTRY_TRIGGER_PCT,
                  target_pct: float = TARGET_PCT, stop_pct: float = STOP_PCT, progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    symbols_df = get_all_symbols(segments, force_refresh=force_refresh)[["Symbol", "Segment"]]
    all_trades = []
    total = len(symbols_df)
    for i, r in enumerate(symbols_df.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        all_trades.extend(backtest_symbol(r.Symbol, r.Segment, timeframe=timeframe,
                                           entry_trigger_pct=entry_trigger_pct, target_pct=target_pct,
                                           stop_pct=stop_pct, force_refresh=force_refresh))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values("EntryDate").reset_index(drop=True)
    return trades


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"total_trades": 0, "closed_trades": 0, "open_trades": 0, "win_rate_pct": float("nan"),
                "avg_return_pct": float("nan"), "avg_win_pct": float("nan"), "avg_loss_pct": float("nan"),
                "max_drawdown_pct": float("nan")}

    closed = trades[trades["ExitReason"] != "Open (end of data)"]
    wins = closed[closed["ReturnPct"] > 0]
    losses = closed[closed["ReturnPct"] <= 0]

    equity_curve = equity_curve_from_trades(closed)
    max_drawdown = drawdown_pct(equity_curve).min() if not equity_curve.empty else float("nan")

    return {
        "total_trades": len(trades),
        "closed_trades": len(closed),
        "open_trades": len(trades) - len(closed),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 2) if len(closed) else float("nan"),
        "avg_return_pct": round(closed["ReturnPct"].mean(), 2) if len(closed) else float("nan"),
        "avg_win_pct": round(wins["ReturnPct"].mean(), 2) if len(wins) else float("nan"),
        "avg_loss_pct": round(losses["ReturnPct"].mean(), 2) if len(losses) else float("nan"),
        "max_drawdown_pct": round(float(max_drawdown), 2) if pd.notna(max_drawdown) else float("nan"),
    }


def equity_curve_from_trades(closed_trades: pd.DataFrame) -> pd.Series:
    """Cumulative growth of 1 unit of capital, compounding each closed trade's
    return in EntryDate order -- i.e. "if you risked your whole stack on each
    signal in turn." A simplification (real trades overlap across symbols),
    but standard for a single equity-curve read on a multi-symbol signal."""
    if closed_trades.empty:
        return pd.Series(dtype=float)
    ordered = closed_trades.sort_values("EntryDate")
    return (1 + ordered["ReturnPct"] / 100).cumprod()


def drawdown_pct(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max * 100
