"""Oliver's 20 & 200 MA Cross backtest.

Strategy (user-specified, 2026-09-15):
- LONG entry: the fast MA (default 20-period) crosses ABOVE the slow MA
  (default 200-period) -- a fresh crossover event, not a persisting state
  filter (unlike the EMA20/200 filter in cpr_ema_backtest_intraday.py).
  SHORT entry: fast MA crosses BELOW the slow MA. Classic Golden Cross /
  Death Cross, applied on 5-minute intraday bars.
- MA type is user-selectable per run: SMA (classic convention) or EMA.
- Exit: fixed stop-loss and target, whichever is hit first, checked
  bar-by-bar from the bar after entry (conservative: if a bar's High/Low
  range touches both, stop-loss is assumed to hit first). Entry fill is
  assumed at the crossover bar's own close -- same simplification used in
  cpr_ema_backtest_intraday.py. SL/target can be set either as a % of the
  entry price, or as absolute points (e.g. NIFTY entry 24000, 20-point
  target -> exit at 24020) -- the user's own convention for index trading,
  where a flat point distance is more natural than a %.
- One trade at a time per symbol: a new crossover signal is ignored while
  a prior trade on that symbol is still open.

Data-source note: Kite's MCP bridge (mcp.kite.trade) only works inside a
Claude session, not from this Streamlit process directly (same constraint
documented in cpr_ema_backtest_intraday.py). So 5-min history is pulled
ONCE per symbol via Kite MCP in a Claude session and cached to
data/kite_5min/<SYMBOL>.csv -- this tab reads that CSV, it does not fetch
live from Kite itself. Ask Claude to re-pull it for a fresher/longer window.
"""
import os

import pandas as pd

from indicators import sma, ema

FAST_PERIOD, SLOW_PERIOD = 20, 200
DEFAULT_SL_PCT, DEFAULT_TARGET_PCT = 1.0, 2.0
DEFAULT_SL_POINTS, DEFAULT_TARGET_POINTS = 20.0, 40.0
KITE_5MIN_DIR = os.path.join(os.path.dirname(__file__), "data", "kite_5min")


def load_kite_5min(symbol: str) -> pd.DataFrame | None:
    path = f"{KITE_5MIN_DIR}/{symbol}.csv"
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except FileNotFoundError:
        return None
    df.index.name = "Date"
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    return df.sort_index()


def _ma(series: pd.Series, ma_type: str, period: int) -> pd.Series:
    return ema(series, period) if ma_type == "EMA" else sma(series, period)


def find_crossovers(frame: pd.DataFrame, ma_type: str, fast: int, slow: int) -> list[tuple[int, str]]:
    """Integer positions where the fast MA crosses above (Long) or below (Short) the slow MA."""
    close = frame["Close"]
    fast_ma = _ma(close, ma_type, fast)
    slow_ma = _ma(close, ma_type, slow)
    diff = fast_ma - slow_ma

    events = []
    for i in range(1, len(diff)):
        prev, cur = diff.iloc[i - 1], diff.iloc[i]
        if pd.isna(prev) or pd.isna(cur):
            continue
        if prev <= 0 and cur > 0:
            events.append((i, "Long"))
        elif prev >= 0 and cur < 0:
            events.append((i, "Short"))
    return events


def backtest_symbol(symbol: str, ma_type: str = "EMA", fast: int = FAST_PERIOD, slow: int = SLOW_PERIOD,
                     sl_value: float = DEFAULT_SL_PCT, target_value: float = DEFAULT_TARGET_PCT,
                     unit: str = "pct") -> list[dict]:
    """unit: "pct" (sl_value/target_value are % of entry price) or "points"
    (sl_value/target_value are an absolute price distance, e.g. NIFTY points)."""
    frame = load_kite_5min(symbol)
    if frame is None or len(frame) < slow + 5:
        return []

    trades = []
    in_trade_until = -1

    for cross_idx, direction in find_crossovers(frame, ma_type, fast, slow):
        if cross_idx <= in_trade_until:
            continue

        entry_price = float(frame["Close"].iloc[cross_idx])
        entry_time = frame.index[cross_idx]

        if unit == "points":
            sl_dist, target_dist = sl_value, target_value
        else:
            sl_dist, target_dist = entry_price * sl_value / 100, entry_price * target_value / 100

        if direction == "Long":
            sl_price = entry_price - sl_dist
            target_price = entry_price + target_dist
        else:
            sl_price = entry_price + sl_dist
            target_price = entry_price - target_dist

        exit_idx, exit_price, exit_reason = None, None, None
        for k in range(cross_idx + 1, len(frame)):
            low, high = frame["Low"].iloc[k], frame["High"].iloc[k]
            if direction == "Long":
                hit_sl = low <= sl_price
                hit_target = high >= target_price
            else:
                hit_sl = high >= sl_price
                hit_target = low <= target_price
            if hit_sl:
                exit_idx, exit_price, exit_reason = k, sl_price, "Stop-loss"
                break
            if hit_target:
                exit_idx, exit_price, exit_reason = k, target_price, "Target"
                break

        if exit_idx is None:
            exit_idx = len(frame) - 1
            exit_price = float(frame["Close"].iloc[-1])
            exit_reason = "Open (end of data)"

        pct_return = (exit_price - entry_price) / entry_price * 100.0
        if direction == "Short":
            pct_return = -pct_return

        trades.append({
            "Symbol": symbol,
            "Direction": direction,
            "MAType": ma_type,
            "EntryTime": entry_time,
            "EntryPrice": round(entry_price, 2),
            "SLPrice": round(sl_price, 2),
            "TargetPrice": round(target_price, 2),
            "ExitTime": frame.index[exit_idx],
            "ExitPrice": round(float(exit_price), 2),
            "ExitReason": exit_reason,
            "ReturnPct": round(pct_return, 2),
            "HoldingBars": exit_idx - cross_idx,
        })
        in_trade_until = exit_idx

    return trades


def run_backtest(symbols: list[str], ma_type: str = "EMA", fast: int = FAST_PERIOD, slow: int = SLOW_PERIOD,
                  sl_value: float = DEFAULT_SL_PCT, target_value: float = DEFAULT_TARGET_PCT,
                  unit: str = "pct") -> pd.DataFrame:
    all_trades = []
    for symbol in symbols:
        all_trades.extend(backtest_symbol(symbol, ma_type=ma_type, fast=fast, slow=slow,
                                           sl_value=sl_value, target_value=target_value, unit=unit))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values("EntryTime").reset_index(drop=True)
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
    return in entry-time order (sequential all-in -- a simplification)."""
    if closed_trades.empty:
        return pd.Series(dtype=float)
    ordered = closed_trades.sort_values("EntryTime")
    return (1 + ordered["ReturnPct"] / 100).cumprod()


def drawdown_pct(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max * 100
