"""CPR Scalper: 20/200 MA cross + RSI filter + nearest-CPR-level exit.

Strategy (user-specified, 2026-09-16):
- LONG entry: fast MA (default 20) crosses ABOVE slow MA (default 200) on
  5-min bars -- a fresh crossover event, same convention as
  oliver_ma_cross_backtest.py. SHORT entry: fast MA crosses below slow MA.
- Confirm: RSI(14) must be above `long_rsi_min` (default 45) on the cross
  bar's close for a Long, or below `short_rsi_max` (default 55) for a Short.
  A crossover that fails this filter is simply not traded -- it is not a
  signal, not a skipped trade with a recorded loss.
- Exit (profit target): the nearest CPR level *above* entry for a Long
  (smallest of that trading day's BC/TC/R1/R2), or *nearest below* entry for
  a Short (largest of BC/TC/S1/S2). CPR for a given trading day is derived
  from the *prior* trading day's High/Low/Close within this same 5-min
  dataset (session-based, not repainted intraday) -- deliberately not
  cpr.py's Yahoo-daily version, so entry/exit and the CPR basis both come
  from the one Kite intraday dataset.
- Exit (stop-loss): not specified by the user -- the cross bar's own Low
  (Long) / High (Short), the "optional" line from the CPR Scalper Playbook
  artifact. Whichever of stop/target is hit first wins; a bar touching both
  is conservatively scored as the stop.
- One trade at a time per symbol, same as oliver_ma_cross_backtest.py.

Reuses oliver_ma_cross_backtest.py for data loading, crossover detection,
and the summary/equity-curve helpers so both strategies share one notion of
"trade" and "summary" -- only the RSI filter and CPR-based target/stop are
new.
"""
import pandas as pd

from indicators import rsi
from oliver_ma_cross_backtest import (
    FAST_PERIOD, SLOW_PERIOD, load_kite_5min, find_crossovers,
    summarize, equity_curve_from_trades, drawdown_pct,
)

DEFAULT_RSI_PERIOD = 14
DEFAULT_LONG_RSI_MIN = 45.0
DEFAULT_SHORT_RSI_MAX = 55.0


def compute_pivot_levels(high: float, low: float, close: float) -> dict:
    """One session's H/L/C -> the full CPR + R1/R2/S1/S2 level set.

    TC/BC swap logic matches cpr.py::compute_cpr (Close in the lower half of
    the day's range flips the raw formula's TC/BC order) -- kept in sync
    with that module rather than imported, since this operates on Kite
    intraday-derived daily bars, not cpr.py's Yahoo-daily ones.
    """
    pivot = (high + low + close) / 3.0
    bc_raw = (high + low) / 2.0
    tc_raw = 2 * pivot - bc_raw
    tc, bc = max(tc_raw, bc_raw), min(tc_raw, bc_raw)
    return {
        "Pivot": pivot, "TC": tc, "BC": bc,
        "R1": 2 * pivot - low, "S1": 2 * pivot - high,
        "R2": pivot + (high - low), "S2": pivot - (high - low),
    }


def nearest_resistance(levels: dict, entry: float) -> tuple[str, float] | None:
    candidates = [(k, levels[k]) for k in ("BC", "TC", "R1", "R2") if levels[k] > entry]
    return min(candidates, key=lambda kv: kv[1]) if candidates else None


def nearest_support(levels: dict, entry: float) -> tuple[str, float] | None:
    candidates = [(k, levels[k]) for k in ("BC", "TC", "S1", "S2") if levels[k] < entry]
    return max(candidates, key=lambda kv: kv[1]) if candidates else None


def daily_levels_by_date(frame: pd.DataFrame) -> dict:
    """date -> that day's CPR level set, derived from the *previous* trading
    day's High/Low/Close in this same intraday frame."""
    daily = frame.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna(how="all")
    dates = list(daily.index)
    levels_by_date = {}
    for i in range(1, len(dates)):
        prior = daily.iloc[i - 1]
        levels_by_date[dates[i].date()] = compute_pivot_levels(
            float(prior["High"]), float(prior["Low"]), float(prior["Close"])
        )
    return levels_by_date


def backtest_symbol(symbol: str, ma_type: str = "EMA", fast: int = FAST_PERIOD, slow: int = SLOW_PERIOD,
                     rsi_period: int = DEFAULT_RSI_PERIOD, long_rsi_min: float = DEFAULT_LONG_RSI_MIN,
                     short_rsi_max: float = DEFAULT_SHORT_RSI_MAX) -> list[dict]:
    frame = load_kite_5min(symbol)
    if frame is None or len(frame) < slow + 5:
        return []

    rsi_series = rsi(frame["Close"], rsi_period)
    levels_by_date = daily_levels_by_date(frame)

    trades = []
    in_trade_until = -1

    for cross_idx, direction in find_crossovers(frame, ma_type, fast, slow):
        if cross_idx <= in_trade_until:
            continue

        rsi_val = rsi_series.iloc[cross_idx]
        if pd.isna(rsi_val):
            continue
        if direction == "Long" and rsi_val <= long_rsi_min:
            continue
        if direction == "Short" and rsi_val >= short_rsi_max:
            continue

        entry_time = frame.index[cross_idx]
        levels = levels_by_date.get(entry_time.date())
        if levels is None:
            continue

        entry_price = float(frame["Close"].iloc[cross_idx])
        if direction == "Long":
            target_hit = nearest_resistance(levels, entry_price)
            sl_price = float(frame["Low"].iloc[cross_idx])
        else:
            target_hit = nearest_support(levels, entry_price)
            sl_price = float(frame["High"].iloc[cross_idx])

        if target_hit is None or sl_price == entry_price:
            continue
        target_label, target_price = target_hit

        exit_idx, exit_price, exit_reason = None, None, None
        for k in range(cross_idx + 1, len(frame)):
            low, high = frame["Low"].iloc[k], frame["High"].iloc[k]
            if direction == "Long":
                hit_sl, hit_target = low <= sl_price, high >= target_price
            else:
                hit_sl, hit_target = high >= sl_price, low <= target_price
            if hit_sl:
                exit_idx, exit_price, exit_reason = k, sl_price, "Stop-loss"
                break
            if hit_target:
                exit_idx, exit_price, exit_reason = k, target_price, f"Target ({target_label})"
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
            "RSIAtEntry": round(float(rsi_val), 1),
            "SLPrice": round(sl_price, 2),
            "TargetPrice": round(target_price, 2),
            "TargetLevel": target_label,
            "ExitTime": frame.index[exit_idx],
            "ExitPrice": round(float(exit_price), 2),
            "ExitReason": exit_reason,
            "ReturnPct": round(pct_return, 2),
            "HoldingBars": exit_idx - cross_idx,
        })
        in_trade_until = exit_idx

    return trades


def run_backtest(symbols: list[str], ma_type: str = "EMA", fast: int = FAST_PERIOD, slow: int = SLOW_PERIOD,
                  rsi_period: int = DEFAULT_RSI_PERIOD, long_rsi_min: float = DEFAULT_LONG_RSI_MIN,
                  short_rsi_max: float = DEFAULT_SHORT_RSI_MAX) -> pd.DataFrame:
    all_trades = []
    for symbol in symbols:
        all_trades.extend(backtest_symbol(
            symbol, ma_type=ma_type, fast=fast, slow=slow, rsi_period=rsi_period,
            long_rsi_min=long_rsi_min, short_rsi_max=short_rsi_max,
        ))
    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values("EntryTime").reset_index(drop=True)
    return trades
