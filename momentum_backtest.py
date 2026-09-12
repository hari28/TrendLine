"""Backtest for momentum_screener.py's entry rules.

The screener only defines ENTRY conditions (see momentum_screener.py's
docstring) -- no exit/stop/target rule was specified. This applies the same
exit house-style already established for this app's Swing Strategy backtest
(swing_backtest.py), for consistency: stop-loss = Entry - 1.5x ATR(14), skip
the trade (don't shrink the stop) if that implies >4.5% risk, target =
max(2x actual risk, 5%). Weekly-mode trades size their stop off WEEKLY
ATR(14) (not part of the screener's weekly entry rule, added here purely
for exit sizing).

A trade enters on any day/week the full rule set passes, as long as no
prior trade on that symbol is still open (same "one trade at a time"
convention as every other backtest module in this app) -- there's no
"only on a fresh transition" gate here, matching backtest.py /
cpr_ema_backtest.py / swing_backtest.py's precedent, not universe_digest.py's
live-alert dedup (a different concern for actual Telegram alerts).

summarize() and simulate_portfolio() are reused as-is from swing_backtest.py
-- they're generic trade-DataFrame analytics (win rate, expectancy,
concurrency-capped no-leverage portfolio simulation) that only need
EntryDate/ExitDate/ExitReason/ReturnPct/RiskPct columns, which these trades
provide.
"""
import pandas as pd

from indicators import atr, rsi, sma
from screener import load_frame
from swing_backtest import simulate_portfolio, summarize  # noqa: F401 (re-exported for callers)

VOL_AVG_PERIOD = 20
VOLUME_MULTIPLE = 1.5
PRICE_MIN, PRICE_MAX = 50.0, 5000.0
ATR_PERIOD = 14
ATR_PCT_MIN = 1.5
SMA_FAST, SMA_SLOW = 20, 50
RSI_PERIOD = 14
RSI_MIN, RSI_MAX = 40.0, 70.0
RETURN_LOOKBACK = 5

WEEKLY_SMA_FAST, WEEKLY_SMA_SLOW = 20, 200
WEEKLY_VOL_AVG_PERIOD = 20
WEEKLY_RSI_MIN, WEEKLY_RSI_MAX = 45.0, 65.0

NIFTY50_SYMBOL = "NIFTY 50"

ATR_MULTIPLE = 1.5
MAX_RISK_PCT = 4.5
MIN_REWARD_PCT = 5.0
RR_MULTIPLE = 2.0

STRATEGY_LABELS = {"daily": "Momentum Screener (Daily)", "weekly": "Momentum Screener (Weekly)"}


def _finalize_stop_target(entry_price: float, atr_value: float) -> tuple | None:
    """Same rule as swing_backtest.py's _finalize_stop_target, long-only."""
    if pd.isna(atr_value) or atr_value <= 0:
        return None
    risk_distance = ATR_MULTIPLE * atr_value
    risk_pct = risk_distance / entry_price * 100.0
    if risk_pct <= 0 or risk_pct > MAX_RISK_PCT:
        return None
    reward_pct = max(RR_MULTIPLE * risk_pct, MIN_REWARD_PCT)
    stop_price = entry_price - risk_distance
    target_price = entry_price * (1 + reward_pct / 100.0)
    return stop_price, target_price, risk_pct, reward_pct


def _scan_exit(df: pd.DataFrame, start_idx: int, stop_price: float, target_price: float) -> tuple:
    for k in range(start_idx, len(df)):
        low, high = df["Low"].iloc[k], df["High"].iloc[k]
        if low <= stop_price:
            return k, stop_price, "Stop-loss"
        if high >= target_price:
            return k, target_price, "Target"
    return None, None, None


def _make_trade(symbol, segment, mode, entry_idx, entry_date, entry_price, stop_price, target_price,
                 risk_pct, reward_pct, df) -> tuple[dict, int]:
    exit_idx, exit_price, exit_reason = _scan_exit(df, entry_idx + 1, stop_price, target_price)
    if exit_idx is None:
        exit_idx = len(df) - 1
        exit_price = float(df["Close"].iloc[-1])
        exit_reason = "Open (end of data)"
    pct_return = (exit_price - entry_price) / entry_price * 100.0
    trade = {
        "Strategy": STRATEGY_LABELS[mode],
        "Symbol": symbol, "Segment": segment,
        "EntryDate": entry_date, "EntryPrice": round(entry_price, 2),
        "StopPrice": round(stop_price, 2), "TargetPrice": round(target_price, 2),
        "RiskPct": round(risk_pct, 2), "RewardPct": round(reward_pct, 2),
        "ExitDate": df.index[exit_idx].date(), "ExitPrice": round(float(exit_price), 2),
        "ExitReason": exit_reason, "ReturnPct": round(float(pct_return), 2),
        "HoldingBars": exit_idx - entry_idx,
    }
    return trade, exit_idx


def _prepare_daily_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["VolAvg20"] = df["Volume"].shift(1).rolling(window=VOL_AVG_PERIOD, min_periods=VOL_AVG_PERIOD).mean()
    df["SMA20"] = sma(df["Close"], SMA_FAST)
    df["SMA50"] = sma(df["Close"], SMA_SLOW)
    df["ATR14"] = atr(df, ATR_PERIOD)
    df["RSI14"] = rsi(df["Close"], RSI_PERIOD)
    df["Return5D"] = df["Close"].pct_change(RETURN_LOOKBACK) * 100.0
    return df


def _nifty50_return_series(force_refresh: bool = False) -> pd.Series:
    frame = load_frame(NIFTY50_SYMBOL, "1D", force_refresh=force_refresh)
    if frame is None:
        return pd.Series(dtype=float)
    return frame["Close"].pct_change(RETURN_LOOKBACK) * 100.0


def _passes_daily(row: pd.Series) -> bool:
    if any(pd.isna(row[c]) for c in ("VolAvg20", "SMA20", "SMA50", "ATR14", "RSI14", "Return5D", "NiftyReturn5D")):
        return False
    atr_pct = row["ATR14"] / row["Close"] * 100.0
    return (
        row["Volume"] > VOLUME_MULTIPLE * row["VolAvg20"]
        and PRICE_MIN <= row["Close"] <= PRICE_MAX
        and atr_pct > ATR_PCT_MIN
        and row["Close"] > row["SMA20"] and row["Close"] > row["SMA50"]
        and RSI_MIN <= row["RSI14"] <= RSI_MAX
        and row["Return5D"] > row["NiftyReturn5D"]
    )


def backtest_symbol_daily(symbol: str, segment: str, start_date: str = "2020-01-01",
                           force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    min_bars = max(SMA_SLOW, RSI_PERIOD, VOL_AVG_PERIOD, ATR_PERIOD) + RETURN_LOOKBACK + 5
    if frame is None or len(frame) < min_bars:
        return []

    df = _prepare_daily_indicators(frame)
    nifty_series = _nifty50_return_series(force_refresh=force_refresh).reindex(df.index).ffill()
    df["NiftyReturn5D"] = nifty_series
    df = df.dropna(subset=["VolAvg20", "SMA20", "SMA50", "ATR14", "RSI14", "Return5D", "NiftyReturn5D"])
    df = df[df.index >= pd.Timestamp(start_date)]
    if df.empty:
        return []

    trades = []
    in_trade_until = -1
    for i in range(len(df)):
        if i <= in_trade_until:
            continue
        row = df.iloc[i]
        if not _passes_daily(row):
            continue

        entry_price = float(row["Close"])
        result = _finalize_stop_target(entry_price, float(row["ATR14"]))
        if result is None:
            continue
        stop_price, target_price, risk_pct, reward_pct = result

        trade, exit_idx = _make_trade(symbol, segment, "daily", i, df.index[i].date(), entry_price,
                                       stop_price, target_price, risk_pct, reward_pct, df)
        trades.append(trade)
        in_trade_until = exit_idx

    return trades


def _prepare_weekly_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["SMA20W"] = sma(df["Close"], WEEKLY_SMA_FAST)
    df["SMA200W"] = sma(df["Close"], WEEKLY_SMA_SLOW)
    df["VolAvg20W"] = df["Volume"].shift(1).rolling(window=WEEKLY_VOL_AVG_PERIOD,
                                                      min_periods=WEEKLY_VOL_AVG_PERIOD).mean()
    df["RSI14"] = rsi(df["Close"], RSI_PERIOD)
    df["ATR14"] = atr(df, ATR_PERIOD)  # not part of the screener's weekly rule -- used only to size the exit
    return df


def _passes_weekly(row: pd.Series) -> bool:
    if any(pd.isna(row[c]) for c in ("SMA20W", "SMA200W", "VolAvg20W", "RSI14", "ATR14")):
        return False
    return (
        row["SMA20W"] > row["SMA200W"]
        and row["Volume"] > row["VolAvg20W"]
        and WEEKLY_RSI_MIN <= row["RSI14"] <= WEEKLY_RSI_MAX
    )


def backtest_symbol_weekly(symbol: str, segment: str, start_date: str = "2020-01-01",
                            force_refresh: bool = False) -> list[dict]:
    frame = load_frame(symbol, "1W", force_refresh=force_refresh)
    min_bars = WEEKLY_SMA_SLOW + WEEKLY_VOL_AVG_PERIOD + 5
    if frame is None or len(frame) < min_bars:
        return []

    df = _prepare_weekly_indicators(frame)
    df = df.dropna(subset=["SMA20W", "SMA200W", "VolAvg20W", "RSI14", "ATR14"])
    df = df[df.index >= pd.Timestamp(start_date)]
    if df.empty:
        return []

    trades = []
    in_trade_until = -1
    for i in range(len(df)):
        if i <= in_trade_until:
            continue
        row = df.iloc[i]
        if not _passes_weekly(row):
            continue

        entry_price = float(row["Close"])
        result = _finalize_stop_target(entry_price, float(row["ATR14"]))
        if result is None:
            continue
        stop_price, target_price, risk_pct, reward_pct = result

        trade, exit_idx = _make_trade(symbol, segment, "weekly", i, df.index[i].date(), entry_price,
                                       stop_price, target_price, risk_pct, reward_pct, df)
        trades.append(trade)
        in_trade_until = exit_idx

    return trades


def run_backtest(mode: str, symbols: list[tuple[str, str]], start_date: str = "2020-01-01", progress_cb=None,
                  force_refresh: bool = False) -> pd.DataFrame:
    """mode: "daily", "weekly", or "both". symbols: list of (Symbol, Segment)."""
    fn_by_mode = {"daily": backtest_symbol_daily, "weekly": backtest_symbol_weekly}
    modes = ["daily", "weekly"] if mode == "both" else [mode]

    all_trades = []
    total = len(symbols) * len(modes)
    done = 0
    for m in modes:
        fn = fn_by_mode[m]
        for symbol, segment in symbols:
            if progress_cb:
                progress_cb(done, total, symbol)
            all_trades.extend(fn(symbol, segment, start_date=start_date, force_refresh=force_refresh))
            done += 1

    trades = pd.DataFrame(all_trades)
    if not trades.empty:
        trades = trades.sort_values(["Strategy", "EntryDate"]).reset_index(drop=True)
    return trades
