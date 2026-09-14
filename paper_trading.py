"""Live paper-trading engine: runs every strategy built in this app forward
from today with real (well -- simulated-real) capital tracking, so their
live performance can be compared apples-to-apples, without waiting months
for enough live history to judge a strategy by.

Design decisions, stated up front:
  - Each strategy gets its OWN independent starting capital (STARTING_CAPITAL,
    Rs 100,000) -- these are NOT one shared pool. Sharing a pool would let
    whichever strategy fires first "claim" capital and starve the others,
    which would confound the comparison this whole feature exists to make.
  - Each strategy's own ENTRY/EXIT/STOP rules are reused EXACTLY as already
    built and backtested in its own module (backtest.py, cpr_ema_backtest.py,
    swing_backtest.py, momentum_backtest.py, bigmoney_swing_backtest.py,
    scalping_backtest.py) -- this module does not redesign any strategy's
    logic. What IS applied uniformly across all of them is the PORTFOLIO-
    LEVEL discipline: 1% of that strategy's own capital risked per trade
    (sized off that trade's own stop distance), a max-concurrent-positions
    cap, and honest drawdown tracking from a real day-by-day capital ledger
    -- the "best of" position sizing/drawdown control this app has
    validated, applied consistently everywhere.
  - CPR+EMA's own backtest partially books a position (50% at 1R, 50% at
    1.5R) -- replicating that multi-leg state here would meaningfully
    complicate every strategy's exit-checking code for one outlier. Paper
    trading instead treats it as a plain full-position stop/target (SLPrice
    / TargetMid), a disclosed simplification.
  - Seven of the eight strategies (Golden Cross, CPR+EMA, Swing Breakout/
    Pullback, Momentum Daily/Weekly, Big Money Swing) hold positions open
    across multiple days, so they're checked ONCE per day, in the post-
    close window -- same convention as universe_digest.py's daily bucket,
    since a day's own bar isn't reliably final before then.
  - EMA Scalping is different: its trades open AND close within the SAME
    session (see scalping_backtest.py) -- by the time the post-close check
    runs, a signal from today has typically already resolved to its own
    exit. So there's no "open position" to carry across days for it; each
    day's check (run_same_day_cycle, not run_daily_cycle) just logs
    whatever trades that day's backtest run produced with EntryDate ==
    today directly as closed trades, entry and exit together.
  - Signal detection for the 7 non-scalping strategies reuses each
    strategy's own backtest_symbol_* function with start_date set to today
    (or this week, for weekly strategies) -- if it returns a trade dated
    today, that's a live signal, using the exact same, already-tested logic
    as the historical backtest with zero duplicated rules. EMA Scalping
    does the same thing but keeps the whole trade dict since it already
    carries entry AND exit fields.

State lives under data/paper_trading/<strategy>/ -- state.json (capital,
high-water mark, open positions) and trades.csv (closed trade log, append-
only, same convention as call_log.csv).
"""
import csv
import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

import backtest as golden_cross_backtest
import bigmoney_swing_backtest
import cpr_ema_backtest
import momentum_backtest
import scalping_backtest
import swing_backtest
from constituents import get_all_symbols
from screener import load_frame

IST = ZoneInfo("Asia/Kolkata")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "paper_trading")

STARTING_CAPITAL = 100_000.0
RISK_PER_TRADE_PCT = 1.0
MAX_CONCURRENT_EQUITY = 20

GOLDEN_CROSS_UNIVERSE = ["Nifty 100 (Large Cap)"]
CPR_EMA_UNIVERSE = [
    ("RELIANCE", "Nifty 100"), ("HDFCBANK", "Nifty 100"), ("ICICIBANK", "Nifty 100"),
    ("INFY", "Nifty 100"), ("TCS", "Nifty 100"), ("SBIN", "Nifty 100"), ("TATASTEEL", "Nifty 100"),
    ("LT", "Nifty 100"), ("AXISBANK", "Nifty 100"), ("MARUTI", "Nifty 100"),
]
SWING_MOMENTUM_UNIVERSE = ["Nifty 100 (Large Cap)", "Nifty Midcap 150"]
BIGMONEY_UNIVERSE = ["Nifty 100 (Large Cap)", "Nifty Midcap 150"]
SCALPING_UNIVERSE = ["Nifty 100 (Large Cap)"]
SCALPING_TIMEFRAME = "5MIN"

STRATEGIES = ["golden_cross", "cpr_ema", "swing_breakout", "swing_pullback",
              "momentum_daily", "momentum_weekly", "bigmoney_swing", "ema_scalping"]
DAILY_BAR_STRATEGIES = ["golden_cross", "cpr_ema", "swing_breakout", "swing_pullback",
                          "momentum_daily", "momentum_weekly", "bigmoney_swing"]
SAME_DAY_STRATEGIES = ["ema_scalping"]
STRATEGY_LABELS = {
    "golden_cross": "Golden Cross",
    "cpr_ema": "CPR + EMA",
    "swing_breakout": "Swing: Volatility Breakout",
    "swing_pullback": "Swing: Trend Pullback",
    "momentum_daily": "Momentum Screener (Daily)",
    "momentum_weekly": "Momentum Screener (Weekly)",
    "bigmoney_swing": "Big Money Swing (Approx.)",
    "ema_scalping": "EMA Scalping",
}

TRADE_COLUMNS = ["strategy", "symbol", "direction", "entry_date", "entry_price", "stop_price",
                  "target_price", "risk_pct", "exit_date", "exit_price", "exit_reason", "return_pct",
                  "pnl_rupees", "capital_after"]


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _strategy_dir(strategy: str) -> str:
    d = os.path.join(DATA_DIR, strategy)
    os.makedirs(d, exist_ok=True)
    return d


def _state_path(strategy: str) -> str:
    return os.path.join(_strategy_dir(strategy), "state.json")


def _trades_path(strategy: str) -> str:
    return os.path.join(_strategy_dir(strategy), "trades.csv")


def load_state(strategy: str) -> dict:
    path = _state_path(strategy)
    if not os.path.exists(path):
        return {"capital": STARTING_CAPITAL, "high_water_mark": STARTING_CAPITAL, "open_positions": [],
                "last_checked_date": None}
    with open(path) as f:
        state = json.load(f)
    return state


def save_state(strategy: str, state: dict) -> None:
    path = _state_path(strategy)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)


def log_trade(strategy: str, row: dict) -> None:
    path = _trades_path(strategy)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TRADE_COLUMNS})


def load_trades(strategy: str) -> pd.DataFrame:
    path = _trades_path(strategy)
    if not os.path.exists(path):
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Shared position sizing / exit helpers (daily-bar equity strategies)
# ---------------------------------------------------------------------------

def _open_equity_position(state: dict, symbol: str, entry_date: str, entry_price: float,
                           stop_price: float, target_price: float) -> None:
    risk_pct = abs(entry_price - stop_price) / entry_price * 100.0
    if risk_pct <= 0:
        return
    risk_amount = state["capital"] * RISK_PER_TRADE_PCT / 100.0
    cash_cap = state["capital"] / MAX_CONCURRENT_EQUITY
    position_size_rupees = min(risk_amount / (risk_pct / 100.0), cash_cap)
    state["open_positions"].append({
        "symbol": symbol, "entry_date": entry_date, "entry_price": entry_price,
        "stop_price": stop_price, "target_price": target_price, "risk_pct": round(risk_pct, 2),
        "position_size_rupees": round(position_size_rupees, 2),
    })


def _check_equity_exits(strategy: str, state: dict, force_refresh: bool = False) -> None:
    """Mutates state in place -- closes any open position whose stop/target
    was hit as of the latest available daily bar."""
    still_open = []
    for pos in state["open_positions"]:
        frame = load_frame(pos["symbol"], "1D", force_refresh=force_refresh)
        if frame is None or frame.empty:
            still_open.append(pos)
            continue
        last = frame.iloc[-1]
        low, high = float(last["Low"]), float(last["High"])
        hit_stop = low <= pos["stop_price"]
        hit_target = high >= pos["target_price"]
        if hit_stop or hit_target:
            exit_price = pos["stop_price"] if hit_stop else pos["target_price"]
            exit_reason = "Stop-loss" if hit_stop else "Target"
            return_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100.0
            pnl_rupees = pos["position_size_rupees"] * (return_pct / 100.0)
            state["capital"] += pnl_rupees
            state["high_water_mark"] = max(state["high_water_mark"], state["capital"])
            log_trade(strategy, {
                "strategy": strategy, "symbol": pos["symbol"], "direction": "Long",
                "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
                "stop_price": pos["stop_price"], "target_price": pos["target_price"],
                "risk_pct": pos["risk_pct"], "exit_date": frame.index[-1].date().isoformat(),
                "exit_price": exit_price, "exit_reason": exit_reason, "return_pct": round(return_pct, 2),
                "pnl_rupees": round(pnl_rupees, 2), "capital_after": round(state["capital"], 2),
            })
        else:
            still_open.append(pos)
    state["open_positions"] = still_open


def _open_symbols(state: dict) -> set:
    return {p["symbol"] for p in state["open_positions"]}


# ---------------------------------------------------------------------------
# Per-strategy signal detection -- each reuses its own backtest module's
# already-tested entry logic, checking only for a trade dated today (or
# this week, for weekly strategies).
# ---------------------------------------------------------------------------

def _find_entries_golden_cross(today: date, existing: set, force_refresh: bool = False) -> list[dict]:
    symbols_df = get_all_symbols(GOLDEN_CROSS_UNIVERSE, force_refresh=force_refresh)[["Symbol", "Segment"]]
    entries = []
    for r in symbols_df.itertuples(index=False):
        if r.Symbol in existing:
            continue
        for t in golden_cross_backtest.backtest_symbol(r.Symbol, r.Segment, force_refresh=force_refresh):
            if t["EntryDate"] == today:
                entry_price = t["EntryPrice"]
                entries.append({
                    "symbol": r.Symbol, "entry_price": entry_price,
                    "stop_price": entry_price * (1 - golden_cross_backtest.STOP_PCT),
                    "target_price": entry_price * (1 + golden_cross_backtest.TARGET_PCT),
                })
    return entries


def _find_entries_cpr_ema(today: date, existing: set, force_refresh: bool = False) -> list[dict]:
    entries = []
    for symbol, segment in CPR_EMA_UNIVERSE:
        if symbol in existing:
            continue
        for t in cpr_ema_backtest.backtest_symbol(symbol, segment, force_refresh=force_refresh):
            if t["EntryDate"] == today and t["Direction"] == "Long":
                entries.append({
                    "symbol": symbol, "entry_price": t["EntryPrice"],
                    "stop_price": t["SLPrice"], "target_price": t["TargetMid"],
                })
    return entries


def _find_entries_swing(mode: str, today: date, existing: set, force_refresh: bool = False) -> list[dict]:
    fn = swing_backtest.backtest_symbol_breakout if mode == "breakout" else swing_backtest.backtest_symbol_pullback
    symbols_df = get_all_symbols(SWING_MOMENTUM_UNIVERSE, force_refresh=force_refresh)[["Symbol", "Segment"]]
    entries = []
    for r in symbols_df.itertuples(index=False):
        if r.Symbol in existing:
            continue
        for t in fn(r.Symbol, r.Segment, start_date=today.isoformat(), force_refresh=force_refresh):
            if t["EntryDate"] == today:
                entries.append({"symbol": r.Symbol, "entry_price": t["EntryPrice"],
                                 "stop_price": t["StopPrice"], "target_price": t["TargetPrice"]})
    return entries


def _find_entries_momentum(mode: str, today: date, existing: set, force_refresh: bool = False) -> list[dict]:
    fn = momentum_backtest.backtest_symbol_daily if mode == "daily" else momentum_backtest.backtest_symbol_weekly
    symbols_df = get_all_symbols(SWING_MOMENTUM_UNIVERSE, force_refresh=force_refresh)[["Symbol", "Segment"]]
    entries = []
    for r in symbols_df.itertuples(index=False):
        if r.Symbol in existing:
            continue
        for t in fn(r.Symbol, r.Segment, start_date=today.isoformat(), force_refresh=force_refresh):
            if t["EntryDate"] == today:
                entries.append({"symbol": r.Symbol, "entry_price": t["EntryPrice"],
                                 "stop_price": t["StopPrice"], "target_price": t["TargetPrice"]})
    return entries


def _find_entries_bigmoney(today: date, existing: set, force_refresh: bool = False) -> list[dict]:
    symbols_df = get_all_symbols(BIGMONEY_UNIVERSE, force_refresh=force_refresh)[["Symbol", "Segment"]]
    entries = []
    for r in symbols_df.itertuples(index=False):
        if r.Symbol in existing:
            continue
        for t in bigmoney_swing_backtest.backtest_symbol(r.Symbol, r.Segment, start_date=today.isoformat(),
                                                           force_refresh=force_refresh):
            if t["EntryDate"] == today:
                entries.append({"symbol": r.Symbol, "entry_price": t["EntryPrice"],
                                 "stop_price": t["StopPrice"], "target_price": t["TargetPrice"]})
    return entries


DAILY_ENTRY_FINDERS = {
    "golden_cross": lambda today, existing, fr: _find_entries_golden_cross(today, existing, fr),
    "cpr_ema": lambda today, existing, fr: _find_entries_cpr_ema(today, existing, fr),
    "swing_breakout": lambda today, existing, fr: _find_entries_swing("breakout", today, existing, fr),
    "swing_pullback": lambda today, existing, fr: _find_entries_swing("pullback", today, existing, fr),
    "momentum_daily": lambda today, existing, fr: _find_entries_momentum("daily", today, existing, fr),
    "momentum_weekly": lambda today, existing, fr: _find_entries_momentum("weekly", today, existing, fr),
    "bigmoney_swing": lambda today, existing, fr: _find_entries_bigmoney(today, existing, fr),
}


def run_daily_cycle(strategy: str, force_refresh: bool = False) -> dict:
    """Called once per strategy, once per trading day (post-close). Checks
    existing open positions for exits first, then looks for fresh entries."""
    today = datetime.now(IST).date()
    state = load_state(strategy)
    if state.get("last_checked_date") == today.isoformat():
        return {"ran": False, "reason": "already checked today"}

    _check_equity_exits(strategy, state, force_refresh=force_refresh)

    if len(state["open_positions"]) < MAX_CONCURRENT_EQUITY:
        finder = DAILY_ENTRY_FINDERS[strategy]
        for entry in finder(today, _open_symbols(state), force_refresh):
            if len(state["open_positions"]) >= MAX_CONCURRENT_EQUITY:
                break
            _open_equity_position(state, entry["symbol"], today.isoformat(), entry["entry_price"],
                                   entry["stop_price"], entry["target_price"])

    state["last_checked_date"] = today.isoformat()
    save_state(strategy, state)
    return {"ran": True, "open_positions": len(state["open_positions"]), "capital": state["capital"]}


def run_same_day_cycle(strategy: str, force_refresh: bool = False) -> dict:
    """For strategies whose trades open AND close within one session (EMA
    Scalping, see scalping_backtest.py) -- there's no "open position" to
    carry across cycles the way run_daily_cycle tracks one, since the
    entry's own resolution (stop/target/session-close) has typically
    already happened by the time this runs. Each call just re-runs that
    day's backtest for the configured universe, picks out any trade whose
    EntryDate is today, and logs it directly as a closed trade -- entry and
    exit together, in one step. Idempotent per day via last_checked_date,
    same as run_daily_cycle."""
    today = datetime.now(IST).date()
    state = load_state(strategy)
    if state.get("last_checked_date") == today.isoformat():
        return {"ran": False, "reason": "already checked today"}

    symbols_df = get_all_symbols(SCALPING_UNIVERSE, force_refresh=force_refresh)[["Symbol", "Segment"]]
    today_trades = []
    for r in symbols_df.itertuples(index=False):
        for t in scalping_backtest.backtest_symbol(r.Symbol, r.Segment, timeframe=SCALPING_TIMEFRAME,
                                                     force_refresh=force_refresh):
            if t["EntryDate"] == today:
                today_trades.append((r.Symbol, t))

    for symbol, t in today_trades:
        risk_pct = t["RiskPct"] if t["RiskPct"] > 0 else 0.01
        risk_amount = state["capital"] * RISK_PER_TRADE_PCT / 100.0
        cash_cap = state["capital"] / MAX_CONCURRENT_EQUITY
        position_size_rupees = min(risk_amount / (risk_pct / 100.0), cash_cap)
        pnl_rupees = position_size_rupees * (t["ReturnPct"] / 100.0)
        state["capital"] += pnl_rupees
        state["high_water_mark"] = max(state["high_water_mark"], state["capital"])
        log_trade(strategy, {
            "strategy": strategy, "symbol": symbol, "direction": t["Direction"],
            "entry_date": t["EntryDate"].isoformat(), "entry_price": t["EntryPrice"],
            "stop_price": t["StopPrice"], "target_price": t["TargetPrice"], "risk_pct": t["RiskPct"],
            "exit_date": t["ExitDate"].isoformat(), "exit_price": t["ExitPrice"],
            "exit_reason": t["ExitReason"], "return_pct": t["ReturnPct"],
            "pnl_rupees": round(pnl_rupees, 2), "capital_after": round(state["capital"], 2),
        })

    state["last_checked_date"] = today.isoformat()
    save_state(strategy, state)
    return {"ran": True, "trades_today": len(today_trades), "capital": state["capital"]}


# ---------------------------------------------------------------------------
# Comparative summary
# ---------------------------------------------------------------------------

def summary_row(strategy: str) -> dict:
    state = load_state(strategy)
    trades = load_trades(strategy)
    closed = len(trades)
    wins = (trades["return_pct"] > 0).sum() if closed else 0
    win_rate = wins / closed * 100 if closed else float("nan")
    total_return = (state["capital"] - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    drawdown = (state["capital"] - state["high_water_mark"]) / state["high_water_mark"] * 100 \
        if state["high_water_mark"] > 0 else 0.0
    open_count = len(state["open_positions"])
    return {
        "Strategy": STRATEGY_LABELS[strategy], "Capital": round(state["capital"], 2),
        "Total Return %": round(total_return, 2), "Closed Trades": closed,
        "Win Rate %": round(win_rate, 1) if pd.notna(win_rate) else None,
        "Current Drawdown %": round(drawdown, 2), "Open Positions": open_count,
    }


def comparison_table() -> pd.DataFrame:
    return pd.DataFrame([summary_row(s) for s in STRATEGIES]).sort_values("Total Return %", ascending=False)
