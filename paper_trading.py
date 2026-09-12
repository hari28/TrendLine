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
    swing_backtest.py, momentum_backtest.py, iron_condor_backtest.py,
    iron_condor_range_backtest.py) -- this module does not redesign any
    strategy's logic. What IS applied uniformly across all of them is the
    PORTFOLIO-LEVEL discipline: 1% of that strategy's own capital risked per
    trade (sized off that trade's own stop distance), a max-concurrent-
    positions cap, and honest drawdown tracking from a real day-by-day
    capital ledger -- the "best of" position sizing/drawdown control this
    app has validated, applied consistently everywhere.
  - CPR+EMA's own backtest partially books a position (50% at 1R, 50% at
    1.5R) -- replicating that multi-leg state here would meaningfully
    complicate every strategy's exit-checking code for one outlier. Paper
    trading instead treats it as a plain full-position stop/target (SLPrice
    / TargetMid), a disclosed simplification.
  - Daily-bar strategies (Golden Cross, CPR+EMA, Swing Breakout/Pullback,
    Momentum Daily/Weekly) are checked ONCE per day, in the post-close
    window -- same convention as universe_digest.py's daily bucket, since a
    day's own bar isn't reliably final before then.
  - Both Iron Condor variants are intraday and are checked on EVERY 15-min
    cycle during market hours, since they need to react to same-day entry
    and exit windows, not just once at day's end.
  - Signal detection for the 6 daily/weekly strategies reuses each
    strategy's own backtest_symbol_* function with start_date set to today
    (or this week, for weekly strategies) -- if it returns a trade dated
    today, that's a live signal, using the exact same, already-tested logic
    as the historical backtest with zero duplicated rules.

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
import cpr_ema_backtest
import iron_condor_backtest
import iron_condor_range_backtest
import momentum_backtest
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
IRON_CONDOR_SYMBOL = "NIFTY 50"
IRON_CONDOR_EXPIRY_WEEKDAY = "Thursday"

STRATEGIES = ["golden_cross", "cpr_ema", "swing_breakout", "swing_pullback",
              "momentum_daily", "momentum_weekly", "iron_condor_0dte", "iron_condor_range"]
DAILY_BAR_STRATEGIES = ["golden_cross", "cpr_ema", "swing_breakout", "swing_pullback",
                          "momentum_daily", "momentum_weekly"]
INTRADAY_STRATEGIES = ["iron_condor_0dte", "iron_condor_range"]
STRATEGY_LABELS = {
    "golden_cross": "Golden Cross",
    "cpr_ema": "CPR + EMA",
    "swing_breakout": "Swing: Volatility Breakout",
    "swing_pullback": "Swing: Trend Pullback",
    "momentum_daily": "Momentum Screener (Daily)",
    "momentum_weekly": "Momentum Screener (Weekly)",
    "iron_condor_0dte": "Iron Condor (0DTE)",
    "iron_condor_range": "Iron Condor (Non-Expiry Range)",
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
                "last_checked_date": None, "intraday": {}}
    with open(path) as f:
        state = json.load(f)
    state.setdefault("intraday", {})
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


DAILY_ENTRY_FINDERS = {
    "golden_cross": lambda today, existing, fr: _find_entries_golden_cross(today, existing, fr),
    "cpr_ema": lambda today, existing, fr: _find_entries_cpr_ema(today, existing, fr),
    "swing_breakout": lambda today, existing, fr: _find_entries_swing("breakout", today, existing, fr),
    "swing_pullback": lambda today, existing, fr: _find_entries_swing("pullback", today, existing, fr),
    "momentum_daily": lambda today, existing, fr: _find_entries_momentum("daily", today, existing, fr),
    "momentum_weekly": lambda today, existing, fr: _find_entries_momentum("weekly", today, existing, fr),
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


# ---------------------------------------------------------------------------
# Intraday Iron Condor strategies -- real-time state machine, checked on
# every 15-min cycle during market hours (not once per day).
# ---------------------------------------------------------------------------

def _run_iron_condor_cycle(strategy: str, engine, force_refresh: bool = False) -> dict:
    """`engine` is iron_condor_backtest or iron_condor_range_backtest --
    both expose the same primitives (_condor_cost_to_close, ENTRY_TIME,
    EXIT_TIME, INSTRUMENT_CONFIG, EXPIRY_WEEKDAY_MAP) this reuses directly,
    so the live state machine can never drift from what was backtested."""
    now_ist = datetime.now(IST)
    today = now_ist.date()
    state = load_state(strategy)
    intr = state["intraday"]

    if intr.get("date") != today.isoformat():
        intr = {"date": today.isoformat(), "entered_today": False, "position": None}

    cfg = engine.INSTRUMENT_CONFIG[IRON_CONDOR_SYMBOL]
    target_weekday = engine.EXPIRY_WEEKDAY_MAP[IRON_CONDOR_EXPIRY_WEEKDAY]
    is_expiry_day = today.weekday() == target_weekday

    daily = load_frame(IRON_CONDOR_SYMBOL, "1D", force_refresh=force_refresh)
    intraday = load_frame(IRON_CONDOR_SYMBOL, "15MIN", force_refresh=force_refresh)
    if daily is None or intraday is None or intraday.empty:
        return {"ran": False, "reason": "no price data"}

    today_bars = intraday[intraday.index.date == today]
    if today_bars.empty:
        return {"ran": False, "reason": "no bars for today yet"}
    latest_bar = today_bars.iloc[-1]
    latest_time = today_bars.index[-1].strftime("%H:%M")
    spot_now = float(latest_bar["Close"])

    if strategy == "iron_condor_0dte":
        applicable_today = is_expiry_day
    else:
        applicable_today = (not is_expiry_day) and today.weekday() < 5

    if not applicable_today:
        state["intraday"] = intr
        save_state(strategy, state)
        return {"ran": False, "reason": "not this strategy's day"}

    if intr["position"] is not None:
        pos = intr["position"]
        if strategy == "iron_condor_0dte":
            t_years = engine._time_fraction_remaining(latest_time)
            sigma = pos["sigma"]
        else:
            t_years = pos["days_to_expiry"] / engine.TRADING_DAYS_PER_YEAR
            sigma = pos["sigma_daily"] * engine._time_of_day_vol_multiplier(intraday).get(latest_time, 1.0) \
                if hasattr(engine, "_time_of_day_vol_multiplier") else pos["sigma"]
        cost_now = engine._condor_cost_to_close(spot_now, t_years, sigma, pos["short_call"], pos["long_call"],
                                                 pos["short_put"], pos["long_put"])
        hit_stop = cost_now >= pos["stop_loss_value"]
        exit_time_reached = latest_time >= engine.EXIT_TIME
        if hit_stop or exit_time_reached:
            pnl_pts = pos["credit"] - cost_now
            max_loss = cfg["spread_width"] - pos["credit"]
            return_pct = pnl_pts / max_loss * 100.0 if max_loss > 0 else 0.0
            risk_amount = state["capital"] * RISK_PER_TRADE_PCT / 100.0
            pnl_rupees = risk_amount * (return_pct / 100.0)
            state["capital"] += pnl_rupees
            state["high_water_mark"] = max(state["high_water_mark"], state["capital"])
            log_trade(strategy, {
                "strategy": strategy, "symbol": IRON_CONDOR_SYMBOL, "direction": "Condor",
                "entry_date": pos["entry_date"], "entry_price": pos["spot_entry"],
                "stop_price": pos["stop_loss_value"], "target_price": 0,
                "risk_pct": round(max_loss / pos["spot_entry"] * 100, 2) if pos["spot_entry"] else 0,
                "exit_date": today.isoformat(), "exit_price": spot_now,
                "exit_reason": "Stop-loss" if hit_stop else "Time-based exit",
                "return_pct": round(return_pct, 2), "pnl_rupees": round(pnl_rupees, 2),
                "capital_after": round(state["capital"], 2),
            })
            intr["position"] = None

    elif not intr["entered_today"] and latest_time >= engine.ENTRY_TIME:
        # Same defensive pattern as cpr.py's _basis_row: don't assume today's
        # own row is or isn't in `daily` (Yahoo sometimes includes a live,
        # not-yet-final placeholder for today) -- explicitly drop anything
        # dated today before computing the rolling estimates, so this always
        # uses only fully-completed PRIOR sessions regardless of that.
        daily_prior = daily[daily.index.date < today]
        sigma_daily = engine._realized_vol_series(daily_prior["Close"]).iloc[-1] if len(daily_prior) > 1 else None
        opened = None
        if sigma_daily is not None and pd.notna(sigma_daily) and sigma_daily > 0:
            step, width = cfg["strike_step"], cfg["spread_width"]
            if strategy == "iron_condor_0dte":
                t_entry = engine._time_fraction_remaining(engine.ENTRY_TIME)
                sc = engine._find_strike_for_delta(spot_now, t_entry, sigma_daily, engine.SHORT_DELTA_TARGET,
                                                    "call", step)
                sp = engine._find_strike_for_delta(spot_now, t_entry, sigma_daily, engine.SHORT_DELTA_TARGET,
                                                    "put", step)
                if sc and sp:
                    lc, lp = sc + width, sp - width
                    credit = engine._condor_cost_to_close(spot_now, t_entry, sigma_daily, sc, lc, sp, lp)
                    if credit > 0 and (width - credit) > 0:
                        opened = {"short_call": sc, "long_call": lc, "short_put": sp, "long_put": lp,
                                  "credit": credit, "sigma": sigma_daily,
                                  "stop_loss_value": credit * engine.STOP_LOSS_MULTIPLE,
                                  "spot_entry": spot_now, "entry_date": today.isoformat()}
            else:
                expiry_days = engine._trading_days_until_expiry(today, target_weekday)
                if expiry_days >= engine.MIN_DAYS_TO_EXPIRY:
                    avg_range = engine._avg_range_series(daily_prior).iloc[-1] if len(daily_prior) > 1 else None
                    day_open = float(today_bars["Open"].iloc[0])
                    if avg_range is not None and pd.notna(avg_range) and avg_range > 0:
                        sc = engine._round_to_step(day_open + avg_range, step)
                        sp = engine._round_to_step(day_open - avg_range, step)
                        if sc > spot_now > sp:
                            lc, lp = sc + width, sp - width
                            vol_mult = engine._time_of_day_vol_multiplier(intraday).get(engine.ENTRY_TIME, 1.0)
                            sigma_entry = sigma_daily * vol_mult
                            t_years = expiry_days / engine.TRADING_DAYS_PER_YEAR
                            credit = engine._condor_cost_to_close(spot_now, t_years, sigma_entry, sc, lc, sp, lp)
                            if credit > 0 and (width - credit) > 0:
                                opened = {"short_call": sc, "long_call": lc, "short_put": sp, "long_put": lp,
                                          "credit": credit, "sigma_daily": sigma_daily,
                                          "days_to_expiry": expiry_days,
                                          "stop_loss_value": credit * engine.STOP_LOSS_MULTIPLE,
                                          "spot_entry": spot_now, "entry_date": today.isoformat()}
        intr["entered_today"] = True
        intr["position"] = opened

    state["intraday"] = intr
    save_state(strategy, state)
    return {"ran": True, "in_position": intr["position"] is not None, "capital": state["capital"]}


def run_iron_condor_cycle(strategy: str, force_refresh: bool = False) -> dict:
    engine = iron_condor_backtest if strategy == "iron_condor_0dte" else iron_condor_range_backtest
    return _run_iron_condor_cycle(strategy, engine, force_refresh=force_refresh)


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
    open_count = len(state["open_positions"]) if strategy in DAILY_BAR_STRATEGIES \
        else (1 if state["intraday"].get("position") else 0)
    return {
        "Strategy": STRATEGY_LABELS[strategy], "Capital": round(state["capital"], 2),
        "Total Return %": round(total_return, 2), "Closed Trades": closed,
        "Win Rate %": round(win_rate, 1) if pd.notna(win_rate) else None,
        "Current Drawdown %": round(drawdown, 2), "Open Positions": open_count,
    }


def comparison_table() -> pd.DataFrame:
    return pd.DataFrame([summary_row(s) for s in STRATEGIES]).sort_values("Total Return %", ascending=False)
