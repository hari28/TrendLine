"""Backtest for the intraday 0DTE Iron Condor strategy discussed with the user.

IMPORTANT scoping note (read before trusting the numbers): TrendLine has no
real options-chain data source at all -- no strikes, no quoted premiums, no
implied volatility, no Greeks. Yahoo (the app's only data source) doesn't
serve NSE index options. So this cannot be a real historical options
backtest. What it does instead: prices a synthetic Iron Condor via
Black-Scholes, fed by the actual Nifty/Bank Nifty INDEX price history, using
trailing REALIZED volatility (20-day annualized stdev of log returns) as a
stand-in for implied volatility. This is a standard technique when real
options history isn't available, but it has real, disclosed limitations:
  - Realized vol is typically LOWER than the market's actual implied vol
    (IV carries a volatility risk premium) -- so this likely UNDERSTATES
    real-world premium collected and overstates how often a "stop-loss"
    would trigger, since real premiums would be richer.
  - No volatility skew is modeled (flat vol across all strikes) -- real NSE
    index options skew, usually pricing OTM puts richer than OTM calls.
  - No intraday IV crush is modeled -- real 0DTE options often see IV drop
    through the session as uncertainty resolves, which would make a real
    short premium position look BETTER than this model shows late in the day.
  - Entry/exit use 15-min bar closes as the observed spot, not tick-level
    prices, and Yahoo only retains ~60 days of 15-min history, so this can
    only ever cover the last ~60 trading days (weekly-expiry days within
    that window), not a multi-year window like the other backtests.
  - "Expiry weekday" isn't looked up from an actual NSE contract calendar
    (not available from this data source) -- it's a user-supplied parameter,
    since NSE has changed weekly-expiry weekdays by circular more than once.

Treat every number here as "what a textbook Black-Scholes model, fed
historical realized vol, would have produced" -- a reasonable proxy for
gut-checking the STRATEGY'S SHAPE (does selling ~15-20 delta strikes with a
defined-risk hedge and a hard stop actually behave the way the strategy
logic intends?), not a promise of real trading P&L.

Strategy rules being modeled:
  - Entry: ENTRY_TIME (09:45 IST) on each matching expiry weekday. Sell a
    call and a put each at SHORT_DELTA_TARGET delta (~0.15-0.20), buy a
    further-OTM call and put SPREAD_WIDTH points out as protection (the
    classic Iron Condor -- defined max loss, not a naked strangle).
  - Exit: whichever comes first --
      (a) mark-to-market cost to close the structure reaches
          STOP_LOSS_MULTIPLE x the credit received (cutting losers before
          they reach the full defined-risk max loss), or
      (b) EXIT_TIME (15:15 IST, effectively end of session for a 0DTE
          contract) -- forced flat, no overnight risk ever.
  - One condor per expiry day (no re-entry same day after a stop-out, since
    a 0DTE stop-out on an index usually means a trending day where selling
    premium again is fighting the tape).
"""
import math

import pandas as pd

from screener import load_frame

ENTRY_TIME = "09:45"
EXIT_TIME = "15:15"
SHORT_DELTA_TARGET = 0.175
STOP_LOSS_MULTIPLE = 1.75
VOL_LOOKBACK = 20  # trading days, for the realized-vol (IV proxy) estimate
RISK_FREE_RATE = 0.065
TRADING_MINUTES_PER_DAY = 375  # 09:15-15:30 IST
TRADING_DAYS_PER_YEAR = 252

INSTRUMENT_CONFIG = {
    "NIFTY 50": {"strike_step": 50, "spread_width": 150},
    "NIFTY BANK": {"strike_step": 100, "spread_width": 300},
}
EXPIRY_WEEKDAY_MAP = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(spot: float, strike: float, t_years: float, sigma: float, option_type: str,
              r: float = RISK_FREE_RATE) -> float:
    if t_years <= 1e-8 or sigma <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "call":
        return spot * _normal_cdf(d1) - strike * math.exp(-r * t_years) * _normal_cdf(d2)
    return strike * math.exp(-r * t_years) * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def _bs_delta(spot: float, strike: float, t_years: float, sigma: float, option_type: str,
              r: float = RISK_FREE_RATE) -> float:
    if t_years <= 1e-8 or sigma <= 0:
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * math.sqrt(t_years))
    return _normal_cdf(d1) if option_type == "call" else _normal_cdf(d1) - 1.0


def _find_strike_for_delta(spot: float, t_years: float, sigma: float, target_delta: float, option_type: str,
                            strike_step: int, search_range_pct: float = 0.08) -> int | None:
    lo = int((spot * (1 - search_range_pct)) // strike_step * strike_step)
    hi = int((spot * (1 + search_range_pct)) // strike_step * strike_step) + strike_step
    best_strike, best_diff = None, float("inf")
    k = lo
    while k <= hi:
        if (option_type == "call" and k > spot) or (option_type == "put" and k < spot):
            diff = abs(abs(_bs_delta(spot, k, t_years, sigma, option_type)) - target_delta)
            if diff < best_diff:
                best_diff, best_strike = diff, k
        k += strike_step
    return best_strike


def _time_fraction_remaining(time_str: str, exit_time: str = EXIT_TIME) -> float:
    cur_h, cur_m = (int(x) for x in time_str.split(":"))
    exit_h, exit_m = (int(x) for x in exit_time.split(":"))
    minutes_remaining = max((exit_h * 60 + exit_m) - (cur_h * 60 + cur_m), 0)
    return (minutes_remaining / TRADING_MINUTES_PER_DAY) / TRADING_DAYS_PER_YEAR


def _realized_vol_series(daily_close: pd.Series, lookback: int = VOL_LOOKBACK) -> pd.Series:
    log_ret = (daily_close / daily_close.shift(1)).apply(math.log)
    return log_ret.rolling(window=lookback, min_periods=lookback).std() * math.sqrt(TRADING_DAYS_PER_YEAR)


def _condor_cost_to_close(spot: float, t_years: float, sigma: float, short_call_k: float, long_call_k: float,
                           short_put_k: float, long_put_k: float) -> float:
    sc = _bs_price(spot, short_call_k, t_years, sigma, "call")
    lc = _bs_price(spot, long_call_k, t_years, sigma, "call")
    sp = _bs_price(spot, short_put_k, t_years, sigma, "put")
    lp = _bs_price(spot, long_put_k, t_years, sigma, "put")
    return (sc - lc) + (sp - lp)


def _simulate_day(day_bars: pd.DataFrame, sigma: float, cfg: dict) -> dict | None:
    entry_matches = day_bars[day_bars.index.strftime("%H:%M") == ENTRY_TIME]
    if entry_matches.empty or pd.isna(sigma) or sigma <= 0:
        return None
    entry_pos = day_bars.index.get_loc(entry_matches.index[0])
    spot_entry = float(day_bars["Close"].iloc[entry_pos])
    t_entry = _time_fraction_remaining(ENTRY_TIME)
    if t_entry <= 0:
        return None

    step, width = cfg["strike_step"], cfg["spread_width"]
    short_call_k = _find_strike_for_delta(spot_entry, t_entry, sigma, SHORT_DELTA_TARGET, "call", step)
    short_put_k = _find_strike_for_delta(spot_entry, t_entry, sigma, SHORT_DELTA_TARGET, "put", step)
    if short_call_k is None or short_put_k is None:
        return None
    long_call_k, long_put_k = short_call_k + width, short_put_k - width

    credit = _condor_cost_to_close(spot_entry, t_entry, sigma, short_call_k, long_call_k, short_put_k, long_put_k)
    if credit <= 0:
        return None
    max_loss = width - credit
    if max_loss <= 0:
        return None
    stop_loss_value = credit * STOP_LOSS_MULTIPLE

    exit_reason = exit_spot = exit_time = exit_cost = None
    for i in range(entry_pos + 1, len(day_bars)):
        t_str = day_bars.index[i].strftime("%H:%M")
        t_now = _time_fraction_remaining(t_str)
        spot_now = float(day_bars["Close"].iloc[i])
        cost_now = _condor_cost_to_close(spot_now, t_now, sigma, short_call_k, long_call_k, short_put_k, long_put_k)
        hit_stop = cost_now >= stop_loss_value
        if hit_stop or t_str >= EXIT_TIME:
            exit_reason = "Stop-loss" if hit_stop else "Session close"
            exit_spot, exit_time, exit_cost = spot_now, day_bars.index[i], cost_now
            break

    if exit_reason is None:
        exit_time = day_bars.index[-1]
        exit_spot = float(day_bars["Close"].iloc[-1])
        exit_cost = _condor_cost_to_close(exit_spot, _time_fraction_remaining(exit_time.strftime("%H:%M")), sigma,
                                           short_call_k, long_call_k, short_put_k, long_put_k)
        exit_reason = "Session close"

    pnl = credit - exit_cost
    return {
        "EntryDate": day_bars.index[entry_pos].date(), "ExitDate": exit_time.date(),
        "EntryTime": day_bars.index[entry_pos], "ExitTime": exit_time,
        "SpotEntry": round(spot_entry, 2), "SpotExit": round(exit_spot, 2),
        "ShortCall": short_call_k, "LongCall": long_call_k,
        "ShortPut": short_put_k, "LongPut": long_put_k,
        "SigmaPct": round(sigma * 100, 1),
        "CreditPts": round(credit, 2), "MaxLossPts": round(max_loss, 2), "PnLPts": round(pnl, 2),
        "RiskPct": round(max_loss / spot_entry * 100, 2),
        "ReturnPct": round(pnl / max_loss * 100, 2),
        "ExitReason": exit_reason,
    }


def backtest_iron_condor(symbol: str, expiry_weekday: str = "Thursday", start_date: str = "2020-01-01",
                          force_refresh: bool = False) -> list[dict]:
    if symbol not in INSTRUMENT_CONFIG:
        raise ValueError(f"Unsupported symbol {symbol!r} -- choose one of {list(INSTRUMENT_CONFIG)}")
    cfg = INSTRUMENT_CONFIG[symbol]
    target_weekday = EXPIRY_WEEKDAY_MAP[expiry_weekday]

    daily = load_frame(symbol, "1D", force_refresh=force_refresh)
    intraday = load_frame(symbol, "15MIN", force_refresh=force_refresh)
    if daily is None or intraday is None or intraday.empty:
        return []

    sigma_shifted = _realized_vol_series(daily["Close"]).shift(1)
    sigma_by_date = {ts.date(): float(v) for ts, v in sigma_shifted.items() if pd.notna(v)}
    start = pd.Timestamp(start_date).date()

    trades = []
    for date, day_bars in intraday.groupby(intraday.index.date):
        if date.weekday() != target_weekday or date < start:
            continue
        sigma = sigma_by_date.get(date)
        if sigma is None:
            continue
        trade = _simulate_day(day_bars, sigma, cfg)
        if trade:
            trade["Symbol"] = symbol
            trades.append(trade)

    return trades


def run_backtest(symbol: str, expiry_weekday: str = "Thursday", start_date: str = "2020-01-01",
                  force_refresh: bool = False) -> pd.DataFrame:
    trades = backtest_iron_condor(symbol, expiry_weekday, start_date, force_refresh=force_refresh)
    df = pd.DataFrame(trades)
    if not df.empty:
        df = df.sort_values("EntryDate").reset_index(drop=True)
    return df


def simulate_portfolio(trades: pd.DataFrame, risk_per_trade_pct: float = 1.0,
                        starting_capital: float = 100.0) -> dict:
    """Deliberately NOT swing_backtest.simulate_portfolio() -- that function's
    same-day tie-break (process an exit event before an entry event on a
    tied date, correct for multi-day swing trades where that tie is rare) is
    wrong here: every single trade in this module both enters AND exits on
    the same calendar date by construction (0DTE), so that ordering would
    close each trade before it ever opened. Since trades here never overlap
    (one condor per expiry day, always flat by session close), this is just
    sequential compounding -- no concurrency model needed at all. ReturnPct
    is already expressed relative to that trade's own max loss, so scaling
    by a fixed risk_amount directly gives the position's P&L in rupees."""
    if trades.empty:
        return {"final_capital": starting_capital, "total_return_pct": 0.0, "max_drawdown_pct": float("nan"),
                "equity_curve": pd.Series(dtype=float)}

    ordered = trades.sort_values("EntryDate")
    capital = starting_capital
    dates, values = [], []
    for _, row in ordered.iterrows():
        risk_amount = capital * risk_per_trade_pct / 100.0
        capital += risk_amount * (row["ReturnPct"] / 100.0)
        dates.append(pd.Timestamp(row["EntryDate"]))
        values.append(capital)

    equity = pd.Series(values, index=dates)
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100

    return {
        "final_capital": round(capital, 2),
        "total_return_pct": round((capital - starting_capital) / starting_capital * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()), 2) if not drawdown.empty else float("nan"),
        "equity_curve": equity,
    }
