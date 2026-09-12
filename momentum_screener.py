"""Momentum + relative-strength screener.

Daily mode (default) -- a stock must pass ALL of:
  - Yesterday's volume > 1.5x the 20-day average
  - Price between Rs 50 and Rs 5000
  - ATR(14) / price > 1.5% (volatile enough to be worth trading)
  - Close above both the 20-day and 50-day SMA
  - RSI(14) between 40 and 70
  - 5-day return beats the Nifty 50's 5-day return over the same window
    (relative strength -- the stock must be outperforming the index, not
    just moving)

Weekly mode (--weekly) -- same universe, weekly timeframe, a DIFFERENT
(not additive) rule set:
  - 20-week SMA > 200-week SMA
  - Weekly volume > the 20-week average
  - Weekly RSI(14) between 45 and 65

Every scan function returns a metrics row for every symbol with enough
history, with individual *OK boolean columns per condition plus an overall
"Passed" column -- so a near-miss is visible (which condition it failed),
not just silently dropped.
"""
import argparse
import sys

import pandas as pd

from constituents import get_all_symbols
from indicators import atr, rsi, sma
from screener import load_frame

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


def nifty50_return(lookback: int = RETURN_LOOKBACK, force_refresh: bool = False) -> float | None:
    """Nifty 50 index's own N-day return, for the relative-strength filter.
    None if the index data isn't available (never raises)."""
    frame = load_frame(NIFTY50_SYMBOL, "1D", force_refresh=force_refresh)
    if frame is None or len(frame) < lookback + 1:
        return None
    close = frame["Close"]
    prior = float(close.iloc[-1 - lookback])
    return (float(close.iloc[-1]) - prior) / prior * 100.0


def scan_symbol_daily(symbol: str, segment: str, nifty_return_5d: float,
                       force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, "1D", force_refresh=force_refresh)
    min_bars = max(SMA_SLOW, RSI_PERIOD, VOL_AVG_PERIOD, ATR_PERIOD) + RETURN_LOOKBACK + 5
    if frame is None or len(frame) < min_bars:
        return None

    close, volume = frame["Close"], frame["Volume"]
    last_close = float(close.iloc[-1])
    last_volume = float(volume.iloc[-1])
    vol_avg20 = volume.shift(1).rolling(window=VOL_AVG_PERIOD, min_periods=VOL_AVG_PERIOD).mean().iloc[-1]
    sma20 = sma(close, SMA_FAST).iloc[-1]
    sma50 = sma(close, SMA_SLOW).iloc[-1]
    atr14 = atr(frame, ATR_PERIOD).iloc[-1]
    rsi14 = rsi(close, RSI_PERIOD).iloc[-1]
    prior_close = float(close.iloc[-1 - RETURN_LOOKBACK])

    if any(pd.isna(v) for v in (vol_avg20, sma20, sma50, atr14, rsi14)) or prior_close <= 0:
        return None

    vol_avg20, sma20, sma50, atr14, rsi14 = map(float, (vol_avg20, sma20, sma50, atr14, rsi14))
    return_5d = (last_close - prior_close) / prior_close * 100.0
    atr_pct = atr14 / last_close * 100.0

    volume_ok = last_volume > VOLUME_MULTIPLE * vol_avg20
    price_ok = PRICE_MIN <= last_close <= PRICE_MAX
    atr_ok = atr_pct > ATR_PCT_MIN
    trend_ok = last_close > sma20 and last_close > sma50
    rsi_ok = RSI_MIN <= rsi14 <= RSI_MAX
    rel_strength_ok = return_5d > nifty_return_5d

    return {
        "Symbol": symbol, "Segment": segment, "AsOf": frame.index[-1], "Close": round(last_close, 2),
        "Volume": last_volume, "VolAvg20": round(vol_avg20, 0), "VolumeOK": volume_ok,
        "PriceOK": price_ok,
        "ATRPct": round(atr_pct, 2), "ATROK": atr_ok,
        "SMA20": round(sma20, 2), "SMA50": round(sma50, 2), "TrendOK": trend_ok,
        "RSI14": round(rsi14, 2), "RSIOK": rsi_ok,
        "Return5D": round(return_5d, 2), "Nifty50Return5D": round(nifty_return_5d, 2),
        "RelStrengthOK": rel_strength_ok,
        "Passed": volume_ok and price_ok and atr_ok and trend_ok and rsi_ok and rel_strength_ok,
    }


def scan_symbol_weekly(symbol: str, segment: str, force_refresh: bool = False) -> dict | None:
    frame = load_frame(symbol, "1W", force_refresh=force_refresh)
    min_bars = WEEKLY_SMA_SLOW + WEEKLY_VOL_AVG_PERIOD + 5
    if frame is None or len(frame) < min_bars:
        return None

    close, volume = frame["Close"], frame["Volume"]
    last_close = float(close.iloc[-1])
    last_volume = float(volume.iloc[-1])
    sma20w = sma(close, WEEKLY_SMA_FAST).iloc[-1]
    sma200w = sma(close, WEEKLY_SMA_SLOW).iloc[-1]
    vol_avg20w = volume.shift(1).rolling(window=WEEKLY_VOL_AVG_PERIOD, min_periods=WEEKLY_VOL_AVG_PERIOD).mean().iloc[-1]
    rsi14w = rsi(close, RSI_PERIOD).iloc[-1]

    if any(pd.isna(v) for v in (sma20w, sma200w, vol_avg20w, rsi14w)):
        return None

    sma20w, sma200w, vol_avg20w, rsi14w = map(float, (sma20w, sma200w, vol_avg20w, rsi14w))
    trend_ok = sma20w > sma200w
    volume_ok = last_volume > vol_avg20w
    rsi_ok = WEEKLY_RSI_MIN <= rsi14w <= WEEKLY_RSI_MAX

    return {
        "Symbol": symbol, "Segment": segment, "AsOf": frame.index[-1], "Close": round(last_close, 2),
        "SMA20W": round(sma20w, 2), "SMA200W": round(sma200w, 2), "TrendOK": trend_ok,
        "Volume": last_volume, "VolAvg20W": round(vol_avg20w, 0), "VolumeOK": volume_ok,
        "RSI14": round(rsi14w, 2), "RSIOK": rsi_ok,
        "Passed": trend_ok and volume_ok and rsi_ok,
    }


def scan_universe_daily(symbols_df: pd.DataFrame, progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    nifty_return_5d = nifty50_return(force_refresh=force_refresh)
    if nifty_return_5d is None:
        raise RuntimeError("Could not fetch Nifty 50 index data -- required for the relative-strength filter.")

    rows = []
    total = len(symbols_df)
    for i, r in enumerate(symbols_df.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_daily(r.Symbol, r.Segment, nifty_return_5d, force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


def scan_universe_weekly(symbols_df: pd.DataFrame, progress_cb=None, force_refresh: bool = False) -> pd.DataFrame:
    rows = []
    total = len(symbols_df)
    for i, r in enumerate(symbols_df.itertuples(index=False)):
        if progress_cb:
            progress_cb(i, total, r.Symbol)
        result = scan_symbol_weekly(r.Symbol, r.Segment, force_refresh=force_refresh)
        if result:
            rows.append(result)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Momentum + relative-strength screener")
    parser.add_argument("--weekly", action="store_true",
                         help="Run the weekly-timeframe rule set instead of the daily one")
    parser.add_argument("--universe", nargs="+",
                         default=["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"],
                         help="Universe segment(s) to scan")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached price history")
    args = parser.parse_args()

    symbols_df = get_all_symbols(args.universe, force_refresh=args.force_refresh)[["Symbol", "Segment"]]

    def _cb(i, total, symbol):
        print(f"\r[{i + 1}/{total}] {symbol}" + " " * 10, end="", file=sys.stderr, flush=True)

    if args.weekly:
        results = scan_universe_weekly(symbols_df, progress_cb=_cb, force_refresh=args.force_refresh)
    else:
        results = scan_universe_daily(symbols_df, progress_cb=_cb, force_refresh=args.force_refresh)
    print(file=sys.stderr)

    if results.empty:
        print("No symbols had enough history to evaluate.")
        return

    passed = results[results["Passed"]]
    print(f"\n{len(passed)} / {len(results)} symbols passed all filters:\n")
    if passed.empty:
        print("(none)")
    else:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(passed.drop(columns=["Passed"]).to_string(index=False))


if __name__ == "__main__":
    main()
