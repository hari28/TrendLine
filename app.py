"""
EMA/SMA screener for Nifty Large Cap (100) / Midcap 150 / Smallcap 250.

Run with:
    streamlit run app.py

Three scan modes, all on ONE chosen timeframe (1H / 4H / 1D / 1W / 1M):

1. Above 200 MA — flags stocks whose latest close sits just above (within
   your chosen band) the 200-period EMA/SMA.
2. Golden Cross / Death Cross — flags stocks where the 50-period EMA/SMA has
   recently crossed the 200-period EMA/SMA, up (golden) or down (death).
3. Unusual Volume — flags stocks with a volume spike vs. their recent
   average, labeled Buying/Selling from the price move on that bar. This is
   the closest free, per-stock proxy for "big institutional activity" --
   NSE only discloses actual FII/DII activity as a market-wide daily total,
   not per stock (shown separately in the Market Pulse panel below).
4. Chart Patterns — heuristic, best-effort detection of Triangle, Channel,
   and Flag & Pole breakouts (see patterns.py for how, and its limits).

Plus a Bulk & Block Deals panel (NSE's disclosed large-trade feed) and a
Chart tab with a Fixed Range Volume Profile.

This is a screening tool, not investment advice. It surfaces candidates for
you to apply your own entry/exit discipline (stop-loss, position size, risk
per trade) to -- it does not decide trades for you.
"""
import glob
import os
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from constituents import get_all_symbols, get_constituents, INDEX_UNIVERSE_LABEL
from screener import (scan_universe, apply_band, apply_band_below, scan_universe_cross, scan_universe_volume,
                       load_frame, scan_symbol, scan_symbol_cross, scan_symbol_volume,
                       scan_universe_pattern, scan_symbol_pattern, scan_universe_rsi, scan_symbol_rsi,
                       scan_universe_aged_ath, scan_symbol_aged_ath)
from indicators import TIMEFRAMES
from market_data import get_fii_dii_activity, deals_for_symbols
from chart import build_candles_with_volume_profile, build_ma_overlay_chart, build_structure_chart, add_ott_overlay
from ott import MA_TYPES as OTT_MA_TYPES
import screener_alert
import cpr
import structure
import fo_universe
import telegram_bot
import backtest
import cpr_ema_backtest
import swing_backtest
import momentum_backtest
import scalping_backtest
import bigmoney_swing_backtest
import oliver_ma_cross_backtest
import cpr_rsi_scalp_backtest
import paper_trading
import call_log

st.set_page_config(page_title="TrendLine", layout="wide")

# TradingView-style chart interaction everywhere: scroll wheel zooms, drag pans (paired with
# dragmode="pan" on each figure's layout) instead of plotly's default click-drag zoom box.
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}

TF_KEYS = list(TIMEFRAMES.keys())
TF_LABELS = [TIMEFRAMES[k]["label"] for k in TF_KEYS]
DEFAULT_TF_INDEX = TF_LABELS.index("1 Day")
UNIVERSES = ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"]
ALL_UNIVERSE_LABEL = "All (Large + Mid + Small Cap)"
UNIVERSE_OPTIONS = UNIVERSES + [ALL_UNIVERSE_LABEL, INDEX_UNIVERSE_LABEL]


def _fmt_asof(series: pd.Series, timeframe: str) -> pd.Series:
    fmt = "%Y-%m-%d %H:%M" if TIMEFRAMES[timeframe]["kind"] == "intraday" else "%Y-%m-%d"
    return pd.to_datetime(series).dt.strftime(fmt)


def _current_scan_params(scan_mode, band_low, band_high, lookback, avg_period, spike_multiple,
                          pattern_types, pattern_lookback, pole_min_move_pct) -> dict:
    if scan_mode in ("Above 200 MA", "Below 200 MA"):
        return {"band_low": band_low, "band_high": band_high}
    if scan_mode.startswith("Golden Cross"):
        return {"lookback": lookback}
    if scan_mode.startswith("Unusual Volume"):
        return {"avg_period": avg_period, "spike_multiple": spike_multiple}
    return {"pattern_types": pattern_types, "pattern_lookback": pattern_lookback,
            "pole_min_move_pct": pole_min_move_pct}


def render_telegram_send_button(key_prefix: str, header_lines: list, records: list) -> None:
    """A "📲 Send to Telegram" section for a single snapshot (not a scan digest --
    those go through screener_alert.py). `records` is a list of flat dicts (one
    per row shown on screen); each renders as its own numbered "Key: value | Key:
    value" block, separated by a blank line, so a multi-row snapshot reads as a
    list rather than a wall of text in Telegram."""
    st.divider()
    st.subheader("📲 Send to Telegram")
    if not records:
        st.caption("Nothing to send yet.")
        return
    if st.button("📲 Send this snapshot now", key=f"{key_prefix}_send_tg"):
        body_blocks = [f"{i}. " + " | ".join(f"{k}: {v}" for k, v in rec.items())
                        for i, rec in enumerate(records, start=1)]
        message = "\n".join(header_lines) + "\n\n" + "\n\n".join(body_blocks)
        with st.spinner("Sending to Telegram..."):
            sent = telegram_bot.send_telegram_message(message)
        if sent:
            st.success("Sent to Telegram.")
        else:
            st.error("Telegram send failed — check .env credentials or your network.")


def render_market_pulse():
    st.subheader("Market Pulse — FII / DII activity (market-wide, today)")
    st.caption(
        "This is NSE's aggregate cash-market figure for the whole exchange, not per stock — "
        "NSE doesn't publish which specific stock an FII or DII bought or sold."
    )
    fd = get_fii_dii_activity()
    if fd is None or fd.empty:
        st.warning("Couldn't reach NSE's FII/DII feed right now (it occasionally rate-limits). Try again shortly.")
        return
    cols = st.columns(len(fd))
    for col, (_, row) in zip(cols, fd.iterrows()):
        label = f"{row['category']} net ({row['date']})"
        col.metric(label, f"₹{row['netValue']:,.0f} Cr",
                   delta=f"Buy ₹{row['buyValue']:,.0f} Cr · Sell ₹{row['sellValue']:,.0f} Cr")


MA_PERIODS = (10, 20, 50, 200)


def render_ma_toggles(key_prefix: str, ma_type: str, periods: tuple = MA_PERIODS,
                       default_on: tuple = MA_PERIODS) -> tuple:
    """One checkbox per MA period so any of them can be switched on/off independently.
    Returns the tuple of periods currently toggled on."""
    st.caption("Moving averages shown:")
    cols = st.columns(len(periods))
    selected = []
    for col, p in zip(cols, periods):
        if col.checkbox(f"{ma_type}{p}", value=p in default_on, key=f"{key_prefix}_toggle_{p}"):
            selected.append(p)
    return tuple(selected)


def render_ott_controls(key_prefix: str) -> dict | None:
    """Optimized Trend Tracker (OTT) toggle + settings, mirroring the original Pine Script's
    defaults (length=2, percent=1.4, MA type VAR). Returns a kwargs dict for add_ott_overlay(),
    or None if the indicator is switched off."""
    show = st.checkbox("Show OTT (Optimized Trend Tracker)", value=False, key=f"{key_prefix}_ott_on")
    if not show:
        return None
    with st.expander("OTT settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        length = c1.number_input("OTT Period", min_value=1, value=2, step=1, key=f"{key_prefix}_ott_length")
        percent = c2.number_input("OTT Percent", min_value=0.0, value=1.4, step=0.1, key=f"{key_prefix}_ott_percent")
        ma_type = c3.selectbox("Moving Average Type", OTT_MA_TYPES, index=0, key=f"{key_prefix}_ott_ma_type")
        c4, c5 = st.columns(2)
        show_support = c4.checkbox("Show Support Line?", value=True, key=f"{key_prefix}_ott_support")
        highlighting = c5.checkbox("Highlighter On/Off?", value=True, key=f"{key_prefix}_ott_highlighting")
        c6, c7, c8 = st.columns(3)
        show_signals_support = c6.checkbox("Show Support Line Crossing Signals?", value=True,
                                            key=f"{key_prefix}_ott_sig_support")
        show_signals_price = c7.checkbox("Show Price/OTT Crossing Signals?", value=False,
                                          key=f"{key_prefix}_ott_sig_price")
        highlight_color_changes = c8.checkbox("Show OTT Color Changes?", value=False,
                                               key=f"{key_prefix}_ott_color_changes")
        show_signals_color_change = st.checkbox("Show OTT Color Change Signals?", value=False,
                                                 key=f"{key_prefix}_ott_sig_color")
    return dict(length=int(length), percent=float(percent), ma_type=ma_type, show_support=show_support,
                highlighting=highlighting, highlight_color_changes=highlight_color_changes,
                show_signals_support=show_signals_support, show_signals_price=show_signals_price,
                show_signals_color_change=show_signals_color_change)


def render_chart_picker(symbols: list, timeframe_key: str, timeframe_label: str, ma_type: str, key_prefix: str,
                         force_refresh: bool = False):
    """A symbol picker + 'Show chart' button rendering a candlestick chart with
    toggleable MA lines (SMA or EMA, matching the current selection)."""
    if not symbols:
        return
    c1, c2 = st.columns([3, 1])
    with c1:
        picked = st.selectbox("Pick a symbol to chart", symbols, key=f"{key_prefix}_pick")
    with c2:
        bars = st.number_input("Bars to show", min_value=50, max_value=1000, value=250, step=25,
                                key=f"{key_prefix}_bars")
    periods = render_ma_toggles(key_prefix, ma_type)

    shown_key = f"{key_prefix}_shown_symbol"
    if st.button(f"📈 Show chart for {picked}", key=f"{key_prefix}_show"):
        st.session_state[shown_key] = picked

    shown_symbol = st.session_state.get(shown_key)
    if shown_symbol:
        with st.spinner(f"Fetching {shown_symbol}..."):
            full = load_frame(shown_symbol, timeframe_key, force_refresh=force_refresh)
        if full is None or full.empty:
            st.error(f"Couldn't fetch usable data for {shown_symbol} on {timeframe_label}.")
        else:
            fig = build_ma_overlay_chart(full, shown_symbol, timeframe_label, ma_type, periods=periods,
                                          display_bars=int(bars))
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fig", config=PLOTLY_CONFIG)


def render_deals_panel(universe_choice: str):
    st.subheader(f"Bulk & Block Deals — {universe_choice} (latest session)")
    st.caption(
        "NSE's disclosed large-trade feed: individual trades above the exchange's bulk/block "
        "thresholds, with the counterparty named. Not officially tagged FII/DII/other — read the "
        "client name yourself — but this is the closest free, per-stock signal for large disclosed "
        "buying or selling."
    )
    if universe_choice == ALL_UNIVERSE_LABEL:
        symbols = sorted(all_universe_symbols["Symbol"].unique().tolist())
    else:
        symbols = list(get_constituents(universe_choice)["Symbol"])
    deals = deals_for_symbols(symbols)
    if deals.empty:
        st.write("No bulk or block deals reported for this universe in the latest session.")
        return
    qty_col = "Quantity Traded" if "Quantity Traded" in deals.columns else None
    if qty_col:
        deals = deals.sort_values(by=qty_col, ascending=False)
    show_cols = [c for c in ["Date", "Symbol", "Deal Type", "Client Name", "Buy/Sell",
                              "Quantity Traded", "Trade Price / Wght. Avg. Price"] if c in deals.columns]
    st.dataframe(deals[show_cols], use_container_width=True, hide_index=True)


all_universe_symbols = get_all_symbols(UNIVERSES + [INDEX_UNIVERSE_LABEL])
all_symbols_sorted = sorted(all_universe_symbols["Symbol"].unique().tolist())

with st.sidebar:
    st.title("📈 TrendLine")
    st.header("Scan settings")

    universe_choice = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0)
    segments = UNIVERSES if universe_choice == ALL_UNIVERSE_LABEL else [universe_choice]

    scan_mode = st.radio(
        "Scan type",
        ["Aged ATH Breakout", "Above 200 MA", "Below 200 MA", "Golden Cross / Death Cross (50 vs 200)",
         "Unusual Volume (Buying/Selling Spike)", "Chart Patterns (Triangle / Channel / Flag & Pole)",
         "RSI Range (Overbought/Oversold Filter)"],
        index=1,
    )

    ma_type = st.radio("Moving average type", ["EMA", "SMA"], horizontal=True)

    tf_choice_label = st.radio("Timeframe", TF_LABELS, index=DEFAULT_TF_INDEX, horizontal=True)  # default 1 Day
    timeframe = TF_KEYS[TF_LABELS.index(tf_choice_label)]

    lookback = avg_period = spike_multiple = band_low = band_high = None
    pattern_types = pattern_lookback = pole_min_move_pct = None
    rsi_period = rsi_min = rsi_max = None
    ath_min_age_years = ath_confirm_bars = None

    if scan_mode == "Above 200 MA":
        band_low, band_high = st.slider(
            f"\"Just above\" band (% above the 200 {ma_type})",
            min_value=0.0, max_value=15.0, value=(0.0, 3.0), step=0.5,
        )
    elif scan_mode == "Below 200 MA":
        band_low, band_high = st.slider(
            f"\"Just broke down\" band (% below the 200 {ma_type})",
            min_value=0.0, max_value=15.0, value=(0.0, 3.0), step=0.5,
        )
    elif scan_mode.startswith("Golden Cross"):
        lookback = st.number_input(
            "Treat a cross as \"recent\" if it happened within the last N bars",
            min_value=1, max_value=50, value=5, step=1,
        )
    elif scan_mode.startswith("Unusual Volume"):
        avg_period = st.number_input(
            "Baseline average volume window (bars)", min_value=5, max_value=100, value=20, step=5,
        )
        spike_multiple = st.number_input(
            "Flag as unusual if volume is at least this many times the average",
            min_value=1.2, max_value=10.0, value=2.0, step=0.1,
        )
    elif scan_mode.startswith("RSI Range"):
        rsi_period = st.number_input("RSI length", min_value=2, max_value=50, value=14, step=1)
        rsi_min = st.number_input("Bottom value (oversold below this)", min_value=1, max_value=49,
                                   value=30, step=1)
        rsi_max = st.number_input("Max value (overbought above this)", min_value=51, max_value=99,
                                   value=70, step=1)
        st.caption(f"Screens for RSI({rsi_period}) currently between {rsi_min} and {rsi_max} — "
                   "neither overbought nor oversold.")
    elif scan_mode.startswith("Aged ATH"):
        st.markdown("**Aged ATH Breakout settings**")
        with st.container(border=True):
            ath_min_age_years = st.number_input(
                "Minimum age of the all-time high being broken (years)", min_value=1.0, max_value=30.0,
                value=5.0, step=0.5,
            )
            ath_confirm_bars = st.slider(
                "Confirmation candles (at least this many consecutive closes above the old high)",
                min_value=1, max_value=10, value=2, step=1,
            )
        st.caption(
            f"Flags a CONFIRMED breakout — at least {ath_confirm_bars} consecutive candle(s) closing "
            f"above the level, not just a single-candle cross that can be a whipsaw/fakeout — over an "
            f"all-time high that is at least {ath_min_age_years:g} years old — a genuinely dormant "
            "multi-year level finally giving way, not just an ordinary new high. Works best on **1 Day, "
            "1 Week, or 1 Month** — intraday timeframes rarely have enough history for a multi-year-old "
            "high to even exist."
        )
    else:
        pattern_types = st.multiselect(
            "Pattern types", ["Triangle", "Channel", "Flag & Pole"],
            default=["Triangle", "Channel", "Flag & Pole"],
        )
        pattern_lookback = st.number_input(
            "Trendline lookback (bars) — for Triangle & Channel", min_value=30, max_value=250, value=80, step=10,
        )
        pole_min_move_pct = st.number_input(
            "Minimum pole move % — for Flag & Pole", min_value=2.0, max_value=30.0, value=8.0, step=1.0,
        )
        st.caption("Heuristic, rule-based detection — expect false positives. Always confirm visually "
                   "on the chart before acting on a hit.")

    min_volume = st.number_input(
        "Minimum volume on this timeframe (liquidity filter)", min_value=0, value=50000, step=10000
    )

    force_refresh_symbols = st.checkbox("Force-refresh index constituent lists", value=False)
    force_refresh_prices = st.checkbox("Force-refresh price history (ignore cache)", value=False)

    run = st.button("Run scan", type="primary", use_container_width=True)

    st.divider()
    st.subheader("🔍 Search a stock")
    st.caption("Any symbol or index, all universes. Opens in the Stock Chart tab, using the settings above.")
    searched_symbol = st.selectbox("Symbol", [""] + all_symbols_sorted, index=0, key="global_search_symbol")

(tab_screener, tab_stock_chart, tab_volume_profile, tab_cpr, tab_structure, tab_backtest,
 tab_calls) = st.tabs(
    ["Screener", "🔍 Stock Chart", "Volume Profile", "🎯 Narrow CPR", "📐 Market Structure",
     "🧪 Backtest", "📞 Call Performance"]
)

with tab_screener:
    if scan_mode == "Above 200 MA":
        st.title(f"Filter Stocks out of {ma_type}200")
    elif scan_mode == "Below 200 MA":
        st.title(f"🔴 Short Setups — Below {ma_type}200")
    elif scan_mode.startswith("Golden Cross"):
        st.title(f"{ma_type}50 / {ma_type}200 Golden Cross & Death Cross Screener")
    elif scan_mode.startswith("Unusual Volume"):
        st.title("Unusual Volume Screener")
    elif scan_mode.startswith("RSI Range"):
        st.title("RSI Range Screener")
    elif scan_mode.startswith("Aged ATH"):
        st.title("Aged All-Time-High Breakout Screener")
    else:
        st.title("Chart Pattern Breakout Screener")

    st.caption(
        f"Scan: {scan_mode} · Timeframe: {tf_choice_label} · Universe: {universe_choice} · "
        "Data: Yahoo Finance (NSE). Screening tool only — apply your own entry/exit discipline before trading."
    )

    with st.expander("📊 Market Pulse & Bulk/Block Deals", expanded=False):
        render_market_pulse()
        st.divider()
        render_deals_panel(universe_choice)

    for key in ("results", "scanned_at", "scanned_tf", "scanned_ma", "scanned_mode", "scanned_timeframe_key",
                "scanned_lookback", "scanned_spike_multiple", "scanned_band_low", "scanned_band_high",
                "last_scan_config"):
        if key not in st.session_state:
            st.session_state[key] = None

    if run:
        symbols_df = get_all_symbols(segments, force_refresh=force_refresh_symbols)
        st.write(f"Scanning {len(symbols_df)} symbols in {universe_choice} on {tf_choice_label} ({ma_type})...")

        progress = st.progress(0.0)
        status = st.empty()
        start = time.time()

        def _cb(i, total, symbol):
            progress.progress((i + 1) / total)
            status.text(f"[{i+1}/{total}] {symbol}")

        if scan_mode in ("Above 200 MA", "Below 200 MA"):
            results = scan_universe(symbols_df[["Symbol", "Segment"]], timeframe, ma_type, progress_cb=_cb,
                                     force_refresh=force_refresh_prices)
        elif scan_mode.startswith("Golden Cross"):
            results = scan_universe_cross(symbols_df[["Symbol", "Segment"]], timeframe, ma_type, lookback,
                                           progress_cb=_cb, force_refresh=force_refresh_prices)
        elif scan_mode.startswith("Unusual Volume"):
            results = scan_universe_volume(symbols_df[["Symbol", "Segment"]], timeframe, avg_period,
                                            spike_multiple, progress_cb=_cb, force_refresh=force_refresh_prices)
        elif scan_mode.startswith("RSI Range"):
            results = scan_universe_rsi(symbols_df[["Symbol", "Segment"]], timeframe, rsi_period, rsi_min,
                                         rsi_max, progress_cb=_cb, force_refresh=force_refresh_prices)
        elif scan_mode.startswith("Aged ATH"):
            results = scan_universe_aged_ath(symbols_df[["Symbol", "Segment"]], timeframe, ath_min_age_years,
                                              min_confirm_bars=int(ath_confirm_bars), progress_cb=_cb,
                                              force_refresh=force_refresh_prices)
        else:
            results = scan_universe_pattern(symbols_df[["Symbol", "Segment"]], timeframe, pattern_types,
                                             pattern_lookback, pole_min_move_pct, progress_cb=_cb,
                                             force_refresh=force_refresh_prices)

        elapsed = time.time() - start
        if scan_mode.startswith("Chart Patterns"):
            status.text(f"Done in {elapsed:.0f}s — {len(results)} pattern breakout(s) found "
                        f"across {len(symbols_df)} symbols.")
        else:
            status.text(f"Done in {elapsed:.0f}s — {len(results)}/{len(symbols_df)} symbols had usable data.")

        scan_config = {
            "universe_choice": universe_choice, "segments": segments, "scan_mode": scan_mode,
            "timeframe": timeframe, "timeframe_label": tf_choice_label, "ma_type": ma_type,
            "min_volume": min_volume, "band_low": band_low, "band_high": band_high, "lookback": lookback,
            "avg_period": avg_period, "spike_multiple": spike_multiple, "pattern_types": pattern_types,
            "pattern_lookback": pattern_lookback, "pole_min_move_pct": pole_min_move_pct,
            "rsi_period": rsi_period, "rsi_min": rsi_min, "rsi_max": rsi_max,
            "ath_min_age_years": ath_min_age_years, "ath_confirm_bars": ath_confirm_bars,
        }
        screener_alert.set_active_scan(scan_config)
        if screener_alert.send_snapshot_now(scan_config, results=results):
            st.caption("📲 Telegram alert sent — this scan will keep re-sending every 15 min until market close.")
        else:
            st.caption("⚠️ Telegram alert not sent (check .env credentials) — will retry on the next 15-min cycle.")

        st.session_state["results"] = results
        st.session_state["last_scan_config"] = scan_config
        st.session_state["scanned_at"] = pd.Timestamp.now()
        st.session_state["scanned_tf"] = tf_choice_label
        st.session_state["scanned_ma"] = ma_type
        st.session_state["scanned_mode"] = scan_mode
        st.session_state["scanned_timeframe_key"] = timeframe
        st.session_state["scanned_lookback"] = lookback
        st.session_state["scanned_spike_multiple"] = spike_multiple
        st.session_state["scanned_band_low"] = band_low
        st.session_state["scanned_band_high"] = band_high

    results = st.session_state["results"]

    if results is None or (results.empty and st.session_state["scanned_mode"] != "Chart Patterns (Triangle / Channel / Flag & Pole)"):
        st.info("Set your filters in the sidebar and click **Run scan**. Intraday timeframes (1H/4H) "
                "and the first daily/weekly/monthly scan of the day are slower (full history fetch per "
                "stock); later runs the same day reuse the local cache and are fast.")
    elif results.empty:
        st.success("Scan complete — no chart pattern breakouts found with these settings. "
                   "Try a different pattern type, a wider trendline lookback, or a lower pole move %.")
    else:
        scanned_ma = st.session_state["scanned_ma"]
        scanned_mode = st.session_state["scanned_mode"]
        scanned_tf_key = st.session_state["scanned_timeframe_key"]

        df = results.copy()
        df = df[df["Volume"] >= min_volume]
        df["AsOf"] = _fmt_asof(df["AsOf"], scanned_tf_key)

        st.caption(
            f"Last scanned: {st.session_state['scanned_at']:%Y-%m-%d %H:%M} · "
            f"{st.session_state['scanned_tf']} · {scanned_ma}"
        )

        if scanned_mode != scan_mode or scanned_tf_key != timeframe or scanned_ma != ma_type:
            st.warning(
                "The sidebar's Scan type / MA / Timeframe has changed since this scan ran — the results "
                "below are still from the settings shown above. Click **Run scan** to refresh."
            )

        with st.expander(f"📈 View chart for a stock in these results ({len(df)} scanned)", expanded=False):
            render_chart_picker(sorted(df["Symbol"].unique().tolist()), scanned_tf_key,
                                 st.session_state["scanned_tf"], scanned_ma, key_prefix="results",
                                 force_refresh=force_refresh_prices)

        if scanned_mode == "Above 200 MA":
            df = apply_band(df, st.session_state["scanned_band_low"], st.session_state["scanned_band_high"])

            in_band = df[df["Status"] == "In band"].copy()
            above = df[df["Status"] == "Above band"].copy()
            below = df[df["Status"] == "Below band"].copy()
            no_hist = df[df["Status"] == "Insufficient history"].copy()

            ma_col = f"{scanned_ma}200"
            display_cols = ["Symbol", "Close", "AsOf", ma_col, "PctAbove", "Volume"]
            rename = {"PctAbove": f"% above {ma_col}"}

            def _fmt(d):
                out = d[display_cols].rename(columns=rename).copy()
                out["Close"] = out["Close"].round(2)
                out[ma_col] = out[ma_col].round(2)
                out[f"% above {ma_col}"] = out[f"% above {ma_col}"].round(2)
                return out.sort_values(by=f"% above {ma_col}")

            st.subheader(f"🟢 Long — fresh cross above {ma_col} on {st.session_state['scanned_tf']} ({len(in_band)})")
            st.caption("Only stocks whose previous bar closed at/below the MA and whose latest bar closed "
                       "above it -- a genuine fresh cross, not just \"currently sitting above.\"")
            if in_band.empty:
                st.write("No stocks currently meet this condition with these settings.")
            else:
                st.dataframe(_fmt(in_band), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download this list (CSV)",
                    _fmt(in_band).to_csv(index=False),
                    file_name=f"in_band_{ma_col}_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            st.subheader(f"🔴 Short — currently under {ma_col} on {st.session_state['scanned_tf']} ({len(below)})")
            if below.empty:
                st.write("No stocks currently meet this condition with these settings.")
            else:
                st.dataframe(_fmt(below), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download this list (CSV)",
                    _fmt(below).to_csv(index=False),
                    file_name=f"below_band_{ma_col}_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"Above band — extended above {ma_col}, or in-range with no fresh cross today "
                              f"({len(above)})"):
                st.dataframe(_fmt(above) if not above.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

            with st.expander(f"Insufficient history for a 200-bar {scanned_ma} on this timeframe ({len(no_hist)})"):
                st.write("These symbols don't have 200 bars of data yet on this timeframe (e.g. a recent "
                         "listing, or 1H/4H data only goes back ~2 years on the free feed).")
                st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

        elif scanned_mode == "Below 200 MA":
            df = apply_band_below(df, st.session_state["scanned_band_low"], st.session_state["scanned_band_high"])
            df["PctBelow"] = -df["PctAbove"]

            in_band = df[df["Status"] == "In band"].copy()
            not_below = df[df["Status"] == "Not below MA"].copy()
            extended = df[df["Status"] == "Extended below"].copy()
            no_hist = df[df["Status"] == "Insufficient history"].copy()

            ma_col = f"{scanned_ma}200"
            display_cols = ["Symbol", "Close", "AsOf", ma_col, "PctBelow", "Volume"]
            rename = {"PctBelow": f"% below {ma_col}"}

            def _fmt(d):
                out = d[display_cols].rename(columns=rename).copy()
                out["Close"] = out["Close"].round(2)
                out[ma_col] = out[ma_col].round(2)
                out[f"% below {ma_col}"] = out[f"% below {ma_col}"].round(2)
                return out.sort_values(by=f"% below {ma_col}")

            st.subheader(f"🔴 Short — fresh breakdown below {ma_col} on {st.session_state['scanned_tf']} "
                         f"({len(in_band)})")
            st.caption("Only stocks whose previous bar closed at/above the MA and whose latest bar closed "
                       "below it -- a genuine fresh breakdown, not just \"currently sitting under it.\"")
            if in_band.empty:
                st.write("No stocks currently meet this condition with these settings.")
            else:
                st.dataframe(_fmt(in_band), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download this list (CSV)",
                    _fmt(in_band).to_csv(index=False),
                    file_name=f"below_200ma_{ma_col}_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"Extended below — further under {ma_col}, or in-range with no fresh break "
                              f"today ({len(extended)})"):
                st.write("Already well below the MA (or sitting in-range without a fresh break today) — a "
                         "short entry here is chasing a move that's further along, not a fresh breakdown.")
                st.dataframe(_fmt(extended) if not extended.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"Not below MA — still trading above {ma_col} ({len(not_below)})"):
                st.dataframe(_fmt(not_below) if not not_below.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"Insufficient history for a 200-bar {scanned_ma} on this timeframe ({len(no_hist)})"):
                st.write("These symbols don't have 200 bars of data yet on this timeframe (e.g. a recent "
                         "listing, or 1H/4H data only goes back ~2 years on the free feed).")
                st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

        elif scanned_mode.startswith("Golden Cross"):
            fast_col, slow_col = f"{scanned_ma}50", f"{scanned_ma}200"
            display_cols = ["Symbol", "Close", "AsOf", fast_col, slow_col, "Volume"]

            def _fmt(d):
                out = d[display_cols].copy()
                out["Close"] = out["Close"].round(2)
                out[fast_col] = out[fast_col].round(2)
                out[slow_col] = out[slow_col].round(2)
                return out.sort_values(by="Symbol")

            golden = df[df["CrossType"] == "Golden Cross"].copy()
            death = df[df["CrossType"] == "Death Cross"].copy()
            no_cross = df[df["CrossType"] == "No recent cross"].copy()
            no_hist = df[df["CrossType"] == "Insufficient history"].copy()

            scanned_lookback = st.session_state["scanned_lookback"]
            st.subheader(f"🟢 Golden Cross — {fast_col} crossed above {slow_col} ({len(golden)})")
            if golden.empty:
                st.write(f"No golden crosses in the last {scanned_lookback} bars with these settings.")
            else:
                st.dataframe(_fmt(golden), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Golden Cross list (CSV)",
                    _fmt(golden).to_csv(index=False),
                    file_name=f"golden_cross_{scanned_ma}_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            st.subheader(f"🔴 Death Cross — {fast_col} crossed below {slow_col} ({len(death)})")
            if death.empty:
                st.write(f"No death crosses in the last {scanned_lookback} bars with these settings.")
            else:
                st.dataframe(_fmt(death), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Death Cross list (CSV)",
                    _fmt(death).to_csv(index=False),
                    file_name=f"death_cross_{scanned_ma}_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"No recent cross — {fast_col}/{slow_col} didn't cross recently ({len(no_cross)})"):
                st.dataframe(_fmt(no_cross) if not no_cross.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

            with st.expander(f"Insufficient history for a 200-bar {scanned_ma} on this timeframe ({len(no_hist)})"):
                st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

        elif scanned_mode.startswith("Unusual Volume"):
            display_cols = ["Symbol", "Close", "AsOf", "Volume", "AvgVolume", "VolumeRatio",
                             "PriceChangePct"]

            def _fmt(d):
                out = d[display_cols].rename(columns={
                    "AvgVolume": "Avg Volume", "VolumeRatio": "Volume x Avg", "PriceChangePct": "Price Chg %",
                }).copy()
                out["Close"] = out["Close"].round(2)
                out["Avg Volume"] = out["Avg Volume"].round(0)
                out["Volume x Avg"] = out["Volume x Avg"].round(2)
                out["Price Chg %"] = out["Price Chg %"].round(2)
                return out.sort_values(by="Volume x Avg", ascending=False)

            buying = df[df["Activity"] == "Unusual Buying"].copy()
            selling = df[df["Activity"] == "Unusual Selling"].copy()
            flat_spike = df[df["Activity"] == "Volume Spike (Flat)"].copy()
            normal = df[df["Activity"] == "Normal"].copy()
            no_hist = df[df["Activity"] == "Insufficient history"].copy()

            st.subheader(f"🟢 Unusual Buying — volume spike + price up ({len(buying)})")
            if buying.empty:
                st.write("No unusual buying activity with these settings.")
            else:
                st.dataframe(_fmt(buying), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Unusual Buying list (CSV)",
                    _fmt(buying).to_csv(index=False),
                    file_name=f"unusual_buying_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            st.subheader(f"🔴 Unusual Selling — volume spike + price down ({len(selling)})")
            if selling.empty:
                st.write("No unusual selling activity with these settings.")
            else:
                st.dataframe(_fmt(selling), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Unusual Selling list (CSV)",
                    _fmt(selling).to_csv(index=False),
                    file_name=f"unusual_selling_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"Volume spike, but price flat ({len(flat_spike)})"):
                st.dataframe(_fmt(flat_spike) if not flat_spike.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

            with st.expander(f"Normal volume ({len(normal)})"):
                st.dataframe(_fmt(normal) if not normal.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

            with st.expander(f"Insufficient history for a volume baseline ({len(no_hist)})"):
                st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

        elif scanned_mode.startswith("RSI Range"):
            display_cols = ["Symbol", "Close", "AsOf", "Volume", "RSI"]

            def _fmt(d):
                out = d[display_cols].copy()
                out["Close"] = out["Close"].round(2)
                out["RSI"] = out["RSI"].round(2)
                return out.sort_values(by="RSI")

            in_range = df[df["Status"] == "In Range"].copy()
            overbought = df[df["Status"] == "Overbought"].copy()
            oversold = df[df["Status"] == "Oversold"].copy()
            no_hist = df[df["Status"] == "Insufficient history"].copy()

            st.subheader(f"🎯 In Range — neither overbought nor oversold ({len(in_range)})")
            if in_range.empty:
                st.write("No stocks currently meet this condition with these settings.")
            else:
                st.dataframe(_fmt(in_range), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download In Range list (CSV)",
                    _fmt(in_range).to_csv(index=False),
                    file_name=f"rsi_in_range_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"🔴 Overbought (RSI above the max) ({len(overbought)})"):
                st.dataframe(_fmt(overbought) if not overbought.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"🟢 Oversold (RSI below the bottom) ({len(oversold)})"):
                st.dataframe(_fmt(oversold) if not oversold.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"Insufficient history for this RSI length ({len(no_hist)})"):
                st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

        elif scanned_mode.startswith("Aged ATH"):
            display_cols = ["Symbol", "Close", "AsOf", "ATH", "ATHDate", "AgeYears", "PctFromATH", "Volume"]
            rename = {"ATHDate": "ATH Date", "AgeYears": "ATH Age (yrs)", "PctFromATH": "% From ATH"}

            def _fmt(d):
                out = d[display_cols].rename(columns=rename).copy()
                out["Close"] = out["Close"].round(2)
                out["ATH"] = out["ATH"].round(2)
                out["ATH Date"] = pd.to_datetime(out["ATH Date"]).dt.strftime("%d-%b-%Y")
                return out.sort_values(by="ATH Age (yrs)", ascending=False)

            fresh_breakout = df[df["Status"] == "Fresh Aged Breakout"].copy()
            forming = df[df["Status"] == "Breakout Forming"].copy()
            above_aged = df[df["Status"] == "Above Aged ATH"].copy()
            too_recent = df[df["Status"] == "ATH Too Recent"].copy()
            below_ath = df[df["Status"] == "Below ATH"].copy()
            no_hist = df[df["Status"] == "Insufficient history"].copy()

            st.subheader(f"🚀 Fresh Aged Breakout — confirmed over a multi-year-old high ({len(fresh_breakout)})")
            if fresh_breakout.empty:
                st.write("No stocks currently meet this condition with these settings.")
            else:
                st.dataframe(_fmt(fresh_breakout), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Fresh Aged Breakout list (CSV)",
                    _fmt(fresh_breakout).to_csv(index=False),
                    file_name=f"aged_ath_breakout_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"⏳ Breakout Forming — above the level, but hasn't held long enough to "
                              f"confirm yet ({len(forming)})"):
                st.dataframe(_fmt(forming) if not forming.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"Above Aged ATH — confirmed on an earlier bar, not fresh today "
                              f"({len(above_aged)})"):
                st.dataframe(_fmt(above_aged) if not above_aged.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"ATH Too Recent — at/near its high, but that high isn't old enough to "
                              f"qualify ({len(too_recent)})"):
                st.dataframe(_fmt(too_recent) if not too_recent.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"Below ATH — hasn't reclaimed its aged high yet ({len(below_ath)})"):
                st.dataframe(_fmt(below_ath) if not below_ath.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

            with st.expander(f"Insufficient history ({len(no_hist)})"):
                st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                             use_container_width=True, hide_index=True)

        else:
            st.caption(
                "Heuristic, rule-based pattern detection — treat these as candidates to confirm "
                "visually on the chart above, not certainties."
            )
            display_cols = ["Symbol", "Pattern", "Direction", "Close", "AsOf", "Volume"]

            def _fmt(d):
                out = d[display_cols].copy()
                out["Close"] = out["Close"].round(2)
                return out.sort_values(by=["Pattern", "Direction", "Symbol"])

            st.subheader(f"📐 Chart Pattern Breakouts ({len(df)})")
            st.dataframe(_fmt(df), use_container_width=True, hide_index=True)
            st.download_button(
                "Download this list (CSV)",
                _fmt(df).to_csv(index=False),
                file_name=f"pattern_breakouts_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
            )

    st.divider()
    st.caption(
        "Reminder: this tool only screens for technical conditions (price structure and volume vs. "
        "their own history). It does not size positions, place stop-losses, or manage risk for you — "
        "that discipline is on you before every entry."
    )

    st.divider()
    st.subheader("📲 Send to Telegram")
    active_scan = screener_alert.get_active_scan()
    if active_scan and active_scan.get("date") == datetime.now(screener_alert.IST).strftime("%Y-%m-%d"):
        st.caption(
            f"🔁 Currently auto-repeating every 15 min until market close: **{active_scan['scan_mode']}** "
            f"on **{active_scan['universe_choice']}** ({active_scan['timeframe_label']}, "
            f"{active_scan['ma_type']}). The button below (re)sends it on demand, independent of that cycle."
        )
    last_scan_config = st.session_state.get("last_scan_config")
    if not last_scan_config or results is None:
        st.caption("Run a scan above first — this sends whatever that scan found.")
    else:
        tg_hits = screener_alert.qualifying_hits(last_scan_config, results)
        st.write(f"**Currently qualifying: {len(tg_hits)}**")
        if tg_hits:
            st.dataframe(
                pd.DataFrame(tg_hits)[["symbol", "detail", "ma_value", "close"]]
                  .rename(columns={"symbol": "Symbol", "detail": "Detail", "ma_value": "Moving Avg",
                                    "close": "Close"}),
                use_container_width=True, hide_index=True,
            )
        if st.button("📲 Send Telegram alert now", type="primary", key="screener_send_tg"):
            with st.spinner("Sending to Telegram..."):
                tg_sent = screener_alert.send_snapshot_now(last_scan_config, results=results)
            if tg_sent:
                st.success("Sent to Telegram.")
            else:
                st.error("Telegram send failed — check .env credentials or your network.")

with tab_stock_chart:
    st.title("Stock Chart")
    st.caption(
        "Search a stock in the sidebar (🔍 Search a stock) to see it here — uses the common Scan "
        "settings (Moving average type & Timeframe) from the sidebar, same as the Screener tab."
    )

    if not searched_symbol:
        st.info("Use the 🔍 Search box in the sidebar to pick a stock.")
    else:
        segment = all_universe_symbols.loc[all_universe_symbols["Symbol"] == searched_symbol, "Segment"].iloc[0]
        st.subheader(f"{searched_symbol} — {segment}")

        is_pattern_mode = scan_mode.startswith("Chart Patterns")
        with st.spinner(f"Fetching {searched_symbol}..."):
            if scan_mode in ("Above 200 MA", "Below 200 MA"):
                row = scan_symbol(searched_symbol, segment, timeframe, ma_type,
                                   force_refresh=force_refresh_prices)
            elif scan_mode.startswith("Golden Cross"):
                row = scan_symbol_cross(searched_symbol, segment, timeframe, ma_type, lookback or 5,
                                         force_refresh=force_refresh_prices)
            elif is_pattern_mode:
                row = scan_symbol_pattern(searched_symbol, segment, timeframe, pattern_types or [],
                                           pattern_lookback or 80, pole_min_move_pct or 8.0,
                                           force_refresh=force_refresh_prices)
            else:
                row = scan_symbol_volume(searched_symbol, segment, timeframe, avg_period or 20,
                                          spike_multiple or 2.0, force_refresh=force_refresh_prices)
            full = load_frame(searched_symbol, timeframe, force_refresh=force_refresh_prices)

        stock_chart_records = []
        if is_pattern_mode:
            if not row:
                st.info(f"No selected pattern found for {searched_symbol} on {tf_choice_label} with these settings.")
            else:
                out = pd.DataFrame(row).copy()
                out["AsOf"] = _fmt_asof(out["AsOf"], timeframe)
                out["Close"] = out["Close"].round(2)
                out = out.drop(columns=["Segment"], errors="ignore")
                st.dataframe(out, use_container_width=True, hide_index=True)
                stock_chart_records = out.to_dict("records")
        elif row is None:
            st.error(f"Couldn't fetch usable data for {searched_symbol} on {tf_choice_label}.")
        else:
            out = pd.DataFrame([row]).copy()
            out["AsOf"] = _fmt_asof(out["AsOf"], timeframe)
            for c in out.columns:
                if out[c].dtype.kind == "f":
                    out[c] = out[c].round(2)
            out = out.drop(columns=["Segment", "BarsAvailable"], errors="ignore")
            st.dataframe(out, use_container_width=True, hide_index=True)
            stock_chart_records = out.to_dict("records")

        bars = st.number_input("Bars to show", min_value=50, max_value=1000, value=250, step=25,
                                key="stock_chart_bars")
        stock_chart_periods = render_ma_toggles("stock_chart", ma_type)
        stock_chart_ott = render_ott_controls("stock_chart")

        if full is None or full.empty:
            st.error(f"Couldn't fetch usable price history for {searched_symbol} on {tf_choice_label}.")
        else:
            fig = build_ma_overlay_chart(full, searched_symbol, tf_choice_label, ma_type,
                                          periods=stock_chart_periods, display_bars=int(bars))
            if stock_chart_ott is not None:
                add_ott_overlay(fig, full, display_bars=int(bars), **stock_chart_ott)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        render_telegram_send_button(
            "stock_chart",
            header_lines=[
                f"{datetime.now():%d-%b-%Y %H:%M} IST",
                f"🔍 Stock Chart snapshot — {searched_symbol}",
                f"Scan: {scan_mode} | Timeframe: {tf_choice_label} | MA: {ma_type}",
            ],
            records=stock_chart_records,
        )

with tab_volume_profile:
    st.title("Chart & Fixed Range Volume Profile")
    st.caption(
        "Candlestick chart with a volume-by-price histogram for the range shown. The orange line "
        "marks the Point of Control (POC) — the price level with the most traded volume in this range."
    )

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        chart_universe = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0, key="chart_universe")
        if chart_universe == ALL_UNIVERSE_LABEL:
            chart_symbols = sorted(all_universe_symbols["Symbol"].unique().tolist())
        else:
            chart_symbols = sorted(get_constituents(chart_universe)["Symbol"].tolist())
        chart_symbol = st.selectbox("Symbol", chart_symbols, key="chart_symbol")
    with c2:
        chart_tf_label = st.selectbox("Timeframe", TF_LABELS, index=DEFAULT_TF_INDEX, key="chart_tf")
        chart_timeframe = TF_KEYS[TF_LABELS.index(chart_tf_label)]
    with c3:
        num_bars = st.number_input("Bars to show", min_value=30, max_value=1000, value=150, step=10)
    with c4:
        num_bins = st.number_input("Price bins", min_value=8, max_value=60, value=24, step=2)

    vp_periods = render_ma_toggles("vp", ma_type)
    st.caption(f"Moving average type follows the sidebar's setting (currently {ma_type}).")

    shown_key = "vp_shown_symbol"
    if st.button("Load chart", type="primary"):
        st.session_state[shown_key] = (chart_symbol, chart_timeframe, chart_tf_label, int(num_bars), int(num_bins))

    shown = st.session_state.get(shown_key)
    vp_records = []
    if shown:
        shown_symbol, shown_timeframe, shown_tf_label, shown_bars, shown_bins = shown
        with st.spinner(f"Fetching {shown_symbol}..."):
            frame = load_frame(shown_symbol, shown_timeframe, force_refresh=force_refresh_prices)
        if frame is None or frame.empty:
            st.error(f"Couldn't fetch usable data for {shown_symbol} on {shown_tf_label}.")
        else:
            fig, poc_price = build_candles_with_volume_profile(frame, shown_symbol, shown_tf_label, shown_bins,
                                                                 ma_type=ma_type, periods=vp_periods,
                                                                 display_bars=shown_bars)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            last_close = float(frame["Close"].iloc[-1])
            vp_records = [{"Symbol": shown_symbol, "Close": round(last_close, 2), "POC": round(poc_price, 2),
                            "Bars": shown_bars}]

    render_telegram_send_button(
        "volume_profile",
        header_lines=[
            f"{datetime.now():%d-%b-%Y %H:%M} IST",
            f"📊 Volume Profile snapshot — {shown[0] if shown else ''}",
            f"Timeframe: {shown[2] if shown else ''}",
        ],
        records=vp_records,
    )

with tab_cpr:
    st.title("🎯 Narrow CPR Scanner")
    st.caption(
        "Central Pivot Range, computed from each stock's most recently completed daily session "
        "(High/Low/Close) and projected for the next session. A narrow CPR (tight TC-BC band) is "
        "read by CPR traders as compressed volatility -- a higher-probability breakout setup -- "
        "while a wide CPR suggests a more range-bound day. Data: Yahoo Finance (NSE), same feed "
        "and cache as the rest of this app. This is a screening tool, not a trade signal by itself."
    )
    st.caption("Narrow: ≤0.5% · Normal: ≤1% · Wide: >1%")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        cpr_universe = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0, key="cpr_universe")
    with c2:
        cpr_width_lt = st.number_input(
            "Width % less than", min_value=0.0, max_value=10.0, value=0.5, step=0.05,
            format="%.2f", key="cpr_width_lt",
        )
    with c3:
        cpr_width_gte = st.number_input(
            "Width % greater than or equal to", min_value=0.0, max_value=10.0, value=0.0, step=0.05,
            format="%.2f", key="cpr_width_gte",
        )

    c4, c5, c6 = st.columns([2, 1, 1])
    with c4:
        cpr_search = st.text_input("Search stock", value="", key="cpr_search")
    with c5:
        cpr_fo_only = st.checkbox("F&O Stocks Only", value=False, key="cpr_fo_only")
    with c6:
        cpr_force_refresh = st.checkbox("Force-refresh price history", value=False, key="cpr_force_refresh")

    cpr_run = st.button("Run CPR scan", type="primary", key="cpr_run")

    if cpr_run:
        cpr_segments = UNIVERSES if cpr_universe == ALL_UNIVERSE_LABEL else [cpr_universe]
        cpr_symbols_df = get_all_symbols(cpr_segments)

        if cpr_fo_only and cpr_universe == INDEX_UNIVERSE_LABEL:
            st.info(
                "\"F&O Stocks Only\" filters to individual-stock derivatives, which don't include "
                "index derivatives like NIFTY/BANK NIFTY (a separate NSE segment) — so it can't "
                "apply to the Indices universe. Showing all indices instead."
            )
            st.write(f"Scanning {len(cpr_symbols_df)} symbols in {cpr_universe}...")
        elif cpr_fo_only:
            try:
                fo_symbols = fo_universe.get_fo_symbols()
                cpr_symbols_df = cpr_symbols_df[cpr_symbols_df["Symbol"].isin(fo_symbols)]
                st.write(f"Scanning {len(cpr_symbols_df)} F&O-eligible symbols "
                         f"(of {len(fo_symbols)} total F&O stocks) in {cpr_universe}...")
            except Exception as e:
                st.error(f"Couldn't load the F&O underlying list ({e}) — scanning full universe instead.")
                st.write(f"Scanning {len(cpr_symbols_df)} symbols in {cpr_universe}...")
        else:
            st.write(f"Scanning {len(cpr_symbols_df)} symbols in {cpr_universe}...")

        cpr_progress = st.progress(0.0)
        cpr_status = st.empty()
        cpr_start = time.time()

        def _cpr_cb(i, total, symbol):
            cpr_progress.progress((i + 1) / total)
            cpr_status.text(f"[{i+1}/{total}] {symbol}")

        cpr_results = cpr.scan_universe_cpr(cpr_symbols_df[["Symbol", "Segment"]], progress_cb=_cpr_cb,
                                             force_refresh=cpr_force_refresh)
        cpr_elapsed = time.time() - cpr_start
        cpr_status.text(f"Done in {cpr_elapsed:.0f}s — {len(cpr_results)}/{len(cpr_symbols_df)} symbols had usable data.")

        st.session_state["cpr_results"] = cpr_results
        st.session_state["cpr_scanned_at"] = pd.Timestamp.now()
        st.session_state["cpr_scanned_universe"] = cpr_universe
        st.session_state["cpr_scanned_fo_only"] = cpr_fo_only and cpr_universe != INDEX_UNIVERSE_LABEL

    cpr_results = st.session_state.get("cpr_results")
    cpr_narrow_records = []

    if cpr_results is None:
        st.info("Set your filters above and click **Run CPR scan**. First run of the day is slower "
                 "(full history fetch per stock); later runs reuse the local cache and are fast.")
    elif cpr_results.empty:
        st.warning("No symbols returned usable data for this universe/filter combination.")
    else:
        fo_tag = " · F&O stocks only" if st.session_state.get("cpr_scanned_fo_only") else ""
        st.caption(
            f"Last scanned: {st.session_state['cpr_scanned_at']:%Y-%m-%d %H:%M} · "
            f"{st.session_state['cpr_scanned_universe']}{fo_tag}"
        )

        stale_count = int(cpr_results["IsStale"].sum()) if "IsStale" in cpr_results.columns else 0
        if stale_count > 0:
            latest_date = pd.Timestamp(cpr_results["LatestAsOf"].iloc[0]).date()
            stale_dates = sorted(pd.to_datetime(cpr_results.loc[cpr_results["IsStale"], "AsOf"]).dt.date.unique())
            stale_dates_str = ", ".join(str(d) for d in stale_dates)
            st.warning(
                f"⚠️ {stale_count} of {len(cpr_results)} symbols ({stale_count/len(cpr_results)*100:.0f}%) are "
                f"showing data from an older session ({stale_dates_str}) instead of the most recent one "
                f"({latest_date}) — Yahoo Finance hasn't caught up for these yet. Their CPR levels may be "
                f"outdated; flagged rows are marked in the **Data** column below."
            )

        cdf = cpr_results.copy()
        if cpr_search.strip():
            # Word-order-independent: each space-separated term must appear somewhere in the
            # symbol, so "bank nifty" matches "NIFTY BANK" even though the words are reversed.
            search_terms = cpr_search.strip().lower().split()
            symbol_lower = cdf["Symbol"].str.lower()
            mask = pd.Series(True, index=cdf.index)
            for term in search_terms:
                mask &= symbol_lower.str.contains(term, na=False)
            cdf = cdf[mask]

        display_cols = ["Symbol", "PrevClose", "PrevHigh", "PrevLow", "Range", "TC", "Pivot", "BC",
                         "WidthPct", "Category", "AsOf", "IsStale"]
        rename = {"PrevClose": "Prev Close", "PrevHigh": "Prev High", "PrevLow": "Prev Low",
                   "WidthPct": "Width %", "IsStale": "Data"}

        def _fmt_cpr(d):
            out = d[display_cols].rename(columns=rename).copy()
            for col in ["Prev Close", "Prev High", "Prev Low", "Range", "TC", "Pivot", "BC"]:
                out[col] = out[col].round(2)
            out["Width %"] = out["Width %"].round(2)
            out["Data"] = out["Data"].map({True: "⚠️ Stale", False: "✅ Current"})
            out["AsOf"] = _fmt_asof(out["AsOf"], "1D")
            return out.sort_values(by="Width %")

        narrow_filtered = cdf[(cdf["WidthPct"] < cpr_width_lt) & (cdf["WidthPct"] >= cpr_width_gte)].copy()

        st.subheader(f"Filtered results ({len(narrow_filtered)} of {len(cdf)} scanned)")
        if narrow_filtered.empty:
            st.write("No stocks currently fall in this Width % range with these settings.")
        else:
            st.dataframe(_fmt_cpr(narrow_filtered), use_container_width=True, hide_index=True)
            st.download_button(
                "Download this list (CSV)",
                _fmt_cpr(narrow_filtered).to_csv(index=False),
                file_name=f"narrow_cpr_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
            )
            cpr_narrow_records = _fmt_cpr(narrow_filtered)[["Symbol", "Width %", "TC", "Pivot", "BC"]] \
                .to_dict("records")

        narrow = cdf[cdf["Category"] == "Narrow"]
        normal = cdf[cdf["Category"] == "Normal"]
        wide = cdf[cdf["Category"] == "Wide"]
        with st.expander(f"All Narrow (≤0.5%) in this universe ({len(narrow)})"):
            st.dataframe(_fmt_cpr(narrow) if not narrow.empty else pd.DataFrame(), use_container_width=True, hide_index=True)
        with st.expander(f"Normal (≤1%) ({len(normal)})"):
            st.dataframe(_fmt_cpr(normal) if not normal.empty else pd.DataFrame(), use_container_width=True, hide_index=True)
        with st.expander(f"Wide (>1%) ({len(wide)})"):
            st.dataframe(_fmt_cpr(wide) if not wide.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

    render_telegram_send_button(
        "cpr",
        header_lines=[
            f"{datetime.now():%d-%b-%Y %H:%M} IST",
            "🎯 Narrow CPR snapshot",
            f"Universe: {cpr_universe} | Width % range: [{cpr_width_gte}, {cpr_width_lt})",
        ],
        records=cpr_narrow_records,
    )

with tab_structure:
    st.title("📐 Market Structure — Trend via Higher Highs / Higher Lows")
    st.caption(
        "Classic price-action swing structure: an **Uptrend** is a sequence of Higher Highs (HH) + "
        "Higher Lows (HL); a **Downtrend** is Lower Highs (LH) + Lower Lows (LL). A **Character Change** "
        "fires the moment price breaks the last swing that was holding the trend up (a Close below the "
        "last Higher Low in an uptrend, or above the last Lower High in a downtrend) — a potential "
        "reversal in progress. 'Uptrend'/'Downtrend' additionally require price to also be on the right "
        f"side of BOTH the {structure.MA_FAST}- and {structure.MA_SLOW}-period MA (using the sidebar's "
        "EMA/SMA setting) — structure alone can lag; this MA filter confirms the trend is live, not stale."
    )

    sc1, sc2, sc3 = st.columns([2, 2, 2])
    with sc1:
        structure_universe = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0, key="structure_universe")
    with sc2:
        structure_tf_label = st.selectbox("Timeframe", TF_LABELS, index=DEFAULT_TF_INDEX, key="structure_tf")
        structure_timeframe = TF_KEYS[TF_LABELS.index(structure_tf_label)]
    with sc3:
        structure_order = st.number_input(
            "Swing strength (bars each side)", min_value=1, max_value=50, value=20, step=1,
            key="structure_order",
        )

    sc4, sc5, sc6 = st.columns([2, 1, 1])
    with sc4:
        structure_search = st.text_input("Search stock", value="", key="structure_search")
    with sc5:
        structure_choch_lookback = st.number_input(
            "\"Recent\" Character Change if within last N bars", min_value=1, max_value=50, value=5, step=1,
            key="structure_choch_lookback",
        )
    with sc6:
        structure_force_refresh = st.checkbox("Force-refresh price history", value=False, key="structure_force_refresh")

    structure_run = st.button("Run Structure scan", type="primary", key="structure_run")

    if structure_run:
        structure_segments = UNIVERSES if structure_universe == ALL_UNIVERSE_LABEL else [structure_universe]
        structure_symbols_df = get_all_symbols(structure_segments)
        st.write(f"Scanning {len(structure_symbols_df)} symbols in {structure_universe} on {structure_tf_label}...")

        structure_progress = st.progress(0.0)
        structure_status = st.empty()
        structure_start = time.time()

        def _structure_cb(i, total, symbol):
            structure_progress.progress((i + 1) / total)
            structure_status.text(f"[{i+1}/{total}] {symbol}")

        structure_results = structure.scan_universe_structure(
            structure_symbols_df[["Symbol", "Segment"]], structure_timeframe, ma_type, int(structure_order),
            int(structure_choch_lookback), progress_cb=_structure_cb, force_refresh=structure_force_refresh,
        )
        structure_elapsed = time.time() - structure_start
        structure_status.text(f"Done in {structure_elapsed:.0f}s — "
                               f"{len(structure_results)}/{len(structure_symbols_df)} symbols had usable data.")

        st.session_state["structure_results"] = structure_results
        st.session_state["structure_scanned_at"] = pd.Timestamp.now()
        st.session_state["structure_scanned_universe"] = structure_universe
        st.session_state["structure_scanned_tf"] = structure_tf_label
        st.session_state["structure_scanned_ma"] = ma_type

    structure_results = st.session_state.get("structure_results")
    structure_records = []

    if structure_results is None:
        st.info("Set your filters above and click **Run Structure scan**. First run of the day is slower "
                 "(full history fetch per stock); later runs reuse the local cache and are fast.")
    elif structure_results.empty:
        st.warning("No symbols returned usable data for this universe/filter combination.")
    else:
        scanned_ma = st.session_state["structure_scanned_ma"]
        st.caption(
            f"Last scanned: {st.session_state['structure_scanned_at']:%Y-%m-%d %H:%M} · "
            f"{st.session_state['structure_scanned_universe']} · {st.session_state['structure_scanned_tf']} · "
            f"{scanned_ma}"
        )

        sdf = structure_results.copy()
        if structure_search.strip():
            search_terms = structure_search.strip().lower().split()
            symbol_lower = sdf["Symbol"].str.lower()
            mask = pd.Series(True, index=sdf.index)
            for term in search_terms:
                mask &= symbol_lower.str.contains(term, na=False)
            sdf = sdf[mask]

        ma_fast_col, ma_slow_col = f"{scanned_ma}{structure.MA_FAST}", f"{scanned_ma}{structure.MA_SLOW}"
        display_cols = ["Symbol", "Close", "AsOf", ma_fast_col, ma_slow_col, "Structure", "Signal", "Volume"]

        def _fmt_structure(d):
            out = d[display_cols].copy()
            out["Close"] = out["Close"].round(2)
            out[ma_fast_col] = out[ma_fast_col].round(2)
            out[ma_slow_col] = out[ma_slow_col].round(2)
            out["AsOf"] = _fmt_asof(out["AsOf"], structure_timeframe)
            return out.sort_values(by="Symbol")

        uptrend = sdf[sdf["Signal"] == "Uptrend"].copy()
        downtrend = sdf[sdf["Signal"] == "Downtrend"].copy()
        choch = sdf[sdf["Signal"].str.startswith("Character Change", na=False)].copy()
        weak_up = sdf[sdf["Signal"].str.startswith("Uptrend (below", na=False)].copy()
        weak_down = sdf[sdf["Signal"].str.startswith("Downtrend (above", na=False)].copy()
        no_structure = sdf[sdf["Signal"] == "No clear structure"].copy()
        no_hist = sdf[sdf["Signal"] == "Insufficient history"].copy()

        st.subheader(f"🔄 Character Change — trend just flipped ({len(choch)})")
        if choch.empty:
            st.write(f"No Character Change events in the last {structure_choch_lookback} bars with these settings.")
        else:
            st.dataframe(_fmt_structure(choch), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Character Change list (CSV)",
                _fmt_structure(choch).to_csv(index=False),
                file_name=f"structure_choch_{structure_timeframe}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
            )

        st.subheader(f"🟢 Uptrend — HH/HL, above {scanned_ma}{structure.MA_FAST}/{structure.MA_SLOW} ({len(uptrend)})")
        if uptrend.empty:
            st.write("No confirmed uptrends with these settings.")
        else:
            st.dataframe(_fmt_structure(uptrend), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Uptrend list (CSV)",
                _fmt_structure(uptrend).to_csv(index=False),
                file_name=f"structure_uptrend_{structure_timeframe}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
            )
            structure_records = _fmt_structure(uptrend)[["Symbol", "Close", "Structure", "Signal"]].to_dict("records")

        st.subheader(f"🔴 Downtrend — LH/LL, below {scanned_ma}{structure.MA_FAST}/{structure.MA_SLOW} ({len(downtrend)})")
        if downtrend.empty:
            st.write("No confirmed downtrends with these settings.")
        else:
            st.dataframe(_fmt_structure(downtrend), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Downtrend list (CSV)",
                _fmt_structure(downtrend).to_csv(index=False),
                file_name=f"structure_downtrend_{structure_timeframe}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
            )

        with st.expander(f"Uptrend structure, but below {scanned_ma}{structure.MA_FAST}/{structure.MA_SLOW} ({len(weak_up)})"):
            st.dataframe(_fmt_structure(weak_up) if not weak_up.empty else pd.DataFrame(),
                         use_container_width=True, hide_index=True)
        with st.expander(f"Downtrend structure, but above {scanned_ma}{structure.MA_FAST}/{structure.MA_SLOW} ({len(weak_down)})"):
            st.dataframe(_fmt_structure(weak_down) if not weak_down.empty else pd.DataFrame(),
                         use_container_width=True, hide_index=True)
        with st.expander(f"No clear structure ({len(no_structure)})"):
            st.dataframe(_fmt_structure(no_structure) if not no_structure.empty else pd.DataFrame(),
                         use_container_width=True, hide_index=True)
        with st.expander(f"Insufficient history ({len(no_hist)})"):
            st.dataframe(no_hist[["Symbol", "AsOf"]] if not no_hist.empty else pd.DataFrame(),
                         use_container_width=True, hide_index=True)

        with st.expander(f"📈 View swing chart for a stock in these results ({len(sdf)} scanned)", expanded=False):
            structure_chart_symbols = sorted(sdf["Symbol"].unique().tolist())
            sc_c1, sc_c2 = st.columns([3, 1])
            with sc_c1:
                structure_chart_pick = st.selectbox("Pick a symbol to chart", structure_chart_symbols,
                                                     key="structure_chart_pick")
            with sc_c2:
                structure_chart_bars = st.number_input("Bars to show", min_value=50, max_value=1000, value=250,
                                                        step=25, key="structure_chart_bars")
            if st.button(f"📈 Show chart for {structure_chart_pick}", key="structure_chart_show"):
                st.session_state["structure_chart_shown"] = structure_chart_pick

            structure_shown_symbol = st.session_state.get("structure_chart_shown")
            if structure_shown_symbol:
                with st.spinner(f"Fetching {structure_shown_symbol}..."):
                    structure_full = load_frame(structure_shown_symbol, structure_timeframe,
                                                 force_refresh=structure_force_refresh)
                if structure_full is None or structure_full.empty:
                    st.error(f"Couldn't fetch usable data for {structure_shown_symbol} on {structure_tf_label}.")
                else:
                    structure_fig = build_structure_chart(structure_full, structure_shown_symbol, structure_tf_label,
                                                            int(structure_order), display_bars=int(structure_chart_bars))
                    st.plotly_chart(structure_fig, use_container_width=True, key="structure_fig",
                                     config=PLOTLY_CONFIG)

    render_telegram_send_button(
        "structure",
        header_lines=[
            f"{datetime.now():%d-%b-%Y %H:%M} IST",
            "📐 Market Structure snapshot — Uptrends",
            f"Universe: {structure_universe} | Timeframe: {structure_tf_label}",
        ],
        records=structure_records,
    )

with tab_backtest:
    st.title("🧪 Backtest")
    bt_strategy = st.radio(
        "Strategy", ["Golden Cross", "CPR + EMA", "Swing Strategy", "Momentum Screener", "EMA Scalping",
                     "Big Money Swing (Approximation)", "Oliver's 20 & 200", "CPR Scalper (MA + RSI + CPR)"],
        horizontal=True, key="bt_strategy")
    st.divider()

if bt_strategy == "Golden Cross":
  with tab_backtest:
    st.subheader("Golden Cross Backtest")
    st.caption(
        "Nifty 100, 50/200 EMA cross on the timeframe below. Entry once price is that much above "
        "the cross bar's close; exit at target or stop-loss, whichever comes first. One trade per "
        "symbol at a time. Uses the same cached price history as the Screener tab."
    )

    bt_tf_options = {"1 Day": "1D", "1 Week": "1W"}
    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    bt_tf_label = bt_col1.radio("Timeframe", list(bt_tf_options.keys()), index=1, horizontal=False,
                                 key="bt_timeframe")
    bt_entry_trigger_pct = bt_col2.number_input("Entry trigger (% above cross)", value=5.0, step=0.5,
                                                 key="bt_entry_trigger") / 100
    bt_target_pct = bt_col3.number_input("Target (%)", value=10.0, step=1.0, key="bt_target") / 100
    bt_stop_pct = bt_col4.number_input("Stop-loss (%)", value=5.0, step=0.5, key="bt_stop") / 100

    bt_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="bt_force_refresh")

    if st.button("▶️ Run Golden Cross Backtest (Nifty 100)", type="primary"):
        bt_progress = st.progress(0.0)
        bt_status = st.empty()
        bt_start = time.time()

        def _bt_cb(i, total, symbol):
            bt_progress.progress((i + 1) / total)
            bt_status.text(f"[{i+1}/{total}] {symbol}")

        bt_trades = backtest.run_backtest(
            ["Nifty 100 (Large Cap)"], timeframe=bt_tf_options[bt_tf_label],
            entry_trigger_pct=bt_entry_trigger_pct, target_pct=bt_target_pct, stop_pct=bt_stop_pct,
            progress_cb=_bt_cb, force_refresh=bt_force_refresh,
        )
        bt_elapsed = time.time() - bt_start
        bt_status.text(f"Done in {bt_elapsed:.0f}s — {len(bt_trades)} trade(s) generated.")

        st.session_state["bt_trades"] = bt_trades
        st.session_state["bt_ran_at"] = pd.Timestamp.now()

    bt_trades = st.session_state.get("bt_trades")

    if bt_trades is None:
        st.info("Click **Run Golden Cross Backtest** above. First run is slow (full history fetch "
                 "per stock); later runs reuse the local cache and are fast.")
    elif bt_trades.empty:
        st.warning("No Golden Cross signals produced a qualifying trade (entry never triggered, or "
                   "not enough weekly history) for this universe.")
    else:
        st.caption(f"Last run: {st.session_state['bt_ran_at']:%Y-%m-%d %H:%M}")

        bt_summary = backtest.summarize(bt_trades)
        bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
        bt_c1.metric("Closed trades", bt_summary["closed_trades"],
                     help=f"{bt_summary['open_trades']} still open (never hit target/stop) as of latest data.")
        bt_c2.metric("Win rate", f"{bt_summary['win_rate_pct']:.1f}%" if pd.notna(bt_summary["win_rate_pct"]) else "—")
        bt_c3.metric("Avg return / trade", f"{bt_summary['avg_return_pct']:.2f}%"
                     if pd.notna(bt_summary["avg_return_pct"]) else "—")
        bt_c4.metric("Max drawdown", f"{bt_summary['max_drawdown_pct']:.2f}%"
                     if pd.notna(bt_summary["max_drawdown_pct"]) else "—")
        st.caption(
            f"Avg winner: {bt_summary['avg_win_pct']:.2f}% · Avg loser: {bt_summary['avg_loss_pct']:.2f}% "
            "· Drawdown is on an equity curve that compounds each closed trade in entry-date order "
            "(a simplification — in reality multiple symbols can be in a trade at once)."
        )

        bt_closed = bt_trades[bt_trades["ExitReason"] != "Open (end of data)"]
        if not bt_closed.empty:
            bt_equity = backtest.equity_curve_from_trades(bt_closed)
            bt_fig = go.Figure()
            bt_fig.add_trace(go.Scatter(
                x=bt_closed.sort_values("EntryDate")["EntryDate"], y=bt_equity,
                mode="lines", name="Equity (x initial capital)",
            ))
            bt_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                  yaxis_title="Equity multiple", xaxis_title="Entry date")
            st.plotly_chart(bt_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")
        st.dataframe(bt_trades, use_container_width=True, hide_index=True)

elif bt_strategy == "CPR + EMA":
  with tab_backtest:
    st.subheader("CPR + EMA Backtest")
    st.warning(
        "**Scoping note:** this was requested on a 15-min timeframe, but that isn't reachable "
        "through the Kite bridge available here — Kite's 15-min history caps each request at "
        "~200 days (so one symbol's full history needs ~13 chunked calls), and tracking every "
        "trade's SL/1R/2R/trailing exit at 15-min precision for its whole holding period would "
        "need thousands more. All of that data has to pass through this chat, which isn't "
        "practical at that volume. **This runs the same rules on daily candles instead** — CPR, "
        "20/50/200 EMA trend filter, breakout entry/SL, 1R/2R partial booking, 20 EMA trailing "
        "exit — which is a faithful read of the strategy (CPR itself is a daily-only indicator, "
        "and the trailing-exit rule was already specified as a *daily* candle close)."
    )
    st.caption(
        "LONG: close above CPR Top (TC) with price above 200 EMA. SHORT: close below CPR Bottom "
        "(BC) with price below 200 EMA. SL = CPR Bottom/Top of the breakout day (never moved). "
        "Exits: 25% at 1R, 30% at 2R, remaining 45% trailed — exit on a daily close back through "
        "the 20 EMA. One trade per symbol at a time. No volume/rejection-candle filter (per your "
        "'objective breakout only' choice)."
    )

    CPR_BT_UNIVERSE = [
        ("RELIANCE", "Nifty 100"), ("HDFCBANK", "Nifty 100"), ("ICICIBANK", "Nifty 100"),
        ("INFY", "Nifty 100"), ("TCS", "Nifty 100"), ("SBIN", "Nifty 100"), ("TATASTEEL", "Nifty 100"),
        ("LT", "Nifty 100"), ("AXISBANK", "Nifty 100"), ("MARUTI", "Nifty 100"),
    ]
    st.caption(f"Universe: {', '.join(s for s, _ in CPR_BT_UNIVERSE)} (10 liquid large caps).")

    cbt_col1, cbt_col2 = st.columns(2)
    cbt_start_date = cbt_col1.date_input("Start date", value=pd.Timestamp("2020-01-01"), key="cbt_start_date")
    cbt_force_refresh = cbt_col2.checkbox("Force-refresh price history (ignore cache)", key="cbt_force_refresh")

    if st.button("▶️ Run CPR + EMA Backtest", type="primary"):
        cbt_progress = st.progress(0.0)
        cbt_status = st.empty()
        cbt_start = time.time()

        def _cbt_cb(i, total, symbol):
            cbt_progress.progress((i + 1) / total)
            cbt_status.text(f"[{i+1}/{total}] {symbol}")

        cbt_trades = cpr_ema_backtest.run_backtest(
            CPR_BT_UNIVERSE, start_date=str(cbt_start_date), progress_cb=_cbt_cb,
            force_refresh=cbt_force_refresh,
        )
        cbt_elapsed = time.time() - cbt_start
        cbt_status.text(f"Done in {cbt_elapsed:.0f}s — {len(cbt_trades)} trade(s) generated.")

        st.session_state["cbt_trades"] = cbt_trades
        st.session_state["cbt_ran_at"] = pd.Timestamp.now()

    cbt_trades = st.session_state.get("cbt_trades")

    if cbt_trades is None:
        st.info("Click **Run CPR + EMA Backtest** above. First run is slow (full history fetch "
                 "per stock); later runs reuse the local cache and are fast.")
    elif cbt_trades.empty:
        st.warning("No qualifying trades for this universe/date range.")
    else:
        st.caption(f"Last run: {st.session_state['cbt_ran_at']:%Y-%m-%d %H:%M}")

        cbt_summary = cpr_ema_backtest.summarize(cbt_trades)
        cbt_c1, cbt_c2, cbt_c3, cbt_c4 = st.columns(4)
        cbt_c1.metric("Total trades", cbt_summary["total_trades"])
        cbt_c2.metric("Win rate", f"{cbt_summary['win_rate_pct']:.1f}%"
                      if pd.notna(cbt_summary["win_rate_pct"]) else "—")
        cbt_c3.metric("Avg return / trade", f"{cbt_summary['avg_return_pct']:.2f}%"
                      if pd.notna(cbt_summary["avg_return_pct"]) else "—")
        cbt_c4.metric("Max drawdown", f"{cbt_summary['max_drawdown_pct']:.2f}%"
                      if pd.notna(cbt_summary["max_drawdown_pct"]) else "—")
        st.caption(
            f"Avg winner: {cbt_summary['avg_win_pct']:.2f}% · Avg loser: {cbt_summary['avg_loss_pct']:.2f}% "
            "· Long/Short split: "
            f"{(cbt_trades['Direction'] == 'Long').sum()} / {(cbt_trades['Direction'] == 'Short').sum()} "
            "· Drawdown is on an equity curve that compounds each trade in entry-date order "
            "(a simplification — in reality multiple symbols can be in a trade at once)."
        )

        cbt_equity = (1 + cbt_trades.sort_values("EntryDate")["ReturnPct"] / 100).cumprod()
        cbt_fig = go.Figure()
        cbt_fig.add_trace(go.Scatter(
            x=cbt_trades.sort_values("EntryDate")["EntryDate"], y=cbt_equity,
            mode="lines", name="Equity (x initial capital)",
        ))
        cbt_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                               yaxis_title="Equity multiple", xaxis_title="Entry date")
        st.plotly_chart(cbt_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")
        st.dataframe(cbt_trades, use_container_width=True, hide_index=True)

elif bt_strategy == "Oliver's 20 & 200":
  with tab_backtest:
    st.subheader("Oliver's 20 & 200 — MA Crossover Backtest")
    st.warning(
        "**Scoping note:** 5-minute data comes from your Kite account via the MCP bridge, which "
        "only works inside a Claude session — this Streamlit process can't call it directly. So "
        "the price history below is a one-off snapshot pulled and cached to "
        "`data/kite_5min/<SYMBOL>.csv`, not a live feed. Ask Claude to re-pull it for a fresher or "
        "longer window. Universe: all 100 Nifty 100 stocks + the NIFTY 50 index are cached "
        "(~3 months each, RELIANCE has ~6.5 months). Midcap/Smallcap aren't pulled yet."
    )
    st.caption(
        "LONG: fast MA crosses above slow MA (5-min bars). SHORT: fast MA crosses below slow MA. "
        "Entry at the crossover bar's close. Exit at a fixed stop-loss or target (in % of entry "
        "price, or absolute points), whichever is hit first (stop-loss assumed to win if a bar "
        "touches both). One trade at a time per symbol."
    )

    olv_cache_dir = oliver_ma_cross_backtest.KITE_5MIN_DIR
    olv_available_symbols = sorted(
        os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(olv_cache_dir, "*.csv"))
    ) or ["RELIANCE"]
    olv_nifty100_symbols = [s for s in olv_available_symbols if s in set(
        get_all_symbols(["Nifty 100 (Large Cap)"])["Symbol"]
    )]

    olv_universe_options = ["Nifty 100 (Large Cap)", "NIFTY 50 Index", "Custom selection"]
    olv_universe = st.selectbox("Universe", olv_universe_options, key="olv_universe")

    if olv_universe == "Nifty 100 (Large Cap)":
        olv_symbols = olv_nifty100_symbols
        st.caption(f"{len(olv_symbols)} stocks selected.")
    elif olv_universe == "NIFTY 50 Index":
        olv_symbols = ["NIFTY50"] if "NIFTY50" in olv_available_symbols else []
        if not olv_symbols:
            st.warning("NIFTY 50 index data not cached yet — ask Claude to pull it via Kite.")
    else:
        olv_symbols = st.multiselect("Symbols", olv_available_symbols, default=olv_available_symbols,
                                      key="olv_symbols")

    olv_col1, olv_col2 = st.columns(2)
    olv_ma_type = olv_col1.radio("MA type", ["SMA", "EMA"], horizontal=True, key="olv_ma_type")
    olv_unit_label = olv_col2.radio("SL/Target unit", ["% of entry price", "Points"], horizontal=True,
                                     key="olv_unit_label")
    olv_unit = "points" if olv_unit_label == "Points" else "pct"

    olv_col3, olv_col4 = st.columns(2)
    if olv_unit == "points":
        olv_sl_value = olv_col3.slider("Stop-loss (points)", min_value=1.0, max_value=500.0,
                                        value=oliver_ma_cross_backtest.DEFAULT_SL_POINTS, step=1.0,
                                        key="olv_sl_points")
        olv_target_value = olv_col4.slider("Target (points)", min_value=1.0, max_value=1000.0,
                                            value=oliver_ma_cross_backtest.DEFAULT_TARGET_POINTS, step=1.0,
                                            key="olv_target_points")
    else:
        olv_sl_value = olv_col3.slider("Stop-loss (%)", min_value=0.25, max_value=10.0,
                                        value=oliver_ma_cross_backtest.DEFAULT_SL_PCT, step=0.25,
                                        key="olv_sl_pct")
        olv_target_value = olv_col4.slider("Target (%)", min_value=0.25, max_value=100.0,
                                            value=oliver_ma_cross_backtest.DEFAULT_TARGET_PCT, step=0.25,
                                            key="olv_target_pct")

    olv_col5, olv_col6 = st.columns(2)
    olv_fast = olv_col5.number_input("Fast MA period", value=oliver_ma_cross_backtest.FAST_PERIOD, step=1,
                                      key="olv_fast")
    olv_slow = olv_col6.number_input("Slow MA period", value=oliver_ma_cross_backtest.SLOW_PERIOD, step=1,
                                      key="olv_slow")

    if st.button("▶️ Run Oliver's 20 & 200 Backtest", type="primary", disabled=not olv_symbols):
        olv_trades = oliver_ma_cross_backtest.run_backtest(
            olv_symbols, ma_type=olv_ma_type, fast=int(olv_fast), slow=int(olv_slow),
            sl_value=olv_sl_value, target_value=olv_target_value, unit=olv_unit,
        )
        st.session_state["olv_trades"] = olv_trades
        st.session_state["olv_ran_at"] = pd.Timestamp.now()

    olv_trades = st.session_state.get("olv_trades")

    if olv_trades is None:
        st.info("Click **Run Oliver's 20 & 200 Backtest** above.")
    elif olv_trades.empty:
        st.warning(
            "No crossover signals produced a trade — either no 5-min data is cached yet for this "
            "symbol (ask Claude to pull it via Kite), or not enough bars for the slow MA to warm up."
        )
    else:
        st.caption(f"Last run: {st.session_state['olv_ran_at']:%Y-%m-%d %H:%M}")

        olv_summary = oliver_ma_cross_backtest.summarize(olv_trades)
        olv_c1, olv_c2, olv_c3, olv_c4 = st.columns(4)
        olv_c1.metric("Closed trades", olv_summary["closed_trades"],
                      help=f"{olv_summary['open_trades']} still open (never hit target/stop) as of latest data.")
        olv_c2.metric("Win rate", f"{olv_summary['win_rate_pct']:.1f}%"
                      if pd.notna(olv_summary["win_rate_pct"]) else "—")
        olv_c3.metric("Avg return / trade", f"{olv_summary['avg_return_pct']:.2f}%"
                      if pd.notna(olv_summary["avg_return_pct"]) else "—")
        olv_c4.metric("Max drawdown", f"{olv_summary['max_drawdown_pct']:.2f}%"
                      if pd.notna(olv_summary["max_drawdown_pct"]) else "—")
        st.caption(
            f"Avg winner: {olv_summary['avg_win_pct']:.2f}% · Avg loser: {olv_summary['avg_loss_pct']:.2f}% "
            "· Long/Short split: "
            f"{(olv_trades['Direction'] == 'Long').sum()} / {(olv_trades['Direction'] == 'Short').sum()}"
        )

        if olv_trades["Symbol"].nunique() > 1:
            st.subheader("Per-symbol breakdown")
            olv_per_symbol = []
            for sym, grp in olv_trades.groupby("Symbol"):
                grp_closed = grp[grp["ExitReason"] != "Open (end of data)"]
                olv_per_symbol.append({
                    "Symbol": sym, "Trades": len(grp),
                    "Win rate %": round((grp_closed["ReturnPct"] > 0).mean() * 100, 1) if len(grp_closed) else None,
                    "Avg return %": round(grp_closed["ReturnPct"].mean(), 2) if len(grp_closed) else None,
                })
            st.dataframe(pd.DataFrame(olv_per_symbol).sort_values("Avg return %", ascending=False),
                         use_container_width=True, hide_index=True)

        olv_closed = olv_trades[olv_trades["ExitReason"] != "Open (end of data)"]
        if not olv_closed.empty:
            olv_equity = oliver_ma_cross_backtest.equity_curve_from_trades(olv_closed)
            olv_fig = go.Figure()
            olv_fig.add_trace(go.Scatter(
                x=olv_closed.sort_values("EntryTime")["EntryTime"], y=olv_equity,
                mode="lines", name="Equity (x initial capital)",
            ))
            olv_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                   yaxis_title="Equity multiple", xaxis_title="Entry time")
            st.plotly_chart(olv_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")

        def _olv_fmt_dt(ts) -> str:
            hour = ts.strftime("%I").lstrip("0") or "12"
            return f"{ts.strftime('%d-%b-%Y')} {hour}.{ts.strftime('%M')} {ts.strftime('%p').lower()}"

        olv_trades_show = pd.DataFrame({
            "Symbol": olv_trades["Symbol"],
            "Direction": olv_trades["Direction"],
            "MA Type": [f"{r.MAType}, {_olv_fmt_dt(r.EntryTime)}" for r in olv_trades.itertuples()],
            "Entry": olv_trades["EntryPrice"],
            "SL": olv_trades["SLPrice"],
            "Target": olv_trades["TargetPrice"],
            "Exit Price": [f"{r.ExitPrice:.2f}, {_olv_fmt_dt(r.ExitTime)}" for r in olv_trades.itertuples()],
            "Exit Reason": olv_trades["ExitReason"],
            "Return %": olv_trades["ReturnPct"],
        })
        st.dataframe(
            olv_trades_show, use_container_width=True, hide_index=True,
            column_config={
                "Entry": st.column_config.NumberColumn(format="%.2f"),
                "SL": st.column_config.NumberColumn(format="%.2f"),
                "Target": st.column_config.NumberColumn(format="%.2f"),
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

elif bt_strategy == "CPR Scalper (MA + RSI + CPR)":
  with tab_backtest:
    st.subheader("CPR Scalper — MA Cross + RSI + Nearest CPR Level")
    st.warning(
        "**Scoping note:** same one-off Kite 5-min snapshot as Oliver's 20 & 200 "
        "(`data/kite_5min/<SYMBOL>.csv`), not a live feed — ask Claude to re-pull it for a fresher "
        "or longer window."
    )
    st.caption(
        "LONG: fast MA crosses above slow MA (5-min bars), AND RSI(14) is above the Long threshold "
        "on the cross bar's close. SHORT: fast MA crosses below slow MA, AND RSI(14) is below the "
        "Short threshold. A crossover that fails the RSI check is not traded at all. "
        "**Target:** the nearest CPR level above entry for a Long / below entry for a Short "
        "(BC, TC, R1 or R2 — CPR for the day is derived from the prior day's High/Low/Close in this "
        "same 5-min dataset). **Stop-loss:** the cross bar's own Low (Long) / High (Short) — not "
        "specified in the original rules, a practical default. One trade at a time per symbol."
    )

    cs_cache_dir = oliver_ma_cross_backtest.KITE_5MIN_DIR
    cs_available_symbols = sorted(
        os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(cs_cache_dir, "*.csv"))
    ) or ["RELIANCE"]
    cs_nifty100_symbols = [s for s in cs_available_symbols if s in set(
        get_all_symbols(["Nifty 100 (Large Cap)"])["Symbol"]
    )]

    cs_universe_options = ["Nifty 100 (Large Cap)", "NIFTY 50 Index", "Custom selection"]
    cs_universe = st.selectbox("Universe", cs_universe_options, key="cs_universe")

    if cs_universe == "Nifty 100 (Large Cap)":
        cs_symbols = cs_nifty100_symbols
        st.caption(f"{len(cs_symbols)} stocks selected.")
    elif cs_universe == "NIFTY 50 Index":
        cs_symbols = ["NIFTY50"] if "NIFTY50" in cs_available_symbols else []
        if not cs_symbols:
            st.warning("NIFTY 50 index data not cached yet — ask Claude to pull it via Kite.")
    else:
        cs_symbols = st.multiselect("Symbols", cs_available_symbols, default=cs_available_symbols,
                                     key="cs_symbols")

    cs_col1, cs_col2 = st.columns(2)
    cs_ma_type = cs_col1.radio("MA type", ["SMA", "EMA"], horizontal=True, key="cs_ma_type")
    cs_col1b, cs_col2b = st.columns(2)
    cs_fast = cs_col1b.number_input("Fast MA period", value=oliver_ma_cross_backtest.FAST_PERIOD, step=1,
                                     key="cs_fast")
    cs_slow = cs_col2b.number_input("Slow MA period", value=oliver_ma_cross_backtest.SLOW_PERIOD, step=1,
                                     key="cs_slow")

    cs_col3, cs_col4 = st.columns(2)
    cs_long_rsi_min = cs_col3.slider("Long RSI threshold (must be above)", min_value=0.0, max_value=100.0,
                                      value=cpr_rsi_scalp_backtest.DEFAULT_LONG_RSI_MIN, step=1.0,
                                      key="cs_long_rsi_min")
    cs_short_rsi_max = cs_col4.slider("Short RSI threshold (must be below)", min_value=0.0, max_value=100.0,
                                       value=cpr_rsi_scalp_backtest.DEFAULT_SHORT_RSI_MAX, step=1.0,
                                       key="cs_short_rsi_max")

    if st.button("▶️ Run CPR Scalper Backtest", type="primary", disabled=not cs_symbols):
        cs_trades = cpr_rsi_scalp_backtest.run_backtest(
            cs_symbols, ma_type=cs_ma_type, fast=int(cs_fast), slow=int(cs_slow),
            long_rsi_min=cs_long_rsi_min, short_rsi_max=cs_short_rsi_max,
        )
        st.session_state["cs_trades"] = cs_trades
        st.session_state["cs_ran_at"] = pd.Timestamp.now()

    cs_trades = st.session_state.get("cs_trades")

    if cs_trades is None:
        st.info("Click **Run CPR Scalper Backtest** above.")
    elif cs_trades.empty:
        st.warning(
            "No trade cleared both filters — either no 5-min data is cached yet for this symbol "
            "(ask Claude to pull it via Kite), not enough bars for the slow MA to warm up, every "
            "crossover failed the RSI check, or price was already past every CPR level on the day."
        )
    else:
        st.caption(f"Last run: {st.session_state['cs_ran_at']:%Y-%m-%d %H:%M}")

        cs_summary = oliver_ma_cross_backtest.summarize(cs_trades)
        cs_c1, cs_c2, cs_c3, cs_c4 = st.columns(4)
        cs_c1.metric("Closed trades", cs_summary["closed_trades"],
                      help=f"{cs_summary['open_trades']} still open (never hit target/stop) as of latest data.")
        cs_c2.metric("Win rate", f"{cs_summary['win_rate_pct']:.1f}%"
                      if pd.notna(cs_summary["win_rate_pct"]) else "—")
        cs_c3.metric("Avg return / trade", f"{cs_summary['avg_return_pct']:.2f}%"
                      if pd.notna(cs_summary["avg_return_pct"]) else "—")
        cs_c4.metric("Max drawdown", f"{cs_summary['max_drawdown_pct']:.2f}%"
                      if pd.notna(cs_summary["max_drawdown_pct"]) else "—")
        st.caption(
            f"Avg winner: {cs_summary['avg_win_pct']:.2f}% · Avg loser: {cs_summary['avg_loss_pct']:.2f}% "
            "· Long/Short split: "
            f"{(cs_trades['Direction'] == 'Long').sum()} / {(cs_trades['Direction'] == 'Short').sum()}"
        )

        if cs_trades["Symbol"].nunique() > 1:
            st.subheader("Per-symbol breakdown")
            cs_per_symbol = []
            for sym, grp in cs_trades.groupby("Symbol"):
                grp_closed = grp[grp["ExitReason"] != "Open (end of data)"]
                cs_per_symbol.append({
                    "Symbol": sym, "Trades": len(grp),
                    "Win rate %": round((grp_closed["ReturnPct"] > 0).mean() * 100, 1) if len(grp_closed) else None,
                    "Avg return %": round(grp_closed["ReturnPct"].mean(), 2) if len(grp_closed) else None,
                })
            st.dataframe(pd.DataFrame(cs_per_symbol).sort_values("Avg return %", ascending=False),
                         use_container_width=True, hide_index=True)

        cs_closed = cs_trades[cs_trades["ExitReason"] != "Open (end of data)"]
        if not cs_closed.empty:
            cs_equity = oliver_ma_cross_backtest.equity_curve_from_trades(cs_closed)
            cs_fig = go.Figure()
            cs_fig.add_trace(go.Scatter(
                x=cs_closed.sort_values("EntryTime")["EntryTime"], y=cs_equity,
                mode="lines", name="Equity (x initial capital)",
            ))
            cs_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                  yaxis_title="Equity multiple", xaxis_title="Entry time")
            st.plotly_chart(cs_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")

        def _cs_fmt_dt(ts) -> str:
            hour = ts.strftime("%I").lstrip("0") or "12"
            return f"{ts.strftime('%d-%b-%Y')} {hour}.{ts.strftime('%M')} {ts.strftime('%p').lower()}"

        cs_trades_show = pd.DataFrame({
            "Symbol": cs_trades["Symbol"],
            "Direction": cs_trades["Direction"],
            "MA Type": [f"{r.MAType}, {_cs_fmt_dt(r.EntryTime)}" for r in cs_trades.itertuples()],
            "Entry": cs_trades["EntryPrice"],
            "RSI@Entry": cs_trades["RSIAtEntry"],
            "SL": cs_trades["SLPrice"],
            "Target": cs_trades["TargetPrice"],
            "Target Level": cs_trades["TargetLevel"],
            "Exit Price": [f"{r.ExitPrice:.2f}, {_cs_fmt_dt(r.ExitTime)}" for r in cs_trades.itertuples()],
            "Exit Reason": cs_trades["ExitReason"],
            "Return %": cs_trades["ReturnPct"],
        })
        st.dataframe(
            cs_trades_show, use_container_width=True, hide_index=True,
            column_config={
                "Entry": st.column_config.NumberColumn(format="%.2f"),
                "RSI@Entry": st.column_config.NumberColumn(format="%.1f"),
                "SL": st.column_config.NumberColumn(format="%.2f"),
                "Target": st.column_config.NumberColumn(format="%.2f"),
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

elif bt_strategy == "Swing Strategy":
  with tab_backtest:
    st.subheader("Swing Strategy Backtest")
    st.caption(
        "Two positional-swing entries, both built around one rule: stop-loss = Entry − 1.5× ATR(14), "
        "scaled to each stock's own volatility — skip the trade if that implies >4.5% risk, rather "
        "than shrinking the stop to fit (an artificially tight stop just gets whipsawed). "
        "Target = max(2× actual risk, 5%). **Volatility Contraction Breakout** — 200 EMA trend filter "
        "+ Narrow CPR + close above CPR Top on volume ≥1.5× average. **Trend Pullback Entry** — "
        "Market Structure uptrend + pullback to the rising 20 EMA + reclaim above the pullback bar's "
        "high. One trade per symbol at a time, long-only."
    )

    swbt_strategy_label = st.radio(
        "Which strategy", ["Volatility Contraction Breakout", "Trend Pullback Entry", "Both"],
        horizontal=True, key="swbt_strategy_label",
    )
    swbt_strategy_key = {"Volatility Contraction Breakout": "breakout", "Trend Pullback Entry": "pullback",
                          "Both": "both"}[swbt_strategy_label]

    swbt_universes = st.multiselect(
        "Universe", ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"],
        default=["Nifty 100 (Large Cap)", "Nifty Midcap 150"], key="swbt_universes",
    )

    swbt_col1, swbt_col2, swbt_col3 = st.columns(3)
    swbt_start_date = swbt_col1.date_input("Start date", value=pd.Timestamp("2020-01-01"), key="swbt_start_date")
    swbt_risk_per_trade = swbt_col2.number_input(
        "Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="swbt_risk_per_trade",
    )
    swbt_max_concurrent = swbt_col3.number_input(
        "Portfolio sim: max concurrent positions", min_value=1, max_value=100, value=20, step=1,
        key="swbt_max_concurrent",
    )
    swbt_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="swbt_force_refresh")

    if st.button("▶️ Run Swing Strategy Backtest", type="primary"):
        if not swbt_universes:
            st.warning("Pick at least one universe first.")
        else:
            swbt_symbols_df = get_all_symbols(swbt_universes, force_refresh=swbt_force_refresh)[["Symbol", "Segment"]]
            swbt_symbols = list(swbt_symbols_df.itertuples(index=False, name=None))

            swbt_progress = st.progress(0.0)
            swbt_status = st.empty()
            swbt_start = time.time()

            def _swbt_cb(i, total, symbol):
                swbt_progress.progress((i + 1) / total)
                swbt_status.text(f"[{i+1}/{total}] {symbol}")

            swbt_trades = swing_backtest.run_backtest(
                swbt_strategy_key, swbt_symbols, start_date=str(swbt_start_date), progress_cb=_swbt_cb,
                force_refresh=swbt_force_refresh,
            )
            swbt_elapsed = time.time() - swbt_start
            swbt_status.text(f"Done in {swbt_elapsed:.0f}s — {len(swbt_trades)} trade(s) generated.")

            st.session_state["swbt_trades"] = swbt_trades
            st.session_state["swbt_ran_at"] = pd.Timestamp.now()

    swbt_trades = st.session_state.get("swbt_trades")

    if swbt_trades is None:
        st.info("Click **Run Swing Strategy Backtest** above. First run is slow (full history fetch "
                 "per stock); later runs reuse the local cache and are fast.")
    elif swbt_trades.empty:
        st.warning("No qualifying trades for this universe/date range.")
    else:
        st.caption(f"Last run: {st.session_state['swbt_ran_at']:%Y-%m-%d %H:%M}")

        for swbt_label in swbt_trades["Strategy"].unique():
            swbt_sub = swbt_trades[swbt_trades["Strategy"] == swbt_label]
            swbt_summary = swing_backtest.summarize(swbt_sub)
            swbt_port = swing_backtest.simulate_portfolio(
                swbt_sub, risk_per_trade_pct=swbt_risk_per_trade, max_concurrent=int(swbt_max_concurrent),
            )

            st.markdown(f"#### {swbt_label}")
            swbt_c1, swbt_c2, swbt_c3, swbt_c4, swbt_c5 = st.columns(5)
            swbt_c1.metric("Closed trades", swbt_summary["closed_trades"])
            swbt_c2.metric("Win rate", f"{swbt_summary['win_rate_pct']:.1f}%"
                           if pd.notna(swbt_summary["win_rate_pct"]) else "—")
            swbt_c3.metric("Expectancy / trade", f"{swbt_summary['expectancy_pct']:.2f}%"
                           if pd.notna(swbt_summary["expectancy_pct"]) else "—")
            swbt_c4.metric("Portfolio return", f"{swbt_port['total_return_pct']:.1f}%",
                           help=f"Started at 100, risking {swbt_risk_per_trade}% per trade, "
                                f"max {int(swbt_max_concurrent)} concurrent, no leverage.")
            swbt_c5.metric("Portfolio max drawdown", f"{swbt_port['max_drawdown_pct']:.2f}%"
                           if pd.notna(swbt_port["max_drawdown_pct"]) else "—")
            st.caption(
                f"Avg winner: {swbt_summary['avg_win_pct']:.2f}% · Avg loser: {swbt_summary['avg_loss_pct']:.2f}% "
                f"· Trades taken: {swbt_port['trades_taken']} · skipped (no free slot): {swbt_port['trades_skipped']}"
            )

            if not swbt_port["equity_curve"].empty:
                swbt_fig = go.Figure()
                swbt_fig.add_trace(go.Scatter(
                    x=swbt_port["equity_curve"].index, y=swbt_port["equity_curve"].values,
                    mode="lines", name="Capital",
                ))
                swbt_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                        yaxis_title="Capital (started at 100)", xaxis_title="Exit date")
                st.plotly_chart(swbt_fig, use_container_width=True, config=PLOTLY_CONFIG)

        if swbt_strategy_key == "both":
            swbt_port_combined = swing_backtest.simulate_portfolio(
                swbt_trades, risk_per_trade_pct=swbt_risk_per_trade, max_concurrent=int(swbt_max_concurrent),
            )
            st.markdown("#### Combined (both strategies, shared capital & slots)")
            swbt_cc1, swbt_cc2 = st.columns(2)
            swbt_cc1.metric("Portfolio return", f"{swbt_port_combined['total_return_pct']:.1f}%")
            swbt_cc2.metric("Portfolio max drawdown", f"{swbt_port_combined['max_drawdown_pct']:.2f}%"
                            if pd.notna(swbt_port_combined["max_drawdown_pct"]) else "—")

        st.subheader("Trades")
        st.dataframe(swbt_trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all trades (CSV)", swbt_trades.to_csv(index=False),
            file_name=f"swing_backtest_{swbt_strategy_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )

elif bt_strategy == "Momentum Screener":
  with tab_backtest:
    st.subheader("Momentum Screener Backtest")
    st.caption(
        "**Daily**: yesterday's volume > 1.5× the 20-day average, price ₹50–₹5000, ATR(14)/price "
        "> 1.5%, close above both the 20-day and 50-day SMA, RSI(14) between 40–70, and 5-day "
        "return beats the Nifty 50's 5-day return. **Weekly**: a separate rule set — 20-week SMA "
        "> 200-week SMA, weekly volume > the 20-week average, weekly RSI(14) between 45–65."
    )
    st.warning(
        "The screener only defines entry conditions, not an exit — this reuses the same house-style "
        "exit as the Swing Strategy backtest for consistency: stop-loss = Entry − 1.5× ATR(14), skip "
        "the trade if that implies >4.5% risk, target = max(2× actual risk, 5%)."
    )

    mom_mode_label = st.radio("Which mode", ["Daily", "Weekly", "Both"], horizontal=True, key="mom_mode_label")
    mom_mode_key = {"Daily": "daily", "Weekly": "weekly", "Both": "both"}[mom_mode_label]

    mom_universes = st.multiselect(
        "Universe", ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"],
        default=["Nifty 100 (Large Cap)", "Nifty Midcap 150"], key="mom_universes",
    )

    mom_col1, mom_col2, mom_col3 = st.columns(3)
    mom_start_date = mom_col1.date_input("Start date", value=pd.Timestamp("2020-01-01"), key="mom_start_date")
    mom_risk_per_trade = mom_col2.number_input(
        "Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="mom_risk_per_trade",
    )
    mom_max_concurrent = mom_col3.number_input(
        "Portfolio sim: max concurrent positions", min_value=1, max_value=100, value=20, step=1,
        key="mom_max_concurrent",
    )
    mom_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="mom_force_refresh")

    if st.button("▶️ Run Momentum Screener Backtest", type="primary"):
        if not mom_universes:
            st.warning("Pick at least one universe first.")
        else:
            mom_symbols_df = get_all_symbols(mom_universes, force_refresh=mom_force_refresh)[["Symbol", "Segment"]]
            mom_symbols = list(mom_symbols_df.itertuples(index=False, name=None))

            mom_progress = st.progress(0.0)
            mom_status = st.empty()
            mom_start = time.time()

            def _mom_cb(i, total, symbol):
                mom_progress.progress((i + 1) / total)
                mom_status.text(f"[{i+1}/{total}] {symbol}")

            mom_trades = momentum_backtest.run_backtest(
                mom_mode_key, mom_symbols, start_date=str(mom_start_date), progress_cb=_mom_cb,
                force_refresh=mom_force_refresh,
            )
            mom_elapsed = time.time() - mom_start
            mom_status.text(f"Done in {mom_elapsed:.0f}s — {len(mom_trades)} trade(s) generated.")

            st.session_state["mom_trades"] = mom_trades
            st.session_state["mom_ran_at"] = pd.Timestamp.now()

    mom_trades = st.session_state.get("mom_trades")

    if mom_trades is None:
        st.info("Click **Run Momentum Screener Backtest** above. First run is slow (full history "
                 "fetch per stock); later runs reuse the local cache and are fast.")
    elif mom_trades.empty:
        st.warning("No qualifying trades for this universe/date range — this is a strict multi-"
                    "condition filter, so few or zero hits on a given universe/period is expected, "
                    "not necessarily a bug.")
    else:
        st.caption(f"Last run: {st.session_state['mom_ran_at']:%Y-%m-%d %H:%M}")

        for mom_label in mom_trades["Strategy"].unique():
            mom_sub = mom_trades[mom_trades["Strategy"] == mom_label]
            mom_summary = momentum_backtest.summarize(mom_sub)
            mom_port = momentum_backtest.simulate_portfolio(
                mom_sub, risk_per_trade_pct=mom_risk_per_trade, max_concurrent=int(mom_max_concurrent),
            )

            st.markdown(f"#### {mom_label}")
            mom_c1, mom_c2, mom_c3, mom_c4, mom_c5 = st.columns(5)
            mom_c1.metric("Closed trades", mom_summary["closed_trades"])
            mom_c2.metric("Win rate", f"{mom_summary['win_rate_pct']:.1f}%"
                          if pd.notna(mom_summary["win_rate_pct"]) else "—")
            mom_c3.metric("Expectancy / trade", f"{mom_summary['expectancy_pct']:.2f}%"
                          if pd.notna(mom_summary["expectancy_pct"]) else "—")
            mom_c4.metric("Portfolio return", f"{mom_port['total_return_pct']:.1f}%",
                          help=f"Started at 100, risking {mom_risk_per_trade}% per trade, "
                               f"max {int(mom_max_concurrent)} concurrent, no leverage.")
            mom_c5.metric("Portfolio max drawdown", f"{mom_port['max_drawdown_pct']:.2f}%"
                          if pd.notna(mom_port["max_drawdown_pct"]) else "—")
            st.caption(
                f"Avg winner: {mom_summary['avg_win_pct']:.2f}% · Avg loser: {mom_summary['avg_loss_pct']:.2f}% "
                f"· Trades taken: {mom_port['trades_taken']} · skipped (no free slot): {mom_port['trades_skipped']}"
            )

            if not mom_port["equity_curve"].empty:
                mom_fig = go.Figure()
                mom_fig.add_trace(go.Scatter(
                    x=mom_port["equity_curve"].index, y=mom_port["equity_curve"].values,
                    mode="lines", name="Capital",
                ))
                mom_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                       yaxis_title="Capital (started at 100)", xaxis_title="Exit date")
                st.plotly_chart(mom_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")
        st.dataframe(mom_trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all trades (CSV)", mom_trades.to_csv(index=False),
            file_name=f"momentum_backtest_{mom_mode_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )

elif bt_strategy == "EMA Scalping":
  with tab_backtest:
    st.subheader("9/21 EMA Cross Scalping Backtest")
    st.caption(
        "1-min or 5-min bars. **Long**: 9 EMA crosses above 21 EMA on a bar where price is also above "
        "both EMAs and RSI(14) is between 50 and 70 — entered at that bar's close. **Short** is the "
        "exact mirror (cross down, price below both EMAs, RSI between 30 and 50). Stop-loss = a small "
        "buffer beyond the recent swing low/high (last 10 bars, same session); target = risk × the R:R "
        "below. Forced flat at the session's last bar if neither is hit first — this is an "
        "intraday-only strategy, no overnight holds."
    )
    st.caption(
        "The 30/70 bounds are the same RSI Range used on the Screener tab's home-page RSI screener — "
        "applied here so an entry is never taken once RSI has already run into overbought/oversold "
        "territory, not just \"which side is in control\" (the original >50/<50 rule alone)."
    )
    st.warning(
        "**Yahoo's intraday retention caps this backtest's window hard**: 1-minute bars go back only "
        "~7 days, 5-minute bars ~60 days — this cannot be a multi-year backtest like the daily-bar "
        "strategies above. Read any result here as a read on *recent* market behavior, not a long-run "
        "edge. \"A few ticks\" beyond the swing point (from the source rules) has no real tick-size "
        "data behind it here — approximated as a small % buffer instead."
    )

    ema_universes = st.multiselect(
        "Universe", ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"],
        default=["Nifty 100 (Large Cap)"], key="ema_universes",
        help="Scalping wants liquid names — the source rules call for major indices or liquid "
             "large-caps specifically. Mid/Smallcap are offered but expect wider slippage in reality.",
    )

    ema_col1, ema_col2, ema_col3 = st.columns(3)
    ema_timeframe_label = ema_col1.radio("Timeframe", ["1 Minute", "5 Minute"], index=1,
                                          horizontal=True, key="ema_timeframe_label")
    ema_timeframe = {"1 Minute": "1MIN", "5 Minute": "5MIN"}[ema_timeframe_label]
    ema_rr_label = ema_col2.radio("Risk : Reward", ["1 : 1", "1 : 1.5"], index=1,
                                   horizontal=True, key="ema_rr_label")
    ema_rr_multiple = {"1 : 1": 1.0, "1 : 1.5": 1.5}[ema_rr_label]
    ema_risk_per_trade = ema_col3.number_input(
        "Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="ema_risk_per_trade",
    )
    ema_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="ema_force_refresh")

    if st.button("▶️ Run EMA Scalping Backtest", type="primary"):
        if not ema_universes:
            st.warning("Pick at least one universe first.")
        else:
            ema_symbols_df = get_all_symbols(ema_universes, force_refresh=ema_force_refresh)[["Symbol", "Segment"]]
            ema_symbols = list(ema_symbols_df.itertuples(index=False, name=None))

            ema_progress = st.progress(0.0)
            ema_status = st.empty()
            ema_start = time.time()

            def _ema_cb(i, total, symbol):
                ema_progress.progress((i + 1) / total)
                ema_status.text(f"[{i+1}/{total}] {symbol}")

            ema_trades = scalping_backtest.run_backtest(
                ema_symbols, timeframe=ema_timeframe, rr_multiple=ema_rr_multiple, progress_cb=_ema_cb,
                force_refresh=ema_force_refresh,
            )
            ema_elapsed = time.time() - ema_start
            ema_status.text(f"Done in {ema_elapsed:.0f}s — {len(ema_trades)} trade(s) generated.")

            st.session_state["ema_trades"] = ema_trades
            st.session_state["ema_ran_at"] = pd.Timestamp.now()

    ema_trades = st.session_state.get("ema_trades")

    if ema_trades is None:
        st.info("Click **Run EMA Scalping Backtest** above. First run is slow (full intraday history "
                 "fetch per stock); later runs reuse the local cache and are fast.")
    elif ema_trades.empty:
        st.warning("No qualifying trades — try the other timeframe, a wider universe, or check that "
                    "price history has cached in first (run a scan on this universe/timeframe first).")
    else:
        st.caption(f"Last run: {st.session_state['ema_ran_at']:%Y-%m-%d %H:%M}")

        ema_summary = scalping_backtest.summarize(ema_trades)
        ema_port = scalping_backtest.simulate_portfolio(ema_trades, risk_per_trade_pct=ema_risk_per_trade)

        ema_c1, ema_c2, ema_c3, ema_c4, ema_c5 = st.columns(5)
        ema_c1.metric("Closed trades", ema_summary["closed_trades"])
        ema_c2.metric("Win rate", f"{ema_summary['win_rate_pct']:.1f}%"
                      if pd.notna(ema_summary["win_rate_pct"]) else "—")
        ema_c3.metric("Avg return / trade", f"{ema_summary['avg_return_pct']:.2f}%"
                      if pd.notna(ema_summary["avg_return_pct"]) else "—")
        ema_c4.metric("Portfolio return", f"{ema_port['total_return_pct']:.2f}%",
                      help=f"Started at 100, risking {ema_risk_per_trade}% of capital per trade, "
                           "ordered by full entry/exit timestamp (not just date) since scalp trades "
                           "often open and close same-day.")
        ema_c5.metric("Portfolio max drawdown", f"{ema_port['max_drawdown_pct']:.2f}%"
                      if pd.notna(ema_port["max_drawdown_pct"]) else "—")
        st.caption(
            f"Avg winner: {ema_summary['avg_win_pct']:.2f}% · Avg loser: {ema_summary['avg_loss_pct']:.2f}% "
            f"· Long/Short split: {(ema_trades['Direction'] == 'Long').sum()} / "
            f"{(ema_trades['Direction'] == 'Short').sum()} "
            f"· Trades taken: {ema_port['trades_taken']} · skipped (no free slot): {ema_port['trades_skipped']}"
        )

        if not ema_port["equity_curve"].empty:
            ema_fig = go.Figure()
            ema_fig.add_trace(go.Scatter(
                x=ema_port["equity_curve"].index, y=ema_port["equity_curve"].values,
                mode="lines", name="Capital",
            ))
            ema_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                   yaxis_title="Capital (started at 100)", xaxis_title="Exit time")
            st.plotly_chart(ema_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")
        st.dataframe(ema_trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all trades (CSV)", ema_trades.to_csv(index=False),
            file_name=f"scalping_backtest_{ema_timeframe}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )

elif bt_strategy == "Big Money Swing (Approximation)":
  with tab_backtest:
    st.subheader("Big Money Swing Backtest — an Approximation")
    st.error(
        "**This is NOT CA Afzal Lokhandwala's actual trading system.** His specific mechanical rules "
        "(what precisely counts as \"big money\" price/volume behavior, his exact anticipation-entry "
        "trigger, his stop-loss method) are taught in his paid Champion Trading Course and aren't "
        "published anywhere publicly. Every public source (his own site, interviews, YouTube) only "
        "describes the philosophy below in general terms — there's no public spec to implement "
        "faithfully. This is TrendLine's own mechanical translation of that public philosophy, built "
        "to be testable. Read results as evidence about *this specific approximation*, never as "
        "evidence about his real, undisclosed system."
    )
    st.caption(
        "**Publicly stated philosophy this approximates**: price/volume-only technical analysis (no "
        "fundamentals), swing entries taken in *anticipation* of a breakout rather than after it "
        "confirms, no intraday/options, 1–4 week holds targeting 10–30% gains, end-of-day-only "
        "scanning. **Rules implemented**: Close > 50 EMA (uptrend) + price within 5% below its 20-day "
        "high but not yet through it (the \"anticipation zone\") + that day's volume ≥ 1.5× the 20-day "
        "average (the \"big money\" proxy) → entry at that day's close. Stop = ATR-based (same "
        "house-style convention as this app's other backtests); target = the % you set below; forced "
        "time-exit if neither hits within ~4 weeks (20 trading days)."
    )

    bm_universes = st.multiselect(
        "Universe", ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"],
        default=["Nifty 100 (Large Cap)", "Nifty Midcap 150"], key="bm_universes",
    )
    bm_col1, bm_col2, bm_col3 = st.columns(3)
    bm_start_date = bm_col1.date_input("Start date", value=pd.Timestamp("2020-01-01"), key="bm_start_date")
    bm_target_pct = bm_col2.slider(
        "Target % (publicly stated range is 10–30%)", min_value=10.0, max_value=30.0, value=20.0, step=1.0,
        key="bm_target_pct",
    )
    bm_risk_per_trade = bm_col3.number_input(
        "Risk per trade (% of capital)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        key="bm_risk_per_trade",
    )
    bm_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="bm_force_refresh")

    if st.button("▶️ Run Big Money Swing Backtest", type="primary"):
        if not bm_universes:
            st.warning("Pick at least one universe first.")
        else:
            bm_symbols_df = get_all_symbols(bm_universes, force_refresh=bm_force_refresh)[["Symbol", "Segment"]]
            bm_symbols = list(bm_symbols_df.itertuples(index=False, name=None))

            bm_progress = st.progress(0.0)
            bm_status = st.empty()
            bm_start = time.time()

            def _bm_cb(i, total, symbol):
                bm_progress.progress((i + 1) / total)
                bm_status.text(f"[{i+1}/{total}] {symbol}")

            bm_trades = bigmoney_swing_backtest.run_backtest(
                bm_symbols, target_pct=bm_target_pct, start_date=str(bm_start_date), progress_cb=_bm_cb,
                force_refresh=bm_force_refresh,
            )
            bm_elapsed = time.time() - bm_start
            bm_status.text(f"Done in {bm_elapsed:.0f}s — {len(bm_trades)} trade(s) generated.")

            st.session_state["bm_trades"] = bm_trades
            st.session_state["bm_ran_at"] = pd.Timestamp.now()

    bm_trades = st.session_state.get("bm_trades")

    if bm_trades is None:
        st.info("Click **Run Big Money Swing Backtest** above. First run is slow (full history fetch "
                 "per stock); later runs reuse the local cache and are fast.")
    elif bm_trades.empty:
        st.warning("No qualifying trades for this universe/date range — this is a multi-condition "
                    "filter (uptrend + near-high + volume pickup), so few or zero hits on a given "
                    "universe/period is expected, not necessarily a bug.")
    else:
        st.caption(f"Last run: {st.session_state['bm_ran_at']:%Y-%m-%d %H:%M}")

        bm_summary = bigmoney_swing_backtest.summarize(bm_trades)
        bm_port = bigmoney_swing_backtest.simulate_portfolio(bm_trades, risk_per_trade_pct=bm_risk_per_trade)

        bm_c1, bm_c2, bm_c3, bm_c4, bm_c5 = st.columns(5)
        bm_c1.metric("Closed trades", bm_summary["closed_trades"])
        bm_c2.metric("Win rate", f"{bm_summary['win_rate_pct']:.1f}%"
                      if pd.notna(bm_summary["win_rate_pct"]) else "—")
        bm_c3.metric("Avg return / trade", f"{bm_summary['avg_return_pct']:.2f}%"
                      if pd.notna(bm_summary["avg_return_pct"]) else "—")
        bm_c4.metric("Portfolio return", f"{bm_port['total_return_pct']:.2f}%",
                      help=f"Started at 100, risking {bm_risk_per_trade}% of capital per trade.")
        bm_c5.metric("Portfolio max drawdown", f"{bm_port['max_drawdown_pct']:.2f}%"
                      if pd.notna(bm_port["max_drawdown_pct"]) else "—")
        st.caption(
            f"Avg winner: {bm_summary['avg_win_pct']:.2f}% · Avg loser: {bm_summary['avg_loss_pct']:.2f}% "
            f"· Avg holding days: {bm_trades['HoldingDays'].mean():.1f} "
            f"· Trades taken: {bm_port['trades_taken']} · skipped (no free slot): {bm_port['trades_skipped']}"
        )

        if not bm_port["equity_curve"].empty:
            bm_fig = go.Figure()
            bm_fig.add_trace(go.Scatter(
                x=bm_port["equity_curve"].index, y=bm_port["equity_curve"].values,
                mode="lines", name="Capital",
            ))
            bm_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                                   yaxis_title="Capital (started at 100)", xaxis_title="Exit date")
            st.plotly_chart(bm_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Trades")
        st.dataframe(bm_trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all trades (CSV)", bm_trades.to_csv(index=False),
            file_name=f"bigmoney_swing_backtest_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )

with tab_calls:
    st.title("📞 Call Performance")
    st.caption(
        "Every Above 200 EMA / Golden Cross call TrendLine has sent to Telegram, logged the moment it "
        "first qualified and marked to market against the latest daily close since. Read-only — this "
        "tab never sends alerts, it just tracks what already went out."
    )

    calls_df = call_log.load_calls()
    if calls_df.empty:
        st.info(
            "No calls logged yet. They're recorded automatically the next time a stock newly "
            "qualifies for Above 200 EMA or Golden Cross in the 15-minute background alert cycle "
            "(local launchd or GitHub Actions) — see README for how that job runs."
        )
    else:
        call_filter_col1, call_filter_col2 = st.columns([2, 1])
        with call_filter_col1:
            call_signal_filter = st.multiselect(
                "Signal type", ["Above 200 EMA", "Golden Cross"],
                default=["Above 200 EMA", "Golden Cross"], key="call_signal_filter",
            )
        with call_filter_col2:
            call_date_filter = st.date_input(
                "Call date", value=None, key="call_date_filter", format="DD-MM-YYYY",
                help="Pick a date to see only calls generated that day. Leave blank for all dates.",
            )

        calls_filtered = calls_df[calls_df["signal_type"].isin(call_signal_filter)] if call_signal_filter \
            else calls_df.iloc[0:0]
        if call_date_filter:
            calls_filtered = calls_filtered[calls_filtered["date"].dt.date == call_date_filter]

        if calls_filtered.empty:
            st.warning("No calls match these filters.")
        else:
            with st.spinner("Marking calls to market..."):
                calls_perf = call_log.with_performance(calls_filtered)
                calls_perf = call_log.with_target_sl_exit(calls_perf)

            total_calls = len(calls_perf)
            winners = int((calls_perf["change_pct"] > 0).sum())
            win_rate = winners / total_calls * 100.0
            avg_change = calls_perf["change_pct"].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Calls", total_calls)
            m2.metric("Win rate", f"{win_rate:.1f}%", help="Share of calls currently above their call price")
            m3.metric("Avg return", f"{avg_change:+.2f}%")
            m4.metric("Best call", f"{calls_perf['change_pct'].max():+.2f}%")

            calls_show = calls_perf.rename(columns={
                "date": "Date", "time": "Time", "symbol": "Symbol", "segment": "Segment",
                "signal_type": "Signal", "timeframe": "Timeframe", "entry_price": "Entry",
                "target_price": "Target", "sl_price": "SL", "sold_at_price": "Sold At",
                "current_close": "Close Now", "change_pct": "Change %", "days_since": "Days Since",
            })
            calls_show["Sold At"] = calls_show["Sold At"].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else "Open")
            calls_show = calls_show[["Date", "Time", "Symbol", "Signal", "Timeframe", "Entry", "Target",
                                      "SL", "Sold At", "Close Now", "Change %", "Days Since"]].sort_values(
                ["Date", "Time"], ascending=False)
            calls_show["Date"] = calls_show["Date"].dt.strftime("%d-%b-%Y")

            st.caption(
                f"Target/SL applied uniformly to every call ({call_log.TARGET_PCT*100:.0f}% target / "
                f"{call_log.STOP_PCT*100:.0f}% stop from the call's own price) — Golden Cross's own rule, "
                "reused here for Above 200 EMA too since that signal has no target/SL of its own. "
                "\"Sold At\" walks forward on daily closes from the call date; \"Open\" means neither "
                "has been hit yet in the cached price history."
            )
            st.dataframe(
                calls_show, use_container_width=True, hide_index=True,
                column_config={
                    "Entry": st.column_config.NumberColumn(format="%.2f"),
                    "Target": st.column_config.NumberColumn(format="%.2f"),
                    "SL": st.column_config.NumberColumn(format="%.2f"),
                    "Close Now": st.column_config.NumberColumn(format="%.2f"),
                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

    st.divider()
    st.header("📊 Strategy Signal Tracker (Entry / SL / Target / Exit, Daily)")
    st.caption(
        f"Every backtest strategy in this app, run forward live from today with its own independent "
        f"₹{paper_trading.STARTING_CAPITAL:,.0f} paper account — full Entry/Stop-Loss/Target/Exit for "
        f"every signal, not just a mark-to-market price change like the calls above."
    )
    st.info(
        f"**Position sizing** (applied uniformly to all {len(paper_trading.STRATEGIES)}): risk "
        f"{paper_trading.RISK_PER_TRADE_PCT}% of that strategy's own current capital per trade, sized "
        "off that trade's own stop-loss distance — same discipline validated in every backtest on this "
        "app. Each strategy keeps its **own** entry/exit/stop rules exactly as already built and tested "
        "(nothing was redesigned here) — this only adds the capital ledger and position sizing on top. "
        "**Automation**: all strategies are checked once per trading day, after the close — the seven "
        "that hold positions across days get a fresh entry/exit check; EMA Scalping's trades resolve "
        "within the same session, so its check logs that day's already-resolved trades directly. Runs "
        "automatically via the same background job that already sends your Telegram alerts (local "
        "launchd / GitHub Actions) — nothing here requires the Streamlit app itself to be open."
    )

    stc1, stc2 = st.columns([3, 1])
    with stc2:
        if st.button("▶️ Run all checks now"):
            with st.spinner("Running every strategy's daily check..."):
                for s in paper_trading.DAILY_BAR_STRATEGIES:
                    try:
                        paper_trading.run_daily_cycle(s)
                    except Exception as e:
                        st.warning(f"{paper_trading.STRATEGY_LABELS[s]}: {e}")
                for s in paper_trading.SAME_DAY_STRATEGIES:
                    try:
                        paper_trading.run_same_day_cycle(s)
                    except Exception as e:
                        st.warning(f"{paper_trading.STRATEGY_LABELS[s]}: {e}")
            st.success("Done.")
            st.rerun()

    st.subheader("Comparative view")
    comparison = paper_trading.comparison_table()
    st.dataframe(
        comparison, use_container_width=True, hide_index=True,
        column_config={
            "Capital": st.column_config.NumberColumn(format="₹%.2f"),
            "Total Return %": st.column_config.NumberColumn(format="%.2f%%"),
            "Win Rate %": st.column_config.NumberColumn(format="%.1f%%"),
            "Current Drawdown %": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    all_strategy_trades = {s: paper_trading.load_trades(s) for s in paper_trading.STRATEGIES}
    any_trades = any(not t.empty for t in all_strategy_trades.values())
    if any_trades:
        st.subheader("Capital over time (all strategies)")
        st_fig = go.Figure()
        for s in paper_trading.STRATEGIES:
            t = all_strategy_trades[s]
            if t.empty:
                continue
            t = t.sort_values("exit_date")
            st_fig.add_trace(go.Scatter(
                x=pd.to_datetime(t["exit_date"]), y=t["capital_after"],
                mode="lines+markers", name=paper_trading.STRATEGY_LABELS[s],
            ))
        st_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), dragmode="pan",
                              yaxis_title="Capital (₹)", xaxis_title="Exit date")
        st.plotly_chart(st_fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.info("No closed trades yet on any strategy — the comparison chart will fill in as trades close. "
                "This is expected on day one.")

    st.subheader("Per-strategy detail — Entry / SL / Target / Exit")
    strat_selected = st.selectbox("Strategy", paper_trading.STRATEGIES,
                                   format_func=lambda s: paper_trading.STRATEGY_LABELS[s], key="strat_selected")
    strat_state = paper_trading.load_state(strat_selected)
    strat_trades = all_strategy_trades[strat_selected]

    if strat_state["open_positions"]:
        st.write("**Open positions**")
        st.dataframe(pd.DataFrame(strat_state["open_positions"]), use_container_width=True, hide_index=True)
    else:
        st.caption("No open positions right now.")

    st.write("**Closed trades (Entry / SL / Target / Exit)**")
    if strat_trades.empty:
        st.caption("No closed trades yet for this strategy.")
    else:
        st.dataframe(strat_trades.sort_values("exit_date", ascending=False), use_container_width=True,
                     hide_index=True)
        st.download_button(
            "Download trades (CSV)", strat_trades.to_csv(index=False),
            file_name=f"strategy_trades_{strat_selected}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )
