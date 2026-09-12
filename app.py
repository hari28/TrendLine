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
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from constituents import get_all_symbols, get_constituents, INDEX_UNIVERSE_LABEL
from screener import (scan_universe, apply_band, apply_band_below, scan_universe_cross, scan_universe_volume,
                       load_frame, scan_symbol, scan_symbol_cross, scan_symbol_volume,
                       scan_universe_pattern, scan_symbol_pattern)
from indicators import TIMEFRAMES
from market_data import get_fii_dii_activity, deals_for_symbols
from chart import build_candles_with_volume_profile, build_ma_overlay_chart, build_structure_chart
import watchlist
import screener_alert
import cpr
import structure
import fo_universe
import telegram_bot
import backtest
import cpr_ema_backtest
import swing_backtest
import momentum_backtest
import iron_condor_backtest
import iron_condor_range_backtest
import paper_trading
import call_log

st.set_page_config(page_title="TrendLine", layout="wide")

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
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_fig", config={'scrollZoom': True})


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
        ["Above 200 MA", "Below 200 MA", "Golden Cross / Death Cross (50 vs 200)",
         "Unusual Volume (Buying/Selling Spike)", "Chart Patterns (Triangle / Channel / Flag & Pole)"],
        index=0,
    )

    ma_type = st.radio("Moving average type", ["EMA", "SMA"], horizontal=True)

    tf_choice_label = st.radio("Timeframe", TF_LABELS, index=DEFAULT_TF_INDEX, horizontal=True)  # default 1 Day
    timeframe = TF_KEYS[TF_LABELS.index(tf_choice_label)]

    lookback = avg_period = spike_multiple = band_low = band_high = None
    pattern_types = pattern_lookback = pole_min_move_pct = None

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

    st.divider()
    st.subheader("⭐ Add to watchlist")
    if searched_symbol:
        if st.button(f"➕ Add {searched_symbol} (current scan settings)", use_container_width=True):
            mode_code = watchlist.SCAN_MODE_CODES[scan_mode]
            wl_params = _current_scan_params(scan_mode, band_low, band_high, lookback, avg_period,
                                              spike_multiple, pattern_types, pattern_lookback, pole_min_move_pct)
            seg = all_universe_symbols.loc[all_universe_symbols["Symbol"] == searched_symbol, "Segment"].iloc[0]
            watchlist.add_entry(searched_symbol, seg, universe_choice, timeframe, ma_type, mode_code, wl_params)
            st.success(f"Added {searched_symbol} — will watch for: {scan_mode}.")
    else:
        st.caption("Pick a symbol above first.")

(tab_screener, tab_stock_chart, tab_volume_profile, tab_cpr, tab_structure, tab_iron_condor, tab_backtest,
 tab_calls, tab_paper, tab_watchlist) = st.tabs(
    ["Screener", "🔍 Stock Chart", "Volume Profile", "🎯 Narrow CPR", "📐 Market Structure", "🦅 Iron Condor",
     "🧪 Backtest", "📞 Call Performance", "📝 Paper Trading", "⭐ Watchlist"]
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

        if full is None or full.empty:
            st.error(f"Couldn't fetch usable price history for {searched_symbol} on {tf_choice_label}.")
        else:
            fig = build_ma_overlay_chart(full, searched_symbol, tf_choice_label, ma_type,
                                          periods=stock_chart_periods, display_bars=int(bars))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

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
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
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
                                     config={'scrollZoom': True})

    render_telegram_send_button(
        "structure",
        header_lines=[
            f"{datetime.now():%d-%b-%Y %H:%M} IST",
            "📐 Market Structure snapshot — Uptrends",
            f"Universe: {structure_universe} | Timeframe: {structure_tf_label}",
        ],
        records=structure_records,
    )

with tab_iron_condor:
    st.title("🦅 Intraday Iron Condor")
    st.caption(
        "An intraday, defined-risk options-selling strategy on Nifty 50 / Bank Nifty weekly options — "
        "sell a high-probability OTM call and put, buy further-OTM options as insurance, and be flat "
        "before the close every single day."
    )

    st.warning(
        "**TrendLine has no real options-chain data source** (no strikes, no live premiums, no IV, no "
        "Greeks — Yahoo doesn't serve NSE index options). Everything below is a **Black-Scholes model** "
        "fed by the actual index price history, using trailing realized volatility as a stand-in for "
        "implied volatility. Treat this as a way to gut-check the strategy's shape, not as a source of "
        "tradeable prices — always cross-check strikes and premiums against your broker's live option "
        "chain before placing a real order."
    )

    st.markdown("### How it works")

    st.markdown("""
**1. Instrument & timing**
- Nifty 50 or Bank Nifty **weekly options**, traded on the day of expiry itself (0DTE) — this is what
  makes it genuinely *intraday*: same-day expiry means the position is naturally flat by the close,
  with no overnight gap risk.

**2. Entry — around 09:45 AM**
- Wait out the first ~30 minutes of opening volatility before selecting strikes.
- Sell a **Call** and a **Put**, each roughly **15–20 delta** (≈80–85% statistical probability of
  expiring worthless) — these are the "Short Call" and "Short Put".
- Buy one further-OTM Call and one further-OTM Put as protection, spread-width away from each short
  strike — these cap your maximum possible loss on the trade *before you even enter it*. This 4-leg
  structure is the **Iron Condor**.
- Only take the trade if the credit collected is worth it relative to the spread width (roughly a third
  of the width or more) — skip the day if premiums are too thin.

**3. Exit — whichever comes first**
- **Stop-loss**: if the cost to close the position (buy back the shorts, sell the longs) reaches about
  **1.5–2× the credit you collected**, exit immediately. Don't wait for the full defined-risk max loss
  to play out — this keeps your average loss well inside the theoretical worst case.
- **Time-based exit**: square off everything by **3:15 PM**, every day, no exceptions. This is what
  guarantees zero overnight risk.

**4. Position sizing — the actual lever that controls drawdown**
- Risk a small, fixed % of capital per trade (e.g. 1%), sized off that trade's own defined max loss.
- This is what turns "occasional losing trade" into "5–10% portfolio drawdown" instead of a blown
  account — even several losses in a row stay bounded by construction, since no single trade can ever
  cost more than its position-sized share of capital.

**Why this combination targets >50% win rate with controlled drawdown**: selling 15–20 delta strikes
gives a naturally high win rate (you need the index to *break* a boundary to lose, not just move) — the
defined-risk hedge and the stop-loss are what keep the *size* of a loss bounded when it does happen,
and position sizing is what keeps a string of bounded losses from ever becoming a large drawdown.
    """)

    st.divider()
    st.markdown("### Variant: Non-Expiry Day (Range) Iron Condor")
    st.caption(
        "0DTE only trades one day a week (expiry day). This variant fills the other days by harvesting "
        "a *different* edge — since there's barely any theta decay with days still left, it instead "
        "targets **intraday volatility crush** (implied vol is typically elevated at the open and "
        "compresses through the morning) and **range containment**, sized off the recent average daily "
        "range instead of delta."
    )
    st.markdown("""
**Entry — ~09:45 AM, on any day that is NOT the chosen expiry weekday**
- Short strikes set at the day's open ± the recent 10-day average daily range (not delta-based — delta
  on a multi-day option barely moves within one session, so it's not a useful strike-selection tool here)
- Same Iron Condor hedge structure (buy further-OTM Call/Put to cap max loss)

**Exit — whichever comes first**
- **Time-based**: flat by **~1:15 PM** — earlier than 0DTE's 3:15 PM, since the vol-crush edge is
  front-loaded to the first half of the session; holding longer adds days-to-expiry risk for no extra edge
- **Stop-loss**: ≈1.4× the credit collected

**A real finding from backtesting this, not a design choice**: 1-day-to-expiry entries performed far
worse than 2-4 day entries (credit shrinks a lot at 1 day left while full-session range risk doesn't) —
so entries with fewer than 2 days to expiry are skipped by default.
    """)
    st.warning(
        "**This strategy is regime-dependent** — like any non-directional premium-selling approach, it "
        "profits when the index stays range-bound and loses when it trends hard in one direction. "
        "Backtesting found exactly this split: Nifty 50 (relatively range-bound in the tested window) "
        "showed a 94% win rate, while Bank Nifty (which trended down hard in the same window) showed "
        "only 34%, with the short put side getting run over repeatedly. This isn't a bug — it's the "
        "genuine risk of the strategy, and it's exactly why position sizing (not the win rate alone) is "
        "what actually protects your capital."
    )

    st.divider()
    st.markdown("### Today's model strikes (Black-Scholes, not live market prices)")

    ic_col1, ic_col2 = st.columns(2)
    ic_symbol = ic_col1.selectbox("Index", list(iron_condor_backtest.INSTRUMENT_CONFIG.keys()), key="ic_symbol")
    ic_weekday = ic_col2.selectbox("Weekly expiry weekday for this index",
                                    list(iron_condor_backtest.EXPIRY_WEEKDAY_MAP.keys()), index=3, key="ic_weekday")

    if st.button("Compute today's model strikes", key="ic_compute"):
        ic_daily = load_frame(ic_symbol, "1D", force_refresh=False)
        ic_intraday = load_frame(ic_symbol, "15MIN", force_refresh=False)
        if ic_daily is None or ic_intraday is None or ic_intraday.empty:
            st.error(f"Couldn't load price history for {ic_symbol}.")
        else:
            ic_sigma = iron_condor_backtest._realized_vol_series(ic_daily["Close"]).iloc[-1]
            ic_spot = float(ic_intraday["Close"].iloc[-1])
            ic_t = iron_condor_backtest._time_fraction_remaining(
                iron_condor_backtest.ENTRY_TIME, iron_condor_backtest.EXIT_TIME)
            if pd.isna(ic_sigma) or ic_sigma <= 0:
                st.warning("Not enough history to estimate realized volatility yet.")
            else:
                ic_cfg = iron_condor_backtest.INSTRUMENT_CONFIG[ic_symbol]
                ic_sc = iron_condor_backtest._find_strike_for_delta(
                    ic_spot, ic_t, ic_sigma, iron_condor_backtest.SHORT_DELTA_TARGET, "call", ic_cfg["strike_step"])
                ic_sp = iron_condor_backtest._find_strike_for_delta(
                    ic_spot, ic_t, ic_sigma, iron_condor_backtest.SHORT_DELTA_TARGET, "put", ic_cfg["strike_step"])
                if ic_sc is None or ic_sp is None:
                    st.warning("Couldn't locate strikes at the target delta for the latest spot/volatility.")
                else:
                    ic_lc = ic_sc + ic_cfg["spread_width"]
                    ic_lp = ic_sp - ic_cfg["spread_width"]
                    ic_credit = iron_condor_backtest._condor_cost_to_close(
                        ic_spot, ic_t, ic_sigma, ic_sc, ic_lc, ic_sp, ic_lp)
                    ic_max_loss = ic_cfg["spread_width"] - ic_credit

                    st.caption(f"Latest cached spot: {ic_spot:,.2f} · trailing realized vol: {ic_sigma*100:.1f}% "
                               f"(annualized, {iron_condor_backtest.VOL_LOOKBACK}-day) — NOT live data, based on "
                               f"whatever is currently cached.")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sell Call", ic_sc)
                    m2.metric("Buy Call", ic_lc)
                    m3.metric("Sell Put", ic_sp)
                    m4.metric("Buy Put", ic_lp)
                    m5, m6 = st.columns(2)
                    m5.metric("Model credit (points)", f"{ic_credit:.2f}")
                    m6.metric("Model max loss (points)", f"{ic_max_loss:.2f}")

with tab_backtest:
    st.title("🧪 Backtest")
    bt_strategy = st.radio(
        "Strategy", ["Golden Cross", "CPR + EMA", "Swing Strategy", "Momentum Screener", "Iron Condor"],
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
            bt_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                                  yaxis_title="Equity multiple", xaxis_title="Entry date")
            st.plotly_chart(bt_fig, use_container_width=True)

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
        cbt_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                               yaxis_title="Equity multiple", xaxis_title="Entry date")
        st.plotly_chart(cbt_fig, use_container_width=True)

        st.subheader("Trades")
        st.dataframe(cbt_trades, use_container_width=True, hide_index=True)

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
        "Portfolio sim: risk % of capital per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
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
                swbt_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                        yaxis_title="Capital (started at 100)", xaxis_title="Exit date")
                st.plotly_chart(swbt_fig, use_container_width=True)

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
        "Portfolio sim: risk % of capital per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
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
                mom_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                       yaxis_title="Capital (started at 100)", xaxis_title="Exit date")
                st.plotly_chart(mom_fig, use_container_width=True)

        st.subheader("Trades")
        st.dataframe(mom_trades, use_container_width=True, hide_index=True)
        st.download_button(
            "Download all trades (CSV)", mom_trades.to_csv(index=False),
            file_name=f"momentum_backtest_{mom_mode_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )

elif bt_strategy == "Iron Condor":
  with tab_backtest:
    st.subheader("Intraday Iron Condor Backtest")
    st.warning(
        "**No real options-chain data exists for this app** — this prices a synthetic Iron Condor via "
        "Black-Scholes, fed by the actual index history, using trailing realized volatility as an IV "
        "stand-in. Also: Yahoo only retains ~60 days of 15-min history, so this can only cover the last "
        "~60 trading days, not a multi-year window like the other backtests — and \"expiry weekday\" is "
        "a parameter you set, not looked up from a real contract calendar (NSE has changed this by "
        "circular before)."
    )

    ibt_variant = st.radio("Which variant", ["0DTE (expiry day)", "Non-Expiry Day (Range)"],
                            horizontal=True, key="ibt_variant")

    if ibt_variant == "0DTE (expiry day)":
        with st.expander("Rules being backtested (Entry / Exit / Risk controls)", expanded=True):
            st.markdown("""
**Entry — around 09:45 AM, on the chosen weekly expiry day**
- Sell a **Call** and a **Put**, each ≈15–20 delta (≈80–85% probability of expiring worthless)
- Buy one further-OTM Call and one further-OTM Put, spread-width away — this caps max loss up front
- Only enter if the credit collected is worth it relative to the spread width; skip thin days

**Exit — whichever comes first**
- **Stop-loss**: cost to close the position reaches ≈1.5–2× the credit collected → exit immediately
- **Time-based**: forced flat at 3:15 PM every day — no exceptions, no overnight risk ever

**Risk control — what actually bounds the drawdown**
- Risk a fixed % of capital per trade (set below), sized off that trade's own defined max loss
- One condor per expiry day — no re-entry same day after a stop-out
            """)

        ibt_col1, ibt_col2, ibt_col3 = st.columns(3)
        ibt_symbol = ibt_col1.selectbox("Index", list(iron_condor_backtest.INSTRUMENT_CONFIG.keys()), key="ibt_symbol")
        ibt_weekday = ibt_col2.selectbox("Weekly expiry weekday",
                                          list(iron_condor_backtest.EXPIRY_WEEKDAY_MAP.keys()),
                                          index=3, key="ibt_weekday")
        ibt_risk_per_trade = ibt_col3.number_input(
            "Portfolio sim: risk % of capital per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key="ibt_risk_per_trade",
        )
        ibt_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="ibt_force_refresh")

        if st.button("▶️ Run Iron Condor Backtest", type="primary"):
            ibt_trades = iron_condor_backtest.run_backtest(
                ibt_symbol, expiry_weekday=ibt_weekday, start_date="2020-01-01", force_refresh=ibt_force_refresh,
            )
            st.session_state["ibt_trades"] = ibt_trades
            st.session_state["ibt_ran_at"] = pd.Timestamp.now()

        ibt_trades = st.session_state.get("ibt_trades")

        if ibt_trades is None:
            st.info("Click **Run Iron Condor Backtest** above.")
        elif ibt_trades.empty:
            st.warning(f"No {ibt_weekday} sessions with enough cached 15-min + daily history to simulate. "
                       "Try a different weekday, or run other scans first to warm up the price cache.")
        else:
            st.caption(f"Last run: {st.session_state['ibt_ran_at']:%Y-%m-%d %H:%M}")

            ibt_summary = swing_backtest.summarize(ibt_trades)
            ibt_port = iron_condor_backtest.simulate_portfolio(ibt_trades, risk_per_trade_pct=ibt_risk_per_trade)

            ibt_c1, ibt_c2, ibt_c3, ibt_c4, ibt_c5 = st.columns(5)
            ibt_c1.metric("Trades", ibt_summary["closed_trades"])
            ibt_c2.metric("Win rate", f"{ibt_summary['win_rate_pct']:.1f}%"
                           if pd.notna(ibt_summary["win_rate_pct"]) else "—")
            ibt_c3.metric("Avg return on risk", f"{ibt_summary['avg_return_pct']:.1f}%"
                           if pd.notna(ibt_summary["avg_return_pct"]) else "—")
            ibt_c4.metric("Portfolio return", f"{ibt_port['total_return_pct']:.2f}%",
                           help=f"Started at 100, risking {ibt_risk_per_trade}% of capital per trade, "
                                "compounding sequentially (trades never overlap by construction).")
            ibt_c5.metric("Portfolio max drawdown", f"{ibt_port['max_drawdown_pct']:.2f}%"
                           if pd.notna(ibt_port["max_drawdown_pct"]) else "—")
            st.caption(f"Avg winner: {ibt_summary['avg_win_pct']:.1f}% of max loss · "
                       f"Avg loser: {ibt_summary['avg_loss_pct']:.1f}% of max loss")

            if not ibt_port["equity_curve"].empty:
                ibt_fig = go.Figure()
                ibt_fig.add_trace(go.Scatter(
                    x=ibt_port["equity_curve"].index, y=ibt_port["equity_curve"].values,
                    mode="lines+markers", name="Capital",
                ))
                ibt_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                       yaxis_title="Capital (started at 100)", xaxis_title="Expiry date")
                st.plotly_chart(ibt_fig, use_container_width=True)

            st.subheader("Trades")
            st.dataframe(ibt_trades, use_container_width=True, hide_index=True)
            st.download_button(
                "Download all trades (CSV)", ibt_trades.to_csv(index=False),
                file_name=f"iron_condor_backtest_{ibt_symbol}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
            )

    else:
        st.warning(
            "**This variant is regime-dependent, confirmed by backtesting it, not just theory**: it "
            "profits when the index stays range-bound and loses when it trends hard. In testing, Nifty "
            "50 (range-bound in the tested window) showed a 94% win rate; Bank Nifty (trending down hard "
            "in the same window) showed only 34%, with repeated stop-losses on the put side. Check both "
            "before trusting either number."
        )
        with st.expander("Rules being backtested (Entry / Exit / Risk controls)", expanded=True):
            st.markdown("""
**Entry — ~09:45 AM, on any day that is NOT the chosen expiry weekday, with ≥2 days left to expiry**
- Short strikes = day's open ± the recent 10-day average daily range (range-based, not delta-based)
- Same Iron Condor hedge structure (buy further-OTM Call/Put to cap max loss)
- Entries with only 1 day left to expiry are skipped — backtesting found those specifically underperform
  (credit shrinks a lot while full-session range risk doesn't)

**Exit — whichever comes first**
- **Time-based**: flat by ~1:15 PM — earlier than 0DTE, since the edge here (intraday vol crush) is
  front-loaded to the first half of the session
- **Stop-loss**: cost to close reaches ≈1.4× the credit collected

**The edge being harvested is different from 0DTE**: with days still left, there's barely any theta
decay in one session, so this targets intraday **implied volatility crush** (elevated at the open,
compressing through the morning) instead — modeled from each 15-min time bucket's own historical
volatility relative to the session average. With only ~60 days of history, that per-bucket estimate has
a small sample and should be read as illustrative of the mechanism, not a precise curve.
            """)

        ibtr_col1, ibtr_col2, ibtr_col3 = st.columns(3)
        ibtr_symbol = ibtr_col1.selectbox("Index", list(iron_condor_backtest.INSTRUMENT_CONFIG.keys()),
                                           key="ibtr_symbol")
        ibtr_weekday = ibtr_col2.selectbox("Weekly expiry weekday",
                                            list(iron_condor_backtest.EXPIRY_WEEKDAY_MAP.keys()),
                                            index=3, key="ibtr_weekday")
        ibtr_risk_per_trade = ibtr_col3.number_input(
            "Portfolio sim: risk % of capital per trade", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key="ibtr_risk_per_trade",
        )
        ibtr_force_refresh = st.checkbox("Force-refresh price history (ignore cache)", key="ibtr_force_refresh")

        if st.button("▶️ Run Non-Expiry Range Condor Backtest", type="primary"):
            ibtr_trades = iron_condor_range_backtest.run_backtest(
                ibtr_symbol, expiry_weekday=ibtr_weekday, start_date="2020-01-01",
                force_refresh=ibtr_force_refresh,
            )
            st.session_state["ibtr_trades"] = ibtr_trades
            st.session_state["ibtr_ran_at"] = pd.Timestamp.now()

        ibtr_trades = st.session_state.get("ibtr_trades")

        if ibtr_trades is None:
            st.info("Click **Run Non-Expiry Range Condor Backtest** above.")
        elif ibtr_trades.empty:
            st.warning(f"No non-{ibtr_weekday} sessions with enough cached 15-min + daily history to "
                       "simulate. Try a different weekday, or run other scans first to warm the cache.")
        else:
            st.caption(f"Last run: {st.session_state['ibtr_ran_at']:%Y-%m-%d %H:%M}")

            ibtr_summary = swing_backtest.summarize(ibtr_trades)
            ibtr_port = iron_condor_range_backtest.simulate_portfolio(
                ibtr_trades, risk_per_trade_pct=ibtr_risk_per_trade)

            ibtr_c1, ibtr_c2, ibtr_c3, ibtr_c4, ibtr_c5 = st.columns(5)
            ibtr_c1.metric("Trades", ibtr_summary["closed_trades"])
            ibtr_c2.metric("Win rate", f"{ibtr_summary['win_rate_pct']:.1f}%"
                           if pd.notna(ibtr_summary["win_rate_pct"]) else "—")
            ibtr_c3.metric("Avg return on risk", f"{ibtr_summary['avg_return_pct']:.1f}%"
                           if pd.notna(ibtr_summary["avg_return_pct"]) else "—")
            ibtr_c4.metric("Portfolio return", f"{ibtr_port['total_return_pct']:.2f}%",
                           help=f"Started at 100, risking {ibtr_risk_per_trade}% of capital per trade, "
                                "compounding sequentially (trades never overlap by construction).")
            ibtr_c5.metric("Portfolio max drawdown", f"{ibtr_port['max_drawdown_pct']:.2f}%"
                           if pd.notna(ibtr_port["max_drawdown_pct"]) else "—")
            st.caption(f"Avg winner: {ibtr_summary['avg_win_pct']:.1f}% of max loss · "
                       f"Avg loser: {ibtr_summary['avg_loss_pct']:.1f}% of max loss")

            if not ibtr_port["equity_curve"].empty:
                ibtr_fig = go.Figure()
                ibtr_fig.add_trace(go.Scatter(
                    x=ibtr_port["equity_curve"].index, y=ibtr_port["equity_curve"].values,
                    mode="lines+markers", name="Capital",
                ))
                ibtr_fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                       yaxis_title="Capital (started at 100)", xaxis_title="Entry date")
                st.plotly_chart(ibtr_fig, use_container_width=True)

            st.subheader("Trades")
            st.dataframe(ibtr_trades, use_container_width=True, hide_index=True)
            st.download_button(
                "Download all trades (CSV)", ibtr_trades.to_csv(index=False),
                file_name=f"iron_condor_range_backtest_{ibtr_symbol}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
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
                "signal_type": "Signal", "timeframe": "Timeframe", "close_at_call": "Close @ Call",
                "current_close": "Close Now", "change_pct": "Change %", "days_since": "Days Since",
            })
            calls_show = calls_show[["Date", "Time", "Symbol", "Signal", "Timeframe", "Close @ Call",
                                      "Close Now", "Change %", "Days Since"]].sort_values(
                ["Date", "Time"], ascending=False)
            calls_show["Date"] = calls_show["Date"].dt.strftime("%d-%b-%Y")

            st.dataframe(
                calls_show, use_container_width=True, hide_index=True,
                column_config={
                    "Close @ Call": st.column_config.NumberColumn(format="%.2f"),
                    "Close Now": st.column_config.NumberColumn(format="%.2f"),
                    "Change %": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

with tab_paper:
    st.title("📝 Paper Trading")
    st.caption(
        f"Every strategy built in this app, running forward live from today with its own independent "
        f"₹{paper_trading.STARTING_CAPITAL:,.0f} paper account — so their real, live performance can be "
        f"compared apples-to-apples instead of guessing from backtests alone."
    )
    st.info(
        f"**Position sizing** (applied uniformly to all 8): risk {paper_trading.RISK_PER_TRADE_PCT}% of "
        f"that strategy's own current capital per trade, sized off that trade's own stop-loss distance — "
        "same discipline validated in every backtest on this app. Each strategy keeps its **own** entry/"
        "exit/stop rules exactly as already built and tested (nothing was redesigned here) — this only "
        "adds the capital ledger and position sizing on top. "
        "**Automation**: the 6 daily/weekly equity strategies are checked once per trading day, after "
        "the close; the 2 Iron Condor strategies are checked every 15 minutes during market hours, since "
        "they react to same-day entry/exit windows. Runs automatically via the same background job that "
        "already sends your Telegram alerts (local launchd / GitHub Actions) — nothing here requires the "
        "Streamlit app itself to be open."
    )

    ptc1, ptc2 = st.columns([3, 1])
    with ptc2:
        if st.button("▶️ Run all checks now"):
            with st.spinner("Running every strategy's daily + intraday check..."):
                for s in paper_trading.DAILY_BAR_STRATEGIES:
                    try:
                        paper_trading.run_daily_cycle(s)
                    except Exception as e:
                        st.warning(f"{paper_trading.STRATEGY_LABELS[s]}: {e}")
                for s in paper_trading.INTRADAY_STRATEGIES:
                    try:
                        paper_trading.run_iron_condor_cycle(s)
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

    all_trades = {s: paper_trading.load_trades(s) for s in paper_trading.STRATEGIES}
    any_trades = any(not t.empty for t in all_trades.values())
    if any_trades:
        st.subheader("Capital over time (all strategies)")
        pt_fig = go.Figure()
        for s in paper_trading.STRATEGIES:
            t = all_trades[s]
            if t.empty:
                continue
            t = t.sort_values("exit_date")
            pt_fig.add_trace(go.Scatter(
                x=pd.to_datetime(t["exit_date"]), y=t["capital_after"],
                mode="lines+markers", name=paper_trading.STRATEGY_LABELS[s],
            ))
        pt_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                              yaxis_title="Capital (₹)", xaxis_title="Exit date")
        st.plotly_chart(pt_fig, use_container_width=True)
    else:
        st.info("No closed trades yet on any strategy — the comparison chart will fill in as trades close. "
                "This is expected on day one.")

    st.subheader("Per-strategy detail")
    pt_selected = st.selectbox("Strategy", paper_trading.STRATEGIES,
                                format_func=lambda s: paper_trading.STRATEGY_LABELS[s], key="pt_selected")
    pt_state = paper_trading.load_state(pt_selected)
    pt_trades = all_trades[pt_selected]

    if pt_selected in paper_trading.DAILY_BAR_STRATEGIES:
        if pt_state["open_positions"]:
            st.write("**Open positions**")
            st.dataframe(pd.DataFrame(pt_state["open_positions"]), use_container_width=True, hide_index=True)
        else:
            st.caption("No open positions right now.")
    else:
        pos = pt_state["intraday"].get("position")
        if pos:
            st.write("**Open position (today)**")
            st.json(pos)
        else:
            st.caption("No open position right now.")

    st.write("**Closed trades**")
    if pt_trades.empty:
        st.caption("No closed trades yet for this strategy.")
    else:
        st.dataframe(pt_trades.sort_values("exit_date", ascending=False), use_container_width=True,
                     hide_index=True)
        st.download_button(
            "Download trades (CSV)", pt_trades.to_csv(index=False),
            file_name=f"paper_trading_{pt_selected}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        )

with tab_watchlist:
    st.title("⭐ Watchlist")
    st.caption(
        "Each symbol keeps the scan rule it was added with (from the sidebar's settings at the "
        "time you added it). Status below is a live, read-only check — it never sends alerts on "
        "page load. Real alerts come from the background launchd job (see README), or on-demand "
        "with the button below."
    )

    wl_entries = watchlist.load_watchlist()
    if not wl_entries:
        st.info("No symbols yet. Use the 🔍 Search box in the sidebar to pick a symbol, then "
                 "'➕ Add to watchlist' below it — it saves whichever Scan type and settings are "
                 "currently selected in the sidebar.")
    else:
        wl_rows = []
        for wl_entry in wl_entries:
            wl_result = watchlist.evaluate_entry(wl_entry)
            wl_rows.append({
                "Symbol": wl_entry["symbol"],
                "Scan mode": watchlist.SCAN_MODE_LABELS.get(wl_entry["scan_mode"], wl_entry["scan_mode"]),
                "Timeframe": TIMEFRAMES[wl_entry["timeframe"]]["label"],
                "MA": wl_entry["ma_type"],
                "Status": wl_result["status"] if wl_result["ok"] else f"ERROR: {wl_result.get('reason')}",
                "Added": wl_entry["added_at"],
            })
        st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)

        st.divider()
        wl_remove_choice = st.selectbox("Remove a symbol", [""] + [e["symbol"] for e in wl_entries],
                                         key="wl_remove_choice")
        if wl_remove_choice and st.button(f"🗑️ Remove {wl_remove_choice}"):
            wl_eid = next(e["id"] for e in wl_entries if e["symbol"] == wl_remove_choice)
            watchlist.remove_entry(wl_eid)
            st.rerun()

        st.divider()
        st.subheader("📲 Send to Telegram")
        st.caption(
            "Runs the exact same check-and-alert logic as the background launchd job, synchronously "
            "— useful to test Telegram delivery without waiting for the next scheduled run. Only "
            "sends for a symbol whose condition newly triggered since the last check (not every time "
            "it stays true)."
        )
        if st.button("📲 Check now & send alerts", type="primary"):
            with st.spinner("Checking all watchlist entries..."):
                wl_check_results = watchlist.check_all(send_alerts=True)
            wl_triggered = [r for r in wl_check_results if r.get("triggered")]
            st.success(f"Checked {len(wl_check_results)} entries — {len(wl_triggered)} new alert(s) sent.")
            if wl_triggered:
                st.dataframe(
                    pd.DataFrame([{"Symbol": r["entry"]["symbol"], "Status": r["status"],
                                    "Telegram sent": r.get("alert_sent", False)} for r in wl_triggered]),
                    use_container_width=True, hide_index=True,
                )
