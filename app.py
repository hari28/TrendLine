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
import pandas as pd
import streamlit as st

from constituents import get_all_symbols, get_constituents
from screener import (scan_universe, apply_band, scan_universe_cross, scan_universe_volume, load_frame,
                       scan_symbol, scan_symbol_cross, scan_symbol_volume,
                       scan_universe_pattern, scan_symbol_pattern)
from indicators import TIMEFRAMES
from market_data import get_fii_dii_activity, deals_for_symbols
from chart import build_candles_with_volume_profile, build_ma_overlay_chart
import watchlist

st.set_page_config(page_title="TrendLine", layout="wide")

TF_KEYS = list(TIMEFRAMES.keys())
TF_LABELS = [TIMEFRAMES[k]["label"] for k in TF_KEYS]
DEFAULT_TF_INDEX = TF_LABELS.index("1 Day")
UNIVERSES = ["Nifty 100 (Large Cap)", "Nifty Midcap 150", "Nifty Smallcap 250"]
ALL_UNIVERSE_LABEL = "All (Large + Mid + Small Cap)"
UNIVERSE_OPTIONS = UNIVERSES + [ALL_UNIVERSE_LABEL]


def _fmt_asof(series: pd.Series, timeframe: str) -> pd.Series:
    fmt = "%Y-%m-%d %H:%M" if TIMEFRAMES[timeframe]["kind"] == "intraday" else "%Y-%m-%d"
    return pd.to_datetime(series).dt.strftime(fmt)


def _current_scan_params(scan_mode, band_low, band_high, lookback, avg_period, spike_multiple,
                          pattern_types, pattern_lookback, pole_min_move_pct) -> dict:
    if scan_mode == "Above 200 MA":
        return {"band_low": band_low, "band_high": band_high}
    if scan_mode.startswith("Golden Cross"):
        return {"lookback": lookback}
    if scan_mode.startswith("Unusual Volume"):
        return {"avg_period": avg_period, "spike_multiple": spike_multiple}
    return {"pattern_types": pattern_types, "pattern_lookback": pattern_lookback,
            "pole_min_move_pct": pole_min_move_pct}


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


all_universe_symbols = get_all_symbols(UNIVERSES)
all_symbols_sorted = sorted(all_universe_symbols["Symbol"].unique().tolist())

with st.sidebar:
    st.title("📈 TrendLine")
    st.header("Scan settings")

    universe_choice = st.selectbox("Universe", UNIVERSE_OPTIONS, index=0)
    segments = UNIVERSES if universe_choice == ALL_UNIVERSE_LABEL else [universe_choice]

    scan_mode = st.radio(
        "Scan type",
        ["Above 200 MA", "Golden Cross / Death Cross (50 vs 200)", "Unusual Volume (Buying/Selling Spike)",
         "Chart Patterns (Triangle / Channel / Flag & Pole)"],
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
    st.caption("Any symbol, all 3 universes. Opens in the Stock Chart tab, using the settings above.")
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

tab_screener, tab_stock_chart, tab_volume_profile, tab_watchlist = st.tabs(
    ["Screener", "🔍 Stock Chart", "Volume Profile", "⭐ Watchlist"]
)

with tab_screener:
    if scan_mode == "Above 200 MA":
        st.title(f"Filter Stocks out of {ma_type}200")
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
                "scanned_lookback", "scanned_spike_multiple", "scanned_band_low", "scanned_band_high"):
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

        if scan_mode == "Above 200 MA":
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

        st.session_state["results"] = results
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

            st.subheader(f"In band — just above {ma_col} on {st.session_state['scanned_tf']} ({len(in_band)})")
            if in_band.empty:
                st.write("No stocks currently meet this condition with these settings.")
            else:
                st.dataframe(_fmt(in_band), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download this list (CSV)",
                    _fmt(in_band).to_csv(index=False),
                    file_name=f"in_band_{ma_col}_{scanned_tf_key}_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
                )

            with st.expander(f"Above band — further extended above {ma_col} ({len(above)})"):
                st.dataframe(_fmt(above) if not above.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

            with st.expander(f"Below band — currently under {ma_col} ({len(below)})"):
                st.dataframe(_fmt(below) if not below.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

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
            if scan_mode == "Above 200 MA":
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

        if is_pattern_mode:
            if not row:
                st.info(f"No selected pattern found for {searched_symbol} on {tf_choice_label} with these settings.")
            else:
                out = pd.DataFrame(row).copy()
                out["AsOf"] = _fmt_asof(out["AsOf"], timeframe)
                out["Close"] = out["Close"].round(2)
                st.dataframe(out.drop(columns=["Segment"], errors="ignore"), use_container_width=True,
                             hide_index=True)
        elif row is None:
            st.error(f"Couldn't fetch usable data for {searched_symbol} on {tf_choice_label}.")
        else:
            out = pd.DataFrame([row]).copy()
            out["AsOf"] = _fmt_asof(out["AsOf"], timeframe)
            for c in out.columns:
                if out[c].dtype.kind == "f":
                    out[c] = out[c].round(2)
            st.dataframe(out.drop(columns=["Segment", "BarsAvailable"], errors="ignore"), use_container_width=True,
                         hide_index=True)

        bars = st.number_input("Bars to show", min_value=50, max_value=1000, value=250, step=25,
                                key="stock_chart_bars")
        stock_chart_periods = render_ma_toggles("stock_chart", ma_type)

        if full is None or full.empty:
            st.error(f"Couldn't fetch usable price history for {searched_symbol} on {tf_choice_label}.")
        else:
            fig = build_ma_overlay_chart(full, searched_symbol, tf_choice_label, ma_type,
                                          periods=stock_chart_periods, display_bars=int(bars))
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

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
    if shown:
        shown_symbol, shown_timeframe, shown_tf_label, shown_bars, shown_bins = shown
        with st.spinner(f"Fetching {shown_symbol}..."):
            frame = load_frame(shown_symbol, shown_timeframe, force_refresh=force_refresh_prices)
        if frame is None or frame.empty:
            st.error(f"Couldn't fetch usable data for {shown_symbol} on {shown_tf_label}.")
        else:
            fig = build_candles_with_volume_profile(frame, shown_symbol, shown_tf_label, shown_bins,
                                                      ma_type=ma_type, periods=vp_periods,
                                                      display_bars=shown_bars)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

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
        st.subheader("Send alerts now")
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
