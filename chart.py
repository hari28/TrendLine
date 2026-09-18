"""Candlestick charts:
- build_candles_with_volume_profile: Fixed Range Volume Profile (volume-by-price) + optional MA lines
- build_ma_overlay_chart: candlesticks + moving average lines + volume
- build_structure_chart: candlesticks + swing-high/low markers (HH/LH/HL/LL) + Character Change marker

The first two accept `periods` (which MA lines to draw) so callers can wire up
on/off toggles -- pass a shorter tuple, or () to hide all MA lines, without
changing the underlying data fetch.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators import sma, ema
from structure import analyze_structure
from ott import calc_ott, crossover, crossunder

_MA_COLORS = {10: "#FFA500", 20: "#008000", 50: "#FF0000", 200: "#000000"}  # orange, green, red, black


def _ma_lines(full_frame: pd.DataFrame, ma_type: str, periods: tuple, display_bars: int) -> dict:
    if not ma_type or not periods:
        return {}
    ma_fn = ema if ma_type == "EMA" else sma
    return {p: ma_fn(full_frame["Close"], p).tail(display_bars) for p in periods}


def _add_ma_traces(fig, frame: pd.DataFrame, ma_lines: dict, ma_type: str, periods: tuple, row: int, col: int):
    for p in periods:
        if p not in ma_lines:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame.index, y=ma_lines[p], mode="lines", name=f"{ma_type}{p}",
                line=dict(width=1.5, color=_MA_COLORS.get(p)),
            ),
            row=row, col=col,
        )


def _apply_crosshair(fig, timeframe_label: str, rows: int):
    is_intraday = "Hour" in timeframe_label or "Minute" in timeframe_label
    date_fmt = "%b %d, %Y %H:%M" if is_intraday else "%b %d, %Y"
    spike_style = dict(showspikes=True, spikemode="across", spikesnap="cursor",
                        spikecolor="#888", spikethickness=1, spikedash="dot")
    for r in range(1, rows + 1):
        fig.update_xaxes(hoverformat=date_fmt, **spike_style, row=r, col=1)
    fig.update_yaxes(hoverformat=".2f", **spike_style, row=1, col=1)
    if rows > 1:
        fig.update_yaxes(**spike_style, row=2, col=1)
    fig.update_traces(
        hovertemplate="Open: %{open:.2f}<br>High: %{high:.2f}<br>Low: %{low:.2f}<br>Close: %{close:.2f}<extra></extra>",
        selector=dict(type="candlestick"),
    )
    fig.update_layout(hovermode="x")


def build_candles_with_volume_profile(full_frame: pd.DataFrame, symbol: str, timeframe_label: str,
                                       num_bins: int = 24, ma_type: str = None, periods: tuple = (),
                                       display_bars: int = 250) -> go.Figure:
    """full_frame: the COMPLETE OHLCV history (DatetimeIndex) -- MAs are computed on the full
    series so they're accurate from the first displayed bar; the volume profile itself is built
    only from the last `display_bars` bars (the "fixed range")."""
    frame = full_frame.tail(display_bars)
    ma_lines = _ma_lines(full_frame, ma_type, periods, display_bars)

    lo, hi = frame["Low"].min(), frame["High"].max()
    if hi <= lo:
        hi = lo + 1e-6

    edges = np.linspace(lo, hi, num_bins + 1)
    bin_volume = np.zeros(num_bins)

    # Distribute each bar's volume across the price bins its High-Low range touches,
    # weighted by overlap -- a standard approximation for a fixed range volume profile
    # when only OHLCV bars (not tick data) are available.
    for _, row in frame.iterrows():
        bar_lo, bar_hi, vol = row["Low"], row["High"], row["Volume"]
        if vol <= 0 or bar_hi <= bar_lo:
            idx = np.clip(np.searchsorted(edges, row["Close"]) - 1, 0, num_bins - 1)
            bin_volume[idx] += vol
            continue
        first_bin = max(0, np.searchsorted(edges, bar_lo, side="right") - 1)
        last_bin = min(num_bins - 1, np.searchsorted(edges, bar_hi, side="right") - 1)
        touched = list(range(first_bin, last_bin + 1)) or [first_bin]
        overlaps = []
        for b in touched:
            seg_lo, seg_hi = max(bar_lo, edges[b]), min(bar_hi, edges[b + 1])
            overlaps.append(max(0.0, seg_hi - seg_lo))
        total_overlap = sum(overlaps) or 1.0
        for b, ov in zip(touched, overlaps):
            bin_volume[b] += vol * (ov / total_overlap)

    bin_centers = (edges[:-1] + edges[1:]) / 2
    poc_idx = int(np.argmax(bin_volume))  # point of control: highest-volume price level

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2], horizontal_spacing=0.01,
    )

    fig.add_trace(
        go.Candlestick(
            x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
            name=symbol, showlegend=False,
        ),
        row=1, col=1,
    )

    _add_ma_traces(fig, frame, ma_lines, ma_type, periods, row=1, col=1)

    colors = ["#e08214" if i == poc_idx else "rgba(99, 110, 250, 0.55)" for i in range(num_bins)]
    fig.add_trace(
        go.Bar(
            x=bin_volume, y=bin_centers, orientation="h", marker_color=colors,
            showlegend=False, name="Volume Profile",
            hovertemplate="Price ~%{y:.2f}<br>Volume: %{x:,.0f}<extra></extra>",
        ),
        row=1, col=2,
    )

    fig.add_hline(y=bin_centers[poc_idx], line_dash="dot", line_color="#e08214",
                   annotation_text="POC", annotation_position="top left", row=1, col=1)

    _apply_crosshair(fig, timeframe_label, rows=1)

    title = f"{symbol} — {timeframe_label} — Fixed Range Volume Profile ({len(frame)} bars)"
    if ma_type and periods:
        title += f" — {ma_type} " + "/".join(str(p) for p in periods)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        height=520,
        margin=dict(l=40, r=20, t=50, b=30),
        bargap=0.05,
        legend=dict(orientation="h", y=1.06),
    )
    fig.update_xaxes(title_text="Volume", row=1, col=2)
    return fig, float(bin_centers[poc_idx])


def build_ma_overlay_chart(full_frame: pd.DataFrame, symbol: str, timeframe_label: str, ma_type: str,
                            periods: tuple = (10, 20, 50, 200), display_bars: int = 250) -> go.Figure:
    """full_frame: the COMPLETE OHLCV history (DatetimeIndex) -- MAs are computed on the full
    series so they're accurate from the first displayed bar, then only the last `display_bars`
    bars are plotted."""
    ma_lines = _ma_lines(full_frame, ma_type, periods, display_bars)
    frame = full_frame.tail(display_bars)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
            name=symbol, showlegend=False,
        ),
        row=1, col=1,
    )

    _add_ma_traces(fig, frame, ma_lines, ma_type, periods, row=1, col=1)

    fig.add_trace(
        go.Bar(x=frame.index, y=frame["Volume"], marker_color="rgba(99, 110, 250, 0.45)",
               showlegend=False, name="Volume"),
        row=2, col=1,
    )

    _apply_crosshair(fig, timeframe_label, rows=2)

    title = f"{symbol} — {timeframe_label}"
    if ma_type and periods:
        title += f" — {ma_type} " + "/".join(str(p) for p in periods)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        height=560,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def _add_signal_markers(fig, frame: pd.DataFrame, ott: pd.Series, buy_mask: pd.Series,
                         sell_mask: pd.Series, row: int, col: int):
    buy_y = ott.where(buy_mask) * 0.995
    sell_y = ott.where(sell_mask) * 1.005
    fig.add_trace(
        go.Scatter(x=frame.index, y=buy_y, mode="markers+text", text="Buy", textposition="bottom center",
                   textfont=dict(color="white", size=9), marker=dict(symbol="triangle-up", size=11, color="green"),
                   showlegend=False, name="OTT Buy", hoverinfo="skip"),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(x=frame.index, y=sell_y, mode="markers+text", text="Sell", textposition="top center",
                   textfont=dict(color="white", size=9), marker=dict(symbol="triangle-down", size=11, color="red"),
                   showlegend=False, name="OTT Sell", hoverinfo="skip"),
        row=row, col=col,
    )


def add_ott_overlay(fig, full_frame: pd.DataFrame, display_bars: int, length: int = 2, percent: float = 1.4,
                     ma_type: str = "VAR", show_support: bool = True, highlighting: bool = True,
                     highlight_color_changes: bool = False, show_signals_support: bool = True,
                     show_signals_price: bool = False, show_signals_color_change: bool = False,
                     row: int = 1, col: int = 1):
    """Adds Anil Ozeksi's Optimized Trend Tracker (OTT) to an existing candlestick figure
    (e.g. from build_ma_overlay_chart) -- ported 1:1 from the original Pine Script v4 source,
    including its default look: a blue "Support Line" (the underlying MA), a purple OTT trend
    line, a green/red fill between price and the OTT line, and Buy/Sell labels where the
    Support Line crosses the OTT line. full_frame must be the COMPLETE OHLC history so the
    stop levels/crossovers are accurate from the first displayed bar."""
    ott_full = calc_ott(full_frame, length=length, percent=percent, ma_type=ma_type)
    ohlc4_full = (full_frame["Open"] + full_frame["High"] + full_frame["Low"] + full_frame["Close"]) / 4

    buy_support_full = crossover(ott_full["MAvg"], ott_full["OTT"])
    sell_support_full = crossunder(ott_full["MAvg"], ott_full["OTT"])
    buy_price_full = crossover(full_frame["Close"], ott_full["OTT"])
    sell_price_full = crossunder(full_frame["Close"], ott_full["OTT"])
    buy_color_full = crossover(ott_full["OTT"], ott_full["OTT"].shift(1))
    sell_color_full = crossunder(ott_full["OTT"], ott_full["OTT"].shift(1))

    frame = full_frame.tail(display_bars)
    ott = ott_full.tail(display_bars)
    ohlc4 = ohlc4_full.tail(display_bars)

    if show_support:
        fig.add_trace(
            go.Scatter(x=frame.index, y=ott["MAvg"], mode="lines", name="OTT Support",
                       line=dict(width=2, color="#0585E1")),
            row=row, col=col,
        )

    if highlighting:
        up_ott = ott["OTT"].where(ott["MAvg"] > ott["OTT"])
        down_ott = ott["OTT"].where(ott["MAvg"] < ott["OTT"])
        for band, fillcolor in ((up_ott, "rgba(0,128,0,0.15)"), (down_ott, "rgba(255,0,0,0.15)")):
            fig.add_trace(go.Scatter(x=frame.index, y=ohlc4, mode="lines", line=dict(width=0),
                                      showlegend=False, hoverinfo="skip"), row=row, col=col)
            fig.add_trace(go.Scatter(x=frame.index, y=band, mode="lines", line=dict(width=0),
                                      fill="tonexty", fillcolor=fillcolor, showlegend=False,
                                      hoverinfo="skip"), row=row, col=col)

    if highlight_color_changes:
        up_line = ott["OTT"].where(ott["trend_up"])
        down_line = ott["OTT"].where(~ott["trend_up"])
        fig.add_trace(go.Scatter(x=frame.index, y=up_line, mode="lines", name="OTT",
                                  line=dict(width=2, color="#00A800")), row=row, col=col)
        fig.add_trace(go.Scatter(x=frame.index, y=down_line, mode="lines", name="OTT",
                                  line=dict(width=2, color="#FF0000"), showlegend=False), row=row, col=col)
    else:
        fig.add_trace(go.Scatter(x=frame.index, y=ott["OTT"], mode="lines", name="OTT",
                                  line=dict(width=2, color="#B800D9")), row=row, col=col)

    if show_signals_support:
        _add_signal_markers(fig, frame, ott["OTT"], buy_support_full.tail(display_bars),
                             sell_support_full.tail(display_bars), row, col)
    if show_signals_price:
        _add_signal_markers(fig, frame, ott["OTT"], buy_price_full.tail(display_bars),
                             sell_price_full.tail(display_bars), row, col)
    if show_signals_color_change:
        _add_signal_markers(fig, frame, ott["OTT"], buy_color_full.tail(display_bars),
                             sell_color_full.tail(display_bars), row, col)

    return fig


def build_structure_chart(full_frame: pd.DataFrame, symbol: str, timeframe_label: str, order: int,
                           display_bars: int = 250) -> go.Figure:
    """full_frame: the COMPLETE OHLC history -- swings are found over the full series
    (a swing point near the edge of a truncated window can't be confirmed correctly),
    then only swings landing within the last `display_bars` are plotted."""
    result = analyze_structure(full_frame, order)
    frame = full_frame.tail(display_bars)
    cutoff = len(full_frame) - len(frame)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"],
            name=symbol, showlegend=False,
        ),
        row=1, col=1,
    )

    highs_x, highs_y, highs_text = [], [], []
    lows_x, lows_y, lows_text = [], [], []
    for sp in result["swing_points"]:
        if sp["index"] < cutoff:
            continue
        idx = full_frame.index[sp["index"]]
        if sp["kind"] == "high":
            highs_x.append(idx), highs_y.append(sp["price"]), highs_text.append(sp["label"] or "H")
        else:
            lows_x.append(idx), lows_y.append(sp["price"]), lows_text.append(sp["label"] or "L")

    fig.add_trace(
        go.Scatter(x=highs_x, y=highs_y, mode="markers+text", text=highs_text, textposition="top center",
                   marker=dict(symbol="triangle-down", size=9, color="#FF0000"), name="Swing High",
                   showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=lows_x, y=lows_y, mode="markers+text", text=lows_text, textposition="bottom center",
                   marker=dict(symbol="triangle-up", size=9, color="#008000"), name="Swing Low",
                   showlegend=False),
        row=1, col=1,
    )

    if result["choch_bars_ago"] is not None:
        choch_pos = len(full_frame) - 1 - result["choch_bars_ago"]
        if choch_pos >= cutoff:
            fig.add_vline(x=frame.index[choch_pos - cutoff], line_dash="dash", line_color="#9b59b6",
                          annotation_text=f"CHoCH → {result['choch_type']}", annotation_position="top",
                          row=1, col=1)

    fig.add_trace(
        go.Bar(x=frame.index, y=frame["Volume"], marker_color="rgba(99, 110, 250, 0.45)",
               showlegend=False, name="Volume"),
        row=2, col=1,
    )

    _apply_crosshair(fig, timeframe_label, rows=2)

    title = f"{symbol} — {timeframe_label} — Market Structure (Trend: {result['trend'] or 'Establishing'})"
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        height=560,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.06),
    )
    return fig
