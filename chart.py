"""Candlestick charts:
- build_candles_with_volume_profile: Fixed Range Volume Profile (volume-by-price) + optional MA lines
- build_ma_overlay_chart: candlesticks + moving average lines + volume

Both accept `periods` (which MA lines to draw) so callers can wire up on/off
toggles -- pass a shorter tuple, or () to hide all MA lines, without changing
the underlying data fetch.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators import sma, ema

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
        height=560,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", y=1.06),
    )
    return fig
