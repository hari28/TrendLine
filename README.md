# TrendLine

Screens Nifty 100 (Large Cap) / Midcap 150 / Smallcap 250 on a timeframe you
choose (1/5/15 Minute, 1 Hour, 4 Hour, 1 Day, 1 Week, or 1 Month), in one of
four modes:

- **Above 200 MA** — latest close sits just above the 200-period EMA/SMA
- **Golden Cross / Death Cross** — the 50-period EMA/SMA has recently crossed
  the 200-period EMA/SMA, up (golden) or down (death)
- **Unusual Volume** — a volume spike vs. the stock's own recent average,
  labeled Buying/Selling from the price move on that bar
- **Chart Patterns** — heuristic, best-effort detection of Triangle, Channel,
  and Flag & Pole breakouts (see the honesty note below)

Plus, always available:

- A **Market Pulse** panel — today's aggregate FII/DII net buy/sell for the
  whole market (from NSE's own feed)
- A **Bulk & Block Deals** panel — NSE's disclosed large-trade feed, filtered
  to whichever universe you have selected
- A **🔍 Search a stock** box in the sidebar — any symbol across all 3
  universes, opens automatically in the **Stock Chart** tab with a candlestick
  chart and 10/20/50/200-period MA lines (using the common EMA/SMA + Timeframe
  settings from the sidebar)
- A **Volume Profile** tab — candlestick chart with a Fixed Range Volume
  Profile for any one symbol, with its own independent universe/symbol/bins
  controls (separate from the common sidebar settings)

**Important honesty note on FII/DII:** NSE does not publish which specific
stock an FII or DII bought or sold — that breakdown only exists as a
market-wide daily total (shown in Market Pulse). The Bulk & Block Deals feed
is the closest free, per-stock proxy: it discloses individual large trades
with the counterparty's name, but isn't officially tagged FII/DII/other — you
have to read the client name yourself. The Unusual Volume scan mode is a
third, independent proxy built from price/volume data alone (no institutional
attribution at all, just "more volume than usual, and which way price moved").

**Important honesty note on Chart Patterns:** Triangle, Channel, and Flag &
Pole aren't precisely defined mathematically — even professional chartists
draw them differently by eye, and there's no universal algorithm. This mode
fits trendlines through swing highs/lows and checks for convergence
(Triangle), parallelism (Channel), or a sharp move + tight consolidation
(Flag & Pole), then flags a breakout when price closes outside those lines.
Expect false positives; treat hits as candidates to confirm visually on the
chart, not certainties. See `patterns.py` for the exact rules.

This is a plain local project folder — it lives permanently on this Mac at
`~/Documents/Claude_PRO/ma_screener/` regardless of Claude. It only *runs*
while you start it; nothing runs in the background on its own.

## Setup (one-time only)

Already done for you in this folder (a `venv/` virtual environment with all
dependencies installed, including `plotly` for the volume profile chart). If
you ever move this folder or set it up on another machine, recreate it with:

**macOS / Linux:**

```bash
cd ma_screener
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell or cmd):**

```bat
cd ma_screener
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Daily use

**macOS — easiest:** double-click `Run_Screener.command` in Finder. A
Terminal window opens, and your default browser opens automatically to the
app.

**macOS — from Terminal:**

```bash
cd ~/Documents/Claude_PRO/ma_screener
source venv/bin/activate
streamlit run app.py
```

**Windows — from PowerShell or cmd:**

```bat
cd path\to\ma_screener
venv\Scripts\activate
streamlit run app.py
```

Opens at http://localhost:8501. The sidebar (Scan settings + Search) is common
to every tab. Three tabs:

### Sidebar — common to all tabs
- Pick universe from the dropdown (Large Cap, Midcap, or Smallcap — one at a time)
- Pick **Scan type**: "Above 200 MA", "Golden Cross / Death Cross (50 vs 200)",
  "Unusual Volume (Buying/Selling Spike)", or "Chart Patterns (Triangle /
  Channel / Flag & Pole)"
- Pick **EMA or SMA**
- Pick a **Timeframe**: 1 Minute, 5 Minute, 15 Minute, 1 Hour, 4 Hour, 1 Day,
  1 Week, or 1 Month
- Mode-specific controls: the "just above" band, the cross lookback window,
  the volume-spike threshold, or the pattern types + trendline lookback +
  minimum pole move %, depending on Scan type
- Set a minimum volume filter, then click **Run scan** for a full-universe scan
- **🔍 Search a stock** — pick any symbol from all 3 universes; it appears
  automatically in the **Stock Chart** tab using the MA type & Timeframe above,
  independent of which Universe is selected for bulk scanning

### Screener tab
Shows the full-universe scan results for whichever Scan type is selected. The
page title and results layout adapt to the mode. An expandable
**"Market Pulse & Bulk/Block Deals"** panel always shows today's FII/DII
aggregate and any disclosed large trades in your selected universe — no scan
needed to see that. Within the results, a **"View chart for a stock in these
results"** expander lets you pick any scanned symbol and see its candlestick
chart with 10/20/50/200 MA lines.

First scan on a given timeframe is slow (fetches full history per stock —
roughly 5–8 min for ~500 stocks on Daily/Weekly/Monthly, and can be slower on
1H/4H since intraday data is a separate, more heavily-throttled fetch).
Re-running the same timeframe later reuses the local cache (`cache/`) and is
fast — daily data is cached 12h, intraday (1H/4H) is cached 1h. Index
constituent lists (`data/`) refresh weekly; FII/DII and deals data refresh
every 1–6 hours.

### 🔍 Stock Chart tab
Shows whatever symbol you picked in the sidebar's Search box: its scan-mode
details table plus a candlestick chart with 10/20/50/200-period MA lines
(SMA or EMA, matching the sidebar). Updates live as you change the sidebar's
MA type or Timeframe — no separate "load" step.

### Volume Profile tab
Independent controls (its own universe, symbol, timeframe, bars, and price
bins) since this is a different kind of one-off lookup — MA type still
follows the sidebar. Pick your settings and click **Load chart** to render a
candlestick chart (with MA lines overlaid) plus a horizontal volume-by-price
histogram; the orange line marks the Point of Control (the price level with
the most traded volume in that range).

### Moving average toggles
Everywhere MA lines appear (Stock Chart tab, the Screener tab's "View chart"
picker, and the Volume Profile tab), a row of checkboxes — one per period
(10/20/50/200) — lets you switch any of them on or off independently. They
update the chart immediately, no re-run or "load" click needed.

### Chart zoom
All charts support scroll-wheel / trackpad zoom (previously off by default —
now explicitly enabled), plus the standard Plotly gestures: click-and-drag to
box-zoom into a range, and double-click to reset back to the full view.

**To stop the app:** close the Terminal (or PowerShell/cmd) window it opened,
or press Control+C (Ctrl+C on Windows) inside it. Closing the browser tab
alone does not stop the server.

## Notes

- Data source: Yahoo Finance for NSE (`SYMBOL.NS`) for prices, and NSE's own
  public feeds for FII/DII and bulk/block deals — none of this is your
  broker's live feed, treat it as recent EOD/intraday data, not live ticks.
- NSE's FII/DII and deals endpoints occasionally rate-limit scripted access;
  the app shows a warning and keeps working with cached data if a fetch fails.
- A 200-bar 200-Month MA needs ~17 years of listed history; a 200-bar 1H/4H
  MA needs history the free intraday feed only carries ~2 years back. Symbols
  without enough bars on the chosen timeframe are shown separately under
  "Insufficient history", not silently dropped.
- Yahoo caps how far back each intraday interval goes: 1-Minute is limited to
  the last 8 days, 5-Minute and 15-Minute to the last 60 days, 1-Hour to
  ~729 days. That's still hundreds to thousands of bars for a 200-period MA
  since intraday bars accumulate fast, but don't expect months of 1-minute
  history — it isn't available from this free feed at any price.
- Bulk-scanning the full universe on 1/5/15-Minute works, but is slower and
  more heavily throttled by Yahoo than Daily/Weekly/Monthly — expect it to
  take longer, especially the first run before the cache warms up.
- This screens for technical conditions only (price, volume, and disclosed
  deals vs. history). It does not size positions, place stops, or manage
  risk — apply your own entry/exit discipline before acting on anything it
  surfaces.
