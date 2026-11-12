"""Recent news headlines for a stock -- used to flag catalysts on Narrow CPR stocks.

A narrow CPR means compressed volatility; a fresh news item (results, order win,
block deal, regulatory action...) is often what triggers the breakout, so the
CPR tab surfaces headlines for every narrow stock.

Source: Google News RSS (free, no API key, good coverage of Indian business
media), with yfinance's Ticker.news as a fallback when Google returns nothing.
"""
import re
import email.utils
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import requests

NEWS_VERSION = 2  # bump whenever the filtering changes, so cached session news is refetched

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_SUFFIX = re.compile(r"\b(ltd|limited|corporation|corp|co|company|india|inc|plc)\b\.?", re.I)


def _clean_name(company_name: str | None) -> str:
    if not company_name or not isinstance(company_name, str):
        return ""
    name = _SUFFIX.sub("", company_name).replace("&", "and")
    return re.sub(r"\s+", " ", name).strip(" .,-")


def _google_news(query: str, days: int, max_items: int) -> list[dict]:
    url = (f"https://news.google.com/rss/search?q={quote_plus(query + f' when:{days}d')}"
           f"&hl=en-IN&gl=IN&ceid=IN:en")
    resp = requests.get(url, headers=_UA, timeout=10)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        source = (it.findtext("source") or "").strip()
        if source and title.endswith(" - " + source):
            title = title[: -(len(source) + 3)]
        try:
            published = email.utils.parsedate_to_datetime(it.findtext("pubDate"))
        except Exception:
            published = None
        if published and published < cutoff:
            continue
        items.append({"title": title, "source": source, "link": it.findtext("link") or "",
                      "published": published})
    items.sort(key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items[:max_items]


def _yf_news(symbol: str, days: int, max_items: int) -> list[dict]:
    import yfinance as yf
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    t = yf.Ticker(f"{symbol}.NS")
    raw = t.get_news(count=max_items) if hasattr(t, "get_news") else t.news
    for n in (raw or []):
        c = n.get("content", n)  # yfinance >=0.2.50 nests fields under "content"
        title = c.get("title")
        if not title:
            continue
        pub = c.get("pubDate") or c.get("providerPublishTime")
        try:
            published = (datetime.fromtimestamp(pub, timezone.utc) if isinstance(pub, (int, float))
                         else datetime.fromisoformat(str(pub).replace("Z", "+00:00")))
        except Exception:
            published = None
        if published and published < cutoff:
            continue
        link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        source = (c.get("provider") or {}).get("displayName", "") if isinstance(c.get("provider"), dict) else c.get("publisher", "")
        out.append({"title": title, "source": source or "Yahoo Finance", "link": link or "", "published": published})
    return out[:max_items]


_GENERIC = {"the", "and", "of", "bank", "india", "indian", "national", "industries", "finance", "financial",
            "services", "power", "energy", "steel", "tata", "bajaj", "adani", "hindustan", "bharat", "state",
            "solar", "green", "cement", "cements", "auto", "motors", "pharma", "pharmaceuticals", "capital",
            "chemicals", "infra", "metal", "metals", "oil", "gas", "life", "insurance", "housing", "general",
            "united", "new", "global", "first", "max", "union", "central", "punjab", "indraprastha", "gujarat"}


def _is_relevant(title: str, symbol: str, name: str) -> bool:
    """Google News matches a quoted phrase anywhere in the article body, so loosely related
    stories (e.g. a Nava Ltd result for a 'Solar Industries' query) slip in. Keep a headline
    only if it actually names the company: the full cleaned name, its first two words, the
    NSE symbol as a whole word, or a distinctive first word (not generic like Tata/Bank/India)."""
    t = " " + re.sub(r"[^a-z0-9& ]", " ", title.lower().replace("&", " and ")) + " "
    t = re.sub(r"\s+", " ", t)
    words = name.lower().split()
    checks = []
    if words:
        checks.append(" ".join(words))
        if len(words) >= 2:
            checks.append(" ".join(words[:2]))
        if len(words[0]) >= 4 and words[0] not in _GENERIC:
            checks.append(words[0])
    if len(symbol) >= 3:
        checks.append(symbol.lower())
    return any(f" {c} " in t for c in checks)


def fetch_news(symbol: str, company_name: str | None = None, days: int = 3, max_items: int = 5) -> list[dict]:
    """Latest headlines for one NSE symbol within the last `days` days. Never raises.
    Only headlines that actually name the company (see _is_relevant) are kept."""
    name = _clean_name(company_name)
    query = f'"{name}" share OR stock OR "{symbol}"' if name else f'"{symbol}" NSE share'
    try:
        items = [n for n in _google_news(query, days, 50) if _is_relevant(n["title"], symbol, name or symbol)]
        if items:
            return items[:max_items]
    except Exception:
        pass
    try:
        # Yahoo's feed for NSE tickers is often generic market news (verified: SOLARINDS.NS returned a
        # Nava press release), so the same "must name the company" filter applies here too.
        items = [n for n in _yf_news(symbol, days, 50) if _is_relevant(n["title"], symbol, name or symbol)]
        return items[:max_items]
    except Exception:
        return []


def fetch_news_bulk(pairs: list[tuple[str, str | None]], days: int = 3, max_items: int = 5,
                    workers: int = 8, progress_cb=None) -> dict[str, list[dict]]:
    """pairs: [(symbol, company_name), ...] -> {symbol: [news items]} fetched in parallel."""
    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_news, s, n, days, max_items): s for s, n in pairs}
        for i, fut in enumerate(futures):
            sym = futures[fut]
            results[sym] = fut.result()
            if progress_cb:
                progress_cb(i, len(futures), sym)
    return results
