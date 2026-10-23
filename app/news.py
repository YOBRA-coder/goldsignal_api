"""Economic news: calendar (CPI, NFP, FOMC, ...), headlines, and the trading blackout windows the engine uses.

Why: a CPI / NFP / FOMC print moves gold and the dollar pairs by several times a normal candle, in both directions, in
seconds. A technically perfect setup that fires 10 minutes before one is a coin-flip with a bigger stop-out chance.
The engine therefore treats "high-impact news about to hit / just hit" as a hard gate (see strategy.evaluate_entry,
check key "news"), and the charts show every event so you can see what shifted the market.

Sources (all optional - the app works without them):
  * the public ForexFactory weekly JSON (this week + next week), refreshed every 30 min and STORED in news_events
  * rule-based recurring US events (NFP, FOMC decisions, weekly claims) so backtests over past weeks still know about
    them even though the feed only ever shows the current week
  * manual events you add yourself (a speech, an auction, an unscheduled meeting)
  * RSS headlines (FXStreet, ForexLive) for context
"""
from __future__ import annotations

import bisect
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from sqlalchemy.orm import Session

from . import models, settings
from .database import SessionLocal

log = logging.getLogger("goldsignal.news")
NY = ZoneInfo("America/New_York")
FEEDS = ("https://nfs.faireconomy.media/ff_calendar_thisweek.json", "https://nfs.faireconomy.media/ff_calendar_nextweek.json")
RSS = (("FXStreet", "https://www.fxstreet.com/rss/news"), ("ForexLive", "https://www.forexlive.com/feed/news"))
REFRESH_SECONDS = 30 * 60
_status = {"last_ok": None, "last_try": None, "error": None, "events": 0}
_head_cache: dict = {"ts": 0.0, "items": []}
_lock = threading.Lock()

# events that move EVERYTHING: wider blackout window
MAJOR = re.compile(r"\b(cpi|consumer price|non-?farm|nfp|employment change|fomc|federal funds|interest rate|rate decision|"
                   r"powell|core pce|pce price|gdp|retail sales|unemployment rate|ecb press|boe|boj|monetary policy)\b", re.I)
# FOMC decision days (second day of each meeting, 14:00 New York), published Fed calendar
FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
FOMC_2025 = ["2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10"]


# --------------------------------------------------------------------------------------------- symbols -> currencies
def currencies_for(symbol: str) -> set[str]:
    s = symbol.upper().replace("=X", "").replace("=F", "").replace("/", "")
    if len(s) == 6 and s.isalpha() and not s.startswith(("XAU", "XAG")):
        return {s[:3], s[3:]}
    return {"USD"}          # gold, silver, oil, indices: priced in dollars


def _utc(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ------------------------------------------------------------------------------------------------------ rule events
def _first_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def rule_events(from_ts: int, to_ts: int) -> list[dict]:
    """Recurring US releases that are calendar-fixed. Marked source='rule' (estimated); a feed event for the same
    release replaces them. Weekly claims are 'medium' (they only matter when the filter is set to 'all')."""
    out: list[dict] = []
    a, b = _utc(from_ts).date() - timedelta(days=1), _utc(to_ts).date() + timedelta(days=1)

    def add(d: date, hh: int, mm: int, title: str, impact: str):
        ts = int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY).timestamp())
        if from_ts <= ts <= to_ts:
            out.append({"ts": ts, "currency": "USD", "title": title, "impact": impact, "forecast": None, "previous": None,
                        "actual": None, "source": "rule"})
    y, m = a.year, a.month
    while date(y, m, 1) <= b:
        add(_first_friday(y, m), 8, 30, "Non-Farm Employment Change", "high")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    for ds in FOMC_2025 + FOMC_2026:
        d = date.fromisoformat(ds)
        if a <= d <= b:
            add(d, 14, 0, "FOMC Federal Funds Rate decision", "high")
            add(d, 14, 30, "FOMC Press Conference", "high")
    d = a
    while d <= b:
        if d.weekday() == 3:
            add(d, 8, 30, "Unemployment Claims", "medium")
        d += timedelta(days=1)
    return out


# ----------------------------------------------------------------------------------------------------- feed + store
def _uid(cur: str, title: str, ts: int) -> str:
    return f"{cur}|{re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()}|{ts}"


def parse_feed(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows or []:
        try:
            ts = int(datetime.fromisoformat(str(r["date"])).timestamp())
        except Exception:  # noqa: BLE001
            continue
        imp = str(r.get("impact", "")).lower()
        if imp not in ("high", "medium", "low"):
            continue                                    # "Holiday" and unknown rows
        out.append({"ts": ts, "currency": str(r.get("country", "")).upper()[:3], "title": str(r.get("title", "")).strip(),
                    "impact": imp, "forecast": (r.get("forecast") or None), "previous": (r.get("previous") or None),
                    "actual": (r.get("actual") or None), "source": "feed"})
    return out


def upsert(db: Session, events: list[dict]) -> int:
    n = 0
    for e in events:
        uid = _uid(e["currency"], e["title"], e["ts"])
        row = db.query(models.NewsEvent).filter(models.NewsEvent.uid == uid).first()
        if row is None:
            db.add(models.NewsEvent(uid=uid, **{k: e.get(k) for k in ("ts", "currency", "title", "impact", "forecast", "previous", "actual", "source")}))
            n += 1
        else:
            for k in ("impact", "forecast", "previous", "actual"):
                if e.get(k) is not None and e.get(k) != getattr(row, k):
                    setattr(row, k, e[k])
    db.commit()
    return n


def refresh(db: Session | None = None, force: bool = False) -> dict:
    """Fetch the public calendar and store it. Never raises: a dead feed only means 'no fresh events'."""
    own = db is None
    db = db or SessionLocal()
    try:
        with _lock:
            if not force and _status["last_ok"] and time.time() - _status["last_ok"] < REFRESH_SECONDS:
                return dict(_status)
            _status["last_try"] = time.time()
        total, err = 0, None
        for url in FEEDS:
            try:
                r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0 GoldSignal"})
                r.raise_for_status()
                total += len(parse_feed(r.json()))
                upsert(db, parse_feed(r.json()))
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:120]}"
                log.warning("news feed failed (%s): %s", url, err)
        with _lock:
            _status["error"] = err
            _status["events"] = total
            if total:
                _status["last_ok"] = time.time()
        return dict(_status)
    finally:
        if own:
            db.close()


def status() -> dict:
    with _lock:
        return dict(_status)


# ------------------------------------------------------------------------------------------------------ query
def _is_dup_of_feed(rule: dict, feed: list[dict]) -> bool:
    key = "non-farm" if "non-farm" in rule["title"].lower() else "fomc" if "fomc" in rule["title"].lower() else rule["title"].lower()[:12]
    for e in feed:
        if e["currency"] == rule["currency"] and abs(e["ts"] - rule["ts"]) <= 3 * 3600 and (
                key in e["title"].lower() or (key == "fomc" and "federal funds" in e["title"].lower())
                or (key == "non-farm" and ("nfp" in e["title"].lower() or "non farm" in e["title"].lower()))):
            return True
    return False


def events_between(db: Session, symbol: str | None, from_ts: int, to_ts: int, impact: str = "all",
                   include_rules: bool = True) -> list[dict]:
    """Calendar events in [from_ts, to_ts]. symbol=None -> every currency; otherwise only the pair's currencies.
    impact: 'high' | 'medium' (medium and high) | 'all'."""
    q = db.query(models.NewsEvent).filter(models.NewsEvent.ts >= from_ts, models.NewsEvent.ts <= to_ts).all()
    ev = [{"id": r.id, "ts": r.ts, "currency": r.currency, "title": r.title, "impact": r.impact, "forecast": r.forecast,
           "previous": r.previous, "actual": r.actual, "source": r.source} for r in q]
    if include_rules:
        feed = [e for e in ev if e["source"] != "rule"]
        ev += [r for r in rule_events(from_ts, to_ts) if not _is_dup_of_feed(r, feed)]
    if symbol:
        cur = currencies_for(symbol)
        ev = [e for e in ev if e["currency"] in cur]
    rank = {"high": 3, "medium": 2, "low": 1}
    floor = {"high": 3, "medium": 2, "all": 1}.get(impact, 1)
    ev = [e for e in ev if rank.get(e["impact"], 1) >= floor]
    ev.sort(key=lambda e: e["ts"])
    return ev


def window_for(e: dict, before: int, after: int, level: str) -> tuple[int, int] | None:
    """Blackout (start, end) epoch seconds for one event under the filter level, or None if it does not block."""
    if level == "off":
        return None
    major = bool(MAJOR.search(e["title"]))
    if e["impact"] == "high":
        k = 1.5 if major else 1.0
        return e["ts"] - int(before * 60 * k), e["ts"] + int(after * 60 * k)
    if e["impact"] == "medium" and level == "all":
        return e["ts"] - 10 * 60, e["ts"] + 5 * 60
    return None


def windows(db: Session, symbol: str, from_ts: int, to_ts: int, level: str | None = None,
            before: int | None = None, after: int | None = None) -> list[tuple]:
    """Merged blackout windows [(start, end, label, impact)] for the engine. Looks 1 day beyond the span on both sides."""
    pr = settings.news_prefs()
    level = level or pr["filter"]
    if level == "off":
        return []
    before = pr["before"] if before is None else before
    after = pr["after"] if after is None else after
    ev = events_between(db, symbol, from_ts - 86400, to_ts + 86400, "medium" if level == "all" else "high")
    out = []
    for e in ev:
        w = window_for(e, before, after, level)
        if w:
            out.append((w[0], w[1], f"{e['currency']} {e['title']}", e["impact"]))
    out.sort()
    merged: list[list] = []
    for s, t, lab, imp in out:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t)
            if lab not in merged[-1][2]:
                merged[-1][2] += f" + {lab}"
        else:
            merged.append([s, t, lab, imp])
    return [tuple(m) for m in merged]


def hit(wins: tuple | list, ts: int):
    """The blackout window containing ts (or None) and the start of the next one."""
    if not wins:
        return None, None
    starts = [w[0] for w in wins]
    k = bisect.bisect_right(starts, ts) - 1
    cur = wins[k] if k >= 0 and wins[k][1] >= ts else None
    nxt = wins[k + 1] if k + 1 < len(wins) else None
    return cur, nxt


# ---------------------------------------------------------------------------------------------------- headlines
_TAG = re.compile(r"<[^>]+>")
_BOILER = re.compile(r"(the post .{0,200}? appeared first on .*$|\bappeared first on\b.*$|read more.*$|continue reading.*$|\[…\]|\[\.\.\.\])", re.I | re.S)


def clean_summary(raw: str | None, title: str = "", max_chars: int = 420) -> str:
    """RSS description/content -> a short plain-text summary (whole sentences, no HTML, no 'appeared first on')."""
    import html
    if not raw:
        return ""
    t = html.unescape(_TAG.sub(" ", raw))
    t = _BOILER.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or t.lower().rstrip(".") == title.lower().rstrip("."):
        return ""
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[:end + 1] if end > max_chars * 0.4 else cut.rsplit(" ", 1)[0].rstrip(",;:") + "…"


def headlines(symbol: str | None = None, limit: int = 20) -> list[dict]:
    now = time.time()
    with _lock:
        fresh = now - _head_cache["ts"] < 600
        items = list(_head_cache["items"])
    if not fresh:
        items = []
        for name, url in RSS:
            try:
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 GoldSignal"})
                r.raise_for_status()
                root = ET.fromstring(r.content)
                for it in root.iter("item"):
                    title = (it.findtext("title") or "").strip()
                    if not title:
                        continue
                    try:
                        from email.utils import parsedate_to_datetime
                        ts = int(parsedate_to_datetime(it.findtext("pubDate")).timestamp())
                    except Exception:  # noqa: BLE001
                        ts = int(now)
                    full = it.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or it.findtext("description")
                    items.append({"ts": ts, "title": title, "link": (it.findtext("link") or "").strip(), "source": name,
                                  "major": bool(MAJOR.search(title)), "summary": clean_summary(full, title)})
            except Exception as e:  # noqa: BLE001
                log.warning("headlines %s failed: %s", name, e)
        items.sort(key=lambda x: -x["ts"])
        with _lock:
            if items:
                _head_cache.update(ts=now, items=items)
            else:
                _head_cache["ts"] = now - 540        # retry in a minute, keep the old items
                items = list(_head_cache["items"])
    if symbol:
        cur = currencies_for(symbol)
        words = {"USD": r"dollar|usd|fed|powell|cpi|nfp|payroll|treasury|yield", "EUR": r"euro|eur|ecb|lagarde", "GBP": r"pound|gbp|sterling|boe|bailey",
                 "JPY": r"yen|jpy|boj|ueda", "AUD": r"aussie|aud|rba", "NZD": r"nzd|rbnz", "CAD": r"cad|loonie|boc", "CHF": r"chf|snb"}
        pat = re.compile("|".join(words.get(c, c.lower()) for c in cur) + (r"|gold|xau" if symbol.upper().startswith(("GC", "XAU")) else ""), re.I)
        items = [x for x in items if x["major"] or pat.search(x["title"])]
    return items[:limit]


# ------------------------------------------------------------------------------------------------ background thread
def _loop():
    while True:
        try:
            refresh()
        except Exception:  # noqa: BLE001
            log.exception("news refresh crashed")
        time.sleep(REFRESH_SECONDS)


def start() -> None:
    import os
    if os.environ.get("GOLDSIGNAL_NEWS", "1") == "0" or os.environ.get("GOLDSIGNAL_DEMO") == "1":
        return
    threading.Thread(target=_loop, name="news-refresh", daemon=True).start()
