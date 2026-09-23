"""
Market-data layer.

Why this file exists (the "JSONDecodeError('Expecting value: line 1 column 1')" bug)
------------------------------------------------------------------------------------
Yahoo Finance regularly answers with an HTML / plain-text "Too Many Requests"
or consent page instead of JSON.  Old yfinance releases (the project pinned
0.2.43) crash while decoding it and  yf.download()  swallows the error and just
returns an empty frame  ->  "1 Failed download: ['GC=F']: JSONDecodeError".

What we do about it:
  1. requirements.txt now asks for a modern yfinance (curl_cffi based, browser
     impersonation) - the actual fix.  Upgrade with:
         pip install -U -r requirements.txt
  2. Every request goes through a chain of independent download methods
     (direct Yahoo chart API -> yfinance Ticker.history -> yf.download); the
     first that returns data wins.
  3. Results are cached (memory + disk) with a per-interval TTL; if every method
     fails we serve the last good cache (marked stale) instead of a 500.
  4. Failures are remembered for a short cool-down so the dashboard polling
     cannot hammer Yahoo while it is throttling us.
  5. If there is truly nothing, a DataUnavailable exception is raised and the
     routers translate it into a clean HTTP 503 with diagnostics.

Set GOLDSIGNAL_DEMO=1 to use synthetic data (offline UI/strategy testing).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import settings

log = logging.getLogger("goldsignal.data")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DEMO = os.environ.get("GOLDSIGNAL_DEMO", "").lower() in ("1", "true", "yes")

# public timeframe -> Yahoo interval
YF_INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
               "1h": "1h", "4h": "1h", "1d": "1h", "1w": "1h"}  # 4h/1d/1w are cut from 1h (broker-aligned)
INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800,
                    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
SUPPORTED = list(INTERVAL_SECONDS)

# 1m is the "live tape": every other timeframe is patched with it (see merge_live), so the
# slower frames can be cached longer without the newest candle ever lagging.
DEFAULT_PERIOD = {"1m": "2d", "5m": "30d", "15m": "30d", "30m": "45d",
                  "1h": "365d", "1d": "2y", "1wk": "5y"}
TTL = {"1m": 12, "5m": 90, "15m": 120, "30m": 180, "1h": 300, "1d": 120, "1wk": 600}
FAIL_COOLDOWN = 25  # seconds before retrying a failed download

SYMBOL_RE = re.compile(r"^[A-Za-z0-9=^.\-]{1,20}$")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


class DataUnavailable(Exception):
    def __init__(self, message: str, diagnostics: list[str] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


# ------------------------------------------------------------------ utilities
def check_symbol(symbol: str) -> str:
    if not SYMBOL_RE.match(symbol or ""):
        raise DataUnavailable(f"Invalid symbol '{symbol}'")
    return symbol


def is_futures(symbol: str) -> bool:
    return not symbol.endswith("=X")


def period_days(period: str) -> float:
    m = re.fullmatch(r"(\d+)(d|wk|mo|y)", period.strip().lower())
    if not m:
        raise ValueError(f"bad period {period!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 1, "wk": 7, "mo": 30.5, "y": 365.25}[unit]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    need = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in need):
        raise ValueError(f"missing OHLC columns: {list(df.columns)}")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df = df[need + ["Volume"]].astype(float)
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df["Volume"] = df["Volume"].fillna(0.0)
    df = df.dropna(subset=need)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "time"
    return df


# ------------------------------------------------------------- download methods
def parse_chart_json(payload: dict) -> pd.DataFrame:
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(str(chart["error"])[:120])
    res = (chart.get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        raise RuntimeError("empty chart result")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
         "Close": q.get("close"), "Volume": q.get("volume")},
        index=pd.to_datetime(res["timestamp"], unit="s", utc=True),
    )
    return _normalize(df)


def _fetch_direct(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
    """Plain HTTP call to Yahoo's chart endpoint (no cookies/crumb needed)."""
    p2 = int(time.time()) + 60
    p1 = int(p2 - period_days(period) * 86400)
    last: Exception | None = None
    for host in ("query1", "query2"):
        url = f"https://{host}.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe='')}"
        try:
            r = requests.get(
                url,
                params={"period1": p1, "period2": p2, "interval": yf_interval,
                        "includePrePost": "false", "events": "div,splits"},
                headers={"User-Agent": UA, "Accept": "application/json"},
                timeout=12,
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]!r}")
            return parse_chart_json(r.json())
        except Exception as e:  # noqa: BLE001
            last = e
    raise last  # type: ignore[misc]


def _fetch_yf_history(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(symbol).history(period=period, interval=yf_interval,
                                   auto_adjust=False, actions=False, timeout=15)
    if df is None or df.empty:
        raise RuntimeError("empty frame")
    return _normalize(df)


def _fetch_yf_download(symbol: str, yf_interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=yf_interval, progress=False,
                     auto_adjust=False, threads=False, timeout=15)
    if df is None or df.empty:
        raise RuntimeError("empty frame")
    return _normalize(df)


# yfinance impersonates a browser (curl_cffi) and copes with Yahoo's cookie/crumb -> preferred.
# The raw "direct" call is a fallback: from many networks Yahoo answers it with HTTP 429.
METHODS = (("yfinance.history", _fetch_yf_history), ("direct", _fetch_direct),
           ("yfinance.download", _fetch_yf_download))

_method_block: dict[str, float] = {}      # method -> unix time until which we skip it
_gate = threading.Semaphore(1)            # one Yahoo request at a time (yfinance/curl_cffi is not thread-safe)
_last_call = [0.0]
_call_lock = threading.Lock()
BLOCK_SECONDS = 120


def _throttle() -> None:
    """Space Yahoo requests >= 0.2 s apart so bursts of dashboard polling don't trigger 429."""
    with _call_lock:
        wait = 0.2 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def _download(symbol: str, yf_interval: str, period: str) -> tuple[pd.DataFrame, str, list[str]]:
    diag: list[str] = []
    now = time.time()
    order = [m for m in METHODS if _method_block.get(m[0], 0) <= now] or list(METHODS)
    for name, fn in order:
        try:
            with _gate:
                _throttle()
                df = fn(symbol, yf_interval, period)
            if len(df):
                return df, name, diag
            diag.append(f"{name}: no rows")
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:110]}"
            diag.append(f"{name}: {msg}")
            if "429" in msg or "Too Many" in msg or "Rate" in msg:
                _method_block[name] = time.time() + BLOCK_SECONDS  # stop poking a method that is being throttled
    raise DataUnavailable(f"Could not download {symbol} {yf_interval} from Yahoo Finance", diag)


# ---------------------------------------------------------------------- caching
_mem: dict[tuple, tuple[float, pd.DataFrame]] = {}
_meta: dict[tuple, dict] = {}
_fail_until: dict[tuple, tuple[float, list[str]]] = {}
_locks: dict[tuple, threading.Lock] = {}
_glock = threading.Lock()
LAST_DIAGNOSTICS: dict = {"ok": None, "at": None, "detail": []}


def _lock(key: tuple) -> threading.Lock:
    with _glock:
        return _locks.setdefault(key, threading.Lock())


def _disk_path(symbol: str, interval: str, period: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]", "", symbol)
    return os.path.join(CACHE_DIR, f"{safe}_{interval}_{period}.csv")


def _read_disk(path: str) -> pd.DataFrame | None:
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "time"
        return df if len(df) else None
    except Exception:  # noqa: BLE001
        return None


def _get_raw(symbol: str, yf_interval: str, period: str | None = None) -> pd.DataFrame:
    """Cached fetch of a native Yahoo interval (1m,5m,15m,30m,1h,1d,1wk)."""
    period = period or DEFAULT_PERIOD[yf_interval]
    key = (symbol, yf_interval, period)
    ttl = TTL.get(yf_interval, 120)

    hit = _mem.get(key)
    if hit and time.time() - hit[0] <= ttl:
        return hit[1]

    with _lock(key):
        hit = _mem.get(key)
        if hit and time.time() - hit[0] <= ttl:
            return hit[1]

        # ---- demo mode
        if DEMO:
            from .demo_data import demo_candles
            df = demo_candles(symbol, yf_interval, period_days(period))
            _mem[key] = (time.time(), df)
            _meta[key] = {"source": "demo", "fetched_at": int(time.time()), "stale": False}
            return df

        # ---- recently failed -> don't hammer Yahoo
        stale = hit[1] if hit else _read_disk(_disk_path(symbol, yf_interval, period))
        cool = _fail_until.get(key)
        if cool and time.time() < cool[0]:
            if stale is not None:
                _meta[key] = {"source": "cache", "stale": True, "error": cool[1][:3]}
                return stale
            raise DataUnavailable(f"Data feed unavailable for {symbol} {yf_interval}", cool[1])

        try:
            df, method, diag = _download(symbol, yf_interval, period)
            _mem[key] = (time.time(), df)
            _meta[key] = {"source": method, "fetched_at": int(time.time()), "stale": False}
            LAST_DIAGNOSTICS.update(ok=True, at=int(time.time()),
                                    detail=[f"{symbol} {yf_interval}: {method}"] + diag)
            try:
                df.to_csv(_disk_path(symbol, yf_interval, period))
            except OSError:
                pass
            _fail_until.pop(key, None)
            return df
        except DataUnavailable as e:
            _fail_until[key] = (time.time() + FAIL_COOLDOWN, e.diagnostics)
            LAST_DIAGNOSTICS.update(ok=False, at=int(time.time()), detail=[f"{symbol} {yf_interval}"] + e.diagnostics)
            log.warning("Data download failed %s %s: %s", symbol, yf_interval, e.diagnostics)
            if stale is not None:
                _meta[key] = {"source": "cache", "stale": True, "error": e.diagnostics[:3]}
                return stale
            raise


# ------------------------------------------------------------------- public API
def resample_ohlc(df: pd.DataFrame, rule: str, symbol: str = "GC=F") -> pd.DataFrame:
    """Cut 4h / 1d / 1w candles the way MetaTrader 5 does (see settings.py):
    ny_close -> broker server time = New York time + 7 h (GMT+2 winter / GMT+3 summer, DST follows the US),
                so the trading day starts at 17:00 New York and 4H candles start 17,21,01,05,09,13 NY time
    utc      -> candles cut on UTC boundaries"""
    if df.empty:
        return df
    idx = df.index
    utc_naive = idx.tz_convert("UTC").tz_localize(None)
    if settings.anchor() == "ny_close":
        et = idx.tz_convert("America/New_York").tz_localize(None)
        off = et - utc_naive                       # negative: NY is behind UTC
        shift = pd.Timedelta(hours=7)
        srv = et + shift
    else:
        off = utc_naive - utc_naive                # zero timedeltas
        shift = pd.Timedelta(0)
        srv = utc_naive
    if rule == "4h":
        key = srv.floor("4h")
    elif rule == "1d":
        key = srv.floor("D")
    elif rule == "1w":
        d = srv.floor("D")
        key = d - pd.to_timedelta(d.dayofweek, unit="D")   # week starts Monday 00:00 server time
    else:
        raise ValueError(rule)
    tmp = df.reset_index(drop=True).assign(_k=key, _off=off)
    g = tmp.groupby("_k")
    out = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                        "Close": g["Close"].last(), "Volume": g["Volume"].sum(), "_off": g["_off"].first()})
    start = out.index.to_series() - shift - out["_off"]     # back to naive UTC
    out = out.drop(columns="_off")
    out.index = pd.DatetimeIndex(start.to_numpy()).tz_localize("UTC")
    out.index.name = "time"
    return out


def merge_live(base: pd.DataFrame, interval: str, symbol: str) -> pd.DataFrame:
    """Patch a (cached, possibly minutes-old) frame with the live 1-minute tape so the newest
    candle of EVERY timeframe keeps moving, and brand-new candles appear the moment they open.

    Only the TAIL is touched (the last cached bar plus whatever the 1m tape covers since then) and
    every rebuilt bar is cut on an EPOCH-aligned boundary (floor(unix_time / step) * step) - the same
    grid Yahoo's own 5m/15m/30m/1h bars sit on. Bucketing relative to "the last cached bar's start"
    instead of the epoch was the bug that made 1m/5m candles look slightly misaligned / jump on refresh:
    if the cached base bar wasn't itself sitting exactly on a round boundary, every synthetic bar drifted
    with it and snapped back the moment a fresh native fetch replaced the cache.
    """
    if interval == "1m" or base.empty:
        return base
    try:
        m1 = _get_raw(symbol, "1m")
    except DataUnavailable:
        return base
    if m1.empty:
        return base
    step = INTERVAL_SECONDS[interval]
    df = base.copy()
    last_start = df.index[-1]
    # Only rebuild from one bar back (safety margin in case the cached last bar was itself partial) -
    # never touches older history, so this stays cheap even with a 2-day 1m tape.
    window_start = last_start - timedelta(seconds=step)
    seg = m1[m1.index >= window_start]
    if seg.empty:
        return df

    secs = seg.index.as_unit("s").asi8
    bucket = (secs // step) * step
    g = seg.assign(_k=bucket).groupby("_k")
    new_bars = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                             "Close": g["Close"].last(), "Volume": g["Volume"].sum()})
    new_bars.index = pd.to_datetime(new_bars.index, unit="s", utc=True)
    new_bars.index.name = "time"
    if new_bars.empty:
        return df

    cutoff = new_bars.index.min()
    df = pd.concat([df[df.index < cutoff], new_bars]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _candles(symbol: str, interval: str, period: str | None, live: bool) -> pd.DataFrame:
    if interval in ("4h", "1d", "1w"):
        base = _candles(symbol, "1h", period, live)
        return resample_ohlc(base, interval, symbol)
    df = _get_raw(symbol, YF_INTERVAL[interval], period)
    return merge_live(df, interval, symbol) if live else df


def get_candles(symbol: str, interval: str = "15m", period: str | None = None, live: bool = True) -> pd.DataFrame:
    """OHLCV DataFrame (UTC index, columns Open High Low Close Volume).
    Includes the still-forming last bar; use drop_incomplete() for analysis.
    live=True patches the newest bar(s) with the 1-minute feed (ignored when a custom period is asked for).
    4h/1d/1w are cut from 1h candles on MT5-style boundaries and every price gets the calibration offset."""
    check_symbol(symbol)
    if interval not in INTERVAL_SECONDS:
        raise DataUnavailable(f"Unsupported timeframe '{interval}'")
    df = _candles(symbol, interval, period, live and period is None)
    off = settings.offset(symbol)
    if off:
        df = df.copy()
        df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] + off
    return df


def raw_last_price(symbol: str) -> float:
    """Latest Yahoo price WITHOUT the calibration offset (used to calibrate)."""
    m1 = _get_raw(symbol, "1m")
    return float(m1["Close"].iloc[-1])


CLOSE_GRACE = 15  # seconds after a bar's end before it counts as closed (lets the last 1m bar arrive)


def drop_incomplete(df: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    """Remove the last bar if it has not closed yet."""
    if df.empty:
        return df
    from .sessions import now_utc
    now = now or now_utc()
    end = df.index[-1] + timedelta(seconds=INTERVAL_SECONDS[interval] + CLOSE_GRACE)
    return df.iloc[:-1] if now < end else df


def get_meta(symbol: str, interval: str) -> dict:
    yf_i = YF_INTERVAL.get(interval, interval)
    for (s, i, _p), m in _meta.items():
        if s == symbol and i == yf_i:
            return m
    return {}


STALE_AFTER = 12 * 60  # seconds without a new 1m bar (while the market is open) before we call the feed stale


def data_status(symbol: str, interval: str, df: pd.DataFrame) -> dict:
    """Freshness info for the UI. Lag is measured on the 1-minute tape (the freshest thing we have)."""
    from .sessions import market_open, now_utc
    meta = get_meta(symbol, interval)
    now = now_utc()
    lag = None
    try:
        m1 = _get_raw(symbol, "1m")
        if len(m1):
            lag = (now - m1.index[-1].to_pydatetime()).total_seconds() - 60
    except DataUnavailable:
        pass
    if lag is None and len(df):
        lag = (now - df.index[-1].to_pydatetime()).total_seconds() - INTERVAL_SECONDS[interval]
    mkt = market_open(None, is_futures(symbol))
    stale = bool(not DEMO and mkt and lag is not None and lag > STALE_AFTER)
    return {
        "source": meta.get("source", "unknown"),
        "last_bar": int(df.index[-1].timestamp()) if len(df) else None,
        "lag_sec": int(max(0, lag)) if lag is not None else None,
        "stale": stale,
        "market_open": mkt,
        "demo": DEMO,
    }


def feed_health(symbol: str = "GC=F") -> dict:
    """Live connectivity test used by /market/health (bypasses caches)."""
    if DEMO:
        return {"demo": True, "ok": True, "methods": [{"method": "demo", "ok": True}]}
    out = []
    for name, fn in METHODS:
        t0 = time.time()
        try:
            df = fn(symbol, "1h", "5d")
            out.append({"method": name, "ok": True, "rows": len(df), "ms": int((time.time() - t0) * 1000),
                        "last": df.index[-1].isoformat()})
        except Exception as e:  # noqa: BLE001
            out.append({"method": name, "ok": False, "error": f"{type(e).__name__}: {str(e)[:140]}",
                        "ms": int((time.time() - t0) * 1000)})
    try:
        import yfinance
        ver = yfinance.__version__
    except Exception:  # noqa: BLE001
        ver = "not installed"
    ok = any(m["ok"] for m in out)
    return {"demo": False, "ok": ok, "yfinance_version": ver, "methods": out,
            "hint": None if ok else
            "All download methods failed. Run: pip install -U -r requirements.txt (yfinance must be recent), "
            "then check this machine can reach query1.finance.yahoo.com (VPN / firewall / DNS)."}


def clear_caches() -> None:
    _mem.clear()
    _fail_until.clear()
