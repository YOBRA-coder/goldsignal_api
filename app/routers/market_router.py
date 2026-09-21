from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import auth, models, sessions
from ..data_fetcher import (DataUnavailable, INTERVAL_SECONDS, LAST_DIAGNOSTICS, data_status, drop_incomplete,
                            feed_health, get_candles, is_futures)
from ..levels import key_levels
from ..strategy import own_structure, quick_bias
from ..structure import atr_array

router = APIRouter(prefix="/market", tags=["market"])

TF = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
MTF_ORDER = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]


def _503(e: DataUnavailable) -> HTTPException:
    return HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics,
                                      "hint": "Run `pip install -U -r requirements.txt` and open /market/health."})


@router.get("/candles")
def candles(
    symbol: str = "GC=F",
    interval: TF = "15m",
    limit: int = Query(500, ge=50, le=1500),
    structure: bool = True,
    user: models.User = Depends(auth.get_current_user),
):
    try:
        full = get_candles(symbol, interval)
    except DataUnavailable as e:
        raise _503(e)
    struct = None
    if structure:
        try:  # structure of THIS timeframe (BOS/CHoCH, zones, liquidity...) from closed bars
            struct = own_structure(drop_incomplete(full, interval), interval)
        except Exception:  # noqa: BLE001
            struct = None
    df = full.tail(limit)
    times = df.index.as_unit("s").asi8
    intraday = INTERVAL_SECONDS[interval] < 86400
    tags = sessions.tag_sessions(df.index) if intraday else None
    rows = []
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    for k in range(len(df)):
        r = {"time": int(times[k]), "open": o[k], "high": h[k], "low": l[k], "close": c[k], "volume": v[k]}
        if tags:
            r["session"] = tags[k]
        rows.append(r)
    return {"symbol": symbol, "interval": interval, "candles": rows, "data": data_status(symbol, interval, df),
            "has_volume": bool(np.nansum(v[-80:]) > 0), "structure": struct}


def _quote(symbol: str, live: bool = True) -> dict:
    """Last price + day change. live=True uses the 1-minute tape (updates every few seconds)."""
    d = get_candles(symbol, "1d", live=live)
    if len(d) < 2:
        raise DataUnavailable("not enough daily data")
    last_bar = d.iloc[-1]
    done = drop_incomplete(d, "1d")
    prev_close = float(d.iloc[-2]["Close"]) if len(done) == len(d) else float(done.iloc[-1]["Close"])
    last = float(last_bar["Close"])
    out = {"symbol": symbol, "last": last, "prev_close": prev_close, "change": last - prev_close,
           "change_pct": (last / prev_close - 1) * 100 if prev_close else 0.0,
           "day_high": float(last_bar["High"]), "day_low": float(last_bar["Low"])}
    return out


@router.get("/quote")
def quote(symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    """Fast live quote for the market-watch panel / top bar (poll every few seconds)."""
    try:
        q = _quote(symbol, live=True)
        m1 = get_candles(symbol, "1m")
    except DataUnavailable as e:
        raise _503(e)
    return {**q, "data": data_status(symbol, "1m", m1), "bar": {
        "time": int(m1.index[-1].timestamp()), "open": float(m1["Open"].iloc[-1]), "high": float(m1["High"].iloc[-1]),
        "low": float(m1["Low"].iloc[-1]), "close": float(m1["Close"].iloc[-1])}}


@router.get("/watchlist")
def watchlist(symbols: str = "GC=F,XAUUSD=X,SI=F,EURUSD=X,GBPUSD=X,USDJPY=X", user: models.User = Depends(auth.get_current_user)):
    syms = [s.strip() for s in symbols.split(",") if s.strip()][:12]

    def one(s):
        try:
            return {**_quote(s, live=False), "ok": True}
        except Exception as e:  # noqa: BLE001
            return {"symbol": s, "ok": False, "error": str(e)[:80]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        return {"quotes": list(ex.map(one, syms))}


@router.get("/overview")
def overview(symbol: str = "GC=F", user: models.User = Depends(auth.get_current_user)):
    """Market-watch data for one symbol: quote, ATR, key levels, multi-timeframe bias, sessions."""
    try:
        quote = _quote(symbol, live=True)
        d1h = get_candles(symbol, "1h")
    except DataUnavailable as e:
        raise _503(e)

    def tf_bias(tf):
        try:
            df = drop_incomplete(get_candles(symbol, tf), tf)
            return tf, quick_bias(df, 3 if tf in ("4h", "1d", "1w") else 2)
        except Exception as e:  # noqa: BLE001
            return tf, {"bias": "n/a", "error": str(e)[:60], "last_event": None, "labels": []}

    with ThreadPoolExecutor(max_workers=4) as ex:
        mtf = dict(ex.map(tf_bias, MTF_ORDER))

    try:
        daily, weekly = get_candles(symbol, "1d"), get_candles(symbol, "1w")
    except DataUnavailable:
        daily = weekly = None
    levels = key_levels(daily, weekly, d1h, quote["last"])

    atr1h = float(atr_array(d1h["High"].to_numpy(float), d1h["Low"].to_numpy(float), d1h["Close"].to_numpy(float))[-1])
    atrd = None
    if daily is not None and len(daily) > 15:
        atrd = float(atr_array(daily["High"].to_numpy(float), daily["Low"].to_numpy(float), daily["Close"].to_numpy(float))[-1])
    votes = [mtf[t]["bias"] for t in ("15m", "1h", "4h", "1d")]
    bull, bear = votes.count("bullish"), votes.count("bearish")
    return {
        "symbol": symbol, "quote": quote, "atr_1h": atr1h, "atr_daily": atrd, "levels": levels,
        "mtf": [{"tf": t, **mtf[t]} for t in MTF_ORDER],
        "alignment": {"bull": bull, "bear": bear, "of": len(votes)},
        "sessions": sessions.session_state(futures=is_futures(symbol)),
        "data": data_status(symbol, "1h", d1h),
    }


@router.get("/sessions")
def sessions_now(symbol: str = "GC=F"):
    return sessions.session_state(futures=is_futures(symbol))


@router.get("/health")
def health(symbol: str = "GC=F"):
    """No login needed: live test of every data-download method (helps debug the feed)."""
    return {**feed_health(symbol), "last_result": LAST_DIAGNOSTICS}
