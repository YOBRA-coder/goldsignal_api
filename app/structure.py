"""
Price-action primitives used by the strategy engine.

Everything here works on plain numpy arrays (class `Bars`) and returns plain
dicts, so results serialise straight to JSON for the chart overlays.

Terminology
-----------
swing         fractal high/low, cleaned into an alternating zig-zag
BOS           break of structure  = close beyond the last swing in trend direction
CHoCH         change of character = close beyond the last swing AGAINST the trend
order block   last opposite candle before the impulse that produced a BOS/CHoCH
              ("refined": candle body + the wick on the reaction side)
FVG           fair value gap (3-candle imbalance), shrinks as it is filled
zone state    fresh (never revisited) / tested (revisited) / broken (closed through)
liquidity     equal highs/lows + unswept swing points; a "sweep" is a wick beyond
              a swing that closes back inside
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BULL, BEAR = 1, -1


# ------------------------------------------------------------------- utilities
def num(x):
    """float for JSON (None for NaN/inf)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


class Bars:
    """Numpy view of an OHLCV frame (UTC index)."""

    def __init__(self, df: pd.DataFrame, atr_period: int = 14):
        self.n = len(df)
        self.t = df.index.as_unit("s").asi8.astype(np.int64) if self.n else np.zeros(0, np.int64)
        self.o = df["Open"].to_numpy(float)
        self.h = df["High"].to_numpy(float)
        self.l = df["Low"].to_numpy(float)
        self.c = df["Close"].to_numpy(float)
        self.v = df["Volume"].to_numpy(float)
        self.atr = atr_array(self.h, self.l, self.c, atr_period)
        self.has_volume = bool(self.n and np.nansum(self.v[-80:]) > 0)


def atr_array(h, l, c, period: int = 14) -> np.ndarray:
    if len(c) == 0:
        return np.zeros(0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


# ---------------------------------------------------------------------- swings
def find_swings(b: Bars, left: int = 2, right: int = 2) -> list[tuple[int, int, float]]:
    """Fractal swings -> [(index, type(+1 high / -1 low), price)] (vectorised)."""
    w = left + right + 1
    if b.n < w + 1:
        return []
    raw: list[tuple[int, int, float]] = []
    hw = sliding_window_view(b.h, w)
    lw = sliding_window_view(b.l, w)
    is_h = (hw[:, left] > hw[:, :left].max(axis=1)) & (hw[:, left] >= hw[:, left + 1:].max(axis=1))
    is_l = (lw[:, left] < lw[:, :left].min(axis=1)) & (lw[:, left] <= lw[:, left + 1:].min(axis=1))
    for k in np.nonzero(is_h)[0]:
        raw.append((int(k) + left, BULL, float(b.h[k + left])))
    for k in np.nonzero(is_l)[0]:
        raw.append((int(k) + left, BEAR, float(b.l[k + left])))
    raw.sort(key=lambda x: x[0])
    return raw


def zigzag(raw, right: int) -> list[dict]:
    """Alternate high/low; consecutive same-type swings keep the extreme one."""
    zz: list[dict] = []
    for idx, typ, price in raw:
        if zz and zz[-1]["type"] == typ:
            better = price >= zz[-1]["price"] if typ == BULL else price <= zz[-1]["price"]
            if better:
                zz[-1].update(idx=idx, price=price)
        else:
            zz.append({"idx": idx, "type": typ, "price": price})
    for s in zz:
        s["conf"] = s["idx"] + right  # bar on which the swing becomes known
    return zz


# ------------------------------------------------------------------- structure
def analyze_structure(b: Bars, left: int = 2, right: int = 2) -> dict:
    swings = zigzag(find_swings(b, left, right), right)

    last = {BULL: None, BEAR: None}
    for s in swings:
        prev = last[s["type"]]
        if prev is None:
            s["label"] = None
        elif s["type"] == BULL:
            s["label"] = "HH" if s["price"] > prev else "LH"
        else:
            s["label"] = "HL" if s["price"] > prev else "LL"
        last[s["type"]] = s["price"]

    events: list[dict] = []
    trend, sh, sl, p = 0, None, None, 0
    c = b.c
    for i in range(b.n):
        while p < len(swings) and swings[p]["conf"] <= i:
            s = swings[p]
            p += 1
            if s["type"] == BULL:
                sh = s
            else:
                sl = s
        if sh is not None and c[i] > sh["price"]:
            kind = "CHoCH" if trend == BEAR else "BOS"
            events.append({"type": kind, "dir": BULL, "level": sh["price"], "level_idx": sh["idx"], "idx": i})
            trend, sh = BULL, None
        elif sl is not None and c[i] < sl["price"]:
            kind = "CHoCH" if trend == BULL else "BOS"
            events.append({"type": kind, "dir": BEAR, "level": sl["price"], "level_idx": sl["idx"], "idx": i})
            trend, sl = BEAR, None

    highs = [s for s in swings if s["type"] == BULL]
    lows = [s for s in swings if s["type"] == BEAR]
    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None
    eq = None
    if last_high and last_low and last_high["price"] > last_low["price"]:
        eq = (last_high["price"] + last_low["price"]) / 2
    run = 0
    for ev in reversed(events):
        if ev["dir"] == trend:
            run += 1
        else:
            break
    return {"swings": swings, "events": events, "trend": trend, "last_high": last_high,
            "last_low": last_low, "eq": eq, "run": run}


def bias_name(trend: int) -> str:
    return "bullish" if trend == BULL else "bearish" if trend == BEAR else "neutral"


# ----------------------------------------------------------------------- zones
def _track_zone(b: Bars, z: dict, start: int) -> dict:
    """Fill z['state'], z['touches'], z['end_idx'] from bars after `start`."""
    demand = z["type"] == "demand"
    z["state"], z["touches"], z["end_idx"] = "fresh", 0, None
    if start >= b.n:
        return z
    if demand:
        touched = b.l[start:] <= z["top"]
        broke = b.c[start:] < z["bottom"]
    else:
        touched = b.h[start:] >= z["bottom"]
        broke = b.c[start:] > z["top"]
    if broke.any():
        z["state"], z["end_idx"] = "broken", start + int(np.argmax(broke))
        return z
    if touched.any():
        z["state"] = "tested"
        z["touches"] = int(np.count_nonzero(touched[1:] & ~touched[:-1]) + (1 if touched[0] else 0))
    return z


def zones_from_events(b: Bars, events: list[dict], min_leg_atr: float = 1.0, max_events: int = 14) -> list[dict]:
    """Refined order blocks: the last opposite candle at the origin of the leg
    that broke structure."""
    out = []
    for ev in events[-max_events:]:
        d, k, i = ev["dir"], ev["level_idx"], ev["idx"]
        if i <= k:
            continue
        if d == BULL:
            m = k + int(np.argmin(b.l[k:i + 1]))
            leg = float(b.h[m:i + 1].max() - b.l[m])
        else:
            m = k + int(np.argmax(b.h[k:i + 1]))
            leg = float(b.h[m] - b.l[m:i + 1].min())
        atr = b.atr[m]
        if atr <= 0 or leg < min_leg_atr * atr:
            continue
        lo, hi = max(k, m - 2), min(i - 1, m + 2)
        cands = [j for j in range(lo, hi + 1) if (b.c[j] < b.o[j] if d == BULL else b.c[j] > b.o[j])]
        if cands:
            j = min(cands, key=lambda q: b.l[q]) if d == BULL else max(cands, key=lambda q: b.h[q])
        else:
            j = m
        if d == BULL:
            bottom, top = float(b.l[j]), float(max(b.o[j], b.c[j]))
            top = max(top, bottom + 0.15 * atr)
        else:
            top, bottom = float(b.h[j]), float(min(b.o[j], b.c[j]))
            bottom = min(bottom, top - 0.15 * atr)
        z = {"type": "demand" if d == BULL else "supply", "source": "OB", "top": top, "bottom": bottom,
             "idx": j, "formed_idx": i, "strength": leg / atr, "event": ev["type"]}
        out.append(_track_zone(b, z, i + 1))
    return out


def base_zones(b: Bars, max_bars: int = 260, impulse_atr: float = 1.4) -> list[dict]:
    """Classic supply/demand: tight base followed by an impulsive candle that
    closes out of it."""
    out = []
    for i in range(max(3, b.n - max_bars), b.n - 1):
        atr = b.atr[i - 1]
        body, rng = abs(b.c[i] - b.o[i]), b.h[i] - b.l[i]
        if atr <= 0 or body < impulse_atr * atr or body < 0.6 * rng:
            continue
        d = BULL if b.c[i] > b.o[i] else BEAR
        base, k = [], i - 1
        while k >= max(0, i - 3) and (b.h[k] - b.l[k]) <= 1.0 * b.atr[k]:
            base.append(k)
            k -= 1
        if not base:
            base = [i - 1]
        top, bottom = float(max(b.h[j] for j in base)), float(min(b.l[j] for j in base))
        if (d == BULL and b.c[i] <= top) or (d == BEAR and b.c[i] >= bottom):
            continue
        z = {"type": "demand" if d == BULL else "supply", "source": "BASE", "top": top, "bottom": bottom,
             "idx": min(base), "formed_idx": i, "strength": float(body / atr), "event": None}
        out.append(_track_zone(b, z, i + 1))
    return out


def merge_zones(zones: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for z in sorted(zones, key=lambda q: -q["strength"]):
        dup = False
        for k in kept:
            if k["type"] != z["type"]:
                continue
            ov = min(k["top"], z["top"]) - max(k["bottom"], z["bottom"])
            small = min(k["top"] - k["bottom"], z["top"] - z["bottom"])
            if small > 0 and ov / small >= 0.6:
                dup = True
                break
        if not dup:
            kept.append(z)
    return sorted(kept, key=lambda q: q["formed_idx"])


def find_fvgs(b: Bars, max_bars: int = 220, min_atr: float = 0.12) -> list[dict]:
    """Unfilled fair value gaps (partially filled ones are shrunk to what remains)."""
    out = []
    h, l = b.h, b.l
    for i in range(max(2, b.n - max_bars), b.n):
        atr = b.atr[i]
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= min_atr * atr:  # bullish gap
            top0, bot = float(l[i]), float(h[i - 2])
            fut = l[i + 1:]
            top, state = top0, "fresh"
            if fut.size:
                mn = float(fut.min())
                if mn <= bot:
                    continue
                if mn < top0:
                    top, state = mn, "tested"
            out.append({"type": "demand", "source": "FVG", "top": top, "bottom": bot, "idx": i - 2,
                        "formed_idx": i, "state": state, "touches": int(state == "tested"),
                        "strength": float((top0 - bot) / atr), "end_idx": None, "event": None})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= min_atr * atr:  # bearish gap
            bot0, top = float(h[i]), float(l[i - 2])
            fut = h[i + 1:]
            bot, state = bot0, "fresh"
            if fut.size:
                mx = float(fut.max())
                if mx >= top:
                    continue
                if mx > bot0:
                    bot, state = mx, "tested"
            out.append({"type": "supply", "source": "FVG", "top": top, "bottom": bot, "idx": i - 2,
                        "formed_idx": i, "state": state, "touches": int(state == "tested"),
                        "strength": float((top - bot0) / atr), "end_idx": None, "event": None})
    return out


# ------------------------------------------------------------------- liquidity
def liquidity(b: Bars, swings: list[dict], max_swings: int = 50) -> tuple[list[dict], list[dict]]:
    """Returns (pools, sweeps).

    pools  : unswept equal highs/lows (EQH/EQL) and lone swing highs/lows (BSL/SSL)
    sweeps : wick beyond a swing that closed back inside (a liquidity raid)
    """
    if b.n < 5:
        return [], []
    sw = swings[-max_swings:]
    tol = max(0.15 * b.atr[-1], b.c[-1] * 0.0001)
    pools, sweeps = [], []

    for s in sw:
        j0 = s["conf"] + 1
        if s["type"] == BULL:
            beyond = b.h[j0:] > s["price"]
        else:
            beyond = b.l[j0:] < s["price"]
        s["swept_idx"] = None
        if j0 < b.n and beyond.any():
            j = j0 + int(np.argmax(beyond))
            s["swept_idx"] = j
            closed_back = b.c[j] < s["price"] if s["type"] == BULL else b.c[j] > s["price"]
            if closed_back:
                sweeps.append({"type": "bsl_sweep" if s["type"] == BULL else "ssl_sweep",
                               "dir": BEAR if s["type"] == BULL else BULL,
                               "level": s["price"], "idx": j, "level_idx": s["idx"],
                               "wick": float(b.h[j] if s["type"] == BULL else b.l[j])})

    for typ, name_eq, name_one in ((BULL, "EQH", "BSL"), (BEAR, "EQL", "SSL")):
        pts = sorted([s for s in sw if s["type"] == typ], key=lambda s: s["price"])
        groups, cur = [], []
        for s in pts:
            if cur and s["price"] - cur[0]["price"] > tol:
                groups.append(cur)
                cur = []
            cur.append(s)
        if cur:
            groups.append(cur)
        for g in groups:
            unswept = [s for s in g if s["swept_idx"] is None]
            if len(g) >= 2 and len(unswept) >= 2:
                price = max(s["price"] for s in g) if typ == BULL else min(s["price"] for s in g)
                pools.append({"type": name_eq, "price": float(price), "idx": min(s["idx"] for s in g),
                              "idx2": max(s["idx"] for s in g), "count": len(g), "swept": False})
            elif len(g) == 1 and g[0]["swept_idx"] is None:
                pools.append({"type": name_one, "price": float(g[0]["price"]), "idx": g[0]["idx"],
                              "idx2": g[0]["idx"], "count": 1, "swept": False})
    return pools, sweeps


# ------------------------------------------------------------------ trend lines
def trendlines(b: Bars, swings: list[dict]) -> list[dict]:
    """Ascending line through the last two rising lows / descending line through
    the last two falling highs (must not have been cut by a close in between)."""
    out = []
    if b.n < 10:
        return out
    for typ, kind in ((BEAR, "up"), (BULL, "down")):
        pts = [s for s in swings if s["type"] == typ][-5:]
        for a, c2 in reversed(list(zip(pts[:-1], pts[1:]))):
            rising = c2["price"] > a["price"] if kind == "up" else c2["price"] < a["price"]
            if not rising or c2["idx"] - a["idx"] < 3:
                continue
            slope = (c2["price"] - a["price"]) / (c2["idx"] - a["idx"])
            idxs = np.arange(a["idx"], b.n)
            line = a["price"] + slope * (idxs - a["idx"])
            tol = 0.1 * b.atr[-1]
            mid = slice(0, c2["idx"] - a["idx"] + 1)
            if kind == "up":
                bad_mid = (b.c[a["idx"]:c2["idx"] + 1] < line[mid] - tol).any()
                cut = b.c[a["idx"]:] < line - tol
            else:
                bad_mid = (b.c[a["idx"]:c2["idx"] + 1] > line[mid] + tol).any()
                cut = b.c[a["idx"]:] > line + tol
            if bad_mid:
                continue
            after = cut[c2["idx"] - a["idx"] + 1:]
            broken_idx = c2["idx"] + 1 + int(np.argmax(after)) if after.any() else None
            out.append({"type": kind, "idx1": a["idx"], "p1": float(a["price"]), "idx2": c2["idx"],
                        "p2": float(c2["price"]), "slope": float(slope),
                        "price_now": float(a["price"] + slope * (b.n - 1 - a["idx"])),
                        "broken": broken_idx is not None, "broken_idx": broken_idx})
            break
    return out


# ------------------------------------------------------- lower-timeframe triggers
def candle_pattern(b: Bars, i: int, d: int) -> dict | None:
    """Bullish/bearish engulfing or rejection wick on closed candle i."""
    if i < 1 or d == 0:
        return None
    o, h, l, c = b.o[i], b.h[i], b.l[i], b.c[i]
    po, pc = b.o[i - 1], b.c[i - 1]
    body, rng = abs(c - o), max(h - l, 1e-12)
    upper, lower = h - max(o, c), min(o, c) - l
    atr = b.atr[i]
    if d == BULL:
        if c > o and pc < po and c >= po and o <= pc and body >= 0.8 * abs(pc - po) and rng >= 0.4 * atr:
            return {"type": "bullish_engulfing", "label": "Bullish engulfing", "idx": i}
        if lower >= 1.5 * body and lower / rng >= 0.45 and c >= l + 0.55 * rng and rng >= 0.4 * atr:
            return {"type": "bullish_rejection", "label": "Bullish rejection wick", "idx": i}
    else:
        if c < o and pc > po and c <= po and o >= pc and body >= 0.8 * abs(pc - po) and rng >= 0.4 * atr:
            return {"type": "bearish_engulfing", "label": "Bearish engulfing", "idx": i}
        if upper >= 1.5 * body and upper / rng >= 0.45 and c <= h - 0.55 * rng and rng >= 0.4 * atr:
            return {"type": "bearish_rejection", "label": "Bearish rejection wick", "idx": i}
    return None


def mini_bos(b: Bars, i: int, d: int, window: int = 30, recent: int = 3) -> dict | None:
    """Micro break of structure on the entry timeframe: within the last `recent`
    bars price closed beyond the most recent minor swing (lower high for longs,
    higher low for shorts)."""
    lo = max(0, i - window)
    if i - lo < 8:
        return None
    sub = Bars.__new__(Bars)
    sub.n = i + 1 - lo
    sub.h, sub.l, sub.c = b.h[lo:i + 1], b.l[lo:i + 1], b.c[lo:i + 1]
    raw = find_swings(sub, 2, 2)
    typ = BULL if d == BULL else BEAR
    cand = [(k, p) for k, t, p in raw if t == typ and k + 2 <= sub.n - 1]
    if not cand:
        return None
    k, lvl = cand[-1]
    for j in range(max(1, sub.n - recent), sub.n):
        crossed = (sub.c[j] > lvl and sub.c[j - 1] <= lvl) if d == BULL else (sub.c[j] < lvl and sub.c[j - 1] >= lvl)
        if crossed:
            return {"level": float(lvl), "idx": lo + j, "level_idx": lo + k}
    return None


def volume_spike(b: Bars, i: int, mult: float = 1.5, lookback: int = 20) -> dict | None:
    if not b.has_volume or i < lookback:
        return None
    avg = float(np.mean(b.v[i - lookback:i]))
    if avg <= 0:
        return None
    cur = float(max(b.v[i], b.v[i - 1]))
    return {"ratio": cur / avg, "spike": cur >= mult * avg}
