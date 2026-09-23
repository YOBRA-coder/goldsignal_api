"""
Top-down strategy engine.

  STEP 1  4H  set the direction     -> market structure bias (BOS/CHoCH, HH/HL vs LH/LL)
  STEP 2  4H  mark the map          -> fresh supply above / demand below, last BOS/CHoCH,
                                        recent swing H/L, PDH/PDL/PWH/PWL, round numbers
  STEP 3  1H  find the setup        -> structure must ALIGN with 4H (else no bias, no trade);
                                        refined order block / FVG, liquidity, trend lines
  STEP 4  5m/15m pull the trigger   -> price inside the 1H OB/FVG + engulfing or rejection wick,
                                        mini BOS, volume spike, London/New York session only

Every rule is a "check" with a weight.  Some checks are GATES (mandatory).  A signal
fires only when every gate passes AND the weighted agreement >= min_agreement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .structure import (BULL, BEAR, Bars, _pattern_from_ohlc, analyze_structure, base_zones, bias_name,
                        candle_pattern, find_fvgs, find_swings, liquidity, merge_zones, mini_bos, num,
                        trendlines, volume_spike, zigzag, zones_from_events)

MIN_BARS = {"4h": 40, "1h": 80, "entry": 60}

# Reference timeframe every "how many bars back should I look" constant below was originally tuned
# against: 15m.  Below is 900s (15m) any check window would cover LESS real market time than it did
# when the app was tuned on 15m/30m, so the same bar-count window is far noisier -> more false
# triggers -> more losing entries.  We scale bar-count windows UP for faster entry timeframes so they
# cover the same real-world time as they always have on 15m/30m, which are left untouched.
_REF_SECONDS = 900  # 15m


def _tf_scale(entry_seconds: int, bars: int) -> int:
    """Stretch a bar-count window so faster-than-15m entry timeframes (5m, 1m) cover the same amount
    of real time as they do on 15m. No-op for 15m and slower (returns `bars` unchanged)."""
    if entry_seconds >= _REF_SECONDS:
        return bars
    return max(bars, round(bars * _REF_SECONDS / entry_seconds))


def effective_min_agreement(base: float, rr: float, entry_seconds: int = _REF_SECONDS) -> float:
    """Scale the required agreement with the reward you are asking for, so signal frequency tracks
    risk:reward instead of staying fixed: a 1:1 scalp target is easy to reach, so more setups qualify;
    a 1:4 swing target needs much more room to work, so only the cleanest setups should fire.
    rr=2 (the default) leaves `base` untouched; +/-6 points of agreement per point of R away from 2,
    clamped to a sane 35-92% band either way.

    Faster entry timeframes see more price noise per check, so they get a small extra agreement
    requirement on top of that (+3 pts below 15m) to keep the same effective setup quality. This is
    intentionally small: the confluence gate and tighter candle-pattern thresholds (see
    structure._pattern_from_ohlc) already do the real noise filtering on 5m, and 5m has far more
    candles per day to work with than 15m/30m - stacking a big extra agreement penalty on top would
    mostly just throw away good setups rather than catch more bad ones."""
    out = base + (rr - 2.0) * 6.0
    if entry_seconds < _REF_SECONDS:
        out += 3.0
    return max(35.0, min(92.0, out))


@dataclass
class Cfg:
    rr: float = 2.0
    min_agreement: float = 70.0
    sessions_only: bool = True
    entry_seconds: int = 900
    entry_interval: str = "15m"
    breakeven_at_r: float = 1.0  # once a trade is +this many R in favor, move the stop to entry


# ------------------------------------------------------------------- contexts
class H4Ctx:
    """4H analysis (direction + map)."""

    def __init__(self, df4: pd.DataFrame):
        self.b = Bars(df4.tail(350))
        self.st = analyze_structure(self.b, 3, 3)
        self.zones = merge_zones(zones_from_events(self.b, self.st["events"]) + base_zones(self.b))
        self.fvgs = find_fvgs(self.b, 120)
        self.pools, self.sweeps = liquidity(self.b, self.st["swings"])
        self.tl = trendlines(self.b, self.st["swings"])

    @property
    def trend(self) -> int:
        return self.st["trend"]


class H1Ctx:
    """1H analysis (structure + liquidity + refined OB/FVG)."""

    def __init__(self, df1: pd.DataFrame):
        self.b = Bars(df1.tail(700))
        self.st = analyze_structure(self.b, 2, 2)
        self.zones = zones_from_events(self.b, self.st["events"], min_leg_atr=1.0)
        self.fvgs = find_fvgs(self.b, 200)
        self.pools, self.sweeps = liquidity(self.b, self.st["swings"])
        self.tl = trendlines(self.b, self.st["swings"])

    @property
    def trend(self) -> int:
        return self.st["trend"]


class TfCtx:
    """Structure + zones + liquidity for ANY timeframe (drawn on the chart of that timeframe)."""

    def __init__(self, df: pd.DataFrame, left: int = 2, right: int = 2, tail: int = 500):
        self.b = Bars(df.tail(tail))
        self.st = analyze_structure(self.b, left, right)
        self.zones = merge_zones(zones_from_events(self.b, self.st["events"]) + base_zones(self.b))
        self.fvgs = find_fvgs(self.b, 200)
        self.pools, self.sweeps = liquidity(self.b, self.st["swings"])
        self.tl = trendlines(self.b, self.st["swings"])


# ------------------------------------------------------------------ json helpers
def zone_json(b: Bars, z: dict) -> dict:
    return {"type": z["type"], "source": z["source"], "top": num(z["top"]), "bottom": num(z["bottom"]),
            "ts": int(b.t[z["idx"]]), "formed_ts": int(b.t[min(z["formed_idx"], b.n - 1)]),
            "end_ts": int(b.t[z["end_idx"]]) if z.get("end_idx") is not None else None,
            "state": z["state"], "touches": z.get("touches", 0), "strength": num(z["strength"]),
            "event": z.get("event")}


def swing_json(b: Bars, s: dict) -> dict:
    return {"ts": int(b.t[s["idx"]]), "price": num(s["price"]), "type": "high" if s["type"] == BULL else "low",
            "label": s.get("label")}


def event_json(b: Bars, e: dict) -> dict:
    return {"type": e["type"], "dir": "bull" if e["dir"] == BULL else "bear", "level": num(e["level"]),
            "level_ts": int(b.t[e["level_idx"]]), "ts": int(b.t[e["idx"]])}


def pool_json(b: Bars, p: dict) -> dict:
    return {"type": p["type"], "price": num(p["price"]), "ts": int(b.t[p["idx"]]), "ts2": int(b.t[p["idx2"]]),
            "count": p["count"]}


def sweep_json(b: Bars, s: dict) -> dict:
    return {"type": s["type"], "dir": "bull" if s["dir"] == BULL else "bear", "level": num(s["level"]),
            "level_ts": int(b.t[s["level_idx"]]), "ts": int(b.t[s["idx"]]), "wick": num(s["wick"])}


def tl_json(b: Bars, t: dict) -> dict:
    ts1, ts2 = int(b.t[t["idx1"]]), int(b.t[t["idx2"]])
    slope_s = (t["p2"] - t["p1"]) / max(1, ts2 - ts1)
    return {"type": t["type"], "ts1": ts1, "p1": num(t["p1"]), "ts2": ts2, "p2": num(t["p2"]),
            "slope_per_sec": slope_s, "price_now": num(t["price_now"]), "broken": t["broken"],
            "broken_ts": int(b.t[t["broken_idx"]]) if t["broken_idx"] is not None else None}


def select_zones(b: Bars, zones: list[dict], price: float, max_n: int = 8, horizon: int = 260) -> list[dict]:
    """Relevant, still-valid zones: not broken, not ancient, nearest to price first."""
    live = [z for z in zones if z["state"] != "broken" and z["formed_idx"] >= b.n - horizon]
    atr = max(b.atr[-1], 1e-9)
    live.sort(key=lambda z: (0 if z["state"] == "fresh" else 1, abs((z["top"] + z["bottom"]) / 2 - price) / atr))
    return live[:max_n]


def nearest_key_zones(h4: H4Ctx, price: float) -> dict:
    fresh = [z for z in h4.zones if z["state"] == "fresh"]
    sup = [z for z in fresh if z["type"] == "supply" and z["bottom"] > price]
    dem = [z for z in fresh if z["type"] == "demand" and z["top"] < price]
    s = min(sup, key=lambda z: z["bottom"] - price) if sup else None
    d = max(dem, key=lambda z: z["top"]) if dem else None
    return {"supply_above": zone_json(h4.b, s) if s else None, "demand_below": zone_json(h4.b, d) if d else None}


def tf_json(ctx, price: float, name: str) -> dict:
    b, st = ctx.b, ctx.st
    lh, ll = st["last_high"], st["last_low"]
    eq = st["eq"]
    out = {
        "tf": name, "bias": bias_name(st["trend"]), "run": st["run"],
        "swings": [swing_json(b, s) for s in st["swings"][-14:]],
        "events": [event_json(b, e) for e in st["events"][-8:]],
        "last_event": event_json(b, st["events"][-1]) if st["events"] else None,
        "last_high": swing_json(b, lh) if lh else None, "last_low": swing_json(b, ll) if ll else None,
        "equilibrium": num(eq),
        "location": None if eq is None else ("premium" if price > eq else "discount"),
        "zones": [zone_json(b, z) for z in select_zones(b, ctx.zones, price)],
        "fvgs": [zone_json(b, z) for z in select_zones(b, ctx.fvgs, price, 6, 120)],
        "pools": [pool_json(b, p) for p in sorted(ctx.pools, key=lambda p: abs(p["price"] - price))[:8]],
        "sweeps": [sweep_json(b, s) for s in ctx.sweeps[-5:]],
        "trendlines": [tl_json(b, t) for t in ctx.tl],
        "atr": num(b.atr[-1]),
    }
    if name == "4h":
        out["key_zones"] = nearest_key_zones(ctx, price)
    return out


# ---------------------------------------------------------------- entry evaluation
def _add(checks: list, key, label, step, ok, detail, weight, gate=False, tf=None):
    checks.append({"key": key, "label": label, "step": step, "tf": tf, "ok": ok, "detail": detail,
                   "weight": weight, "gate": gate})


def realistic_target(h1: "H1Ctx", h4: "H4Ctx", d: int, price: float, risk: float, rr: float,
                     levels: list[dict]) -> tuple[float, str]:
    """Aim the take-profit at the nearest REAL opposing level ahead of price (1H zone/pool first, then
    4H zone/pool, then a daily/weekly key level) instead of a blind R-multiple.

    A blind rr*risk target can sit well past real resistance/support (price statistically won't reach
    it) or well short of it (the setup gets marked a 'loss' by the backtest even though it did
    everything right and stalled at a real wall a hair under the arbitrary number). This never asks for
    MORE than the requested rr*risk - only for less, and only when a real level sits closer than 1R
    away is it worth shrinking to (a level inside 1R means the setup barely had room to begin with)."""
    blind = rr * risk
    candidates: list[tuple[float, str]] = []
    for ctx, tag in ((h1, "1H"), (h4, "4H")):
        for z in ctx.zones:
            if z["state"] == "broken":
                continue
            if d == BULL and z["type"] == "supply" and z["bottom"] > price:
                candidates.append((z["bottom"] - price, f"{tag} supply {z['bottom']:.2f}"))
            if d == BEAR and z["type"] == "demand" and z["top"] < price:
                candidates.append((price - z["top"], f"{tag} demand {z['top']:.2f}"))
        for p in ctx.pools:
            if d == BULL and p["type"] in ("EQH", "BSL") and p["price"] > price:
                candidates.append((p["price"] - price, f"{tag} {p['type']} {p['price']:.2f}"))
            if d == BEAR and p["type"] in ("EQL", "SSL") and p["price"] < price:
                candidates.append((price - p["price"], f"{tag} {p['type']} {p['price']:.2f}"))
    for lv in levels:
        if lv.get("kind") in ("daily", "weekly"):
            if d == BULL and lv["price"] > price:
                candidates.append((lv["price"] - price, lv["label"]))
            if d == BEAR and lv["price"] < price:
                candidates.append((price - lv["price"], lv["label"]))
    viable = [(dist, name) for dist, name in candidates if dist >= 1.0 * risk]
    if not viable:
        return blind, "blind R-multiple (no real level in range)"
    dist, name = min(viable, key=lambda x: x[0])
    return min(dist, blind), name


def _sub_bars(b: Bars, lo: int, hi: int) -> Bars:
    s = Bars.__new__(Bars)
    s.n = hi - lo
    s.t, s.o, s.h, s.l, s.c, s.v = (a[lo:hi] for a in (b.t, b.o, b.h, b.l, b.c, b.v))
    s.atr, s.has_volume = b.atr[lo:hi], b.has_volume
    return s


def evaluate_entry(h4: H4Ctx, h1: H1Ctx, be: Bars, i: int, cfg: Cfg, sess_ok: bool, market_ok: bool = True,
                   levels: list | None = None, fast: bool = False) -> dict:
    """Evaluate the whole rule-set on closed entry-TF candle `i` (uses data <= i only)."""
    levels = levels or []
    d = h4.trend
    bias4, bias1 = bias_name(d), bias_name(h1.trend)
    price, t = float(be.c[i]), int(be.t[i])
    a_e, a_1, a_4 = float(be.atr[i]), float(h1.b.atr[-1]), float(h4.b.atr[-1])
    checks: list[dict] = []
    tf_e = cfg.entry_interval
    want = "demand" if d == BULL else "supply"
    sign = "bullish" if d == BULL else "bearish"

    # ---- STEP 1: 4H direction
    bias_ok = d != 0
    _add(checks, "bias_4h", "4H structure sets direction", 1, bias_ok,
         f"4H is {bias4.upper()}" + (f" ({'HH/HL' if d == BULL else 'LH/LL'}, {h4.st['run']} break(s) in a row)" if bias_ok
                                     else " - no clear structure, stand aside"), 2.0, True, "4H")

    # ---- STEP 3: 1H alignment
    align_ok = bias_ok and h1.trend == d
    _add(checks, "align_1h", "1H structure aligns with 4H", 3, align_ok,
         f"1H is {bias1.upper()} vs 4H {bias4.upper()}" + ("" if align_ok else " - not aligned = no bias, no trade"),
         2.0, True, "1H")

    # ---- zone candidates on 1H (refined OB / FVG of the right type)
    cands = []
    if bias_ok:
        for z in h1.zones + h1.fvgs:
            if z["type"] == want and z["state"] != "broken" and z["formed_idx"] >= h1.b.n - 220 \
                    and h1.b.t[min(z["formed_idx"], h1.b.n - 1)] <= t:
                cands.append(z)
    tapped, tap_idx = None, None
    lo_i = max(0, i - _tf_scale(cfg.entry_seconds, 5))
    for z in sorted(cands, key=lambda q: (0 if q["source"] == "OB" else 1, -q["strength"])):
        if d == BULL:
            hit = (be.l[lo_i:i + 1] <= z["top"]) & (be.h[lo_i:i + 1] >= z["bottom"])
            valid = be.c[i] >= z["bottom"]
        else:
            hit = (be.h[lo_i:i + 1] >= z["bottom"]) & (be.l[lo_i:i + 1] <= z["top"])
            valid = be.c[i] <= z["top"]
        if hit.any() and valid:
            tapped, tap_idx = z, lo_i + int(np.argmax(hit))
            break
    zone_ok = tapped is not None
    if zone_ok:
        zdet = f"Price tapped 1H {'demand' if d == BULL else 'supply'} {tapped['source']} " \
               f"{tapped['bottom']:.2f}-{tapped['top']:.2f} ({tapped['state']})"
    elif cands:
        near = min(cands, key=lambda z: abs((z["top"] + z["bottom"]) / 2 - price))
        zdet = f"Waiting for pullback into 1H {near['source']} {near['bottom']:.2f}-{near['top']:.2f}"
    else:
        zdet = "No valid 1H OB/FVG of the right type yet"
    _add(checks, "zone_1h", "Price inside 1H refined OB / FVG", 3, zone_ok, zdet, 2.0, True, "1H")

    # ---- trigger candle on the entry timeframe
    pattern = candle_pattern(be, i, d) if bias_ok else None
    pat_ok = pattern is not None and zone_ok
    _add(checks, "candle", f"{tf_e} engulfing / rejection wick at the zone", 4, pat_ok,
         (pattern["label"] if pat_ok else f"{pattern['label']} - ignored, not at a 1H zone") if pattern
         else "No confirmation candle on the last closed bar", 2.0, True, tf_e)

    # ---- trigger must NOT be standing alone: at least one of mini-BOS / volume spike has to back it
    # up, or it's too easily just noise on the entry timeframe (computed once here, reused below so we
    # don't run these twice).
    mb_window = _tf_scale(cfg.entry_seconds, 30)
    mb_recent = _tf_scale(cfg.entry_seconds, 3)
    bos = mini_bos(be, i, d, window=mb_window, recent=mb_recent) if bias_ok and zone_ok else None
    vs = volume_spike(be, i, lookback=_tf_scale(cfg.entry_seconds, 20))
    confluence_ok = bool(pat_ok and (bos is not None or (vs and vs["spike"])))
    conf_bits = [n for n, ok in (("mini BOS", bos is not None), ("volume spike", bool(vs and vs["spike"]))) if ok]
    _add(checks, "confluence", "Mini BOS or volume spike backs up the trigger candle", 4,
         confluence_ok if pat_ok else None,
         ("Backed by " + " + ".join(conf_bits) if confluence_ok else
          "Trigger candle stands alone - no mini BOS or volume spike behind it" if pat_ok else
          "No confirmation candle yet"), 0.0, True, tf_e)

    # ---- session gate
    sess_gate = cfg.sessions_only
    sess_pass = sess_ok and market_ok
    _add(checks, "session", "London / New York session", 4, sess_pass if sess_gate else (sess_pass or None),
         "London/NY session active" if sess_pass else "Outside London/New York - no entries", 1.5, sess_gate, tf_e)

    # ---- risk model (needs zone + pattern)
    entry = sl = tp = risk = None
    risk_ok = False
    tp_reason = None
    if zone_ok:
        # Below 15m, a fixed-ATR stop that's fine on 15m/30m sits so close to price that ordinary
        # noise (and real-world spread) stops the trade out before the 4H/1H thesis has had room to
        # play out. Widen the buffer/floor for faster entry timeframes; 15m/30m are unchanged.
        fast_tf = cfg.entry_seconds < _REF_SECONDS
        pad_mult, floor_mult = (0.4, 1.0) if fast_tf else (0.25, 0.6)
        seg = slice(max(0, tap_idx - 1), i + 1)
        if d == BULL:
            sl = min(tapped["bottom"], float(be.l[seg].min())) - pad_mult * a_e
            risk = max(price - sl, floor_mult * a_e)
            sl = price - risk
            tgt_dist, tp_reason = realistic_target(h1, h4, d, price, risk, cfg.rr, levels)
            tp = price + tgt_dist
        else:
            sl = max(tapped["top"], float(be.h[seg].max())) + pad_mult * a_e
            risk = max(sl - price, floor_mult * a_e)
            sl = price + risk
            tgt_dist, tp_reason = realistic_target(h1, h4, d, price, risk, cfg.rr, levels)
            tp = price - tgt_dist
        entry = price
        risk_ok = risk <= 3.0 * a_1
    _add(checks, "risk", "Stop distance is sensible (<= 3x 1H ATR)", 4, risk_ok if zone_ok else False,
         (f"Risk {risk:.2f} pts vs 1H ATR {a_1:.2f}" if zone_ok else "Pending zone tap"), 0.0, True, tf_e)

    # ---- protect a winner: at this many R in favor, move the stop to breakeven (informational for
    # live signals - there's no order execution here - and actually enforced in the backtest engine)
    breakeven_price = None
    if zone_ok and risk:
        breakeven_price = price + cfg.breakeven_at_r * risk if d == BULL else price - cfg.breakeven_at_r * risk

    # ---- early preview: 4H direction is already set but 1H hasn't confirmed it yet. The zone/pattern/
    # risk work above only ever depended on bias_ok (4H), never on align_ok (1H) - so if 1H is just
    # lagging, we already know exactly what this timeframe's setup looks like. Never used to fire a
    # trade (align_1h is still a hard gate below) - purely a heads-up on what's forming.
    h1_preview = None
    if bias_ok and not align_ok:
        h1_preview = {
            "would_be_direction": "BUY" if d == BULL else "SELL",
            "zone_tapped": zone_ok, "trigger_ready": pat_ok,
            "zone": zone_json(h1.b, tapped) if tapped else None,
            "pattern": pattern["label"] if pattern else None,
            "entry": entry, "stop_loss": sl, "take_profit": tp, "tp_reason": tp_reason,
            "note": (f"4H is {bias4} and the {tf_e} setup is "
                     f"{'ready to fire' if pat_ok else 'tapped, waiting on a trigger candle' if zone_ok else 'not tapped yet'} "
                     f"- still needs 1H to flip {bias4.lower()} (currently {bias1.lower()}) before anything can fire"),
        }

    gates_ok = all(c["ok"] for c in checks if c["gate"])
    if fast and not gates_ok:
        return {"direction": "WAIT", "gates_ok": False}

    # ---- STEP 2: 4H map (soft checks)
    price_zone = None
    if bias_ok:
        near_zone = [z for z in h4.zones if z["type"] == want and z["state"] != "broken"
                     and z["bottom"] - 0.5 * a_4 <= price <= z["top"] + 0.5 * a_4]
        eq4 = h4.st["eq"]
        good_side = eq4 is not None and (price < eq4 if d == BULL else price > eq4)
        price_zone = near_zone[0] if near_zone else None
        _add(checks, "htf_location", "Price at 4H fresh zone / correct side of 4H range", 2, bool(near_zone) or good_side,
             (f"Inside 4H {want} zone {price_zone['bottom']:.2f}-{price_zone['top']:.2f}" if near_zone else
              ("In 4H discount" if d == BULL and good_side else "In 4H premium" if good_side else
               ("In 4H premium - poor long location" if d == BULL else "In 4H discount - poor short location"))),
             1.5, False, "4H")
        targets = []
        for z in h4.zones:
            if z["state"] != "broken":
                if d == BULL and z["type"] == "supply" and z["bottom"] > price:
                    targets.append(("4H supply", z["bottom"]))
                if d == BEAR and z["type"] == "demand" and z["top"] < price:
                    targets.append(("4H demand", z["top"]))
        for lv in levels:
            if lv["kind"] in ("daily", "weekly"):
                if d == BULL and lv["price"] > price:
                    targets.append((lv["label"], lv["price"]))
                if d == BEAR and lv["price"] < price:
                    targets.append((lv["label"], lv["price"]))
        for p in h4.pools:
            if d == BULL and p["type"] in ("EQH", "BSL") and p["price"] > price:
                targets.append(("4H " + p["type"], p["price"]))
            if d == BEAR and p["type"] in ("EQL", "SSL") and p["price"] < price:
                targets.append(("4H " + p["type"], p["price"]))
        if risk and targets:
            name, lvl = min(targets, key=lambda q: abs(q[1] - price))
            room = abs(lvl - price) / risk
            _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, room >= 2.0,
                 f"Next obstacle {name} at {lvl:.2f} = {room:.1f}R away", 2.0, False, "4H")
        elif risk:
            _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, True, "Clear path - no obstacle found", 2.0, False, "4H")
        else:
            _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, False, "Pending zone tap", 2.0, False, "4H")
    else:
        _add(checks, "htf_location", "Price at 4H fresh zone / correct side of 4H range", 2, False, "No 4H bias", 1.5, False, "4H")
        _add(checks, "htf_room", "Room to next 4H obstacle (>= 2R)", 2, False, "No 4H bias", 2.0, False, "4H")

    # ---- STEP 3 soft checks
    n1 = h1.b.n
    last_ev = h1.st["events"][-1] if h1.st["events"] else None
    brk_ok = bool(last_ev and last_ev["dir"] == d and last_ev["idx"] >= n1 - 40)
    _add(checks, "h1_break", "Recent 1H BOS/CHoCH in the 4H direction", 3, brk_ok,
         (f"Last 1H {last_ev['type']} {'up' if last_ev['dir'] == BULL else 'down'} {n1 - 1 - last_ev['idx']} bars ago"
          if last_ev else "No 1H structure break"), 1.0, False, "1H")

    sw1 = [s for s in h1.sweeps if s["dir"] == d and s["idx"] >= n1 - 24]
    lo = max(0, i - 60)
    sb = _sub_bars(be, lo, i + 1)
    _, esw = liquidity(sb, zigzag(find_swings(sb, 2, 2), 2))
    esw = [s for s in esw if s["dir"] == d and s["idx"] >= sb.n - 12]
    sweep_e = esw[-1] if esw else None
    sweep_e_ts = int(sb.t[sweep_e["idx"]]) if sweep_e else None
    liq_ok = bool(sw1) or sweep_e is not None
    _add(checks, "liquidity", "Liquidity sweep before the reaction", 3, liq_ok,
         (f"{'Sell-side' if d == BULL else 'Buy-side'} liquidity swept on {'1H' if sw1 else tf_e}" if liq_ok
          else "No aligned liquidity sweep in recent bars"), 1.5, False, "1H")

    tl_ok, tl_txt = False, "No valid trend line"
    for tl in h1.tl:
        if d == BULL and tl["type"] == "up" and not tl["broken"] and abs(price - tl["price_now"]) <= 1.0 * a_1:
            tl_ok, tl_txt = True, f"Holding 1H ascending trend line ~{tl['price_now']:.2f}"
        if d == BULL and tl["type"] == "down" and tl["broken"] and tl["broken_idx"] >= n1 - 30:
            tl_ok, tl_txt = True, "Broke 1H descending trend line"
        if d == BEAR and tl["type"] == "down" and not tl["broken"] and abs(price - tl["price_now"]) <= 1.0 * a_1:
            tl_ok, tl_txt = True, f"Rejecting 1H descending trend line ~{tl['price_now']:.2f}"
        if d == BEAR and tl["type"] == "up" and tl["broken"] and tl["broken_idx"] >= n1 - 30:
            tl_ok, tl_txt = True, "Broke 1H ascending trend line"
    _add(checks, "trendline", "1H trend line support / break", 3, tl_ok, tl_txt, 1.0, False, "1H")

    eq1 = h1.st["eq"]
    pd_ok = bool(eq1 is not None and bias_ok and (price < eq1 if d == BULL else price > eq1))
    _add(checks, "pd_1h", "Discount (buys) / premium (sells) of 1H range", 3, pd_ok,
         ("In 1H discount" if d == BULL else "In 1H premium") if pd_ok else "On the wrong side of the 1H range", 1.0, False, "1H")

    # ---- STEP 4 soft checks (bos/vs already computed above, alongside the confluence gate)
    _add(checks, "mini_bos", f"Mini BOS on {tf_e}", 4, bos is not None,
         f"Closed beyond minor swing {bos['level']:.2f}" if bos else "No micro break of structure yet", 2.0, False, tf_e)
    _add(checks, "volume", "Volume spike on entry", 4, None if vs is None else vs["spike"],
         "No volume data for this symbol" if vs is None else f"Volume {vs['ratio']:.1f}x the 20-bar average",
         1.5, False, tf_e)

    applicable = [c for c in checks if c["ok"] is not None and c["weight"] > 0]
    tot = sum(c["weight"] for c in applicable)
    got = sum(c["weight"] for c in applicable if c["ok"])
    agreement = 100.0 * got / tot if tot else 0.0
    grade = "A+" if agreement >= 85 else "A" if agreement >= 70 else "B" if agreement >= 55 else "C"

    gates_ok = all(c["ok"] for c in checks if c["gate"])
    fired = gates_ok and agreement >= cfg.min_agreement

    # ---- status / headline
    if not market_ok:
        status, head = "market_closed", "Market closed - no signals until it reopens"
    elif not bias_ok:
        status, head = "no_bias", "No clear 4H direction - stand aside"
    elif not align_ok:
        status = "not_aligned"
        head = f"1H ({bias1}) fights 4H ({bias4}) - no setup"
        if pat_ok:
            head += f" (but the {tf_e} setup is already ready - see preview)"
    elif not zone_ok:
        status, head = "waiting_zone", zdet
    elif not pat_ok:
        status, head = "waiting_trigger", f"Inside 1H zone - waiting for {sign} engulfing / rejection on {tf_e}"
    elif sess_gate and not sess_pass:
        status, head = "out_of_session", "Setup ready but outside London / New York"
    elif not risk_ok:
        status, head = "risk_too_wide", "Setup found but the stop would be too wide"
    elif not fired:
        status, head = "low_agreement", f"Trigger present but agreement {agreement:.0f}% < {cfg.min_agreement:.0f}% minimum"
    else:
        status = "signal"
        head = f"{'BUY' if d == BULL else 'SELL'} - {pattern['label']} in 1H {tapped['source']}"

    step_ok = [bias_ok, bias_ok, align_ok and zone_ok, pat_ok and (sess_pass or not sess_gate)]
    stage = 0
    for ok in step_ok:
        if not ok:
            break
        stage += 1

    watch = None
    if bias_ok and cands and not zone_ok:
        near = min(cands, key=lambda z: abs((z["top"] + z["bottom"]) / 2 - price))
        watch = {**zone_json(h1.b, near), "distance_atr": abs((near["top"] + near["bottom"]) / 2 - price) / max(a_1, 1e-9)}

    # ---- highlight reel: every pattern / reversal signal actually detected this bar, in plain English
    highlights = []
    if pattern:
        highlights.append(f"{tf_e} {pattern['label']}" + (" at the 1H zone" if zone_ok else " (not at a 1H zone yet)"))
    if bos:
        highlights.append(f"Mini break of structure on {tf_e} through {bos['level']:.2f}")
    if sweep_e:
        highlights.append(f"Liquidity sweep on {tf_e} right before the reaction")
    if sw1:
        highlights.append("1H liquidity sweep backing the move")
    if tl_ok:
        highlights.append(tl_txt)
    if vs and vs.get("spike"):
        highlights.append(f"Volume spike ({vs['ratio']:.1f}x the {tf_e} average)")

    return {
        "direction": ("BUY" if d == BULL else "SELL") if fired else "WAIT",
        "status": status, "headline": head, "stage": stage, "steps_ok": step_ok,
        "bias_4h": bias4, "bias_1h": bias1, "agreement": round(agreement, 1), "grade": grade,
        "gates_ok": gates_ok, "checks": checks, "entry": entry if zone_ok else None,
        "stop_loss": sl if zone_ok else None, "take_profit": tp if zone_ok else None,
        "risk": risk if zone_ok else None, "rr": cfg.rr,
        "tp_reason": tp_reason, "reward_r": round(abs(tp - price) / risk, 2) if (zone_ok and risk) else None,
        "breakeven_at_r": cfg.breakeven_at_r, "breakeven_price": breakeven_price,
        "h1_preview": h1_preview, "highlights": highlights,
        "candle_ts": t, "signal_time": t + cfg.entry_seconds, "candle_idx": i,
        "zone": zone_json(h1.b, tapped) if tapped else None, "watch_zone": watch,
        "pattern": ({**pattern, "ts": t} if pattern else None),
        "mini_bos": ({"level": bos["level"], "ts": int(be.t[bos["idx"]]), "level_ts": int(be.t[bos["level_idx"]])} if bos else None),
        "volume": ({"ratio": vs["ratio"], "spike": vs["spike"], "ts": t} if vs else None),
        "sweep_entry": ({"level": sweep_e["level"], "ts": sweep_e_ts} if sweep_e else None),
        "tap_ts": int(be.t[tap_idx]) if tap_idx is not None else None,
        "session_ok": bool(sess_pass), "price": price,
    }


# -------------------------------------------------------------------- forming-candle preview
def forming_watch(be: Bars, forming: dict | None, latest: dict, d: int) -> dict | None:
    """Early heads-up only: peek at the STILL-OPEN entry-timeframe candle to see whether it is already
    shaping up as the trigger the real signal is waiting on.

    This never fires a signal by itself and never feeds back into evaluate_entry/build_signal - the
    real BUY/SELL only ever comes from a fully closed candle (see evaluate_entry), so there is no
    repainting risk. It only activates when everything else (4H bias, 1H alignment, the 1H zone tap)
    is ALREADY true on closed data and the only thing missing is this candle finishing - i.e. exactly
    the app's own 'waiting_trigger' status. Because the candle can still change shape until it closes,
    this can flip or disappear on the next poll; that's expected and is why it's labelled 'forming'."""
    if forming is None or d == 0 or latest.get("status") != "waiting_trigger":
        return None
    i = be.n - 1
    prev_o, prev_c = float(be.o[i]), float(be.c[i])
    atr = float(be.atr[i])
    pat = _pattern_from_ohlc(prev_o, prev_c, forming["Open"], forming["High"], forming["Low"],
                             forming["Close"], atr, d)
    if pat is None:
        return None
    return {
        "status": "forming", "direction": "BUY" if d == BULL else "SELL",
        "pattern": pat["label"],
        "detail": f"{pat['label']} shaping up on the still-open candle - "
                  f"only confirms if it closes this way, can still change",
        "zone": latest.get("zone"), "would_be_price": float(forming["Close"]),
    }


# -------------------------------------------------------------------- live signal
def build_signal(df4: pd.DataFrame, df1: pd.DataFrame, df_entry: pd.DataFrame, cfg: Cfg,
                 levels: list[dict], sess_mask: np.ndarray, market_ok: bool, lookback: int = 3,
                 forming_bar: dict | None = None) -> tuple[dict, dict]:
    """df_* must contain CLOSED bars only.  Returns (signal, analysis_for_chart).
    forming_bar (optional): {"Open","High","Low","Close"} of the entry-timeframe candle that is
    currently still open (not in df_entry) - used only for the informational 'forming' preview."""
    h4, h1 = H4Ctx(df4), H1Ctx(df1)
    be = Bars(df_entry.tail(400))
    sess = sess_mask[-be.n:]
    n = be.n
    price = float(be.c[-1])
    # How many closed bars back we still treat a trigger as "live" (price hasn't moved more than 1R
    # away yet). A fixed bar count here has the exact same real-time-coverage problem the confirmation
    # windows had: on 5m, 3 bars is only 15 minutes, so plenty of perfectly good, still-live setups
    # were being thrown away right after they fired. Scale it like everything else in this file.
    lookback = _tf_scale(cfg.entry_seconds, lookback)

    best = None
    for k in range(min(lookback, n - 1)):
        i = n - 1 - k
        res = evaluate_entry(h4, h1, be, i, cfg, bool(sess[i]), market_ok, levels)
        if k == 0:
            latest = res
        if res["direction"] in ("BUY", "SELL"):
            d = 1 if res["direction"] == "BUY" else -1
            moved = (price - res["entry"]) * d / max(res["risk"], 1e-9)
            if -1.0 < moved < 1.0:
                best = {**res, "age_bars": k, "moved_r": moved}
                break
    sig = best or {**latest, "age_bars": 0, "moved_r": 0.0}
    sig["last_price"] = price
    sig["confidence"] = sig["agreement"] / 100.0
    sig["bias_4h"] = bias_name(h4.trend)
    sig["entry_price"] = sig.pop("entry", None)
    sig["risk_reward"] = cfg.rr
    sig["reason"] = " | ".join(f"{'OK' if c['ok'] else ('n/a' if c['ok'] is None else 'NO')}: {c['detail']}" for c in sig["checks"])
    sig["forming"] = forming_watch(be, forming_bar, latest, h4.trend)

    analysis = {
        "4h": tf_json(h4, price, "4h"),
        "1h": tf_json(h1, price, "1h"),
        "entry": {
            "tf": cfg.entry_interval,
            "swings": [swing_json(be, s) for s in analyze_structure(be, 2, 2)["swings"][-10:]],
            "pattern": sig.get("pattern"), "mini_bos": sig.get("mini_bos"), "volume": sig.get("volume"),
            "tap_ts": sig.get("tap_ts"), "candle_ts": sig["candle_ts"],
        },
        "levels": levels,
        "trade": {"direction": sig["direction"], "entry": sig["entry_price"], "sl": sig["stop_loss"],
                  "tp": sig["take_profit"], "rr_requested": sig["rr"], "reward_r": sig.get("reward_r"),
                  "tp_reason": sig.get("tp_reason"), "breakeven_price": sig.get("breakeven_price"),
                  "ts": sig["candle_ts"]} if sig["direction"] in ("BUY", "SELL") else None,
        "watch_zone": sig.get("watch_zone"),
        "zone_hit": sig.get("zone"),
        "forming": sig.get("forming"),
        "h1_preview": sig.get("h1_preview"),
        "highlights": sig.get("highlights"),
    }
    return sig, analysis


def own_structure(df: pd.DataFrame, tf: str) -> dict | None:
    """Overlay data for the chart's own timeframe (closed bars only)."""
    if len(df) < 30:
        return None
    k = 3 if tf in ("4h", "1d", "1w") else 2
    ctx = TfCtx(df, k, k)
    return tf_json(ctx, float(ctx.b.c[-1]), tf)


def quick_bias(df: pd.DataFrame, left: int = 2) -> dict:
    """Light structure read used by the multi-timeframe table."""
    if len(df) < 25:
        return {"bias": "n/a", "last_event": None, "labels": []}
    b = Bars(df.tail(300))
    st = analyze_structure(b, left, left)
    ev = st["events"][-1] if st["events"] else None
    return {"bias": bias_name(st["trend"]), "run": st["run"],
            "last_event": event_json(b, ev) if ev else None,
            "labels": [s["label"] for s in st["swings"][-4:] if s.get("label")],
            "price": float(b.c[-1])}
