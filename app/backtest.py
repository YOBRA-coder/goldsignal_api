"""
Walk-forward backtest of the exact same rule-set the live engine uses.

No look-ahead:
  * 1H / 4H bars are built from the entry-timeframe data and a higher-timeframe
    bar is only visible once it has fully CLOSED (bar end <= close of the current
    entry candle).
  * Signals are evaluated on closed entry candles only; the trade is entered at
    that candle's close and managed from the NEXT candle on.
  * Once a trade is `breakeven_at_r` R in favor, its stop moves to entry (see run_backtest) - a
    losing candle after that scores a "breakeven" scratch (r=0), not a full loss.
  * If SL and TP are both touched inside one candle, the winner is approximated by which level the
    candle's open was closer to (see run_backtest) instead of always scoring it a loss.
  * The take-profit itself is not always the blind risk_reward*risk distance - evaluate_entry aims it
    at the nearest real opposing level when one sits closer (see strategy.realistic_target), so a
    winning trade's `r` can be less than risk_reward (never more).
Spread / slippage / commission are not modelled.  The daily/weekly key levels are
not used (they need data the entry frame does not contain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sessions
from .data_fetcher import INTERVAL_SECONDS, resample_ohlc
from .strategy import Cfg, H1Ctx, H4Ctx, MIN_BARS, effective_min_agreement, evaluate_entry
from .structure import Bars


def summarize(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("result") in ("win", "loss", "breakeven")]
    wins = [t for t in closed if t["result"] == "win"]
    losses = [t for t in closed if t["result"] == "loss"]
    breakevens = [t for t in closed if t["result"] == "breakeven"]
    net = sum(t["r"] for t in closed)
    gp, gl = sum(t["r"] for t in wins), -sum(t["r"] for t in losses)
    eq, peak, mdd, cur = [0.0], 0.0, 0.0, 0.0
    best_streak = worst_streak = streak = 0
    for t in closed:
        cur += t["r"]
        eq.append(cur)
        peak = max(peak, cur)
        mdd = max(mdd, peak - cur)
        streak = (streak + 1 if streak >= 0 else 1) if t["result"] == "win" else (streak - 1 if streak <= 0 else -1) \
            if t["result"] == "loss" else streak  # a breakeven scratch doesn't extend or break a streak
        best_streak, worst_streak = max(best_streak, streak), min(worst_streak, streak)
    by_session: dict[str, dict] = {}
    for t in closed:
        s = by_session.setdefault(t.get("session", "?"), {"trades": 0, "wins": 0, "net_r": 0.0})
        s["trades"] += 1
        s["wins"] += t["result"] == "win"
        s["net_r"] = round(s["net_r"] + t["r"], 2)
    by_dir = {}
    for t in closed:
        s = by_dir.setdefault(t["direction"], {"trades": 0, "wins": 0, "net_r": 0.0})
        s["trades"] += 1
        s["wins"] += t["result"] == "win"
        s["net_r"] = round(s["net_r"] + t["r"], 2)
    n = len(closed)
    return {
        "total_trades": n, "wins": len(wins), "losses": len(losses), "breakevens": len(breakevens),
        "win_rate": round(100 * len(wins) / n, 2) if n else 0.0,
        "net_r": round(net, 2), "avg_r": round(net / n, 3) if n else 0.0,
        "profit_factor": round(gp / gl, 2) if gl > 0 else (None if not gp else 999.0),
        "max_drawdown_r": round(mdd, 2), "best_streak": best_streak, "worst_streak": worst_streak,
        "equity_curve": [round(x, 3) for x in eq], "by_session": by_session, "by_direction": by_dir,
    }


def _resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    return (df.resample("1h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna(subset=["Open", "Close"]))


def run_backtest(df_entry: pd.DataFrame, entry_interval: str = "15m", risk_reward: float = 2.0,
                 min_agreement: float = 70.0, sessions_only: bool = True, symbol: str = "GC=F",
                 df_1h: pd.DataFrame | None = None, breakeven_at_r: float = 1.0) -> dict:
    """df_1h: real hourly history (e.g. 180 days) used to build the 1H/4H context, exactly like the live
    engine.  Without it the 1H/4H bars are derived from the entry candles (only ~59 days of context).
    breakeven_at_r: once a trade is this many R in favor, its stop is moved to entry (a scratch instead
    of a full loss if price comes back) - set to 0 to disable and trade with a static stop."""
    eff_min = effective_min_agreement(min_agreement, risk_reward, INTERVAL_SECONDS[entry_interval])
    cfg = Cfg(rr=risk_reward, min_agreement=eff_min, sessions_only=sessions_only,
              entry_seconds=INTERVAL_SECONDS[entry_interval], entry_interval=entry_interval,
              breakeven_at_r=breakeven_at_r)
    df1 = df_1h if df_1h is not None else _resample_1h(df_entry)
    df4 = resample_ohlc(df1, "4h", symbol)
    end1 = df1.index.as_unit("s").asi8 + 3600
    end4 = df4.index.as_unit("s").asi8 + 14400
    be = Bars(df_entry)
    n = be.n
    sess = sessions.allowed_mask(df_entry.index)

    trades: list[dict] = []
    open_t = None
    c1 = c4 = -1
    h1 = h4 = None
    for i in range(30, n):
        T = int(be.t[i]) + cfg.entry_seconds  # close time of candle i
        if open_t is not None:
            hi, lo = be.h[i], be.l[i]
            up = open_t["direction"] == "BUY"
            # ---- protect a winner: move the stop to breakeven once price is far enough in our favor
            if breakeven_at_r > 0 and not open_t["be_moved"]:
                fav = (hi - open_t["entry"]) if up else (open_t["entry"] - lo)
                if fav / max(open_t["risk0"], 1e-9) >= breakeven_at_r:
                    open_t["sl"] = open_t["entry"]
                    open_t["be_moved"] = True
            hit_tp = hi >= open_t["tp"] if up else lo <= open_t["tp"]
            hit_sl = lo <= open_t["sl"] if up else hi >= open_t["sl"]
            if hit_tp or hit_sl:
                if hit_tp and hit_sl:
                    # Both levels traded inside one candle - we don't actually know which came
                    # first. Scoring this as an automatic loss (the old behaviour) is a pessimistic
                    # bias that inflates the loss count on every timeframe, worst on fast ones where
                    # SL/TP are close together and this case is common. Approximate instead: whichever
                    # level the candle's open sat closer to needed less distance to be touched first.
                    win = abs(be.o[i] - open_t["tp"]) <= abs(be.o[i] - open_t["sl"])
                else:
                    win = hit_tp
                if win:
                    result, r = "win", open_t["reward_r"]
                elif open_t["be_moved"]:
                    result, r = "breakeven", 0.0
                else:
                    result, r = "loss", -1.0
                open_t.update(exit_ts=int(be.t[i]), result=result, r=round(r, 3),
                              exit_price=open_t["tp"] if win else open_t["sl"])
                trades.append(open_t)
                open_t = None
            continue

        k1, k4 = int(np.searchsorted(end1, T, side="right")), int(np.searchsorted(end4, T, side="right"))
        if k1 < MIN_BARS["1h"] or k4 < MIN_BARS["4h"]:
            continue
        if k1 != c1:
            h1, c1 = H1Ctx(df1.iloc[:k1]), k1
        if k4 != c4:
            h4, c4 = H4Ctx(df4.iloc[:k4]), k4
        if h4.trend == 0 or h1.trend != h4.trend:
            continue
        res = evaluate_entry(h4, h1, be, i, cfg, bool(sess[i]), True, [], fast=True)
        if res["direction"] in ("BUY", "SELL"):
            tag = sessions.tag_sessions(df_entry.index[i:i + 1])[0]
            open_t = {"entry_ts": int(be.t[i]), "direction": res["direction"], "entry": res["entry"],
                      "sl": res["stop_loss"], "tp": res["take_profit"], "agreement": res["agreement"],
                      "grade": res["grade"], "session": tag, "pattern": (res["pattern"] or {}).get("type"),
                      "zone": (res["zone"] or {}).get("source"), "risk0": res["risk"],
                      "reward_r": res.get("reward_r") or risk_reward, "be_moved": False}
    if open_t is not None:
        open_t["result"] = "open"
        trades.append(open_t)

    out = summarize(trades)
    out["trades"] = trades
    out["equity_ts"] = [int(be.t[30])] + [t["exit_ts"] for t in trades if t.get("result") in ("win", "loss", "breakeven")]
    out["bars"] = n
    out["htf_source"] = "real 1H history" if df_1h is not None else "derived from entry candles"
    out["min_agreement_requested"] = min_agreement
    out["min_agreement_effective"] = round(eff_min, 1)
    out["breakeven_at_r"] = breakeven_at_r
    out["from_ts"], out["to_ts"] = int(be.t[0]), int(be.t[-1])
    return out
