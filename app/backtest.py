"""
Walk-forward backtest of the exact same rule-set the live engine uses.

No look-ahead:
  * 1H / 4H bars are built from the entry-timeframe data and a higher-timeframe
    bar is only visible once it has fully CLOSED (bar end <= close of the current
    entry candle).
  * Signals are evaluated on closed entry candles only; the trade is entered at
    that candle's close and managed from the NEXT candle on.
  * If SL and TP are both touched inside one candle the trade is scored as a loss.
Spread / slippage / commission are not modelled.  The daily/weekly key levels are
not used (they need data the entry frame does not contain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sessions
from .data_fetcher import INTERVAL_SECONDS, resample_ohlc
from .strategy import Cfg, H1Ctx, H4Ctx, MIN_BARS, evaluate_entry
from .structure import Bars


def summarize(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("result") in ("win", "loss")]
    wins = [t for t in closed if t["result"] == "win"]
    losses = [t for t in closed if t["result"] == "loss"]
    net = sum(t["r"] for t in closed)
    gp, gl = sum(t["r"] for t in wins), -sum(t["r"] for t in losses)
    eq, peak, mdd, cur = [0.0], 0.0, 0.0, 0.0
    best_streak = worst_streak = streak = 0
    for t in closed:
        cur += t["r"]
        eq.append(cur)
        peak = max(peak, cur)
        mdd = max(mdd, peak - cur)
        streak = (streak + 1 if streak >= 0 else 1) if t["result"] == "win" else (streak - 1 if streak <= 0 else -1)
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
        "total_trades": n, "wins": len(wins), "losses": len(losses),
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
                 df_1h: pd.DataFrame | None = None) -> dict:
    """df_1h: real hourly history (e.g. 180 days) used to build the 1H/4H context, exactly like the live
    engine.  Without it the 1H/4H bars are derived from the entry candles (only ~59 days of context)."""
    cfg = Cfg(rr=risk_reward, min_agreement=min_agreement, sessions_only=sessions_only,
              entry_seconds=INTERVAL_SECONDS[entry_interval], entry_interval=entry_interval)
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
            hit_tp = hi >= open_t["tp"] if up else lo <= open_t["tp"]
            hit_sl = lo <= open_t["sl"] if up else hi >= open_t["sl"]
            if hit_tp or hit_sl:
                win = hit_tp and not hit_sl
                open_t.update(exit_ts=int(be.t[i]), result="win" if win else "loss",
                              r=round(risk_reward if win else -1.0, 3), exit_price=open_t["tp"] if win else open_t["sl"])
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
                      "zone": (res["zone"] or {}).get("source")}
    if open_t is not None:
        open_t["result"] = "open"
        trades.append(open_t)

    out = summarize(trades)
    out["trades"] = trades
    out["equity_ts"] = [int(be.t[30])] + [t["exit_ts"] for t in trades if t.get("result") in ("win", "loss")]
    out["bars"] = n
    out["htf_source"] = "real 1H history" if df_1h is not None else "derived from entry candles"
    out["from_ts"], out["to_ts"] = int(be.t[0]), int(be.t[-1])
    return out
