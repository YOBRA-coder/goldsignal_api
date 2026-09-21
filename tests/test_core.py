"""Run:  GOLDSIGNAL_DEMO=1 pytest -q      (from the backend/ folder)"""
import os
os.environ.setdefault("GOLDSIGNAL_DEMO", "1")

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app import backtest as bt
from app import data_fetcher as f
from app import sessions
from app.structure import Bars, analyze_structure, find_fvgs, zones_from_events


def _frame(closes, spread=1.0):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq="1h", tz="UTC")
    c = np.array(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
                         "Close": c, "Volume": 100.0}, index=idx)


def test_parse_yahoo_chart_json():
    payload = {"chart": {"result": [{"timestamp": [1700000000, 1700003600, 1700007200],
               "indicators": {"quote": [{"open": [1, 2, None], "high": [2, 3, None], "low": [0.5, 1.5, None],
                                         "close": [1.5, 2.5, None], "volume": [10, None, 5]}]}}], "error": None}}
    df = f.parse_chart_json(payload)
    assert len(df) == 2 and str(df.index.tz) == "UTC" and df["Volume"].iloc[1] == 0


def test_market_closed_on_weekend_and_sessions():
    sat = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
    assert not sessions.market_open(sat)
    tue = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    st = sessions.session_state(tue)
    assert st["market_open"] and st["trade_allowed"] and "london" in st["active"] and "ny" in st["active"]
    night = datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc)
    assert not sessions.session_state(night)["trade_allowed"]


def test_bullish_structure_bos_then_choch():
    # up-trend with pullbacks, then a break down
    pts = [100, 110, 104, 116, 108, 122, 114, 128, 118, 105, 98, 90]
    closes = []
    for a, b in zip(pts[:-1], pts[1:]):
        closes += list(np.linspace(a, b, 6, endpoint=False))
    closes.append(pts[-1])
    st = analyze_structure(Bars(_frame(closes)), 2, 2)
    kinds = [(e["type"], e["dir"]) for e in st["events"]]
    assert ("BOS", 1) in kinds and ("CHoCH", -1) in kinds
    assert st["trend"] == -1


def test_zones_and_fvg_state():
    closes = [100] * 10 + [100, 99, 98, 99, 105, 112, 118, 121, 120, 121, 125, 130, 128, 131]
    b = Bars(_frame(closes, 0.3))
    st = analyze_structure(b, 2, 2)
    zs = zones_from_events(b, st["events"], 0.5)
    assert all(z["state"] in ("fresh", "tested", "broken") for z in zs)
    assert isinstance(find_fvgs(b, 50, 0.05), list)


def test_backtest_has_no_lookahead():
    """Trades found on truncated data must be identical to the same trades on the full data."""
    df = f.drop_incomplete(f.get_candles("GC=F", "15m", "40d"), "15m")
    full = bt.run_backtest(df, "15m")
    cut = bt.run_backtest(df.iloc[:-800], "15m")
    last_ts = int(df.index[-801].timestamp())
    a = [(t["entry_ts"], t["direction"]) for t in full["trades"] if t["entry_ts"] < last_ts - 86400]
    b = [(t["entry_ts"], t["direction"]) for t in cut["trades"] if t["entry_ts"] < last_ts - 86400]
    assert a == b and len(a) > 0


def test_signal_engine_runs_all_entry_tfs():
    from app import strategy as S
    from app.levels import key_levels
    for iv in ("5m", "15m", "30m"):
        d4 = f.drop_incomplete(f.get_candles("GC=F", "4h"), "4h")
        d1 = f.drop_incomplete(f.get_candles("GC=F", "1h"), "1h")
        de = f.drop_incomplete(f.get_candles("GC=F", iv), iv)
        lv = key_levels(f.get_candles("GC=F", "1d"), f.get_candles("GC=F", "1w"), d1, float(de.Close.iloc[-1]))
        sig, an = S.build_signal(d4, d1, de, S.Cfg(entry_seconds=f.INTERVAL_SECONDS[iv], entry_interval=iv), lv,
                                 sessions.allowed_mask(de.index), True)
        assert sig["direction"] in ("BUY", "WAIT") or sig["direction"] == "SELL"
        assert 0 <= sig["agreement"] <= 100 and an["4h"]["bias"] in ("bullish", "bearish", "neutral")


def test_live_merge_rebuilds_missing_and_forming_candles():
    """A stale cached 15m frame is patched from the 1-minute tape: missing bars reappear."""
    base = f._get_raw("GC=F", "15m")
    stale = base.iloc[:-3]
    merged = f.merge_live(stale, "15m", "GC=F")
    assert len(merged) == len(base)
    assert np.allclose(merged[["Open", "High", "Low", "Close"]].tail(3).to_numpy(),
                       base[["Open", "High", "Low", "Close"]].tail(3).to_numpy())


def test_own_structure_for_every_timeframe():
    from app import strategy as S
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"):
        df = f.drop_incomplete(f.get_candles("GC=F", tf), tf)
        st = S.own_structure(df, tf)
        assert st is None or st["bias"] in ("bullish", "bearish", "neutral")
