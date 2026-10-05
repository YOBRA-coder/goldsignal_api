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


def test_candles_are_cut_like_metatrader5():
    """ny_close anchor: server = New York + 7h  ->  4H candles start 17,21,01,05,09,13 NY time,
    D1 starts 17:00 NY; utc anchor -> 00,04,08... UTC."""
    from app import settings
    idx = pd.date_range("2026-07-06 00:00", "2026-07-10 20:00", freq="1h", tz="UTC")     # summer (EDT)
    df = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 1.0}, index=idx)
    settings.set_anchor("ny_close")
    h4 = f.resample_ohlc(df, "4h")
    assert set(h4.index.hour) == {21, 1, 5, 9, 13, 17}          # UTC hours in summer == NY 17,21,01,05,09,13
    d1 = f.resample_ohlc(df, "1d")
    assert set(d1.index.hour) == {21}                            # day opens 17:00 EDT = 21:00 UTC
    w1 = f.resample_ohlc(df, "1w")
    assert w1.index[0].weekday() in (6, 0) and w1.index[0].hour == 21
    settings.set_anchor("utc")
    assert set(f.resample_ohlc(df, "4h").index.hour) == {0, 4, 8, 12, 16, 20}
    assert set(f.resample_ohlc(df, "1d").index.hour) == {0}
    settings.set_anchor("ny_close")


def test_price_offset_shifts_every_timeframe():
    from app import settings
    base = f.get_candles("GC=F", "15m")["Close"].iloc[-1]
    settings.set_offset("GC=F", 12.5)
    try:
        assert abs(f.get_candles("GC=F", "15m")["Close"].iloc[-1] - base - 12.5) < 1e-6
        assert abs(f.get_candles("GC=F", "4h")["High"].iloc[-1] - f.resample_ohlc(f._candles("GC=F", "1h", None, True), "4h")["High"].iloc[-1] - 12.5) < 1e-6
    finally:
        settings.set_offset("GC=F", None)


# ------------------------------------------------------------- trade tracker / confirmations / indicators
def test_replay_breakeven_win_loss():
    from app import trades
    t = np.arange(0, 6 * 900, 900)
    tr = {"direction": "BUY", "entry": 100.0, "sl": 99.0, "tp": 102.0, "start_ts": 0}
    # +1R then back to entry -> breakeven, 0R
    o = np.array([100, 100.5, 101.2, 100.5, 100, 100.0]); h = o + 0.3; l = o - 0.1
    l = np.array([99.9, 100.2, 100.9, 100.2, 99.9, 99.9]); h = np.array([100.3, 101.2, 101.4, 100.8, 100.2, 100.1])
    r = trades.replay(tr, t, o, h, l, o)
    assert r["status"] == "breakeven" and r["result_r"] == 0.0 and r["be_moved"]
    # straight to target -> win 2R
    h2 = np.array([100.5, 101.5, 102.1, 102.2, 102.3, 102.4]); l2 = np.array([99.9, 100.4, 101.4, 101.9, 102, 102])
    r = trades.replay(tr, t, o, h2, l2, o)
    assert r["status"] == "won" and r["result_r"] == 2.0
    # straight down -> loss -1R
    h3 = np.array([100.1, 100.0, 99.5, 99, 99, 99]); l3 = np.array([99.5, 98.9, 98.5, 98, 98, 98])
    r = trades.replay(tr, t, o, h3, l3, o)
    assert r["status"] == "lost" and r["result_r"] == -1.0
    # SELL mirror
    s = {"direction": "SELL", "entry": 100.0, "sl": 101.0, "tp": 98.0, "start_ts": 0}
    r = trades.replay(s, t, o, np.array([100.1, 99.9, 99.5, 99.5, 99.5, 99.5]), np.array([99.5, 97.9, 97, 97, 97, 97]), o)
    assert r["status"] == "won"


def test_can_open_rules():
    from app import trades
    run = [{"direction": "BUY", "entry": 100.0, "sl": 99.0, "plan": {}}]
    new = {"direction": "BUY", "entry": 100.2, "sl": 99.2}
    assert not trades.can_open(run, new)[0]                       # same level
    assert trades.can_open(run, {"direction": "BUY", "entry": 103.0, "sl": 102.0})[0]
    assert trades.can_open(run, {"direction": "SELL", "entry": 100.1, "sl": 101.1})[0]
    assert not trades.can_open(run * 3, {"direction": "BUY", "entry": 120.0, "sl": 119.0})[0]   # max 3


def test_indicators_are_causal():
    from app import indicators as ix
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 300))
    h, l = c + 1, c - 1
    full = (ix.rsi(c), ix.adx(h, l, c)[0], ix.macd(c)[2], ix.efficiency_ratio(c))
    cut = (ix.rsi(c[:200]), ix.adx(h[:200], l[:200], c[:200])[0], ix.macd(c[:200])[2], ix.efficiency_ratio(c[:200]))
    for a, b in zip(full, cut):
        assert np.allclose(a[:200], b, atol=1e-9)                 # value at bar i never depends on later bars
    assert 0 <= full[0].min() and full[0].max() <= 100


def test_backtest_concurrent_and_styles_run():
    df = _frame(list(100 + 5 * np.sin(np.arange(1500) / 20.0)))
    for kw in ({"max_open": 1}, {"max_open": 3}, {"style": "scalp", "max_open": 2}):
        out = bt.run_backtest(df, "15m", 2.0, 60, False, "EURUSD=X", None, 1.0, **kw)
        assert "total_trades" in out and "breakevens" in out
