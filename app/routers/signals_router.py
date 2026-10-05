from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import List, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import auth, models, schemas, sessions, trades
from .. import strategy as strat
from ..confirm import candle_patterns
from ..strategy import effective_min_agreement
from ..data_fetcher import (DataUnavailable, INTERVAL_SECONDS, data_status, drop_incomplete, get_candles,
                            is_futures)
from ..database import get_db
from ..levels import key_levels
from ..models import utcnow
from ..structure import Bars

router = APIRouter(prefix="/signals", tags=["signals"])

EXPIRE_HOURS = 72  # open signals older than this are closed as "expired"
MAX_OPEN_PER_SYMBOL = 3  # trades that may run at the same time on one pair


def unavailable(e: DataUnavailable) -> HTTPException:
    return HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics,
                                      "hint": "Run `pip install -U -r requirements.txt` (yfinance must be recent) "
                                              "and open /market/health for a live connectivity test."})


# ---------------------------------------------------------------- trade tracker
def _to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _jload(txt, default):
    try:
        return json.loads(txt) if txt else default
    except (TypeError, ValueError):
        return default


def _trade_dict(rec: models.SignalRecord) -> dict:
    sec = INTERVAL_SECONDS.get(rec.entry_interval or "15m", 900)
    start = (rec.trigger_ts + sec) if rec.trigger_ts else int(rec.created_at.replace(tzinfo=timezone.utc).timestamp())
    return {"direction": rec.direction, "entry": rec.entry_price, "sl": rec.stop_loss, "tp": rec.take_profit,
            "start_ts": start, "style": rec.style or "swing", "plan": _jload(rec.plan_json, {}), "be_at": 1.0}


_CTX_CACHE: dict[tuple, tuple[float, dict]] = {}


def trade_context(symbol: str, entry_interval: str, direction: str) -> dict:
    """Market state NOW, used to judge whether an open trade's reasons still hold (cached ~20 s per pair)."""
    key = (symbol, entry_interval, direction)
    hit = _CTX_CACHE.get(key)
    if hit and time.time() - hit[0] < 20:
        return hit[1]
    ctx: dict = {}
    try:
        d4 = drop_incomplete(get_candles(symbol, "4h"), "4h")
        d1 = drop_incomplete(get_candles(symbol, "1h"), "1h")
        ctx["bias_4h"] = strat.quick_bias(d4, 3)["bias"]
        ctx["bias_1h"] = strat.quick_bias(d1, 2)["bias"]
        de = drop_incomplete(get_candles(symbol, entry_interval), entry_interval)
        be = Bars(de.tail(120))
        opp = -1 if direction == "BUY" else 1
        pats = [p for p in candle_patterns(be, be.n - 1, opp) if p.get("strength", 0) >= 3]
        ctx["opposing_pattern"] = pats[0]["label"] if pats else None
        vr = strat._volatility_regime(be, be.n - 1)
        ctx["vol_spike"] = bool(vr and vr["spike"])
    except Exception:  # noqa: BLE001 - context is advisory
        pass
    fut = is_futures(symbol)
    ctx["market_open"] = sessions.market_open(None, fut)
    ctx["mins_to_close"] = sessions.minutes_to_market_close(None, fut)
    _CTX_CACHE[key] = (time.time(), ctx)
    return ctx


def track_open(db: Session, user: models.User) -> tuple[list[models.SignalRecord], list[dict]]:
    """Replay every open trade against the candles since entry (TP / SL / breakeven), judge its health against the
    market now, store the result, raise alerts on changes. Returns (trades closed now, snapshot of every open one)."""
    opens = db.query(models.SignalRecord).filter(
        models.SignalRecord.user_id == user.id, models.SignalRecord.status == "open").all()
    if not opens:
        return [], []
    frames: dict[str, object] = {}
    closed: list[models.SignalRecord] = []
    snaps: list[dict] = []
    now = utcnow()
    for rec in opens:
        if not (rec.entry_price and rec.stop_loss and rec.take_profit):
            continue
        try:
            if rec.symbol not in frames:
                frames[rec.symbol] = get_candles(rec.symbol, "5m")
            df = frames[rec.symbol]
        except DataUnavailable:
            continue
        td = _trade_dict(rec)
        t = df.index.as_unit("s").asi8
        st = trades.replay(td, t, df["Open"].to_numpy(float), df["High"].to_numpy(float), df["Low"].to_numpy(float),
                           df["Close"].to_numpy(float), be_at=1.0)
        price = float(df["Close"].iloc[-1])
        was_be = bool(rec.be_moved)
        rec.max_r, rec.min_r, rec.be_moved, rec.current_sl = st["max_r"], st["min_r"], st["be_moved"], st["current_sl"]
        events = _jload(rec.events_json, [])
        known = {(e.get("ts"), e.get("kind")) for e in events}
        for e in st["events"]:
            if (e["ts"], e["kind"]) not in known:
                events.append(e)
        age_h = (now - rec.created_at).total_seconds() / 3600

        if st["status"] in ("won", "lost", "breakeven"):
            rec.status, rec.result_r, rec.outcome_price = st["status"], st["result_r"], st["outcome_price"]
            rec.closed_at, rec.outcome, rec.health, rec.health_note = _to_dt(st["closed_ts"]), "auto", None, None
            win = st["status"] == "won"
            be = st["status"] == "breakeven"
            sign = "+" if (st["result_r"] or 0) > 0 else ""
            db.add(models.Alert(
                user_id=user.id, kind="win" if win else ("info" if be else "loss"), symbol=rec.symbol, signal_id=rec.id,
                title=(f"TAKE PROFIT - {rec.symbol} {rec.direction}" if win else
                       f"BREAKEVEN exit - {rec.symbol} {rec.direction}" if be else f"STOP LOSS - {rec.symbol} {rec.direction}"),
                message=f"{rec.direction} from {rec.entry_price:.5g} closed at {rec.outcome_price:.5g} ({sign}{st['result_r']}R)"))
            closed.append(rec)
        elif age_h > EXPIRE_HOURS:
            rec.status, rec.closed_at, rec.outcome, rec.health = "expired", now, "auto", None
            db.add(models.Alert(user_id=user.id, kind="info", symbol=rec.symbol, signal_id=rec.id,
                                title=f"Signal expired - {rec.symbol} {rec.direction}",
                                message=f"Neither TP nor SL was reached within {EXPIRE_HOURS}h."))
            closed.append(rec)
        else:
            m = trades.live_metrics(td, st, price)
            m["tp_r"] = abs(rec.take_profit - rec.entry_price) / (abs(rec.entry_price - rec.stop_loss) or 1e-9)
            ctx = trade_context(rec.symbol, rec.entry_interval or "15m", rec.direction)
            ctx["age_hours"] = age_h
            h = trades.assess(td, st, m, ctx)
            if st["be_moved"] and not was_be:
                db.add(models.Alert(user_id=user.id, kind="update", symbol=rec.symbol, signal_id=rec.id,
                                    title=f"Stop to breakeven - {rec.symbol} {rec.direction}",
                                    message=f"+1R reached. Move your stop to the entry {rec.entry_price:.5g}; the trade can no longer lose."))
            if h["key"] != (rec.last_event or "") and h["state"] in ("caution", "danger", "near_target"):
                db.add(models.Alert(user_id=user.id, kind="update", symbol=rec.symbol, signal_id=rec.id,
                                    title=f"{'WARNING' if h['state'] == 'danger' else 'Update'} - {rec.symbol} {rec.direction}",
                                    message=h["notes"][0]))
                events.append({"ts": int(time.time()), "kind": "health", "text": h["notes"][0]})
            rec.last_event, rec.health, rec.health_note = h["key"], h["state"], " | ".join(h["notes"][:3])
            snaps.append({"id": rec.id, **m, "health": h["state"], "notes": h["notes"], "be_moved": st["be_moved"],
                          "current_sl": st["current_sl"], "max_r": st["max_r"], "min_r": st["min_r"], "age_hours": round(age_h, 1)})
        rec.events_json = json.dumps(events[-40:])
    db.commit()
    return closed, snaps


def resolve_open(db: Session, user: models.User) -> list[models.SignalRecord]:
    """Backwards-compatible name: track every open trade, return the ones that just closed."""
    return track_open(db, user)[0]


def check_bias_shift(db: Session, user: models.User, symbol: str, bias_4h: str, bias_1h: str) -> None:
    """Early-warning alert: fires the moment 4H flips, or 1H moves in/out of agreement with 4H -
    well before the full 4-step checklist would produce a BUY/SELL signal."""
    st = db.query(models.BiasState).filter(models.BiasState.user_id == user.id,
                                           models.BiasState.symbol == symbol).first()
    aligned = bias_4h == bias_1h and bias_4h != "neutral"
    if st is None:
        db.add(models.BiasState(user_id=user.id, symbol=symbol, bias_4h=bias_4h, bias_1h=bias_1h, aligned=aligned))
        db.commit()
        return
    msgs = []
    if st.bias_4h and st.bias_4h != bias_4h and bias_4h != "neutral" and st.bias_4h != "neutral":
        msgs.append(f"4H flipped {st.bias_4h} -> {bias_4h.upper()}")
    elif st.bias_4h and st.bias_4h != bias_4h:
        msgs.append(f"4H structure turned {bias_4h}")
    if st.aligned != aligned:
        msgs.append(f"1H {'now aligns with' if aligned else 'no longer aligns with'} 4H ({bias_1h})" if aligned
                    else f"1H drifted out of sync with 4H (now {bias_1h} vs {bias_4h})")
    if msgs:
        db.add(models.Alert(user_id=user.id, kind="shift", symbol=symbol,
                            title=f"Bias shift - {symbol}", message=" · ".join(msgs)))
    st.bias_4h, st.bias_1h, st.aligned, st.updated_at = bias_4h, bias_1h, aligned, utcnow()
    db.commit()


def _rec_out(rec: models.SignalRecord, snap: dict | None = None) -> dict:
    out = schemas.SignalOut.model_validate(rec).model_dump(mode="json")
    out["plan"] = _jload(rec.plan_json, {})
    out["events"] = _jload(rec.events_json, [])
    if snap:
        out.update({k: snap[k] for k in ("live_r", "progress", "to_tp", "to_sl", "price", "notes", "age_hours") if k in snap})
    return out


# ------------------------------------------------------------------------- live
# One place that decides which of the new engine switches the live signal AND the backtest use.
LIVE_FLAGS = dict(no_chase=True, min_confirms=1, ext_patterns=False, ext_confirms=False, fvg_extra=False)


def _make_cfg(style: str, rr: float, min_agreement: float, sessions_only: bool, entry_interval: str,
              breakeven_at_r: float) -> strat.Cfg:
    scalp = style == "scalp"
    rr_eff = min(rr, 1.5) if scalp else rr
    eff_min = effective_min_agreement(min_agreement, rr_eff, INTERVAL_SECONDS[entry_interval])
    return strat.Cfg(rr=rr, min_agreement=eff_min, sessions_only=sessions_only,
                     entry_seconds=INTERVAL_SECONDS[entry_interval], entry_interval=entry_interval,
                     breakeven_at_r=breakeven_at_r, style=style, **LIVE_FLAGS)


def _plan_json(sig: dict) -> str:
    return json.dumps({
        "bias_4h": sig.get("bias_4h"), "bias_1h": sig.get("bias_1h"), "zone": sig.get("zone"),
        "confirmations": sig.get("confirmations"), "pattern": (sig.get("pattern") or {}).get("label"),
        "tp_reason": sig.get("tp_reason"), "reward_r": sig.get("reward_r"), "style": sig.get("style"),
        "highlights": sig.get("highlights"), "retest_entry": sig.get("retest_entry")}, default=str)


@router.get("/live")
def live_signal(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    min_agreement: float = Query(70, ge=0, le=100),
    sessions_only: bool = True,
    rr: float = Query(2.0, ge=0.5, le=10),
    breakeven_at_r: float = Query(1.0, ge=0, le=5),
    scalp: bool = False,
    persist: bool = True,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    try:
        raw4 = get_candles(symbol, "4h")
        raw1 = get_candles(symbol, "1h")
        rawe = get_candles(symbol, entry_interval)
    except DataUnavailable as e:
        raise unavailable(e)

    d4, d1, de = (drop_incomplete(raw4, "4h"), drop_incomplete(raw1, "1h"), drop_incomplete(rawe, entry_interval))
    if len(d4) < strat.MIN_BARS["4h"] or len(d1) < strat.MIN_BARS["1h"] or len(de) < strat.MIN_BARS["entry"]:
        raise HTTPException(503, detail={"message": f"Not enough history for {symbol} "
                                         f"(4H {len(d4)}, 1H {len(d1)}, {entry_interval} {len(de)} bars)."})

    forming_bar = None
    if len(rawe) > len(de):
        r = rawe.iloc[-1]
        forming_bar = {"Open": float(r["Open"]), "High": float(r["High"]), "Low": float(r["Low"]),
                       "Close": float(r["Close"])}

    daily = weekly = None
    try:
        daily, weekly = get_candles(symbol, "1d"), get_candles(symbol, "1w")
    except DataUnavailable:
        pass
    price = float(rawe["Close"].iloc[-1])
    levels = key_levels(daily, weekly, d1, price)

    futures = is_futures(symbol)
    status = data_status(symbol, entry_interval, rawe)
    market_ok = status["market_open"] and not status["stale"]
    sess_mask = sessions.allowed_mask(de.index)
    ctx = strat.make_ctx(d4, d1)            # built once, shared by the swing and the scalp evaluation

    cfg = _make_cfg("swing", rr, min_agreement, sessions_only, entry_interval, breakeven_at_r)
    sig, analysis = strat.build_signal(d4, d1, de, cfg, levels, sess_mask, market_ok, forming_bar=forming_bar,
                                       ctx=ctx, futures=futures)
    sig["min_agreement_requested"] = min_agreement
    sig["min_agreement_effective"] = round(cfg.min_agreement, 1)

    scalp_sig = None
    if scalp:
        scfg = _make_cfg("scalp", rr, min_agreement, sessions_only, entry_interval, breakeven_at_r)
        scalp_sig, _ = strat.build_signal(d4, d1, de, scfg, levels, sess_mask, market_ok, ctx=ctx, futures=futures)
        scalp_sig["min_agreement_effective"] = round(scfg.min_agreement, 1)
    if not market_ok:
        lag_min = round((status.get("lag_sec") or 0) / 60)
        sig["headline"] = (f"Price feed is ~{lag_min} min behind - signals paused until it catches up"
                           if status["market_open"] else "Market closed - no signals until it reopens")
    sess = sessions.session_state(futures=futures)
    sig["session"] = {"label": sess["label"], "active": sess["active"], "allowed": sess["trade_allowed"]}

    # ---- early-warning: did the 4H/1H bias itself shift? (independent of whether a full signal fired)
    if market_ok:
        try:
            check_bias_shift(db, user, symbol, sig["bias_4h"], sig["bias_1h"])
        except Exception:  # noqa: BLE001 - never let an alert-side bug break the live endpoint
            db.rollback()

    # ---- scan log: record every evaluation (fired or not) so a quiet day is explainable, not opaque.
    previews = {"forming": sig.get("forming"), "h1_preview": sig.get("h1_preview"),
                "one_h_preview": sig.get("one_h_preview"), "entry_shift_preview": sig.get("entry_shift_preview"),
                "counter_watch": sig.get("counter_watch"), "highlights": sig.get("highlights")}
    try:
        db.add(models.ScanLog(
            user_id=user.id, symbol=symbol, entry_interval=entry_interval, status=sig["status"],
            headline=sig["headline"], direction=sig["direction"] if sig["direction"] in ("BUY", "SELL") else None,
            bias_4h=sig["bias_4h"], bias_1h=sig["bias_1h"], agreement=sig["agreement"], grade=sig.get("grade"),
            momentum_agree=(sig.get("momentum") or {}).get("agree"), volatility_spike=(sig.get("volatility") or {}).get("spike"),
            checks_json=json.dumps(sig["checks"]), previews_json=json.dumps(previews, default=str)))
        stale = db.query(models.ScanLog.id).filter(
            models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol,
            models.ScanLog.entry_interval == entry_interval).order_by(desc(models.ScanLog.ts)).offset(500).all()
        if stale:
            db.query(models.ScanLog).filter(models.ScanLog.id.in_([s_.id for s_ in stale])).delete(synchronize_session=False)
        db.commit()
    except Exception:  # noqa: BLE001 - the scan log is diagnostic only, never block the live endpoint on it
        db.rollback()

    # ---- track every running trade (TP / SL / breakeven / health), then add a fresh one if this scan produced it
    newly_closed, snaps = [], []
    try:
        newly_closed, snaps = track_open(db, user)
    except Exception:  # noqa: BLE001 - a feed hiccup on another pair must not hide this pair's signal
        db.rollback()

    record_ids: list[int] = []
    skipped: list[str] = []
    if persist and market_ok:
        for cand in (sig, scalp_sig):
            if not cand or cand["direction"] not in ("BUY", "SELL"):
                continue
            if db.query(models.SignalRecord.id).filter(
                    models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                    models.SignalRecord.trigger_ts == cand["candle_ts"],
                    models.SignalRecord.style == cand.get("style")).first():
                continue                                             # this exact candle was already recorded
            running = db.query(models.SignalRecord).filter(
                models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
                models.SignalRecord.status == "open").all()
            ok, why = trades.can_open(
                [{"direction": r_.direction, "entry": r_.entry_price, "plan": _jload(r_.plan_json, {})} for r_ in running],
                {"direction": cand["direction"], "entry": cand["entry_price"], "sl": cand["stop_loss"], "zone": cand.get("zone")},
                MAX_OPEN_PER_SYMBOL)
            if not ok:
                skipped.append(why)
                continue
            note = ""
            opp = [r_ for r_ in running if r_.direction != cand["direction"]]
            if opp:
                note = f" (note: you also have an open {opp[0].direction} on this pair)"
            rec = models.SignalRecord(
                user_id=user.id, symbol=symbol, direction=cand["direction"], bias_4h=cand["bias_4h"],
                entry_price=cand["entry_price"], stop_loss=cand["stop_loss"], take_profit=cand["take_profit"],
                confidence=cand["confidence"], reason=cand["reason"], status="open",
                entry_interval=entry_interval, trigger_ts=cand["candle_ts"], agreement=cand["agreement"],
                grade=cand["grade"], session_label=sess["label"], risk_reward=cand.get("rr", rr),
                checks_json=json.dumps(cand["checks"]), style=cand.get("style"), trade_type=cand.get("trade_type"),
                current_sl=cand["stop_loss"], be_moved=False, max_r=0.0, min_r=0.0, plan_json=_plan_json(cand),
                events_json=json.dumps([{"ts": cand["signal_time"], "kind": "open",
                                         "text": f"{cand.get('trade_type') or cand['direction']} opened at {cand['entry_price']:.5g}"}]))
            db.add(rec)
            db.flush()
            db.add(models.Alert(
                user_id=user.id, kind="signal", symbol=symbol, signal_id=rec.id,
                title=f"{cand.get('trade_type') or cand['direction']} signal - {symbol}",
                message=f"{cand['headline']} | entry {cand['entry_price']:.5g}  SL {cand['stop_loss']:.5g}  "
                        f"TP {cand['take_profit']:.5g} | agreement {cand['agreement']:.0f}% ({cand['grade']}){note}"))
            db.commit()
            record_ids.append(rec.id)
        if record_ids:
            try:
                _, snaps = track_open(db, user)
            except Exception:  # noqa: BLE001
                db.rollback()

    snap_by_id = {sn["id"]: sn for sn in snaps}
    open_recs = db.query(models.SignalRecord).filter(
        models.SignalRecord.user_id == user.id, models.SignalRecord.status == "open"
    ).order_by(desc(models.SignalRecord.created_at)).all()
    active_all = []
    for r_ in open_recs:
        sn = snap_by_id.get(r_.id)
        if sn is None and r_.symbol == symbol:
            sn = trades.live_metrics(_trade_dict(r_), {"current_sl": r_.current_sl or r_.stop_loss}, price)
        active_all.append(_rec_out(r_, sn))
    active_here = [a_ for a_ in active_all if a_["symbol"] == symbol]

    return {
        "signal": sig,
        "scalp_signal": scalp_sig,
        "analysis": analysis,
        "record_id": record_ids[0] if record_ids else None,
        "record_ids": record_ids,
        "skipped": skipped,
        "active_signal": active_here[0] if active_here else None,
        "active_signals": active_here,
        "open_trades": active_all,
        "closed_now": [r_.id for r_ in newly_closed],
        "sessions": sess,
        "data": status,
        "updated_at": int(datetime.now(timezone.utc).timestamp()),
    }


# ---------------------------------------------------------------------- scan log
@router.get("/scan-log")
def scan_log(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] | None = None,
    limit: int = Query(100, ge=1, le=500),
    detail: bool = False,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Every evaluation the engine made (fired or not), newest first - the answer to 'what's it been
    doing all day' and 'why hasn't this pair signalled yet'. Pass detail=true for the full per-check
    breakdown and every preview object (forming candle, early previews, counter-signal watch) instead
    of just the one-line headline - nothing the engine saw is held back."""
    q = db.query(models.ScanLog).filter(models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol)
    if entry_interval:
        q = q.filter(models.ScanLog.entry_interval == entry_interval)
    rows = q.order_by(desc(models.ScanLog.ts)).limit(limit).all()
    out = []
    for r in rows:
        row = {"ts": int(r.ts.replace(tzinfo=timezone.utc).timestamp()), "entry_interval": r.entry_interval,
               "status": r.status, "headline": r.headline, "direction": r.direction,
               "bias_4h": r.bias_4h, "bias_1h": r.bias_1h, "agreement": r.agreement, "grade": r.grade,
               "momentum_agree": r.momentum_agree, "volatility_spike": r.volatility_spike}
        if detail:
            row["checks"] = json.loads(r.checks_json) if r.checks_json else []
            row["previews"] = json.loads(r.previews_json) if r.previews_json else {}
        out.append(row)
    return out


# ---------------------------------------------------------------------- history
@router.get("/history", response_model=List[schemas.SignalOut])
def history(
    status_filter: str | None = None,
    symbol: str | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    try:
        resolve_open(db, user)
    except Exception:  # noqa: BLE001  (history must load even if the feed is down)
        pass
    q = db.query(models.SignalRecord).filter(models.SignalRecord.user_id == user.id)
    if status_filter:
        q = q.filter(models.SignalRecord.status == status_filter)
    if symbol:
        q = q.filter(models.SignalRecord.symbol == symbol)
    return q.order_by(desc(models.SignalRecord.created_at)).limit(300).all()


@router.get("/{signal_id}/checks")
def signal_checks(signal_id: int, db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    rec = db.query(models.SignalRecord).filter(
        models.SignalRecord.id == signal_id, models.SignalRecord.user_id == user.id).first()
    if not rec:
        raise HTTPException(404, "Signal not found")
    return {"checks": json.loads(rec.checks_json) if rec.checks_json else [], "reason": rec.reason}


@router.post("/{signal_id}/close")
def close_signal(
    signal_id: int, result: Literal["won", "lost", "breakeven", "cancelled"] = Query(...),
    result_r: float | None = None,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Manual override (auto-tracking normally closes signals for you)."""
    rec = db.query(models.SignalRecord).filter(
        models.SignalRecord.id == signal_id, models.SignalRecord.user_id == user.id).first()
    if not rec:
        raise HTTPException(404, "Signal not found")
    rec.status = result
    if result_r is None and result in ("won", "lost", "breakeven") and rec.entry_price and rec.stop_loss and rec.take_profit:
        risk = abs(rec.entry_price - rec.stop_loss)
        result_r = round(abs(rec.take_profit - rec.entry_price) / risk, 2) if result == "won" else (0.0 if result == "breakeven" else -1.0)
    rec.result_r = result_r
    rec.closed_at = utcnow()
    rec.outcome = "manual"
    db.commit()
    return {"ok": True}
