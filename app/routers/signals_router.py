from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import auth, models, schemas, sessions
from .. import strategy as strat
from ..strategy import effective_min_agreement
from ..data_fetcher import (DataUnavailable, INTERVAL_SECONDS, data_status, drop_incomplete, get_candles,
                            is_futures)
from ..database import get_db
from ..levels import key_levels
from ..models import utcnow

router = APIRouter(prefix="/signals", tags=["signals"])

EXPIRE_HOURS = 72  # open signals older than this are closed as "expired"


def unavailable(e: DataUnavailable) -> HTTPException:
    return HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics,
                                      "hint": "Run `pip install -U -r requirements.txt` (yfinance must be recent) "
                                              "and open /market/health for a live connectivity test."})


# ---------------------------------------------------------------- auto win / loss
def _to_dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def resolve_open(db: Session, user: models.User) -> list[models.SignalRecord]:
    """Check every open signal against price action since it fired.  TP first -> won,
    SL first -> lost (both inside one candle counts as lost).  Creates alerts."""
    opens = db.query(models.SignalRecord).filter(
        models.SignalRecord.user_id == user.id, models.SignalRecord.status == "open").all()
    if not opens:
        return []
    frames: dict[str, object] = {}
    closed: list[models.SignalRecord] = []
    for rec in opens:
        try:
            if rec.symbol not in frames:
                frames[rec.symbol] = get_candles(rec.symbol, "5m")
            df = frames[rec.symbol]
        except DataUnavailable:
            continue
        sec = INTERVAL_SECONDS.get(rec.entry_interval or "15m", 900)
        start = (rec.trigger_ts + sec) if rec.trigger_ts else int(rec.created_at.replace(tzinfo=timezone.utc).timestamp())
        t = df.index.as_unit("s").asi8
        m = t >= start
        if m.any() and rec.entry_price and rec.stop_loss and rec.take_profit:
            hi, lo = df["High"].to_numpy()[m], df["Low"].to_numpy()[m]
            up = rec.direction == "BUY"
            tp_hit = hi >= rec.take_profit if up else lo <= rec.take_profit
            sl_hit = lo <= rec.stop_loss if up else hi >= rec.stop_loss
            any_hit = tp_hit | sl_hit
            if any_hit.any():
                k = int(np.argmax(any_hit))
                win = bool(tp_hit[k]) and not bool(sl_hit[k])
                risk = abs(rec.entry_price - rec.stop_loss)
                rec.status = "won" if win else "lost"
                rec.result_r = round(abs(rec.take_profit - rec.entry_price) / risk, 2) if win else -1.0
                rec.outcome_price = rec.take_profit if win else rec.stop_loss
                rec.closed_at = _to_dt(int(t[m][k]))
                rec.outcome = "auto"
                sign = "+" if win else ""
                db.add(models.Alert(
                    user_id=user.id, kind="win" if win else "loss", symbol=rec.symbol, signal_id=rec.id,
                    title=f"{'TAKE PROFIT' if win else 'STOP LOSS'} - {rec.symbol} {rec.direction}",
                    message=f"{rec.direction} from {rec.entry_price:.2f} closed at {rec.outcome_price:.2f} "
                            f"({sign}{rec.result_r}R)"))
                closed.append(rec)
                continue
        if (utcnow() - rec.created_at).total_seconds() > EXPIRE_HOURS * 3600:
            rec.status, rec.closed_at, rec.outcome = "expired", utcnow(), "auto"
            db.add(models.Alert(user_id=user.id, kind="info", symbol=rec.symbol, signal_id=rec.id,
                                title=f"Signal expired - {rec.symbol} {rec.direction}",
                                message=f"Neither TP nor SL was reached within {EXPIRE_HOURS}h."))
            closed.append(rec)
    if closed:
        db.commit()
    return closed


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


def _live_r(rec: models.SignalRecord, price: float) -> float | None:
    if not (rec.entry_price and rec.stop_loss):
        return None
    risk = abs(rec.entry_price - rec.stop_loss)
    d = 1 if rec.direction == "BUY" else -1
    return round((price - rec.entry_price) * d / risk, 2) if risk else None


# ------------------------------------------------------------------------- live
@router.get("/live")
def live_signal(
    symbol: str = "GC=F",
    entry_interval: Literal["1m", "5m", "15m", "30m"] = "15m",
    min_agreement: float = Query(70, ge=0, le=100),
    sessions_only: bool = True,
    rr: float = Query(2.0, ge=0.5, le=10),
    breakeven_at_r: float = Query(1.0, ge=0, le=5),
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
    eff_min = effective_min_agreement(min_agreement, rr, INTERVAL_SECONDS[entry_interval])
    cfg = strat.Cfg(rr=rr, min_agreement=eff_min, sessions_only=sessions_only,
                    entry_seconds=INTERVAL_SECONDS[entry_interval], entry_interval=entry_interval,
                    breakeven_at_r=breakeven_at_r)
    sig, analysis = strat.build_signal(d4, d1, de, cfg, levels, sessions.allowed_mask(de.index), market_ok,
                                       forming_bar=forming_bar)
    sig["min_agreement_requested"] = min_agreement
    sig["min_agreement_effective"] = round(eff_min, 1)
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

    # ---- scan log: record every evaluation (fired or not) so a quiet day is explainable, not opaque
    db.add(models.ScanLog(
        user_id=user.id, symbol=symbol, entry_interval=entry_interval, status=sig["status"],
        headline=sig["headline"], direction=sig["direction"] if sig["direction"] in ("BUY", "SELL") else None,
        bias_4h=sig["bias_4h"], bias_1h=sig["bias_1h"], agreement=sig["agreement"]))
    stale = db.query(models.ScanLog.id).filter(
        models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol,
        models.ScanLog.entry_interval == entry_interval).order_by(desc(models.ScanLog.ts)).offset(500).all()
    if stale:
        db.query(models.ScanLog).filter(models.ScanLog.id.in_([s.id for s in stale])).delete(synchronize_session=False)
    try:
        db.commit()
    except Exception:  # noqa: BLE001 - the scan log is diagnostic only, never block the live endpoint on it
        db.rollback()

    # ---- resolve old signals (win/loss alerts) then persist a fresh one
    newly_closed = resolve_open(db, user)
    record_id = None
    if persist and sig["direction"] in ("BUY", "SELL") and market_ok:
        dup = db.query(models.SignalRecord).filter(
            models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
            models.SignalRecord.status == "open").first()
        dup_same = db.query(models.SignalRecord).filter(
            models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
            models.SignalRecord.trigger_ts == sig["candle_ts"]).first()
        if not dup and not dup_same:
            rec = models.SignalRecord(
                user_id=user.id, symbol=symbol, direction=sig["direction"], bias_4h=sig["bias_4h"],
                entry_price=sig["entry_price"], stop_loss=sig["stop_loss"], take_profit=sig["take_profit"],
                confidence=sig["confidence"], reason=sig["reason"], status="open",
                entry_interval=entry_interval, trigger_ts=sig["candle_ts"], agreement=sig["agreement"],
                grade=sig["grade"], session_label=sess["label"], risk_reward=rr,
                checks_json=json.dumps(sig["checks"]))
            db.add(rec)
            db.flush()
            db.add(models.Alert(
                user_id=user.id, kind="signal", symbol=symbol, signal_id=rec.id,
                title=f"{sig['direction']} signal - {symbol}",
                message=f"{sig['headline']} | entry {sig['entry_price']:.2f}  SL {sig['stop_loss']:.2f}  "
                        f"TP {sig['take_profit']:.2f} | agreement {sig['agreement']:.0f}% ({sig['grade']})"))
            db.commit()
            db.refresh(rec)
            record_id = rec.id

    active = db.query(models.SignalRecord).filter(
        models.SignalRecord.user_id == user.id, models.SignalRecord.symbol == symbol,
        models.SignalRecord.status == "open").order_by(desc(models.SignalRecord.created_at)).first()
    active_out = None
    if active:
        active_out = {**schemas.SignalOut.model_validate(active).model_dump(mode="json"), "live_r": _live_r(active, price)}

    return {
        "signal": sig,
        "analysis": analysis,
        "record_id": record_id,
        "active_signal": active_out,
        "closed_now": [r.id for r in newly_closed],
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
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    """Every evaluation the engine made (fired or not), newest first - the answer to 'what's it been
    doing all day' and 'why hasn't this pair signalled yet'."""
    q = db.query(models.ScanLog).filter(models.ScanLog.user_id == user.id, models.ScanLog.symbol == symbol)
    if entry_interval:
        q = q.filter(models.ScanLog.entry_interval == entry_interval)
    rows = q.order_by(desc(models.ScanLog.ts)).limit(limit).all()
    return [{"ts": int(r.ts.replace(tzinfo=timezone.utc).timestamp()), "entry_interval": r.entry_interval,
            "status": r.status, "headline": r.headline, "direction": r.direction,
            "bias_4h": r.bias_4h, "bias_1h": r.bias_1h, "agreement": r.agreement} for r in rows]


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
    signal_id: int, result: Literal["won", "lost", "cancelled"] = Query(...),
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
    if result_r is None and result in ("won", "lost") and rec.entry_price and rec.stop_loss and rec.take_profit:
        risk = abs(rec.entry_price - rec.stop_loss)
        result_r = round(abs(rec.take_profit - rec.entry_price) / risk, 2) if result == "won" else -1.0
    rec.result_r = result_r
    rec.closed_at = utcnow()
    rec.outcome = "manual"
    db.commit()
    return {"ok": True}
