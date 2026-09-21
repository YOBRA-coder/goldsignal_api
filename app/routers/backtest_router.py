import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc
from sqlalchemy.orm import Session

from .. import models, schemas, auth, backtest as bt
from ..data_fetcher import DataUnavailable, drop_incomplete, get_candles
from ..database import get_db

router = APIRouter(prefix="/backtest", tags=["backtest"])


def _out(r: models.BacktestRun) -> schemas.BacktestOut:
    stats = json.loads(r.stats_json) if r.stats_json else None
    return schemas.BacktestOut(
        id=r.id, symbol=r.symbol, period=r.period, entry_interval=r.entry_interval,
        total_trades=r.total_trades, wins=r.wins, losses=r.losses,
        win_rate=r.win_rate, net_r=r.net_r, avg_r=r.avg_r,
        equity_curve=json.loads(r.equity_curve_json), trades=json.loads(r.trades_json),
        created_at=r.created_at, stats=stats,
    )


@router.post("/run", response_model=schemas.BacktestOut)
def run(
    payload: schemas.BacktestRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(auth.get_current_user),
):
    try:
        df = get_candles(payload.symbol, payload.entry_interval, payload.period, live=False)
        df1 = get_candles(payload.symbol, "1h", "180d", live=False)
    except DataUnavailable as e:
        raise HTTPException(503, detail={"message": str(e), "diagnostics": e.diagnostics})
    df = drop_incomplete(df, payload.entry_interval)
    df1 = drop_incomplete(df1, "1h")
    if len(df) < 600:
        raise HTTPException(400, f"Only {len(df)} bars available - need at least 600 for a meaningful backtest.")
    result = bt.run_backtest(df, payload.entry_interval, payload.risk_reward, payload.min_agreement,
                             payload.sessions_only, payload.symbol, df1)

    stats = {k: result[k] for k in ("profit_factor", "max_drawdown_r", "best_streak", "worst_streak",
                                    "by_session", "by_direction", "equity_ts", "bars", "from_ts", "to_ts", "htf_source")}
    stats["params"] = payload.model_dump()
    rec = models.BacktestRun(
        user_id=user.id, symbol=payload.symbol, period=payload.period,
        entry_interval=payload.entry_interval,
        total_trades=result["total_trades"], wins=result["wins"], losses=result["losses"],
        win_rate=result["win_rate"], net_r=result["net_r"], avg_r=result["avg_r"],
        equity_curve_json=json.dumps(result["equity_curve"]),
        trades_json=json.dumps(result["trades"]), stats_json=json.dumps(stats),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return _out(rec)


@router.get("/history", response_model=List[schemas.BacktestOut])
def history(db: Session = Depends(get_db), user: models.User = Depends(auth.get_current_user)):
    rows = db.query(models.BacktestRun).filter(
        models.BacktestRun.user_id == user.id
    ).order_by(desc(models.BacktestRun.created_at)).limit(30).all()
    return [_out(r) for r in rows]
