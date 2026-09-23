from sqlalchemy import (
    Column, Integer, String, DateTime, Float, Text, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base


def utcnow() -> datetime:
    """Naive UTC timestamp (SQLite friendly). API responses re-attach the UTC tzinfo."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    signals = relationship("SignalRecord", back_populates="owner")
    backtests = relationship("BacktestRun", back_populates="owner")


class SignalRecord(Base):
    """A signal the engine generated for a user (persisted so history/win-rate works)."""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, default="GC=F")
    direction = Column(String)          # BUY / SELL
    bias_4h = Column(String)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    confidence = Column(Float)          # agreement / 100 (kept for backwards compatibility)
    reason = Column(Text)               # human-readable breakdown of the checklist
    status = Column(String, default="open")  # open / won / lost / cancelled / expired
    result_r = Column(Float, nullable=True)  # realised R multiple once closed
    created_at = Column(DateTime, default=utcnow)
    closed_at = Column(DateTime, nullable=True)

    # --- added in v2 (auto-migrated) ---
    entry_interval = Column(String, nullable=True)
    trigger_ts = Column(Integer, nullable=True)     # open time (epoch s) of the trigger candle
    agreement = Column(Float, nullable=True)        # 0-100
    grade = Column(String, nullable=True)
    session_label = Column(String, nullable=True)
    risk_reward = Column(Float, nullable=True)
    checks_json = Column(Text, nullable=True)
    outcome_price = Column(Float, nullable=True)
    outcome = Column(String, nullable=True)         # auto / manual

    owner = relationship("User", back_populates="signals")


class Alert(Base):
    """In-app alert feed: new signals, take-profit hits (win) and stop-loss hits (loss)."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    kind = Column(String)               # signal / win / loss / info
    title = Column(String)
    message = Column(Text)
    symbol = Column(String, nullable=True)
    signal_id = Column(Integer, nullable=True)
    seen = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utcnow)


class BiasState(Base):
    """Last known 4H/1H bias per user+symbol, so we can detect a SHIFT (early-warning alert)
    the moment structure flips, well before all 4 steps line up into a full BUY/SELL signal."""
    __tablename__ = "bias_state"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    symbol = Column(String, index=True)
    bias_4h = Column(String, nullable=True)
    bias_1h = Column(String, nullable=True)
    aligned = Column(Boolean, nullable=True)     # was 1H matching 4H?
    updated_at = Column(DateTime, default=utcnow)


class ScanLog(Base):
    """One row per /signals/live evaluation (fired or not) - lets 'why hasn't this pair signalled
    today' be answered by looking at the log instead of guessing. Pruned to the most recent rows per
    user/symbol/entry_interval so it never grows unbounded."""
    __tablename__ = "scan_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    entry_interval = Column(String)
    ts = Column(DateTime, default=utcnow, index=True)
    status = Column(String)          # e.g. waiting_trigger, not_aligned, signal, ...
    headline = Column(Text)
    direction = Column(String, nullable=True)
    bias_4h = Column(String, nullable=True)
    bias_1h = Column(String, nullable=True)
    agreement = Column(Float, nullable=True)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    period = Column(String)
    entry_interval = Column(String)
    total_trades = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    win_rate = Column(Float)
    net_r = Column(Float)
    avg_r = Column(Float)
    equity_curve_json = Column(Text)     # JSON list of running R
    trades_json = Column(Text)           # JSON list of individual trades
    created_at = Column(DateTime, default=utcnow)

    # --- added in v2 ---
    stats_json = Column(Text, nullable=True)   # profit factor, drawdown, per-session breakdown, params

    owner = relationship("User", back_populates="backtests")
