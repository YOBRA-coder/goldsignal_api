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
