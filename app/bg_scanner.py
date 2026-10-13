"""
Background scanner.

The old app only looked for signals when a browser tab polled /signals/live. Close the tab (or let the laptop sleep)
and every trigger that printed meanwhile was simply never evaluated - and signals that appeared between two polls
were only ever seen as previews. This thread keeps scanning every pair the user has looked at in the last few days,
on the same code path as the page (signals_router.run_scan), so signals, setup alerts and win / loss tracking carry on
while nobody is watching.

Environment:  GOLDSIGNAL_BG_SCAN=0  turns it off   |   GOLDSIGNAL_SCAN_SECONDS=45  changes the interval
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import timedelta

from . import models, sessions
from .data_fetcher import DataUnavailable, is_futures
from .database import SessionLocal
from .models import utcnow

log = logging.getLogger("goldsignal.bgscan")
INTERVAL = max(15, int(os.environ.get("GOLDSIGNAL_SCAN_SECONDS", "45")))
ACTIVE_DAYS = 3          # keep scanning for a pair that a page asked about within this many days
_started = False


def tick() -> int:
    """One pass over every active (user, pair, entry timeframe). Returns how many scans ran."""
    from .routers import signals_router as sr
    db = SessionLocal()
    n = 0
    try:
        rows = db.query(models.ScanState).filter(
            models.ScanState.seen_at >= utcnow() - timedelta(days=ACTIVE_DAYS)).all()
        done: set[tuple] = set()
        for r in rows:
            key = (r.user_id, r.symbol, r.entry_interval)
            if key in done:
                continue
            done.add(key)
            if not sessions.market_open(None, is_futures(r.symbol)):
                continue                                   # weekend / daily break: nothing can happen, spare the feed
            user = db.get(models.User, r.user_id)
            if user is None:
                continue
            p = json.loads(r.params_json or "{}")
            try:
                sr.run_scan(db, user, r.symbol, r.entry_interval, float(p.get("min_agreement", 70)),
                            bool(p.get("sessions_only", True)), float(p.get("rr", 2.0)), float(p.get("breakeven_at_r", 1.0)),
                            bool(p.get("scalp", False)), p.get("sens", "normal"), True, background=True)
                n += 1
            except DataUnavailable:
                pass
            except Exception:  # noqa: BLE001 - one bad pair must not stop the loop
                log.exception("background scan failed for %s %s", r.symbol, r.entry_interval)
                db.rollback()
    finally:
        db.close()
    return n


def _loop() -> None:
    time.sleep(20)                                          # let the server finish starting
    while True:
        try:
            tick()
        except Exception:  # noqa: BLE001
            log.exception("background scanner tick failed")
        time.sleep(INTERVAL)


def start() -> None:
    global _started
    if _started or os.environ.get("GOLDSIGNAL_BG_SCAN", "1") == "0":
        return
    _started = True
    threading.Thread(target=_loop, name="goldsignal-bg-scan", daemon=True).start()
    log.info("background scanner started (every %ss)", INTERVAL)
