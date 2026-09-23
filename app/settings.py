"""Persistent user-level data settings (single-user tool -> one JSON file next to the DB).

anchor  : how 4H / 1D / 1W candles are cut so they line up with your MetaTrader 5 chart
            "ny_close" -> MT5 servers on GMT+2 (winter) / GMT+3 (summer): day starts 17:00 New York
                          (this is what most brokers use)
            "utc"      -> servers on GMT+0 (day starts 00:00 UTC)
offsets : per-symbol price shift added to every candle so Yahoo prices match your broker's quote
            (e.g. futures-vs-spot difference). Set it from the app: Profile -> Match MetaTrader 5.
"""
from __future__ import annotations

import json
import os
import threading

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "settings.json")
_lock = threading.Lock()
_state: dict | None = None
DEFAULTS = {"anchor": "ny_close", "offsets": {}, "offset_set_at": {}}


def _load() -> dict:
    global _state
    if _state is None:
        try:
            with open(_PATH) as f:
                _state = {**DEFAULTS, **json.load(f)}
        except Exception:  # noqa: BLE001
            _state = json.loads(json.dumps(DEFAULTS))
    return _state


def get() -> dict:
    with _lock:
        return json.loads(json.dumps(_load()))


def _save() -> None:
    try:
        with open(_PATH, "w") as f:
            json.dump(_state, f, indent=2)
    except OSError:
        pass


def anchor() -> str:
    with _lock:
        return _load().get("anchor", "ny_close")


def set_anchor(value: str) -> None:
    with _lock:
        _load()["anchor"] = "utc" if value == "utc" else "ny_close"
        _save()


def offset(symbol: str) -> float:
    with _lock:
        return float(_load().get("offsets", {}).get(symbol, 0.0))


def set_offset(symbol: str, value: float | None, at: int | None = None) -> None:
    with _lock:
        st = _load()
        if not value:
            st["offsets"].pop(symbol, None)
            st["offset_set_at"].pop(symbol, None)
        else:
            st["offsets"][symbol] = float(value)
            st["offset_set_at"][symbol] = at
        _save()
