# goldsignal_api

## v4 changes

- **Scalp mode** follows the 1H (falls back to the 4H) instead of only firing when 1H and 4H disagree. Against the 4H it needs one extra confirmation. It may also use the entry timeframe's own order blocks / gaps.
- **Entry sensitivity** (`sens`: strict / normal / high) in Profile, live scanner, replay and backtest. `strict` = old v3 rules. Switches live in `strategy.engine_flags()`.
- **Catch-up + background scanner**: every closed candle since the last scan is evaluated (`run_scan`), signals that had already resolved are stored silently, and `bg_scanner.py` keeps scanning for 3 days after the last page visit (`GOLDSIGNAL_BG_SCAN=0` disables). A `setup` alert fires when a zone is tapped and a trigger is pending.
- **Backtest = live engine**: same flags, daily/weekly levels, `trades.replay` rules (stop wins a both-hit candle), entry on the next candle's open, spread charged in R.
- **Candle store** (`cache/candles.db`, `candle_store.py`): every closed candle is kept, history grows beyond Yahoo's window. `clean_candles()` aligns, fills holes, fixes OHLC and snaps tiny open gaps on <=30m. `merge_live` no longer shrinks candles.
- **Chart**: scroll back loads older pages (`/market/candles?before=`), `/signals/replay` draws past signals, outcomes and near-misses.
