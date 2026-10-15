# goldsignal_api

## v4 changes

- **Scalp mode** follows the 1H (falls back to the 4H) instead of only firing when 1H and 4H disagree. Against the 4H it needs one extra confirmation. It may also use the entry timeframe's own order blocks / gaps.
- **Entry sensitivity** (`sens`: strict / normal / high) in Profile, live scanner, replay and backtest. `strict` = old v3 rules. Switches live in `strategy.engine_flags()`.
- **Catch-up + background scanner**: every closed candle since the last scan is evaluated (`run_scan`), signals that had already resolved are stored silently, and `bg_scanner.py` keeps scanning for 3 days after the last page visit (`GOLDSIGNAL_BG_SCAN=0` disables). A `setup` alert fires when a zone is tapped and a trigger is pending.
- **Backtest = live engine**: same flags, daily/weekly levels, `trades.replay` rules (stop wins a both-hit candle), entry on the next candle's open, spread charged in R.
- **Candle store** (`cache/candles.db`, `candle_store.py`): every closed candle is kept, history grows beyond Yahoo's window. `clean_candles()` aligns, fills holes, fixes OHLC and snaps tiny open gaps on <=30m. `merge_live` no longer shrinks candles.
- **Chart**: scroll back loads older pages (`/market/candles?before=`), `/signals/replay` draws past signals, outcomes and near-misses.

## v5 notes - fewer losses, news protection

**Losses / small wins.** The old engine took any setup with a wall 1.0R away (target ~1.1R), scalps aimed at 1.5R, and after a
stop-out the same idea re-entered a few candles later. Now:
* **Room gate** - the next real wall (1H/4H zone, liquidity pool, daily/weekly level) must be >= 1.5R away (1.2R for short R:R); otherwise the status is `no_room` instead of a tiny win.
* **Scalp target 2.0R** (was 1.5R). **Chop gate** (efficiency ratio over 40 bars < 0.07 = no entries). **Cost gate** (spread must be <= 0.15R of the stop).
* **Cool-down** - after a stop-out no new entry in the same direction for ~6 bars (live tracker and backtest).
* Backtest page has **"Why trades lose"**: win rate needed to break even, losers that were once +0.5R, weak buckets (agreement / pattern / zone / session) and hints.
* Numbers on the bundled demo data are synthetic (random walk): they check the mechanics, not profitability. Judge it with a backtest on real Yahoo data.

**News.** `app/news.py` + `/news/*`: ForexFactory weekly calendar (this + next week, refreshed every 30 min, stored in `news_events`),
a built-in schedule for NFP / FOMC / weekly claims (so backtests over past weeks still see them), manual events, RSS headlines.
`GOLDSIGNAL_NEWS=0` disables the refresher. Hard **news** gate: no new signals from 30 min before to 20 min after a high-impact release for the
pair's currencies (CPI / NFP / FOMC / rate decisions x1.5). Settings: News page -> News protection. The same windows feed the live scan,
the chart replay and the backtest. Alert "News: ..." 15 min before a release. Charts: the **News** layer (line, blackout band, price reaction).

## v6 - installable PWA + background push

The frontend is a full PWA: `manifest.webmanifest`, icons (incl. maskable + Apple touch), `sw.js` (offline app shell, "new version ready" banner;
market data is never cached), install button (top bar + Profile -> "Install as an app", iPhone steps included).

**Install needs https** (or `localhost` on the same computer):
* Deploy: frontend on Vercel (`frontend/`, `npm run build`, env `VITE_API_URL=https://<your-backend>`), backend on Railway / a VPS
  (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`). Mount a volume for `goldsignal.db` and `cache/` (SQLite is wiped on redeploy otherwise).
* Try it on a phone from your PC: `cloudflared tunnel --url http://localhost:5173` (or ngrok) gives an https URL.

**Background alerts:** Profile -> Install as an app -> "Background alerts on this device". The backend encrypts and sends Web Push itself
(`app/push.py`, no extra package) for every alert: signals, TP/SL, setups, news. The VAPID key is created on first start in `backend/vapid_private.pem`;
on hosts with a wiped disk set `GOLDSIGNAL_VAPID_PEM` (PEM text, newlines as `\n`) so subscriptions survive redeploys.
`GOLDSIGNAL_PUSH_SUBJECT=mailto:you@yourdomain.com` sets the contact; `GOLDSIGNAL_PUSH=0` turns sending off. iPhone: only after "Add to Home Screen" (iOS 16.4+).
The server keeps scanning with the browser closed (`bg_scanner.py`), so alerts keep coming while the backend runs.
