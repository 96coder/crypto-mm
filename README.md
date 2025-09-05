Crypto Myanmar — Market Tracker & Signals

Overview
- Tracks top crypto prices using CoinGecko public API (no API key).
- Shows 7-day sparklines, 24h change, market cap and volume.
- Adds indicators and signals:
  - EMA: EMA(7), EMA(14) trend check
  - RSI: RSI(14) value
  - MACD: MACD(12,26,9) with histogram momentum
  - BB/ATR/StochRSI: Bollinger Bands %B/width, ATR% approx, StochRSI
  - Multi-timeframe: 1h base with 4h/daily EMA trend confluence
  - Combined signal: blends EMA trend, MACD relation/momentum, RSI/BB/ATR/StochRSI, higher-timeframe confluence; regression adds confidence and 24h forecast
- 1h TP/SL: Per-coin 1h targets using recent hourly ATR-like volatility
- Pure static site: open in a browser or serve as static files.

Guides
- A guides hub is available at `guides.html` with a sidebar list, search, and an article reader.
- The reader supports headings, lists, code blocks, links, blockquotes, and builds a simple in-page TOC.
- Add your `.md` files under `guide/` or `guides/`. A GitHub Action auto-builds `index.json` for the sidebar.
  - Script: `scripts/build_guides_index.py`
  - Workflow: `.github/workflows/guides.yml`
  - On each push (or manual run), it scans markdown, extracts the first `# Heading` as title and first paragraph as description, and writes `guide/index.json` or `guides/index.json`.
- The `guides.html` page tries `guide/index.json` first, then `guides/index.json`.

Troubleshooting (Guides)
- Serve over HTTP. Opening via file:// blocks fetch(). Run: `python3 -m http.server 5173` then open `http://localhost:5173/guides.html`.
- Build the index: `python3 scripts/build_guides_index.py` to create `guide/index.json` or `guides/index.json`.
- Fallback: If no index is found, the app tries `guide/README.md` or `guides/README.md`. If still blank, ensure your markdown files are inside `guide/` or `guides/` and named correctly.

Files
- `index.html`: Main page and controls (currency, search, refresh).
- `styles.css`: Dark-theme styling and card layout.
- `app.js`: Data fetching, rendering, sparkline drawing, and predictions.
  - Also computes 1h TP/SL using recent hourly ATR: TP1/TP2/TP3 at 0.5×/1.0×/1.5× ATR in the signal direction; SL at 1.0× ATR opposite.

Run Locally
Option 1: Open `index.html` directly in a browser.

Option 2 (recommended): Serve over a local web server for best compatibility.
- Python 3: `python3 -m http.server 5173`
- Then open: `http://localhost:5173/`

Notes on Data & Predictions
- Data source: CoinGecko `/coins/markets` endpoint with `sparkline=true`.
- Currency: You can switch `USD/MMK/EUR/SGD/JPY`. Availability depends on CoinGecko support for the chosen currency.
- Predictions: Heuristic and educational only — not financial advice.
  - EMA: Bullish bias if EMA(7) > EMA(14); bearish if opposite.
  - MACD: Bias from MACD vs Signal and histogram momentum (rising/falling).
  - RSI: Zone-based contribution (55–70 supportive, <45 weak, >70 overbought, <30 mean-reversion bias).
  - Forecast: Linear regression over ~last 60 sparkline points (≈ 60 hours) extrapolated ~24 hours.
  - Confidence: Blend of signal strength and regression R² (5–95%).
  - 1h TP/SL: Derived from recent hourly close-to-close ATR approximation; direction uses model probability if present (>55% long, <45% short), else combined signal.

Customize
- Change `PER_PAGE` in `app.js` to show more/less top coins.
- Style tweaks in `styles.css`.
- Tune indicator weights in `computeCombinedSignal` inside `app.js`.

Deploy to GitHub Pages
Option A: Use included GitHub Actions workflow (recommended)
- Push the repo to GitHub with the default branch named `main`.
- In GitHub, go to Settings → Pages and ensure Source is set to "GitHub Actions".
- The workflow at `.github/workflows/pages.yml` deploys on every push to `main`.

Option B: Deploy from branch (no Actions)
- In GitHub Settings → Pages, choose "Deploy from a branch" and select `main` and directory `/`.
- Pages will serve the static files from the repository root.

Predictions Pipeline (optional)
- Workflow: `.github/workflows/predict.yml` runs every 6 hours (and on manual dispatch).
- Script: `scripts/predict.py` builds a dataset per coin:
  - Data: Paginates Binance 1h klines to ~5000 bars (~200 days) for USDT pairs; gracefully falls back to CoinGecko hourly prices if missing.
  - Features: EMA(7/14/50), RSI(14), MACD(12/26/9), Bollinger width, ATR% (Wilder), ADX(14) with +DI/−DI, 24h momentum, volume change, plus BTC regime features (BTC EMA7/EMA14 trend, BTC 1h/24h momentum, relative momentum vs BTC, BTC ATR%).
  - Models:
    - Classifier: HistGradientBoostingClassifier, walk‑forward AUC; outputs `prob_up`, `exp_return`.
    - Quantile regressors: 1h forward return quantiles (q20/q50/q80) via GradientBoostingRegressor; outputs under `q1h` per coin.
  - Output: `predictions.json` with `prob_up`, `exp_return`, `auc`, optional `adx`/`+di`/`-di`, and `q1h`.
- Frontend: If `predictions.json` exists, the app shows model probability and expected 24h return per coin and blends it into the combined signal.
  - Also surfaces ADX and DI direction from predictions when available.
  - TP/SL: 1h TP/SL targets prefer quantile returns (`q1h`) when present; falls back to ATR bands otherwise.
- Tuning: Adjust TOP_N, MAX_BARS, and model hyperparams in the script as desired.

 Notes
 - ADX and ATR use true OHLC ranges; these require exchange OHLCV (Binance). If a symbol is unavailable, the script falls back gracefully.
 - The pipeline paginates Binance klines to increase history (~200 days by default).

Disclaimer
This project is for educational purposes only. No financial advice.

Support / Donations
- USDT (TRC20): `TLbwVrZyaZujcTCXAb94t6k7BrvChVfxzi`
Alerts (Telegram/Discord)
- The predictions workflow can post alerts when strong signals appear (default: prob_up ≥ 0.65 or ≤ 0.35).
- Add repo secrets/vars and the workflow will send alerts automatically:
  - Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` and/or `DISCORD_WEBHOOK_URL`
  - Optional repo variables: `ALERT_PROB_HIGH`, `ALERT_PROB_LOW` to tune thresholds
  - Messages include top 5 strongest signals with probability, expected return, and ADX if available
