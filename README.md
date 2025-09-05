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
- Pure static site: open in a browser or serve as static files.

Files
- `index.html`: Main page and controls (currency, search, refresh).
- `styles.css`: Dark-theme styling and card layout.
- `app.js`: Data fetching, rendering, sparkline drawing, and predictions.

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
- Script: `scripts/predict.py` fetches hourly prices for top 25 coins from CoinGecko and writes `predictions.json`.
- Frontend: If `predictions.json` exists, the app shows model probability and expected 24h return per coin and blends it into the signal.
- Tuning: The script uses a heuristic score mapped to probability; you can replace it with a trained model later.

Disclaimer
This project is for educational purposes only. No financial advice.
