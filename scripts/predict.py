#!/usr/bin/env python3
"""
Predictions pipeline with OHLCV and a trained classifier.

Data sources:
- CoinGecko: top coins list (id, symbol)
- Binance: hourly OHLCV (1h klines) for SYMBOLUSDT pairs

Model:
- Features from OHLCV: EMA, RSI, MACD, BB width, ATR%, ADX, momentum
- Label: forward 24h return > 0 (binary)
- Classifier: Logistic Regression with walk-forward validation (TimeSeriesSplit)

Output:
- predictions.json at repo root with per-coin prob_up and exp_return

Note: This runs on GitHub Actions (network available). Locally, it may fail without network.
"""
import json, math, time, urllib.request, urllib.error
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

CG_API = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com/api/v3"
VS = "usd"
TOP_N = 25
KLINES_LIMIT = 1000  # per request
MAX_BARS = 5000      # target bars (~208 days)
H_FWD = 24

def http_get_json(url: str):
    req = urllib.request.Request(url, headers={"accept":"application/json","user-agent":"cm-forecast/2.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def top_coins(n=TOP_N) -> List[Tuple[str,str]]:
    url = f"{CG_API}/coins/markets?vs_currency={VS}&order=market_cap_desc&per_page={n}&page=1&sparkline=false"
    data = http_get_json(url)
    return [(c["id"], c["symbol"].upper()) for c in data]

def binance_klines(symbol: str, interval: str = "1h", limit: int = KLINES_LIMIT, max_bars: int = MAX_BARS) -> Optional[Dict[str, List[float]]]:
    all_rows = []
    end_time = None
    while len(all_rows) < max_bars:
        url = f"{BINANCE}/klines?symbol={symbol}&interval={interval}&limit={limit}"
        if end_time is not None:
            url += f"&endTime={end_time}"
        try:
            arr = http_get_json(url)
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
        if not isinstance(arr, list) or not arr:
            break
        all_rows = arr + all_rows  # prepend older chunk
        # next page ends before earliest openTime
        end_time = arr[0][0] - 1
        if len(arr) < limit:
            break
    if not all_rows:
        return None
    o = [float(x[1]) for x in all_rows]
    h = [float(x[2]) for x in all_rows]
    l = [float(x[3]) for x in all_rows]
    c = [float(x[4]) for x in all_rows]
    v = [float(x[5]) for x in all_rows]
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}

def coingecko_prices(id: str, days: int = 90) -> Optional[List[float]]:
    url = f"{CG_API}/coins/{id}/market_chart?vs_currency={VS}&days={days}&interval=hourly"
    try:
        data = http_get_json(url)
    except Exception:
        return None
    prices = [p[1] for p in data.get("prices", [])]
    return prices if prices else None

# Indicator helpers (numpy-based)
def ema(arr: np.ndarray, period: int) -> np.ndarray:
    if arr.size == 0:
        return np.array([])
    k = 2 / (period + 1)
    out = np.zeros_like(arr)
    out[:] = np.nan
    seed = np.nanmean(arr[:period]) if arr.size >= period else np.nanmean(arr)
    out[period - 1] = seed
    for i in range(period, len(arr)):
        prev = out[i - 1]
        out[i] = arr[i] * k + prev * (1 - k)
    return out

def rsi(arr: np.ndarray, period: int = 14) -> np.ndarray:
    if arr.size < period + 1:
        return np.full_like(arr, np.nan)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.zeros_like(arr); al = np.zeros_like(arr)
    ag[:] = np.nan; al[:] = np.nan
    ag[period] = gains[:period].mean()
    al[period] = losses[:period].mean()
    for i in range(period + 1, len(arr)):
        g = gains[i - 1]
        l = losses[i - 1]
        ag[i] = (ag[i - 1] * (period - 1) + g) / period
        al[i] = (al[i - 1] * (period - 1) + l) / period
    rs = ag / np.where(al == 0, np.nan, al)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def macd(arr: np.ndarray, fast=12, slow=26, signal=9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ef = ema(arr, fast)
    es = ema(arr, slow)
    macd_line = ef - es
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bb_width(arr: np.ndarray, period=20, k=2) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        s = arr[i - period + 1:i + 1]
        m = np.nanmean(s)
        sd = np.nanstd(s)
        if m:
            out[i] = (2 * k * sd) / m
    return out

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    # Wilder smoothing
    atr = np.zeros_like(close); atr[:] = np.nan
    atr[period] = np.nanmean(tr[:period])
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus_dm = np.zeros_like(close)
    minus_dm = np.zeros_like(close)
    for i in range(1, len(close)):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
    tr = np.zeros_like(close)
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    # Wilder's smoothing for TR and DMs
    atr_w = np.zeros_like(close); atr_w[:] = np.nan
    plus_dm_s = np.zeros_like(close); plus_dm_s[:] = np.nan
    minus_dm_s = np.zeros_like(close); minus_dm_s[:] = np.nan
    atr_w[period] = np.nanmean(tr[1:period+1])
    plus_dm_s[period] = np.nanmean(plus_dm[1:period+1])
    minus_dm_s[period] = np.nanmean(minus_dm[1:period+1])
    for i in range(period + 1, len(close)):
        atr_w[i] = (atr_w[i - 1] * (period - 1) + tr[i]) / period
        plus_dm_s[i] = (plus_dm_s[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_dm_s[i] = (minus_dm_s[i - 1] * (period - 1) + minus_dm[i]) / period
    plus_di = 100 * (plus_dm_s / atr_w)
    minus_di = 100 * (minus_dm_s / atr_w)
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = np.zeros_like(close); adx[:] = np.nan
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])
    for i in range(2*period + 1, len(close)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, plus_di, minus_di

def build_features(ohlcv: Dict[str, List[float]], btc_ohlcv: Optional[Dict[str, List[float]]] = None):
    c = np.array(ohlcv["close"], dtype=float)
    h = np.array(ohlcv["high"], dtype=float)
    l = np.array(ohlcv["low"], dtype=float)
    v = np.array(ohlcv["volume"], dtype=float)
    # Align with BTC history if provided (use common tail length)
    if btc_ohlcv is not None and "close" in btc_ohlcv:
        bc = np.array(btc_ohlcv["close"], dtype=float)
        bh = np.array(btc_ohlcv["high"], dtype=float)
        bl = np.array(btc_ohlcv["low"], dtype=float)
        m = int(min(len(c), len(bc)))
        if m >= 100:
            c = c[-m:]; h = h[-m:]; l = l[-m:]; v = v[-m:]
            bc = bc[-m:]; bh = bh[-m:]; bl = bl[-m:]
        else:
            bc = None; bh = None; bl = None
    else:
        bc = None; bh = None; bl = None
    ema7 = ema(c, 7)
    ema14 = ema(c, 14)
    ema50 = ema(c, 50)
    rsi14 = rsi(c, 14)
    macd_line, macd_sig, macd_hist = macd(c, 12, 26, 9)
    bbw = bb_width(c, 20, 2)
    atr14 = atr(h, l, c, 14)
    adx14, plus_di, minus_di = adx(h, l, c, 14)
    mom24 = np.concatenate(([np.nan]*H_FWD, c[H_FWD:] / c[:-H_FWD] - 1))
    vol_change = np.concatenate(([np.nan], v[1:] / np.where(v[:-1]==0, np.nan, v[:-1]) - 1))
    cols = [
        (c / ema7) - 1,
        (ema7 / ema14) - 1,
        (ema14 / ema50) - 1,
        rsi14 / 100,
        macd_line, macd_sig, macd_hist,
        bbw,
        atr14 / c,
        adx14 / 100,
        plus_di / 100, minus_di / 100,
        vol_change,
    ]
    # BTC regime features if available
    if bc is not None:
        btc_ema7 = ema(bc, 7); btc_ema14 = ema(bc, 14)
        btc_trend = (btc_ema7 / btc_ema14) - 1
        btc_mom24 = np.concatenate(([np.nan]*H_FWD, bc[H_FWD:] / bc[:-H_FWD] - 1))
        btc_atr14 = atr(bh, bl, bc, 14) / bc
        rel_mom24 = mom24 - btc_mom24
        cols.extend([btc_trend, btc_mom24, rel_mom24, btc_atr14])
    X = np.column_stack(cols)
    # Label: forward 24h return sign
    y = np.where(mom24 > 0, 1, 0)
    # valid mask: no NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    ret24 = mom24[mask]
    # last index mapping for later
    last_idx = mask.nonzero()[0][-1] if mask.any() else None
    return X, y, ret24, last_idx, {
        "adx": float(adx14[~np.isnan(adx14)][-1]) if np.any(~np.isnan(adx14)) else None,
        "+di": float(plus_di[~np.isnan(plus_di)][-1]) if np.any(~np.isnan(plus_di)) else None,
        "-di": float(minus_di[~np.isnan(minus_di)][-1]) if np.any(~np.isnan(minus_di)) else None,
    }

def train_and_predict(X: np.ndarray, y: np.ndarray, ret24: np.ndarray) -> Tuple[float, float, float]:
    if len(y) < 200:
        # not enough data for walk-forward; fallback simple model
        clf = HistGradientBoostingClassifier(
            max_depth=7,
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
        )
        clf.fit(X, y)
        prob = float(clf.predict_proba(X[-1:])[0, 1])
        auc = float('nan')
        med_abs = float(np.nanmedian(np.abs(ret24))) if ret24.size else 0.02
        exp_ret = (2 * prob - 1) * med_abs
        return prob, exp_ret, auc
    tscv = TimeSeriesSplit(n_splits=5)
    probs = np.zeros_like(y, dtype=float)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        clf = HistGradientBoostingClassifier(
            max_depth=7,
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.1,
        )
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict_proba(X[test_idx])[:, 1]
        probs[test_idx] = p
        try:
            aucs.append(roc_auc_score(y[test_idx], p))
        except ValueError:
            pass
    auc = float(np.nanmean(aucs)) if aucs else float('nan')
    # fit on all and predict last
    clf = HistGradientBoostingClassifier(
        max_depth=7,
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X, y)
    prob = float(clf.predict_proba(X[-1:])[0, 1])
    med_abs = float(np.nanmedian(np.abs(ret24))) if ret24.size else 0.02
    exp_ret = float((2 * prob - 1) * med_abs)
    return prob, exp_ret, auc

def fallback_prices_predict(prices: List[float]):
    # Simple heuristic fallback if Binance data unavailable
    if len(prices) < 60:
        return None
    last = prices[-1]
    # slope via linear regression
    n = min(60, len(prices))
    s = prices[-n:]
    x = np.arange(n)
    mx = x.mean(); my = np.mean(s)
    num = float(((x - mx) * (np.array(s) - my)).sum())
    den = float(((x - mx) ** 2).sum())
    slope = 0.0 if den == 0 else num / den
    slope24 = slope * 24
    score = 0.6 if slope24 > 0 else -0.6
    prob = 1 / (1 + math.exp(-score))
    exp_return = slope24 / last if last else 0.0
    return {"prob_up": float(prob), "exp_return": float(exp_return), "auc": None}

def main():
    coins = top_coins(TOP_N)
    # Fetch BTC regime data once
    btc_ohlcv = binance_klines("BTCUSDT")
    out = {"as_of": int(time.time()), "horizon_hours": H_FWD, "coins": {}}
    for cid, sym in coins:
        symbol = f"{sym}USDT"
        ohlcv = binance_klines(symbol)
        if ohlcv:
            try:
                X, y, ret24, last_idx, extras = build_features(ohlcv, btc_ohlcv)
                if len(y) >= 50:
                    prob, exp_ret, auc = train_and_predict(X, y, ret24)
                    coin_entry = {"prob_up": float(prob), "exp_return": float(exp_ret), "auc": auc}
                    if extras.get("adx") is not None:
                        coin_entry.update({"adx": extras.get("adx"), "+di": extras.get("+di"), "-di": extras.get("-di")})
                    out["coins"][cid] = coin_entry
                    continue
            except Exception:
                pass
        # fallback to CoinGecko prices
        prices = coingecko_prices(cid, days=30)
        fb = fallback_prices_predict(prices or [])
        if fb:
            out["coins"][cid] = fb
    with open("predictions.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))

if __name__ == "__main__":
    main()
