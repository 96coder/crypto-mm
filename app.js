// Crypto Myanmar — Market tracker with TA indicators (RSI, MACD, EMA, BB, ATR, StochRSI)
// Data: CoinGecko public API (no key). Not financial advice.

const API_BASE = 'https://api.coingecko.com/api/v3';
const DEFAULT_VS = 'usd';
const PER_PAGE = 20; // top coins by market cap

const els = {
  status: document.getElementById('statusText'),
  cards: document.getElementById('cards'),
  currency: document.getElementById('currency'),
  search: document.getElementById('search'),
  refreshBtn: document.getElementById('refreshBtn'),
  autorefresh: document.getElementById('autorefresh'),
};

let state = {
  vs: DEFAULT_VS,
  markets: [],
  filter: '',
  timer: null,
  predictions: null,
};

function fmtNumber(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const abs = Math.abs(x);
  if (abs >= 1e12) return (x / 1e12).toFixed(2) + 'T';
  if (abs >= 1e9) return (x / 1e9).toFixed(2) + 'B';
  if (abs >= 1e6) return (x / 1e6).toFixed(2) + 'M';
  if (abs >= 1e3) return (x / 1e3).toFixed(2) + 'K';
  return Number(x).toFixed(digits);
}

function fmtCurrency(x, vs) {
  const symbolMap = { usd: '$', eur: '€', sgd: 'S$', jpy: '¥', mmk: 'Ks' };
  const sym = symbolMap[vs] || '';
  // Dynamic precision: more decimals for very small prices
  let digits = 2;
  if (x > 0 && x < 1) digits = 4;
  if (x > 0 && x < 0.01) digits = 6;
  return sym + Number(x).toFixed(digits);
}

async function fetchMarkets(vs) {
  const url = `${API_BASE}/coins/markets?vs_currency=${encodeURIComponent(vs)}&order=market_cap_desc&per_page=${PER_PAGE}&page=1&sparkline=true&price_change_percentage=1h,24h,7d`;
  const res = await fetch(url, { headers: { 'accept': 'application/json' } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// Math utils for predictions
function sma(values, window) {
  if (!values || values.length < window) return null;
  let sum = 0;
  for (let i = values.length - window; i < values.length; i++) sum += values[i];
  return sum / window;
}

function emaLast(values, period) {
  if (!values || values.length === 0) return null;
  const k = 2 / (period + 1);
  const seedLen = Math.min(period, values.length);
  let ema = values.slice(0, seedLen).reduce((a, b) => a + b, 0) / seedLen;
  for (let i = seedLen; i < values.length; i++) {
    ema = values[i] * k + ema * (1 - k);
  }
  return ema;
}

function rsiLast(values, period = 14) {
  if (!values || values.length < period + 1) return null;
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = values[i] - values[i - 1];
    if (diff >= 0) gains += diff; else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  for (let i = period + 1; i < values.length; i++) {
    const diff = values[i] - values[i - 1];
    const gain = Math.max(0, diff);
    const loss = Math.max(0, -diff);
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

function macdLast(values, fast = 12, slow = 26, signalPeriod = 9) {
  if (!values || values.length < slow + signalPeriod) return null;
  const n = values.length;
  // EMA fast series
  const kFast = 2 / (fast + 1);
  const emaFastSeries = Array(n).fill(null);
  let seedFast = values.slice(0, fast).reduce((a, b) => a + b, 0) / fast;
  emaFastSeries[fast - 1] = seedFast;
  for (let i = fast; i < n; i++) {
    emaFastSeries[i] = values[i] * kFast + emaFastSeries[i - 1] * (1 - kFast);
  }
  // EMA slow series
  const kSlow = 2 / (slow + 1);
  const emaSlowSeries = Array(n).fill(null);
  let seedSlow = values.slice(0, slow).reduce((a, b) => a + b, 0) / slow;
  emaSlowSeries[slow - 1] = seedSlow;
  for (let i = slow; i < n; i++) {
    emaSlowSeries[i] = values[i] * kSlow + emaSlowSeries[i - 1] * (1 - kSlow);
  }
  const macdSeries = [];
  for (let i = slow - 1; i < n; i++) {
    const ef = emaFastSeries[i];
    const es = emaSlowSeries[i];
    if (ef != null && es != null) macdSeries.push(ef - es);
  }
  if (macdSeries.length < signalPeriod + 1) return null;
  const kSig = 2 / (signalPeriod + 1);
  let signal = macdSeries.slice(0, signalPeriod).reduce((a, b) => a + b, 0) / signalPeriod;
  for (let i = signalPeriod; i < macdSeries.length; i++) {
    signal = macdSeries[i] * kSig + signal * (1 - kSig);
  }
  const macd = macdSeries[macdSeries.length - 1];
  // previous signal by rolling one step earlier
  let prevSignal = macdSeries.slice(0, signalPeriod).reduce((a, b) => a + b, 0) / signalPeriod;
  for (let i = signalPeriod; i < macdSeries.length - 1; i++) {
    prevSignal = macdSeries[i] * kSig + prevSignal * (1 - kSig);
  }
  const prevMacd = macdSeries[macdSeries.length - 2];
  const hist = macd - signal;
  const prevHist = prevMacd - prevSignal;
  return { macd, signal, hist, prevHist };
}

function stddevLast(values, period) {
  if (!values || values.length < period) return null;
  const slice = values.slice(-period);
  const m = slice.reduce((a,b)=>a+b,0) / period;
  const v = slice.reduce((a,b)=>a + (b - m) * (b - m), 0) / period;
  return Math.sqrt(v);
}

function bbLast(values, period = 20, k = 2) {
  const mid = sma(values, period);
  const sd = stddevLast(values, period);
  if (mid == null || sd == null) return null;
  const upper = mid + k * sd;
  const lower = mid - k * sd;
  const last = values[values.length - 1];
  const pctB = upper === lower ? 0.5 : (last - lower) / (upper - lower);
  const width = (upper - lower) / mid;
  return { mid, upper, lower, pctB, width };
}

function atrApproxLast(values, period = 14) {
  if (!values || values.length < period + 1) return null;
  let sum = 0;
  for (let i = values.length - period; i < values.length; i++) {
    sum += Math.abs(values[i] - values[i - 1]);
  }
  return sum / period;
}

function rsiSeries(values, period = 14) {
  if (!values || values.length < period + 1) return [];
  const out = [];
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const d = values[i] - values[i - 1];
    if (d >= 0) gains += d; else losses -= d;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  out.push(100 - 100 / (1 + (avgLoss === 0 ? 1e9 : avgGain / avgLoss)));
  for (let i = period + 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    const gain = Math.max(0, d);
    const loss = Math.max(0, -d);
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    out.push(100 - 100 / (1 + (avgLoss === 0 ? 1e9 : avgGain / avgLoss)));
  }
  return out;
}

function stochRsiLast(values, period = 14, smoothK = 3, smoothD = 3) {
  const rsi = rsiSeries(values, period);
  if (!rsi.length) return null;
  const window = Math.min(period, rsi.length);
  const base = rsi.slice(-window);
  const minR = Math.min(...base);
  const maxR = Math.max(...base);
  const lastRsi = rsi[rsi.length - 1];
  const raw = maxR === minR ? 0.5 : (lastRsi - minR) / (maxR - minR);
  const kVals = [raw];
  for (let i = 2; i <= smoothK; i++) kVals.push(raw);
  const k = kVals.reduce((a,b)=>a+b,0) / kVals.length;
  const dVals = [k];
  for (let i = 2; i <= smoothD; i++) dVals.push(k);
  const d = dVals.reduce((a,b)=>a+b,0) / dVals.length;
  return { k: k * 100, d: d * 100 };
}

function downsample(values, factor) {
  if (!values || values.length < factor) return values || [];
  const out = [];
  for (let i = values.length % factor; i < values.length; i += factor) {
    const slice = values.slice(i, i + factor);
    if (slice.length) out.push(slice.reduce((a,b)=>a+b,0) / slice.length);
  }
  return out;
}

function linearRegression(y) {
  // x = 0..n-1
  const n = y.length;
  if (n < 2) return { slope: 0, intercept: y[n - 1] || 0, r2: 0 };
  const x = [...Array(n).keys()];
  const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const mx = mean(x);
  const my = mean(y);
  let num = 0, den = 0, ssTot = 0, ssReg = 0, ssRes = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    num += dx * (y[i] - my);
    den += dx * dx;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = my - slope * mx;
  for (let i = 0; i < n; i++) {
    const yi = y[i];
    const fi = slope * x[i] + intercept;
    ssTot += (yi - my) ** 2;
    ssReg += (fi - my) ** 2;
    ssRes += (yi - fi) ** 2;
  }
  const r2 = ssTot === 0 ? 0 : 1 - (ssRes / ssTot);
  return { slope, intercept, r2 };
}

function computeSignalFromSparkline(prices) {
  // prices: last 7d hourly samples (~168 points)
  if (!prices || prices.length < 20) return {
    label: 'Neutral', cls: 'neutral', confidence: 0.3, forecast: null
  };
  const last = prices[prices.length - 1];
  const s7 = sma(prices, 7);
  const s14 = sma(prices, 14);
  const { slope, r2 } = linearRegression(prices.slice(-60)); // last ~60h
  const slope24h = slope * 24; // approximate 24h forward
  const forecast = last + slope24h;
  let label = 'Neutral', cls = 'neutral';
  const trendUp = s7 && s14 && s7 > s14 && slope > 0;
  const trendDown = s7 && s14 && s7 < s14 && slope < 0;
  if (trendUp) { label = 'Bullish'; cls = 'bull'; }
  else if (trendDown) { label = 'Bearish'; cls = 'bear'; }
  const confidence = Math.max(0.05, Math.min(0.95, 0.4 + 0.6 * r2));
  return { label, cls, confidence, forecast };
}

function computeIndicators(prices) {
  if (!prices || prices.length < 30) return null;
  const ema7 = emaLast(prices, 7);
  const ema14 = emaLast(prices, 14);
  const ema20 = emaLast(prices, 20);
  const rsi = rsiLast(prices, 14);
  const macd = macdLast(prices, 12, 26, 9);
  const bb = bbLast(prices, 20, 2);
  const atr = atrApproxLast(prices, 14);
  const stoch = stochRsiLast(prices, 14, 3, 3);
  const prices4h = downsample(prices, 4);
  const prices1d = downsample(prices, 24);
  const ema7_4h = emaLast(prices4h, 7), ema14_4h = emaLast(prices4h, 14);
  const ema7_1d = emaLast(prices1d, 7), ema14_1d = emaLast(prices1d, 14);
  return { ema7, ema14, ema20, rsi, macd, bb, atr, stoch, ema7_4h, ema14_4h, ema7_1d, ema14_1d };
}

function computeCombinedSignal(prices) {
  const ind = computeIndicators(prices);
  if (!ind) {
    const basic = computeSignalFromSparkline(prices);
    return { ...basic, indicators: null, score: 0 };
  }
  const { ema7, ema14, rsi, macd, bb, atr, stoch, ema7_4h, ema14_4h, ema7_1d, ema14_1d } = ind;
  let score = 0;
  if (ema7 != null && ema14 != null) score += ema7 > ema14 ? 0.5 : -0.5;
  if (macd) score += macd.macd > macd.signal ? 0.4 : -0.4;
  if (macd) score += macd.hist > macd.prevHist ? 0.2 : -0.2;
  if (rsi != null) {
    if (rsi >= 55 && rsi <= 70) score += 0.3; else if (rsi <= 45 && rsi >= 30) score -= 0.3;
    if (rsi > 70) score -= 0.1;
    if (rsi < 30) score += 0.1;
  }
  if (bb) {
    if (bb.pctB > 0.8) score -= 0.1;
    if (bb.pctB < 0.2) score += 0.1;
    if (bb.width < 0.04) score -= 0.05;
  }
  if (atr != null && prices.length) {
    const last = prices[prices.length - 1];
    const atrPct = atr / last;
    if (atrPct < 0.01) score -= 0.05;
    if (atrPct > 0.03) score += 0.05;
  }
  if (stoch) {
    if (stoch.k > 80 && stoch.d > 80) score -= 0.1;
    if (stoch.k < 20 && stoch.d < 20) score += 0.1;
  }
  let confluence = 0;
  if (ema7_4h != null && ema14_4h != null) confluence += ema7_4h > ema14_4h ? 0.15 : -0.15;
  if (ema7_1d != null && ema14_1d != null) confluence += ema7_1d > ema14_1d ? 0.2 : -0.2;
  score += confluence;
  score = Math.max(-1, Math.min(1, score));
  let label = 'Neutral', cls = 'neutral';
  if (score >= 0.5) { label = 'Bullish'; cls = 'bull'; }
  else if (score <= -0.5) { label = 'Bearish'; cls = 'bear'; }
  const { slope, r2 } = linearRegression(prices.slice(-60));
  const last = prices[prices.length - 1];
  const forecast = last + slope * 24;
  const confidence = Math.max(0.05, Math.min(0.95, 0.5 * Math.abs(score) + 0.5 * r2));
  return { label, cls, confidence, forecast, indicators: ind, score };
}

function drawSparkline(canvas, data, color) {
  if (!canvas || !data || data.length < 2) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const pad = 6 * devicePixelRatio;
  const sx = (w - pad * 2) / (data.length - 1);
  const sy = max === min ? 0 : (h - pad * 2) / (max - min);
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.strokeStyle = color;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad + i * sx;
    const y = h - pad - (data[i] - min) * sy;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function render() {
  const q = state.filter.trim().toLowerCase();
  const items = state.markets.filter(c => {
    if (!q) return true;
    return (
      c.name.toLowerCase().includes(q) ||
      c.symbol.toLowerCase().includes(q)
    );
  });
  els.cards.innerHTML = '';
  for (const c of items) {
    const card = document.createElement('article');
    card.className = 'card';
    const chg = c.price_change_percentage_24h;
    const chgCls = chg >= 0 ? 'pos' : 'neg';
    const spark = (c.sparkline_in_7d && c.sparkline_in_7d.price) || [];
    let sig = computeCombinedSignal(spark);
    const pred = state.predictions?.coins?.[c.id];
    if (pred) {
      const prob = pred.prob_up;
      const expRet = pred.exp_return;
      const label = prob > 0.6 ? 'Bullish' : prob < 0.4 ? 'Bearish' : 'Neutral';
      const cls = label === 'Bullish' ? 'bull' : label === 'Bearish' ? 'bear' : 'neutral';
      const forecast = c.current_price * (1 + expRet);
      const confidence = Math.max(sig.confidence, Math.abs(prob - 0.5) * 2 * 0.8 + 0.2);
      sig = { ...sig, label, cls, forecast, confidence, model: { prob, expRet } };
    }
    const forecastPct = sig.forecast ? ((sig.forecast - c.current_price) / c.current_price) * 100 : null;
    card.innerHTML = `
      <div class="row">
        <div class="coin">
          <img src="${c.image}" alt="${c.symbol}" loading="lazy" />
          <div>
            <div class="name">${c.name}</div>
            <div class="sym">${c.symbol}</div>
          </div>
        </div>
        <div class="price">${fmtCurrency(c.current_price, state.vs)}</div>
      </div>
      <div class="row">
        <div class="meta">MC: ${fmtNumber(c.market_cap)} • Vol: ${fmtNumber(c.total_volume)}</div>
        <div class="chg ${chgCls}">${chg?.toFixed(2) ?? '—'}%</div>
      </div>
      <canvas class="spark"></canvas>
      <div class="predict">
        <div class="signal">
          <span class="pill ${sig.cls}">${sig.label}</span>
          <span class="small">Conf: ${(sig.confidence * 100).toFixed(0)}%</span>
        </div>
        <div class="small">${sig.forecast ? `24h: ${fmtCurrency(sig.forecast, state.vs)} (${forecastPct.toFixed(1)}%)` : '24h: —'}</div>
      </div>
      <div class="indicators">
        <div title="Exponential Moving Averages">EMA7/14: <span class="${sig.indicators && sig.indicators.ema7 > sig.indicators.ema14 ? 'pos' : 'neg'}">${sig.indicators ? (sig.indicators.ema7 > sig.indicators.ema14 ? '↑' : '↓') : '—'}</span></div>
        <div title="Relative Strength Index">RSI14: <strong>${sig.indicators?.rsi ? sig.indicators.rsi.toFixed(0) : '—'}</strong></div>
        <div title="MACD Histogram">MACD: <span class="${sig.indicators?.macd?.hist >= 0 ? 'pos' : 'neg'}">${sig.indicators?.macd ? sig.indicators.macd.hist.toFixed(2) : '—'}</span></div>
        <div title="Bollinger %B">BB%B: <strong>${sig.indicators?.bb ? (sig.indicators.bb.pctB * 100).toFixed(0) : '—'}</strong></div>
        <div title="ATR approx %">ATR%: <strong>${sig.indicators?.atr && c.current_price ? ((sig.indicators.atr / c.current_price) * 100).toFixed(1) : '—'}</strong></div>
        ${sig.model ? `<div title="Model probability">Model: <strong>${(sig.model.prob * 100).toFixed(0)}%</strong> ↑ • Exp: <strong>${(sig.model.expRet * 100).toFixed(1)}%</strong></div>` : ''}
      </div>
    `;
    els.cards.appendChild(card);
    const canvas = card.querySelector('canvas.spark');
    drawSparkline(canvas, spark, chg >= 0 ? '#1cc8a0' : '#ef476f');
  }
  els.status.textContent = `${items.length} coins • vs ${state.vs.toUpperCase()}`;
}

async function refresh() {
  els.status.textContent = 'Loading markets…';
  try {
    const markets = await fetchMarkets(state.vs);
    state.markets = markets;
    render();
  } catch (err) {
    console.error(err);
    els.status.textContent = `Failed to load (${String(err.message || err)})`;
  }
}

async function loadPredictionsOnce() {
  if (state.predictions !== null) return; // already attempted
  try {
    const res = await fetch('predictions.json', { cache: 'no-store' });
    if (res.ok) {
      state.predictions = await res.json();
    } else {
      state.predictions = undefined; // mark attempted
    }
  } catch {
    state.predictions = undefined;
  }
}

function setAutoRefresh(seconds) {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  if (seconds > 0) {
    state.timer = setInterval(refresh, seconds * 1000);
  }
}

// Event wiring
els.currency.addEventListener('change', () => {
  state.vs = els.currency.value;
  refresh();
});
els.search.addEventListener('input', () => {
  state.filter = els.search.value;
  render();
});
els.refreshBtn.addEventListener('click', () => refresh());
els.autorefresh.addEventListener('change', () => setAutoRefresh(Number(els.autorefresh.value)));

// Initial load
loadPredictionsOnce().finally(refresh);
