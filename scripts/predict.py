#!/usr/bin/env python3
"""
Build predictions.json using CoinGecko hourly prices (no key).
Heuristic model: engineered features + logistic-like score -> probability.

Output: predictions.json at repo root.
"""
import json, math, sys, time, urllib.request

API = "https://api.coingecko.com/api/v3"
VS = "usd"
TOP_N = 25
HOURS_LOOKBACK = 90 * 24  # attempt 90 days hourly

def get(url):
    req = urllib.request.Request(url, headers={"accept":"application/json","user-agent":"cm-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def top_coins(n=TOP_N):
    url = f"{API}/coins/markets?vs_currency={VS}&order=market_cap_desc&per_page={n}&page=1&sparkline=false"
    return [c["id"] for c in get(url)]

def market_chart(id, days=90):
    url = f"{API}/coins/{id}/market_chart?vs_currency={VS}&days={days}&interval=hourly"
    data = get(url)
    prices = [p[1] for p in data.get("prices", [])]
    return prices

def sma(vals, w):
    if len(vals) < w: return None
    return sum(vals[-w:]) / w

def ema_last(vals, p):
    if not vals: return None
    k = 2/(p+1)
    seed = sum(vals[:min(p,len(vals))]) / min(p,len(vals))
    e = seed
    for v in vals[min(p,len(vals)):]:
        e = v*k + e*(1-k)
    return e

def std_last(vals, w):
    if len(vals) < w: return None
    s = vals[-w:]
    m = sum(s)/w
    return math.sqrt(sum((v-m)*(v-m) for v in s)/w)

def rsi_last(vals, p=14):
    if len(vals) < p+1: return None
    gains=0;losses=0
    for i in range(1,p+1):
        d=vals[i]-vals[i-1]
        gains += d if d>0 else 0
        losses += -d if d<0 else 0
    ag=gains/p; al=losses/p
    for i in range(p+1, len(vals)):
        d=vals[i]-vals[i-1]
        g=d if d>0 else 0
        l=-d if d<0 else 0
        ag=(ag*(p-1)+g)/p
        al=(al*(p-1)+l)/p
    if al==0: return 100.0
    rs=ag/al
    return 100-100/(1+rs)

def lr_slope(vals, last_n):
    s=vals[-last_n:]
    n=len(s)
    if n<2: return 0.0
    mx=(n-1)/2
    my=sum(s)/n
    num=0; den=0
    for i,y in enumerate(s):
        dx=i-mx
        num+=dx*(y-my)
        den+=dx*dx
    return 0.0 if den==0 else num/den

def predict_for_series(prices):
    if len(prices)<60: return None
    last=prices[-1]
    ema7=ema_last(prices,7); ema14=ema_last(prices,14)
    rsi=rsi_last(prices,14)
    sd=std_last(prices,20)
    bb_width=(2*2*sd/ (sma(prices,20) or (last or 1))) if sd is not None else 0
    slope=lr_slope(prices,60)
    slope24=slope*24
    score=0.0
    if ema7 is not None and ema14 is not None:
        score += 1.0 if ema7>ema14 else -1.0
    if rsi is not None:
        if 55<=rsi<=70: score += 0.6
        if 30<=rsi<=45: score -= 0.6
        if rsi>70: score -= 0.2
        if rsi<30: score += 0.2
    if slope24>0: score += 0.6
    if slope24<0: score -= 0.6
    if bb_width<0.04: score -= 0.2
    # squash to probability
    prob = 1/(1+math.exp(-score))
    exp_return = (slope24/last) if last else 0.0
    return {"prob_up": float(max(0,min(1,prob))), "exp_return": float(exp_return)}

def main():
    ids = top_coins(TOP_N)
    out = {"as_of": int(time.time()), "horizon_hours": 24, "coins": {}}
    for cid in ids:
        try:
            px = market_chart(cid, days=90)
            pred = predict_for_series(px)
            if pred:
                out["coins"][cid] = pred
        except Exception as e:
            # skip coin on errors
            pass
    with open("predictions.json","w") as f:
        json.dump(out, f, separators=(",",":"))

if __name__ == "__main__":
    main()

