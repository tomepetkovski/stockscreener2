"""
upside_engine.py  —  Upside Probability Engine
===============================================
Plug-in for "Institutional Breakout & Investment Terminal v4.0".

Pillars (each scored 0-100, then weight-averaged over the pillars that have data):
    trend       Golden-cross state + recency, EMA50/200 daily, EMA10/40 weekly (MTF)
    continuity  3-5 year continuity of revenue / EPS / free-cash-flow / equity growth
    analyst     Finnhub recommendation trend + price-target upside + EPS-surprise streak
    momentum    Relative strength vs SPY, distance to 52w high, up/down-volume, OBV
    sentiment   Yahoo news lexicon + Finnhub company news + Finnhub insider sentiment
    options     Open interest (put/call OI), volume P/C, call/put walls  (yfinance chain)

The result is a *heuristic* Upside Score (0-100), NOT a calibrated probability.
Hard-fail rules remove names that fail the basics (death-cross structure, bearish
analysts, loss-making / shrinking earnings in strict mode).

Finnhub free tier = 60 calls/min  ->  a global rate limiter + cache is built in.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("UpsideEngine")

# ---------------------------------------------------------------------------
# Finnhub keys.  Env var wins; the hard-coded strings are the fallback.
# NOTE: the string you sent is two 20-char keys glued together -> split here.
# ---------------------------------------------------------------------------
_FINNHUB_KEYS: List[str] = [
    k for k in (
        os.environ.get("FINNHUB_API_KEY", "").strip(),
        "dbms6khr01qn1o670s3g",
        "dbms6khr01qn1o670s40",
    ) if k
]


def _num(x, default: float = 0.0) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else default
    except Exception:
        return default


# =========================================================
# FINNHUB CLIENT  (rate-limited, cached, premium-endpoint aware)
# =========================================================
class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"
    MIN_INTERVAL = 1.1          # s between calls  (~54/min, under the 60/min cap)

    TTL = {
        "/stock/metric": 12 * 3600,
        "/stock/recommendation": 6 * 3600,
        "/stock/earnings": 12 * 3600,
        "/stock/price-target": 6 * 3600,
        "/stock/insider-sentiment": 12 * 3600,
        "/company-news": 3600,
    }

    _rate_lock = threading.Lock()
    _cache_lock = threading.Lock()
    _last_call = 0.0
    _key_idx = 0
    _disabled: set = set()      # endpoints that returned 403 (premium) this session
    _cache: Dict[Any, Any] = {}

    @classmethod
    def available(cls) -> bool:
        return bool(_FINNHUB_KEYS)

    @classmethod
    def get(cls, endpoint: str, **params) -> Optional[Any]:
        if not _FINNHUB_KEYS or endpoint in cls._disabled:
            return None

        ck = (endpoint, tuple(sorted(params.items())))
        ttl = cls.TTL.get(endpoint, 3600)
        with cls._cache_lock:
            hit = cls._cache.get(ck)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]

        for attempt in range(3):
            with cls._rate_lock:
                wait = cls.MIN_INTERVAL - (time.monotonic() - cls._last_call)
                if wait > 0:
                    time.sleep(wait)
                cls._last_call = time.monotonic()
                key = _FINNHUB_KEYS[cls._key_idx % len(_FINNHUB_KEYS)]
            try:
                r = requests.get(
                    cls.BASE + endpoint,
                    params={**params, "token": key},
                    timeout=15,
                )
            except requests.RequestException as exc:
                logger.debug("Finnhub %s network error: %s", endpoint, exc)
                time.sleep(1.5)
                continue

            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    return None
                with cls._cache_lock:
                    cls._cache[ck] = (time.time(), data)
                return data
            if r.status_code == 429:                      # rate limited
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 401:                      # bad key -> rotate
                cls._key_idx += 1
                logger.warning("Finnhub 401 — rotating to next key")
                continue
            if r.status_code == 403:                      # premium endpoint
                cls._disabled.add(endpoint)
                logger.info("Finnhub %s needs a paid plan — disabled for this session", endpoint)
                return None
            logger.debug("Finnhub %s HTTP %s", endpoint, r.status_code)
            break
        return None


# =========================================================
# ANALYSTS (Finnhub)  -> also usable as a source in AnalystConsensusEngine
# =========================================================
class FinnhubAnalystEngine:

    @staticmethod
    def get_rating(symbol: str, price: Optional[float] = None) -> Dict[str, Any]:
        recs = FinnhubClient.get("/stock/recommendation", symbol=symbol)
        if not isinstance(recs, list) or not recs:
            return {}

        def _score(r: Dict[str, Any]):
            sb, b, h = _num(r.get("strongBuy")), _num(r.get("buy")), _num(r.get("hold"))
            s, ss = _num(r.get("sell")), _num(r.get("strongSell"))
            n = sb + b + h + s + ss
            if n <= 0:
                return None, 0
            return (sb * 100 + b * 75 + h * 50 + s * 25 + ss * 0) / n, int(n)

        latest, n = _score(recs[0])
        if latest is None or n < 5:
            return {}

        # trend: compare with ~3 months ago (monthly snapshots, newest first)
        trend_delta = 0.0
        if len(recs) >= 4:
            old, _ = _score(recs[3])
            if old is not None:
                trend_delta = latest - old
        score = float(np.clip(latest + np.clip(trend_delta, -16, 16) * 0.5, 0, 100))

        out: Dict[str, Any] = {
            "score": round(score, 1),
            "source": "Finnhub",
            "analysts": n,
            "raw_score": round(latest, 1),
            "trend_delta": round(trend_delta, 1),
            "strong_buy": int(_num(recs[0].get("strongBuy"))),
            "buy": int(_num(recs[0].get("buy"))),
            "hold": int(_num(recs[0].get("hold"))),
            "sell": int(_num(recs[0].get("sell"))),
            "strong_sell": int(_num(recs[0].get("strongSell"))),
        }

        pt = FinnhubClient.get("/stock/price-target", symbol=symbol)   # may be premium
        if isinstance(pt, dict) and pt.get("targetMean"):
            out["target_mean"] = _num(pt.get("targetMean"))
            out["target_high"] = _num(pt.get("targetHigh"))
            out["target_low"] = _num(pt.get("targetLow"))
            if price and price > 0:
                out["upside_pct"] = round((out["target_mean"] / price - 1) * 100, 1)
        return out


# =========================================================
# TREND STRUCTURE  (golden cross + MTF EMA, from data already in memory)
# =========================================================
class TrendStructureEngine:

    @staticmethod
    def _weekly(df: pd.DataFrame, df_weekly: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_weekly is not None and len(df_weekly) >= 45:
            return df_weekly
        try:
            w = df.resample("W").agg({"open": "first", "high": "max", "low": "min",
                                      "close": "last", "volume": "sum"}).dropna()
            return w if len(w) >= 45 else None
        except Exception:
            return None

    @classmethod
    def analyze(cls, df: pd.DataFrame, df_weekly: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if df is None or len(df) < 210:
            return {"status": "NO_DATA", "score": None}

        c = df["close"].astype(float)
        price = float(c.iloc[-1])
        e50 = c.ewm(span=50, adjust=False).mean()
        e200 = c.ewm(span=200, adjust=False).mean()
        s50 = c.rolling(50).mean()
        s200 = c.rolling(200).mean()

        diff = (s50 - s200).dropna()
        golden_state = bool(diff.iloc[-1] > 0)
        sgn = np.sign(diff)
        flips = sgn[sgn.diff().fillna(0) != 0]
        days_since = int(len(diff) - 1 - diff.index.get_loc(flips.index[-1])) if len(flips) else None

        above50 = price > float(e50.iloc[-1])
        above200 = price > float(e200.iloc[-1])
        e200_rising = float(e200.iloc[-1]) > float(e200.iloc[-21])
        dist50 = (price / float(e50.iloc[-1]) - 1) * 100

        raw, pos, neg = 0.0, [], []

        if above200:
            raw += 12
        else:
            neg.append("Price below EMA200")
        if float(e50.iloc[-1]) > float(e200.iloc[-1]):
            raw += 12
            pos.append("EMA50 > EMA200")
        if golden_state:
            pos.append("Golden-cross state (SMA50>SMA200)")
            if days_since is not None and days_since <= 60:
                raw += 6
                pos.append(f"Fresh golden cross ({days_since}d ago)")
        else:
            neg.append("Death-cross state (SMA50<SMA200)")
            if days_since is not None and days_since <= 60:
                raw -= 6
                neg.append(f"Recent death cross ({days_since}d ago)")
        if above50:
            raw += 8
        if e200_rising:
            raw += 8
            pos.append("EMA200 rising")
        if 0 <= dist50 <= 8:
            raw += 6
            pos.append("Healthy distance above EMA50")
        elif dist50 > 20:
            raw -= 8
            neg.append(f"Overextended (+{dist50:.0f}% over EMA50)")

        # weekly (EMA10 / EMA40 weeks ~ EMA50 / EMA200 days)
        w = cls._weekly(df, df_weekly)
        weekly_ok = None
        if w is not None:
            wc = w["close"].astype(float)
            w10 = wc.ewm(span=10, adjust=False).mean()
            w40 = wc.ewm(span=40, adjust=False).mean()
            weekly_ok = bool(w10.iloc[-1] > w40.iloc[-1])
            if weekly_ok:
                raw += 14
                pos.append("Weekly EMA10 > EMA40")
            else:
                neg.append("Weekly EMA10 < EMA40")
            if float(wc.iloc[-1]) > float(w10.iloc[-1]):
                raw += 8
            if len(w40) > 5 and float(w40.iloc[-1]) > float(w40.iloc[-5]):
                raw += 8
                pos.append("Weekly EMA40 rising")

        score = float(np.clip(raw / 80.0 * 100.0, 0, 100))
        return {
            "status": "OK",
            "score": round(score, 1),
            "golden_state": golden_state,
            "days_since_cross": days_since,
            "above_ema50": above50,
            "above_ema200": above200,
            "weekly_bull": weekly_ok,
            "dist_ema50_pct": round(dist50, 2),
            "death_structure": (not golden_state) and (not above200),
            "pos": pos,
            "neg": neg,
        }


# =========================================================
# FUNDAMENTAL CONTINUITY  (3-5 years of rising profit / capital)
# =========================================================
class FundamentalContinuityEngine:

    ROW_NAMES = {
        "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
        "earnings": ["Diluted EPS", "Basic EPS", "Net Income", "Net Income Common Stockholders"],
        "net_income": ["Net Income", "Net Income Common Stockholders"],
        "fcf": ["Free Cash Flow"],
        "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    }
    FH_KEYS = {          # Finnhub series.annual candidates
        "revenue": ["salesPerShare"],
        "earnings": ["eps"],
        "equity": ["bookValue", "tangibleBookValue"],
    }
    WEIGHTS = {"revenue": 0.25, "earnings": 0.35, "fcf": 0.25, "equity": 0.15}
    _CACHE: Dict[str, Any] = {}
    _TTL = 6 * 3600

    @staticmethod
    def _yf_series(frame: Optional[pd.DataFrame], names: List[str]) -> List[float]:
        if frame is None or getattr(frame, "empty", True):
            return []
        for n in names:
            if n in frame.index:
                s = frame.loc[n].dropna()
                try:
                    s = s.astype(float)
                    s.index = pd.to_datetime(s.index)
                    return [float(v) for v in s.sort_index().values][-6:]
                except Exception:
                    continue
        return []

    @staticmethod
    def _fh_series(metric_blob: Optional[dict], keys: List[str]) -> List[float]:
        try:
            annual = ((metric_blob or {}).get("series") or {}).get("annual") or {}
            for k in keys:
                rows = annual.get(k)
                if isinstance(rows, list) and rows:
                    rows = sorted(rows, key=lambda r: r.get("period", ""))
                    vals = [float(r["v"]) for r in rows if r.get("v") is not None]
                    if vals:
                        return vals[-6:]
        except Exception:
            pass
        return []

    @staticmethod
    def _streak(vals: List[float]) -> Dict[str, Any]:
        """vals oldest -> newest."""
        if len(vals) < 3:
            return {}
        ups = [vals[i + 1] > vals[i] for i in range(len(vals) - 1)]
        streak = 0
        for u in reversed(ups):
            if u:
                streak += 1
            else:
                break
        cagr = None
        if vals[0] > 0 and vals[-1] > 0:
            cagr = (vals[-1] / vals[0]) ** (1.0 / (len(vals) - 1)) - 1
        return {"years": len(vals), "n_up": int(sum(ups)), "n_total": len(ups),
                "streak": streak, "cagr": cagr, "latest": vals[-1]}

    @classmethod
    def analyze(cls, symbol: str) -> Dict[str, Any]:
        hit = cls._CACHE.get(symbol)
        if hit and time.time() - hit[0] < cls._TTL:
            return hit[1]

        inc = cf = bs = None
        try:
            t = yf.Ticker(symbol)
            inc, cf, bs = t.income_stmt, t.cashflow, t.balance_sheet
        except Exception as exc:
            logger.debug("[%s] yfinance statements failed: %s", symbol, exc)

        blob = FinnhubClient.get("/stock/metric", symbol=symbol, metric="all")
        fh_metric = (blob or {}).get("metric") or {}

        series: Dict[str, List[float]] = {}
        for key in ("revenue", "earnings", "fcf", "equity"):
            src = {"revenue": inc, "earnings": inc, "fcf": cf, "equity": bs}[key]
            yv = cls._yf_series(src, cls.ROW_NAMES[key])
            fv = cls._fh_series(blob, cls.FH_KEYS.get(key, []))
            series[key] = yv if len(yv) >= len(fv) else fv        # keep the longer history
        ni = cls._yf_series(inc, cls.ROW_NAMES["net_income"])

        pos: List[str] = []
        neg: List[str] = []
        parts, wsum = 0.0, 0.0
        detail: Dict[str, Any] = {}
        label = {"revenue": "Revenue", "earnings": "EPS/earnings", "fcf": "Free cash flow", "equity": "Equity/book value"}

        for key, vals in series.items():
            st = cls._streak(vals)
            if not st:
                continue
            consistency = st["n_up"] / st["n_total"]
            sc = 100 * (0.6 * consistency + 0.4 * min(st["streak"], 4) / 4)
            if st["cagr"] is not None:
                if st["cagr"] > 0.10:
                    sc += 10
                elif st["cagr"] < 0:
                    sc -= 10
            if key in ("earnings", "fcf") and st["latest"] < 0:
                sc -= 20
            sc = float(np.clip(sc, 0, 100))
            detail[key] = {**st, "score": round(sc, 1)}
            parts += cls.WEIGHTS[key] * sc
            wsum += cls.WEIGHTS[key]
            tag = f"{label[key]} up {st['n_up']}/{st['n_total']} yrs (streak {st['streak']})"
            (pos if consistency >= 0.67 and st["streak"] >= 2 else neg if consistency < 0.5 else []).append(tag)

        # Finnhub multi-year growth rates as a cross-check
        bonus = 0.0
        for k, thr in (("epsGrowth5Y", 10), ("epsGrowth3Y", 10), ("revenueGrowth5Y", 8), ("revenueGrowth3Y", 8)):
            v = fh_metric.get(k)
            if v is not None:
                bonus += 2.5 if _num(v) >= thr else -2.5 if _num(v) < 0 else 0
        if (_num(fh_metric.get("roeTTM"), 0)) >= 15:
            bonus += 2.5
        if (_num(fh_metric.get("netProfitMarginTTM"), 0)) >= 10:
            bonus += 2.5

        if wsum < 0.5 or sum(1 for d in detail.values() if d["years"] >= 3) < 2:
            out = {"status": "INSUFFICIENT", "score": None, "detail": detail,
                   "pos": pos, "neg": neg, "loss_making": bool(ni and ni[-1] < 0)}
        else:
            score = float(np.clip(parts / wsum + bonus, 0, 100))
            out = {
                "status": "OK",
                "score": round(score, 1),
                "detail": detail,
                "pos": pos,
                "neg": neg,
                "loss_making": bool(ni and ni[-1] < 0),
                "profit_up_streak": detail.get("earnings", {}).get("streak", 0),
                "fh": {k: fh_metric.get(k) for k in
                       ("epsGrowth3Y", "epsGrowth5Y", "revenueGrowth3Y", "revenueGrowth5Y", "roeTTM", "netProfitMarginTTM")},
            }
        cls._CACHE[symbol] = (time.time(), out)
        return out


# =========================================================
# OPEN INTEREST  (yfinance option chain)
# =========================================================
class OptionsOIEngine:
    _CACHE: Dict[str, Any] = {}
    _TTL = 3600

    @classmethod
    def analyze(cls, symbol: str, price: float) -> Dict[str, Any]:
        hit = cls._CACHE.get(symbol)
        if hit and time.time() - hit[0] < cls._TTL:
            return hit[1]

        out: Dict[str, Any] = {"status": "NO_DATA", "score": None}
        try:
            t = yf.Ticker(symbol)
            exps = list(t.options or [])
            today = datetime.now(timezone.utc).date()
            picks = []
            for e in exps:
                dte = (pd.Timestamp(e).date() - today).days
                if 14 <= dte <= 150:
                    picks.append(e)
            picks = picks[:3]
            if picks:
                c_oi = p_oi = c_vol = p_vol = otm_c_oi = 0.0
                call_strikes: Dict[float, float] = {}
                put_strikes: Dict[float, float] = {}
                for e in picks:
                    ch = t.option_chain(e)
                    calls, puts = ch.calls.fillna(0), ch.puts.fillna(0)
                    c_oi += float(calls["openInterest"].sum())
                    p_oi += float(puts["openInterest"].sum())
                    c_vol += float(calls["volume"].sum())
                    p_vol += float(puts["volume"].sum())
                    otm_c_oi += float(calls.loc[calls["strike"] > price * 1.02, "openInterest"].sum())
                    for _, r in calls.iterrows():
                        call_strikes[r["strike"]] = call_strikes.get(r["strike"], 0) + r["openInterest"]
                    for _, r in puts.iterrows():
                        put_strikes[r["strike"]] = put_strikes.get(r["strike"], 0) + r["openInterest"]

                total_oi = c_oi + p_oi
                if total_oi >= 1000 and c_oi > 0:
                    pcr_oi = p_oi / c_oi
                    pcr_vol = (p_vol / c_vol) if c_vol > 0 else None
                    call_wall = max(call_strikes, key=call_strikes.get) if call_strikes else None
                    put_wall = max(put_strikes, key=put_strikes.get) if put_strikes else None

                    sc, pos, neg = 50.0, [], []
                    if pcr_oi < 0.5:
                        sc += 15; pos.append(f"Put/Call OI {pcr_oi:.2f} (call-heavy)")
                    elif pcr_oi < 0.7:
                        sc += 10; pos.append(f"Put/Call OI {pcr_oi:.2f}")
                    elif pcr_oi < 0.9:
                        sc += 3
                    elif pcr_oi > 1.3:
                        sc -= 15; neg.append(f"Put/Call OI {pcr_oi:.2f} (put-heavy)")
                    elif pcr_oi > 1.0:
                        sc -= 8; neg.append(f"Put/Call OI {pcr_oi:.2f}")
                    if pcr_vol is not None and (c_vol + p_vol) >= 200:
                        if pcr_vol < 0.6:
                            sc += 10; pos.append(f"Put/Call volume {pcr_vol:.2f}")
                        elif pcr_vol > 1.2:
                            sc -= 10; neg.append(f"Put/Call volume {pcr_vol:.2f}")
                    if c_oi > 0 and otm_c_oi / c_oi > 0.45:
                        sc += 5; pos.append("OTM call OI concentration (upside positioning)")
                    if call_wall and call_wall > price * 1.05:
                        sc += 5; pos.append(f"Call-OI wall ${call_wall:.0f} leaves room")
                    if put_wall and put_wall < price:
                        sc += 5; pos.append(f"Put-OI support ${put_wall:.0f}")
                    out = {
                        "status": "OK", "score": float(np.clip(sc, 0, 100)),
                        "pcr_oi": round(pcr_oi, 2),
                        "pcr_vol": None if pcr_vol is None else round(pcr_vol, 2),
                        "total_oi": int(total_oi), "call_wall": call_wall, "put_wall": put_wall,
                        "expiries": picks, "pos": pos, "neg": neg,
                    }
        except Exception as exc:
            logger.debug("[%s] options OI failed: %s", symbol, exc)

        cls._CACHE[symbol] = (time.time(), out)
        return out


# =========================================================
# SENTIMENT (Finnhub news + insider sentiment)
# =========================================================
_BULL = {"beat": 2, "beats": 2, "surge": 2, "surges": 2, "upgrade": 2, "upgraded": 2, "raises": 1.5,
         "record": 1.5, "growth": 1.5, "rally": 2, "bullish": 2.5, "outperform": 2, "soar": 2,
         "breakthrough": 2, "buyback": 1.5, "expands": 1.2, "profit": 1.2, "strong": 1.2}
_BEAR = {"miss": -2, "misses": -2, "drop": -2, "plunge": -2.5, "downgrade": -2, "downgraded": -2,
         "loss": -2, "weak": -1.5, "crash": -3, "bearish": -2.5, "lawsuit": -2, "probe": -2,
         "investigation": -2, "layoff": -2, "layoffs": -2, "cuts": -1.5, "warning": -1.5,
         "recall": -2, "fraud": -3, "bankruptcy": -3, "decline": -1.5}


class FinnhubSentimentEngine:

    @staticmethod
    def news_score(symbol: str, days: int = 14) -> Optional[float]:
        to = datetime.now(timezone.utc).date()
        frm = to - timedelta(days=days)
        items = FinnhubClient.get("/company-news", symbol=symbol, **{"from": str(frm), "to": str(to)})
        if not isinstance(items, list) or not items:
            return None
        nets = []
        for it in items[:25]:
            text = f"{it.get('headline', '')} {it.get('summary', '')}".lower()
            words = [w.strip(".,!?;:\"'()[]") for w in text.split()]
            nets.append(sum(_BULL.get(w, 0) + _BEAR.get(w, 0) for w in words))
        return float(np.clip(50 + np.mean(nets) * 12, 10, 90))

    @staticmethod
    def insider_score(symbol: str) -> Optional[float]:
        to = datetime.now(timezone.utc).date()
        frm = to - timedelta(days=120)
        blob = FinnhubClient.get("/stock/insider-sentiment", symbol=symbol,
                                 **{"from": str(frm), "to": str(to)})
        rows = (blob or {}).get("data") if isinstance(blob, dict) else None
        if not rows:
            return None
        mspr = np.mean([_num(r.get("mspr")) for r in rows[-3:]])        # -100 .. +100
        return float(np.clip(50 + mspr / 2.0, 0, 100))


# =========================================================
# MOMENTUM / FLOW
# =========================================================
def _momentum_flow(df: pd.DataFrame, spy_roc_60: float) -> Dict[str, Any]:
    c = df["close"].astype(float)
    v = df["volume"].astype(float)
    n = len(c)
    if n < 130:
        return {"score": None}
    pos, neg = [], []
    sc = 50.0

    roc60 = (c.iloc[-1] / c.iloc[-61] - 1) * 100
    roc120 = (c.iloc[-1] / c.iloc[-121] - 1) * 100
    rs60 = roc60 - spy_roc_60
    sc += float(np.clip(rs60, -15, 15))
    sc += float(np.clip(roc120 * 0.15, -10, 10))
    if rs60 > 5:
        pos.append(f"Outperforming SPY by {rs60:.1f}% (60d)")
    elif rs60 < -5:
        neg.append(f"Lagging SPY by {abs(rs60):.1f}% (60d)")

    hi252 = float(df["high"].iloc[-252:].max())
    near = float(c.iloc[-1]) / hi252
    if near >= 0.95:
        sc += 10; pos.append("Within 5% of 52w high")
    elif near >= 0.85:
        sc += 5
    elif near < 0.60:
        sc -= 10; neg.append("More than 40% below 52w high")

    ret = c.diff()
    up_v = v[ret > 0].iloc[-50:].sum()
    dn_v = v[ret < 0].iloc[-50:].sum()
    ud = float(up_v / dn_v) if dn_v > 0 else 1.0
    if ud > 1.3:
        sc += 10; pos.append(f"Up/down volume {ud:.2f}")
    elif ud > 1.1:
        sc += 5
    elif ud < 0.8:
        sc -= 10; neg.append(f"Up/down volume {ud:.2f} (distribution)")

    obv = (np.sign(ret).fillna(0) * v).cumsum()
    if obv.iloc[-1] > obv.iloc[-50]:
        sc += 5
    else:
        sc -= 5; neg.append("OBV falling (50d)")

    return {"score": float(np.clip(sc, 0, 100)), "rs_60d": round(rs60, 2),
            "near_52w_high": round(near, 3), "updown_vol": round(ud, 2), "pos": pos, "neg": neg}


# =========================================================
# MAIN ENGINE
# =========================================================
class UpsideProbabilityEngine:

    WEIGHTS = {"trend": 0.25, "continuity": 0.22, "analyst": 0.16,
               "momentum": 0.15, "sentiment": 0.10, "options": 0.12}

    @classmethod
    def evaluate(
        cls,
        symbol: str,
        df: pd.DataFrame,
        df_weekly: Optional[pd.DataFrame] = None,
        *,
        sentiment: Optional[Dict[str, Any]] = None,
        analysts: Optional[Dict[str, Any]] = None,
        funds: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
        deep: bool = True,         # Finnhub + options calls (rate limited)
        strict: bool = True,       # apply hard-fail rules on earnings continuity
    ) -> Dict[str, Any]:

        price = float(df["close"].iloc[-1])
        sentiment, analysts, funds = sentiment or {}, analysts or {}, funds or {}
        pos: List[str] = []
        neg: List[str] = []
        fails: List[str] = []
        pillars: Dict[str, Optional[float]] = {}

        # 1) trend / golden cross / MTF
        trend = TrendStructureEngine.analyze(df, df_weekly)
        pillars["trend"] = trend.get("score")
        pos += trend.get("pos", [])
        neg += trend.get("neg", [])
        if trend.get("death_structure"):
            fails.append("Death-cross structure and price below EMA200")

        # 2) momentum / flow
        spy60 = _num(((market_regime or {}).get("SPY") or {}).get("roc_60d"), 0.0)
        mom = _momentum_flow(df, spy60)
        pillars["momentum"] = mom.get("score")
        pos += mom.get("pos", [])
        neg += mom.get("neg", [])

        cont, opt, fh_an = {}, {}, {}
        news_s = ins_s = None
        if deep:
            # 3) multi-year continuity
            cont = FundamentalContinuityEngine.analyze(symbol)
            pillars["continuity"] = cont.get("score")
            pos += cont.get("pos", [])
            neg += cont.get("neg", [])
            if strict and cont.get("loss_making"):
                fails.append("Latest annual net income is negative")
            if strict and cont.get("score") is not None and cont["score"] < 35:
                fails.append(f"Weak multi-year growth continuity ({cont['score']:.0f}/100)")

            # 4) analysts (Finnhub + existing consensus + price target + EPS beats)
            fh_an = FinnhubAnalystEngine.get_rating(symbol, price)
            an_scores, an_w = [], []
            if fh_an.get("score") is not None:
                an_scores.append(fh_an["score"]); an_w.append(0.55)
            if analysts.get("score") is not None and analysts.get("active_sources", 1):
                an_scores.append(_num(analysts["score"], 50)); an_w.append(0.45)
            if an_scores:
                a = float(np.average(an_scores, weights=an_w))
                up = fh_an.get("upside_pct")
                if up is None:
                    tgt, px = _num(funds.get("target_mean_price")), _num(funds.get("current_price")) or price
                    up = (tgt / px - 1) * 100 if tgt and px else None
                if up is not None:
                    a += 10 if up > 25 else 5 if up > 10 else -10 if up < 0 else 0
                    if up > 10:
                        pos.append(f"Analyst target upside {up:.0f}%")
                    elif up < 0:
                        neg.append(f"Price above analyst target ({up:.0f}%)")
                eps = FinnhubClient.get("/stock/earnings", symbol=symbol)
                if isinstance(eps, list) and eps:
                    beats = sum(1 for r in eps[:4] if _num(r.get("surprisePercent")) > 0)
                    a += 8 if beats == 4 else 4 if beats == 3 else -6 if beats <= 1 else 0
                    if beats >= 3:
                        pos.append(f"Beat EPS in {beats}/4 quarters")
                    elif beats <= 1:
                        neg.append(f"Beat EPS in only {beats}/4 quarters")
                pillars["analyst"] = float(np.clip(a, 0, 100))
                if fh_an and fh_an.get("trend_delta", 0) > 8:
                    pos.append("Analyst ratings upgrading (3m trend)")
                if fh_an.get("analysts", 0) >= 5 and fh_an.get("raw_score", 50) < 40:
                    fails.append("Analyst consensus bearish")
            else:
                pillars["analyst"] = None

            # 5) sentiment
            parts, w = [], []
            if sentiment.get("score") is not None:
                parts.append(_num(sentiment["score"], 50)); w.append(0.45)
            news_s = FinnhubSentimentEngine.news_score(symbol)
            if news_s is not None:
                parts.append(news_s); w.append(0.35)
            ins_s = FinnhubSentimentEngine.insider_score(symbol)
            if ins_s is not None:
                parts.append(ins_s); w.append(0.20)
                if ins_s > 60:
                    pos.append("Net insider buying (MSPR)")
                elif ins_s < 40:
                    neg.append("Net insider selling (MSPR)")
            pillars["sentiment"] = float(np.average(parts, weights=w)) if parts else None

            # 6) open interest
            opt = OptionsOIEngine.analyze(symbol, price)
            pillars["options"] = opt.get("score")
            pos += opt.get("pos", [])
            neg += opt.get("neg", [])
        else:
            if analysts.get("score") is not None:
                pillars["analyst"] = _num(analysts["score"], 50)
            if sentiment.get("score") is not None:
                pillars["sentiment"] = _num(sentiment["score"], 50)

        # weighted score over pillars that have data
        avail = {k: v for k, v in pillars.items() if v is not None}
        wsum = sum(cls.WEIGHTS[k] for k in avail)
        score = sum(cls.WEIGHTS[k] * v for k, v in avail.items()) / wsum if wsum > 0 else 0.0
        coverage = wsum / sum(cls.WEIGHTS.values())
        if coverage < 0.5:                       # too little evidence -> don't trust
            score *= 0.85
            neg.append(f"Low data coverage ({coverage:.0%})")

        grade = "A+" if score >= 80 else "A" if score >= 70 else "B" if score >= 60 else "C"
        return {
            "upside_score": round(float(score), 1),
            "grade": grade,
            "coverage": round(coverage, 2),
            "pillars": {k: (None if v is None else round(v, 1)) for k, v in pillars.items()},
            "pos": pos[:10],
            "neg": neg[:8],
            "hard_fail": bool(fails),
            "fail_reasons": fails,
            "trend": trend,
            "continuity": cont,
            "options": opt,
            "finnhub_analyst": fh_an,
            "momentum": mom,
        }
