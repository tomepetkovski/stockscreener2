"""
decision_context.py - "do the independent signals agree with the trade?" layer

Plugs in AFTER ProposalEngine/DecisionEngine produce a proposal. For each finalist it checks:
  1. AnalystIntel        rating mix + 3-month trend, recent upgrades/downgrades (recency-weighted),
                         price-target upside & dispersion, EPS-estimate revisions, earnings-surprise record
  2. OptionsFlowEngine   open-interest put/call, volume put/call, unusual contracts, OI walls, max pain, IV skew
  3. NewsIntel           phrase/negation lexicon, recency-weighted, news velocity, event flags
  4. VolumeFlowEngine    signed volume z-score (volume spike *in the direction of price*), up/down volume,
                         CMF, OBV divergence, breakout-on-volume / selling-on-volume
  5. MTFAlignment        daily/weekly EMA conditions, golden-cross age, extension (chasing) check,
                         optional MTFEMAEngine result
Result: per-factor CONFIRM / NEUTRAL / CONFLICT rows, an alignment score 0-100, a verdict
(ALIGNED / MIXED / CONFLICTED), a size multiplier and a confidence delta.

Everything is free-data (yfinance). Options data is delayed and open interest is end-of-previous-day;
OI shows positioning size, NOT who is buying vs selling. Treat it as context, not a trigger.
"""
from __future__ import annotations

import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # keeps the module importable for tests
    yf = None

_LOCK = threading.Lock()
_CACHE: Dict[Any, Any] = {}


# ---------------------------------------------------------------- helpers
def _ticker(symbol: str):
    return yf.Ticker(symbol)


def _num(x: Any, d: float = 0.0) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else d
    except Exception:
        return d


def _safe(fn: Callable, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _cached(key, ttl: int, fn: Callable):
    now = time.time()
    with _LOCK:
        h = _CACHE.get(key)
        if h and now - h[0] < ttl:
            return h[1]
    try:
        v = fn()
    except Exception:
        v = None
    with _LOCK:
        _CACHE[key] = (now, v)
    return v


def _label(score: float) -> str:
    return ("STRONG BUY" if score >= 85 else "BUY" if score >= 65 else
            "HOLD" if score >= 45 else "SELL" if score >= 25 else "STRONG SELL")


def _trailing(mask: np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0
    last = arr[-1]
    idx = np.flatnonzero(arr != last)
    return int(arr.size if idx.size == 0 else arr.size - idx[-1] - 1)


# ================================================================ 1. ANALYSTS
class AnalystIntel:
    WEIGHTS = {"rating_mix": .25, "rating_trend": .10, "actions": .20,
               "targets": .20, "revisions": .15, "surprises": .10}
    TTL = 3600

    @classmethod
    def get(cls, symbol: str, price: Optional[float] = None) -> Dict[str, Any]:
        return _cached(("analyst", symbol), cls.TTL, lambda: cls._build(symbol, price)) \
            or {"status": "NO_DATA", "score": None, "comments": [], "flags": [], "components": {}}

    @classmethod
    def _build(cls, symbol: str, price: Optional[float]) -> Dict[str, Any]:
        t = _ticker(symbol)
        info = _safe(lambda: t.info, {}) or {}
        price = price or _num(info.get("currentPrice") or info.get("regularMarketPrice")) or None
        comp: Dict[str, Dict[str, Any]] = {}
        comments: List[str] = []
        flags: List[str] = []
        n_analysts = int(_num(info.get("numberOfAnalystOpinions")))

        # --- rating mix and 3-month trend
        rs = _safe(lambda: t.recommendations_summary)
        if isinstance(rs, pd.DataFrame) and not rs.empty:
            rs = rs.set_index("period") if "period" in rs.columns else rs

            def dist(row):
                sb, b, h, s, ss = [_num(row.get(k)) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")]
                tot = sb + b + h + s + ss
                if tot <= 0:
                    return None
                x = (2 * sb + b - s - 2 * ss) / tot
                return {"score": (x + 2) / 4 * 100, "buy": (sb + b) / tot * 100,
                        "sell": (s + ss) / tot * 100, "n": int(tot)}

            cur = dist(rs.loc["0m"]) if "0m" in rs.index else dist(rs.iloc[0])
            old = dist(rs.loc["-3m"]) if "-3m" in rs.index else None
            if cur:
                n_analysts = n_analysts or cur["n"]
                comp["rating_mix"] = {"score": cur["score"],
                                      "detail": f"{cur['buy']:.0f}% buy / {cur['sell']:.0f}% sell of {cur['n']} analysts"}
                if old:
                    d = cur["score"] - old["score"]
                    comp["rating_trend"] = {"score": 50 + float(np.clip(d * 2, -40, 40)),
                                            "detail": f"mix {d:+.1f} pts vs 3 months ago"}
                    if d >= 5:
                        flags.append("RATING_IMPROVING")
                        comments.append(f"Rating mix improved {d:+.0f} pts over 3 months")
                    elif d <= -5:
                        flags.append("RATING_DETERIORATING")
                        comments.append(f"Rating mix deteriorated {d:+.0f} pts over 3 months")

        # --- recency-weighted upgrades / downgrades (last 90d)
        ud = _safe(lambda: t.upgrades_downgrades)
        if isinstance(ud, pd.DataFrame) and not ud.empty:
            d = ud.copy()
            d.index = pd.to_datetime(d.index, errors="coerce")
            d = d[d.index.notna()].sort_index(ascending=False)
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            ages = (pd.Timestamp.now().normalize() - d.index.normalize()).days.to_numpy()
            m = ages <= 90
            d, ages = d[m], ages[m]
            if len(d):
                w = np.where(ages <= 30, 1.0, np.where(ages <= 60, 0.6, 0.3))
                act = d["Action"].astype(str).str.lower().to_numpy() if "Action" in d else np.array([""] * len(d))
                pta = d["priceTargetAction"].astype(str).str.lower().to_numpy() if "priceTargetAction" in d else np.array([""] * len(d))
                ups, dns = float(w[act == "up"].sum()), float(w[act == "down"].sum())
                pt_up, pt_dn = float(w[pta == "raises"].sum()), float(w[pta == "lowers"].sum())
                net = (ups - dns) + 0.5 * (pt_up - pt_dn)
                n_up, n_dn = int((act == "up").sum()), int((act == "down").sum())
                comp["actions"] = {"score": 50 + float(np.clip(net * 8, -40, 40)),
                                   "detail": f"90d: {n_up} upgrades / {n_dn} downgrades, PT raised {int((pta == 'raises').sum())} / cut {int((pta == 'lowers').sum())}"}
                if n_dn >= 3 and n_dn > 2 * n_up:
                    flags.append("DOWNGRADE_CLUSTER")
                if n_up >= 3 and n_up > 2 * n_dn:
                    flags.append("UPGRADE_CLUSTER")
                for (_, row), age in list(zip(d.iterrows(), ages))[:3]:
                    firm = str(row.get("Firm", "?"))
                    to_g, fr_g = str(row.get("ToGrade", "")), str(row.get("FromGrade", ""))
                    pt = _num(row.get("currentPriceTarget"))
                    arrow = f"{fr_g} -> {to_g}" if fr_g and fr_g != "nan" and fr_g != to_g else to_g
                    comments.append(f"{int(age)}d ago {firm}: {arrow}" + (f", PT ${pt:.0f}" if pt else ""))

        # --- price targets
        apt = _safe(lambda: t.analyst_price_targets, {}) or {}
        mean = _num(apt.get("mean") or info.get("targetMeanPrice"))
        med = _num(apt.get("median") or info.get("targetMedianPrice")) or mean
        hi = _num(apt.get("high") or info.get("targetHighPrice"))
        lo = _num(apt.get("low") or info.get("targetLowPrice"))
        tgt: Dict[str, Any] = {}
        if price and med:
            up = med / price - 1
            s = 50 + float(np.clip(up * 100, -25, 40)) * 1.1
            spread = (hi - lo) / mean if (hi and lo and mean) else None
            if spread is not None and spread > 0.7:
                s -= 5
                flags.append("WIDE_TARGET_SPREAD")
            if lo and lo > price:
                s += 5
                flags.append("ALL_TARGETS_ABOVE_PRICE")
            if mean and mean < price:
                flags.append("PRICE_ABOVE_MEAN_TARGET")
            tgt = {"upside_median_pct": round(up * 100, 1), "low": lo or None, "high": hi or None,
                   "mean": mean or None, "spread": None if spread is None else round(spread, 2)}
            comp["targets"] = {"score": float(np.clip(s, 0, 100)),
                               "detail": f"median PT upside {up * 100:+.0f}%" +
                                         (f", range ${lo:.0f}-${hi:.0f}" if lo and hi else "")}

        # --- EPS estimate revisions
        er = _safe(lambda: t.eps_revisions)
        if isinstance(er, pd.DataFrame) and not er.empty:
            up_n = dn_n = 0.0
            for idx in ("0y", "+1y"):
                if idx in er.index:
                    up_n += _num(er.loc[idx].get("upLast30days"))
                    dn_n += _num(er.loc[idx].get("downLast30days"))
            if up_n + dn_n > 0:
                net = (up_n - dn_n) / (up_n + dn_n)
                s = 50 + 25 * net
                det = f"30d EPS revisions: {int(up_n)} up / {int(dn_n)} down"
                et = _safe(lambda: t.eps_trend)
                if isinstance(et, pd.DataFrame) and "+1y" in et.index:
                    cur_e, ago = _num(et.loc["+1y"].get("current")), _num(et.loc["+1y"].get("90daysAgo"))
                    if ago:
                        drift = (cur_e - ago) / abs(ago) * 100
                        s += float(np.clip(drift, -15, 15))
                        det += f", next-yr EPS est {drift:+.1f}% in 90d"
                comp["revisions"] = {"score": float(np.clip(s, 0, 100)), "detail": det}
                if net <= -0.5:
                    flags.append("ESTIMATES_FALLING")
                    comments.append("Analysts are cutting EPS estimates")
                elif net >= 0.5:
                    flags.append("ESTIMATES_RISING")
                    comments.append("Analysts are raising EPS estimates")

        # --- earnings-surprise record
        eh = _safe(lambda: t.earnings_history)
        if isinstance(eh, pd.DataFrame) and {"epsActual", "epsEstimate"} <= set(eh.columns):
            e = eh.dropna(subset=["epsActual", "epsEstimate"]).tail(4)
            if len(e):
                beats = int((e["epsActual"] > e["epsEstimate"]).sum())
                sur = ((e["epsActual"] - e["epsEstimate"]) / e["epsEstimate"].abs().replace(0, np.nan)).mean()
                sur = 0.0 if not math.isfinite(_num(sur)) else float(sur)
                comp["surprises"] = {"score": float(np.clip(50 + (beats - len(e) / 2) * 12 + np.clip(sur * 100, -10, 10), 0, 100)),
                                     "detail": f"beat EPS in {beats}/{len(e)} last quarters (avg surprise {sur * 100:+.1f}%)"}

        if not comp:
            return {"status": "NO_DATA", "score": None, "comments": [], "flags": [], "components": {}}
        wsum = sum(cls.WEIGHTS[k] for k in comp)
        score = sum(cls.WEIGHTS[k] * comp[k]["score"] for k in comp) / wsum
        conf = "HIGH" if len(comp) >= 5 and n_analysts >= 15 else "MEDIUM" if len(comp) >= 3 and n_analysts >= 5 else "LOW"
        return {"status": "OK", "score": round(score, 1), "label": _label(score), "confidence": conf,
                "n_analysts": n_analysts, "components": comp, "targets": tgt,
                "comments": comments, "flags": flags}


# ================================================================ 2. OPTIONS / OPEN INTEREST
class OptionsFlowEngine:
    TTL = 1800

    @classmethod
    def analyze(cls, symbol: str, price: Optional[float] = None, max_expiries: int = 3,
                min_days: int = 5, max_days: int = 75) -> Dict[str, Any]:
        return _cached(("opt", symbol), cls.TTL, lambda: cls._build(symbol, price, max_expiries, min_days, max_days)) \
            or {"status": "NO_DATA"}

    @staticmethod
    def _build(symbol, price, max_expiries, min_days, max_days) -> Dict[str, Any]:
        t = _ticker(symbol)
        exps = list(_safe(lambda: t.options, []) or [])
        if not exps:
            return {"status": "NO_OPTIONS"}
        today = pd.Timestamp.now().normalize()
        dte = {e: (pd.Timestamp(e) - today).days for e in exps}
        sel = [e for e in exps if min_days <= dte[e] <= max_days] or exps[:2]
        sel = sel[:max_expiries]
        if not price:
            info = _safe(lambda: t.info, {}) or {}
            price = _num(info.get("currentPrice") or info.get("regularMarketPrice")) or None
        if not price:
            return {"status": "NO_PRICE"}

        c_oi = p_oi = c_vol = p_vol = 0.0
        coi, poi = {}, {}
        unusual: List[Dict[str, Any]] = []
        iv_c, iv_p = [], []
        pain = None
        for k, e in enumerate(sel):
            ch = _safe(lambda: t.option_chain(e))
            if ch is None:
                continue
            for side, df in (("C", ch.calls), ("P", ch.puts)):
                if df is None or len(df) == 0:
                    continue
                df = df.copy()
                for col in ("openInterest", "volume", "impliedVolatility", "strike"):
                    df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)
                oi, vol = float(df["openInterest"].sum()), float(df["volume"].sum())
                book = coi if side == "C" else poi
                for s_, o_ in zip(df["strike"], df["openInterest"]):
                    book[s_] = book.get(s_, 0.0) + o_
                if side == "C":
                    c_oi += oi; c_vol += vol
                else:
                    p_oi += oi; p_vol += vol
                u = df[(df["volume"] >= 500) & (df["volume"] >= 2 * df["openInterest"].clip(lower=1))]
                for _, r in u.nlargest(2, "volume").iterrows():
                    unusual.append({"type": side, "exp": e, "strike": float(r["strike"]),
                                    "vol": int(r["volume"]), "oi": int(r["openInterest"])})
                near = df.iloc[(df["strike"] - price).abs().argsort()[:1]]
                iv = _num(near["impliedVolatility"].iloc[0])
                if iv > 0 and dte[e] <= 45:
                    (iv_c if side == "C" else iv_p).append(iv)
            if k == 0 and ch.calls is not None and ch.puts is not None:
                try:
                    ks = np.array(sorted(set(coi) | set(poi)))
                    cs = np.array([coi.get(x, 0.0) for x in ks]); ps = np.array([poi.get(x, 0.0) for x in ks])
                    tot = [(cs * np.clip(K - ks, 0, None)).sum() + (ps * np.clip(ks - K, 0, None)).sum() for K in ks]
                    pain = float(ks[int(np.argmin(tot))])
                except Exception:
                    pain = None

        total_oi, total_vol = c_oi + p_oi, c_vol + p_vol
        out: Dict[str, Any] = {"status": "OK", "expiries": sel, "call_oi": int(c_oi), "put_oi": int(p_oi),
                               "pc_oi": round(p_oi / max(c_oi, 1), 2), "pc_vol": round(p_vol / max(c_vol, 1), 2),
                               "unusual": sorted(unusual, key=lambda x: -x["vol"])[:3], "max_pain": pain}
        above = {k: v for k, v in coi.items() if k > price}
        below = {k: v for k, v in poi.items() if k < price}
        out["call_wall"] = max(above, key=above.get) if above else None
        out["put_wall"] = max(below, key=below.get) if below else None
        iv_skew = (np.mean(iv_p) - np.mean(iv_c)) if iv_c and iv_p else None
        out["iv_skew"] = None if iv_skew is None else round(float(iv_skew), 3)
        flags: List[str] = []
        if total_oi < 2000 or total_vol < 200:
            out.update(score=50.0, bias="ILLIQUID", flags=["LOW_OPTIONS_LIQUIDITY"], liquid=False)
            return out
        s = 50.0
        s += float(np.clip((1.0 - out["pc_oi"]) * 25, -25, 25))
        s += float(np.clip((1.0 - out["pc_vol"]) * 20, -20, 20))
        uc = sum(u["vol"] for u in unusual if u["type"] == "C")
        up_ = sum(u["vol"] for u in unusual if u["type"] == "P")
        if uc + up_ > 0:
            s += 8 * (uc - up_) / (uc + up_)
            flags.append("UNUSUAL_CALL_ACTIVITY" if uc > up_ else "UNUSUAL_PUT_ACTIVITY")
        if iv_skew is not None:
            s += -5 if iv_skew > 0.08 else 3 if iv_skew < -0.02 else 0
        if out["call_wall"] and out["call_wall"] / price - 1 < 0.03:
            s -= 4; flags.append("CALL_WALL_OVERHEAD")
        if out["put_wall"] and 1 - out["put_wall"] / price < 0.03:
            s += 3; flags.append("PUT_WALL_SUPPORT")
        if out["pc_oi"] > 2.0:
            flags.append("EXTREME_PUT_POSITIONING (hedging or fear; ambiguous)")
        s = float(np.clip(s, 0, 100))
        out.update(score=round(s, 1), bias="BULLISH" if s >= 60 else "BEARISH" if s <= 40 else "NEUTRAL",
                   flags=flags, liquid=True)
        return out


# ================================================================ 3. NEWS
_PHRASES = {
    "raises guidance": 2.5, "raised guidance": 2.5, "raises outlook": 2.0, "beats estimates": 2.5, "beat estimates": 2.5,
    "tops estimates": 2.5, "record revenue": 2.0, "record earnings": 2.0, "price target raised": 1.5,
    "raises price target": 1.5, "upgraded": 2.0, "upgrade": 2.0, "share repurchase": 1.2, "buyback": 1.2,
    "fda approval": 3.0, "fda approves": 3.0, "wins contract": 2.0, "contract win": 2.0,
    "strategic partnership": 1.5, "partnership": 1.0, "dividend increase": 1.2, "all-time high": 1.2,
    "insider buying": 1.5, "soars": 2.0, "surges": 2.0, "jumps": 1.5,
    "cuts guidance": -2.5, "lowers guidance": -2.5, "lowered guidance": -2.5, "misses estimates": -2.5,
    "missed estimates": -2.5, "falls short": -2.0, "price target cut": -1.5, "lowers price target": -1.5,
    "price target lowered": -1.5, "downgraded": -2.0, "downgrade": -2.0, "sec investigation": -3.0,
    "sec probe": -3.0, "class action": -2.0, "lawsuit": -1.5, "recall": -2.0, "bankruptcy": -4.0,
    "going concern": -4.0, "fda rejects": -3.0, "complete response letter": -3.0, "layoffs": -1.5,
    "job cuts": -1.5, "short seller": -3.0, "accounting irregularities": -4.0, "delisting": -3.0,
    "insider selling": -1.0, "plunges": -2.0, "plunge": -2.0, "tumbles": -1.8, "slumps": -2.0, "sinks": -2.0,
}
_PHRASE_ORDER = sorted(_PHRASES, key=len, reverse=True)
_POS_W = {"strong", "growth", "beats", "rally", "bullish", "outperform", "expands", "wins", "approval", "record"}
_NEG_W = {"weak", "miss", "misses", "decline", "declines", "bearish", "underperform", "fraud", "probe",
          "warning", "concern", "concerns", "slowdown"}
_NEGATORS = {"not", "no", "fails", "failed", "without", "unable", "denies", "despite"}
_EVENTS = ("earnings", "fda", "pdufa", "merger", "acquisition", "spin-off", "fomc", "investigation", "split")


class NewsIntel:
    TTL = 900

    @classmethod
    def analyze(cls, symbol: str) -> Dict[str, Any]:
        return _cached(("news", symbol), cls.TTL, lambda: cls._build(symbol)) or {"status": "NO_DATA"}

    @staticmethod
    def _score_text(text: str) -> float:
        t = " " + re.sub(r"[^a-z0-9\- ]", " ", text.lower()) + " "
        s = 0.0
        for ph in _PHRASE_ORDER:
            if ph in t:
                s += _PHRASES[ph]
                t = t.replace(ph, " ")
        toks = t.split()
        for i, w in enumerate(toks):
            val = 1.0 if w in _POS_W else -1.0 if w in _NEG_W else 0.0
            if val and any(x in _NEGATORS for x in toks[max(0, i - 3):i]):
                val = -0.7 * val
            s += val
        return s

    @classmethod
    def _build(cls, symbol: str) -> Dict[str, Any]:
        items = _safe(lambda: _ticker(symbol).news, []) or []
        if not items:
            return {"status": "NO_DATA"}
        now = pd.Timestamp.now(tz="UTC")
        seen, rows = set(), []
        for it in items[:25]:
            c = it.get("content") or {}
            title = str(it.get("title") or c.get("title") or "").strip()
            summ = str(c.get("summary") or "")
            key = re.sub(r"\W+", "", title.lower())[:60]
            if not title or key in seen:
                continue
            seen.add(key)
            ts = pd.to_datetime(c.get("pubDate"), utc=True, errors="coerce") if c.get("pubDate") else pd.NaT
            if pd.isna(ts) and it.get("providerPublishTime"):
                ts = pd.to_datetime(it["providerPublishTime"], unit="s", utc=True, errors="coerce")
            age_h = (now - ts).total_seconds() / 3600 if not pd.isna(ts) else 72.0
            raw = cls._score_text(title) + 0.5 * cls._score_text(summ)
            rows.append({"title": title, "age_h": max(age_h, 0.0), "raw": raw,
                         "w": max(0.1, math.exp(-max(age_h, 0.0) / 96.0))})
        if not rows:
            return {"status": "NO_DATA"}
        wsum = sum(r["w"] for r in rows)
        x = sum(r["w"] * r["raw"] for r in rows) / (wsum + 1.5)
        score = 50 + 45 * math.tanh(x)
        n24 = sum(1 for r in rows if r["age_h"] <= 24)
        n7 = sum(1 for r in rows if r["age_h"] <= 168)
        flags = []
        if n24 >= 5 or (n7 and n24 > 0.5 * n7 and n24 >= 3):
            flags.append("NEWS_SPIKE")
        ev = sorted({k for r in rows[:10] for k in _EVENTS if k in r["title"].lower()})
        if ev:
            flags.append("EVENT_KEYWORDS: " + ", ".join(ev))
        by = sorted(rows, key=lambda r: r["w"] * r["raw"])
        return {"status": "OK", "score": round(score, 1),
                "label": "BULLISH" if score >= 60 else "BEARISH" if score <= 40 else "NEUTRAL",
                "n": len(rows), "n_24h": n24, "n_7d": n7, "flags": flags,
                "worst": by[0]["title"] if by[0]["raw"] < 0 else None,
                "best": by[-1]["title"] if by[-1]["raw"] > 0 else None}


# ================================================================ 4. VOLUME FLOW
class VolumeFlowEngine:
    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or len(df) < 40 or df["volume"].tail(20).sum() <= 0:
            return {"status": "NO_DATA"}
        c, h, lo, v = (df[k].astype(float) for k in ("close", "high", "low", "volume"))
        v = v.clip(lower=0)
        z = (v - v.rolling(20).mean().shift(1)) / (v.rolling(20).std().shift(1) + 1e-9)
        ret = c.pct_change()
        sg = z * np.sign(ret)
        z1, sz1, sz3 = float(z.iloc[-1]), float(sg.iloc[-1]), float(sg.iloc[-3:].mean())
        r20 = ret.tail(20)
        v20 = v.tail(20)
        ratio = float(v20[r20 > 0].sum() / max(v20[r20 < 0].sum(), 1.0))
        mfm = ((c - lo) - (h - c)) / (h - lo + 1e-9)
        cmf = float((mfm * v).tail(20).sum() / (v.tail(20).sum() + 1e-9))
        obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
        obv_slope = float((obv.iloc[-1] - obv.iloc[-21]) / (v.tail(20).sum() + 1))
        px_chg = float(c.iloc[-1] / c.iloc[-21] - 1)
        div = "BULLISH_DIV" if (px_chg < 0 and obv_slope > 0.05) else "BEARISH_DIV" if (px_chg > 0 and obv_slope < -0.05) else None

        r_last = float(ret.iloc[-1])
        at_high = c.iloc[-1] >= c.iloc[-21:-1].max()
        if at_high and z1 >= 1.0:
            signal = "BREAKOUT_ON_VOLUME"
        elif r_last > 0 and z1 >= 1.0:
            signal = "UP_ON_HIGH_VOLUME"
        elif r_last < 0 and z1 >= 1.5:
            signal = "SELLING_ON_HIGH_VOLUME"
        elif r_last > 0 and z1 <= -0.7:
            signal = "LOW_VOLUME_RALLY"
        else:
            signal = "NORMAL"

        s = 50 + np.clip(sz3 * 8, -20, 20) + np.clip((ratio - 1) * 20, -15, 15) \
            + np.clip(cmf * 60, -15, 15) + np.clip(obv_slope * 40, -10, 10)
        s += 8 if div == "BULLISH_DIV" else -8 if div == "BEARISH_DIV" else 0
        s += {"BREAKOUT_ON_VOLUME": 6, "SELLING_ON_HIGH_VOLUME": -8, "LOW_VOLUME_RALLY": -3}.get(signal, 0)
        return {"status": "OK", "score": round(float(np.clip(s, 0, 100)), 1), "signal": signal,
                "vol_z": round(z1, 2), "signed_vol_z_1d": round(sz1, 2), "signed_vol_z_3d": round(sz3, 2),
                "up_down_vol_ratio": round(ratio, 2), "cmf20": round(cmf, 3), "obv_divergence": div}


# ================================================================ 5. MTF EMA ALIGNMENT
class MTFAlignment:
    @staticmethod
    def analyze(df: pd.DataFrame, df_weekly: Optional[pd.DataFrame] = None,
                mtf_result: Optional[Dict[str, Any]] = None, max_extension_pct: float = 15.0) -> Dict[str, Any]:
        if df is None or len(df) < 210:
            return {"status": "NO_DATA"}
        c = df["close"].astype(float)
        e21, e50, e200 = (c.ewm(span=n, adjust=False).mean() for n in (21, 50, 200))
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        wk = df_weekly["close"].astype(float) if df_weekly is not None and len(df_weekly) >= 30 \
            else c.resample("W-FRI").last().dropna()
        conds: List[Dict[str, Any]] = []

        def add(name, ok, weight, detail):
            conds.append({"name": name, "ok": bool(ok), "w": weight, "detail": detail})

        dist50 = (c.iloc[-1] / e50.iloc[-1] - 1) * 100
        add("Daily close > EMA50", c.iloc[-1] > e50.iloc[-1], 20, f"{dist50:+.1f}% from EMA50")
        gc_state = bool(sma50.iloc[-1] > sma200.iloc[-1])
        days = _trailing((sma50 > sma200).to_numpy())
        add("Golden-cross state (SMA50>SMA200)", gc_state, 15, f"{'above' if gc_state else 'below'} for {days} sessions")
        if len(wk) >= 30:
            we21 = wk.ewm(span=21, adjust=False).mean()
            add("Weekly close > weekly EMA21", wk.iloc[-1] > we21.iloc[-1], 20, f"{(wk.iloc[-1] / we21.iloc[-1] - 1) * 100:+.1f}%")
            add("Weekly EMA21 rising (4w)", we21.iloc[-1] > we21.iloc[-5], 15, f"{(we21.iloc[-1] / we21.iloc[-5] - 1) * 100:+.1f}% in 4 weeks")
        add("Not extended (chasing check)", dist50 <= max_extension_pct, 15,
            f"{dist50:.1f}% above EMA50 (limit {max_extension_pct:.0f}%)")
        if mtf_result and mtf_result.get("status") in ("MATCH", "WATCH", "FAILED"):
            add("MTFEMAEngine rules", mtf_result["status"] in ("MATCH", "WATCH"), 15,
                f"{mtf_result.get('status')} ({mtf_result.get('passed', 0)}/{mtf_result.get('total', 0)} timeframes)")
        tw = sum(x["w"] for x in conds)
        score = 100.0 * sum(x["w"] for x in conds if x["ok"]) / tw
        pullback = bool(0 <= dist50 <= 5 and c.iloc[-1] > e21.iloc[-1] * 0.99 and gc_state)
        return {"status": "OK", "score": round(score, 1), "conditions": conds,
                "extended": bool(dist50 > max_extension_pct), "pullback_in_uptrend": pullback,
                "dist_ema50_pct": round(float(dist50), 2)}


# ================================================================ COMBINER
class DecisionContext:
    FACTOR_WEIGHTS = {"Analysts": .20, "Options OI": .15, "News": .15, "Volume flow": .25, "MTF EMA": .25}

    @classmethod
    def build(cls, symbol: str, df: pd.DataFrame, direction: str = "LONG", price: Optional[float] = None,
              df_weekly: Optional[pd.DataFrame] = None, mtf_result: Optional[Dict[str, Any]] = None,
              include_options: bool = True) -> Dict[str, Any]:
        direction = direction.upper()
        price = price or float(df["close"].iloc[-1])
        raw = {
            "analyst": AnalystIntel.get(symbol, price),
            "options": OptionsFlowEngine.analyze(symbol, price) if include_options else {"status": "SKIPPED"},
            "news": NewsIntel.analyze(symbol),
            "volume": VolumeFlowEngine.analyze(df),
            "mtf": MTFAlignment.analyze(df, df_weekly, mtf_result),
        }
        src = {"Analysts": raw["analyst"], "Options OI": raw["options"], "News": raw["news"],
               "Volume flow": raw["volume"], "MTF EMA": raw["mtf"]}
        rows, wsum, acc = [], 0.0, 0.0
        for name, d in src.items():
            sc = d.get("score")
            usable = d.get("status") == "OK" and sc is not None and d.get("liquid", True)
            if not usable:
                rows.append({"factor": name, "status": "N/A", "score": None, "detail": d.get("status", "N/A")})
                continue
            ds = sc if direction == "LONG" else 100 - sc
            st = "CONFIRM" if ds >= 60 else "CONFLICT" if ds <= 40 else "NEUTRAL"
            if name == "Volume flow" and direction == "LONG" and d.get("signal") == "SELLING_ON_HIGH_VOLUME":
                st = "CONFLICT"
            if name == "Volume flow" and direction == "SHORT" and d.get("signal") in ("BREAKOUT_ON_VOLUME", "UP_ON_HIGH_VOLUME"):
                st = "CONFLICT"
            rows.append({"factor": name, "status": st, "score": round(ds, 1), "detail": cls._detail(name, d)})
            w = cls.FACTOR_WEIGHTS[name]
            wsum += w
            acc += w * ds
        align = acc / wsum if wsum else 50.0
        conflicts = sum(r["status"] == "CONFLICT" for r in rows)
        confirms = sum(r["status"] == "CONFIRM" for r in rows)
        by = {r["factor"]: r["status"] for r in rows}
        if by.get("Volume flow") == "CONFLICT" and by.get("MTF EMA") == "CONFLICT":
            verdict = "CONFLICTED"
        elif align >= 65 and conflicts == 0:
            verdict = "ALIGNED"
        elif align < 42 or conflicts >= 2:
            verdict = "CONFLICTED"
        else:
            verdict = "MIXED"
        return {"symbol": symbol, "direction": direction, "alignment_score": round(align, 1),
                "verdict": verdict, "confirms": confirms, "conflicts": conflicts, "rows": rows,
                "size_multiplier": {"ALIGNED": 1.10, "MIXED": 0.90, "CONFLICTED": 0.50}[verdict],
                "confidence_delta": {"ALIGNED": 4.0, "MIXED": 0.0, "CONFLICTED": -8.0}[verdict],
                "reject": verdict == "CONFLICTED" and align < 35,
                "flags": sum((d.get("flags") or [] for d in (raw["analyst"], raw["options"], raw["news"])), []),
                "raw": raw}

    @staticmethod
    def _detail(name: str, d: Dict[str, Any]) -> str:
        if name == "Analysts":
            tg = d.get("targets") or {}
            return (f"{d.get('label')} ({d.get('confidence')} conf, {d.get('n_analysts')} analysts)"
                    + (f", PT upside {tg['upside_median_pct']:+.0f}%" if tg.get("upside_median_pct") is not None else ""))
        if name == "Options OI":
            return (f"P/C OI {d.get('pc_oi')}, P/C vol {d.get('pc_vol')}, call wall {d.get('call_wall')}, "
                    f"put wall {d.get('put_wall')}, max pain {d.get('max_pain')}")
        if name == "News":
            return f"{d.get('label')} - {d.get('n')} headlines ({d.get('n_24h')} in 24h)"
        if name == "Volume flow":
            return (f"{d.get('signal')}, signed vol-z(3d) {d.get('signed_vol_z_3d'):+.1f}, "
                    f"up/down vol {d.get('up_down_vol_ratio')}, CMF {d.get('cmf20'):+.2f}")
        if name == "MTF EMA":
            ok = sum(c["ok"] for c in d["conditions"])
            return f"{ok}/{len(d['conditions'])} conditions" + (", EXTENDED" if d["extended"] else "") + \
                   (", pullback in uptrend" if d["pullback_in_uptrend"] else "")
        return ""


# ================================================================ INTEGRATION HELPERS
def apply_to_proposal(p: Any, ctx: Dict[str, Any], reject_on_conflict: bool = True):
    """Adjust size / confidence, attach the context, and blend richer analyst data. Returns None if rejected."""
    if reject_on_conflict and ctx.get("reject"):
        return None
    p.position_size_pct = round(_num(getattr(p, "position_size_pct", 0)) * ctx["size_multiplier"], 2)
    p.ai_confidence = float(np.clip(_num(getattr(p, "ai_confidence", 50)) + ctx["confidence_delta"], 0, 100))
    p.context = ctx
    sig = list(getattr(p, "signals", []) or [])
    sig.append(f"Alignment {ctx['verdict']} ({ctx['alignment_score']:.0f}/100, {ctx['confirms']} confirm / {ctx['conflicts']} conflict)")
    p.signals = sig
    intel = ctx["raw"].get("analyst") or {}
    if intel.get("status") == "OK":
        ac = dict(getattr(p, "analyst_consensus", {}) or {})
        old = _num(ac.get("score"), -1)
        new = intel["score"] if (old < 0 or not ac.get("active_sources")) else 0.5 * old + 0.5 * intel["score"]
        ac.update(score=round(new, 2), label=_label(new), intel=intel, comments=intel["comments"])
        p.analyst_consensus = ac
        p.analyst_comments = intel["comments"]
    return p


def enrich_finalists(proposals: List[Any], df_fn: Callable[[str], Optional[pd.DataFrame]], top_n: int = 15,
                     workers: int = 4, include_options: bool = True, reject_conflicted: bool = True) -> List[Any]:
    """Run the (slower) context checks only on the top_n finalists. df_fn(symbol) -> daily OHLCV DataFrame."""
    head, tail = list(proposals[:top_n]), list(proposals[top_n:])

    def one(p):
        df = df_fn(p.symbol)
        if df is None or df.empty:
            return p, None
        return p, DecisionContext.build(p.symbol, df, getattr(p, "direction", "LONG"),
                                        price=_num(getattr(p, "entry_price", 0)) or None,
                                        df_weekly=None, mtf_result=None, include_options=include_options)

    kept = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, p) for p in head]):
            try:
                p, ctx = f.result()
            except Exception:
                continue
            if ctx is None:
                kept.append(p)
                continue
            q = apply_to_proposal(p, ctx, reject_conflicted)
            if q is not None:
                kept.append(q)
    order = {id(p): i for i, p in enumerate(head)}
    kept.sort(key=lambda p: order.get(id(p), 10 ** 6))
    return kept + tail


def telegram_block(ctx: Dict[str, Any]) -> str:
    ic = {"CONFIRM": "✅", "NEUTRAL": "➖", "CONFLICT": "⚠️", "N/A": "▫️"}
    head = {"ALIGNED": "🟢", "MIXED": "🟡", "CONFLICTED": "🔴"}[ctx["verdict"]]
    lines = [f"🧭 <b>Alignment:</b> {head} {ctx['verdict']} ({ctx['alignment_score']:.0f}/100)"]
    for r in ctx["rows"]:
        if r["status"] != "N/A":
            lines.append(f"  {ic[r['status']]} {r['factor']}: {r['detail']}")
    intel = ctx["raw"].get("analyst") or {}
    for cmt in (intel.get("comments") or [])[:3]:
        lines.append(f"  💬 {cmt}")
    return "\n".join(lines)


def render_streamlit(st, ctx: Dict[str, Any]) -> None:
    icon = {"CONFIRM": "✅", "NEUTRAL": "➖", "CONFLICT": "⚠️", "N/A": "▫️"}
    st.markdown(f"**🧭 Alignment: {ctx['verdict']} ({ctx['alignment_score']:.0f}/100)** — "
                f"{ctx['confirms']} confirm · {ctx['conflicts']} conflict")
    st.dataframe(pd.DataFrame([{"": icon[r["status"]], "Factor": r["factor"], "Score": r["score"],
                                "Detail": r["detail"]} for r in ctx["rows"]]), hide_index=True, width="stretch")
    if ctx["flags"]:
        st.caption("Flags: " + " · ".join(ctx["flags"]))
    intel = ctx["raw"].get("analyst") or {}
    for cmt in (intel.get("comments") or [])[:4]:
        st.caption(f"💬 {cmt}")
