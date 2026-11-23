"""
macro_sentiment_engine.py
=========================
Macro + sentiment awareness layer for the Institutional Breakout Terminal.

What it adds
------------
1. Macro snapshot (cached 30 min): trend, VIX level / spike / term structure,
   yields (10y 5-day move, 10y-3m curve), dollar, credit (HYG/LQD), safe-haven
   flows (TLT, GLD), risk appetite (XLY/XLP, IWM/SPY), oil.
2. Shock detector (what already happened) + early-warning list (what is building)
   -> stress_level 0..3 -> label RISK_ON / NEUTRAL / RISK_OFF / STRESS / SHOCK.
3. Market-wide news sentiment (time-decayed, negation-aware, macro lexicon).
4. Event calendar (FOMC / NFP built in, CPI etc. via env MACRO_EVENT_DATES).
5. Per-symbol overlay: regression sensitivity to rates / dollar / oil, beta in
   stress, sector tilts.
6. Policy outputs used by the terminal: score_adj, position_multiplier,
   long_threshold_bump, min_confidence_bump, block_new_longs.
7. Optional background watcher that pushes a Telegram alert when stress rises.

Optional env: FINNHUB_API_KEY (extra market news),
              MACRO_EVENT_DATES="2026-12-10:CPI,2026-12-17:PCE"
"""
from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

logger = logging.getLogger("MacroSentiment")

# ---------------------------------------------------------------------------
# tiny cache
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Tuple[float, Any]] = {}
_LOCK = threading.Lock()


def _cached(key: str, ttl: float, fn: Callable[[], Any]) -> Any:
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _LOCK:
        _CACHE[key] = (time.time(), val)
    return val


def _pct(s: Optional[pd.Series], n: int) -> float:
    if s is None:
        return 0.0
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[-n - 1] - 1) * 100) if len(s) > n else 0.0


def _chg(s: Optional[pd.Series], n: int) -> float:
    if s is None:
        return 0.0
    s = s.dropna()
    return float(s.iloc[-1] - s.iloc[-n - 1]) if len(s) > n else 0.0


def _last(s: Optional[pd.Series], default: float = 0.0) -> float:
    if s is None:
        return default
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else default


# ---------------------------------------------------------------------------
# news lexicon (phrase -> weight).  Positive = bullish for equities.
# ---------------------------------------------------------------------------
_LEXICON: Dict[str, float] = {
    # macro / policy - bearish
    "rate hike": -2.5, "rate hikes": -2.5, "hawkish": -2.0, "hot inflation": -2.5,
    "inflation surge": -2.5, "inflation jump": -2.0, "stagflation": -3.0,
    "trade war": -2.5, "tariff": -1.5, "shutdown": -1.5, "recession": -2.5,
    "credit crunch": -3.0, "bank failure": -3.0, "contagion": -2.5, "default": -2.5,
    "margin call": -2.5, "yields surge": -2.0, "yields jump": -2.0,
    "yields spike": -2.0, "volatility spike": -2.0, "sanctions": -1.5,
    "war": -2.0, "missile": -2.0, "invasion": -2.5, "weak jobs": -2.0,
    "job cuts": -1.5, "layoff": -1.5, "slowdown": -1.5, "bubble": -1.5,
    # market moves - bearish
    "sell-off": -2.5, "selloff": -2.5, "plunge": -2.5, "crash": -3.0,
    "tumble": -2.0, "slump": -2.0, "sink": -1.5, "rout": -2.5, "panic": -2.5,
    "fears": -1.0, "worries": -1.0, "downgrade": -2.0, "miss": -2.0,
    "loss": -1.5, "warning": -1.0, "cut guidance": -2.5, "profit warning": -2.5,
    "investigation": -1.5, "lawsuit": -1.2, "recall": -1.5, "bankruptcy": -3.0,
    "plummet": -2.5, "drop": -1.5, "decline": -1.2, "bearish": -2.0, "weak": -1.2,
    # bullish
    "rate cut": 2.5, "rate cuts": 2.5, "dovish": 2.0, "cooling inflation": 2.5,
    "inflation cools": 2.5, "soft landing": 2.0, "ceasefire": 2.0, "trade deal": 2.5,
    "stimulus": 2.0, "rally": 2.0, "record high": 2.0, "surge": 1.5, "soar": 2.0,
    "rebound": 1.5, "optimism": 1.5, "easing": 1.5, "yields fall": 1.5,
    "resilient": 1.2, "strong earnings": 2.0, "beat": 1.5, "upgrade": 2.0,
    "raises guidance": 2.5, "buyback": 1.2, "breakthrough": 2.0, "approval": 1.5,
    "bullish": 2.0, "outperform": 1.8, "growth": 1.0, "profit": 1.0, "record": 1.2,
    "expansion": 1.2, "dividend": 0.8,
}
_PATTERNS: List[Tuple[re.Pattern, float, str]] = sorted(
    [(re.compile(r"\b" + re.escape(p) + r"(?:s|es|ed|d|ing)?\b"), w, p) for p, w in _LEXICON.items()],
    key=lambda x: -len(x[2]),
)
_NEG_RE = re.compile(r"\b(?:no|not|without|avoids?|avoided|ends?|fades?|fading)\s+(?:\w+\s+)?$")


def _score_title(title: str) -> Tuple[float, List[str]]:
    t = " " + title.lower() + " "
    total, hits = 0.0, []
    for pat, w, phrase in _PATTERNS:
        for m in list(pat.finditer(t)):
            if not t[m.start():m.end()].strip():
                continue                      # already consumed by a longer phrase
            neg = bool(_NEG_RE.search(t[max(0, m.start() - 18):m.start()]))
            total += -w if neg else w
            hits.append(phrase)
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    return float(np.clip(total, -4.0, 4.0)), hits


def _parse_news(items: Any) -> List[Dict[str, Any]]:
    """Works for old and new yfinance payloads and Finnhub-style dicts."""
    out, seen = [], set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        c = it.get("content") or {}
        title = it.get("title") or c.get("title") or it.get("headline") or ""
        if not title:
            continue
        key = re.sub(r"\W+", "", title.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        dt = None
        try:
            ts = it.get("providerPublishTime") or it.get("datetime")
            if ts:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            else:
                pub = c.get("pubDate") or c.get("displayTime")
                if pub:
                    dt = pd.Timestamp(pub).to_pydatetime()
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = None
        link = (it.get("link") or it.get("url")
                or (c.get("canonicalUrl") or {}).get("url")
                or (c.get("clickThroughUrl") or {}).get("url") or "")
        out.append({"title": title, "link": link, "time": dt})
    return out


def _aggregate_sentiment(items: List[Dict[str, Any]], half_life_h: float = 36.0) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    rows = []
    for it in items:
        s, hits = _score_title(it["title"])
        age_h = 24.0 if it["time"] is None else max(0.0, (now - it["time"]).total_seconds() / 3600.0)
        if age_h > 24 * 7:
            continue
        rows.append((s, 0.5 ** (age_h / half_life_h), age_h, it))
    if not rows:
        return {"score": 50.0, "label": "NEUTRAL", "n": 0, "panic_count": 0, "momentum": 0.0,
                "headlines": [], "top_negative": [], "top_positive": []}
    w = np.array([r[1] for r in rows])
    s = np.array([r[0] for r in rows])
    mean = float((w * s).sum() / max(w.sum(), 1e-9))
    score = float(np.clip(50 + mean * 12, 0, 100))
    recent = [r[0] for r in rows if r[2] <= 24]
    older = [r[0] for r in rows if r[2] > 24]
    momentum = (float(np.mean(recent)) - float(np.mean(older))) if recent and older else 0.0
    rows.sort(key=lambda r: r[0])
    return {
        "score": round(score, 1),
        "label": "BULLISH" if score > 60 else "BEARISH" if score < 40 else "NEUTRAL",
        "n": len(rows),
        "panic_count": int(sum(1 for r in rows if r[0] <= -3 and r[2] <= 48)),
        "momentum": round(momentum, 2),
        "top_negative": [r[3]["title"] for r in rows[:3] if r[0] < 0],
        "top_positive": [r[3]["title"] for r in rows[::-1][:3] if r[0] > 0],
        "headlines": [{"title": r[3]["title"], "link": r[3]["link"],
                       "published": r[3]["time"].isoformat() if r[3]["time"] else ""}
                      for r in sorted(rows, key=lambda r: r[2])[:10]],
    }


def symbol_sentiment(symbol: str) -> Dict[str, Any]:
    """Drop-in replacement for SentimentEngine.analyze (same keys + extras)."""
    try:
        items = _parse_news(yf.Ticker(symbol).news)[:15]
        agg = _aggregate_sentiment(items)
        return {"score": agg["score"], "label": agg["label"], "headline_count": agg["n"],
                "headlines": agg["headlines"], "momentum": agg["momentum"],
                "panic_count": agg["panic_count"]}
    except Exception:
        return {"score": 50, "label": "NEUTRAL", "headline_count": 0, "headlines": []}


def _market_news() -> Dict[str, Any]:
    def fetch():
        items: List[Dict[str, Any]] = []
        for t in ("SPY", "QQQ", "^GSPC", "^VIX", "TLT"):
            try:
                items += _parse_news(yf.Ticker(t).news)
            except Exception:
                pass
        key = os.environ.get("FINNHUB_API_KEY", "")
        if key and requests is not None:
            try:
                r = requests.get("https://finnhub.io/api/v1/news",
                                 params={"category": "general", "token": key}, timeout=10)
                if r.status_code == 200:
                    items += _parse_news(r.json()[:40])
            except Exception:
                pass
        return _aggregate_sentiment(_parse_news([{"title": i["title"], "link": i["link"],
                                                  "providerPublishTime": i["time"].timestamp() if i["time"] else None}
                                                 for i in items]))
    return _cached("market_news", 900, fetch)


# ---------------------------------------------------------------------------
# event calendar
# ---------------------------------------------------------------------------
# FOMC decision days 2026 - verify against federalreserve.gov
_FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
              "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]


def _first_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _event_calendar(year: int) -> List[Tuple[date, str]]:
    ev: List[Tuple[date, str]] = [(date.fromisoformat(d), "FOMC") for d in _FOMC_2026]
    for y in (year, year + 1):
        ev += [(_first_friday(y, m), "NFP (approx)") for m in range(1, 13)]
    for tok in os.environ.get("MACRO_EVENT_DATES", "").split(","):
        if ":" in tok:
            d, n = tok.strip().split(":", 1)
            try:
                ev.append((date.fromisoformat(d), n.strip()))
            except ValueError:
                pass
    return ev


def upcoming_events(today: Optional[date] = None, horizon_days: int = 3) -> List[Dict[str, Any]]:
    today = today or datetime.now(timezone.utc).date()
    out = []
    for d, name in _event_calendar(today.year):
        dd = (d - today).days
        if 0 <= dd <= horizon_days:
            out.append({"event": name, "date": d.isoformat(), "days": dd})
    return sorted(out, key=lambda e: e["days"])


# ---------------------------------------------------------------------------
# macro engine
# ---------------------------------------------------------------------------
FACTORS = {
    "SPY": "SPY", "TNX": "^TNX", "IRX": "^IRX", "VIX": "^VIX", "VIX3M": "^VIX3M",
    "DXY": "DX-Y.NYB", "HYG": "HYG", "LQD": "LQD", "TLT": "TLT", "GLD": "GLD",
    "OIL": "CL=F", "IWM": "IWM", "XLY": "XLY", "XLP": "XLP",
}
_POLICY = {  # label -> (position multiplier, long-threshold bump)
    "RISK_ON": (1.0, 0.0), "NEUTRAL": (1.0, 0.0), "RISK_OFF": (0.75, 3.0),
    "STRESS": (0.5, 6.0), "SHOCK": (0.25, 10.0),
}
_SECTOR_STRESS = {"Consumer Staples": 2.0, "Healthcare": 1.5, "Utilities": 1.5,
                  "Consumer Disc.": -2.0, "Semiconductors": -2.0, "Technology": -1.0}


class MacroSentimentEngine:
    # set to safe_yf_download from the terminal to reuse its back-off logic
    downloader: Optional[Callable[..., pd.DataFrame]] = None

    # ----- data -----------------------------------------------------------
    @classmethod
    def _closes(cls) -> Optional[pd.DataFrame]:
        def fetch():
            try:
                dl = cls.downloader or (lambda *a, **k: yf.download(*a, progress=False, threads=False, **k))
                raw = dl(list(FACTORS.values()), period="1y", interval="1d", auto_adjust=True)
                if raw is None or raw.empty:
                    return None
                cl = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
                cl = cl.rename(columns={v: k for k, v in FACTORS.items()}).ffill()
                if getattr(cl.index, "tz", None) is not None:
                    cl.index = cl.index.tz_localize(None)
                cl.index = pd.to_datetime(cl.index).normalize()
                return cl
            except Exception as e:
                logger.warning("macro download failed: %s", e)
                return None
        return _cached("macro_closes", 1800, fetch)

    @classmethod
    def _factor_returns(cls) -> Optional[pd.DataFrame]:
        cl = cls._closes()
        if cl is None:
            return None
        parts = {}
        if "SPY" in cl: parts["SPY"] = cl["SPY"].pct_change()
        if "TNX" in cl: parts["TNX"] = cl["TNX"].diff()
        if "DXY" in cl: parts["DXY"] = cl["DXY"].pct_change()
        if "OIL" in cl: parts["OIL"] = cl["OIL"].pct_change()
        f = pd.DataFrame(parts).dropna()
        return f if len(f) > 100 else None

    # ----- snapshot -------------------------------------------------------
    @classmethod
    def snapshot(cls, force: bool = False) -> Dict[str, Any]:
        if force:
            with _LOCK:
                _CACHE.pop("macro_snapshot", None)
                _CACHE.pop("macro_closes", None)
                _CACHE.pop("market_news", None)
        return _cached("macro_snapshot", 1800, cls._build)

    @staticmethod
    def _fallback(note: str = "macro data unavailable") -> Dict[str, Any]:
        return {"data_ok": False, "macro_score": 50.0, "label": "NEUTRAL", "stress_level": 0,
                "shock_flags": [], "early_warnings": [note], "reasons": [note], "metrics": {},
                "news": {"score": 50.0, "label": "NEUTRAL", "n": 0, "panic_count": 0, "headlines": []},
                "events": [], "score_adj": 0.0, "position_multiplier": 1.0,
                "long_threshold_bump": 0.0, "min_confidence_bump": 0.0, "block_new_longs": False,
                "sector_tilt": {}, "timestamp": datetime.now(timezone.utc).isoformat()}

    @classmethod
    def _build(cls) -> Dict[str, Any]:
        cl = cls._closes()
        if cl is None or "SPY" not in cl or cl["SPY"].dropna().shape[0] < 60:
            return cls._fallback()

        S = lambda n: cl[n].dropna() if n in cl else None
        spy, vix, vix3m, tnx, irx = S("SPY"), S("VIX"), S("VIX3M"), S("TNX"), S("IRX")
        dxy, tlt, gld, oil = S("DXY"), S("TLT"), S("GLD"), S("OIL")
        pts: List[Tuple[float, str]] = []
        shock: List[str] = []
        early: List[str] = []
        m: Dict[str, float] = {}

        # --- equity trend / drawdown
        m["spy_1d_min5"] = float(spy.pct_change().iloc[-5:].min() * 100)
        m["spy_5d"], m["spy_20d"] = _pct(spy, 5), _pct(spy, 20)
        m["spy_dd"] = float((spy.iloc[-1] / spy.iloc[-252:].max() - 1) * 100)
        sma50 = float(spy.rolling(50).mean().iloc[-1])
        sma200 = float(spy.rolling(200).mean().iloc[-1]) if len(spy) >= 200 else sma50
        px = float(spy.iloc[-1])
        pts.append((8 if px > sma200 else -10, "SPY above 200DMA" if px > sma200 else "SPY below 200DMA"))
        pts.append((4 if px > sma50 else -6, "SPY above 50DMA" if px > sma50 else "SPY below 50DMA"))
        pts.append((float(np.clip(m["spy_20d"] * 0.8, -6, 6)), f"SPY 20d {m['spy_20d']:+.1f}%"))
        if m["spy_5d"] <= -4:
            shock.append(f"SPY {m['spy_5d']:.1f}% in 5d")
        elif m["spy_5d"] <= -2.5:
            early.append(f"SPY {m['spy_5d']:.1f}% in 5d")
        if m["spy_1d_min5"] <= -2.5:
            shock.append(f"SPY single-day {m['spy_1d_min5']:.1f}%")
        rv = spy.pct_change().rolling(5).std().dropna() * np.sqrt(252) * 100
        if len(rv) > 60:
            m["rv_z"] = float((rv.iloc[-1] - rv.iloc[-250:].mean()) / (rv.iloc[-250:].std() + 1e-9))
            if m["rv_z"] > 2:
                pts.append((-4, f"realized vol z {m['rv_z']:.1f}"))
                early.append("realized volatility spike")

        # --- VIX
        if vix is not None and len(vix) > 6:
            v = float(vix.iloc[-1]); m["vix"] = v; m["vix_5d"] = _pct(vix, 5)
            if v < 16: pts.append((6, f"VIX {v:.0f} calm"))
            elif v < 20: pts.append((2, f"VIX {v:.0f}"))
            elif v < 25: pts.append((-4, f"VIX {v:.0f} elevated"))
            elif v < 35: pts.append((-10, f"VIX {v:.0f} high"))
            else: pts.append((-16, f"VIX {v:.0f} panic"))
            if v >= 40 or m["vix_5d"] >= 50:
                shock.append(f"VIX {v:.0f} ({m['vix_5d']:+.0f}% 5d)")
            elif v >= 30:
                shock.append(f"VIX {v:.0f}")
            elif m["vix_5d"] >= 25:
                early.append(f"VIX +{m['vix_5d']:.0f}% in 5d")
                pts.append((-8, "VIX spike"))
            if vix3m is not None and len(vix3m):
                ratio = v / float(vix3m.iloc[-1]); m["vix_term"] = ratio
                if ratio > 1.0:
                    pts.append((-8, f"VIX term structure inverted ({ratio:.2f})"))
                    early.append(f"VIX backwardation {ratio:.2f}")
                elif ratio < 0.9:
                    pts.append((3, "VIX contango"))

        # --- rates
        if tnx is not None and len(tnx) > 6:
            m["tnx"] = float(tnx.iloc[-1]); m["tnx_5d_bp"] = _chg(tnx, 5) * 100
            bp = m["tnx_5d_bp"]
            if bp >= 30:
                shock.append(f"10y yield +{bp:.0f}bp in 5d")
                pts.append((-9, "rate shock"))
            elif bp >= 15:
                early.append(f"10y yield +{bp:.0f}bp in 5d")
                pts.append((-5, "rising yields"))
            if irx is not None and len(irx):
                m["curve_10y3m"] = float(tnx.iloc[-1] - irx.iloc[-1])
                if m["curve_10y3m"] < 0:
                    pts.append((-3, "inverted 10y-3m curve"))

        # --- dollar / oil
        if dxy is not None and len(dxy) > 21:
            m["dxy_20d"] = _pct(dxy, 20)
            if m["dxy_20d"] > 2: pts.append((-4, f"USD +{m['dxy_20d']:.1f}% 20d")); early.append("strong dollar")
            elif m["dxy_20d"] < -2: pts.append((3, "USD weakening"))
        if oil is not None and len(oil) > 21:
            m["oil_20d"] = _pct(oil, 20)
            if m["oil_20d"] > 15: pts.append((-4, f"oil +{m['oil_20d']:.0f}% 20d")); early.append("oil spike")

        # --- credit / safe haven / risk appetite
        if "HYG" in cl and "LQD" in cl:
            ratio = (cl["HYG"] / cl["LQD"]).dropna()
            ch = ratio.pct_change(10).dropna()
            if len(ch) > 60:
                z = float((ch.iloc[-1] - ch.iloc[-250:].mean()) / (ch.iloc[-250:].std() + 1e-9)); m["credit_z"] = z
                if z <= -2.2: shock.append(f"credit spreads widening (z {z:.1f})"); pts.append((-10, "credit stress"))
                elif z <= -1.2: early.append(f"credit weakening (z {z:.1f})"); pts.append((-6, "credit weakening"))
                elif z >= 0.5: pts.append((3, "credit supportive"))
        if tlt is not None and len(tlt) > 6:
            m["haven_5d"] = _pct(tlt, 5) - m["spy_5d"]
            if m["haven_5d"] > 3: pts.append((-4, "flight to bonds")); early.append("flight to safety (TLT vs SPY)")
        if gld is not None and len(gld) > 21:
            m["gld_20d"] = _pct(gld, 20)
        if "XLY" in cl and "XLP" in cl:
            m["cyc_def_20d"] = _pct(cl["XLY"] / cl["XLP"], 20)
            pts.append((3 if m["cyc_def_20d"] > 1 else -4 if m["cyc_def_20d"] < -2 else 0, "XLY/XLP"))
        if "IWM" in cl:
            m["small_vs_large_20d"] = _pct(cl["IWM"] / cl["SPY"], 20)
            pts.append((2 if m["small_vs_large_20d"] > 1 else -3 if m["small_vs_large_20d"] < -2 else 0, "IWM/SPY"))

        # --- news
        news = _market_news()
        pts.append((float(np.clip((news["score"] - 50) * 0.3, -9, 6)), f"news {news['score']:.0f}"))
        if news["score"] < 38 or news["panic_count"] >= 3:
            early.append(f"news panic ({news['panic_count']} severe headlines, score {news['score']:.0f})")

        # --- events
        events = upcoming_events()
        event_mult = 1.0
        for e in events:
            sev = 0.6 if e["days"] == 0 else 0.75 if e["days"] == 1 else 0.9
            if "NFP" in e["event"]: sev = min(1.0, sev + 0.1)
            event_mult = min(event_mult, sev)
            early.append(f"{e['event']} in {e['days']}d")

        score = float(np.clip(50 + sum(p for p, _ in pts), 0, 100))
        n_shock, n_early = len(shock), len(early)
        if n_shock >= 3 or m.get("vix", 0) >= 40:
            lvl = 3
        elif n_shock >= 2 or (n_shock >= 1 and n_early >= 2) or score < 30:
            lvl = 2
        elif n_shock >= 1 or n_early >= 2 or score < 45:
            lvl = 1
        else:
            lvl = 0
        label = ("SHOCK" if lvl == 3 else "STRESS" if lvl == 2 else
                 "RISK_ON" if (score >= 62 and lvl == 0) else "NEUTRAL" if score >= 45 else "RISK_OFF")
        mult, bump = _POLICY[label]
        mult *= event_mult

        tilt: Dict[str, float] = {}
        def tl(sec: str, v: float): tilt[sec] = tilt.get(sec, 0.0) + v
        bp = m.get("tnx_5d_bp", 0.0)
        if bp >= 15: [tl(s, v) for s, v in (("Utilities", -3), ("Real Estate", -3), ("Technology", -1.5), ("Semiconductors", -1.5), ("Financials", 1))]
        elif bp <= -15: [tl(s, v) for s, v in (("Utilities", 2), ("Real Estate", 2), ("Technology", 1.5))]
        if m.get("oil_20d", 0) >= 10: [tl(s, v) for s, v in (("Energy", 3), ("Consumer Disc.", -1.5), ("Industrials", -1))]
        elif m.get("oil_20d", 0) <= -10: [tl(s, v) for s, v in (("Energy", -3), ("Consumer Disc.", 1))]
        if m.get("credit_z", 0) <= -1.5: tl("Financials", -2)
        if m.get("dxy_20d", 0) >= 2: [tl(s, -1.0) for s in ("Materials", "Energy")]
        if lvl >= 2: [tl(s, v) for s, v in _SECTOR_STRESS.items()]

        score_adj = float(np.clip((score - 50) * 0.25, -12, 8)) - (4.0 if events and events[0]["days"] <= 1 else 0.0)
        return {
            "data_ok": True, "macro_score": round(score, 1), "label": label, "stress_level": lvl,
            "shock_flags": shock, "early_warnings": early,
            "reasons": [t for p, t in sorted(pts, key=lambda x: abs(x[0]), reverse=True)[:6] if p],
            "metrics": {k: round(v, 3) for k, v in m.items()}, "news": news, "events": events,
            "score_adj": round(score_adj, 2), "position_multiplier": round(mult, 3),
            "long_threshold_bump": bump, "min_confidence_bump": bump,
            "block_new_longs": lvl >= 3, "sector_tilt": tilt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ----- per-symbol overlay ---------------------------------------------
    @classmethod
    def symbol_overlay(cls, symbol: str, df: Optional[pd.DataFrame], snap: Dict[str, Any],
                       sector: str = "") -> Dict[str, Any]:
        """Score adjustment (points added to the AI score) + human-readable signals."""
        out = {"score_adj": float(snap.get("score_adj", 0.0)), "signals": [], "beta": None}
        sig: List[str] = out["signals"]
        if snap.get("label") not in (None, "NEUTRAL"):
            sig.append(f"Macro {snap['label']} (score {snap.get('macro_score', 50):.0f}, adj {out['score_adj']:+.1f})")
        adj = out["score_adj"]
        lvl = int(snap.get("stress_level", 0))

        t = (snap.get("sector_tilt") or {}).get(sector, 0.0)
        if t:
            adj += t; sig.append(f"Sector macro tilt {sector}: {t:+.1f}")

        try:
            fr = cls._factor_returns()
            if fr is not None and df is not None and len(df) >= 100:
                r = df["close"].pct_change().dropna()
                idx = pd.to_datetime(r.index)
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_localize(None)
                r.index = idx.normalize()
                X = fr.reindex(r.index).dropna()
                y = r.reindex(X.index).iloc[-120:]
                X = X.loc[y.index]
                if len(y) >= 80:
                    A = np.column_stack([np.ones(len(y)), X.values])
                    coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
                    pred = A @ coef
                    r2 = 1 - ((y.values - pred) ** 2).sum() / (((y.values - y.values.mean()) ** 2).sum() + 1e-12)
                    b = dict(zip(X.columns, coef[1:]))
                    beta = float(b.get("SPY", 1.0)); out["beta"] = round(beta, 2)
                    if r2 > 0.05:
                        f5 = {"TNX": float(fr["TNX"].iloc[-5:].sum()) if "TNX" in fr else 0.0,
                              "DXY": float(fr["DXY"].iloc[-5:].sum()) if "DXY" in fr else 0.0,
                              "OIL": float(fr["OIL"].iloc[-5:].sum()) if "OIL" in fr else 0.0}
                        impulse = 100 * sum(b.get(k, 0.0) * v for k, v in f5.items())
                        imp_adj = float(np.clip(impulse, -5, 5))
                        if abs(imp_adj) >= 1:
                            adj += imp_adj
                            sig.append(f"Macro factor impulse (rates/USD/oil): {impulse:+.1f}% -> {imp_adj:+.1f}")
                    if lvl >= 2 and beta > 1.3:
                        pen = float(np.clip((beta - 1.0) * 4, 0, 6)); adj -= pen
                        sig.append(f"High beta {beta:.2f} in {snap.get('label')}: -{pen:.1f}")
                    elif lvl >= 2 and beta < 0.7:
                        adj += 2; sig.append(f"Defensive beta {beta:.2f} in {snap.get('label')}: +2")
        except Exception as e:
            logger.debug("symbol_overlay %s: %s", symbol, e)

        out["score_adj"] = float(np.clip(adj, -15, 10))
        return out


# ---------------------------------------------------------------------------
# presentation helpers
# ---------------------------------------------------------------------------
_EMOJI = {"RISK_ON": "🟢", "NEUTRAL": "🟡", "RISK_OFF": "🟠", "STRESS": "🔴", "SHOCK": "🚨"}


def telegram_line(snap: Dict[str, Any]) -> str:
    e = _EMOJI.get(snap.get("label"), "⚪")
    w = "; ".join((snap.get("shock_flags") or snap.get("early_warnings") or [])[:2])
    return (f"{e} <b>Macro:</b> {html.escape(str(snap.get('label')))} "
            f"(score {snap.get('macro_score', 50):.0f}, news {snap.get('news', {}).get('score', 50):.0f}, "
            f"size x{snap.get('position_multiplier', 1):.2f})"
            + (f"\n⚠️ {html.escape(w)}" if w else ""))


def alert_text(snap: Dict[str, Any]) -> str:
    lines = [f"{_EMOJI.get(snap['label'], '⚪')} <b>MACRO ALERT — {snap['label']}</b> (stress {snap['stress_level']}/3)",
             f"Macro score {snap['macro_score']:.0f} | News {snap['news'].get('score', 50):.0f} | "
             f"size x{snap['position_multiplier']:.2f} | longs threshold +{snap['long_threshold_bump']:.0f}"]
    for s in snap["shock_flags"][:4]:
        lines.append(f"🔻 {html.escape(s)}")
    for s in snap["early_warnings"][:4]:
        lines.append(f"⚠️ {html.escape(s)}")
    for h in snap["news"].get("top_negative", [])[:2]:
        lines.append(f"📰 {html.escape(h[:110])}")
    return "\n".join(lines)


def render_macro_panel(st, snap: Dict[str, Any]) -> None:
    mt = snap.get("metrics", {})
    with st.container(border=True):
        st.markdown(f"### 🧭 Macro & Sentiment Awareness — {_EMOJI.get(snap['label'], '')} {snap['label']}")
        c = st.columns(5)
        c[0].metric("Macro score", f"{snap['macro_score']:.0f}")
        c[1].metric("Stress level", f"{snap['stress_level']}/3")
        c[2].metric("News sentiment", f"{snap['news'].get('score', 50):.0f}", snap["news"].get("label"))
        c[3].metric("Size multiplier", f"x{snap['position_multiplier']:.2f}")
        c[4].metric("Long threshold", f"+{snap['long_threshold_bump']:.0f}")
        c = st.columns(5)
        c[0].metric("VIX", f"{mt.get('vix', 0):.1f}", f"{mt.get('vix_5d', 0):+.0f}% 5d")
        c[1].metric("VIX/VIX3M", f"{mt.get('vix_term', 0):.2f}")
        c[2].metric("10y yield", f"{mt.get('tnx', 0):.2f}%", f"{mt.get('tnx_5d_bp', 0):+.0f}bp 5d")
        c[3].metric("Credit z", f"{mt.get('credit_z', 0):.1f}")
        c[4].metric("SPY 5d", f"{mt.get('spy_5d', 0):+.1f}%")
        if snap["block_new_longs"]:
            st.error("SHOCK: new longs blocked unless confidence >= 80.")
        for s in snap["shock_flags"]:
            st.error(f"🔻 {s}")
        for s in snap["early_warnings"]:
            st.warning(f"⚠️ {s}")
        if snap["events"]:
            st.info("📅 " + " · ".join(f"{e['event']} in {e['days']}d" for e in snap["events"]))
        with st.expander("Drivers & headlines", expanded=False):
            st.caption("Drivers: " + " | ".join(snap["reasons"]))
            for h in snap["news"].get("headlines", [])[:8]:
                st.markdown(f"• [{h['title']}]({h['link']})" if h.get("link") else f"• {h['title']}")


def start_macro_watch(send_fn: Callable[[str], Any], interval_s: int = 1800) -> threading.Thread:
    """Background thread: alerts when stress level rises (or on first run if >= 2)."""
    def loop():
        prev: Optional[int] = None
        while True:
            try:
                snap = MacroSentimentEngine.snapshot(force=True)
                lvl = snap["stress_level"]
                if (prev is None and lvl >= 2) or (prev is not None and lvl > prev):
                    send_fn(alert_text(snap))
                prev = lvl
            except Exception:
                logger.exception("macro watcher error")
            time.sleep(interval_s)
    th = threading.Thread(target=loop, daemon=True, name="MacroWatch")
    th.start()
    return th
