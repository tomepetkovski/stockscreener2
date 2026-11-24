"""
finnhub_client.py — drop-in Finnhub layer for the Institutional Breakout Terminal.

Fixes the endless "Finnhub 401 — rotating to next key" loop:
  * splits concatenated keys (40 chars -> 2 x 20) and de-duplicates
  * a key that returns 401/403 is marked DEAD once and never retried
  * 429 puts only that key on a 60 s cool-down
  * global per-key throttle (free tier = 60 calls/min/key) shared by all threads
  * response cache (analyst data 6 h, fundamentals 24 h)

Adds:
  * analyst_rating(symbol)      -> dict usable as the "finnhub" source in AnalystConsensusEngine
  * MultiYearGrowthFilter       -> 3-5 year earnings / revenue / margin / ROE continuity score

Keys: env FINNHUB_API_KEYS="key1,key2"  or  .streamlit/secrets.toml:
        [finnhub]
        api_keys = "key1,key2"
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("finnhub_client")

# Fallback only if env / secrets are empty. Rotate these keys (they were pasted in chat)
# and prefer FINNHUB_API_KEYS instead of hard-coding.
_FALLBACK_KEYS = "dbms6khr01qn1o670s3g,dbms6khr01qn1o670s40"


# ---------------------------------------------------------------------------
# KEY LOADING
# ---------------------------------------------------------------------------
def load_finnhub_keys() -> List[str]:
    raw = os.environ.get("FINNHUB_API_KEYS", "") or os.environ.get("FINNHUB_API_KEY", "")
    if not raw:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "finnhub" in st.secrets:
                raw = str(st.secrets["finnhub"].get("api_keys", ""))
        except Exception:
            pass
    raw = raw or _FALLBACK_KEYS

    keys: List[str] = []
    for tok in re.split(r"[,\s;]+", raw.strip()):
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) > 20 and len(tok) % 20 == 0:          # accidentally concatenated
            keys += [tok[i:i + 20] for i in range(0, len(tok), 20)]
        else:
            keys.append(tok)
    return list(dict.fromkeys(keys))


# ---------------------------------------------------------------------------
# CLIENT
# ---------------------------------------------------------------------------
class FinnhubClient:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, keys: Optional[List[str]] = None, min_interval: float = 1.1):
        self.keys = keys or load_finnhub_keys()
        self.min_interval = min_interval
        self._state: Dict[str, Dict[str, Any]] = {
            k: {"next_ok": 0.0, "cool": 0.0, "dead": False} for k in self.keys
        }
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, Tuple], Tuple[float, Any]] = {}
        logger.info("Finnhub client: %d key(s) loaded (…%s)",
                    len(self.keys), ", …".join(k[-4:] for k in self.keys))

    # -- internals ---------------------------------------------------------
    def _pick(self) -> Tuple[Optional[str], float]:
        with self._lock:
            now = time.time()
            alive = [k for k, s in self._state.items() if not s["dead"]]
            if not alive:
                return None, 0.0
            k = min(alive, key=lambda x: max(self._state[x]["next_ok"], self._state[x]["cool"]))
            s = self._state[k]
            start = max(now, s["next_ok"], s["cool"])
            s["next_ok"] = start + self.min_interval
            return k, start - now

    def _mark_dead(self, key: str, code: int) -> None:
        with self._lock:
            if not self._state[key]["dead"]:
                self._state[key]["dead"] = True
                logger.error("Finnhub key …%s rejected (HTTP %s) — disabled for this session.",
                             key[-4:], code)

    def _cooldown(self, key: str, secs: float = 60.0) -> None:
        with self._lock:
            self._state[key]["cool"] = time.time() + secs

    # -- public ------------------------------------------------------------
    def alive_keys(self) -> List[str]:
        return [k for k, s in self._state.items() if not s["dead"]]

    def validate(self) -> List[str]:
        """Probe every key once; dead ones are disabled. Returns usable keys."""
        for k in list(self.keys):
            try:
                r = requests.get(f"{self.BASE}/quote",
                                 params={"symbol": "AAPL", "token": k}, timeout=8)
                logger.info("Finnhub key …%s → HTTP %s", k[-4:], r.status_code)
                if r.status_code in (401, 403):
                    self._mark_dead(k, r.status_code)
            except Exception as exc:
                logger.warning("Finnhub probe failed for …%s: %s", k[-4:], exc)
        return self.alive_keys()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None,
            ttl: int = 6 * 3600) -> Optional[Any]:
        params = dict(params or {})
        ck = (path, tuple(sorted(params.items())))
        hit = self._cache.get(ck)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]

        for _ in range(max(2, len(self.keys) * 2)):
            key, wait = self._pick()
            if key is None:
                return None                              # all keys dead — fail fast, no log spam
            if wait > 0:
                time.sleep(min(wait, 65.0))
            try:
                r = requests.get(f"{self.BASE}{path}",
                                 params={**params, "token": key}, timeout=15)
            except requests.RequestException as exc:
                logger.debug("Finnhub network error: %s", exc)
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    return None
                self._cache[ck] = (time.time(), data)
                return data
            if r.status_code in (401, 403):
                self._mark_dead(key, r.status_code)
                continue
            if r.status_code == 429:
                self._cooldown(key)
                continue
            logger.debug("Finnhub %s → HTTP %s", path, r.status_code)
            return None
        return None


_CLIENT: Optional[FinnhubClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> FinnhubClient:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = FinnhubClient()
        return _CLIENT


# ---------------------------------------------------------------------------
# ANALYST RATING (replacement for the "finnhub" source)
# ---------------------------------------------------------------------------
def analyst_rating(symbol: str) -> Dict[str, Any]:
    """Returns {} when unavailable, else {"score": 0-100, "source": "Finnhub", ...}."""
    data = get_client().get("/stock/recommendation", {"symbol": symbol.upper()}, ttl=6 * 3600)
    if not isinstance(data, list) or not data:
        return {}
    latest = data[0]                                    # newest period first
    sb, b, h = latest.get("strongBuy", 0), latest.get("buy", 0), latest.get("hold", 0)
    s, ss = latest.get("sell", 0), latest.get("strongSell", 0)
    n = sb + b + h + s + ss
    if n < 3:
        return {}
    mean = (5 * sb + 4 * b + 3 * h + 2 * s + 1 * ss) / n          # 1..5
    score = (mean - 1.0) / 4.0 * 100.0

    # trend: did bullish share rise vs. 3 months ago?
    trend = None
    if len(data) >= 4:
        old = data[3]
        on = sum(old.get(k, 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")) or 1
        old_bull = (old.get("strongBuy", 0) + old.get("buy", 0)) / on
        new_bull = (sb + b) / n
        trend = round((new_bull - old_bull) * 100, 1)
        score = max(0.0, min(100.0, score + max(-6, min(6, trend * 0.3))))

    return {
        "score": round(score, 1),
        "source": "Finnhub",
        "analysts": n,
        "period": latest.get("period"),
        "strong_buy": sb, "buy": b, "hold": h, "sell": s, "strong_sell": ss,
        "bull_share_change_3m_pct": trend,
    }


# ---------------------------------------------------------------------------
# 3-5 YEAR GROWTH CONTINUITY
# ---------------------------------------------------------------------------
def _annual(series: Dict[str, Any], name: str, n: int = 5) -> List[float]:
    rows = ((series or {}).get("annual") or {}).get(name) or []
    rows = sorted((r for r in rows if r.get("v") is not None), key=lambda r: r.get("period", ""))
    return [float(r["v"]) for r in rows][-n:]


def _num(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


class MultiYearGrowthFilter:
    """
    Score 0-100 for multi-year continuity of profit / revenue / returns.

    Components (max):
      EPS growth 3Y>0 (15) + 5Y>0 (15)        revenue growth 3Y>0 (15) + 5Y>0 (15)
      profitable years in last 5 (15)         net margin not shrinking (10)
      ROE 5Y avg >10% (10) + ROE TTM >10% (5)
    hard_fail when growth is negative on both EPS and revenue over 3Y,
    or fewer than half of observed years were profitable.
    With strict=False hard_fail is never raised (score only).
    """

    @staticmethod
    def evaluate(symbol: str, strict: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {"available": False, "score": 50.0, "hard_fail": False,
                               "reasons": [], "label": "NO_DATA"}
        data = get_client().get("/stock/metric", {"symbol": symbol.upper(), "metric": "all"},
                                ttl=24 * 3600)
        if not isinstance(data, dict) or not data.get("metric"):
            return out

        m, series = data["metric"], data.get("series") or {}
        eps3, eps5 = _num(m.get("epsGrowth3Y")), _num(m.get("epsGrowth5Y"))
        rev3, rev5 = _num(m.get("revenueGrowth3Y")), _num(m.get("revenueGrowth5Y"))
        roe5, roe_ttm = _num(m.get("roe5Y")), _num(m.get("roeTTM"))

        score, reasons = 0.0, []

        def pts(cond: Optional[bool], p: float, ok: str, bad: str):
            nonlocal score
            if cond is None:
                return
            if cond:
                score += p
                reasons.append("✓ " + ok)
            else:
                reasons.append("✗ " + bad)

        pts(None if eps3 is None else eps3 > 0, 15, f"EPS growth 3Y {eps3:+.1f}%" if eps3 is not None else "", f"EPS growth 3Y {eps3:+.1f}%" if eps3 is not None else "")
        pts(None if eps5 is None else eps5 > 0, 15, f"EPS growth 5Y {eps5:+.1f}%" if eps5 is not None else "", f"EPS growth 5Y {eps5:+.1f}%" if eps5 is not None else "")
        pts(None if rev3 is None else rev3 > 0, 15, f"Revenue growth 3Y {rev3:+.1f}%" if rev3 is not None else "", f"Revenue growth 3Y {rev3:+.1f}%" if rev3 is not None else "")
        pts(None if rev5 is None else rev5 > 0, 15, f"Revenue growth 5Y {rev5:+.1f}%" if rev5 is not None else "", f"Revenue growth 5Y {rev5:+.1f}%" if rev5 is not None else "")

        margins = _annual(series, "netMargin", 5)
        profitable = sum(1 for v in margins if v > 0)
        observed = len(margins)
        if observed >= 3:
            score += 15.0 * profitable / observed
            reasons.append(f"{'✓' if profitable == observed else '✗'} profitable {profitable}/{observed} yrs")
            half = max(1, observed // 2)
            early, late = sum(margins[:half]) / half, sum(margins[-half:]) / half
            if late >= early - 1.0:                     # margin stable or expanding (pp)
                score += 10.0
                reasons.append("✓ net margin stable/expanding")
            else:
                reasons.append("✗ net margin shrinking")

        pts(None if roe5 is None else roe5 > 10, 10, f"ROE 5Y avg {roe5:.1f}%" if roe5 is not None else "", f"ROE 5Y avg {roe5:.1f}%" if roe5 is not None else "")
        pts(None if roe_ttm is None else roe_ttm > 10, 5, f"ROE TTM {roe_ttm:.1f}%" if roe_ttm is not None else "", f"ROE TTM {roe_ttm:.1f}%" if roe_ttm is not None else "")

        # normalise by the points that were actually measurable
        max_pts = 0.0
        max_pts += 15 if eps3 is not None else 0
        max_pts += 15 if eps5 is not None else 0
        max_pts += 15 if rev3 is not None else 0
        max_pts += 15 if rev5 is not None else 0
        max_pts += 25 if observed >= 3 else 0
        max_pts += 10 if roe5 is not None else 0
        max_pts += 5 if roe_ttm is not None else 0
        if max_pts < 40:                                 # too little evidence
            out["reasons"] = ["insufficient multi-year data"]
            return out

        final = round(score / max_pts * 100.0, 1)
        hard = bool(
            (eps3 is not None and rev3 is not None and eps3 < 0 and rev3 < 0)
            or (observed >= 4 and profitable < observed / 2)
        )
        out.update({
            "available": True,
            "score": final,
            "hard_fail": hard if strict else False,
            "reasons": reasons,
            "label": "STRONG" if final >= 75 else "SOLID" if final >= 55 else "WEAK",
            "eps_growth_3y": eps3, "eps_growth_5y": eps5,
            "revenue_growth_3y": rev3, "revenue_growth_5y": rev5,
            "profitable_years": f"{profitable}/{observed}" if observed else None,
        })
        return out
