"""
investment_quality.py — drop-in upgrades for the Institutional Breakout Terminal.

Adds:
  1. LiquidityGate        – price / dollar-volume / staleness / volatility filter
  2. QualityEngine        – REAL Piotroski F (9 tests) + Altman Z from statements,
                            FCF yield, PEG, balance-sheet veto
  3. GrowthContinuityEngineFixed – fixes inverted yfinance column order, dead
                            hard-fail, and the -12.5 penalty on missing data
  4. risk_adjusted_score  – ranking without the small-cap / "BREAKOUT" bias
  5. PortfolioBudget      – enforces regime gross cap, per-position cap, sector cap
  6. finalize_proposals   – ONE place for rank -> journal sizing -> portfolio
                            construction -> budget -> journal log (UI + auto-scan)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("investment_quality")

__all__ = [
    "LiquidityGate", "QualityEngine", "GrowthContinuityEngineFixed",
    "risk_adjusted_score", "PortfolioBudget", "finalize_proposals",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row(df: Optional[pd.DataFrame], names: List[str]) -> Optional[pd.Series]:
    """First matching statement row, returned OLDEST -> NEWEST (yfinance gives newest first)."""
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if len(s):
                return s.sort_index()
    return None


def _ratio(a: float, b: float) -> float:
    a, b = float(a), float(b)
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        raise ValueError("bad ratio")
    return a / b


# ---------------------------------------------------------------------------
# 1. LIQUIDITY / DATA-QUALITY GATE
# ---------------------------------------------------------------------------
class LiquidityGate:
    MIN_PRICE = 5.0
    MIN_DOLLAR_VOL = 10_000_000      # median 20-day $ volume
    MAX_STALE_DAYS = 5
    MAX_ATR_PCT = 8.0                # >8% daily ATR is not a 3-12 month holding

    @classmethod
    def check(cls, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if df is None or df.empty or len(df) < 60:
            return {"ok": False, "reasons": ["insufficient data"]}
        reasons: List[str] = []
        last = float(df["close"].iloc[-1])
        dv = float((df["close"] * df["volume"]).iloc[-20:].median())
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        age = (today - pd.Timestamp(df.index[-1]).normalize()).days
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        atr_pct = float(tr.rolling(14).mean().iloc[-1] / last * 100) if last > 0 else 99.0

        if last < cls.MIN_PRICE:
            reasons.append(f"price ${last:.2f} < ${cls.MIN_PRICE:.0f}")
        if dv < cls.MIN_DOLLAR_VOL:
            reasons.append(f"$vol {dv/1e6:.1f}M < {cls.MIN_DOLLAR_VOL/1e6:.0f}M")
        if age > cls.MAX_STALE_DAYS:
            reasons.append(f"stale data ({age}d)")
        if atr_pct > cls.MAX_ATR_PCT:
            reasons.append(f"ATR {atr_pct:.1f}% too volatile")
        return {"ok": not reasons, "reasons": reasons,
                "dollar_volume": dv, "atr_pct": atr_pct, "age_days": age}


# ---------------------------------------------------------------------------
# 2. QUALITY ENGINE (real Piotroski + Altman)
# ---------------------------------------------------------------------------
class QualityEngine:
    _CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    _LOCK = threading.Lock()
    _TTL = 24 * 3600

    @staticmethod
    def _statements(symbol: str):
        t = yf.Ticker(symbol)
        try:
            fin = t.get_income_stmt()
        except Exception:
            fin = getattr(t, "financials", None)
        bs = getattr(t, "balance_sheet", None)
        cf = getattr(t, "cashflow", None)
        return fin, bs, cf

    @staticmethod
    def _piotroski(fin, bs, cf) -> Dict[str, bool]:
        ni = _row(fin, ["Net Income", "Net Income Common Stockholders"])
        rev = _row(fin, ["Total Revenue", "Revenue", "Operating Revenue"])
        gp = _row(fin, ["Gross Profit"])
        ta = _row(bs, ["Total Assets"])
        ca = _row(bs, ["Current Assets", "Total Current Assets"])
        cl = _row(bs, ["Current Liabilities", "Total Current Liabilities"])
        ltd = _row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        sh = _row(bs, ["Ordinary Shares Number", "Share Issued"])
        cfo = _row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                        "Total Cash From Operating Activities"])
        tests: Dict[str, bool] = {}

        def t(name, fn):
            try:
                v = fn()
            except Exception:
                return
            if v is not None:
                tests[name] = bool(v)

        def two(*series):
            return all(s is not None and len(s) >= 2 for s in series)

        if ni is not None and ta is not None:
            t("roa_positive", lambda: _ratio(ni.iloc[-1], ta.iloc[-1]) > 0)
        if two(ni, ta):
            t("roa_improving", lambda: _ratio(ni.iloc[-1], ta.iloc[-1]) > _ratio(ni.iloc[-2], ta.iloc[-2]))
        if cfo is not None:
            t("cfo_positive", lambda: cfo.iloc[-1] > 0)
        if cfo is not None and ni is not None:
            t("accruals_ok", lambda: cfo.iloc[-1] > ni.iloc[-1])
        if two(ltd, ta):
            t("leverage_down", lambda: _ratio(ltd.iloc[-1], ta.iloc[-1]) <= _ratio(ltd.iloc[-2], ta.iloc[-2]))
        if two(ca, cl):
            t("liquidity_up", lambda: _ratio(ca.iloc[-1], cl.iloc[-1]) > _ratio(ca.iloc[-2], cl.iloc[-2]))
        if two(sh):
            t("no_dilution", lambda: sh.iloc[-1] <= sh.iloc[-2] * 1.01)
        if two(gp, rev):
            t("gross_margin_up", lambda: _ratio(gp.iloc[-1], rev.iloc[-1]) > _ratio(gp.iloc[-2], rev.iloc[-2]))
        if two(rev, ta):
            t("asset_turnover_up", lambda: _ratio(rev.iloc[-1], ta.iloc[-1]) > _ratio(rev.iloc[-2], ta.iloc[-2]))
        return tests

    @staticmethod
    def _altman(fin, bs, info) -> Optional[float]:
        if str(info.get("sector", "")) == "Financial Services":
            return None   # Z-score is not meaningful for banks / insurers
        try:
            ta = _row(bs, ["Total Assets"]).iloc[-1]
            ca = _row(bs, ["Current Assets", "Total Current Assets"]).iloc[-1]
            cl = _row(bs, ["Current Liabilities", "Total Current Liabilities"]).iloc[-1]
            re_ = _row(bs, ["Retained Earnings"]).iloc[-1]
            tl = _row(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"]).iloc[-1]
            ebit = _row(fin, ["EBIT", "Operating Income"]).iloc[-1]
            sales = _row(fin, ["Total Revenue", "Revenue"]).iloc[-1]
            mc = float(info.get("marketCap") or 0)
            if mc <= 0:
                return None
            z = (1.2 * _ratio(ca - cl, ta) + 1.4 * _ratio(re_, ta) + 3.3 * _ratio(ebit, ta)
                 + 0.6 * _ratio(mc, tl) + 1.0 * _ratio(sales, ta))
            return float(z)
        except Exception:
            return None

    @classmethod
    def evaluate(cls, symbol: str, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with cls._LOCK:
            hit = cls._CACHE.get(symbol)
            if hit and time.time() - hit[0] < cls._TTL:
                return hit[1]

        info = info or {}
        out: Dict[str, Any] = {
            "quality_score": 50.0, "piotroski": None, "piotroski_tests": 0,
            "altman_z": None, "altman_zone": "N/A", "fcf_yield_pct": None,
            "peg": None, "flags": [], "veto": False, "available": False,
        }
        try:
            fin, bs, cf = cls._statements(symbol)
            tests = cls._piotroski(fin, bs, cf)
            n = len(tests)
            score = 50.0
            if n >= 5:                                   # need enough tests to trust it
                f = sum(tests.values())
                f_scaled = f / n * 9.0
                out.update(piotroski=round(f_scaled, 1), piotroski_tests=n, available=True)
                score += (f_scaled - 4.5) / 4.5 * 20.0
                if f_scaled <= 3:
                    out["flags"].append(f"weak Piotroski {f_scaled:.1f}/9")

            z = cls._altman(fin, bs, info)
            if z is not None:
                out["altman_z"] = round(z, 2)
                if z >= 2.99:
                    out["altman_zone"] = "SAFE"; score += 8
                elif z >= 1.81:
                    out["altman_zone"] = "GREY"
                else:
                    out["altman_zone"] = "DISTRESS"; score -= 20
                    out["flags"].append(f"Altman Z {z:.2f} (distress zone)")

            mc = float(info.get("marketCap") or 0)
            fcf_s = _row(cf, ["Free Cash Flow"])
            fcf = float(fcf_s.iloc[-1]) if fcf_s is not None else float(info.get("freeCashflow") or 0)
            if mc > 0 and fcf:
                y = fcf / mc * 100
                out["fcf_yield_pct"] = round(y, 2)
                score += 10 if y > 5 else 5 if y > 2 else -10 if y < 0 else 0
                if y < 0:
                    out["flags"].append("negative FCF")

            peg = info.get("trailingPegRatio") or info.get("pegRatio")
            if peg:
                peg = float(peg)
                out["peg"] = round(peg, 2)
                if 0 < peg < 1.5:
                    score += 6
                elif peg > 3:
                    score -= 6
                    out["flags"].append(f"PEG {peg:.1f} rich")

            out["quality_score"] = float(np.clip(score, 0, 100))
            # Hard veto only on the combination that actually predicts blow-ups
            out["veto"] = bool(out["altman_zone"] == "DISTRESS"
                               and (out["piotroski"] is not None and out["piotroski"] <= 3))
        except Exception as e:
            logger.debug("[%s] QualityEngine: %s", symbol, e)

        with cls._LOCK:
            cls._CACHE[symbol] = (time.time(), out)
        return out


# ---------------------------------------------------------------------------
# 3. FIXED GROWTH CONTINUITY
# ---------------------------------------------------------------------------
class GrowthContinuityEngineFixed:
    """Same output keys as GrowthContinuityEngine, but:
       * statement rows are sorted oldest->newest (yfinance returns newest first,
         which inverted every growth rate and CAGR in the original)
       * hard-fail is reachable (original required years>=6; annual data has <=4 intervals)
       * missing data -> neutral 50, not 0 (0 cost every such stock -12.5 AI points)
    """

    @staticmethod
    def _cagr(s: pd.Series) -> Optional[float]:
        if s is None or len(s) < 3:
            return None
        a, b = float(s.iloc[0]), float(s.iloc[-1])
        if a <= 0 or b <= 0:
            return None
        return ((b / a) ** (1 / (len(s) - 1)) - 1) * 100

    @staticmethod
    def _consec_up(s: pd.Series) -> int:
        n = 0
        vals = s.values.astype(float)
        for i in range(len(vals) - 1, 0, -1):          # newest backwards
            if vals[i] > vals[i - 1]:
                n += 1
            else:
                break
        return n

    @classmethod
    def evaluate(cls, symbol: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "growth_score": 50.0, "growth_available": False, "years_with_data": 0,
            "consecutive_growth_years": 0, "revenue_cagr_3y": 0.0, "eps_cagr_3y": 0.0,
            "hard_fail": False, "reasons": [],
        }
        try:
            t = yf.Ticker(symbol)
            try:
                inc = t.get_income_stmt()
            except Exception:
                inc = getattr(t, "financials", None)
            rev = _row(inc, ["Total Revenue", "Revenue", "Operating Revenue"])
            eps = _row(inc, ["Diluted EPS", "Basic EPS"])
            ni = _row(inc, ["Net Income", "Net Income Common Stockholders"])
            if rev is None or len(rev) < 3:
                out["reasons"].append("not enough revenue history")
                return out

            earn = eps if eps is not None and len(eps) >= 3 else ni
            rev_cagr = cls._cagr(rev)
            earn_cagr = cls._cagr(earn) if earn is not None else None
            consec = cls._consec_up(rev)
            profitable = bool(ni is not None and len(ni) >= 3 and (ni.tail(3) > 0).all())

            out.update(
                growth_available=True,
                years_with_data=len(rev) - 1,
                consecutive_growth_years=consec,
                revenue_cagr_3y=round(rev_cagr, 2) if rev_cagr is not None else 0.0,
                eps_cagr_3y=round(earn_cagr, 2) if earn_cagr is not None else 0.0,
            )

            score = 20.0
            score += 35 if consec >= 3 else 20 if consec == 2 else 8 if consec == 1 else 0
            if rev_cagr is not None:
                score += 15 if rev_cagr > 10 else 8 if rev_cagr > 0 else 0
            if earn_cagr is not None:
                score += 15 if earn_cagr > 10 else 8 if earn_cagr > 0 else 0
            if profitable:
                score += 10
            out["growth_score"] = round(min(100.0, score), 1)

            if consec == 0 and rev_cagr is not None and rev_cagr < -5:
                out["hard_fail"] = True
                out["reasons"].append(f"revenue shrinking (CAGR {rev_cagr:.1f}%)")
            if ni is not None and len(ni) >= 3 and (ni.tail(3) < 0).all() and (rev_cagr or 0) < 5:
                out["hard_fail"] = True
                out["reasons"].append("3 loss-making years without strong revenue growth")
            return out
        except Exception as e:
            out["reasons"].append(f"error: {e}")
            return out


# ---------------------------------------------------------------------------
# 4. RISK-ADJUSTED RANKING
# ---------------------------------------------------------------------------
def _f(v, d=0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else d
    except Exception:
        return d


def risk_adjusted_score(p: Any, regime: str = "NEUTRAL") -> float:
    """Blend of conviction and *payoff*, minus risk. No cap-tier / opp-type bonus."""
    dec = (getattr(p, "decision", None) or {}).get("score", 0)
    base = (0.35 * _f(getattr(p, "ai_confidence", 0))
            + 0.25 * _f(dec)
            + 0.25 * _f(getattr(p, "upside_score", 0))
            + 0.15 * _f(getattr(p, "short_term_score", 0)))

    rr = min(max(_f(getattr(p, "risk_reward", 0)), 0.0), 4.0)
    base += rr / 4.0 * 8.0                                         # payoff asymmetry: 0..+8

    n_bt = int(_f(getattr(p, "bt_trades", 0)))
    if n_bt >= 12:                                                 # only trust a real sample
        base += float(np.clip(_f(getattr(p, "bt_expectancy_R", 0)), -0.5, 1.0)) * 8.0

    base += (_f(getattr(p, "fundamental_quality", 45), 45) - 45) * 0.08
    base -= _f(getattr(p, "event_risk_score", 0)) * 0.10
    base -= (1.0 - _f(getattr(p, "crash_position_adjustment", 1.0), 1.0)) * 12.0

    entry, sl = _f(getattr(p, "entry_price", 0)), _f(getattr(p, "stop_loss", 0))
    if entry > 0 and sl > 0:
        stop_pct = abs(entry - sl) / entry * 100
        if stop_pct > 12:
            base -= min(8.0, (stop_pct - 12) * 0.8)                # very wide stops = hidden risk
    return round(base, 3)


# ---------------------------------------------------------------------------
# 5. PORTFOLIO RISK BUDGET
# ---------------------------------------------------------------------------
class PortfolioBudget:
    REGIME_GROSS_CAP = {            # max total long exposure, % of equity
        "STRONG_BULL": 90.0, "BULL": 80.0, "NEUTRAL": 65.0,
        "CAUTIOUS": 45.0, "BEAR": 30.0, "STRONG_BEAR": 15.0,
    }
    SECTOR_CAP = 25.0               # % of equity per sector

    @classmethod
    def apply(cls, proposals: List[Any], market_regime: Optional[Dict[str, Any]]):
        """Proposals must already be in priority order. position_size_pct is % of equity."""
        mr = market_regime or {}
        regime = mr.get("_overall", "NEUTRAL")
        per_pos = _f((mr.get("_bias") or {}).get("max_position_size", 5), 5.0) or 5.0
        gross_cap = cls.REGIME_GROSS_CAP.get(regime, 60.0)

        used, sector_used = 0.0, {}
        kept: List[Any] = []
        notes: List[str] = []
        for p in proposals:
            sym = getattr(p, "symbol", "?")
            raw = _f(getattr(p, "position_size_pct", 0))
            sec = getattr(p, "sector_exposure", "DEFAULT") or "DEFAULT"
            want = min(raw, per_pos)
            size = min(want, gross_cap - used, cls.SECTOR_CAP - sector_used.get(sec, 0.0))
            if size <= 0 or size < 0.25 * want:
                notes.append(f"{sym}: dropped (budget exhausted: gross {used:.1f}/{gross_cap:.0f}%, {sec} {sector_used.get(sec, 0):.1f}%)")
                continue
            if size < raw:
                notes.append(f"{sym}: size {raw:.1f}% -> {size:.1f}%")
            try:
                p.position_size_pct = round(size, 2)
            except Exception:
                pass
            used += size
            sector_used[sec] = sector_used.get(sec, 0.0) + size
            kept.append(p)
        return kept, {"regime": regime, "gross_cap": gross_cap, "gross_used": round(used, 2),
                      "per_position_cap": per_pos, "sector_used": sector_used, "notes": notes}


# ---------------------------------------------------------------------------
# 6. ONE FINALIZATION PIPELINE
# ---------------------------------------------------------------------------
def finalize_proposals(
    proposals: List[Any],
    market_regime: Optional[Dict[str, Any]],
    journal: Any = None,
    constructor: Any = None,
    top_n: Optional[int] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """rank -> journal sizing -> portfolio construction -> budget -> journal log."""
    regime = (market_regime or {}).get("_overall", "NEUTRAL")
    ranked = sorted(proposals or [], key=lambda p: -risk_adjusted_score(p, regime))
    for p in ranked:
        setattr(p, "_final_score", risk_adjusted_score(p, regime))

    if journal is not None:
        for p in ranked:
            if getattr(p, "_journal_sized", False):
                continue
            try:
                p.position_size_pct = _f(p.position_size_pct) * _f(journal.size_multiplier(p.ai_confidence), 1.0)
                p._journal_sized = True
            except Exception:
                pass

    if constructor is not None:
        try:
            sel = constructor.select(ranked, regime).get("selected") or ranked
            keep = {id(x) for x in sel}
            ranked = [p for p in ranked if id(p) in keep]
        except Exception as e:
            logger.debug("constructor failed: %s", e)

    if top_n:
        ranked = ranked[:top_n * 2]          # leave room for the budget to drop some
    kept, report = PortfolioBudget.apply(ranked, market_regime)
    if top_n:
        kept = kept[:top_n]

    if journal is not None:
        for p in kept:
            if getattr(p, "_journaled", False):
                continue
            try:
                journal.log(p, regime)
                p._journaled = True
            except Exception:
                pass
    return kept, report
