"""
decision_engine.py
==================
Fuses every analysis pillar of the Institutional Breakout Terminal into ONE
investment decision:

    trend (EMA/MA/Ichimoku/VWAP) · breakout · order flow · volume direction + z-score
    technicals (RSI/MACD/stretch z) · AI + ML models · fundamentals · analyst rating
    sentiment · macro risk (regime/VIX/DXY/sector flow)

Outputs
-------
score / verdict / conviction / agreement between pillars
expected price for 1 week, 1 month, 6 months (median, 1-sigma band, P(up))
entry style (buy now / pullback / break trigger), size multiplier, vetoes,
top pros and cons, invalidation level.

Self-contained: needs only numpy + pandas. No import from the terminal file,
so there is no circular-import risk.

IMPORTANT: expected prices are *scenario estimates* (shrunk historical drift +
a bounded tilt from the composite score + volatility cone). They are not
forecasts and carry no guarantee.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- tunables
MAX_ANNUAL_TILT = 0.30        # max extra annual drift the composite can add (+/-30 %)
HIST_DRIFT_SHRINK = 0.25      # keep only 25 % of the stock's own historical drift
HORIZONS = {"1W": 5, "1M": 21, "6M": 126}   # trading days
LEVELS = ["AVOID", "REDUCE", "HOLD", "ACCUMULATE", "BUY", "STRONG BUY"]


def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip(x: float, lo: float = -100.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


Pillar = Tuple[Optional[float], List[str]]     # (score -100..+100 or None=no data, notes)


class DecisionEngine:

    # weights sum to 1.0; pillars with no data are dropped and the rest renormalised
    WEIGHTS: Dict[str, float] = {
        "trend": 0.14,
        "breakout": 0.14,
        "order_flow": 0.12,
        "volume": 0.10,
        "technical": 0.08,
        "ai_model": 0.12,
        "fundamental": 0.12,
        "analyst": 0.08,
        "sentiment": 0.04,
        "macro": 0.06,
    }

    # ================================================================ PUBLIC
    @classmethod
    def decide(
        cls,
        df: pd.DataFrame,
        features: Dict[str, Any],
        ai: Optional[Dict[str, Any]] = None,
        ml: Optional[Dict[str, Any]] = None,
        analysts: Optional[Dict[str, Any]] = None,
        funds: Optional[Dict[str, Any]] = None,
        sentiment: Optional[Dict[str, Any]] = None,
        market_regime: Optional[Dict[str, Any]] = None,
        sector_flow: Optional[Dict[str, Any]] = None,
        crash: Optional[Dict[str, Any]] = None,
        event_risk: Optional[Dict[str, Any]] = None,
        rr: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        f = features or {}
        c = df["close"].astype(float)
        px = float(c.iloc[-1])

        raw: Dict[str, Pillar] = {
            "trend": cls._trend(f),
            "breakout": cls._breakout(f),
            "order_flow": cls._order_flow(f),
            "volume": cls._volume(df, f),
            "technical": cls._technical(df, f),
            "ai_model": cls._ai(ai or {}, ml or {}),
            "fundamental": cls._fundamental(funds or {}),
            "analyst": cls._analyst(analysts or {}, funds or {}),
            "sentiment": cls._sentiment(sentiment or {}),
            "macro": cls._macro(market_regime or {}, sector_flow or {}, funds or {}),
        }
        avail = {k: v for k, v in raw.items() if v[0] is not None}
        wsum = sum(cls.WEIGHTS[k] for k in avail)
        S = sum(cls.WEIGHTS[k] * avail[k][0] for k in avail) / wsum if wsum > 0 else 0.0

        n = len(avail)
        bull = sum(1 for k in avail if avail[k][0] > 15)
        bear = sum(1 for k in avail if avail[k][0] < -15)
        agreement = (max(bull, bear) / n) if n else 0.0

        # ---------- base verdict
        if S >= 45:
            verdict = "STRONG BUY"
        elif S >= 25:
            verdict = "BUY"
        elif S >= 10:
            verdict = "ACCUMULATE"
        elif S > -10:
            verdict = "HOLD"
        elif S > -25:
            verdict = "REDUCE"
        else:
            verdict = "AVOID"
        if verdict == "STRONG BUY" and agreement < 0.6:
            verdict = "BUY"

        # ---------- expected prices
        expected = cls.expected_prices(df, S, ml, funds)

        # ---------- vetoes (each one caps the verdict)
        vetoes: List[str] = []
        caps: List[str] = []
        size_mult_extra = 1.0

        if wsum < 0.60:
            caps.append("HOLD"); vetoes.append(f"Only {wsum:.0%} of the evidence has data")
        if bull >= 3 and bear >= 3:
            caps.append("ACCUMULATE"); vetoes.append(f"Pillars conflict ({bull} bullish vs {bear} bearish)")
        if rr is not None and _f(rr) < 1.5:
            caps.append("ACCUMULATE"); vetoes.append(f"Risk:reward only {_f(rr):.1f}")
        if expected:
            if expected["6M"]["exp_return_pct"] < 0:
                caps.append("HOLD"); vetoes.append("Expected 6-month return is negative")
            if expected["1M"]["prob_up"] < 0.48:
                caps.append("ACCUMULATE"); vetoes.append("1-month drift is negative (P(up) < 48%)")
        ev = event_risk or {}
        adv = ev.get("position_advice", "STANDARD")
        if adv == "AVOID_NEW":
            caps.append("HOLD"); vetoes.append("Earnings/event risk HIGH: " + ", ".join(ev.get("flags", [])[:2]))
        elif adv == "REDUCE_SIZE":
            caps.append("BUY"); size_mult_extra *= 0.6; vetoes.append("Event risk MEDIUM: size reduced")
        cr = crash or {}
        crash_lvl = cr.get("warning_level", "LOW")
        crash_adj = _f(cr.get("position_adjustment"), 1.0)
        if crash_lvl == "CRITICAL":
            caps.append("HOLD"); vetoes.append("Crash risk CRITICAL: no new positions")
        elif crash_lvl == "HIGH":
            caps.append("ACCUMULATE"); vetoes.append("Crash risk HIGH")

        for cap in caps:
            if LEVELS.index(verdict) > LEVELS.index(cap):
                verdict = cap

        # ---------- entry plan
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        stretch = cls._stretch_z(c)
        rsi = _f(f.get("rsi"), 50)
        rh = _f(f.get("range_high"))
        bt = str(f.get("breakout_type") or "")
        if verdict in ("HOLD", "REDUCE", "AVOID"):
            entry = {"style": "NO NEW ENTRY", "trigger": None}
        elif stretch > 2.0 or rsi > 72:
            entry = {"style": "WAIT FOR PULLBACK", "trigger": round(ema21, 2)}
        elif f.get("pre_breakout") and rh > 0:
            entry = {"style": "BUY ON BREAK OF RANGE", "trigger": round(rh * 1.002, 2)}
        elif "BULLISH" in bt and f.get("volume_confirmed"):
            entry = {"style": "BUY NOW (confirmed breakout)", "trigger": round(px, 2)}
        else:
            entry = {"style": "SCALE IN (1/3 now, 2/3 on pullback)", "trigger": round(px, 2)}
        entry["stretch_z"] = round(stretch, 2)
        entry["ema21"] = round(ema21, 2)

        base_size = {"STRONG BUY": 1.0, "BUY": 0.8, "ACCUMULATE": 0.5}.get(verdict, 0.0)
        size_mult = round(base_size * (0.6 + 0.4 * agreement) * crash_adj * size_mult_extra, 3)

        # ---------- pros / cons
        contrib = sorted(
            ((k, cls.WEIGHTS[k] * avail[k][0]) for k in avail), key=lambda x: x[1], reverse=True
        )

        def _line(k: str) -> str:
            note = avail[k][1][0] if avail[k][1] else ""
            return f"{k.replace('_', ' ')} ({avail[k][0]:+.0f}): {note}".strip()

        pros = [_line(k) for k, v in contrib if avail[k][0] > 10][:3]
        cons = [_line(k) for k, v in reversed(contrib) if avail[k][0] < -10][:3]

        if abs(S) >= 40 and agreement >= 0.7 and not vetoes:
            conviction = "HIGH"
        elif abs(S) >= 20 and agreement >= 0.5:
            conviction = "MEDIUM"
        else:
            conviction = "LOW"

        return {
            "score": round((S + 100) / 2.0, 1),        # 0-100 for ranking
            "raw": round(S, 1),                          # -100..+100
            "verdict": verdict,
            "conviction": conviction,
            "agreement": round(agreement, 2),
            "bull_pillars": bull,
            "bear_pillars": bear,
            "data_coverage": round(wsum, 2),
            "pillars": {
                k: {
                    "score": None if raw[k][0] is None else round(raw[k][0], 1),
                    "weight": cls.WEIGHTS[k],
                    "notes": raw[k][1],
                }
                for k in raw
            },
            "expected": expected,
            "entry": entry,
            "size_multiplier": size_mult,
            "vetoes": vetoes,
            "pros": pros,
            "cons": cons,
            "invalidation": None if stop_loss is None else round(_f(stop_loss), 2),
        }

    # ============================================================ EXPECTED PRICE
    @classmethod
    def expected_prices(
        cls,
        df: pd.DataFrame,
        S: float,
        ml: Optional[Dict[str, Any]] = None,
        funds: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        c = df["close"].astype(float).dropna()
        if len(c) < 60:
            return {}
        px = float(c.iloc[-1])
        r = np.log(c).diff().dropna()
        s60 = float(r.iloc[-60:].std())
        s252 = float(r.iloc[-252:].std()) if len(r) >= 120 else s60
        sig = 0.6 * s60 + 0.4 * s252
        if not math.isfinite(sig) or sig <= 0:
            return {}

        mu_hist = float(r.iloc[-252:].mean())
        mu_d = HIST_DRIFT_SHRINK * mu_hist + (S / 100.0) * MAX_ANNUAL_TILT / 252.0

        # short-term shift from the ML model, discounted by its holdout reliability
        ml_shift = 0.0
        ml = ml or {}
        bal = ml.get("balanced_accuracy_holdout")
        if ml.get("trained") and bal is not None and _f(bal) >= 0.40:
            rel = _clip((_f(bal) - 0.33) / 0.20, 0.0, 1.0)
            ml_shift = (_f(ml.get("p_long"), 1 / 3) - _f(ml.get("p_short"), 1 / 3)) * 0.5 * sig * math.sqrt(10) * rel

        # analyst 12-month target, halved for a 6-month view and weighted lightly
        target_ret = None
        funds = funds or {}
        t, p0 = _f(funds.get("target_mean_price")), _f(funds.get("current_price"), px) or px
        if t > 0 and p0 > 0:
            target_ret = math.log(t / p0)

        out: Dict[str, Dict[str, float]] = {}
        for name, h in HORIZONS.items():
            mu = mu_d * h
            if h <= 21:
                mu += ml_shift
            if h >= 100 and target_ret is not None:
                mu = 0.8 * mu + 0.2 * _clip(0.5 * target_ret, -0.25, 0.30)
            sd = sig * math.sqrt(h)
            out[name] = {
                "horizon_days": h,
                "median": round(px * math.exp(mu), 2),
                "low_1sd": round(px * math.exp(mu - sd), 2),
                "high_1sd": round(px * math.exp(mu + sd), 2),
                "exp_return_pct": round((math.exp(mu) - 1) * 100, 2),
                "sigma_pct": round(sd * 100, 2),
                "prob_up": round(_phi(mu / sd), 3),
            }
        return out

    # ================================================================ PILLARS
    @staticmethod
    def _trend(f: Dict[str, Any]) -> Pillar:
        notes: List[str] = []
        t = f.get("ema_trend", "NEUTRAL")
        ema = 60.0 if t == "BULLISH" else -60.0 if t == "BEARISH" else 0.0
        ema += 40.0 if f.get("above_sma200") else -40.0
        ma = (_f(f.get("ma_confluence")) / max(1.0, _f(f.get("max_ma_confluence"), 5)) - 0.5) * 200
        ichi = (_f(f.get("ichimoku_score"), 50) - 50) * 2
        vw = {"DEEP_ABOVE": 30, "ABOVE_VWAP": 60, "AT_VWAP": 0, "BELOW_VWAP": -60, "DEEP_BELOW": -90}.get(
            str(f.get("vwap_tag")), 0
        )
        s = 0.35 * ema + 0.25 * ma + 0.25 * ichi + 0.15 * vw
        notes.append(f"EMA stack {t}, {'above' if f.get('above_sma200') else 'below'} SMA200")
        if f.get("golden_cross"):
            s += 8; notes.append("fresh golden cross")
        if f.get("death_cross"):
            s -= 8; notes.append("fresh death cross")
        return _clip(s), notes

    @staticmethod
    def _breakout(f: Dict[str, Any]) -> Pillar:
        bt = str(f.get("breakout_type") or "")
        sign = 1 if "BULLISH" in bt else -1 if "BEARISH" in bt else 0
        notes: List[str] = []
        if sign:
            s = sign * (35 + 0.4 * _f(f.get("breakout_score")))
            notes.append(f"{bt} score {_f(f.get('breakout_score')):.0f}")
            if f.get("smc_alignment"):
                s += sign * 10; notes.append("SMC structure aligned")
            if f.get("higher_timeframe_alignment"):
                s += sign * 10; notes.append("weekly trend aligned")
            if f.get("volume_confirmed"):
                s += sign * 10; notes.append("volume confirmed")
            rq = f.get("retest_quality")
            if rq == "FAILED":
                s -= sign * 25; notes.append("retest failed")
            elif rq == "GOOD":
                s += sign * 10; notes.append("clean retest")
        else:
            s = 0.0
            if f.get("pre_breakout"):
                s += 25; notes.append("pre-breakout: tight base under resistance")
            trend = str(f.get("trend", "")).upper()
            s += 15 if "BULL" in trend else -15 if "BEAR" in trend else 0
            if f.get("squeeze_detected") and s > 0:
                s += 5; notes.append("volatility squeeze")
            if not notes:
                notes.append("no breakout")
        return _clip(s), notes

    @staticmethod
    def _order_flow(f: Dict[str, Any]) -> Pillar:
        if "of_imbalance" not in f or str(f.get("of_summary", "")).startswith("Insufficient"):
            return None, ["order-flow proxy not computed"]
        imb = _f(f.get("of_imbalance_20", f.get("of_imbalance")), 0.5)
        s = _clip((imb - 0.5) * 400, -60, 60)
        notes = [f"20-bar buy-pressure {imb:.2f}"]
        dt = str(f.get("of_delta_trend", "NEUTRAL"))
        s += {"BULLISH_ACCELERATING": 30, "BULLISH_DECELERATING": 10,
              "BEARISH_ACCELERATING": -30, "BEARISH_DECELERATING": -10}.get(dt, 0)
        if dt != "NEUTRAL":
            notes.append(f"delta {dt.lower().replace('_', ' ')}")
        bias = f.get("of_bias")
        if f.get("of_absorption_flag"):
            s += 10 if bias == "BULLISH" else -10 if bias == "BEARISH" else 0
            notes.append("absorption")
        s += 5 if f.get("of_delta_positive") else -5
        return _clip(s), notes

    @staticmethod
    def _volume(df: pd.DataFrame, f: Dict[str, Any]) -> Pillar:
        c, v = df["close"].astype(float), df["volume"].astype(float)
        if v.iloc[-20:].sum() <= 0:
            return None, ["no volume data"]
        d = c.diff().iloc[-20:]
        vv = v.iloc[-20:]
        up, dn = float(vv[d > 0].sum()), float(vv[d < 0].sum())
        ud = (up + 1) / (dn + 1)
        s = _clip(math.log(ud) * 80, -60, 60)
        notes = [f"up/down volume ratio {ud:.2f} (20d)"]
        vz = _f(f.get("vol_z"))
        if vz > 1.0:
            last_up = c.iloc[-1] > c.iloc[-2]
            s += 12 if last_up else -12
            notes.append(f"volume z-score {vz:+.1f} on {'up' if last_up else 'down'} bar")
        if f.get("is_accumulating"):
            s += 12; notes.append("A/D line accumulating")
        else:
            s -= 6
        if f.get("obv_bullish_divergence"):
            s += 12; notes.append("OBV bullish divergence")
        if f.get("obv_bearish_divergence"):
            s -= 12; notes.append("OBV bearish divergence")
        if f.get("volume_increasing") and ud > 1:
            s += 6
        return _clip(s), notes

    @staticmethod
    def _stretch_z(c: pd.Series) -> float:
        if len(c) < 60:
            return 0.0
        m, sd = c.rolling(50).mean().iloc[-1], c.rolling(50).std().iloc[-1]
        return float((c.iloc[-1] - m) / sd) if sd and math.isfinite(sd) and sd > 0 else 0.0

    @classmethod
    def _technical(cls, df: pd.DataFrame, f: Dict[str, Any]) -> Pillar:
        rsi = _f(f.get("rsi"), 50)
        notes = [f"RSI {rsi:.0f}"]
        if 45 <= rsi <= 65:
            s = 20.0
        elif 65 < rsi <= 72:
            s = 5.0
        elif 72 < rsi <= 78:
            s = -15.0
        elif rsi > 78:
            s = -35.0
        elif 30 <= rsi < 45:
            s = -10.0
        else:
            s = 10.0
        mc = f.get("macd_cross")
        s += 30 if mc == "BULLISH" else -30 if mc == "BEARISH" else 0
        s += 15 if _f(f.get("macd_hist")) > 0 else -15
        if f.get("macd_bullish_divergence"):
            s += 20; notes.append("MACD bullish divergence")
        if f.get("macd_bearish_divergence"):
            s -= 20; notes.append("MACD bearish divergence")
        z = cls._stretch_z(df["close"].astype(float))
        notes.append(f"price z-score vs 50d mean {z:+.1f}")
        if z > 2.5:
            s -= 30
        elif z > 2.0:
            s -= 15
        elif z < -2.0:
            s += 10
        if _f(f.get("bb_position"), 0.5) > 1.0:
            s -= 10
        return _clip(s), notes

    @staticmethod
    def _ai(ai: Dict[str, Any], ml: Dict[str, Any]) -> Pillar:
        if not ai:
            return None, ["AI ensemble missing"]
        conf = _f(ai.get("confidence"), 50)
        a = _clip((conf - 50) * 2.5)
        notes = [f"rule-based ensemble {conf:.0f}/100"]
        bal = ml.get("balanced_accuracy_holdout")
        if ml.get("trained") and bal is not None and _f(bal) >= 0.40:
            rel = _clip((_f(bal) - 0.33) / 0.20, 0.0, 1.0)
            pl, ps = _f(ml.get("p_long"), 1 / 3), _f(ml.get("p_short"), 1 / 3)
            m = _clip((pl - ps) * 150 * rel)
            notes.append(f"ML p(long) {pl:.2f} vs p(short) {ps:.2f} (reliability {rel:.0%})")
            return _clip(0.65 * a + 0.35 * m), notes
        return a, notes

    @staticmethod
    def _fundamental(fu: Dict[str, Any]) -> Pillar:
        if "buffett_score" not in fu:
            return None, ["fundamentals unavailable"]
        s_f = (_f(fu.get("score"), 50) - 50) * 2
        b = _clip((_f(fu.get("buffett_score"), 45) - 45) / 45 * 100)
        pio = _clip((_f(fu.get("piotroski_f"), 3) - 3) / 3 * 100)
        s = 0.40 * s_f + 0.35 * b + 0.25 * pio
        notes = [f"quality score {_f(fu.get('buffett_score')):.0f}/90, Piotroski {int(_f(fu.get('piotroski_f')))}/6"]
        if _f(fu.get("revenue_growth")) > 10:
            notes.append(f"revenue growth {_f(fu.get('revenue_growth')):.0f}%")
        if fu.get("near_52w_high"):
            s += 8; notes.append("near 52-week high")
        return _clip(s), notes

    @staticmethod
    def _analyst(an: Dict[str, Any], fu: Dict[str, Any]) -> Pillar:
        if not an or _f(an.get("active_sources"), 1) == 0:
            return None, ["no analyst data"]
        s = (_f(an.get("score"), 50) - 50) * 2
        notes = [f"consensus {an.get('label', 'N/A')} ({_f(an.get('score'), 50):.0f})"]
        t, p = _f(fu.get("target_mean_price")), _f(fu.get("current_price"))
        if t > 0 and p > 0:
            up = (t / p - 1) * 100
            s = 0.6 * s + 0.4 * _clip(up * 2.5, -50, 50)
            notes.append(f"mean target {up:+.0f}% from price")
        return _clip(s), notes

    @staticmethod
    def _sentiment(se: Dict[str, Any]) -> Pillar:
        if not se or not se.get("headline_count"):
            return None, ["no headlines"]
        return _clip((_f(se.get("score"), 50) - 50) * 2), [
            f"{se.get('label', 'NEUTRAL')} across {se.get('headline_count')} headlines"
        ]

    @staticmethod
    def _macro(mr: Dict[str, Any], sf: Dict[str, Any], fu: Dict[str, Any]) -> Pillar:
        if not mr:
            return None, ["market regime unavailable"]
        ov = mr.get("_overall", "NEUTRAL")
        reg = {"STRONG_BULL": 80, "BULL": 50, "NEUTRAL": 0, "CAUTIOUS": -30,
               "BEAR": -65, "STRONG_BEAR": -90}.get(ov, 0)
        vix_r = (mr.get("VIX") or {}).get("regime", "NORMAL")
        vix = {"COMPLACENT": -5, "NORMAL": 0, "ELEVATED": -20, "HIGH_FEAR": -45, "PANIC": -70}.get(vix_r, 0)
        dxy = -15 if (mr.get("DXY") or {}).get("trending_up") else 10
        bias = str(sf.get("bias", ""))
        sec = 20 if "LEADING" in bias else -20 if "LAGGING" in bias else 0
        s = 0.50 * reg + 0.25 * vix + 0.10 * dxy + 0.15 * sec
        if s < 0:
            s *= float(np.clip(_f(fu.get("beta"), 1.0), 0.7, 1.5))   # high beta suffers more
        notes = [f"regime {ov}, VIX {vix_r}" + (", sector leading" if sec > 0 else ", sector lagging" if sec < 0 else "")]
        return _clip(s), notes

    # ================================================================ FORMAT
    @staticmethod
    def expected_line(dec: Dict[str, Any]) -> str:
        e = dec.get("expected") or {}
        if not e:
            return "n/a"
        return "  ".join(
            f"{k} {e[k]['median']:.2f} ({e[k]['exp_return_pct']:+.1f}%, P↑{e[k]['prob_up']:.0%})"
            for k in ("1W", "1M", "6M") if k in e
        )
