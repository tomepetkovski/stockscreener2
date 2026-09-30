"""
Enhanced Breakout Engine v2 (stocks)
Drop-in replacement:
  * same class name, same analyze(df, smc_data, df_weekly) signature
  * every original output key is still returned (new keys are additive)
  * private helpers keep their original names

Key changes vs v1 (see notes in the chat reply for details):
  * fixed crashes (staticmethod/cls bug, OBV key mismatch, unbound variables)
  * breakout is detected on the breakout BAR (last N bars), so the range,
    volume, and hold checks are measured correctly instead of self-invalidating
  * ATR-scaled buffer, pre-breakout squeeze, close-location, extension (anti-chase),
    daily trend filter, liquidity filter, volume gate, failed-breakout detection
  * trade plan (entry / stop / targets / R:R) and an `actionable` flag
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancedBreakoutThresholds:
    """Adaptive thresholds for breakout detection"""
    # --- original fields (unchanged names) ---
    min_breakout_buffer_pct: float = 0.001   # 0.1% minimum
    max_breakout_buffer_pct: float = 0.005   # 0.5% (percentile-based cap)
    min_volume_multiplier: float = 1.3
    max_volume_multiplier: float = 3.0
    min_squeeze_percentile: float = 0.20
    post_breakout_hold_bars: int = 3
    min_volatility_expansion: float = 1.2    # breakout bar TR vs prior ATR
    min_atr_percentile_for_tight_buffer: float = 0.30

    # --- new ---
    hard_max_buffer_pct: float = 0.015       # cap for ATR-scaled buffer (volatile stocks)
    atr_buffer_fraction: float = 0.20        # buffer ~ 20% of ATR%
    strong_volume_multiplier: float = 2.0
    range_lookback: int = 30
    max_breakout_age_bars: int = 3           # breakout bar may be up to N-1 bars ago
    atr_rank_window: int = 120
    min_close_location: float = 0.70         # close in top 30% of bar (bull)
    max_extension_atr: float = 2.0           # beyond this = chasing
    hard_extension_atr: float = 3.0
    min_avg_dollar_volume: float = 1_000_000  # 20d avg $ volume (liquidity floor)
    low_volume_score_cap: int = 60
    # risk / trade plan
    stop_atr_buffer: float = 0.10
    min_risk_atr: float = 0.75
    max_stop_atr: float = 1.5
    target_1_r: float = 2.0
    target_2_r: float = 3.0
    trail_atr_mult: float = 2.5
    actionable_score: int = 60


class EnhancedBreakoutEngine:
    """
    Enhanced breakout engine with:
    - ATR-adaptive breakout buffer
    - Breakout-bar volume / close-location / volatility-expansion confirmation
    - Post-breakout hold + retest check, failed-breakout detection
    - Trend (50/200 SMA) and higher-timeframe (10/30 week) alignment
    - Anti-chase extension filter and liquidity filter
    - Trade plan with stop, targets and R:R
    """

    CONFIG = EnhancedBreakoutThresholds()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        smc_data: Dict,
        df_weekly: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Enhanced breakout analysis with multi-layer confirmation."""
        if df is None or len(df) < 60:
            return {"status": "NO_DATA"}

        try:
            cfg = cls.CONFIG
            smc_data = smc_data or {}
            d = cls._prepare(df)
            if len(d) < 60:
                return {"status": "NO_DATA"}

            n = len(d)
            close = d["close"]
            current_close = float(close.iloc[-1])
            atr = cls._atr(d)
            atr_now = float(atr.iloc[-1])
            atr_pct_series = (atr / close).replace([np.inf, -np.inf], np.nan)
            weekly = cls._resolve_weekly(d, df_weekly)

            event = cls._find_breakout_event(d, atr)
            breakout_type = event.get("type")

            # Defaults (also used when there is no breakout)
            sustain: Dict[str, Any] = {"sustained": False, "reason": "NO_BREAKOUT"}
            vol_expansion: Dict[str, Any] = {"expanded": False, "reason": "NO_BREAKOUT", "expansion_ratio": 0}
            htf_confirmation = False
            htf_available = False
            trend: Dict[str, Any] = {"aligned": False, "counter_trend": False, "strong": False}
            pattern_data: Dict[str, Any] = {"pattern": None}
            smc_alignment = False
            trade_plan: Optional[Dict[str, Any]] = None
            extension_atr = 0.0
            close_location = 0.5
            flags = []

            if breakout_type:
                bi = event["bar"]
                age = event["age"]
                range_high = event["range_high"]
                range_low = event["range_low"]
                adaptive_buffer = event["buffer"]
                atr_rank = event["atr_rank"]
                atr_pre = float(atr.iloc[bi - 1])
                bull = "BULLISH" in breakout_type

                volume_data = cls._enhanced_volume_analysis(d, weekly, bi)

                bar_h, bar_l, bar_c = (float(d["high"].iloc[bi]), float(d["low"].iloc[bi]),
                                       float(d["close"].iloc[bi]))
                close_location = (bar_c - bar_l) / (bar_h - bar_l + 1e-9)

                trend = cls._trend_alignment(d, breakout_type, bi)

                base = d.iloc[max(0, bi - cfg.range_lookback):bi]
                if bull:
                    touches = int((base["high"] >= range_high * 0.995).sum())
                else:
                    touches = int((base["low"] <= range_low * 1.005).sum())

                pattern_data = cls._detect_patterns(d.iloc[:bi])

                breakout_data = cls._detect_adaptive_breakout(
                    close=bar_c,
                    range_high=range_high,
                    range_low=range_low,
                    volume_data=volume_data,
                    adaptive_buffer=adaptive_buffer,
                    atr_percentile=atr_rank,
                    close_location=close_location,
                    trend=trend,
                    touches=touches,
                )
                breakout_score = float(breakout_data["score"])
                flags = list(breakout_data["flags"])

                # ---------------- False breakout filtering ----------------
                sustain = cls._check_post_breakout_sustainability(
                    d, breakout_type, range_high, range_low, bi, atr_pre
                )
                vol_expansion = cls._check_volatility_expansion(d, atr, bi)
                htf_available = weekly is not None and len(weekly) >= 30
                htf_confirmation = cls._multi_timeframe_confirmation(breakout_type, weekly)

                if not sustain["sustained"]:
                    breakout_score -= 15
                    flags.append("FAILED_SUSTAINABILITY")
                elif age > 0:
                    breakout_score += 6
                    flags.append("HELD_AFTER_BREAKOUT")
                    if sustain.get("retested"):
                        breakout_score += 3
                        flags.append("RETEST_HELD")
                if age > 0:
                    breakout_score -= 3 * age          # stale signals decay

                if not vol_expansion["expanded"]:
                    breakout_score -= 10
                    flags.append("NO_VOL_EXPANSION")

                if htf_available and not htf_confirmation:
                    breakout_score -= 8
                    flags.append("NO_HTF_CONFIRMATION")

                # ---------------- Anti-chase (extension) ----------------
                if atr_pre > 0:
                    if bull:
                        extension_atr = (current_close - range_high) / atr_pre
                    else:
                        extension_atr = (range_low - current_close) / atr_pre
                if extension_atr > cfg.hard_extension_atr:
                    breakout_score -= 20
                    flags.append("OVEREXTENDED")
                elif extension_atr > cfg.max_extension_atr:
                    breakout_score -= 10
                    flags.append("EXTENDED")

                # ---------------- Liquidity ----------------
                dollar_vol = float((d["close"] * d["volume"]).iloc[max(0, bi - 20):bi].mean())
                if dollar_vol < cfg.min_avg_dollar_volume:
                    breakout_score -= 15
                    flags.append("LOW_LIQUIDITY")

                # ---------------- Patterns (direction aware) ----------------
                pat = pattern_data["pattern"]
                if pat:
                    bullish_pats = {"ASCENDING_TRIANGLE", "CUP_AND_HANDLE"}
                    bearish_pats = {"DESCENDING_TRIANGLE"}
                    if pat == "VOLATILITY_CONTRACTION":
                        breakout_score += 6
                    elif (bull and pat in bullish_pats) or ((not bull) and pat in bearish_pats):
                        breakout_score += 8

                # ---------------- SMC alignment ----------------
                smc_alignment = cls._smc_alignment(breakout_type, smc_data)
                if smc_alignment:
                    breakout_score += 8

                # ---------------- Volume gate ----------------
                if not volume_data["volume_confirmed"] and breakout_score > cfg.low_volume_score_cap:
                    breakout_score = cfg.low_volume_score_cap
                    flags.append("LOW_VOLUME_CAP")

                trade_plan = cls._build_trade_plan(
                    breakout_type, current_close, range_high, range_low, bar_h, bar_l, atr_pre
                )
            else:
                # No breakout on the recent bars
                rng = cls._range_structure(d, cfg.range_lookback)
                range_high, range_low = rng["range_high"], rng["range_low"]
                atr_rank = cls._percentile_rank(atr_pct_series, min(cfg.atr_rank_window, n))
                adaptive_buffer = cls._calculate_adaptive_buffer(atr_rank, float(atr_pct_series.iloc[-1]))
                volume_data = cls._enhanced_volume_analysis(d, weekly)
                breakout_score = 0.0
                if event.get("failed"):
                    flags.append(event["failed"])

            breakout_score = int(round(max(0, min(100, breakout_score))))
            confidence = cls._classify_confidence(breakout_score)

            setup = None if breakout_type else cls._detect_setup(d, atr_now, atr_rank)

            score_ok = breakout_score >= cfg.actionable_score
            actionable = bool(
                breakout_type
                and score_ok
                and volume_data["volume_confirmed"]
                and "OVEREXTENDED" not in flags
                and "LOW_LIQUIDITY" not in flags
                and "FAILED_SUSTAINABILITY" not in flags
            )

            return {
                "status": "OK",
                "breakout_type": breakout_type,
                "breakout_score": breakout_score,
                "confidence": confidence,
                "adaptive_buffer_pct": round(adaptive_buffer * 100, 3),
                "atr_percentile": round(float(atr_rank), 3),
                "range_high": float(range_high),
                "range_low": float(range_low),
                "volume_confirmed": volume_data["volume_confirmed"],
                "relative_volume": volume_data["relative_volume"],
                "weekly_volume_confirmed": volume_data.get("weekly_confirmed", False),
                "sustainability": bool(sustain.get("sustained", False)),
                "volatility_expansion": bool(vol_expansion.get("expanded", False)),
                "htf_confirmed": bool(htf_confirmation),
                "pattern": pattern_data["pattern"],
                "smc_alignment": bool(smc_alignment),
                "flags": flags,
                "signal_quality": cls._assess_signal_quality(
                    breakout_score, volume_data, htf_confirmation, smc_alignment
                ),
                # ---- additive keys ----
                "actionable": actionable,
                "breakout_age_bars": event.get("age") if breakout_type else None,
                "extension_atr": round(float(extension_atr), 2),
                "close_location": round(float(close_location), 2),
                "trend_aligned": bool(trend.get("aligned", False)),
                "htf_available": bool(htf_available),
                "trade_plan": trade_plan,
                "setup": setup,
            }

        except Exception as e:
            logger.exception(f"Enhanced breakout analysis failed: {e}")
            return {"status": "ERROR", "error": str(e)}

    # ------------------------------------------------------------------ #
    # Data helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV (keeps original index)."""
        cols = ["high", "low", "close", "volume"]
        d = df[cols].apply(pd.to_numeric, errors="coerce")
        d = d.replace([np.inf, -np.inf], np.nan).dropna()
        d = d[(d["high"] > 0) & (d["low"] > 0) & (d["close"] > 0)]
        return d

    @staticmethod
    def _resolve_weekly(d: pd.DataFrame, df_weekly: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Use supplied weekly data; else derive it from daily if the index is datetime."""
        if df_weekly is not None and len(df_weekly) >= 30:
            return df_weekly
        if isinstance(d.index, pd.DatetimeIndex):
            try:
                w = d.resample("W").agg(
                    {"high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna()
                if len(w) >= 30:
                    return w
            except Exception:
                pass
        return df_weekly

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _percentile_rank(series: pd.Series, window: int) -> float:
        """Percentile rank of the last value within the trailing window."""
        s = series.dropna()
        if len(s) < 20:
            return 1.0
        w = s.iloc[-window:]
        return float((w <= w.iloc[-1]).mean())

    @staticmethod
    def _calculate_adaptive_buffer(atr_percentile: float, atr_pct: Optional[float] = None) -> float:
        """
        Adaptive breakout buffer (fraction of price).
        Blends the original percentile interpolation with an ATR-scaled term so
        volatile stocks are not triggered by noise.
        """
        cfg = EnhancedBreakoutEngine.CONFIG
        lin = cfg.min_breakout_buffer_pct + (
            cfg.max_breakout_buffer_pct - cfg.min_breakout_buffer_pct
        ) * atr_percentile
        if atr_pct is None or not np.isfinite(atr_pct):
            return float(lin)
        vol_based = cfg.atr_buffer_fraction * atr_pct
        buf = 0.5 * lin + 0.5 * vol_based
        return float(min(cfg.hard_max_buffer_pct, max(cfg.min_breakout_buffer_pct, buf)))

    @staticmethod
    def _range_structure(df: pd.DataFrame, lookback: int = 30, bar_idx: Optional[int] = None) -> Dict[str, Any]:
        """Range of the `lookback` bars BEFORE bar_idx (default: last bar)."""
        bi = len(df) - 1 if bar_idx is None else bar_idx
        base = df.iloc[max(0, bi - lookback):bi]
        range_high = float(base["high"].max())
        range_low = float(base["low"].min())
        px = float(df["close"].iloc[bi])
        range_size = range_high - range_low
        consolidation_ratio = range_size / px if px > 0 else 1.0
        return {
            "range_high": range_high,
            "range_low": range_low,
            "range_size": float(range_size),
            "consolidation_ratio": float(consolidation_ratio),
            "tight": consolidation_ratio < 0.08,
        }

    # ------------------------------------------------------------------ #
    # Breakout detection
    # ------------------------------------------------------------------ #
    @classmethod
    def _find_breakout_event(cls, df: pd.DataFrame, atr: pd.Series) -> Dict[str, Any]:
        """
        Look for a close beyond the prior range within the last N bars.
        The range is always measured BEFORE the breakout bar, so hold/retest
        checks on the following bars are meaningful.
        A breakout whose later closes fell back into the range is a failed breakout.
        """
        cfg = cls.CONFIG
        n = len(df)
        close = df["close"]
        atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
        failed = None

        for age in range(cfg.max_breakout_age_bars - 1, -1, -1):
            bi = n - 1 - age
            if bi - cfg.range_lookback < 1:
                continue
            rng = cls._range_structure(df, cfg.range_lookback, bi)
            rh, rl = rng["range_high"], rng["range_low"]
            rank = cls._percentile_rank(atr_pct.iloc[:bi], min(cfg.atr_rank_window, bi))
            buf = cls._calculate_adaptive_buffer(rank, float(atr_pct.iloc[bi - 1]))
            c = float(close.iloc[bi])

            if c > rh * (1 + buf):
                btype = "BULLISH_BREAKOUT"
            elif c < rl * (1 - buf):
                btype = "BEARISH_BREAKOUT"
            else:
                continue

            post = close.iloc[bi + 1:]
            if btype == "BULLISH_BREAKOUT":
                valid = bool((post > rh).all())
            else:
                valid = bool((post < rl).all())
            if not valid:
                failed = "FAILED_BULLISH_BREAKOUT" if btype == "BULLISH_BREAKOUT" else "FAILED_BEARISH_BREAKOUT"
                continue

            return {
                "type": btype, "age": age, "bar": bi,
                "range_high": rh, "range_low": rl,
                "buffer": buf, "atr_rank": rank,
            }

        return {"type": None, "failed": failed}

    @classmethod
    def _detect_adaptive_breakout(
        cls,
        close: float,
        range_high: float,
        range_low: float,
        volume_data: Dict,
        adaptive_buffer: float,
        atr_percentile: float,
        close_location: float = 0.5,
        trend: Optional[Dict] = None,
        touches: int = 0,
    ) -> Dict[str, Any]:
        """Score a breakout bar (base signals). Confirmation penalties are applied in analyze()."""
        cfg = cls.CONFIG
        trend = trend or {}
        flags = []

        bullish = close > range_high * (1 + adaptive_buffer)
        bearish = close < range_low * (1 - adaptive_buffer)
        if not (bullish or bearish):
            return {"type": None, "score": 0, "flags": flags}

        breakout_type = "BULLISH_BREAKOUT" if bullish else "BEARISH_BREAKOUT"
        score = 25

        if volume_data["volume_confirmed"]:
            score += 15
            flags.append("VOLUME_CONFIRMED")
            if volume_data["relative_volume"] >= cfg.strong_volume_multiplier:
                score += 7
                flags.append("VOLUME_SURGE")

        if volume_data.get("weekly_confirmed"):
            score += 5
            flags.append("WEEKLY_VOLUME_CONFIRMED")

        if atr_percentile < cfg.min_squeeze_percentile:
            score += 10
            flags.append("SQUEEZE")

        # Close location: conviction of the breakout bar
        if bullish:
            strong_close = close_location >= cfg.min_close_location
            weak_close = close_location < 0.5
        else:
            strong_close = close_location <= 1 - cfg.min_close_location
            weak_close = close_location > 0.5
        if strong_close:
            score += 5
            flags.append("STRONG_CLOSE")
        elif weak_close:
            score -= 5
            flags.append("WEAK_CLOSE")

        # Money flow
        if bullish:
            if volume_data["is_accumulating"]:
                score += 5
                flags.append("ACCUMULATION")
            if volume_data["obv_confirmed"]:
                score += 5
                flags.append("OBV_CONFIRMED")
            if volume_data["obv_bullish_div"]:
                score += 3
                flags.append("OBV_BULLISH_DIV")
        else:
            if not volume_data["is_accumulating"]:
                score += 5
                flags.append("DISTRIBUTION")
            if volume_data["obv_confirmed"]:
                score += 5
                flags.append("OBV_CONFIRMED")
            if volume_data["obv_bearish_div"]:
                score += 3
                flags.append("OBV_BEARISH_DIV")

        # A level tested several times carries more stored energy
        if touches >= 3:
            score += 4
            flags.append("MULTI_TOUCH_LEVEL")

        # Trade with the daily trend
        if trend.get("aligned"):
            score += 8
            flags.append("TREND_ALIGNED")
            if trend.get("strong"):
                score += 2
                flags.append("STAGE_2_TREND" if bullish else "STAGE_4_TREND")
        elif trend.get("counter_trend"):
            score -= 8
            flags.append("COUNTER_TREND")

        return {"type": breakout_type, "score": score, "flags": flags}

    @staticmethod
    def _enhanced_volume_analysis(
        df: pd.DataFrame,
        df_weekly: Optional[pd.DataFrame] = None,
        bar_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Volume / money-flow analysis measured on the breakout bar (default: last bar)."""
        cfg = EnhancedBreakoutEngine.CONFIG
        n = len(df)
        bi = n - 1 if bar_idx is None else bar_idx
        volume = df["volume"]
        close = df["close"]

        prior = volume.iloc[max(0, bi - 20):bi]          # excludes the bar itself
        ma = float(prior.mean())
        sd = float(prior.std()) if len(prior) > 1 else 0.0
        cur = float(volume.iloc[bi])

        relative_volume = cur / (ma + 1e-9)
        volume_zscore = (cur - ma) / (sd + 1e-9)
        adaptive_vol_mult = min(cfg.max_volume_multiplier, relative_volume)
        volume_confirmed = relative_volume >= cfg.min_volume_multiplier

        # Weekly volume: last 2 weeks vs the 10 before (current week may be partial)
        weekly_confirmed = False
        if df_weekly is not None and len(df_weekly) >= 20:
            wv = df_weekly["volume"]
            recent = float(wv.iloc[-2:].mean())
            base = float(wv.iloc[-12:-2].mean())
            weekly_confirmed = bool(recent / (base + 1e-9) >= 1.1)

        # Accumulation / Distribution line
        rng = df["high"] - df["low"]
        mfm = ((close - df["low"]) - (df["high"] - close)) / (rng + 1e-9)
        ad_line = (mfm * volume).cumsum()
        ad_ma = ad_line.rolling(20).mean()
        is_accumulating = bool(ad_line.iloc[bi] > ad_ma.iloc[bi]) if pd.notna(ad_ma.iloc[bi]) else False

        # OBV
        obv = (np.sign(close.diff()).fillna(0) * volume).cumsum()
        lo = max(0, bi - 20)
        obv_high = bool(obv.iloc[bi] >= obv.iloc[lo:bi].max()) if bi > lo else False
        obv_low = bool(obv.iloc[bi] <= obv.iloc[lo:bi].min()) if bi > lo else False
        if bi >= 21:
            price_change = float(close.iloc[bi - 1] - close.iloc[bi - 21])
            obv_change = float(obv.iloc[bi - 1] - obv.iloc[bi - 21])
        else:
            price_change = obv_change = 0.0
        obv_bullish_div = price_change < 0 and obv_change > 0
        obv_bearish_div = price_change > 0 and obv_change < 0

        return {
            "volume_confirmed": bool(volume_confirmed),
            "volume_zscore": round(float(volume_zscore), 2),
            "relative_volume": round(float(relative_volume), 2),
            "adaptive_volume_mult": round(float(adaptive_vol_mult), 2),
            "weekly_confirmed": weekly_confirmed,
            "is_accumulating": is_accumulating,
            "obv_confirmed": obv_high or obv_low,
            "obv_bullish_div": bool(obv_bullish_div),
            "obv_bearish_div": bool(obv_bearish_div),
            # original key names kept for compatibility
            "obv_bullish_divergence": bool(obv_bullish_div),
            "obv_bearish_divergence": bool(obv_bearish_div),
        }

    # ------------------------------------------------------------------ #
    # Confirmation filters
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_post_breakout_sustainability(
        df: pd.DataFrame,
        breakout_type: Optional[str],
        range_high: float,
        range_low: float,
        bar_idx: Optional[int] = None,
        atr_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Did price hold the broken level on the bars AFTER the breakout bar?
        A fresh breakout (age 0) has nothing to test yet: neutral (no bonus).
        Wicks back to the level are tolerated (0.5 ATR) - a retest that holds is bullish.
        """
        if breakout_type is None or len(df) < 5:
            return {"sustained": False, "reason": "INSUFFICIENT_DATA"}

        cfg = EnhancedBreakoutEngine.CONFIG
        n = len(df)
        bi = n - 1 if bar_idx is None else bar_idx
        age = n - 1 - bi
        if age == 0:
            return {"sustained": True, "reason": "FRESH_BREAKOUT", "retested": False}

        if atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
            atr_value = float((df["high"] - df["low"]).iloc[-14:].mean())
        tol = 0.5 * atr_value

        post = df.iloc[bi + 1:][-cfg.post_breakout_hold_bars:] if age > 0 else df.iloc[0:0]
        if "BULLISH" in breakout_type:
            sustained = bool((post["close"] > range_high).all() and (post["low"] > range_high - tol).all())
            retested = bool(sustained and (post["low"] <= range_high + 0.25 * atr_value).any())
            reason = "PRICE_HELD_ABOVE_BREAKOUT" if sustained else "PRICE_FAILED_TO_HOLD"
        elif "BEARISH" in breakout_type:
            sustained = bool((post["close"] < range_low).all() and (post["high"] < range_low + tol).all())
            retested = bool(sustained and (post["high"] >= range_low - 0.25 * atr_value).any())
            reason = "PRICE_HELD_BELOW_BREAKOUT" if sustained else "PRICE_FAILED_TO_HOLD"
        else:
            return {"sustained": False, "reason": "NO_BREAKOUT"}

        return {"sustained": sustained, "reason": reason, "retested": retested}

    @staticmethod
    def _check_volatility_expansion(
        df: pd.DataFrame, atr: pd.Series, bar_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Valid breakouts show an expanded true range on the breakout bar
        relative to the ATR that preceded it.
        """
        n = len(df)
        bi = n - 1 if bar_idx is None else bar_idx
        if bi < 10 or len(atr) < 10:
            return {"expanded": False, "reason": "INSUFFICIENT_ATR_HISTORY", "expansion_ratio": 0}

        min_expansion = EnhancedBreakoutEngine.CONFIG.min_volatility_expansion
        prev_close = float(df["close"].iloc[bi - 1])
        h, l = float(df["high"].iloc[bi]), float(df["low"].iloc[bi])
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        prior_atr = float(atr.iloc[bi - 1])

        if prior_atr > 0:
            ratio = tr / prior_atr
            return {
                "expanded": bool(ratio >= min_expansion),
                "reason": f"ATR_EXPANSION_{ratio:.2f}x",
                "expansion_ratio": float(ratio),
            }
        return {"expanded": False, "reason": "ZERO_PRIOR_ATR", "expansion_ratio": 0}

    @staticmethod
    def _multi_timeframe_confirmation(
        breakout_type: Optional[str],
        df_weekly: Optional[pd.DataFrame],
    ) -> bool:
        """Weekly trend confirmation: price vs 10w / 30w SMA and 30w slope."""
        if df_weekly is None or len(df_weekly) < 30 or breakout_type is None:
            return False

        close = df_weekly["close"]
        sma10 = close.rolling(10).mean()
        sma30 = close.rolling(30).mean()
        cur, s10, s30 = close.iloc[-1], sma10.iloc[-1], sma30.iloc[-1]
        s30_prev = sma30.iloc[-6] if len(sma30) >= 6 else np.nan
        if pd.isna(s10) or pd.isna(s30) or pd.isna(s30_prev):
            return False

        if "BULLISH" in breakout_type:
            return bool(cur > s10 and cur > s30 and s30 > s30_prev)
        if "BEARISH" in breakout_type:
            return bool(cur < s10 and cur < s30 and s30 < s30_prev)
        return False

    @staticmethod
    def _trend_alignment(df: pd.DataFrame, breakout_type: Optional[str], bar_idx: Optional[int] = None) -> Dict[str, Any]:
        """Daily trend filter using 50 / 200 SMA at the breakout bar."""
        neutral = {"aligned": False, "counter_trend": False, "strong": False}
        if breakout_type is None:
            return neutral
        close = df["close"]
        bi = len(df) - 1 if bar_idx is None else bar_idx
        sma50 = close.rolling(50).mean()
        if bi < 10 or pd.isna(sma50.iloc[bi]) or pd.isna(sma50.iloc[bi - 10]):
            return neutral
        px, s50, s50_prev = float(close.iloc[bi]), float(sma50.iloc[bi]), float(sma50.iloc[bi - 10])
        sma200 = close.rolling(200).mean()
        s200 = sma200.iloc[bi]
        has200 = pd.notna(s200)

        if "BULLISH" in breakout_type:
            aligned = px > s50 and s50 > s50_prev
            return {
                "aligned": bool(aligned),
                "counter_trend": bool(px < s50),
                "strong": bool(aligned and has200 and s50 > s200),
            }
        aligned = px < s50 and s50 < s50_prev
        return {
            "aligned": bool(aligned),
            "counter_trend": bool(px > s50),
            "strong": bool(aligned and has200 and s50 < s200),
        }

    @staticmethod
    def _smc_alignment(breakout_type: Optional[str], smc_data: Dict) -> bool:
        """Check Smart Money Concepts alignment"""
        if breakout_type is None:
            return False

        structure = str(smc_data.get("structure", "")).upper()
        trend = str(smc_data.get("trend", "")).upper()

        bullish = "UP" in structure or "BULL" in trend
        bearish = "DOWN" in structure or "BEAR" in trend

        if "BULLISH" in breakout_type:
            return bullish
        if "BEARISH" in breakout_type:
            return bearish
        return False

    # ------------------------------------------------------------------ #
    # Patterns / setups
    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns on the PRE-breakout base."""
        found = []
        if len(df) < 30:
            return {"pattern": None, "patterns": found}

        highs = df["high"].iloc[-30:]
        lows = df["low"].iloc[-30:]
        h_cv = highs.std() / (highs.mean() + 1e-9)
        l_cv = lows.std() / (lows.mean() + 1e-9)

        # Ascending triangle: flat resistance + rising lows
        if h_cv < 0.03 and lows.iloc[-10:].mean() > lows.iloc[:10].mean() * 1.02:
            found.append("ASCENDING_TRIANGLE")

        # Descending triangle: flat support + falling highs
        if l_cv < 0.03 and highs.iloc[-10:].mean() < highs.iloc[:10].mean() * 0.98:
            found.append("DESCENDING_TRIANGLE")

        # Cup & handle: rounded base, recovery to the left rim, tight handle
        if len(df) >= 60:
            cup = df.iloc[-60:]
            cup_low = cup["low"].min()
            left_high = cup["high"].iloc[:20].max()
            right_high = cup["high"].iloc[-20:].max()
            recovery = right_high / (left_high + 1e-9)
            handle = cup.iloc[-10:]
            handle_range = (handle["high"].max() - handle["low"].min()) / (right_high + 1e-9)
            if 0.90 <= recovery <= 1.10 and cup_low < left_high * 0.85 and handle_range < 0.10:
                found.append("CUP_AND_HANDLE")

        # Volatility contraction (VCP-style): each third of the base tighter than the last
        thirds = [df.iloc[-30:-20], df.iloc[-20:-10], df.iloc[-10:]]
        ranges = [float(t["high"].max() - t["low"].min()) for t in thirds]
        if ranges[0] > ranges[1] > ranges[2] > 0:
            found.append("VOLATILITY_CONTRACTION")

        return {"pattern": found[0] if found else None, "patterns": found}

    @classmethod
    def _detect_setup(cls, df: pd.DataFrame, atr_now: float, atr_rank: float) -> Optional[Dict[str, Any]]:
        """Watchlist helper: price coiled right under resistance / above support, no breakout yet."""
        cfg = cls.CONFIG
        if atr_now <= 0:
            return None
        rng = cls._range_structure(df, cfg.range_lookback)
        px = float(df["close"].iloc[-1])
        d_up = (rng["range_high"] - px) / atr_now
        d_dn = (px - rng["range_low"]) / atr_now
        if 0 <= d_up <= 1.0 and d_up <= d_dn:
            return {"state": "BULLISH_SETUP", "distance_atr": round(d_up, 2),
                    "coiled": bool(atr_rank <= 0.35), "trigger": rng["range_high"]}
        if 0 <= d_dn <= 1.0:
            return {"state": "BEARISH_SETUP", "distance_atr": round(d_dn, 2),
                    "coiled": bool(atr_rank <= 0.35), "trigger": rng["range_low"]}
        return None

    # ------------------------------------------------------------------ #
    # Trade plan
    # ------------------------------------------------------------------ #
    @classmethod
    def _build_trade_plan(
        cls,
        breakout_type: str,
        entry: float,
        range_high: float,
        range_low: float,
        bar_high: float,
        bar_low: float,
        atr: float,
    ) -> Optional[Dict[str, Any]]:
        """Structure-based stop (below breakout bar / level), R-multiple targets, measured move."""
        cfg = cls.CONFIG
        if not np.isfinite(atr) or atr <= 0:
            return None
        size = range_high - range_low
        bull = "BULLISH" in breakout_type

        if bull:
            stop = max(bar_low, range_high - cfg.max_stop_atr * atr) - cfg.stop_atr_buffer * atr
            risk = entry - stop
            if risk < cfg.min_risk_atr * atr:
                stop = entry - cfg.min_risk_atr * atr
                risk = entry - stop
            t1 = entry + cfg.target_1_r * risk
            t2 = entry + cfg.target_2_r * risk
            measured = range_high + size
            rr_measured = (measured - entry) / risk
        else:
            stop = min(bar_high, range_low + cfg.max_stop_atr * atr) + cfg.stop_atr_buffer * atr
            risk = stop - entry
            if risk < cfg.min_risk_atr * atr:
                stop = entry + cfg.min_risk_atr * atr
                risk = stop - entry
            t1 = entry - cfg.target_1_r * risk
            t2 = entry - cfg.target_2_r * risk
            measured = range_low - size
            rr_measured = (entry - measured) / risk

        return {
            "direction": "LONG" if bull else "SHORT",
            "entry": round(entry, 4),
            "stop": round(stop, 4),
            "risk_per_share": round(risk, 4),
            "risk_pct": round(risk / entry * 100, 2),
            "target_1": round(t1, 4),
            "target_2": round(t2, 4),
            "measured_move": round(measured, 4),
            "rr_to_measured_move": round(rr_measured, 2),
            "trail_atr_distance": round(cfg.trail_atr_mult * atr, 4),
        }

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify_confidence(score: int) -> str:
        """Classify confidence score"""
        if score >= 85:
            return "A+"
        if score >= 70:
            return "A"
        if score >= 55:
            return "B"
        if score >= 40:
            return "C"
        return "LOW"

    @staticmethod
    def _assess_signal_quality(
        score: int,
        volume_data: Dict,
        htf: bool,
        smc: bool,
    ) -> str:
        """Assess overall signal quality"""
        if score >= 80 and volume_data["volume_confirmed"] and htf and smc:
            return "INSTITUTIONAL_GRADE"
        if score >= 65:
            return "HIGH_QUALITY"
        if score >= 50:
            return "TRADABLE"
        return "WEAK"
