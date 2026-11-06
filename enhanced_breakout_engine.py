"""
Enhanced Breakout Engine - Improved accuracy with false breakout filtering
P0 Priority Improvement

v2 fixes (public API, class/method names and all previous output keys unchanged):
  * post-breakout sustainability no longer compares inside-range bars with the range high
    (it used to be False for virtually every breakout => constant -15 penalty)
  * volatility expansion uses the true range of the breakout bars vs ATR *before* them
    (EWM-smoothed ATR almost never moved 20 % in 3 bars)
  * weekly confirmation needs >= 50 weekly bars (SMA50 was NaN => always False)
  * pattern bonus only applies when a breakout exists (no more score 10 with no breakout)
  * volume "adaptive multiplier" tautology removed (behaviour identical: rel. volume >= 1.3)
  * NEW keys the terminal / AIEnsemble / DecisionEngine read but this engine never produced:
    is_accumulating, obv_bullish_divergence, obv_bearish_divergence, volume_zscore,
    volume_increasing, volume_strength, squeeze_detected, squeeze_duration, atr_rank,
    higher_timeframe_alignment, retest_quality, pre_breakout, breakout_strength,
    sustained, fresh_breakout, sustainability_detail
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancedBreakoutThresholds:
    """Adaptive thresholds for breakout detection"""
    min_breakout_buffer_pct: float = 0.001  # 0.1% minimum
    max_breakout_buffer_pct: float = 0.005  # 0.5% maximum
    min_volume_multiplier: float = 1.3       # Reduced from 1.5 for more sensitivity
    max_volume_multiplier: float = 3.0       # Cap extreme volume
    min_squeeze_percentile: float = 0.20    # More aggressive squeeze detection
    post_breakout_hold_bars: int = 3         # Bars to hold for sustainability check
    min_volatility_expansion: float = 1.2    # 20% ATR expansion required
    min_atr_percentile_for_tight_buffer: float = 0.30


class EnhancedBreakoutEngine:
    """
    Enhanced breakout engine with:
    - Adaptive breakout buffer based on volatility
    - Multi-timeframe volume confirmation
    - Post-breakout sustainability check
    - Volatility expansion confirmation
    - False breakout filtering
    """

    CONFIG = EnhancedBreakoutThresholds()
    RANGE_LOOKBACK = 30

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        smc_data: Dict,
        df_weekly: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Enhanced breakout analysis with multi-layer confirmation.
        """
        if df is None or len(df) < 60:
            return {"status": "NO_DATA"}

        try:
            close = df["close"]
            current_close = float(close.iloc[-1])

            # --- ATR & Volatility Analysis ---
            atr = cls._atr(df)
            atr_rank = cls._percentile_rank(atr, 60)

            # Adaptive breakout buffer based on volatility
            adaptive_buffer = cls._calculate_adaptive_buffer(atr_rank)

            is_squeeze = atr_rank < cls.CONFIG.min_squeeze_percentile
            squeeze_duration = cls._count_squeeze_bars(atr)

            # --- Range Structure ---
            range_data = cls._range_structure(df, lookback=cls.RANGE_LOOKBACK)
            range_high = range_data["range_high"]
            range_low = range_data["range_low"]

            # --- Volume Analysis ---
            volume_data = cls._enhanced_volume_analysis(df, df_weekly)

            # --- Breakout Detection with Adaptive Buffer ---
            breakout_data = cls._detect_adaptive_breakout(
                close=current_close,
                range_high=range_high,
                range_low=range_low,
                volume_data=volume_data,
                adaptive_buffer=adaptive_buffer,
                atr_percentile=atr_rank,
            )

            breakout_type = breakout_data["type"]
            breakout_score = breakout_data["score"]

            # --- False Breakout Filtering ---
            sustainability = {"sustained": False, "fresh": False, "reason": "NO_BREAKOUT",
                              "retest_quality": "NONE"}
            vol_expansion = {"expanded": False, "reason": "NO_BREAKOUT"}
            htf_confirmation = False
            if breakout_type:
                # Check post-breakout sustainability (fresh breakouts cannot fail yet -> no penalty)
                sustainability = cls._check_post_breakout_sustainability(
                    df, breakout_type, range_high, range_low
                )

                # Check volatility expansion
                vol_expansion = cls._check_volatility_expansion(df, atr)

                # Multi-timeframe confirmation
                htf_confirmation = cls._multi_timeframe_confirmation(
                    breakout_type, df_weekly
                )

                # Apply penalties for failed confirmations
                if not sustainability["sustained"] and not sustainability.get("fresh"):
                    breakout_score -= 15
                    breakout_data["flags"].append("FAILED_SUSTAINABILITY")

                if not vol_expansion["expanded"]:
                    breakout_score -= 10
                    breakout_data["flags"].append("NO_VOL_EXPANSION")

                if not htf_confirmation:
                    breakout_score -= 8
                    breakout_data["flags"].append("NO_HTF_CONFIRMATION")

                # Flags the terminal's thesis builder looks for
                if volume_data.get("volume_zscore", 0) > 0:
                    breakout_data["flags"].append("VOLUME_ZSCORE_ALIGNED")
                if volume_data.get("volume_increasing"):
                    breakout_data["flags"].append("VOLUME_INCREASING")
                if volume_data.get("relative_volume", 0) >= 2.0:
                    breakout_data["flags"].append("STRONG_VOLUME")

            # --- Pattern Detection (bonus only when a breakout exists) ---
            pattern_data = cls._detect_patterns(df)
            if pattern_data["pattern"] and breakout_type:
                breakout_score += 10

            # --- SMC Alignment ---
            smc_alignment = cls._smc_alignment(breakout_type, smc_data)
            if smc_alignment:
                breakout_score += 8

            # --- Final Score Clamping ---
            breakout_score = max(0, min(100, breakout_score))
            confidence = cls._classify_confidence(breakout_score)

            # --- Early signal: tight base under resistance with accumulation ---
            pre_breakout = bool(
                breakout_type is None
                and current_close >= range_high * 0.97
                and range_data["tight"]
                and volume_data.get("is_accumulating", False)
            )

            return {
                "status": "OK",
                "breakout_type": breakout_type,
                "breakout_score": breakout_score,
                "confidence": confidence,
                "adaptive_buffer_pct": round(adaptive_buffer * 100, 3),
                "atr_percentile": round(atr_rank, 3),
                "atr_rank": round(atr_rank, 3),
                "range_high": float(range_high),
                "range_low": float(range_low),
                "tight_consolidation": bool(range_data["tight"]),
                "volume_confirmed": volume_data.get("volume_confirmed", False),
                "relative_volume": volume_data.get("relative_volume", 0.0),
                "volume_zscore": volume_data.get("volume_zscore", 0.0),
                "volume_strength": volume_data.get("relative_volume", 0.0),
                "volume_increasing": volume_data.get("volume_increasing", False),
                "weekly_volume_confirmed": volume_data.get("weekly_confirmed", False),
                "is_accumulating": volume_data.get("is_accumulating", False),
                "obv_bullish_divergence": volume_data.get("obv_bullish_divergence", False),
                "obv_bearish_divergence": volume_data.get("obv_bearish_divergence", False),
                "squeeze_detected": bool(is_squeeze),
                "squeeze_duration": squeeze_duration,
                # NOTE: bool, as before. The terminal's thesis only says "sustained with
                # follow-through" when this is True; fresh breakouts report False but are not penalised.
                "sustainability": sustainability.get("sustained", False),
                "sustained": sustainability.get("sustained", False),
                "fresh_breakout": sustainability.get("fresh", False),
                "sustainability_detail": sustainability,
                "retest_quality": sustainability.get("retest_quality", "NONE"),
                "volatility_expansion": vol_expansion.get("expanded", False),
                "htf_confirmed": htf_confirmation,
                "higher_timeframe_alignment": htf_confirmation,
                "pattern": pattern_data["pattern"],
                "smc_alignment": smc_alignment,
                "pre_breakout": pre_breakout,
                "breakout_strength": cls._classify_breakout_strength(
                    breakout_score, volume_data.get("relative_volume", 0.0)
                ),
                "flags": breakout_data.get("flags", []),
                "signal_quality": cls._assess_signal_quality(
                    breakout_score, volume_data, htf_confirmation, smc_alignment
                ),
            }

        except Exception as e:
            logger.exception(f"Enhanced breakout analysis failed: {e}")
            return {"status": "ERROR", "error": str(e)}

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
        """Calculate percentile rank"""
        ranked = series.rolling(window).rank(pct=True)
        value = ranked.iloc[-1]
        return float(value) if pd.notna(value) else 1.0

    @staticmethod
    def _count_squeeze_bars(atr: pd.Series, threshold: float = 0.30) -> int:
        """Consecutive most-recent bars with ATR percentile rank below threshold."""
        rank = atr.rolling(60).rank(pct=True)
        n = 0
        for below in (rank < threshold).to_numpy()[::-1]:
            if below:
                n += 1
            else:
                break
        return int(n)

    @staticmethod
    def _calculate_adaptive_buffer(atr_percentile: float) -> float:
        """
        Calculate adaptive breakout buffer based on ATR percentile.
        Low volatility = tighter buffer, High volatility = wider buffer
        """
        min_buf = EnhancedBreakoutEngine.CONFIG.min_breakout_buffer_pct
        max_buf = EnhancedBreakoutEngine.CONFIG.max_breakout_buffer_pct

        # Linear interpolation
        adaptive_buffer = min_buf + (max_buf - min_buf) * atr_percentile
        return adaptive_buffer

    @staticmethod
    def _range_structure(df: pd.DataFrame, lookback: int = 30) -> Dict[str, Any]:
        """Calculate range structure"""
        range_high = df["high"].iloc[-lookback:-1].max()
        range_low = df["low"].iloc[-lookback:-1].min()
        current_close = df["close"].iloc[-1]
        range_size = range_high - range_low
        consolidation_ratio = range_size / current_close if current_close > 0 else 1

        return {
            "range_high": float(range_high),
            "range_low": float(range_low),
            "range_size": float(range_size),
            "consolidation_ratio": float(consolidation_ratio),
            "tight": consolidation_ratio < 0.08,
        }

    @staticmethod
    def _classify_breakout_strength(score: float, relative_volume: float) -> str:
        if score >= 80 and relative_volume >= 2.0:
            return "STRONG"
        if score >= 65 and relative_volume >= 1.5:
            return "MODERATE"
        if score >= 50:
            return "WEAK"
        return "UNDEFINED"

    @staticmethod
    def _enhanced_volume_analysis(
        df: pd.DataFrame,
        df_weekly: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Enhanced volume analysis with weekly confirmation"""
        volume = df["volume"]
        vol_ma20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std()
        current_volume = volume.iloc[-1]

        relative_volume = current_volume / (vol_ma20.iloc[-1] + 1e-9)
        volume_zscore = (current_volume - vol_ma20.iloc[-1]) / (vol_std20.iloc[-1] + 1e-9)

        # Volume multiplier: floor at the configured minimum (the previous "adaptive" version
        # compared relative_volume with itself and was equivalent to this)
        vol_mult_min = EnhancedBreakoutEngine.CONFIG.min_volume_multiplier
        vol_mult_max = EnhancedBreakoutEngine.CONFIG.max_volume_multiplier
        adaptive_vol_mult = min(vol_mult_max, max(vol_mult_min, relative_volume))

        volume_confirmed = relative_volume >= vol_mult_min

        # Weekly volume confirmation if available
        weekly_confirmed = False
        if df_weekly is not None and len(df_weekly) >= 20:
            weekly_vol = df_weekly["volume"]
            weekly_ma = weekly_vol.rolling(10).mean()
            weekly_relative = weekly_vol.iloc[-1] / (weekly_ma.iloc[-1] + 1e-9)
            weekly_confirmed = bool(weekly_relative >= 1.2)

        # Volume trend
        volume_increasing = bool(volume.rolling(5).mean().iloc[-1] > volume.rolling(10).mean().iloc[-1])

        # Accumulation/Distribution
        mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-9)
        mfv = mfm * volume
        ad_line = mfv.cumsum()
        is_accumulating = ad_line.iloc[-1] > ad_line.rolling(20).mean().iloc[-1]

        # OBV divergence
        obv = (np.sign(df["close"].diff()).fillna(0) * volume).cumsum()
        price_change = df["close"].iloc[-1] - df["close"].iloc[-20]
        obv_change = obv.iloc[-1] - obv.iloc[-20]
        obv_bullish_div = price_change < 0 and obv_change > 0
        obv_bearish_div = price_change > 0 and obv_change < 0

        return {
            "volume_confirmed": bool(volume_confirmed),
            "volume_zscore": round(float(volume_zscore), 2),
            "relative_volume": round(float(relative_volume), 2),
            "adaptive_volume_mult": round(float(adaptive_vol_mult), 2),
            "weekly_confirmed": weekly_confirmed,
            "volume_increasing": volume_increasing,
            "is_accumulating": bool(is_accumulating),
            "obv_bullish_divergence": bool(obv_bullish_div),
            "obv_bearish_divergence": bool(obv_bearish_div),
        }

    @staticmethod
    def _detect_adaptive_breakout(
        close: float,
        range_high: float,
        range_low: float,
        volume_data: Dict,
        adaptive_buffer: float,
        atr_percentile: float,
    ) -> Dict[str, Any]:
        """Detect breakout with adaptive buffer"""
        score = 0
        breakout_type = None
        flags = []

        bullish_breakout = close > range_high * (1 + adaptive_buffer)
        bearish_breakout = close < range_low * (1 - adaptive_buffer)

        # Squeeze detection bonus
        is_squeeze = atr_percentile < EnhancedBreakoutEngine.CONFIG.min_squeeze_percentile

        if bullish_breakout:
            breakout_type = "BULLISH_BREAKOUT"
            score += 30  # Base score

            if volume_data.get("volume_confirmed"):
                score += 25
                flags.append("VOLUME_CONFIRMED")

            if volume_data.get("weekly_confirmed"):
                score += 8
                flags.append("WEEKLY_VOLUME_CONFIRMED")

            if is_squeeze:
                score += 12
                flags.append("SQUEEZE")

            if volume_data.get("is_accumulating"):
                score += 10
                flags.append("ACCUMULATION")

            if volume_data.get("obv_bullish_divergence"):
                score += 8
                flags.append("OBV_BULLISH_DIV")

        elif bearish_breakout:
            breakout_type = "BEARISH_BREAKOUT"
            score += 30

            if volume_data.get("volume_confirmed"):
                score += 25
                flags.append("VOLUME_CONFIRMED")

            if is_squeeze:
                score += 12
                flags.append("SQUEEZE")

            if volume_data.get("obv_bearish_divergence"):
                score += 10
                flags.append("OBV_BEARISH_DIV")

        return {
            "type": breakout_type,
            "score": score,
            "flags": flags,
        }

    @staticmethod
    def _check_post_breakout_sustainability(
        df: pd.DataFrame,
        breakout_type: Optional[str],
        range_high: float,
        range_low: float,
    ) -> Dict[str, Any]:
        """
        Did price hold the breakout level after the first close beyond it?

        The breakout level is rebuilt from the base *before* the hold window, because
        range_high/range_low include the last bars themselves (comparing those bars'
        lows with their own range high can essentially never pass).

        Returns sustained (held >= 2 bars beyond the level), fresh (first bar beyond the level,
        cannot be judged yet) and retest_quality (GOOD / ACCEPTABLE / FAILED / NONE).
        """
        hold = EnhancedBreakoutEngine.CONFIG.post_breakout_hold_bars
        look = EnhancedBreakoutEngine.RANGE_LOOKBACK
        if breakout_type is None or len(df) < look + 2:
            return {"sustained": False, "fresh": False, "reason": "INSUFFICIENT_DATA",
                    "retest_quality": "NONE"}

        window = df.iloc[-hold:]
        base = df.iloc[-look:-hold]

        if "BULLISH" in breakout_type:
            level = float(base["high"].max())
            beyond = (window["close"] > level).to_numpy()
            if not beyond.any():
                return {"sustained": False, "fresh": False, "reason": "NEVER_CLOSED_ABOVE_LEVEL",
                        "retest_quality": "NONE", "level": level}
            after = window.iloc[int(beyond.argmax()):]
            if len(after) == 1:
                return {"sustained": False, "fresh": True, "reason": "FRESH_BREAKOUT",
                        "retest_quality": "NONE", "level": level}
            held = bool((after["low"] > level * 0.995).all() and after["close"].iloc[-1] > level)
            touched = bool((after["low"] <= level * 1.005).any())
            if not held:
                rq, reason = "FAILED", "PRICE_FAILED_TO_HOLD"
            elif touched:
                rq, reason = "GOOD", "RETESTED_AND_HELD"
            else:
                rq, reason = "ACCEPTABLE", "PRICE_HELD_ABOVE_BREAKOUT"
            return {"sustained": held, "fresh": False, "reason": reason,
                    "retest_quality": rq, "level": level}

        if "BEARISH" in breakout_type:
            level = float(base["low"].min())
            beyond = (window["close"] < level).to_numpy()
            if not beyond.any():
                return {"sustained": False, "fresh": False, "reason": "NEVER_CLOSED_BELOW_LEVEL",
                        "retest_quality": "NONE", "level": level}
            after = window.iloc[int(beyond.argmax()):]
            if len(after) == 1:
                return {"sustained": False, "fresh": True, "reason": "FRESH_BREAKOUT",
                        "retest_quality": "NONE", "level": level}
            held = bool((after["high"] < level * 1.005).all() and after["close"].iloc[-1] < level)
            touched = bool((after["high"] >= level * 0.995).any())
            if not held:
                rq, reason = "FAILED", "PRICE_FAILED_TO_HOLD"
            elif touched:
                rq, reason = "GOOD", "RETESTED_AND_HELD"
            else:
                rq, reason = "ACCEPTABLE", "PRICE_HELD_BELOW_BREAKOUT"
            return {"sustained": held, "fresh": False, "reason": reason,
                    "retest_quality": rq, "level": level}

        return {"sustained": False, "fresh": False, "reason": "NO_BREAKOUT", "retest_quality": "NONE"}

    @staticmethod
    def _check_volatility_expansion(df: pd.DataFrame, atr: pd.Series) -> Dict[str, Any]:
        """
        Valid breakouts should show a range expansion: the largest true range of the last
        3 bars versus the ATR that existed *before* those bars.
        (EWM ATR itself moves too slowly to show a 20 % jump within 3 bars.)
        """
        if len(atr) < 20:
            return {"expanded": False, "reason": "INSUFFICIENT_ATR_HISTORY", "expansion_ratio": 0}

        min_expansion = EnhancedBreakoutEngine.CONFIG.min_volatility_expansion

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)

        recent_tr = float(tr.iloc[-3:].max())
        prior_atr = float(atr.iloc[-4])

        if prior_atr > 0:
            expansion_ratio = recent_tr / prior_atr
            expanded = expansion_ratio >= min_expansion
            reason = f"TR_EXPANSION_{expansion_ratio:.2f}x"
        else:
            expansion_ratio, expanded, reason = 0.0, False, "ZERO_PRIOR_ATR"

        return {"expanded": bool(expanded), "reason": reason, "expansion_ratio": expansion_ratio}

    @staticmethod
    def _multi_timeframe_confirmation(
        breakout_type: Optional[str],
        df_weekly: Optional[pd.DataFrame],
    ) -> bool:
        """Check if weekly timeframe confirms the breakout (needs 50 weekly bars for SMA50)"""
        if df_weekly is None or len(df_weekly) < 50 or breakout_type is None:
            return False

        close = df_weekly["close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]

        if "BULLISH" in breakout_type:
            return bool(current > sma20 and current > sma50)
        if "BEARISH" in breakout_type:
            return bool(current < sma20 and current < sma50)

        return False

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

    @staticmethod
    def _detect_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect chart patterns"""
        pattern = None
        highs = df["high"].iloc[-30:]
        lows = df["low"].iloc[-30:]

        # Ascending triangle
        high_std = highs.std()
        high_mean = highs.mean()
        flat_resistance = (high_std / (high_mean + 1e-9)) < 0.03
        rising_lows = lows.iloc[-10:].mean() > lows.iloc[:10].mean() * 1.02

        if flat_resistance and rising_lows:
            pattern = "ASCENDING_TRIANGLE"

        # Cup & Handle
        if len(df) >= 60:
            cup = df.iloc[-60:]
            cup_low = cup["low"].min()
            left_high = cup["high"].iloc[:20].max()
            right_high = cup["high"].iloc[-20:].max()
            recovery = right_high / (left_high + 1e-9)

            if 0.90 <= recovery <= 1.10 and cup_low < left_high * 0.85:
                pattern = pattern or "CUP_AND_HANDLE"

        return {"pattern": pattern}

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
        if score >= 80 and volume_data.get("volume_confirmed") and htf and smc:
            return "INSTITUTIONAL_GRADE"
        if score >= 65:
            return "HIGH_QUALITY"
        if score >= 50:
            return "TRADABLE"
        return "WEAK"
