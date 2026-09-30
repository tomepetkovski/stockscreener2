"""
Enhanced Breakout Engine - Improved accuracy with false breakout filtering
P0 Priority Improvement
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
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            current_close = float(close.iloc[-1])

            # --- ATR & Volatility Analysis ---
            atr = cls._atr(df)
            atr_percent = atr / current_close * 100
            atr_rank = cls._percentile_rank(atr, 60)
            
            # Adaptive breakout buffer based on volatility
            adaptive_buffer = cls._calculate_adaptive_buffer(atr_rank)
            
            # --- Range Structure ---
            range_data = cls._range_structure(df, lookback=30)
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
            if breakout_type:
                # Check post-breakout sustainability
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
                if not sustainability["sustained"]:
                    breakout_score -= 15
                    breakout_data["flags"].append("FAILED_SUSTAINABILITY")
                
                if not vol_expansion["expanded"]:
                    breakout_score -= 10
                    breakout_data["flags"].append("NO_VOL_EXPANSION")
                
                if not htf_confirmation:
                    breakout_score -= 8
                    breakout_data["flags"].append("NO_HTF_CONFIRMATION")

            # --- Pattern Detection ---
            pattern_data = cls._detect_patterns(df)
            if pattern_data["pattern"]:
                breakout_score += 10

            # --- SMC Alignment ---
            smc_alignment = cls._smc_alignment(breakout_type, smc_data)
            if smc_alignment:
                breakout_score += 8

            # --- Final Score Clamping ---
            breakout_score = max(0, min(100, breakout_score))
            confidence = cls._classify_confidence(breakout_score)

            return {
                "status": "OK",
                "breakout_type": breakout_type,
                "breakout_score": breakout_score,
                "confidence": confidence,
                "adaptive_buffer_pct": round(adaptive_buffer * 100, 3),
                "atr_percentile": round(atr_rank, 3),
                "range_high": float(range_high),
                "range_low": float(range_low),
                "volume_confirmed": volume_data["volume_confirmed"],
                "relative_volume": volume_data["relative_volume"],
                "weekly_volume_confirmed": volume_data.get("weekly_confirmed", False),
                "sustainability": sustainability.get("sustained", False),
                "volatility_expansion": vol_expansion.get("expanded", False),
                "htf_confirmed": htf_confirmation,
                "pattern": pattern_data["pattern"],
                "smc_alignment": smc_alignment,
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
        
        # Adaptive volume multiplier
        vol_mult_min = EnhancedBreakoutEngine.CONFIG.min_volume_multiplier
        vol_mult_max = EnhancedBreakoutEngine.CONFIG.max_volume_multiplier
        adaptive_vol_mult = min(vol_mult_max, max(vol_mult_min, relative_volume))
        
        volume_confirmed = relative_volume >= adaptive_vol_mult
        
        # Weekly volume confirmation if available
        weekly_confirmed = False
        if df_weekly is not None and len(df_weekly) >= 20:
            weekly_vol = df_weekly["volume"]
            weekly_ma = weekly_vol.rolling(10).mean()
            weekly_relative = weekly_vol.iloc[-1] / (weekly_ma.iloc[-1] + 1e-9)
            weekly_confirmed = weekly_relative >= 1.2
        
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
            "adaptive_volume_mult": round(adaptive_vol_mult, 2),
            "weekly_confirmed": weekly_confirmed,
            "is_accumulating": bool(is_accumulating),
            "obv_bullish_divergence": bool(obv_bullish_div),
            "obv_bearish_divergence": bool(obv_bearish_div),
        }

    @staticmethod
    def _detect_adaptive_breakout(
        cls,
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
        is_squeeze = atr_percentile < cls.CONFIG.min_squeeze_percentile
        
        if bullish_breakout:
            breakout_type = "BULLISH_BREAKOUT"
            score += 30  # Base score
            
            if volume_data["volume_confirmed"]:
                score += 25
                flags.append("VOLUME_CONFIRMED")
            
            if volume_data.get("weekly_confirmed"):
                score += 8
                flags.append("WEEKLY_VOLUME_CONFIRMED")
            
            if is_squeeze:
                score += 12
                flags.append("SQUEEZE")
            
            if volume_data["is_accumulating"]:
                score += 10
                flags.append("ACCUMULATION")
            
            if volume_data["obv_bullish_div"]:
                score += 8
                flags.append("OBV_BULLISH_DIV")
        
        elif bearish_breakout:
            breakout_type = "BEARISH_BREAKOUT"
            score += 30
            
            if volume_data["volume_confirmed"]:
                score += 25
                flags.append("VOLUME_CONFIRMED")
            
            if is_squeeze:
                score += 12
                flags.append("SQUEEZE")
            
            if volume_data["obv_bearish_div"]:
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
        Check if price sustains above/below breakout level for N bars.
        This filters out fake breakouts that immediately reverse.
        """
        if breakout_type is None or len(df) < 5:
            return {"sustained": False, "reason": "INSUFFICIENT_DATA"}
        
        hold_bars = EnhancedBreakoutEngine.CONFIG.post_breakout_hold_bars
        recent_bars = min(hold_bars, len(df) - 1)
        
        if "BULLISH" in breakout_type:
            # Check if recent lows stayed above breakout level
            recent_lows = df["low"].iloc[-recent_bars:]
            sustained = (recent_lows > range_high * 0.998).all()
            reason = "PRICE_HELD_ABOVE_BREAKOUT" if sustained else "PRICE_FAILED_TO_HOLD"
        elif "BEARISH" in breakout_type:
            # Check if recent highs stayed below breakout level
            recent_highs = df["high"].iloc[-recent_bars:]
            sustained = (recent_highs < range_low * 1.002).all()
            reason = "PRICE_HELD_BELOW_BREAKOUT" if sustained else "PRICE_FAILED_TO_HOLD"
        else:
            return {"sustained": False, "reason": "NO_BREAKOUT"}
        
        return {"sustained": sustained, "reason": reason}

    @staticmethod
    def _check_volatility_expansion(df: pd.DataFrame, atr: pd.Series) -> Dict[str, Any]:
        """
        Check if volatility has expanded after potential breakout.
        Valid breakouts should be accompanied by volatility increase.
        """
        if len(atr) < 10:
            return {"expanded": False, "reason": "INSUFFICIENT_ATR_HISTORY"}
        
        min_expansion = EnhancedBreakoutEngine.CONFIG.min_volatility_expansion
        
        # Compare recent ATR to pre-breakout ATR
        recent_atr = atr.iloc[-3:].mean()
        prior_atr = atr.iloc[-10:-3].mean()
        
        if prior_atr > 0:
            expansion_ratio = recent_atr / prior_atr
            expanded = expansion_ratio >= min_expansion
            reason = f"ATR_EXPANSION_{expansion_ratio:.2f}x"
        else:
            expanded = False
            reason = "ZERO_PRIOR_ATR"
        
        return {"expanded": expanded, "reason": reason, "expansion_ratio": expansion_ratio if prior_atr > 0 else 0}

    @staticmethod
    def _multi_timeframe_confirmation(
        breakout_type: Optional[str],
        df_weekly: Optional[pd.DataFrame],
    ) -> bool:
        """Check if weekly timeframe confirms the breakout"""
        if df_weekly is None or len(df_weekly) < 20 or breakout_type is None:
            return False
        
        close = df_weekly["close"]
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        
        if "BULLISH" in breakout_type:
            return current > sma20 and current > sma50
        if "BEARISH" in breakout_type:
            return current < sma20 and current < sma50
        
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
        if score >= 80 and volume_data["volume_confirmed"] and htf and smc:
            return "INSTITUTIONAL_GRADE"
        if score >= 65:
            return "HIGH_QUALITY"
        if score >= 50:
            return "TRADABLE"
        return "WEAK"
