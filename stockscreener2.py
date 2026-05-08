"""
Institutional Breakout & Investment Terminal v4.0
Enhanced for 3-12 month investment opportunity detection.
thanks to Kicko Ognenovski, Nikola Stojcevski, Altj Sulejman, Dejan Butevski, Anastas Dzurovski

Features:
  - Market Regime Detection (Bull/Bear/Range via SPY/QQQ/VIX)
  - Multi-Timeframe Analysis (Weekly + Daily confirmation)
  - Sector Rotation Scanner (11 SPDR sector ETFs)
  - Advanced Breakout Engine (Cup&Handle, Ascending Triangle, Consolidation)
  - Institutional Flow (Accumulation/Distribution, OBV Divergence)
  - Macroeconomic Overlay (DXY, Treasury yields proxy)
  - 3-12 Month Investment Thesis Generator
  - Market Breadth (% above 200DMA, advance-decline proxy)
  - Enhanced AI Ensemble with Confluence Weighting
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
import logging
import re
import time
import json

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple



# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InstitutionalEngine_v4")
from typing import Dict, Optional
import pandas as pd
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed


# ===============================================
# 1. DATA FOUNDATION
# ===============================================

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safe OHLCV normalization (institutional-grade defensive version)
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if "adj close" in df.columns and "close" not in df.columns:
        df.rename(columns={"adj close": "close"}, inplace=True)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return None

    df = df[required]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    if len(df) < 60:
        return None

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)
    
    if isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index

    return df

def _download_timeframe(
    symbol: str,
    period: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """
    Download and normalize OHLCV data for a single timeframe.
    """
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
            prepost=False,
            repair=True,
        )

        if df.empty:
            logger.warning(
                f"No data returned for {symbol} [{interval}]"
            )
            return None

        df = normalize_ohlcv(df)

        if df is None or df.empty:
            logger.warning(
                f"Normalization failed for {symbol} [{interval}]"
            )
            return None

        # Ensure sorted datetime index
        df = df.sort_index()

        # Remove duplicate candles if any
        df = df[~df.index.duplicated(keep="last")]

        return df

    except Exception as e:
        logger.exception(
            f"Failed downloading {symbol} [{interval}] : {e}"
        )
        return None


def fetch_multi_timeframe(
    symbol: str,
    *,
    daily_period: str = "1y",
    weekly_period: str = "2y",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch multiple timeframes for top-down analysis.

    Returns:
        {
            "daily": pd.DataFrame,
            "weekly": pd.DataFrame,
        }

    Features:
    - Parallel downloads
    - Validation
    - Duplicate cleanup
    - Robust logging
    - Configurable periods
    """

    configs = {
        "daily": {
            "period": daily_period,
            "interval": "1d",
        },
        "weekly": {
            "period": weekly_period,
            "interval": "1wk",
        },
    }

    results: Dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        futures = {
            executor.submit(
                _download_timeframe,
                symbol,
                cfg["period"],
                cfg["interval"],
            ): tf
            for tf, cfg in configs.items()
        }

        for future in as_completed(futures):
            timeframe = futures[future]

            try:
                df = future.result()

                if df is not None and not df.empty:
                    results[timeframe] = df
                    logger.info(
                        f"Fetched {symbol} [{timeframe}] "
                        f"{len(df)} rows"
                    )
                else:
                    logger.warning(
                        f"No usable {timeframe} data for {symbol}"
                    )

            except Exception as e:
                logger.exception(
                    f"Unhandled error for {symbol} [{timeframe}] : {e}"
                )

    return results




# =========================================================
# MARKET REGIME ENGINE (ENHANCED)
# =========================================================

@dataclass(frozen=True)
class RegimeConfig:
    period: str
    interval: str
    min_bars: int


class MarketRegimeEngine:
    """
    Institutional-style market regime engine.

    Features:
    - Parallelized downloads
    - Standardized indicator calculations
    - Volatility-aware regime scoring
    - Breadth + momentum logic
    - Safer error handling
    - Weighted composite scoring
    - Cleaner extensibility
    """

    BENCHMARKS = {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "^VIX": "Volatility Index",
        "DX-Y.NYB": "US Dollar Index",
    }

    CONFIG = {
        "SPY": RegimeConfig("1y", "1d", 200),
        "QQQ": RegimeConfig("1y", "1d", 200),
        "^VIX": RegimeConfig("6mo", "1d", 50),
        "DX-Y.NYB": RegimeConfig("1y", "1d", 200),
    }

    # -----------------------------------------------------
    # PUBLIC
    # -----------------------------------------------------

    @classmethod
    def analyze(cls) -> Dict[str, Any]:
        """
        Main regime analysis pipeline.
        """

        market_data = cls._fetch_all()
        regime_data: Dict[str, Any] = {}

        # -------------------------------------------------
        # EQUITY INDEXES
        # -------------------------------------------------

        for ticker in ["SPY", "QQQ"]:
            df = market_data.get(ticker)

            if df is None:
                continue

            regime_data[ticker] = cls._analyze_equity_index(df)

        # -------------------------------------------------
        # VIX
        # -------------------------------------------------

        vix_df = market_data.get("^VIX")

        if vix_df is not None:
            regime_data["VIX"] = cls._analyze_vix(vix_df)

        # -------------------------------------------------
        # DXY
        # -------------------------------------------------

        dxy_df = market_data.get("DX-Y.NYB")

        if dxy_df is not None:
            regime_data["DXY"] = cls._analyze_dxy(dxy_df)

        # -------------------------------------------------
        # COMPOSITE REGIME
        # -------------------------------------------------

        overall, score = cls._determine_regime(regime_data)

        bias = cls._get_investment_bias(
            regime=overall,
            score=score,
            data=regime_data,
        )

        regime_data["_overall"] = overall
        regime_data["_score"] = score
        regime_data["_bias"] = bias
        regime_data["_timestamp"] = pd.Timestamp.utcnow().isoformat()

        return regime_data

    # -----------------------------------------------------
    # FETCHING
    # -----------------------------------------------------

    @classmethod
    def _fetch_all(cls) -> Dict[str, pd.DataFrame]:
        """
        Fetch all benchmark datasets in parallel.
        """

        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:

            futures = {
                executor.submit(
                    cls._download_symbol,
                    symbol,
                    config,
                ): symbol
                for symbol, config in cls.CONFIG.items()
            }

            for future in as_completed(futures):

                symbol = futures[future]

                try:
                    df = future.result()

                    if df is not None and not df.empty:
                        results[symbol] = df

                        logger.info(
                            f"Loaded {symbol} "
                            f"({len(df)} rows)"
                        )

                except Exception as e:
                    logger.exception(
                        f"Unhandled fetch error for {symbol}: {e}"
                    )

        return results

    @staticmethod
    def _download_symbol(
        symbol: str,
        config: RegimeConfig,
    ) -> Optional[pd.DataFrame]:

        try:
            df = yf.download(
                tickers=symbol,
                period=config.period,
                interval=config.interval,
                progress=False,
                auto_adjust=True,
                repair=True,
                threads=False,
            )

            if df.empty:
                logger.warning(f"{symbol}: empty dataframe")
                return None

            df = normalize_ohlcv(df)

            if df is None or len(df) < config.min_bars:
                logger.warning(
                    f"{symbol}: insufficient data "
                    f"({0 if df is None else len(df)} rows)"
                )
                return None

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            return df

        except Exception as e:
            logger.exception(f"{symbol} download failed: {e}")
            return None

    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    @classmethod
    def _analyze_equity_index(
        cls,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:

        close = df["close"]

        current = float(close.iloc[-1])

        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])

        ema9 = float(close.ewm(span=9).mean().iloc[-1])
        ema21 = float(close.ewm(span=21).mean().iloc[-1])

        roc20 = cls._roc(close, 20)
        roc60 = cls._roc(close, 60)

        volatility = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

        trend_score = 0

        if current > sma50:
            trend_score += 1

        if current > sma200:
            trend_score += 1

        if ema9 > ema21:
            trend_score += 1

        if roc20 > 0:
            trend_score += 1

        if roc60 > 0:
            trend_score += 1

        if trend_score >= 4:
            trend = "BULL"
        elif trend_score <= 1:
            trend = "BEAR"
        else:
            trend = "NEUTRAL"

        return {
            "price": round(current, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "roc_20d": round(roc20, 2),
            "roc_60d": round(roc60, 2),
            "annualized_volatility": round(volatility * 100, 2),
            "trend_score": trend_score,
            "trend": trend,
            "above_sma50": current > sma50,
            "above_sma200": current > sma200,
            "golden_cross": sma50 > sma200,
            "ema_stack": (
                "BULL"
                if ema9 > ema21 > sma50
                else "BEAR"
                if ema9 < ema21 < sma50
                else "NEUTRAL"
            ),
        }

    @staticmethod
    def _analyze_vix(df: pd.DataFrame) -> Dict[str, Any]:

        close = df["close"]

        current = float(close.iloc[-1])

        ma20 = float(close.rolling(20).mean().iloc[-1])

        if current >= 35:
            regime = "PANIC"
        elif current >= 25:
            regime = "HIGH_FEAR"
        elif current >= 20:
            regime = "ELEVATED"
        elif current >= 15:
            regime = "NORMAL"
        else:
            regime = "COMPLACENT"

        return {
            "level": round(current, 2),
            "ma20": round(ma20, 2),
            "above_20ma": current > ma20,
            "regime": regime,
        }

    @classmethod
    def _analyze_dxy(
        cls,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:

        close = df["close"]

        current = float(close.iloc[-1])

        sma50 = float(close.rolling(50).mean().iloc[-1])

        roc20 = cls._roc(close, 20)

        bullish = current > sma50 and roc20 > 0

        return {
            "level": round(current, 2),
            "roc_20d": round(roc20, 2),
            "trending_up": bullish,
            "impact": (
                "Bearish for risk assets"
                if bullish
                else "Supportive for risk assets"
            ),
        }

    # -----------------------------------------------------
    # REGIME LOGIC
    # -----------------------------------------------------

    @classmethod
    def _determine_regime(
        cls,
        data: Dict[str, Any],
    ) -> Tuple[str, int]:

        score = 0

        # SPY
        spy = data.get("SPY", {})
        score += spy.get("trend_score", 0)

        # QQQ
        qqq = data.get("QQQ", {})
        score += qqq.get("trend_score", 0)

        # VIX adjustment
        vix = data.get("VIX", {})
        vix_regime = vix.get("regime", "NORMAL")

        if vix_regime == "PANIC":
            score -= 6
        elif vix_regime == "HIGH_FEAR":
            score -= 4
        elif vix_regime == "ELEVATED":
            score -= 2

        # Dollar adjustment
        dxy = data.get("DXY", {})

        if dxy.get("trending_up"):
            score -= 1

        # Final mapping
        if score >= 8:
            regime = "STRONG_BULL"
        elif score >= 5:
            regime = "BULL"
        elif score >= 2:
            regime = "NEUTRAL"
        elif score >= -2:
            regime = "CAUTIOUS"
        elif score >= -5:
            regime = "BEAR"
        else:
            regime = "STRONG_BEAR"

        return regime, score

    # -----------------------------------------------------
    # INVESTMENT BIAS
    # -----------------------------------------------------

    @staticmethod
    def _get_investment_bias(
        regime: str,
        score: int,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:

        templates = {
            "STRONG_BULL": {
                "direction": "AGGRESSIVE_LONG",
                "risk_level": "MODERATE",
                "max_position_size": 8,
                "cash_allocation": "5-10%",
            },
            "BULL": {
                "direction": "TACTICAL_LONG",
                "risk_level": "MODERATE",
                "max_position_size": 6,
                "cash_allocation": "15-20%",
            },
            "NEUTRAL": {
                "direction": "SELECTIVE",
                "risk_level": "LOW-MODERATE",
                "max_position_size": 5,
                "cash_allocation": "30%",
            },
            "CAUTIOUS": {
                "direction": "DEFENSIVE",
                "risk_level": "HIGH",
                "max_position_size": 4,
                "cash_allocation": "40-50%",
            },
            "BEAR": {
                "direction": "RISK_REDUCTION",
                "risk_level": "HIGH",
                "max_position_size": 3,
                "cash_allocation": "60%+",
            },
            "STRONG_BEAR": {
                "direction": "CAPITAL_PRESERVATION",
                "risk_level": "EXTREME",
                "max_position_size": 2,
                "cash_allocation": "70-90%",
            },
        }

        result = templates.get(regime, templates["NEUTRAL"]).copy()

        result["confidence"] = min(
            95,
            max(35, abs(score) * 10),
        )

        result["score"] = score

        result["advice"] = MarketRegimeEngine._generate_advice(
            regime,
            data,
        )

        return result

    # -----------------------------------------------------
    # HELPERS
    # -----------------------------------------------------

    @staticmethod
    def _roc(
        series: pd.Series,
        periods: int,
    ) -> float:

        if len(series) <= periods:
            return 0.0

        return float(
            (series.iloc[-1] / series.iloc[-periods - 1] - 1) * 100
        )

    @staticmethod
    def _generate_advice(
        regime: str,
        data: Dict[str, Any],
    ) -> str:

        advice = {
            "STRONG_BULL": (
                "Strong trend confirmation across indexes. "
                "Favor momentum breakouts and trend continuation setups."
            ),
            "BULL": (
                "Bullish conditions intact. "
                "Buy pullbacks into support and focus on leaders."
            ),
            "NEUTRAL": (
                "Mixed market internals. "
                "Reduce exposure and prioritize selectivity."
            ),
            "CAUTIOUS": (
                "Elevated volatility and deteriorating breadth. "
                "Trade smaller with tighter stops."
            ),
            "BEAR": (
                "Risk conditions worsening. "
                "Focus on defense and capital preservation."
            ),
            "STRONG_BEAR": (
                "Broad market weakness with high volatility. "
                "Avoid aggressive long exposure."
            ),
        }

        return advice.get(regime, advice["NEUTRAL"])
# ===============================================
# 3. SECTOR ROTATION ENGINE (NEW)
# ===============================================

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services"
}


class SectorRotationEngine:
    """
    Analyzes sector rotation to identify leading/lagging sectors.
    Uses relative strength vs SPY and momentum ranking.
    """

    @staticmethod
    def analyze() -> Dict[str, Any]:
        sector_data = {}

        try:
            # Fetch SPY for relative strength
            spy = yf.download("SPY", period="6mo", interval="1d", progress=False, auto_adjust=True)
            spy = normalize_ohlcv(spy)
            spy_return = (spy['close'].iloc[-1] / spy['close'].iloc[0] - 1) * 100 if spy is not None else 0
        except:
            spy_return = 0

        for symbol, name in SECTOR_ETFS.items():
            try:
                df = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)
                df = normalize_ohlcv(df)
                if df is None or len(df) < 50:
                    continue

                close = df['close']

                # Returns
                ret_1m = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) > 22 else 0
                ret_3m = (close.iloc[-1] / close.iloc[-65] - 1) * 100 if len(close) > 65 else 0
                ret_6m = (close.iloc[-1] / close.iloc[0] - 1) * 100

                # Relative Strength vs SPY
                rel_strength = round(ret_6m - spy_return, 2)

                # Trend
                sma50 = close.rolling(50).mean().iloc[-1]
                ema9 = close.ewm(span=9).mean().iloc[-1]
                ema21 = close.ewm(span=21).mean().iloc[-1]
                
                trend = "STRONG" if ema9 > ema21 > sma50 and ret_1m > 0 else \
                        "WEAK" if ema9 < ema21 < sma50 and ret_1m < 0 else "NEUTRAL"

                # Volume trend
                vol_ma = df['volume'].rolling(20).mean()
                recent_vol = df['volume'].iloc[-10:].mean()
                vol_trend = "RISING" if recent_vol > vol_ma.iloc[-1] * 1.1 else "FALLING" if recent_vol < vol_ma.iloc[-1] * 0.9 else "NORMAL"

                sector_data[symbol] = {
                    "name": name,
                    "return_1m": round(ret_1m, 2),
                    "return_3m": round(ret_3m, 2),
                    "return_6m": round(ret_6m, 2),
                    "relative_strength": rel_strength,
                    "trend": trend,
                    "volume_trend": vol_trend,
                    "momentum_score": round((ret_1m * 0.3 + ret_3m * 0.3 + ret_6m * 0.4), 2)
                }
            except Exception as e:
                logger.debug(f"Sector {symbol} failed: {e}")

        # Rank sectors by momentum
        ranked = sorted(sector_data.items(), key=lambda x: x[1].get("momentum_score", 0), reverse=True)

        leading = [s[0] for s in ranked[:3]]
        lagging = [s[0] for s in ranked[-3:]]

        return {
            "sectors": sector_data,
            "ranked": ranked,
            "leading": leading,
            "lagging": lagging,
            "leading_names": [sector_data[s]["name"] for s in leading],
            "lagging_names": [sector_data[s]["name"] for s in lagging]
        }


# ===============================================
# 4. SMART MONEY CONCEPTS ENGINE (ENHANCED)
# ===============================================

class SMCEngine:

    @staticmethod
    def get_swing_points(df: pd.DataFrame, order=5) -> Tuple[List, List]:
        highs = df['high'].values
        lows = df['low'].values
        swing_highs = []
        swing_lows = []
        
        for i in range(order, len(highs)-order):
            window_highs = highs[i-order:i+order+1]
            if highs[i] == max(window_highs):
                swing_highs.append((df.index[i], highs[i]))
            
            window_lows = lows[i-order:i+order+1]
            if lows[i] == min(window_lows):
                swing_lows.append((df.index[i], lows[i]))
        
        return swing_highs, swing_lows

    @staticmethod
    def detect_structure(df: pd.DataFrame) -> dict:
        df = normalize_ohlcv(df)
        if df is None or len(df) < 50:
            return {}

        swing_highs, swing_lows = SMCEngine.get_swing_points(df, order=5)
        
        trend = "RANGE"
        structure = "NEUTRAL"
        bos = False
        choch = False
        
        last_sh = swing_highs[-1] if swing_highs else (df.index[-1], df['high'].iloc[-1])
        last_sl = swing_lows[-1] if swing_lows else (df.index[-1], df['low'].iloc[-1])
        
        current_price = df['close'].iloc[-1]
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if last_sh[1] > swing_highs[-2][1] and last_sl[1] > swing_lows[-2][1]:
                trend = "BULLISH"
            elif last_sh[1] < swing_highs[-2][1] and last_sl[1] < swing_lows[-2][1]:
                trend = "BEARISH"
        
        if current_price > last_sh[1]:
            structure = "BOS_UP"
            bos = True
        elif current_price < last_sl[1]:
            structure = "BOS_DOWN"
            bos = True
        
        if len(swing_highs) >= 3:
            prev_trend = "RANGE"
            if swing_highs[-2][1] < swing_highs[-3][1]:
                prev_trend = "BEARISH"
                if current_price > last_sh[1] and trend != "BULLISH":
                    choch = True
                    structure = "CHoCH_UP"
            elif swing_highs[-2][1] > swing_highs[-3][1]:
                prev_trend = "BULLISH"
                if current_price < last_sl[1] and trend != "BEARISH":
                    choch = True
                    structure = "CHoCH_DOWN"

        # --- NEW: Higher Timeframe Structure ---
        htf_trend = "UNKNOWN"
        if len(swing_highs) >= 4:
            # Use older swings for macro trend
            macro_sh = swing_highs[:len(swing_highs)//2]
            macro_sl = swing_lows[:len(swing_lows)//2]
            if len(macro_sh) >= 2 and len(macro_sl) >= 2:
                if macro_sh[-1][1] > macro_sh[-2][1] and macro_sl[-1][1] > macro_sl[-2][1]:
                    htf_trend = "BULLISH"
                elif macro_sh[-1][1] < macro_sh[-2][1] and macro_sl[-1][1] < macro_sl[-2][1]:
                    htf_trend = "BEARISH"
                else:
                    htf_trend = "RANGE"

        return {
            "trend": trend,
            "structure": structure,
            "bos": bos,
            "choch": choch,
            "last_swing_high": float(last_sh[1]),
            "last_swing_low": float(last_sl[1]),
            "htf_trend": htf_trend,
            "swing_count": (len(swing_highs), len(swing_lows))
        }


# =========================================================
# ADVANCED BREAKOUT ENGINE (INSTITUTIONAL ENHANCED)
# =========================================================




@dataclass(frozen=True)
class BreakoutThresholds:
    squeeze_percentile: float = 0.25
    breakout_volume_mult: float = 1.5
    tight_range_threshold: float = 0.08
    breakout_buffer: float = 0.002
    min_breakout_score: int = 50


class BreakoutEngine:
    """
    Institutional breakout engine.

    Features:
    - Volatility contraction detection
    - Multi-factor breakout scoring
    - Volume expansion analysis
    - OBV / accumulation logic
    - Pattern recognition
    - Higher timeframe alignment
    - Smart Money structure confluence
    - False breakout filtering
    - Retest quality analysis
    """

    CONFIG = BreakoutThresholds()

    # -----------------------------------------------------
    # PUBLIC
    # -----------------------------------------------------

    @classmethod
    def analyze(
        cls,
        df: pd.DataFrame,
        smc_data: Dict,
        df_weekly: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:

        if df is None or len(df) < 80:
            return {
                "status": "NO_DATA",
            }

        try:

            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            current_close = float(close.iloc[-1])

            # =================================================
            # CORE METRICS
            # =================================================

            atr = cls._atr(df)
            atr_rank = cls._percentile_rank(atr, 60)

            is_squeeze = atr_rank < cls.CONFIG.squeeze_percentile

            squeeze_duration = cls._count_consecutive(
                atr_rank_series=atr.rolling(60).rank(pct=True),
                threshold=0.30,
            )

            # =================================================
            # RANGE STRUCTURE
            # =================================================

            range_data = cls._range_structure(df)

            range_high = range_data["range_high"]
            range_low = range_data["range_low"]

            # =================================================
            # VOLUME ANALYSIS
            # =================================================

            volume_data = cls._volume_analysis(df)

            # =================================================
            # BREAKOUT DETECTION
            # =================================================

            breakout_data = cls._detect_breakout(
                close=current_close,
                range_high=range_high,
                range_low=range_low,
                volume_data=volume_data,
                squeeze=is_squeeze,
                accumulation=volume_data["is_accumulating"],
            )

            breakout_type = breakout_data["type"]
            breakout_score = breakout_data["score"]

            # =================================================
            # PATTERN DETECTION
            # =================================================

            pattern_data = cls._detect_patterns(df)

            # =================================================
            # HIGHER TIMEFRAME ALIGNMENT
            # =================================================

            htf_alignment = cls._higher_timeframe_alignment(
                breakout_type,
                df_weekly,
            )

            # =================================================
            # SMC ALIGNMENT
            # =================================================

            smc_alignment = cls._smc_alignment(
                breakout_type,
                smc_data,
            )

            # =================================================
            # RETEST QUALITY
            # =================================================

            retest_quality = cls._retest_quality(
                df,
                breakout_type,
                range_high,
                range_low,
            )

            # =================================================
            # PRE-BREAKOUT
            # =================================================

            pre_breakout = cls._pre_breakout_signal(
                current_close=current_close,
                range_high=range_high,
                consolidation_ratio=range_data["consolidation_ratio"],
                accumulation=volume_data["is_accumulating"],
                breakout_type=breakout_type,
            )

            # =================================================
            # FINAL COMPOSITE SCORING
            # =================================================

            final_score = breakout_score

            if pattern_data["pattern"]:
                final_score += 10

            if htf_alignment:
                final_score += 10

            if smc_alignment:
                final_score += 10

            if retest_quality == "GOOD":
                final_score += 5

            final_score = min(100, final_score)

            confidence = cls._classify_confidence(final_score)

            # =================================================
            # FINAL OUTPUT
            # =================================================

            return {
                "status": "OK",

                # Core breakout
                "breakout_type": breakout_type,
                "breakout_score": final_score,
                "confidence": confidence,

                # Volatility
                "squeeze_detected": is_squeeze,
                "squeeze_duration": squeeze_duration,
                "atr_rank": round(float(atr_rank), 3),

                # Range
                "range_high": float(range_high),
                "range_low": float(range_low),
                "range_width_pct": round(
                    range_data["consolidation_ratio"] * 100,
                    2,
                ),
                "tight_consolidation": range_data["tight"],

                # Volume
                "volume_confirmed": volume_data["volume_confirmed"],
                "volume_zscore": volume_data["volume_zscore"],
                "relative_volume": volume_data["relative_volume"],
                "is_accumulating": volume_data["is_accumulating"],

                # OBV
                "obv_bullish_divergence":
                    volume_data["obv_bullish_div"],
                "obv_bearish_divergence":
                    volume_data["obv_bearish_div"],

                # Pattern
                "pattern": pattern_data["pattern"],

                # Alignment
                "smc_alignment": smc_alignment,
                "higher_timeframe_alignment": htf_alignment,

                # Retest
                "retest_quality": retest_quality,

                # Early signal
                "pre_breakout": pre_breakout,

                # Metadata
                "volatility_expansion":
                    volume_data["volatility_expansion"],

                "signal_quality": cls._signal_quality(
                    final_score,
                    volume_data,
                    htf_alignment,
                    smc_alignment,
                ),
            }

        except Exception as e:

            logger.exception(f"Breakout analysis failed: {e}")

            return {
                "status": "ERROR",
                "error": str(e),
            }

    # =====================================================
    # ATR
    # =====================================================

    @staticmethod
    def _atr(
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.Series:

        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    # =====================================================
    # PERCENTILE RANK
    # =====================================================

    @staticmethod
    def _percentile_rank(
        series: pd.Series,
        window: int,
    ) -> float:

        ranked = series.rolling(window).rank(pct=True)

        value = ranked.iloc[-1]

        return float(value) if pd.notna(value) else 1.0

    # =====================================================
    # CONSECUTIVE COUNT
    # =====================================================

    @staticmethod
    def _count_consecutive(
        atr_rank_series: pd.Series,
        threshold: float,
    ) -> int:

        count = 0

        for val in reversed(atr_rank_series.dropna().tolist()):

            if val < threshold:
                count += 1
            else:
                break

        return count

    # =====================================================
    # RANGE STRUCTURE
    # =====================================================

    @classmethod
    def _range_structure(
        cls,
        df: pd.DataFrame,
        lookback: int = 30,
    ) -> Dict[str, Any]:

        range_high = df["high"].iloc[-lookback:-1].max()
        range_low = df["low"].iloc[-lookback:-1].min()

        current_close = df["close"].iloc[-1]

        range_size = range_high - range_low

        consolidation_ratio = (
            range_size / current_close
            if current_close > 0 else 1
        )

        return {
            "range_high": float(range_high),
            "range_low": float(range_low),
            "range_size": float(range_size),
            "consolidation_ratio": float(consolidation_ratio),
            "tight": (
                consolidation_ratio
                < cls.CONFIG.tight_range_threshold
            ),
        }

    # =====================================================
    # VOLUME ANALYSIS
    # =====================================================

    @classmethod
    def _volume_analysis(
        cls,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:

        volume = df["volume"]
        close = df["close"]

        vol_ma20 = volume.rolling(20).mean()
        vol_std20 = volume.rolling(20).std()

        current_volume = volume.iloc[-1]

        relative_volume = (
            current_volume / (vol_ma20.iloc[-1] + 1e-9)
        )

        volume_zscore = (
            (current_volume - vol_ma20.iloc[-1])
            / (vol_std20.iloc[-1] + 1e-9)
        )

        volume_confirmed = (
            relative_volume
            >= cls.CONFIG.breakout_volume_mult
        )

        # AD Line
        mfm = (
            (
                (close - df["low"])
                - (df["high"] - close)
            )
            / (df["high"] - df["low"] + 1e-9)
        )

        mfv = mfm * volume

        ad_line = mfv.cumsum()

        is_accumulating = (
            ad_line.iloc[-1]
            > ad_line.rolling(20).mean().iloc[-1]
        )

        # OBV
        obv = (
            np.sign(close.diff()).fillna(0)
            * volume
        ).cumsum()

        price_change = close.iloc[-1] - close.iloc[-20]
        obv_change = obv.iloc[-1] - obv.iloc[-20]

        obv_bullish_div = (
            price_change < 0
            and obv_change > 0
        )

        obv_bearish_div = (
            price_change > 0
            and obv_change < 0
        )

        return {
            "volume_confirmed": bool(volume_confirmed),
            "volume_zscore": round(float(volume_zscore), 2),
            "relative_volume": round(float(relative_volume), 2),
            "is_accumulating": bool(is_accumulating),
            "obv_bullish_div": bool(obv_bullish_div),
            "obv_bearish_div": bool(obv_bearish_div),
            "volatility_expansion": (
                volume_zscore > 1.5
            ),
        }

    # =====================================================
    # BREAKOUT DETECTION
    # =====================================================

    @classmethod
    def _detect_breakout(
        cls,
        close: float,
        range_high: float,
        range_low: float,
        volume_data: Dict,
        squeeze: bool,
        accumulation: bool,
    ) -> Dict[str, Any]:

        score = 0
        breakout_type = None

        buffer = cls.CONFIG.breakout_buffer

        bullish_breakout = (
            close > range_high * (1 + buffer)
        )

        bearish_breakout = (
            close < range_low * (1 - buffer)
        )

        if bullish_breakout:

            breakout_type = "BULLISH_BREAKOUT"
            score += 35

            if volume_data["volume_confirmed"]:
                score += 25

            if squeeze:
                score += 15

            if accumulation:
                score += 10

            if volume_data["obv_bullish_div"]:
                score += 10

        elif bearish_breakout:

            breakout_type = "BEARISH_BREAKOUT"
            score += 35

            if volume_data["volume_confirmed"]:
                score += 25

            if squeeze:
                score += 15

            if volume_data["obv_bearish_div"]:
                score += 10

        return {
            "type": breakout_type,
            "score": score,
        }

    # =====================================================
    # PATTERN DETECTION
    # =====================================================

    @staticmethod
    def _detect_patterns(
        df: pd.DataFrame,
    ) -> Dict[str, Any]:

        pattern = None

        highs = df["high"].iloc[-30:]
        lows = df["low"].iloc[-30:]

        # Ascending triangle
        high_std = highs.std()
        high_mean = highs.mean()

        flat_resistance = (
            high_std / (high_mean + 1e-9)
        ) < 0.03

        rising_lows = (
            lows.iloc[-10:].mean()
            > lows.iloc[:10].mean() * 1.02
        )

        if flat_resistance and rising_lows:
            pattern = "ASCENDING_TRIANGLE"

        # Cup & Handle
        if len(df) >= 60:

            cup = df.iloc[-60:]

            cup_low = cup["low"].min()

            left_high = cup["high"].iloc[:20].max()
            right_high = cup["high"].iloc[-20:].max()

            recovery = (
                right_high / (left_high + 1e-9)
            )

            if (
                0.90 <= recovery <= 1.10
                and cup_low < left_high * 0.85
            ):
                pattern = pattern or "CUP_AND_HANDLE"

        return {
            "pattern": pattern,
        }

    # =====================================================
    # HTF ALIGNMENT
    # =====================================================

    @staticmethod
    def _higher_timeframe_alignment(
        breakout_type: Optional[str],
        weekly: Optional[pd.DataFrame],
    ) -> bool:

        if (
            weekly is None
            or len(weekly) < 50
            or breakout_type is None
        ):
            return False

        close = weekly["close"]

        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]

        current = close.iloc[-1]

        if "BULLISH" in breakout_type:
            return current > sma20 and current > sma50

        if "BEARISH" in breakout_type:
            return current < sma20 and current < sma50

        return False

    # =====================================================
    # SMC ALIGNMENT
    # =====================================================

    @staticmethod
    def _smc_alignment(
        breakout_type: Optional[str],
        smc_data: Dict,
    ) -> bool:

        if breakout_type is None:
            return False

        structure = str(
            smc_data.get("structure", "")
        ).upper()

        trend = str(
            smc_data.get("trend", "")
        ).upper()

        bullish = (
            "UP" in structure
            or "BULL" in trend
        )

        bearish = (
            "DOWN" in structure
            or "BEAR" in trend
        )

        if "BULLISH" in breakout_type:
            return bullish

        if "BEARISH" in breakout_type:
            return bearish

        return False

    # =====================================================
    # RETEST QUALITY
    # =====================================================

    @staticmethod
    def _retest_quality(
        df: pd.DataFrame,
        breakout_type: Optional[str],
        range_high: float,
        range_low: float,
    ) -> str:

        if breakout_type is None or len(df) < 5:
            return "NONE"

        recent_lows = df["low"].iloc[-3:]
        recent_highs = df["high"].iloc[-3:]

        if "BULLISH" in breakout_type:

            if recent_lows.min() > range_high:
                return "GOOD"

            if recent_lows.min() >= range_high * 0.995:
                return "ACCEPTABLE"

            return "FAILED"

        if "BEARISH" in breakout_type:

            if recent_highs.max() < range_low:
                return "GOOD"

            if recent_highs.max() <= range_low * 1.005:
                return "ACCEPTABLE"

            return "FAILED"

        return "NONE"

    # =====================================================
    # PRE-BREAKOUT
    # =====================================================

    @staticmethod
    def _pre_breakout_signal(
        current_close: float,
        range_high: float,
        consolidation_ratio: float,
        accumulation: bool,
        breakout_type: Optional[str],
    ) -> bool:

        if breakout_type is not None:
            return False

        near_resistance = (
            current_close >= range_high * 0.97
        )

        tight_range = consolidation_ratio < 0.08

        return (
            near_resistance
            and tight_range
            and accumulation
        )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    @staticmethod
    def _classify_confidence(score: int) -> str:

        if score >= 85:
            return "A+"

        if score >= 70:
            return "A"

        if score >= 55:
            return "B"

        if score >= 40:
            return "C"

        return "LOW"

    # =====================================================
    # SIGNAL QUALITY
    # =====================================================

    @staticmethod
    def _signal_quality(
        score: int,
        volume_data: Dict,
        htf: bool,
        smc: bool,
    ) -> str:

        if (
            score >= 80
            and volume_data["volume_confirmed"]
            and htf
            and smc
        ):
            return "INSTITUTIONAL_GRADE"

        if score >= 65:
            return "HIGH_QUALITY"

        if score >= 50:
            return "TRADABLE"

        return "WEAK"


# ===============================================
# 6. TECHNICAL ENGINE (ENHANCED)
# ===============================================

class TechnicalEngine:
    @staticmethod
    def get_indicators(df: pd.DataFrame) -> dict:
        df = normalize_ohlcv(df)
        if df is None or len(df) < 50:
            return {}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # --- EMAs ---
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        sma50 = close.rolling(50).mean() if len(close) >= 50 else None
        sma100 = close.rolling(100).mean() if len(close) >= 100 else None
        sma200 = close.rolling(200).mean() if len(close) >= 200 else None

        ema_trend = "BULLISH" if ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] else \
                    "BEARISH" if ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1] else "NEUTRAL"

        # MA Confluence Score (how many MAs are aligned)
        ma_confluence = 0
        if close.iloc[-1] > ema9.iloc[-1]: ma_confluence += 1
        if close.iloc[-1] > ema21.iloc[-1]: ma_confluence += 1
        if close.iloc[-1] > ema50.iloc[-1]: ma_confluence += 1
        if sma100 is not None and close.iloc[-1] > sma100.iloc[-1]: ma_confluence += 1
        if sma200 is not None and close.iloc[-1] > sma200.iloc[-1]: ma_confluence += 1

        # Golden/Death Cross
        golden_cross = False
        death_cross = False
        if sma50 is not None and sma200 is not None:

            if (
                sma50.iloc[-1] > sma200.iloc[-1]
                and sma50.iloc[-2] <= sma200.iloc[-2]
            ):
                golden_cross = True

            elif (
                sma50.iloc[-1] < sma200.iloc[-1]
                and sma50.iloc[-2] >= sma200.iloc[-2]
            ):
                death_cross = True

        # --- RSI ---
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        rsi_status = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"

        # --- MACD ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        macd_cross = "BULLISH" if macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0 else \
                     "BEARISH" if macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0 else "NONE"

        # MACD divergence
        # Price higher high but MACD lower high = bearish divergence
        recent_prices = close.iloc[-30:]
        recent_macd = macd_hist.iloc[-30:]
        if len(recent_prices) >= 20:
            ph1, ph2 = recent_prices.iloc[0], recent_prices.iloc[-1]
            mh1, mh2 = recent_macd.iloc[0], recent_macd.iloc[-1]
            macd_bearish_div = ph2 > ph1 and mh2 < mh1
            macd_bullish_div = ph2 < ph1 and mh2 > mh1
        else:
            macd_bearish_div = False
            macd_bullish_div = False

        # --- ATR ---
        # === ATR Series ===
        # --- ATR ---
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr_series = tr.ewm(span=14, adjust=False).mean()

        atr = atr_series.iloc[-1]

        atr_percent = (atr / close.iloc[-1]) * 100
        

        
        # --- Bollinger Bands ---
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_sma.iloc[-1] * 100
        bb_position = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-9)

        # --- Volume Z-Score ---
        vol_z = ((volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-9)).iloc[-1]

        # --- Rate of Change (momentum) ---
        roc_20 = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else 0
        roc_50 = (close.iloc[-1] / close.iloc[-51] - 1) * 100 if len(close) > 51 else 0

        return {
            "ema_trend": ema_trend,
            "ma_confluence": ma_confluence,
            "max_ma_confluence": 5,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "rsi": float(rsi) if not np.isnan(rsi) else 50,
            "rsi_status": rsi_status,
            "macd_cross": macd_cross,
            "macd_hist": round(float(macd_hist.iloc[-1]), 4),
            "macd_bullish_divergence": macd_bullish_div,
            "macd_bearish_divergence": macd_bearish_div,
            "atr": float(atr),
            "atr_percent": round(float(atr_percent), 2),
            "bb_width": round(float(bb_width), 2),
            "bb_position": round(float(bb_position), 4),
            "vol_z": float(vol_z),
            "roc_20d": round(float(roc_20), 2),
            "roc_50d": round(float(roc_50), 2),
            "close": float(close.iloc[-1]),
            "above_sma200": sma200 is not None and close.iloc[-1] > sma200.iloc[-1],
            "above_sma100": sma100 is not None and close.iloc[-1] > sma100.iloc[-1]
        }


# ===============================================
# 7. FUNDAMENTAL ENGINE (ENHANCED)
# ===============================================

class FundamentalEngine:
    @staticmethod
    def get_fundamentals(symbol: str) -> Dict:
        try:
            t = yf.Ticker(symbol)
            info = t.info
            
            pe = info.get("forwardPE", info.get("trailingPE", 0)) or 0
            pb = info.get("priceToBook", 0) or 0
            ps = info.get("priceToSalesTrailing12Months", 0) or 0
            growth = info.get("earningsQuarterlyGrowth", 0) or 0
            revenue_growth = info.get("revenueGrowth", 0) or 0
            profit_margin = info.get("profitMargins", 0) or 0
            beta = info.get("beta", 1.0) or 1.0
            market_cap = info.get("marketCap", 0) or 0
            debt_to_equity = info.get("debtToEquity", 0) or 0
            
            # 52-week range
            week52_high = info.get("fiftyTwoWeekHigh", 0) or 0
            week52_low = info.get("fiftyTwoWeekLow", 0) or 0
            current_price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0) or 0
            
            # Distance from 52-week high (good for breakout context)
            dist_from_high = ((current_price - week52_high) / week52_high * 100) if week52_high > 0 else 0
            near_52w_high = dist_from_high > -5  # Within 5% of 52-week high

            # Composite fundamental score (0-100)
            score = 50
            
            # Valuation
            if 0 < pe < 20: score += 12
            elif 20 <= pe < 35: score += 5
            elif pe > 50: score -= 10
            
            # Growth
            if growth > 0.25: score += 15
            elif growth > 0.10: score += 8
            elif growth < 0: score -= 10
            
            # Revenue growth
            if revenue_growth > 0.20: score += 10
            elif revenue_growth > 0.05: score += 5
            elif revenue_growth < 0: score -= 8
            
            # Profitability
            if profit_margin > 0.20: score += 8
            elif profit_margin > 0.10: score += 4
            
            # Balance sheet
            if 0 < debt_to_equity < 50: score += 5
            elif debt_to_equity > 200: score -= 10
            
            # 52-week proximity (near highs = bullish momentum context)
            if near_52w_high: score += 5

            # Analyst consensus
            rec = info.get("recommendationKey", "")
            if rec in ["strong_buy", "buy"]: score += 5
            elif rec in ["strong_sell", "sell"]: score -= 8

            return {
                "score": min(100, max(0, score)),
                "pe": round(pe, 2),
                "pb": round(pb, 2),
                "ps": round(ps, 2),
                "growth": round(growth * 100, 2),
                "revenue_growth": round(revenue_growth * 100, 2),
                "profit_margin": round(profit_margin * 100, 2),
                "beta": beta,
                "market_cap": market_cap,
                "debt_to_equity": debt_to_equity,
                "near_52w_high": near_52w_high,
                "dist_from_52w_high": round(dist_from_high, 2),
                "analyst_rec": rec
            }
        except Exception as e:
            logger.debug(f"Fundamentals failed for {symbol}: {e}")
            return {"score": 50, "beta": 1.0}


# ===============================================
# 8. SENTIMENT ENGINE
# ===============================================

class SentimentEngine:
    BULLISH_WORDS = {"beat": 2.0, "surge": 2.0, "buy": 1.5, "upgrade": 2.0, "growth": 1.5,
                     "profit": 1.5, "rally": 2.0, "bullish": 2.5, "outperform": 2.0,
                     "record": 1.5, "innovation": 1.0, "expansion": 1.5, "boom": 2.0,
                     "soar": 2.0, "breakthrough": 2.0, "dividend": 1.0}
    BEARISH_WORDS = {"miss": -2.0, "drop": -2.0, "sell": -1.5, "downgrade": -2.0,
                     "loss": -2.0, "weak": -1.5, "crash": -3.0, "bearish": -2.5,
                     "recession": -2.0, "cut": -1.5, "layoff": -2.0, "decline": -1.5,
                     "slump": -2.0, "warning": -1.0, "risk": -1.0, "debt": -1.5}

    @staticmethod
    def analyze(symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                return {"score": 50, "label": "NEUTRAL", "headline_count": 0}

            score = 50
            headlines = []
            for item in news[:8]:
                title = item.get("title", "").lower()
                headlines.append(title)
                for w in title.split():
                    w_clean = w.strip(".,!?;:\"'()[]{}")
                    if w_clean in SentimentEngine.BULLISH_WORDS:
                        score += SentimentEngine.BULLISH_WORDS[w_clean]
                    if w_clean in SentimentEngine.BEARISH_WORDS:
                        score += SentimentEngine.BEARISH_WORDS[w_clean]
            
            score = float(np.clip(score, 0, 100))
            label = "BULLISH" if score > 60 else "BEARISH" if score < 40 else "NEUTRAL"
            return {"score": score, "label": label, "headline_count": len(headlines)}
        except:
            return {"score": 50, "label": "NEUTRAL", "headline_count": 0}


# ===============================================
# 9. AI ENSEMBLE (ENHANCED WITH CONFLUENCE)
# ===============================================

class AIEnsemble:
    """
    Enhanced AI Ensemble with multi-factor confluence scoring.
    Designed for 3-12 month investment decisions.
    """

    @staticmethod
    def predict(features: Dict, market_regime: Dict = None) -> Dict:
        score = 50
        signals = []  # Track individual signal sources
        weights = {}  # Track weights for transparency

        # --- 1. TREND (Weight: 15) ---
        trend = features.get("ema_trend", "NEUTRAL")
        trend_score = 0
        if trend == "BULLISH":
            trend_score = 15
            signals.append("EMA stack BULLISH")
        elif trend == "BEARISH":
            trend_score = -15
            signals.append("EMA stack BEARISH")
        weights["trend"] = trend_score
        score += trend_score

        # --- 2. BREAKOUT (Weight: 30) ---
        breakout = features.get("breakout_type")
        aligned = features.get("structure_aligned", False)
        htf_aligned = features.get("htf_aligned", False)
        breakout_strength = features.get("breakout_strength", 0)
        
        breakout_score = 0
        if breakout:
            if "BULLISH" in breakout:
                breakout_score = 15
                if aligned: breakout_score += 8
                if htf_aligned: breakout_score += 7
                if "SQUEEZE" in breakout: breakout_score += 5
                if "LOW_VOL" not in breakout: breakout_score += 3
                signals.append(f"Bullish breakout (strength: {breakout_strength})")
            elif "BEARISH" in breakout:
                breakout_score = -15
                if aligned: breakout_score -= 8
                if htf_aligned: breakout_score -= 7
                if "SQUEEZE" in breakout: breakout_score -= 5
                signals.append(f"Bearish breakout (strength: {breakout_strength})")
        
        # Pre-breakout bonus (consolidation near resistance + accumulation)
        if features.get("pre_breakout"):
            breakout_score += 8
            signals.append("Pre-breakout setup (tight consolidation + accumulation)")
        
        weights["breakout"] = breakout_score
        score += breakout_score

        # --- 3. SQUEEZE (Weight: 8) ---
        if features.get("squeeze_detected"):
            squeeze_bonus = 8
            if features.get("squeeze_days", 0) > 10:
                squeeze_bonus += 4  # Extended squeeze = more energy
            weights["squeeze"] = squeeze_bonus
            score += squeeze_bonus
            signals.append(f"Volatility squeeze ({features.get('squeeze_days', 0)} days)")

        # --- 4. MACD (Weight: 10) ---
        macd_cross = features.get("macd_cross", "NONE")
        macd_score = 0
        if macd_cross == "BULLISH":
            macd_score = 10
            signals.append("MACD bullish crossover")
        elif macd_cross == "BEARISH":
            macd_score = -10
            signals.append("MACD bearish crossover")
        
        if features.get("macd_bullish_divergence"):
            macd_score += 5
            signals.append("MACD bullish divergence")
        elif features.get("macd_bearish_divergence"):
            macd_score -= 5
            signals.append("MACD bearish divergence")
        
        weights["macd"] = macd_score
        score += macd_score

        # --- 5. RSI Context (Weight: 7) ---
        rsi = features.get("rsi", 50)
        rsi_score = 0
        if 40 <= rsi <= 65:
            rsi_score = 5  # Healthy range for uptrend continuation
        elif rsi > 75:
            rsi_score = -7  # Overbought risk for new entries
        elif rsi < 30:
            rsi_score = 7  # Oversold = potential opportunity
        weights["rsi"] = rsi_score
        score += rsi_score

        # --- 6. MA Confluence (Weight: 8) ---
        ma_con = features.get("ma_confluence", 0)
        max_con = features.get("max_ma_confluence", 5)
        ma_score = ((ma_con / max_con) - 0.4) * 16  # Scale: 0 confluence = -6.4, all = +9.6
        weights["ma_confluence"] = round(ma_score, 1)
        score += ma_score

        # --- 7. Volume / Institutional Flow (Weight: 7) ---
        flow_score = 0
        if features.get("is_accumulating"):
            flow_score += 5
            signals.append("Accumulation detected (A/D line rising)")
        if features.get("obv_bullish_divergence"):
            flow_score += 4
            signals.append("OBV bullish divergence")
        elif features.get("obv_bearish_divergence"):
            flow_score -= 4
            signals.append("OBV bearish divergence")
        if features.get("vol_above_avg"):
            flow_score += 2
        weights["flow"] = flow_score
        score += flow_score

        # --- 8. Momentum (Weight: 5) ---
        roc20 = features.get("roc_20d", 0)
        roc50 = features.get("roc_50d", 0)
        mom_score = 0
        if roc20 > 5 and roc50 > 10:
            mom_score = 8
        elif roc20 > 2 and roc50 > 0:
            mom_score = 4
        elif roc20 < -5:
            mom_score = -6
        weights["momentum"] = mom_score
        score += mom_score

        # --- 9. Pattern Recognition (Weight: 5) ---
        pattern = features.get("pattern")
        pattern_score = 0
        if pattern == "ASCENDING_TRIANGLE":
            pattern_score = 6
            signals.append("Ascending Triangle pattern")
        elif pattern == "CUP_AND_HANDLE":
            pattern_score = 7
            signals.append("Cup & Handle pattern")
        weights["pattern"] = pattern_score
        score += pattern_score

        # --- 10. Fundamentals (Weight: 10) ---
        fund_score_val = features.get("fundamental_score", 50)
        fund_score = (fund_score_val - 50) * 0.2  # Scale to -10 to +10
        weights["fundamentals"] = round(fund_score, 1)
        score += fund_score

        # --- 11. Sentiment (Weight: 5) ---
        sent_score = features.get("sentiment_score", 50)
        sent_adjust = (sent_score - 50) * 0.1  # Scale to -5 to +5
        weights["sentiment"] = round(sent_adjust, 1)
        score += sent_adjust

        # --- 12. Market Regime Filter (Weight: Variable) ---
        regime_adjust = 0
        if market_regime:
            bias = market_regime.get("_bias", {})
            regime_dir = bias.get("direction", "SELECTIVE")
            regime_conf = bias.get("confidence", 50)
            
            if regime_dir in ["PRESERVATION", "RISK_OFF"]:
                # Dampen long signals in bear markets
                if score > 50:
                    regime_adjust = -(regime_conf * 0.2)
                    signals.append(f"Market regime {market_regime.get('_overall')}: reducing long exposure")
            elif regime_dir in ["STRONG_BULL", "BULL"]:
                # Boost long signals in bull markets
                if score > 50:
                    regime_adjust = (regime_conf * 0.1)
                    signals.append(f"Market regime {market_regime.get('_overall')}: favoring longs")
        
        weights["regime"] = round(regime_adjust, 1)
        score += regime_adjust

        # --- FINAL SCORING ---
        score = float(np.clip(score, 0, 100))
        
        if score >= 70:
            direction = "LONG"
        elif score <= 30:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        # Investment horizon confidence
        # Higher confluence = better for 3-12 month hold
        confluence_count = sum(1 for s in [trend_score, breakout_score, macd_score, flow_score, pattern_score] if abs(s) > 3)
        horizon_confidence = "HIGH" if confluence_count >= 4 and score > 65 else \
                           "MEDIUM" if confluence_count >= 3 else "LOW"

        return {
            "direction": direction,
            "confidence": score,
            "breakout_triggered": breakout is not None,
            "signals": signals,
            "weights": weights,
            "confluence_count": confluence_count,
            "horizon_confidence": horizon_confidence
        }


# ===============================================
# 10. PROPOSAL ENGINE (ENHANCED)
# ===============================================

@dataclass
class InvestmentProposal:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    tp_1: float
    tp_2: float
    tp_3: float  # NEW: Extended target for 3-12 month horizon
    risk_reward: float
    risk_reward_extended: float  # NEW: Extended R:R
    ai_confidence: float
    ai_grade: str
    thesis: str
    setup_type: str
    position_size_pct: float
    hold_period: str  # NEW: Estimated hold period
    horizon_confidence: str  # NEW: Quality of setup for 3-12 month
    sector_exposure: str  # NEW: Sector context
    chart_data: Any
    signals: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)


from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
import logging
import math
import pandas as pd
import html



logger = logging.getLogger(__name__)

class FeatureRegistry:

    ALLOWED_FEATURES = {
        # Technicals
        "rsi",
        "adx",
        "atr_pct",
        "ema_20_distance",
        "ema_50_distance",
        "volume_relative",

        # Structure
        "trend_strength",
        "bos_count",
        "liquidity_sweep",

        # Breakout
        "compression_ratio",
        "range_position",
        "breakout_strength",

        # Fundamentals
        "fundamental_score",
        "earnings_growth",

        # Sentiment
        "sentiment_score",

        # Context
        "weekly_alignment",
        "market_regime",
    }

    
class ProposalEngine:
    """
    Generates institutional-style investment proposals using:
    - Technicals
    - Smart Money Concepts (SMC)
    - Breakout analysis
    - Fundamentals
    - Sentiment
    - AI ensemble scoring
    """

    MIN_BARS = 60
    DEFAULT_RISK_MULTIPLIER = 2.0

    # =========================================================
    # PUBLIC API
    # =========================================================
    @staticmethod
    def build(
        symbol: str,
        df: pd.DataFrame,
        market_regime: Dict[str, Any],
        sector_rotation: Optional[Dict[str, Any]] = None,
        df_weekly: Optional[pd.DataFrame] = None,
    ) -> Optional[InvestmentProposal]:

        try:
            # -------------------------------------------------
            # Validation
            # -------------------------------------------------
            if not ProposalEngine._is_valid_dataframe(df):
                logger.warning(f"[{symbol}] Invalid dataframe supplied.")
                return None

            df_norm = normalize_ohlcv(df)

            if df_norm is None or df_norm.empty:
                logger.warning(f"[{symbol}] Failed to normalize OHLCV.")
                return None

            # -------------------------------------------------
            # Data Collection
            # -------------------------------------------------
            techs = TechnicalEngine.get_indicators(df_norm) or {}
            smc = SMCEngine.detect_structure(df_norm) or {}
            breakout = BreakoutEngine.analyze(df_norm, smc, df_weekly) or {}
            funds = FundamentalEngine.get_fundamentals(symbol) or {}
            sentiment = SentimentEngine.analyze(symbol) or {}

            close = ProposalEngine._safe_float(
                techs.get("close"),
                fallback=df_norm["close"].iloc[-1]
            )

            atr = ProposalEngine._safe_float(
                techs.get("atr"),
                fallback=max(close * 0.01, 0.01)
            )

            # -------------------------------------------------
            # AI Feature Engineering
            # -------------------------------------------------
            features = ProposalEngine._build_features(
                techs=techs,
                smc=smc,
                breakout=breakout,
                funds=funds,
                sentiment=sentiment
            )

            ai = AIEnsemble.predict(features, market_regime)

            if not ai or ai.get("direction") == "NEUTRAL":
                logger.info(f"[{symbol}] AI returned NEUTRAL.")
                return None

            direction = ai.get("direction", "NEUTRAL")
            confidence = ProposalEngine._clamp(
                ProposalEngine._safe_float(ai.get("confidence"), 50),
                0,
                100
            )

            # -------------------------------------------------
            # Trade Construction
            # -------------------------------------------------
            trade = ProposalEngine._build_trade_levels(
                direction=direction,
                close=close,
                atr=atr,
                breakout=breakout,
                smc=smc,
                df=df_norm,
            )

            sl = trade["sl"]
            tp1 = trade["tp1"]
            tp2 = trade["tp2"]
            tp3 = trade["tp3"]
            risk = trade["risk"]
            setup_type = trade["setup_type"]

            # -------------------------------------------------
            # Risk Metrics
            # -------------------------------------------------
            rr = ProposalEngine._calculate_rr(close, sl, tp2)
            rr_ext = ProposalEngine._calculate_rr(close, sl, tp3)

            # -------------------------------------------------
            # Position Sizing
            # -------------------------------------------------
            pos_size = ProposalEngine._calculate_position_size(
                confidence=confidence,
                rr=rr,
                market_regime=market_regime,
                horizon_confidence=ai.get("horizon_confidence")
            )

            # -------------------------------------------------
            # Hold Period
            # -------------------------------------------------
            hold_period = ProposalEngine._estimate_hold_period(
                setup_type=setup_type,
                breakout=breakout,
                rr_ext=rr_ext
            )

            # -------------------------------------------------
            # Sector Context
            # -------------------------------------------------
            sector_exp = ProposalEngine._build_sector_context(
                symbol=symbol,
                sector_rotation=sector_rotation
            )

            # -------------------------------------------------
            # Thesis
            # -------------------------------------------------
            thesis = ProposalEngine._build_thesis(
                setup_type=setup_type,
                breakout=breakout,
                techs=techs,
                funds=funds,
                ai=ai
            )

            # -------------------------------------------------
            # Chart Generation
            # -------------------------------------------------
            fig = ChartEngine.plot_setup(
                df_norm,
                smc,
                breakout,
                techs,
                direction,
                close,
                sl,
                tp1,
                tp2,
                tp3,
                df_weekly,
            )

            # -------------------------------------------------
            # AI Grade
            # -------------------------------------------------
            grade = ProposalEngine._calculate_grade(
                confidence=confidence,
                horizon_confidence=ai.get("horizon_confidence")
            )

            # -------------------------------------------------
            # Final Proposal
            # -------------------------------------------------
            return InvestmentProposal(
                symbol=symbol,
                direction=direction,
                entry_price=round(close, 2),
                stop_loss=round(sl, 2),
                tp_1=round(tp1, 2),
                tp_2=round(tp2, 2),
                tp_3=round(tp3, 2),
                risk_reward=rr,
                risk_reward_extended=rr_ext,
                ai_confidence=round(confidence, 2),
                ai_grade=grade,
                thesis=thesis,
                setup_type=setup_type,
                position_size_pct=round(pos_size, 2),
                hold_period=hold_period,
                horizon_confidence=ai.get("horizon_confidence", "LOW"),
                sector_exposure=sector_exp,
                chart_data=fig,
                signals=ai.get("signals", []),
                weights=ai.get("weights", {}),
            )

        except Exception as e:
            logger.exception(f"[{symbol}] Failed to build proposal: {e}")
            return None

    # =========================================================
    # VALIDATION
    # =========================================================
    @staticmethod
    def _is_valid_dataframe(df: Optional[pd.DataFrame]) -> bool:
        required_cols = {"open", "high", "low", "close", "volume"}

        return (
            df is not None
            and isinstance(df, pd.DataFrame)
            and len(df) >= ProposalEngine.MIN_BARS
            and required_cols.issubset(df.columns)
        )

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================
    @staticmethod
    def _build_features(
        techs: Dict,
        smc: Dict,
        breakout: Dict,
        funds: Dict,
        sentiment: Dict
    ) -> Dict[str, Any]:

        features = {
            **techs,
            **smc,
            **breakout,
            "fundamental_score": funds.get("score", 50),
            "sentiment_score": sentiment.get("score", 50),
            "earnings_growth": funds.get("growth", 0),
            "institutional_accumulation": int(
                breakout.get("is_accumulating", False)
            ),
            "weekly_alignment": int(
                breakout.get("weekly_aligned", False)
            ),
        }

        return features

    # =========================================================
    # TRADE CONSTRUCTION
    # =========================================================
    @staticmethod
    def _build_trade_levels(direction, close, atr, breakout, smc, df):

        smc = smc or {}
        breakout = breakout or {}

        setup_type = "TREND_CONTINUATION"

        breakout_type = (breakout.get("breakout_type") or "").upper()

        # safe df extraction
        last_low = df["low"].iloc[-1] if df is not None and len(df) > 0 and "low" in df else close - atr * 2
        last_high = df["high"].iloc[-1] if df is not None and len(df) > 0 and "high" in df else close + atr * 2

        if direction == "LONG":

            if "BREAKOUT" in breakout_type and "BULLISH" in breakout_type:

                range_low = breakout.get("range_low")
                range_low = range_low if isinstance(range_low, (int, float)) else close - atr * 2

                sl = min(last_low, range_low)
                setup_type = "BREAKOUT_LONG"

            else:

                sl = smc.get("last_swing_low", close - atr * ProposalEngine.DEFAULT_RISK_MULTIPLIER)

                if smc.get("trend") == "BULLISH":
                    setup_type = "PULLBACK_BUY"

            raw_risk = close - sl
            if raw_risk <= 0:
                raw_risk = atr * 0.75

            risk = max(raw_risk, atr * 0.5)

            tp1 = close + risk * 1.5
            tp2 = close + risk * 3.0
            tp3 = close + risk * 5.0

        else:

            if "BREAKOUT" in breakout_type and "BEARISH" in breakout_type:

                range_high = breakout.get("range_high")
                range_high = range_high if isinstance(range_high, (int, float)) else close + atr * 2

                sl = max(last_high, range_high)
                setup_type = "BREAKOUT_SHORT"

            else:

                sl = smc.get("last_swing_high", close + atr * ProposalEngine.DEFAULT_RISK_MULTIPLIER)

            raw_risk = sl - close
            if raw_risk <= 0:
                raw_risk = atr * 0.75

            risk = max(raw_risk, atr * 0.5)

            tp1 = close - risk * 1.5
            tp2 = close - risk * 3.0
            tp3 = close - risk * 5.0

        return {
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "risk": risk,
            "setup_type": setup_type
        }

    # =========================================================
    # POSITION SIZING
    # =========================================================
    @staticmethod
    def _calculate_position_size(
        confidence: float,
        rr: float,
        market_regime: Dict,
        horizon_confidence: Optional[str]
    ) -> float:

        bias = market_regime.get("_bias", {})
        max_pos = bias.get("max_position_size", 5)

        win_prob = confidence / 100

        # Kelly Criterion
        kelly = (
            (win_prob * rr - (1 - win_prob)) / rr
            if rr > 0
            else 0
        )

        # Conservative scaling
        kelly_lite = max(0, kelly * 0.5)

        pos_size = max(
            0.5,
            min(max_pos, kelly_lite * 100)
        )

        # Boost for high timeframe alignment
        if horizon_confidence == "HIGH":
            pos_size *= 1.2

        return min(pos_size, max_pos)

    # =========================================================
    # HOLD PERIOD
    # =========================================================
    @staticmethod
    def _estimate_hold_period(
        setup_type: str,
        breakout: Dict,
        rr_ext: float
    ) -> str:

        if setup_type in {"BREAKOUT_LONG", "BREAKOUT_SHORT"}:

            if breakout.get("squeeze_days", 0) > 10:
                return "3-6 months (Post-squeeze breakout)"

            return "1-3 months (Momentum breakout)"

        if setup_type == "PULLBACK_BUY":
            return "3-6 months (Trend continuation)"

        if rr_ext >= 4:
            return "6-12 months (Swing position)"

        return "1-3 months (Standard swing)"

    # =========================================================
    # SECTOR CONTEXT
    # =========================================================
    @staticmethod
    def _build_sector_context(
        symbol: str,
        sector_rotation: Optional[Dict]
    ) -> str:

        if not sector_rotation:
            return "N/A"

        leading = sector_rotation.get("leading", [])

        if not leading:
            return "Neutral sector environment"

        return f"Leading sectors: {', '.join(leading[:3])}"

    # =========================================================
    # THESIS
    # =========================================================
    @staticmethod
    def _build_thesis(
        setup_type: str,
        breakout: Dict,
        techs: Dict,
        funds: Dict,
        ai: Dict
    ) -> str:

        thesis_parts = [setup_type]

        mappings = [
            ("breakout_type", "Breakout"),
            ("pattern", "Pattern"),
        ]

        for key, label in mappings:
            value = breakout.get(key)

            if value:
                thesis_parts.append(f"{label}: {value}")

        if breakout.get("squeeze_detected"):
            thesis_parts.append(
                f"Squeeze: {breakout.get('squeeze_days', 0)}d compression"
            )

        if breakout.get("is_accumulating"):
            thesis_parts.append("Institutional accumulation")

        if ai.get("htf_aligned"):
            thesis_parts.append("Weekly timeframe aligned")

        if techs.get("golden_cross"):
            thesis_parts.append("Golden Cross")

        if techs.get("macd_bullish_divergence"):
            thesis_parts.append("MACD bullish divergence")

        growth = funds.get("growth", 0)

        if growth > 20:
            thesis_parts.append(f"Earnings growth: {growth}%")

        if funds.get("near_52w_high"):
            thesis_parts.append("Near 52-week high")

        return ". ".join(thesis_parts) + "."

    # =========================================================
    # GRADE
    # =========================================================
    @staticmethod
    def _calculate_grade(
        confidence: float,
        horizon_confidence: Optional[str]
    ) -> str:

        if confidence >= 85 and horizon_confidence == "HIGH":
            return "A+"

        if confidence >= 75:
            return "A"

        if confidence >= 65:
            return "B+"

        if confidence >= 55:
            return "B"

        return "C"

    # =========================================================
    # HELPERS
    # =========================================================
    @staticmethod
    def _calculate_rr(
        close: float,
        sl: float,
        target: float
    ) -> float:

        risk = abs(close - sl)

        if risk <= 0:
            return 1.0

        return round(abs(target - close) / risk, 2)

    @staticmethod
    def _safe_float(value: Any, fallback: float = 0.0) -> float:

        try:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return fallback

            return float(value)

        except Exception:
            return fallback

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

# ===============================================
# 11. CHART ENGINE (ENHANCED)
# ===============================================

class ChartEngine:
    @staticmethod
    def plot_setup(df, smc, breakout, techs, direction, entry, sl, tp1, tp2, tp3, df_weekly=None):
        df_plt = df.iloc[-90:].copy()
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.65, 0.15, 0.20],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        # --- Row 1: Candlesticks + Indicators ---
        fig.add_trace(go.Candlestick(
            x=df_plt.index, open=df_plt['open'], high=df_plt['high'],
            low=df_plt['low'], close=df_plt['close'], name='OHLC',
            increasing_line_color='#00c853', decreasing_line_color='#ff1744'
        ), row=1, col=1)

        # EMA overlays
        if len(df_plt) >= 21:
            ema9 = df_plt['close'].ewm(span=9).mean()
            ema21 = df_plt['close'].ewm(span=21).mean()
            ema50 = df_plt['close'].ewm(span=50).mean() if len(df_plt) >= 50 else None
            
            fig.add_trace(go.Scatter(x=df_plt.index, y=ema9, name='EMA 9',
                                     line=dict(color='yellow', width=1), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plt.index, y=ema21, name='EMA 21',
                                     line=dict(color='orange', width=1), opacity=0.7), row=1, col=1)
            if ema50 is not None:
                fig.add_trace(go.Scatter(x=df_plt.index, y=ema50, name='EMA 50',
                                         line=dict(color='purple', width=1), opacity=0.7), row=1, col=1)

        # Range box (if breakout or consolidation detected)
        if breakout.get("breakout_type") or breakout.get("is_tight_consolidation"):
            r_high = breakout.get("range_high")
            r_low = breakout.get("range_low")
            
            # Shaded range zone
            fig.add_hrect(y0=r_low, y1=r_high, fillcolor="blue", opacity=0.08,
                         line_width=0, row=1, col=1)
            fig.add_hline(y=r_high, line_dash="dot", line_color="blue", opacity=0.5,
                         row=1, col=1, annotation_text="Resistance")
            fig.add_hline(y=r_low, line_dash="dot", line_color="blue", opacity=0.5,
                         row=1, col=1, annotation_text="Support")

        # Pre-breakout zone (price near resistance)
        if breakout.get("pre_breakout"):
            r_high = breakout.get("range_high")
            fig.add_hrect(y0=r_high * 0.97, y1=r_high, fillcolor="orange", opacity=0.15,
                         line_width=0, row=1, col=1)

        # Entry / SL / TP lines
        fig.add_hline(y=entry, line_dash="dash", line_color="#ffeb3b",
                      row=1, col=1, annotation_text="Entry")
        fig.add_hline(y=sl, line_dash="dash", line_color="#f44336",
                      row=1, col=1, annotation_text="SL")
        fig.add_hline(y=tp2, line_dash="dot", line_color="#4caf50",
                      row=1, col=1, annotation_text="TP2")
        fig.add_hline(y=tp3, line_dash="longdash", line_color="#00e676",
                      row=1, col=1, annotation_text="TP3 (Extended)")

        # Volume
        colors = ['#00c853' if x > 0 else '#ff1744' for x in df_plt['close'] - df_plt['open']]
        vol_ma = df_plt['volume'].rolling(20).mean()
        fig.add_trace(go.Bar(x=df_plt.index, y=df_plt['volume'], marker_color=colors, opacity=0.5, name='Volume'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=df_plt.index, y=vol_ma, name='Vol MA20',
                                 line=dict(color='white', width=1, dash='dash'), opacity=0.5), row=2, col=1)

        # --- Row 3: RSI ---
        # Calculate RSI for the plot data
        delta = df_plt['close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))

        fig.add_trace(go.Scatter(x=df_plt.index, y=rsi, name='RSI',
                                 line=dict(color='#ab47bc', width=1.5)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.05, row=3, col=1)

        fig.update_layout(
            height=650,
            xaxis_rangeslider_visible=False,
            margin=dict(t=20, b=10, l=10, r=10),
            paper_bgcolor="#0e1117",
            font_color="white",
            plot_bgcolor="#0e1117",
            legend=dict(font=dict(size=9), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor='#1a1a2e'),
            yaxis=dict(gridcolor='#1a1a2e'),
            xaxis2=dict(gridcolor='#1a1a2e'),
            yaxis2=dict(gridcolor='#1a1a2e'),
            xaxis3=dict(gridcolor='#1a1a2e'),
            yaxis3=dict(gridcolor='#1a1a2e')
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#1a1a2e')
        return fig


# ===============================================
# 12. MARKET BREADTH ENGINE (NEW)
# ===============================================

class MarketBreadthEngine:
    """
    Proxies market breadth by analyzing a basket of major stocks.
    """

    BREADTH_BASKET = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V",
        "UNH", "JNJ", "WMT", "PG", "MA", "HD", "XOM", "BAC", "PFE", "KO",
        "DIS", "INTC", "CSCO", "PEP", "ABT", "CRM", "AVGO", "NFLX", "COST",
        "TMO", "MRK"
    ]

    @staticmethod
    def analyze() -> Dict[str, Any]:
        try:
            # Fetch all at once for efficiency
            df = yf.download(
                MarketBreadthEngine.BREADTH_BASKET,
                period="3mo", interval="1d", progress=False, auto_adjust=True
            )
            
            if isinstance(df.columns, pd.MultiIndex):
                closes = df['Close']
            else:
                closes = df[['Close']] if 'Close' in df.columns else df
            
            if closes.empty:
                return {"pct_above_50dma": 50, "pct_above_200dma": 50, "advance_decline": "NEUTRAL"}

            # Calculate % above moving averages
            sma50 = closes.rolling(50).mean()
            sma200 = closes.rolling(200).mean()
            
            latest = closes.iloc[-1]
            above_50 = (latest > sma50.iloc[-1]).sum()
            above_200 = (latest > sma200.iloc[-1]).sum() if len(closes) >= 200 else 0
            total = len(closes.columns)
            
            pct_50 = round((above_50 / total) * 100, 1)
            pct_200 = round((above_200 / total) * 100, 1) if len(closes) >= 200 else 50

            # Advance-decline proxy (recent performance)
            if len(closes) >= 5:
                week_return = (closes.iloc[-1] / closes.iloc[-6] - 1) * 100
                advancers = (week_return > 0).sum()
                decliners = (week_return < 0).sum()
                ad_ratio = round(advancers / (decliners + 1), 2)
            else:
                ad_ratio = 1.0

            # Breadth assessment
            if pct_50 > 70:
                breadth = "STRONG"  # Broad participation = healthy bull
            elif pct_50 > 55:
                breadth = "HEALTHY"
            elif pct_50 > 40:
                breadth = "MIXED"
            elif pct_50 > 25:
                breadth = "WEAK"
            else:
                breadth = "VERY_WEAK"  # Capitulation territory

            return {
                "pct_above_50dma": pct_50,
                "pct_above_200dma": pct_200,
                "advance_decline_ratio": ad_ratio,
                "breadth": breadth,
                "interpretation": {
                    "STRONG": "Broad market participation — strong foundation for breakout trades.",
                    "HEALTHY": "Most stocks trending well — selective breakouts favored.",
                    "MIXED": "Choppy market — only highest-conviction setups.",
                    "WEAK": "Declining breadth — reduce exposure, tighten stops.",
                    "VERY_WEAK": "Market distress — preservation mode."
                }.get(breadth, "Neutral market conditions.")
            }
        except Exception as e:
            logger.warning(f"Breadth analysis failed: {e}")
            return {"pct_above_50dma": 50, "pct_above_200dma": 50, "breadth": "MIXED",
                    "interpretation": "Unable to calculate breadth."}


# ===============================================
# 13. ASSET UNIVERSE DEFINITIONS
# ===============================================

STOCKS_ALL = [
    # Mega Cap Tech
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "NFLX", "INTU", "QCOM", "AMD", "IBM", "NOW", "INTC", "MU", "LRCX",

    # Finance
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB",
    "SPGI", "COIN",

    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "ISRG", "GILD", "VRTX", "MDT", "DXCM", "NVO",

    # Consumer & Retail
    "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "COST", "PG", "KO", "PEP",
    "WMT", "PM",

    # Industrial & Energy
    "XOM", "CVX", "COP", "BA", "CAT", "HON", "UPS", "RTX", "LMT", "GE", "DE",

    # ETFs / Indices
    "SPY",

    # Growth / Tech / Speculative
    "SOFI", "RBLX", "RIVN", "AI", "BABA", "OKLO", "MSTR", "OPEN", "SMR", "IONQ",
    "RDDT", "QBTS", "SMCI", "RUM", "TTD", "ON",

    # Small / Mid / Biotech / Other
    "CDE", "PSTV", "RNAZ", "EH", "AAOI", "CHPT", "HOTH", "DVLT", "TOVX", "BSOL",
    "AAL", "BBAR", "APH", "NGD", "ACHR", "PONY", "BZAI", "NNE", "RBLX",

    # Real Estate / Infrastructure
    "PLD", "AMT", "EQIX", "INVH",

    # Additional industrial / diversified
    "HON", "DE",

    # Existing misc / dual listings kept once
    "BRK-B"
]

CRYPTO_ALL = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "DOT-USD",
    "MATIC-USD", "SHIB-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "ATOM-USD", "UNI-USD",
    "XMR-USD", "ETC-USD", "BCH-USD", "NEAR-USD", "FIL-USD", "APT-USD", "ARB-USD", "OP-USD"
]

FOREX_ALL = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "NZDUSD=X", "USDCAD=X",
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURAUD=X", "EURNZD=X"
]

# ===============================================
# 14. MAIN UI
# ===============================================

def _render_market_dashboard(market_regime: Dict, sector_rotation: Dict, breadth: Dict):
    """Render the market overview dashboard at the top of the page."""
    
    overall = market_regime.get("_overall", "NEUTRAL")
    bias = market_regime.get("_bias", {})
    
    # Color mapping for regimes
    regime_colors = {
        "STRONG_BULL": "#00e676", "BULL": "#69f0ae", "NEUTRAL": "#ffeb3b",
        "BEAR": "#ff5252", "STRONG_BEAR": "#d50000", "RISK_OFF": "#d50000", "CAUTIOUS": "#ff9800"
    }
    regime_color = regime_colors.get(overall, "#ffeb3b")
    
    # Market Regime Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-left: 4px solid {regime_color};
                padding: 20px; border-radius: 8px; margin-bottom: 15px;">
        <h3 style="color: {regime_color}; margin: 0 0 10px 0;">
            🌍 Market Regime: {overall.replace('_', ' ')}
        </h3>
        <p style="color: #ccc; margin: 5px 0;">
            <b>Strategy:</b> {bias.get('direction', 'SELECTIVE')} | 
            <b>Risk Level:</b> {bias.get('risk_level', 'MODERATE')} |
            <b>Max Position:</b> {bias.get('max_position_size', 5)}%
        </p>
        <p style="color: #999; font-size: 0.9em; margin: 5px 0;">
            💡 {bias.get('advice', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics in columns
    cols = st.columns(5)
    
    # SPY
    spy = market_regime.get("SPY", {})
    with cols[0]:
        trend_color = "#00e676" if spy.get("trend") == "BULL" else "#ff5252" if spy.get("trend") == "BEAR" else "#ffeb3b"
        st.markdown(f"""
        <div style="text-align:center; padding:10px; background:#1a1a2e; border-radius:8px;">
            <small style="color:#999;">S&P 500 (SPY)</small><br>
            <span style="color:white; font-size:1.2em;"><b>${spy.get('price', 0):.0f}</b></span><br>
            <span style="color:{trend_color};">{spy.get('trend', '?')}</span> · 
            <span style="color:{'#00e676' if spy.get('roc_20d', 0) > 0 else '#ff5252'};">
                {spy.get('roc_20d', 0):+.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # QQQ
    qqq = market_regime.get("QQQ", {})
    with cols[1]:
        trend_color = "#00e676" if qqq.get("trend") == "BULL" else "#ff5252" if qqq.get("trend") == "BEAR" else "#ffeb3b"
        st.markdown(f"""
        <div style="text-align:center; padding:10px; background:#1a1a2e; border-radius:8px;">
            <small style="color:#999;">Nasdaq (QQQ)</small><br>
            <span style="color:white; font-size:1.2em;"><b>${qqq.get('price', 0):.0f}</b></span><br>
            <span style="color:{trend_color};">{qqq.get('trend', '?')}</span> · 
            <span style="color:{'#00e676' if qqq.get('roc_20d', 0) > 0 else '#ff5252'};">
                {qqq.get('roc_20d', 0):+.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # VIX
    vix = market_regime.get("VIX", {})
    vix_color = "#ff5252" if vix.get("regime") in ["HIGH_FEAR", "ELEVATED"] else "#00e676"
    with cols[2]:
        st.markdown(f"""
        <div style="text-align:center; padding:10px; background:#1a1a2e; border-radius:8px;">
            <small style="color:#999;">VIX</small><br>
            <span style="color:white; font-size:1.2em;"><b>{vix.get('level', 0):.1f}</b></span><br>
            <span style="color:{vix_color};">{vix.get('regime', '?')}</span>
        </div>
        """, unsafe_allow_html=True)

    # DXY
    dxy = market_regime.get("DXY", {})
    with cols[3]:
        st.markdown(f"""
        <div style="text-align:center; padding:10px; background:#1a1a2e; border-radius:8px;">
            <small style="color:#999;">Dollar (DXY)</small><br>
            <span style="color:white; font-size:1.2em;"><b>{dxy.get('level', 0):.1f}</b></span><br>
            <span style="color:#999; font-size:0.8em;">{dxy.get('impact', '?')}</span>
        </div>
        """, unsafe_allow_html=True)

    # Market Breadth
    with cols[4]:
        breadth_color = "#00e676" if breadth.get("breadth") in ["STRONG", "HEALTHY"] else \
                        "#ff5252" if breadth.get("breadth") in ["WEAK", "VERY_WEAK"] else "#ffeb3b"
        st.markdown(f"""
        <div style="text-align:center; padding:10px; background:#1a1a2e; border-radius:8px;">
            <small style="color:#999;">Market Breadth</small><br>
            <span style="color:white; font-size:1.2em;"><b>{breadth.get('pct_above_50dma', 50)}%</b></span><br>
            <span style="color:{breadth_color};">{breadth.get('breadth', '?')}</span>
        </div>
        """, unsafe_allow_html=True)

    # Sector Rotation Table
    
    st.markdown("---")
    st.subheader("📊 Sector Rotation Map")
    
    sectors = sector_rotation.get("sectors", {})
    ranked = sector_rotation.get("ranked", [])
    
    if ranked:
        sector_rows = []
        for symbol, data in ranked:
            trend_color = "🟢" if data.get("trend") == "STRONG" else "🟡" if data.get("trend") == "NEUTRAL" else "🔴"
            sector_rows.append({
                "Sector": data.get("name", symbol),
                "ETF": symbol,
                "1M": f"{data.get('return_1m', 0):+.1f}%",
                "3M": f"{data.get('return_3m', 0):+.1f}%",
                "6M": f"{data.get('return_6m', 0):+.1f}%",
                "RS vs SPY": f"{data.get('relative_strength', 0):+.1f}%",
                "Momentum": f"{data.get('momentum_score', 0):+.1f}",
                "Trend": f"{trend_color} {data.get('trend', '?')}",
                "Volume": data.get("volume_trend", "?")
            })
        
        df_sector = pd.DataFrame(sector_rows)
        st.dataframe(df_sector, use_container_width=True, hide_index=True,
                     column_config={
                         "1M": st.column_config.NumberColumn(format="%.1f%%"),
                         "3M": st.column_config.NumberColumn(format="%.1f%%"),
                         "6M": st.column_config.NumberColumn(format="%.1f%%"),
                     })
        
        leading = sector_rotation.get("leading_names", [])
        lagging = sector_rotation.get("lagging_names", [])
        st.markdown(f"**🟢 Leading:** {', '.join(leading)} | **🔴 Lagging:** {', '.join(lagging)}")


def main_ui():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    st.set_page_config(layout="wide", page_title="Institutional Breakout Terminal v4")

    st.title("🚀 Institutional Breakout & Investment Terminal")
    st.caption("3–12 Month Investment Opportunity Scanner | Multi-Timeframe | Market Regime Aware")

    # =========================================================
    # SIDEBAR CONFIG
    # =========================================================
    st.sidebar.header("⚙️ Configuration")

    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Quick Scan (Watchlist)", "Full Market Scan (All Assets)"]
    )

    asset_class = st.sidebar.selectbox("Asset Class", ["Stocks", "Crypto", "Forex"])

    if scan_mode == "Quick Scan (Watchlist)":
        default_symbols = (
            "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, AMD, META, JPM, LLY, AVGO, CRM"
            if asset_class == "Stocks"
            else "BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD"
            if asset_class == "Crypto"
            else "EURUSD=X, GBPUSD=X, USDJPY=X"
        )
    else:
        default_symbols = (
            ", ".join(STOCKS_ALL) if asset_class == "Stocks"
            else ", ".join(CRYPTO_ALL) if asset_class == "Crypto"
            else ", ".join(FOREX_ALL)
        )

    symbols_raw = st.sidebar.text_area("Symbols", default_symbols, height=120)
    symbols = sorted({s.strip().upper() for s in symbols_raw.split(",") if s.strip()})

    horizon = st.sidebar.selectbox(
        "Investment Horizon",
        ["3-6 months (Swing)", "6-12 months (Position)", "Any (Show All)"]
    )

    min_confidence = st.sidebar.slider("Min AI Confidence", 40, 90, 55)

    # =========================================================
    # HELPERS
    # =========================================================
    def safe(x, default=0):
        if x is None:
            return default
        return x

    def safe_str(x):
        return str(x) if x is not None else ""

    # =========================================================
    # MAIN SCAN BUTTON
    # =========================================================
    if st.button("🚀 SCAN FOR OPPORTUNITIES"):

        if not symbols:
            st.warning("No symbols provided")
            return

        # =========================================================
        # MARKET ANALYSIS
        # =========================================================
        st.subheader("🌍 Market Analysis")

        with st.spinner("Market regime..."):
            market_regime = MarketRegimeEngine.analyze()

        with st.spinner("Sector rotation..."):
            sector_rotation = SectorRotationEngine.analyze()

        with st.spinner("Market breadth..."):
            breadth = MarketBreadthEngine.analyze()

        _render_market_dashboard(market_regime, sector_rotation, breadth)

        # =========================================================
        # SCAN SYMBOLS
        # =========================================================
        st.markdown("---")
        st.subheader("📡 Scanning Assets")

        progress = st.progress(0)
        proposals = []

        def fetch(sym):
            try:
                tf = fetch_multi_timeframe(sym)
                df = tf.get("daily")
                if df is None or len(df) == 0:
                    return None

                return ProposalEngine.build(
                    sym,
                    df,
                    market_regime,
                    sector_rotation=sector_rotation,
                    df_weekly=tf.get("weekly")
                )
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(fetch, s): s for s in symbols}

            for i, f in enumerate(as_completed(futures)):
                progress.progress((i + 1) / len(symbols))
                res = f.result()
                if res and res.ai_confidence >= min_confidence:
                    proposals.append(res)

        progress.empty()

        # =========================================================
        # NO RESULTS
        # =========================================================
        if not proposals:
            st.error("No valid setups found. Lower confidence or expand universe.")
            return

        proposals.sort(key=lambda x: x.ai_confidence, reverse=True)

        # =========================================================
        # SUMMARY
        # =========================================================
        st.success(f"Found {len(proposals)} setups")

        longs = sum(p.direction == "LONG" for p in proposals)
        shorts = sum(p.direction == "SHORT" for p in proposals)
        avg_conf = np.mean([p.ai_confidence for p in proposals])

        st.markdown(f"""
        **LONG:** {longs} | **SHORT:** {shorts} | **Avg Confidence:** {avg_conf:.1f}%
        """)

        # =========================================================
        # TABLE VIEW (SAFE)
        # =========================================================
        st.subheader("📊 Opportunity Table")

        table = []
        for p in proposals[:20]:
            table.append({
                "Symbol": p.symbol,
                "Direction": p.direction,
                "Setup": safe_str(p.setup_type),
                "Confidence": p.ai_confidence,
                "Entry": p.entry_price,
                "SL": p.stop_loss,
                "TP2": p.tp_2,
                "TP3": p.tp_3,
                "R:R": f"1:{p.risk_reward}"
            })

        df = pd.DataFrame(table)
        st.dataframe(df, use_container_width=True)

        # =========================================================
        # DETAIL CARDS (STREAMLIT-NATIVE, NO HTML)
        # =========================================================
        st.subheader("📈 Detailed Setups")

        cols = st.columns(2)

        for i, p in enumerate(proposals[:30]):

            with cols[i % 2]:

                border = "🟢" if "BREAKOUT" in safe_str(p.setup_type) else "🟡"

                st.markdown(f"## {border} {p.symbol}")
                st.markdown(f"**{p.direction} | {p.setup_type} | {p.ai_grade} ({p.ai_confidence}%)**")

                c1, c2, c3 = st.columns(3)

                c1.metric("Entry", p.entry_price)
                c2.metric("Stop", p.stop_loss)
                c3.metric("Size %", p.position_size_pct)

                c1.metric("TP2", p.tp_2)
                c2.metric("TP3", p.tp_3)
                c3.metric("R:R", f"1:{p.risk_reward}")

                st.write("📌 Thesis:")
                st.write(safe_str(p.thesis))

                if getattr(p, "chart_data", None) is not None:
                    try:
                        st.plotly_chart(p.chart_data, use_container_width=True)
                    except Exception:
                        st.warning("Chart unavailable")

                st.markdown("---")

        # =========================================================
        # PORTFOLIO SUMMARY
        # =========================================================
        st.subheader("💼 Portfolio Summary")

        long_exp = sum(p.position_size_pct for p in proposals if p.direction == "LONG")
        short_exp = sum(p.position_size_pct for p in proposals if p.direction == "SHORT")

        c1, c2, c3 = st.columns(3)

        c1.metric("Long Exposure", f"{long_exp:.1f}%")
        c2.metric("Short Exposure", f"{short_exp:.1f}%")
        c3.metric("Net", f"{long_exp - short_exp:+.1f}%")
if __name__ == "__main__":
    main_ui()
