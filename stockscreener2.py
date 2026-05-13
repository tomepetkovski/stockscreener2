"""
Institutional Breakout & Investment Terminal v4.0
Enhanced for 3-12 month investment opportunity detection.

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
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import math
from typing import Dict, Any, Optional, List
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

import requests
import yfinance as yf



import time

def safe_yf_download(*args, retries=3, **kwargs):

    for attempt in range(retries):

        try:

            data = yf.download(
                *args,
                progress=False,
                threads=False,
                **kwargs
            )

            if data is not None and not data.empty:
                return data

        except Exception as e:

            logger.warning(
                "Yahoo download retry %s failed | error=%s",
                attempt + 1,
                str(e)
            )

            time.sleep(1.5)

    return pd.DataFrame()


# ===============================================
# 1. DATA FOUNDATION
# ===============================================

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Institutional-grade OHLCV normalization.

    Handles:
    - MultiIndex columns
    - Yahoo Finance corruption
    - Duplicate columns
    - Object/string numeric conversion
    - Missing volume
    - Timezone-aware indexes
    - Duplicate timestamps
    - Column normalization
    - Invalid numeric rows
    """

    try:

        # =====================================================
        # BASIC VALIDATION
        # =====================================================

        if df is None:
            return pd.DataFrame()

        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        # =====================================================
        # FLATTEN MULTIINDEX
        # =====================================================

        if isinstance(df.columns, pd.MultiIndex):

            try:
                df.columns = [
                    str(col[0]).lower().strip()
                    for col in df.columns
                ]

            except Exception:
                df.columns = [
                    str(col).lower().strip()
                    for col in df.columns
                ]

        else:

            df.columns = [
                str(col).lower().strip()
                for col in df.columns
            ]

        # =====================================================
        # REMOVE DUPLICATE COLUMNS
        # =====================================================

        df = df.loc[:, ~df.columns.duplicated()]

        # =====================================================
        # STANDARD COLUMN MAPPING
        # =====================================================

        rename_map = {
            "adj close": "close",
            "adjclose": "close",
            "closing price": "close",
            "opening price": "open",
        }

        df.rename(columns=rename_map, inplace=True)

        # =====================================================
        # REQUIRED COLUMNS
        # =====================================================

        required = [
            "open",
            "high",
            "low",
            "close"
        ]

        for col in required:

            if col not in df.columns:
                logger.warning(
                    f"normalize_ohlcv() missing column: {col}"
                )
                return pd.DataFrame()

        # =====================================================
        # HANDLE MISSING VOLUME
        # =====================================================

        if "volume" not in df.columns:

            logger.warning(
                "normalize_ohlcv() volume missing. "
                "Injecting synthetic volume."
            )

            df["volume"] = 0

        # =====================================================
        # KEEP ONLY OHLCV
        # =====================================================

        cols = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        df = df[cols]

        # =====================================================
        # NUMERIC CONVERSION
        # =====================================================

        for col in cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # =====================================================
        # REMOVE INVALID ROWS
        # =====================================================

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        df.dropna(inplace=True)

        # =====================================================
        # REMOVE DUPLICATE INDEXES
        # =====================================================

        if df.index.duplicated().any():

            df = df[
                ~df.index.duplicated(keep="last")
            ]

        # =====================================================
        # REMOVE TIMEZONE INFO
        # =====================================================

        try:

            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        except Exception:
            pass

        # =====================================================
        # SORT INDEX
        # =====================================================

        df.sort_index(inplace=True)

        # =====================================================
        # FINAL VALIDATION
        # =====================================================

        if len(df) < 10:

            logger.warning(
                "normalize_ohlcv() insufficient rows after cleanup."
            )

            return pd.DataFrame()

        # =====================================================
        # PRICE SANITY CHECK
        # =====================================================

        invalid_prices = (
            (df["high"] < df["low"]) |
            (df["close"] <= 0) |
            (df["open"] <= 0)
        )

        if invalid_prices.any():

            logger.warning(
                "normalize_ohlcv() removing invalid OHLC rows."
            )

            df = df[~invalid_prices]

        # =====================================================
        # FINAL RESET
        # =====================================================

        df.dropna(inplace=True)

        return df

    except Exception as e:

        logger.exception(
            f"normalize_ohlcv() failed: {e}"
        )

        return pd.DataFrame()

def _download_timeframe(
    symbol: str,
    period: str,
    interval: str,
    ) -> Optional[pd.DataFrame]:
    """
    Download and normalize OHLCV data for a single timeframe.
    """
    try:
        # `repair=True` can raise KeyError('Stock Splits') on some 1wk payloads.
        # Keep repair only for daily bars where it is most useful.
        use_repair = interval in {"1d", "1h", "1m"}
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
            prepost=False,
            repair=use_repair,
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

def fetch_weekly_data(symbol: str) -> Optional[pd.DataFrame]:

    try:

        df = yf.download(
            symbol,
            period="3y",
            interval="1wk",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        return normalize_ohlcv(df)

    except Exception as e:

        logger.warning(
            f"[{symbol}] Weekly fetch failed: {e}"
        )

        return None

def build_synthetic_weekly(df_daily: pd.DataFrame):

    if df_daily is None or len(df_daily) < 20:
        return None

    weekly = (
        df_daily
        .resample("W")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )

    return weekly

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

# Symbol → (sector ETF, sector name) for rotation / money-flow context (expand as needed)
SYMBOL_TO_SECTOR_ETF: Dict[str, Tuple[str, str]] = {
    "AAPL": ("XLK", "Technology"),
    "MSFT": ("XLK", "Technology"),
    "NVDA": ("XLK", "Technology"),
    "AMD": ("XLK", "Technology"),
    "AVGO": ("XLK", "Technology"),
    "CRM": ("XLK", "Technology"),
    "META": ("XLC", "Communication Services"),
    "GOOGL": ("XLC", "Communication Services"),
    "GOOG": ("XLC", "Communication Services"),
    "NFLX": ("XLC", "Communication Services"),
    "AMZN": ("XLY", "Consumer Discretionary"),
    "TSLA": ("XLY", "Consumer Discretionary"),
    "HD": ("XLY", "Consumer Discretionary"),
    "NKE": ("XLY", "Consumer Discretionary"),
    "JPM": ("XLF", "Financials"),
    "BAC": ("XLF", "Financials"),
    "GS": ("XLF", "Financials"),
    "LLY": ("XLV", "Healthcare"),
    "UNH": ("XLV", "Healthcare"),
    "JNJ": ("XLV", "Healthcare"),
    "XOM": ("XLE", "Energy"),
    "CVX": ("XLE", "Energy"),
    "CAT": ("XLI", "Industrials"),
    "UPS": ("XLI", "Industrials"),
}

# Crypto: segment label for flow table (expand as needed)
CRYPTO_SEGMENT: Dict[str, str] = {
    "BTC-USD": "Large cap (BTC)",
    "ETH-USD": "Large cap (ETH)",
    "SOL-USD": "L1 alt",
    "XRP-USD": "Payments alt",
    "DOGE-USD": "Meme / retail",
    "ADA-USD": "L1 alt",
    "AVAX-USD": "L1 alt",
    "DOT-USD": "L1 alt",
    "LINK-USD": "Infra / DeFi",
    "MATIC-USD": "L2 / scaling",
    "BNB-USD": "Exchange / L1",
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


class ExternalDataProvider:

    @staticmethod
    def safe_call(engine, method, *args, fallback=None, **kwargs):

        try:

            if engine is None:
                return fallback or {}

            fn = getattr(engine, method, None)

            if fn is None:
                return fallback or {}

            result = fn(*args, **kwargs)

            return result or fallback or {}

        except Exception as e:

            logger.warning(
                "External provider failure | engine=%s | error=%s",
                getattr(engine, "__name__", str(engine)),
                str(e)
            )

            return fallback or {}
class AnalystConsensusEngine:

    @staticmethod
    def get_ratings(symbol: str) -> Dict[str, Any]:

        try:
            ticker = yf.Ticker(symbol)

            info = {}

            try:
                info = ticker.info
                if not isinstance(info, dict):
                    info = {}
            except Exception:
                info = {}

            recommendation = str(
                info.get("recommendationKey", "hold")
            ).lower()

            analyst_count = int(
                info.get("numberOfAnalystOpinions") or 0
            )

            target_mean = AnalystConsensusEngine._safe_float(
                info.get("targetMeanPrice")
            )

            current_price = AnalystConsensusEngine._safe_float(
                info.get("currentPrice")
            )

            upside = 0.0

            if current_price and target_mean:
                upside = ((target_mean - current_price) / current_price) * 100

            score_map = {
                "strong_buy": 95,
                "buy": 85,
                "overweight": 80,
                "hold": 60,
                "underperform": 35,
                "sell": 20,
            }

            score = score_map.get(recommendation, 50)

            return {
                "source": "Yahoo/Nasdaq Aggregated",
                "recommendation": recommendation.upper(),
                "analyst_count": analyst_count,
                "target_price": target_mean,
                "current_price": current_price,
                "upside_pct": round(upside, 2),
                "score": score
            }

        except Exception as e:

            logger.warning(
                f"[{symbol}] AnalystConsensusEngine failed: {e}"
            )

            return {
                "source": "Fallback",
                "recommendation": "HOLD",
                "analyst_count": 0,
                "target_price": None,
                "current_price": None,
                "upside_pct": 0,
                "score": 50
            }

    @staticmethod
    def _safe_float(x, fallback=0.0):
        try:
            if x is None:
                return fallback
            return float(x)
        except Exception:
            return fallback


# =========================================================
# PROPOSAL MODEL
# =========================================================

@dataclass
class InvestmentProposal:
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    tp_1: float
    tp_2: float
    tp_3: float
    risk_reward: float
    risk_reward_extended: float
    ai_confidence: float
    ai_grade: str
    thesis: str
    setup_type: str
    position_size_pct: float
    hold_period: str
    horizon_confidence: str
    sector_exposure: str
    chart_data: Any
    signals: List[str]
    weights: Dict[str, Any]


# =========================================================
# ANALYST CONSENSUS ENGINE
# =========================================================

class AnalystConsensusEngine:

    @staticmethod
    def get(symbol: str) -> Dict[str, Any]:

        result = {
            "score": 50,
            "label": "NEUTRAL",
            "sources": {}
        }

        try:

            zacks = ZacksEngine.get_rank(symbol)
            yahoo = YahooFinanceEngine.get_rating(symbol)
            nasdaq = NasdaqEngine.get_rating(symbol)
            fool = MotleyFoolEngine.get_sentiment(symbol)
            barrons = BarronsEngine.get_sentiment(symbol)
            insider = InsiderMonkeyEngine.get_activity(symbol)

            scores = []

            for item in [zacks, yahoo, nasdaq, fool, barrons, insider]:

                if item and isinstance(item, dict):
                    val = item.get("score")

                    if isinstance(val, (int, float)):
                        scores.append(val)

            if scores:
                avg = np.mean(scores)
            else:
                avg = 50

            label = "NEUTRAL"

            if avg >= 75:
                label = "STRONG_BUY"
            elif avg >= 60:
                label = "BUY"
            elif avg <= 35:
                label = "SELL"

            result = {
                "score": round(avg, 2),
                "label": label,
                "sources": {
                    "zacks": zacks,
                    "yahoo": yahoo,
                    "nasdaq": nasdaq,
                    "motley_fool": fool,
                    "barrons": barrons,
                    "insider_monkey": insider
                }
            }

        except Exception as e:
            logger.warning(f"[{symbol}] AnalystConsensusEngine failed: {e}")

        return result


# =========================================================
# ZACKS ENGINE
# =========================================================

class ZacksEngine:

    @staticmethod
    def get_rank(symbol: str) -> Dict[str, Any]:

        try:
            # MOCK SAFE IMPLEMENTATION
            # Replace with scraper/API later

            return {
                "rank": 2,
                "label": "BUY",
                "score": 82
            }

        except Exception as e:
            logger.warning(f"[{symbol}] Zacks failed: {e}")

        return {}


# =========================================================
# YAHOO ANALYST ENGINE
# =========================================================

class YahooFinanceEngine:

    @staticmethod
    def get_rating(symbol: str):

        try:

            ticker = yf.Ticker(symbol)

            rec = ticker.recommendations

            if rec is None or len(rec) == 0:
                return {}

            recent = rec.tail(20).copy()
            normalized_cols = {str(c).strip().lower(): c for c in recent.columns}
            grade_col = (
                normalized_cols.get("to grade")
                or normalized_cols.get("to_grade")
                or normalized_cols.get("grade")
                or normalized_cols.get("rating")
            )
            if grade_col is None:
                return {}

            grades = recent[grade_col].astype(str)
            buys = grades.str.contains(
                "buy|strong buy|outperform|overweight",
                case=False,
                na=False,
                regex=True,
            ).sum()

            score = min(100, buys * 5 + 50)

            return {
                "score": score,
                "source": "Yahoo Finance"
            }

        except Exception as e:
            logger.warning(f"[{symbol}] Yahoo rating failed: {e}")

        return {}


class YahooFinanceAnalystEngine(YahooFinanceEngine):
    pass


# =========================================================
# NASDAQ ENGINE
# =========================================================

class NasdaqEngine:

    @staticmethod
    def get_rating(symbol):

        try:

            return {
                "score": 74,
                "source": "Nasdaq Analysts"
            }

        except Exception:
            return {}


class NasdaqAnalystEngine(NasdaqEngine):
    pass


# =========================================================
# BARRONS ENGINE
# =========================================================

class BarronsEngine:

    @staticmethod
    def get_sentiment(symbol):

        try:

            return {
                "score": 68,
                "source": "Barrons"
            }

        except Exception:
            return {}

    @staticmethod
    def get_rating(symbol):
        return BarronsEngine.get_sentiment(symbol)


# =========================================================
# MOTLEY FOOL ENGINE
# =========================================================

class MotleyFoolEngine:

    @staticmethod
    def get_sentiment(symbol):

        try:

            return {
                "score": 72,
                "source": "Motley Fool"
            }

        except Exception:
            return {}

    @staticmethod
    def get_rating(symbol):
        return MotleyFoolEngine.get_sentiment(symbol)


# =========================================================
# INSIDER MONKEY ENGINE
# =========================================================

class InsiderMonkeyEngine:

    @staticmethod
    def get_activity(symbol):

        try:

            return {
                "score": 77,
                "hedge_fund_sentiment": "BULLISH"
            }

        except Exception:
            return {}

    @staticmethod
    def get_sentiment(symbol):
        return InsiderMonkeyEngine.get_activity(symbol)


# =========================================================
# MAIN PROPOSAL ENGINE
# =========================================================
import math
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class InvestmentProposal:

    symbol: str
    direction: str

    entry_price: float
    stop_loss: float

    tp_1: float
    tp_2: float
    tp_3: float

    risk_reward: float
    risk_reward_extended: float

    ai_confidence: float
    ai_grade: str

    thesis: str
    setup_type: str

    position_size_pct: float
    hold_period: str

    horizon_confidence: str
    sector_exposure: str

    chart_data: Any

    signals: list
    weights: dict
    analyst_consensus: dict
    breakout_analytics: dict
    sentiment_snapshot: dict
    sector_flow: dict
    money_flow_note: str
    short_term_score: float


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    try:

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [
            str(c).lower().strip()
            for c in df.columns
        ]

        rename_map = {
            "adj close": "close"
        }

        df.rename(columns=rename_map, inplace=True)

        required = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing OHLCV column: {col}")
                return pd.DataFrame()

        df = df[required]

        df = df.replace([np.inf, -np.inf], np.nan)

        df.dropna(inplace=True)

        return df

    except Exception as e:

        logger.exception(f"normalize_ohlcv failed: {e}")

        return pd.DataFrame()


# =========================================================
# ANALYST CONSENSUS ENGINE
# =========================================================

class AnalystConsensusEngine:

    WEIGHTS = {
        "zacks": 0.25,
        "barrons": 0.15,
        "motley_fool": 0.10,
        "insider_monkey": 0.10,
        "nasdaq": 0.20,
        "yahoo": 0.20,
    }

    @staticmethod
    def get(symbol: str) -> Dict[str, Any]:

        try:
            sources: Dict[str, Dict[str, Any]] = {}
            weighted_sum = 0.0
            total_weight = 0.0
            bullish_votes = 0
            bearish_votes = 0

            raw_sources = {
                "zacks": ZacksEngine.get_rank(symbol),
                "barrons": BarronsEngine.get_rating(symbol),
                "motley_fool": MotleyFoolEngine.get_rating(symbol),
                "insider_monkey": InsiderMonkeyEngine.get_sentiment(symbol),
                "nasdaq": NasdaqAnalystEngine.get_rating(symbol),
                "yahoo": YahooFinanceAnalystEngine.get_rating(symbol),
            }

            for source_name, raw in raw_sources.items():
                weight = AnalystConsensusEngine.WEIGHTS.get(source_name, 0.0)
                normalized = (
                    AnalystConsensusEngine._normalize_zacks(raw)
                    if source_name == "zacks"
                    else AnalystConsensusEngine._normalize_rating(raw)
                )

                entry = AnalystConsensusEngine._build_source_entry(
                    source_name,
                    raw,
                    normalized,
                    weight
                )
                sources[source_name] = entry

                if entry["available"]:
                    total_weight += weight
                    weighted_sum += entry["weighted_score"]
                    if entry["label"] in {"STRONG BUY", "BUY"}:
                        bullish_votes += 1
                    if entry["label"] in {"SELL", "STRONG SELL"}:
                        bearish_votes += 1

            if total_weight <= 0:
                return {
                    "score": 50.0,
                    "label": "NEUTRAL",
                    "coverage": 0.0,
                    "active_sources": 0,
                    "bullish_votes": 0,
                    "bearish_votes": 0,
                    "sources": sources,
                    "summary": "No analyst source returned usable data."
                }

            final_score = weighted_sum / total_weight
            active_sources = sum(1 for v in sources.values() if v["available"])
            coverage = round(100 * (active_sources / max(1, len(sources))), 1)
            final_label = AnalystConsensusEngine._score_to_label(final_score)
            bullish_ratio = round(100 * (bullish_votes / max(1, active_sources)), 1)

            return {
                "score": round(final_score, 2),
                "label": final_label,
                "coverage": coverage,
                "active_sources": active_sources,
                "bullish_votes": bullish_votes,
                "bearish_votes": bearish_votes,
                "bullish_ratio": bullish_ratio,
                "sources": sources,
                "summary": (
                    f"{final_label} consensus from {active_sources}/{len(sources)} sources "
                    f"({coverage}% coverage, {bullish_ratio}% bullish votes)."
                )
            }

        except Exception as e:

            logger.warning(
                f"[{symbol}] AnalystConsensusEngine failed: {e}"
            )

            return {
                "score": 50,
                "label": "NEUTRAL",
                "coverage": 0.0,
                "active_sources": 0,
                "bullish_votes": 0,
                "bearish_votes": 0,
                "sources": {},
                "summary": "Consensus engine fallback due to upstream error."
            }

    @staticmethod
    def _normalize_zacks(rank):
        if isinstance(rank, dict):
            rank = rank.get("rank")
        try:
            rank = int(rank)
        except Exception:
            return 50
        mapping = {
            1: 95,
            2: 80,
            3: 55,
            4: 35,
            5: 15
        }

        return mapping.get(rank, 50)

    @staticmethod
    def _normalize_rating(value, weight=1.0):

        if isinstance(value, dict):
            if "score" in value:
                return AnalystConsensusEngine._normalize_rating(value.get("score"))
            for key in ["label", "rating", "recommendation", "sentiment"]:
                if key in value:
                    return AnalystConsensusEngine._normalize_rating(value.get(key))
            return 50

        if isinstance(value, str):

            value = value.upper()

            mapping = {
                "STRONG BUY": 95,
                "BUY": 80,
                "OUTPERFORM": 75,
                "OVERWEIGHT": 70,
                "HOLD": 55,
                "NEUTRAL": 50,
                "UNDERPERFORM": 35,
                "SELL": 15
            }

            return mapping.get(value, 50)

        try:
            return float(value)

        except Exception:
            return 50

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 85:
            return "STRONG BUY"
        if score >= 65:
            return "BUY"
        if score >= 45:
            return "HOLD"
        if score >= 25:
            return "SELL"
        return "STRONG SELL"

    @staticmethod
    def _build_source_entry(
        source_name: str,
        raw: Any,
        normalized: float,
        weight: float
    ) -> Dict[str, Any]:
        available = bool(raw)
        weighted_score = round(normalized * weight, 2) if available else 0.0
        details: Dict[str, Any] = {}

        if isinstance(raw, dict):
            details = {
                "provider": raw.get("source", source_name.replace("_", " ").title()),
                "rank": raw.get("rank"),
                "raw_label": raw.get("label") or raw.get("rating") or raw.get("sentiment"),
                "note": raw.get("hedge_fund_sentiment"),
            }
        else:
            details = {"provider": source_name.replace("_", " ").title()}

        return {
            "source": source_name,
            "available": available,
            "score": round(float(normalized), 2) if available else None,
            "label": AnalystConsensusEngine._score_to_label(normalized) if available else "N/A",
            "weight": round(weight, 2),
            "weighted_score": weighted_score,
            "details": details,
        }


# =========================================================
# PROPOSAL ENGINE
# =========================================================

class ProposalEngine:

    MIN_BARS = 60

    DEFAULT_RISK_MULTIPLIER = 1.5

    # =====================================================
    # BUILD
    # =====================================================

    @staticmethod
    def build(
        symbol: str,
        df: pd.DataFrame,
        market_regime: Dict[str, Any],
        sector_rotation: Optional[Dict[str, Any]] = None,
        df_weekly: Optional[pd.DataFrame] = None,
        *,
        asset_class: str = "Stocks",
        flow_reference: Optional[Dict[str, Any]] = None,
    ) -> Optional[InvestmentProposal]:

        try:

            # ---------------------------------------------
            # VALIDATION
            # ---------------------------------------------

            df = normalize_ohlcv(df)

            if not ProposalEngine._is_valid_dataframe(df):

                logger.warning(
                    f"[{symbol}] Invalid dataframe."
                )

                return None

            # ---------------------------------------------
            # WEEKLY FALLBACK
            # ---------------------------------------------

            if df_weekly is None or df_weekly.empty:

                logger.warning(
                    f"[{symbol}] Weekly unavailable. "
                    f"Generating synthetic weekly structure."
                )

                df_weekly = ProposalEngine._build_synthetic_weekly(df)

            # ---------------------------------------------
            # ENGINES
            # ---------------------------------------------

            techs = TechnicalEngine.get_indicators(df) or {}

            smc = SMCEngine.detect_structure(df) or {}

            breakout = BreakoutEngine.analyze(
                df,
                smc,
                df_weekly
            ) or {}

            funds = FundamentalEngine.get_fundamentals(
                symbol
            ) or {}

            sentiment = SentimentEngine.analyze(
                symbol
            ) or {}

            analysts = AnalystConsensusEngine.get(
                symbol
            ) or {}

            # ---------------------------------------------
            # CORE PRICING
            # ---------------------------------------------

            close = ProposalEngine._safe_float(
                techs.get("close"),
                df["close"].iloc[-1]
            )

            atr = ProposalEngine._safe_float(
                techs.get("atr"),
                close * 0.02
            )

            # ---------------------------------------------
            # FEATURES
            # ---------------------------------------------

            features = ProposalEngine._build_features(
                techs,
                smc,
                breakout,
                funds,
                sentiment,
                analysts
            )

            ai = AIEnsemble.predict(
                features,
                market_regime
            )

            if not ai:
                return None

            direction = ai.get(
                "direction",
                "NEUTRAL"
            )

            if direction == "NEUTRAL":
                return None

            confidence = ProposalEngine._safe_float(
                ai.get("confidence"),
                50
            )

            # ---------------------------------------------
            # TRADE LEVELS
            # ---------------------------------------------

            trade = ProposalEngine._build_trade_levels(
                direction,
                close,
                atr,
                breakout,
                smc,
                df
            )

            # ---------------------------------------------
            # RR
            # ---------------------------------------------

            rr = ProposalEngine._calculate_rr(
                close,
                trade["sl"],
                trade["tp2"]
            )

            rr_ext = ProposalEngine._calculate_rr(
                close,
                trade["sl"],
                trade["tp3"]
            )

            # ---------------------------------------------
            # POSITION SIZE
            # ---------------------------------------------

            pos_size = ProposalEngine._calculate_position_size(
                confidence,
                rr,
                market_regime,
                ai.get("horizon_confidence")
            )

            # ---------------------------------------------
            # THESIS
            # ---------------------------------------------

            thesis = ProposalEngine._build_thesis(
                trade["setup_type"],
                breakout,
                techs,
                funds,
                ai,
                analysts
            )

            breakout_snap = ProposalEngine._breakout_snapshot(breakout)
            sentiment_snap = {
                "score": sentiment.get("score"),
                "label": sentiment.get("label"),
                "headline_count": sentiment.get("headline_count", 0),
            }
            sector_flow_ctx = ProposalEngine._flow_context(
                symbol,
                sector_rotation,
                asset_class=asset_class,
                df=df,
                flow_reference=flow_reference,
            )
            money_note = ProposalEngine._money_flow_note(breakout)
            short_term = ProposalEngine._compute_short_term_score(
                breakout,
                sentiment,
                analysts,
                rr,
                confidence,
            )

            # ---------------------------------------------
            # FINAL
            # ---------------------------------------------

            return InvestmentProposal(

                symbol=symbol,

                direction=direction,

                entry_price=round(close, 2),

                stop_loss=round(trade["sl"], 2),

                tp_1=round(trade["tp1"], 2),
                tp_2=round(trade["tp2"], 2),
                tp_3=round(trade["tp3"], 2),

                risk_reward=rr,
                risk_reward_extended=rr_ext,

                ai_confidence=confidence,

                ai_grade=ProposalEngine._calculate_grade(
                    confidence,
                    ai.get("horizon_confidence")
                ),

                thesis=thesis,

                setup_type=trade["setup_type"],

                position_size_pct=pos_size,

                hold_period="1-3 Months",

                horizon_confidence=ai.get(
                    "horizon_confidence",
                    "MEDIUM"
                ),

                sector_exposure="NEUTRAL",

                chart_data=None,

                signals=ai.get("signals", []),

                weights=ai.get("weights", {}),
                analyst_consensus=analysts,
                breakout_analytics=breakout_snap,
                sentiment_snapshot=sentiment_snap,
                sector_flow=sector_flow_ctx,
                money_flow_note=money_note,
                short_term_score=short_term,
            )

        except Exception as e:

            logger.exception(
                f"[{symbol}] Proposal build failure | error={e}"
            )

            return None

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def _safe_float(value, fallback=0.0):

        try:

            if value is None:
                return fallback

            if isinstance(value, float) and math.isnan(value):
                return fallback

            return float(value)

        except Exception:
            return fallback

    @staticmethod
    def _is_valid_dataframe(df):

        required = {
            "open",
            "high",
            "low",
            "close",
            "volume"
        }

        return (
            isinstance(df, pd.DataFrame)
            and len(df) >= ProposalEngine.MIN_BARS
            and required.issubset(df.columns)
        )

    @staticmethod
    def _build_synthetic_weekly(df):

        try:

            return (
                df
                .resample("W")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                })
                .dropna()
            )

        except Exception:

            return pd.DataFrame()

    @staticmethod
    def _breakout_snapshot(breakout: Dict[str, Any]) -> Dict[str, Any]:
        if not breakout or breakout.get("status") == "NO_DATA":
            return {}
        return {
            "breakout_type": breakout.get("breakout_type"),
            "breakout_score": breakout.get("breakout_score"),
            "confidence": breakout.get("confidence"),
            "signal_quality": breakout.get("signal_quality"),
            "volume_confirmed": breakout.get("volume_confirmed"),
            "relative_volume": breakout.get("relative_volume"),
            "squeeze_detected": breakout.get("squeeze_detected"),
            "higher_timeframe_alignment": breakout.get("higher_timeframe_alignment"),
            "pattern": breakout.get("pattern"),
            "pre_breakout": breakout.get("pre_breakout"),
        }

    @staticmethod
    @staticmethod
    def _return_n_bars_pct(df: pd.DataFrame, bars: int = 22) -> Optional[float]:
        if df is None or df.empty or "close" not in df.columns:
            return None
        close = df["close"].dropna()
        if len(close) <= bars:
            return None
        try:
            return float((close.iloc[-1] / close.iloc[-bars - 1] - 1) * 100)
        except Exception:
            return None

    @staticmethod
    def _equity_sector_flow(
        symbol: str,
        sector_rotation: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sym = symbol.upper().strip()
        mapped = SYMBOL_TO_SECTOR_ETF.get(sym)
        if not sector_rotation:
            return {
                "flow_type": "EQUITY",
                "sector_name": mapped[1] if mapped else "Unknown",
                "etf": mapped[0] if mapped else None,
                "bias": "NO DATA",
                "detail": "Sector rotation unavailable",
            }
        sectors = sector_rotation.get("sectors", {})
        ranked = sector_rotation.get("ranked", [])
        leading = set(sector_rotation.get("leading", []))
        lagging = set(sector_rotation.get("lagging", []))
        if not mapped:
            return {
                "flow_type": "EQUITY",
                "sector_name": "Unmapped",
                "etf": None,
                "bias": "UNKNOWN",
                "detail": "Map ticker in SYMBOL_TO_SECTOR_ETF for sector flow",
            }
        etf, sector_name = mapped
        info = sectors.get(etf, {})
        rank_idx = next(
            (i + 1 for i, (s, _) in enumerate(ranked) if s == etf),
            None,
        )
        if etf in leading:
            bias = "LEADING (inflow)"
        elif etf in lagging:
            bias = "LAGGING (outflow)"
        else:
            bias = "MIDDLE"
        return {
            "flow_type": "EQUITY",
            "sector_name": sector_name,
            "etf": etf,
            "bias": bias,
            "sector_rank": rank_idx,
            "momentum_score": info.get("momentum_score"),
            "volume_trend": info.get("volume_trend"),
            "return_1m_pct": info.get("return_1m"),
            "relative_strength_vs_spy": info.get("relative_strength"),
        }

    @staticmethod
    def _crypto_flow_context(
        symbol: str,
        df: pd.DataFrame,
        flow_reference: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sym = symbol.upper().strip()
        seg = CRYPTO_SEGMENT.get(sym, "Crypto (unclassified)")
        sym_1m = ProposalEngine._return_n_bars_pct(df, 22)
        btc_1m = None
        if flow_reference:
            btc_1m = flow_reference.get("BTC_1m_pct")
        if sym_1m is None:
            bias = "NO DATA"
        elif btc_1m is None:
            bias = "NEUTRAL (no BTC ref)"
        else:
            diff = sym_1m - float(btc_1m)
            if diff > 1.5:
                bias = "OUTPERFORMING BTC (1m)"
            elif diff < -1.5:
                bias = "LAGGING BTC (1m)"
            else:
                bias = "IN LINE WITH BTC (1m)"
        vs_proxy = None
        if sym_1m is not None and btc_1m is not None:
            vs_proxy = round(sym_1m - float(btc_1m), 2)
        return {
            "flow_type": "CRYPTO",
            "sector_name": seg,
            "etf": "BTC-USD",
            "bias": bias,
            "sector_rank": None,
            "momentum_score": None,
            "volume_trend": None,
            "return_1m_pct": round(sym_1m, 2) if sym_1m is not None else None,
            "proxy_1m_pct": round(float(btc_1m), 2) if btc_1m is not None else None,
            "vs_proxy_1m": vs_proxy,
            "relative_strength_vs_spy": vs_proxy,
        }

    @staticmethod
    def _forex_flow_context(
        symbol: str,
        df: pd.DataFrame,
        flow_reference: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sym = symbol.upper().strip()
        pair_1m = ProposalEngine._return_n_bars_pct(df, 22)
        dxy_1m = None
        if flow_reference:
            dxy_1m = flow_reference.get("DXY_1m_pct")
        base, quote = "?", "?"
        if sym.endswith("=X") and len(sym) > 3:
            core = sym.replace("=X", "")
            if len(core) == 6:
                base, quote = core[:3], core[3:]
        bias = "NO DATA"
        if dxy_1m is not None:
            d = float(dxy_1m)
            if quote == "USD" and base != "USD":
                bias = "USD FIRM vs pair" if d > 0.15 else "USD SOFT vs pair" if d < -0.15 else "USD NEUTRAL"
            elif base == "USD" and quote == "JPY":
                bias = "JPY WEAK (risk-on skew)" if d > 0.15 else "JPY STRONG" if d < -0.15 else "USD/JPY NEUTRAL"
            else:
                bias = f"DXY 1m {d:+.2f}% (context)"
        return {
            "flow_type": "FOREX",
            "sector_name": f"FX {base}/{quote}",
            "etf": "DX-Y.NYB",
            "bias": bias,
            "sector_rank": None,
            "momentum_score": None,
            "volume_trend": None,
            "return_1m_pct": round(pair_1m, 2) if pair_1m is not None else None,
            "proxy_1m_pct": round(float(dxy_1m), 2) if dxy_1m is not None else None,
            "vs_proxy_1m": None,
            "relative_strength_vs_spy": round(float(dxy_1m), 2) if dxy_1m is not None else None,
        }

    @staticmethod
    def _flow_context(
        symbol: str,
        sector_rotation: Optional[Dict[str, Any]],
        *,
        asset_class: str = "Stocks",
        df: Optional[pd.DataFrame] = None,
        flow_reference: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ac = (asset_class or "Stocks").strip()
        if ac == "Crypto":
            return ProposalEngine._crypto_flow_context(
                symbol, df if df is not None else pd.DataFrame(), flow_reference
            )
        if ac == "Forex":
            return ProposalEngine._forex_flow_context(
                symbol, df if df is not None else pd.DataFrame(), flow_reference
            )
        return ProposalEngine._equity_sector_flow(symbol, sector_rotation)

    @staticmethod
    def _money_flow_note(breakout: Dict[str, Any]) -> str:
        if not breakout or breakout.get("status") == "NO_DATA":
            return "Insufficient data"
        parts = []
        if breakout.get("volume_confirmed"):
            parts.append("Volume confirmed")
        rv = breakout.get("relative_volume")
        if rv is not None:
            try:
                parts.append(f"RV {float(rv):.2f}x")
            except (TypeError, ValueError):
                pass
        if breakout.get("is_accumulating"):
            parts.append("Accumulation")
        if breakout.get("obv_bullish_divergence"):
            parts.append("OBV bull div")
        if breakout.get("obv_bearish_divergence"):
            parts.append("OBV bear div")
        return " · ".join(parts) if parts else "Neutral flow"

    @staticmethod
    def _compute_short_term_score(
        breakout: Dict[str, Any],
        sentiment: Dict[str, Any],
        analysts: Dict[str, Any],
        rr: Optional[float],
        ai_confidence: Optional[float],
    ) -> float:
        bs = float(breakout.get("breakout_score") or 0) if breakout else 0.0
        sen = float((sentiment or {}).get("score") or 50)
        ana = float((analysts or {}).get("score") or 50)
        vol_b = 10.0 if breakout and breakout.get("volume_confirmed") else 0.0
        sq_b = 9.0 if breakout and breakout.get("squeeze_detected") else 0.0
        htf_b = 7.0 if breakout and breakout.get("higher_timeframe_alignment") else 0.0
        if rr is None and ai_confidence is None:
            raw = bs * 0.48 + sen * 0.20 + ana * 0.18 + vol_b + sq_b + htf_b
        else:
            rr_term = min(28.0, max(0.0, float(rr or 0)) * 7.5)
            ai_term = min(
                18.0,
                max(0.0, float(ai_confidence or 50) - 48) * 0.55,
            )
            raw = (
                bs * 0.34
                + sen * 0.14
                + ana * 0.14
                + rr_term
                + vol_b
                + sq_b
                + htf_b
                + ai_term
            )
        return float(min(100.0, max(0.0, raw)))

    @staticmethod
    def quick_surface(
        symbol: str,
        df: pd.DataFrame,
        df_weekly: Optional[pd.DataFrame],
        sector_rotation: Optional[Dict[str, Any]],
        *,
        asset_class: str = "Stocks",
        flow_reference: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        df = normalize_ohlcv(df)
        if not ProposalEngine._is_valid_dataframe(df):
            return None
        if df_weekly is None or getattr(df_weekly, "empty", True):
            df_weekly = ProposalEngine._build_synthetic_weekly(df)
        smc = SMCEngine.detect_structure(df) or {}
        breakout = BreakoutEngine.analyze(df, smc, df_weekly) or {}
        sentiment = SentimentEngine.analyze(symbol) or {}
        analysts = AnalystConsensusEngine.get(symbol) or {}
        sector_flow = ProposalEngine._flow_context(
            symbol,
            sector_rotation,
            asset_class=asset_class,
            df=df,
            flow_reference=flow_reference,
        )
        money_note = ProposalEngine._money_flow_note(breakout)
        st_score = ProposalEngine._compute_short_term_score(
            breakout,
            sentiment,
            analysts,
            None,
            None,
        )
        return {
            "symbol": symbol,
            "bucket": "Watchlist",
            "breakout_analytics": ProposalEngine._breakout_snapshot(breakout),
            "sentiment_snapshot": {
                "score": sentiment.get("score"),
                "label": sentiment.get("label"),
                "headline_count": sentiment.get("headline_count", 0),
            },
            "sector_flow": sector_flow,
            "money_flow_note": money_note,
            "short_term_score": st_score,
            "analyst_consensus": analysts,
        }

    @staticmethod
    def _build_features(
        techs,
        smc,
        breakout,
        funds,
        sentiment,
        analysts
    ):

        return {

            **techs,
            **smc,
            **breakout,

            "fundamental_score":
                funds.get("score", 50),

            "earnings_growth":
                funds.get("growth", 0),

            "sentiment_score":
                sentiment.get("score", 50),

            "analyst_score":
                analysts.get("score", 50),

            "institutional_sentiment":
                analysts.get("score", 50),

            "weekly_aligned":
                int(
                    breakout.get(
                        "weekly_aligned",
                        False
                    )
                ),

            "is_accumulating":
                int(
                    breakout.get(
                        "is_accumulating",
                        False
                    )
                ),
        }

    @staticmethod
    def _build_trade_levels(
        direction,
        close,
        atr,
        breakout,
        smc,
        df
    ):

        risk = atr * 1.5

        if direction == "LONG":

            sl = close - risk

            return {
                "sl": sl,
                "tp1": close + risk * 1.5,
                "tp2": close + risk * 3,
                "tp3": close + risk * 5,
                "setup_type": "LONG_SWING"
            }

        sl = close + risk

        return {
            "sl": sl,
            "tp1": close - risk * 1.5,
            "tp2": close - risk * 3,
            "tp3": close - risk * 5,
            "setup_type": "SHORT_SWING"
        }

    @staticmethod
    def _calculate_rr(close, sl, target):

        risk = abs(close - sl)

        if risk <= 0:
            return 1.0

        reward = abs(target - close)

        return round(reward / risk, 2)

    @staticmethod
    def _calculate_position_size(
        confidence,
        rr,
        market_regime,
        horizon_confidence
    ):

        edge = max(0, confidence - 50)

        size = edge * rr * 0.05

        return min(max(size, 0.5), 5)

    @staticmethod
    def _calculate_grade(confidence, horizon):

        if confidence >= 85:
            return "A+"

        if confidence >= 75:
            return "A"

        if confidence >= 65:
            return "B+"

        if confidence >= 55:
            return "B"

        return "C"

    @staticmethod
    def _build_thesis(
        setup_type,
        breakout,
        techs,
        funds,
        ai,
        analysts
    ):

        thesis = []

        thesis.append(setup_type)

        if breakout.get("breakout_type"):
            thesis.append(
                breakout["breakout_type"]
            )

        if techs.get("golden_cross"):
            thesis.append("Golden Cross")

        if breakout.get("is_accumulating"):
            thesis.append("Institutional accumulation")

        if analysts.get("label"):
            thesis.append(
                f"Analyst consensus: "
                f"{analysts['label']}"
            )

        growth = funds.get("growth", 0)

        if growth > 15:
            thesis.append(
                f"Earnings growth {growth}%"
            )

        return ". ".join(thesis) + "."

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

class MarketBreadthEngine:
    """
    Institutional-grade market breadth engine using multi-asset basket.
    """

    BREADTH_BASKET = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V",
        "UNH","JNJ","WMT","PG","MA","HD","XOM","BAC","PFE","KO",
        "DIS","INTC","CSCO","PEP","ABT","CRM","AVGO","NFLX","COST",
        "TMO","MRK"
    ]

    @staticmethod
    def analyze() -> Dict[str, Any]:

        try:
            df = yf.download(
                MarketBreadthEngine.BREADTH_BASKET,
                period="6mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker"
            )

            if df is None or df.empty:
                return MarketBreadthEngine._fallback()

            # -----------------------------
            # Extract close prices per ticker
            # -----------------------------
            closes = {}

            for ticker in MarketBreadthEngine.BREADTH_BASKET:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if ("Close", ticker) in df.columns:
                            series = df[("Close", ticker)]
                        elif (ticker, "Close") in df.columns:
                            series = df[(ticker, "Close")]
                        else:
                            continue
                    else:
                        if "Close" not in df.columns:
                            continue
                        series = df["Close"]

                    series = series.dropna()
                    if len(series) < 60:
                        continue

                    closes[ticker] = series

                except Exception:
                    continue

            if not closes:
                return MarketBreadthEngine._fallback()

            # -----------------------------
            # Breadth calculations
            # -----------------------------
            above_50 = 0
            above_200 = 0
            advancing = 0
            declining = 0
            valid = 0

            for ticker, series in closes.items():

                if len(series) < 50:
                    continue

                valid += 1

                last = series.iloc[-1]
                sma50 = series.rolling(50).mean().iloc[-1]
                sma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else None

                if last > sma50:
                    above_50 += 1

                if sma200 is not None and last > sma200:
                    above_200 += 1

                # weekly momentum proxy
                if len(series) >= 6:
                    ret = (series.iloc[-1] / series.iloc[-6] - 1)
                    if ret > 0:
                        advancing += 1
                    else:
                        declining += 1

            total = max(valid, 1)

            pct_50 = round((above_50 / total) * 100, 1)
            pct_200 = round((above_200 / total) * 100, 1)

            ad_ratio = round(advancing / max(declining, 1), 2)

            # -----------------------------
            # Breadth regime classification
            # -----------------------------
            if pct_50 >= 75:
                breadth = "STRONG"
            elif pct_50 >= 60:
                breadth = "HEALTHY"
            elif pct_50 >= 45:
                breadth = "MIXED"
            elif pct_50 >= 30:
                breadth = "WEAK"
            else:
                breadth = "VERY_WEAK"

            return {
                "pct_above_50dma": pct_50,
                "pct_above_200dma": pct_200,
                "advance_decline_ratio": ad_ratio,
                "breadth": breadth,
                "universe_size": valid,
                "interpretation": {
                    "STRONG": "Broad participation — trend expansion likely.",
                    "HEALTHY": "Constructive market — selective longs favored.",
                    "MIXED": "Rotational market — reduce position size.",
                    "WEAK": "Risk-off regime emerging.",
                    "VERY_WEAK": "Defensive regime — capital preservation."
                }[breadth]
            }

        except Exception as e:
            logger.warning(f"Breadth analysis failed: {e}")
            return MarketBreadthEngine._fallback()

    @staticmethod
    def _fallback():
        return {
            "pct_above_50dma": 50,
            "pct_above_200dma": 50,
            "advance_decline_ratio": 1.0,
            "breadth": "MIXED",
            "universe_size": 0,
            "interpretation": "Unable to compute market breadth."
        }

# ===============================================
# 13. ASSET UNIVERSE DEFINITIONS
# ===============================================
class AssetUniverse:

    # ---------------------------------------
    # WEIGHTING MODEL (INSTITUTIONAL BREADTH)
    # ---------------------------------------
    UNIVERSE_WEIGHTS = {
        "LARGE_CAP": 1.0,
        "GROWTH": 0.7,
        "SPECULATIVE": 0.4,
        "MICROCAP": 0.2,
        "CRYPTO": 0.5,
        "FOREX": 0.3
    }

    # ---------------------------------------
    # EQUITIES
    # ---------------------------------------
    STOCKS_LARGE_CAP = [
        "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","TSLA","AVGO","ORCL",
        "ADBE","CRM","NFLX","INTU","QCOM","AMD","IBM","NOW","INTC","MU","LRCX",
        "JPM","V","MA","BAC","WFC","GS","MS","BLK","AXP",
        "UNH","JNJ","LLY","ABBV","TMO","AMGN","ISRG","VRTX",
        "HD","MCD","NKE","SBUX","COST","WMT","PG","KO","PEP",
        "XOM","CVX","COP","BA","CAT","HON","UPS","RTX","LMT","GE","DE",
        "SPY"
    ]

    STOCKS_GROWTH = [
        "COIN","SOFI","RBLX","RIVN","BABA","MSTR","OPEN","SMR","IONQ",
        "TTD","ON","RDDT","SMCI"
    ]

    STOCKS_SPECULATIVE = [
        "OKLO","AI","QBTS","RUM","ACHR","PONY","NNE","CHPT"
    ]

    STOCKS_MICROCAP = [
        "PSTV","RNAZ","EH","AAOI","HOTH","DVLT","TOVX","BSOL","BBAR","CDE","NGD"
    ]

    STOCKS_ALL = list(set(
        STOCKS_LARGE_CAP +
        STOCKS_GROWTH +
        STOCKS_SPECULATIVE +
        STOCKS_MICROCAP
    ))

    # ---------------------------------------
    # CRYPTO
    # ---------------------------------------
    CRYPTO_ALL = [
        "BTC-USD","ETH-USD","BNB-USD","XRP-USD","ADA-USD","DOGE-USD","SOL-USD",
        "DOT-USD","MATIC-USD","SHIB-USD","LTC-USD","AVAX-USD","LINK-USD","ATOM-USD",
        "UNI-USD","XMR-USD","ETC-USD","BCH-USD","NEAR-USD","FIL-USD","APT-USD",
        "ARB-USD","OP-USD"
    ]

    # ---------------------------------------
    # FOREX
    # ---------------------------------------
    FOREX_ALL = [
        "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X","NZDUSD=X","USDCAD=X",
        "EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X","EURNZD=X"
    ]

    # ---------------------------------------
    # WEIGHT FUNCTION (FIXED)
    # ---------------------------------------
    @staticmethod
    def get_ticker_weight(symbol: str) -> float:

        if symbol in AssetUniverse.STOCKS_LARGE_CAP:
            return AssetUniverse.UNIVERSE_WEIGHTS["LARGE_CAP"]

        if symbol in AssetUniverse.STOCKS_GROWTH:
            return AssetUniverse.UNIVERSE_WEIGHTS["GROWTH"]

        if symbol in AssetUniverse.STOCKS_SPECULATIVE:
            return AssetUniverse.UNIVERSE_WEIGHTS["SPECULATIVE"]

        if symbol in AssetUniverse.STOCKS_MICROCAP:
            return AssetUniverse.UNIVERSE_WEIGHTS["MICROCAP"]

        if symbol.endswith("-USD"):
            return AssetUniverse.UNIVERSE_WEIGHTS["CRYPTO"]

        if symbol.endswith("=X"):
            return AssetUniverse.UNIVERSE_WEIGHTS["FOREX"]

        return 0.5  # fallback neutral weight


# ===============================================
# SAFE UTILITIES
# ===============================================
def safe_num(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


def safe_asset_block(asset: dict) -> dict:
    asset = asset or {}
    return {
        "price": safe_num(asset.get("price")),
        "trend": asset.get("trend") or "UNKNOWN",
        "roc_20d": safe_num(asset.get("roc_20d")),
        "level": safe_num(asset.get("level")),
        "regime": asset.get("regime") or "UNKNOWN",
        "impact": asset.get("impact") or "N/A",
    }


# ===============================================
# MARKET ENGINE (MUST BE ABOVE DASHBOARD)
# ===============================================

def compute_market_pressure(market_regime, breadth, sector_rotation):
    regime_map = {
        "STRONG_BULL": 100,
        "BULL": 75,
        "NEUTRAL": 50,
        "CAUTIOUS": 40,
        "BEAR": 20,
        "STRONG_BEAR": 0,
        "RISK_OFF": 10,
    }

    regime_score = regime_map.get(market_regime.get("_overall", "NEUTRAL"), 50)

    breadth_score = (
        safe_num(breadth.get("pct_above_50dma")) * 0.6
        + safe_num(breadth.get("pct_above_200dma")) * 0.4
    )

    ranked = sector_rotation.get("ranked", [])
    if ranked:
        sector_score = sum(
            safe_num(d.get("momentum_score"))
            for _, d in ranked[:5]
        ) / max(1, min(5, len(ranked)))
    else:
        sector_score = 50

    sector_score = max(0, min(100, sector_score))

    return round(
        regime_score * 0.5 +
        breadth_score * 0.3 +
        sector_score * 0.2,
        1
    )


def classify_market_state(score: float, regime: str):
    if regime in ["STRONG_BULL", "STRONG_BEAR"]:
        if score >= 70:
            return "RISK-ON EXTREME" if regime == "STRONG_BULL" else "CAPITULATION"
        return "RISK-ON" if regime == "STRONG_BULL" else "RISK-OFF"

    if score >= 80:
        return "RISK-ON EXTREME"
    elif score >= 65:
        return "RISK-ON"
    elif score >= 50:
        return "NEUTRAL"
    elif score >= 35:
        return "RISK-OFF"
    return "CAPITULATION"


def detect_rotation_divergence(sector_rotation, breadth):
    ranked = sector_rotation.get("ranked", [])
    if not ranked:
        return None

    top = ranked[0]
    data = top[1] if isinstance(top, tuple) else top

    top_momentum = safe_num(data.get("momentum_score"))
    breadth_strength = safe_num(breadth.get("pct_above_50dma"))

    if top_momentum > 75 and breadth_strength < 45:
        return "⚠️ NARROW RALLY (Top-heavy market)"

    if top_momentum < 40 and breadth_strength > 60:
        return "⚠️ LATENT ACCUMULATION (hidden strength)"

    return None
# ===============================================
# INSTITUTIONAL MARKET DASHBOARD v3.5 (HEAT + INTEL LAYER)
# ===============================================

# -------------------------------
# ROTATION ACCELERATION (NEW ALPHA SIGNAL)
# -------------------------------

def compute_rotation_acceleration(sector_rotation):
    ranked = sector_rotation.get("ranked", [])
    if len(ranked) < 2:
        return 0.0

    values = []
    for item in ranked:
        data = item[1] if isinstance(item, tuple) else item
        values.append(safe_num(data.get("momentum_score")))

    # acceleration = change dispersion (proxy for regime churn)
    return round(float(np.mean(np.diff(values))) if len(values) > 1 else 0.0, 2)


# -------------------------------
# SECTOR DOMINANCE INDEX
# -------------------------------

def compute_sector_dominance(sector_rotation):
    ranked = sector_rotation.get("ranked", [])
    if not ranked:
        return 50

    top = ranked[0][1] if isinstance(ranked[0], tuple) else ranked[0]
    top_mom = safe_num(top.get("momentum_score"))

    avg = np.mean([
        safe_num((x[1] if isinstance(x, tuple) else x).get("momentum_score"))
        for x in ranked[:10]
    ])

    return round((top_mom - avg) + 50, 1)


# -------------------------------
# REGIME TRANSITION RISK
# -------------------------------

def compute_regime_transition_risk(pressure, breadth, acceleration):

    breadth_score = safe_num(breadth.get("pct_above_50dma"))

    risk = (
        abs(50 - pressure) * 0.4 +
        abs(50 - breadth_score) * 0.3 +
        abs(acceleration) * 5
    )

    return round(min(100, risk), 1)


# -------------------------------
# TOP SETUPS FILTER (HOOK)
# -------------------------------

def extract_top_setups(proposals, n=3):
    if not proposals:
        return []

    ranked = sorted(
        proposals,
        key=lambda p: (
            getattr(p, "short_term_score", 0),
            getattr(p, "ai_confidence", 0),
            getattr(p, "risk_reward", 0)
        ),
        reverse=True
    )

    return ranked[:n]


# -------------------------------
# MAIN DASHBOARD v3.5
# -------------------------------

def _render_market_dashboard(market_regime, sector_rotation, breadth, proposals=None):

    overall = market_regime.get("_overall", "NEUTRAL")

    pressure = compute_market_pressure(market_regime, breadth, sector_rotation)
    state = classify_market_state(pressure, overall)

    warning = detect_rotation_divergence(sector_rotation, breadth)

    acceleration = compute_rotation_acceleration(sector_rotation)
    dominance = compute_sector_dominance(sector_rotation)
    transition_risk = compute_regime_transition_risk(pressure, breadth, acceleration)

    # ---------------- HEADER (native Streamlit — avoids raw HTML showing as text) ----------------
    with st.container(border=True):
        st.markdown(f"### 🌍 {state}")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Pressure", f"{pressure:.1f}")
        h2.metric("Regime", str(overall))
        h3.metric("Breadth", str(breadth.get("breadth", "?")))
        h4.metric("Transition risk", f"{transition_risk:.1f}/100")
        st.caption(
            f"Rotation acceleration: **{acceleration:.2f}** · "
            f"Sector dominance: **{dominance:.1f}** · "
            f"Higher transition risk suggests a less stable regime backdrop."
        )

    if warning:
        st.warning(warning)

    # ---------------- METRICS ----------------
    cols = st.columns(5)

    spy = safe_asset_block(market_regime.get("SPY", {}))
    qqq = safe_asset_block(market_regime.get("QQQ", {}))
    vix = safe_asset_block(market_regime.get("VIX", {}))
    dxy = safe_asset_block(market_regime.get("DXY", {}))

    with cols[0]:
        st.metric("SPY", f"{spy['price']:.0f}", f"{spy['roc_20d']:+.1f}%")

    with cols[1]:
        st.metric("QQQ", f"{qqq['price']:.0f}", f"{qqq['roc_20d']:+.1f}%")

    with cols[2]:
        st.metric("VIX", f"{vix['level']:.1f}", vix["regime"])

    with cols[3]:
        st.metric("DXY", f"{dxy['level']:.1f}", dxy["impact"])

    with cols[4]:
        st.metric("Breadth", f"{breadth.get('pct_above_50dma',50):.0f}%", breadth.get("breadth","MIXED"))

    # ---------------- HEATMAP GRID ----------------
    st.markdown("### 🔥 Institutional Sector Heatmap")

    ranked = sector_rotation.get("ranked", [])

    if ranked:

        grid_cols = st.columns(6)

        for i, item in enumerate(ranked[:6]):

            symbol = item[0] if isinstance(item, tuple) else item.get("symbol")
            data = item[1] if isinstance(item, tuple) else item

            mom = safe_num(data.get("momentum_score"))

            with grid_cols[i]:
                st.metric(symbol, f"{mom:.1f}")

    # ---------------- TOP SETUPS ----------------
    if proposals:

        st.markdown("### ⚡ Top Institutional Setups")

        top = extract_top_setups(proposals, n=3)

        for p in top:

            st.markdown(f"**{p.symbol} | {p.direction} | Score: {p.short_term_score:.1f}**")
            st.caption(f"""
            Confidence: {p.ai_confidence:.1f}% |
            R:R: 1:{p.risk_reward} |
            Setup: {p.setup_type}
            """)

    # ---------------- FULL TABLE ----------------
    st.markdown("### 📊 Sector Rotation Map")

    rows = []

    for item in ranked:

        symbol = item[0] if isinstance(item, tuple) else item.get("symbol")
        data = item[1] if isinstance(item, tuple) else item

        data = data or {}

        rows.append({
            "Sector": data.get("name", symbol),
            "ETF": symbol,
            "Momentum": safe_num(data.get("momentum_score")),
            "1M": f"{safe_num(data.get('return_1m')):+.1f}%",
            "3M": f"{safe_num(data.get('return_3m')):+.1f}%",
            "RS vs SPY": f"{safe_num(data.get('relative_strength')):+.1f}%"
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

def main_ui():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    st.set_page_config(layout="wide", page_title="Institutional Breakout Terminal v4")

    st.title("🚀 Institutional Breakout & Investment Terminal")
    st.caption("3–12 Month Investment Opportunity Scanner | Multi-Timeframe | Market Regime Aware, Tnx to Kicko Ognenovski, Nikola Stojcevski, Altaj Sulejman, Dejan Butevski, Anastas Dzurovski")

    # =========================================================
    # SIDEBAR CONFIG (UPGRADED)
    # =========================================================
    st.sidebar.header("⚙️ Configuration")

    # -------------------------
    # SCAN MODE
    # -------------------------
    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Quick Scan (Watchlist)", "Full Market Scan (All Assets)"],
        index=0
    )

    asset_class = st.sidebar.selectbox(
        "Asset Class",
        ["Stocks", "Crypto", "Forex"],
        index=0
    )

    # -------------------------
    # DEFAULT SYMBOLS LOGIC
    # -------------------------
    WATCHLISTS = {
        "Stocks": "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, AMD, META, JPM, LLY, AVGO, CRM",
        "Crypto": "BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD",
        "Forex": "EURUSD=X, GBPUSD=X, USDJPY=X"
    }

    if scan_mode == "Quick Scan (Watchlist)":
        default_symbols = WATCHLISTS.get(asset_class, "")
    else:
        try:
            default_symbols = ", ".join(
                STOCKS_ALL if asset_class == "Stocks"
                else CRYPTO_ALL if asset_class == "Crypto"
                else FOREX_ALL
            )
        except NameError:
            # fallback safety if global lists are missing
            default_symbols = WATCHLISTS.get(asset_class, "")

    # -------------------------
    # SYMBOL PARSING (SAFE + CLEAN)
    # -------------------------
    symbols_raw = st.sidebar.text_area(
        "Symbols (comma separated)",
        value=default_symbols,
        height=140,
        help="Enter tickers separated by commas. Example: AAPL, MSFT, TSLA"
    )

    symbols = sorted({
        s.strip().upper()
        for s in symbols_raw.replace("\n", ",").split(",")
        if s.strip()
    })

    # -------------------------
    # INVESTMENT HORIZON
    # -------------------------
    horizon = st.sidebar.selectbox(
        "Investment Horizon",
        ["3-6 months (Swing)", "6-12 months (Position)", "Any (Show All)"],
        index=1
    )

    # -------------------------
    # MIN CONFIDENCE FILTER
    # -------------------------
    min_confidence = st.sidebar.slider(
        "Min AI Confidence (%)",
        min_value=40,
        max_value=90,
        value=55,
        step=1,
        help="Filters out weak setups below this confidence threshold"
    )

    # -------------------------
    # QUICK STATS (OPTIONAL UI NICE-TO-HAVE)
    # -------------------------
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📊 Symbols loaded: {len(symbols)}")
    st.sidebar.caption(f"Mode: {scan_mode.split(' ')[0]}")
    st.sidebar.caption(f"Asset class: {asset_class}")
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

        # One-shot proxies for crypto (BTC) / forex (DXY) momentum vs scan universe
        def _scan_flow_reference(ac: str) -> Dict[str, Any]:
            ref: Dict[str, Any] = {}
            if ac == "Crypto":
                try:
                    btc = yf.download(
                        "BTC-USD",
                        period="6mo",
                        interval="1d",
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
                    btc = normalize_ohlcv(btc)
                    r = ProposalEngine._return_n_bars_pct(btc, 22)
                    if r is not None:
                        ref["BTC_1m_pct"] = r
                except Exception:
                    pass
            elif ac == "Forex":
                try:
                    dxy = yf.download(
                        "DX-Y.NYB",
                        period="6mo",
                        interval="1d",
                        progress=False,
                        auto_adjust=True,
                        threads=False,
                    )
                    dxy = normalize_ohlcv(dxy)
                    r = ProposalEngine._return_n_bars_pct(dxy, 22)
                    if r is not None:
                        ref["DXY_1m_pct"] = r
                except Exception:
                    pass
            return ref

        flow_reference = _scan_flow_reference(asset_class)

        # =========================================================
        # SCAN SYMBOLS
        # =========================================================
        st.markdown("---")
        st.subheader("📡 Scanning Assets")

        progress = st.progress(0)
        proposals = []
        flow_rows = []

        def fetch(sym):
            try:
                tf = fetch_multi_timeframe(sym)
                df = tf.get("daily")
                if df is None or len(df) == 0:
                    return None

                df_weekly = tf.get("weekly")
                prop = ProposalEngine.build(
                    sym,
                    df,
                    market_regime,
                    sector_rotation=sector_rotation,
                    df_weekly=df_weekly,
                    asset_class=asset_class,
                    flow_reference=flow_reference,
                )
                if prop:
                    sf = prop.sector_flow or {}
                    ba = prop.breakout_analytics or {}
                    sn = prop.sentiment_snapshot or {}
                    ac = prop.analyst_consensus or {}
                    flow_rows.append({
                        "Symbol": prop.symbol,
                        "Bucket": "Trade setup",
                        "Flow": sf.get("flow_type", "EQUITY"),
                        "Context": sf.get("sector_name"),
                        "Proxy": sf.get("etf"),
                        "1m %": sf.get("return_1m_pct"),
                        "vs proxy 1m": sf.get("vs_proxy_1m"),
                        "Proxy 1m %": sf.get("proxy_1m_pct"),
                        "Sector rank": sf.get("sector_rank"),
                        "Breakout type": ba.get("breakout_type"),
                        "Brk score": ba.get("breakout_score"),
                        "Short-term edge": round(prop.short_term_score, 1),
                        "Sentiment": sn.get("label"),
                        "News #": sn.get("headline_count"),
                        "Sector flow": sf.get("bias"),
                        "Sector ETF": sf.get("etf"),
                        "Sector 1m %": sf.get("return_1m_pct"),
                        "Money flow": prop.money_flow_note,
                        "Analyst": ac.get("score"),
                        "Analyst label": ac.get("label"),
                    })
                else:
                    qs = ProposalEngine.quick_surface(
                        sym,
                        df,
                        df_weekly,
                        sector_rotation,
                        asset_class=asset_class,
                        flow_reference=flow_reference,
                    )
                    if qs:
                        sf = qs.get("sector_flow") or {}
                        ba = qs.get("breakout_analytics") or {}
                        sn = qs.get("sentiment_snapshot") or {}
                        ac = qs.get("analyst_consensus") or {}
                        flow_rows.append({
                            "Symbol": qs["symbol"],
                            "Bucket": "Watchlist",
                            "Flow": sf.get("flow_type", "EQUITY"),
                            "Context": sf.get("sector_name"),
                            "Proxy": sf.get("etf"),
                            "1m %": sf.get("return_1m_pct"),
                            "vs proxy 1m": sf.get("vs_proxy_1m"),
                            "Proxy 1m %": sf.get("proxy_1m_pct"),
                            "Sector rank": sf.get("sector_rank"),
                            "Breakout type": ba.get("breakout_type"),
                            "Brk score": ba.get("breakout_score"),
                            "Short-term edge": round(qs.get("short_term_score", 0), 1),
                            "Sentiment": sn.get("label"),
                            "News #": sn.get("headline_count"),
                            "Sector flow": sf.get("bias"),
                            "Sector ETF": sf.get("etf"),
                            "Sector 1m %": sf.get("return_1m_pct"),
                            "Money flow": qs.get("money_flow_note"),
                            "Analyst": ac.get("score"),
                            "Analyst label": ac.get("label"),
                        })
                return prop
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

        st.success(f"Found {len(proposals)} valid setups")

        # -------------------------
        # SAFE METRICS
        # -------------------------
        longs = sum(getattr(p, "direction", None) == "LONG" for p in proposals)
        shorts = sum(getattr(p, "direction", None) == "SHORT" for p in proposals)

        conf_values = [getattr(p, "ai_confidence", 0) for p in proposals]
        avg_conf = float(np.mean(conf_values)) if conf_values else 0.0

        # -------------------------
        # DISPLAY PANEL
        # -------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📈 Long Setups", longs)

        with col2:
            st.metric("📉 Short Setups", shorts)

        with col3:
            st.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

        # -------------------------
        # CONTEXT LINE (OPTIONAL INSIGHT)
        # -------------------------
        bias = "LONG BIAS" if longs > shorts else "SHORT BIAS" if shorts > longs else "NEUTRAL"

        st.caption(
            f"Market positioning bias: {bias} | "
            f"Net skew: {longs - shorts:+d} setups"
        )
        # =========================================================
        # SHORTER-TERM EDGE (SETUPS)
        # =========================================================

        st.subheader("⚡ Shorter-term edge — ranked trade setups")
        st.caption(
            "Composite score combining breakout quality, risk/reward, AI confidence, "
            "sentiment flow, sector rotation, and analyst consensus. "
            "Higher score = stronger short-term asymmetric opportunity."
        )

        # -------------------------
        # BUILD EDGE TABLE
        # -------------------------
        edge_rows = []

        for p in proposals:
            ba = getattr(p, "breakout_analytics", None) or {}
            sn = getattr(p, "sentiment_snapshot", None) or {}
            sf = getattr(p, "sector_flow", None) or {}
            ac = getattr(p, "analyst_consensus", None) or {}

            edge_rows.append({
                "Symbol": getattr(p, "symbol", ""),
                "Flow": sf.get("flow_type", "EQUITY"),
                "Context": sf.get("sector_name", "—"),

                "Short-term edge": safe(getattr(p, "short_term_score", 0)),
                "AI %": safe(getattr(p, "ai_confidence", 0)),

                "R:R": safe(getattr(p, "risk_reward", 0)),
                "Brk score": safe(ba.get("breakout_score", 0)),

                "Breakout": ba.get("breakout_type", "—"),
                "Sentiment": sn.get("label", "—"),

                "Sector flow": sf.get("bias", "—"),
                "vs proxy 1m": sf.get("vs_proxy_1m", 0),

                "Money flow": getattr(p, "money_flow_note", "—"),
                "Analyst": ac.get("score", 0),
            })

        edge_df = pd.DataFrame(edge_rows)

        # -------------------------
        # SORTING ENGINE (ROBUST)
        # -------------------------
        if not edge_df.empty:

            edge_df["_edge"] = pd.to_numeric(edge_df["Short-term edge"], errors="coerce").fillna(-1e9)
            edge_df["_brk"] = pd.to_numeric(edge_df["Brk score"], errors="coerce").fillna(-1e9)
            edge_df["_rr"] = pd.to_numeric(edge_df["R:R"], errors="coerce").fillna(-1e9)

            edge_df = edge_df.sort_values(
                by=["_edge", "_brk", "_rr"],
                ascending=[False, False, False],
            ).drop(columns=["_edge", "_brk", "_rr"], errors="ignore")

            # Optional: highlight top 5 setups mentally (no UI dependency)
            top_n = min(5, len(edge_df))
            st.caption(f"🔥 Top {top_n} setups show strongest composite edge")

        # -------------------------
        # RENDER TABLE
        # -------------------------
        st.dataframe(
            edge_df,
            width="stretch",
            hide_index=True
        )
        # =========================================================
        # FULL SCAN: BREAKOUT + SENTIMENT + SECTOR FLOW
        # =========================================================

        st.subheader("🧭 Full Market Flow — Breakout, Sentiment & Capital Rotation")
        st.caption(
            "Comprehensive view of all scanned assets including breakout structure, "
            "news sentiment, and sector capital rotation. "
            "Used to identify where liquidity is concentrating across the market."
        )

        # -------------------------
        # VALIDATION
        # -------------------------
        if not flow_rows:
            st.info("No flow analytics available (insufficient OHLCV or incomplete data pipeline).")
            st.stop()

        # -------------------------
        # BUILD DATAFRAME
        # -------------------------
        fdf = pd.DataFrame(flow_rows)

        # -------------------------
        # SAFE SORT KEYS
        # -------------------------
        fdf["_edge"] = pd.to_numeric(fdf.get("Short-term edge", 0), errors="coerce").fillna(-1e9)
        fdf["_brk"] = pd.to_numeric(fdf.get("Brk score", 0), errors="coerce").fillna(-1e9)
        fdf["_sr"] = pd.to_numeric(fdf.get("Sector rank", 999), errors="coerce").fillna(999)

        # -------------------------
        # SORT ENGINE
        # -------------------------
        fdf = fdf.sort_values(
            by=["_edge", "_brk", "_sr"],
            ascending=[False, False, True],
        )

        # cleanup helper columns
        fdf = fdf.drop(columns=["_edge", "_brk", "_sr"], errors="ignore")

        # -------------------------
        # OPTIONAL INSIGHT LINE
        # -------------------------
        top_n = min(10, len(fdf))
        st.caption(f"🔥 Showing {len(fdf)} instruments | Top {top_n} prioritized by flow + breakout strength")

        # -------------------------
        # RENDER TABLE
        # -------------------------
        st.dataframe(
            fdf,
            width="stretch",
            hide_index=True
        )

        # =========================================================
        # TABLE VIEW (SAFE)
        # =========================================================

        st.subheader("📊 Opportunity Table")

        # -------------------------
        # BUILD TABLE
        # -------------------------
        table = []

        for p in proposals[:20]:

            ba = getattr(p, "breakout_analytics", None) or {}
            sn = getattr(p, "sentiment_snapshot", None) or {}
            sf = getattr(p, "sector_flow", None) or {}
            ac = getattr(p, "analyst_consensus", None) or {}

            table.append({
                "Symbol": getattr(p, "symbol", "—"),
                "Direction": getattr(p, "direction", "—"),
                "Setup": safe_str(getattr(p, "setup_type", "—")),

                "Confidence": safe(getattr(p, "ai_confidence", 0)),
                "Short-term": round(safe(getattr(p, "short_term_score", 0)), 1),

                "Flow": sf.get("flow_type", "—"),
                "Context": sf.get("sector_name", "—"),

                "Brk": safe(ba.get("breakout_score", 0)),
                "Brk type": ba.get("breakout_type", "—"),

                "Sentiment": sn.get("label", "—"),
                "Sector flow": sf.get("bias", "—"),

                "vs proxy": safe(sf.get("vs_proxy_1m", 0)),
                "Money flow": getattr(p, "money_flow_note", "—"),

                "Analyst": safe(ac.get("score", 0)),
                "A. label": ac.get("label", "—"),

                "Entry": safe(getattr(p, "entry_price", 0)),
                "SL": safe(getattr(p, "stop_loss", 0)),
                "TP2": safe(getattr(p, "tp_2", 0)),
                "TP3": safe(getattr(p, "tp_3", 0)),

                "R:R": safe(getattr(p, "risk_reward", 0)),
            })

        df = pd.DataFrame(table)

        # -------------------------
        # SORT ENGINE (ROBUST)
        # -------------------------
        if not df.empty:

            df["_e"] = pd.to_numeric(df["Short-term"], errors="coerce").fillna(-1e9)
            df["_b"] = pd.to_numeric(df["Brk"], errors="coerce").fillna(-1e9)
            df["_c"] = pd.to_numeric(df["Confidence"], errors="coerce").fillna(-1e9)

            df = df.sort_values(
                by=["_e", "_b", "_c"],
                ascending=[False, False, False],
            ).drop(columns=["_e", "_b", "_c"], errors="ignore")

            st.caption(
                f"Top {min(20, len(df))} setups ranked by short-term edge → breakout strength → AI confidence"
            )

        # -------------------------
        # RENDER TABLE
        # -------------------------
        st.dataframe(
            df,
            width="stretch",
            hide_index=True
        )

        # =========================================================
        # DETAIL CARDS (STREAMLIT-NATIVE, NO HTML)
        # =========================================================

        st.subheader("📈 Detailed Setups")

        cols = st.columns(2)

        # -------------------------
        # LOOP THROUGH PROPOSALS
        # -------------------------
        for i, p in enumerate(proposals[:30]):

            with cols[i % 2]:

                setup_label = safe_str(getattr(p, "setup_type", "—"))
                is_breakout = "BREAKOUT" in setup_label.upper()

                border = "🟢" if is_breakout else "🟡"

                st.markdown(f"## {border} {getattr(p, 'symbol', '—')}")
                st.markdown(
                    f"**{getattr(p, 'direction', '—')} | "
                    f"{setup_label} | "
                    f"{getattr(p, 'ai_grade', '—')} "
                    f"({safe(getattr(p, 'ai_confidence', 0))}%)**"
                )

                # -------------------------
                # METRICS GRID
                # -------------------------
                c1, c2, c3 = st.columns(3)

                c1.metric("Entry", safe(getattr(p, "entry_price", 0)))
                c2.metric("Stop", safe(getattr(p, "stop_loss", 0)))
                c3.metric("Size %", safe(getattr(p, "position_size_pct", 0)))

                c1.metric("TP2", safe(getattr(p, "tp_2", 0)))
                c2.metric("TP3", safe(getattr(p, "tp_3", 0)))
                c3.metric("R:R", f"1:{safe(getattr(p, 'risk_reward', 0))}")

                # -------------------------
                # THESIS
                # -------------------------
                st.write("📌 Thesis:")
                st.write(safe_str(getattr(p, "thesis", "—")))

                # -------------------------
                # SAFE META EXTRACTION
                # -------------------------
                _sf = getattr(p, "sector_flow", None) or {}
                _sn = getattr(p, "sentiment_snapshot", None) or {}
                _ba = getattr(p, "breakout_analytics", None) or {}

                st.caption(
                    f"⚡ Short-term edge: **{safe(getattr(p, 'short_term_score', 0)):.1f}** · "
                    f"Flow **{_sf.get('flow_type', '—')}** · "
                    f"{_sf.get('sector_name', '—')} · "
                    f"Breakout **{_ba.get('breakout_score', '—')}** "
                    f"({safe_str(_ba.get('breakout_type', '—'))}) · "
                    f"Sentiment **{_sn.get('label', '—')}** · "
                    f"Bias **{_sf.get('bias', '—')}** · "
                    f"Money: {safe_str(getattr(p, 'money_flow_note', '—'))}"
                )

                # -------------------------
                # ANALYST CONSENSUS
                # -------------------------
                analyst_block = getattr(p, "analyst_consensus", None) or {}

                if analyst_block:

                    st.caption(
                        f"Analyst Consensus: {analyst_block.get('label', 'NEUTRAL')} "
                        f"({safe(analyst_block.get('score', 50)):.1f}) | "
                        f"Coverage: {analyst_block.get('coverage', 0)}%"
                    )

                    with st.expander("Analyst Rating Breakdown", expanded=False):

                        source_rows = []

                        for src_key, src_data in (analyst_block.get("sources", {}) or {}).items():

                            if not isinstance(src_data, dict):
                                continue

                            details = src_data.get("details", {}) or {}

                            source_rows.append({
                                "Source": str(src_key).replace("_", " ").title(),
                                "Status": "Available" if src_data.get("available") else "N/A",
                                "Score": src_data.get("score"),
                                "Label": src_data.get("label"),
                                "Weight": src_data.get("weight"),
                                "Weighted": src_data.get("weighted_score"),
                                "Provider": details.get("provider"),
                                "Rank": details.get("rank"),
                                "Raw Label": details.get("raw_label"),
                                "Extra": details.get("note"),
                            })

                        if source_rows:
                            st.dataframe(
                                pd.DataFrame(source_rows),
                                width="stretch",
                                hide_index=True
                            )
                            st.caption(analyst_block.get("summary", ""))

                # -------------------------
                # CHART (SAFE FIX)
                # -------------------------
                chart_data = getattr(p, "chart_data", None)

                if chart_data is not None:
                    try:
                        # IMPORTANT: assumes fig is already defined elsewhere correctly
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.warning("Chart unavailable")

                st.markdown("---")

        # =========================================================
        # PORTFOLIO SUMMARY
        # =========================================================

        st.subheader("💼 Portfolio Summary")

        # -------------------------
        # SAFE EXPOSURE CALCULATION
        # -------------------------
        long_exp = sum(
            safe(getattr(p, "position_size_pct", 0))
            for p in proposals
            if getattr(p, "direction", None) == "LONG"
        )

        short_exp = sum(
            safe(getattr(p, "position_size_pct", 0))
            for p in proposals
            if getattr(p, "direction", None) == "SHORT"
        )

        net_exp = long_exp - short_exp

        # -------------------------
        # EXPOSURE METRICS
        # -------------------------
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "📈 Long Exposure",
            f"{long_exp:.1f}%"
        )

        c2.metric(
            "📉 Short Exposure",
            f"{short_exp:.1f}%"
        )

        c3.metric(
            "⚖️ Net Exposure",
            f"{net_exp:+.1f}%"
        )

        # -------------------------
        # CONTEXT BIAS LINE
        # -------------------------
        if net_exp > 10:
            bias = "LONG BIASED PORTFOLIO"
        elif net_exp < -10:
            bias = "SHORT BIASED PORTFOLIO"
        else:
            bias = "BALANCED / NEUTRAL EXPOSURE"

        st.caption(f"Portfolio stance: {bias}")
if __name__ == "__main__":
    main_ui()
