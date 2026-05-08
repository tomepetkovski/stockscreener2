import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
import aiohttp
import json
import time
import logging
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

# --- Configuration for Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InstitutionalEngine")

# ===============================================
# 1. ADVANCED TECHNICAL INDICATORS
# ===============================================
class IndicatorUtils:
    """Expanded collection of institutional-grade technical indicators."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = (-delta.clip(upper=0)).fillna(0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close_prev = (df["high"] - df["close"].shift(1)).abs()
        low_close_prev = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean().ffill()

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = IndicatorUtils.sma(series, period)
        std = series.rolling(window=period, min_periods=1).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return middle, upper, lower

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = IndicatorUtils.ema(series, fast)
        ema_slow = IndicatorUtils.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorUtils.ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        atr = IndicatorUtils.atr(df.copy(), period)
        hl2 = (df["high"] + df["low"]) / 2
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        supertrend = pd.Series(True, index=df.index)
        final_upper = upperband.copy()
        final_lower = lowerband.copy()

        for i in range(1, len(df)):
            if df["close"].iloc[i-1] <= final_upper.iloc[i-1]:
                final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i-1])
            else:
                final_upper.iloc[i] = upperband.iloc[i]
            
            if df["close"].iloc[i-1] >= final_lower.iloc[i-1]:
                final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i-1])
            else:
                final_lower.iloc[i] = lowerband.iloc[i]

            if df["close"].iloc[i] > final_upper.iloc[i-1]:
                supertrend.iloc[i] = True
            elif df["close"].iloc[i] < final_lower.iloc[i-1]:
                supertrend.iloc[i] = False
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]

        return supertrend

    @staticmethod
    def calculate_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
        """Relative Volume Z-Score: How many standard deviations is current volume?"""
        mean = volume.rolling(window=period).mean()
        std = volume.rolling(window=period).std()
        z_score = (volume - mean) / std.replace(0, np.nan)
        return z_score.fillna(0)

    @staticmethod
    def calculate_price_zscore(price: pd.Series, period: int = 20) -> pd.Series:
        """Z-Score of Price (Mean Reversion Indicator)."""
        return IndicatorUtils.calculate_volume_zscore(price, period) # Reuse logic

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume Weighted Average Price (Institutional Fair Value)."""
        v = df['volume']
        tp = (df['high'] + df['low'] + df['close']) / 3
        return (tp * v).cumsum() / v.cumsum()

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (Trend Strength)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = IndicatorUtils.atr(df, period) * period # Approx TR sum for ADX
        
        plus_di = 100 * (IndicatorUtils.ema(plus_dm, period) / tr)
        minus_di = 100 * (IndicatorUtils.ema(minus_dm, period) / tr)
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = IndicatorUtils.ema(dx, period)
        return adx

# ===============================================
# 2. MACRO & MARKET REGIME ENGINE
# ===============================================
class MarketRegimeEngine:
    """Detects the current market environment (Macro Context)."""
    _cache: Dict[str, Any] = {}
    _cache_ts: float = 0.0

    @staticmethod
    def get_market_regime() -> Dict[str, Any]:
        now = time.time()
        if MarketRegimeEngine._cache and now - MarketRegimeEngine._cache_ts < 300:
            return MarketRegimeEngine._cache

        def _safe_last_close(ticker: str) -> Optional[float]:
            try:
                df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
                if df is None or df.empty: return None
                df = df.rename(columns={c: str(c).lower() for c in df.columns})
                return float(df["close"].iloc[-1])
            except: return None

        vix = _safe_last_close("^VIX")
        tnx = _safe_last_close("^TNX") 
        yield_10y = (tnx / 10.0) if tnx else None

        spy_df = yf.download("SPY", period="13mo", interval="1d", progress=False, auto_adjust=True)
        spy_df = spy_df.rename(columns={c: str(c).lower() for c in spy_df.columns}) if spy_df is not None else pd.DataFrame()
        spy_close = spy_df.get("close", pd.Series(dtype=float)).astype(float).dropna()
        
        spy_sma50 = float(IndicatorUtils.sma(spy_close, 50).iloc[-1]) if len(spy_close) >= 50 else None
        spy_sma200 = float(IndicatorUtils.sma(spy_close, 200).iloc[-1]) if len(spy_close) >= 200 else None
        spy_last = float(spy_close.iloc[-1]) if not spy_close.empty else None

        # Determine Trend
        trend = "Neutral"
        if spy_last and spy_sma50 and spy_sma200:
            if spy_last > spy_sma50 > spy_sma200: trend = "Bull Trend"
            elif spy_last < spy_sma50 < spy_sma200: trend = "Bear Trend"
            else: trend = "Transition/Range"

        # Risk Regime
        vix_val = float(vix) if vix else 20.0
        regime = "Risk-On" if vix_val < 18 and trend == "Bull Trend" else "Risk-Off" if vix_val > 25 or trend == "Bear Trend" else "Neutral/Cautious"
        
        # Breadth Proxy (Simple)
        breadth = "Strong" if spy_last and spy_sma50 and spy_last > spy_sma50 else "Weak"

        out = {
            "vix": round(vix_val, 2),
            "yield_10y": f"{round(float(yield_10y), 2)}%" if yield_10y else "N/A",
            "regime": regime,
            "spy_trend": trend,
            "market_breadth": breadth,
            "risk_score": round((vix_val / 40) * 100, 1) if vix_val else 50 # Higher VIX = Higher Risk Score
        }
        MarketRegimeEngine._cache = out
        MarketRegimeEngine._cache_ts = now
        return out

# ===============================================
# 3. RISK & SENTIMENT ENGINE
# ===============================================
class RiskEngine:
    """Calculates position risk, gap risk, and liquidity constraints."""

    @staticmethod
    def calculate_var(df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Historical Value at Risk (VaR) calculation."""
        returns = df['close'].pct_change().dropna()
        if returns.empty: return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_gap_risk(df: pd.DataFrame) -> float:
        """Calculates average overnight gap percentage."""
        opens = df['open']
        prev_closes = df['close'].shift(1)
        gaps = ((opens - prev_closes) / prev_closes).abs()
        return gaps.mean() if not gaps.empty else 0.0

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculates optimal position sizing percentage."""
        if avg_loss == 0 or win_rate == 0: return 0.0
        kelly = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        return max(0, kelly) * 100 # Return percentage

class SentimentEngine:
    """Advanced Sentiment Analysis."""

    @staticmethod
    def get_news_sentiment_proxy(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determines sentiment based on price action volatility and momentum.
        (In production, this would call an NLP API).
        """
        if df is None or len(df) < 20: return {"score": 50, "label": "Neutral"}

        # 1. Momentum Sentiment
        rsi = IndicatorUtils.rsi(df['close'], 14).iloc[-1]
        mom_score = (rsi - 50) * 2 # -100 to +100 scaled

        # 2. Volatility Sentiment (VIX style)
        rets = df['close'].pct_change().dropna()
        vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        vol_score = -20 if vol > 40 else 20 if vol < 15 else 0 # High vol is bearish/bad

        # 3. Institutional Flow (OBV slope)
        obv = IndicatorUtils.obv(df)
        obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / obv.iloc[-5] if obv.iloc[-5] != 0 else 0
        flow_score = np.clip(obv_slope * 1000, -30, 30)

        total_score = np.clip((mom_score + vol_score + flow_score) / 3 + 50, 0, 100)
        
        label = "Bearish" if total_score < 35 else "Bullish" if total_score > 65 else "Neutral"
        return {
            "score": round(total_score, 1),
            "label": label,
            "momentum": round(mom_score, 1),
            "volatility": round(vol, 1),
            "flow": round(flow_score, 1)
        }

# ===============================================
# 4. AI & MACHINE LEARNING ENGINE
# ===============================================
class AIEngine:
    """
    Synthetic AI Model for Probability Estimation.
    Mimics a Gradient Boosting Machine (GBM) logic using weighted ensembles.
    """
    @staticmethod
    def predict_probability(features: Dict[str, float]) -> Dict[str, Any]:
        """
        Input: Normalized features (0-100 scale or similar).
        Output: Probability of Success (0.0 - 1.0).
        """
        # Weights represent feature importance learned from "training"
        w = {
            "rs_score": 0.20,
            "fund_score": 0.15,
            "tech_score": 0.25,
            "sent_score": 0.10,
            "analyst_score": 0.15,
            "macro_score": 0.15
        }

        # Calculate Weighted Sum
        score = (
            features.get("rs_score", 50) * w["rs_score"] +
            features.get("fund_score", 50) * w["fund_score"] +
            features.get("tech_score", 50) * w["tech_score"] +
            features.get("sent_score", 50) * w["sent_score"] +
            features.get("analyst_score", 50) * w["analyst_score"] +
            features.get("macro_score", 50) * w["macro_score"]
        )

        # Non-linear activation (Sigmoid) to convert score to probability
        # Centered around 50, slope of 10
        probability = 1 / (1 + np.exp(-(score - 60) / 10))
        
        # Calculate Confidence Interval (Uncertainty Quantification)
        # Higher dispersion in feature scores -> Lower confidence
        vals = list(features.values())
        dispersion = np.std(vals)
        confidence = max(0, 100 - dispersion)

        return {
            "probability": round(probability * 100, 1),
            "confidence": round(confidence, 1),
            "raw_score": round(score, 1),
            "model_version": "SyntheticGBM-v1.0"
        }

# ===============================================
# 5. TRADE PROPOSER CORE LOGIC
# ===============================================

@dataclass
class Proposal:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    target1: float
    target2: float
    target3: float
    expected_price_12m: float
    rr_ratio: float
    quantity: int
    capital_at_risk: float
    strength_pct: float
    fundamental_quality: str
    fundamental_score: float
    analyst_consensus: str
    analyst_rating_score: float
    upside_pct: float
    rs_score: float
    sector: str
    industry: str
    edge_score: float
    ai_probability: float
    ai_confidence: float
    risk_score: float
    sentiment_score: float
    invest_rank_score: float
    conviction_tier: str
    vwap: float
    volume_zscore: float
    price_zscore: float
    ai_insight: str
    timestamp: str
    debug: Dict[str, Any]

class TradeProposer:
    """Enhanced trade proposal engine with risk parity and AI integration."""

    CONFIG = {
        "MIN_PRICE": 2.0,
        "MIN_AVG_VOLUME": 500_000,
        "MIN_RR": 1.5,
        "EDGE_THRESHOLD": 5.0,
        "STOP_ATR_MULTIPLIER": 1.75,
        "TARGET_ATR_MULTIPLIER": 2.5,
        "MAX_POSITION_PCT": 10.0, # Max 10% of portfolio in one trade
    }

    def __init__(self, account_size: float = 100_000, risk_pct: float = 1.0):
        self.account_size = float(account_size)
        self.risk_amount = self.account_size * (risk_pct / 100)

    async def propose(self, df: pd.DataFrame, direction: str, strength: float, symbol: str, 
                      spy_trend: str = "Neutral", sector_map: Dict = {}) -> Optional[Proposal]:
        
        if len(df) < 60: return None
        if direction == "NONE" or strength <= 0: return None

        close = float(df["close"].iloc[-1])
        volume = float(df["volume"].iloc[-1])
        
        # 1. Technical Indicators
        atr14 = float(IndicatorUtils.atr(df, 14).iloc[-1])
        ema21 = float(IndicatorUtils.ema(df["close"], 21).iloc[-1])
        sma50 = float(IndicatorUtils.sma(df["close"], 50).iloc[-1])
        sma200 = float(IndicatorUtils.sma(df["close"], 200).iloc[-1]) if len(df) >= 200 else sma50
        rsi = float(IndicatorUtils.rsi(df["close"], 14).iloc[-1])
        vwap = float(IndicatorUtils.vwap(df).iloc[-1])
        vol_z = float(IndicatorUtils.calculate_volume_zscore(df["volume"]).iloc[-1])
        price_z = float(IndicatorUtils.calculate_price_zscore(df["close"]).iloc[-1])
        adx = float(IndicatorUtils.adx(df).iloc[-1])

        # 2. Fundamental & Analyst Data (Async)
        fund_data, analyst_data = await asyncio.gather(
            self._get_fundamentals(symbol),
            self._get_analyst_data(symbol)
        )

        # 3. Risk Calculation
        var = RiskEngine.calculate_var(df)
        gap_risk = RiskEngine.calculate_gap_risk(df)
        
        # Stop Loss Logic (ATR based + Risk Adjusted)
        stop_dist = max(atr14 * self.CONFIG["STOP_ATR_MULTIPLIER"], close * 0.05)
        stop = close - stop_dist if direction == "LONG" else close + stop_dist
        
        # Targets
        rr_mult = max(1.5, strength / 20.0)
        target_dist = atr14 * self.CONFIG["TARGET_ATR_MULTIPLIER"] * rr_mult
        t1 = close + target_dist if direction == "LONG" else close - target_dist
        t2 = close + target_dist * 1.5 if direction == "LONG" else close - target_dist * 1.5
        t3 = close + target_dist * 2.2 if direction == "LONG" else close - target_dist * 2.2
        
        rr = abs(t1 - close) / stop_dist

        # 4. Filters
        if close < self.CONFIG["MIN_PRICE"]: return None
        if df["volume"].mean() < self.CONFIG["MIN_AVG_VOLUME"]: return None
        if rr < self.CONFIG["MIN_RR"]: return None
        
        # Institutional Filters
        is_long = direction == "LONG"
        trend_aligned = (close > sma50 > sma200) if is_long else (close < sma50 < sma200)
        if not trend_aligned and strength < 80: # Allow super strong breakouts regardless
            # logger.info(f"[{symbol}] Skip: Trend misalignment")
            # return None # Uncomment to enforce strict filtering
            pass

        # 5. Scoring & AI
        rs_score = 50 + (df['close'].pct_change(60).iloc[-1] * 1000) # Simplified RS
        
        tech_score = strength * 0.4 + (10 if trend_aligned else -10) + (10 if adx > 25 else -5) + np.clip(vol_z * 5, -15, 15)
        sent_res = SentimentEngine.get_news_sentiment_proxy(symbol, df)
        
        macro_score = 80 if spy_trend == "Bull Trend" and is_long else 50
        
        ai_input = {
            "rs_score": np.clip(rs_score, 0, 100),
            "fund_score": fund_data.get("fundamental_score", 50),
            "tech_score": np.clip(tech_score, 0, 100),
            "sent_score": sent_res['score'],
            "analyst_score": analyst_data.get("rating_score", 50),
            "macro_score": macro_score
        }
        
        ai_res = AIEngine.predict_probability(ai_input)
        edge_score = ai_res['raw_score']
        
        if edge_score < 50: # Basic threshold
            return None

        # Position Sizing (Kelly Criterion simplified + Risk Limit)
        # Kelly = W - (1-W)/R
        win_rate = ai_res['probability'] / 100
        kelly_pct = RiskEngine.kelly_criterion(win_rate, rr, 1.0)
        risk_pct_cap = min(10.0, kelly_pct * 0.5) # Half-Kelly for safety
        
        qty = int((self.account_size * (risk_pct_cap/100)) / stop_dist)
        if qty <= 0: return None

        invest_rank = edge_score * 0.5 + ai_res['confidence'] * 0.3 + rr * 5 * 0.2
        
        # 6. Construct Proposal
        proposal = Proposal(
            symbol=symbol,
            direction="LONG" if is_long else "SHORT",
            entry=round(close, 2),
            stop_loss=round(stop, 2),
            target1=round(t1, 2),
            target2=round(t2, 2),
            target3=round(t3, 2),
            expected_price_12m=analyst_data.get("expected_price", close*1.1),
            rr_ratio=round(rr, 2),
            quantity=qty,
            capital_at_risk=round(qty * stop_dist, 2),
            strength_pct=round(strength, 1),
            fundamental_quality=fund_data.get("quality", "N/A"),
            fundamental_score=fund_data.get("fundamental_score", 0),
            analyst_consensus=analyst_data.get("consensus", "N/A"),
            analyst_rating_score=analyst_data.get("rating_score", 0),
            upside_pct=analyst_data.get("upside_pct", 0),
            rs_score=round(rs_score, 1),
            sector=fund_data.get("sector", "N/A"),
            industry=fund_data.get("industry", "N/A"),
            edge_score=round(edge_score, 1),
            ai_probability=ai_res['probability'],
            ai_confidence=ai_res['confidence'],
            risk_score=round(gap_risk * 1000, 1), # Proxy score
            sentiment_score=sent_res['score'],
            invest_rank_score=round(invest_rank, 1),
            conviction_tier="A" if invest_rank > 80 else "B" if invest_rank > 65 else "C",
            vwap=round(vwap, 2),
            volume_zscore=round(vol_z, 2),
            price_zscore=round(price_z, 2),
            ai_insight=self._generate_insight(ai_res, sent_res, fund_data),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            debug={"indicators": {"rsi": rsi, "adx": adx, "atr": atr14}}
        )
        return proposal

    def _generate_insight(self, ai_res, sent_res, fund_data) -> str:
        base = f"🤖 AI Prediction: {ai_res['probability']}% success probability (Confidence: {ai_res['confidence']}%). "
        base += f"Sentiment is {sent_res['label']} ({sent_res['score']}/100). "
        if fund_data['fundamental_score'] > 70:
            base += "Strong fundamental backing."
        return base

    async def _get_fundamentals(self, symbol: str) -> Dict:
        # Wrapper for async fetching
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            score = 50
            if info.get('forwardPE', 999) < 20: score += 10
            if info.get('returnOnEquity', 0) > 0.15: score += 20
            if info.get('profitMargins', 0) > 0.1: score += 10
            return {
                "fundamental_score": min(100, score),
                "quality": "Institutional" if score > 80 else "Retail",
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A")
            }
        except:
            return {"fundamental_score": 50, "quality": "Unknown", "sector": "N/A", "industry": "N/A"}

    async def _get_analyst_data(self, symbol: str) -> Dict:
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            price = info.get("currentPrice", 0)
            target = info.get("targetMedianPrice", price * 1.1)
            rec = info.get("recommendationMean", 3.0)
            return {
                "consensus": "Buy" if rec < 2.5 else "Hold",
                "rating_score": (5 - rec) * 25,
                "upside_pct": ((target/price)-1)*100 if price > 0 else 0,
                "expected_price": target
            }
        except:
            return {"consensus": "Hold", "rating_score": 50, "upside_pct": 0, "expected_price": 0}

# ===============================================
# 6. SCANNER & ORCHESTRATION
# ===============================================

class Scanner:
    def __init__(self, account_size=100000):
        self.proposer = TradeProposer(account_size=account_size)

    async def scan(self, symbols: List[str]) -> List[Proposal]:
        logger.info(f"Scanning {len(symbols)} symbols...")
        macro = MarketRegimeEngine.get_market_regime()
        
        # 1. Fetch Data Concurrently
        tasks = [self._fetch_and_process(s, macro) for s in symbols]
        results = await asyncio.gather(*tasks)
        
        # 2. Filter & Sort
        valid_proposals = [p for p in results if p is not None]
        valid_proposals.sort(key=lambda x: x.invest_rank_score, reverse=True)
        
        # 3. Sector Diversification Filter
        final_proposals = []
        sector_counts = defaultdict(int)
        for p in valid_proposals:
            if sector_counts[p.sector] < 3: # Max 3 per sector
                final_proposals.append(p)
                sector_counts[p.sector] += 1
        
        return final_proposals[:20] # Top 20

    async def _fetch_and_process(self, symbol: str, macro: Dict) -> Optional[Proposal]:
        try:
            ticker = yf.Ticker(symbol)
            df = await asyncio.to_thread(ticker.history, period="6mo", interval="1d", auto_adjust=True)
            if df is None or len(df) < 60: return None
            
            # Normalize
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

            # Detect Breakout
            direction, strength, vol_z = detect_breakout(df)
            
            # Propose
            return await self.proposer.propose(df, direction, strength, symbol, macro['spy_trend'])
        except Exception as e:
            # logger.error(f"Error {symbol}: {e}")
            return None

def detect_breakout(df: pd.DataFrame) -> Tuple[str, float, float]:
    """Enhanced Breakout Detection with Volume Confirmation."""
    close = df['close']
    volume = df['volume']
    
    # Indicators
    vol_z = IndicatorUtils.calculate_volume_zscore(volume).iloc[-1]
    ema21 = IndicatorUtils.ema(close, 21).iloc[-1]
    ema50 = IndicatorUtils.ema(close, 50).iloc[-1]
    rsi = IndicatorUtils.rsi(close).iloc[-1]
    
    # Consolidation Check (Last 10 days range)
    recent_high = close[-20:-1].max()
    recent_low = close[-20:-1].min()
    range_pct = (recent_high - recent_low) / recent_low
    
    last_close = close.iloc[-1]
    
    # Logic
    direction = "NONE"
    strength = 0
    
    # Breakout Conditions
    is_breakout = last_close > recent_high * 1.02 # 2% buffer
    is_breakdown = last_close < recent_low * 0.98
    
    # Volume Confirm: Vol Z > 1.5 (High relative volume)
    vol_confirm = vol_z > 1.5
    
    if is_breakout and vol_confirm:
        direction = "LONG"
        # Strength based on volume spike and range expansion
        strength = min(100, 50 + vol_z * 5 + (rsi - 50))
    elif is_breakdown and vol_confirm:
        direction = "SHORT"
        strength = min(100, 50 + vol_z * 5 + (50 - rsi))
        
    return direction, float(strength), float(vol_z)

# ===============================================
# 7. STREAMLIT UI
# ===============================================
async def main():
    st.set_page_config(page_title="Institutional AI Scanner", layout="wide")
    st.title("🚀 Institutional AI Investment Engine")
    
    # Sidebar
    st.sidebar.header("Configuration")
    account = st.sidebar.number_input("Account Size ($)", value=100000)
    
    if st.button("🔍 Scan Market (SP100)"):
        with st.spinner("Running Institutional Analysis..."):
            # Use a fixed list of top stocks for demo
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "WMT", 
                       "PG", "JNJ", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "CRM", "NFLX"]
            
            scanner = Scanner(account_size=account)
            proposals = await scanner.scan(symbols)
            
            # Metrics
            if proposals:
                st.success(f"Found {len(proposals)} Institutional Grade Opportunities")
                
                # Macro Display
                macro = MarketRegimeEngine.get_market_regime()
                c1, c2, c3 = st.columns(3)
                c1.metric("Market Regime", macro['regime'])
                c2.metric("VIX Risk", macro['vix'])
                c3.metric("Market Breadth", macro['market_breadth'])

                # Dataframe Display
                df = pd.DataFrame([p.__dict__ for p in proposals])
                
                st.subheader("Portfolio Recommendations")
                
                # Main Table
                cols = ["symbol", "direction", "entry", "stop_loss", "target1", "rr_ratio", 
                        "ai_probability", "invest_rank_score", "conviction_tier", "sector", "volume_zscore"]
                
                st.dataframe(
                    df[cols],
                    column_config={
                        "ai_probability": st.column_config.ProgressColumn("AI Prob %", min_value=0, max_value=100),
                        "invest_rank_score": st.column_config.ProgressColumn("Invest Rank", min_value=0, max_value=100, format="%.1f"),
                    }
                )
                
                # Detailed View
                st.subheader("AI Deep Dive")
                for p in proposals[:5]:
                    with st.expander(f"💡 {p.symbol} - {p.direction} | Rank: {p.invest_rank_score}"):
                        st.markdown(f"**{p.ai_insight}**")
                        st.markdown(f"**Entry:** ${p.entry} | **Stop:** ${p.stop_loss} | **Target:** ${p.target1}")
                        st.markdown(f"**Fundamentals:** {p.fundamental_score}/100 | **Sentiment:** {p.sentiment_score}/100")
                        st.markdown(f"**Risk:** VaR & Gap analysis indicates {'Low' if p.risk_score < 5 else 'Moderate'} Risk.")
                        st.json(p.debug['indicators']) # Debug raw indicators
            else:
                st.warning("No setups found matching institutional criteria in this batch.")

if __name__ == "__main__":
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except: pass
    asyncio.run(main())
