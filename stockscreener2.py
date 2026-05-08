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
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
# --- Configuration for Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FinanceEngine")
STOCKS_ALL = []
SKIP_SYMBOLS = set()
def fetch_and_build(symbol):
    return symbol
# ===============================================
# 1. TECHNICAL INDICATORS
# ===============================================
class IndicatorUtils:
    """Collection of high-performance technical indicators."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average (SMA), ensuring min_periods=1 for early data."""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average (EMA)."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index (RSI) using SMA smoothing."""
        delta = series.diff()
        gain = delta.clip(lower=0).fillna(0)
        loss = (-delta.clip(upper=0)).fillna(0)
        
        # Calculate Average Gain and Average Loss (using SMA)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range (ATR). Requires 'high', 'low', and 'close' columns."""
        if any(col not in df.columns for col in ['high', 'low', 'close']):
             raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns for ATR.")
        
        high_low = df["high"] - df["low"]
        high_close_prev = (df["high"] - df["close"].shift(1)).abs()
        low_close_prev = (df["low"] - df["close"].shift(1)).abs()
        
        # True Range is the maximum of the three components
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # ATR is the SMA of the True Range
        return tr.rolling(window=period, min_periods=1).mean().ffill()

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (middle_band, upper_band, lower_band)"""
        middle = IndicatorUtils.sma(series, period)
        std = series.rolling(window=period, min_periods=1).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return middle, upper, lower

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (macd_line, signal_line, histogram)"""
        ema_fast = IndicatorUtils.ema(series, fast)
        ema_slow = IndicatorUtils.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorUtils.ema(macd_line, signal)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """
        Calculates the Supertrend indicator (True=Uptrend, False=Downtrend).
        Uses a loop (as in the original code), which is less efficient but correct.
        """
        atr = IndicatorUtils.atr(df.copy(), period)
        hl2 = (df["high"] + df["low"]) / 2
        upperband = hl2 + multiplier * atr
        lowerband = hl2 - multiplier * atr

        supertrend = pd.Series(True, index=df.index)
        final_upper = upperband.copy()
        final_lower = lowerband.copy()

        for i in range(1, len(df)):
            # Original iterative logic:
            final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i-1]) if df["close"].iloc[i-1] <= final_upper.iloc[i-1] else upperband.iloc[i]
            final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i-1]) if df["close"].iloc[i-1] >= final_lower.iloc[i-1] else lowerband.iloc[i]

            if df["close"].iloc[i] > final_upper.iloc[i-1]:
                supertrend.iloc[i] = True
            elif df["close"].iloc[i] < final_lower.iloc[i-1]:
                supertrend.iloc[i] = False
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]

        return supertrend

    @staticmethod
    def calculate_volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculates the Z-Score of volume relative to its moving average."""
        mean = volume.rolling(window=period).mean()
        std = volume.rolling(window=period).std()
        z_score = (volume - mean) / std.replace(0, np.nan)
        return z_score.fillna(0)

# ===============================================
# 2. NEWS + FUNDAMENTAL SCORING ENGINE
# ===============================================
class MarketSentimentEngine:
    """Sentiment and fundamental scoring engine."""

    @staticmethod
    async def get_news_sentiment(symbol: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Build a deterministic sentiment proxy from price/volume behavior.
        This is safer than using random values when no reliable news feed is wired in.
        """
        await asyncio.sleep(0)
        if df is None or len(df) < 20:
            return {
                "sentiment_score": 0.0,
                "news_impact": "Neutral",
                "headlines_today": 0
            }

        close = df["close"].astype(float)
        volume = df["volume"].astype(float)
        ret_5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0.0
        ret_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) > 20 else ret_5
        vol_z = float(IndicatorUtils.calculate_volume_zscore(volume).iloc[-1])

        score = np.clip(ret_5 * 1.4 + ret_20 * 0.6 + vol_z * 3.5, -25, 25)
        impact = "Neutral"
        if score > 15: impact = "Strong Bullish"
        elif score > 7: impact = "Bullish"
        elif score < -12: impact = "Strong Bearish"
        elif score < -7: impact = "Bearish"

        return {
            "sentiment_score": round(score, 1),
            "news_impact": impact,
            "headlines_today": int(max(0, round(abs(vol_z))))
        }

    @staticmethod
    def get_fundamental_score(info: Dict) -> Dict[str, Any]:
        """Calculates a fundamental quality score based on institutional quality metrics."""
        score = 0
        
        # 1. Growth & Profitability
        roe = info.get("returnOnEquity") or 0
        growth = info.get("earningsQuarterlyGrowth") or 0
        profit_margin = info.get("profitMargins") or 0
        
        if growth > 0.2: score += 20
        if roe > 0.15: score += 15
        if profit_margin > 0.10: score += 10
        
        # 2. Safety & Debt
        debt_to_equity = info.get("debtToEquity") or 999
        current_ratio = info.get("currentRatio") or 0
        
        if debt_to_equity < 80: score += 15
        if current_ratio > 1.5: score += 10
        
        # 3. Valuation (Middle-Long Term)
        forward_pe = info.get("forwardPE") or 999
        peg_ratio = info.get("pegRatio") or 999
        
        if 5 < forward_pe < 25: score += 15
        if 0 < peg_ratio < 1.5: score += 15
        
        # 4. Dividends (Stability)
        dividend_yield = info.get("dividendYield") or 0
        if dividend_yield > 0.02: score += 10

        quality = "Institutional" if score >= 80 else "Strong" if score >= 60 else "Average" if score >= 40 else "Speculative"
        return {
            "fundamental_score": min(score, 100),
            "quality": quality,
            "roe": round(roe * 100, 1),
            "pe": round(forward_pe, 1)
        }

class AnalystAggregator:
    """Aggregates analyst ratings and price targets from Yahoo/Nasdaq proxies."""
    
    @staticmethod
    async def get_analyst_outlook(symbol: str) -> Dict[str, Any]:
        """Fetches analyst consensus and price target upside."""
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            
            current_price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            target_median = info.get("targetMedianPrice") or 0
            upside = ((target_median / current_price) - 1) * 100 if current_price > 0 and target_median > 0 else 0
            
            rec_mean = info.get("recommendationMean") or 3.0
            # Scale: 1 is Strong Buy, 5 is Sell
            consensus = "Strong Buy" if rec_mean <= 1.5 else "Buy" if rec_mean <= 2.5 else "Hold" if rec_mean <= 3.5 else "Underperform"
            
            rating_score = (5 - rec_mean) * 25 # Convert 1-5 scale to 0-100 score
            
            # Expanded Institutional Logic (Mocking specific source weights based on consensus)
            zacks = "3-Hold"
            if rating_score >= 80: zacks = "1-Strong Buy"
            elif rating_score >= 65: zacks = "2-Buy"
            elif rating_score <= 30: zacks = "5-Strong Sell"
            
            mf = "Neutral"
            if upside > 25 and rating_score > 70: mf = "Top Pick"
            elif upside > 15: mf = "Starter Stock"
            
            mb_score = f"{rating_score/20:.1f}/5.0"

            return {
                "consensus": consensus,
                "upside_pct": round(upside, 1),
                "rating_score": round(rating_score, 1),
                "expected_price": round(target_median, 2),
                "expected_price_3m": round(current_price + (target_median - current_price) * 0.40, 2) if current_price > 0 and target_median > 0 else 0.0,
                "expected_price_12m": round(target_median, 2) if target_median > 0 else 0.0,
                "nasdaq_consensus": consensus,  # Proxy label aligned with Yahoo recommendation mean
                "zacks_rank": zacks,
                "motley_fool_view": mf,
                "marketbeat_score": mb_score,
                "analyst_count": info.get("numberOfAnalystOpinions") or 0
            }
        except Exception:
            return {
                "consensus": "Neutral", "upside_pct": 0, "rating_score": 50, 
                "expected_price": 0, "expected_price_3m": 0.0, "expected_price_12m": 0.0,
                "nasdaq_consensus": "Neutral", "zacks_rank": "3-Hold", "motley_fool_view": "Neutral",
                "marketbeat_score": "2.5/5.0", "analyst_count": 0
            }

# ===============================================
# 3. MACHINE LEARNING PLACEHOLDER
# ===============================================
class MLScoringEngine:
    """Predictive ML engine for trade scoring."""
    def predict_score(self, features: Dict[str, Any]) -> float:
        """Returns a non-linear prediction score based on multi-factor input."""
        # Simulated Neural Network Weights
        w_tech = 0.35
        w_fund = 0.30
        w_sent = 0.20
        w_rs = 0.15
        
        base = features.get("rs_score", 50) * w_rs + \
               features.get("fund_score", 50) * w_fund + \
               features.get("analyst_score", 50) * w_sent + \
               (features.get("strength", 0) * 0.5) * w_tech
               
        # Add risk-adjusted modifier (S-curve boost for better RR)
        rr = features.get("rr_ratio", 1.0)
        risk_mod = np.tanh(rr - 2.0) * 5
        
        return max(0, min(100, base + risk_mod + 30)) # Offset to center around 70-80 for good picks

    @staticmethod
    def generate_ai_insight(p: Dict) -> str:
        """Generates a text-based AI insight for the stock."""
        score = p.get("edge_score", 50)
        upside = p.get("upside_pct", 0)
        consensus = p.get("analyst_consensus", "Neutral")
        
        if score > 80:
            return f"🧠 **AI ANALYSIS:** High institutional confluence detected. The model predicts a high probability of outperformance over the next 22 sessions. {consensus} consensus supports the current {upside}% upside target."
        elif score > 65:
            return f"📈 **AI ANALYSIS:** Positive momentum build-up. Fundamental stability ({p.get('fundamental_quality')}) aligns with current technical breakout strength. Low-to-moderate risk profile."
        else:
            return f"⚖️ **AI ANALYSIS:** Mixed signal environment. While {consensus} view is stable, technical volatility remains elevated. Suggest waiting for high-volume confirmation."

# ===============================================
# 3.5 STRATEGY PERFORMANCE ANALYTICS
# ===============================================
class StrategyBacktester:
    """Backtesting utilities (deterministic, candle-based)."""
    @staticmethod
    def get_strategy_performance() -> Dict[str, Any]:
        """
        Uses recent SPY trend/volatility to display realistic baseline expectations.
        This is NOT a true backtest of your exact scanner rules (would require event-based simulation).
        """
        try:
            df = yf.download("SPY", period="9mo", interval="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                raise ValueError("No SPY data")
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            close = df["close"].astype(float).dropna()
            rets = close.pct_change().dropna()
            if rets.empty:
                raise ValueError("No returns")

            ann_vol = float(rets.std() * np.sqrt(252) * 100)
            last_126 = close.tail(126)
            trend_6m = float((last_126.iloc[-1] / last_126.iloc[0] - 1) * 100) if len(last_126) > 10 else 0.0

            # Conservative, regime-aware heuristics
            base_win = 52.0 + np.clip(trend_6m / 3.5, -6, 8)
            dd_proxy = 6.0 + np.clip(ann_vol / 6.0, 2, 12)
            pf_proxy = 1.25 + np.clip(trend_6m / 35.0, -0.15, 0.55)
            avg_pnl_proxy = 1.2 + np.clip(trend_6m / 25.0, -0.3, 1.8)

            equity = [100000]
            step = np.clip(trend_6m / 9.0, -1.5, 2.5) / 100
            for _ in range(9):
                equity.append(round(equity[-1] * (1 + step), 2))

            return {
                "win_rate": round(float(np.clip(base_win, 40, 68)), 1),
                "profit_factor": round(float(np.clip(pf_proxy, 1.05, 2.2)), 2),
                "avg_trade_pnl": round(float(np.clip(avg_pnl_proxy, 0.4, 3.2)), 2),  # %
                "max_drawdown": round(float(np.clip(dd_proxy, 6, 22)), 1),  # %
                "total_trades": int(np.clip(110 + trend_6m, 80, 180)),
                "equity_growth": equity
            }
        except Exception:
            # Deterministic fallback (no randomness)
            return {
                "win_rate": 55.0,
                "profit_factor": 1.45,
                "avg_trade_pnl": 1.6,  # %
                "max_drawdown": 14.0,  # %
                "total_trades": 120,
                "equity_growth": [100000, 100800, 101200, 102200, 101600, 103400, 104800, 104200, 105700, 106900]
            }

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize yfinance OHLCV into ['open','high','low','close','volume'] and drop NaNs."""
        if df is None or df.empty:
            return None

        def flatten(c):
            if isinstance(c, tuple):
                for part in c:
                    p_str = str(part).lower()
                    if any(x in p_str for x in ['open', 'high', 'low', 'close', 'volume']):
                        for base in ['open', 'high', 'low', 'close', 'volume']:
                            if base in p_str:
                                return base
                return str(c[0]).lower()
            return str(c).lower()

        df = df.copy()
        df.columns = [flatten(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in required):
            return None
        out = df[required].copy()
        for c in required:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna()
        return out if not out.empty else None

    @staticmethod
    def backtest_breakout(
        df: pd.DataFrame,
        vol_z_threshold: float = 1.0,
        lookback_days: int = 20,
        max_hold_days: int = 20,
        stop_atr_mult: float = 1.75,
        target_atr_mult: float = 2.3,
        entry_mode: str = "next_open",
    ) -> Dict[str, Any]:
        """
        Event-based backtest of the breakout detector.
        - Signal: `detect_breakout()` on data up to day t (inclusive)
        - Entry: next day's open (default) or same-day close
        - Exit: stop, target1, or time-based exit at max_hold_days
        """
        df = StrategyBacktester._normalize_ohlcv(df)
        if df is None or len(df) < 120:
            return {"trades": [], "total_trades": 0}

        trades: List[Dict[str, Any]] = []
        close = df["close"].astype(float)
        atr_series = IndicatorUtils.atr(df[["high", "low", "close"]], 14).astype(float)

        i = max(60, lookback_days + 20)
        while i < len(df) - 2:
            window = df.iloc[: i + 1]
            direction, strength, vol_z = detect_breakout(window, max_lookback_days=lookback_days, vol_z_threshold=vol_z_threshold)
            if direction == "NONE" or strength <= 0:
                i += 1
                continue

            is_long = direction == "BULLISH"
            entry_idx = i + 1
            if entry_idx >= len(df):
                break

            entry = float(df["open"].iloc[entry_idx]) if entry_mode == "next_open" else float(df["close"].iloc[i])
            atr = float(atr_series.iloc[i])
            if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0 or entry <= 0:
                i += 1
                continue

            stop_dist = max(atr * stop_atr_mult, entry * 0.036)
            stop = entry - stop_dist if is_long else entry + stop_dist

            rr_mult = max(1.45, float(strength) / 20.0)
            target_dist = atr * target_atr_mult * rr_mult
            target = entry + target_dist if is_long else entry - target_dist

            exit_price = float(df["close"].iloc[entry_idx])
            exit_idx = entry_idx
            exit_reason = "TIME"

            end_idx = min(len(df) - 1, entry_idx + max_hold_days)
            for j in range(entry_idx, end_idx + 1):
                hi = float(df["high"].iloc[j])
                lo = float(df["low"].iloc[j])
                cl = float(df["close"].iloc[j])

                if is_long:
                    hit_stop = lo <= stop
                    hit_target = hi >= target
                    if hit_stop and hit_target:
                        # Conservative: assume stop first on the day of conflict.
                        exit_price, exit_idx, exit_reason = float(stop), j, "STOP"
                        break
                    if hit_stop:
                        exit_price, exit_idx, exit_reason = float(stop), j, "STOP"
                        break
                    if hit_target:
                        exit_price, exit_idx, exit_reason = float(target), j, "TARGET"
                        break
                    exit_price, exit_idx = cl, j
                else:
                    hit_stop = hi >= stop
                    hit_target = lo <= target
                    if hit_stop and hit_target:
                        exit_price, exit_idx, exit_reason = float(stop), j, "STOP"
                        break
                    if hit_stop:
                        exit_price, exit_idx, exit_reason = float(stop), j, "STOP"
                        break
                    if hit_target:
                        exit_price, exit_idx, exit_reason = float(target), j, "TARGET"
                        break
                    exit_price, exit_idx = cl, j

            pnl_pct = ((exit_price / entry) - 1) * 100 if is_long else ((entry / exit_price) - 1) * 100
            trades.append({
                "entry_idx": int(entry_idx),
                "exit_idx": int(exit_idx),
                "direction": "LONG" if is_long else "SHORT",
                "entry": round(entry, 4),
                "exit": round(exit_price, 4),
                "pnl_pct": round(float(pnl_pct), 3),
                "reason": exit_reason,
                "strength": float(strength),
                "vol_z": float(vol_z),
            })

            # Avoid overlapping trades: skip ahead to exit day + 1
            i = exit_idx + 1

        return {"trades": trades, "total_trades": len(trades)}

    @staticmethod
    def summarize_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not trades:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "equity_growth": [100000],
            }

        pnl = np.array([t["pnl_pct"] for t in trades], dtype=float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float((pnl > 0).mean() * 100)
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum()) if losses.size else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        avg_trade = float(pnl.mean())

        equity = [100000.0]
        for x in pnl:
            equity.append(equity[-1] * (1 + (x / 100.0)))
        peak = equity[0]
        max_dd = 0.0
        for v in equity:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100 if peak else 0.0
            max_dd = max(max_dd, dd)

        return {
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else 99.0,
            "avg_trade_pnl": round(avg_trade, 3),
            "max_drawdown": round(max_dd, 2),
            "total_trades": int(len(trades)),
            "equity_growth": [round(x, 2) for x in equity[-10:]],
        }

# ===============================================
# 3.7 MACRO & MARKET REGIME ENGINE
# ===============================================
class MarketRegimeEngine:
    """Detects the current market environment (Macro Context)."""
    _cache: Dict[str, Any] = {}
    _cache_ts: float = 0.0

    @staticmethod
    def get_market_regime() -> Dict[str, Any]:
        """Returns the current macro state (cached)."""
        now = time.time()
        if MarketRegimeEngine._cache and now - MarketRegimeEngine._cache_ts < 300:
            return MarketRegimeEngine._cache

        def _safe_last_close(ticker: str) -> Optional[float]:
            try:
                df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=True)
                if df is None or df.empty:
                    return None
                df = df.rename(columns={c: str(c).lower() for c in df.columns})
                close = df.get("close")
                if close is None or close.dropna().empty:
                    return None
                return float(close.dropna().iloc[-1])
            except Exception:
                return None

        vix = _safe_last_close("^VIX")
        tnx = _safe_last_close("^TNX")  # ^TNX is yield * 10
        yield_10y = (tnx / 10.0) if tnx is not None else None

        spy = yf.download("SPY", period="13mo", interval="1d", progress=False, auto_adjust=True)
        spy = spy.rename(columns={c: str(c).lower() for c in spy.columns}) if spy is not None else pd.DataFrame()
        spy_close = spy.get("close", pd.Series(dtype=float)).astype(float).dropna()
        spy_sma50 = float(IndicatorUtils.sma(spy_close, 50).iloc[-1]) if len(spy_close) >= 50 else None
        spy_sma200 = float(IndicatorUtils.sma(spy_close, 200).iloc[-1]) if len(spy_close) >= 200 else None
        spy_last = float(spy_close.iloc[-1]) if not spy_close.empty else None

        trend = "Unknown"
        if spy_last is not None and spy_sma50 is not None and spy_sma200 is not None:
            if spy_last > spy_sma50 > spy_sma200:
                trend = "Uptrend"
            elif spy_last < spy_sma50 < spy_sma200:
                trend = "Downtrend"
            else:
                trend = "Range"

        vix_val = float(vix) if vix is not None else 20.0
        regime = "Risk-On" if vix_val < 18 and trend == "Uptrend" else "Risk-Off" if vix_val > 25 or trend == "Downtrend" else "Neutral"
        sentiment = "Greed" if vix_val < 15 else "Fear" if vix_val > 22 else "Neutral"

        out = {
            "vix": round(vix_val, 2),
            "yield_10y": f"{round(float(yield_10y), 2)}%" if yield_10y is not None else "N/A",
            "regime": regime,
            "sentiment": sentiment,
            "spy_trend": trend,
            "advancing_sectors": ["Technology", "Energy", "Industrials"]  # placeholder; can be computed later
        }
        MarketRegimeEngine._cache = out
        MarketRegimeEngine._cache_ts = now
        return out

# ===============================================
# 3.9 VIRTUAL TRADE & JOURNAL MANAGER
# ===============================================
class TradeJournal:
    """Manages virtual/paper trades and performance tracking."""
    JOURNAL_FILE = Path("trade_journal.json")

    @classmethod
    def load_journal(cls) -> List[Dict]:
        if not cls.JOURNAL_FILE.exists(): return []
        try: return json.loads(cls.JOURNAL_FILE.read_text())
        except: return []

    @classmethod
    def save_trade(cls, trade: Dict):
        journal = cls.load_journal()
        trade["id"] = int(time.time())
        trade["status"] = "ACTIVE"
        trade.setdefault("opened_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        trade.setdefault("current_price", None)
        trade.setdefault("current_pnl_pct", 0.0)
        journal.append(trade)
        cls.JOURNAL_FILE.write_text(json.dumps(journal, indent=4))

    @classmethod
    def mark_to_market(cls) -> List[Dict]:
        """Refreshes current prices & PnL for ACTIVE trades (best-effort)."""
        journal = cls.load_journal()
        if not journal:
            return []

        active = [t for t in journal if str(t.get("status", "")).upper() == "ACTIVE" and t.get("symbol")]
        if not active:
            return journal

        symbols = sorted({t["symbol"] for t in active})
        try:
            df = yf.download(" ".join(symbols), period="7d", interval="1d", progress=False, group_by="ticker", auto_adjust=True, threads=True)
        except Exception:
            df = None

        def _last_close(sym: str) -> Optional[float]:
            if df is None or getattr(df, "empty", True):
                return None
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    c = df[(sym, "Close")] if (sym, "Close") in df.columns else df[(sym, "close")]
                else:
                    c = df["Close"] if "Close" in df.columns else df["close"]
                c = pd.to_numeric(c, errors="coerce").dropna()
                return float(c.iloc[-1]) if not c.empty else None
            except Exception:
                return None

        price_map = {s: _last_close(s) for s in symbols}

        for t in journal:
            if str(t.get("status", "")).upper() != "ACTIVE":
                continue
            sym = t.get("symbol")
            last = price_map.get(sym)
            if last is None:
                continue
            entry = float(t.get("entry") or 0)
            direction = str(t.get("direction", "LONG")).upper()
            if entry <= 0:
                continue
            pnl = ((last / entry) - 1) * 100 if direction == "LONG" else ((entry / last) - 1) * 100
            t["current_price"] = round(last, 4)
            t["current_pnl_pct"] = round(float(pnl), 2)

        cls.JOURNAL_FILE.write_text(json.dumps(journal, indent=4))
        return journal

# ===============================================
# 4. TRADE PROPOSER CORE LOGIC
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
    expected_price: float # Target Median
    expected_price_3m: float
    expected_price_12m: float
    rr_ratio: float
    quantity: int
    capital_at_risk: float
    volume_surge: float
    strength_pct: float
    news_impact: str
    fundamental_quality: str
    fundamental_score: float
    roe_pct: float
    forward_pe: float
    analyst_consensus: str
    nasdaq_consensus: str
    analyst_rating_score: float
    analyst_count: int
    zacks_rank: str
    motley_fool_view: str
    marketbeat_score: str
    upside_pct: float
    rs_score: float # Relative Strength vs SPY
    sector: str
    industry: str
    edge_score: float
    enhanced_score: float
    ml_score: float
    confidence: float
    invest_rank_score: float
    conviction_tier: str
    backtest_score: float # Historical success rate for this symbol/setup
    ai_insight: str
    timestamp: str
    debug: Dict[str, Any]

class TradeProposer:
    """Core trade proposal engine with multi-factor scoring."""

    CONFIG = {
        "MIN_PRICE": 2.0,
        "MIN_AVG_VOLUME": 5_000,
        "MIN_DOLLAR_VOLUME": 250_000,
        "MIN_RR": 1.1,
        "EDGE_THRESHOLD": 5.0,
        "MIN_VOLUME_SURGE": 0.0, # Handled by Z-Score now
        "MAX_SPREAD_PCT": 5.0,
        "STOP_ATR_MULTIPLIER": 1.75,
        "TARGET_ATR_MULTIPLIER": 2.3,
        "MAX_POSITION_PCT": 20.0,
        "MAX_ADV_PARTICIPATION": 0.01,
    }

    def __init__(self, account_size: float = 25_000, risk_pct: float = 1.0):
        self.account_size = float(account_size)
        self.risk_amount = self.account_size * (risk_pct / 100)
        self.ml = MLScoringEngine()

    async def safe_get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetches fundamentals safely."""
        try:
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            base_data = MarketSentimentEngine.get_fundamental_score(info)
            # Add extra metadata
            base_data["sector"] = info.get("sector", "Unknown")
            base_data["industry"] = info.get("industry", "Unknown")
            base_data["dividend_yield"] = info.get("dividendYield", 0)
            return base_data
        except Exception:
            return {"fundamental_score": 50, "quality": "Neutral", "sector": "N/A", "industry": "N/A", "dividend_yield": 0}

    @staticmethod
    async def safe_get_news(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Builds a deterministic sentiment proxy from price/volume action."""
        return await MarketSentimentEngine.get_news_sentiment(symbol, df)
    
    def _get_scalar(self, df: pd.DataFrame, column: str):
        """Safely extract the last scalar value from a DataFrame column."""
        try:
            val = df[column].iloc[-1]
            return float(val.item() if hasattr(val, 'item') else val)
        except Exception:
            raise ValueError(f"Could not extract scalar from {column} for proposal.")

    async def propose(self, df: pd.DataFrame, direction: str, strength: float, symbol: str, rs_score: float = 50, vol_z: float = 0) -> Optional[Proposal]:
        """Generates a trade proposal."""
        if len(df) < 40:
            logger.info(f"[{symbol}] Skip: insufficient data ({len(df)})")
            return None
        if direction.upper() == "NONE" or strength <= 0:
            logger.info(f"[{symbol}] Skip: no qualified breakout signal")
            return None
            
        try:
            close = self._get_scalar(df, "close")
            volume = self._get_scalar(df, "volume")
        except ValueError as e:
            logger.error(f"[{symbol}] Data extraction failed: {e}")
            return None

        avg_vol = df["volume"].tail(40).mean()
        vol_surge = volume / avg_vol if avg_vol > 0 else 0

        if not self._validate_market(symbol, close, avg_vol, vol_surge):
            return None

        indicators = self._compute_indicators(df, close, direction)
        
        if indicators is None:
            logger.info(f"[{symbol}] Skip: Technical indicators misaligned.")
            return None

        news, fund, outlook = await asyncio.gather(
            self.safe_get_news(symbol, df),
            self.safe_get_fundamentals(symbol),
            AnalystAggregator.get_analyst_outlook(symbol)
        )

        if fund.get("fundamental_score", 50) < 25 and outlook.get("rating_score", 50) < 30:
            logger.info(f"[{symbol}] Skip: fundamentals and analyst outlook poor")
            return None

        atr14 = indicators["atr14"]
        stop, t1, t2, t3, rr, qty, risk_per_share = self._compute_stops_targets(
            close, atr14, direction, strength, avg_vol
        )
        
        if rr < self.CONFIG["MIN_RR"]:
            logger.info(f"[{symbol}] Skip: RR ratio too low ({rr:.2f})")
            return None

        edge_score, enhanced_score, ml_score, confidence = self._compute_scores(
            indicators, news, fund, outlook, rr, strength, rs_score, vol_z
        )

        if edge_score < self.CONFIG["EDGE_THRESHOLD"]:
            logger.info(f"[{symbol}] Skip: edge_score={edge_score:.1f} below threshold")
            return None

        # Build Proposal
        proposal = Proposal(
            symbol=symbol,
            direction="LONG" if direction.upper() in ["LONG", "UP", "BULLISH"] else "SHORT",
            entry=round(close, 3),
            stop_loss=round(stop, 3),
            target1=round(t1, 3),
            target2=round(t2, 3),
            target3=round(t3, 3),
            expected_price=outlook.get("expected_price", 0),
            expected_price_3m=outlook.get("expected_price_3m", 0),
            expected_price_12m=outlook.get("expected_price_12m", 0),
            rr_ratio=round(rr, 2),
            quantity=qty,
            capital_at_risk=round(qty * risk_per_share, 2),
            volume_surge=round(vol_surge, 2),
            strength_pct=round(strength, 1),
            news_impact=news.get("news_impact", "Neutral"),
            fundamental_quality=fund.get("quality", "Neutral"),
            fundamental_score=round(float(fund.get("fundamental_score", 50)), 1),
            roe_pct=round(float(fund.get("roe", 0)), 1),
            forward_pe=round(float(fund.get("pe", 0)), 1),
            analyst_consensus=outlook.get("consensus", "Hold"),
            nasdaq_consensus=outlook.get("nasdaq_consensus", outlook.get("consensus", "Neutral")),
            analyst_rating_score=round(float(outlook.get("rating_score", 50)), 1),
            analyst_count=int(outlook.get("analyst_count", 0)),
            zacks_rank=outlook.get("zacks_rank", "3-Hold"),
            motley_fool_view=outlook.get("motley_fool_view", "Neutral"),
            marketbeat_score=outlook.get("marketbeat_score", "2.5/5.0"),
            upside_pct=outlook.get("upside_pct", 0),
            rs_score=round(rs_score, 1),
            sector=fund.get("sector", "N/A"),
            industry=fund.get("industry", "N/A"),
            edge_score=round(edge_score, 1),
            enhanced_score=round(enhanced_score, 1),
            ml_score=round(ml_score, 1),
            confidence=round(confidence, 1),
            invest_rank_score=round(self._compute_invest_rank(edge_score, confidence, rr, fund.get("fundamental_score", 50), outlook.get("rating_score", 50), outlook.get("upside_pct", 0)), 1),
            conviction_tier="B",
            backtest_score=round(self._estimate_setup_quality(edge_score, rr, strength, rs_score, vol_z), 1),
            ai_insight="Generating...",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            debug={**indicators, "vol_z": vol_z, "news_score": news.get("sentiment_score", 50), "upside": outlook.get("upside_pct")}
        )
        proposal.conviction_tier = self._conviction_tier(proposal.invest_rank_score)
        
        # Add the real insight after creation for ease
        proposal.ai_insight = MLScoringEngine.generate_ai_insight(proposal.__dict__)
        logger.info(f"[{symbol}] ✓ Trade Proposed | RR={rr:.2f} | Edge={edge_score:.1f} | ML={ml_score:.1f} | Conf={confidence:.1f}")
        return proposal

    def _validate_market(self, symbol: str, close: float, avg_vol: float, vol_surge: float) -> bool:
        """Returns True if a symbol passes basic market prefilters."""
        if close < self.CONFIG.get("MIN_PRICE", 1.0):
            logger.info(f"[{symbol}] Skip: low price ({close:.2f})")
            return False
        if avg_vol < self.CONFIG.get("MIN_AVG_VOLUME", 1000):
            logger.info(f"[{symbol}] Skip: avg volume too low ({avg_vol:.0f})")
            return False
        if close * avg_vol < self.CONFIG.get("MIN_DOLLAR_VOLUME", 0):
            logger.info(f"[{symbol}] Skip: dollar volume too low ({close * avg_vol:.0f})")
            return False
        return True

    def _compute_indicators(self, df: pd.DataFrame, close: float, direction: str) -> Optional[Dict]:
        """Computes necessary indicators and checks for directional alignment."""
        try:
            ema21 = float(IndicatorUtils.ema(df["close"], 21).iloc[-1])
            sma50 = float(IndicatorUtils.sma(df["close"], 50).iloc[-1])
            rsi14 = float(IndicatorUtils.rsi(df["close"], 14).iloc[-1])
            atr14 = float(IndicatorUtils.atr(df, 14).iloc[-1])
            supertrend = bool(IndicatorUtils.supertrend(df, 10, 3.0).iloc[-1])
        except Exception as e:
            logger.warning(f"Indicator computation failed: {e}")
            return None
            
        is_long = direction.lower() in ["up", "long"]
        
        # We no longer hard-gate the SMA/RSI for institutional discovery,
        # but we track alignment for scoring.
        sma_aligned = (close > sma50) if is_long else (close < sma50)
        rsi_extreme = (rsi14 > 78) if is_long else (rsi14 < 22)

        return {
            "ema21": ema21,
            "sma50": sma50, 
            "rsi14": rsi14, 
            "atr14": atr14, 
            "atr_pct": (atr14 / close) * 100 if close else 0.0,
            "is_long": is_long, 
            "supertrend_aligned": supertrend if is_long else not supertrend,
            "sma_aligned": sma_aligned,
            "rsi_extreme": rsi_extreme
        }
    
    def _compute_stops_targets(self, close: float, atr14: float, direction: str, strength: float, avg_vol: float) -> Tuple[float, float, float, float, float, int, float]:
        """Calculates risk parameters with multiple take-profit levels."""
        is_long = direction.lower() in ["up", "long"]
        
        stop_dist = max(atr14 * self.CONFIG["STOP_ATR_MULTIPLIER"], close * 0.036)
        stop = close - stop_dist if is_long else close + stop_dist

        rr_multiplier = max(1.45, strength / 20)
        base_target_dist = atr14 * self.CONFIG["TARGET_ATR_MULTIPLIER"] * rr_multiplier
        
        # 3 Take-Profit Levels
        t1 = close + base_target_dist if is_long else close - base_target_dist
        t2 = close + base_target_dist * 1.5 if is_long else close - base_target_dist * 1.5
        t3 = close + base_target_dist * 2.2 if is_long else close - base_target_dist * 2.2

        risk_per_share = max(abs(close - stop), 0.01)
        qty_from_risk = max(1, int(self.risk_amount / risk_per_share))
        max_position_value = self.account_size * (self.CONFIG["MAX_POSITION_PCT"] / 100)
        qty_from_capital = max(1, int(max_position_value / close))
        qty_from_liquidity = max(1, int(avg_vol * self.CONFIG["MAX_ADV_PARTICIPATION"]))
        qty = max(1, min(qty_from_risk, qty_from_capital, qty_from_liquidity))
        rr = abs(t1 - close) / risk_per_share
            
        return stop, t1, t2, t3, rr, qty, risk_per_share

    def _compute_scores(self, indicators: Dict, news: Dict, fund: Dict, outlook: Dict, rr: float, strength: float, rs_score: float = 50, vol_z: float = 0) -> Tuple[float, float, float, float]:
        """Calculates Multi-Factor Edge, Enhanced, ML, and Confidence scores."""
        
        ml_features = {
            **indicators, 
            "rr_ratio": rr, 
            "strength": strength, 
            "news_score": news.get("sentiment_score", 0), 
            "fund_score": fund.get("fundamental_score", 50),
            "analyst_score": outlook.get("rating_score", 50),
            "rs_score": rs_score
        }
        ml_score = self.ml.predict_score(ml_features)

        tech_score = (
            strength * 0.35
            + (12 if indicators.get("sma_aligned", False) else -4)
            + (10 if indicators.get("supertrend_aligned", False) else -8)
            + (-8 if indicators.get("rsi_extreme", False) else 8)
            + min(10, max(-10, vol_z * 3))
        )
        
        news_score = news.get("sentiment_score", 0)
        fund_score = fund.get("fundamental_score", 50)
        analyst_score = outlook.get("rating_score", 50)
        
        # Multi-Factor Edge Score with Alpha (RS) weight
        edge_score = max(0, min(100, 
            tech_score * 0.30 + 
            fund_score * 0.27 + 
            analyst_score * 0.18 +
            rs_score * 0.15 +
            news_score * 0.05 +
            min(10, rr * 2.5) +
            min(8, max(0, vol_z) * 2.0)
        ))

        enhanced_score = max(0, min(100, edge_score * 0.50 + ml_score * 0.22 + rr * 7.0 + strength * 0.18 + max(0, vol_z) * 2.0))
        confidence = max(0, min(100, (enhanced_score * 0.42 + rr * 9 + strength * 0.22 + ml_score * 0.24 + analyst_score * 0.12) / 3.0))

        return edge_score, enhanced_score, ml_score, confidence

    @staticmethod
    def _estimate_setup_quality(edge_score: float, rr: float, strength: float, rs_score: float, vol_z: float) -> float:
        """Deterministic setup-quality proxy used instead of random backtest values."""
        score = 35 + edge_score * 0.35 + rr * 8 + strength * 0.12 + rs_score * 0.10 + max(0, vol_z) * 3.0
        return max(0, min(100, score))

    @staticmethod
    def _compute_invest_rank(edge_score: float, confidence: float, rr: float, fund_score: float, analyst_score: float, upside_pct: float) -> float:
        """Composite ranking for deciding where to invest first."""
        rank = (
            edge_score * 0.30
            + confidence * 0.25
            + max(0, min(100, fund_score)) * 0.20
            + max(0, min(100, analyst_score)) * 0.15
            + max(0.0, min(20.0, rr * 5.0))
            + max(0.0, min(12.0, upside_pct * 0.25))
        )
        return max(0.0, min(100.0, rank))

    @staticmethod
    def _conviction_tier(invest_rank_score: float) -> str:
        if invest_rank_score >= 85:
            return "A+"
        if invest_rank_score >= 75:
            return "A"
        if invest_rank_score >= 60:
            return "B"
        return "C"

# ===============================================
# 5. ASYNC BATCH PROCESSING
# ===============================================
class TradeProposerAsyncWrapper(TradeProposer):
    """
    Async batch processing wrapper around TradeProposer logic.
    """

    def __init__(self, account_size: float = 25_000, risk_pct: float = 1.0, max_concurrency: int = 20):
        super().__init__(account_size, risk_pct)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def propose_for_symbol(self, symbol: str, df: pd.DataFrame, direction: str, strength: float, rs_score: float = 50, vol_z: float = 0) -> Optional[Dict[str, Any]]:
        """Propose a trade for a single symbol safely with semaphore control."""
        async with self.semaphore:
            try:
                # df is already normalized in _fetch_with_cache
                return await self.propose(df, direction, strength, symbol, rs_score, vol_z)
            except Exception as e:
                logger.error(f"[{symbol}] Proposal failed: {e}")
                return None

    async def propose_batch(self, symbols_data: List[Dict[str, Any]]) -> List[Proposal]:
        """Processes a list of potential trade candidates concurrently."""
        tasks = [
            self.propose_for_symbol(
                data["symbol"], 
                data["df"], 
                data["direction"], 
                data["strength"], 
                data.get("rs_score", 50),
                data.get("vol_z", 0)
            )
            for data in symbols_data
        ]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res is not None]

# ===============================================
# 6. STANDALONE UTILS AND FETCHERS
# ===============================================
def to_float(x: Any) -> float:
    """Safely convert a pandas/numpy scalar, series, or generic value to float."""
    try:
        if isinstance(x, pd.Series) and not x.empty:
            x = x.iloc[0]
        elif isinstance(x, np.ndarray) and x.size > 0:
            x = x[0]
        return float(x.item() if hasattr(x, 'item') else x)
    except Exception:
        return 0.0
 
# This function is responsible for detecting the breakout patterns.
def detect_breakout(
    df: pd.DataFrame,
    max_lookback_days: int = 20,
    vol_z_threshold: float = 1.0,
    breakout_buffer_atr: float = 0.15,
    base_tight_atr_mult: float = 3.2
) -> tuple[str, float, float]:
    """
    Analyzes the price action of a given DataFrame (OHLCV data) to detect
    a strong breakout, handling potential MultiIndex column names.

    :param df: Pandas DataFrame with OHLCV data.
    :param max_lookback_days: Maximum number of days to look back for consolidation.
    :return: Tuple of (direction: str, strength: float, vol_z: float)
    """

    # --- Column Normalization (Handles MultiIndex tuples) ---
    def normalize_col(col):
        """Helper to flatten MultiIndex column names (tuples) into strings."""
        if isinstance(col, tuple):
            # Join the elements of the tuple (e.g., ('Close', 'Symbol') -> 'close_symbol')
            # The filter ensures empty levels (like the symbol level) don't create ambiguity
            return '_'.join(str(x) for x in col if x is not None and str(x).strip() != '').lower()
        # For standard columns, just convert to string and lowercase
        return str(col).lower()

    # Apply the normalization across all columns
    df_normalized = df.rename(columns={c: normalize_col(c) for c in df.columns})
    # -----------------------------------------------------------------------

    # --- Initial Check for sufficient data for ATR (Window size is 14) ---
    if len(df_normalized) < 15: # Need at least 14 points + 1 for proper calculation
        print(f"Warning: Insufficient data points ({len(df_normalized)} rows) for 14-period ATR calculation.")
        return "NONE", 0.0, 0.0

    # --- Column Mapping (Ensures ATR calculation uses correct dynamic column names) ---
    col_map = {}
    current_cols = df_normalized.columns
    base_cols = {'open', 'high', 'low', 'close', 'volume'}

    for req in base_cols:
        if req in current_cols:
            col_map[req] = req # Use simple name if available
        else:
            # Search for the column, e.g., 'ticker_close' if the symbol name was prepended
            # Avoid 'adjusted' closes which are less common for futures/daily analysis
            found_col = next((c for c in current_cols if req in c and 'adj' not in c), None)
            if found_col:
                col_map[req] = found_col # Use the dynamically found name
            else:
                col_map[req] = None # Mark as missing

    # Guard check for critical columns needed for the breakout model
    critical_cols = ['high', 'low', 'close', 'volume']
    if not all(col_map.get(c) for c in critical_cols):
        print(f"Warning: Missing critical OHLC data after normalization. Missing: {[c for c in critical_cols if not col_map.get(c)]}")
        return "NONE", 0.0, 0.0
    
    # Helper to ensure we get a single Series, resolving potential duplicated column names
    def _get_single_series(df, col_name):
        """Safely extracts a single Series from a DataFrame, even if the column name is duplicated."""
        if col_name in df.columns:
            col_data = df.loc[:, col_name]
            # If the column name is duplicated, Pandas returns a DataFrame. We take the first instance.
            if isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0]
            return col_data
        
        # This branch should not be reached if the earlier guard passed
        return pd.Series([], dtype='float64')

    # Extract mapped column names for clarity
    high_col = col_map['high']
    low_col = col_map['low']
    close_col = col_map['close']
    
    # --- CRITICAL FIX: Reconstruct Clean DataFrame ---
    # To fix persistent KeyError/ValueError issues caused by duplicate columns or index issues,
    # we create a brand new DataFrame containing ONLY the strict Series we need.
    try:
        high_series = _get_single_series(df_normalized, high_col)
        low_series = _get_single_series(df_normalized, low_col)
        close_series = _get_single_series(df_normalized, close_col)
        
        # Build the clean working DataFrame
        work_df = pd.DataFrame({
            'high': high_series,
            'low': low_series,
            'close': close_series
        })
        
        # Ensure proper index alignment and deduplication
        work_df = work_df.loc[~work_df.index.duplicated(keep='last')]
        work_df = work_df.sort_index()

        # Check sufficiency again after reconstruction
        if len(work_df) < max_lookback_days + 2:
            return "NONE", 0.0, 0.0

        # --- ATR Calculation on Clean Data ---
        work_df['tr1'] = work_df['high'] - work_df['low']
        work_df['tr2'] = abs(work_df['high'] - work_df['close'].shift(1))
        work_df['tr3'] = abs(work_df['low'] - work_df['close'].shift(1))
        work_df['tr'] = work_df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Simple smoothing average (SMA) for ATR
        work_df['atr'] = work_df['tr'].rolling(window=14).mean()
        
        # Drop initial NaN values from ATR calculation
        work_df = work_df.dropna(subset=['atr'])

        if work_df.empty:
            return "NONE", 0.0, 0.0

        vol_series = _get_single_series(df_normalized, col_map['volume'])
        vol_zscores = IndicatorUtils.calculate_volume_zscore(vol_series)
        current_vol_z = float(vol_zscores.iloc[-1])

        # Current price and volatility (last values)
        current_close = float(work_df['close'].iloc[-1])
        current_atr = float(work_df['atr'].iloc[-1])

        # Lookback period for consolidation check (default is max_lookback_days)
        # CRITICAL FIX: Lookback should EXCLUDE the current candle to detect a breakout FROM the past.
        # We look at the previous 'max_lookback_days' candles (from -max_lookback_days-1 to -1)
        lookback_df = work_df.iloc[-(max_lookback_days + 1):-1]

        if lookback_df.empty:
             return "NONE", 0.0, 0.0

        range_high = lookback_df['high'].max()
        range_low = lookback_df['low'].min()
        price_range = range_high - range_low
        range_mid = (range_high + range_low) / 2
        ema21 = float(IndicatorUtils.ema(work_df['close'], 21).iloc[-1])
        sma50 = float(IndicatorUtils.sma(work_df['close'], 50).iloc[-1])

        # Prefer compressed bases that expand on abnormal volume.
        base_is_tight = price_range <= current_atr * float(max(1.0, base_tight_atr_mult))
        breakout_buffer = current_atr * float(max(0.01, breakout_buffer_atr))
        vol_threshold = float(vol_z_threshold)

        direction = "NONE"
        strength = 0.0

        if current_vol_z >= vol_threshold and base_is_tight:
            bullish_break = current_close > (range_high + breakout_buffer) and current_close > ema21 > sma50
            bearish_break = current_close < (range_low - breakout_buffer) and current_close < ema21 < sma50

            if bullish_break:
                direction = "BULLISH"
                depth = (current_close - range_high) / max(current_atr, 0.01)
                compression_bonus = max(0, (current_atr * float(max(1.0, base_tight_atr_mult)) - price_range) / max(current_atr, 0.01))
                strength = depth * 55 + max(0, current_vol_z) * 14 + compression_bonus * 6 + max(0, (current_close - range_mid) / max(current_atr, 0.01)) * 5
            elif bearish_break:
                direction = "BEARISH"
                depth = (range_low - current_close) / max(current_atr, 0.01)
                compression_bonus = max(0, (current_atr * float(max(1.0, base_tight_atr_mult)) - price_range) / max(current_atr, 0.01))
                strength = depth * 55 + max(0, current_vol_z) * 14 + compression_bonus * 6 + max(0, (range_mid - current_close) / max(current_atr, 0.01)) * 5

        if strength > 0:
            score = min(strength, 100.0)
            return direction, round(score, 2), round(current_vol_z, 2)
            
        return "NONE", 0.0, 0.0

    except Exception as e:
        print(f"Error in calculation block for a symbol: {e}")
        return "NONE", 0.0, 0.0


class SymbolFetcher:
    """Fetches a list of tradable symbols, prioritizing cache/remote."""
    CACHE_FILE = Path("symbols_cache.json")

    async def get_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Fetches symbols, first trying cache, then remote API, then fallback."""
        symbols = []
        if self.CACHE_FILE.exists():
            try:
                data = json.loads(self.CACHE_FILE.read_text(encoding="utf-8"))
                symbols = data.get("symbols", [])
            except: pass

        if not symbols:
            # Remote Source (NASDAQ Listings)
            url = "https://raw.githubusercontent.com/datasets/nasdaq-listings/master/data/nasdaq-listed.json"
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=50)) as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            raw = await resp.json()
                            symbols = sorted({x["Symbol"] for x in raw if x.get("Symbol") and x["Symbol"].isalpha() and 1 <= len(x["Symbol"]) <= 5})
                            if symbols:
                                self.CACHE_FILE.write_text(json.dumps({"symbols": symbols}), encoding="utf-8")
            except Exception as e:
                logger.warning(f"Remote symbol list download failed: {e}")

        if not symbols:
            logger.warning("Using fallback symbol list.")
            symbols = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","JPM","V","WMT",
                        "BRK-B","BRK-A","AVGO","TSM","BABA","TCEHY","ORCL","XOM","CVX",
                        "JNJ","PG","KO","PEP","MCD","COST","DIS","NFLX","MA","BAC","WFC","C","GS","MS",
                        "UBS","SAP","UL","LLY","NKE","IBM","INTC","AMD","QCOM","HD","LOW","RTX","BA",
                        "CAT","GM","F","UBER","LYFT","COIN","ASML","SONY","ADBE","CRM","SNOW","ZM",
                        "SQ","PYPL","SBUX","MO","PM","BTI","RIO","BHP","VALE","SHEL","TTE","BP",
                        "CVE","ENB","PLTR","TXN","SNPS","GILD","MRK","PFE","ABBV","UNH","CVS","TMO",
                        "MDT","BMY","CL","CLX","KMB","GIS","MNST","MRNA","AMGN","AXP","HSBC","SAN",
                        "LVMUY","NSRGY","NVS","RHHBY","VWAGY","TM","HMC","CHL","TCS","INFY","IBN","HDB","CMCSA",
                        "T","VZ","TMUS","VOD","GOLD","NEM","FCX"]
        
        # Always ensure SPY is included for RS calculation
        if "SPY" not in symbols: symbols.insert(0, "SPY")
        return symbols[:limit] if limit else symbols

# ===============================================
# 7. MAIN SCANNER CLASS
# ===============================================
@dataclass
class ScanResult:
    longs: List[Dict]
    shorts: List[Dict]
    institutional_quality: List[Dict] # Top stocks by fundamental + analyst score
    total_scanned: int
    valid_breakouts: int
    duration_seconds: float
    diagnostics: Dict[str, Any]

class Scanner:
    """The main entry point for the trade scanning and proposal pipeline."""
    MAX_CONCURRENT_FETCH = 18
    CACHE_TTL = 3600  # 1 hour
    SIGNAL_FREQ_BOOST_PCT = 8.0  # More aggressive default to avoid zero-signal regimes
    AUTO_TUNE_MIN_BOOST = 3.0
    AUTO_TUNE_MAX_BOOST = 18.0
    AUTO_TUNE_TARGET_HIT_RATE = 0.10
    AUTO_TUNE_DEADBAND = 0.015
    AUTO_TUNE_STEP = 0.5
    _recent_hit_rates: List[float] = []

    def __init__(self, account_size: float = 25_000, risk_pct: float = 1.0):
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_FETCH)
        self.data_cache: Dict[str, Tuple[pd.DataFrame, float]] = {} 

        self.proposer = TradeProposerAsyncWrapper(
            account_size, 
            risk_pct, 
            max_concurrency=20 
        )

    async def _fetch_with_cache(self, 
                                symbol: str, 
                                period: str = "6mo", 
                                interval: str = "1d") -> Optional[pd.DataFrame]:
        """Fetches OHLCV data using yfinance (blocking) via asyncio.to_thread with caching."""
        now = time.time()
        if symbol in self.data_cache and now - self.data_cache[symbol][1] < self.CACHE_TTL:
            return self.data_cache[symbol][0]

        async with self.semaphore:
            try:
                # Use per-symbol history fetch to avoid cross-symbol contamination and
                # intermittent yfinance thread-safety issues seen with concurrent downloads.
                ticker = yf.Ticker(symbol)
                df = await asyncio.to_thread(
                    ticker.history,
                    period=period,
                    interval=interval,
                    auto_adjust=True
                )
                if df is None or df.empty: return None

                # 1. Normalize columns
                df.columns = [str(c).lower() for c in df.columns]
                
                # Deduplicate columns (take first occurrence)
                df = df.loc[:, ~df.columns.duplicated()]
                
                # 2. Extract only what we need and drop NaNs
                required = ['open', 'high', 'low', 'close', 'volume']
                if not all(c in df.columns for c in required):
                    return None
                    
                df_normalized = df[required].copy().dropna()
                for col in required:
                    df_normalized[col] = pd.to_numeric(df_normalized[col], errors="coerce")
                df_normalized = df_normalized.dropna()

                if len(df_normalized) < 60: return None
                # Guard against poisoned frames where price/volume are effectively constant.
                if df_normalized["close"].nunique() < 5:
                    logger.warning(f"[{symbol}] Skip: suspiciously low close variability in fetched data.")
                    return None

                self.data_cache[symbol] = (df_normalized, now)
                return df_normalized

            except Exception as e:
                logger.error(f"[{symbol}] Download failed: {e}")
                return None

    async def scan_for_breakouts(
        self,
        limit: int = 500,
        min_edge_score: float = 68.0,
        vol_z_threshold: float = 1.0,
        manual_signal_boost_pct: Optional[float] = None
    ) -> ScanResult:
        """Runs the entire scanning and proposal pipeline."""
        start = time.time()
        logger.info(f"🚀 Starting elite breakout scan | Target symbols: {limit}")

        fetcher = SymbolFetcher()
        symbols = await fetcher.get_symbols(limit)
        total = len(symbols)

        # 1. PARALLEL OHLCV FETCH
        tasks = [self._fetch_with_cache(sym) for sym in symbols]
        ohlcv_results = await asyncio.gather(*tasks)
        valid_datasets = {sym: df for sym, df in zip(symbols, ohlcv_results) if df is not None}
        
        if not valid_datasets:
            logger.error("❌ CRITICAL: No valid OHLCV datasets retrieved. Check internet or API status.")
            return ScanResult([], [], [], total, 0, 0, {"valid_datasets": 0})
            
        logger.info(f"📊 Valid OHLCV datasets: {len(valid_datasets)}/{total}")

        # 3. ALPHA (RS) CALCULATION (Relative to SPY)
        spy_df = valid_datasets.get("SPY")
        spy_perf = 0.0
        if spy_df is not None:
             spy_perf = (spy_df["close"].iloc[-1] / spy_df["close"].iloc[0] - 1) * 100

        # 4. ANALYSIS PIPELINE (Parallelized)
        logger.info(f"🔍 Analyzing {len(valid_datasets)} symbols for institutional quality...")
        
        # Adaptive frequency control: keep signal rate near target without over-loosening.
        requested_boost = self.SIGNAL_FREQ_BOOST_PCT if manual_signal_boost_pct is None else float(manual_signal_boost_pct)
        current_boost = max(self.AUTO_TUNE_MIN_BOOST, min(requested_boost, self.AUTO_TUNE_MAX_BOOST))
        relax = current_boost / 100.0
        effective_vol_z_threshold = max(0.1, float(vol_z_threshold) * (1.0 - relax))
        effective_breakout_buffer_atr = 0.15 * (1.0 - relax)

        analysis_input = []
        no_signal = 0
        for sym, df in valid_datasets.items():
            if sym == "SPY": continue
            direction, strength, vol_z = detect_breakout(
                df,
                vol_z_threshold=effective_vol_z_threshold,
                breakout_buffer_atr=effective_breakout_buffer_atr,
                base_tight_atr_mult=3.2
            )
            if direction == "NONE" or strength <= 0:
                no_signal += 1
                continue
            
            # Calculate RS Score (Outperformance)
            stock_perf = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            rs_score = max(0, min(100, 50 + (stock_perf - spy_perf) * 2))

            analysis_input.append({
                "symbol": sym, 
                "df": df, 
                "direction": direction,
                "strength": strength,
                "rs_score": rs_score,
                "vol_z": vol_z
            })

        fallback_triggered = False
        if not analysis_input:
            # Emergency fallback: only used when the primary pass finds zero setups.
            fallback_triggered = True
            looser_vol_z = max(0.35, effective_vol_z_threshold * 0.80)
            looser_breakout_buffer = max(0.08, effective_breakout_buffer_atr * 0.80)
            for sym, df in valid_datasets.items():
                if sym == "SPY":
                    continue
                direction, strength, vol_z = detect_breakout(
                    df,
                    vol_z_threshold=looser_vol_z,
                    breakout_buffer_atr=looser_breakout_buffer,
                    base_tight_atr_mult=3.8
                )
                if direction == "NONE" or strength <= 0:
                    continue
                stock_perf = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
                rs_score = max(0, min(100, 50 + (stock_perf - spy_perf) * 2))
                analysis_input.append({
                    "symbol": sym,
                    "df": df,
                    "direction": direction,
                    "strength": strength,
                    "rs_score": rs_score,
                    "vol_z": vol_z
                })
        
        proposals_raw = await self.proposer.propose_batch(analysis_input)
        def normalize(p):
            if p is None:
                return None
        
            if isinstance(p, dict):
                return p
        
            if hasattr(p, "__dict__"):
                return vars(p)
        
            return None
        
        proposals_raw = [normalize(p) for p in proposals_raw]
        proposals_raw = [p for p in proposals_raw if p is not None]
        proposals = [
            p for p in proposals_raw
            if (p.edge_score if hasattr(p, "edge_score") else p.get("edge_score", 0)) >= min_edge_score
        ]

        diagnostics = {
            "symbols_requested": total,
            "valid_datasets": len(valid_datasets),
            "no_signal": no_signal,
            "breakout_candidates": len(analysis_input),
            "proposals_raw": len(proposals_raw),
            "min_edge_score": float(min_edge_score),
            "proposals_after_edge": len(proposals),
            "vol_z_threshold": float(vol_z_threshold),
            "effective_vol_z_threshold": round(float(effective_vol_z_threshold), 4),
            "effective_breakout_buffer_atr": round(float(effective_breakout_buffer_atr), 4),
            "signal_freq_boost_pct": round(current_boost, 2),
            "fallback_triggered": fallback_triggered,
            "manual_boost_enabled": manual_signal_boost_pct is not None,
        }

        # Auto-tune boost based on realized hit-rate in this scan.
        analyzed = max(1, len(valid_datasets) - (1 if "SPY" in valid_datasets else 0))
        hit_rate = len(proposals) / analyzed
        self._recent_hit_rates.append(float(hit_rate))
        if len(self._recent_hit_rates) > 8:
            self._recent_hit_rates = self._recent_hit_rates[-8:]
        avg_hit_rate = float(sum(self._recent_hit_rates) / len(self._recent_hit_rates))

        new_boost = current_boost
        if avg_hit_rate < (self.AUTO_TUNE_TARGET_HIT_RATE - self.AUTO_TUNE_DEADBAND):
            new_boost = min(self.AUTO_TUNE_MAX_BOOST, current_boost + self.AUTO_TUNE_STEP)
        elif avg_hit_rate > (self.AUTO_TUNE_TARGET_HIT_RATE + self.AUTO_TUNE_DEADBAND):
            new_boost = max(self.AUTO_TUNE_MIN_BOOST, current_boost - self.AUTO_TUNE_STEP)
        self.SIGNAL_FREQ_BOOST_PCT = round(new_boost, 2)

        diagnostics["scan_hit_rate"] = round(hit_rate, 4)
        diagnostics["avg_hit_rate_8"] = round(avg_hit_rate, 4)
        diagnostics["next_signal_freq_boost_pct"] = self.SIGNAL_FREQ_BOOST_PCT
        diagnostics["target_hit_rate"] = self.AUTO_TUNE_TARGET_HIT_RATE

        proposals = [
            p for p in proposals
            if p is not None
        ]
        
        # 5. DIVERSIFIED INSTITUTIONAL SELECTION
        # Sort by edge_score but ensure sector variety in the top 15
        # Using a flexible lambda to handle both Proposal objects and potential dictionaries
        def get_score(x):
            if hasattr(x, 'edge_score'): return x.edge_score
            if isinstance(x, dict): return x.get('edge_score', 0)
            return 0

        sorted_props = sorted(proposals, key=get_score, reverse=True)
        diversified = []
        seen_sectors = set()
        
        # Priority 1: Top stock from each unique sector
        for p in sorted_props:
            p_dict = p.__dict__ if hasattr(p, '__dict__') else p
            sector = p_dict.get('sector', 'N/A')
            if sector not in seen_sectors:
                diversified.append(p_dict)
                seen_sectors.add(sector)
        
        # Priority 2: Fill remaining with top stocks regardless of sector
        for p in sorted_props:
            if len(diversified) >= 15: break
            p_dict = p.__dict__ if hasattr(p, '__dict__') else p
            if p_dict not in diversified:
                diversified.append(p_dict)
        
        inst_quality = diversified[:15]
        longs = [p.__dict__ if hasattr(p, '__dict__') else p for p in proposals if (p.direction if hasattr(p, 'direction') else p.get('direction', '')) == "LONG" and (p.strength_pct if hasattr(p, 'strength_pct') else p.get('strength_pct', 0)) > 1.0]
        shorts = [p.__dict__ if hasattr(p, '__dict__') else p for p in proposals if (p.direction if hasattr(p, 'direction') else p.get('direction', '')) == "SHORT" and (p.strength_pct if hasattr(p, 'strength_pct') else p.get('strength_pct', 0)) > 1.0]

        duration = round(time.time() - start, 1)
        logger.info(f"✅ SCAN COMPLETE | Alpha Candidates: {len(proposals)} | Duration: {duration}s")

        return ScanResult(
            longs=longs[:50], 
            shorts=shorts[:50], 
            institutional_quality=inst_quality,
            total_scanned=total, 
            valid_breakouts=len(proposals), 
            duration_seconds=duration,
            diagnostics=diagnostics
        )


# ===============================================
# 12. MAIN UI
# ===============================================

def main_ui():
    st.set_page_config(layout="wide", page_title="Institutional Breakout Terminal v4")

    # --- Header ---
    st.title("🚀 Institutional Breakout Terminal v4")
    st.markdown("*Smart Money Concepts + Breakout Detection + AI Ensemble Scoring*")

    st.sidebar.header("⚙️ Configuration")

    # Scan mode
    scan_mode = st.sidebar.radio(
        "Scan Mode",
        ["Quick Scan (Watchlist)", "Full Market Scan (All Assets)"],
        help="Full Market scans all tickers in the selected asset class.",
    )

    asset_class = st.sidebar.selectbox("Asset Class", ["Stocks", "Crypto", "Forex"])

    # Dynamic symbol loading
    if scan_mode == "Quick Scan (Watchlist)":
        if asset_class == "Stocks":
            default_symbols = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, AMD, META, JPM"
        elif asset_class == "Crypto":
            default_symbols = "BTC-USD, ETH-USD, SOL-USD, XRP-USD, DOGE-USD"
        else:
            default_symbols = "EURUSD=X, GBPUSD=X, USDJPY=X"
    else:
        if asset_class == "Stocks":
            default_symbols = ", ".join(STOCKS_ALL)
        elif asset_class == "Crypto":
            default_symbols = ", ".join(CRYPTO_ALL)
        else:
            default_symbols = ", ".join(FOREX_ALL)

    symbols_raw = st.sidebar.text_area(
        "Symbols (comma separated)",
        default_symbols,
        height=150 if scan_mode == "Full Market Scan (All Assets)" else 100,
    )

    # Parse and clean
    symbols = sorted({s.strip().upper() for s in symbols_raw.split(",") if s.strip()})
    symbols = [s for s in symbols if s and s not in SKIP_SYMBOLS]

    if len(symbols) > 50:
        st.sidebar.warning(f"⚠️ {len(symbols)} assets queued. This may take a minute...")

    # Confidence threshold
    min_confidence = st.sidebar.slider("Min AI Confidence", min_value=0, max_value=100, value=65, step=5)

    # --- Run scan ---
    if st.button("🚀 SCAN FOR BREAKOUTS", type="primary"):
        if not symbols:
            st.warning("No symbols provided.")
            return

        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        proposals = []
        errors = []

        # Use 3 workers max to avoid Yahoo rate limits
        workers = min(3, len(symbols))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_and_build, s): s for s in symbols}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                sym = futures[future]
                progress_bar.progress(
                    int((completed / len(symbols)) * 100),
                    text=f"Analyzing {sym}... ({completed}/{len(symbols)})",
                )
                try:
                    res = future.result()
                    if res:
                        proposals.append(res)
                except Exception as e:
                    errors.append(f"{sym}: {str(e)[:60]}")

        progress_bar.empty()
        def get_confidence(x):
            if isinstance(x, dict):
                return x.get("ai_confidence", 0)
            return getattr(x, "ai_confidence", 0)
        
        def get_confidence(x):
            if isinstance(x, dict):
                return x.get("ai_confidence", 0)
            return getattr(x, "ai_confidence", 0)
        
        # Remove None values (safety)
        proposals = [p for p in proposals if p is not None]
        
        # Single safe sort (DO NOT duplicate this anywhere else)
        proposals.sort(key=get_confidence, reverse=True)

        st.success(f"🎯 Found **{len(proposals)}** qualifying setups out of **{len(symbols)}** scanned.")

        # --- Summary Table ---
        st.subheader("📊 Setup Summary")

        def get_field(p, key, default="UNKNOWN"):
            if isinstance(p, dict):
                return p.get(key, default)
            return getattr(p, key, default)
        
        summary_data = []
        
        for p in proposals:
            summary_data.append({
                "Symbol": get_field(p, "symbol"),
                "Setup": get_field(p, "setup_type"),
                "Direction": get_field(p, "direction"),
                "Confidence": f"{get_field(p, 'ai_confidence', 0)}%",
                "Grade": get_field(p, "ai_grade"),
                "Entry": get_field(p, "entry_price"),
                "Stop Loss": get_field(p, "stop_loss"),
                "TP1": get_field(p, "tp_1"),
                "TP2": get_field(p, "tp_2"),
                "R:R": f"1:{get_field(p, 'risk_reward', 0)}",
                "Size%": f"{get_field(p, 'position_size_pct', 0)}%",
            })
        
        df_summary = pd.DataFrame(summary_data)

        # Style
        def highlight_direction(row):
            if row["Direction"] == "LONG":
                return ["color: #00ff00; font-weight: bold" if c == "Direction" else "" for c in df_summary.columns]
            elif row["Direction"] == "SHORT":
                return ["color: #ff4444; font-weight: bold" if c == "Direction" else "" for c in df_summary.columns]
            return [""] * len(df_summary.columns)

        st.dataframe(
            df_summary.style.apply(highlight_direction, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # --- Detailed Charts ---
        st.subheader("📈 Detailed Setup Charts")
        display_limit = 15

        for idx, p in enumerate(proposals):
            if idx >= display_limit:
                st.info(f"ℹ️ Showing top {display_limit}. {len(proposals) - display_limit} more results available.")
                break

            border_color = "#00ff00" if "BREAKOUT" in p.setup_type else "#ffa500"
            dir_color = "#00ff00" if p.direction == "LONG" else "#ff4444"

            with st.container():
                st.markdown(f"""
                <div style="border:2px solid {border_color}; padding:12px; border-radius:8px; margin-bottom:8px; background-color:#0e1117;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <h3 style="color:white; margin:0;">{p.symbol}
                                <span style="font-size:0.5em; color:gray;">{p.setup_type}</span>
                            </h3>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:1.8em; font-weight:bold; color:{dir_color};">{p.direction}</span>
                            <br>
                            <span style="color:gray;">Grade: <b style="color:white;">{p.ai_grade}</b> ({p.ai_confidence}%)</span>
                        </div>
                    </div>
                    <div style="display:flex; gap:20px; margin-top:8px; font-size:0.85em;">
                        <span>Entry: <b>{p.entry_price}</b></span>
                        <span style="color:#ff4444;">SL: <b>{p.stop_loss}</b></span>
                        <span style="color:#44ff44;">TP1: <b>{p.tp_1}</b></span>
                        <span style="color:#44ff44;">TP2: <b>{p.tp_2}</b></span>
                        <span>R:R <b>1:{p.risk_reward}</b></span>
                        <span>Size: <b>{p.position_size_pct}%</b></span>
                    </div>
                    <div style="margin-top:6px; color:#aaa; font-size:0.8em;">
                        {p.thesis}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Reasons
                if p.reasons:
                    st.caption("📋 " + " | ".join(p.reasons))

                st.plotly_chart(p.chart_data, use_container_width=True)
                st.divider()


if __name__ == "__main__":
    main_ui()
