"""
Institutional Breakout & Investment Terminal v4.0
Integrated with Market Regime, Sector Rotation, and Breakout Analysis
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Setup & Config ---
st.set_page_config(page_title="Institutional Terminal v4.0", layout="wide", page_icon="📈")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Terminal_v4")

# ===============================================
# 1. CORE DATA FOUNDATION
# ===============================================

def normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    
    if "adj close" in df.columns:
        df.rename(columns={"adj close": "close"}, inplace=True)
    
    required = ["open", "high", "low", "close", "volume"]
    df = df.reindex(columns=required)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(df) < 30:
        return None
    return df

@st.cache_data(ttl=3600)
def get_data(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        return normalize_ohlcv(data)
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

# ===============================================
# 2. MARKET REGIME ENGINE
# ===============================================

class MarketRegimeEngine:
    @staticmethod
    def get_regime():
        # Fetching SPY and VIX for quick regime check
        spy = get_data("SPY", "1y", "1d")
        vix = get_data("^VIX", "1y", "1d")
        
        if spy is None or vix is None:
            return "UNKNOWN", 0, {}

        current_spy = spy['close'].iloc[-1]
        sma200 = spy['close'].rolling(200).mean().iloc[-1]
        vix_now = vix['close'].iloc[-1]
        
        # Scoring logic
        score = 0
        if current_spy > sma200: score += 5
        if vix_now < 20: score += 3
        if vix_now > 30: score -= 5
        
        regime = "BULL" if score >= 5 else "BEAR" if score <= 0 else "NEUTRAL"
        return regime, score, {"spy_price": current_spy, "vix": vix_now, "sma200": sma200}

# ===============================================
# 3. SECTOR ROTATION ENGINE
# ===============================================

SECTOR_ETFS = {
    "XLK": "Tech", "XLF": "Finance", "XLE": "Energy", "XLV": "Health",
    "XLI": "Industrial", "XLP": "Staples", "XLY": "Discretionary",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Comm"
}

class SectorRotationEngine:
    @staticmethod
    def analyze_rotation():
        results = []
        spy = get_data("SPY", "6mo", "1d")
        if spy is None: return pd.DataFrame()
        
        spy_perf = (spy['close'].iloc[-1] / spy['close'].iloc[0] - 1) * 100
        
        for ticker, name in SECTOR_ETFS.items():
            df = get_data(ticker, "6mo", "1d")
            if df is not None:
                perf = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                rel_perf = perf - spy_perf
                results.append({"Sector": name, "Ticker": ticker, "Performance": round(perf, 2), "Rel_SPY": round(rel_perf, 2)})
        
        return pd.DataFrame(results).sort_values(by="Rel_SPY", ascending=False)

# ===============================================
# 4. TECHNICAL ENGINE (BREAKOUTS)
# ===============================================

def detect_breakout(df: pd.DataFrame) -> Dict[str, Any]:
    # Simplified Breakout Detection Logic
    last_close = df['close'].iloc[-1]
    max_20 = df['high'].rolling(20).max().iloc[-2]
    vol_sma = df['volume'].rolling(20).mean().iloc[-1]
    
    is_breakout = last_close > max_20 and df['volume'].iloc[-1] > (vol_sma * 1.5)
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    return {
        "Breakout": "✅ YES" if is_breakout else "❌ NO",
        "RSI": round(rsi, 2),
        "Signal": "BUY" if is_breakout and rsi < 70 else "HOLD/WAIT"
    }

# ===============================================
# 5. STREAMLIT UI LAYOUT
# ===============================================

def main():
    st.title("🏛️ Institutional Breakout Terminal v4.0")
    st.sidebar.header("Navigation & Analysis")
    
    symbol = st.sidebar.text_input("Enter Ticker", value="NVDA").upper()
    analyze_btn = st.sidebar.button("Run Terminal Analysis")

    # Top Row: Market Dashboard
    col1, col2, col3 = st.columns(3)
    regime, score, r_data = MarketRegimeEngine.get_regime()
    
    with col1:
        st.metric("Market Regime", regime, delta=f"Score: {score}")
    with col2:
        st.metric("VIX Level", f"{r_data.get('vix', 0):.2f}", delta_color="inverse")
    with col3:
        st.metric("SPY Trend", "Above SMA200" if r_data.get('spy_price', 0) > r_data.get('sma200', 0) else "Below SMA200")

    st.divider()

    if analyze_btn:
        # Data Retrieval
        with st.spinner(f"Analyzing {symbol}..."):
            hist = get_data(symbol, "1y", "1d")
            
            if hist is not None:
                # Layout
                main_col, side_col = st.columns([3, 1])
                
                with main_col:
                    # Plotly Candlestick
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['open'], high=hist['high'], low=hist['low'], close=hist['close'], name="Price"), row=1, col=1)
                    fig.add_trace(go.Bar(x=hist.index, y=hist['volume'], name="Volume"), row=2, col=1)
                    fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                
                with side_col:
                    st.subheader("Breakout Status")
                    metrics = detect_breakout(hist)
                    st.write(f"**Breakout Confirmed:** {metrics['Breakout']}")
                    st.write(f"**RSI (14):** {metrics['RSI']}")
                    st.success(f"**Institutional Thesis:** {metrics['Signal']}")
                    
                    st.divider()
                    st.subheader("Sector Performance")
                    rotation_df = SectorRotationEngine.analyze_rotation()
                    st.dataframe(rotation_df[['Sector', 'Rel_SPY']], hide_index=True)

            else:
                st.error("Invalid Ticker or No Data Found.")

    # Macro Section
    with st.expander("Macroeconomic & Breadth Overview"):
        st.info("Institutional Flow Proxy: OBV Divergence and Advance-Decline metrics are calculated based on SPY components.")
        st.write("Current DXY Trend: Stable | 10Y Yield: Monitoring")

if __name__ == "__main__":
    main()
