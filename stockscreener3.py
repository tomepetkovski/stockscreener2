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
import os
import numpy as np
import math
import requests
import schedule
import threading
import time
from typing import Dict, Any, Optional, List
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from types import SimpleNamespace



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


# =====================================================
# YAHOO FINANCE STDERR-SILENCE HELPERS
# =====================================================
import io as _io
import sys as _sys

_YAHOO_SPAM = (
    "HTTP Error 401",
    "Invalid Crumb",
    "Too Many Requests",
    "YFRateLimitError",
    "CRUMB",
)

_orig_stderr = _sys.stderr

class _YFSilentBuffer(_io.StringIO):
    def write(self, s: str) -> int:
        if any(pat in s for pat in _YAHOO_SPAM):
            return 0
        _orig_stderr.write(s)
        return 0

    def flush(self) -> None:
        pass

_yf_buf = _YFSilentBuffer()

def _yf_call(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with sys.stderr pointed at Yahoo-spam filter."""
    old = _sys.stderr
    _sys.stderr = _yf_buf
    try:
        return fn(*args, **kwargs)
    finally:
        _sys.stderr = old


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
# AUTO-SCAN + TELEGRAM NOTIFICATION INFRASTRUCTURE
# ===============================================

# --- Timing ---
_AUTO_SCAN_INTERVAL_HOURS: float = 4.0
_AUTO_SCAN_INTERVAL_SECS: int  = int(_AUTO_SCAN_INTERVAL_HOURS * 3600)

# --- Telegram configuration (override via env-var) ---
_TG_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "") or "7608970630:AAH5YDKlFdRrxp5pLAwNvoCq4yqIQTTm0yE"
_TG_CHAT_ID:   str = os.environ.get("TELEGRAM_CHAT_ID",   "") or "-1002385326575"
_TG_API_URL:   str = f"https://api.telegram.org/bot{_TG_BOT_TOKEN}"


# ---------------------------------------------------------------------------
# HTML escape helper (used by Telegram notifier)
# ---------------------------------------------------------------------------

def _esc_html(text: str) -> str:
    return text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")


# ---------------------------------------------------------------------------
# TelegramNotifier
# ---------------------------------------------------------------------------

class TelegramNotifier:
    """Thin wrapper around the Telegram Bot HTTP API (HTML parse-mode)."""

    def __init__(self, token: str = _TG_BOT_TOKEN, chat_id: str = _TG_CHAT_ID):
        self.token   = token or _TG_BOT_TOKEN
        self.chat_id = chat_id or _TG_CHAT_ID
        self._base   = f"https://api.telegram.org/bot{self.token}"

    def send(self, text: str) -> bool:
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured — skipping send.")
            return False
        try:
            r = requests.post(
                f"{self._base}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )
            ok = r.status_code == 200
            if not ok:
                logger.error("Telegram HTTP %s: %s", r.status_code, r.text[:200])
            return ok
        except Exception as exc:
            logger.error("Telegram send error: %s", exc)
            return False

    # ------------------------------------------------------------------
    def send_opportunity(self, rank: int, proposal: Any, top_n: int = 10) -> bool:
        """Format one InvestmentProposal as an HTML Telegram block."""
        try:
            p = proposal
            dir_up = str(getattr(p,"direction","")).upper() == "LONG"
            emoji_dir = "🟢 LONG" if dir_up else "🔴 SHORT"
            grade_emoji = {
                "A+":"⭐⭐⭐⭐⭐","A":"⭐⭐⭐⭐","B+":"⭐⭐⭐",
                "B":"⭐⭐","C":"⭐","LOW":"⚪",
            }.get(getattr(p,"ai_grade",""), "⚪")

            def fmt(v, dp=2):
                try: return f"{float(v):.{dp}f}"
                except Exception: return str(v) if v is not None else "N/A"

            tp1_s, tp2_s, tp3_s = fmt(getattr(p,"tp_1",0)), fmt(getattr(p,"tp_2",0)), fmt(getattr(p,"tp_3",0))
            rr_lbl   = f"{safe_num(getattr(p,'risk_reward',1)):.1f}:1"
            rr_ext   = f"{safe_num(getattr(p,'risk_reward_extended',1)):.1f}:1"
            conff    = f"{safe_num(getattr(p,'ai_confidence',0)):.0f}%"
            sym      = _esc_html(getattr(p,"symbol","?"))
            grade    = _esc_html(getattr(p,"ai_grade","—"))

            thesis_lines = (getattr(p,"thesis","") or "No thesis available").strip().splitlines()
            thesis_md    = "\n".join(f"  • {_esc_html(l)}" for l in thesis_lines[:3])
            top_sigs     = (getattr(p,"signals",[]) or [])[:4]
            sig_md       = "\n".join(f"  ✓ {_esc_html(s)}" for s in top_sigs) if top_sigs else "  —"

            ac         = getattr(p,"analyst_consensus",{}) or {}
            ac_label   = ac.get("label","N/A")
            ac_score   = fmt(ac.get("score",0),0)
            hold_per   = _esc_html(getattr(p,"hold_period","—") or "—")

            block = [
                f"<b>#{rank}  {sym}</b>  {emoji_dir}",
                "",
                f"💰 <b>Entry:</b>  <code>{fmt(getattr(p,'entry_price',0))}</code>",
                f"🛑 <b>Stop Loss:</b> <code>{fmt(getattr(p,'stop_loss',0))}</code>",
                f"🎯 <b>TP1:</b> <code>{tp1_s}</code>  <b>TP2:</b> <code>{tp2_s}</code>  <b>TP3:</b> <code>{tp3_s}</code>",
                f"📊 <b>RR:</b> {rr_lbl} (ext: {rr_ext})",
                f"⭐ <b>Grade:</b> {grade_emoji}  {grade}  ({conff})",
                f"📰 <b>Analysts:</b> {ac_label} ({ac_score})",
                f"⏱ <b>Hold:</b> {hold_per}",
                "",
                f"📌 <b>Thesis:</b>",
                thesis_md,
                "",
                f"🔑 <b>Signals:</b>",
                sig_md,
                "",
                f"<i>For informational purposes only — not financial advice.</i>",
            ]
            return self.send("\n".join(block))
        except Exception as exc:
            logger.error("send_opportunity error: %s", exc)
            return False

    # ------------------------------------------------------------------
    def send_header(self, market_regime: Dict[str, Any], total_scanned: int = 0,
                    top10_mix: Optional[Dict[str, int]] = None, top_n: int = 10) -> bool:
        regime = str(market_regime.get("_overall","UNKNOWN"))
        score  = str(market_regime.get("_score","?"))
        bias   = (market_regime.get("_bias") or {}).get("direction","—")
        ts     = _esc_html(
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        )

        emoji = {
            "STRONG_BULL":"🚀","BULL":"📈","NEUTRAL":"➡️",
            "CAUTIOUS":"⚠️","BEAR":"📉","STRONG_BEAR":"🔻",
        }.get(regime,"❓")

        lines = [f"{emoji} <b>Scanner Report — Top {top_n} Opportunities</b>",
                 f"📊 Market: <b>{_esc_html(regime)}</b> (score {score})  |  Bias: <b>{_esc_html(str(bias))}</b>"]
        if top10_mix:
            mix_str = "  ".join(f"<b>{k}</b>: {v}" for k,v in sorted(top10_mix.items(), key=lambda x:-x[1]))
            lines.append(f"🏗 Top-10 mix → {mix_str}")
        lines.append(f"🌐 Universe: {total_scanned} US equities  |  🕐 {ts}")
        return self.send("\n".join(lines))

    def send_footer(self, n_shown: int) -> bool:
        return self.send(
            f"ℹ️ Showing top {n_shown} opportunities.\n"
            f"⚠️ <i>For informational purposes only. Not financial advice.</i>"
        )


# ---------------------------------------------------------------------------
# Stock universe (sourced from telegram_scanner.py — single source of truth)
# ---------------------------------------------------------------------------

_UNIVERSE_CATALOG: Dict[str, Dict[str, str]] = {
    # ── MEGA TECHNOLOGY ($300B+) ──────────────────────────────────────────────
    "AAPL":{"opp":"MOMENTUM","cap":"MEGA","sector":"Technology"},
    "MSFT":{"opp":"GROWTH","cap":"MEGA","sector":"Technology"},
    "NVDA":{"opp":"BREAKOUT","cap":"MEGA","sector":"Semiconductors"},
    "GOOGL":{"opp":"MOMENTUM","cap":"MEGA","sector":"Technology"},
    "GOOG":{"opp":"MOMENTUM","cap":"MEGA","sector":"Technology"},
    "AMZN":{"opp":"BREAKOUT","cap":"MEGA","sector":"Consumer Disc."},
    "META":{"opp":"BREAKOUT","cap":"MEGA","sector":"Technology"},
    "TSLA":{"opp":"BREAKOUT","cap":"MEGA","sector":"Consumer Disc."},
    "AVGO":{"opp":"GROWTH","cap":"MEGA","sector":"Semiconductors"},
    "ORCL":{"opp":"MOMENTUM","cap":"MEGA","sector":"Technology"},
    "BRK-B":{"opp":"VALUE","cap":"MEGA","sector":"Financials"},
    "TSM":{"opp":"GROWTH","cap":"MEGA","sector":"Semiconductors"},
    "CRM":{"opp":"MOMENTUM","cap":"LARGE","sector":"Technology"},
    "ADBE":{"opp":"GROWTH","cap":"LARGE","sector":"Technology"},
    "AMD":{"opp":"BREAKOUT","cap":"LARGE","sector":"Semiconductors"},
    "INTC":{"opp":"TURNAROUND","cap":"LARGE","sector":"Semiconductors"},
    "PLTR":{"opp":"BREAKOUT","cap":"LARGE","sector":"Technology"},
    "NOW":{"opp":"GROWTH","cap":"LARGE","sector":"Technology"},
    "UBER":{"opp":"MOMENTUM","cap":"LARGE","sector":"Technology"},
    "SHOP":{"opp":"BREAKOUT","cap":"LARGE","sector":"Technology"},
    "PANW":{"opp":"MOMENTUM","cap":"LARGE","sector":"Technology"},
    "SNOW":{"opp":"GROWTH","cap":"LARGE","sector":"Technology"},
    "ARM":{"opp":"GROWTH","cap":"LARGE","sector":"Semiconductors"},
    "MSTR":{"opp":"BREAKOUT","cap":"LARGE","sector":"Financials"},
    "COIN":{"opp":"BREAKOUT","cap":"LARGE","sector":"Financials"},
    "JPM":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "BAC":{"opp":"VALUE","cap":"LARGE","sector":"Financials"},
    "WFC":{"opp":"TURNAROUND","cap":"LARGE","sector":"Financials"},
    "GS":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "MS":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "BLK":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "SCHW":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "AXP":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "C":{"opp":"VALUE","cap":"LARGE","sector":"Financials"},
    "V":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "MA":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "PYPL":{"opp":"TURNAROUND","cap":"LARGE","sector":"Financials"},
    "UNH":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "LLY":{"opp":"BREAKOUT","cap":"LARGE","sector":"Healthcare"},
    "JNJ":{"opp":"VALUE","cap":"LARGE","sector":"Healthcare"},
    "PFE":{"opp":"TURNAROUND","cap":"LARGE","sector":"Healthcare"},
    "ABBV":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "MRK":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "ABT":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "TMO":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "DHR":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "NVO":{"opp":"GROWTH","cap":"LARGE","sector":"Healthcare"},
    "AZN":{"opp":"GROWTH","cap":"LARGE","sector":"Healthcare"},
    "BMY":{"opp":"VALUE","cap":"LARGE","sector":"Healthcare"},
    "GILD":{"opp":"TURNAROUND","cap":"LARGE","sector":"Healthcare"},
    "BIIB":{"opp":"BREAKOUT","cap":"LARGE","sector":"Healthcare"},
    "VRTX":{"opp":"BREAKOUT","cap":"LARGE","sector":"Healthcare"},
    "WMT":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Staples"},
    "COST":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Staples"},
    "PG":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Staples"},
    "KO":{"opp":"VALUE","cap":"LARGE","sector":"Consumer Staples"},
    "PEP":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Staples"},
    "MCD":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Disc."},
    "NKE":{"opp":"TURNAROUND","cap":"LARGE","sector":"Consumer Disc."},
    "HD":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Disc."},
    "LOW":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Disc."},
    "TGT":{"opp":"TURNAROUND","cap":"LARGE","sector":"Consumer Disc."},
    "SBUX":{"opp":"TURNAROUND","cap":"MID","sector":"Consumer Disc."},
    "CMG":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Disc."},
    "CAT":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "DE":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "GE":{"opp":"BREAKOUT","cap":"LARGE","sector":"Industrials"},
    "UPS":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "BA":{"opp":"TURNAROUND","cap":"LARGE","sector":"Industrials"},
    "HON":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "RTX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "LMT":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "XOM":{"opp":"MOMENTUM","cap":"LARGE","sector":"Energy"},
    "CVX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Energy"},
    "COP":{"opp":"MOMENTUM","cap":"LARGE","sector":"Energy"},
    "SLB":{"opp":"BREAKOUT","cap":"LARGE","sector":"Energy"},
    "TXN":{"opp":"MOMENTUM","cap":"LARGE","sector":"Semiconductors"},
    "QCOM":{"opp":"MOMENTUM","cap":"LARGE","sector":"Semiconductors"},
    "MU":{"opp":"BREAKOUT","cap":"LARGE","sector":"Semiconductors"},
    "AMAT":{"opp":"BREAKOUT","cap":"LARGE","sector":"Semiconductors"},
    "LRCX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Semiconductors"},
    "KLAC":{"opp":"MOMENTUM","cap":"LARGE","sector":"Semiconductors"},
    "ASML":{"opp":"GROWTH","cap":"LARGE","sector":"Semiconductors"},
    "NFLX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Comm. Services"},
    "DIS":{"opp":"TURNAROUND","cap":"LARGE","sector":"Comm. Services"},
    "CMCSA":{"opp":"VALUE","cap":"LARGE","sector":"Comm. Services"},
    "VZ":{"opp":"VALUE","cap":"LARGE","sector":"Comm. Services"},
    "NEE":{"opp":"GROWTH","cap":"LARGE","sector":"Utilities"},
    "DUK":{"opp":"VALUE","cap":"LARGE","sector":"Utilities"},
    "SO":{"opp":"VALUE","cap":"LARGE","sector":"Utilities"},
    "D":{"opp":"VALUE","cap":"LARGE","sector":"Utilities"},
    "AEP":{"opp":"VALUE","cap":"LARGE","sector":"Utilities"},
    "EXC":{"opp":"VALUE","cap":"LARGE","sector":"Utilities"},
    "CEG":{"opp":"BREAKOUT","cap":"LARGE","sector":"Utilities"},
    "SRE":{"opp":"MOMENTUM","cap":"LARGE","sector":"Utilities"},
    "LIN":{"opp":"MOMENTUM","cap":"LARGE","sector":"Materials"},
    "APD":{"opp":"MOMENTUM","cap":"LARGE","sector":"Materials"},
    "SHW":{"opp":"MOMENTUM","cap":"LARGE","sector":"Materials"},
    "FCX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Materials"},
    "PLD":{"opp":"MOMENTUM","cap":"LARGE","sector":"Real Estate"},
    "EQIX":{"opp":"GROWTH","cap":"LARGE","sector":"Real Estate"},
    "CRWD":{"opp":"BREAKOUT","cap":"MID","sector":"Technology"},
    "ZS":{"opp":"GROWTH","cap":"MID","sector":"Technology"},
    "DDOG":{"opp":"GROWTH","cap":"MID","sector":"Technology"},
    "FTNT":{"opp":"BREAKOUT","cap":"MID","sector":"Technology"},
    "NET":{"opp":"BREAKOUT","cap":"MID","sector":"Technology"},
    "APP":{"opp":"BREAKOUT","cap":"MID","sector":"Technology"},
    "DKNG":{"opp":"BREAKOUT","cap":"LARGE","sector":"Consumer Disc."},
    "CCL":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Disc."},
    "MAR":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Disc."},
    "HLT":{"opp":"GROWTH","cap":"LARGE","sector":"Consumer Disc."},
    "LULU":{"opp":"BREAKOUT","cap":"LARGE","sector":"Consumer Disc."},
    "OXY":{"opp":"MOMENTUM","cap":"LARGE","sector":"Energy"},
    "SMCI":{"opp":"BREAKOUT","cap":"MID","sector":"Technology"},
    "LLY1":{"opp":"BREAKOUT","cap":"MID","sector":"Healthcare"},
    "BMRN":{"opp":"BREAKOUT","cap":"LARGE","sector":"Healthcare"},
    "VRTX1":{"opp":"BREAKOUT","cap":"LARGE","sector":"Healthcare"},
    "SYK":{"opp":"MOMENTUM","cap":"LARGE","sector":"Healthcare"},
    "CATY":{"opp":"MOMENTUM","cap":"MID","sector":"Financials"},
    "FITB":{"opp":"VALUE","cap":"LARGE","sector":"Financials"},
    "PGR":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "TRV":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "ALL":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "SPGI":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "MSCI":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "ICE":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "CME":{"opp":"MOMENTUM","cap":"LARGE","sector":"Financials"},
    "MDB":{"opp":"BREAKOUT","cap":"MID","sector":"Technology"},
    "MSTR":{"opp":"BREAKOUT","cap":"LARGE","sector":"Financials"},
    "DOCU":{"opp":"TURNAROUND","cap":"LARGE","sector":"Technology"},
    "TEAM":{"opp":"TURNAROUND","cap":"MID","sector":"Technology"},
    "ZM":{"opp":"TURNAROUND","cap":"LARGE","sector":"Technology"},
    "SQ":{"opp":"TURNAROUND","cap":"LARGE","sector":"Technology"},
    "PYPL":{"opp":"TURNAROUND","cap":"LARGE","sector":"Financials"},
    "COST":{"opp":"MOMENTUM","cap":"LARGE","sector":"Consumer Staples"},
    "TXN":{"opp":"MOMENTUM","cap":"LARGE","sector":"Semiconductors"},
    "WMB":{"opp":"GROWTH","cap":"LARGE","sector":"Energy"},
    "FDX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "FAST":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "PAYX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Technology"},
    "EFX":{"opp":"MOMENTUM","cap":"LARGE","sector":"Technology"},
    "CTAS":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
    "ODFL":{"opp":"MOMENTUM","cap":"LARGE","sector":"Industrials"},
}

# Canonical, deduplicated, ordered scan list
_STOCK_UNIVERSE: List[str] = list(dict.fromkeys(_UNIVERSE_CATALOG.keys()))

# Rank/score belt thresholds (echo telegram_scanner.py)
_CAP_TIER_WEIGHT: Dict[str, float] = {"MEGA":1.0,"LARGE":2.0,"MID":5.0,"SMALL":9.0}
_OPP_BOOST:       Dict[str, float] = {
    "BREAKOUT":12.0,"MOMENTUM":9.0,"GROWTH":7.0,"VALUE":4.0,"TURNAROUND":6.0,
}

# Shared scan constants
_SCAN_WORKERS:       int = 8
_SCAN_TIMEOUT_S:     int = 30
_MIN_CONFIDENCE:     float = 48.0

# Auto-scan state
_LAST_SCAN_AT:   Optional[datetime] = None
_AUTO_SCAN_LOCK = threading.Lock()          # prevents overlapping runs
_auto_scan_log:  List[str] = []             # bounded circular buffer of log lines


# ---------------------------------------------------------------------------
# Reusable scan runner + Telegram dispatcher (called by Streamlit toggle AND
# by telegram_scanner.py --once / --scheduled modes)
# ---------------------------------------------------------------------------

def _run_scan_job(
    notifier:   TelegramNotifier,
    scanner:    "MarketScanner",
    top_n:      int = 10,
    min_conf:   float = _MIN_CONFIDENCE,
    send_empty: bool = False,
) -> None:
    """Execute one full scan and push results to Telegram."""
    global _LAST_SCAN_AT

    if not _AUTO_SCAN_LOCK.acquire(blocking=False):
        logger.info("Scan already in progress — skipping this tick.")
        return

    ts_start = datetime.now(timezone.utc)
    try:
        logger.info(
            "Scan job fired — %s", ts_start.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        _auto_scan_log.append(
            f"[{ts_start.strftime('%H:%M:%S')}] Scan started"
        )
        if len(_auto_scan_log) > 50:
            _auto_scan_log.pop(0)

        ranked = scanner.scan()

        if not ranked:
            if send_empty:
                notifier.send(
                    f"📭 <b>Scan complete — no qualifying opportunities</b>\n"
                    f"🕐 {ts_start.strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"All {len(scanner.universe)} symbols below confidence floor."
                )
            logger.info("No proposals. Silent skip (send_empty=%s).", send_empty)
            return

        # Category mix for header
        mix: Dict[str, int] = {}
        for p in ranked[:top_n]:
            key = f"{getattr(p,'_cap_tier','?')}/{getattr(p,'_opp_type','?')}"
            mix[key] = mix.get(key, 0) + 1

        # Header → each opportunity → footer
        notifier.send_header(
            {"_overall":"RUN","_score":0,"_bias":{"direction":"—"}},
            total_scanned=len(scanner.universe),
            top10_mix=mix,
            top_n=top_n,
        )
        for rank, p in enumerate(ranked[:top_n], 1):
            notifier.send_opportunity(rank, p, top_n=top_n)
            time.sleep(0.4)  # Telegram rate-limit friendly
        notifier.send_footer(min(top_n, len(ranked)))

        logger.info(
            "Telegram batch sent — %d proposals, %d shared.",
            len(ranked[:top_n]), len(mix),
        )
        _auto_scan_log.append(
            f"[{ts_start.strftime('%H:%M:%S')}] Sent {min(top_n,len(ranked))} / {len(ranked)}"
        )
    except Exception:
        logger.exception("scan_job crashed")
        _auto_scan_log.append(
            f"[{ts_start.strftime('%H:%M:%S')}] CRASHED — see logs"
        )
    finally:
        _LAST_SCAN_AT = datetime.now(timezone.utc)
        _AUTO_SCAN_LOCK.release()


# ---------------------------------------------------------------------------
# Background daemon thread — runs schedule loop forever
# ---------------------------------------------------------------------------

def _auto_scan_daemon() -> None:
    """Background thread: scan every _AUTO_SCAN_INTERVAL_HOURS and push Telegram alerts."""
    sched = schedule.Scheduler()

    notifier = TelegramNotifier()
    scanner  = MarketScanner(workers=_SCAN_WORKERS)

    # Fire immediately so the first scan isn't delayed by one interval
    sched.every(_AUTO_SCAN_INTERVAL_HOURS).hours.do(
        _run_scan_job, notifier, scanner
    )
    logger.info(
        "Auto-scan: every %.1f h  |  Freq: %d workers  |  Min confidence: %.0f",
        _AUTO_SCAN_INTERVAL_HOURS, _SCAN_WORKERS, _MIN_CONFIDENCE,
    )

    while True:
        try:
            sched.run_pending()
        except Exception:
            logger.exception("Auto-scan scheduler error")
        time.sleep(30)


def start_auto_scan(
    interval_hours: float = _AUTO_SCAN_INTERVAL_HOURS,
    workers:        int   = _SCAN_WORKERS,
    min_conf:       float = _MIN_CONFIDENCE,
) -> threading.Thread | None:
    """Start the background auto-scan daemon thread (once only)."""
    global _AUTO_SCAN_INTERVAL_HOURS, _AUTO_SCAN_INTERVAL_SECS
    global _SCAN_WORKERS, _MIN_CONFIDENCE

    _AUTO_SCAN_INTERVAL_HOURS = float(interval_hours)
    _AUTO_SCAN_INTERVAL_SECS  = int(_AUTO_SCAN_INTERVAL_HOURS * 3600)
    _SCAN_WORKERS              = int(workers)
    _MIN_CONFIDENCE            = float(min_conf)

    th = threading.Thread(target=_auto_scan_daemon, daemon=True, name="AutoScan")
    th.start()
    logger.info(
        "Auto-scan daemon started — interval=%.1f h", _AUTO_SCAN_INTERVAL_HOURS,
    )
    return th


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

        if df is None or len(df) < 60:
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
            # Additional fundamentals for quality scoring
            roe = info.get("returnOnEquity", 0) or 0
            net_income = info.get("netIncomeToCommon", 0) or info.get("netIncome", 0) or 0
            
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

            # --- Buffett-style quality score ---
            buffett = BuffettAnalyzer.analyze(info)
            buffett_score = buffett["buffett_score"]

            # --- Piotroski F-Score (0-9) ---
            # Simplified checks (no full balance sheet history available)
            f_score = 0
            try:
                # Profitability
                if net_income and net_income > 0:
                    f_score += 1
                if info.get("operatingCashflow", 0) > 0:
                    f_score += 1
                if roe and roe > 0.10:
                    f_score += 1
                # Leverage / Liquidity
                if info.get("debtToEquity", 999) < 0.5:
                    f_score += 1
                if info.get("currentRatio", 0) > 1.0:
                    f_score += 1
                # Operating efficiency (ROA improvement proxy)
                if roe and roe > 0.12:
                    f_score += 1
            except Exception:
                pass

            # --- Altman Z-Score (approximation) ---
            z_score = None
            try:
                ta = info.get("totalAssets") or 1e9
                wc = (info.get("totalCurrentAssets", 0) or 0) - (info.get("totalCurrentLiabilities", 0) or 0)
                x1 = wc / ta
                x2 = (info.get("retainedEarnings", 0) or 0) / ta
                ebit = (info.get("ebit", 0) or 0)
                x3 = ebit / ta
                tde = (info.get("totalDebt", 0) or 0)
                x4 = ((info.get("marketCap", 0) or 1) / (tde or 1))
                rev = (info.get("totalRevenue", 0) or 0)
                x5 = rev / ta
                z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
                z_score = float(z)
            except Exception:
                z_score = None

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
                "analyst_rec": rec,
                # Buffett / Munger quality
                "buffett_score": buffett_score,
                "buffett_rating": buffett["buffett_rating"],
                "piotroski_f": f_score,
                "altman_z": round(z_score, 2) if z_score else None,
                # Additional fields needed by BuffettAnalyzer
                "roe": roe,
                "eps_growth": growth * 100,
                "free_cashflow": info.get("freeCashflow"),
                "current_price": info.get("currentPrice"),
                "target_mean_price": info.get("targetMeanPrice"),
                "held_percent_insiders": info.get("heldPercentInsiders"),
                "held_percent_institutions": info.get("heldPercentInstitutions"),
                "operating_margin": profit_margin,
            }
        except Exception as e:
            logger.debug(f"Fundamentals failed for {symbol}: {e}")
            return {"score": 50, "beta": 1.0}


# ===============================================
# 8. BUFFETT / MUNGER QUALITY ANALYZER
# ===============================================

class BuffettAnalyzer:
    """
    Institutional-quality fundamental scoring inspired by Warren Buffett.
    Nine pillars: ROE, Debt, EPS Growth, FCF, Operating Margin, P/E,
    Competitive Moat, Management Quality, Intrinsic Value Margin.
    Returns total 0-100 score + per-pillar breakdown +Rating.
    """

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @classmethod
    def analyze(cls, info: Dict[str, Any]) -> Dict[str, Any]:
        scores = {}

        # --- 1. ROE ---
        roe = cls._safe_float(info.get("returnOnEquity"))
        if roe >= 0.20:
            scores["roe"] = 10
        elif roe >= 0.15:
            scores["roe"] = 8
        elif roe >= 0.10:
            scores["roe"] = 5
        else:
            scores["roe"] = 0

        # --- 2. Debt-to-Equity ---
        dte = cls._safe_float(info.get("debtToEquity"))
        if dte < 0.3:
            scores["debt"] = 10
        elif dte < 0.5:
            scores["debt"] = 8
        elif dte < 1.0:
            scores["debt"] = 5
        else:
            scores["debt"] = 0

        # --- 3. EPS Growth (YoY) ---
        eps_g = cls._safe_float(info.get("earningsQuarterlyGrowth"), 0) * 100
        if eps_g > 15:
            scores["eps_growth"] = 10
        elif eps_g > 10:
            scores["eps_growth"] = 8
        elif eps_g > 5:
            scores["eps_growth"] = 5
        else:
            scores["eps_growth"] = 0

        # --- 4. Free Cash Flow (positive) ---
        fcf = cls._safe_float(info.get("freeCashflow"))
        scores["fcf"] = 10 if fcf > 0 else 0

        # --- 5. Operating Margin ---
        op_margin = cls._safe_float(info.get("profitMargins"))
        if op_margin > 0.25:
            scores["margin"] = 10
        elif op_margin > 0.15:
            scores["margin"] = 7
        elif op_margin > 0.10:
            scores["margin"] = 5
        else:
            scores["margin"] = 0

        # --- 6. P/E Ratio ---
        pe = cls._safe_float(info.get("forwardPE"))
        if pe < 15:
            scores["pe"] = 10
        elif pe < 25:
            scores["pe"] = 7
        elif pe < 35:
            scores["pe"] = 4
        else:
            scores["pe"] = 0

        # --- 7. Competitive Moat (durability proxy) ---
        moat_score = 0
        if roe > 0.15 and op_margin > 0.15 and eps_g > 5:
            moat_score = 10  # durable competitive advantages
        elif roe > 0.12 and op_margin > 0.10:
            moat_score = 6
        else:
            moat_score = 2
        scores["moat"] = moat_score

        # --- 8. Management Quality (insider + institutional ownership) ---
        mgmt_score = 0
        inst_held = cls._safe_float(info.get("heldPercentInstitutions"))
        ins_held = cls._safe_float(info.get("heldPercentInsiders"))
        if ins_held > 0.1 or inst_held > 0.7:
            mgmt_score = 10  # strong alignment
        elif ins_held > 0.05 or inst_held > 0.5:
            mgmt_score = 6
        else:
            mgmt_score = 3
        scores["management"] = mgmt_score

        # --- 9. Intrinsic Value Margin (analyst target vs current) ---
        current_price = cls._safe_float(info.get("currentPrice"))
        target_price = cls._safe_float(info.get("targetMeanPrice"))
        if current_price > 0 and target_price and target_price > 0:
            margin = (target_price - current_price) / current_price
            if margin > 0.40:
                scores["intrinsic"] = 10
            elif margin > 0.25:
                scores["intrinsic"] = 8
            elif margin > 0.10:
                scores["intrinsic"] = 5
            else:
                scores["intrinsic"] = 0
        else:
            scores["intrinsic"] = 0

        total = sum(scores.values())

        return {
            "buffett_score": round(total, 2),
            "buffett_max": 90.0,
            "buffett_rating": cls._rating(total),
            "pillars": scores,
        }

    @staticmethod
    def _rating(total_score: float) -> str:
        if total_score >= 75:
            return "STRONG BUY"
        if total_score >= 60:
            return "BUY"
        if total_score >= 45:
            return "HOLD"
        return "AVOID"


# ===============================================
# 9. SENTIMENT ENGINE
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
    def predict(
        features: Dict,
        market_regime: Dict = None,
        ml_forecast: Dict = None,
    ) -> Dict:
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
        aligned = features.get("smc_alignment", False)
        htf_aligned = features.get("higher_timeframe_alignment", False)
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
            squeeze_duration = features.get("squeeze_duration", 0)
            if squeeze_duration > 10:
                squeeze_bonus += 4  # Extended squeeze = more energy
            weights["squeeze"] = squeeze_bonus
            score += squeeze_bonus
            signals.append(f"Volatility squeeze ({squeeze_duration} days)")

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
        if features.get("volume_confirmed"):
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

        # --- 12. Buffett Quality Score (Weight: 15) ---
        quality_score_val = features.get("fundamental_quality", 50)  # 0-90 scale
        quality_edge = (quality_score_val - 45) * 0.35  # 45→0, 90→+15.75 capped
        quality_edge = max(-12.0, min(15.0, quality_edge))
        weights["quality"] = round(quality_edge, 1)
        score += quality_edge

        # --- 13. Market Regime Filter (Weight: Variable) ---
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

        # --- 13. ORDER-FLOW PROXY (Tape / Bar Microstructure) ---
        of_score = 0.0
        try:
            # Core metrics
            imb = float(features.get("of_imbalance", 0.5) or 0.5)
            imb_20 = float(features.get("of_imbalance_20", 0.5) or 0.5)
            delta_trend = features.get("of_delta_trend", "NEUTRAL")
            absorption = features.get("of_absorption_flag", False)
            absorption_strength = float(features.get("of_absorption_strength", 0.0) or 0.0)
            inst_footprint = features.get("of_institutional_footprint", "NEUTRAL")
            liquidity_grab = features.get("of_liquidity_grab", False)
            stop_run = features.get("of_stop_run", False)
            bias_lbl = features.get("of_bias", "NEUTRAL")
            delta_pos = features.get("of_delta_positive", False)

            # Base imbalance scoring (short-term pressure)
            of_score = (imb - 0.5) * 14.0

            # Trend-stacked imbalance (medium-term pressure)
            if imb_20 > 0.56:
                of_score += 6.0  # Sustained buying pressure
                signals.append(f"Imbalance 20-bar: {imb_20:.2f} (bullish)")
            elif imb_20 < 0.44:
                of_score -= 6.0  # Sustained selling pressure
                signals.append(f"Imbalance 20-bar: {imb_20:.2f} (bearish)")

            # Delta trend acceleration
            if delta_trend == "BULLISH_ACCELERATING":
                of_score += 5.0
                signals.append("Delta acceleration BULLISH")
            elif delta_trend == "BEARISH_ACCELERATING":
                of_score -= 5.0
                signals.append("Delta acceleration BEARISH")
            elif delta_trend == "BULLISH_DECELERATING":
                of_score += 2.0
                signals.append("Delta decelerating (bullish)")
            elif delta_trend == "BEARISH_DECELERATING":
                of_score -= 2.0
                signals.append("Delta decelerating (bearish)")

            # Absorption quality
            if absorption:
                inst_add = 0.0
                if inst_footprint == "HEAVY":
                    inst_add = 5.0
                    signals.append("Heavy institutional footprint + absorption")
                elif inst_footprint == "MODERATE":
                    inst_add = 3.0
                    signals.append("Moderate institutional footprint")
                else:
                    inst_add = 1.5
                of_score += inst_add if bias_lbl == "BULLISH" else -inst_add

            # Liquidity grab / stop run
            if liquidity_grab:
                if stop_run:
                    of_score += 3.0 if bias_lbl == "BULLISH" else -3.0
                    signals.append("Stop-run detected (institutional sweep)")
                else:
                    of_score += 1.5 if bias_lbl == "BULLISH" else -1.5
                    signals.append("Liquidity grab detected")

            # Delta direction
            if delta_pos and imb > 0.52:
                of_score += 2.5
                signals.append("Positive signed volume delta")
            elif not delta_pos and imb < 0.48:
                of_score -= 2.5
                signals.append("Negative signed volume delta")

            # Bias summary
            if bias_lbl == "BULLISH":
                signals.append("Order-flow bias: BULLISH")
            elif bias_lbl == "BEARISH":
                signals.append("Order-flow bias: BEARISH")

            # Cap order-flow contribution
            of_score = max(-12.0, min(12.0, of_score))
            weights["order_flow"] = round(of_score, 1)
            score += of_score
        except Exception:
            of_score = 0.0
            weights["order_flow"] = 0.0

        # --- 14. ML prognosis (history-trained probabilities) ---
        if ml_forecast and ml_forecast.get("trained"):
            pl = float(ml_forecast.get("p_long", 1.0 / 3.0))
            ps = float(ml_forecast.get("p_short", 1.0 / 3.0))
            ml_edge = (pl - ps) * 35.0
            weights["ml_ensemble"] = round(ml_edge, 1)
            score += ml_edge * 0.35
            if pl > 0.42 and pl >= ps:
                cv_a = ml_forecast.get("cv_accuracy_mean")
                cv_s = f"{cv_a:.3f}" if cv_a is not None else "n/a"
                signals.append(f"ML ensemble favors LONG (p={pl:.2f}; CV acc≈{cv_s})")
            elif ps > 0.42 and ps > pl:
                cv_a = ml_forecast.get("cv_accuracy_mean")
                cv_s = f"{cv_a:.3f}" if cv_a is not None else "n/a"
                signals.append(f"ML ensemble favors SHORT (p={ps:.2f}; CV acc≈{cv_s})")

        # --- FINAL SCORING ---
        score = float(np.clip(score, 0, 100))

        if score >= 50:
            direction = "LONG"
        elif score <= 35:
            direction = "SHORT"
        else:
            direction = "NEUTRAL"

        if ml_forecast and ml_forecast.get("trained"):
            pl = float(ml_forecast.get("p_long", 1.0 / 3.0))
            ps = float(ml_forecast.get("p_short", 1.0 / 3.0))
            if direction == "LONG" and pl < 0.26 and ps > pl + 0.08:
                if score >= 58:  # 8-pt buffer keeps ≥50 after penalty
                    score = max(0.0, score - 8.0)
                    signals.append("ML conflict: probabilities lean SHORT vs rule-based LONG")
            elif direction == "SHORT" and ps < 0.26 and pl > ps + 0.08:
                score = max(0.0, score - 8.0)  # Safe: lowers score further into SHORT
                signals.append("ML conflict: probabilities lean LONG vs rule-based SHORT")
            score = float(np.clip(score, 0, 100))
            if score >= 50:
                direction = "LONG"
            elif score <= 35:
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
# 9b. ORDER FLOW (OHLCV / TAPE PROXY)
# ===============================================

class OrderFlowEngine:
    """
    Bar-level order-flow proxies when tick data is unavailable:
    buying pressure, signed volume delta, cumulative delta slope, absorption.
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        base = {
            "of_imbalance": 0.5,
            "of_buy_pressure_20": 0.5,
            "of_delta_10": 0.0,
            "of_cum_delta_slope": 0.0,
            "of_absorption_flag": False,
            "of_bias": "NEUTRAL",
            "of_score_component": 0.0,
            "of_vol_ratio": 1.0,
            "of_delta_positive": False,
            "of_summary": "Insufficient data",
            # New detailed metrics
            "of_imbalance_5": 0.5,
            "of_imbalance_20": 0.5,
            "of_delta_trend": "NEUTRAL",
            "of_absorption_strength": 0.0,
            "of_liquidity_grab": False,
            "of_stop_run": False,
            "of_institutional_footprint": "NEUTRAL",
            "of_effort_ratio": 0.0,
        }
        df = normalize_ohlcv(df)
        if df is None or df.empty or len(df) < 15:
            return dict(base)
        try:
            c = df["close"].astype(float)
            h = df["high"].astype(float)
            low = df["low"].astype(float)
            v = df["volume"].astype(float).clip(lower=0.0)
            rng = (h - low).replace(0, np.nan)
            bp = ((c - low) / rng).clip(0.0, 1.0).fillna(0.5)
            signed = ((2.0 * bp - 1.0) * v).fillna(0.0)
            d10 = float(signed.iloc[-10:].sum())
            cum = signed.cumsum()
            slope = float(cum.iloc[-1] - cum.iloc[-6]) if len(cum) > 6 else 0.0
            slope_20 = float(cum.iloc[-1] - cum.iloc[-20]) if len(cum) > 20 else 0.0

            # Determine delta trend
            if slope > 0 and slope_20 > 0:
                delta_trend = "BULLISH_ACCELERATING"
            elif slope > 0 and slope_20 <= 0:
                delta_trend = "BULLISH_DECELERATING"
            elif slope < 0 and slope_20 < 0:
                delta_trend = "BEARISH_ACCELERATING"
            elif slope < 0 and slope_20 >= 0:
                delta_trend = "BEARISH_DECELERATING"
            else:
                delta_trend = "NEUTRAL"

            imb_ser = (bp * v).rolling(5, min_periods=3).sum() / (
                v.rolling(5, min_periods=3).sum() + 1e-12
            )
            imb = float(np.clip(imb_ser.iloc[-1], 0.0, 1.0))

            # Multi-timeframe imbalance
            imb_ser_20 = (bp * v).rolling(20, min_periods=10).sum() / (
                v.rolling(20, min_periods=10).sum() + 1e-12
            )
            imb_20 = float(np.clip(imb_ser_20.iloc[-1], 0.0, 1.0))

            buy_p20 = float(bp.iloc[-20:].mean()) if len(bp) >= 20 else float(bp.mean())

            tr = (h - low).rolling(14, min_periods=5).mean()
            rng_ratio = float((h.iloc[-1] - low.iloc[-1]) / (float(tr.iloc[-1]) + 1e-12))
            vma = float(v.rolling(20).mean().iloc[-1]) + 1e-12
            vol_ratio = float(v.iloc[-1] / vma)

            # Absorption strength (volume vs range)
            absorption = rng_ratio < 0.65 and vol_ratio > 1.25
            absorption_strength = (vol_ratio / (rng_ratio + 1e-12)) if absorption else 0.0

            # Liquidity grab detection (wicks vs body)
            upper_wick = h - c
            lower_wick = c - low
            body = abs(c - c.shift(1).fillna(c))
            liquidity_grab = (upper_wick.iloc[-1] > 2 * body.iloc[-1]) or (lower_wick.iloc[-1] > 2 * body.iloc[-1])

            # Stop run detection (wicks + volume spike)
            stop_run = absorption and liquidity_grab and vol_ratio > 1.5

            # Institutional footprint (large trader proxy)
            # High volume + low range + price near high/low suggests institutional activity
            inst_score = 0.0
            if vol_ratio > 1.3 and rng_ratio < 0.5:
                inst_score += 2.0
            if abs(slope_20) > v.iloc[-20:].sum() * 0.1:
                inst_score += 1.0
            if inst_score >= 2.5:
                inst_footprint = "HEAVY"
            elif inst_score >= 1.5:
                inst_footprint = "MODERATE"
            else:
                inst_footprint = "LIGHT"

            # Effort ratio (volume relative to price movement)
            price_change = abs(c.iloc[-1] - c.iloc[-5]) if len(c) >= 5 else 0
            effort_ratio = (v.iloc[-5:].sum() / (price_change + 1e-12)) if price_change > 0 else 0.0

            if imb > 0.56 and slope >= 0:
                bias = "BULLISH"
            elif imb < 0.44 and slope <= 0:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            comp = (imb - 0.5) * 22.0 + float(np.sign(slope)) * min(
                abs(slope) / (v.iloc[-10:].sum() + 1e-12) * 800.0, 8.0
            )
            if absorption:
                comp += 3.0 if c.iloc[-1] >= c.iloc[-2] else -3.0

            summary = (
                f"Imb {imb:.2f} | signed Δ10 {d10:.0f} | cumΔ slope {slope:.0f} | "
                f"vol×{vol_ratio:.2f} | trend {delta_trend}"
            )

            out = dict(base)
            out.update(
                {
                    "of_imbalance": imb,
                    "of_imbalance_5": imb,
                    "of_imbalance_20": imb_20,
                    "of_buy_pressure_20": buy_p20,
                    "of_delta_10": d10,
                    "of_cum_delta_slope": slope,
                    "of_cum_delta_slope_20": slope_20,
                    "of_absorption_flag": bool(absorption),
                    "of_absorption_strength": float(absorption_strength),
                    "of_bias": bias,
                    "of_score_component": float(comp),
                    "of_vol_ratio": vol_ratio,
                    "of_delta_positive": d10 > 0,
                    "of_delta_trend": delta_trend,
                    "of_liquidity_grab": bool(liquidity_grab),
                    "of_stop_run": bool(stop_run),
                    "of_institutional_footprint": inst_footprint,
                    "of_effort_ratio": float(effort_ratio),
                    "of_summary": summary,
                }
            )
            return out
        except Exception as e:
            logger.debug("OrderFlowEngine failed: %s", e)
            return dict(base)


# ===============================================
# 9c. ML PROGNOSIS (HISTORY-TRAINED ENSEMBLE)
# ===============================================

class MLPrognosisEngine:
    """
    Trains a small soft-voting ensemble on OHLCV-derived rows to estimate
    P(LONG), P(SHORT), P(NEUTRAL) for a forward horizon, and reports CV / holdout metrics.
    """

    FORWARD_H = 10
    RET_TH = 0.006
    MIN_SAMPLES = 80  # reduced to ensure training on shorter histories
    CACHE_SUBDIR = "ml_prognosis_cache"

    @classmethod
    def _cache_path(cls, symbol: str) -> str:
        import os

        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), cls.CACHE_SUBDIR)
        os.makedirs(root, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol.upper())
        return os.path.join(root, f"{safe}_ensemble.joblib")

    @classmethod
    def _sklearn_bundle(cls):
        try:
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                RandomForestClassifier,
                VotingClassifier,
            )
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            return (
                True,
                VotingClassifier,
                RandomForestClassifier,
                GradientBoostingClassifier,
                TimeSeriesSplit,
                cross_val_score,
                Pipeline,
                StandardScaler,
            )
        except Exception:
            return (False,) + (None,) * 7

    @classmethod
    def _build_training_frame(cls, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        df = normalize_ohlcv(df)
        if df is None or df.empty or len(df) < cls.MIN_SAMPLES + cls.FORWARD_H + 5:
            return None
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        low = df["low"].astype(float)
        v = df["volume"].astype(float).clip(lower=0.0)
        rng = (h - low).replace(0, np.nan)
        bp = ((c - low) / rng).clip(0.0, 1.0).fillna(0.5)
        signed = ((2.0 * bp - 1.0) * v).fillna(0.0)
        r1 = c.pct_change(1)
        r5 = c.pct_change(5)
        r20 = c.pct_change(20)
        delta = c.diff()
        gain = delta.clip(lower=0).fillna(0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).fillna(0).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).fillna(100)
        vz = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
        tr = pd.concat(
            [
                h - low,
                (h - c.shift()).abs(),
                (low - c.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()
        atr_pct = atr / (c + 1e-12) * 100.0
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        gap20 = c / (sma20 + 1e-12) - 1.0
        gap50 = c / (sma50 + 1e-12) - 1.0
        imb5 = (bp * v).rolling(5, min_periods=3).sum() / (v.rolling(5, min_periods=3).sum() + 1e-12)
        cd20 = signed.rolling(20, min_periods=5).sum()
        cdslope = cd20.diff(5)

        # Order flow features (additional)
        # Imbalance over 20 bars
        imb20 = (bp * v).rolling(20, min_periods=10).sum() / (v.rolling(20, min_periods=10).sum() + 1e-12)
        # Cumulative delta over 10 and 20 bars
        cd10 = signed.rolling(10, min_periods=5).sum()
        cdslope10 = cd10.diff(3) if len(cd10) > 3 else cd10 * 0
        # Absorbtion proxy (volume vs range)
        vol_ratio = v / (v.rolling(20).mean() + 1e-12)
        rng_ratio = (h - low) / (atr + 1e-12)
        absorption = ((rng_ratio < 0.65) & (vol_ratio > 1.25)).astype(float)
        # Institutional footprint proxy
        inst_footprint = ((vol_ratio > 1.3) & (rng_ratio < 0.5)).astype(float)

        fwd = c.shift(-cls.FORWARD_H) / (c + 1e-12) - 1.0
        y = np.where(
            fwd > cls.RET_TH,
            2,
            np.where(fwd < -cls.RET_TH, 0, 1),
        )
        mat = pd.DataFrame(
            {
                "r1": r1,
                "r5": r5,
                "r20": r20,
                "rsi": rsi,
                "vz": vz,
                "atr_pct": atr_pct,
                "gap20": gap20,
                "gap50": gap50,
                "imb5": imb5,
                "imb20": imb20,
                "cdslope": cdslope,
                "cdslope10": cdslope10,
                "absorption": absorption,
                "inst_footprint": inst_footprint,
                "y": y,
            }
        )
        mat = mat.iloc[: -cls.FORWARD_H].dropna()
        if mat is None or len(mat) < cls.MIN_SAMPLES:
            return None
        return mat

    @classmethod
    def forecast(
        cls,
        symbol: str,
        df: pd.DataFrame,
        *,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        import os

        import joblib

        empty = {
            "p_long": 1.0 / 3.0,
            "p_short": 1.0 / 3.0,
            "p_neutral": 1.0 / 3.0,
            "direction_hint": "NEUTRAL",
            "cv_accuracy_mean": None,
            "holdout_accuracy": None,
            "balanced_accuracy_holdout": None,
            "macro_f1_holdout": None,
            "win_rate_proxy": None,
            "precision_long": None,
            "precision_short": None,
            "precision_neutral": None,
            "recall_long": None,
            "recall_short": None,
            "recall_neutral": None,
            "f1_long": None,
            "f1_short": None,
            "f1_neutral": None,
            "brier_score": None,
            "of_accuracy_proxy": None,
            "of_win_rate_proxy": None,
            "feature_importance": {},
            "models": [],
            "trained": False,
            "n_samples": 0,
            "error": None,
        }
        ok, VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, TimeSeriesSplit, cross_val_score, Pipeline, StandardScaler = cls._sklearn_bundle()
        if not ok:
            empty["error"] = "scikit-learn not available"
            return empty

        mat = cls._build_training_frame(df)
        if mat is None or mat.empty:
            empty["error"] = "insufficient_history"
            return empty

        feature_cols = [c for c in mat.columns if c != "y"]
        X = mat[feature_cols].to_numpy(dtype=float)
        y = mat["y"].to_numpy(dtype=int)
        n = len(mat)
        last_ts = str(mat.index[-1]) if mat.index is not None else str(n)

        path = cls._cache_path(symbol)
        if not force_retrain and os.path.isfile(path):
            try:
                blob = joblib.load(path)
                if (
                    isinstance(blob, dict)
                    and blob.get("last_ts") == last_ts
                    and int(blob.get("n_samples", 0)) == n
                    and blob.get("pipeline") is not None
                ):
                    return blob["report"]
            except Exception:
                pass

        hold = max(int(n * 0.15), 30)
        X_train, y_train = X[:-hold], y[:-hold]
        X_test, y_test = X[-hold:], y[-hold:]

        # Initialize additional metrics
        of_accuracy_proxy = None
        of_win_rate_proxy = None

        clf = VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=14,
                        min_samples_leaf=6,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.06,
                        random_state=42,
                    ),
                ),
            ],
            voting="soft",
            n_jobs=-1,
        )
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

        cv_mean = None
        try:
            tsc = TimeSeriesSplit(n_splits=min(5, max(2, n // 80)))
            scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=tsc,
                scoring="accuracy",
                n_jobs=-1,
            )
            cv_mean = float(np.mean(scores))
        except Exception as e:
            logger.debug("ML CV failed %s: %s", symbol, e)

        hold_acc = bal_acc = macro_f1 = win_proxy = None
        precision_long = precision_short = precision_neutral = None
        recall_long = recall_short = recall_neutral = None
        f1_long = f1_short = f1_neutral = None
        brier_score = None
        try:
            from sklearn.metrics import (
                balanced_accuracy_score,
                f1_score,
                accuracy_score,
                precision_score,
                recall_score,
                brier_score_loss,
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            hold_acc = float(accuracy_score(y_test, pred))
            bal_acc = float(balanced_accuracy_score(y_test, pred))
            macro_f1 = float(
                f1_score(y_test, pred, average="macro", zero_division=0)
            )
            win_proxy = float(np.mean(pred == y_test))

            # Per-class metrics (labels: 0=SHORT, 1=NEUTRAL, 2=LONG)
            # Use fixed label order to ensure indices align
            labels = [0, 1, 2]
            precisions = precision_score(y_test, pred, labels=labels, average=None, zero_division=0)
            recalls = recall_score(y_test, pred, labels=labels, average=None, zero_division=0)
            f1s = f1_score(y_test, pred, labels=labels, average=None, zero_division=0)
            precision_short = float(precisions[0]) if len(precisions) > 0 else 0.0
            precision_neutral = float(precisions[1]) if len(precisions) > 1 else 0.0
            precision_long = float(precisions[2]) if len(precisions) > 2 else 0.0
            recall_short = float(recalls[0]) if len(recalls) > 0 else 0.0
            recall_neutral = float(recalls[1]) if len(recalls) > 1 else 0.0
            recall_long = float(recalls[2]) if len(recalls) > 2 else 0.0
            f1_short = float(f1s[0]) if len(f1s) > 0 else 0.0
            f1_neutral = float(f1s[1]) if len(f1s) > 1 else 0.0
            f1_long = float(f1s[2]) if len(f1s) > 2 else 0.0

            # Brier score (calibration) - only for probabilistic predictions
            try:
                proba = pipe.predict_proba(X_test)
                brier_score = float(brier_score_loss(y_test, proba))
            except Exception:
                brier_score = None

            # Order flow accuracy proxy: How well do order flow signals predict forward returns?
            # We evaluate on the test set only to avoid data leakage
            try:
                of_accuracy = None
                of_win_rate = None
                if len(mat) > hold:
                    of_test = mat.iloc[-hold:].copy()
                    # Bullish order flow: imbalance > 0.56, positive delta trend, or absorption in bullish direction
                    bullish_of = (
                        (of_test.get('imb20', 0.5) > 0.56) |
                        (of_test.get('cdslope', 0) > 0) |
                        ((of_test.get('absorption', 0) == 1) & (of_test.get('inst_footprint', 0) == 1))
                    )
                    bearish_of = (
                        (of_test.get('imb20', 0.5) < 0.44) |
                        (of_test.get('cdslope', 0) < 0) |
                        ((of_test.get('absorption', 0) == 1) & (of_test.get('inst_footprint', 0) == 0))
                    )
                    y_test_series = pd.Series(y_test, index=of_test.index)
                    # Win rate: when order flow was bullish, what % had positive returns?
                    if bullish_of.sum() > 0:
                        bull_returns = y_test_series[bullish_of]
                        of_win_rate = float(np.mean(bull_returns == 2))  # Class 2 = LONG (positive return)
                    # Accuracy: How often does order flow direction match the actual forward direction?
                    # Simple: order flow bullish -> predict LONG, bearish -> predict SHORT
                    of_preds = np.where(bullish_of, 2, np.where(bearish_of, 0, 1))
                    of_accuracy = float(np.mean(of_preds == y_test))
                of_accuracy_proxy = of_accuracy
                of_win_rate_proxy = of_win_rate
            except Exception:
                of_accuracy_proxy = None
                of_win_rate_proxy = None
        except Exception as e:
            logger.debug("ML holdout failed %s: %s", symbol, e)

        try:
            pipe.fit(X, y)
        except Exception as e:
            logger.debug("ML full-sample refit failed %s: %s", symbol, e)

        try:
            proba = pipe.predict_proba(X[-1:].reshape(1, -1))[0]
            classes = list(pipe.named_steps["clf"].classes_)
            pi = {int(classes[j]): float(proba[j]) for j in range(len(classes))}
            p_long = float(pi.get(2, 0.0))
            p_short = float(pi.get(0, 0.0))
            p_neutral = float(pi.get(1, max(0.0, 1.0 - p_long - p_short)))

            # Feature importance (average across ensemble)
            clf = pipe.named_steps["clf"]
            if hasattr(clf, 'estimators_'):
                importances = []
                for est in clf.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        importances.append(est.feature_importances_)
                if importances:
                    avg_importance = np.mean(importances, axis=0)
                    feature_names = [f"f{i}" for i in range(len(feature_cols))]
                    # Map importance to feature names
                    feature_importance = {str(feature_cols[i]): float(avg_importance[i]) for i in range(min(len(feature_cols), len(avg_importance)))}
                else:
                    feature_importance = {}
            else:
                feature_importance = {}
        except Exception:
            p_long = p_short = p_neutral = 1.0 / 3.0
            feature_importance = {}

        s = p_long + p_short + p_neutral + 1e-12
        p_long, p_short, p_neutral = p_long / s, p_short / s, p_neutral / s
        hint = (
            "LONG"
            if p_long >= p_short and p_long >= p_neutral
            else "SHORT"
            if p_short >= p_neutral
            else "NEUTRAL"
        )

        report = {
            "p_long": round(p_long, 4),
            "p_short": round(p_short, 4),
            "p_neutral": round(p_neutral, 4),
            "direction_hint": hint,
            "cv_accuracy_mean": None if cv_mean is None else round(cv_mean, 4),
            "holdout_accuracy": None if hold_acc is None else round(hold_acc, 4),
            "balanced_accuracy_holdout": None if bal_acc is None else round(bal_acc, 4),
            "macro_f1_holdout": None if macro_f1 is None else round(macro_f1, 4),
            "win_rate_proxy": None if win_proxy is None else round(win_proxy, 4),
            "precision_long": None if precision_long is None else round(precision_long, 4),
            "precision_short": None if precision_short is None else round(precision_short, 4),
            "precision_neutral": None if precision_neutral is None else round(precision_neutral, 4),
            "recall_long": None if recall_long is None else round(recall_long, 4),
            "recall_short": None if recall_short is None else round(recall_short, 4),
            "recall_neutral": None if recall_neutral is None else round(recall_neutral, 4),
            "f1_long": None if f1_long is None else round(f1_long, 4),
            "f1_short": None if f1_short is None else round(f1_short, 4),
            "f1_neutral": None if f1_neutral is None else round(f1_neutral, 4),
            "brier_score": None if brier_score is None else round(brier_score, 4),
            "of_accuracy_proxy": None if of_accuracy_proxy is None else round(of_accuracy_proxy, 4),
            "of_win_rate_proxy": None if of_win_rate_proxy is None else round(of_win_rate_proxy, 4),
            "feature_importance": feature_importance,
            "models": ["RandomForest", "GradientBoosting", "VotingSoft"],
            "trained": True,
            "n_samples": int(n),
            "error": None,
        }

        try:
            joblib.dump(
                {
                    "pipeline": pipe,
                    "report": report,
                    "last_ts": last_ts,
                    "n_samples": n,
                },
                path,
            )
        except Exception as e:
            logger.debug("ML cache write failed %s: %s", symbol, e)

        return report


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
    # Fundamental quality metrics
    fundamental_quality: float = 50.0
    piotroski_f: int = 0
    altman_z: Optional[float] = None


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

    order_flow: dict
    ml_prognosis: dict

    # Fundamental quality metrics
    fundamental_quality: float = 50.0
    piotroski_f: int = 0
    altman_z: Optional[float] = None


# =========================================================
# NORMALIZATION
# =========================================================




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
        use_order_flow: bool = True,
        use_ml: bool = True,
        force_ml_retrain: bool = False,
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

            order_flow_ctx: Dict[str, Any] = {}
            if use_order_flow:
                order_flow_ctx = OrderFlowEngine.analyze(df)

            ml_forecast: Dict[str, Any] = {}
            if use_ml:
                ml_forecast = MLPrognosisEngine.forecast(
                    symbol,
                    df,
                    force_retrain=force_ml_retrain,
                )

            features = ProposalEngine._build_features(
                techs,
                smc,
                breakout,
                funds,
                sentiment,
                analysts,
                order_flow=order_flow_ctx,
            )

            ai = AIEnsemble.predict(
                features,
                market_regime,
                ml_forecast if use_ml else None,
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
                ai.get("horizon_confidence"),
                direction=direction,
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
                order_flow=order_flow_ctx,
                ml_prognosis=ml_forecast,
                fundamental_quality=funds.get("buffett_score", 45.0),
                piotroski_f=funds.get("piotroski_f", 0),
                altman_z=funds.get("altman_z"),
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
        funds = FundamentalEngine.get_fundamentals(symbol) or {}
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
        # Infer direction
        direction = "NEUTRAL"
        if breakout.get("breakout_type"):
            direction = "LONG" if "BULLISH" in breakout["breakout_type"] else "SHORT"
        elif st_score >= 50:
            direction = "LONG"
        elif st_score <= 35:
            direction = "SHORT"

        # --- Compute trade levels and position size ---
        close = df["close"].iloc[-1]
        # ATR (14-period)
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = close * 0.02
        risk = atr * 1.5
        if direction == "LONG":
            sl = close - risk
            tp1 = close + risk * 1.5
            tp2 = close + risk * 3
            tp3 = close + risk * 5
        else:
            sl = close + risk
            tp1 = close - risk * 1.5
            tp2 = close - risk * 3
            tp3 = close - risk * 5
        rr = ProposalEngine._calculate_rr(close, sl, tp2)
        rr_ext = ProposalEngine._calculate_rr(close, sl, tp3)
        pos_size = ProposalEngine._calculate_position_size(
            st_score, rr, {}, "MEDIUM", direction=direction
        )
        ai_grade = ProposalEngine._calculate_grade(st_score, "MEDIUM")
        setup_type = f"{direction}_SWING" if direction in ("LONG", "SHORT") else "WATCHLIST"
        thesis = f"{direction} setup"
        if breakout.get("breakout_type"):
            thesis += f": {breakout['breakout_type']}"
        if breakout.get("is_accumulating"):
            thesis += " · Institutional accumulation"
        if analysts.get("label"):
            thesis += f" · Analyst {analysts['label']}"

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
            "fundamentals": funds,  # include full fundamentals
            "ai_confidence": st_score,
            "direction": direction,
            # Fundamental quality
            "fundamental_quality": funds.get("buffett_score", 45.0),
            "piotroski_f": funds.get("piotroski_f", 0),
            "altman_z": funds.get("altman_z"),
            # Trade levels ... (as before)
            # Trade levels
            "entry_price": round(close, 2),
            "stop_loss": round(sl, 2),
            "tp_1": round(tp1, 2),
            "tp_2": round(tp2, 2),
            "tp_3": round(tp3, 2),
            "risk_reward": rr,
            "risk_reward_extended": rr_ext,
            "position_size_pct": pos_size,
            "ai_grade": ai_grade,
            "thesis": thesis,
            "setup_type": setup_type,
            "hold_period": "1-3 Months",
            "horizon_confidence": "MEDIUM",
            "sector_exposure": "NEUTRAL",
            "chart_data": None,
            "signals": [],
            "weights": {},
            "ml_prognosis": None,
        }

    @staticmethod
    def _build_features(
        techs,
        smc,
        breakout,
        funds,
        sentiment,
        analysts,
        order_flow: Optional[Dict[str, Any]] = None,
    ):

        merged: Dict[str, Any] = {
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

            # Buffett/Munger quality score (0-90 scale → normalize to 0-50 weight)
            "fundamental_quality":
                funds.get("buffett_score", 45),  # default ~50% quality

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
        if order_flow:
            for _k, _v in order_flow.items():
                if isinstance(_k, str) and _k.startswith("of_"):
                    merged[_k] = _v
        return merged

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
        confidence,  # raw 0-100 score
        rr,
        market_regime,
        horizon_confidence,
        direction="LONG"  # add direction to invert SHORT scores
    ):
        # Convert to effective confidence: SHORTs use inverted scale
        if direction == "SHORT":
            effective = 100 - confidence
        else:
            effective = confidence

        edge = max(0, effective - 50)

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
        "MIDCAP": 0.6,
        "MICROCAP": 0.2,
        "CRYPTO": 0.5,
        "FOREX": 0.3
    }

    # ---------------------------------------
    # EQUITIES
    # ---------------------------------------
    STOCKS_LARGE_CAP = [
        # Mega-cap tech
        "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","TSLA","AVGO","ORCL",
        "ADBE","CRM","NFLX","INTU","QCOM","AMD","IBM","NOW","INTC","MU","LRCX",
        "CSCO","TXN","ADI","KLAC","AMAT","MCHP","SNPS","CDNS","ASML","PANW",
        # Financials
        "JPM","V","MA","BAC","WFC","GS","MS","BLK","AXP","C","PFG","MET",
        "SCHW","BX","TROW","STT","NTRS","FITB","USB","PNC","KEY",
        # Healthcare
        "UNH","JNJ","LLY","ABBV","TMO","AMGN","ISRG","VRTX","ABT","DHR","PFE",
        "MRK","BMY","GILD","REGN","VRTX","ILMN","MRNA","BIIB","CVS","CI",
        # Consumer
        "HD","MCD","NKE","SBUX","COST","WMT","PG","KO","PEP","TGT","WBA",
        "CL","KMB","GIS","K","KHC","MDLZ","CPB","CAG","SJM",
        # Industrials & Defense
        "XOM","CVX","COP","BA","CAT","HON","UPS","RTX","LMT","GE","DE",
        "CAT","MMM","EMR","ETN","ITW","NOC","LHX","GD","HII","TXT",
        # Telecom & Media
        "T","VZ","CMCSA","DIS","NFLX","CHTR","FOXA","PARA","WBD",
        # Utilities
        "NEE","DUK","SO","D","AEP","EXC","SRE","ED","PEG","XEL",
        # Real Estate
        "PLD","AMT","EQIX","DLR","PSA","O","WY","SPG","VNO","SLG",
        "SPY",  # ETF placeholder
        # Expanded large-cap additions (increased stock universe)
        "BRK.B","BKNG","LOW","TJX","ROST","ORLY","AZO","GM","DELL","HPQ","HPE","COF","AIG","PRU","ALL","TRV","ADP","FISV","FIS","GPN","DFS","HCA","MDT","BAX","ZTS","BDX","KDP","STZ","BF.B","LULU","CMG","DRI","YUM","QSR","EA","SWKS","QRVO","NUE","STLD","VMC","MLM","BLL","IP","PPG","KMI","EPD","MMP","BKR","SLB","HAL","FDX","CSX","UNP","NSC","IR","GWW","WM","RSG","HWM","PH","IT","ANET","FTNT","ENPH","PAYC","AXON"
    ]

    STOCKS_GROWTH = [
        "COIN","SOFI","RBLX","RIVN","BABA","MSTR","OPEN","SMR","IONQ",
        "TTD","ON","RDDT","SMCI","PDD","JD","BILI","NIO","XPENG","LI",
        "PTON","UBER","LYFT","DASH","SNOW","PLTR","CRWD","ZS","NET","OKTA",
        "TEAM","MDB","DDOG","AYX","TWLO","MELI","SHOP","SE","BIDU","BKKT",
        "ASAN","UPST","AFRM","SQ","PYPL","COUP","VEEV","DOCU","ZM","TWOU",
        "U","MNDY","DUOL","ESTC","GTLB","HCP","RAMP","M","MSTR","HOOD",
        "SOUN","RKLB","ACLS","BEAM","DNA","EXAS","IONS","CRSP","EDIT","NTLA",
        "VCEL","ALGN","INMD","ISRG","IRTC","NVCR","TMDX","UTRS","INSP","LMND",
        "UPWK","PINS","SWK","W","BYND","TTCF","IMGN","NVTA","CRNX","VRTX"
    ]

    STOCKS_SPECULATIVE = [
        "OKLO","AI","QBTS","RUM","ACHR","PONY","NNE","CHPT","RGT","ARVL",
        "LAZR","MAXR","BLNK","BLNKW","SPCE","VORB","RDW","ALGM","ALGT",
        "ATSG","GXO","FLYG","HA","JBLU","UAL","AAL","DAL","LUV","RYAAY",
        "SAVE","ALK","SKYW","VRM","CAR","ABUS","AIM","AREN","BCOV","BMRN"
    ]

    STOCKS_MIDCAP = [
        # S&P 400 mid-caps
        "ETSY","TTWO","VRSK","CSGP","CPRT","CDW","FANG","DLTR","CEG","ROL",
        "MSCI","SPGI","NEM","CF","MOS","APD","SHW","LIN","CTAS","EXPD",
        "ADSK","PAYX","ANSS","ONS","MORN","ZBRA","DXCM","BRO","FAST","NDSN",
        "HTLF","BANC","CIVI","O","VTR","WELL","ELS","REG","FRT","KIM",
        "CAG","K","SJM","TAP","CCL","RCL","EXPE","MAR","HLT","H",
        "CNP","ATO","AEE","XEL","ES","DTE","PPL","FE","ETR","WEC",
        "VST","NRG","SRE","DTE","PEG","ED","EIX","AEP","D","DUK",
        "NEE","SO","EXC","PWR","AES","NRG","VST","OGE","OKE","WMB",
        "HES","COTY","PVH","RL","GPS","LEVI","FL","DECK","ZUMZ","BKE",
        "CHD","COTY","EL","OLPX","NWL","CLX","HRL","SJM","CAG","K",
        "MKC","KMB","PG","CL","KO","PEP","WMT","TGT","COST","DG","DLTR",
        "KR","WBA","BG","KHC","GIS","CPB","CAG","SJM","MKC","KMB",
        "BIG","OLLI","CURLF","SNDX","CXDO","CERT","TMDX","INSP","AMN","ACAD"
    ]

    STOCKS_MICROCAP = [
        "PSTV","RNAZ","EH","AAOI","HOTH","DVLT","TOVX","BSOL","BBAR","CDE","NGD",
        "AUY","GDX","GDXJ","NEM","KGC","ABX","GG","GFI","OR","AU","AEM",
        "Bathon","BTG","KGC","NG","F","VGZ","EXK","SSRM","SAND","PLL","HL",
        "CLF","Vale","RNO","SBSW","SBGL","GGB","CX","TGB","SCCO","FCX",
        "ATNM","UEC","URG","DNN","UUU","ISR","UC","NXE","LEU","URRE","NFEC",
        "MNTS","ZEPP","MVIS","VYGR","AVO","MEIP","RVP","PVC","TKAT","RSLS"
    ]

    STOCKS_ALL = list(set(
        STOCKS_LARGE_CAP +
        STOCKS_GROWTH +
        STOCKS_SPECULATIVE +
        STOCKS_MIDCAP +
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

        if symbol in AssetUniverse.STOCKS_MIDCAP:
            return AssetUniverse.UNIVERSE_WEIGHTS["MIDCAP"]

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
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

# Alias for backward compatibility
safe = safe_num


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
    st.caption("3–12 Month Investment Opportunity Scanner, with AI, Technical Indicators, SMC & Market Regime, Berkshire Hathaway criteria, Tnx to knowledges from: Dr. Anastas Dzurovski, Kicko Ognenovski, Nikola Stojcevski, Altaj Sulejman, Dejan Butevski")

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
        index=1
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
                AssetUniverse.STOCKS_ALL if asset_class == "Stocks"
                else AssetUniverse.CRYPTO_ALL if asset_class == "Crypto"
                else AssetUniverse.FOREX_ALL
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
        value=40,  # lowered to capture more borderline setups
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

    use_order_flow = st.sidebar.checkbox(
        "Order-flow tape proxy (OHLCV)",
        value=True,
        help="Uses bar microstructure proxies (signed volume delta, imbalance, absorption) for extra confluence.",
    )
    use_ml = st.sidebar.checkbox(
        "ML prognosis (train on history)",
        value=True,
        help="Fits a small ensemble on past bars, reports CV / holdout metrics, and blends probabilities into the AI score.",
    )
    force_ml_retrain = st.sidebar.checkbox(
        "Force ML retrain (ignore disk cache)",
        value=False,
    )

    # ── Auto-Scan (Telegram alerts every 4 h when enabled) ───────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Auto-Scan (Telegram)")

    def _ensure_auto_scan_thread() -> None:
        """Start the background auto-scan daemon (every interval_hours, guarded to one thread)."""
        guard = st.session_state.get("auto_scan_thread")
        if guard is not None:
            return   # daemon already running
        try:
            interval_h: float = float(st.session_state.get("auto_scan_interval_h", 4.0))
            _th = start_auto_scan(interval_hours=interval_h)
            st.session_state.auto_scan_thread = _th
            logger.info("Auto-scan daemon interval set to %.1f h", interval_h)
        except Exception as _exc:
            logger.error("auto_scan start failed: %s", _exc)

    # Toggle and interval live in session_state so change-callbacks are stable
    # across Streamlit reruns
    if "auto_scan_enabled" not in st.session_state:
        st.session_state.auto_scan_enabled = False
    if "auto_scan_interval_h" not in st.session_state:
        st.session_state.auto_scan_interval_h = 4.0

    st.sidebar.checkbox(
        "Enable auto-scan",
        value=st.session_state.auto_scan_enabled,
        key="auto_scan_enabled",
        on_change=_ensure_auto_scan_thread,
        help=(
            "Runs the stock universe scan in the background every "
            f"{st.session_state.auto_scan_interval_h:.0f} h and pushes the "
            "top 10 opportunities to the Telegram chat."
        ),
    )

    st.sidebar.number_input(
        "Scan interval (hours)",
        min_value=1.0,
        max_value=24.0,
        value=float(st.session_state.auto_scan_interval_h),
        step=0.5,
        key="auto_scan_interval_h",
        on_change=_ensure_auto_scan_thread,
    )

    if st.session_state.get("auto_scan_enabled"):
        last_at = _LAST_SCAN_AT
        if last_at is not None:
            ago  = datetime.now(timezone.utc) - last_at
            ago_h   = ago.total_seconds() / 3600.0
            next_in = max(0.0, _AUTO_SCAN_INTERVAL_SECS - ago.total_seconds())
            next_h  = next_in / 3600.0
            st.sidebar.caption(
                f"Last run: **{ago_h:.1f} h ago**  ·  Next in: **{next_h:.1f} h**"
            )
            if "auto_scan_thread" not in st.session_state:
                _ensure_auto_scan_thread()
        else:
            st.sidebar.caption("No scan yet — first run will fire shortly.")
            _ensure_auto_scan_thread()

        if _auto_scan_log:
            with st.sidebar.expander("Recent auto-scan log", expanded=False):
                for _line in reversed(_auto_scan_log[-10:]):
                    st.caption(_line)
    else:
        st.sidebar.caption("Auto-scan is **OFF**.")
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
                    btc = _yf_call(
                        yf.download,
                        "BTC-USD", period="6mo", interval="1d",
                        progress=False, auto_adjust=True, threads=False,
                    )
                    btc = normalize_ohlcv(btc)
                    r = ProposalEngine._return_n_bars_pct(btc, 22)
                    if r is not None:
                        ref["BTC_1m_pct"] = r
                except Exception:
                    pass
            elif ac == "Forex":
                try:
                    dxy = _yf_call(
                        yf.download,
                        "DX-Y.NYB", period="6mo", interval="1d",
                        progress=False, auto_adjust=True, threads=False,
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
                tf = _yf_call(fetch_multi_timeframe, sym)
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
                    use_order_flow=use_order_flow,
                    use_ml=use_ml,
                    force_ml_retrain=force_ml_retrain,
                )
                if prop:
                    sf = prop.sector_flow or {}
                    ba = prop.breakout_analytics or {}
                    sn = prop.sentiment_snapshot or {}
                    ac = prop.analyst_consensus or {}
                    ofx = {}
                    if use_order_flow:
                        ofx = OrderFlowEngine.analyze(df) or {}
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
                        "Buffett": safe(getattr(prop, "fundamental_quality", 0)),
                        "Piotroski F": getattr(prop, "piotroski_f", "—"),
                        "Altman Z": round(getattr(prop, "altman_z", np.nan), 2) if getattr(prop, "altman_z", None) is not None else np.nan,
                        "OF bias": ofx.get("of_bias", "—"),
                        "OF trend": ofx.get("of_delta_trend", "—"),
                        "Imbalance": round(ofx.get("of_imbalance", 0.5), 3),
                        "Inst footp.": ofx.get("of_institutional_footprint", "—"),
                    })
                    return prop  # <-- CRITICAL: return successful proposal
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
                        # Quick surface doesn't compute full order flow, but we can fetch it optionally
                        ofx = {}
                        if use_order_flow:
                            ofx = OrderFlowEngine.analyze(df) or {}
                        money_note = qs.get("money_flow_note", "—")
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
                            "Money flow": money_note,
                            "Analyst": ac.get("score"),
                            "Analyst label": ac.get("label"),
                "Buffett": safe(qs.get("fundamental_quality", 0)),
                "Piotroski F": qs.get("piotroski_f", "—"),
                "Altman Z": round(qs.get("altman_z", np.nan), 2) if qs.get("altman_z") is not None else np.nan,
                "OF bias": ofx.get("of_bias", "—"),
                            "OF trend": ofx.get("of_delta_trend", "—"),
                            "Imbalance": round(ofx.get("of_imbalance", 0.5), 3),
                            "Inst footp.": ofx.get("of_institutional_footprint", "—"),
                        })
                        # Convert quick_surface dict to proposal object
                        proposal_obj = SimpleNamespace()
                        for _k, _v in qs.items():
                            setattr(proposal_obj, _k, _v)
                        proposal_obj.order_flow = ofx  # attach computed order flow
                        return proposal_obj
                return None
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = {ex.submit(fetch, s): s for s in symbols}

            for i, f in enumerate(as_completed(futures)):
                progress.progress((i + 1) / len(symbols))
                res = f.result()
                if res:
                    # Extract direction and confidence (handles both object and dict)
                    direction = getattr(res, "direction", None)
                    if direction is None and isinstance(res, dict):
                        direction = res.get("direction")
                    conf = getattr(res, "ai_confidence", None)
                    if conf is None and isinstance(res, dict):
                        conf = res.get("ai_confidence")
                    if conf is not None:
                        if direction == "LONG":
                            effective = conf
                        elif direction == "SHORT":
                            effective = 100 - conf
                        else:
                            effective = 0
                        if effective >= min_confidence:
                            proposals.append(res)

        progress.empty()

        # =========================================================
        # NO RESULTS
        # =========================================================
        if not proposals:
            st.error("No valid setups found. Lower confidence or expand universe.")
            return

        proposals.sort(key=lambda x: x.ai_confidence, reverse=True)

        # Ensure all proposals have valid position sizing (safety net for quick_surface or legacy)
        for _p in proposals:
            current = getattr(_p, "position_size_pct", None)
            if current is None or float(current) <= 0:
                direction = getattr(_p, "direction", "LONG")
                conf = getattr(_p, "ai_confidence", 50)
                rr = getattr(_p, "risk_reward", 3.0)
                if not isinstance(rr, (int, float)) or rr <= 0:
                    rr = 3.0
                new_pos = ProposalEngine._calculate_position_size(
                    conf, rr, {}, "MEDIUM", direction=direction
                )
                try:
                    setattr(_p, "position_size_pct", new_pos)
                except Exception:
                    pass

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

        ml_cv_vals = []
        ml_hold_vals = []
        ml_prec_l_vals = []
        ml_rec_l_vals = []
        ml_f1_l_vals = []
        ml_brier_vals = []
        of_acc_vals = []
        of_wr_vals = []
        for _p in proposals:
            _m = getattr(_p, "ml_prognosis", None) or {}
            if _m.get("cv_accuracy_mean") is not None:
                ml_cv_vals.append(float(_m["cv_accuracy_mean"]))
            if _m.get("holdout_accuracy") is not None:
                ml_hold_vals.append(float(_m["holdout_accuracy"]))
            if _m.get("precision_long") is not None:
                ml_prec_l_vals.append(float(_m["precision_long"]))
            if _m.get("recall_long") is not None:
                ml_rec_l_vals.append(float(_m["recall_long"]))
            if _m.get("f1_long") is not None:
                ml_f1_l_vals.append(float(_m["f1_long"]))
            if _m.get("brier_score") is not None:
                ml_brier_vals.append(float(_m["brier_score"]))
            if _m.get("of_accuracy_proxy") is not None:
                of_acc_vals.append(float(_m["of_accuracy_proxy"]))
            if _m.get("of_win_rate_proxy") is not None:
                of_wr_vals.append(float(_m["of_win_rate_proxy"]))

        with st.expander("AI / ML prognosis quality (this scan)", expanded=False):
            if ml_cv_vals:
                st.write(
                    f"Mean time-series CV accuracy: **{float(np.mean(ml_cv_vals)):.1%}** "
                    f"(min {float(np.min(ml_cv_vals)):.1%}, max {float(np.max(ml_cv_vals)):.1%})"
                )
            else:
                st.caption("ML metrics unavailable (disabled or insufficient history).")
            if ml_hold_vals:
                st.write(
                    f"Mean holdout accuracy (tail slice): **{float(np.mean(ml_hold_vals)):.1%}**"
                )
            if ml_prec_l_vals:
                st.write(
                    f"Mean precision (LONG class): **{float(np.mean(ml_prec_l_vals)):.3f}** · "
                    f"recall (LONG): **{float(np.mean(ml_rec_l_vals)):.3f}** · "
                    f"F1 (LONG): **{float(np.mean(ml_f1_l_vals)):.3f}**"
                )
            if ml_brier_vals:
                st.write(
                    f"Mean Brier score (calibration, lower is better): **{float(np.mean(ml_brier_vals)):.4f}**"
                )
            if of_acc_vals:
                st.write(
                    f"Order-flow directional accuracy (historical proxy): **{float(np.mean(of_acc_vals)):.1%}**"
                )
            if of_wr_vals:
                st.write(
                    f"Order-flow long win-rate (bullish signals → profit): **{float(np.mean(of_wr_vals)):.1%}**"
                )
            st.caption(
                "Models: RandomForest + GradientBoosting (soft vote), trained on OHLCV + order-flow rows. "
                "Labels use forward-return buckets (LONG/SHORT/NEUTRAL within 10 bars); CV uses TimeSeriesSplit. "
                "Displayed metrics: precision/recall/F1 per class, Brier score (calibration), feature importance, order flow accuracy proxy. "
                "For research context only — past performance does not guarantee future results."
            )
            st.caption(
                "Models: RandomForest + GradientBoosting (soft vote), trained on OHLCV + order-flow rows. "
                "Labels use forward-return buckets (LONG/SHORT/NEUTRAL within 10 bars); CV uses TimeSeriesSplit. "
                "Displayed metrics: precision/recall/F1 per class, Brier score (calibration), feature importance. "
                "For research context only — past performance does not guarantee future results."
            )

        # =========================================================
        # FEATURE IMPORTANCE SUMMARY (AGGREGATE)
        # =========================================================
        all_feat_importances = {}
        for _p in proposals:
            _m = getattr(_p, "ml_prognosis", None) or {}
            _fi = _m.get("feature_importance", {})
            if isinstance(_fi, dict) and _fi:
                for feat, imp in _fi.items():
                    if feat not in all_feat_importances:
                        all_feat_importances[feat] = []
                    all_feat_importances[feat].append(float(imp))

        if all_feat_importances:
            # Average importances across models
            avg_importances = {k: np.mean(v) for k, v in all_feat_importances.items()}
            sorted_feats = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            with st.expander("🔬 Top 10 Predictive Features (Aggregate)", expanded=False):
                if sorted_feats:
                    st.caption("Features ranked by average importance across all ML models in this scan:")
                    for feat, imp in sorted_feats:
                        st.write(f"- **{feat}**: {imp:.4f}")
                else:
                    st.caption("No feature importance data available.")

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
            mlx = getattr(p, "ml_prognosis", None) or {}
            ofx = getattr(p, "order_flow", None) or {}
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

                "Buffett": safe(getattr(p, "fundamental_quality", 0)),
                "Piotroski F": safe(getattr(p, "piotroski_f", 0)),
                "Altman Z": round(getattr(p, "altman_z", np.nan), 2) if getattr(p, "altman_z", None) is not None else np.nan,

                "ML CV": (
                    f"{float(mlx['cv_accuracy_mean']):.1%}"
                    if mlx.get("cv_accuracy_mean") is not None
                    else "—"
                ),
                "OF bias": ofx.get("of_bias", "—"),
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

            # Optional: highlight top 10 setups mentally (no UI dependency)
            top_n = min(10, len(edge_df))
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

        for p in proposals[:50]:

            ba = getattr(p, "breakout_analytics", None) or {}
            sn = getattr(p, "sentiment_snapshot", None) or {}
            sf = getattr(p, "sector_flow", None) or {}
            ac = getattr(p, "analyst_consensus", None) or {}
            ofx = getattr(p, "order_flow", None) or {}
            mlx = getattr(p, "ml_prognosis", None) or {}

            table.append({
                "Symbol": getattr(p, "symbol", "—"),
                "Direction": getattr(p, "direction", "—"),
                "Setup": safe_str(getattr(p, "setup_type", "—")),

                "Confidence": safe(getattr(p, "ai_confidence", 0)),
                "Short-term": round(safe(getattr(p, "short_term_score", 0)), 1),

                "OF bias": ofx.get("of_bias", "—"),
                "OF trend": ofx.get("of_delta_trend", "—"),
                "Inst footprint": ofx.get("of_institutional_footprint", "—"),

                "ML p(L)": round(safe(mlx.get("p_long", 0)), 3),
                "ML CV acc": (
                    f"{float(mlx['cv_accuracy_mean']):.1%}"
                    if mlx.get("cv_accuracy_mean") is not None
                    else "—"
                ),
                "ML Prec(L)": (
                    f"{float(mlx.get('precision_long', 0)):.3f}"
                    if mlx.get("precision_long") is not None
                    else "—"
                ),
                "ML Rec(L)": (
                    f"{float(mlx.get('recall_long', 0)):.3f}"
                    if mlx.get("recall_long") is not None
                    else "—"
                ),
                "ML Brier": (
                    f"{float(mlx.get('brier_score', 0)):.3f}"
                    if mlx.get("brier_score") is not None
                    else "—"
                ),

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

                "Buffett": safe(getattr(p, "fundamental_quality", 0)),
                "Piotroski F": safe(getattr(p, "piotroski_f", 0)),
                "Altman Z": round(getattr(p, "altman_z", np.nan), 2) if getattr(p, "altman_z", None) is not None else np.nan,

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

                ofx = getattr(p, "order_flow", None) or {}
                mlx = getattr(p, "ml_prognosis", None) or {}
                o1, o2, o3 = st.columns(3)
                o1.metric("Order-flow bias", str(ofx.get("of_bias", "—")))
                o2.metric("ML P(LONG)", f"{safe(mlx.get('p_long', 0)):.3f}")
                _cv = mlx.get("cv_accuracy_mean")
                o3.metric(
                    "ML CV overall acc",
                    f"{float(_cv):.1%}" if _cv is not None else "n/a",
                )

                # Additional ML metrics row
                _prec_l = mlx.get("precision_long")
                _rec_l = mlx.get("recall_long")
                _f1_l = mlx.get("f1_long")
                _prec_s = mlx.get("precision_short")
                _rec_s = mlx.get("recall_short")
                _f1_s = mlx.get("f1_short")
                _brier = mlx.get("brier_score")

                if any(v is not None for v in [_prec_l, _rec_l, _f1_l, _prec_s, _rec_s, _f1_s, _brier]):
                    m1, m2, m3, m4, m5 = st.columns(5)
                    if _prec_l is not None:
                        m1.metric("Prec(L)", f"{_prec_l:.3f}")
                    if _rec_l is not None:
                        m2.metric("Recall(L)", f"{_rec_l:.3f}")
                    if _f1_l is not None:
                        m3.metric("F1(L)", f"{_f1_l:.3f}")
                    if _brier is not None:
                        m4.metric("Brier", f"{_brier:.3f}")
                    if _prec_s is not None:
                        m5.metric("Prec(S)", f"{_prec_s:.3f}")

                # Holdout accuracy and feature importance snippet
                _ho = mlx.get("holdout_accuracy")
                _f1m = mlx.get("macro_f1_holdout")
                _of_acc = mlx.get("of_accuracy_proxy")
                _of_wr = mlx.get("of_win_rate_proxy")
                if _ho is not None or _f1m is not None or _of_acc is not None or _of_wr is not None:
                    parts = []
                    if _ho is not None:
                        parts.append(f"ML holdout {float(_ho):.1%}")
                    if _f1m is not None:
                        parts.append(f"macro-F1 {float(_f1m):.3f}")
                    if _of_acc is not None:
                        parts.append(f"OF acc {float(_of_acc):.1%}")
                    if _of_wr is not None:
                        parts.append(f"OF win-rate {float(_of_wr):.1%}")
                    st.caption(" · ".join(parts))

                # Feature importance (top 5)
                feat_imp = mlx.get("feature_importance", {})
                if feat_imp and isinstance(feat_imp, dict):
                    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
                    if sorted_feats:
                        st.caption("Top 5 ML features: " + ", ".join([f"{k} ({v:.3f})" for k,v in sorted_feats]))

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
