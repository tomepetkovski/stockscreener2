"""
terminal_enhancements.py  -  drop-in add-on for Institutional Breakout Terminal v4.0

What it adds (all standalone, no changes to your engines required):
  1. sanitize_universe()      - fixes dot-tickers, renamed/delisted symbols, auto-quarantines dead tickers
  2. SignalJournal            - logs every proposal, later scores it (SL/TP walk), calibrates confidence -> real hit rate
  3. PortfolioConstructor     - ranks by quality, caps sector count / correlation / gross exposure / open risk by regime
  4. AlertDeduper             - stops Telegram re-sending the same setup every 4h
  5. count_trailing_true()    - correct squeeze-duration helper (the original returned wrong numbers)
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
JOURNAL_DB = os.path.join(_HERE, "signal_journal.db")
QUARANTINE_FILE = os.path.join(_HERE, "ticker_quarantine.json")
ALERT_STATE_FILE = os.path.join(_HERE, "alert_state.json")
_LOCK = threading.Lock()


def _num(x: Any, d: float = 0.0) -> float:
    try:
        f = float(x)
        return f if math.isfinite(f) else d
    except Exception:
        return d


# =====================================================================
# 1. UNIVERSE HYGIENE
# =====================================================================
# Yahoo wants hyphens, not dots. Renames/delistings below are from my knowledge
# and may be stale - the quarantine below is the self-healing part.
TICKER_RENAMES = {
    "BRK.B": "BRK-B", "BF.B": "BF-B",
    "SQ": "XYZ",            # Block
    "PARA": "PSKY",         # Paramount Skydance
    "BLL": "BALL",          # Ball Corp
    "MATIC-USD": "POL-USD",
}
KNOWN_DEAD = {
    "LLY1", "VRTX1", "Bathon", "COUP", "TWOU", "AYX", "MAXR", "ANSS",
    "WBA", "HES", "HCP", "VORB", "ARVL", "SAVE", "BIG",
}


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _save_json(path: str, obj) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)
    except Exception:
        pass


def sanitize_universe(symbols: Iterable[str], quarantine_after: int = 3) -> List[str]:
    """Rename, drop known-dead and quarantined symbols; dedupe preserving order."""
    q = _load_json(QUARANTINE_FILE, {})
    out, seen = [], set()
    for s in symbols:
        s = str(s).strip()
        if not s:
            continue
        s = TICKER_RENAMES.get(s.upper(), TICKER_RENAMES.get(s, s)).upper()
        if "." in s and not s.endswith("=X"):
            s = s.replace(".", "-")
        if s in {k.upper() for k in KNOWN_DEAD}:
            continue
        if q.get(s, 0) >= quarantine_after:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def report_download_result(symbol: str, ok: bool) -> None:
    """Call after each data fetch. 3 consecutive failures -> symbol is skipped in future scans."""
    with _LOCK:
        q = _load_json(QUARANTINE_FILE, {})
        if ok:
            if symbol in q:
                q.pop(symbol)
        else:
            q[symbol] = q.get(symbol, 0) + 1
        _save_json(QUARANTINE_FILE, q)


# =====================================================================
# 2. SIGNAL JOURNAL  (the feedback loop the terminal was missing)
# =====================================================================
class SignalJournal:
    """
    Logs proposals, scores them against what the market did next, and tells you
    whether "78% confidence" has actually meant anything.
    Entry is assumed at the signal close; stop is checked before targets on the
    same bar (conservative). Targets are scaled out in thirds, no break-even move.
    """

    def __init__(self, path: str = JOURNAL_DB):
        self.path = path
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS signals(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, ts TEXT, direction TEXT,
                    entry REAL, sl REAL, tp1 REAL, tp2 REAL, tp3 REAL,
                    conf REAL, grade TEXT, regime TEXT, size_pct REAL,
                    status TEXT DEFAULT 'OPEN', exit_ts TEXT, exit_reason TEXT,
                    r_result REAL, mfe_r REAL)"""
            )

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    # ---- logging -------------------------------------------------------
    def log(self, p: Any, regime: str = "NEUTRAL", cooldown_days: int = 5) -> bool:
        sym = getattr(p, "symbol", None)
        if not sym:
            return False
        now = datetime.now(timezone.utc)
        with _LOCK, self._conn() as c:
            row = c.execute(
                "SELECT ts FROM signals WHERE symbol=? ORDER BY id DESC LIMIT 1", (sym,)
            ).fetchone()
            if row:
                try:
                    last = datetime.fromisoformat(row[0])
                    if (now - last).days < cooldown_days:
                        return False
                except Exception:
                    pass
            c.execute(
                """INSERT INTO signals(symbol,ts,direction,entry,sl,tp1,tp2,tp3,conf,grade,regime,size_pct)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    sym, now.isoformat(), str(getattr(p, "direction", "LONG")),
                    _num(getattr(p, "entry_price", 0)), _num(getattr(p, "stop_loss", 0)),
                    _num(getattr(p, "tp_1", 0)), _num(getattr(p, "tp_2", 0)), _num(getattr(p, "tp_3", 0)),
                    _num(getattr(p, "ai_confidence", 0)), str(getattr(p, "ai_grade", "")),
                    regime, _num(getattr(p, "position_size_pct", 0)),
                ),
            )
        return True

    # ---- scoring -------------------------------------------------------
    @staticmethod
    def _score_one(r: dict, bars: pd.DataFrame, max_days: int) -> Optional[dict]:
        sign = 1.0 if r["direction"].upper() == "LONG" else -1.0
        entry, sl = r["entry"], r["sl"]
        risk = (entry - sl) * sign
        tps = [r["tp1"], r["tp2"], r["tp3"]]
        if risk <= 0 or any(t <= 0 for t in tps) or bars is None or bars.empty:
            return None
        rr = [((t - entry) * sign) / risk for t in tps]
        hit = [False, False, False]
        mfe = 0.0
        for ts, bar in bars.iterrows():
            lo, hi = float(bar["low"]), float(bar["high"])
            worst = lo if sign > 0 else hi
            best = hi if sign > 0 else lo
            if (worst - sl) * sign <= 0:  # stop first
                realized = sum(rr[i] / 3 for i in range(3) if hit[i]) - (3 - sum(hit)) / 3
                return dict(status="CLOSED", exit_ts=str(ts), exit_reason="SL", r=realized, mfe=mfe)
            mfe = max(mfe, ((best - entry) * sign) / risk)
            for i in range(3):
                if not hit[i] and (best - tps[i]) * sign >= 0:
                    hit[i] = True
            if all(hit):
                return dict(status="CLOSED", exit_ts=str(ts), exit_reason="TP3", r=sum(rr) / 3, mfe=mfe)
        age = (pd.Timestamp.now() - pd.Timestamp(r["ts"]).tz_localize(None)).days
        if age >= max_days:
            last_c = float(bars["close"].iloc[-1])
            mtm = ((last_c - entry) * sign) / risk
            realized = sum(rr[i] / 3 for i in range(3) if hit[i]) + (3 - sum(hit)) / 3 * mtm
            return dict(status="CLOSED", exit_ts=str(bars.index[-1]), exit_reason="EXPIRED", r=realized, mfe=mfe)
        return None

    def evaluate(self, fetch_fn: Callable[[str, pd.Timestamp], Optional[pd.DataFrame]], max_days: int = 120) -> int:
        """fetch_fn(symbol, start_ts) -> DataFrame[open, high, low, close] indexed by date."""
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            rows = [dict(x) for x in c.execute("SELECT * FROM signals WHERE status='OPEN'")]
        closed = 0
        for r in rows:
            try:
                start = pd.Timestamp(r["ts"]).tz_localize(None).normalize() + pd.Timedelta(days=1)
                res = self._score_one(r, fetch_fn(r["symbol"], start), max_days)
                if res:
                    with _LOCK, self._conn() as c:
                        c.execute(
                            "UPDATE signals SET status=?,exit_ts=?,exit_reason=?,r_result=?,mfe_r=? WHERE id=?",
                            (res["status"], res["exit_ts"], res["exit_reason"], res["r"], res["mfe"], r["id"]),
                        )
                    closed += 1
            except Exception:
                continue
        return closed

    # ---- calibration ---------------------------------------------------
    def calibration(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT conf, regime, r_result FROM signals WHERE status='CLOSED'", c)
        if df.empty:
            return pd.DataFrame(columns=["bucket", "n", "win_rate", "avg_R"])
        df["bucket"] = pd.cut(df["conf"], [0, 60, 70, 80, 101], labels=["<60", "60-70", "70-80", "80+"])
        g = df.groupby("bucket", observed=True)["r_result"]
        return pd.DataFrame(
            {"n": g.size(), "win_rate": g.apply(lambda s: (s > 0).mean()).round(3), "avg_R": g.mean().round(3)}
        ).reset_index()

    def size_multiplier(self, conf: float, min_n: int = 20) -> float:
        """0.5-1.2 scale on position size, driven by realised results of this confidence bucket."""
        cal = self.calibration()
        if cal.empty:
            return 1.0
        label = "<60" if conf < 60 else "60-70" if conf < 70 else "70-80" if conf < 80 else "80+"
        row = cal[cal["bucket"].astype(str) == label]
        if row.empty or int(row["n"].iloc[0]) < min_n:
            return 1.0
        avg_r = float(row["avg_R"].iloc[0])
        return float(np.clip(1.0 + avg_r * 0.5, 0.5, 1.2))


# =====================================================================
# 3. PORTFOLIO CONSTRUCTION
# =====================================================================
REGIME_MAX_GROSS = {  # % of equity allowed in new longs
    "STRONG_BULL": 90, "BULL": 80, "NEUTRAL": 60,
    "CAUTIOUS": 45, "BEAR": 30, "STRONG_BEAR": 15,
}


def quality_rank(p: Any) -> float:
    """0-100 ranking that rewards proven edge, not just a high indicator score."""
    conf = _num(getattr(p, "ai_confidence", 0))
    dec = _num((getattr(p, "decision", {}) or {}).get("score", 0))
    ups = _num(getattr(p, "upside_score", 0))
    n_bt = int(_num(getattr(p, "bt_trades", 0)))
    exp_r = getattr(p, "bt_expectancy_R", None)
    bt_term = float(np.clip(_num(exp_r) * 40 + 50, 0, 100)) if (exp_r is not None and n_bt >= 12) else 50.0
    mc = _num(getattr(p, "mc_ev_r", 0))
    mc_term = float(np.clip(mc * 40 + 50, 0, 100)) if mc else 50.0
    rr = float(np.clip(_num(getattr(p, "risk_reward", 1)) / 3.0, 0, 1)) * 100
    score = 0.30 * conf + 0.20 * dec + 0.20 * ups + 0.15 * bt_term + 0.10 * mc_term + 0.05 * rr
    score -= 0.25 * _num(getattr(p, "event_risk_score", 0)) / 4  # up to -6
    if str(getattr(p, "crash_warning_level", "LOW")) in ("HIGH", "CRITICAL"):
        score -= 8
    return round(float(score), 2)


class PortfolioConstructor:
    def __init__(self, max_positions: int = 10, max_per_sector: int = 2,
                 corr_threshold: float = 0.80, max_open_risk_pct: float = 6.0,
                 max_single_pct: float = 8.0):
        self.max_positions = max_positions
        self.max_per_sector = max_per_sector
        self.corr_threshold = corr_threshold
        self.max_open_risk_pct = max_open_risk_pct
        self.max_single_pct = max_single_pct

    @staticmethod
    def _risk_pct(p: Any) -> float:
        e, s = _num(getattr(p, "entry_price", 0)), _num(getattr(p, "stop_loss", 0))
        if e <= 0:
            return 0.0
        return _num(getattr(p, "position_size_pct", 0)) * abs(e - s) / e

    def select(self, proposals: List[Any], regime: str = "NEUTRAL",
               returns: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        returns: optional DataFrame of daily returns (columns = symbols) for the correlation filter.
        Mutates position_size_pct on the kept proposals (scaled to the caps) and returns a report.
        """
        notes: List[str] = []
        ranked = sorted(proposals, key=quality_rank, reverse=True)
        corr = returns.corr() if returns is not None and returns.shape[1] > 1 else None
        kept: List[Any] = []
        sector_ct: Dict[str, int] = {}
        for p in ranked:
            if len(kept) >= self.max_positions:
                break
            sym = getattr(p, "symbol", "")
            sec = getattr(p, "sector_exposure", "DEFAULT") or "DEFAULT"
            if sector_ct.get(sec, 0) >= self.max_per_sector:
                notes.append(f"{sym}: skipped (sector '{sec}' already has {self.max_per_sector})")
                continue
            if corr is not None and sym in corr.index:
                peers = [getattr(k, "symbol", "") for k in kept if getattr(k, "symbol", "") in corr.columns]
                if peers and float(corr.loc[sym, peers].abs().max()) > self.corr_threshold:
                    notes.append(f"{sym}: skipped (corr>{self.corr_threshold} with an already-selected name)")
                    continue
            kept.append(p)
            sector_ct[sec] = sector_ct.get(sec, 0) + 1

        for p in kept:  # per-name cap
            p.position_size_pct = min(_num(p.position_size_pct), self.max_single_pct)

        gross_cap = REGIME_MAX_GROSS.get(regime, 60)
        gross = sum(_num(p.position_size_pct) for p in kept)
        if gross > gross_cap and gross > 0:
            f = gross_cap / gross
            for p in kept:
                p.position_size_pct = round(_num(p.position_size_pct) * f, 2)
            notes.append(f"Gross exposure scaled {gross:.1f}% -> {gross_cap}% (regime {regime})")

        heat = sum(self._risk_pct(p) for p in kept)
        if heat > self.max_open_risk_pct and heat > 0:
            f = self.max_open_risk_pct / heat
            for p in kept:
                p.position_size_pct = round(_num(p.position_size_pct) * f, 2)
            notes.append(f"Open risk scaled {heat:.2f}% -> {self.max_open_risk_pct}% of equity")

        return {
            "selected": kept,
            "gross_pct": round(sum(_num(p.position_size_pct) for p in kept), 2),
            "open_risk_pct": round(sum(self._risk_pct(p) for p in kept), 2),
            "cash_pct": round(100 - sum(_num(p.position_size_pct) for p in kept), 2),
            "notes": notes,
        }


# =====================================================================
# 4. TELEGRAM DE-DUPLICATION
# =====================================================================
class AlertDeduper:
    """Only re-alert if: new symbol, >N days old, confidence moved >=8 pts, or price moved >5%."""

    def __init__(self, path: str = ALERT_STATE_FILE, min_days: float = 3.0):
        self.path, self.min_days = path, min_days

    def filter_new(self, proposals: List[Any], commit: bool = True) -> List[Any]:
        state = _load_json(self.path, {})
        now, out = time.time(), []
        for p in proposals:
            sym = getattr(p, "symbol", "")
            prev = state.get(sym)
            conf, px = _num(getattr(p, "ai_confidence", 0)), _num(getattr(p, "entry_price", 0))
            fresh = (
                prev is None
                or (now - prev["ts"]) / 86400 >= self.min_days
                or abs(conf - prev["conf"]) >= 8
                or (prev["px"] > 0 and abs(px / prev["px"] - 1) > 0.05)
            )
            if fresh:
                out.append(p)
                state[sym] = {"ts": now, "conf": conf, "px": px}
        if commit:
            _save_json(self.path, state)
        return out


# =====================================================================
# 5. CORRECT SQUEEZE DURATION
# =====================================================================
def count_trailing_true(mask: pd.Series) -> int:
    """Number of consecutive True values ending at the last bar."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or not arr[-1]:
        return 0
    idx = np.flatnonzero(~arr)
    return int(arr.size if idx.size == 0 else arr.size - idx[-1] - 1)
