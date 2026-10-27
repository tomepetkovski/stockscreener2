from __future__ import annotations

import importlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # optional dependency
    st_autorefresh = None

st.set_page_config(page_title="Stock Screener 2 · MTF EMA", page_icon="📈", layout="wide")

# ── Style ───────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px; padding: 10px 14px;}
.hero {background: linear-gradient(90deg,#0f766e,#1d4ed8); color:#fff; padding:16px 22px;
    border-radius:14px; margin-bottom:14px;}
.hero h2 {margin:0; font-size:1.6rem;} .hero span {opacity:.85; font-size:.9rem;}
.lvl {display:flex; justify-content:space-between; padding:7px 12px; margin:4px 0;
    border-radius:8px; font-weight:600; font-size:.95rem;}
.lvl small {font-weight:400; opacity:.85;}
.entry {background:rgba(59,130,246,.18); border-left:4px solid #3b82f6;}
.sl {background:rgba(239,68,68,.18); border-left:4px solid #ef4444;}
.tp {background:rgba(34,197,94,.16); border-left:4px solid #22c55e;}
.exp {background:rgba(245,158,11,.18); border-left:4px solid #f59e0b;}
</style>
""", unsafe_allow_html=True)

# ── Criteria ────────────────────────────────────────────────────
TFS = ["15m", "1h", "4h", "1d", "1w"]
STAGE_ORDER = ["1w", "1d", "4h", "1h", "15m"]          # cheapest / most selective first
PRICE_OPTS = ["Any", "Above", "Below"]
CROSS_OPTS = ["Any", "50>200", "50<200"]

DEFAULT_RULES = {
    "15m": ("Any", "Any", "Any"),
    "1h":  ("Any", "Below", "Any"),
    "4h":  ("Any", "Any", "Any"),
    "1d":  ("Above", "Any", "Any"),
    "1w":  ("Above", "Any", "Any"),
}

# (Price vs EMA50, Price vs EMA200, EMA50 vs EMA200)
PRESETS = {
    "Custom": None,
    "Full stack (4H–1W)": {
        "15m": ("Any", "Any", "Any"), "1h": ("Any", "Any", "Any"),
        "4h": ("Above", "Above", "50>200"), "1d": ("Above", "Above", "50>200"),
        "1w": ("Above", "Above", "50>200"),
    },
    "Pullback in uptrend": {
        "15m": ("Any", "Any", "Any"),
        "1h": ("Below", "Above", "50>200"), "4h": ("Below", "Above", "50>200"),
        "1d": ("Above", "Above", "50>200"), "1w": ("Above", "Above", "50>200"),
    },
    "4H reclaim, 1D still below": {
        "15m": ("Any", "Any", "Any"), "1h": ("Any", "Any", "Any"),
        "4h": ("Above", "Above", "50>200"), "1d": ("Any", "Below", "Any"),
        "1w": ("Any", "Any", "Any"),
    },
    "Full bear stack": {
        "15m": ("Any", "Any", "Any"), "1h": ("Any", "Any", "Any"),
        "4h": ("Below", "Below", "50<200"), "1d": ("Below", "Below", "50<200"),
        "1w": ("Below", "Below", "50<200"),
    },
    "Investor trend (1D–1W) + 4H pullback": {      # extra preset for 3–12 month holds
        "15m": ("Any", "Any", "Any"), "1h": ("Any", "Any", "Any"),
        "4h": ("Below", "Any", "Any"),
        "1d": ("Above", "Above", "50>200"), "1w": ("Above", "Above", "50>200"),
    },
}

# yfinance interval / period per timeframe (4h is built from 1h)
TF_SPEC = {
    "15m": ("15m", "59d"),
    "1h": ("1h", "729d"),
    "4h": ("1h", "729d"),
    "1d": ("1d", "3y"),
    "1w": ("1wk", "10y"),
}
INTRADAY = {"15m", "1h", "4h"}

# ── Universe ────────────────────────────────────────────────────
DEFAULT_UNIVERSE = (
    "AAPL MSFT NVDA GOOGL GOOG AMZN META TSLA AVGO ORCL BRK-B TSM CRM ADBE AMD INTC PLTR NOW UBER SHOP "
    "PANW SNOW ARM MSTR JPM BAC WFC GS MS BLK SCHW AXP C V MA PYPL UNH LLY JNJ PFE ABBV MRK ABT TMO DHR "
    "NVO AZN BMY GILD BIIB VRTX WMT COST PG KO PEP MCD NKE HD LOW TGT SBUX CMG CAT DE GE UPS BA HON RTX "
    "LMT XOM CVX COP SLB TXN QCOM MU AMAT LRCX KLAC ASML NFLX DIS CMCSA VZ NEE DUK SO D AEP EXC CEG SRE "
    "LIN APD SHW FCX PLD EQIX CRWD ZS DDOG FTNT NET APP DKNG CCL MAR HLT LULU OXY SMCI BMRN SYK PGR TRV "
    "ALL SPGI MSCI ICE CME MDB DOCU TEAM ZM SQ WMB FDX FAST PAYX CTAS ODFL COIN HOOD IONQ RKLB SOFI RDDT "
    "ANET ISRG REGN MRVL SNPS CDNS INTU NOC GD ETN PH TJX ORLY BKNG ABNB"
).split()


@st.cache_resource(show_spinner=False)
def load_terminal(modname: str):
    """Import your Institutional Breakout Terminal module (optional)."""
    try:
        return importlib.import_module(modname), None
    except Exception as exc:                       # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# ── Data layer (yfinance, batched + cached) ─────────────────────
def _frames(raw: pd.DataFrame, syms: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    cols = ["open", "high", "low", "close"]
    if not isinstance(raw.columns, pd.MultiIndex):
        d = raw.copy()
        d.columns = [str(c).lower() for c in d.columns]
        if len(syms) == 1 and all(c in d.columns for c in cols):
            out[syms[0]] = d[cols].dropna(subset=["close"])
        return out
    lv0 = set(raw.columns.get_level_values(0))
    swap = not (set(syms) & lv0)                   # handles both (ticker, field) / (field, ticker)
    for s in syms:
        try:
            d = raw.xs(s, axis=1, level=1 if swap else 0).copy()
        except KeyError:
            continue
        d.columns = [str(c).lower() for c in d.columns]
        if not all(c in d.columns for c in cols):
            continue
        d = d[cols].dropna(subset=["close"])
        if len(d):
            out[s] = d
    return out


def to_4h(d: pd.DataFrame) -> pd.DataFrame:
    """1h -> 4h bars anchored to the 09:30 NY open (09:30-13:30, 13:30-17:30)."""
    r = d.resample("4h", offset="1h30min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return r


@st.cache_data(ttl=300, show_spinner=False)
def download_tf(syms: tuple, tf: str) -> dict[str, pd.DataFrame]:
    interval, period = TF_SPEC[tf]
    res: dict[str, pd.DataFrame] = {}
    for i in range(0, len(syms), 80):
        chunk = list(syms[i:i + 80])
        raw = None
        for _ in range(2):
            try:
                raw = yf.download(chunk, period=period, interval=interval, auto_adjust=True,
                                  progress=False, group_by="ticker", threads=True)
                if raw is not None and not raw.empty:
                    break
            except Exception:                      # noqa: BLE001
                time.sleep(1.0)
        for s, d in _frames(raw, chunk).items():
            try:
                res[s] = to_4h(d) if tf == "4h" else d
            except Exception:                      # noqa: BLE001
                continue
    return res


# ── Rule engine ─────────────────────────────────────────────────
def evaluate(rule, p, e50, e200, max_dist):
    m50, m200, mx = rule
    ok50 = m50 == "Any" or (p > e50 if m50 == "Above" else p < e50)
    ok200 = m200 == "Any" or (p > e200 if m200 == "Above" else p < e200)
    okx = mx == "Any" or (e50 > e200 if mx == "50>200" else e50 < e200)
    d50, d200 = (p / e50 - 1) * 100, (p / e200 - 1) * 100
    dok50 = max_dist == 0 or m50 == "Any" or abs(d50) <= max_dist
    dok200 = max_dist == 0 or m200 == "Any" or abs(d200) <= max_dist
    return bool(ok50 and ok200 and okx and dok50 and dok200), d50, d200


@st.cache_data(ttl=300, show_spinner=False)
def scan_rows(symbols: tuple, rules: tuple, lf: int, ls: int, max_dist: float,
              live: bool, fast: bool) -> list[dict]:
    active = [(tf, r) for tf, r in rules if any(x != "Any" for x in r)]
    if not active:
        return []
    total, rmap = len(active), dict(active)
    stages = [tf for tf in STAGE_ORDER if tf in rmap]
    state = {s: {"Symbol": s, "_p": 0, "_f": 0} for s in symbols}
    alive = list(symbols)

    for tf in stages:
        if not alive:
            break
        data = download_tf(tuple(alive), tf)
        nxt = []
        for s in alive:
            d = data.get(s)
            if d is None or len(d) < ls + 2:       # not enough history -> skip symbol
                state.pop(s, None)
                continue
            c = d["close"]
            e50 = c.ewm(span=lf, adjust=False).mean()
            e200 = c.ewm(span=ls, adjust=False).mean()
            i = -1 if live else -2
            p, a, b = float(c.iloc[i]), float(e50.iloc[i]), float(e200.iloc[i])
            ok, d50, d200 = evaluate(rmap[tf], p, a, b, max_dist)
            row = state[s]
            row[tf] = "✅" if ok else "❌"
            row[f"{tf} %50"] = round(d50, 2)
            row[f"{tf} %200"] = round(d200, 2)
            row["Price"] = p
            row["_p"] += int(ok)
            row["_f"] += int(not ok)
            if not (fast and row["_f"] >= 2):      # 2 fails => can't be MATCH/WATCH any more
                nxt.append(s)
        alive = nxt

    rows = []
    for row in state.values():
        passed = row.pop("_p")
        row.pop("_f")
        row["Passed"] = f"{passed}/{total}"
        row["Status"] = ("MATCH" if passed == total
                         else "WATCH" if total >= 2 and passed == total - 1 else "-")
        row["_score"] = passed / total
        rows.append(row)
    return rows


def _dte(sym: str):
    try:
        cal = yf.Ticker(sym).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if not dates:
            return None
        if not isinstance(dates, (list, tuple)):
            dates = [dates]
        today = pd.Timestamp.now().date()
        ds = [(pd.Timestamp(x).date() - today).days for x in dates]
        ds = [x for x in ds if x >= 0]
        return min(ds) if ds else None
    except Exception:                              # noqa: BLE001
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def earnings_days(symbols: tuple) -> dict:
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(zip(symbols, ex.map(_dte, symbols)))


def telegram(msg: str):
    try:
        tok, chat = st.secrets["TG_TOKEN"], st.secrets["TG_CHAT"]
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": msg}, timeout=10)
    except Exception:                              # noqa: BLE001
        pass


# ── Trade plan + Monte Carlo ────────────────────────────────────
def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def trade_plan(df, direction, sl_mode, atr_mult, rr):
    entry = float(df["close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    long = direction == "Long"
    if sl_mode == "Swing (10 bars)":
        sl = df["low"].iloc[-10:].min() - 0.1 * a if long else df["high"].iloc[-10:].max() + 0.1 * a
    else:
        sl = entry - atr_mult * a if long else entry + atr_mult * a
    risk = abs(entry - sl)
    sign = 1 if long else -1
    return entry, float(sl), [entry + sign * risk * r for r in rr], float(risk)


def monte_carlo(close, entry, sl, tps, rr, long, horizon, n_paths, drift, seed=42):
    rng = np.random.default_rng(seed)
    r = np.diff(np.log(close))[-500:]
    if not drift:
        r = r - r.mean()
    paths = entry * np.exp(np.cumsum(r[rng.integers(0, len(r), (n_paths, horizon))], axis=1))

    def first(mask):
        hit = mask.any(axis=1)
        return np.where(hit, mask.argmax(axis=1), horizon + 1)

    t_sl = first(paths <= sl) if long else first(paths >= sl)
    t_tp = [first(paths >= tp) if long else first(paths <= tp) for tp in tps]
    term = paths[:, -1]
    sign = 1 if long else -1
    risk = abs(entry - sl)
    term_r = np.clip(sign * (term - entry) / risk, -1, None)
    ev = 0.0                                       # 1/3 of the position exits at each TP
    for t, R in zip(t_tp, rr):
        out = np.where((t < t_sl) & (t <= horizon), R,
                       np.where((t_sl < t) & (t_sl <= horizon), -1.0, term_r))
        ev += out.mean() / len(rr)
    return {
        "paths": paths,
        "p_tp": [float(((t < t_sl) & (t <= horizon)).mean()) for t in t_tp],
        "p_sl": float(((t_sl < np.minimum.reduce(t_tp)) & (t_sl <= horizon)).mean()),
        "exp_price": float(term.mean()), "median": float(np.median(term)),
        "p5": float(np.percentile(term, 5)), "p95": float(np.percentile(term, 95)),
        "ev_r": float(ev),
    }


def fmt(x: float) -> str:
    return f"{x:,.4f}" if x < 1 else (f"{x:,.3f}" if x < 10 else f"{x:,.2f}")


def build_chart(df, lf, ls, entry, sl, tps, mc, tf, long, bars=160):
    """Integer x-axis (bar index) so overnight / weekend gaps never show on intraday charts."""
    c = df["close"]
    d = df.assign(e50=c.ewm(span=lf, adjust=False).mean(),
                  e200=c.ewm(span=ls, adjust=False).mean()).iloc[-bars:]
    n, H = len(d), mc["paths"].shape[1]
    x = list(range(n))
    ft = list(range(n - 1, n + H))
    pct = np.percentile(mc["paths"], [5, 25, 50, 75, 95], axis=0)
    pct = np.hstack([np.full((5, 1), entry), pct])
    lab = d.index.strftime("%d %b %H:%M" if tf in INTRADAY else "%d %b %Y").tolist()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 text=lab, name="Price",
                                 increasing_line_color="#22c55e", decreasing_line_color="#ef4444"))
    fig.add_trace(go.Scatter(x=x, y=d["e50"], name=f"EMA{lf}", line=dict(color="#f59e0b", width=1.5)))
    fig.add_trace(go.Scatter(x=x, y=d["e200"], name=f"EMA{ls}", line=dict(color="#a855f7", width=1.5)))
    fig.add_trace(go.Scatter(x=ft, y=pct[4], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[0], fill="tonexty", fillcolor="rgba(96,165,250,.14)",
                             line=dict(width=0), name="MC 5–95%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[3], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[1], fill="tonexty", fillcolor="rgba(96,165,250,.28)",
                             line=dict(width=0), name="MC 25–75%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ft, y=pct[2], name="MC median", line=dict(color="#60a5fa", dash="dot", width=2)))
    fig.add_trace(go.Scatter(x=[ft[0], ft[-1]], y=[entry, mc["exp_price"]], name="Expected price",
                             line=dict(color="#f59e0b", width=3)))
    x1 = ft[-1]
    fig.add_hrect(y0=min(entry, sl), y1=max(entry, sl), fillcolor="rgba(239,68,68,.10)", line_width=0)
    fig.add_hrect(y0=min(entry, tps[-1]), y1=max(entry, tps[-1]), fillcolor="rgba(34,197,94,.07)", line_width=0)
    levels = [("ENTRY", entry, "#3b82f6"), ("SL", sl, "#ef4444")] + \
             [(f"TP{i + 1}", tp, "#22c55e") for i, tp in enumerate(tps)] + [("EXP", mc["exp_price"], "#f59e0b")]
    for name, y, col in levels:
        fig.add_shape(type="line", x0=0, x1=x1, y0=y, y1=y,
                      line=dict(color=col, width=1.4, dash="solid" if name in ("ENTRY", "SL") else "dash"))
        fig.add_annotation(x=x1, y=y, text=f"<b>{name}</b> {fmt(y)}", showarrow=False, xanchor="left",
                           font=dict(color=col, size=11))
    step = max(1, n // 8)
    tv = list(range(0, n, step))
    fig.update_layout(height=640, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=130, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=1.04), title=f"{'LONG' if long else 'SHORT'} plan · {tf}",
                      xaxis=dict(tickmode="array", tickvals=tv, ticktext=[lab[i] for i in tv]))
    return fig


# ── Terminal confluence (optional) ──────────────────────────────
def terminal_regime(mod):
    try:
        return mod.MarketRegimeEngine.analyze()
    except Exception:                              # noqa: BLE001
        return {"_overall": "NEUTRAL", "_score": 0, "_bias": {"direction": "SELECTIVE"}}


def terminal_one(mod, sym: str, regime: dict) -> dict:
    try:
        df = mod._download_timeframe(sym, "2y", "1d")
        if df is None or df.empty:
            return {"Symbol": sym, "Terminal": "no data"}
        p = mod.ProposalEngine.build(sym, df, regime, sector_rotation=None,
                                     use_order_flow=True, use_ml=False)
        if p is None:
            return {"Symbol": sym, "Terminal": "filtered (no setup / event risk / regime gate)"}
        ac = p.analyst_consensus or {}
        return {"Symbol": sym, "Terminal": "setup", "Dir": p.direction, "AI %": round(p.ai_confidence, 1),
                "Grade": p.ai_grade, "R:R": p.risk_reward, "Analyst": ac.get("label", "—"),
                "Buffett": p.fundamental_quality, "Event risk": p.event_risk_score,
                "Crash": p.crash_warning_level, "Hold": p.hold_period,
                "BT win": p.bt_win_rate, "BT exp R": p.bt_expectancy_R}
    except Exception as exc:                       # noqa: BLE001
        return {"Symbol": sym, "Terminal": f"error: {exc}"}


# ── Sidebar ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
live = st.sidebar.toggle("Use live price (off = last closed candle)", True)
lf = st.sidebar.number_input("EMA fast", 1, 500, 50)
ls = st.sidebar.number_input("EMA slow", 1, 500, 200)
max_dist = st.sidebar.number_input("Max distance from EMA (%), 0 = off", 0.0, 100.0, 0.0, 0.5)
min_price = st.sidebar.number_input("Min price ($)", 0.0, 10000.0, 5.0, 1.0)
fast = st.sidebar.toggle("Fast mode (stop after 2 failed TFs)", True,
                         help="Far fewer downloads; MATCH/WATCH results are identical.")
show_earn = st.sidebar.toggle("Show days to earnings (MATCH/WATCH)", True)
refresh = st.sidebar.slider("Auto-refresh (minutes, 0 = off)", 0, 30, 0)
alerts = st.sidebar.toggle("Telegram alerts (needs secrets)", False)

st.sidebar.subheader("Universe")
term_mod_name = st.sidebar.text_input(
    "Terminal module name (optional)", "stockscreener",
    help="File name of your Institutional Breakout Terminal without .py. Enables its universe, "
         "sectors and AI grading. Leave it if the import fails - the screener works standalone.")
term, term_err = load_terminal(term_mod_name.strip()) if term_mod_name.strip() else (None, None)
uni_opts = ["Built-in (~190 large caps)", "Custom list"]
if term is not None and getattr(term, "_STOCK_UNIVERSE", None):
    uni_opts.insert(0, "Terminal universe")
uni_mode = st.sidebar.selectbox("Universe source", uni_opts)
if uni_mode == "Terminal universe":
    universe = list(term._STOCK_UNIVERSE)
elif uni_mode == "Custom list":
    raw_u = st.sidebar.text_area("Tickers (comma / space / newline)", "AAPL, MSFT, NVDA, AMD, MU", height=120)
    universe = [t for t in raw_u.replace(",", " ").upper().split() if t]
else:
    universe = list(DEFAULT_UNIVERSE)
universe = list(dict.fromkeys(universe))
max_syms = st.sidebar.slider("Max symbols to scan", 20, max(len(universe), 40), min(len(universe), 400), 10)
universe = universe[:max_syms]
if term_mod_name.strip() and term is None:
    st.sidebar.caption(f"Terminal not loaded: {term_err}")

st.sidebar.subheader("EMA criteria presets")
preset_name = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0)

st.sidebar.subheader("Rules per timeframe")
manual_rules = []
for tf in TFS:
    d0 = DEFAULT_RULES[tf]
    st.sidebar.markdown(f"**{tf}**")
    c1, c2, c3 = st.sidebar.columns(3)
    a = c1.selectbox("Price / EMA50", PRICE_OPTS, PRICE_OPTS.index(d0[0]), key=f"a{tf}")
    b = c2.selectbox("Price / EMA200", PRICE_OPTS, PRICE_OPTS.index(d0[1]), key=f"b{tf}")
    c = c3.selectbox("EMA50 / EMA200", CROSS_OPTS, CROSS_OPTS.index(d0[2]), key=f"c{tf}")
    manual_rules.append((tf, (a, b, c)))

if PRESETS[preset_name] is None:
    rules = manual_rules
else:
    rules = [(tf, PRESETS[preset_name][tf]) for tf in TFS]
    st.sidebar.caption(f"Active preset: {preset_name}")

if refresh and st_autorefresh:
    st_autorefresh(interval=refresh * 60_000, key="auto")

# ── Main ────────────────────────────────────────────────────────
st.markdown('<div class="hero"><h2>📈 Stock Screener 2 · MTF EMA</h2>'
            '<span>Multi-timeframe EMA criteria on stocks · trade planner · Monte Carlo · '
            'optional Institutional Terminal grading</span></div>', unsafe_allow_html=True)
if st.button("🔄 Scan now"):
    scan_rows.clear()
    download_tf.clear()

if not universe:
    st.warning("Universe is empty.")
    st.stop()
if not any(x != "Any" for _, r in rules for x in r):
    st.warning("All rules are 'Any' - set at least one criterion.")
    st.stop()

t0 = time.time()
with st.spinner(f"Scanning {len(universe)} stocks (staged: 1w → 1d → 4h → 1h → 15m)…"):
    rows = scan_rows(tuple(universe), tuple(rules), lf, ls, max_dist, live, fast)

rows = [r for r in rows if r.get("Price", 0) >= min_price]
if not rows:
    st.error("No data returned. Yahoo may be rate-limiting - retry in a minute or reduce the universe.")
    st.stop()

df = pd.DataFrame(rows).sort_values(["_score", "Symbol"], ascending=[False, True]).drop(columns="_score")

# sector column (from terminal catalog if available)
cat = getattr(term, "_UNIVERSE_CATALOG", None) if term is not None else None
if cat:
    df.insert(1, "Sector", df["Symbol"].map(lambda s: cat.get(s, {}).get("sector", "—")))

match = df[df["Status"] == "MATCH"].copy()
watch = df[df["Status"] == "WATCH"].copy()

if show_earn and (len(match) + len(watch)):
    ed = earnings_days(tuple(list(match["Symbol"]) + list(watch["Symbol"]))[:80])
    for part in (match, watch):
        part["Earnings (d)"] = part["Symbol"].map(ed)
    df["Earnings (d)"] = df["Symbol"].map(ed)

lead = [c for c in ["Symbol", "Sector", "Status", "Passed", "Price", "Earnings (d)"] if c in df.columns]
tf_cols = [c for tf in TFS for c in (tf, f"{tf} %50", f"{tf} %200") if c in df.columns]


def tidy(x: pd.DataFrame) -> pd.DataFrame:
    return x[lead + tf_cols]


m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Scanned", len(df))
m2.metric("🟢 MATCH", len(match))
m3.metric("👀 WATCH", len(watch))
m4.metric("Price basis", "Live" if live else "Last close")
m5.metric("Scan time", f"{time.time() - t0:.1f}s")
st.caption(f"Updated {time.strftime('%H:%M:%S')} · 4h bars are resampled from 1h (09:30 NY anchor) · "
           "intraday data is limited by Yahoo (15m ≈ 60 days) and delayed outside US market hours")

now = set(match["Symbol"])
prev = st.session_state.get("prev_matches")
if alerts and prev is not None:
    for s in now - prev:
        telegram(f"ENTRY: {s} - all MTF EMA rules matched")
    for s in prev - now:
        telegram(f"EXIT: {s} - MTF EMA rules no longer matched")
st.session_state["prev_matches"] = now

pct_cols = {c: st.column_config.NumberColumn(c, format="%.2f%%") for c in df.columns if "%" in c}
cfg = {"Price": st.column_config.NumberColumn("Price", format="$%.2f"), **pct_cols}
t1, t2, t3, t4, t5 = st.tabs(["✅ MATCH", "👀 WATCH", "All", "🎯 Trade planner", "🏛 Terminal grading"])
for key, tab, data in (("match", t1, match), ("watch", t2, watch), ("all", t3, df)):
    with tab:
        if data.empty:
            st.info("Nothing here right now.")
        else:
            st.dataframe(tidy(data), use_container_width=True, hide_index=True, column_config=cfg)
            st.download_button("⬇️ CSV", tidy(data).to_csv(index=False), f"stock_screener_{key}.csv",
                               key=f"dl_{key}")

pool = list(match["Symbol"]) + list(watch["Symbol"])

# ── Trade planner ───────────────────────────────────────────────
with t4:
    if not pool:
        st.info("No MATCH/WATCH stocks to plan right now.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        sym = k1.selectbox("Stock", pool, format_func=lambda s: f"{'🟢' if s in now else '👀'} {s}")
        ctf = k2.selectbox("Chart / entry TF", TFS, index=3)
        dirn = k3.selectbox("Direction", ["Auto", "Long", "Short"])
        sl_mode = k4.selectbox("Stop-loss", ["ATR", "Swing (10 bars)"])
        k5, k6, k7, k8 = st.columns(4)
        atr_mult = k5.number_input("ATR multiple", 0.5, 10.0, 2.0, 0.25)
        rr_txt = k6.text_input("TP in R (TP1,TP2,TP3)", "1,2,3")
        horizon = k7.slider("MC horizon (bars)", 10, 200, 60, 5)
        n_paths = k8.select_slider("MC paths", [1000, 2000, 5000, 10000], 5000)
        k9, k10, k11 = st.columns(3)
        drift = k9.toggle("Include historical drift", True)
        equity = k10.number_input("Account size ($)", 100.0, 1e9, 10000.0, 500.0)
        risk_pct = k11.number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1)

        try:
            rr = [float(x) for x in rr_txt.split(",")][:3]
            assert len(rr) == 3 and all(x > 0 for x in rr)
        except Exception:                          # noqa: BLE001
            st.error("Enter three positive R multiples, e.g. 1,2,3")
            st.stop()

        cdf = download_tf((sym,), ctf).get(sym)
        if cdf is None or len(cdf) < ls + 5:
            st.error("Could not load enough candles for this stock/timeframe.")
        else:
            closes = cdf["close"]
            e50, e200 = float(closes.ewm(span=lf, adjust=False).mean().iloc[-1]), \
                float(closes.ewm(span=ls, adjust=False).mean().iloc[-1])
            p = float(closes.iloc[-1])
            auto = "Long" if (p > e200 and e50 >= e200) else "Short" if (p < e200 and e50 <= e200) else \
                ("Long" if p > e200 else "Short")
            direction = auto if dirn == "Auto" else dirn
            long = direction == "Long"
            entry, sl, tps, risk = trade_plan(cdf, direction, sl_mode, atr_mult, rr)
            mc = monte_carlo(closes.values, entry, sl, tps, rr, long, horizon, n_paths, drift)
            shares = int((equity * risk_pct / 100) / risk) if risk > 0 else 0

            left, right = st.columns([3, 1.1])
            left.plotly_chart(build_chart(cdf, lf, ls, entry, sl, tps, mc, ctf, long), use_container_width=True)
            with right:
                st.markdown(f"### {sym} · {'🟢 LONG' if long else '🔴 SHORT'}")
                st.markdown(f'<div class="lvl entry">ENTRY<span>{fmt(entry)}</span></div>'
                            f'<div class="lvl sl">SL<span>{fmt(sl)} '
                            f'<small>({abs(sl / entry - 1) * 100:.2f}%)</small></span></div>',
                            unsafe_allow_html=True)
                for i, (tp, R, pr) in enumerate(zip(tps, rr, mc["p_tp"])):
                    st.markdown(f'<div class="lvl tp">TP{i + 1} · {R:g}R<span>{fmt(tp)} '
                                f'<small>P={pr * 100:.0f}%</small></span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="lvl exp">Expected price<span>{fmt(mc["exp_price"])} '
                            f'<small>({(mc["exp_price"] / entry - 1) * 100:+.2f}%)</small></span></div>',
                            unsafe_allow_html=True)
                st.metric("P(SL before any TP)", f"{mc['p_sl'] * 100:.0f}%")
                st.metric("Expected value (R)", f"{mc['ev_r']:+.2f}R",
                          help="1/3 of the position exits at each TP; leftovers marked at horizon end.")
                st.metric("Median / 5–95% range", fmt(mc["median"]),
                          f"{fmt(mc['p5'])} – {fmt(mc['p95'])}", delta_color="off")
                st.metric("Position size", f"{shares:,} sh",
                          f"≈ ${shares * entry:,.0f} · risk ${equity * risk_pct / 100:,.0f}", delta_color="off")
            st.caption("Monte Carlo = bootstrap of the last ≤500 log-returns on the chosen timeframe; hits are "
                       "checked on closes and ignore overnight gaps beyond the sampled returns. A statistical "
                       "estimate, not a prediction or financial advice.")

# ── Terminal grading ────────────────────────────────────────────
with t5:
    if term is None:
        st.info("Terminal module not loaded. Put this file next to your Institutional Breakout Terminal "
                "(and its enhanced_breakout_engine.py / enhanced_risk_manager.py) and enter its file name "
                "in the sidebar.")
    elif not pool:
        st.info("No MATCH/WATCH stocks to grade right now.")
    else:
        n_grade = st.slider("How many MATCH/WATCH stocks to grade", 3, min(30, len(pool)), min(12, len(pool)))
        if st.button("🏛 Grade with Institutional Terminal (regime · events · analysts · backtest)"):
            with st.spinner("Running terminal ProposalEngine…"):
                regime = terminal_regime(term)
                with ThreadPoolExecutor(max_workers=3) as ex:
                    res = list(ex.map(lambda s: terminal_one(term, s, regime), pool[:n_grade]))
            st.session_state["terminal_res"] = (regime, pd.DataFrame(res))
        if st.session_state.get("terminal_res"):
            regime, gdf = st.session_state["terminal_res"]
            st.caption(f"Market regime: **{regime.get('_overall')}** (score {regime.get('_score')}) · "
                       f"bias {(regime.get('_bias') or {}).get('direction', '—')}")
            st.dataframe(gdf, use_container_width=True, hide_index=True)
            st.caption("'filtered' = the terminal rejected it (AI score below the LONG threshold, high earnings/"
                       "event risk, negative backtest expectancy, bear-regime gate or crash protection). "
                       "A stock that passes both the EMA criteria and the terminal is the strongest candidate.")
