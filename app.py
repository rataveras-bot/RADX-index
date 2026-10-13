"""
Copper-Brass Radiator Cost Index (CBRI)
=======================================
A daily, chain-linked Laspeyres cost index for the raw materials, energy and labor that go
into a heavy-duty copper-brass radiator. Single file: Streamlit + Plotly + Pandas.

Run:      streamlit run app.py
Storage:  index_config.json  (components, weights, rebalance history, settings)
          price_history.csv  (daily price per component; seeded with simulated data on first run)
Secrets:  ADMIN_PASSWORD (optional). When set, the sidebar admin panel asks for it.
"""
from __future__ import annotations

import datetime as dt
import hmac
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =============================================================================
# Constants
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "index_config.json"
PRICES_PATH = APP_DIR / "price_history.csv"

INDEX_NAME = "Copper-Brass Radiator Cost Index"
TICKER = "CBRI"
BASE_LEVEL = 100.0
HISTORY_DAYS = 365
LB_PER_T = 2204.62262
LIVE_REFRESH_HOURS = 6
PALETTE = ["#C8743B", "#C9A227", "#7C8A99", "#2F6F8F", "#4E8F5A", "#8E5BA8", "#B4505A", "#5B7DB4",
           "#9C6B3A", "#3E9C9C"]

# Default basket. "live" tells the optional live-data refresh how to build each price:
#   alloy  -> sum(share * price * factor)  (mass fractions, all in USD/t), also used for single series
#   basket -> 100 * sum(share * price / price_at_start)  (cost shares of unlike units)
DEFAULT_COMPONENTS = {
    "Copper": dict(
        unit="USD/t", description="Fin strip and tube stock (C11000/C12200). Proxy: COMEX HG copper.",
        live=dict(method="alloy", parts=[dict(source="yfinance", ticker="HG=F", factor=LB_PER_T, share=1.0)])),
    "Brass": dict(
        unit="USD/t", description="Header plates, tanks, side plates (CuZn35-37), priced as 65% Cu + 35% Zn.",
        live=dict(method="alloy", parts=[dict(source="yfinance", ticker="HG=F", factor=LB_PER_T, share=0.65),
                                         dict(source="fred", ticker="PZINCUSDM", factor=1.0, share=0.35)])),
    "Solder": dict(
        unit="USD/t", description="Pb-Sn solder, 70/30 by mass, for tube-to-header and tank joints.",
        live=dict(method="alloy", parts=[dict(source="fred", ticker="PLEADUSDM", factor=1.0, share=0.70),
                                         dict(source="fred", ticker="PTINUSDM", factor=1.0, share=0.30)])),
    "Energy": dict(
        unit="basket pts", description="45% natural gas (ovens), 35% diesel (freight), 20% Brent (oil-linked inputs).",
        live=dict(method="basket", parts=[dict(source="fred", ticker="DHHNGSP", share=0.45),
                                          dict(source="fred", ticker="DDFUELNYH", share=0.35),
                                          dict(source="fred", ticker="DCOILBRENTEU", share=0.20)])),
    "Labor": dict(
        unit="USD/hr", description="Direct wages. Proxy: BLS manufacturing avg hourly earnings (CES3000000008).",
        live=dict(method="alloy", parts=[dict(source="fred", ticker="CES3000000008", factor=1.0, share=1.0)])),
}
DEFAULT_WEIGHTS = {"Copper": 42.0, "Brass": 20.0, "Solder": 8.0, "Energy": 10.0, "Labor": 20.0}


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def get_secret(key: str, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


# =============================================================================
# Mock data generator
# =============================================================================
def generate_default_mock(dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """Correlated geometric Brownian motion for the underlying markets, then the five components."""
    names = ["CU", "ZN", "PB", "SN", "GAS", "DIESEL", "BRENT"]
    s0 = np.array([10800, 2900, 2050, 48000, 3.2, 2.45, 72.0])
    vol = np.array([0.22, 0.24, 0.20, 0.28, 0.55, 0.32, 0.30])
    drift = np.array([0.06, 0.03, 0.01, 0.08, 0.00, 0.02, 0.00])
    corr = np.array([
        # CU   ZN   PB   SN   GAS  DSL  BRT
        [1.00, 0.65, 0.50, 0.45, 0.15, 0.25, 0.30],
        [0.65, 1.00, 0.60, 0.40, 0.15, 0.20, 0.25],
        [0.50, 0.60, 1.00, 0.35, 0.10, 0.20, 0.20],
        [0.45, 0.40, 0.35, 1.00, 0.10, 0.15, 0.20],
        [0.15, 0.15, 0.10, 0.10, 1.00, 0.35, 0.30],
        [0.25, 0.20, 0.20, 0.15, 0.35, 1.00, 0.80],
        [0.30, 0.25, 0.20, 0.20, 0.30, 0.80, 1.00],
    ])
    rng = np.random.default_rng(seed)
    n, step = len(dates), 1 / 252
    z = rng.standard_normal((n, len(names))) @ np.linalg.cholesky(corr).T
    log_ret = (drift - 0.5 * vol ** 2) * step + vol * np.sqrt(step) * z
    log_ret[0] = 0.0
    px_ = pd.DataFrame(s0 * np.exp(np.cumsum(log_ret, axis=0)), index=dates, columns=names)

    # Labor: monthly BLS-style prints (~4%/yr wage growth), flat within the month
    months = (dates.year - dates[0].year) * 12 + (dates.month - dates[0].month)
    labor = 29.40 * (1.0033 ** months) * (1 + rng.normal(0, 0.002, months.max() + 1)[months])

    out = pd.DataFrame(index=dates)
    out["Copper"] = px_["CU"]
    out["Brass"] = 0.65 * px_["CU"] + 0.35 * px_["ZN"]
    out["Solder"] = 0.70 * px_["PB"] + 0.30 * px_["SN"]
    out["Energy"] = 100 * (0.45 * px_["GAS"] / s0[4] + 0.35 * px_["DIESEL"] / s0[5] + 0.20 * px_["BRENT"] / s0[6])
    out["Labor"] = labor
    return out.round(4)


def bridge_path(dates: pd.DatetimeIndex, start: pd.Timestamp, p0: float, pt: float,
                vol: float, seed: int) -> pd.Series:
    """Simulated history for a user-added component: a log Brownian bridge from P0 (at start) to Pt
    (at the last date), flat at P0 before start."""
    rng = np.random.default_rng(seed)
    s = pd.Series(float(p0), index=dates)
    seg = dates[dates >= start]
    n = len(seg)
    if n < 2:
        s.iloc[-1] = pt
        return s
    w = np.concatenate([[0.0], np.cumsum(rng.normal(0, np.sqrt(1 / 252), n - 1))])
    t = np.linspace(0, 1, n)
    bridge = w - t * w[-1]
    s.loc[seg] = np.exp(np.log(p0) + t * (np.log(pt) - np.log(p0)) + vol * bridge)
    return s.round(4)


# =============================================================================
# Persistence
# =============================================================================
def new_default_config(base_date: pd.Timestamp) -> dict:
    comps = {}
    for i, (name, spec) in enumerate(DEFAULT_COMPONENTS.items()):
        comps[name] = dict(spec, color=PALETTE[i % len(PALETTE)], seed=None)
    return dict(
        index_name=INDEX_NAME, ticker=TICKER, base_date=str(base_date.date()), base_level=BASE_LEVEL,
        reference_unit_cost=1000.0, auto_live=False, last_live_refresh=None,
        components=comps,
        data_status={n: "simulated" for n in comps},
        versions=[dict(id=1, effective_date=str(base_date.date()), weights=dict(DEFAULT_WEIGHTS),
                       note="Initial baseline weights", mode="base", created=now_iso())],
        archived_versions=[],
    )


def history_calendar(end=None) -> pd.DatetimeIndex:
    end = pd.Timestamp(end or dt.date.today()).normalize()
    return pd.bdate_range(end - pd.Timedelta(days=HISTORY_DAYS), end)


def ensure_columns(cfg: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Every configured component must have a price column; fill gaps with simulated history."""
    dates = prices.index
    mock = None
    for name, comp in cfg["components"].items():
        if name in prices.columns:
            continue
        if name in DEFAULT_COMPONENTS:
            mock = generate_default_mock(dates) if mock is None else mock
            prices[name] = mock[name]
        else:
            sd = comp.get("seed") or {}
            prices[name] = bridge_path(dates, pd.Timestamp(cfg["base_date"]), sd.get("p0", 100.0),
                                       sd.get("pt", 100.0), sd.get("vol", 0.2), sd.get("seed", 7))
        cfg["data_status"][name] = "simulated"
    return prices


def save_config(cfg: dict):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, default=str))


def save_prices(prices: pd.DataFrame):
    prices.rename_axis("date").to_csv(PRICES_PATH, float_format="%.6f")


def load_state() -> tuple[dict, pd.DataFrame]:
    cfg = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else None
    prices = None
    if PRICES_PATH.exists():
        prices = pd.read_csv(PRICES_PATH, index_col="date", parse_dates=["date"]).sort_index()
    if cfg is None:
        dates = history_calendar() if prices is None else prices.index
        cfg = new_default_config(dates[0])
        if prices is None:
            prices = generate_default_mock(dates)
        save_config(cfg)
    if prices is None:
        prices = pd.DataFrame(index=history_calendar())
        if pd.Timestamp(cfg["base_date"]) < prices.index[0]:
            cfg["base_date"] = str(prices.index[0].date())
    before = set(prices.columns)
    prices = ensure_columns(cfg, prices)
    if set(prices.columns) != before or not PRICES_PATH.exists():
        save_prices(prices)
        save_config(cfg)
    return cfg, prices


# =============================================================================
# Optional live data (FRED needs no key; Yahoo via yfinance is unofficial)
# =============================================================================
def fetch_fred(ticker: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}&cosd={start}"
    r = requests.get(url, timeout=30, headers={"User-Agent": "CBRI-index/1.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
    df.columns = ["date", "value"]  # header is DATE or observation_date depending on FRED version
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return pd.Series(df["value"].values, index=pd.to_datetime(df["date"])).sort_index()


def fetch_yfinance(ticker: str, start: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    idx = close.index.tz_localize(None) if close.index.tz is not None else close.index
    return pd.Series(close.values, index=pd.DatetimeIndex(idx)).dropna().sort_index()


def refresh_live(cfg: dict, prices: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    today = pd.Timestamp(dt.date.today())
    start = min(pd.Timestamp(cfg["base_date"]), today - pd.Timedelta(days=HISTORY_DAYS))
    fetch_start = (start - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    cache, raw, log, last_obs = {}, {}, [], []

    def get(part):
        key = (part["source"], part["ticker"])
        if key not in cache:
            cache[key] = fetch_fred(part["ticker"], fetch_start) if part["source"] == "fred" \
                else fetch_yfinance(part["ticker"], fetch_start)
        return cache[key]

    for name, comp in cfg["components"].items():
        spec = comp.get("live")
        if not spec:
            continue
        try:
            raw[name] = [(p, get(p)) for p in spec["parts"]]
            last_obs += [s.index.max() for _, s in raw[name]]
        except Exception as e:
            log.append(dict(component=name, status=f"error: {str(e)[:120]}"))
    if not raw:
        return prices, log

    cal = pd.bdate_range(start, min(max(last_obs), today))
    new = prices.reindex(prices.index.union(cal)).ffill().bfill().reindex(cal)
    for name, parts in raw.items():
        spec = cfg["components"][name]["live"]
        aligned = [(p, s.reindex(s.index.union(cal)).ffill().reindex(cal)) for p, s in parts]
        if spec["method"] == "basket":
            val = 100 * sum(p["share"] * s / s.iloc[0] for p, s in aligned)
        else:
            val = sum(p["share"] * p.get("factor", 1.0) * s for p, s in aligned)
        if val.isna().any():
            log.append(dict(component=name, status="error: a source has no data at the start of the window"))
            continue
        new[name] = val.round(6)
        cfg["data_status"][name] = "live"
        log.append(dict(component=name, status="ok", last=str(val.index[-1].date()), value=round(val.iloc[-1], 4)))
    return new, log


# =============================================================================
# Index engine
# =============================================================================
def active_versions(cfg: dict, dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, dict]]:
    anchors = []
    for v in sorted(cfg["versions"], key=lambda v: (v["effective_date"], v["id"])):
        pos = dates.searchsorted(max(pd.Timestamp(v["effective_date"]), dates[0]))
        if pos >= len(dates):
            continue  # scheduled beyond available data
        T = dates[pos]
        if anchors and anchors[-1][0] == T:
            anchors[-1] = (T, v)  # same anchor date: the later version wins
        else:
            anchors.append((T, v))
    if not anchors:
        raise ValueError("no weight version is effective within the data range")
    anchors[0] = (dates[0], anchors[0][1])
    return anchors


def compute_index(prices: pd.DataFrame, cfg: dict):
    """
    Chain-linked Laspeyres. For rebalance period k starting at T_k:
        I_t = I_{T_k} * sum_i w_i^k * P_i,t / P_i,T_k,      I_{T_0} = base level (100)
    Equivalent divisor form: I_t = sum_i q_i^k P_i,t / D_k with q_i^k = w_i^k U / P_i,T_k
    (units of input i in a radiator costing U at T_k) and D_k = U / I_{T_k}.
    """
    base = pd.Timestamp(cfg["base_date"])
    p = prices.loc[prices.index >= base]
    if p.empty:
        raise ValueError("no price data on or after the base date")
    dates = p.index
    U = float(cfg.get("reference_unit_cost", 1000.0))
    anchors = active_versions(cfg, dates)
    names = list(dict.fromkeys(n for _, v in anchors for n, w in v["weights"].items() if w > 0))

    index = pd.Series(np.nan, index=dates, name=TICKER)
    contrib = pd.DataFrame(0.0, index=dates, columns=names)
    eff_w = pd.DataFrame(0.0, index=dates, columns=names)
    periods = []
    level = float(cfg.get("base_level", BASE_LEVEL))
    for k, (T, v) in enumerate(anchors):
        T_next = anchors[k + 1][0] if k + 1 < len(anchors) else None
        seg = dates[(dates >= T) & (dates <= T_next)] if T_next is not None else dates[dates >= T]
        write = seg[seg < T_next] if T_next is not None else seg
        w = {n: x for n, x in v["weights"].items() if x > 0}
        tot = sum(w.values())
        bad = [n for n in w if n not in p.columns or pd.isna(p.at[T, n]) or p.at[T, n] <= 0]
        if bad:
            raise ValueError(f"version {v['id']}: no valid price on {T.date()} for {bad}")
        seg_val = pd.Series(0.0, index=seg)
        parts = {}
        for n, x in w.items():
            parts[n] = level * (x / tot) * p.loc[seg, n] / p.at[T, n]
            seg_val += parts[n]
        index.loc[write] = seg_val.loc[write]
        for n, s in parts.items():
            contrib.loc[write, n] = s.loc[write]
            eff_w.loc[write, n] = (s / seg_val).loc[write]
        periods.append(dict(version=v["id"], anchor=T, level=level, divisor=U / level,
                            weights={n: x / tot for n, x in w.items()},
                            spend={n: x / tot * U for n, x in w.items()},
                            qty={n: x / tot * U / p.at[T, n] for n, x in w.items()},
                            note=v.get("note", ""), mode=v.get("mode", "")))
        level = float(seg_val.iloc[-1])  # I at T_{k+1}; sum w' * 1 = 1 makes the link seamless
    return index, contrib, eff_w, periods


def current_weights(cfg: dict) -> dict:
    v = sorted(cfg["versions"], key=lambda v: (v["effective_date"], v["id"]))[-1]
    return {n: float(v["weights"].get(n, 0.0)) for n in cfg["components"]}


def apply_weights(cfg: dict, weights: dict, mode: str, eff_date: dt.date, note: str):
    """mode: 'chain' (new period from eff_date, history kept) or 'restate' (one period from base)."""
    w = {n: round(float(x), 6) for n, x in weights.items() if x > 0}
    next_id = max(v["id"] for v in cfg["versions"] + cfg.get("archived_versions", [])) + 1
    if mode == "restate":
        cfg.setdefault("archived_versions", []).extend(cfg["versions"])
        cfg["versions"] = [dict(id=next_id, effective_date=cfg["base_date"], weights=w,
                                note=note or "Restated history", mode="restate", created=now_iso())]
    else:
        eff = str(eff_date)
        last = max(cfg["versions"], key=lambda v: v["effective_date"])
        if eff < last["effective_date"]:
            raise ValueError(f"effective date must be on or after {last['effective_date']}")
        if eff == last["effective_date"]:  # same day: replace that period instead of stacking
            cfg["versions"].remove(last)
            cfg.setdefault("archived_versions", []).append(last)
        cfg["versions"].append(dict(id=next_id, effective_date=eff, weights=w,
                                    note=note or "Rebalance", mode="chain", created=now_iso()))


def rescale(weights: dict, keep_total: float = 100.0) -> dict:
    tot = sum(weights.values())
    return {n: x / tot * keep_total for n, x in weights.items()} if tot > 0 else weights


# =============================================================================
# Charts
# =============================================================================
def color_of(cfg, name, i=0):
    return cfg["components"].get(name, {}).get("color", PALETTE[i % len(PALETTE)])


def index_chart(cfg, index: pd.Series, contrib: pd.DataFrame, periods):
    cols = [c for c in contrib.columns if contrib[c].abs().sum() > 0]
    lines = "<br>".join(f"{c}: %{{customdata[{j}]:,.2f}} pts" for j, c in enumerate(cols))
    fig = go.Figure(go.Scatter(
        x=index.index, y=index.values, mode="lines", name=TICKER,
        line=dict(color="#C8743B", width=2.4), customdata=contrib[cols].values,
        hovertemplate="<b>%{x|%a %d %b %Y}</b><br>" + TICKER + " <b>%{y:,.2f}</b><br>"
                      "<span style='font-size:11px'>Contribution</span><br>" + lines + "<extra></extra>"))
    for per in periods[1:]:
        fig.add_vline(x=per["anchor"], line_dash="dot", line_color="#8a8a8a", opacity=0.7)
        fig.add_annotation(x=per["anchor"], y=1, yref="paper", text=f"v{per['version']}", showarrow=False,
                           yanchor="bottom", font=dict(size=10, color="#8a8a8a"))
    fig.add_hline(y=cfg.get("base_level", BASE_LEVEL), line_color="#999", line_width=1, opacity=0.35)
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=36, b=10), hovermode="x", showlegend=False,
        yaxis_title="Index level",
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.07), type="date",
                   rangeselector=dict(buttons=[
                       dict(count=1, label="1D", step="day", stepmode="backward"),
                       dict(count=1, label="1M", step="month", stepmode="backward"),
                       dict(count=6, label="6M", step="month", stepmode="backward"),
                       dict(count=1, label="1Y", step="year", stepmode="backward"),
                       dict(step="all", label="ALL")])))
    return fig


def donut(cfg, weights: dict, title: str):
    names = [n for n, w in weights.items() if w > 0]
    fig = go.Figure(go.Pie(
        labels=names, values=[weights[n] for n in names], hole=0.58, sort=False,
        marker=dict(colors=[color_of(cfg, n, i) for i, n in enumerate(names)]),
        texttemplate="%{label}<br>%{percent:.1%}", hovertemplate="%{label}: %{percent:.2%}<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      annotations=[dict(text=title, showarrow=False, font=dict(size=13))])
    return fig


def stacked_area(cfg, contrib: pd.DataFrame):
    fig = go.Figure()
    for i, c in enumerate(contrib.columns):
        fig.add_trace(go.Scatter(x=contrib.index, y=contrib[c], name=c, stackgroup="one",
                                 line=dict(width=0.5, color=color_of(cfg, c, i)),
                                 hovertemplate=f"{c}: %{{y:,.2f}} pts<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="Index points", legend=dict(orientation="h", y=-0.15))
    return fig


def relatives_chart(cfg, prices: pd.DataFrame):
    rel = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, c in enumerate(rel.columns):
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=c, mode="lines",
                                 line=dict(color=color_of(cfg, c, i)),
                                 customdata=prices[c].values,
                                 hovertemplate=f"{c}: %{{y:.1f}} (price %{{customdata:,.2f}})<extra></extra>"))
    fig.add_hline(y=100, line_color="#999", line_width=1, opacity=0.35)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title="Price relative (base = 100)", legend=dict(orientation="h", y=-0.15))
    return fig


# =============================================================================
# Public dashboard
# =============================================================================
def metric(col, label, value, delta=None, **kw):
    try:
        col.metric(label, value, delta, **kw)
    except TypeError:  # older Streamlit without sparkline support
        col.metric(label, value, delta)


def value_at_or_before(s: pd.Series, d: pd.Timestamp):
    s = s.loc[:d].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def page_dashboard(cfg, prices, index, contrib, eff_w, periods):
    U = float(cfg.get("reference_unit_cost", 1000.0))
    base_level = float(cfg.get("base_level", BASE_LEVEL))
    last_d, last = index.index[-1], float(index.iloc[-1])
    prev = float(index.iloc[-2]) if len(index) > 1 else last
    chg, chg_pct = last - prev, (last / prev - 1) * 100
    v30 = value_at_or_before(index, last_d - pd.Timedelta(days=30))
    trend30 = (last / v30 - 1) * 100 if not np.isnan(v30) else np.nan
    spark = index.loc[last_d - pd.Timedelta(days=30):].round(3).tolist()

    c = st.columns(4)
    metric(c[0], f"{TICKER} · {last_d:%d %b %Y}", f"{last:,.2f}", f"{chg:+.2f} pts")
    metric(c[1], "1-day change (per radiator)", f"${U * chg / base_level:+,.2f}", f"{chg_pct:+.2f}%",
           help=f"Index change applied to the reference cost of ${U:,.0f} per radiator at base.")
    metric(c[2], "30-day trend", f"{trend30:+.2f}%", f"{last - v30:+.2f} pts",
           chart_data=spark, chart_type="line")
    metric(c[3], "Implied material+conversion cost", f"${U * last / base_level:,.2f}",
           f"{(last / base_level - 1) * 100:+.2f}% vs base")

    st.plotly_chart(index_chart(cfg, index, contrib, periods), width="stretch")

    t1, t2, t3 = st.tabs(["Weight breakdown", "Contributions over time", "Component prices"])
    with t1:
        cw = {n: w for n, w in current_weights(cfg).items() if w > 0}
        left, right = st.columns([1, 1])
        with left:
            show_drift = st.toggle("Show drifted (today's) weights", help=(
                "Target weights are set at each rebalance. Drifted weights are the cost shares today, "
                "after relative price moves: if copper rallies, it is a bigger slice of the radiator."))
            if show_drift:
                ew = eff_w.iloc[-1]
                st.plotly_chart(donut(cfg, {n: float(ew.get(n, 0)) for n in cw}, "Drifted"), width="stretch")
            else:
                st.plotly_chart(donut(cfg, cw, "Target"), width="stretch")
        with right:
            per = periods[-1]
            rows = []
            for n, wt in per["weights"].items():
                unit = cfg["components"].get(n, {}).get("unit", "")
                q = per["qty"][n]
                qty = f"{q * 1000:,.2f} kg" if unit == "USD/t" else (f"{q:,.2f} h" if unit == "USD/hr" else "-")
                rows.append({"Component": n, "Target %": wt * 100, "Drifted %": float(eff_w.iloc[-1].get(n, 0)) * 100,
                             "Spend at rebalance ($)": per["spend"][n], "Implied qty / radiator": qty})
            st.caption(f"Implied bill of materials for a radiator costing ${U:,.0f} at the last rebalance "
                       f"({per['anchor']:%d %b %Y}). Use it to sanity-check weights against your real BOM.")
            st.dataframe(pd.DataFrame(rows).style.format(
                {"Target %": "{:.2f}", "Drifted %": "{:.2f}", "Spend at rebalance ($)": "{:,.2f}"}),
                hide_index=True, width="stretch")
    with t2:
        st.caption("Index points contributed by each component. The stacked areas add up to the index.")
        st.plotly_chart(stacked_area(cfg, contrib), width="stretch")
    with t3:
        st.caption("Each component's price relative to the first day shown (= 100). Hover for actual prices.")
        p = prices.loc[index.index[0]:, [c for c in prices.columns if c in cfg["components"]]]
        st.plotly_chart(relatives_chart(cfg, p), width="stretch")

    # ---- Historical table ----
    st.subheader("Historical data")
    hist = pd.DataFrame({TICKER: index, "Change (pts)": index.diff(), "Change (%)": index.pct_change() * 100,
                         "Implied cost ($)": U * index / base_level})
    units = {n: c.get("unit", "") for n, c in cfg["components"].items()}
    raw = prices.loc[index.index, [c for c in prices.columns if c in units]]
    hist = hist.join(raw.rename(columns={c: f"{c} ({units[c]})" for c in raw.columns}))
    if st.toggle("Include contribution columns"):
        hist = hist.join(contrib.add_suffix(" (pts)"))
    hist.index.name = "Date"
    lo_default = max(index.index[0], last_d - pd.Timedelta(days=90)).date()
    lo, hi = st.slider("Date range", min_value=index.index[0].date(), max_value=last_d.date(),
                       value=(lo_default, last_d.date()), format="DD MMM YYYY")
    view = hist.loc[str(lo):str(hi)].sort_index(ascending=False)
    st.dataframe(view.style.format("{:,.2f}", na_rep="-"), width="stretch", height=360)
    st.download_button(f"Download {TICKER} history (CSV)", hist.round(6).to_csv().encode(),
                       f"{TICKER}_history_{last_d:%Y%m%d}.csv", "text/csv")


# =============================================================================
# Methodology tab
# =============================================================================
def page_methodology(cfg, periods):
    st.markdown(r"""
#### Index formula
Base date $t=0$, base value $I_0 = 100$. With weights $w_i$ (summing to 1) and prices $P_{i,t}$:

$$I_t = 100 \times \sum_i w_i \frac{P_{i,t}}{P_{i,0}}$$

This is a fixed-basket **Laspeyres** index: it prices the *same* radiator every day.

#### Rebalancing without breaking history (chain-linking)
When weights change on date $T_k$, the new basket starts from the old index value:

$$I_t = I_{T_k} \times \sum_i w_i^{(k)} \frac{P_{i,t}}{P_{i,T_k}}, \qquad t \ge T_k$$

On $T_k$ itself every ratio equals 1 and $\sum_i w_i^{(k)} = 1$, so the index is identical under old and
new weights: no jump, and values before $T_k$ never change. Adding a component means giving it a weight
in a new period. Removing one means setting its weight to 0. Both are chain-linked the same way.
**Restate** mode instead re-runs the whole history with the new weights from the base date,
which is useful while you are still calibrating, but past published values will change.

#### Divisor (Dow-style) view
Equivalently $I_t = \sum_i q_i^{(k)} P_{i,t} \, / \, D_k$, where $q_i^{(k)} = w_i^{(k)} U / P_{i,T_k}$ are the
implied physical quantities in a radiator costing $U$ at the rebalance, and $D_k = U / I_{T_k}$. Each
rebalance resets the quantities and adjusts the divisor so the level is continuous.

#### Drifted weights
Between rebalances the actual cost share of component $i$ is $w_i R_{i,t} / \sum_j w_j R_{j,t}$, where
$R_{i,t} = P_{i,t}/P_{i,T_k}$.

#### Calendar & gaps
Business days. A series without a new print (holiday, monthly release) carries its last value forward.
""")
    st.markdown("#### Rebalance periods")
    rows = [dict(version=p["version"], anchor=p["anchor"].date(), level_at_anchor=round(p["level"], 4),
                 divisor=round(p["divisor"], 6), mode=p["mode"],
                 weights=", ".join(f"{n} {w * 100:.2f}%" for n, w in p["weights"].items()), note=p["note"])
            for p in periods]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if cfg.get("archived_versions"):
        with st.expander("Archived (superseded) versions"):
            st.json(cfg["archived_versions"], expanded=False)
    st.markdown("#### Components")
    st.dataframe(pd.DataFrame([dict(component=n, unit=c.get("unit"), data=cfg["data_status"].get(n),
                                    description=c.get("description", "")) for n, c in cfg["components"].items()]),
                 hide_index=True, width="stretch")


# =============================================================================
# Sidebar admin panel
# =============================================================================
def admin_unlocked() -> bool:
    pw = get_secret("ADMIN_PASSWORD")
    if not pw:
        st.sidebar.caption("Admin is open. Set `ADMIN_PASSWORD` in secrets to lock it.")
        return True
    if st.session_state.get("admin_ok"):
        return True
    entered = st.sidebar.text_input("Admin password", type="password")
    if st.sidebar.button("Unlock"):
        if hmac.compare_digest(entered, str(pw)):
            st.session_state["admin_ok"] = True
            st.rerun()
        st.sidebar.error("Wrong password.")
    return False


def commit(cfg, prices=None, msg=None):
    save_config(cfg)
    if prices is not None:
        save_prices(prices)
    if msg:
        st.session_state["flash"] = msg
    st.rerun()


def sidebar_admin(cfg: dict, prices: pd.DataFrame):
    st.sidebar.header("Admin panel")
    if not admin_unlocked():
        return
    last_date = prices.index[-1].date()
    base_date = pd.Timestamp(cfg["base_date"]).date()
    last_eff = max(pd.Timestamp(v["effective_date"]).date() for v in cfg["versions"])
    # widget keys follow the stored weights, so any change (rebalance, add, delete) resets the inputs
    vkey = abs(hash(json.dumps(cfg["versions"], sort_keys=True, default=str))) % 10 ** 10

    # ---- Rebalance settings shared by all actions ----
    with st.sidebar.expander("Rebalance settings", expanded=True):
        mode_label = st.radio("When weights or components change",
                              ["Chain-link (preserve history)", "Restate full history"],
                              help="Chain-link starts a new weight period on the effective date. "
                                   "Restate recomputes every day since the base date with the new weights.")
        mode = "chain" if mode_label.startswith("Chain") else "restate"
        eff = st.date_input("Effective date", value=last_date, min_value=max(last_eff, base_date),
                            max_value=last_date, disabled=(mode == "restate"),
                            help="Same date as the latest version replaces it instead of stacking.")
        note = st.text_input("Reason / note", placeholder="e.g. Q4 BOM re-costing")

    # ---- Edit weights ----
    with st.sidebar.expander("Edit weights", expanded=True):
        cw = current_weights(cfg)
        keys = {n: f"w_{vkey}_{n}" for n in cfg["components"]}
        for n, k in keys.items():
            st.session_state.setdefault(k, round(cw.get(n, 0.0), 4))

        def _normalize():
            vals = {n: st.session_state[k] for n, k in keys.items()}
            tot = sum(vals.values())
            if tot > 0:
                for n, k in keys.items():
                    st.session_state[k] = round(vals[n] / tot * 100, 4)

        for n, k in keys.items():
            st.number_input(f"{n} (%)", min_value=0.0, max_value=100.0, step=0.5, key=k,
                            help=cfg["components"][n].get("description"))
        total = sum(st.session_state[k] for k in keys.values())
        if abs(total - 100) > 0.01:
            st.warning(f"Weights sum to {total:.2f}%, not 100%.")
        else:
            st.success("Weights sum to 100%.")
        b1, b2 = st.columns(2)
        b1.button("Normalize", on_click=_normalize, width="stretch")
        if b2.button("Apply", type="primary", disabled=abs(total - 100) > 0.01, width="stretch"):
            new_w = {n: st.session_state[k] for n, k in keys.items()}
            if all(v == 0 for v in new_w.values()):
                st.error("At least one component needs a weight.")
            else:
                try:
                    apply_weights(cfg, new_w, mode, eff, note)
                    commit(cfg, msg="Weights applied and index recalculated.")
                except ValueError as e:
                    st.error(str(e))

    # ---- Add component ----
    with st.sidebar.expander("Add component"):
        with st.form("add_component", clear_on_submit=True):
            name = st.text_input("Name", placeholder="Aluminum side plates")
            unit = st.text_input("Unit", value="USD/t")
            desc = st.text_input("Description", placeholder="What it represents and its price proxy")
            c1, c2 = st.columns(2)
            p0 = c1.number_input("Baseline price P0", min_value=0.0, value=2600.0, format="%.4f")
            pt = c2.number_input("Current price Pt", min_value=0.0, value=2750.0, format="%.4f")
            c3, c4 = st.columns(2)
            w_new = c3.number_input("Initial weight %", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            vol = c4.number_input("Volatility %/yr", min_value=0.0, max_value=200.0, value=20.0, step=1.0,
                                  help="Only shapes the simulated path between P0 and Pt.")
            make_room = st.checkbox("Scale other weights down to make room", value=True)
            if st.form_submit_button("Add component"):
                name = name.strip()
                if not name or name in cfg["components"]:
                    st.error("Name is empty or already exists.")
                elif p0 <= 0 or pt <= 0:
                    st.error("Prices must be positive.")
                else:
                    seed = int(abs(hash(name)) % 10_000)
                    cfg["components"][name] = dict(
                        unit=unit, description=desc, live=None, color=PALETTE[len(cfg["components"]) % len(PALETTE)],
                        seed=dict(p0=p0, pt=pt, vol=vol / 100, seed=seed))
                    cfg["data_status"][name] = "manual"
                    prices[name] = bridge_path(prices.index, pd.Timestamp(cfg["base_date"]), p0, pt, vol / 100, seed)
                    if make_room and w_new > 0:
                        others = rescale({n: x for n, x in current_weights(cfg).items() if n != name},
                                         100 - w_new)
                        others[name] = w_new
                        try:
                            apply_weights(cfg, others, mode, eff, note or f"Added {name}")
                        except ValueError as e:
                            st.error(str(e))
                            return
                        commit(cfg, prices, f"Added {name} at {w_new:.2f}% and rebalanced.")
                    commit(cfg, prices, f"Added {name} with 0% weight. Set its weight in Edit weights.")

    # ---- Remove component ----
    with st.sidebar.expander("Remove component"):
        target = st.selectbox("Component", list(cfg["components"]), key="rm_target")
        how = st.radio("Action", ["Disable (weight 0, keeps history)", "Delete permanently (restates history)"],
                       key="rm_how")
        if st.button("Remove", key="rm_go"):
            cw = current_weights(cfg)
            rest = {n: x for n, x in cw.items() if n != target and x > 0}
            if not rest:
                st.error("Cannot remove the last weighted component.")
            elif how.startswith("Disable"):
                try:
                    apply_weights(cfg, rescale(rest), mode, eff, note or f"Disabled {target}")
                    commit(cfg, msg=f"{target} disabled; remaining weights scaled to 100%.")
                except ValueError as e:
                    st.error(str(e))
            else:
                for v in cfg["versions"]:
                    v["weights"] = rescale({n: x for n, x in v["weights"].items() if n != target})
                cfg["versions"] = [v for v in cfg["versions"] if v["weights"]]
                cfg["components"].pop(target, None)
                cfg["data_status"].pop(target, None)
                commit(cfg, prices.drop(columns=[target], errors="ignore"),
                       f"{target} deleted; history restated without it.")

    # ---- Prices ----
    with st.sidebar.expander("Update prices"):
        comp = st.selectbox("Component", list(cfg["components"]), key="px_comp")
        c1, c2 = st.columns(2)
        d = c1.date_input("Date", value=dt.date.today(), min_value=prices.index[0].date(), key="px_date")
        last_val = float(prices[comp].dropna().iloc[-1]) if comp in prices else 0.0
        val = c2.number_input("Price", min_value=0.0, value=last_val, format="%.4f", key="px_val")
        if st.button("Save price", key="px_save") and val > 0:
            prices = set_prices(prices, comp, pd.Series([val], index=[pd.Timestamp(d)]))
            cfg["data_status"][comp] = "manual" if cfg["data_status"].get(comp) != "live" else "live"
            commit(cfg, prices, f"{comp} = {val:,.4f} on {d}.")
        up = st.file_uploader("Or upload CSV (date,value)", type=["csv"], key="px_csv")
        if up is not None and st.button("Import CSV", key="px_imp"):
            df = pd.read_csv(up)
            df.columns = [c.strip().lower() for c in df.columns]
            s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=pd.to_datetime(df["date"])).dropna()
            prices = set_prices(prices, comp, s)
            cfg["data_status"][comp] = "manual"
            commit(cfg, prices, f"Imported {len(s)} prices into {comp}.")

    # ---- Data & settings ----
    with st.sidebar.expander("Data & settings"):
        U = st.number_input("Reference cost per radiator at base ($)", min_value=1.0,
                            value=float(cfg.get("reference_unit_cost", 1000.0)), step=50.0,
                            help="Turns index points into dollars: implied cost = U x I / 100.")
        if U != cfg.get("reference_unit_cost"):
            cfg["reference_unit_cost"] = U
            commit(cfg)
        auto = st.checkbox("Auto-refresh live market data", value=bool(cfg.get("auto_live")),
                           help=f"Pulls FRED / Yahoo data at most every {LIVE_REFRESH_HOURS} h on page load.")
        if auto != bool(cfg.get("auto_live")):
            cfg["auto_live"] = auto
            commit(cfg)
        if st.button("Refresh live data now"):
            with st.spinner("Fetching FRED / Yahoo..."):
                new_prices, log = refresh_live(cfg, prices)
            cfg["last_live_refresh"] = now_iso()
            st.session_state["live_log"] = log
            commit(cfg, new_prices, "Live refresh finished.")
        if st.session_state.get("live_log"):
            st.dataframe(pd.DataFrame(st.session_state["live_log"]), hide_index=True)
        st.caption(f"Base date {cfg['base_date']} = {cfg.get('base_level', BASE_LEVEL):g}. "
                   f"Last live refresh: {cfg.get('last_live_refresh') or 'never'}.")
        st.download_button("Download index_config.json", json.dumps(cfg, indent=2, default=str).encode(),
                           "index_config.json", "application/json")
        st.download_button("Download price_history.csv", prices.rename_axis("date").to_csv().encode(),
                           "price_history.csv", "text/csv")
        upc = st.file_uploader("Restore index_config.json", type=["json"], key="cfg_up")
        if upc is not None and st.button("Restore config"):
            new_cfg = json.load(upc)
            if not {"components", "versions", "base_date"} <= set(new_cfg):
                st.error("Not a valid index config.")
            else:
                new_cfg.setdefault("data_status", {n: "simulated" for n in new_cfg["components"]})
                new_cfg.setdefault("archived_versions", [])
                prices = ensure_columns(new_cfg, prices)
                commit(new_cfg, prices, "Config restored.")
        if st.button("Reset to demo defaults"):
            st.session_state["confirm_reset"] = True
        if st.session_state.get("confirm_reset"):
            st.warning("This replaces all weights, components and prices with fresh simulated data.")
            if st.button("Yes, reset everything", type="primary"):
                st.session_state.pop("confirm_reset")
                dates = history_calendar()
                commit(new_default_config(dates[0]), generate_default_mock(dates), "Reset to demo defaults.")


def set_prices(prices: pd.DataFrame, comp: str, s: pd.Series) -> pd.DataFrame:
    """Write prices for one component; extends the calendar forward (other series carried forward)."""
    s = s[s.index >= prices.index[0]]
    s = s[s.index.dayofweek < 5]
    if s.empty:
        return prices
    idx = prices.index.union(pd.bdate_range(prices.index[-1], max(s.index.max(), prices.index[-1])))
    out = prices.reindex(idx)
    out.loc[s.index, comp] = s.values
    return out.ffill()


# =============================================================================
# Main
# =============================================================================
def maybe_auto_refresh(cfg, prices):
    if not cfg.get("auto_live") or os.environ.get("CBRI_OFFLINE") == "1":
        return prices
    last = cfg.get("last_live_refresh")
    if last and dt.datetime.now() - dt.datetime.fromisoformat(last) < dt.timedelta(hours=LIVE_REFRESH_HOURS):
        return prices
    with st.spinner("Refreshing live market data..."):
        prices, log = refresh_live(cfg, prices)
    cfg["last_live_refresh"] = now_iso()  # recorded even on failure so a dead feed doesn't retry every run
    save_config(cfg)
    save_prices(prices)
    return prices


def main():
    st.set_page_config(page_title=INDEX_NAME, page_icon="📈", layout="wide")
    cfg, prices = load_state()
    prices = maybe_auto_refresh(cfg, prices)

    st.title(cfg.get("index_name", INDEX_NAME))
    st.caption(f"{TICKER} · daily cost index of copper, brass, solder, energy and labor for a heavy-duty "
               f"copper-brass radiator · base {cfg['base_date']} = {cfg.get('base_level', BASE_LEVEL):g}")
    if msg := st.session_state.pop("flash", None):
        st.toast(msg)

    weighted = [n for n, w in current_weights(cfg).items() if w > 0]
    simulated = [n for n in weighted if cfg["data_status"].get(n) == "simulated"]
    if simulated:
        st.info(f"Demo data: {', '.join(simulated)} {'is' if len(simulated) == 1 else 'are'} simulated. "
                "Use **Refresh live data** in the sidebar, or upload your own prices.", icon="🧪")

    sidebar_admin(cfg, prices)

    tab_dash, tab_method = st.tabs(["Dashboard", "Methodology"])
    try:
        index, contrib, eff_w, periods = compute_index(prices, cfg)
        index = index.dropna()
    except Exception as e:
        with tab_dash:
            st.error(f"The index could not be computed: {e}")
        return
    with tab_dash:
        page_dashboard(cfg, prices, index, contrib.loc[index.index], eff_w.loc[index.index], periods)
    with tab_method:
        page_methodology(cfg, periods)


if __name__ == "__main__":
    main()
