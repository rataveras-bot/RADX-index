"""
Copper-Brass Radiator Cost Index (CBRI)
=======================================
A daily, chain-linked Laspeyres cost index for the raw materials, energy and labor that go
into a heavy-duty copper-brass radiator. Single file: Streamlit + Plotly + Pandas.
English / Spanish interface (sidebar switch, or ?lang=es in the URL).

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
import time
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
CONFIG_PATH = APP_DIR / "index_config.json"   # local app state (weights, manual prices); not in git
PRICES_PATH = APP_DIR / "price_history.csv"
DATA_DIR = APP_DIR / "data"                     # market data, written by the nightly GitHub job
MARKET_PATH = DATA_DIR / "market_data.csv"
MARKET_META_PATH = DATA_DIR / "market_meta.json"
MARKET_BACKFILL_DAYS = 420
METALCHARTS_URL = "https://metalcharts.org"

INDEX_NAME = "Copper-Brass Radiator Cost Index"
TICKER = "CBRI"
BASE_LEVEL = 100.0
HISTORY_DAYS = 365
LB_PER_T = 2204.62262
PALETTE = ["#C8743B", "#C9A227", "#7C8A99", "#2F6F8F", "#4E8F5A", "#8E5BA8", "#B4505A", "#5B7DB4",
           "#9C6B3A", "#3E9C9C"]

# Raw markets, fetched by the nightly job. Sources are in priority order: the first one with a value
# for a date wins, later ones fill the gaps (e.g. MetalCharts daily spot, backfilled by IMF monthly).
# "factor" converts each source into the market's unit. metalcharts tickers/factors are confirmed
# against their API docs before use.
MARKETS = {
    "COPPER": dict(name="Copper, COMEX front month", unit="USD/lb",
                   sources=[dict(source="yfinance", ticker="HG=F")]),
    "ZINC": dict(name="Zinc", unit="USD/t",
                 sources=[dict(source="metalcharts", ticker="zinc"), dict(source="fred", ticker="PZINCUSDM")]),
    "TIN": dict(name="Tin", unit="USD/t",
                sources=[dict(source="metalcharts", ticker="tin"), dict(source="fred", ticker="PTINUSDM")]),
    "LEAD": dict(name="Lead", unit="USD/t",
                 sources=[dict(source="metalcharts", ticker="lead"), dict(source="fred", ticker="PLEADUSDM")]),
    "NATGAS": dict(name="Natural gas, Henry Hub", unit="USD/MMBtu",
                   sources=[dict(source="yfinance", ticker="NG=F"), dict(source="fred", ticker="DHHNGSP")]),
    "DIESEL": dict(name="ULSD diesel, NY Harbor", unit="USD/gal",
                   sources=[dict(source="yfinance", ticker="HO=F"), dict(source="fred", ticker="DDFUELNYH")]),
    "BRENT": dict(name="Brent crude", unit="USD/bbl",
                  sources=[dict(source="yfinance", ticker="BZ=F"), dict(source="fred", ticker="DCOILBRENTEU")]),
    "MFG_WAGES": dict(name="US manufacturing avg hourly earnings", unit="USD/hr",
                      sources=[dict(source="bls", ticker="CES3000000008"),
                               dict(source="fred", ticker="CES3000000008")]),
}

# Default basket. "live" says how to build each component's price from the markets above:
#   alloy  -> sum(share * market * factor)  (mass fractions, all in USD/t), also used for single markets
#   basket -> 100 * sum(share * market / market_at_base_date)  (cost shares of unlike units)
DEFAULT_COMPONENTS = {
    "Copper": dict(
        unit="USD/t", description="Fin strip and tube stock (C11000/C12200). Proxy: COMEX HG copper.",
        live=dict(method="alloy", parts=[dict(market="COPPER", factor=LB_PER_T, share=1.0)])),
    "Brass": dict(
        unit="USD/t", description="Header plates, tanks, side plates (CuZn35-37), priced as 65% Cu + 35% Zn.",
        live=dict(method="alloy", parts=[dict(market="COPPER", factor=LB_PER_T, share=0.65),
                                         dict(market="ZINC", factor=1.0, share=0.35)])),
    "Solder": dict(
        unit="USD/t", description="Pb-Sn solder, 70/30 by mass, for tube-to-header and tank joints.",
        live=dict(method="alloy", parts=[dict(market="LEAD", factor=1.0, share=0.70),
                                         dict(market="TIN", factor=1.0, share=0.30)])),
    "Energy": dict(
        unit="basket pts", description="45% natural gas (ovens), 35% diesel (freight), 20% Brent (oil-linked inputs).",
        live=dict(method="basket", parts=[dict(market="NATGAS", share=0.45),
                                          dict(market="DIESEL", share=0.35),
                                          dict(market="BRENT", share=0.20)])),
    "Labor": dict(
        unit="USD/hr", description="Direct wages. Proxy: BLS manufacturing avg hourly earnings (CES3000000008).",
        live=dict(method="alloy", parts=[dict(market="MFG_WAGES", factor=1.0, share=1.0)])),
}
DEFAULT_WEIGHTS = {"Copper": 42.0, "Brass": 20.0, "Solder": 8.0, "Energy": 10.0, "Labor": 20.0}


# =============================================================================
# Language (English / Spanish)
# =============================================================================
# English text is the key. Anything not listed (e.g. user-added component names) shows as typed.
ES = {
    # --- app shell ---
    INDEX_NAME: "Índice de Costo de Radiador Cobre-Latón",
    "Dashboard": "Panel",
    "Methodology": "Metodología",
    "daily cost index of copper, brass, solder, energy and labor for a heavy-duty copper-brass radiator":
        "índice diario de costo de cobre, latón, soldadura, energía y mano de obra para un radiador "
        "de cobre-latón de servicio pesado",
    "base {date} = {level}": "base {date} = {level}",
    "Demo data: {names} {verb} simulated until the nightly market-data job has run, or upload your own prices.":
        "Datos de demostración: {names} {verb} simulados hasta que corra la actualización nocturna de datos "
        "de mercado, o cargue sus propios precios.",
    "Metal prices by [MetalCharts]({u}).": "Precios de metales por [MetalCharts]({u}).",
    "#### Market data sources": "#### Fuentes de datos de mercado",
    "market": "mercado", "last date": "última fecha", "last value": "último valor", "source": "fuente",
    "Each market uses its first source with a value for that day; later sources fill gaps. "
    "Zinc, tin and lead fall back to IMF monthly averages before daily data exists.":
        "Cada mercado usa su primera fuente con valor para ese día; las siguientes llenan los huecos. "
        "Zinc, estaño y plomo usan promedios mensuales del FMI antes de que existan datos diarios.",
    "is": "está", "are": "están",
    "The index could not be computed: {e}": "No se pudo calcular el índice: {e}",
    # --- components, units, data status ---
    "Copper": "Cobre", "Brass": "Latón", "Solder": "Soldadura", "Energy": "Energía", "Labor": "Mano de obra",
    "basket pts": "pts canasta",
    "simulated": "simulado", "live": "en vivo", "manual": "manual",
    "Fin strip and tube stock (C11000/C12200). Proxy: COMEX HG copper.":
        "Fleje para aletas y tubo (C11000/C12200). Referencia: cobre COMEX HG.",
    "Header plates, tanks, side plates (CuZn35-37), priced as 65% Cu + 35% Zn.":
        "Placas de cabezal, tanques, placas laterales (CuZn35-37), valorado como 65% Cu + 35% Zn.",
    "Pb-Sn solder, 70/30 by mass, for tube-to-header and tank joints.":
        "Soldadura Pb-Sn, 70/30 en masa, para uniones tubo-cabezal y de tanques.",
    "45% natural gas (ovens), 35% diesel (freight), 20% Brent (oil-linked inputs).":
        "45% gas natural (hornos), 35% diésel (fletes), 20% Brent (insumos ligados al petróleo).",
    "Direct wages. Proxy: BLS manufacturing avg hourly earnings (CES3000000008).":
        "Salarios directos. Referencia: salario promedio por hora en manufactura, BLS (CES3000000008).",
    # --- dashboard ---
    "1-day change (per radiator)": "Cambio de 1 día (por radiador)",
    "Index change applied to the reference cost of ${u} per radiator at base.":
        "Cambio del índice aplicado al costo de referencia de ${u} por radiador en la base.",
    "30-day trend": "Tendencia 30 días",
    "Implied material+conversion cost": "Costo implícito (materiales + conversión)",
    "{p}% vs base": "{p}% vs. base",
    "Weight breakdown": "Distribución de pesos",
    "Contributions over time": "Contribuciones en el tiempo",
    "Component prices": "Precios de componentes",
    "Show drifted (today's) weights": "Mostrar pesos derivados (de hoy)",
    "Target weights are set at each rebalance. Drifted weights are the cost shares today, after relative "
    "price moves: if copper rallies, it is a bigger slice of the radiator.":
        "Los pesos objetivo se fijan en cada rebalanceo. Los pesos derivados son la participación en el "
        "costo hoy, tras los movimientos relativos de precios: si el cobre sube, ocupa una porción mayor "
        "del radiador.",
    "Drifted": "Derivado", "Target": "Objetivo",
    "Component": "Componente", "Target %": "% objetivo", "Drifted %": "% derivado",
    "Spend at rebalance ($)": "Gasto al rebalanceo ($)", "Implied qty / radiator": "Cantidad implícita / radiador",
    "Implied bill of materials for a radiator costing ${u} at the last rebalance ({d}). "
    "Use it to sanity-check weights against your real BOM.":
        "Lista de materiales implícita para un radiador de ${u} en el último rebalanceo ({d}). "
        "Úsela para validar los pesos contra su lista de materiales real.",
    "Index points contributed by each component. The stacked areas add up to the index.":
        "Puntos del índice aportados por cada componente. Las áreas apiladas suman el índice.",
    "Each component's price relative to the first day shown (= 100). Hover for actual prices.":
        "Precio de cada componente relativo al primer día mostrado (= 100). Pase el cursor para ver los "
        "precios reales.",
    "Historical data": "Datos históricos",
    "Change (pts)": "Cambio (pts)", "Change (%)": "Cambio (%)", "Implied cost ($)": "Costo implícito ($)",
    "Include contribution columns": "Incluir columnas de contribución",
    "Date": "Fecha", "Date range": "Rango de fechas",
    "Download {t} history (CSV)": "Descargar historial de {t} (CSV)",
    # --- charts ---
    "Contribution": "Contribución", "Index level": "Nivel del índice", "Index points": "Puntos del índice",
    "Price relative (base = 100)": "Precio relativo (base = 100)", "price": "precio",
    "1Y": "1A", "ALL": "TODO",
    # --- methodology ---
    "#### Rebalance periods": "#### Periodos de rebalanceo",
    "Archived (superseded) versions": "Versiones archivadas (reemplazadas)",
    "#### Components": "#### Componentes",
    "version": "versión", "anchor": "fecha ancla", "level_at_anchor": "nivel en ancla", "divisor": "divisor",
    "mode": "modo", "weights": "pesos", "note": "nota", "unit": "unidad", "data": "datos",
    "description": "descripción", "component": "componente",
    # --- admin: access ---
    "Admin panel": "Panel de administración",
    "Admin is open. Set `ADMIN_PASSWORD` in secrets to lock it.":
        "La administración está abierta. Defina `ADMIN_PASSWORD` en los secretos para bloquearla.",
    "Admin password": "Contraseña de administrador", "Unlock": "Desbloquear", "Wrong password.": "Contraseña incorrecta.",
    # --- admin: rebalance ---
    "Rebalance settings": "Configuración de rebalanceo",
    "When weights or components change": "Cuando cambien pesos o componentes",
    "Chain-link (preserve history)": "Encadenar (conservar historial)",
    "Restate full history": "Recalcular todo el historial",
    "Chain-link starts a new weight period on the effective date. "
    "Restate recomputes every day since the base date with the new weights.":
        "Encadenar inicia un nuevo periodo de pesos en la fecha efectiva. "
        "Recalcular vuelve a calcular cada día desde la fecha base con los nuevos pesos.",
    "Effective date": "Fecha efectiva",
    "Same date as the latest version replaces it instead of stacking.":
        "Si coincide con la fecha de la última versión, la reemplaza en vez de agregar otra.",
    "Reason / note": "Motivo / nota", "e.g. Q4 BOM re-costing": "ej. recosteo de BOM del T4",
    # --- admin: weights ---
    "Edit weights": "Editar pesos",
    "Weights sum to {x}%, not 100%.": "Los pesos suman {x}%, no 100%.",
    "Weights sum to 100%.": "Los pesos suman 100%.",
    "Normalize": "Normalizar", "Apply": "Aplicar",
    "At least one component needs a weight.": "Al menos un componente necesita un peso.",
    "Weights applied and index recalculated.": "Pesos aplicados e índice recalculado.",
    "Rebalance": "Rebalanceo", "Restated history": "Historial recalculado",
    # --- admin: add component ---
    "Add component": "Agregar componente",
    "Name": "Nombre", "Aluminum side plates": "Placas laterales de aluminio",
    "Unit": "Unidad", "Description": "Descripción",
    "What it represents and its price proxy": "Qué representa y su referencia de precio",
    "Baseline price P0": "Precio base P0", "Current price Pt": "Precio actual Pt",
    "Initial weight %": "Peso inicial %", "Volatility %/yr": "Volatilidad %/año",
    "Only shapes the simulated path between P0 and Pt.": "Solo define la trayectoria simulada entre P0 y Pt.",
    "Scale other weights down to make room": "Reducir los demás pesos para hacer espacio",
    "Name is empty or already exists.": "El nombre está vacío o ya existe.",
    "Prices must be positive.": "Los precios deben ser positivos.",
    "Added {name}": "Se agregó {name}",
    "Added {name} at {w}% and rebalanced.": "Se agregó {name} con {w}% y se rebalanceó.",
    "Added {name} with 0% weight. Set its weight in Edit weights.":
        "Se agregó {name} con peso 0%. Asigne su peso en Editar pesos.",
    # --- admin: remove component ---
    "Remove component": "Quitar componente", "Action": "Acción",
    "Disable (weight 0, keeps history)": "Desactivar (peso 0, conserva historial)",
    "Delete permanently (restates history)": "Eliminar permanentemente (recalcula historial)",
    "Remove": "Quitar",
    "Cannot remove the last weighted component.": "No se puede quitar el último componente con peso.",
    "Disabled {name}": "Se desactivó {name}",
    "{name} disabled; remaining weights scaled to 100%.": "{name} desactivado; los pesos restantes se ajustaron a 100%.",
    "{name} deleted; history restated without it.": "{name} eliminado; historial recalculado sin él.",
    # --- admin: prices ---
    "Update prices": "Actualizar precios", "Price": "Precio", "Save price": "Guardar precio",
    "{c} = {v} on {d}.": "{c} = {v} el {d}.",
    "Or upload CSV (date,value)": "O cargue un CSV (date,value)", "Import CSV": "Importar CSV",
    "Imported {n} prices into {c}.": "Se importaron {n} precios en {c}.",
    # --- admin: data & settings ---
    "Data & settings": "Datos y configuración",
    "Reference cost per radiator at base ($)": "Costo de referencia por radiador en la base ($)",
    "Turns index points into dollars: implied cost = U x I / 100.":
        "Convierte puntos del índice en dólares: costo implícito = U x I / 100.",
    "Base date {b} = {l}. Market data is updated every weekday evening by the GitHub job. Last update: {r}.":
        "Fecha base {b} = {l}. Los datos de mercado se actualizan cada día hábil en la noche con la tarea de "
        "GitHub. Última actualización: {r}.",
    "{n} source(s) failed in the last update; their backup source was used.":
        "{n} fuente(s) fallaron en la última actualización; se usó su fuente de respaldo.",
    "never": "nunca",
    "Download index_config.json": "Descargar index_config.json",
    "Download price_history.csv": "Descargar price_history.csv",
    "Restore index_config.json": "Restaurar index_config.json", "Restore config": "Restaurar configuración",
    "Not a valid index config.": "No es una configuración de índice válida.",
    "Config restored.": "Configuración restaurada.",
    "Reset to demo defaults": "Restablecer valores de demostración",
    "This replaces all weights, components and prices with fresh simulated data.":
        "Esto reemplaza todos los pesos, componentes y precios con datos simulados nuevos.",
    "Yes, reset everything": "Sí, restablecer todo",
    "Reset to demo defaults.": "Valores de demostración restablecidos.",
}

MONTHS_ES = {"Jan": "ene", "Feb": "feb", "Mar": "mar", "Apr": "abr", "May": "may", "Jun": "jun",
             "Jul": "jul", "Aug": "ago", "Sep": "sep", "Oct": "oct", "Nov": "nov", "Dec": "dic",
             "Mon": "lun", "Tue": "mar", "Wed": "mié", "Thu": "jue", "Fri": "vie", "Sat": "sáb", "Sun": "dom"}


def lang() -> str:
    return st.session_state.get("lang", "en")


def t(text: str, **kw) -> str:
    """Translate a UI string (English is the key) and fill {placeholders}."""
    s = ES.get(text, text) if lang() == "es" else text
    return s.format(**kw) if kw else s


def fdate(d, fmt: str = "%d %b %Y") -> str:
    s = pd.Timestamp(d).strftime(fmt)
    if lang() == "es":
        for en, es in MONTHS_ES.items():
            s = s.replace(en, es)
    return s


def init_language():
    """Pick the language before any output: session first, then ?lang= in the URL."""
    if "lang" not in st.session_state:
        st.session_state["lang"] = "es" if st.query_params.get("lang") == "es" else "en"


def language_switch():
    def _changed():
        st.session_state["lang"] = "es" if st.session_state["lang_choice"] == "Español" else "en"
        st.query_params["lang"] = st.session_state["lang"]

    st.sidebar.radio("Language / Idioma", ["English", "Español"], horizontal=True, key="lang_choice",
                     index=1 if lang() == "es" else 0, on_change=_changed)


def now_iso() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def get_secret(key: str, default=None):
    """Secret from Streamlit secrets or the environment, with stray whitespace removed."""
    val = None
    try:
        if key in st.secrets:
            val = st.secrets[key]
    except Exception:
        pass
    if val is None:
        val = os.environ.get(key, default)
    return val.strip() if isinstance(val, str) else val


SECRET_NAMES = ("FRED_API_KEY", "METALCHARTS_API_KEY", "ADMIN_PASSWORD")


def redact(text: str) -> str:
    """Remove secret values and URL query strings from text that may be saved or shown publicly."""
    import re
    text = re.sub(r"(https?://[^\s?]+)\?\S*", r"\1?…", str(text))
    text = re.sub(r"(?i)(api_key|apikey|key|token)=[^&\s]+", r"\1=***", text)
    for name in SECRET_NAMES:
        val = get_secret(name)
        if isinstance(val, str) and len(val) >= 6:
            text = text.replace(val, "***")
    return text


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
    tt = np.linspace(0, 1, n)
    bridge = w - tt * w[-1]
    s.loc[seg] = np.exp(np.log(p0) + tt * (np.log(pt) - np.log(p0)) + vol * bridge)
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
        reference_unit_cost=1000.0,
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
    migrated = False
    for name, comp in cfg["components"].items():  # migrate recipes saved by older versions of the app
        live = comp.get("live") or {}
        if name in DEFAULT_COMPONENTS and any("market" not in p for p in live.get("parts", [])):
            comp["live"] = DEFAULT_COMPONENTS[name]["live"]
            migrated = True
    for old_key in ("auto_live", "last_live_refresh"):
        migrated |= cfg.pop(old_key, None) is not None
    before = set(prices.columns)
    prices = ensure_columns(cfg, prices)
    if migrated or set(prices.columns) != before or not PRICES_PATH.exists():
        save_prices(prices)
        save_config(cfg)
    return cfg, prices


# =============================================================================
# Market data: fetched by the nightly job (update_prices.py), read by the app
# =============================================================================
def _get(url: str, tries: int = 2, timeout: int = 30, **kw) -> requests.Response:
    """GET with retries and back-off; FRED in particular is slow to answer cloud servers."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (CBRI-index/1.0)"}, **kw)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(5 * (i + 1))
    raise last


def fetch_fred(ticker: str, start: str) -> pd.Series:
    """FRED official API when FRED_API_KEY is set (reliable from cloud servers), else the public CSV."""
    key = get_secret("FRED_API_KEY")
    if key:
        r = _get("https://api.stlouisfed.org/fred/series/observations",
                 params=dict(series_id=ticker, api_key=key, file_type="json", observation_start=start))
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs, columns=["date", "value"])
    else:
        r = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}&cosd={start}")
        df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
        df.columns = ["date", "value"]  # header is DATE or observation_date depending on FRED version
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED marks missing values as "."
    df = df.dropna()
    return pd.Series(df["value"].values, index=pd.to_datetime(df["date"])).sort_index()


def fetch_bls(ticker: str, start: str) -> pd.Series:
    """BLS public API v1 (no key needed), monthly series dated on the 1st of the month."""
    r = _get(f"https://api.bls.gov/publicAPI/v1/timeseries/data/{ticker}")
    js = r.json()
    if js.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS: {js.get('status')} {js.get('message')}")
    rows = [(pd.Timestamp(int(d["year"]), int(d["period"][1:]), 1), float(d["value"]))
            for d in js["Results"]["series"][0]["data"]
            if d["period"].startswith("M") and d["period"] != "M13" and d["value"] not in ("-", "")]
    s = pd.Series(dict(rows)).sort_index()
    return s[s.index >= pd.Timestamp(start) - pd.offsets.MonthBegin(1)]


def fetch_yfinance(ticker: str, start: str) -> pd.Series:
    import yfinance as yf
    raw = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    idx = close.index.tz_localize(None) if close.index.tz is not None else close.index
    return pd.Series(close.values, index=pd.DatetimeIndex(idx).normalize()).dropna().sort_index()


def fetch_metalcharts(ticker: str, start: str) -> pd.Series:
    """MetalCharts spot price (free tier: today's value only, no history).
    Needs METALCHARTS_API_KEY. The endpoint is wired in once the request format from their docs is confirmed."""
    if not get_secret("METALCHARTS_API_KEY"):
        raise RuntimeError("METALCHARTS_API_KEY not set")
    raise NotImplementedError("MetalCharts request format not configured yet")


FETCHERS = {"fred": fetch_fred, "bls": fetch_bls, "yfinance": fetch_yfinance, "metalcharts": fetch_metalcharts}


def source_col(market: str, src: dict) -> str:
    return f"{market}|{src['source']}:{src['ticker']}"


def read_market_data(path: Path = MARKET_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col="date", parse_dates=["date"]).sort_index()


def read_market_meta(path: Path = MARKET_META_PATH) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def update_market_data(path: Path = MARKET_PATH, meta_path: Path = MARKET_META_PATH, today=None) -> list[dict]:
    """Fetch every market source and upsert it into data/market_data.csv (one column per source).
    History-capable sources re-pull the last 30 days to catch revisions; spot-only sources add today."""
    today = pd.Timestamp(today or dt.date.today()).normalize()
    raw = read_market_data(path)
    log = []
    down = {}  # source -> first error; once a source is unreachable, skip it for the rest of this run
    for market, spec in MARKETS.items():
        for src in spec["sources"]:
            col = source_col(market, src)
            if src["source"] in down:
                log.append(dict(market=market, source=f"{src['source']}:{src['ticker']}",
                                status=f"error: skipped, {src['source']} unreachable this run"))
                continue
            have = raw[col].dropna() if col in raw.columns else pd.Series(dtype=float)
            start = (have.index.max() - pd.Timedelta(days=30)) if len(have) else today - pd.Timedelta(days=MARKET_BACKFILL_DAYS)
            try:
                s = FETCHERS[src["source"]](src["ticker"], start.strftime("%Y-%m-%d"))
                s = s[s.index <= today].dropna()
                if s.empty:
                    raise RuntimeError("no observations returned")
                raw = raw.reindex(raw.index.union(s.index))
                if col not in raw.columns:
                    raw[col] = np.nan
                raw.loc[s.index, col] = s.values
                log.append(dict(market=market, source=f"{src['source']}:{src['ticker']}", status="ok",
                                last=str(s.index.max().date()), value=round(float(s.iloc[-1]), 6)))
            except Exception as e:
                if isinstance(e, requests.exceptions.RequestException):
                    down[src["source"]] = e
                log.append(dict(market=market, source=f"{src['source']}:{src['ticker']}",
                                status=f"error: {redact(e)[:160]}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.sort_index().rename_axis("date").to_csv(path, float_format="%.6f")
    meta_path.write_text(json.dumps(dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                                         log=log), indent=2))
    return log


def market_series(raw: pd.DataFrame, market: str) -> tuple[pd.Series, pd.Series]:
    """Merged daily-or-better series for one market, plus which source supplied each date."""
    vals, used = pd.Series(dtype=float), pd.Series(dtype=object)
    for src in MARKETS[market]["sources"]:
        col = source_col(market, src)
        if col not in raw.columns:
            continue
        s = raw[col].dropna() * float(src.get("factor", 1.0))
        new = s.index.difference(vals.index)
        vals = pd.concat([vals, s.loc[new]]).sort_index()
        used = pd.concat([used, pd.Series(src["source"], index=new)]).sort_index()
    return vals, used


def build_live_prices(cfg: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Component prices on a business-day calendar, built from the stored market data."""
    merged = {m: market_series(raw, m)[0] for m in MARKETS}
    merged = {m: s for m, s in merged.items() if len(s)}
    if not merged:
        return pd.DataFrame()
    cal = pd.bdate_range(min(s.index.min() for s in merged.values()), max(s.index.max() for s in merged.values()))
    daily = {m: s.reindex(s.index.union(cal)).ffill().reindex(cal) for m, s in merged.items()}
    base = pd.Timestamp(cfg["base_date"])
    out = pd.DataFrame(index=cal)
    for name, comp in cfg["components"].items():
        spec = comp.get("live")
        if not spec or any(p.get("market") not in daily for p in spec["parts"]):
            continue
        if spec["method"] == "basket":
            val = 0.0
            for p in spec["parts"]:
                m = daily[p["market"]]
                ref = m.loc[:base].dropna()
                ref = ref.iloc[-1] if len(ref) else m.dropna().iloc[0]
                val = val + 100 * p["share"] * m / ref
        else:
            val = sum(p["share"] * p.get("factor", 1.0) * daily[p["market"]] for p in spec["parts"])
        out[name] = val
    return out


def overlay_live(cfg: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Replace simulated/stored prices with market data for every component that has a live recipe.
    Components set to 'manual' (your own uploaded prices) are left alone."""
    raw = read_market_data()
    if raw.empty:
        return prices
    live = build_live_prices(cfg, raw).dropna(how="all")
    if live.empty:
        return prices
    end = max(prices.index.max(), live.index.max())
    cal = prices.index.union(pd.bdate_range(prices.index.min(), end))
    out = prices.reindex(cal).ffill()
    for name in live.columns:
        if cfg["data_status"].get(name) == "manual":
            continue
        s = live[name].dropna()
        merged = s.reindex(cal).combine_first(out[name]) if name in out else s.reindex(cal)
        out[name] = merged.ffill()
        cfg["data_status"][name] = "live"
    return out


def market_status(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, spec in MARKETS.items():
        vals, used = market_series(raw, m)
        rows.append({"market": spec["name"], "unit": spec["unit"],
                     "last date": vals.index.max().date() if len(vals) else None,
                     "last value": round(float(vals.iloc[-1]), 4) if len(vals) else None,
                     "source": used.iloc[-1] if len(used) else None})
    return pd.DataFrame(rows)


def uses_metalcharts(raw: pd.DataFrame) -> bool:
    return any(c.split("|")[1].startswith("metalcharts:") and raw[c].notna().any() for c in raw.columns if "|" in c)


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
                                note=note or t("Restated history"), mode="restate", created=now_iso())]
    else:
        eff = str(eff_date)
        last = max(cfg["versions"], key=lambda v: v["effective_date"])
        if eff < last["effective_date"]:
            raise ValueError(f"effective date must be on or after {last['effective_date']}")
        if eff == last["effective_date"]:  # same day: replace that period instead of stacking
            cfg["versions"].remove(last)
            cfg.setdefault("archived_versions", []).append(last)
        cfg["versions"].append(dict(id=next_id, effective_date=eff, weights=w,
                                    note=note or t("Rebalance"), mode="chain", created=now_iso()))


def rescale(weights: dict, keep_total: float = 100.0) -> dict:
    tot = sum(weights.values())
    return {n: x / tot * keep_total for n, x in weights.items()} if tot > 0 else weights


# =============================================================================
# Charts
# =============================================================================
def date_ticks():
    """Plotly writes month names in English; Spanish uses numeric dates instead."""
    return "%d/%m/%y" if lang() == "es" else None


def color_of(cfg, name, i=0):
    return cfg["components"].get(name, {}).get("color", PALETTE[i % len(PALETTE)])


def index_chart(cfg, index: pd.Series, contrib: pd.DataFrame, periods):
    cols = [c for c in contrib.columns if contrib[c].abs().sum() > 0]
    lines = "<br>".join(f"{t(c)}: %{{customdata[{j}]:,.2f}} pts" for j, c in enumerate(cols))
    labels = [fdate(d, "%a %d %b %Y") for d in index.index]
    fig = go.Figure(go.Scatter(
        x=index.index, y=index.values, mode="lines", name=TICKER,
        line=dict(color="#C8743B", width=2.4),
        customdata=[row + [lab] for row, lab in zip(contrib[cols].values.tolist(), labels)],
        hovertemplate=f"<b>%{{customdata[{len(cols)}]}}</b><br>" + TICKER + " <b>%{y:,.2f}</b><br>"
                      f"<span style='font-size:11px'>{t('Contribution')}</span><br>" + lines + "<extra></extra>"))
    for per in periods[1:]:
        fig.add_vline(x=per["anchor"], line_dash="dot", line_color="#8a8a8a", opacity=0.7)
        fig.add_annotation(x=per["anchor"], y=1, yref="paper", text=f"v{per['version']}", showarrow=False,
                           yanchor="bottom", font=dict(size=10, color="#8a8a8a"))
    fig.add_hline(y=cfg.get("base_level", BASE_LEVEL), line_color="#999", line_width=1, opacity=0.35)
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=36, b=10), hovermode="x", showlegend=False,
        yaxis_title=t("Index level"), separators=".,",
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.07), type="date", tickformat=date_ticks(),
                   rangeselector=dict(buttons=[
                       dict(count=1, label="1D", step="day", stepmode="backward"),
                       dict(count=1, label="1M", step="month", stepmode="backward"),
                       dict(count=6, label="6M", step="month", stepmode="backward"),
                       dict(count=1, label=t("1Y"), step="year", stepmode="backward"),
                       dict(step="all", label=t("ALL"))])))
    return fig


def donut(cfg, weights: dict, title: str):
    names = [n for n, w in weights.items() if w > 0]
    fig = go.Figure(go.Pie(
        labels=[t(n) for n in names], values=[weights[n] for n in names], hole=0.58, sort=False,
        marker=dict(colors=[color_of(cfg, n, i) for i, n in enumerate(names)]),
        texttemplate="%{label}<br>%{percent:.1%}", hovertemplate="%{label}: %{percent:.2%}<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      annotations=[dict(text=title, showarrow=False, font=dict(size=13))])
    return fig


def stacked_area(cfg, contrib: pd.DataFrame):
    fig = go.Figure()
    for i, c in enumerate(contrib.columns):
        fig.add_trace(go.Scatter(x=contrib.index, y=contrib[c], name=t(c), stackgroup="one",
                                 line=dict(width=0.5, color=color_of(cfg, c, i)),
                                 hovertemplate=f"{t(c)}: %{{y:,.2f}} pts<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title=t("Index points"), legend=dict(orientation="h", y=-0.15),
                      xaxis_tickformat=date_ticks(), xaxis_hoverformat="%d/%m/%Y")
    return fig


def relatives_chart(cfg, prices: pd.DataFrame):
    rel = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, c in enumerate(rel.columns):
        fig.add_trace(go.Scatter(x=rel.index, y=rel[c], name=t(c), mode="lines",
                                 line=dict(color=color_of(cfg, c, i)),
                                 customdata=prices[c].values,
                                 hovertemplate=f"{t(c)}: %{{y:.1f}} ({t('price')} %{{customdata:,.2f}})"
                                               "<extra></extra>"))
    fig.add_hline(y=100, line_color="#999", line_width=1, opacity=0.35)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis_title=t("Price relative (base = 100)"), legend=dict(orientation="h", y=-0.15),
                      xaxis_tickformat=date_ticks(), xaxis_hoverformat="%d/%m/%Y")
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
    metric(c[0], f"{TICKER} · {fdate(last_d)}", f"{last:,.2f}", f"{chg:+.2f} pts")
    metric(c[1], t("1-day change (per radiator)"), f"${U * chg / base_level:+,.2f}", f"{chg_pct:+.2f}%",
           help=t("Index change applied to the reference cost of ${u} per radiator at base.", u=f"{U:,.0f}"))
    metric(c[2], t("30-day trend"), f"{trend30:+.2f}%", f"{last - v30:+.2f} pts",
           chart_data=spark, chart_type="line")
    metric(c[3], t("Implied material+conversion cost"), f"${U * last / base_level:,.2f}",
           t("{p}% vs base", p=f"{(last / base_level - 1) * 100:+.2f}"))

    st.plotly_chart(index_chart(cfg, index, contrib, periods), width="stretch")

    t1, t2, t3 = st.tabs([t("Weight breakdown"), t("Contributions over time"), t("Component prices")])
    with t1:
        cw = {n: w for n, w in current_weights(cfg).items() if w > 0}
        left, right = st.columns([1, 1])
        with left:
            show_drift = st.toggle(t("Show drifted (today's) weights"), help=t(
                "Target weights are set at each rebalance. Drifted weights are the cost shares today, after relative "
                "price moves: if copper rallies, it is a bigger slice of the radiator."))
            if show_drift:
                ew = eff_w.iloc[-1]
                st.plotly_chart(donut(cfg, {n: float(ew.get(n, 0)) for n in cw}, t("Drifted")), width="stretch")
            else:
                st.plotly_chart(donut(cfg, cw, t("Target")), width="stretch")
        with right:
            per = periods[-1]
            col_c, col_t, col_d, col_s, col_q = (t("Component"), t("Target %"), t("Drifted %"),
                                                 t("Spend at rebalance ($)"), t("Implied qty / radiator"))
            rows = []
            for n, wt in per["weights"].items():
                unit = cfg["components"].get(n, {}).get("unit", "")
                q = per["qty"][n]
                qty = f"{q * 1000:,.2f} kg" if unit == "USD/t" else (f"{q:,.2f} h" if unit == "USD/hr" else "-")
                rows.append({col_c: t(n), col_t: wt * 100, col_d: float(eff_w.iloc[-1].get(n, 0)) * 100,
                             col_s: per["spend"][n], col_q: qty})
            st.caption(t("Implied bill of materials for a radiator costing ${u} at the last rebalance ({d}). "
                         "Use it to sanity-check weights against your real BOM.",
                         u=f"{U:,.0f}", d=fdate(per["anchor"])))
            st.dataframe(pd.DataFrame(rows).style.format({col_t: "{:.2f}", col_d: "{:.2f}", col_s: "{:,.2f}"}),
                         hide_index=True, width="stretch")
    with t2:
        st.caption(t("Index points contributed by each component. The stacked areas add up to the index."))
        st.plotly_chart(stacked_area(cfg, contrib), width="stretch")
    with t3:
        st.caption(t("Each component's price relative to the first day shown (= 100). Hover for actual prices."))
        p = prices.loc[index.index[0]:, [c for c in prices.columns if c in cfg["components"]]]
        st.plotly_chart(relatives_chart(cfg, p), width="stretch")

    # ---- Historical table ----
    st.subheader(t("Historical data"))
    hist = pd.DataFrame({TICKER: index, t("Change (pts)"): index.diff(), t("Change (%)"): index.pct_change() * 100,
                         t("Implied cost ($)"): U * index / base_level})
    units = {n: c.get("unit", "") for n, c in cfg["components"].items()}
    raw = prices.loc[index.index, [c for c in prices.columns if c in units]]
    hist = hist.join(raw.rename(columns={c: f"{t(c)} ({t(units[c])})" for c in raw.columns}))
    if st.toggle(t("Include contribution columns")):
        hist = hist.join(contrib.rename(columns=lambda c: f"{t(c)} (pts)"))
    hist.index.name = t("Date")
    lo_default = max(index.index[0], last_d - pd.Timedelta(days=90)).date()
    lo, hi = st.slider(t("Date range"), min_value=index.index[0].date(), max_value=last_d.date(),
                       value=(lo_default, last_d.date()), format="DD/MM/YYYY")
    view = hist.loc[str(lo):str(hi)].sort_index(ascending=False)
    st.dataframe(view.style.format("{:,.2f}", na_rep="-"), width="stretch", height=360)
    st.download_button(t("Download {t} history (CSV)", t=TICKER), hist.round(6).to_csv().encode(),
                       f"{TICKER}_history_{last_d:%Y%m%d}.csv", "text/csv")


# =============================================================================
# Methodology tab
# =============================================================================
METHOD_EN = r"""
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
"""

METHOD_ES = r"""
#### Fórmula del índice
Fecha base $t=0$, valor base $I_0 = 100$. Con pesos $w_i$ (que suman 1) y precios $P_{i,t}$:

$$I_t = 100 \times \sum_i w_i \frac{P_{i,t}}{P_{i,0}}$$

Es un índice **Laspeyres** de canasta fija: valora el *mismo* radiador cada día.

#### Rebalanceo sin romper el historial (encadenamiento)
Cuando los pesos cambian en la fecha $T_k$, la nueva canasta parte del valor anterior del índice:

$$I_t = I_{T_k} \times \sum_i w_i^{(k)} \frac{P_{i,t}}{P_{i,T_k}}, \qquad t \ge T_k$$

En el propio $T_k$ cada razón vale 1 y $\sum_i w_i^{(k)} = 1$, así que el índice es idéntico con los pesos
anteriores y los nuevos: no hay salto, y los valores antes de $T_k$ nunca cambian. Agregar un componente
significa darle un peso en un nuevo periodo. Quitarlo significa poner su peso en 0. Ambos se encadenan igual.
El modo **Recalcular** en cambio vuelve a calcular todo el historial con los nuevos pesos desde la fecha base.
Es útil mientras se calibra, pero los valores ya publicados cambiarán.

#### Vista con divisor (estilo Dow)
De forma equivalente $I_t = \sum_i q_i^{(k)} P_{i,t} \, / \, D_k$, donde $q_i^{(k)} = w_i^{(k)} U / P_{i,T_k}$ son
las cantidades físicas implícitas en un radiador que cuesta $U$ en el rebalanceo, y $D_k = U / I_{T_k}$. Cada
rebalanceo reinicia las cantidades y ajusta el divisor para que el nivel sea continuo.

#### Pesos derivados
Entre rebalanceos, la participación real en el costo del componente $i$ es $w_i R_{i,t} / \sum_j w_j R_{j,t}$,
donde $R_{i,t} = P_{i,t}/P_{i,T_k}$.

#### Calendario y faltantes
Días hábiles. Una serie sin dato nuevo (feriado, publicación mensual) mantiene su último valor.
"""


def page_methodology(cfg, periods):
    st.markdown(METHOD_ES if lang() == "es" else METHOD_EN)
    st.markdown(t("#### Rebalance periods"))
    rows = [{t("version"): p["version"], t("anchor"): p["anchor"].date(), t("level_at_anchor"): round(p["level"], 4),
             t("divisor"): round(p["divisor"], 6), t("mode"): p["mode"],
             t("weights"): ", ".join(f"{t(n)} {w * 100:.2f}%" for n, w in p["weights"].items()),
             t("note"): p["note"]}
            for p in periods]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if cfg.get("archived_versions"):
        with st.expander(t("Archived (superseded) versions")):
            st.json(cfg["archived_versions"], expanded=False)
    raw = read_market_data()
    if not raw.empty:
        st.markdown(t("#### Market data sources"))
        ms = market_status(raw)
        ms.columns = [t(c) for c in ms.columns]
        st.dataframe(ms, hide_index=True, width="stretch")
        st.caption(t("Each market uses its first source with a value for that day; later sources fill gaps. "
                     "Zinc, tin and lead fall back to IMF monthly averages before daily data exists."))
    st.markdown(t("#### Components"))
    st.dataframe(pd.DataFrame([{t("component"): t(n), t("unit"): t(c.get("unit", "")),
                                t("data"): t(cfg["data_status"].get(n, "")),
                                t("description"): t(c.get("description", ""))}
                               for n, c in cfg["components"].items()]),
                 hide_index=True, width="stretch")


# =============================================================================
# Sidebar admin panel
# =============================================================================
def admin_unlocked() -> bool:
    pw = get_secret("ADMIN_PASSWORD")
    if not pw:
        st.sidebar.caption(t("Admin is open. Set `ADMIN_PASSWORD` in secrets to lock it."))
        return True
    if st.session_state.get("admin_ok"):
        return True
    entered = st.sidebar.text_input(t("Admin password"), type="password")
    if st.sidebar.button(t("Unlock")):
        if hmac.compare_digest(entered, str(pw)):
            st.session_state["admin_ok"] = True
            st.rerun()
        st.sidebar.error(t("Wrong password."))
    return False


def commit(cfg, prices=None, msg=None):
    save_config(cfg)
    if prices is not None:
        save_prices(prices)
    if msg:
        st.session_state["flash"] = msg
    st.rerun()


def choose(widget, label, values, labels, **kw):
    """Show translated labels, return the internal value. Labels are made unique if a translation collides."""
    shown = [lab if labels.count(lab) == 1 else f"{lab} ({v})" for v, lab in zip(values, labels)]
    picked = widget(label, shown, **kw)
    return values[shown.index(picked)] if picked in shown else values[0]


def sidebar_admin(cfg: dict, prices: pd.DataFrame):
    st.sidebar.header(t("Admin panel"))
    if not admin_unlocked():
        return
    last_date = prices.index[-1].date()
    base_date = pd.Timestamp(cfg["base_date"]).date()
    last_eff = max(pd.Timestamp(v["effective_date"]).date() for v in cfg["versions"])
    # widget keys follow the stored weights, so any change (rebalance, add, delete) resets the inputs
    vkey = abs(hash(json.dumps(cfg["versions"], sort_keys=True, default=str))) % 10 ** 10

    # ---- Rebalance settings shared by all actions ----
    with st.sidebar.expander(t("Rebalance settings"), expanded=True):
        mode = choose(st.radio, t("When weights or components change"), ["chain", "restate"],
                      [t("Chain-link (preserve history)"), t("Restate full history")], key=f"mode_{lang()}",
                        help=t("Chain-link starts a new weight period on the effective date. "
                               "Restate recomputes every day since the base date with the new weights."))
        eff = st.date_input(t("Effective date"), value=last_date, min_value=max(last_eff, base_date),
                            max_value=last_date, disabled=(mode == "restate"), format="DD/MM/YYYY",
                            help=t("Same date as the latest version replaces it instead of stacking."))
        note = st.text_input(t("Reason / note"), placeholder=t("e.g. Q4 BOM re-costing"))

    # ---- Edit weights ----
    with st.sidebar.expander(t("Edit weights"), expanded=True):
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
            st.number_input(f"{t(n)} (%)", min_value=0.0, max_value=100.0, step=0.5, key=k,
                            help=t(cfg["components"][n].get("description") or "") or None)
        total = sum(st.session_state[k] for k in keys.values())
        if abs(total - 100) > 0.01:
            st.warning(t("Weights sum to {x}%, not 100%.", x=f"{total:.2f}"))
        else:
            st.success(t("Weights sum to 100%."))
        b1, b2 = st.columns(2)
        b1.button(t("Normalize"), on_click=_normalize, width="stretch")
        if b2.button(t("Apply"), type="primary", disabled=abs(total - 100) > 0.01, width="stretch"):
            new_w = {n: st.session_state[k] for n, k in keys.items()}
            if all(v == 0 for v in new_w.values()):
                st.error(t("At least one component needs a weight."))
            else:
                try:
                    apply_weights(cfg, new_w, mode, eff, note)
                    commit(cfg, msg=t("Weights applied and index recalculated."))
                except ValueError as e:
                    st.error(str(e))

    # ---- Add component ----
    with st.sidebar.expander(t("Add component")):
        with st.form("add_component", clear_on_submit=True):
            name = st.text_input(t("Name"), placeholder=t("Aluminum side plates"))
            unit = st.text_input(t("Unit"), value="USD/t")
            desc = st.text_input(t("Description"), placeholder=t("What it represents and its price proxy"))
            c1, c2 = st.columns(2)
            p0 = c1.number_input(t("Baseline price P0"), min_value=0.0, value=2600.0, format="%.4f")
            pt = c2.number_input(t("Current price Pt"), min_value=0.0, value=2750.0, format="%.4f")
            c3, c4 = st.columns(2)
            w_new = c3.number_input(t("Initial weight %"), min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            vol = c4.number_input(t("Volatility %/yr"), min_value=0.0, max_value=200.0, value=20.0, step=1.0,
                                  help=t("Only shapes the simulated path between P0 and Pt."))
            make_room = st.checkbox(t("Scale other weights down to make room"), value=True)
            if st.form_submit_button(t("Add component")):
                name = name.strip()
                if not name or name in cfg["components"]:
                    st.error(t("Name is empty or already exists."))
                elif p0 <= 0 or pt <= 0:
                    st.error(t("Prices must be positive."))
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
                            apply_weights(cfg, others, mode, eff, note or t("Added {name}", name=name))
                        except ValueError as e:
                            st.error(str(e))
                            return
                        commit(cfg, prices, t("Added {name} at {w}% and rebalanced.", name=name, w=f"{w_new:.2f}"))
                    commit(cfg, prices, t("Added {name} with 0% weight. Set its weight in Edit weights.", name=name))

    # ---- Remove component ----
    with st.sidebar.expander(t("Remove component")):
        names = list(cfg["components"])
        target = choose(st.selectbox, t("Component"), names, [t(n) for n in names], key=f"rm_target_{lang()}")
        how = choose(st.radio, t("Action"), ["disable", "delete"],
                     [t("Disable (weight 0, keeps history)"), t("Delete permanently (restates history)")],
                     key=f"rm_how_{lang()}")
        if st.button(t("Remove"), key="rm_go"):
            cw = current_weights(cfg)
            rest = {n: x for n, x in cw.items() if n != target and x > 0}
            if not rest:
                st.error(t("Cannot remove the last weighted component."))
            elif how == "disable":
                try:
                    apply_weights(cfg, rescale(rest), mode, eff, note or t("Disabled {name}", name=t(target)))
                    commit(cfg, msg=t("{name} disabled; remaining weights scaled to 100%.", name=t(target)))
                except ValueError as e:
                    st.error(str(e))
            else:
                for v in cfg["versions"]:
                    v["weights"] = rescale({n: x for n, x in v["weights"].items() if n != target})
                cfg["versions"] = [v for v in cfg["versions"] if v["weights"]]
                cfg["components"].pop(target, None)
                cfg["data_status"].pop(target, None)
                commit(cfg, prices.drop(columns=[target], errors="ignore"),
                       t("{name} deleted; history restated without it.", name=t(target)))

    # ---- Prices ----
    with st.sidebar.expander(t("Update prices")):
        names = list(cfg["components"])
        comp = choose(st.selectbox, t("Component"), names, [t(n) for n in names], key=f"px_comp_{lang()}")
        c1, c2 = st.columns(2)
        d = c1.date_input(t("Date"), value=dt.date.today(), min_value=prices.index[0].date(), key="px_date",
                          format="DD/MM/YYYY")
        last_val = float(prices[comp].dropna().iloc[-1]) if comp in prices else 0.0
        val = c2.number_input(t("Price"), min_value=0.0, value=last_val, format="%.4f", key="px_val")
        if st.button(t("Save price"), key="px_save") and val > 0:
            prices = set_prices(prices, comp, pd.Series([val], index=[pd.Timestamp(d)]))
            cfg["data_status"][comp] = "manual" if cfg["data_status"].get(comp) != "live" else "live"
            commit(cfg, prices, t("{c} = {v} on {d}.", c=t(comp), v=f"{val:,.4f}", d=fdate(d)))
        up = st.file_uploader(t("Or upload CSV (date,value)"), type=["csv"], key="px_csv")
        if up is not None and st.button(t("Import CSV"), key="px_imp"):
            df = pd.read_csv(up)
            df.columns = [c.strip().lower() for c in df.columns]
            s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values, index=pd.to_datetime(df["date"])).dropna()
            prices = set_prices(prices, comp, s)
            cfg["data_status"][comp] = "manual"
            commit(cfg, prices, t("Imported {n} prices into {c}.", n=len(s), c=t(comp)))

    # ---- Data & settings ----
    with st.sidebar.expander(t("Data & settings")):
        U = st.number_input(t("Reference cost per radiator at base ($)"), min_value=1.0,
                            value=float(cfg.get("reference_unit_cost", 1000.0)), step=50.0,
                            help=t("Turns index points into dollars: implied cost = U x I / 100."))
        if U != cfg.get("reference_unit_cost"):
            cfg["reference_unit_cost"] = U
            commit(cfg)
        meta = read_market_meta()
        st.caption(t("Base date {b} = {l}. Market data is updated every weekday evening by the GitHub job. "
                     "Last update: {r}.", b=fdate(cfg["base_date"]), l=f"{cfg.get('base_level', BASE_LEVEL):g}",
                     r=fdate(meta["updated"], "%d %b %Y %H:%M UTC") if meta.get("updated") else t("never")))
        if meta.get("log"):
            errs = [r for r in meta["log"] if r.get("status") != "ok"]
            if errs:
                st.warning(t("{n} source(s) failed in the last update; their backup source was used.", n=len(errs)))
                st.dataframe(pd.DataFrame(errs)[["market", "source", "status"]], hide_index=True)
        st.download_button(t("Download index_config.json"), json.dumps(cfg, indent=2, default=str).encode(),
                           "index_config.json", "application/json")
        st.download_button(t("Download price_history.csv"), prices.rename_axis("date").to_csv().encode(),
                           "price_history.csv", "text/csv")
        upc = st.file_uploader(t("Restore index_config.json"), type=["json"], key="cfg_up")
        if upc is not None and st.button(t("Restore config")):
            new_cfg = json.load(upc)
            if not {"components", "versions", "base_date"} <= set(new_cfg):
                st.error(t("Not a valid index config."))
            else:
                new_cfg.setdefault("data_status", {n: "simulated" for n in new_cfg["components"]})
                new_cfg.setdefault("archived_versions", [])
                prices = ensure_columns(new_cfg, prices)
                commit(new_cfg, prices, t("Config restored."))
        if st.button(t("Reset to demo defaults")):
            st.session_state["confirm_reset"] = True
        if st.session_state.get("confirm_reset"):
            st.warning(t("This replaces all weights, components and prices with fresh simulated data."))
            if st.button(t("Yes, reset everything"), type="primary"):
                st.session_state.pop("confirm_reset")
                dates = history_calendar()
                commit(new_default_config(dates[0]), generate_default_mock(dates), t("Reset to demo defaults."))


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
def main():
    init_language()
    st.set_page_config(page_title=t(INDEX_NAME), page_icon="📈", layout="wide")
    language_switch()
    cfg, prices = load_state()
    prices = overlay_live(cfg, prices)

    st.title(t(cfg.get("index_name", INDEX_NAME)))
    st.caption(f"{TICKER} · " + t("daily cost index of copper, brass, solder, energy and labor for a heavy-duty "
                                  "copper-brass radiator")
               + " · " + t("base {date} = {level}", date=fdate(cfg["base_date"]),
                           level=f"{cfg.get('base_level', BASE_LEVEL):g}"))
    if msg := st.session_state.pop("flash", None):
        st.toast(msg)

    weighted = [n for n, w in current_weights(cfg).items() if w > 0]
    simulated = [n for n in weighted if cfg["data_status"].get(n) == "simulated"]
    if simulated:
        st.info(t("Demo data: {names} {verb} simulated until the nightly market-data job has run, "
                  "or upload your own prices.",
                  names=", ".join(t(n) for n in simulated), verb=t("is") if len(simulated) == 1 else t("are")),
                icon="🧪")

    sidebar_admin(cfg, prices)

    tab_dash, tab_method = st.tabs([t("Dashboard"), t("Methodology")])
    try:
        index, contrib, eff_w, periods = compute_index(prices, cfg)
        index = index.dropna()
    except Exception as e:
        with tab_dash:
            st.error(t("The index could not be computed: {e}", e=e))
        return
    with tab_dash:
        page_dashboard(cfg, prices, index, contrib.loc[index.index], eff_w.loc[index.index], periods)
    with tab_method:
        page_methodology(cfg, periods)
    if uses_metalcharts(read_market_data()):
        st.caption(t("Metal prices by [MetalCharts]({u}).", u=METALCHARTS_URL))


if __name__ == "__main__":
    main()
