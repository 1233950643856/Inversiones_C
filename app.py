"""Streamlit App principal - Inversiones PRO."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime

from config import (ASSETS, ASSET_LIST, BROKER_FEES, RISK_PROFILES,
                    HORIZON_ADJUSTMENT, DEFAULT_PREFS, PORTFOLIO_NAMES,
                    PORTFOLIO_LABELS, STRESS_SCENARIOS, DISCLAIMER, now_cet)
from data_loader import load_universe, daily_returns, fetch_eur_usd, benchmark_series
from feature_engineering import universe_features
from ml_predictor import rank_universe
from optimizer import (estimate_covariance, expected_returns_simple,
                       markowitz_optimize, min_cvar_optimize, hrp_optimize,
                       black_litterman, all_weather_weights, balanced_60_40,
                       income_portfolio)
from backtester import backtest_threshold, stress_test
from allocator import allocate_budget
from metrics import full_metrics, sharpe, sortino, max_drawdown, cvar, herfindahl, effective_n, sector_concentration, expected_dividend_income
from profiler import QUESTIONS, evaluate
import logbook
import alerts
import supabase_db
import ai_provider
import coach_ai
import news_ai
import report_ai
from scheduler import start_scheduler, read_flag, next_market_close

st.set_page_config(page_title="Inversiones PRO", page_icon="P",
                   layout="wide", initial_sidebar_state="expanded")

# === PWA: manifest + iOS meta tags ===
import streamlit.components.v1 as _components
_PWA_HTML = """
<link rel="manifest" href="data:application/json;charset=utf-8,%7B%22name%22%3A%22Inversiones%20PRO%22%2C%22short_name%22%3A%22InvPRO%22%2C%22start_url%22%3A%22.%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%230E1117%22%2C%22theme_color%22%3A%22%2300C9A7%22%2C%22icons%22%3A%5B%7B%22src%22%3A%22data%3Aimage%2Fsvg%2Bxml%2C%253Csvg%2520xmlns%253D%2527http%253A%2F%2Fwww.w3.org%2F2000%2Fsvg%2527%2520viewBox%253D%25270%25200%2520192%2520192%2527%253E%253Crect%2520width%253D%2527192%2527%2520height%253D%2527192%2527%2520fill%253D%2527%25230E1117%2527%2F%253E%253Ctext%2520x%253D%252796%2527%2520y%253D%2527130%2527%2520font-family%253D%2527Arial%2527%2520font-size%253D%2527100%2527%2520font-weight%253D%2527bold%2527%2520fill%253D%2527%252300C9A7%2527%2520text-anchor%253D%2527middle%2527%253EP%253C%2Ftext%253E%253C%2Fsvg%253E%22%2C%22sizes%22%3A%22192x192%22%2C%22type%22%3A%22image%2Fsvg%2Bxml%22%7D%5D%7D">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="InvPRO">
<meta name="theme-color" content="#00C9A7">
<link rel="apple-touch-icon" href="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 192 192'><rect width='192' height='192' fill='%230E1117'/><text x='96' y='130' font-family='Arial' font-size='100' font-weight='bold' fill='%2300C9A7' text-anchor='middle'>P</text></svg>">
"""
_components.html(f'<script>parent.document.head.insertAdjacentHTML("beforeend", `{_PWA_HTML}`);</script>', height=0)



# === WEBHOOK CRON-JOB: comprueba drift y envia email ===
def _webhook_check_drift():
    """Endpoint llamado por cron-job.org. Comprueba drift y envia email si procede."""
    import logbook as _lb
    from data_loader import load_universe, fetch_eur_usd
    try:
        _prices, _fund = load_universe(period="3mo")
        if _prices is None or _prices.empty:
            return False, "Sin datos de mercado"
        _eur_usd = fetch_eur_usd()
        # Necesitamos las carteras objetivo - reconstruccion minima de la cartera por defecto
        from optimizer import hrp_optimize
        from data_loader import daily_returns
        _R = daily_returns(_prices)
        _w_target = hrp_optimize(_R)
        _md = _lb.max_drift(_w_target)
        _thr = 0.05
        if _md < _thr:
            alerts.log_alert if False else None
            try:
                supabase_db.log_alert("Cron drift check", f"OK: drift {_md*100:.2f}% < threshold")
            except Exception:
                pass
            return False, f"Drift {_md*100:.2f}% bajo umbral, no envio."
        _pv = _lb.portfolio_value(_prices.iloc[-1], _eur_usd)
        _summary = _pv[1] if _pv else {}
        ok, msg = alerts.check_and_alert(_w_target, "HRP (cron)", threshold=_thr, summary=_summary)
        return ok, msg
    except Exception as e:
        return False, f"Error: {e}"

try:
    _qp = st.query_params
    _action = _qp.get("action", "")
    _token = _qp.get("token", "")
    _expected_token = ""
    try:
        _expected_token = st.secrets.get("CRON_TOKEN", "")
    except Exception:
        pass
    if _action == "check_drift" and _expected_token and _token == _expected_token:
        st.title("Webhook ejecutandose")
        with st.spinner("Comprobando drift..."):
            _ok, _msg = _webhook_check_drift()
        if _ok:
            st.success(f"Email enviado: {_msg}")
        else:
            st.info(f"Sin envio: {_msg}")
        st.caption(f"Timestamp: {now_cet().strftime('%Y-%m-%d %H:%M %Z')}")
        st.stop()
    elif _action == "check_drift":
        st.error("Token invalido o no configurado.")
        st.stop()
except Exception:
    pass

# Inicializar scheduler una sola vez
if "scheduler_started" not in st.session_state:
    try:
        start_scheduler()
        st.session_state.scheduler_started = True
    except Exception:
        st.session_state.scheduler_started = False

# Detectar si el scheduler ha refrescado y limpiar caches
flag = read_flag()
if flag:
    if st.session_state.get("last_flag_ts") != flag.get("ts"):
        st.session_state.last_flag_ts = flag.get("ts")
        st.cache_data.clear()

# Estado por defecto
for k, v in DEFAULT_PREFS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def fmt_pct(x, dec=2):
    if x is None or pd.isna(x): return "-"
    return f"{x*100:.{dec}f}%"

def fmt_eur(x, dec=2):
    if x is None or pd.isna(x): return "-"
    return f"{x:,.{dec}f} EUR"

def fmt_num(x, dec=2):
    if x is None or pd.isna(x): return "-"
    return f"{x:.{dec}f}"


# ============================ SIDEBAR ============================
st.sidebar.title("Inversiones PRO")
st.sidebar.caption("Sistema profesional v3 - Metodologia institucional")

PAGES = ["Inicio + Perfil","Dashboard","Mi cartera real","Asistente IA",
         "Coach de sesgos","Informe mensual","Carteras","Detalle activo",
         "Optimizacion manual","Presupuesto y compras","Stress test",
         "Backtesting","Ranking ML","Alertas email","Configuracion IA","Configuracion"]
page = st.sidebar.radio("Navegacion", PAGES, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Preferencias rapidas")
st.session_state.budget = st.sidebar.number_input("Presupuesto (EUR)",
    min_value=50.0, max_value=100000.0, value=float(st.session_state.budget), step=50.0)
st.session_state.broker = st.sidebar.selectbox("Broker",
    options=list(BROKER_FEES.keys()),
    index=list(BROKER_FEES.keys()).index(st.session_state.broker))
st.session_state.fractional_shares = st.sidebar.checkbox("Fracciones de accion",
    value=st.session_state.fractional_shares)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros del universo**")
with st.sidebar.expander("Filtros avanzados", expanded=False):
    type_filter = st.multiselect("Tipo de activo",
        options=["ETF_EQ","ETF_BOND","STOCK","REIT","COMMODITY"],
        default=["ETF_EQ","ETF_BOND","STOCK","REIT","COMMODITY"])
    sectors_avail = sorted(set(v["sector"] for v in ASSETS.values()))
    sectors_filter = st.multiselect("Sectores", options=sectors_avail, default=sectors_avail)
    regions_avail = sorted(set(v["region"] for v in ASSETS.values()))
    regions_filter = st.multiselect("Regiones", options=regions_avail, default=regions_avail)
    max_price = st.number_input("Precio maximo accion (EUR)",
        min_value=10.0, max_value=2000.0, value=float(st.session_state.max_share_price), step=10.0)
    st.session_state.max_share_price = max_price

# Disclaimer permanente
st.sidebar.markdown("---")
with st.sidebar.expander("Aviso legal"):
    st.caption(DISCLAIMER)


# ============================ DATA LOAD (cached) ============================
@st.cache_data(ttl=14400, show_spinner="Cargando precios y fundamentales...")
def load_all():
    prices, fund = load_universe(period="3y")
    eur_usd = fetch_eur_usd()
    return prices, fund, eur_usd

@st.cache_data(ttl=14400, show_spinner="Construyendo features ML...")
def build_features_cached(_prices_hash, prices):
    return universe_features(prices, target_horizon=21)

@st.cache_data(ttl=14400, show_spinner="Calculando ranking ML (puede tardar 1-2 min)...")
def get_ranking(_features_hash, features):
    return rank_universe(features, min_train=252)


def filter_universe_df(prices, fund, type_filter, sectors_filter, regions_filter,
                       max_price, eur_usd):
    """Aplica filtros sobre el universo y devuelve precios filtrados."""
    valid = []
    for t in prices.columns:
        meta = ASSETS.get(t, {})
        if meta.get("type") not in type_filter: continue
        if meta.get("sector") not in sectors_filter: continue
        if meta.get("region") not in regions_filter: continue
        # Precio en EUR
        suffixes = (".DE",".AS",".MC",".PA",".MI",".SW",".L",".CO")
        is_eur = any(t.endswith(s) for s in suffixes) or t.startswith("VWCE")
        last = float(prices[t].dropna().iloc[-1]) if len(prices[t].dropna())>0 else 0
        price_eur = last if is_eur else last/eur_usd
        if price_eur > max_price and not st.session_state.fractional_shares:
            # Solo filtra duro si no hay fracciones
            continue
        valid.append(t)
    return prices[valid]


@st.cache_data(ttl=14400, show_spinner="Optimizando carteras...")
def compute_all_portfolios(_prices_hash, prices, fund_dict, type_map, sector_map,
                            ml_views_dict, profile_constraints):
    """Genera las 8 carteras."""
    R = daily_returns(prices)
    fund = pd.DataFrame.from_dict(fund_dict, orient="index")
    portfolios = {}
    cons = profile_constraints
    try:
        sigma = estimate_covariance(R, "ledoit_wolf")
        mu = expected_returns_simple(R)
        portfolios["max_sharpe_lw"] = markowitz_optimize(
            mu, sigma, "max_sharpe",
            max_per_asset=cons["max_single_asset"], max_per_sector=cons["max_sector"],
            sector_map=sector_map, type_map=type_map,
            min_bonds=cons["min_bonds"], max_equity=cons["max_equity"])
        portfolios["min_vol"] = markowitz_optimize(
            mu, sigma, "min_vol",
            max_per_asset=cons["max_single_asset"], max_per_sector=cons["max_sector"],
            sector_map=sector_map, type_map=type_map,
            min_bonds=cons["min_bonds"], max_equity=cons["max_equity"])
    except Exception as e:
        portfolios["max_sharpe_lw"] = portfolios.get("max_sharpe_lw", {})
        portfolios["min_vol"] = portfolios.get("min_vol", {})
    try:
        portfolios["min_cvar"] = min_cvar_optimize(R, alpha=0.05,
            max_per_asset=cons["max_single_asset"],
            sector_map=sector_map, max_per_sector=cons["max_sector"])
    except Exception:
        portfolios["min_cvar"] = {}
    try:
        portfolios["hrp"] = hrp_optimize(R)
    except Exception:
        portfolios["hrp"] = {}
    try:
        market_caps = {t: fund.loc[t,"market_cap"] if t in fund.index and pd.notna(fund.loc[t,"market_cap"]) else 1e9
                       for t in R.columns}
        portfolios["black_litterman"] = black_litterman(
            R, market_caps, ml_views_dict, tau=0.05, view_confidence=0.5,
            max_per_asset=cons["max_single_asset"],
            sector_map=sector_map, max_per_sector=cons["max_sector"],
            type_map=type_map, min_bonds=cons["min_bonds"], max_equity=cons["max_equity"])
    except Exception:
        portfolios["black_litterman"] = {}
    try:
        portfolios["balanced_60_40"] = balanced_60_40(R, type_map, sector_map)
    except Exception:
        portfolios["balanced_60_40"] = {}
    try:
        portfolios["income"] = income_portfolio(R, fund, type_map)
    except Exception:
        portfolios["income"] = {}
    try:
        portfolios["all_weather"] = all_weather_weights(list(R.columns), type_map)
    except Exception:
        portfolios["all_weather"] = {}
    return portfolios


def portfolio_returns_series(weights, prices):
    """Serie de retornos de la cartera dada (rebalanceo diario teorico)."""
    cols = [c for c in weights if c in prices.columns]
    if not cols: return pd.Series(dtype=float)
    R = prices[cols].pct_change().dropna()
    w = np.array([weights[c] for c in cols])
    w = w/w.sum() if w.sum() > 0 else np.ones(len(cols))/len(cols)
    return (R.values @ w)


# Cargar datos
prices_full, fund, eur_usd = load_all()
if prices_full.empty:
    st.error("No se pudieron descargar datos. Reintenta en unos segundos.")
    st.stop()

# Aplicar filtros del sidebar
prices = filter_universe_df(prices_full, fund, type_filter, sectors_filter,
                             regions_filter, st.session_state.max_share_price, eur_usd)

if prices.empty or prices.shape[1] < 3:
    st.warning("Los filtros han dejado un universo demasiado pequeno (<3 activos). Ajusta filtros.")
    st.stop()

returns = daily_returns(prices)
type_map = {t: ASSETS.get(t, {}).get("type", "STOCK") for t in prices.columns}
sector_map = {t: ASSETS.get(t, {}).get("sector", "Otros") for t in prices.columns}

# Construir features y ranking ML
prices_hash = hash(tuple(sorted(prices.columns)) + (str(prices.index[-1]),))
features = build_features_cached(prices_hash, prices)
features_hash = hash(tuple(sorted(features.keys())))
ranking = get_ranking(features_hash, features)
ml_views = dict(zip(ranking["ticker"], ranking["predicted_return"])) if not ranking.empty else {}

# Restricciones segun perfil + horizonte
profile = st.session_state.risk_profile
horizon = st.session_state.horizon
profile_cons = dict(RISK_PROFILES[profile])
hadj = HORIZON_ADJUSTMENT.get(horizon, {})
profile_cons["min_bonds"] = max(0.0, profile_cons["min_bonds"] + hadj.get("shift_to_bonds", 0))
profile_cons["max_equity"] = min(profile_cons["max_equity"], hadj.get("max_eq_cap", 1.0))

fund_dict = fund.to_dict(orient="index") if not fund.empty else {}
portfolios = compute_all_portfolios(prices_hash, prices, fund_dict, type_map,
                                     sector_map, ml_views, profile_cons)


# ==================================================================
# PAGINA: INICIO + PERFIL
# ==================================================================
if page == "Inicio + Perfil":
    st.title("Bienvenida a Inversiones PRO")
    st.markdown("""
    **Sistema profesional de soporte a la decision** para inversores con presupuestos
    pequenos (300-700 EUR). Aplica metodologia institucional adaptada a tu perfil:

    - Optimizacion robusta con **Ledoit-Wolf shrinkage** (resuelve la inestabilidad clasica de Markowitz)
    - **Black-Litterman** integra las predicciones del modelo ML como vistas
    - **Hierarchical Risk Parity** (Lopez de Prado, 2016) para diversificacion robusta
    - **Min-CVaR** optimiza directamente la cola peor (Expected Shortfall)
    - **Walk-forward validation** elimina look-ahead bias en el modelo ML
    - **Stress testing** con crisis reales (2008, COVID, dot-com, euro 2011)
    - **Renta fija incluida** (bonos gobierno, corporativos, letras)
    - Rebalanceo por umbral con **fiscalidad espanola estimada**
    """)
    st.markdown("---")
    st.subheader("Cuestionario de perfil de riesgo")
    st.caption("Responde 6 preguntas. La app ajustara automaticamente las restricciones de las carteras a tu perfil.")
    with st.form("perfil_form"):
        answers = {}
        for q in QUESTIONS:
            opts = [o[0] for o in q["options"]]
            choice = st.radio(q["text"], opts, key=f"qa_{q['id']}", index=0)
            answers[q["id"]] = opts.index(choice)
        submitted = st.form_submit_button("Calcular mi perfil", type="primary")
    if submitted:
        result = evaluate(answers)
        st.session_state.risk_profile = result["profile"]
        st.session_state.horizon = result["horizon"]
        st.success(f"Tu perfil: **{result['profile']}** | Horizonte: **{result['horizon']}** | Score: {result['score']:.1f}")
        st.info(RISK_PROFILES[result["profile"]]["description"])

    st.markdown("---")
    st.subheader("Tu configuracion actual")
    c1,c2,c3 = st.columns(3)
    c1.metric("Perfil", st.session_state.risk_profile)
    c2.metric("Horizonte", st.session_state.horizon)
    c3.metric("Presupuesto", fmt_eur(st.session_state.budget))
    cons = profile_cons
    st.write(f"**Restricciones aplicadas:** Max activo individual {fmt_pct(cons['max_single_asset'])} | "
             f"Max sector {fmt_pct(cons['max_sector'])} | "
             f"Min bonos {fmt_pct(cons['min_bonds'])} | "
             f"Max RV {fmt_pct(cons['max_equity'])}")


# ==================================================================
# PAGINA: DASHBOARD
# ==================================================================
elif page == "Dashboard":
    st.title("Dashboard")
    st.caption(f"Universo: {prices.shape[1]} activos | Datos hasta {prices.index[-1].date()}")

    # === Banner de drift si hay cartera real registrada ===
    try:
        rec_map_d = {"Conservador":"min_vol","Moderado":"hrp","Crecimiento":"black_litterman","Agresivo":"max_sharpe_lw"}
        _rec_key_d = rec_map_d.get(st.session_state.risk_profile, "hrp")
        _w_target = portfolios.get(_rec_key_d, {})
        if _w_target and not logbook.list_transactions().empty:
            _md = logbook.max_drift(_w_target)
            _thr = st.session_state.get("rebalance_threshold", 0.05)
            if _md >= _thr:
                _c1, _c2 = st.columns([3,1])
                _c1.warning(f"Tu cartera real ha derivado **{_md*100:.2f}%** del objetivo (umbral {_thr*100:.1f}%). "
                            f"Considera rebalancear. Pagina 'Mi cartera real' para detalles.")
                if alerts.is_configured() and alerts.should_send_drift_alert():
                    if _c2.button("Avisar por email"):
                        _pv = logbook.portfolio_value(prices.iloc[-1], eur_usd)
                        _summary = _pv[1] if _pv else {}
                        ok, msg = alerts.check_and_alert(_w_target,
                            f"Recomendada ({_rec_key_d})", threshold=_thr, summary=_summary)
                        if ok: st.success("Email enviado")
                        else: st.error(msg)
            elif _md > 0.001:
                st.info(f"Drift actual: {_md*100:.2f}% (bajo umbral {_thr*100:.1f}%). Cartera alineada.")
    except Exception:
        pass

    # Recomendacion segun perfil
    rec_map = {
        "Conservador":"min_vol", "Moderado":"hrp",
        "Crecimiento":"black_litterman", "Agresivo":"max_sharpe_lw"
    }
    rec_key = rec_map.get(st.session_state.risk_profile, "hrp")
    st.success(f"**Recomendacion para tu perfil ({st.session_state.risk_profile}):** "
               f"{PORTFOLIO_LABELS[rec_key]}")

    # KPIs principales
    rec_w = portfolios.get(rec_key, {})
    rec_rets = pd.Series(portfolio_returns_series(rec_w, prices), index=returns.index[-len(portfolio_returns_series(rec_w, prices)):]) if rec_w else pd.Series(dtype=float)
    if len(rec_rets) > 30:
        m = full_metrics(rec_rets, benchmark_series(prices), rf=0.02)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Retorno anual", fmt_pct(m["Retorno anual"]))
        c2.metric("Volatilidad", fmt_pct(m["Volatilidad"]))
        c3.metric("Sharpe", fmt_num(m["Sharpe"]))
        c4.metric("CVaR 95%", fmt_pct(m["CVaR 95% (ES)"]))
        c1.metric("Max DD", fmt_pct(m["Max Drawdown"]))
        c2.metric("Sortino", fmt_num(m["Sortino"]))
        c3.metric("Calmar", fmt_num(m["Calmar"]))
        c4.metric("Omega", fmt_num(m["Omega (>0)"]))


    # === Insight diario IA (cacheado 6h) ===
    if ai_provider.is_configured():
        with st.expander("Insight diario (IA)", expanded=True):
            try:
                last_rets = returns.tail(1)
                if not last_rets.empty:
                    today_movers = sorted([(c, float(last_rets.iloc[0][c]))
                                           for c in last_rets.columns
                                           if pd.notna(last_rets.iloc[0][c])],
                                          key=lambda x: -abs(x[1]))[:10]
                    pv = logbook.portfolio_value(prices.iloc[-1], eur_usd)
                    pf_summary = pv[1] if pv else None
                    insight = news_ai.daily_market_insight(today_movers, pf_summary)
                    st.write(insight)
                    st.caption(f"Generado por IA. Cacheado 6 horas.")
            except Exception as _e:
                st.caption(f"Insight no disponible: {_e}")
    else:
        st.info("Activa el Asistente IA en 'Configuracion IA' para ver insights diarios y mucho mas.")

    st.markdown("---")
    st.subheader("Top 10 oportunidades segun el modelo ML")
    if not ranking.empty:
        top = ranking.head(10).copy()
        top["sector"] = top["ticker"].map(lambda t: ASSETS.get(t,{}).get("sector","-"))
        top["region"] = top["ticker"].map(lambda t: ASSETS.get(t,{}).get("region","-"))
        top["nombre"] = top["ticker"].map(lambda t: ASSETS.get(t,{}).get("name",t))
        top["predicted_return"] = top["predicted_return"].apply(fmt_pct)
        top["confidence"] = top["confidence"].apply(lambda x: fmt_num(x,3))
        top["score"] = top["score"].apply(lambda x: fmt_num(x,4))
        st.dataframe(top[["ticker","nombre","sector","region","predicted_return","confidence","score"]],
                     use_container_width=True, hide_index=True)
    else:
        st.warning("Ranking ML aun no disponible.")

    st.markdown("---")
    st.subheader("Cartera recomendada - Composicion")
    if rec_w:
        df_w = pd.DataFrame([
            {"ticker":k, "nombre":ASSETS.get(k,{}).get("name",k),
             "tipo":ASSETS.get(k,{}).get("type","-"),
             "sector":ASSETS.get(k,{}).get("sector","-"),
             "peso":v} for k,v in rec_w.items() if v > 0.001
        ]).sort_values("peso", ascending=False)
        if not df_w.empty:
            colA, colB = st.columns([1,1])
            with colA:
                fig = px.pie(df_w, values="peso", names="ticker", hole=0.4,
                             title="Distribucion por activo")
                st.plotly_chart(fig, use_container_width=True)
            with colB:
                df_sec = df_w.groupby("sector")["peso"].sum().reset_index()
                fig2 = px.pie(df_sec, values="peso", names="sector", hole=0.4,
                              title="Distribucion por sector")
                st.plotly_chart(fig2, use_container_width=True)
            df_w["peso"] = df_w["peso"].apply(fmt_pct)
            st.dataframe(df_w, use_container_width=True, hide_index=True)


# ==================================================================
# PAGINA: CARTERAS (las 8)
# ==================================================================
elif page == "Carteras":
    st.title("Las 8 carteras propuestas")
    st.caption("Cada metodologia tiene fortalezas distintas. Compara y elige segun tu criterio.")
    explanations = {
        "max_sharpe_lw":"Maximiza Sharpe usando covarianza Ledoit-Wolf (mas estable que Markowitz puro).",
        "min_cvar":"Minimiza la perdida esperada en el peor 5% de escenarios. Optima para aversion a perdidas.",
        "hrp":"Hierarchical Risk Parity: agrupa activos correlacionados y reparte riesgo. Robusta sin necesidad de predecir retornos.",
        "black_litterman":"Combina pesos de mercado (prior) con vistas del modelo ML (posterior). Lo mas cercano a metodologia de fondos institucionales.",
        "min_vol":"Minimiza volatilidad. Para perfiles conservadores.",
        "balanced_60_40":"Asignacion clasica 60% renta variable / 40% renta fija. Cada bloque optimizado independientemente.",
        "income":"Sobrepondera dividendos altos y bonos. Para generar renta pasiva.",
        "all_weather":"Inspirada en Bridgewater (Dalio): 30% acciones + 55% bonos + 15% oro. Resistente a multiples regimenes.",
    }
    bench = benchmark_series(prices)
    for key in PORTFOLIO_NAMES:
        w = portfolios.get(key, {})
        with st.expander(f"**{PORTFOLIO_LABELS[key]}**", expanded=(key=="hrp")):
            st.caption(explanations[key])
            w_clean = {k:v for k,v in w.items() if v > 0.001}
            if not w_clean:
                st.warning("No se pudo construir esta cartera con los filtros actuales.")
                continue
            rets = pd.Series(portfolio_returns_series(w_clean, prices))
            if len(rets) > 30:
                rets.index = returns.index[-len(rets):]
                m = full_metrics(rets, bench, rf=0.02)
                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Retorno", fmt_pct(m["Retorno anual"]))
                c2.metric("Vol", fmt_pct(m["Volatilidad"]))
                c3.metric("Sharpe", fmt_num(m["Sharpe"]))
                c4.metric("Max DD", fmt_pct(m["Max Drawdown"]))
                c5.metric("CVaR 95%", fmt_pct(m["CVaR 95% (ES)"]))
                c1.metric("Sortino", fmt_num(m["Sortino"]))
                c2.metric("Calmar", fmt_num(m["Calmar"]))
                c3.metric("Omega", fmt_num(m["Omega (>0)"]))
                c4.metric("Ulcer", fmt_num(m["Ulcer Index"]))
                c5.metric("N efectivo", fmt_num(effective_n(w_clean)))
            df = pd.DataFrame([
                {"ticker":k, "nombre":ASSETS.get(k,{}).get("name",k),
                 "tipo":ASSETS.get(k,{}).get("type",""),
                 "sector":ASSETS.get(k,{}).get("sector",""),
                 "peso":v} for k,v in w_clean.items()
            ]).sort_values("peso", ascending=False)
            df["peso_%"] = df["peso"].apply(fmt_pct)
            st.dataframe(df[["ticker","nombre","tipo","sector","peso_%"]],
                         use_container_width=True, hide_index=True)


# ==================================================================
# PAGINA: DETALLE ACTIVO
# ==================================================================
elif page == "Detalle activo":
    st.title("Detalle de activo")
    sel = st.selectbox("Selecciona un activo", options=list(prices.columns),
                       format_func=lambda t: f"{t} - {ASSETS.get(t,{}).get('name',t)}")
    if sel:
        meta = ASSETS.get(sel, {})
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Tipo", meta.get("type","-"))
        c2.metric("Sector", meta.get("sector","-"))
        c3.metric("Region", meta.get("region","-"))
        c4.metric("Precio (nativa)", fmt_num(float(prices[sel].iloc[-1]),2))
        if sel in fund.index:
            f = fund.loc[sel]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("P/E", fmt_num(f.get("pe")) if pd.notna(f.get("pe")) else "-")
            c2.metric("Div Yield", fmt_pct(f.get("div_yield")) if pd.notna(f.get("div_yield")) else "-")
            c3.metric("Beta", fmt_num(f.get("beta")) if pd.notna(f.get("beta")) else "-")
            mc = f.get("market_cap"); c4.metric("Market Cap", f"{mc/1e9:.1f}B" if pd.notna(mc) and mc else "-")
        # Precio
        st.subheader("Evolucion del precio")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices[sel], mode="lines", name=sel))
        fig.update_layout(height=400, margin=dict(t=30,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)
        # Prediccion ML
        if not ranking.empty and sel in ranking["ticker"].values:
            row = ranking[ranking["ticker"]==sel].iloc[0]
            st.subheader("Prediccion del modelo ML")
            c1,c2,c3 = st.columns(3)
            c1.metric("Retorno predicho (~21d)", fmt_pct(row["predicted_return"]))
            c2.metric("Confianza", fmt_num(row["confidence"],3))
            c3.metric("Score combinado", fmt_num(row["score"],4))

        # === Noticias con IA ===
        st.markdown("---")
        st.subheader("Noticias recientes (IA)")
        if not ai_provider.is_configured():
            st.info("Configura tu API key en 'Configuracion IA' para activar el resumen de noticias.")
        else:
            colA, colB = st.columns([3,1])
            colA.caption("La IA descarga las ultimas noticias de Yahoo Finance, las filtra por relevancia y te las resume en 3-5 bullets.")
            if colB.button("Resumir noticias con IA", key=f"news_btn_{sel}"):
                with st.spinner("Buscando y resumiendo noticias..."):
                    asset_name = ASSETS.get(sel, {}).get("name", sel)
                    res = news_ai.summarize_for_ticker(sel, asset_name)
                if res.get("summary"):
                    st.markdown(res["summary"])
                if res.get("raw"):
                    with st.expander(f"Ver titulares originales ({len(res['raw'])})"):
                        for n in res["raw"]:
                            st.markdown(f"- [{n['title']}]({n['link']})  \n  *{n['date']}*")
        # Metricas historicas
        rets_a = prices[sel].pct_change().dropna()
        if len(rets_a) > 60:
            st.subheader("Metricas historicas (5y)")
            m = full_metrics(rets_a, benchmark_series(prices), rf=0.02)
            # Construir el dataframe ya con strings para evitar TypeError en pandas nuevo
            formatted = {}
            for k, v in m.items():
                if k in ("Retorno anual","Volatilidad","Max Drawdown","VaR 95%","CVaR 95% (ES)","VaR 99%","CVaR 99%","% meses positivos"):
                    formatted[k] = fmt_pct(v)
                elif k == "DD duracion (dias)":
                    try:
                        formatted[k] = f"{int(v) if v else 0}"
                    except Exception:
                        formatted[k] = "-"
                else:
                    formatted[k] = fmt_num(v, 3)
            data = pd.DataFrame.from_dict(formatted, orient="index", columns=["valor"])
            st.dataframe(data, use_container_width=True)


# ==================================================================
# PAGINA: OPTIMIZACION MANUAL
# ==================================================================
elif page == "Optimizacion manual":
    st.title("Optimizacion manual de cartera")
    st.caption("Selecciona los activos que quieras y la app calculara los pesos optimos.")
    selected = st.multiselect("Activos a incluir", options=list(prices.columns),
                              default=list(prices.columns)[:8],
                              format_func=lambda t: f"{t} - {ASSETS.get(t,{}).get('name',t)}")
    method = st.selectbox("Metodologia",
        options=["max_sharpe_lw","min_vol","min_cvar","hrp","black_litterman"],
        format_func=lambda k: PORTFOLIO_LABELS[k])
    max_pa = st.slider("Maximo por activo (%)", 5, 100, 25) / 100.0
    if st.button("Calcular cartera optima") and len(selected) >= 2:
        sub_prices = prices[selected]
        sub_returns = sub_prices.pct_change().dropna()
        sub_sigma = estimate_covariance(sub_returns, "ledoit_wolf")
        sub_mu = expected_returns_simple(sub_returns)
        sub_sector = {t: sector_map[t] for t in selected}
        sub_type = {t: type_map[t] for t in selected}
        if method in ("max_sharpe_lw","min_vol"):
            obj = "max_sharpe" if method == "max_sharpe_lw" else "min_vol"
            w = markowitz_optimize(sub_mu, sub_sigma, obj,
                max_per_asset=max_pa, max_per_sector=1.0,
                sector_map=sub_sector, type_map=sub_type)
        elif method == "min_cvar":
            w = min_cvar_optimize(sub_returns, alpha=0.05, max_per_asset=max_pa)
        elif method == "hrp":
            w = hrp_optimize(sub_returns)
        elif method == "black_litterman":
            mc = {t: fund.loc[t,"market_cap"] if t in fund.index and pd.notna(fund.loc[t,"market_cap"]) else 1e9 for t in selected}
            sub_views = {t: ml_views[t] for t in selected if t in ml_views}
            w = black_litterman(sub_returns, mc, sub_views, max_per_asset=max_pa)
        else:
            w = {}
        if w:
            df = pd.DataFrame([{"ticker":k,"peso":v,"peso_%":fmt_pct(v)}
                               for k,v in w.items() if v>0.001]).sort_values("peso",ascending=False)
            st.dataframe(df[["ticker","peso_%"]], use_container_width=True, hide_index=True)
            rets = pd.Series(portfolio_returns_series(w, sub_prices))
            if len(rets) > 30:
                m = full_metrics(rets, benchmark_series(prices), rf=0.02)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Retorno", fmt_pct(m["Retorno anual"]))
                c2.metric("Vol", fmt_pct(m["Volatilidad"]))
                c3.metric("Sharpe", fmt_num(m["Sharpe"]))
                c4.metric("CVaR 95%", fmt_pct(m["CVaR 95% (ES)"]))


# ==================================================================
# PAGINA: PRESUPUESTO Y COMPRAS
# ==================================================================
elif page == "Presupuesto y compras":
    st.title("Plan de compras concreto")
    st.caption(f"Para tu broker actual: **{st.session_state.broker}** "
               f"({'fracciones SI' if st.session_state.fractional_shares else 'solo enteras'})")
    rec_map = {"Conservador":"min_vol","Moderado":"hrp","Crecimiento":"black_litterman","Agresivo":"max_sharpe_lw"}
    default_idx = PORTFOLIO_NAMES.index(rec_map.get(st.session_state.risk_profile,"hrp"))
    sel_port = st.selectbox("Cartera", options=PORTFOLIO_NAMES,
        index=default_idx, format_func=lambda k: PORTFOLIO_LABELS[k])
    weights = portfolios.get(sel_port, {})
    if not weights:
        st.warning("Esta cartera no se pudo construir.")
        st.stop()
    last_prices = prices.iloc[-1]
    plan = allocate_budget(weights, last_prices, st.session_state.budget,
                           broker=st.session_state.broker,
                           fractional_shares=st.session_state.fractional_shares,
                           eur_usd_rate=eur_usd,
                           max_share_price=st.session_state.max_share_price)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Invertido", fmt_eur(plan["invested"]))
    c2.metric("% del presupuesto", fmt_pct(plan["invested_pct"]))
    c3.metric("Comisiones totales", fmt_eur(plan["total_commission"]))
    c4.metric("Cash sobrante", fmt_eur(plan["cash_left"]))

    st.subheader("Operaciones a ejecutar")
    if plan["operations"]:
        ops = pd.DataFrame(plan["operations"])
        ops["nombre"] = ops["ticker"].map(lambda t: ASSETS.get(t,{}).get("name",t))
        ops["price_eur"] = ops["price_eur"].apply(lambda x: fmt_eur(x,2))
        ops["value_eur"] = ops["value_eur"].apply(lambda x: fmt_eur(x,2))
        ops["commission_eur"] = ops["commission_eur"].apply(lambda x: fmt_eur(x,2))
        ops["weight_target"] = ops["weight_target"].apply(fmt_pct)
        ops["weight_actual"] = ops["weight_actual"].apply(fmt_pct)
        st.dataframe(ops[["ticker","nombre","n_shares","price_eur","value_eur",
                          "commission_eur","weight_target","weight_actual"]],
                     use_container_width=True, hide_index=True)
        # Export
        excel_buf = BytesIO()
        try:
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                pd.DataFrame(plan["operations"]).to_excel(writer, sheet_name="Operaciones", index=False)
                pd.DataFrame([{"k":k,"v":v} for k,v in weights.items() if v>0]).to_excel(
                    writer, sheet_name="Pesos", index=False)
            st.download_button("Descargar plan en Excel", data=excel_buf.getvalue(),
                file_name=f"plan_{sel_port}_{datetime.now().strftime('%Y%m%d')}.xlsx")
        except Exception:
            pass
    if plan["not_affordable"]:
        st.warning("Activos no asignables con tu broker / presupuesto:")
        st.dataframe(pd.DataFrame(plan["not_affordable"]), use_container_width=True, hide_index=True)

    # Renta esperada
    div_income = expected_dividend_income(weights, fund, st.session_state.budget)
    if div_income > 0:
        st.success(f"Ingreso anual estimado por dividendos/cupones: **{fmt_eur(div_income)}** "
                   f"({fmt_pct(div_income/st.session_state.budget)})")


# ==================================================================
# PAGINA: STRESS TEST
# ==================================================================
elif page == "Stress test":
    st.title("Stress test - Como te habria ido en crisis pasadas")
    st.caption("Aplica los retornos historicos reales de cada crisis a las carteras propuestas. Descarga datos largos al pulsar el boton (puede tardar 30-60s la primera vez).")

    @st.cache_data(ttl=86400, show_spinner="Descargando historial largo (max 25y)...")
    def _load_long_history(tickers):
        from data_loader import fetch_prices
        return fetch_prices(list(tickers), period="max")

    if st.button("Ejecutar stress test", type="primary"):
        tickers_needed = set()
        for w in portfolios.values():
            tickers_needed.update([k for k, v in w.items() if v > 0.001])
        if not tickers_needed:
            st.warning("No hay carteras construidas todavia.")
        else:
            long_prices = _load_long_history(tuple(sorted(tickers_needed)))
            if long_prices is None or long_prices.empty:
                st.error("No se pudo descargar historial largo.")
            else:
                rows = []
                for port_key in PORTFOLIO_NAMES:
                    w = portfolios.get(port_key, {})
                    if not w: continue
                    for sname, sdates in STRESS_SCENARIOS.items():
                        r = stress_test(long_prices, w, sdates)
                        if r is None: continue
                        rows.append({
                            "Cartera": PORTFOLIO_LABELS[port_key],
                            "Crisis": sname,
                            "Periodo": f"{r['start']} -> {r['end']}",
                            "Retorno total": fmt_pct(r["total_return"]),
                            "Max DD": fmt_pct(r["max_drawdown"]),
                            "Dias": r["n_days"],
                        })
                if not rows:
                    st.warning("Ningun activo de las carteras tiene historial suficiente para los escenarios. "
                               "Esto suele pasar con ETFs jovenes (post-2018). "
                               "Cambia a una cartera con activos antiguos (AAPL, MSFT, JNJ, KO) en la pagina Carteras.")
                else:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    try:
                        pivot = df.pivot(index="Cartera", columns="Crisis", values="Retorno total")
                        pivot_num = pivot.apply(lambda col: col.str.rstrip("%").astype(float))
                        fig = px.imshow(pivot_num, text_auto=".1f", aspect="auto",
                                        color_continuous_scale="RdYlGn", origin="lower",
                                        title="Retorno por cartera y crisis (%)")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                    st.caption(f"Activos con historial disponible: {long_prices.shape[1]} de {len(tickers_needed)} solicitados.")
    else:
        st.info("Pulsa 'Ejecutar stress test' para empezar. La primera ejecucion descarga 25 anos de historial (cacheado 24h despues).")
        st.markdown("**Crisis evaluadas:**")
        for sname, sdates in STRESS_SCENARIOS.items():
            st.write(f"- **{sname}** ({sdates['start']} a {sdates['end']})")


# ==================================================================
# PAGINA: BACKTESTING
# ==================================================================
elif page == "Backtesting":
    st.title("Backtest historico con rebalanceo por umbral")
    st.caption("Simulacion realista: rebalanceo solo cuando un activo deriva > umbral. Incluye slippage, comisiones y fiscalidad espanola.")
    sel_port = st.selectbox("Cartera a testear", options=PORTFOLIO_NAMES,
        format_func=lambda k: PORTFOLIO_LABELS[k])
    threshold = st.slider("Umbral de rebalanceo (%)", 1, 20, 5) / 100.0
    slippage = st.slider("Slippage por operacion (bps)", 0, 30, 5)
    tax_rate = st.slider("Retencion fiscal sobre plusvalias (%)", 0, 30, 21) / 100.0
    if st.button("Ejecutar backtest"):
        w = portfolios.get(sel_port, {})
        if not w:
            st.warning("Cartera no disponible.")
        else:
            res = backtest_threshold(prices_full, w, threshold=threshold,
                                      slippage_bps=slippage, tax_rate=tax_rate)
            if res is None:
                st.warning("No hay datos suficientes.")
            else:
                rets = res["returns"]
                m = full_metrics(rets, benchmark_series(prices_full), rf=0.02)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Valor final (1 EUR)", f"{res['final_value']:.3f}")
                c2.metric("N rebalanceos", res["n_rebalances"])
                c3.metric("Costes totales", f"{res['total_costs']:.4f}")
                c4.metric("Impuestos pagados", f"{res['total_taxes']:.4f}")
                c1.metric("Retorno anual", fmt_pct(m["Retorno anual"]))
                c2.metric("Vol", fmt_pct(m["Volatilidad"]))
                c3.metric("Sharpe", fmt_num(m["Sharpe"]))
                c4.metric("Max DD", fmt_pct(m["Max Drawdown"]))
                # Grafico
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["value_series"].index, y=res["value_series"],
                                         mode="lines", name=PORTFOLIO_LABELS[sel_port]))
                # Benchmark
                bench = benchmark_series(prices_full)
                bench_cum = (1+bench).cumprod()
                fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum,
                                         mode="lines", name="Benchmark", line=dict(dash="dash")))
                fig.update_layout(height=420, title="Evolucion del valor (1 EUR inicial)")
                st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# PAGINA: RANKING ML
# ==================================================================
elif page == "Ranking ML":
    st.title("Ranking ML completo")
    st.caption("Ensemble XGBoost + LightGBM + RandomForest + Ridge con walk-forward validation.")
    if ranking.empty:
        st.warning("Ranking aun no disponible.")
    else:
        df = ranking.copy()
        df["nombre"] = df["ticker"].map(lambda t: ASSETS.get(t,{}).get("name",t))
        df["sector"] = df["ticker"].map(lambda t: ASSETS.get(t,{}).get("sector","-"))
        df["region"] = df["ticker"].map(lambda t: ASSETS.get(t,{}).get("region","-"))
        df["tipo"] = df["ticker"].map(lambda t: ASSETS.get(t,{}).get("type","-"))
        # Filtros adicionales sobre el ranking
        st.subheader("Filtros del ranking")
        c1,c2 = st.columns(2)
        min_conf = c1.slider("Confianza minima", 0.0, 1.0, 0.0, 0.05)
        min_pred = c2.slider("Retorno predicho minimo (%)", -10.0, 30.0, -5.0, 0.5) / 100.0
        df = df[(df["confidence"] >= min_conf) & (df["predicted_return"] >= min_pred)]
        df_disp = df.copy()
        df_disp["predicted_return"] = df_disp["predicted_return"].apply(fmt_pct)
        df_disp["confidence"] = df_disp["confidence"].apply(lambda x: fmt_num(x,3))
        df_disp["score"] = df_disp["score"].apply(lambda x: fmt_num(x,4))
        st.dataframe(df_disp[["ticker","nombre","tipo","sector","region",
                               "predicted_return","confidence","score"]],
                     use_container_width=True, hide_index=True)


# ==================================================================
# ==================================================================
# PAGINA: MI CARTERA REAL (LOGBOOK)
# ==================================================================
elif page == "Mi cartera real":
    st.title("Mi cartera real")
    st.caption("Registra tus operaciones reales. La app calcula tu rentabilidad y drift.")

    tabs = st.tabs(["Resumen", "Anadir operacion", "Historial", "Backup / Restaurar"])

    with tabs[0]:
        st.caption(f"Backend de almacenamiento: **{logbook.get_backend_info()}**")
        pv = logbook.portfolio_value(prices.iloc[-1], eur_usd)
        if pv is None:
            st.info("Aun no hay operaciones registradas. Pestana 'Anadir operacion' para empezar.")
        else:
            df_pv, summary = pv
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valor actual", fmt_eur(summary["total_value"]))
            c2.metric("Invertido", fmt_eur(summary["total_invested"]))
            c3.metric("P/L total", fmt_eur(summary["total_pl"]),
                      delta=f"{summary['total_pl_pct']*100:+.2f}%")
            c4.metric("Posiciones", len(df_pv))
            if not df_pv.empty:
                df_disp = df_pv.copy()
                df_disp["nombre"] = df_disp["ticker"].map(lambda t: ASSETS.get(t,{}).get("name",t))
                df_disp["cost_basis_eur"] = df_disp["cost_basis_eur"].apply(lambda x: fmt_eur(x,2))
                df_disp["price_now_eur"] = df_disp["price_now_eur"].apply(lambda x: fmt_eur(x,2))
                df_disp["value_eur"] = df_disp["value_eur"].apply(lambda x: fmt_eur(x,2))
                df_disp["invested_eur"] = df_disp["invested_eur"].apply(lambda x: fmt_eur(x,2))
                df_disp["pl_eur"] = df_disp["pl_eur"].apply(lambda x: fmt_eur(x,2))
                df_disp["pl_pct"] = df_disp["pl_pct"].apply(fmt_pct)
                df_disp["weight_actual"] = df_disp["weight_actual"].apply(fmt_pct)
                st.dataframe(df_disp[["ticker","nombre","n_shares","cost_basis_eur",
                    "price_now_eur","value_eur","invested_eur","pl_eur","pl_pct","weight_actual"]],
                    use_container_width=True, hide_index=True)

                st.subheader("Drift vs cartera objetivo")
                rec_map_d = {"Conservador":"min_vol","Moderado":"hrp","Crecimiento":"black_litterman","Agresivo":"max_sharpe_lw"}
                rec_key_d = rec_map_d.get(st.session_state.risk_profile, "hrp")
                target_choice = st.selectbox("Compara con cartera",
                    options=PORTFOLIO_NAMES,
                    index=PORTFOLIO_NAMES.index(rec_key_d),
                    format_func=lambda k: PORTFOLIO_LABELS[k])
                w_target = portfolios.get(target_choice, {})
                drifts = logbook.drift_vs_target(w_target)
                if drifts:
                    drift_rows = []
                    for t, d in sorted(drifts.items(), key=lambda x: -abs(x[1])):
                        if abs(d) < 0.005: continue
                        drift_rows.append({"ticker":t,
                            "nombre":ASSETS.get(t,{}).get("name",t),
                            "drift":fmt_pct(d),
                            "accion":"Vender" if d>0 else "Comprar"})
                    if drift_rows:
                        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)
                    md = logbook.max_drift(w_target)
                    thr = st.session_state.get("rebalance_threshold", 0.05)
                    if md >= thr:
                        st.error(f"Drift maximo: {md*100:.2f}% (>= umbral {thr*100:.1f}%). REBALANCEO RECOMENDADO.")
                    else:
                        st.success(f"Drift maximo: {md*100:.2f}% (< umbral {thr*100:.1f}%). Cartera alineada.")

    with tabs[1]:
        st.subheader("Registrar nueva operacion")
        with st.form("new_tx"):
            c1, c2 = st.columns(2)
            tx_ticker = c1.selectbox("Ticker", options=ASSET_LIST,
                format_func=lambda t: f"{t} - {ASSETS.get(t,{}).get('name',t)}")
            tx_side = c2.selectbox("Tipo", options=["buy","sell"],
                format_func=lambda x: "Compra" if x=="buy" else "Venta")
            c1, c2, c3 = st.columns(3)
            tx_shares = c1.number_input("N. acciones", min_value=0.0001, value=1.0, step=0.0001, format="%.4f")
            tx_price = c2.number_input("Precio (EUR)", min_value=0.01, value=100.0, step=0.01)
            tx_comm = c3.number_input("Comision (EUR)", min_value=0.0, value=1.0, step=0.1)
            tx_date = st.date_input("Fecha")
            tx_notes = st.text_input("Notas (opcional)")
            submit_tx = st.form_submit_button("Anadir operacion", type="primary")
        if submit_tx:
            try:
                logbook.add_transaction(tx_ticker, tx_side, tx_shares, tx_price,
                    commission_eur=tx_comm, broker=st.session_state.broker,
                    notes=tx_notes, date=str(tx_date))
                st.success(f"Operacion registrada: {tx_side.upper()} {tx_shares} {tx_ticker} @ {tx_price} EUR")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    with tabs[2]:
        df_hist = logbook.list_transactions()
        if df_hist.empty:
            st.info("Sin operaciones registradas.")
        else:
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            with st.expander("Eliminar operacion"):
                tx_id_del = st.number_input("ID a eliminar", min_value=1, step=1)
                if st.button("Eliminar"):
                    logbook.delete_transaction(int(tx_id_del))
                    st.success("Eliminada")
                    st.rerun()

    with tabs[3]:
        st.subheader("Backup")
        st.write("Descarga tus operaciones como JSON. Importante en cloud (Streamlit) "
                 "porque la base de datos local se borra cuando la app se reinicia.")
        json_data = logbook.export_json()
        st.download_button("Descargar backup JSON", data=json_data,
            file_name=f"logbook_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json")
        st.subheader("Restaurar desde JSON")
        up = st.file_uploader("Subir archivo JSON", type=["json"])
        if up is not None:
            content = up.read().decode("utf-8")
            n = logbook.import_json(content)
            if n >= 0:
                st.success(f"{n} operaciones importadas")
                st.rerun()
            else:
                st.error("Error al parsear JSON")


# ==================================================================
# PAGINA: ALERTAS EMAIL
# ==================================================================
elif page == "Alertas email":
    st.title("Alertas por email")
    st.caption("Configura tu Gmail para recibir avisos cuando tu cartera real derive del objetivo.")

    st.subheader("Como configurar Gmail (App Password)")
    st.markdown("""
    1. Ve a tu cuenta Google -> Seguridad -> verifica que tienes verificacion en 2 pasos activada.
    2. Entra en https://myaccount.google.com/apppasswords
    3. Genera una contrasena de aplicacion (nombre: "Inversiones PRO").
    4. Copia los 16 caracteres que te da (sin espacios) y pegalo en 'Contrasena Gmail' aqui abajo.
    5. **Importante:** NO uses tu contrasena normal de Gmail, no funciona desde apps externas.
    """)

    st.subheader("Configuracion")
    with st.form("email_cfg"):
        c1, c2 = st.columns(2)
        sender = c1.text_input("Email Gmail (remitente)",
            value=st.session_state.get("email_sender", ""))
        password = c2.text_input("Contrasena de aplicacion (16 chars)",
            value=st.session_state.get("email_password", ""), type="password")
        recipient = st.text_input("Email destinatario",
            value=st.session_state.get("email_recipient", sender or ""))
        save_cfg = st.form_submit_button("Guardar configuracion")
    if save_cfg:
        st.session_state.email_sender = sender
        st.session_state.email_password = password
        st.session_state.email_recipient = recipient
        st.success("Configuracion guardada (en sesion). Para persistir entre sesiones en cloud, "
                   "anade los valores a Streamlit Secrets.")

    st.markdown("---")
    st.subheader("Probar envio")
    if alerts.is_configured():
        if st.button("Enviar email de prueba"):
            ok, msg = alerts.send_email("Inversiones PRO - prueba",
                "<h2>Email de prueba</h2><p>Si lees esto, tu configuracion funciona.</p>")
            if ok: st.success(msg)
            else: st.error(msg)
    else:
        st.info("Configura los 3 campos arriba para activar.")

    st.markdown("---")
    st.subheader("Historial de envios")
    try:
        from pathlib import Path as _P
        import json as _j
        log_path = _P(__file__).parent / "alerts_log.json"
        if log_path.exists():
            log = _j.loads(log_path.read_text())
            if log:
                st.dataframe(pd.DataFrame(log[-20:][::-1]), use_container_width=True, hide_index=True)
            else:
                st.caption("Sin envios registrados.")
        else:
            st.caption("Sin envios registrados aun.")
    except Exception:
        st.caption("Sin historial.")


# ==================================================================
# PAGINA: ASISTENTE IA (CHAT CON CONTEXTO DE TU CARTERA)
# ==================================================================
elif page == "Asistente IA":
    st.title("Asistente IA")
    st.caption("Chat con IA que conoce tu perfil, tu cartera y los datos del mercado.")
    if not ai_provider.is_configured():
        st.warning("Configura tu API key gratuita en 'Configuracion IA' para usar el asistente.")
    else:
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []

        # Mostrar historial
        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Entrada
        user_q = st.chat_input("Pregunta lo que quieras (cartera, metodologia, mercado...)")
        if user_q:
            st.session_state.ai_chat_history.append({"role":"user","content":user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Construir contexto rico
            pv = logbook.portfolio_value(prices.iloc[-1], eur_usd)
            ctx_parts = [
                f"Perfil de la usuaria: {st.session_state.risk_profile}",
                f"Horizonte: {st.session_state.horizon}",
                f"Presupuesto: {st.session_state.budget} EUR",
                f"Broker: {st.session_state.broker}",
                f"Universo activo: {prices.shape[1]} activos",
            ]
            if pv:
                _, _summ = pv
                ctx_parts.append(f"Cartera real: {_summ['total_value']:.2f} EUR (P/L {_summ['total_pl_pct']*100:+.2f}%)")
            # Top recomendaciones del modelo
            if not ranking.empty:
                top5 = ranking.head(5)
                tops = ", ".join([f"{r['ticker']} (pred {r['predicted_return']*100:+.1f}%)"
                                  for _, r in top5.iterrows()])
                ctx_parts.append(f"Top 5 ranking ML: {tops}")
            ctx_str = "\n".join(ctx_parts)

            system = (f"Eres asesora financiera personal de la usuaria. Usa el contexto siguiente "
                      f"para dar respuestas relevantes. Habla en espanol cercano. NO hagas "
                      f"recomendaciones de compra/venta concretas: explica metodologias, ayuda a "
                      f"entender, advierte de riesgos. Si te piden predicciones, recuerda que el "
                      f"mercado es impredecible y enfoca en gestion de riesgo.\n\n"
                      f"CONTEXTO:\n{ctx_str}")

            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    res = ai_provider.ask(user_q, system, max_tokens=600)
                    answer = res.get("text") or f"Error: {res.get('error')}"
                    st.markdown(answer)
                    if res.get("provider"):
                        st.caption(f"Proveedor: {res['provider']}")
            st.session_state.ai_chat_history.append({"role":"assistant","content":answer})

        if st.button("Borrar conversacion"):
            st.session_state.ai_chat_history = []
            st.rerun()


# ==================================================================
# PAGINA: COACH DE SESGOS COGNITIVOS
# ==================================================================
elif page == "Coach de sesgos":
    st.title("Coach de sesgos cognitivos")
    st.caption("Analiza tu comportamiento de inversion y detecta patrones psicologicos perjudiciales.")
    if not ai_provider.is_configured():
        st.warning("Configura tu API key en 'Configuracion IA' para activar el coach.")
    else:
        st.markdown("""
        El coach analiza tres fuentes:
        - Tus operaciones reales (logbook)
        - Tu uso de la app (frecuencia de visitas, paginas que consultas)
        - El comportamiento del mercado en relacion a tus activos

        Detecta los principales sesgos cognitivos identificados en la literatura:
        loss aversion, disposition effect, overtrading, status quo, recency bias, anchoring.
        """)

        coach_ai.log_event("page_view", {"page":"coach"})

        if st.button("Ejecutar analisis", type="primary"):
            transactions_df = logbook.list_transactions()
            pv = logbook.portfolio_value(prices.iloc[-1], eur_usd)
            positions_df = pv[0] if pv else None
            # Returns recientes para detector de recency
            recent_rets = None
            if positions_df is not None and not positions_df.empty:
                cols = [t for t in positions_df["ticker"] if t in prices.columns]
                if cols:
                    weights = {r["ticker"]:r["weight_actual"] for _,r in positions_df.iterrows()
                              if r["ticker"] in cols}
                    recent_rets = pd.Series(portfolio_returns_series(weights, prices))
            # Mercado en caida hoy
            market_drop = False
            try:
                last_day_returns = returns.iloc[-1]
                market_drop = (last_day_returns < -0.01).mean() > 0.5
            except Exception:
                pass

            detections = coach_ai.all_detections(
                positions_df, transactions_df, st.session_state.budget,
                market_dropped_today=market_drop, returns_series=recent_rets,
                expected_monthly_contribution=None,
            )
            if not detections:
                st.success("No se han detectado sesgos significativos. Comportamiento equilibrado.")
            else:
                st.subheader("Sesgos detectados")
                for d in detections:
                    sev_color = {"alto":"red","medio":"orange","bajo":"blue"}.get(d.get("severity","bajo"),"blue")
                    st.markdown(f":{sev_color}[**{d['bias']}**] - {d['evidence']}")
                st.markdown("---")
                with st.spinner("Generando consejo personalizado..."):
                    advice = coach_ai.explain_with_ai(detections)
                st.subheader("Consejo del coach")
                st.write(advice)


# ==================================================================
# PAGINA: INFORME MENSUAL IA
# ==================================================================
elif page == "Informe mensual":
    st.title("Informe mensual con IA")
    st.caption("Genera un informe escrito personalizado de tu cartera con analisis IA.")
    if not ai_provider.is_configured():
        st.warning("Configura tu API key en 'Configuracion IA' para generar informes.")
    else:
        if st.button("Generar informe del mes", type="primary"):
            transactions_df = logbook.list_transactions()
            pv = logbook.portfolio_value(prices.iloc[-1], eur_usd)
            if pv is None:
                st.warning("Necesitas registrar al menos una operacion en 'Mi cartera real' antes.")
            else:
                positions_df, summary = pv
                # Metricas de la cartera real
                cols = [t for t in positions_df["ticker"] if t in prices.columns]
                metrics_dict = {}
                if cols:
                    weights = {r["ticker"]:r["weight_actual"] for _,r in positions_df.iterrows()
                              if r["ticker"] in cols}
                    rets = pd.Series(portfolio_returns_series(weights, prices))
                    if len(rets) > 30:
                        m = full_metrics(rets, benchmark_series(prices), rf=0.02)
                        for k,v in m.items():
                            if k in ("Retorno anual","Volatilidad","Max Drawdown","CVaR 95% (ES)"):
                                metrics_dict[k] = fmt_pct(v)
                            elif isinstance(v, (int,float)):
                                metrics_dict[k] = fmt_num(v,2)
                rec_map_d = {"Conservador":"min_vol","Moderado":"hrp","Crecimiento":"black_litterman","Agresivo":"max_sharpe_lw"}
                target_key = rec_map_d.get(st.session_state.risk_profile, "hrp")

                with st.spinner("Generando informe (puede tardar 10-30s)..."):
                    rep = report_ai.generate_monthly_report(
                        summary, positions_df, transactions_df, metrics_dict,
                        st.session_state.risk_profile, st.session_state.horizon,
                        PORTFOLIO_LABELS.get(target_key, "Recomendada"))
                st.success(f"Informe generado por {rep.get('provider','IA')}")
                st.markdown("---")
                st.markdown(rep["narrative"])
                st.download_button(
                    "Descargar informe HTML",
                    data=rep["html"],
                    file_name=f"informe_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                )


# ==================================================================
# PAGINA: CONFIGURACION IA (API keys)
# ==================================================================
elif page == "Configuracion IA":
    st.title("Configuracion IA")
    st.caption("Conecta proveedores de IA gratuitos. Solo necesitas UNA, las demas son de respaldo.")

    st.markdown("""
    ### Proveedores recomendados (todos GRATIS para uso personal)

    **1. Groq (recomendado, primario):** modelos Llama 3.3 70B, muy rapido (1-2s respuesta).
    Cuota: 14.400 requests/dia.
    - Registro: https://console.groq.com/
    - Ve a "API Keys" -> "Create API Key" -> copia el valor

    **2. Google Gemini (respaldo):** modelo Gemini 1.5 Flash.
    Cuota: 1500 requests/dia.
    - Registro: https://aistudio.google.com/apikey
    - Pulsa "Create API key" -> copia

    **3. Cerebras (segundo respaldo):** Llama 3.1 70B.
    Cuota: 14.400 requests/dia.
    - Registro: https://cloud.cerebras.ai/
    - Ve a API Keys -> Create -> copia
    """)
    st.markdown("---")

    with st.form("ai_keys"):
        groq_key = st.text_input("Groq API Key (recomendada)",
            value=st.session_state.get("GROQ_API_KEY",""), type="password")
        gemini_key = st.text_input("Gemini API Key (opcional)",
            value=st.session_state.get("GEMINI_API_KEY",""), type="password")
        cerebras_key = st.text_input("Cerebras API Key (opcional)",
            value=st.session_state.get("CEREBRAS_API_KEY",""), type="password")
        save = st.form_submit_button("Guardar claves", type="primary")
    if save:
        st.session_state.GROQ_API_KEY = groq_key
        st.session_state.GEMINI_API_KEY = gemini_key
        st.session_state.CEREBRAS_API_KEY = cerebras_key
        st.success("Claves guardadas (en sesion). Para persistir en cloud, anadelas a Streamlit Secrets.")

    st.markdown("---")
    st.subheader("Test de conexion")
    if ai_provider.is_configured():
        if st.button("Probar IA"):
            with st.spinner("Llamando a IA..."):
                res = ai_provider.ask("Di hola en una frase corta.",
                                       system="Eres asistente financiero conciso.",
                                       max_tokens=50)
            if res.get("text"):
                st.success(f"OK ({res['provider']}): {res['text']}")
            else:
                st.error(f"Error: {res.get('error')}")
    else:
        st.info("Anade al menos una clave API y guarda para activar.")


# PAGINA: CONFIGURACION
# ==================================================================
elif page == "Configuracion":
    st.title("Configuracion y diagnostico")
    st.subheader("Estado del sistema")
    flag_now = read_flag()
    last = flag_now.get("ts") if flag_now else "Nunca"
    st.json({
        "Universo total": len(ASSET_LIST),
        "Universo filtrado actual": prices.shape[1],
        "Datos hasta": str(prices.index[-1].date()),
        "Scheduler activo": st.session_state.scheduler_started,
        "Ultima actualizacion auto": last,
        "Proximo refresco programado": next_market_close().strftime("%Y-%m-%d %H:%M %Z"),
        "EUR/USD": eur_usd,
        "Perfil actual": st.session_state.risk_profile,
        "Horizonte": st.session_state.horizon,
    })
    st.markdown("---")
    st.subheader("Persistencia y webhook")
    backend_info = logbook.get_backend_info()
    st.write(f"**Backend logbook actual:** {backend_info}")
    if supabase_db.is_configured():
        st.success("Supabase conectado: tu cartera persiste 24/7 en cloud.")
    else:
        st.info("Supabase NO configurado. En cloud el logbook se borra al dormir la app. "
                "Configura SUPABASE_URL y SUPABASE_ANON_KEY en Streamlit Secrets.")
    try:
        cron_token = st.secrets.get("CRON_TOKEN", "")
        if cron_token:
            st.success("CRON_TOKEN configurado: webhook listo para alertas automaticas.")
            try:
                cur_url = "tu-app.streamlit.app"
                st.caption(f"URL del webhook: https://{cur_url}/?action=check_drift&token=XXXXX (sustituye XXXXX)")
            except Exception:
                pass
        else:
            st.info("CRON_TOKEN no configurado. Anade un token aleatorio en Secrets para activar webhook de cron-job.org.")
    except Exception:
        pass
    st.markdown("---")
    st.subheader("Brokers - comisiones")
    bk = pd.DataFrame.from_dict(BROKER_FEES, orient="index")
    st.dataframe(bk, use_container_width=True)
    if st.button("Forzar refresco de datos"):
        st.cache_data.clear()
        st.success("Cache limpiado. Recarga la pagina.")
    st.markdown("---")
    st.subheader("Aviso legal")
    st.warning(DISCLAIMER)
