"""Universo, brokers, perfiles de riesgo y restricciones (v3 LITE - 35 activos)."""
from datetime import datetime
import pytz

ASSETS = {
    # === ETFs RENTA VARIABLE CORE (5) ===
    "VWCE.DE": {"type":"ETF_EQ","sector":"Diversificado","region":"Mundial","name":"Vanguard FTSE All-World"},
    "VOO":     {"type":"ETF_EQ","sector":"Diversificado","region":"EEUU","name":"S&P 500"},
    "QQQ":     {"type":"ETF_EQ","sector":"Tecnologia","region":"EEUU","name":"Nasdaq 100"},
    "IWDA.AS": {"type":"ETF_EQ","sector":"Diversificado","region":"Mundial","name":"MSCI World"},
    "EXSA.DE": {"type":"ETF_EQ","sector":"Diversificado","region":"Europa","name":"STOXX Europe 600"},
    # === ETFs SECTORIALES (3) ===
    "XLK":     {"type":"ETF_EQ","sector":"Tecnologia","region":"EEUU","name":"Technology Select"},
    "XLV":     {"type":"ETF_EQ","sector":"Salud","region":"EEUU","name":"Health Care Select"},
    "XLF":     {"type":"ETF_EQ","sector":"Financiero","region":"EEUU","name":"Financial Select"},
    # === ETFs RENTA FIJA (4) ===
    "AGGH.MI": {"type":"ETF_BOND","sector":"Bonos Globales","region":"Mundial","name":"iShares Core Global Aggregate","duration":7.0,"rating":"Mixto"},
    "IB01.L":  {"type":"ETF_BOND","sector":"Letras Tesoro","region":"EEUU","name":"iShares 0-3m Treasury","duration":0.2,"rating":"AAA"},
    "BND":     {"type":"ETF_BOND","sector":"Bonos Globales","region":"EEUU","name":"Vanguard Total Bond","duration":6.5,"rating":"Mixto"},
    "TLT":     {"type":"ETF_BOND","sector":"Bonos Gobierno Largo","region":"EEUU","name":"iShares 20+ Treasury","duration":17.0,"rating":"AAA"},
    # === ALTERNATIVOS (1) ===
    "GLD":     {"type":"COMMODITY","sector":"Oro","region":"Global","name":"SPDR Gold Trust"},
    # === MEGACAPS USA (12) ===
    "AAPL":    {"type":"STOCK","sector":"Tecnologia","region":"EEUU","name":"Apple"},
    "MSFT":    {"type":"STOCK","sector":"Tecnologia","region":"EEUU","name":"Microsoft"},
    "GOOGL":   {"type":"STOCK","sector":"Tecnologia","region":"EEUU","name":"Alphabet"},
    "AMZN":    {"type":"STOCK","sector":"Consumo","region":"EEUU","name":"Amazon"},
    "META":    {"type":"STOCK","sector":"Tecnologia","region":"EEUU","name":"Meta"},
    "NVDA":    {"type":"STOCK","sector":"Tecnologia","region":"EEUU","name":"Nvidia"},
    "BRK-B":   {"type":"STOCK","sector":"Financiero","region":"EEUU","name":"Berkshire Hathaway B"},
    "JPM":     {"type":"STOCK","sector":"Financiero","region":"EEUU","name":"JPMorgan Chase"},
    "JNJ":     {"type":"STOCK","sector":"Salud","region":"EEUU","name":"Johnson & Johnson"},
    "LLY":     {"type":"STOCK","sector":"Salud","region":"EEUU","name":"Eli Lilly"},
    "KO":      {"type":"STOCK","sector":"Consumo Defensivo","region":"EEUU","name":"Coca-Cola"},
    "WMT":     {"type":"STOCK","sector":"Consumo Defensivo","region":"EEUU","name":"Walmart"},
    # === EUROPA (5) ===
    "ASML.AS": {"type":"STOCK","sector":"Tecnologia","region":"Europa","name":"ASML"},
    "SAP.DE":  {"type":"STOCK","sector":"Tecnologia","region":"Europa","name":"SAP"},
    "NESN.SW": {"type":"STOCK","sector":"Consumo Defensivo","region":"Europa","name":"Nestle"},
    "NOVO-B.CO":{"type":"STOCK","sector":"Salud","region":"Europa","name":"Novo Nordisk"},
    "MC.PA":   {"type":"STOCK","sector":"Consumo","region":"Europa","name":"LVMH"},
    # === ESPANA (4) ===
    "ITX.MC":  {"type":"STOCK","sector":"Consumo","region":"Espana","name":"Inditex"},
    "IBE.MC":  {"type":"STOCK","sector":"Utilities","region":"Espana","name":"Iberdrola"},
    "SAN.MC":  {"type":"STOCK","sector":"Financiero","region":"Espana","name":"Banco Santander"},
    "BBVA.MC": {"type":"STOCK","sector":"Financiero","region":"Espana","name":"BBVA"},
}

ASSET_LIST = list(ASSETS.keys())

BROKER_FEES = {
    "Trade Republic": {"fixed":1.0,"pct":0.0,"min":1.0,"fractional":True},
    "Interactive Brokers":{"fixed":0.0,"pct":0.0005,"min":1.0,"fractional":True},
    "eToro":         {"fixed":0.0,"pct":0.0,"min":0.0,"fractional":True,"spread":0.0009},
    "XTB":           {"fixed":0.0,"pct":0.0,"min":0.0,"fractional":True},
    "Degiro":        {"fixed":1.0,"pct":0.0003,"min":1.0,"fractional":False},
    "ING Direct":    {"fixed":8.0,"pct":0.002,"min":8.0,"fractional":False},
    "MyInvestor":    {"fixed":3.0,"pct":0.001,"min":3.0,"fractional":False},
}

RISK_PROFILES = {
    "Conservador": {"max_equity":0.40,"min_bonds":0.45,"max_single_asset":0.15,
        "max_sector":0.25,"max_drawdown_target":0.12,
        "description":"Prioriza preservacion de capital. Acepta retornos modestos."},
    "Moderado": {"max_equity":0.65,"min_bonds":0.20,"max_single_asset":0.20,
        "max_sector":0.30,"max_drawdown_target":0.20,
        "description":"Equilibrio entre crecimiento y estabilidad."},
    "Crecimiento": {"max_equity":0.85,"min_bonds":0.05,"max_single_asset":0.25,
        "max_sector":0.35,"max_drawdown_target":0.30,
        "description":"Prioriza crecimiento aceptando volatilidad significativa."},
    "Agresivo": {"max_equity":1.00,"min_bonds":0.00,"max_single_asset":0.30,
        "max_sector":0.45,"max_drawdown_target":0.40,
        "description":"Maximizar retorno a largo plazo. Tolerancia alta al drawdown."},
}

HORIZON_ADJUSTMENT = {
    "Corto (<3 anos)":     {"shift_to_bonds":+0.15, "max_eq_cap":0.40},
    "Medio (3-7 anos)":    {"shift_to_bonds":0.00,  "max_eq_cap":0.75},
    "Largo (7-15 anos)":   {"shift_to_bonds":-0.10, "max_eq_cap":0.95},
    "Muy largo (>15 anos)":{"shift_to_bonds":-0.20, "max_eq_cap":1.00},
}

DEFAULT_PREFS = {
    "budget": 500.0, "currency": "EUR", "broker": "Trade Republic",
    "fractional_shares": True, "max_share_price": 300.0,
    "risk_profile": "Moderado", "horizon": "Largo (7-15 anos)",
    "tax_rate": 0.21, "rebalance_threshold": 0.05, "slippage_bps": 5.0,
}

PORTFOLIO_NAMES = [
    "max_sharpe_lw", "min_cvar", "hrp", "black_litterman",
    "min_vol", "balanced_60_40", "income", "all_weather"
]

PORTFOLIO_LABELS = {
    "max_sharpe_lw":   "Max Sharpe (Ledoit-Wolf)",
    "min_cvar":        "Min CVaR (Expected Shortfall)",
    "hrp":             "Hierarchical Risk Parity",
    "black_litterman": "Black-Litterman (vistas ML)",
    "min_vol":         "Minima Volatilidad",
    "balanced_60_40":  "Balanced 60/40",
    "income":          "Renta (dividendos + cupones)",
    "all_weather":     "All-Weather (Dalio)",
}

STRESS_SCENARIOS = {
    "COVID Crash (Feb-Mar 2020)":      {"start":"2020-02-19","end":"2020-03-23"},
    "Crisis Financiera (2008)":         {"start":"2008-09-15","end":"2009-03-09"},
    "Burbuja Punto.com (2000-2002)":    {"start":"2000-03-10","end":"2002-10-09"},
    "Crisis del Euro (2011)":           {"start":"2011-07-01","end":"2011-10-04"},
    "Inflacion 2022 (subida tipos)":    {"start":"2022-01-03","end":"2022-10-12"},
    "Crash 2018 (Q4)":                  {"start":"2018-09-20","end":"2018-12-24"},
}

CET = pytz.timezone("Europe/Madrid")
def now_cet():
    return datetime.now(CET)

DISCLAIMER = ("AVISO: Esta herramienta es exclusivamente informativa y educativa. "
              "NO constituye asesoramiento financiero, legal ni fiscal. "
              "Las inversiones implican riesgo de perdida. Rentabilidades pasadas "
              "no garantizan rentabilidades futuras. Consulte con un asesor "
              "financiero registrado antes de tomar decisiones de inversion.")
