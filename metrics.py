"""Metricas profesionales: Sharpe, Sortino, Calmar, CVaR, Omega, Ulcer."""
import numpy as np
import pandas as pd

ANNUAL_FACTOR = 252

def annual_return(returns):
    r = returns.dropna()
    if len(r) == 0: return 0.0
    return float((1 + r.mean()) ** ANNUAL_FACTOR - 1)

def annual_vol(returns):
    r = returns.dropna()
    if len(r) == 0: return 0.0
    return float(r.std() * np.sqrt(ANNUAL_FACTOR))

def sharpe(returns, rf=0.02):
    vol = annual_vol(returns)
    if vol < 1e-9: return 0.0
    return (annual_return(returns) - rf) / vol

def sortino(returns, rf=0.02, target=0.0):
    r = returns.dropna()
    downside = r[r < target]
    if len(downside) == 0: return 0.0
    dd = downside.std() * np.sqrt(ANNUAL_FACTOR)
    if dd < 1e-9: return 0.0
    return (annual_return(returns) - rf) / dd

def max_drawdown(returns):
    r = returns.dropna()
    if len(r) == 0: return 0.0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())

def max_drawdown_duration(returns):
    """Dias que tarda la cartera en recuperarse del peor drawdown."""
    r = returns.dropna()
    if len(r) == 0: return 0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    in_dd = dd < 0
    if not in_dd.any(): return 0
    longest = curr = 0
    for v in in_dd:
        if v:
            curr += 1; longest = max(longest, curr)
        else:
            curr = 0
    return int(longest)

def calmar(returns):
    mdd = abs(max_drawdown(returns))
    if mdd < 1e-9: return 0.0
    return annual_return(returns) / mdd

def value_at_risk(returns, alpha=0.05):
    r = returns.dropna()
    if len(r) == 0: return 0.0
    return float(np.quantile(r, alpha))

def cvar(returns, alpha=0.05):
    """Conditional VaR / Expected Shortfall: media de la cola peor del alpha%."""
    r = returns.dropna()
    if len(r) == 0: return 0.0
    threshold = np.quantile(r, alpha)
    tail = r[r <= threshold]
    if len(tail) == 0: return float(threshold)
    return float(tail.mean())

def omega_ratio(returns, threshold=0.0):
    r = returns.dropna() - threshold
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg < 1e-9: return float("inf")
    return float(pos / neg)

def ulcer_index(returns):
    """Mide la profundidad y duracion de los drawdowns conjuntamente."""
    r = returns.dropna()
    if len(r) == 0: return 0.0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd_pct = ((cum - peak) / peak) * 100
    return float(np.sqrt((dd_pct ** 2).mean()))

def information_ratio(returns, benchmark_returns, rf=0.0):
    """Active return / tracking error."""
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(df) == 0: return 0.0
    active = df.iloc[:,0] - df.iloc[:,1]
    te = active.std() * np.sqrt(ANNUAL_FACTOR)
    if te < 1e-9: return 0.0
    return float((active.mean() * ANNUAL_FACTOR - rf) / te)

def beta_to(returns, benchmark_returns):
    df = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(df) < 30: return 1.0
    cov = np.cov(df.iloc[:,0], df.iloc[:,1])[0,1]
    var_b = np.var(df.iloc[:,1])
    if var_b < 1e-9: return 1.0
    return float(cov / var_b)

def positive_months_pct(returns):
    r = returns.dropna()
    if len(r) == 0: return 0.0
    monthly = (1 + r).resample("ME").prod() - 1 if hasattr(r.index, "to_period") else r
    if len(monthly) == 0: return 0.0
    return float((monthly > 0).mean())

def herfindahl(weights):
    """Concentracion de la cartera (1/H = num efectivo de activos)."""
    w = np.array(list(weights.values()) if isinstance(weights, dict) else weights)
    return float((w ** 2).sum())

def effective_n(weights):
    h = herfindahl(weights)
    return 1 / h if h > 0 else 0

def sector_concentration(weights, asset_meta):
    """Devuelve dict sector -> peso total."""
    sec = {}
    for k, w in weights.items():
        s = asset_meta.get(k, {}).get("sector", "Otros")
        sec[s] = sec.get(s, 0) + w
    return sec

def expected_dividend_income(weights, fundamentals, budget):
    """Estima ingreso anual por dividendos basado en yield reportado."""
    income = 0.0
    for k, w in weights.items():
        dy = fundamentals.loc[k, "div_yield"] if k in fundamentals.index else None
        if dy and dy > 0:
            income += w * float(dy) * budget
    return income

def full_metrics(returns, benchmark_returns=None, rf=0.02):
    """Devuelve todas las metricas en un dict listo para mostrar."""
    out = {
        "Retorno anual": annual_return(returns),
        "Volatilidad": annual_vol(returns),
        "Sharpe": sharpe(returns, rf),
        "Sortino": sortino(returns, rf),
        "Calmar": calmar(returns),
        "Max Drawdown": max_drawdown(returns),
        "DD duracion (dias)": max_drawdown_duration(returns),
        "VaR 95%": value_at_risk(returns, 0.05),
        "CVaR 95% (ES)": cvar(returns, 0.05),
        "VaR 99%": value_at_risk(returns, 0.01),
        "CVaR 99%": cvar(returns, 0.01),
        "Omega (>0)": omega_ratio(returns, 0.0),
        "Ulcer Index": ulcer_index(returns),
        "% meses positivos": positive_months_pct(returns),
    }
    if benchmark_returns is not None:
        out["Beta"] = beta_to(returns, benchmark_returns)
        out["Info Ratio"] = information_ratio(returns, benchmark_returns, rf)
    return out
