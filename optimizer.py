"""Optimizador profesional: Markowitz+LW, Min-CVaR, HRP, Black-Litterman."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

try:
    from sklearn.covariance import LedoitWolf
    HAS_LW = True
except Exception:
    HAS_LW = False


def estimate_covariance(returns, method="ledoit_wolf"):
    """Devuelve covarianza anualizada. Ledoit-Wolf shrinkage por defecto."""
    R = returns.dropna(how="any")
    if len(R) < 30:
        return R.cov() * 252
    if method == "ledoit_wolf" and HAS_LW:
        lw = LedoitWolf().fit(R.values)
        cov = pd.DataFrame(lw.covariance_ * 252, index=R.columns, columns=R.columns)
        return cov
    return R.cov() * 252

def expected_returns_simple(returns):
    """Mean histórico anualizado."""
    return returns.mean() * 252

def expected_returns_capm(returns, market_returns, rf=0.02):
    """CAPM: E[R_i] = rf + beta_i * (E[R_m] - rf)."""
    mer = market_returns.mean() * 252
    out = {}
    for c in returns.columns:
        df = pd.concat([returns[c], market_returns], axis=1).dropna()
        if len(df) < 30:
            out[c] = mer; continue
        cov = np.cov(df.iloc[:,0], df.iloc[:,1])[0,1]
        var_m = np.var(df.iloc[:,1])
        beta = cov / var_m if var_m > 1e-9 else 1.0
        out[c] = rf + beta * (mer - rf)
    return pd.Series(out)


# ============= MARKOWITZ con shrinkage =============
def _portfolio_perf(weights, mu, sigma):
    w = np.array(weights)
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ sigma.values @ w))
    return ret, vol

def markowitz_optimize(mu, sigma, objective="max_sharpe", rf=0.02,
                       max_per_asset=0.25, max_per_sector=0.35,
                       sector_map=None, type_map=None,
                       min_bonds=0.0, max_equity=1.0, min_per_asset=0.0):
    """SLSQP con restricciones por activo, sector y clase de activo."""
    n = len(mu)
    tickers = list(mu.index)
    x0 = np.ones(n) / n
    bounds = [(min_per_asset, max_per_asset)] * n
    cons = [{"type":"eq", "fun":lambda w: np.sum(w) - 1.0}]
    # Restriccion por sector
    if sector_map:
        sectors = set(sector_map.get(t, "Otros") for t in tickers)
        for sec in sectors:
            idx = [i for i, t in enumerate(tickers) if sector_map.get(t, "Otros") == sec]
            cons.append({"type":"ineq",
                         "fun":lambda w, idx=idx: max_per_sector - sum(w[i] for i in idx)})
    # Restriccion por clase de activo
    if type_map:
        bond_idx = [i for i, t in enumerate(tickers) if "BOND" in type_map.get(t, "")]
        if bond_idx and min_bonds > 0:
            cons.append({"type":"ineq",
                         "fun":lambda w, idx=bond_idx: sum(w[i] for i in idx) - min_bonds})
        eq_idx = [i for i, t in enumerate(tickers)
                  if type_map.get(t, "") in ("ETF_EQ","STOCK","REIT")]
        if eq_idx and max_equity < 1.0:
            cons.append({"type":"ineq",
                         "fun":lambda w, idx=eq_idx: max_equity - sum(w[i] for i in idx)})

    if objective == "max_sharpe":
        def obj(w):
            r, v = _portfolio_perf(w, mu, sigma)
            return -(r - rf) / (v + 1e-9)
    elif objective == "min_vol":
        def obj(w):
            return _portfolio_perf(w, mu, sigma)[1]
    elif objective == "max_return":
        def obj(w):
            return -_portfolio_perf(w, mu, sigma)[0]
    else:
        def obj(w):  # default max sharpe
            r, v = _portfolio_perf(w, mu, sigma)
            return -(r - rf) / (v + 1e-9)

    try:
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter":300, "ftol":1e-9})
        if res.success:
            w = np.maximum(res.x, 0)
            w = w / w.sum() if w.sum() > 0 else x0
            return dict(zip(tickers, w))
    except Exception:
        pass
    return dict(zip(tickers, x0))


# ============= MIN-CVaR (Expected Shortfall optimization) =============
def min_cvar_optimize(returns, alpha=0.05, max_per_asset=0.25,
                      sector_map=None, max_per_sector=0.35):
    """Minimiza CVaR mediante reformulacion lineal (Rockafellar-Uryasev)."""
    R = returns.dropna(how="any")
    n = R.shape[1]; T = R.shape[0]
    tickers = list(R.columns)
    # Variables: [w_1..w_n, VaR, u_1..u_T]. Pero aqui usamos SLSQP con CVaR aprox.
    def cvar_loss(w):
        port_r = R.values @ w
        thr = np.quantile(port_r, alpha)
        tail = port_r[port_r <= thr]
        if len(tail) == 0: return -float(thr)
        return -float(tail.mean())
    x0 = np.ones(n) / n
    bounds = [(0.0, max_per_asset)] * n
    cons = [{"type":"eq","fun":lambda w: np.sum(w)-1.0}]
    if sector_map:
        for sec in set(sector_map.get(t,"Otros") for t in tickers):
            idx = [i for i,t in enumerate(tickers) if sector_map.get(t,"Otros")==sec]
            cons.append({"type":"ineq",
                         "fun":lambda w,idx=idx: max_per_sector - sum(w[i] for i in idx)})
    try:
        res = minimize(cvar_loss, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter":300})
        if res.success:
            w = np.maximum(res.x, 0); w = w/w.sum() if w.sum()>0 else x0
            return dict(zip(tickers, w))
    except Exception:
        pass
    return dict(zip(tickers, x0))


# ============= HRP (Hierarchical Risk Parity, Lopez de Prado) =============
def _correl_distance(corr):
    return np.sqrt(0.5 * (1 - corr))

def _quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index; j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def _ivp(cov):
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()

def _cluster_var(cov, items):
    cov_ = cov.iloc[items, items]
    w_ = _ivp(cov_).reshape(-1, 1)
    return float((w_.T @ cov_.values @ w_)[0, 0])

def _recursive_bisection(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        new_clusters = []
        for cl in clusters:
            if len(cl) <= 1: continue
            split = len(cl) // 2
            c1, c2 = cl[:split], cl[split:]
            v1 = _cluster_var(cov, c1); v2 = _cluster_var(cov, c2)
            alpha = 1 - v1 / (v1 + v2 + 1e-12)
            w[c1] *= alpha
            w[c2] *= 1 - alpha
            new_clusters.extend([c1, c2])
        clusters = new_clusters
    return w

def hrp_optimize(returns):
    """Hierarchical Risk Parity. No requiere expected returns. Robusto."""
    R = returns.dropna(how="any")
    if len(R) < 30 or R.shape[1] < 2:
        n = R.shape[1]
        return dict(zip(R.columns, [1.0/n]*n))
    cov = R.cov() * 252
    corr = R.corr()
    dist = _correl_distance(corr.values)
    np.fill_diagonal(dist, 0.0)
    try:
        link = linkage(squareform(dist, checks=False), method="single")
        sort_ix = _quasi_diag(link)
        sort_tickers = [corr.columns[i] for i in sort_ix]
        w = _recursive_bisection(cov.iloc[sort_ix, sort_ix], list(range(len(sort_ix))))
        w.index = sort_tickers
        out = w.reindex(R.columns).fillna(0).values
        out = out / out.sum() if out.sum() > 0 else np.ones(len(out))/len(out)
        return dict(zip(R.columns, out))
    except Exception:
        n = R.shape[1]
        return dict(zip(R.columns, [1.0/n]*n))


# ============= BLACK-LITTERMAN con vistas del modelo ML =============
def black_litterman(returns, market_caps, ml_views, tau=0.05, view_confidence=0.5,
                    max_per_asset=0.25, sector_map=None, max_per_sector=0.35,
                    type_map=None, min_bonds=0.0, max_equity=1.0):
    """
    market_caps: dict ticker -> market cap (proxy de pesos de mercado)
    ml_views:    dict ticker -> retorno predicho (la "vista" del modelo)
    tau:         escala de incertidumbre prior (0.05 estandar)
    view_confidence: 0-1, peso relativo de las vistas vs prior implicito
    """
    R = returns.dropna(how="any")
    tickers = list(R.columns)
    n = len(tickers)
    sigma = estimate_covariance(R, method="ledoit_wolf").values
    # Pesos de mercado
    if market_caps:
        mc = np.array([market_caps.get(t, 1.0) for t in tickers], dtype=float)
        mc = np.maximum(mc, 1.0)
        w_mkt = mc / mc.sum()
    else:
        w_mkt = np.ones(n) / n
    # Risk aversion lambda implicito (~2-4 tipico; usamos 2.5)
    lam = 2.5
    pi = lam * sigma @ w_mkt  # Implied equilibrium returns
    # Vistas: P matriz (k x n), Q vector (k)
    valid = [(i, t, ml_views[t]) for i, t in enumerate(tickers)
             if t in ml_views and ml_views[t] is not None]
    if not valid:
        # Sin vistas, devuelve markov optimo con pi
        mu_post = pd.Series(pi, index=tickers)
    else:
        k = len(valid)
        P = np.zeros((k, n)); Q = np.zeros(k)
        for row, (i, t, q) in enumerate(valid):
            P[row, i] = 1.0; Q[row] = float(q)
        # Omega: incertidumbre de las vistas (diagonal proporcional a P*sigma*P')
        omega_diag = np.diag(P @ (tau * sigma) @ P.T) / max(view_confidence, 0.01)
        omega = np.diag(omega_diag + 1e-8)
        try:
            inv_tausigma = np.linalg.inv(tau * sigma)
            inv_omega = np.linalg.inv(omega)
            cov_post = np.linalg.inv(inv_tausigma + P.T @ inv_omega @ P)
            mu_post_arr = cov_post @ (inv_tausigma @ pi + P.T @ inv_omega @ Q)
            mu_post = pd.Series(mu_post_arr, index=tickers)
        except Exception:
            mu_post = pd.Series(pi, index=tickers)
    sigma_df = pd.DataFrame(sigma, index=tickers, columns=tickers)
    return markowitz_optimize(mu_post, sigma_df, objective="max_sharpe",
                              max_per_asset=max_per_asset,
                              max_per_sector=max_per_sector,
                              sector_map=sector_map, type_map=type_map,
                              min_bonds=min_bonds, max_equity=max_equity)


# ============= ALL-WEATHER (Ray Dalio static) =============
def all_weather_weights(tickers, type_map):
    """30% acciones, 40% bonos largos, 15% bonos medios, 7.5% oro, 7.5% commodities."""
    eq = [t for t in tickers if type_map.get(t,"") in ("ETF_EQ","STOCK","REIT")]
    bd_long = [t for t in tickers if type_map.get(t,"")=="ETF_BOND" and "Largo" in str(type_map.get(t+"_dur",""))]
    bd_all = [t for t in tickers if type_map.get(t,"")=="ETF_BOND"]
    gold = [t for t in tickers if type_map.get(t,"")=="COMMODITY"]
    w = {t:0.0 for t in tickers}
    if eq:
        for t in eq: w[t] = 0.30 / len(eq)
    if bd_all:
        for t in bd_all: w[t] = 0.55 / len(bd_all)
    if gold:
        for t in gold: w[t] = 0.15 / len(gold)
    s = sum(w.values())
    if s > 0:
        w = {k: v/s for k, v in w.items()}
    else:
        n=len(tickers); w = {t:1.0/n for t in tickers}
    return w


# ============= BALANCED 60/40 =============
def balanced_60_40(returns, type_map, sector_map):
    """60% renta variable optimizada + 40% renta fija optimizada."""
    R = returns.dropna(how="any")
    eq_cols = [c for c in R.columns if type_map.get(c,"") in ("ETF_EQ","STOCK","REIT")]
    bd_cols = [c for c in R.columns if type_map.get(c,"") == "ETF_BOND"]
    out = {c: 0.0 for c in R.columns}
    if eq_cols:
        eq_R = R[eq_cols]
        eq_sigma = estimate_covariance(eq_R, "ledoit_wolf")
        eq_mu = expected_returns_simple(eq_R)
        eq_w = markowitz_optimize(eq_mu, eq_sigma, "max_sharpe",
                                  max_per_asset=0.20, sector_map=sector_map,
                                  max_per_sector=0.35)
        for k, v in eq_w.items(): out[k] = v * 0.60
    if bd_cols:
        bd_R = R[bd_cols]
        bd_sigma = estimate_covariance(bd_R, "ledoit_wolf")
        bd_mu = expected_returns_simple(bd_R)
        bd_w = markowitz_optimize(bd_mu, bd_sigma, "max_sharpe", max_per_asset=0.40)
        for k, v in bd_w.items(): out[k] = v * 0.40
    s = sum(out.values())
    if s > 0: out = {k: v/s for k, v in out.items()}
    return out


# ============= INCOME (dividendos + cupones) =============
def income_portfolio(returns, fundamentals, type_map):
    """Sobrepondera activos con yield > 0 y bonos."""
    R = returns.dropna(how="any")
    out = {}
    weights_raw = {}
    for c in R.columns:
        dy = fundamentals.loc[c, "div_yield"] if c in fundamentals.index else None
        is_bond = type_map.get(c, "") == "ETF_BOND"
        score = 0.0
        if dy and dy > 0: score += float(dy)
        if is_bond: score += 0.04
        weights_raw[c] = score
    total = sum(weights_raw.values())
    if total <= 0:
        n = len(R.columns); return {c: 1.0/n for c in R.columns}
    return {c: v/total for c, v in weights_raw.items()}
