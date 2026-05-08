"""Backtest profesional: rebalanceo por umbral, slippage, fiscalidad, stress."""
import numpy as np
import pandas as pd

def backtest_threshold(prices, weights_target, threshold=0.05,
                       slippage_bps=5.0, tax_rate=0.21, transaction_cost=0.0):
    """
    Rebalancea solo cuando un activo se desvia >= threshold del peso objetivo.
    slippage_bps: 5 = 0.05% por operacion.
    tax_rate: 0.21 = 21% sobre plusvalias realizadas (Espana general).
    """
    P = prices.dropna(how="any").copy()
    cols = [c for c in weights_target if c in P.columns]
    if not cols: return None
    P = P[cols]
    w_t = np.array([weights_target[c] for c in cols])
    w_t = w_t / w_t.sum() if w_t.sum() > 0 else np.ones(len(cols))/len(cols)
    rets = P.pct_change().fillna(0)
    n_days = len(P)
    n_assets = len(cols)
    # Estado: pesos actuales y precio de coste medio por activo
    w = w_t.copy()
    cost_basis = P.iloc[0].values.copy()  # precio medio de adquisicion
    portfolio_value = 1.0
    history_value = [portfolio_value]
    n_rebalances = 0
    total_taxes = 0.0
    total_costs = 0.0
    for i in range(1, n_days):
        # Drift: pesos cambian con los retornos
        gross = w * (1 + rets.iloc[i].values)
        portfolio_value *= float(gross.sum())
        w = gross / gross.sum() if gross.sum() > 0 else w
        # Comprobar drift maximo
        drift = np.abs(w - w_t).max()
        if drift > threshold:
            # Calcular plusvalias realizadas en activos vendidos
            current_prices = P.iloc[i].values
            w_new = w_t.copy()
            delta_w = w_new - w
            for j in range(n_assets):
                if delta_w[j] < 0:  # vendemos parte
                    sold_value = -delta_w[j] * portfolio_value
                    avg_cost = cost_basis[j]
                    if avg_cost > 0:
                        gain = sold_value * (current_prices[j] / avg_cost - 1) / (current_prices[j]/avg_cost)
                        if gain > 0:
                            tax = gain * tax_rate
                            total_taxes += tax
                            portfolio_value -= tax
                elif delta_w[j] > 0:  # compramos mas, actualiza coste medio
                    bought_value = delta_w[j] * portfolio_value
                    old_units = w[j] * portfolio_value / current_prices[j] if current_prices[j]>0 else 0
                    new_units = bought_value / current_prices[j] if current_prices[j]>0 else 0
                    if old_units + new_units > 0:
                        cost_basis[j] = ((old_units * cost_basis[j]) +
                                         (new_units * current_prices[j])) / (old_units + new_units)
            # Slippage + comision
            turnover = float(np.abs(delta_w).sum()) / 2
            cost = (slippage_bps/10000.0 + transaction_cost) * turnover * portfolio_value
            portfolio_value -= cost
            total_costs += cost
            w = w_new.copy()
            n_rebalances += 1
        history_value.append(portfolio_value)
    series = pd.Series(history_value, index=P.index)
    rets_pf = series.pct_change().dropna()
    return {
        "value_series": series,
        "returns": rets_pf,
        "n_rebalances": n_rebalances,
        "total_taxes": total_taxes,
        "total_costs": total_costs,
        "final_value": float(series.iloc[-1]),
    }


def stress_test(prices, weights, scenario_dates):
    """Aplica los retornos historicos de la crisis a la cartera actual."""
    P = prices.dropna(how="any")
    start = pd.to_datetime(scenario_dates["start"])
    end = pd.to_datetime(scenario_dates["end"])
    sub = P[(P.index >= start) & (P.index <= end)]
    cols = [c for c in weights if c in sub.columns]
    if len(sub) < 2 or not cols:
        return None
    sub = sub[cols]
    rets = sub.pct_change().fillna(0)
    w = np.array([weights[c] for c in cols])
    w = w / w.sum() if w.sum() > 0 else np.ones(len(cols))/len(cols)
    port_returns = rets.values @ w
    cum = float(np.prod(1 + port_returns) - 1)
    mdd = 0.0
    cum_path = (1 + pd.Series(port_returns)).cumprod()
    peak = cum_path.cummax()
    dd = ((cum_path - peak) / peak).min()
    return {
        "total_return": cum,
        "max_drawdown": float(dd) if pd.notna(dd) else 0.0,
        "n_days": len(sub),
        "start": str(sub.index[0].date()),
        "end": str(sub.index[-1].date()),
    }
