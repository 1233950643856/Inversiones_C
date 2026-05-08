"""Asignacion de presupuesto: convierte pesos a numero de acciones reales."""
import math
from config import BROKER_FEES

def allocate_budget(weights, prices, budget, broker="Trade Republic",
                    fractional_shares=True, currency="EUR", eur_usd_rate=1.08,
                    max_share_price=None):
    """Devuelve lista de operaciones concretas: ticker, n_shares, price, value, comm."""
    broker_info = BROKER_FEES.get(broker, BROKER_FEES["Trade Republic"])
    can_fractional = broker_info.get("fractional", False) and fractional_shares
    fix = broker_info.get("fixed", 1.0)
    pct = broker_info.get("pct", 0.0)
    minc = broker_info.get("min", 1.0)
    spread = broker_info.get("spread", 0.0)
    operations = []
    not_affordable = []
    invested = 0.0
    total_comm = 0.0
    for t, w in weights.items():
        if w <= 0.001: continue
        if t not in prices.index: continue
        price_native = float(prices.loc[t])
        # Heuristica: si termina en .DE/.AS/.MC/.PA/.MI/.SW/.L/.CO esta en EUR; si no USD
        suffixes = (".DE",".AS",".MC",".PA",".MI",".SW",".L",".CO")
        is_eur = any(t.endswith(s) for s in suffixes) or t.startswith("VWCE")
        price_local = price_native if is_eur else price_native * (1.0 / eur_usd_rate) * eur_usd_rate
        # En realidad Yahoo da precios en moneda nativa; conversion a EUR:
        price_eur = price_native if is_eur else (price_native / eur_usd_rate)
        if max_share_price and price_eur > max_share_price and not can_fractional:
            not_affordable.append({"ticker":t, "price":price_eur,
                                   "reason":f"Precio > {max_share_price} y broker no permite fracciones"})
            continue
        target = w * budget
        if can_fractional:
            n_shares = round(target / price_eur, 4)
        else:
            n_shares = math.floor(target / price_eur)
            if n_shares < 1:
                not_affordable.append({"ticker":t, "price":price_eur,
                                       "target_eur":target,
                                       "reason":"Insuficiente para 1 accion entera"})
                continue
        value = n_shares * price_eur
        # Comision
        comm = max(fix + value * pct, minc)
        if spread > 0: comm += value * spread
        operations.append({
            "ticker": t, "n_shares": n_shares, "price_eur": price_eur,
            "value_eur": value, "commission_eur": comm,
            "weight_target": w, "weight_actual": value / budget,
        })
        invested += value
        total_comm += comm
    cash_left = budget - invested - total_comm
    return {
        "operations": operations,
        "not_affordable": not_affordable,
        "invested": invested,
        "total_commission": total_comm,
        "cash_left": cash_left,
        "invested_pct": invested / budget if budget > 0 else 0,
    }
