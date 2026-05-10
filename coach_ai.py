"""Coach de sesgos cognitivos: detecta patrones psicologicos y avisa al usuario.

Sesgos cubiertos:
  - Loss aversion (aversion a perdidas): mirar la app obsesivamente cuando cae el mercado
  - Disposition effect: vender ganadores demasiado pronto, mantener perdedores
  - Recency bias: dar excesivo peso a movimientos recientes
  - Overtrading: operar demasiado para presupuesto pequeno
  - Status quo bias: no aportar segun el plan establecido
  - Anchoring: fijacion en precio de compra
  - Confirmation bias: solo mirar metricas que confirman tu decision
  - Herding: seguir al rebano (entrar en un activo despues de subida fuerte)
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import ai_provider

USAGE_LOG = Path(__file__).parent / "usage_log.json"

def log_event(event_type: str, meta: dict = None):
    """Registra evento de uso para analisis posterior de sesgos."""
    try:
        log = []
        if USAGE_LOG.exists():
            log = json.loads(USAGE_LOG.read_text())
        log.append({"ts": datetime.now().isoformat(),
                    "event": event_type, "meta": meta or {}})
        log = log[-500:]
        USAGE_LOG.write_text(json.dumps(log, indent=2))
    except Exception:
        pass

def _load_log():
    if not USAGE_LOG.exists(): return []
    try: return json.loads(USAGE_LOG.read_text())
    except: return []

# === Detectores heuristicos ===
def detect_loss_aversion(market_dropped_today: bool = False):
    """Detecta si la usuaria ha abierto la app >5 veces hoy con caida del mercado."""
    log = _load_log()
    today = datetime.now().date()
    visits_today = sum(1 for e in log
                       if datetime.fromisoformat(e["ts"]).date() == today
                       and e["event"] == "page_view")
    if market_dropped_today and visits_today >= 5:
        return {
            "bias":"Loss aversion (aversion a perdidas)",
            "evidence":f"Has abierto la app {visits_today} veces hoy con el mercado en caida.",
            "severity":"alto",
        }
    return None

def detect_disposition_effect(positions_df):
    """Detecta intencion de vender ganador y mantener perdedor.
    positions_df: cartera real con cost_basis y price_now."""
    if positions_df is None or positions_df.empty: return None
    log = _load_log()
    recent = [e for e in log if e["event"] == "view_position_detail"
              and (datetime.now() - datetime.fromisoformat(e["ts"])).days < 7]
    if not recent: return None
    viewed_tickers = [e["meta"].get("ticker") for e in recent]
    df = positions_df.copy()
    df["pl_pct"] = (df["price_now_eur"] / df["cost_basis_eur"]) - 1
    winners_viewed = df[(df["ticker"].isin(viewed_tickers)) & (df["pl_pct"] > 0.10)]
    losers_held = df[(df["pl_pct"] < -0.10) & (~df["ticker"].isin(viewed_tickers))]
    if len(winners_viewed) >= 1 and len(losers_held) >= 1:
        return {
            "bias":"Disposition effect (vender ganadores, mantener perdedores)",
            "evidence":f"Has revisado {len(winners_viewed)} posiciones ganadoras esta semana, "
                       f"e ignoras {len(losers_held)} con perdidas.",
            "severity":"medio",
        }
    return None

def detect_overtrading(transactions_df, budget):
    """Detecta operar demasiado para el presupuesto."""
    if transactions_df is None or transactions_df.empty: return None
    last_30d = pd.to_datetime(transactions_df["date"]) >= (datetime.now() - timedelta(days=30))
    n_30 = last_30d.sum()
    threshold = max(2, int(budget / 200))  # ~1 op por 200 EUR/mes
    if n_30 > threshold * 3:
        return {
            "bias":"Overtrading",
            "evidence":f"{n_30} operaciones en 30 dias con presupuesto {budget} EUR. "
                       f"Las comisiones erosionan retorno.",
            "severity":"alto",
        }
    return None

def detect_status_quo(transactions_df, expected_monthly_contribution=None):
    """Detecta no aportar segun plan."""
    if expected_monthly_contribution is None: return None
    if transactions_df is None or transactions_df.empty:
        return {
            "bias":"Status quo bias",
            "evidence":f"Llevas sin aportar y planeabas {expected_monthly_contribution} EUR/mes. "
                       f"La inflacion erosiona el dinero parado.",
            "severity":"medio",
        }
    last_60d = pd.to_datetime(transactions_df["date"]) >= (datetime.now() - timedelta(days=60))
    if last_60d.sum() == 0:
        return {
            "bias":"Status quo bias",
            "evidence":"Llevas mas de 2 meses sin aportar al plan establecido.",
            "severity":"medio",
        }
    return None

def detect_recency_bias(returns_series):
    """Detecta si la cartera real esta sobrepesada en activos top recientes."""
    if returns_series is None or len(returns_series) < 30: return None
    last_5d = returns_series.tail(5).mean()
    last_60d = returns_series.tail(60).mean()
    if last_5d > 3 * last_60d and last_5d > 0.005:
        return {
            "bias":"Recency bias",
            "evidence":"Tu cartera tiene rendimiento reciente muy por encima del promedio. "
                       "Cuidado con extrapolar 5 dias buenos a 5 anos.",
            "severity":"bajo",
        }
    return None

def detect_anchoring(positions_df):
    """Detecta fijacion en precios de compra (estancamiento mental)."""
    log = _load_log()
    cost_views = sum(1 for e in log if e["event"] == "view_cost_basis"
                     and (datetime.now() - datetime.fromisoformat(e["ts"])).days < 14)
    if cost_views >= 8:
        return {
            "bias":"Anchoring (precio de compra)",
            "evidence":"Estas mirando muchas veces el precio al que compraste. "
                       "El precio de compra no determina el valor futuro.",
            "severity":"bajo",
        }
    return None

def all_detections(positions_df, transactions_df, budget,
                   market_dropped_today=False, returns_series=None,
                   expected_monthly_contribution=None):
    """Ejecuta todos los detectores y devuelve lista de sesgos detectados."""
    out = []
    for fn, args in [
        (detect_loss_aversion, [market_dropped_today]),
        (detect_disposition_effect, [positions_df]),
        (detect_overtrading, [transactions_df, budget]),
        (detect_status_quo, [transactions_df, expected_monthly_contribution]),
        (detect_recency_bias, [returns_series]),
        (detect_anchoring, [positions_df]),
    ]:
        try:
            r = fn(*args)
            if r: out.append(r)
        except Exception:
            pass
    return out

def explain_with_ai(detections):
    """Usa el LLM para humanizar las detecciones en un mensaje cercano."""
    if not detections:
        return "Tu comportamiento de inversion parece equilibrado esta semana. Bien."
    bias_summary = "\n".join([f"- {d['bias']}: {d['evidence']}" for d in detections])
    system = ("Eres un coach financiero conductual cercano y empatico. Hablas en espanol "
              "neutro, sin tecnicismos innecesarios, sin moralismos. Tu objetivo es que la "
              "inversora reconozca patrones psicologicos que estan perjudicando su rendimiento. "
              "Cita brevemente la literatura academica donde aplique (Kahneman, Thaler, Shefrin) "
              "pero sin ser pedante. Maximo 3 parrafos. No repitas literalmente la lista, "
              "integra los puntos en un consejo coherente.")
    prompt = (f"He detectado estos patrones en mi cartera y uso de la app:\n\n"
              f"{bias_summary}\n\n"
              f"Dame un mensaje breve y constructivo. Si hay varios sesgos, prioriza el mas grave.")
    res = ai_provider.ask(prompt, system, max_tokens=500)
    return res.get("text") or "No se pudo generar respuesta del coach (revisa Configuracion IA)."
