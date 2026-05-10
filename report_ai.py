"""Generador de informe mensual personalizado en PDF (sin dependencias externas pesadas)."""
from datetime import datetime
from io import BytesIO
import streamlit as st
import ai_provider

def generate_monthly_report(portfolio_summary, positions_df, transactions_df,
                             metrics_dict, profile, horizon, target_label):
    """Devuelve dict con 'narrative' (texto LLM) y 'html' (HTML formateado)."""
    pos_summary = ""
    if positions_df is not None and not positions_df.empty:
        pos_lines = []
        for _, r in positions_df.head(10).iterrows():
            pos_lines.append(f"  - {r['ticker']}: {r['n_shares']:.2f} acciones, "
                             f"valor {r['value_eur']:.2f} EUR, "
                             f"P/L {r['pl_pct']*100:+.2f}%")
        pos_summary = "\n".join(pos_lines)
    metrics_summary = "\n".join([f"  - {k}: {v}" for k,v in (metrics_dict or {}).items()][:8])
    n_tx = len(transactions_df) if transactions_df is not None else 0

    system = ("Eres asesora financiera personal. Escribes un informe mensual amigable pero "
              "profesional, en espanol, dirigido a una inversora particular con presupuesto "
              "pequeno. Estructura: 1) Resumen ejecutivo, 2) Que ha funcionado, 3) Que "
              "preocupa, 4) Recomendaciones concretas para el proximo mes. No hagas "
              "predicciones de precios. Maximo 350 palabras totales.")
    prompt = (f"Informe mensual a fecha {datetime.now().strftime('%Y-%m-%d')}.\n"
              f"Perfil: {profile} | Horizonte: {horizon} | Cartera objetivo: {target_label}\n"
              f"Resumen cartera real:\n"
              f"  Valor: {portfolio_summary.get('total_value',0):.2f} EUR\n"
              f"  Invertido: {portfolio_summary.get('total_invested',0):.2f} EUR\n"
              f"  P/L: {portfolio_summary.get('total_pl',0):+.2f} EUR ({portfolio_summary.get('total_pl_pct',0)*100:+.2f}%)\n"
              f"Top posiciones:\n{pos_summary}\n"
              f"Operaciones del periodo: {n_tx}\n"
              f"Metricas:\n{metrics_summary}\n\n"
              f"Escribe el informe.")
    res = ai_provider.ask(prompt, system, max_tokens=900)
    narrative = res.get("text") or "No se pudo generar el informe (configura IA primero)."

    html = f"""
    <html><head><meta charset='utf-8'><style>
    body{{font-family:Arial,sans-serif;color:#222;max-width:800px;margin:20px auto;line-height:1.5}}
    h1{{color:#00876d;border-bottom:2px solid #00C9A7}}
    h2{{color:#444;margin-top:24px}}
    .summary{{background:#f4f4f4;padding:12px;border-radius:6px;margin:12px 0}}
    .footer{{color:#888;font-size:11px;margin-top:30px;border-top:1px solid #ddd;padding-top:8px}}
    </style></head><body>
    <h1>Inversiones PRO - Informe mensual</h1>
    <p><b>Fecha:</b> {datetime.now().strftime('%d de %B de %Y')} |
       <b>Perfil:</b> {profile} | <b>Horizonte:</b> {horizon}</p>
    <div class='summary'>
        <b>Cartera real:</b> {portfolio_summary.get('total_value',0):.2f} EUR
        ({portfolio_summary.get('total_pl_pct',0)*100:+.2f}% acumulado)<br>
        <b>Invertido:</b> {portfolio_summary.get('total_invested',0):.2f} EUR<br>
        <b>Operaciones del periodo:</b> {n_tx}
    </div>
    <h2>Analisis IA</h2>
    <div>{narrative.replace(chr(10), '<br>')}</div>
    <div class='footer'>
        Este informe es generado automaticamente con asistencia de IA. No constituye
        asesoramiento financiero. Para decisiones reales consulta con un asesor registrado.
    </div>
    </body></html>
    """
    return {"narrative": narrative, "html": html, "provider": res.get("provider")}
