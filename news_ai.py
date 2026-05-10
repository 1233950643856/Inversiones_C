"""Analiza noticias financieras gratuitas (RSS de Yahoo Finance) y las resume con LLM."""
import requests
import xml.etree.ElementTree as ET
import streamlit as st
from datetime import datetime
import ai_provider

@st.cache_data(ttl=10800, show_spinner=False)
def fetch_yahoo_rss(ticker: str, n: int = 8):
    """Devuelve lista de noticias recientes desde el RSS publico de Yahoo Finance."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=10,
            headers={"User-Agent":"Mozilla/5.0 InvPRO"})
        if r.status_code != 200: return []
        root = ET.fromstring(r.content)
        items = []
        for it in root.findall(".//item")[:n]:
            items.append({
                "title": it.findtext("title", "").strip(),
                "link":  it.findtext("link", "").strip(),
                "date":  it.findtext("pubDate", "").strip(),
                "summary": (it.findtext("description", "") or "").strip()[:500],
            })
        return items
    except Exception:
        return []

@st.cache_data(ttl=10800, show_spinner=False)
def summarize_for_ticker(ticker: str, asset_name: str = ""):
    """Devuelve resumen de noticias relevantes para inversion en 3-5 bullets."""
    news = fetch_yahoo_rss(ticker, n=8)
    if not news:
        return {"summary":"No hay noticias recientes disponibles.", "raw":[]}
    headlines = "\n".join([f"- {n['title']}" for n in news[:8]])
    system = ("Eres analista financiero. Resumes noticias para una inversora particular en "
              "3-5 bullets en espanol, priorizando lo relevante para decisiones de inversion. "
              "Ignora noticias triviales o promocionales. Si una noticia es muy importante, "
              "destacalo con [IMPORTANTE]. Si una es bullish o bearish, indicalo.")
    prompt = (f"Activo: {ticker} ({asset_name}). Titulares recientes:\n\n{headlines}\n\n"
              f"Dame los 3-5 puntos mas relevantes para inversion.")
    res = ai_provider.ask(prompt, system, max_tokens=400)
    return {"summary": res.get("text") or "Resumen no disponible.", "raw": news}

@st.cache_data(ttl=21600, show_spinner=False)
def daily_market_insight(top_movers: list, portfolio_summary: dict = None):
    """Insight diario para el Dashboard. top_movers: [(ticker, ret_today)]."""
    movers_str = "\n".join([f"- {t}: {r*100:+.2f}%" for t, r in top_movers[:10]])
    pf_str = ""
    if portfolio_summary:
        pf_str = (f"\n\nMi cartera real: valor {portfolio_summary.get('total_value',0):.0f} EUR, "
                  f"P/L {portfolio_summary.get('total_pl_pct',0)*100:+.2f}%.")
    system = ("Eres asesor financiero conciso. Das un insight diario en 2-3 frases sobre el "
              "mercado, en espanol, sin floritura. Mencionas riesgo solo si lo ves. No haces "
              "predicciones, solo observaciones contextualizadas. Maximo 60 palabras.")
    prompt = (f"Movimientos del dia ({datetime.now().strftime('%Y-%m-%d')}):\n{movers_str}{pf_str}"
              f"\n\nDame insight breve y util.")
    res = ai_provider.ask(prompt, system, max_tokens=200)
    return res.get("text") or "Insight no disponible."
