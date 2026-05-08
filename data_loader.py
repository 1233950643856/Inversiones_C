"""Descarga de precios y fundamentales con cache y fallback robusto."""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from config import ASSET_LIST, ASSETS

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_prices(tickers, period="3y", interval="1d"):
    """Descarga precios ajustados. Maneja errores por ticker."""
    if isinstance(tickers, str):
        tickers = [tickers]
    valid = {}
    for t in tickers:
        try:
            df = yf.download(t, period=period, interval=interval,
                             progress=False, auto_adjust=True, threads=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            if "Close" in df.columns:
                valid[t] = df["Close"]
        except Exception:
            continue
    if not valid:
        return pd.DataFrame()
    out = pd.DataFrame(valid)
    out = out.dropna(how="all").ffill().bfill()
    return out

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fundamentals(ticker):
    """Saca fundamentales de yfinance.info. Devuelve dict aunque falle."""
    out = {"ticker":ticker,"price":None,"pe":None,"div_yield":None,
           "beta":None,"market_cap":None,"sector_yf":None}
    try:
        info = yf.Ticker(ticker).info or {}
        out["price"] = info.get("currentPrice") or info.get("regularMarketPrice")
        out["pe"] = info.get("trailingPE")
        dy = info.get("dividendYield")
        out["div_yield"] = float(dy) if dy is not None else None
        out["beta"] = info.get("beta")
        out["market_cap"] = info.get("marketCap")
        out["sector_yf"] = info.get("sector") or info.get("category")
    except Exception:
        pass
    return out

@st.cache_data(ttl=14400, show_spinner=False)
def fetch_eur_usd():
    """FX EUR/USD. 1.08 fallback razonable."""
    try:
        fx = yf.download("EURUSD=X", period="5d", progress=False, threads=False)
        if not fx.empty:
            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = [c[0] for c in fx.columns]
            return float(fx["Close"].iloc[-1])
    except Exception:
        pass
    return 1.08

@st.cache_data(ttl=14400, show_spinner=False)
def load_universe(period="3y"):
    """Carga precios de todo el universo y fundamentales basicos."""
    prices = fetch_prices(ASSET_LIST, period=period)
    fund_rows = []
    for t in prices.columns:
        f = fetch_fundamentals(t)
        f.update(ASSETS.get(t, {}))
        fund_rows.append(f)
    fund = pd.DataFrame(fund_rows).set_index("ticker") if fund_rows else pd.DataFrame()
    return prices, fund

def daily_returns(prices):
    return prices.pct_change().dropna(how="all")

def benchmark_series(prices, benchmark="VWCE.DE"):
    if benchmark in prices.columns:
        return prices[benchmark].pct_change().dropna()
    if "VOO" in prices.columns:
        return prices["VOO"].pct_change().dropna()
    return prices.iloc[:,0].pct_change().dropna()
