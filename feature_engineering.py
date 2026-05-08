"""Genera features tecnicos para el modelo ML."""
import numpy as np
import pandas as pd

def _rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def _atr_proxy(series, n=14):
    """Volatilidad rolling como proxy de ATR (sin OHLC)."""
    return series.pct_change().rolling(n).std() * np.sqrt(252)

def build_features(price_series, target_horizon=21):
    """price_series: Series 1D. Devuelve DataFrame con features alineados."""
    df = pd.DataFrame(index=price_series.index)
    df["price"] = price_series
    df["ret_1d"] = price_series.pct_change()
    for w in (5, 10, 21, 63, 126, 252):
        df[f"ret_{w}d"] = price_series.pct_change(w)
        df[f"vol_{w}d"] = price_series.pct_change().rolling(w).std() * np.sqrt(252)
    df["sma_20"] = price_series.rolling(20).mean()
    df["sma_50"] = price_series.rolling(50).mean()
    df["sma_200"] = price_series.rolling(200).mean()
    df["price_to_sma20"] = price_series / df["sma_20"] - 1
    df["price_to_sma50"] = price_series / df["sma_50"] - 1
    df["price_to_sma200"] = price_series / df["sma_200"] - 1
    df["sma_20_50"] = df["sma_20"] / df["sma_50"] - 1
    df["rsi_14"] = _rsi(price_series, 14)
    df["atr_14"] = _atr_proxy(price_series, 14)
    df["mom_3m"] = price_series.pct_change(63)
    df["mom_6m"] = price_series.pct_change(126)
    df["mom_12m_minus_1m"] = price_series.pct_change(252) - price_series.pct_change(21)
    # Skewness y kurtosis rolling (60d)
    rets = price_series.pct_change()
    df["skew_60"] = rets.rolling(60).skew()
    df["kurt_60"] = rets.rolling(60).kurt()
    # Target: retorno futuro sobre target_horizon dias
    df["target"] = price_series.pct_change(target_horizon).shift(-target_horizon)
    return df.dropna()

def universe_features(prices, target_horizon=21):
    """Construye features para todo el universo."""
    out = {}
    for t in prices.columns:
        try:
            f = build_features(prices[t].dropna(), target_horizon=target_horizon)
            if len(f) >= 100:
                out[t] = f
        except Exception:
            continue
    return out
