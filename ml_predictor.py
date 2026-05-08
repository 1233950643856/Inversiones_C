"""Ensemble ML LITE: XGB + RF + Ridge (sin LightGBM, mas rapido)."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

FEATURE_COLS = [
    "ret_1d","ret_5d","ret_10d","ret_21d","ret_63d","ret_126d","ret_252d",
    "vol_5d","vol_10d","vol_21d","vol_63d","vol_126d","vol_252d",
    "price_to_sma20","price_to_sma50","price_to_sma200","sma_20_50",
    "rsi_14","atr_14","mom_3m","mom_6m","mom_12m_minus_1m",
    "skew_60","kurt_60",
]

# Solo 3 modelos en el ensemble (LITE)
ENSEMBLE_WEIGHTS = {"xgb":0.50, "rf":0.30, "ridge":0.20}

def _train_models(X_train, y_train):
    models = {}
    if HAS_XGB:
        m = XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.08,
                         subsample=0.85, colsample_bytree=0.85, random_state=42,
                         verbosity=0, n_jobs=1, tree_method="hist")
        m.fit(X_train, y_train)
        models["xgb"] = m
    rf = RandomForestRegressor(n_estimators=60, max_depth=6, random_state=42,
                                n_jobs=1, min_samples_leaf=8)
    rf.fit(X_train, y_train)
    models["rf"] = rf
    sc = StandardScaler().fit(X_train)
    rd = Ridge(alpha=1.0).fit(sc.transform(X_train), y_train)
    models["ridge"] = (rd, sc)
    return models

def _predict(models, X):
    preds = {}
    if "xgb" in models:    preds["xgb"]   = models["xgb"].predict(X)
    if "rf"  in models:    preds["rf"]    = models["rf"].predict(X)
    if "ridge" in models:
        rd, sc = models["ridge"]
        preds["ridge"] = rd.predict(sc.transform(X))
    avail = {k: ENSEMBLE_WEIGHTS[k] for k in preds}
    total = sum(avail.values())
    if total <= 0: return None
    out = np.zeros(len(X))
    for k, p in preds.items():
        out += (avail[k] / total) * p
    return out

def walk_forward_predict(features_df, n_splits=2, min_train=200):
    """Walk-forward LITE: solo 2 folds para velocidad."""
    df = features_df.dropna()
    if len(df) < min_train + 50:
        return None, None
    X = df[FEATURE_COLS].values
    y = df["target"].values
    n = len(df)
    fold_size = max(20, (n - min_train) // n_splits)
    preds = np.full(n, np.nan)
    errors = []
    start = min_train
    while start < n:
        end = min(start + fold_size, n)
        Xtr, ytr = X[:start], y[:start]
        Xte, yte = X[start:end], y[start:end]
        try:
            models = _train_models(Xtr, ytr)
            yhat = _predict(models, Xte)
            if yhat is None:
                start = end; continue
            preds[start:end] = yhat
            errors.append(float(np.sqrt(np.mean((yte - yhat) ** 2))))
        except Exception:
            pass
        start = end
    rmse = float(np.mean(errors)) if errors else None
    return pd.Series(preds, index=df.index), rmse

def latest_forecast(features_df, min_train=200):
    """Entrena UNA vez con todo y predice. Lo usa rank_universe (rapido)."""
    df = features_df.dropna()
    if len(df) < min_train + 20:
        return None, None
    X = df[FEATURE_COLS].values
    y = df["target"].values
    train_X, train_y = X[:-1], y[:-1]
    last_X = X[-1:].reshape(1, -1)
    try:
        models = _train_models(train_X, train_y)
        pred = _predict(models, last_X)
        if pred is None: return None, None
        individuals = []
        if "xgb" in models: individuals.append(models["xgb"].predict(last_X)[0])
        if "rf"  in models: individuals.append(models["rf"].predict(last_X)[0])
        confidence = 1.0 / (1.0 + float(np.std(individuals))) if individuals else 0.5
        return float(pred[0]), float(confidence)
    except Exception:
        return None, None

def rank_universe(features_dict, min_train=200):
    rows = []
    for t, fdf in features_dict.items():
        pred, conf = latest_forecast(fdf, min_train=min_train)
        if pred is None: continue
        rows.append({"ticker":t, "predicted_return":pred, "confidence":conf})
    if not rows:
        return pd.DataFrame(columns=["ticker","predicted_return","confidence","score"])
    df = pd.DataFrame(rows)
    df["score"] = df["predicted_return"] * np.sqrt(df["confidence"].clip(lower=0.01))
    return df.sort_values("score", ascending=False).reset_index(drop=True)
