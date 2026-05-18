"""Diario de operaciones: registro de compras/ventas reales del usuario.
Persistencia: Supabase (cloud) > SQLite (local PC) > session_state (fallback)."""
import sqlite3, json, os
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st

try:
    import supabase_db
    HAS_SUPABASE = True
except Exception:
    HAS_SUPABASE = False

DB_PATH = Path(__file__).parent / "logbook.db"


def _backend():
    """Devuelve 'supabase', 'sqlite' o 'session' segun lo disponible."""
    if HAS_SUPABASE and supabase_db.is_configured():
        return "supabase"
    try:
        # Probar SQLite (puede fallar en cloud por filesystem efimero)
        DB_PATH.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.close()
        return "sqlite"
    except Exception:
        return "session"


def _init_sqlite():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            n_shares REAL NOT NULL,
            price_eur REAL NOT NULL,
            commission_eur REAL DEFAULT 0,
            broker TEXT,
            notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _save_session():
    if "logbook_backup" not in st.session_state:
        st.session_state.logbook_backup = []


def add_transaction(ticker, side, n_shares, price_eur, commission_eur=0,
                    broker=None, notes=None, date=None):
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    backend = _backend()
    if backend == "supabase":
        row = supabase_db.insert_transaction(date, ticker, side, float(n_shares),
            float(price_eur), float(commission_eur or 0), broker, notes)
        return row.get("id") if row else None
    if backend == "sqlite":
        try:
            conn = _init_sqlite()
            cur = conn.execute(
                "INSERT INTO transactions(date,ticker,side,n_shares,price_eur,commission_eur,broker,notes,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (date, ticker, side, float(n_shares), float(price_eur),
                 float(commission_eur or 0), broker, notes,
                 datetime.now().isoformat()))
            conn.commit()
            new_id = cur.lastrowid
            conn.close()
            return new_id
        except Exception:
            pass
    # Session fallback
    _save_session()
    row = {"id": len(st.session_state.logbook_backup) + 1,
           "date": date, "ticker": ticker, "side": side,
           "n_shares": float(n_shares), "price_eur": float(price_eur),
           "commission_eur": float(commission_eur or 0),
           "broker": broker, "notes": notes,
           "created_at": datetime.now().isoformat()}
    st.session_state.logbook_backup.append(row)
    return row["id"]


def list_transactions():
    backend = _backend()
    if backend == "supabase":
        rows = supabase_db.list_transactions()
        if not rows:
            return pd.DataFrame(columns=["id","date","ticker","side","n_shares",
                "price_eur","commission_eur","broker","notes"])
        return pd.DataFrame(rows)
    if backend == "sqlite" and DB_PATH.exists():
        try:
            conn = _init_sqlite()
            df = pd.read_sql_query(
                "SELECT id,date,ticker,side,n_shares,price_eur,commission_eur,broker,notes "
                "FROM transactions ORDER BY date DESC, id DESC", conn)
            conn.close()
            return df
        except Exception:
            pass
    _save_session()
    rows = st.session_state.logbook_backup
    if not rows:
        return pd.DataFrame(columns=["id","date","ticker","side","n_shares",
            "price_eur","commission_eur","broker","notes"])
    return pd.DataFrame(rows)


def delete_transaction(tx_id):
    backend = _backend()
    if backend == "supabase":
        return supabase_db.delete_transaction(int(tx_id))
    if backend == "sqlite" and DB_PATH.exists():
        try:
            conn = _init_sqlite()
            conn.execute("DELETE FROM transactions WHERE id=?", (tx_id,))
            conn.commit(); conn.close()
            return True
        except Exception:
            pass
    _save_session()
    st.session_state.logbook_backup = [r for r in st.session_state.logbook_backup if r.get("id") != tx_id]
    return True


def current_positions():
    df = list_transactions()
    if df.empty:
        return pd.DataFrame(columns=["ticker","n_shares","cost_basis_eur","invested_eur"])
    out = {}
    for _, r in df.iterrows():
        t = r["ticker"]
        if t not in out:
            out[t] = {"shares": 0.0, "invested": 0.0}
        sign = 1 if r["side"] == "buy" else -1
        out[t]["shares"] += sign * float(r["n_shares"])
        out[t]["invested"] += sign * (float(r["n_shares"]) * float(r["price_eur"]) + float(r["commission_eur"] or 0))
    rows = []
    for t, d in out.items():
        if abs(d["shares"]) < 1e-6: continue
        cost_basis = d["invested"] / d["shares"] if d["shares"] > 0 else 0
        rows.append({
            "ticker": t,
            "n_shares": d["shares"],
            "cost_basis_eur": cost_basis,
            "invested_eur": d["invested"],
        })
    return pd.DataFrame(rows)


def portfolio_value(prices_eur, fx_eur_usd=1.08):
    pos = current_positions()
    if pos.empty: return None
    rows = []
    suffixes = (".DE",".AS",".MC",".PA",".MI",".SW",".L",".CO")
    total_value = 0.0
    total_invested = 0.0
    for _, r in pos.iterrows():
        t = r["ticker"]
        if t not in prices_eur.index: continue
        price_native = float(prices_eur.loc[t])
        is_eur = any(t.endswith(s) for s in suffixes) or t.startswith("VWCE")
        price_eur_now = price_native if is_eur else (price_native / fx_eur_usd)
        value = r["n_shares"] * price_eur_now
        pl = value - r["invested_eur"]
        pl_pct = pl / r["invested_eur"] if r["invested_eur"] > 0 else 0
        rows.append({
            "ticker": t, "n_shares": r["n_shares"],
            "cost_basis_eur": r["cost_basis_eur"],
            "price_now_eur": price_eur_now,
            "value_eur": value, "invested_eur": r["invested_eur"],
            "pl_eur": pl, "pl_pct": pl_pct,
        })
        total_value += value
        total_invested += r["invested_eur"]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["weight_actual"] = df["value_eur"] / total_value if total_value > 0 else 0
    summary = {
        "total_value": total_value,
        "total_invested": total_invested,
        "total_pl": total_value - total_invested,
        "total_pl_pct": (total_value - total_invested)/total_invested if total_invested>0 else 0,
    }
    return df, summary


def export_json():
    df = list_transactions()
    if df.empty: return "[]"
    return df.to_json(orient="records", indent=2)


def import_json(json_str):
    try:
        data = json.loads(json_str)
        count = 0
        for row in data:
            add_transaction(
                ticker=row["ticker"], side=row["side"],
                n_shares=row["n_shares"], price_eur=row["price_eur"],
                commission_eur=row.get("commission_eur",0),
                broker=row.get("broker"), notes=row.get("notes"),
                date=row.get("date"))
            count += 1
        return count
    except Exception:
        return -1


def drift_vs_target(target_weights):
    pos = current_positions()
    if pos.empty: return {}
    total_invested = pos["invested_eur"].sum()
    if total_invested <= 0: return {}
    actual = {r["ticker"]: r["invested_eur"]/total_invested for _, r in pos.iterrows()}
    drifts = {}
    all_tickers = set(actual.keys()) | set(target_weights.keys())
    for t in all_tickers:
        a = actual.get(t, 0.0)
        tg = target_weights.get(t, 0.0)
        drifts[t] = a - tg
    return drifts


def max_drift(target_weights):
    drifts = drift_vs_target(target_weights)
    if not drifts: return 0.0
    return max(abs(v) for v in drifts.values())


def get_backend_info():
    """Para mostrar al usuario que backend se esta usando."""
    backend = _backend()
    if backend == "supabase":
        return "Supabase (persistente cloud)"
    elif backend == "sqlite":
        return "SQLite local (PC)"
    return "Sesion (efimero, descarga JSON antes de cerrar)"
