"""Wrapper minimo de Supabase REST API (PostgREST). Persistencia de logbook + alerts."""
import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import requests
import streamlit as st


def _get_secret(name: str) -> str:
    """Lee de st.secrets, session_state o env."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    if name in st.session_state and st.session_state[name]:
        return str(st.session_state[name])
    return os.environ.get(name, "")


def get_url() -> str:
    return _get_secret("SUPABASE_URL")


def get_key() -> str:
    return _get_secret("SUPABASE_ANON_KEY")


def is_configured() -> bool:
    return bool(get_url()) and bool(get_key())


def _headers():
    key = get_key()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _user_id() -> str:
    """User ID actual. Por ahora solo tu, pero soporta multi-user en el futuro."""
    return _get_secret("USER_ID") or "cristina"


# ===================== TRANSACTIONS =====================
def insert_transaction(date: str, ticker: str, side: str, n_shares: float,
                       price_eur: float, commission_eur: float = 0,
                       broker: str = None, notes: str = None) -> Optional[Dict]:
    if not is_configured():
        return None
    url = f"{get_url()}/rest/v1/transactions"
    payload = {
        "user_id": _user_id(),
        "date": date,
        "ticker": ticker,
        "side": side,
        "n_shares": n_shares,
        "price_eur": price_eur,
        "commission_eur": commission_eur or 0,
        "broker": broker,
        "notes": notes,
    }
    try:
        r = requests.post(url, headers=_headers(), json=payload, timeout=15)
        if r.status_code in (200, 201):
            data = r.json()
            return data[0] if isinstance(data, list) and data else data
        return None
    except Exception:
        return None


def list_transactions() -> List[Dict]:
    if not is_configured():
        return []
    url = f"{get_url()}/rest/v1/transactions"
    params = {
        "user_id": f"eq.{_user_id()}",
        "select": "id,date,ticker,side,n_shares,price_eur,commission_eur,broker,notes,created_at",
        "order": "date.desc,id.desc",
    }
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def delete_transaction(tx_id: int) -> bool:
    if not is_configured():
        return False
    url = f"{get_url()}/rest/v1/transactions"
    params = {"id": f"eq.{tx_id}", "user_id": f"eq.{_user_id()}"}
    try:
        r = requests.delete(url, headers=_headers(), params=params, timeout=15)
        return r.status_code in (200, 204)
    except Exception:
        return False


# ===================== ALERTS LOG =====================
def log_alert(subject: str, status: str) -> bool:
    if not is_configured():
        return False
    url = f"{get_url()}/rest/v1/alerts_log"
    payload = {"user_id": _user_id(), "subject": subject, "status": status}
    try:
        r = requests.post(url, headers=_headers(), json=payload, timeout=15)
        return r.status_code in (200, 201)
    except Exception:
        return False


def last_alert_time(subject_prefix: str = "") -> Optional[datetime]:
    if not is_configured():
        return None
    url = f"{get_url()}/rest/v1/alerts_log"
    params = {
        "user_id": f"eq.{_user_id()}",
        "status": "eq.OK",
        "select": "ts,subject",
        "order": "ts.desc",
        "limit": "20",
    }
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=15)
        if r.status_code != 200:
            return None
        for row in r.json():
            if row["subject"].startswith(subject_prefix):
                return datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
    except Exception:
        pass
    return None


def alerts_history(limit: int = 20) -> List[Dict]:
    if not is_configured():
        return []
    url = f"{get_url()}/rest/v1/alerts_log"
    params = {
        "user_id": f"eq.{_user_id()}",
        "select": "ts,subject,status",
        "order": "ts.desc",
        "limit": str(limit),
    }
    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []
