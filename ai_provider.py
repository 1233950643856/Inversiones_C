"""Proveedor de IA con failover gratis: Groq -> Gemini -> Cerebras -> error."""
import os, json, time
from typing import Optional, List, Dict
import requests
import streamlit as st

# === Configuracion: lee de st.secrets, st.session_state o env ===
def _get_key(name: str) -> str:
    try:
        if name in st.secrets:
            v = st.secrets[name]
            if v: return str(v)
        if "ai_keys" in st.secrets and name in st.secrets["ai_keys"]:
            v = st.secrets["ai_keys"][name]
            if v: return str(v)
    except Exception:
        pass
    if name in st.session_state and st.session_state[name]:
        return str(st.session_state[name])
    return os.environ.get(name, "")

def is_configured() -> bool:
    """True si al menos UN proveedor tiene clave."""
    return any([_get_key("GROQ_API_KEY"),
                _get_key("GEMINI_API_KEY"),
                _get_key("CEREBRAS_API_KEY")])

# === Llamadas a cada proveedor ===
def _call_groq(prompt: str, system: str = "", max_tokens: int = 800,
               model: str = "llama-3.3-70b-versatile") -> Optional[str]:
    key = _get_key("GROQ_API_KEY")
    if not key: return None
    try:
        msgs = []
        if system: msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":model,"messages":msgs,"max_tokens":max_tokens,"temperature":0.6},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None

def _call_gemini(prompt: str, system: str = "", max_tokens: int = 800,
                 model: str = "gemini-1.5-flash") -> Optional[str]:
    key = _get_key("GEMINI_API_KEY")
    if not key: return None
    try:
        full = (system + "\n\n" + prompt) if system else prompt
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        r = requests.post(url,
            headers={"Content-Type":"application/json"},
            json={
                "contents":[{"parts":[{"text":full}]}],
                "generationConfig":{"maxOutputTokens":max_tokens,"temperature":0.6},
            },
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        return None
    except Exception:
        return None

def _call_cerebras(prompt: str, system: str = "", max_tokens: int = 800,
                   model: str = "llama3.1-70b") -> Optional[str]:
    key = _get_key("CEREBRAS_API_KEY")
    if not key: return None
    try:
        msgs = []
        if system: msgs.append({"role":"system","content":system})
        msgs.append({"role":"user","content":prompt})
        r = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
            json={"model":model,"messages":msgs,"max_tokens":max_tokens,"temperature":0.6},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        return None
    except Exception:
        return None

PROVIDERS = [("Groq", _call_groq), ("Gemini", _call_gemini), ("Cerebras", _call_cerebras)]

def ask(prompt: str, system: str = "", max_tokens: int = 800) -> Dict:
    """Pregunta con failover automatico. Devuelve {'text', 'provider', 'error'}."""
    if not is_configured():
        return {"text":"", "provider":None,
                "error":"No hay API key configurada. Ve a 'Configuracion IA' para anadir una."}
    errors = []
    for name, func in PROVIDERS:
        try:
            resp = func(prompt, system, max_tokens)
            if resp:
                return {"text":resp.strip(), "provider":name, "error":None}
            else:
                errors.append(f"{name}: sin respuesta")
        except Exception as e:
            errors.append(f"{name}: {e}")
    return {"text":"", "provider":None, "error":"Todos los proveedores fallaron: " + " | ".join(errors)}

@st.cache_data(ttl=86400, show_spinner=False)
def cached_ask(prompt_hash: str, prompt: str, system: str = "", max_tokens: int = 800) -> Dict:
    """Versiones cacheadas (una al dia) para insights diarios y similares."""
    return ask(prompt, system, max_tokens)
