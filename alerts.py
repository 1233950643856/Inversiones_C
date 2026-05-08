"""Alertas de drift y comunicacion por email (Gmail SMTP)."""
import smtplib, ssl, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st

ALERTS_LOG = Path(__file__).parent / "alerts_log.json"

def get_config():
    """Lee configuracion email de st.secrets (cloud) o session_state (local)."""
    cfg = {}
    try:
        if "email" in st.secrets:
            cfg["sender"] = st.secrets["email"].get("sender", "")
            cfg["password"] = st.secrets["email"].get("password", "")
            cfg["recipient"] = st.secrets["email"].get("recipient", "")
    except Exception:
        pass
    if "email_sender" in st.session_state:
        cfg["sender"] = st.session_state.email_sender
    if "email_password" in st.session_state:
        cfg["password"] = st.session_state.email_password
    if "email_recipient" in st.session_state:
        cfg["recipient"] = st.session_state.email_recipient
    return cfg

def is_configured():
    cfg = get_config()
    return bool(cfg.get("sender") and cfg.get("password") and cfg.get("recipient"))

def send_email(subject, body_html, body_text=None):
    """Envia email via Gmail SMTP. Devuelve (ok, mensaje)."""
    cfg = get_config()
    if not is_configured():
        return False, "Configuracion incompleta (sender, password, recipient)"
    sender = cfg["sender"]
    password = cfg["password"]
    recipient = cfg["recipient"]
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        if body_text:
            msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx, timeout=20) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        _log_send(subject, "OK")
        return True, "Enviado correctamente"
    except smtplib.SMTPAuthenticationError:
        _log_send(subject, "AUTH_ERROR")
        return False, "Error de autenticacion. Para Gmail necesitas un 'App Password' (no tu contrasena normal)."
    except Exception as e:
        _log_send(subject, f"ERROR: {e}")
        return False, f"Error: {e}"

def _log_send(subject, status):
    try:
        log = []
        if ALERTS_LOG.exists():
            log = json.loads(ALERTS_LOG.read_text())
        log.append({"ts": datetime.now().isoformat(), "subject": subject, "status": status})
        log = log[-100:]  # solo ultimos 100
        ALERTS_LOG.write_text(json.dumps(log, indent=2))
    except Exception:
        pass

def last_send_time(subject_prefix=""):
    if not ALERTS_LOG.exists(): return None
    try:
        log = json.loads(ALERTS_LOG.read_text())
        for entry in reversed(log):
            if entry["subject"].startswith(subject_prefix) and entry["status"] == "OK":
                return datetime.fromisoformat(entry["ts"])
    except Exception:
        pass
    return None

def should_send_drift_alert(cooldown_hours=24):
    """No spamear: minimo cooldown_hours entre envios de drift."""
    last = last_send_time("Alerta drift")
    if last is None: return True
    return datetime.now() - last > timedelta(hours=cooldown_hours)

def build_drift_email(drift_dict, max_drift, threshold, target_label, summary):
    """Construye email HTML con detalle del drift."""
    rows_html = ""
    for t, d in sorted(drift_dict.items(), key=lambda x: -abs(x[1])):
        if abs(d) < 0.005: continue
        color = "#d9534f" if abs(d) >= threshold else "#f0ad4e" if abs(d) >= threshold/2 else "#5cb85c"
        rows_html += f'<tr><td>{t}</td><td style="color:{color};text-align:right">{d*100:+.2f}%</td></tr>'
    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#222">
    <h2>Inversiones PRO - Alerta de drift</h2>
    <p>Tu cartera <b>{target_label}</b> se ha desviado significativamente del objetivo.</p>
    <p><b>Drift maximo:</b> {max_drift*100:.2f}% (umbral configurado: {threshold*100:.1f}%)</p>
    <h3>Resumen de la cartera real</h3>
    <ul>
        <li>Valor actual: <b>{summary.get('total_value', 0):.2f} EUR</b></li>
        <li>Invertido total: {summary.get('total_invested', 0):.2f} EUR</li>
        <li>P/L: <span style="color:{'#5cb85c' if summary.get('total_pl', 0)>=0 else '#d9534f'}">{summary.get('total_pl', 0):+.2f} EUR ({summary.get('total_pl_pct', 0)*100:+.2f}%)</span></li>
    </ul>
    <h3>Drifts por activo (orden de mayor a menor)</h3>
    <table border="1" cellpadding="6" style="border-collapse:collapse">
        <thead><tr><th>Ticker</th><th>Drift</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    <p style="margin-top:20px;color:#888;font-size:12px">
        Drift positivo = pesa mas de lo objetivo (vender). Drift negativo = pesa menos (comprar).
    </p>
    <p style="color:#888;font-size:11px">
        Este aviso es informativo. NO constituye asesoramiento financiero.
        Inversiones PRO - {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </p>
    </body></html>
    """
    text = f"Drift maximo: {max_drift*100:.2f}% (umbral {threshold*100:.1f}%). " \
           f"Valor cartera: {summary.get('total_value', 0):.2f} EUR. " \
           f"P/L: {summary.get('total_pl', 0):+.2f} EUR."
    return html, text

def check_and_alert(target_weights, target_label, threshold=0.05, summary=None):
    """Comprueba drift y envia email si supera umbral. Devuelve (sent, message)."""
    from logbook import drift_vs_target, max_drift as mdrift
    drifts = drift_vs_target(target_weights)
    if not drifts:
        return False, "Sin posiciones reales."
    md = mdrift(target_weights)
    if md < threshold:
        return False, f"Drift bajo umbral ({md*100:.2f}% < {threshold*100:.1f}%)"
    if not is_configured():
        return False, "Email no configurado."
    if not should_send_drift_alert():
        return False, "Cooldown activo (ultimo aviso < 24h)."
    html, text = build_drift_email(drifts, md, threshold, target_label, summary or {})
    ok, msg = send_email(f"Alerta drift {md*100:.1f}% - Inversiones PRO", html, text)
    return ok, msg
