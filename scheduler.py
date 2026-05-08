"""APScheduler en background: refresca datos al cierre de mercado europeo."""
from pathlib import Path
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

CET = pytz.timezone("Europe/Madrid")
FLAG_FILE = Path(__file__).parent / "last_update.json"

def write_flag():
    FLAG_FILE.write_text(json.dumps({
        "ts": datetime.now(CET).isoformat(),
        "source": "scheduler"
    }))

def read_flag():
    try:
        return json.loads(FLAG_FILE.read_text())
    except Exception:
        return None

def next_market_close():
    """Proximo lunes-viernes 22:05 CET."""
    now = datetime.now(CET)
    target = now.replace(hour=22, minute=5, second=0, microsecond=0)
    if now >= target or now.weekday() >= 5:
        from datetime import timedelta
        days_ahead = 1
        nxt = now + timedelta(days=days_ahead)
        while nxt.weekday() >= 5:
            nxt = nxt + timedelta(days=1)
        target = nxt.replace(hour=22, minute=5, second=0, microsecond=0)
    return target

_scheduler = None

def start_scheduler():
    global _scheduler
    if _scheduler is not None: return _scheduler
    sch = BackgroundScheduler(timezone=CET)
    sch.add_job(write_flag, CronTrigger(day_of_week="mon-fri", hour=22, minute=5,
                                         timezone=CET), id="market_close_refresh")
    sch.start()
    _scheduler = sch
    return sch
