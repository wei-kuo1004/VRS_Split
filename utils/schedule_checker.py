#schedule_checker.py
from datetime import datetime
import logging

_last_status = None

PERIODS = [
     ("00:30", "02:30"), ("02:40", "03:30"), ("04:00", "06:00"),
     ("06:10", "08:00"), ("08:30", "09:50"), ("10:00", "11:40"),
     ("12:10", "14:30"), ("14:40", "16:00"), ("16:30", "18:30"),
     ("19:00", "21:00"), ("21:10", "22:50"), ("23:00", "00:00"),
 ]
#PERIODS = [
#    ("08:00", "22:30")
#]

def is_within_time_period():
    global _last_status
    now = datetime.now().time()
    status = False
    matched = None
    for start_str, end_str in PERIODS:
        s = datetime.strptime(start_str, "%H:%M").time()
        e = datetime.strptime(end_str, "%H:%M").time()
        if s <= now <= e:
            status = True
            matched = f"{start_str}~{end_str}"
            break

    if status != _last_status:
        if status:
            logging.info(f"✅ 偵測啟用時段 {matched}")
        else:
            logging.info(f"⏸️ 非偵測時段")
        _last_status = status
    return status

def get_current_status():
    return "WORK" if is_within_time_period() else "SKIP"
