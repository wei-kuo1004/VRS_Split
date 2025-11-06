#uploader.py
#使用https://lineapi.pcbut.com.tw:888/api/account/login 取得TOKEN
#VRS聊天室群組ID chatId 83D9B831-E46E-46D2-A985-9CDB1175D462
#測試用chatId 2F0177B1-2AB0-471B-9001-E40B134F4D0F
#使用https://lineapi.pcbut.com.tw:888/api/notify-with-img 發送訊息與圖片

# ==========================================================
# uploader.py (v1.1)
# 改進內容：
# 1. 新增多執行緒上傳池 (3~5 worker threads)
# 2. 加入 requests timeout、重試機制與錯誤復原
# 3. 增加佇列健康監控 log
# ==========================================================
# uploader.py (v1.2)
# 改進內容：
# 1. 新增Email通知功能
# ==========================================================

import os
import cv2
import time
import json
import uuid
import logging
import threading
import requests
from datetime import datetime
from queue import Queue, Empty
from utils.helpers import safe_mkdir, get_timestamp, uuid_suffix

# ==========================================================
# 🔐 API 登入與傳送設定
# ==========================================================
LOGIN_URL = "https://lineapi.pcbut.com.tw:888/api/account/login"
NOTIFY_URL = "https://lineapi.pcbut.com.tw:888/api/Push/notify-with-img"
EMAIL_UPLOAD_URL = "https://eip.pcbut.com.tw/File/UploadYoloImage"

USERNAME = "utbot"
PASSWORD = "mi2@admin5566"
#CHAT_ID = "83D9B831-E46E-46D2-A985-9CDB1175D462"  # VRS聊天室群組ID
CHAT_ID = "2F0177B1-2AB0-471B-9001-E40B134F4D0F"  # 測試用群組ID

# ==========================================================
# ⚙️ 佇列與 Token 管理
# ==========================================================
upload_queue = Queue(maxsize=500)
_token_cache = {"token": None, "expire_time": 0}
_queue_log_timer = 0

# ==========================================================
# 📘 警報類型對應字典
# ==========================================================
valid_result_msgs = {
    "EYES CLOSED": "疑似閉眼過久，專注度下降。",
    "HEAD TURNED": "疑似長時間轉頭未注意作業方向。",
    "MISSING CAP": "疑似未配戴無塵帽，請現場確認。",
    "MISSING MASK": "疑似未戴口罩或配戴不正確，請同仁儘速查看。",
}

# ==========================================================
# 🧩 Token 管理
# ==========================================================
def get_line_token(force_refresh=False):
    """取得或刷新 Bearer Token"""
    global _token_cache
    now = time.time()

    if not force_refresh and _token_cache["token"] and now < _token_cache["expire_time"]:
        return _token_cache["token"]

    try:
        logging.info("🔐 正在向 LineAPI 伺服器登入以取得新 Token ...")
        response = requests.post(
            LOGIN_URL,
            json={"username": USERNAME, "password": PASSWORD},
            verify=False,
            timeout=(8, 10),
        )

        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            if not token:
                raise ValueError("登入回應中缺少 token 欄位")

            expire = now + 3600 * 24 * 365  # 一年有效期
            _token_cache = {"token": token, "expire_time": expire}
            logging.info("✅ LineAPI Token 取得成功")
            return token
        else:
            logging.error(f"❌ 登入失敗 ({response.status_code}): {response.text}")
            return None

    except Exception as e:
        logging.error(f"❌ 取得 LineAPI Token 發生錯誤: {e}")
        return None

# ==========================================================
# 📨 傳送訊息與圖片到 LineGPT 群組
# ==========================================================
def send_line_message(message: str, file_path: str = None, retries: int = 3):
    token = get_line_token()
    if not token:
        logging.error("❌ 無法取得有效 Token，略過此次發送")
        return False

    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message, "chatId": CHAT_ID}

    for attempt in range(1, retries + 1):
        files = None
        try:
            if file_path and os.path.exists(file_path):
                files = {"file": open(file_path, "rb")}

            response = requests.post(
                NOTIFY_URL,
                headers=headers,
                data=data,
                files=files,
                verify=False,
                timeout=(10, 15),
            )

            if response.status_code in (200, 201):
                logging.info(f"📤 LineGPT 通知成功：{message[:40]}...")
                return True
            elif response.status_code == 401 and attempt < retries:
                logging.warning("🔁 Token 可能過期，重新登入刷新 Token")
                get_line_token(force_refresh=True)
            else:
                logging.error(f"❌ 發送失敗 ({response.status_code}): {response.text}")

        except requests.exceptions.Timeout:
            logging.error(f"⚠️ LineGPT 傳送逾時 (第 {attempt}/{retries} 次)")
        except Exception as e:
            logging.error(f"⚠️ LineGPT 發送錯誤 (第 {attempt}/{retries} 次): {e}")
        finally:
            if files:
                try:
                    files["file"].close()
                except Exception:
                    pass

        time.sleep(2 ** attempt * 0.5)

    return False

# ==========================================================
# ✉️ 新增：寄信通知 API 函式
# ==========================================================
def send_email_notification(config, alert_type, file_path, result_msg):
    """呼叫 EIP API 觸發後端寄信"""
    try:
        api_payload = {
            "cameraId": config.get("camera_id", ""),
            "location": config.get("location", ""),
            "eventName": "專注度辨識",
            "eventDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": alert_type,
            "fileName": os.path.basename(file_path),
            "result": result_msg,
        }

        with open(file_path, "rb") as img_file:
            files = {"files": (os.path.basename(file_path), img_file, "image/jpeg")}
            resp = requests.post(
                EMAIL_UPLOAD_URL, data=api_payload, files=files, verify=False, timeout=10
            )

        if resp.status_code == 200:
            logging.info(f"📧 已通知後端寄信成功 ({config.get('location','')} | {alert_type})")
        else:
            logging.warning(f"⚠️ 寄信 API 回應異常：{resp.status_code} - {resp.text}")

    except Exception as e:
        logging.error(f"❌ 寄信 API 發送錯誤：{e}")

# ==========================================================
# 📦 上傳主執行緒
# ==========================================================
def upload_worker(worker_id=1):
    global _queue_log_timer

    while True:
        try:
            annotated_image, config, alert_type = upload_queue.get(timeout=2)
        except Empty:
            now = time.time()
            if now - _queue_log_timer > 60:
                _queue_log_timer = now
                logging.info(f"📦 Queue idle | current size: {upload_queue.qsize()}")
            continue

        try:
            if alert_type not in valid_result_msgs:
                result_msg = f"[錯誤] 未知警報類型：{alert_type}"
                logging.error(f"⛔ 不明警報類型：{alert_type}")
            else:
                result_msg = valid_result_msgs[alert_type]

            date_folder = datetime.now().strftime("%Y%m%d")
            folder = os.path.join("capture", date_folder)
            safe_mkdir(folder)

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            filename = (
                f"{config['camera_id']}_{get_timestamp()}_{uuid_suffix()}_{alert_type}.jpg"
            )
            file_path = os.path.join(folder, filename)
            cv2.imwrite(file_path, annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            message = (
                "【影像辨識通知】\n"
                "系統已偵測到疑似違規行為或潛在安全風險：\n"
                f"📍 地點：{config.get('location', '品保四課VRS')}\n"
                f"🕒 時間：{timestamp_str}\n"
                "🧠 特徵項目：專注度辨識\n"
                f"📄 內容：{result_msg}\n\n"
                "請儘速處理此事件並依據公司規定採取適當行動。\n"
                "如需更多詳細資料，可聯絡資訊處系統一課調閱更詳細影像。\n"
                "問題回報表單：https://forms.gle/rFZXVRP1aUxqQNG97"
            )

            success = send_line_message(message, file_path=file_path)

            # === 新增呼叫寄信 API ===
            send_email_notification(config, alert_type, file_path, result_msg)

            if success:
                logging.info(f"✅ [Worker {worker_id}] 任務完成：{config['location']} | {alert_type}")
            else:
                logging.error(f"❌ [Worker {worker_id}] LineGPT 推播失敗：{config['location']} | {alert_type}")

        except Exception as e:
            logging.error(f"❌ [Worker {worker_id}] 上傳工作發生錯誤：{e}")
        finally:
            upload_queue.task_done()

# ==========================================================
# 🚀 啟動多執行緒上傳池
# ==========================================================
def start_upload_workers(num_workers=3):
    for i in range(num_workers):
        t = threading.Thread(target=upload_worker, args=(i + 1,), daemon=True)
        t.start()
    logging.info(f"🧵 已啟動 {num_workers} 個上傳工作執行緒")
