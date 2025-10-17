#uploader.py
#使用https://lineapi.pcbut.com.tw:888/api/account/login 取得TOKEN
#VRS聊天室群組ID chatId 83D9B831-E46E-46D2-A985-9CDB1175D462
#測試用chatId C5778A16-C191-408D-A9F6-16483ED57F3E
#使用https://lineapi.pcbut.com.tw:888/api/notify-with-img 發送訊息與圖片

import os
import cv2
import json
import time
import logging
import requests
from datetime import datetime
from queue import Queue
from utils.helpers import safe_mkdir, get_timestamp, uuid_suffix

# ==========================================================
# 🔐 API 登入與傳送設定
# ==========================================================
LOGIN_URL = "https://lineapi.pcbut.com.tw:888/api/account/login"
NOTIFY_URL = "https://lineapi.pcbut.com.tw:888/api/Push/notify-with-img"

USERNAME = "utbot"
PASSWORD = "mi2@admin5566"
CHAT_ID = "C5778A16-C191-408D-A9F6-16483ED57F3E"  # ⚠️ 測試用聊天室群組ID

# ==========================================================
# ⚙️ 佇列與 Token 管理
# ==========================================================
upload_queue = Queue(maxsize=500)
_token_cache = {"token": None, "expire_time": 0}


# ==========================================================
# 📘 警報類型對應字典
# ==========================================================
valid_result_msgs = {
    "EYES CLOSED": "疑似閉眼過久，專注度下降。",
    "HEAD TURNED": "疑似長時間轉頭未注意作業方向。",
    "MISSING CAP": "疑似未配戴安全帽，請現場確認。",
    "MISSING MASK": "疑似未戴口罩，請同仁儘速查看。",
}


# ==========================================================
# 🧩 Token 管理
# ==========================================================
def get_line_token(force_refresh=False):
    """取得或刷新 Bearer Token"""
    global _token_cache
    now = time.time()

    # 若仍在有效期內，直接使用舊 Token
    if not force_refresh and _token_cache["token"] and now < _token_cache["expire_time"]:
        return _token_cache["token"]

    try:
        logging.info("🔐 正在向 LineAPI 伺服器登入以取得新 Token ...")
        response = requests.post(
            LOGIN_URL,
            json={"username": USERNAME, "password": PASSWORD},
            verify=False,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            if not token:
                raise ValueError("登入回應中缺少 token 欄位")

            expire = now + 3600 * 24 * 365  # 有效期一年
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
# 📨 發送訊息到 LineGPT
# ==========================================================
def send_line_message(message: str, file_path: str = None, retries: int = 3):
    """傳送訊息與圖片到 LineGPT 群組"""
    token = get_line_token()
    if not token:
        logging.error("❌ 無法取得有效 Token，略過此次發送")
        return False

    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message, "chatId": CHAT_ID}

    files = None
    if file_path and os.path.exists(file_path):
        files = {"file": open(file_path, "rb")}

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                NOTIFY_URL,
                headers=headers,
                data=data,
                files=files,
                verify=False,
                timeout=15,
            )

            if response.status_code in (200, 201):
                logging.info(f"📤 LineGPT 通知成功: {message[:50]}...")
                return True
            elif response.status_code == 401 and attempt < retries:
                logging.warning("🔁 Token 可能失效，嘗試重新登入刷新 Token")
                get_line_token(force_refresh=True)
            else:
                logging.error(f"❌ 發送失敗 ({response.status_code}): {response.text}")

        except Exception as e:
            logging.error(f"⚠️ LineGPT 發送錯誤 (嘗試 {attempt}/{retries}): {e}")

        time.sleep(2 ** attempt * 0.5)

    if files:
        files["file"].close()
    return False


# ==========================================================
# 📦 上傳主執行緒（整合 YOLO 截圖）
# ==========================================================
def upload_worker():
    """
    處理 upload_queue 中的任務，並自動推送到 LineGPT 聊天室。
    """
    while True:
        annotated_image, config, alert_type = upload_queue.get()
        try:
            # === 防呆警報類型檢查 ===
            if alert_type not in valid_result_msgs:
                logging.error(
                    f"⛔ 不明警報類型：{alert_type}，請檢查程式邏輯與來源！"
                )
                result_msg = f"[錯誤] 未知警報類型：{alert_type}"
            else:
                result_msg = valid_result_msgs[alert_type]

            # === 建立本地圖片檔 ===
            date_folder = datetime.now().strftime("%Y%m%d")
            folder = os.path.join("capture", date_folder)
            safe_mkdir(folder)

            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            filename = (
                f"{config['camera_id']}_{get_timestamp()}_{uuid_suffix()}_{alert_type}.jpg"
            )
            file_path = os.path.join(folder, filename)
            cv2.imwrite(file_path, annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # === 組成正式通知訊息 ===
            message = (
                "【影像辨識通知】\n"
                "系統已偵測到疑似違規行為或潛在安全風險：\n"
                f"📍 地點：{config.get('location', '品保四課VRS')}\n"
                f"🕒 時間：{timestamp_str}\n"
                "🧠 特徵點項目：專注度辨識\n"
                f"📄 內容：{result_msg}\n\n"
                "請儘速處理此事件並依據公司規定採取適當行動。\n"
                "如需更多詳細資料，可聯絡資訊處AI課調閱更詳細的資訊。\n"
                "問題處理回報：https://forms.gle/rFZXVRP1aUxqQNG97"
            )

            # === 發送至 LineGPT ===
            success = send_line_message(message, file_path=file_path)

            if success:
                logging.info(
                    f"✅ LineGPT 推播完成：{config['location']} | {alert_type}"
                )
            else:
                logging.error(
                    f"❌ LineGPT 推播失敗：{config['location']} | {alert_type}"
                )

        except Exception as e:
            logging.error(f"❌ 上傳工作發生錯誤：{e}")
        finally:
            upload_queue.task_done()
