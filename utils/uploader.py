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

# 預設聊天群組 ID（若未載入外部檔案即使用此值）
# 可改為正式群組： "83D9B831-E46E-46D2-A985-9CDB1175D462"
# 測試用 chatId： "2F0177B1-2AB0-471B-9001-E40B134F4D0F"
CHAT_ID = "2F0177B1-2AB0-471B-9001-E40B134F4D0F"

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
# 讀取 LineGPT Chat ID（支援多筆）
# ==========================================================
def load_linegpt_chat_ids(file_path: str | None = None):
    """
    預設會讀取與 uploader.py 同一目錄下的 LineGptChatRoomId.txt。
    若傳入 file_path：
      - 絕對路徑會直接使用
      - 相對路徑會相對於本模組所在目錄解析 (utils/)
    """
    chat_ids = []
    try:
        base_dir = os.path.dirname(__file__)  # utils 資料夾
        if file_path is None:
            file_path = os.path.join(base_dir, "LineGptChatRoomId.txt")
        else:
            if not os.path.isabs(file_path):
                file_path = os.path.join(base_dir, file_path)

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    v = line.strip()
                    if v:
                        chat_ids.append(v)
            print(f"✔ 已載入 {len(chat_ids)} 個 LINEGPT chat_id：{chat_ids} (from {file_path})")
        else:
            print(f"ℹ️ chat_id 檔案不存在，使用預設 CHAT_ID ({CHAT_ID})：{file_path}")
    except Exception as e:
        print(f"❌ 無法讀取 chat_id 檔案：{e}")
    return chat_ids

# ✅ 測試程式碼放在函數定義之後
print("=" * 60)
print("🚀 程式開始執行...")
print(f"📂 當前工作目錄：{os.getcwd()}")
print(f"📋 檔案列表：{os.listdir('.')}")
print("=" * 60)


# 只載入一次
LINEGPT_CHAT_IDS = load_linegpt_chat_ids()


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
        print("🔐 正在向 LineAPI 伺服器登入以取得新 Token ...")
        response = requests.post(
            LOGIN_URL,
            json={"username": USERNAME, "password": PASSWORD},
            verify=False,
            timeout=(8, 10),
            proxies={}  # 不使用代理
        )

        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            if not token:
                raise ValueError("登入回應中缺少 token 欄位")

            expire = now + 3600 * 24 * 365  # 一年有效期
            _token_cache = {"token": token, "expire_time": expire}
            print("✅ LineAPI Token 取得成功")
            return token
        else:
            print(f"❌ 登入失敗 ({response.status_code}): {response.text}")
            return None

    except Exception as e:
        print(f"❌ 取得 LineAPI Token 發生錯誤: {e}")
        return None

# ==========================================================
# 📨 傳送訊息與圖片到 LineGPT 群組
# ==========================================================
def send_line_message(message: str, file_path: str = None, retries: int = 3):
    token = get_line_token()
    if not token:
        print("❌ 無法取得有效 Token，略過此次發送")
        return False

    headers = {"Authorization": f"Bearer {token}"}

    # 決定要送到哪些 chat id（若外部檔案有載入，則使用外部清單；否則使用預設 CHAT_ID）
    target_chat_ids = LINEGPT_CHAT_IDS if LINEGPT_CHAT_IDS else [CHAT_ID]

    # 只要其中一個目標成功即可視為成功（但會嘗試對每個目標發送）
    overall_success = False

    for attempt in range(1, retries + 1):
        try:
            # 依序對每個 chat_id 發送（每個 chat_id 使用獨立 request）
            for chat_id in target_chat_ids:
                files = None
                try:
                    data = {"message": message, "chatId": chat_id}
                    if file_path and os.path.exists(file_path):
                        files = {"file": open(file_path, "rb")}

                    response = requests.post(
                        NOTIFY_URL,
                        headers=headers,
                        data=data,
                        files=files,
                        verify=False,
                        timeout=(10, 15),
                        proxies={}  # 不使用代理
                    )

                    if response.status_code in (200, 201):
                        print(f"📤 LineGPT 通知成功 (chatId={chat_id})：{message[:40]}...")
                        overall_success = True
                    elif response.status_code == 401:
                        print("🔁 Token 可能過期，重新登入刷新 Token")
                        get_line_token(force_refresh=True)
                        # 若尚未到達最大重試次數，會在下一輪再次嘗試
                    else:
                        print(f"❌ 發送失敗 (chatId={chat_id}) ({response.status_code}): {response.text}")
                except requests.exceptions.Timeout:
                    print(f"⚠️ LineGPT 傳送逾時 (chatId={chat_id}, 第 {attempt}/{retries} 次)")
                except Exception as e:
                    print(f"⚠️ LineGPT 發送錯誤 (chatId={chat_id}, 第 {attempt}/{retries} 次): {e}")
                finally:
                    if files:
                        try:
                            files["file"].close()
                        except Exception:
                            pass

            # 若已至少有一個成功，直接回傳 True
            if overall_success:
                return True

        except Exception as e:
            print(f"⚠️ LineGPT 發送流程錯誤 (第 {attempt}/{retries} 次): {e}")

        time.sleep(2 ** attempt * 0.5)

    return overall_success
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
            print(f"📧 已通知後端寄信成功 ({config.get('location','')} | {alert_type})")
        else:
            print(f"⚠️ 寄信 API 回應異常：{resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"❌ 寄信 API 發送錯誤：{e}")

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
                print(f"📦 Queue idle | current size: {upload_queue.qsize()}")
            continue

        try:
            if alert_type not in valid_result_msgs:
                result_msg = f"[錯誤] 未知警報類型：{alert_type}"
                print(f"⛔ 不明警報類型：{alert_type}")
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
                print(f"✅ [Worker {worker_id}] 任務完成：{config['location']} | {alert_type}")
            else:
                print(f"❌ [Worker {worker_id}] LineGPT 推播失敗：{config['location']} | {alert_type}")

        except Exception as e:
            print(f"❌ [Worker {worker_id}] 上傳工作發生錯誤：{e}")
        finally:
            upload_queue.task_done()

# ==========================================================
# 🚀 啟動多執行緒上傳池
# ==========================================================
def start_upload_workers(num_workers=3):
    for i in range(num_workers):
        t = threading.Thread(target=upload_worker, args=(i + 1,), daemon=True)
        t.start()
    print(f"🧵 已啟動 {num_workers} 個上傳工作執行緒")


def test_get_line_token():
    print("=" * 60)
    print("🚀 開始測試 TOKEN 獲取...")
    
    token = get_line_token(force_refresh=True)  # 強制刷新以獲取新 TOKEN
    if token:
        print(f"✅ 獲取 TOKEN 成功: {token}")
    else:
        print("❌ 獲取 TOKEN 失敗")

# 執行測試
if __name__ == "__main__":
    test_get_line_token()