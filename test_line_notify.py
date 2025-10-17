import requests
import os
import sys

# === 1️⃣ 登入帳號以取得 Bearer Token ===
LOGIN_URL = "https://lineapi.pcbut.com.tw:888/api/account/login"
NOTIFY_URL = "https://lineapi.pcbut.com.tw:888/api/Push/notify-with-img"

LOGIN_DATA = {
    "username": "utbot",
    "password": "mi2@admin5566"
}

print("🔐 登入 LineGPT API 中...")

try:
    login_response = requests.post(LOGIN_URL, json=LOGIN_DATA, timeout=10)
    login_response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"❌ 登入失敗: {e}")
    sys.exit(1)

if login_response.status_code != 200:
    print(f"❌ 登入失敗，HTTP 狀態碼: {login_response.status_code}")
    print(login_response.text)
    sys.exit(1)

login_json = login_response.json()
token = login_json.get("token")

if not token:
    print("❌ 無法取得 Token，請檢查帳號密碼或伺服器設定。")
    print("伺服器回應:", login_json)
    sys.exit(1)

print("✅ 成功登入，已取得 Token。")


# === 2️⃣ 準備要發送的訊息與圖片 ===
CHAT_ID = "C5778A16-C191-408D-A9F6-16483ED57F3E"
IMAGE_PATH = r"D:\My Documents\wei-kuo\桌面\AI生成圖庫\2025-10-17 105303.jpg"

headers = {
    "Authorization": f"Bearer {token}"
}

data = {
    "message": "📢 一臉屌樣 ",
    "chatId": CHAT_ID
}

files = {}
if os.path.exists(IMAGE_PATH):
    files = {"file": open(IMAGE_PATH, "rb")}
else:
    print("⚠️ 找不到指定圖片，將僅發送文字訊息。")

# === 3️⃣ 發送圖片訊息 ===
print("\n🚀 傳送訊息與圖片中...")
try:
    response = requests.post(NOTIFY_URL, headers=headers, data=data, files=files, timeout=20)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"❌ 傳送失敗: {e}")
    sys.exit(1)

print(f"\nHTTP 狀態碼: {response.status_code}")
print(f"伺服器回應: {response.text}")

if response.status_code == 201:
    print("✅ 成功發送訊息與圖片！請至 LineGPT 聊天室確認。")
else:
    print("⚠️ 發送未成功，請檢查伺服器或參數設定。")

# === 4️⃣ 清理 ===
if "file" in files:
    files["file"].close()
