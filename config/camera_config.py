import os
import json
import logging

def create_default_config(path="cameras_config.txt"):
    default = [
        {
            "rtsp_url": "rtsp://hikvision:Unitech0815!@10.20.233.40/Streaming/Channels/101",
            "camera_id": "202001",
            "location": "UT2-2F-01",
            "upload_url": "https://eip.pcbut.com.tw/File/UploadYoloImage",
            "cooldown_seconds": 300,
            "eye_close_threshold": 0.015,
            "close_threshold_frames": 30,
            "head_turn_threshold": 50,
            "head_turn_frames": 100,
            "missing_cap_frames": 50,
            "missing_mask_frames": 50
        }
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(default, f, ensure_ascii=False, indent=4)
    logging.info(f"✅ 已建立預設設定檔 {path}")

def load_cameras_config(path="cameras_config14.txt"):
    if not os.path.exists(path):
        create_default_config(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            logging.info(f"✅ 成功載入攝影機設定，共 {len(cfg)} 台")
            return cfg
    except Exception as e:
        logging.error(f"❌ 無法讀取設定檔 {path}: {e}")
        return []
