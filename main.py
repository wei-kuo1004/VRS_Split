import threading
import logging
import time
from config.camera_config import load_cameras_config
from models.model_loader import load_models
from cameras.camera_monitor import CameraMonitor
from cameras.rtsp_checker import check_camera_signal
from utils.uploader import upload_worker
from utils.logger import setup_logger


def main():
    setup_logger()
    logging.info("🚀 系統啟動中...")

    # 載入攝影機設定
    cameras_config = load_cameras_config()
    if not cameras_config:
        logging.critical("❌ 無法載入攝影機設定，系統結束")
        return

    # 載入模型
    pose_model, maskcap_model = load_models()

    # 啟動上傳背景執行緒
    from utils.uploader import start_upload_workers
    start_upload_workers(num_workers=3)  # 建議3~5個執行緒

    valid_configs = []
    for cam in cameras_config:
        if check_camera_signal(cam["rtsp_url"]):
            valid_configs.append(cam)
            logging.info(f"✅ 攝影機可用：{cam['location']}")
        else:
            logging.warning(f"🚫 攝影機無法連線：{cam['location']}")

    if not valid_configs:
        logging.critical("⛔ 無可用攝影機，結束執行")
        return

    # 啟動每台攝影機監控
    for idx, cfg in enumerate(valid_configs):
        monitor = CameraMonitor(cfg, idx, pose_model, maskcap_model)
        threading.Thread(target=monitor.read_thread_func, daemon=True).start()
        threading.Thread(target=monitor.process_thread_func, daemon=True).start()
        threading.Thread(target=monitor.display_thread_func, daemon=True).start()


    logging.info(f"🟢 實際啟用 {len(valid_configs)} 台攝影機")
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
