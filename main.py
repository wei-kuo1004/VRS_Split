import threading
import logging
import time
from config.camera_config import load_cameras_config
from models.model_loader import load_models
from cameras.camera_monitor import CameraMonitor
from cameras.rtsp_checker import check_camera_signal
from utils.uploader import upload_worker


def main():
    #setup_logger()
    print("🚀 系統啟動中...")

    # 載入模型
    print("🔍 正在載入模型...")
    pose_model, maskcap_model, n_model = load_models()
    print("🔍 載入模型完成...")


    # 載入攝影機設定
    cameras_config = load_cameras_config()
    if not cameras_config:
        print("❌ 無法載入攝影機設定，系統結束")
        return


    # 啟動上傳背景執行緒
    from utils.uploader import start_upload_workers
    start_upload_workers(num_workers=3)  # 建議3~5個執行緒

    valid_configs = []
    for cam in cameras_config:
        if check_camera_signal(cam["rtsp_url"]):
            valid_configs.append(cam)
            print(f"✅ 攝影機可用：{cam['location']}")
        else:
            print(f"🚫 攝影機無法連線：{cam['location']}")

    if not valid_configs:
        print("⛔ 無可用攝影機，結束執行")
        return

    # 啟動每台攝影機監控
    for idx, cfg in enumerate(valid_configs):
        monitor = CameraMonitor(cfg, idx, pose_model, maskcap_model, n_model)
        threading.Thread(target=monitor.read_thread_func, daemon=True).start()
        threading.Thread(target=monitor.process_thread_func, daemon=True).start()
        threading.Thread(target=monitor.display_thread_func, daemon=True).start()


    print(f"🟢 實際啟用 {len(valid_configs)} 台攝影機")
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
