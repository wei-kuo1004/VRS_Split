# 📁 Project Structure Overview

project_root/
│
├── main.py # 入口點：初始化、啟動主程序
│
├── config/
│ └── camera_config.py # 攝影機設定檔載入與預設生成
│
├── models/
│ ├── model_loader.py # 模型載入與初始化（YOLO / MediaPipe）
│ └── alert_cooldown.py # 警報冷卻時間管理類別
│
├── utils/
│ ├── logger.py # 日誌設定與格式化
│ ├── schedule_checker.py # 偵測啟用時段判斷
│ ├── head_angle.py # 頭部角度與轉頭判定
│ ├── helpers.py # 通用輔助函式（resource_path、目錄建立）
│ └── uploader.py # 上傳佇列與背景上傳執行緒
│
├── cameras/
│ ├── camera_monitor.py # 主體 CameraMonitor 類別：讀取、辨識、顯示、截圖
│ └── rtsp_checker.py # RTSP 連線預檢
│
├── capture/ # 📁 截圖資料夾（執行時生成）
│
└── model/ # 📁 模型資料夾（存放 .pt 與 .task）
├── yolo11n-pose.pt
├── mask_cap/best.pt
├── face_landmarker.task
└── pose_landmarker_full.task

yaml
複製程式碼

---

## 🔧 模組職責概要

### **main.py**
- 初始化日誌與環境變數  
- 載入攝影機設定  
- 初始化模型（由 `model_loader` 提供）  
- 檢查 RTSP 連線可用性  
- 啟動 `CameraMonitor` 執行緒  
- 啟動上傳執行緒  

---

### **config/camera_config.py**
- 提供 `load_cameras_config()` 與 `create_default_config()`  
- 管理 `cameras_config.txt` JSON 檔  
- 自動建立預設設定檔（RTSP URL、閾值、上傳 URL 等）

---

### **models/model_loader.py**
- 載入 YOLO 與 MediaPipe 模型  
- 封裝為：
  - `load_pose_model()`
  - `load_maskcap_model()`  
- 自動檢查模型是否存在與初始化（含黑圖 warm-up）

---

### **models/alert_cooldown.py**
- 定義 `AlertCooldownManager` 類別  
- 管理每台攝影機與警報類型的冷卻計時  
- 提供方法：
  - `is_in_cooldown(camera_id, alert_type)`
  - `update_last_time(camera_id, alert_type)`
  - `get_remaining_time(camera_id, alert_type)`

---

### **utils/logger.py**
- 統一日誌格式  
- 控制全域 logging level（info / debug）  
- 支援多執行緒 lock-safe 輸出  

---

### **utils/schedule_checker.py**
- 管理工作時段啟用邏輯  
- 提供：
  - `is_within_time_period()`
  - `get_current_status()`  
- 讓畫面能顯示 “WORK / SKIP” 狀態  

---

### **utils/head_angle.py**
- 封裝頭部角度計算與轉頭判斷邏輯  
- 提供：
  - `calculate_head_angle(keypoints, frame_width, frame_height)`
  - `is_head_turned(horizontal_angle, face_asymmetry, side_turn_ratio)`  
- 集中所有角度與閾值調整  

---

### **utils/helpers.py**
- 提供通用工具函式：
  - `resource_path(relative_path)`（支援 PyInstaller）
  - `safe_mkdir(path)`
  - `get_timestamp()`
  - `uuid_suffix()`

---

### **utils/uploader.py**
- 封裝 `upload_worker()` 上傳執行緒  
- 處理：
  - JPG 檔案壓縮與儲存  
  - 組成 POST 請求  
  - 寫入上傳日誌  

---

### **cameras/camera_monitor.py**
- `CameraMonitor` 主類別：
  - PyAV / OpenCV 自動切換  
  - MediaPipe 臉部辨識  
  - YOLO pose + mask/cap 推論  
  - 狀態顯示、關鍵點繪製、冷卻控制  
  - 事件觸發與截圖入佇列  
- 高度模組化，獨立於 `main.py`

---

### **cameras/rtsp_checker.py**
- 函式：`check_camera_signal(rtsp_url, timeout=5)`  
- 確認 RTSP 串流是否可開啟  
- 提供連線檢測用於 `main.py` 的啟動階段