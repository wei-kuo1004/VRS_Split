import os
# ✅ 在程式最開頭新增 FFmpeg 優化設定
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1048576|max_delay;500000"
os.environ["FFMPEG_LOG_LEVEL"] = "warning"  # 降低 FFmpeg 日誌等級

import cv2
import time
import threading
import numpy as np
import os
import logging
import requests
from datetime import datetime
from queue import Queue
from ultralytics import YOLO
import mediapipe as mp
print("✅ mediapipe import success")
import torch
import subprocess as sp
import math
import json
import uuid  
import av 
print(f"✅ PyAV Version: {av.__version__}")
print("✅ Linked FFmpeg Libraries:")
for k, v in av.library_versions.items():
    print(f"  - {k}: {v}")

import traceback
import sys

def resource_path(relative_path):
    """打包成 .exe 時取得資源實體路徑"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 取得目前腳本所在資料夾
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ 模型路徑（打包後支援 PyInstaller）
pose_model_path = resource_path("model/yolo11n-pose.pt")
maskcap_model_path = resource_path("model/mask_cap/best.pt")
face_model_path = resource_path("model/face_landmarker.task")
pose_task_path = resource_path("model/pose_landmarker_full.task")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")

# === 🔐 多執行緒保護用 Lock（新增） ===
logfile_lock = threading.Lock()
alert_times_lock = threading.Lock()

# 每種類別（眼睛閉合、轉頭）的冷卻時間（秒）
alert_cooldowns = {
    "EYES CLOSED": 300,
    "HEAD TURNED": 300,
    "MISSING CAP": 600,
    "MISSING MASK": 600,
}

# 每支攝影機的上次觸發時間記錄
last_alert_times = {}

# === 裝置運算模式設定 ===
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用設備: {DEVICE}")  

# 設定 log 格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === 讀取攝影機設定檔案 ===
def load_cameras_config(config_file="cameras_config.txt"):
    """
    從外部檔案讀取攝影機設定
    支援JSON格式的設定檔
    """
    try:
        if not os.path.exists(config_file):
            # 如果檔案不存在，建立預設設定檔
            create_default_config(config_file)
            
        with open(config_file, 'r', encoding='utf-8') as f:
            cameras_config = json.load(f)
            
        logging.info(f"✅ 成功讀取設定檔: {config_file}, 共 {len(cameras_config)} 台攝影機")
        return cameras_config
        
    except json.JSONDecodeError as e:
        logging.error(f"❌ 設定檔格式錯誤: {e}")
        logging.info("💡 請檢查 cameras_config.txt 是否為有效的JSON格式")
        return []
    except Exception as e:
        logging.error(f"❌ 讀取設定檔失敗: {e}")
        return []

def create_default_config(config_file="cameras_config.txt"):
    """
    建立預設的攝影機設定檔案
    """
    default_config = [
        {
            "rtsp_url": "rtsp://hikvision:Unitech0815!@10.20.233.40/Streaming/Channels/101",
            "cooldown_seconds": 300,
            "eye_close_threshold": 0.015,
            "close_threshold_frames": 30,
            "head_turn_threshold": 50,
            "head_turn_frames": 100,
            "missing_cap_frames": 50,
            "missing_mask_frames": 50,
            "camera_id": "202001",
            "location": "UT2-2F-01",
            "upload_url": "https://eip.pcbut.com.tw/File/UploadYoloImage"
        }
    ]
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
        logging.info(f"✅ 已建立預設設定檔: {config_file}")
    except Exception as e:
        logging.error(f"❌ 建立預設設定檔失敗: {e}")

# 載入攝影機設定
cameras_config = load_cameras_config()

# 如果沒有成功載入設定，程式結束
if not cameras_config:
    logging.error("❌ 無法載入攝影機設定，程式結束")
    exit(1)

# ✅ 檢查模型是否存在（執行階段）
for path in [pose_model_path, maskcap_model_path, face_model_path, pose_task_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到模型檔案：{path}")
    
# ✅ 初始化 pose_model，並關閉 fuse 防止 bn 錯誤
pose_model = YOLO(pose_model_path).to(DEVICE)
pose_model.model.fuse = lambda *args, **kwargs: pose_model.model  # 禁用 fuse
pose_model.predict(np.zeros((480, 640, 3), dtype=np.uint8))  # 預先觸發初始化（黑圖）
logging.info("✅ Pose 模型已初始化（含 fuse disabled）")

# ✅ 初始化 maskcap_model（同樣保險起見也關閉 fuse）
maskcap_model = YOLO(maskcap_model_path).to(DEVICE)
maskcap_model.model.fuse = lambda *args, **kwargs: maskcap_model.model
maskcap_model.predict(np.zeros((480, 640, 3), dtype=np.uint8))  # 預推論一次
logging.info("✅ Mask+Cap 模型已初始化")

upload_queue = Queue(maxsize=1000)

# ⬅️ 全域變數，記錄前一次狀態
last_detection_status = None

def is_within_time_period():
    global last_detection_status
    current_time = datetime.now().time()
    periods = [
        ("00:30", "02:30"), ("02:40", "03:30"), ("04:00", "06:00"), ("06:10", "08:00"),
        ("08:30", "09:50"), ("10:00", "11:40"), ("12:10", "14:30"), ("14:40", "16:00"),
        ("16:30", "18:30"), ("19:00", "21:00"), ("21:10", "22:50"), ("23:00", "00:00"),
    ]

    now_status = False
    matched_period = None
    for start_str, end_str in periods:
        start = datetime.strptime(start_str, "%H:%M").time()
        end = datetime.strptime(end_str, "%H:%M").time()
        if start <= current_time <= end:
            now_status = True
            matched_period = f"{start_str}~{end_str}"
            break

    if last_detection_status != now_status:
        if now_status:
            logging.info(f"[時段檢查] ✅ 目前時間 {current_time.strftime('%H:%M:%S')} 屬於偵測啟用時段（{matched_period}）")
        else:
            logging.info(f"[時段檢查] ⏸️ 目前時間 {current_time.strftime('%H:%M:%S')} 不在偵測時段，暫停辨識")
        last_detection_status = now_status  # 更新狀態

    return now_status

# === 新增：取得目前狀態的函數 ===
def get_current_status():
    """取得目前是WORK還是SKIP狀態"""
    return "WORK" if is_within_time_period() else "SKIP"

# === 新增：計算頭部轉向角度的函數 ===
def calculate_head_angle(keypoints, frame_width=640, frame_height=480, confidence_threshold=0.5):
    """
    使用YOLO pose關鍵點計算頭部轉向角度，加入畫面位置補償
    YOLO pose關鍵點索引:
    0: 鼻尖, 1: 左眼, 2: 右眼, 3: 左耳, 4: 右耳
    """
    try:
        # 檢查關鍵點置信度
        if len(keypoints) < 5:
            return None, None, None
        
        # 取得頭部關鍵點座標 (x, y, confidence)
        nose = keypoints[0][:2] if keypoints[0][2] > confidence_threshold else None
        left_eye = keypoints[1][:2] if keypoints[1][2] > confidence_threshold else None
        right_eye = keypoints[2][:2] if keypoints[2][2] > confidence_threshold else None
        left_ear = keypoints[3][:2] if keypoints[3][2] > confidence_threshold else None
        right_ear = keypoints[4][:2] if keypoints[4][2] > confidence_threshold else None
        
        # 至少需要鼻子和雙眼
        if nose is None or left_eye is None or right_eye is None:
            return None, None, None
            
        # 計算頭部中心點
        head_center_x = (left_eye[0] + right_eye[0]) / 2
        head_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # 計算人物在畫面中的位置偏移
        center_offset_x = head_center_x - frame_width / 2
        center_offset_ratio = center_offset_x / (frame_width / 2)  # -1到1之間
        
        # 方法1：使用眼睛計算水平轉向角度（加入位置補償）
        horizontal_angle = None
        if left_eye is not None and right_eye is not None:
            # 計算雙眼連線與水平線的夾角
            eye_vector = np.array(right_eye) - np.array(left_eye)
            raw_angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
            
            # 位置補償：當人物偏離中心時，調整角度判斷
            position_compensation = center_offset_ratio * 15  # 最大補償15度
            horizontal_angle = raw_angle - position_compensation
            
        # 方法2：使用眼睛間距和鼻子位置判斷側面轉向
        face_asymmetry = None
        if left_eye is not None and right_eye is not None and nose is not None:
            # 計算鼻子到雙眼的距離比例
            left_eye_to_nose = np.linalg.norm(np.array(nose) - np.array(left_eye))
            right_eye_to_nose = np.linalg.norm(np.array(nose) - np.array(right_eye))
            
            if left_eye_to_nose + right_eye_to_nose > 0:
                # 正面時應該接近0，側面時會有明顯差異
                face_asymmetry = (right_eye_to_nose - left_eye_to_nose) / (left_eye_to_nose + right_eye_to_nose)
        
        # 方法3：使用耳朵和鼻子計算側面轉向程度（作為輔助）
        side_turn_ratio = None
        if left_ear is not None and right_ear is not None and nose is not None:
            nose_x = nose[0]
            left_ear_x = left_ear[0]
            right_ear_x = right_ear[0]
            
            # 計算鼻子相對於雙耳中點的偏移比例
            ear_center_x = (left_ear_x + right_ear_x) / 2
            ear_width = abs(right_ear_x - left_ear_x)
            
            if ear_width > 0:
                side_turn_ratio = (nose_x - ear_center_x) / ear_width
        
        return horizontal_angle, face_asymmetry, side_turn_ratio
        
    except Exception as e:
        logging.error(f"計算頭部角度時發生錯誤: {e}")
        return None, None, None

def is_head_turned(horizontal_angle, face_asymmetry, side_turn_ratio, 
                  threshold_angle=25, asymmetry_threshold=0.15, side_threshold=0.4):
    """
    判斷是否轉頭（使用多重判斷標準）
    threshold_angle: 水平角度閾值（度）- 降低以減少誤判
    asymmetry_threshold: 臉部不對稱閾值
    side_threshold: 側面轉向比例閾值
    """
    turned_indicators = 0
    total_indicators = 0
    
    # 檢查水平角度（主要指標）
    if horizontal_angle is not None:
        total_indicators += 1
        if abs(horizontal_angle) > threshold_angle:
            turned_indicators += 1
    
    # 檢查臉部不對稱度（更可靠的指標）
    if face_asymmetry is not None:
        total_indicators += 1
        if abs(face_asymmetry) > asymmetry_threshold:
            turned_indicators += 1
    
    # 檢查側面轉向比例（輔助指標）
    if side_turn_ratio is not None:
        total_indicators += 1
        if abs(side_turn_ratio) > side_threshold:
            turned_indicators += 1
    
    # 至少需要2個指標都判斷為轉頭才認定為轉頭
    if total_indicators >= 2:
        return turned_indicators >= 2
    elif total_indicators == 1:
        # 只有一個指標時，需要更嚴格的標準
        if horizontal_angle is not None and abs(horizontal_angle) > 35:
            return True
        if face_asymmetry is not None and abs(face_asymmetry) > 0.25:
            return True
            
    return False

class AlertCooldownManager:
    def __init__(self):
        self.cooldowns = alert_cooldowns.copy()
        self.last_times = {}

    def is_in_cooldown(self, camera_id, alert_type):
        key = f"{camera_id}_{alert_type}"
        now = time.time()
        last = self.last_times.get(key, 0)
        duration = self.cooldowns.get(alert_type, 300)
        return now - last < duration

    def update_last_time(self, camera_id, alert_type):
        key = f"{camera_id}_{alert_type}"
        self.last_times[key] = time.time()

    def get_remaining_time(self, camera_id, alert_type):
        key = f"{camera_id}_{alert_type}"
        last = self.last_times.get(key, 0)
        duration = self.cooldowns.get(alert_type, 300)
        return max(0, duration - (time.time() - last))
    
alert_cooldown_mgr = AlertCooldownManager()

class CameraMonitor:
    def __init__(self, config, index):
        self.config = config
        self.camera_index = index
        self.frame = np.ones((480, 640, 3), dtype=np.uint8)
        self.last_notification_time = 0
        self.eye_close_counter = 0
        self.head_turn_counter = 0
        self.l_eye_value = 0.0
        self.r_eye_value = 0.0
        self.head_angle = 0.0
        self.face_asymmetry = 0.0
        self.side_turn_ratio = 0.0
        self.running = True
        self.maskcap_model = maskcap_model
        self.missing_cap_count = 0
        self.missing_mask_count = 0
        
        # ✅ 新增：智能讀取模式選擇
        self.use_pyav = True  # 預設使用 PyAV
        self.pyav_error_count = 0  # PyAV 錯誤計數
        self.max_pyav_errors = 50  # PyAV 錯誤上限，超過則切換 OpenCV
        
        # 初始化 MediaPipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def reset_counter(self, alert_type):
        if alert_type == "EYES CLOSED":
            self.eye_close_counter = 0
        elif alert_type == "HEAD TURNED":
            self.head_turn_counter = 0
        elif alert_type == "MISSING CAP":
            self.missing_cap_count = 0
        elif alert_type == "MISSING MASK":
            self.missing_mask_count = 0

    def read_thread_func(self):
        """智能選擇讀取方法"""
        if self.use_pyav:
            try:
                self.read_with_pyav()
            except Exception as e:
                logging.error(f"❌ {self.config['location']} PyAV 完全失敗，切換到 OpenCV: {e}")
                self.use_pyav = False
                self.read_with_opencv()
        else:
            self.read_with_opencv()

    def read_with_pyav(self):
        """使用優化的 PyAV 讀取（含錯誤監控）"""
        retry_count = 0
        max_retry = 20

        while self.running and retry_count < max_retry:
            try:
                logging.info(f"📡 [PyAV] 連接攝影機 {self.config['camera_id']} ({self.config['location']})")
                
                # ✅ 優化後的 PyAV 設定 - 移除硬體加速避免 MB 錯誤
                container = av.open(self.config["rtsp_url"], options={
                    "rtsp_transport": "tcp",
                    "buffer_size": "1048576",        # 1MB 緩衝區
                    "max_delay": "500000",           # 最大延遲 500ms
                    "stimeout": "5000000",           # 連線超時 5s
                    "fflags": "nobuffer+fastseek+flush_packets",  # 降低延遲
                    "flags": "low_delay",            # 低延遲模式
                    "analyzeduration": "1000000",    # 分析時長 1s
                    "probesize": "1048576",         # 探測大小 1MB
                    "threads": "2"                   # 增加解碼線程
                    # ❌ 移除硬體加速避免 MB 解碼錯誤
                    # "hwaccel": "cuda",
                    # "hwaccel_output_format": "cuda",
                })
                
                retry_count = 0  # 成功開啟後重置計數
                frame_count = 0
                consecutive_decode_errors = 0
                max_consecutive_errors = 20  # 連續錯誤上限
                
                for frame in container.decode(video=0):
                    if not self.running:
                        break
                        
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        self.frame = cv2.resize(img, (640, 480))
                        frame_count += 1
                        consecutive_decode_errors = 0  # 成功則重置錯誤計數
                        
                    except Exception as decode_err:
                        consecutive_decode_errors += 1
                        self.pyav_error_count += 1
                        
                        # 只記錄前幾次錯誤，避免日誌洪水
                        if consecutive_decode_errors <= 3:
                            logging.warning(f"⚠️ {self.config['location']} 解碼失敗 ({consecutive_decode_errors}): {decode_err}")
                        
                        # ❌ 錯誤過多時切換到 OpenCV
                        if self.pyav_error_count >= self.max_pyav_errors:
                            logging.warning(f"🔄 {self.config['location']} PyAV 錯誤累計過多 ({self.pyav_error_count})，切換到 OpenCV")
                            self.use_pyav = False
                            container.close()
                            return self.read_with_opencv()
                        
                        # 連續錯誤過多時重新連線
                        if consecutive_decode_errors >= max_consecutive_errors:
                            logging.error(f"💥 {self.config['location']} 連續解碼錯誤過多，重新連線")
                            break
                            
                        continue

                    # ✅ 控制幀率，避免過度處理
                    time.sleep(0.033)  # 約30 FPS

                container.close()
                logging.info(f"📴 [PyAV] 鏡頭 {self.config['camera_id']} 結束串流 (共處理 {frame_count} 幀)")

            except Exception as e:
                retry_count += 1
                logging.error(f"❌ {self.config['location']} PyAV 接收錯誤 (嘗試 {retry_count}): {e}")
                
                # ✅ 根據錯誤類型調整重試策略
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    retry_interval = 3  # 網路問題快速重試
                else:
                    retry_interval = 8  # 其他問題延長間隔
                
                time.sleep(retry_interval)

        # PyAV 完全失敗，切換 OpenCV
        if retry_count >= max_retry:
            logging.warning(f"🔄 {self.config['location']} PyAV 重試次數耗盡，永久切換到 OpenCV")
            self.use_pyav = False
            self.read_with_opencv()

    def read_with_opencv(self):
        """使用 OpenCV 備用讀取方法"""
        retry_count = 0
        max_retry = 9999
        
        while self.running:
            cap = None
            try:
                logging.info(f"📡 [OpenCV備用] 連接攝影機 {self.config['camera_id']} ({self.config['location']})")
                
                # ✅ OpenCV 優化設定
                cap = cv2.VideoCapture(self.config["rtsp_url"], cv2.CAP_FFMPEG)
                
                # 設定緩衝區大小和幀率
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)
                
                # 檢查是否成功開啟
                if not cap.isOpened():
                    raise Exception("OpenCV 無法開啟 RTSP 串流")
                    
                retry_count = 0
                frame_count = 0
                consecutive_fails = 0
                max_consecutive_fails = 15

                while self.running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        consecutive_fails += 1
                        if consecutive_fails >= max_consecutive_fails:
                            logging.warning(f"⚠️ {self.config['location']} 連續讀取失敗 {consecutive_fails} 次，重新連線")
                            break
                        time.sleep(0.1)
                        continue
                        
                    consecutive_fails = 0
                    frame_count += 1
                    
                    # 調整畫面大小
                    self.frame = cv2.resize(frame, (640, 480))
                    
                    # 控制幀率
                    time.sleep(0.04)  # 約25 FPS

                logging.info(f"📴 [OpenCV備用] 結束串流 (共處理 {frame_count} 幀)")

            except Exception as e:
                retry_count += 1
                logging.error(f"❌ {self.config['location']} OpenCV 接收錯誤 (嘗試 {retry_count}): {e}")
                
            finally:
                if cap:
                    cap.release()

            if retry_count >= max_retry:
                logging.critical(f"⛔ {self.config['location']} OpenCV 也達到最大重試次數")
                break

            retry_interval = 5
            logging.info(f"🔄 {self.config['location']} OpenCV備用方案 {retry_interval}秒後重試")
            time.sleep(retry_interval)

    def process_thread_func(self):
        while self.running:
            frame = self.frame.copy()

            # ✅ 初始化每幀的檢測狀態
            cap_detected = False
            mask_detected = False
            mouth_detected = False

            if not is_within_time_period():
                time.sleep(1)
                continue
            try:
                # === 步驟 1：使用 MediaPipe 確認有臉 ===
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_detector.process(rgb)
                if not face_results.multi_face_landmarks:
                    self.head_turn_counter = 0
                    self.eye_close_counter = 0
                    self.missing_cap_count = 0
                    self.missing_mask_count = 0
                    time.sleep(0.1)
                    continue

                # === 步驟 2：YOLO Pose 取得主體 keypoints ===
                results = pose_model(frame, verbose=False)
                main_person = None
                min_pose_confidence = 0.90
                for r in results:
                    if r.keypoints is not None:
                        for kp in r.keypoints.data:
                            keypoints = kp.cpu().numpy()
                            high_conf_pts = [pt for pt in keypoints if pt[2] >= min_pose_confidence]
                            if len(high_conf_pts) >= 5:
                                main_person = keypoints
                                break
                    if main_person is not None:
                        break

                if main_person is None:
                    self.head_turn_counter = 0
                    self.eye_close_counter = 0
                    self.missing_cap_count = 0
                    self.missing_mask_count = 0
                    time.sleep(0.1)
                    continue

                # === 頭部角度判斷 ===
                horizontal_angle, face_asymmetry, side_turn_ratio = calculate_head_angle(main_person, 640, 480)
                self.head_angle = horizontal_angle or 0.0
                self.face_asymmetry = face_asymmetry or 0.0
                self.side_turn_ratio = side_turn_ratio or 0.0
                head_turned_current_frame = is_head_turned(horizontal_angle, face_asymmetry, side_turn_ratio)
                if head_turned_current_frame:
                    if not alert_cooldown_mgr.is_in_cooldown(self.config["camera_id"], "HEAD TURNED"):
                        self.head_turn_counter += 1
                    else:
                        logging.debug("👀 HEAD TURNED 冷卻中，跳過累計")
                else:
                    self.head_turn_counter = 0

                eyes_closed_current_frame = False
                min_valid_eye_open = 0.005  # 最小有效距離：小於此值視為遮擋或偵測異常

                for lm in face_results.multi_face_landmarks:
                    l_eye = abs(lm.landmark[145].y - lm.landmark[159].y)
                    r_eye = abs(lm.landmark[374].y - lm.landmark[386].y)
                    self.l_eye_value = l_eye
                    self.r_eye_value = r_eye

                    # ✅ 僅當雙眼皆低於設定門檻，且大於最小偵測值，才視為閉眼
                    if (
                        min_valid_eye_open < l_eye < self.config["eye_close_threshold"] and
                        min_valid_eye_open < r_eye < self.config["eye_close_threshold"]
                    ):
                        eyes_closed_current_frame = True
                    else:
                        eyes_closed_current_frame = False

                if eyes_closed_current_frame:
                    if not alert_cooldown_mgr.is_in_cooldown(self.config["camera_id"], "EYES CLOSED"):
                        self.eye_close_counter += 1
                    else:
                        logging.debug("😴 EYES CLOSED 冷卻中，跳過累計")
                else:
                    self.eye_close_counter = 0

                # === 畫出關鍵點 ===
                self.draw_keypoints(frame, main_person)

                # === YOLO Cap+Mask 模型推論 ===
                maskcap_results = self.maskcap_model(frame, verbose=False)
                for r in maskcap_results:
                    if hasattr(r, "boxes") and r.boxes is not None:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            if cls_id == 0 and conf >= 0.5:  # cap
                                cap_detected = True
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Cap {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            elif cls_id == 1 and conf >= 0.5:  # mask
                                mask_detected = True
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(frame, f"Mask {conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            elif cls_id == 4 and conf >= 0.5:  # mouth
                                mouth_detected = True

                # ✅ 改寫後：只有 mouth 可見 且 mask 未偵測 才算沒戴口罩
                if mouth_detected and not mask_detected:
                    if not alert_cooldown_mgr.is_in_cooldown(self.config["camera_id"], "MISSING MASK"):
                        self.missing_mask_count += 1
                else:
                    self.missing_mask_count = 0

                if cap_detected:
                    self.missing_cap_count = 0
                else:
                    if not alert_cooldown_mgr.is_in_cooldown(self.config["camera_id"], "MISSING CAP"):
                        self.missing_cap_count += 1

                # === 警報條件 ===
                alert_types = [
                    ("EYES CLOSED", self.eye_close_counter, self.config["close_threshold_frames"]),
                    ("HEAD TURNED", self.head_turn_counter, self.config["head_turn_frames"]),
                    ("MISSING CAP", self.missing_cap_count, self.config["missing_cap_frames"]),
                    ("MISSING MASK", self.missing_mask_count, self.config["missing_mask_frames"]),
                ]

                for alert_type, counter, threshold in alert_types:
                    if counter >= threshold:
                        if not alert_cooldown_mgr.is_in_cooldown(self.config["camera_id"], alert_type):
                            alert_cooldown_mgr.update_last_time(self.config["camera_id"], alert_type)
                            self.reset_counter(alert_type)
                            self.take_screenshot(frame, alert_type)

                # === 畫面提示 ===
                if eyes_closed_current_frame:
                    cv2.putText(frame, "EYES CLOSED", (5, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if head_turned_current_frame:
                    cv2.putText(frame, "HEAD TURNED", (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                cv2.putText(frame, f"Mask Detected: {int(mask_detected)}", (5, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"cap Detected: {int(cap_detected)}", (5, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if mouth_detected and not mask_detected:
                    cv2.putText(frame, "Mouth+No Mask", (5, 480),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                logging.debug(
                    f"[{self.config['location']}] EyeCheck: L={l_eye:.4f}, R={r_eye:.4f}, Threshold={self.config['eye_close_threshold']}, Counter={self.eye_close_counter}"
                )    

            except Exception as e:
                logging.error(f"[{self.config['location']}] Frame processing error: {e}")
                traceback_str = traceback.format_exc()
                logging.error(traceback_str)

            time.sleep(0.1)
    
    def draw_keypoints(self, frame, keypoints, confidence_threshold=0.5):
        """在畫面上繪製 YOLO pose 關鍵點與骨架連線"""
        try:
            # === COCO 骨架關節連線對應表（Ultralytics 預設順序） ===
            skeleton_pairs = [
                (5, 7), (7, 9),      # 左臂
                (6, 8), (8, 10),     # 右臂
                (11, 13), (13, 15),  # 左腿
                (12, 14), (14, 16),  # 右腿
                (5, 6),              # 雙肩
                (11, 12),            # 髖部
                (5, 11), (6, 12),    # 軀幹
                (0, 1), (0, 2),      # 鼻尖到雙眼
                (1, 3), (2, 4),      # 雙眼到雙耳
            ]

            # === 畫出骨架線段 ===
            for i, j in skeleton_pairs:
                if (
                    i < len(keypoints) and j < len(keypoints) and
                    keypoints[i][2] > confidence_threshold and
                    keypoints[j][2] > confidence_threshold
                ):
                    pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                    pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            # === 畫出頭部關鍵點（含名稱） ===
            head_points = {
                0: ("Nose", (0, 255, 0)),      # 鼻尖 - 綠
                1: ("L_Eye", (255, 0, 0)),     # 左眼 - 藍
                2: ("R_Eye", (255, 0, 0)),     # 右眼 - 藍
                3: ("L_Ear", (0, 255, 255)),   # 左耳 - 黃
                4: ("R_Ear", (0, 255, 255)),   # 右耳 - 黃
            }

            for idx, (name, color) in head_points.items():
                if idx < len(keypoints) and keypoints[idx][2] > confidence_threshold:
                    x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                    cv2.circle(frame, (x, y), 3, color, -1)
                    cv2.putText(frame, name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        except Exception as e:
            logging.error(f"繪製關鍵點時發生錯誤: {e}")

    def display_thread_func(self):
        while self.running:
            disp = self.frame.copy()

            # === 新增：顯示WORK/SKIP狀態 ===
            current_status = get_current_status()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 根據狀態設定顏色
            if current_status == "WORK":
                status_color = (0, 255, 0)  # 綠色
                bg_color = (0, 100, 0)      # 深綠色背景
            else:
                status_color = (0, 255, 255)  # 黃色
                bg_color = (0, 100, 100)      # 深黃色背景
            
            # 繪製狀態背景
            cv2.rectangle(disp, (450, 10), (630, 50), bg_color, -1)
            cv2.rectangle(disp, (450, 10), (630, 50), status_color, 2)
            
            # 顯示狀態文字
            cv2.putText(disp, f"Status: {current_status}", (465, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(disp, f"Time: {current_time}", (465, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)

            # 顯示使用的解碼方式
            decode_method = "PyAV" if self.use_pyav else "OpenCV"
            cv2.putText(disp, f"Decoder: {decode_method}", (450, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 顯示眼部資訊
            cv2.putText(disp, f"L Eye: {self.l_eye_value:.3f}  R Eye: {self.r_eye_value:.3f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 顯示頭部轉向資訊
            cv2.putText(disp, f"Head Angle: {self.head_angle:.1f}deg", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(disp, f"Face Asym: {self.face_asymmetry:.3f}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(disp, f"Side Turn: {self.side_turn_ratio:.2f}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 顯示計數器
            cv2.putText(disp, f"Eye Close: {self.eye_close_counter}/{self.config['close_threshold_frames']}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(disp, f"Head Turn: {self.head_turn_counter}/{self.config['head_turn_frames']}", 
                       (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow(self.config["location"], disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyWindow(self.config["location"])

    def take_screenshot(self, annotated_image, alert_type):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        date_folder = datetime.now().strftime("%Y%m%d")
        unique_id = uuid.uuid4().hex[:6]

        # 📁 建立 capture/YYYYMMDD 路徑
        folder_path = os.path.join("capture", date_folder)
        os.makedirs(folder_path, exist_ok=True)

        # === 儲存乾淨畫面（未繪製）===
        filename_clean = f"clean_{self.config['camera_id']}_{self.config['location']}_{timestamp}_{unique_id}_{alert_type}.png"
        filepath_clean = os.path.join(folder_path, filename_clean)
        clean_image = self.frame.copy()
        try:
            cv2.imwrite(filepath_clean, clean_image)
        except Exception as e:
            logging.error(f"❌ 儲存 clean 圖失敗：{e}")
            return

        # === 不再儲存 annotated 圖，但保留於記憶體中供轉成 JPG 上傳 ===

        try:
            with logfile_lock:
                with open("FocusDetectionLog.txt", "a") as f:
                    f.write(f"{timestamp}, {self.config['location']}, {filepath_clean}, {alert_type}\n")
        except Exception as e:
            logging.warning(f"⚠️ 寫入日誌失敗：{e}")

        try:
            upload_queue.put((annotated_image.copy(), self.config, alert_type))  # ✅ 傳 frame，而不是 file path
            logging.info(f"📷 {self.config['location']}：{alert_type} 已截圖並加入上傳佇列")
        except Exception as e:
            logging.error(f"❌ 加入佇列失敗: {e}")

def upload_worker():
    valid_result_msgs = {
        "HEAD TURNED": "檢測到持續轉頭行為，疑似不專注狀況",
        "EYES CLOSED": "檢測到持續閉眼行為，疑似不專注狀況",
        "MISSING CAP": "疑似未戴無塵帽，請同仁儘速查看",
        "MISSING MASK": "疑似未戴口罩，請同仁儘速查看",
    }

    while True:
        annotated_image, config, alert_type = upload_queue.get()
        try:
            # 🕒 建立時間與路徑
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            date_folder = datetime.now().strftime("%Y%m%d")
            unique_id = uuid.uuid4().hex[:6]
            folder_path = os.path.join("capture", date_folder)
            os.makedirs(folder_path, exist_ok=True)

            # 📸 儲存為 JPG
            filename_jpg = f"screenshot_{config['camera_id']}_{config['location']}_{timestamp}_{unique_id}_{alert_type}.jpg"
            jpg_path = os.path.join(folder_path, filename_jpg)
            cv2.imwrite(jpg_path, annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # ✅ 嚴格檢查 alert_type 是否合規
            if alert_type not in valid_result_msgs:
                logging.error(f"⛔ 不明警報類型：{alert_type}，請檢查程式邏輯與來源！")
                result_msg = f"[錯誤] 未知警報類型：{alert_type}"
            else:
                result_msg = valid_result_msgs[alert_type]

            # 📝 準備 POST 參數
            model = {
                "cameraId": config["camera_id"],
                "location": config["location"],
                "eventName": "專注度辨識",
                "eventDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "notes": alert_type,
                "fileName": os.path.basename(jpg_path),
                "result": result_msg
            }

            # 📤 執行上傳
            with open(jpg_path, "rb") as f:
                files = {"files": (os.path.basename(jpg_path), f, 'image/jpeg')}
                r = requests.post(config["upload_url"], data=model, files=files, verify=False, timeout=10)

                if r.status_code == 200:
                    logging.info(f"📤 上傳成功：{config['location']} | {alert_type} | {result_msg}")
                else:
                    logging.warning(f"⚠️ 上傳失敗：{config['location']} | HTTP {r.status_code} | {alert_type}")
        except Exception as e:
            logging.error(f"❌ 上傳錯誤（{alert_type}）：{e}")
        finally:
            upload_queue.task_done()

def check_camera_signal(rtsp_url, timeout=5):
    """
    嘗試開啟 RTSP 串流確認有無影像回傳
    """
    try:
        container = av.open(rtsp_url, timeout=timeout, options={"rtsp_transport": "tcp", "threads": "1"})
        for frame in container.decode(video=0):
            if frame:
                container.close()
                return True
        container.close()
    except Exception as e:
        logging.warning(f"❌ RTSP 預檢失敗: {rtsp_url} | 原因: {e}")
    return False     

def main():
    # 啟動上傳背景執行緒
    upload_thread = threading.Thread(target=upload_worker, daemon=True)
    upload_thread.start()

    camera_monitors = []
    valid_configs = []

    # === RTSP 預檢階段 ===
    for idx, cam_config in enumerate(cameras_config):
        rtsp_url = cam_config["rtsp_url"]
        location = cam_config.get("location", f"Cam-{idx}")

        logging.info(f"🔍 預檢攝影機連線：{location}")
        try:
            if check_camera_signal(rtsp_url):
                valid_configs.append(cam_config)
                logging.info(f"✅ 通過：{location}")
            else:
                logging.warning(f"🚫 無法連線，略過：{location}")
        except Exception as e:
            logging.error(f"❌ 預檢例外：{location} | {e}")

    logging.info(f"🟢 啟動成功攝影機數量：{len(valid_configs)} / {len(cameras_config)}")

    if not valid_configs:
        logging.critical("⛔ 沒有可用的攝影機，系統結束")
        return

    # === 初始化 CameraMonitor 與啟動執行緒 ===
    for idx, cam_config in enumerate(valid_configs):
        monitor = CameraMonitor(cam_config, idx)
        camera_monitors.append(monitor)

        try:
            threading.Thread(target=monitor.read_thread_func, daemon=True, name=f"{cam_config['camera_id']}-reader").start()
            threading.Thread(target=monitor.process_thread_func, daemon=True, name=f"{cam_config['camera_id']}-processor").start()
            # 若需開啟畫面：
            #threading.Thread(target=monitor.display_thread_func, daemon=True, name=f"{cam_config['camera_id']}-display").start()
        except Exception as e:
            logging.error(f"❌ 執行緒啟動失敗：{cam_config['location']} | {e}")

    print(f"🔍 實際啟用 {len(valid_configs)} 台攝影機（原始設定數: {len(cameras_config)}）")
    print("📁 攝影機設定從 cameras_config.txt 讀取")
    print("📱 畫面上會顯示目前的 WORK/SKIP 狀態")
    print("🔄 自動偵測解碼問題並切換至穩定模式")
    print("📌 按 'q' 可關閉任一視窗")

    try:
        # 主程式持續運行 + 監控執行緒狀態
        while True:
            live_threads = [t.name for t in threading.enumerate() if t.name != "MainThread"]
            logging.debug(f"🧵 活動中的執行緒：{len(live_threads)} | {live_threads}")
            time.sleep(10)
    except KeyboardInterrupt:
        print("🛑 中斷執行")
    except Exception as e:
        logging.critical(f"💥 主執行緒異常終止：{e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"💥 執行錯誤：{e}", exc_info=True)