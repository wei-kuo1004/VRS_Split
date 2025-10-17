import cv2
import av
import time
import logging
import threading
import numpy as np
import mediapipe as mp
import traceback
from datetime import datetime

from models.alert_cooldown import AlertCooldownManager
from utils.schedule_checker import is_within_time_period, get_current_status
from utils.head_angle import calculate_head_angle
from utils.helpers import safe_mkdir, get_timestamp, uuid_suffix
from utils.uploader import upload_queue

# === 警報冷卻設定 ===
ALERT_COOLDOWNS = {
    "EYES CLOSED": 300,
    "HEAD TURNED": 300,
    "MISSING CAP": 600,
    "MISSING MASK": 600,
}

class CameraMonitor:
    def __init__(self, config, index, pose_model, maskcap_model):
        self.config = config
        self.camera_index = index
        self.pose_model = pose_model
        self.maskcap_model = maskcap_model
        self.frame = np.ones((480, 640, 3), dtype=np.uint8)
        self.running = True

        self.eye_close_counter = 0
        self.head_turn_counter = 0
        self.missing_cap_count = 0
        self.missing_mask_count = 0

        self.l_eye_value = 0.0
        self.r_eye_value = 0.0
        self.head_angle = 0.0
        self.face_asymmetry = 0.0
        self.side_turn_ratio = 0.0

        self.use_pyav = True
        self.pyav_error_count = 0
        self.max_pyav_errors = 30

        self.cooldown_mgr = AlertCooldownManager(ALERT_COOLDOWNS)

        mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # =============================
    # 主流程區：讀取與處理
    # =============================

    def read_thread_func(self):
        """自動選擇 PyAV 或 OpenCV 串流方式"""
        if self.use_pyav:
            try:
                self.read_with_pyav()
            except Exception as e:
                logging.error(f"PyAV 發生錯誤，切換 OpenCV：{e}")
                self.use_pyav = False
                self.read_with_opencv()
        else:
            self.read_with_opencv()

    def read_with_pyav(self):
        retry = 0
        while self.running:
            try:
                logging.info(f"📡 [PyAV] 連線攝影機 {self.config['location']}")
                container = av.open(self.config["rtsp_url"], options={
                    "rtsp_transport": "tcp",
                    "buffer_size": "1048576",
                    "max_delay": "500000",
                    "stimeout": "5000000",
                    "flags": "low_delay",
                    "fflags": "nobuffer+fastseek",
                })

                for frame in container.decode(video=0):
                    if not self.running:
                        break
                    try:
                        img = frame.to_ndarray(format="bgr24")
                        self.frame = cv2.resize(img, (640, 480))
                    except Exception as e:
                        self.pyav_error_count += 1
                        if self.pyav_error_count > self.max_pyav_errors:
                            logging.warning(f"PyAV 錯誤過多，切換 OpenCV")
                            self.use_pyav = False
                            container.close()
                            return self.read_with_opencv()
                        continue

                    time.sleep(0.03)
                container.close()
            except Exception as e:
                retry += 1
                logging.warning(f"PyAV 鏡頭錯誤 (第{retry}次)：{e}")
                time.sleep(3)

    def read_with_opencv(self):
        while self.running:
            try:
                cap = cv2.VideoCapture(self.config["rtsp_url"], cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 25)

                if not cap.isOpened():
                    raise Exception("OpenCV 無法開啟 RTSP 串流")

                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.2)
                        continue
                    self.frame = cv2.resize(frame, (640, 480))
                    time.sleep(0.04)

            except Exception as e:
                logging.error(f"OpenCV 讀取錯誤：{e}")
                time.sleep(5)
            finally:
                if cap:
                    cap.release()

    # =============================
    # 辨識主邏輯
    # =============================

    def process_thread_func(self):
        while self.running:
            frame = self.frame.copy()

            if not is_within_time_period():
                time.sleep(1)
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_results = self.face_detector.process(rgb)

                if not face_results.multi_face_landmarks:
                    self.reset_counters()
                    time.sleep(0.1)
                    continue

                # --- YOLO Pose ---
                results = self.pose_model(frame, verbose=False)
                main_person = None
                for r in results:
                    if r.keypoints is not None:
                        for kp in r.keypoints.data:
                            keypoints = kp.cpu().numpy()
                            if np.count_nonzero(keypoints[:, 2] > 0.8) >= 5:
                                main_person = keypoints
                                break
                    if main_person is not None:
                        break

                if main_person is None:
                    self.reset_counters()
                    continue

                # --- 頭部角度計算 ---
                h_angle, asym, side = calculate_head_angle(main_person)
                self.head_angle = h_angle or 0
                self.face_asymmetry = asym or 0
                self.side_turn_ratio = side or 0
                head_turned = self.is_head_turned(h_angle, asym, side)

                if head_turned:
                    if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "HEAD TURNED"):
                        self.head_turn_counter += 1
                    else:
                        logging.debug("HEAD TURNED 冷卻中")
                else:
                    self.head_turn_counter = 0

                # --- MediaPipe 眼睛閉合判斷 ---
                eyes_closed = False
                for lm in face_results.multi_face_landmarks:
                    l_eye = abs(lm.landmark[145].y - lm.landmark[159].y)
                    r_eye = abs(lm.landmark[374].y - lm.landmark[386].y)
                    self.l_eye_value = l_eye
                    self.r_eye_value = r_eye

                    if 0.005 < l_eye < self.config["eye_close_threshold"] and \
                       0.005 < r_eye < self.config["eye_close_threshold"]:
                        eyes_closed = True

                if eyes_closed:
                    if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "EYES CLOSED"):
                        self.eye_close_counter += 1
                else:
                    self.eye_close_counter = 0

                # --- YOLO Mask+Cap ---
                maskcap_results = self.maskcap_model(frame, verbose=False)
                cap_detected = False
                mask_detected = False
                mouth_detected = False

                for r in maskcap_results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if cls_id == 0 and conf >= 0.5:
                            cap_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        elif cls_id == 1 and conf >= 0.5:
                            mask_detected = True
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        elif cls_id == 4 and conf >= 0.5:
                            mouth_detected = True

                # --- 口罩與帽子邏輯 ---
                if mouth_detected and not mask_detected:
                    if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "MISSING MASK"):
                        self.missing_mask_count += 1
                else:
                    self.missing_mask_count = 0

                if cap_detected:
                    self.missing_cap_count = 0
                else:
                    if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "MISSING CAP"):
                        self.missing_cap_count += 1

                # --- 觸發條件 ---
                self.check_alerts(frame)

            except Exception as e:
                logging.error(f"[{self.config['location']}] 處理錯誤：{e}")
                logging.debug(traceback.format_exc())
            time.sleep(0.1)

    # =============================
    # 警報邏輯
    # =============================

    def check_alerts(self, frame):
        alerts = [
            ("EYES CLOSED", self.eye_close_counter, self.config["close_threshold_frames"]),
            ("HEAD TURNED", self.head_turn_counter, self.config["head_turn_frames"]),
            ("MISSING CAP", self.missing_cap_count, self.config["missing_cap_frames"]),
            ("MISSING MASK", self.missing_mask_count, self.config["missing_mask_frames"]),
        ]
        for a_type, counter, threshold in alerts:
            if counter >= threshold:
                if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], a_type):
                    self.cooldown_mgr.update(self.config["camera_id"], a_type)
                    self.reset_counter(a_type)
                    self.take_screenshot(frame, a_type)

    def take_screenshot(self, frame, alert_type):
        date_folder = datetime.now().strftime("%Y%m%d")
        folder = f"capture/{date_folder}"
        safe_mkdir(folder)

        filename = f"{self.config['camera_id']}_{get_timestamp()}_{uuid_suffix()}_{alert_type}.png"
        path = f"{folder}/{filename}"
        cv2.imwrite(path, frame)
        logging.info(f"📷 {self.config['location']} {alert_type} 截圖完成")

        upload_queue.put((frame.copy(), self.config, alert_type))

    def reset_counter(self, alert_type):
        if alert_type == "EYES CLOSED":
            self.eye_close_counter = 0
        elif alert_type == "HEAD TURNED":
            self.head_turn_counter = 0
        elif alert_type == "MISSING CAP":
            self.missing_cap_count = 0
        elif alert_type == "MISSING MASK":
            self.missing_mask_count = 0

    def reset_counters(self):
        self.eye_close_counter = 0
        self.head_turn_counter = 0
        self.missing_cap_count = 0
        self.missing_mask_count = 0

    # =============================
    # 頭部轉動判斷
    # =============================

    def is_head_turned(self, angle, asym, side):
        turned = 0
        total = 0
        if angle is not None:
            total += 1
            if abs(angle) > 25:
                turned += 1
        if asym is not None:
            total += 1
            if abs(asym) > 0.15:
                turned += 1
        if side is not None:
            total += 1
            if abs(side) > 0.4:
                turned += 1
        return turned >= 2 if total >= 2 else abs(angle or 0) > 35

    # =============================
    # 顯示畫面資訊
    # =============================

    def display_thread_func(self):
        while self.running:
            frame = self.frame.copy()
            status = get_current_status()
            now = datetime.now().strftime("%H:%M:%S")
            color = (0, 255, 0) if status == "WORK" else (0, 255, 255)

            cv2.rectangle(frame, (450, 10), (630, 50), (0, 100, 0), -1)
            cv2.putText(frame, f"Status: {status}", (460, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Time: {now}", (460, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, f"L_eye:{self.l_eye_value:.3f} R_eye:{self.r_eye_value:.3f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Head:{self.head_angle:.1f} Asym:{self.face_asymmetry:.3f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow(self.config["location"], frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break
        cv2.destroyWindow(self.config["location"])
