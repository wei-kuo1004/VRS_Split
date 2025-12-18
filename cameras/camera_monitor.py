# cameras/camera_monitor.py
import cv2
import av
import time
import logging
import numpy as np
import mediapipe as mp
import traceback
from datetime import datetime

from models.alert_cooldown import AlertCooldownManager
from utils.schedule_checker import is_within_time_period, get_current_status
from utils.head_angle import calculate_head_angle
from utils.helpers import safe_mkdir, get_timestamp, uuid_suffix
from utils.uploader import upload_queue
import screeninfo

# === 警報冷卻設定 ===
ALERT_COOLDOWNS = {
    "EYES CLOSED": 300,
    "HEAD TURNED": 300,
    "MISSING CAP": 300,
    "MISSING MASK": 300,
}


class CameraMonitor:
    def __init__(self, config, index, pose_model, maskcap_model,n_model=None):
        self.config = config
        self.camera_index = index
        self.pose_model = pose_model
        self.maskcap_model = maskcap_model
        self.n_model = n_model

        self.frame = np.ones((480, 640, 3), dtype=np.uint8)
        self.display_frame = self.frame.copy()
        self.running = True

        # 狀態
        self.eye_close_counter = 0
        self.head_turn_counter = 0
        self.missing_cap_count = 0
        self.missing_mask_count = 0

        # 即時數值
        self.l_eye_value = 0.0
        self.r_eye_value = 0.0
        self.head_angle = 0.0
        self.face_asymmetry = 0.0
        self.side_turn_ratio = 0.0

        # 串流
        self.use_pyav = True
        self.pyav_error_count = 0
        self.max_pyav_errors = 30

        self.cooldown_mgr = AlertCooldownManager(ALERT_COOLDOWNS)
        self.face_detector = None

    # =============================
    # 串流讀取
    # =============================

    def read_thread_func(self):
        if self.use_pyav:
            try:
                self.read_with_pyav()
            except Exception as e:
                logging.error(f"[{self.config['camera_id']}] PyAV 發生錯誤，改用 OpenCV：{e}")
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
                    img = frame.to_ndarray(format="bgr24")
                    self.frame = cv2.resize(img, (640, 480))
                    time.sleep(0.03)
                container.close()
            except Exception as e:
                retry += 1
                logging.warning(f"[{self.config['camera_id']}] PyAV 連線錯誤 (第{retry}次)：{e}")
                time.sleep(3)

    def read_with_opencv(self):
        while self.running:
            cap = None
            try:
                logging.info(f"📡 [OpenCV] 連線攝影機 {self.config['location']}")
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
                logging.error(f"[{self.config['camera_id']}] OpenCV 錯誤：{e}")
                time.sleep(5)
            finally:
                if cap:
                    cap.release()

    # =============================
    # 主推論流程
    # =============================

    def process_thread_func(self):
        if self.face_detector is None:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_detector = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logging.info(f"[{self.config['camera_id']}] Mediapipe 初始化完成")

        while self.running:
            try:
                base = self.frame.copy()
                vis = base.copy()
                self.draw_status_header(vis)

                if not is_within_time_period():
                    self.display_frame = vis
                    time.sleep(0.5)
                    continue

                # ====== MediaPipe 臉部 ======
                rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
                face_results = self.face_detector.process(rgb)

                has_face = face_results and face_results.multi_face_landmarks
                if has_face:
                    lm = face_results.multi_face_landmarks[0]
                    self.draw_face_landmarks(vis, lm)

                    l_eye = abs(lm.landmark[145].y - lm.landmark[159].y)
                    r_eye = abs(lm.landmark[374].y - lm.landmark[386].y)
                    self.l_eye_value = l_eye
                    self.r_eye_value = r_eye

                    eyes_closed = (
                        l_eye < self.config["eye_close_threshold"] and
                        r_eye < self.config["eye_close_threshold"]
                    )
                    if eyes_closed:
                        if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "EYES CLOSED"):
                            self.eye_close_counter += 1
                    else:
                        self.eye_close_counter = 0
                else:
                    # ===== 若未偵測到人臉，半透明清空畫面以避免鬼影 =====
                    #vis = cv2.addWeighted(base, 0.3, np.zeros_like(base), 0.7, 0)
                    #self.reset_counters_face()
                    #self.display_frame = vis
                    #time.sleep(0.1)
                    #continue
                    self.reset_counters_face()
                    # vis 保持為 base（不改變）

                # ====== YOLO Pose ======
                pose_results = self.pose_model(base, verbose=False)
                main_person = None
                for r in pose_results:
                    if r.keypoints is not None:
                        for kp in r.keypoints.data:
                            keypoints = kp.cpu().numpy()
                            if np.count_nonzero(keypoints[:, 2] > 0.8) >= 5:
                                main_person = keypoints
                                break
                    if main_person is not None:
                        break

                if main_person is not None:
                    self.draw_pose_keypoints(vis, main_person)
                    h_angle, asym, side = calculate_head_angle(main_person)
                    self.head_angle = h_angle or 0.0
                    self.face_asymmetry = asym or 0.0
                    self.side_turn_ratio = side or 0.0

                    head_turned = self.is_head_turned(h_angle, asym, side)
                    if head_turned:
                        if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "HEAD TURNED"):
                            self.head_turn_counter += 1
                    else:
                        self.head_turn_counter = 0
                else:
                    # ===== 若未偵測到姿勢，半透明清空畫面以避免上一幀殘留 =====
                    #vis = cv2.addWeighted(base, 0.1, np.zeros_like(base), 0.7, 0)
                    #self.reset_counters_pose()
                    #self.display_frame = vis
                    # 若未偵測到姿勢：不再壓暗畫面
                    # 只重置 pose 相關計數，保留 vis 為原始畫面，之後由 n_model 判定是否為 person
                    self.reset_counters_pose()

                # ============================================================
                # 🧠 改進版防呆條件：若未偵測到 person (class_id == 0)，直接跳過
                # person 的 class_id = 0，需達到最低信心值才算有效偵測
                # ============================================================
                # 使用 yolo11n 作 person gate（若有提供）
                person_detected = False
                person_conf_thr = 0.50
                if self.n_model is not None:
                    try:
                        n_results = self.n_model(base, conf=person_conf_thr, verbose=False)
                        # 只檢查並繪製 person (class_id == 0)
                        person_detected = self.draw_persons(vis, n_results, conf_thr=person_conf_thr)
                        logging.debug(f"[{self.config.get('camera_id','?')}] n_model person_detected={person_detected}")
                    except Exception as e:
                        logging.warning(f"[{self.config.get('camera_id','?')}] n_model 偵測錯誤：{e}")
                else:
                    # fallback: 使用原本的 pose-based 判定（若 main_person 已偵測到視為有人）
                    # 若你原本有更複雜的 pose 判定邏輯，可在此保留/複製過來
                    if main_person is not None:
                        person_detected = True

                if not person_detected:
                    # 若沒有偵測到 person，畫面先押暗，重置口罩/安全帽計數並跳過後續檢測
                    vis = cv2.addWeighted(base, 0.4, np.zeros_like(base), 0.2, 0)
                    self.missing_mask_count = 0
                    self.missing_cap_count = 0
                    self.display_frame = vis
                    time.sleep(0.1)
                    continue

                # ====== YOLO Mask/Cap ======
                cap_detected = False
                mask_detected = False
                mouth_detected = False
                mask_conf = 0.0  # 初始化信心值

                maskcap_results = self.maskcap_model(base, conf=0.25, iou=0.7, verbose=False)
                for r in maskcap_results:
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label_name = self.maskcap_model.names[cls_id].lower()

                        # ===== 類別邏輯與信心門檻 =====
                        if "cap" in label_name and conf >= 0.55:
                            cap_detected = True
                            color = (0, 255, 0)  # 綠色
                        elif "mask" in label_name:
                            mask_detected = True
                            mask_conf = conf
                            color = (255, 255, 255)  # 白色
                        elif "mouth" in label_name and conf >= 0.55:
                            mouth_detected = True
                            color = (0, 0, 255)  # 紅色
                        else:
                            color = (255, 255, 0)  # 黃色作為預設

                        self.draw_box(vis, (x1, y1, x2, y2), f"{label_name} {conf:.2f}", color)

                # ============================================================
                # 🔧 改進版邏輯：若嘴巴偵測到且口罩信心度低，視為「未戴口罩」
                # ============================================================
                #if mouth_detected and (not mask_detected or mask_conf < 0.8):
                if mouth_detected:
                    if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "MISSING MASK"):
                        self.missing_mask_count += 1
                else:
                    self.missing_mask_count = 0

                # ============================================================
                # 缺帽邏輯：cap 信心值需 >= 0.55 才算有戴
                # ============================================================
                if not cap_detected:
                    if not self.cooldown_mgr.is_in_cooldown(self.config["camera_id"], "MISSING CAP"):
                        self.missing_cap_count += 1
                else:
                    self.missing_cap_count = 0


                # ====== 警報觸發 ======
                self.check_alerts(vis)
                self.display_frame = vis

            except Exception as e:
                logging.error(f"[{self.config['camera_id']}] 推論錯誤：{e}")
                logging.debug(traceback.format_exc())
            time.sleep(0.08)

    # =============================
    # 顯示畫面
    # =============================

    def display_thread_func(self):
        win_name = self.config["location"]

        # === Step 1. 取得螢幕解析度 ===
        screen = screeninfo.get_monitors()[0]
        screen_w, screen_h = screen.width, screen.height

        # === Step 2. 自動計算行列與每個視窗大小 ===
        total_cameras = self.config.get("total_cameras", 16)  # 若主程式可提供總攝影機數，可傳入此參數
        max_cols = min(4, total_cameras)                      # 每行最多 4 個，可依需求調整
        rows = (total_cameras + max_cols - 1) // max_cols     # 自動換行數
        aspect_ratio = 4 / 3                                  # 保持監控畫面比例
        margin_x, margin_y = 10, 10                           # 視窗間距

        # 每個視窗寬高（含邊距）
        win_w = int((screen_w - (max_cols + 1) * margin_x) / max_cols)
        win_h = int(win_w / aspect_ratio)

        # 若總高度超過螢幕，則縮小比例
        total_height = rows * (win_h + margin_y)
        if total_height > screen_h:
            scale = screen_h / total_height
            win_w = int(win_w * scale)
            win_h = int(win_h * scale)

        # === Step 3. 計算當前攝影機座標 ===
        row = self.camera_index // max_cols
        col = self.camera_index % max_cols
        x = margin_x + col * (win_w + margin_x)
        y = margin_y + row * (win_h + margin_y)

        # === Step 4. 建立視窗並移動位置 ===
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, win_w, win_h)
        cv2.moveWindow(win_name, x, y)

        logging.info(
            f"[{self.config['camera_id']}] 顯示 → ({x},{y}) "
            f"大小 {win_w}x{win_h}, 螢幕 {screen_w}x{screen_h}"
        )

        # === Step 5. 顯示主迴圈 ===
        while self.running:
            try:
                disp = self.display_frame.copy()
                self.draw_status_footer(disp)
                cv2.imshow(win_name, disp)
                if cv2.waitKey(2) & 0xFF == ord("q"):
                    self.running = False
                    break
            except Exception as e:
                logging.error(f"[{self.config['camera_id']}] 顯示錯誤：{e}")
                time.sleep(0.2)
        cv2.destroyWindow(win_name)

    # =============================
    # 警報邏輯
    # =============================

    def check_alerts(self, annotated_frame):
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
                    self.take_screenshot(annotated_frame, a_type)

    def take_screenshot(self, annotated_frame, alert_type):
        date_folder = datetime.now().strftime("%Y%m%d")
        folder = f"capture/{date_folder}"
        safe_mkdir(folder)
        filename = f"{self.config['camera_id']}_{get_timestamp()}_{uuid_suffix()}_{alert_type}.png"
        path = f"{folder}/{filename}"
        cv2.imwrite(path, annotated_frame)
        logging.info(f"📷 {self.config['location']} {alert_type} 截圖完成 → {path}")
        upload_queue.put((annotated_frame.copy(), self.config, alert_type))

    # =============================
    # 繪圖輔助
    # =============================

    def draw_status_header(self, img):
        status = get_current_status()
        now = datetime.now().strftime("%H:%M:%S")
        color = (0, 255, 0) if status == "WORK" else (0, 255, 255)
        cv2.rectangle(img, (440, 8), (635, 56), (0, 80, 0), -1)
        cv2.putText(img, f"Status: {status}", (448, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"Time: {now}", (448, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def draw_status_footer(self, img):
        cv2.putText(img, f"L_eye:{self.l_eye_value:.3f} R_eye:{self.r_eye_value:.3f}",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(img, f"Head:{self.head_angle:.1f}  Asym:{self.face_asymmetry:.3f}",
                    (10, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def draw_pose_keypoints(self, img, keypoints, conf_thr=0.5):
        skeleton_pairs = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12),
            (0, 1), (0, 2), (1, 3), (2, 4),
        ]
        for i, j in skeleton_pairs:
            if i < len(keypoints) and j < len(keypoints):
                if keypoints[i][2] > conf_thr and keypoints[j][2] > conf_thr:
                    cv2.line(img, (int(keypoints[i][0]), int(keypoints[i][1])),
                             (int(keypoints[j][0]), int(keypoints[j][1])), (0, 255, 255), 1)
        head_points = {0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear"}
        for idx, name in head_points.items():
            if idx < len(keypoints) and keypoints[idx][2] > conf_thr:
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
                cv2.circle(img, (x, y), 3, (0, 200, 255), -1)
                cv2.putText(img, name, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    def draw_face_landmarks(self, img, landmarks):
        ih, iw = img.shape[:2]
        draw_idxs = {33, 133, 159, 145, 362, 263, 386, 374, 13, 14, 87, 317, 1, 2, 98}
        for idx, lm in enumerate(landmarks.landmark):
            if idx in draw_idxs:
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(img, (x, y), 2, (255, 160, 0), -1)

    def draw_box(self, image, box, label, color=None):
        """
        在畫面上繪製標註框與標籤文字
        - cap / mouth / head：文字顯示在框上方
        - mask / others：文字顯示在框下方
        - cap：綠色框 (0,255,0)
        - mask：白色框 (255,255,255)
        """
        x1, y1, x2, y2 = map(int, box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1  # 改為 1，符合需求
        text_color = (0, 0, 0)  # 黑色文字
        (w, h), _ = cv2.getTextSize(label, font, scale, thickness)

        label_lower = label.lower()

        # ===== 框顏色邏輯 =====
        if "cap" in label_lower:
            color = (0, 255, 0)          # 綠色
        elif "mask" in label_lower:
            color = (255, 255, 255)      # 白色
        elif "mouth" in label_lower:
            color = (0, 0, 255)          # 紅色
        else:
            color = color or (0, 255, 0)  # 預設保留綠色（可由呼叫方覆寫）

        # ===== 繪製框線 =====
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # ===== 文字位置邏輯 =====
        if any(k in label_lower for k in ["cap", "mouth", "head"]):
            # 顯示在框上方
            y_text = max(y1 - 5, h + 5)
        else:
            # 顯示在框下方
            y_text = min(y2 + h + 10, image.shape[0] - 5)

        # ===== 背景方塊（與框同色） =====
        cv2.rectangle(image, (x1, y_text - h - 6), (x1 + w + 6, y_text), color, -1)
        cv2.putText(image, label, (x1 + 3, y_text - 4), font, scale, text_color, thickness)


    def draw_persons(self, image, n_results, conf_thr=0.30):
        """
        從 yolo (n_model) 的結果畫出 person bounding boxes（只畫 person）。
        參數:
          image: 要繪製的影像 (會就地繪製)
          n_results: n_model(...) 回傳的結果 iterable
          conf_thr: person 信心門檻
        回傳:
          bool - 是否至少偵測到一個 person
        """
        person_found = False
        if n_results is None:
            return False
        for r in n_results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for box in r.boxes:
                try:
                    # 兼容 tensor/list 型態
                    cls_val = box.cls
                    conf_val = box.conf
                    cls_id = int(cls_val[0]) if hasattr(cls_val, "__len__") else int(cls_val)
                    conf = float(conf_val[0]) if hasattr(conf_val, "__len__") else float(conf_val)
                except Exception:
                    continue

                # 只處理 person class_id == 0
                if cls_id != 0 or conf < conf_thr:
                    continue

                # 取得 xyxy 座標
                try:
                    xy = box.xyxy[0]
                    if hasattr(xy, "cpu"):
                        xy = xy.cpu().numpy()
                    x1, y1, x2, y2 = map(int, xy)
                except Exception:
                    # 若無法取得座標，略過此 box
                    continue

                # 畫框與標籤：person 改為黃色 (BGR: 0,255,255)
                label = f"person {conf:.2f}"
                self.draw_box(image, (x1, y1, x2, y2), label, color=(0, 255, 255))
                person_found = True
        return person_found

    def draw_n_results(self, image, n_results, conf_thr=0.30):
        """
        把 yolo11n (n_model) 回傳的所有偵測項目繪製到 image。
        - 會顯示 <label> <conf>，並用不同顏色區分 class id。
        - 返回繪製的總 box 數量（可用於判斷是否有偵測到任何物件）。
        """
        if n_results is None:
            return 0

        # 取得 class name 字典（若 n_model 有提供）
        names = {}
        if hasattr(self, "n_model") and self.n_model is not None:
            names = getattr(self.n_model, "names", None) or getattr(getattr(self.n_model, "model", None), "names", {}) or {}

        # 簡單 palette，會根據 class id 取模
        palette = [
            (255,  80,  80), ( 80,255,  80), ( 80, 80,255), (255,255, 80),
            (255, 80,255), ( 80,255,255), (200,120, 70), (120,200, 70),
            (70,120,200), (200, 70,120)
        ]

        drawn = 0
        for r in n_results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for box in r.boxes:
                try:
                    cls_val = box.cls
                    conf_val = box.conf
                    cls_id = int(cls_val[0]) if hasattr(cls_val, "__len__") else int(cls_val)
                    conf = float(conf_val[0]) if hasattr(conf_val, "__len__") else float(conf_val)
                except Exception:
                    continue

                if conf < conf_thr:
                    continue

                # 取得坐標 (優先 xyxy，否則用 xywh 轉換)
                try:
                    if hasattr(box, "xyxy"):
                        xy = box.xyxy[0]
                        if hasattr(xy, "cpu"):
                            xy = xy.cpu().numpy()
                        x1, y1, x2, y2 = map(int, xy)
                    elif hasattr(box, "xywh"):
                        arr = box.xywh[0]
                        if hasattr(arr, "cpu"):
                            arr = arr.cpu().numpy()
                        cx, cy, w, h = arr
                        x1 = int(cx - w / 2); y1 = int(cy - h / 2)
                        x2 = int(cx + w / 2); y2 = int(cy + h / 2)
                    else:
                        continue
                except Exception:
                    continue

                label = names.get(cls_id, str(cls_id))
                text = f"{label} {conf:.2f}"
                color = palette[cls_id % len(palette)]
                # 使用既有 draw_box（會繪製標籤背景），傳 color 以覆寫預設
                self.draw_box(image, (x1, y1, x2, y2), text, color=color)
                drawn += 1

        return drawn

    def reset_counter(self, alert_type):
        if alert_type == "EYES CLOSED":
            self.eye_close_counter = 0
        elif alert_type == "HEAD TURNED":
            self.head_turn_counter = 0
        elif alert_type == "MISSING CAP":
            self.missing_cap_count = 0
        elif alert_type == "MISSING MASK":
            self.missing_mask_count = 0

    def reset_counters_face(self):
        self.eye_close_counter = 0

    def reset_counters_pose(self):
        self.head_turn_counter = 0

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
