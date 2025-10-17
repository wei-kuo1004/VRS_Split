import numpy as np
import math
import logging

def calculate_head_angle(kp, w=640, h=480, conf=0.5):
    """
    根據 YOLO Pose 關鍵點估算頭部轉向角度、臉部不對稱度與側轉比率。
    """
    try:
        if len(kp) < 5:
            return None, None, None

        # 取出前五個關鍵點 (鼻尖、左眼、右眼、左耳、右耳)
        nose = kp[0][:2] if kp[0][2] > conf else None
        left_eye = kp[1][:2] if kp[1][2] > conf else None
        right_eye = kp[2][:2] if kp[2][2] > conf else None
        left_ear = kp[3][:2] if kp[3][2] > conf else None
        right_ear = kp[4][:2] if kp[4][2] > conf else None

        # 若缺少主要關鍵點，直接略過
        if nose is None or left_eye is None or right_eye is None:
            return None, None, None

        # --- 水平角度 ---
        head_center_x = (left_eye[0] + right_eye[0]) / 2
        offset_ratio = (head_center_x - w / 2) / (w / 2)
        eye_vector = np.array(right_eye) - np.array(left_eye)
        raw_angle = math.degrees(math.atan2(eye_vector[1], eye_vector[0]))
        horizontal_angle = raw_angle - offset_ratio * 15  # 補償畫面偏移

        # --- 臉部不對稱度 ---
        left_eye_to_nose = np.linalg.norm(np.array(nose) - np.array(left_eye))
        right_eye_to_nose = np.linalg.norm(np.array(nose) - np.array(right_eye))
        denom = left_eye_to_nose + right_eye_to_nose
        asymmetry = 0.0
        if denom > 0:
            asymmetry = (right_eye_to_nose - left_eye_to_nose) / denom

        # --- 側轉比率 (耳朵輔助判斷) ---
        side_ratio = None
        if left_ear is not None and right_ear is not None:
            ear_center_x = (left_ear[0] + right_ear[0]) / 2
            ear_width = abs(right_ear[0] - left_ear[0])
            if ear_width > 1e-5:  # 避免除以零
                side_ratio = (nose[0] - ear_center_x) / ear_width

        return horizontal_angle, asymmetry, side_ratio

    except Exception as e:
        logging.error(f"頭部角度計算錯誤: {e}")
        return None, None, None
