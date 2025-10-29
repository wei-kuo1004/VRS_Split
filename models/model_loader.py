import os
import torch
import numpy as np
import logging
from ultralytics import YOLO
from utils.helpers import resource_path

def load_models():
    pose_path = resource_path("model/yolo11n-pose.pt")
    mask_path = resource_path("model/mask_cap/best.pt")
    n_path = resource_path("model/yolo11n.pt")  # 新增 yolo11n 路徑

    # 1) 路徑檢查
    for p in [pose_path, mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到模型：{p}")

    # 2) 設備選擇
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"✅ 使用設備: {device}")

    # 3) 載入模型（先建立，再做任何與 model 相關的 log）
    pose_model = YOLO(pose_path).to(device)
    mask_model = YOLO(mask_path).to(device)
    n_model = YOLO(n_path).to(device)  # 載入 yolo11n

    # 4) 關閉 fuse（避免某些環境下 BN fuse 問題）
    pose_model.model.fuse = lambda *a, **k: pose_model.model
    mask_model.model.fuse = lambda *a, **k: mask_model.model
    n_model.model.fuse = lambda *a, **k: n_model.model

    # 5) 暖機（建議保留，能降低第一幀延遲、避免初始化時的偶發錯誤）
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        pose_model.predict(dummy, verbose=False)
    except Exception as e:
        logging.warning(f"⚠️ Pose 模型暖機警告：{e}")

    try:
        mask_model.predict(dummy, verbose=False)
    except Exception as e:
        logging.warning(f"⚠️ MaskCap 模型暖機警告：{e}")

    try:
        n_model.predict(dummy, verbose=False)
    except Exception as e:
        logging.warning(f"⚠️ yolo11n 模型暖機警告：{e}")    

    # 6) 類別表印出（此時 model 一定存在，不會 UnboundLocalError）
    try:
        logging.info(f"✅ Pose 模型類別表: {pose_model.names}")
    except Exception:
        pass
    try:
        logging.info(f"✅ MaskCap 模型類別表: {mask_model.names}")
    except Exception:
        pass
    try:
        logging.info(f"✅ yolo11n 模型類別表: {n_model.names}")
    except Exception:
        pass

    logging.info("✅ 模型初始化完成 (Pose + MaskCap + yolo11n)")
    return pose_model, mask_model, n_model
