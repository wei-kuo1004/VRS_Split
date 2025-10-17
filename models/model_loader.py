import os
import torch
import numpy as np
import logging
from ultralytics import YOLO
from utils.helpers import resource_path

def load_models():
    pose_path = resource_path("model/yolo11n-pose.pt")
    mask_path = resource_path("model/mask_cap/best.pt")

    for p in [pose_path, mask_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到模型：{p}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"✅ 使用設備: {device}")

    pose_model = YOLO(pose_path).to(device)
    pose_model.model.fuse = lambda *a, **k: pose_model.model
    pose_model.predict(np.zeros((480, 640, 3), dtype=np.uint8))

    mask_model = YOLO(mask_path).to(device)
    mask_model.model.fuse = lambda *a, **k: mask_model.model
    mask_model.predict(np.zeros((480, 640, 3), dtype=np.uint8))

    logging.info("✅ 模型初始化完成 (Pose + MaskCap)")
    return pose_model, mask_model
