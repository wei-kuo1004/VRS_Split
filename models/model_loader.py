import os
import torch
import numpy as np
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
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = "cuda:1"  # 使用第二張 GPU
            print("✅ 使用第二張 GPU: cuda:1")
        else:
            device = "cuda:0"  # 使用第一張 GPU
            print("✅ 只有一張 GPU，使用: cuda:0")
    else:
        device = "cpu"  # 使用 CPU
        print("❌ 沒有可用的 GPU，使用 CPU")

    # 打印所選擇的設備
    print(f"📦 當前使用的設備: {device}")

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
        print(f"⚠️ Pose 模型暖機警告：{e}")

    try:
        mask_model.predict(dummy, verbose=False)
    except Exception as e:
        print(f"⚠️ MaskCap 模型暖機警告：{e}")

    try:
        n_model.predict(dummy, verbose=False)
    except Exception as e:
        print(f"⚠️ yolo11n 模型暖機警告：{e}")    

    # 6) 類別表印出（此時 model 一定存在，不會 UnboundLocalError）
    try:
        print(f"✅ Pose 模型類別表: {pose_model.names}")
    except Exception:
        pass
    try:
        print(f"✅ MaskCap 模型類別表: {mask_model.names}")
    except Exception:
        pass
    try:
        print(f"✅ yolo11n 模型類別表: {n_model.names}")
    except Exception:
        pass

    print("✅ 模型初始化完成 (Pose + MaskCap + yolo11n)")
    return pose_model, mask_model, n_model