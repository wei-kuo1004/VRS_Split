import os
import sys
from datetime import datetime
import uuid

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def uuid_suffix():
    return uuid.uuid4().hex[:6]

def resource_path(relative_path):
    # PyInstaller runtime 位置
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    
    # 開發模式 (本地執行)
    return os.path.join(os.path.abspath("."), relative_path)
