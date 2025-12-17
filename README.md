# VRS_Split

## 1. 專案簡介（Overview）

VRS_Split 是一套多路 RTSP 監視器之**安全帽/口罩穿戴與疲勞監控系統**,整合 YOLO 系列模型、MediaPipe 與排程管控機制,可同時對多支攝影機進行即時人員偵測、姿態分析、異常行為判斷,並於觸發事件時自動截圖與上傳告警。

本系統主要解決**工廠/產線場域**中,需要 24 小時多路監控、即時偵測未戴安全帽、未戴口罩、閉眼疲勞、頭部偏轉等高風險行為,並將事件即時推播至 LineGPT 與內部 EIP API 的需求。

### 適用場域

- 工廠產線安全監控
- 勞安稽核輔助
- AI CCTV 即時告警系統
- 長時間無人值守之監控環境

## 2. 系統架構說明（Architecture）

### 整體流程

1. 系統啟動時載入模型與攝影機設定,並初始化事件截圖上傳佇列。
2. 預先檢查 RTSP 訊號可用性,建立有效攝影機清單。
3. 每支攝影機建立三個 Daemon 執行緒:
   - **讀取執行緒**: 使用 PyAV 讀取 RTSP 串流,失敗時自動回退至 OpenCV（FFMPEG）,並將畫面縮放至 640x480。
   - **處理執行緒**:
     - 檢查是否在允許運行時段
     - MediaPipe: 眼睛閉合偵測
     - YOLO Pose: 姿態與頭部角度分析
     - YOLO11n: 人形存在檢查
     - YOLO mask/cap: 安全帽與口罩穿戴判斷
     - 累積影格達門檻後觸發警示
   - **顯示執行緒**: 依螢幕大小與攝影機索引計算視窗位置,繪製偵測框、關鍵點與狀態資訊。
4. 警示觸發時:
   - 畫面截圖存檔
   - 將事件送入上傳佇列
   - 由背景 Worker 批次上傳至 LineGPT 與 EIP API

### 模組關係

```
main.py
 ├─ config.camera_config      → 攝影機設定
 ├─ models.model_loader       → 模型載入
 ├─ cameras.camera_monitor    → 單攝影機處理
 ├─ utils.schedule_checker    → 排程控制
 └─ utils.uploader            → 背景上傳
```

### 多執行緒角色

- 每支攝影機: 3 個執行緒（讀取/處理/顯示）
- 全域: 多個上傳 Worker（預設 3）
- 執行緒間以 Queue 傳遞事件資料

## 3. 專案目錄結構（Project Structure）

```
VRS_Split/
├── main.py                     # 系統入口,啟動流程與執行緒
├── main.spec                   # PyInstaller 打包設定
├── config/
│   └── camera_config.py        # 攝影機 JSON 設定讀寫
├── cameras/
│   ├── camera_monitor.py       # 單一攝影機監控邏輯
│   └── rtsp_checker.py         # RTSP 訊號檢查
├── models/
│   ├── model_loader.py         # YOLO 模型載入與 warmup
│   └── alert_cooldown.py       # 警示冷卻時間管理
├── utils/
│   ├── head_angle.py           # 頭部角度與對稱度計算
│   ├── helpers.py              # 路徑、時間戳、UUID 等工具
│   ├── schedule_checker.py     # 工作時段判斷
│   ├── uploader.py             # 上傳佇列與通知
│   └── LineGptChatRoomId.txt   # LineGPT Chat ID 清單
├── model/
│   ├── yolo11n-pose.pt
│   ├── yolo11n.pt
│   ├── mask_cap/best.pt
│   └── blaze_face_short_range.tflite
├── capture/                    # 事件截圖輸出目錄（依日期）
├── cameras_config_*.txt        # 攝影機設定檔
├── requirements5080.txt        # CUDA 12.8 套件需求
├── requirementsL4.txt          # CUDA 12.6 套件需求
├── build/, dist/               # PyInstaller 產物
└── env5080/, envL4/            # 虛擬環境（非程式邏輯）
```

## 4. 核心功能模組說明（Core Modules）

### main.py
- **功能**: 系統啟動、模型載入、攝影機初始化、上傳 Worker 啟動
- **關鍵邏輯**: RTSP 預檢,若無可用攝影機則直接結束程式

### config/camera_config.py
- **功能**: 讀取指定攝影機設定檔,不存在時自動建立預設檔
- **重要欄位**: `rtsp_url`、`camera_id`、`location`、`eye_close_threshold`、`head_turn_threshold`、`missing_cap_frames`、`missing_mask_frames`

### models/model_loader.py
- **功能**: 裝置選擇（GPU/CPU）、YOLO 模型載入與 warmup
- **策略**: 優先使用 CUDA,多 GPU 環境下指定 line 1

### models/alert_cooldown.py
- **功能**: 依攝影機與警示類型管理冷卻時間,避免重複告警

### cameras/camera_monitor.py
- **功能**: 單一攝影機完整流程（讀取/偵測/顯示/截圖）
- **偵測項目**:
  - 閉眼疲勞
  - 頭部偏轉
  - 未戴安全帽
  - 未戴口罩

### utils/head_angle.py
- **功能**: 依 YOLO Pose 關鍵點計算頭部角度與側轉指標
- **輸出**: 角度、對稱度、側轉比值（不足條件回傳 None）

### utils/schedule_checker.py
- **功能**: 依硬編時間區段判斷是否為工作時段（WORK/SKIP）

### utils/uploader.py
- **功能**: 事件截圖儲存、LineGPT 與 EIP API 上傳
- **注意事項**: 帳密與 Token 目前為程式內硬編

## 5. 執行方式（How to Run）

### 環境需求
- Python 3.10
- Windows 作業系統
- CUDA GPU（非必要,系統可自動降級 CPU）

### 套件安裝

```bash
python -m venv env
.\env\Scripts\Activate.ps1
pip install -U pip setuptools wheel
pip install -r requirements5080.txt
```

（CUDA 12.6 環境請改用 `requirementsL4.txt`）

### 啟動系統

```bash
python main.py
```

## 6. 設定與參數說明（Configuration）

- **攝影機設定**: `cameras_config_*.txt`
- **排程時段**: `utils/schedule_checker.py`
- **模型路徑**: `models/model_loader.py`
- **上傳與通知**: `utils/uploader.py`
- **Chat ID**: `utils/LineGptChatRoomId.txt`

## 7. 日誌與除錯方式（Logging & Debug）

- 預設僅輸出 WARNING 以上等級 log
- 事件截圖存於 `capture/YYYYMMDD/`
- **常見問題排查**:
  - RTSP 無法連線
  - CUDA 不可用導致效能不足
  - Token 過期或 API 連線失敗
  - 排程時間未落在 WORK 區段

## 8. 維運與交接注意事項（Maintenance Notes）

- 更換模型檔需維持相同檔名或同步修改程式
- 重啟程式不會保留佇列狀態
- 建議將帳密與 Token 改為環境變數或安全存放
- PyInstaller 打包時需確認模型與設定檔一併納入
- 視窗排列依螢幕解析度與攝影機數量自動計算,異動需調整顯示邏輯

## 9. 已知限制與待改善事項（Known Issues / TODO）

- LineGPT/EIP 帳密與 Token 為硬編
- 部分 MediaPipe/BlazeFace 模型尚未使用
- Logging 尚未全面標準化
- 尚未支援 Linux/Headless 部署模式