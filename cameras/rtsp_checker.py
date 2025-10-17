import av
import logging

def check_camera_signal(rtsp_url, timeout=5):
    try:
        container = av.open(rtsp_url, timeout=timeout, options={"rtsp_transport": "tcp"})
        for frame in container.decode(video=0):
            container.close()
            return True
    except Exception as e:
        logging.warning(f"❌ RTSP 連線失敗: {rtsp_url} | {e}")
    return False
