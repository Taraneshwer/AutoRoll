"""
RTSP Video Stream Camera Source implementation for network IP cameras.
"""

import queue
import threading
import time
from typing import Any, Tuple

import cv2
import numpy as np

from app.core.logger import get_logger
from app.camera.base import CameraSource

logger = get_logger("rtsp_camera")


class RTSPCameraSource(CameraSource):
    """
    OpenCV VideoCapture wrapper for RTSP IP camera streams.
    """

    def __init__(self, rtsp_url: str, target_fps: int = 25):
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.cap: cv2.VideoCapture | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.thread: threading.Thread | None = None
        self.frames_captured = 0
        self.frames_dropped = 0
        self.actual_capture_fps = 0.0
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.is_running:
            return

        logger.info(f"Opening RTSP Stream: '{self.rtsp_url}'...")
        self.cap = cv2.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            err = f"Failed to connect to RTSP stream: '{self.rtsp_url}'"
            logger.error(err)
            raise RuntimeError(err)

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        interval = 1.0 / max(1, self.target_fps)
        last_fps_calc = time.time()
        fps_count = 0

        while self.is_running and self.cap and self.cap.isOpened():
            loop_start = time.time()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self.frames_captured += 1
            fps_count += 1

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)

            now = time.time()
            if now - last_fps_calc >= 1.0:
                with self.lock:
                    self.actual_capture_fps = fps_count / (now - last_fps_calc)
                fps_count = 0
                last_fps_calc = now

            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read_frame(self) -> Tuple[bool, np.ndarray | None]:
        if not self.is_running or self.frame_queue.empty():
            return False, None
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"RTSP Stream '{self.rtsp_url}' stopped.")

    def is_opened(self) -> bool:
        return self.is_running and self.cap is not None and self.cap.isOpened()


    def get_metrics(self) -> dict[str, Any]:
        with self.lock:
            return {
                "source_type": "rtsp",
                "rtsp_url": self.rtsp_url,
                "is_opened": self.is_opened(),
                "capture_fps": round(self.actual_capture_fps, 1),
                "target_fps": self.target_fps,
                "frames_captured": self.frames_captured,
                "frames_dropped": self.frames_dropped,
                "queue_depth": self.frame_queue.qsize(),
            }
