"""
Local Laptop / USB Webcam Camera Source implementation.
Runs a dedicated background capture thread with bounded queue buffering
to eliminate frame backlog and prefer the latest frame.
"""

import queue
import threading
import time
from typing import Any, Tuple

import cv2
import numpy as np

from app.core.config import get_settings
from app.core.logger import get_logger
from app.camera.base import CameraSource

logger = get_logger("local_camera")


class LocalCameraSource(CameraSource):
    """
    OpenCV VideoCapture wrapper for local USB webcams.
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: int = 30,
        width: int = 1280,
        height: int = 720,
    ):
        settings = get_settings()
        self.camera_index = camera_index
        self.target_fps = target_fps or getattr(settings, "AUTOROLL_CAMERA_FPS", 30)
        self.width = width
        self.height = height

        self.cap: cv2.VideoCapture | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.thread: threading.Thread | None = None

        # Telemetry metrics
        self.frames_captured = 0
        self.frames_dropped = 0
        self.actual_capture_fps = 0.0
        self.last_capture_time = time.time()
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.is_running:
            return

        logger.info(f"Opening Local Webcam Source (Index: {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # Try default backend if DSHOW fails
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            err = f"Failed to open local camera device at index {self.camera_index}."
            logger.error(err)
            raise RuntimeError(err)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Local Webcam Opened | Resolution: {actual_w}x{actual_h} | Target FPS: {self.target_fps}")

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

            # Bounded queue strategy: drop oldest frame if buffer is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)

            # Update capture FPS metric
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
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Local Webcam Source (Index: {self.camera_index}) stopped.")

    def is_opened(self) -> bool:
        return self.is_running and self.cap is not None and self.cap.isOpened()


    def get_metrics(self) -> dict[str, Any]:
        with self.lock:
            return {
                "source_type": "local_webcam",
                "camera_index": self.camera_index,
                "is_opened": self.is_opened(),
                "capture_fps": round(self.actual_capture_fps, 1),
                "target_fps": self.target_fps,
                "frames_captured": self.frames_captured,
                "frames_dropped": self.frames_dropped,
                "queue_depth": self.frame_queue.qsize(),
                "resolution": f"{self.width}x{self.height}",
            }
