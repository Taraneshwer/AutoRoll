"""
Direct RTSP Camera Stream Client with Threaded Frame Decoding & Auto-Reconnection.
"""

import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from app.core.logger import get_logger

logger = get_logger("rtsp_stream_client")


class RTSPStreamClient:
    """
    Direct RTSP stream reader thread. Decodes frames locally and handles automatic reconnection.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        frame_callback: Callable[[str, np.ndarray, int], None],
        reconnect_interval_sec: float = 3.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.frame_callback = frame_callback
        self.reconnect_interval = reconnect_interval_sec

        self.running = False
        self.thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None
        self.frame_index = 0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._stream_loop, name=f"rtsp_{self.camera_id}", daemon=True
        )
        self.thread.start()
        logger.info(f"RTSP Stream Client started for Camera '{self.camera_id}' ({self.rtsp_url})")

    def _stream_loop(self) -> None:
        while self.running:
            logger.info(f"Connecting to RTSP stream for Camera '{self.camera_id}'...")
            self.cap = cv2.VideoCapture(self.rtsp_url)

            if not self.cap.isOpened():
                logger.warning(
                    f"Failed to open RTSP stream for Camera '{self.camera_id}'. "
                    f"Retrying in {self.reconnect_interval}s..."
                )
                time.sleep(self.reconnect_interval)
                continue

            logger.info(f"RTSP stream connected successfully for Camera '{self.camera_id}'")

            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(
                        f"Stream disconnect detected on Camera '{self.camera_id}'. Reconnecting..."
                    )
                    break

                self.frame_index += 1
                try:
                    self.frame_callback(self.camera_id, frame, self.frame_index)
                except Exception as e:
                    logger.error(
                        f"Error processing frame callback for Camera '{self.camera_id}': {e}"
                    )

            if self.cap:
                self.cap.release()
                self.cap = None

            if self.running:
                time.sleep(self.reconnect_interval)

    def stop(self) -> None:
        """
        Releases stream resources cleanly when camera is unassigned or worker shuts down.
        """
        logger.info(f"Stopping RTSP Stream Client for Camera '{self.camera_id}'...")
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info(f"RTSP Stream Client for Camera '{self.camera_id}' stopped.")
