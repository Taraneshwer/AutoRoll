"""
Abstract Base Interface for Camera Video Stream Sources.
Defines the contract for Local (USB/Webcam), RTSP, and future ESP32-CAM ingestion.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class CameraSource(ABC):
    """
    Abstract contract for video frame capture sources.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Starts the camera capture thread/stream.
        """
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, np.ndarray | None]:
        """
        Reads the most recent frame from the bounded buffer.
        Returns (success, frame_bgr).
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Stops the camera capture stream and releases hardware resources.
        """
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """
        Returns True if the camera stream is active and opened.
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Returns camera telemetry (resolution, capture FPS, drop count, queue depth).
        """
        pass
