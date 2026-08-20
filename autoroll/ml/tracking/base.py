"""
Abstract Base Interface for Face Multi-Object Tracking.
"""

from abc import ABC, abstractmethod

import numpy as np

from autoroll.common.schemas import DetectionResult, TrackedFace


class BaseTracker(ABC):
    """
    Abstract contract for tracking faces across continuous frames (e.g. ByteTrack / IoU tracker).
    """

    @abstractmethod
    def update(self, detections: list[DetectionResult], frame: np.ndarray) -> list[TrackedFace]:
        """
        Updates internal tracker state with new frame detections and returns active face tracks.
        """
        pass
