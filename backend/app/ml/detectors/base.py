"""
Abstract Base Interface for Face Detection.
"""

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.common import DetectionResult


class BaseFaceDetector(ABC):
    """
    Abstract contract for face detectors (e.g., SCRFD).
    Input: BGR numpy image frame.
    Output: List of DetectionResult objects containing bounding box and 5-point landmarks.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, score_threshold: float = 0.5) -> list[DetectionResult]:
        """
        Detect faces in frame.
        """
        pass
