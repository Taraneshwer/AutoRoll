"""
Abstract Base Interface for Presentation Attack Detection (Liveness).
"""

from abc import ABC, abstractmethod

import numpy as np

from autoroll.common.schemas import LivenessResult


class BaseLivenessDetector(ABC):
    """
    Abstract contract for anti-spoofing / liveness detection.
    """

    @abstractmethod
    def predict(self, face_chip: np.ndarray) -> LivenessResult:
        """
        Evaluate single face crop for presentation attack artifacts.
        """
        pass

    @abstractmethod
    def predict_sequence(self, face_chips: list[np.ndarray]) -> LivenessResult:
        """
        Evaluate multi-frame temporal sequence of crops for liveness analysis.
        """
        pass
