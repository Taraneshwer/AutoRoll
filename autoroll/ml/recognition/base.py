"""
Abstract Base Interface for Face Recognition (ArcFace Embedding Extractor).
"""

from abc import ABC, abstractmethod

import numpy as np

from autoroll.common.schemas import RecognitionResult


class BaseFaceRecognizer(ABC):
    """
    Abstract contract for face feature extraction (e.g. ArcFace IResNet50).
    """

    @abstractmethod
    def extract_embedding(self, aligned_face: np.ndarray) -> RecognitionResult:
        """
        Given a 112x112 aligned face chip, returns a 512-dimensional normalized embedding vector.
        """
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """
        Returns model version string tag (e.g. 'arcface_iresnet50_v1').
        """
        pass
