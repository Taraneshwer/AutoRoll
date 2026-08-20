"""
Abstract Base Interface for Face Recognition (ArcFace Embedding Extractor).
"""

from abc import ABC, abstractmethod

import numpy as np

from app.schemas.common import RecognitionResult


class BaseFaceRecognizer(ABC):
    """
    Abstract contract for face feature extraction (e.g. ArcFace IResNet50).
    """

    @abstractmethod
    def extract_embedding(self, aligned_face: np.ndarray) -> RecognitionResult:
        """
        Given a 112x112 aligned face chip, returns a 512-dimensional L2-normalized embedding vector.
        """
        pass

    @abstractmethod
    def extract_embeddings_batch(
        self, aligned_faces: list[np.ndarray]
    ) -> list[RecognitionResult]:
        """
        Extracts embeddings across multiple 112x112 aligned face chips.
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """
        Returns model identifier tag (e.g. 'pretrained', 'autoroll_v1').
        """
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """
        Returns model version string tag (e.g. 'arcface_r50_v1', 'autoroll_arcface_r50_epoch1').
        """
        pass

    @abstractmethod
    def get_recognition_threshold(self) -> float:
        """
        Returns model-specific validated decision threshold (e.g. 0.0440 or 0.0540).
        """
        pass


FaceRecognitionModel = BaseFaceRecognizer

