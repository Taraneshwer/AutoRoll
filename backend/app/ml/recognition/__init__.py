"""
Face Recognition Package.
"""

from app.ml.recognition.autoroll_recognizer import AutoRollArcFaceRecognizer
from app.ml.recognition.base import BaseFaceRecognizer, FaceRecognitionModel
from app.ml.recognition.factory import get_recognizer
from app.ml.recognition.pretrained_recognizer import PretrainedArcFaceRecognizer

__all__ = [
    "BaseFaceRecognizer",
    "FaceRecognitionModel",
    "PretrainedArcFaceRecognizer",
    "AutoRollArcFaceRecognizer",
    "get_recognizer",
]
