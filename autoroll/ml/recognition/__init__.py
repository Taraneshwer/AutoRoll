"""
Recognition package.
"""

from autoroll.ml.recognition.arcface_iresnet import ArcFaceRecognizer
from autoroll.ml.recognition.base import BaseFaceRecognizer

__all__ = ["BaseFaceRecognizer", "ArcFaceRecognizer"]
