"""
Detectors package.
"""

from app.ml.detectors.aligner import BaseFaceAligner, FaceAligner
from app.ml.detectors.base import BaseFaceDetector
from app.ml.detectors.scrfd import SCRFDDetector

__all__ = ["BaseFaceDetector", "BaseFaceAligner", "SCRFDDetector", "FaceAligner"]
