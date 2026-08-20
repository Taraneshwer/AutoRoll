"""
Detectors package.
"""

from autoroll.ml.detectors.aligner import BaseFaceAligner, FaceAligner
from autoroll.ml.detectors.base import BaseFaceDetector
from autoroll.ml.detectors.scrfd import SCRFDDetector

__all__ = ["BaseFaceDetector", "BaseFaceAligner", "SCRFDDetector", "FaceAligner"]
