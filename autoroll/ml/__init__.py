"""
AutoRoll Standalone ML Engine Abstractions & Interfaces.
"""

from autoroll.ml.detectors.aligner import BaseFaceAligner
from autoroll.ml.detectors.base import BaseFaceDetector
from autoroll.ml.liveness.base import BaseLivenessDetector
from autoroll.ml.pipeline import BasePipeline
from autoroll.ml.recognition.base import BaseFaceRecognizer
from autoroll.ml.tracking.base import BaseTracker

__all__ = [
    "BaseFaceDetector",
    "BaseFaceAligner",
    "BaseLivenessDetector",
    "BaseFaceRecognizer",
    "BaseTracker",
    "BasePipeline",
]
