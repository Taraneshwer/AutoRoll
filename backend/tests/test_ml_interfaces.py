"""
Unit tests for ML module abstractions.
"""

from app.ml.detectors.base import BaseFaceDetector
from app.ml.liveness.base import BaseLivenessDetector
from app.ml.recognition.base import BaseFaceRecognizer


def test_abstract_interfaces_imported():
    assert BaseFaceDetector is not None
    assert BaseLivenessDetector is not None
    assert BaseFaceRecognizer is not None
