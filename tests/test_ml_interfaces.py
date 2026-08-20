"""
Unit tests for ML module abstractions.
"""

from autoroll.ml.detectors.base import BaseFaceDetector
from autoroll.ml.liveness.base import BaseLivenessDetector
from autoroll.ml.recognition.base import BaseFaceRecognizer


def test_abstract_interfaces_imported():
    assert BaseFaceDetector is not None
    assert BaseLivenessDetector is not None
    assert BaseFaceRecognizer is not None
