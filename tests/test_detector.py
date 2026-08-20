"""
Unit tests for SCRFD face detector module.
"""

import numpy as np

from autoroll.common.schemas import DetectionResult
from autoroll.ml.detectors.scrfd import SCRFDDetector


def test_scrfd_detector_initialization():
    detector = SCRFDDetector(device="cpu")
    assert detector.device == "cpu"


def test_scrfd_detector_detect():
    detector = SCRFDDetector(device="cpu")
    # Synthetic image 480x640 with a dummy box
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 128
    detections = detector.detect(img)
    assert isinstance(detections, list)
    for det in detections:
        assert isinstance(det, DetectionResult)
        assert det.bbox.width > 0
        assert det.landmarks.validate_5point() is True
