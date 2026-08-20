"""
Unit tests for Face Quality Filter module.
"""

import numpy as np

from app.schemas.common import BoundingBox, DetectionResult, FaceLandmarks
from app.ml.preprocessing.quality import FaceQualityFilter


def test_quality_filter_resolution_rejection():
    qfilter = FaceQualityFilter(min_face_size=40)
    img = np.zeros((100, 100, 3), dtype=np.uint8) + 128

    # Small face bbox 20x20
    bbox = BoundingBox(x1=10, y1=10, x2=30, y2=30, confidence=0.9)
    landmarks = FaceLandmarks(points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    det = DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=0.9)

    res = qfilter.evaluate(img, det)
    assert res.passed is False
    assert "below minimum" in res.reason


def test_quality_filter_confidence_rejection():
    qfilter = FaceQualityFilter(min_confidence=0.8)
    img = np.zeros((200, 200, 3), dtype=np.uint8) + 128

    # Low confidence 0.4
    bbox = BoundingBox(x1=10, y1=10, x2=100, y2=100, confidence=0.4)
    landmarks = FaceLandmarks(points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    det = DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=0.4)

    res = qfilter.evaluate(img, det)
    assert res.passed is False
    assert "below threshold" in res.reason


def test_quality_filter_passed():
    qfilter = FaceQualityFilter(min_face_size=30, min_blur_score=1.0, min_confidence=0.5)
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    bbox = BoundingBox(x1=10, y1=10, x2=100, y2=100, confidence=0.9)
    landmarks = FaceLandmarks(points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    det = DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=0.9)

    res = qfilter.evaluate(img, det)
    assert res.passed is True
