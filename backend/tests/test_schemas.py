"""
Unit tests for shared Pydantic DTOs and Schemas.
"""

from app.schemas.common import BoundingBox, DetectionResult, FaceLandmarks, RecognitionResult


def test_bounding_box_calculations():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0, confidence=0.95)
    assert bbox.width == 100.0
    assert bbox.height == 100.0
    assert bbox.area == 10000.0
    assert bbox.to_list() == [10.0, 20.0, 110.0, 120.0]


def test_face_landmarks_5point():
    landmarks = FaceLandmarks(
        points=[
            (30.0, 40.0),
            (70.0, 40.0),
            (50.0, 60.0),
            (35.0, 80.0),
            (65.0, 80.0),
        ]
    )
    assert landmarks.validate_5point() is True


def test_detection_and_recognition_schemas():
    bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.9)
    landmarks = FaceLandmarks(points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
    det = DetectionResult(bbox=bbox, landmarks=landmarks, det_confidence=0.9)
    assert det.det_confidence == 0.9

    rec = RecognitionResult(embedding=[0.1] * 512, student_id="std_123", similarity_score=0.88)
    assert len(rec.embedding) == 512
    assert rec.student_id == "std_123"
