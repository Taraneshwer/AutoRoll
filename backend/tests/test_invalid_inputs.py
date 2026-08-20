"""
Unit tests for invalid input handling across ML components.
"""

import numpy as np
import pytest

from app.schemas.common import FaceLandmarks
from app.ml.detectors.aligner import FaceAligner
from app.ml.detectors.scrfd import SCRFDDetector
from app.ml.pipeline import AutoRollMLPipeline
from app.ml.recognition.arcface_iresnet import ArcFaceRecognizer


def test_detector_invalid_input():
    detector = SCRFDDetector(device="cpu")
    with pytest.raises(ValueError, match="Invalid or empty"):
        detector.detect(None)

    with pytest.raises(ValueError, match="Invalid or empty"):
        detector.detect(np.array([]))


def test_aligner_invalid_input():
    aligner = FaceAligner()
    valid_landmarks = FaceLandmarks(points=[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    with pytest.raises(ValueError, match="empty or invalid"):
        aligner.align(None, valid_landmarks)

    invalid_landmarks = FaceLandmarks(points=[(1, 1), (2, 2)])
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must contain exactly 5 points"):
        aligner.align(img, invalid_landmarks)


def test_recognizer_invalid_input():
    recognizer = ArcFaceRecognizer(device="cpu")
    with pytest.raises(ValueError, match="Invalid or empty"):
        recognizer.extract_embedding(None)

    with pytest.raises(ValueError, match="Invalid or empty"):
        recognizer.extract_embedding(np.array([]))


def test_pipeline_invalid_input():
    pipeline = AutoRollMLPipeline(device="cpu")
    with pytest.raises(ValueError, match="empty or invalid"):
        pipeline.process_frame(None)
