"""
Unit tests for 5-point face aligner module.
"""

import numpy as np

from app.schemas.common import FaceLandmarks
from app.ml.detectors.aligner import FaceAligner


def test_face_aligner_output_dimensions():
    aligner = FaceAligner(target_size=(112, 112))
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 100

    landmarks = FaceLandmarks(
        points=[
            (200.0, 200.0),  # Left eye
            (300.0, 200.0),  # Right eye
            (250.0, 250.0),  # Nose tip
            (220.0, 300.0),  # Left mouth corner
            (280.0, 300.0),  # Right mouth corner
        ]
    )

    aligned = aligner.align(img, landmarks)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (112, 112, 3)
