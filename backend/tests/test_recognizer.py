"""
Unit tests for ArcFace recognizer module.
"""

import numpy as np

from app.ml.recognition.arcface_iresnet import ArcFaceRecognizer


def test_arcface_recognizer_embedding_shape_and_norm():
    recognizer = ArcFaceRecognizer(device="cpu")
    aligned_face = np.zeros((112, 112, 3), dtype=np.uint8) + 150

    result = recognizer.extract_embedding(aligned_face)

    assert result is not None
    assert len(result.embedding) == 512
    assert result.model_version == "arcface_iresnet50_v1"

    # L2 norm of normalized vector should equal ~1.0
    arr = np.array(result.embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    assert abs(norm - 1.0) < 1e-4
