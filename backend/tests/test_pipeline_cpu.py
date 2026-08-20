"""
End-to-end CPU inference and latency recording unit tests.
"""

import cv2
import numpy as np

from app.schemas.common import FrameProcessingResult
from app.ml.pipeline import AutoRollMLPipeline


def test_pipeline_cpu_end_to_end():
    pipeline = AutoRollMLPipeline(device="cpu")
    assert pipeline.device == "cpu"

    # Synthetic image with face shape
    img = np.zeros((480, 640, 3), dtype=np.uint8) + 200
    cv2.ellipse(img, (320, 240), (80, 120), 0, 0, 360, (150, 150, 150), -1)

    result = pipeline.process_frame(img, camera_id="cam_test_cpu")

    assert isinstance(result, FrameProcessingResult)
    assert result.camera_id == "cam_test_cpu"
    assert result.processing_time_ms >= 0.0

    for face in result.faces:
        assert face.bbox.width > 0
        assert face.recognition is not None
        assert len(face.recognition.embedding) == 512
        assert face.recognition.model_version == "arcface_iresnet50_v1"
