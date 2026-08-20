"""
Unit tests for UnifiedInferencePipeline.
"""

import numpy as np

from autoroll.ml.inference.pipeline import UnifiedInferencePipeline
from autoroll.ml.inference.result import UnifiedFrameResult


def test_unified_pipeline_process_frame():
    pipeline = UnifiedInferencePipeline(device="cpu", recognition_interval=5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8) + 180

    result = pipeline.process_frame(frame, frame_index=1)

    assert isinstance(result, UnifiedFrameResult)
    assert result.frame_index == 1
    assert result.total_latency_ms >= 0.0
    assert result.fps >= 0.0
