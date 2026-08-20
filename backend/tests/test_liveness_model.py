"""
Unit tests for Spatial Passive Anti-Spoofing Model and Pipeline.
"""

import numpy as np

from app.schemas.common import LivenessResult
from app.ml.liveness.passive_fas import PassiveAntiSpoofingModel
from app.ml.liveness.pipeline import LivenessPipeline


def test_passive_fas_model_predict():
    model = PassiveAntiSpoofingModel(device="cpu")
    chip = np.zeros((112, 112, 3), dtype=np.uint8) + 150

    score = model.predict_spatial_score(chip)
    assert 0.0 <= score <= 1.0


def test_liveness_pipeline_single_frame():
    pipeline = LivenessPipeline(device="cpu", liveness_threshold=0.5)
    chip = np.zeros((112, 112, 3), dtype=np.uint8) + 180

    result = pipeline.predict(chip)
    assert isinstance(result, LivenessResult)
    assert 0.0 <= result.liveness_score <= 1.0
    assert result.method == "passive_mini_fas_temporal"
    assert "latency_ms" in result.details
    assert "decision" in result.details


def test_liveness_pipeline_sequence():
    pipeline = LivenessPipeline(device="cpu")
    chips = [np.zeros((112, 112, 3), dtype=np.uint8) + (150 + i * 5) for i in range(5)]

    result = pipeline.predict_sequence(chips)
    assert isinstance(result, LivenessResult)
    assert result.details["frame_count"] == 5
