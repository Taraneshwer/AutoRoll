"""
Unit tests for UnifiedDecisionEngine.
"""

from autoroll.common.schemas import BoundingBox
from autoroll.ml.inference.decision import UnifiedDecisionEngine
from autoroll.ml.inference.tracker import FaceTrack


def test_decision_engine_evaluation():
    track = FaceTrack(
        track_id=1,
        bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.95),
        landmarks=[],
        confidence=0.95,
        frame_index=1,
    )

    res = UnifiedDecisionEngine.evaluate_track_decision(
        track=track,
        is_live=True,
        liveness_score=0.95,
        liveness_decision="REAL",
        embedding=[0.1] * 512,
    )

    assert res.track_id == 1
    assert res.is_live is True
    assert res.liveness_decision == "REAL"
    assert res.recognition_status == "RECOGNIZED"
    assert len(res.embedding) == 512


def test_decision_engine_error_resilience():
    track = FaceTrack(
        track_id=2,
        bbox=BoundingBox(x1=0, y1=0, x2=50, y2=50, confidence=0.95),
        landmarks=[],
        confidence=0.95,
        frame_index=1,
    )

    res = UnifiedDecisionEngine.evaluate_track_decision(
        track=track,
        is_live=False,
        liveness_score=0.20,
        liveness_decision="SPOOF",
        recognition_error="Model extraction failed",
    )

    assert res.track_id == 2
    assert res.is_live is False
    assert res.liveness_decision == "SPOOF"
    assert "FAILED" in res.recognition_status
