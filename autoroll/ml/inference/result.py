"""
Structured Dataclasses and Pydantic Schemas for Unified Inference Engine Results.
"""


from pydantic import BaseModel, Field

from autoroll.common.schemas import BoundingBox


class TrackedFaceResult(BaseModel):
    """
    Inference result for a single tracked face within a frame.
    """

    track_id: int
    bbox: BoundingBox
    detection_confidence: float
    landmarks: list[tuple[float, float]] = Field(default_factory=list)
    embedding: list[float] | None = None
    is_live: bool = False
    liveness_score: float = 0.0
    liveness_decision: str = "PENDING"
    recognition_status: str = "PENDING"  # PENDING / RECOGNIZED / SKIPPED / FAILED
    frames_tracked: int = 1


class UnifiedFrameResult(BaseModel):
    """
    Comprehensive inference result for a single frame containing zero or more faces.
    """

    frame_index: int
    timestamp_ms: float
    num_faces_detected: int
    num_faces_live: int
    faces: list[TrackedFaceResult] = Field(default_factory=list)
    detection_latency_ms: float = 0.0
    recognition_latency_ms: float = 0.0
    liveness_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fps: float = 0.0
