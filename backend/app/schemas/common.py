"""
Common Data Transfer Objects (DTOs) and Schemas for AutoRoll.
"""

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class FaceLandmarks(BaseModel):
    """
    Standard 5-point facial landmarks:
    [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
    Each point is (x, y).
    """
    points: list[tuple[float, float]]

    def validate_5point(self) -> bool:
        return len(self.points) == 5


class DetectionResult(BaseModel):
    bbox: BoundingBox
    landmarks: FaceLandmarks
    det_confidence: float


class LivenessResult(BaseModel):
    is_live: bool
    liveness_score: float  # Combined final score
    ml_liveness_score: float = 0.0
    auxiliary_heuristic_score: float = 0.0
    combined_liveness_score: float = 0.0
    method: str = "passive_fas"
    details: dict[str, Any] = Field(default_factory=dict)


class RecognitionResult(BaseModel):
    embedding: list[float]  # 512-dimensional normalized vector
    student_id: str | None = None
    similarity_score: float | None = None
    model_id: str = "glint360k"
    model_version: str = "arcface_iresnet50_v1"
    embedding_dimension: int = 512
    backend: str = "ONNXRuntime"
    device: str = "cpu"
    inference_latency_ms: float = 0.0


class TrackedFace(BaseModel):
    track_id: int
    bbox: BoundingBox
    landmarks: FaceLandmarks
    liveness: LivenessResult | None = None
    recognition: RecognitionResult | None = None
    last_updated: float = Field(default_factory=time.time)


class FrameProcessingResult(BaseModel):
    camera_id: str
    timestamp: float
    frame_number: int
    processing_time_ms: float
    faces: list[TrackedFace]
