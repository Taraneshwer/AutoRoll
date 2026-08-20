"""
Attendance Pydantic Schemas.
"""


from pydantic import BaseModel, Field


class RecognitionEventPayload(BaseModel):
    worker_id: str
    camera_id: str | None = None
    frame_index: int = 1
    timestamp_ms: float = 0.0
    track_id: int = 0
    embedding: list[float] = Field(..., min_length=512, max_length=512)
    liveness_score: float = 0.0
    liveness_decision: str = "REAL"
    detection_confidence: float = 0.0


class AttendanceRecordResponse(BaseModel):
    id: str
    student_id: str
    camera_id: str | None = None
    similarity_score: float
    liveness_score: float
    model_version: str
    verification_status: str
    timestamp: str
