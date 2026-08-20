"""
Privacy-Preserving Enrollment Schemas and Results DTO.
"""

from typing import Any

from pydantic import BaseModel, Field


class EnrollmentSampleMetadata(BaseModel):
    sample_index: int
    passed: bool
    reason: str
    blur_score: float
    face_width: float
    face_height: float
    detection_confidence: float


class EnrollmentResult(BaseModel):
    success: bool
    student_code: str
    full_name: str
    samples_processed: int
    samples_accepted: int
    rejection_reasons: list[str] = Field(default_factory=list)
    aggregated_embedding: list[float] | None = None
    model_version: str = "iresnet50_arcface_v1"
    quality_metadata: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
