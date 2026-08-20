"""
Student Pydantic Schemas.
"""


from pydantic import BaseModel, Field


class StudentCreate(BaseModel):
    student_code: str
    full_name: str
    department: str | None = None


class EnrollmentPayload(BaseModel):
    embedding: list[float] = Field(..., min_length=512, max_length=512)
    model_version: str = "iresnet50_arcface_v1"


class StudentResponse(BaseModel):
    id: str
    student_code: str
    full_name: str
    department: str | None = None
    is_active: bool
