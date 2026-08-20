"""
SQLAlchemy Database ORM Models for AutoRoll.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import relationship

from server.app.db.session import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default="MANAGER")  # ADMIN, MANAGER, VIEWER
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Student(Base):
    __tablename__ = "students"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    student_code = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    embeddings = relationship(
        "StudentEmbedding", back_populates="student", cascade="all, delete-orphan"
    )
    attendance = relationship("AttendanceRecord", back_populates="student")


class StudentEmbedding(Base):
    __tablename__ = "student_embeddings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    student_id = Column(String(36), ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    embedding_vector = Column(LargeBinary, nullable=False)  # Serialized 512 float32 bytes
    model_version = Column(String(50), nullable=False)
    is_primary = Column(Boolean, default=True)
    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    student = relationship("Student", back_populates="embeddings")


class WorkerNode(Base):
    __tablename__ = "worker_nodes"

    id = Column(String(50), primary_key=True)
    state = Column(String(20), nullable=False, default="STARTING")
    cpu_percent = Column(Float, default=0.0)
    ram_used_mb = Column(Float, default=0.0)
    ram_percent = Column(Float, default=0.0)
    gpu_available = Column(Boolean, default=False)
    gpu_name = Column(String(100), nullable=True)
    gpu_utilization_percent = Column(Float, nullable=True)
    gpu_memory_used_mb = Column(Float, nullable=True)
    fps = Column(Float, default=0.0)
    avg_inference_latency_ms = Column(Float, default=0.0)
    last_heartbeat_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False)
    rtsp_url = Column(String(255), nullable=False)
    location = Column(String(100), nullable=True)
    target_fps = Column(Integer, default=15)
    is_active = Column(Boolean, default=True)
    assigned_worker_id = Column(String(50), nullable=True)


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=False)
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=True)
    worker_id = Column(String(50), nullable=True)
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    similarity_score = Column(Float, nullable=False)
    liveness_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    verification_status = Column(String(20), default="CONFIRMED")

    student = relationship("Student", back_populates="attendance")


class AnalyticsEvent(Base):
    __tablename__ = "analytics_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    event_type = Column(String(50), nullable=False)
    student_id = Column(String(36), ForeignKey("students.id"), nullable=True)
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=True)
    worker_id = Column(String(50), nullable=True)
    similarity_score = Column(Float, default=0.0)
    liveness_score = Column(Float, default=0.0)
    model_version = Column(String(50), nullable=False, default="iresnet50_arcface_v1")
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    admin_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(50), nullable=True)
    details = Column(String(255), nullable=True)
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_version = Column(String(50), unique=True, nullable=False)
    backbone = Column(String(50), nullable=False, default="iresnet50")
    dataset_version = Column(String(50), nullable=False, default="MS1MV2_ArcFace")
    embedding_dim = Column(Integer, default=512)
    similarity_threshold = Column(Float, default=0.65)
    liveness_threshold = Column(Float, default=0.90)
    is_active_default = Column(Boolean, default=False)
