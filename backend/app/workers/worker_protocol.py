"""
Worker Protocol & Event Schemas — AutoRoll Phase 14
Defines strongly typed event payloads for registration, heartbeats, failover, and recognition telemetry.
"""

from enum import Enum
import time
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class WorkerEventType(str, Enum):
    WORKER_REGISTERED = "WORKER_REGISTERED"
    WORKER_ONLINE = "WORKER_ONLINE"
    WORKER_HEARTBEAT = "WORKER_HEARTBEAT"
    WORKER_DEGRADED = "WORKER_DEGRADED"
    WORKER_OFFLINE = "WORKER_OFFLINE"
    WORKER_DRAINING = "WORKER_DRAINING"
    WORKER_FAILOVER = "WORKER_FAILOVER"

    CAMERA_ASSIGNED = "CAMERA_ASSIGNED"
    CAMERA_UNASSIGNED = "CAMERA_UNASSIGNED"
    CAMERA_ONLINE = "CAMERA_ONLINE"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"

    FACE_RECOGNIZED = "FACE_RECOGNIZED"
    FACE_UNKNOWN = "FACE_UNKNOWN"
    SPOOF_DETECTED = "SPOOF_DETECTED"
    LOW_QUALITY = "LOW_QUALITY"
    ATTENDANCE_CONFIRMED = "ATTENDANCE_CONFIRMED"


class BaseWorkerEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: WorkerEventType
    timestamp: float = Field(default_factory=time.time)
    worker_id: str
    camera_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class WorkerFailoverEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: WorkerEventType = WorkerEventType.WORKER_FAILOVER
    timestamp: float = Field(default_factory=time.time)
    camera_id: str
    old_worker_id: Optional[str]
    new_worker_id: str
    reason: str
    failover_latency_ms: float = 0.0


class RecognitionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: WorkerEventType = WorkerEventType.FACE_RECOGNIZED
    timestamp: float = Field(default_factory=time.time)
    worker_id: str
    camera_id: str
    student_id: Optional[str] = None
    student_code: Optional[str] = None
    full_name: Optional[str] = None
    similarity: float = 0.0
    liveness_score: float = 0.0
    decision: str = "CONFIRMED"  # CONFIRMED, UNKNOWN, SPOOF, REJECTED
    processing_latency_ms: float = 0.0
    quality_score: float = 1.0
