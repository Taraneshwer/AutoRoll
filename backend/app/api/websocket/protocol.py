"""
WebSocket Real-Time Monitoring Event Protocol Specification and Envelope Schema.
Decouples metadata/analytics telemetry from binary media streams.
"""

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WebSocketEventType(str, Enum):
    CAMERA_STATUS_CHANGED = "CAMERA_STATUS_CHANGED"
    WORKER_STATUS_CHANGED = "WORKER_STATUS_CHANGED"
    FACE_DETECTED = "FACE_DETECTED"
    RECOGNITION_RESULT = "RECOGNITION_RESULT"
    LIVENESS_RESULT = "LIVENESS_RESULT"
    ATTENDANCE_CONFIRMED = "ATTENDANCE_CONFIRMED"
    SPOOF_ATTEMPT = "SPOOF_ATTEMPT"
    SYSTEM_METRICS_UPDATED = "SYSTEM_METRICS_UPDATED"


class WebSocketEventEnvelope(BaseModel):
    """
    Standardized Real-Time Telemetry Envelope for AutoRoll Control Plane.
    Injects event_id, timestamp_ms, sequence_number, and event_type.
    """

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: WebSocketEventType
    timestamp_ms: float = Field(default_factory=lambda: round(time.time() * 1000.0, 2))
    sequence_number: int = 0
    data: dict[str, Any] = Field(default_factory=dict)
