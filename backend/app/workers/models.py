"""
Worker State & Request Data Models — AutoRoll Phase 14
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkerStatus(str, Enum):
    STARTING = "STARTING"
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    DRAINING = "DRAINING"
    OFFLINE = "OFFLINE"
    ERROR = "ERROR"


class WorkerRegistrationRequest(BaseModel):
    worker_id: str = Field(..., description="Unique worker node identifier")
    hostname: str = Field(default="localhost", description="Worker host name or IP")
    ip_address: Optional[str] = Field(default="127.0.0.1", description="IPv4/IPv6 address")
    port: int = Field(default=8001, description="Worker HTTP/WS port")
    secret: str = Field(..., description="Authentication token")
    gpu_name: str = Field(default="CPU", description="GPU graphics card model name")
    gpu_memory_total: float = Field(default=0.0, description="Total VRAM in MB")
    cuda_version: Optional[str] = Field(default=None, description="CUDA runtime version")
    python_version: Optional[str] = Field(default="3.11", description="Python environment version")
    model_id: str = Field(default="autoroll_v1", description="ML model architecture ID")
    model_version: str = Field(default="autoroll_arcface_r50_epoch1", description="Model weights version")
    embedding_dimension: int = Field(default=512, description="Output embedding dimension")
    supported_camera_types: List[str] = Field(default_factory=lambda: ["RTSP", "USB", "MJPEG"])
    max_camera_capacity: int = Field(default=4, description="Maximum concurrent RTSP stream capacity")


class WorkerRegistrationResponse(BaseModel):
    worker_id: str
    status: WorkerStatus
    message: str
    assigned_cameras: List[str] = Field(default_factory=list)


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str
    secret: str
    timestamp: float
    status: WorkerStatus = WorkerStatus.ONLINE
    active_cameras: List[str] = Field(default_factory=list)
    queue_depth: int = 0
    inference_fps: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    temperature: Optional[float] = None


class WorkerHeartbeatResponse(BaseModel):
    worker_id: str
    acknowledged: bool
    status: WorkerStatus
    command: Optional[str] = None  # e.g., "DRAIN", "STOP", "ASSIGN_CAMERAS"
    assigned_cameras: List[str] = Field(default_factory=list)


class WorkerNodeInfo(BaseModel):
    worker_id: str
    hostname: str
    ip_address: Optional[str] = "127.0.0.1"
    port: int = 8001
    status: WorkerStatus = WorkerStatus.ONLINE
    gpu_name: str = "CPU"
    gpu_memory_total: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_utilization: float = 0.0
    cpu_percent: float = 0.0
    model_id: str = "autoroll_v1"
    model_version: str = "autoroll_arcface_r50_epoch1"
    embedding_dimension: int = 512
    assigned_cameras: List[str] = Field(default_factory=list)
    max_camera_capacity: int = 4
    load_score: float = 0.0
    queue_depth: int = 0
    inference_fps: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_heartbeat: float = 0.0


class CameraAssignmentRequest(BaseModel):
    camera_id: str
    worker_id: Optional[str] = None


class CameraAssignmentResponse(BaseModel):
    camera_id: str
    assigned_worker_id: Optional[str]
    status: str
    message: str
