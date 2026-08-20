"""
Worker State Enum and System Health Metrics Schema.
"""

from enum import Enum

from pydantic import BaseModel, Field


class WorkerState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    BUSY = "BUSY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    STOPPING = "STOPPING"


class WorkerHealthMetrics(BaseModel):
    """
    Exposes worker hardware and operational health metrics.
    """

    worker_id: str
    state: WorkerState
    cpu_percent: float
    ram_used_mb: float
    ram_percent: float
    gpu_available: bool
    gpu_name: str | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    active_cameras_count: int = 0
    active_camera_ids: list[str] = Field(default_factory=list)
    fps: float = 0.0
    queue_backlog: int = 0
    avg_inference_latency_ms: float = 0.0
    model_versions: dict[str, str] = Field(default_factory=dict)
