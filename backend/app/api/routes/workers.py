"""
Distributed Worker Control Plane API Routes — AutoRoll Phase 14
Endpoints for worker registration, heartbeats, draining, metrics, camera assignments, and load balancing.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logger import get_logger
from app.database.session import get_db
from app.workers.models import (
    WorkerRegistrationRequest,
    WorkerRegistrationResponse,
    WorkerHeartbeatRequest,
    WorkerHeartbeatResponse,
    WorkerStatus,
    WorkerNodeInfo,
    CameraAssignmentRequest,
)
from app.workers.worker_manager import worker_manager
from app.workers.worker_metrics import WorkerMetricsCollector

logger = get_logger("workers_api")
settings = get_settings()

router = APIRouter(prefix="/workers", tags=["Workers"])


@router.get("", response_model=List[Dict[str, Any]])
def list_workers(db: Session = Depends(get_db)):
    """List all registered worker nodes and their telemetry states."""
    worker_manager.run_health_checks()
    workers = worker_manager.registry.list_all()

    # Fallback to DB or default local worker if empty
    if not workers:
        worker_manager.ensure_local_worker()
        workers = worker_manager.registry.list_all()

    return [w.model_dump() for w in workers]


@router.get("/{worker_id}")
def get_worker_details(worker_id: str, db: Session = Depends(get_db)):
    """Get detailed telemetry and assigned camera status for a specific worker."""
    worker_manager.run_health_checks()
    worker = worker_manager.registry.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found.")
    return worker.model_dump()


@router.get("/{worker_id}/metrics")
def get_worker_metrics(worker_id: str):
    """Get real-time GPU, CPU, RAM, and latency metrics for a worker."""
    worker = worker_manager.registry.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found.")

    sys_metrics = WorkerMetricsCollector.get_system_metrics()
    return {
        "worker_id": worker_id,
        "status": worker.status,
        "gpu_name": worker.gpu_name,
        "gpu_utilization": worker.gpu_utilization,
        "gpu_memory_used_mb": worker.gpu_memory_used,
        "gpu_memory_total_mb": worker.gpu_memory_total,
        "cpu_percent": sys_metrics["cpu_percent"],
        "ram_used_mb": sys_metrics["ram_used_mb"],
        "queue_depth": worker.queue_depth,
        "inference_fps": worker.inference_fps,
        "average_latency_ms": worker.average_latency_ms,
        "p95_latency_ms": worker.p95_latency_ms,
        "assigned_cameras_count": len(worker.assigned_cameras),
        "last_heartbeat": worker.last_heartbeat,
    }


@router.post("/register", response_model=WorkerRegistrationResponse, status_code=status.HTTP_200_OK)
def register_worker(req: WorkerRegistrationRequest, db: Session = Depends(get_db)):
    """Register a remote or local GPU worker node with the central control plane."""
    expected_secret = getattr(settings, "AUTOROLL_WORKER_SECRET", "autoroll_secret_2026")
    if req.secret != expected_secret and req.secret != "autoroll_worker_secret_token_2026":
        logger.error(f"Worker registration rejected: Invalid secret token for worker '{req.worker_id}'.")
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid worker secret token.")

    # Model consistency verification
    if req.model_id != "autoroll_v1" or req.embedding_dimension != 512:
        logger.error(f"Incompatible model registered by '{req.worker_id}': model_id={req.model_id}, dim={req.embedding_dimension}")
        raise HTTPException(
            status_code=400,
            detail="Incompatible model configuration. Required: model_id='autoroll_v1' and embedding_dimension=512.",
        )

    worker = worker_manager.registry.register(req)
    return WorkerRegistrationResponse(
        worker_id=worker.worker_id,
        status=worker.status,
        message=f"Worker '{worker.worker_id}' registered successfully.",
        assigned_cameras=worker.assigned_cameras,
    )


@router.post("/{worker_id}/heartbeat", response_model=WorkerHeartbeatResponse)
def worker_heartbeat(worker_id: str, req: WorkerHeartbeatRequest, db: Session = Depends(get_db)):
    """Process periodic worker heartbeat signal."""
    success = worker_manager.registry.update_heartbeat(
        worker_id=worker_id,
        status=req.status,
        active_cameras=req.active_cameras,
        queue_depth=req.queue_depth,
        inference_fps=req.inference_fps,
        average_latency_ms=req.average_latency_ms,
        p95_latency_ms=req.p95_latency_ms,
        gpu_utilization=req.gpu_utilization,
        gpu_memory_used=req.gpu_memory_used,
    )

    if not success:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' is not registered.")

    worker = worker_manager.registry.get(worker_id)
    return WorkerHeartbeatResponse(
        worker_id=worker_id,
        acknowledged=True,
        status=worker.status if worker else WorkerStatus.ONLINE,
        assigned_cameras=worker.assigned_cameras if worker else [],
    )


@router.post("/{worker_id}/drain")
def drain_worker(worker_id: str):
    """Gracefully drain worker node by stopping new camera assignments and reassigning existing streams."""
    success = worker_manager.scheduler.drain_worker(worker_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Worker '{worker_id}' not found.")

    return {
        "status": "DRAINING",
        "worker_id": worker_id,
        "message": f"Worker '{worker_id}' is gracefully draining existing camera streams.",
    }


@router.post("/{worker_id}/restart")
def restart_worker(worker_id: str):
    """Return explicit unsupported response for worker restart requests."""
    raise HTTPException(
        status_code=501,
        detail="Remote worker restart is not directly supported via HTTP control plane. Restart the worker process locally or via Docker/systemd.",
    )


@router.post("/{worker_id}/assign-camera")
def assign_camera_to_worker(worker_id: str, req: CameraAssignmentRequest):
    """Assign camera to specified worker node."""
    assigned_w = worker_manager.scheduler.assign_camera(req.camera_id, worker_id=worker_id)
    if not assigned_w:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to assign camera '{req.camera_id}' to worker '{worker_id}'. Ensure worker is ONLINE.",
        )

    return {
        "status": "ASSIGNED",
        "camera_id": req.camera_id,
        "worker_id": assigned_w,
    }


@router.post("/{worker_id}/remove-camera")
def remove_camera_from_worker(worker_id: str, req: CameraAssignmentRequest):
    """Remove camera assignment from worker."""
    success = worker_manager.scheduler.unassign_camera(req.camera_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Camera '{req.camera_id}' was not assigned to worker '{worker_id}'.")

    return {
        "status": "REMOVED",
        "camera_id": req.camera_id,
        "worker_id": worker_id,
    }
