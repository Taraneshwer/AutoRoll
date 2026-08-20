"""
API Routes for Central Control Plane Worker & Camera Management.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.worker_service import (
    CameraRegistrationRequest,
    RecognitionEvent,
    WorkerRegistrationRequest,
    worker_control_plane,
)

router = APIRouter(prefix="/api/v1", tags=["workers"])


class HeartbeatPayload(BaseModel):
    queue_depth: int = 0
    recent_latency_ms: float = 0.0
    gpu_utilization: float = 0.0


class AssignCameraRequest(BaseModel):
    camera_id: str


@router.post("/workers/register", status_code=status.HTTP_201_CREATED)
def register_worker(req: WorkerRegistrationRequest) -> Dict[str, Any]:
    try:
        return worker_control_plane.register_worker(req)
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err))


@router.post("/workers/{worker_id}/heartbeat")
def record_heartbeat(worker_id: str, payload: HeartbeatPayload) -> Dict[str, Any]:
    success = worker_control_plane.record_heartbeat(
        worker_id=worker_id,
        queue_depth=payload.queue_depth,
        recent_latency_ms=payload.recent_latency_ms,
        gpu_utilization=payload.gpu_utilization,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker node not found.")
    return {"status": "ACK", "worker_id": worker_id}


@router.get("/workers")
def list_workers() -> List[Dict[str, Any]]:
    return worker_control_plane.list_workers()


@router.get("/workers/{worker_id}")
def get_worker(worker_id: str) -> Dict[str, Any]:
    workers = worker_control_plane.list_workers()
    for w in workers:
        if w["worker_id"] == worker_id:
            return w
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker node not found.")


@router.post("/workers/{worker_id}/assign-camera")
def assign_camera(worker_id: str, req: AssignCameraRequest) -> Dict[str, Any]:
    if worker_id not in worker_control_plane.workers:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker node not found.")
    w_node = worker_control_plane.workers[worker_id]
    if req.camera_id not in w_node.assigned_cameras:
        w_node.assigned_cameras.append(req.camera_id)
    return {"status": "ASSIGNED", "worker_id": worker_id, "camera_id": req.camera_id}


@router.post("/workers/{worker_id}/remove-camera")
def remove_camera(worker_id: str, req: AssignCameraRequest) -> Dict[str, Any]:
    if worker_id not in worker_control_plane.workers:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker node not found.")
    w_node = worker_control_plane.workers[worker_id]
    if req.camera_id in w_node.assigned_cameras:
        w_node.assigned_cameras.remove(req.camera_id)
    return {"status": "REMOVED", "worker_id": worker_id, "camera_id": req.camera_id}


@router.post("/workers/events")
def ingest_event(event: RecognitionEvent) -> Dict[str, Any]:
    return worker_control_plane.ingest_recognition_event(event)


@router.get("/cameras")
def list_cameras() -> List[Dict[str, Any]]:
    return worker_control_plane.list_cameras()


@router.post("/cameras", status_code=status.HTTP_201_CREATED)
def register_camera(req: CameraRegistrationRequest) -> Dict[str, Any]:
    return worker_control_plane.register_camera(req)
