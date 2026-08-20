"""
Camera Management and Manual Assignment API Endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.models import Camera
from app.database.session import get_db
from app.services.scheduler.scheduler import DistributedCameraScheduler

router = APIRouter(prefix="/cameras", tags=["Cameras"])
scheduler = DistributedCameraScheduler()


class CameraCreatePayload(BaseModel):
    name: str
    rtsp_url: str
    location: str | None = None


class CameraAssignPayload(BaseModel):
    worker_id: str | None = None


@router.post("", status_code=status.HTTP_201_CREATED)
def create_camera(payload: CameraCreatePayload, db: Session = Depends(get_db)):
    camera = Camera(
        name=payload.name,
        rtsp_url=payload.rtsp_url,
        location=payload.location,
    )
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return {"camera_id": camera.id, "name": camera.name, "rtsp_url": camera.rtsp_url}


@router.get("")
def list_cameras(db: Session = Depends(get_db)):
    cameras = db.query(Camera).all()
    return [
        {
            "camera_id": c.id,
            "name": c.name,
            "rtsp_url": c.rtsp_url,
            "location": c.location,
            "is_active": c.is_active,
            "assigned_worker_id": c.assigned_worker_id,
        }
        for c in cameras
    ]


@router.post("/{camera_id}/assign")
def assign_camera(
    camera_id: str,
    payload: CameraAssignPayload | None = None,
    db: Session = Depends(get_db),
):
    target_worker = payload.worker_id if payload else None
    assigned_worker_id = scheduler.assign_camera(
        camera_id=camera_id, worker_id=target_worker, db=db
    )

    if not assigned_worker_id:
        raise HTTPException(
            status_code=400,
            detail=f"Could not assign camera '{camera_id}'. No available worker capacity.",
        )

    return {"status": "assigned", "camera_id": camera_id, "assigned_worker_id": assigned_worker_id}


@router.post("/{camera_id}/unassign")
def unassign_camera(camera_id: str, db: Session = Depends(get_db)):
    success = scheduler.unassign_camera(camera_id, db)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Camera '{camera_id}' not found or not assigned."
        )
    return {"status": "unassigned", "camera_id": camera_id}


@router.get("/{camera_id}/worker")
def get_camera_worker(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
    return {
        "camera_id": camera_id,
        "assigned_worker_id": camera.assigned_worker_id,
        "status": "ASSIGNED" if camera.assigned_worker_id else "UNASSIGNED",
    }


@router.post("/{camera_id}/reassign")
def reassign_camera(
    camera_id: str,
    payload: CameraAssignPayload | None = None,
    db: Session = Depends(get_db),
):
    target_worker = payload.worker_id if payload else None
    assigned_worker_id = scheduler.assign_camera(
        camera_id=camera_id, worker_id=target_worker, db=db
    )

    if not assigned_worker_id:
        raise HTTPException(
            status_code=400,
            detail=f"Could not reassign camera '{camera_id}'. No available worker capacity.",
        )

    return {"status": "reassigned", "camera_id": camera_id, "assigned_worker_id": assigned_worker_id}

