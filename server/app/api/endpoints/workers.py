"""
Worker Registration and Heartbeat API Endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from server.app.db.session import get_db
from server.app.scheduler.scheduler import DistributedCameraScheduler

router = APIRouter(prefix="/workers", tags=["Workers"])
scheduler = DistributedCameraScheduler()


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_worker(payload: dict[str, Any], db: Session = Depends(get_db)):
    if "worker_id" not in payload:
        raise HTTPException(status_code=400, detail="Missing worker_id in payload.")
    worker = scheduler.register_worker(payload, db)
    return {"status": "registered", "worker_id": worker.id, "state": worker.state}


@router.post("/heartbeat")
def worker_heartbeat(payload: dict[str, Any], db: Session = Depends(get_db)):
    if "worker_id" not in payload:
        raise HTTPException(status_code=400, detail="Missing worker_id in payload.")
    result = scheduler.update_heartbeat(payload, db)
    return result


@router.get("")
def list_workers(db: Session = Depends(get_db)):
    status_info = scheduler.get_scheduler_status(db)
    return status_info["workers"]
