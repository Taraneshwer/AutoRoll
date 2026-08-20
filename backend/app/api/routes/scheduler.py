"""
Scheduler Status and Rebalance API Endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.scheduler.scheduler import DistributedCameraScheduler

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])
scheduler = DistributedCameraScheduler()


@router.get("/status")
def get_scheduler_status(db: Session = Depends(get_db)):
    """
    Exposes scheduler status, camera assignment matrix, and worker capacity metrics.
    """
    return scheduler.get_scheduler_status(db)


@router.post("/rebalance")
def trigger_rebalance(db: Session = Depends(get_db)):
    """
    Triggers automatic camera workload rebalancing across active workers.
    """
    result = scheduler.rebalance_workload(db)
    return {"status": "rebalanced", "result": result}
