"""
Dashboard Metrics API Endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.get("/dashboard")
def get_dashboard_metrics(db: Session = Depends(get_db)):
    metrics_service = MetricsService(db)
    return metrics_service.get_dashboard_stats()
